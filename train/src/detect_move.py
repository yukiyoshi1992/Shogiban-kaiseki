"""
差分検出・指し手推定スクリプト
- 連続する2枚の盤面画像を比較して指し手を推定
- python-shogiで合法手チェック
- 対局フォルダの全画像を順番に処理してKIFを生成

使い方:
  python detect_move.py \
    --images "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/対局/" \
    --calib  "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/対局_calib/IMG_20260615_144922_calib.json" \
    --model  "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/models/" \
    --out    "//YukiYoshiNAS/Shogiban-kaiseki-tool/runtime/result/output.kif"
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
import shogi
import shogi.KIF

# -------------------------------------------------------
# ラベル定義
# -------------------------------------------------------
ALL_LABELS = [
    "empty",
    "sente_fu","sente_kyo","sente_kei","sente_gin","sente_kin",
    "sente_kaku","sente_hi","sente_ou",
    "sente_tokin","sente_nari_kyo","sente_nari_kei","sente_nari_gin",
    "sente_uma","sente_ryu",
    "gote_fu","gote_kyo","gote_kei","gote_gin","gote_kin",
    "gote_kaku","gote_hi","gote_ou",
    "gote_tokin","gote_nari_kyo","gote_nari_kei","gote_nari_gin",
    "gote_uma","gote_ryu",
]
NUM_CLASSES = len(ALL_LABELS)
LABEL_TO_IDX = {l: i for i, l in enumerate(ALL_LABELS)}

# ラベル → shogi駒情報
LABEL_TO_PIECE = {
    "fu": shogi.PAWN, "kyo": shogi.LANCE, "kei": shogi.KNIGHT,
    "gin": shogi.SILVER, "kin": shogi.GOLD, "kaku": shogi.BISHOP,
    "hi": shogi.ROOK, "ou": shogi.KING,
    "tokin": shogi.PROM_PAWN, "nari_kyo": shogi.PROM_LANCE,
    "nari_kei": shogi.PROM_KNIGHT, "nari_gin": shogi.PROM_SILVER,
    "uma": shogi.PROM_BISHOP, "ryu": shogi.PROM_ROOK,
}

# 成り駒 → 元の駒
PROMOTED_TO_BASE = {
    "tokin": "fu", "nari_kyo": "kyo", "nari_kei": "kei",
    "nari_gin": "gin", "uma": "kaku", "ryu": "hi",
}

# -------------------------------------------------------
# モデル
# -------------------------------------------------------
def load_model(model_folder):
    model_path = Path(model_folder) / "best_model.pth"
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_features, 256),
        nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_board(model, warped, cell_px=100, grid_size=9):
    """補正済み盤面画像から9×9のラベルグリッドと信頼度グリッドを返す"""
    labels = [[None]*grid_size for _ in range(grid_size)]
    confs  = [[0.0]*grid_size for _ in range(grid_size)]
    for row in range(grid_size):
        for col in range(grid_size):
            x1, y1 = col*cell_px, row*cell_px
            cell = warped[y1:y1+cell_px, x1:x1+cell_px]
            img_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                out = model(tensor)
                probs = torch.softmax(out, dim=1)
                conf, pred = probs.max(1)
            labels[row][col] = ALL_LABELS[pred.item()]
            confs[row][col]  = conf.item()
    return labels, confs

# -------------------------------------------------------
# 画像処理
# -------------------------------------------------------
def load_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def warp_image(img, calib):
    M = np.array(calib["perspective_matrix"], dtype="float32")
    cell_px   = calib.get("cell_px", 100)
    grid_size = calib.get("grid_size", 9)
    side = cell_px * grid_size
    return cv2.warpPerspective(img, M, (side, side)), cell_px, grid_size

# -------------------------------------------------------
# 青三角検出（盤の向き判定）
# -------------------------------------------------------
BLUE_LOWER = np.array([90,  60,  40])
BLUE_UPPER = np.array([130, 255, 255])

def detect_blue_triangle_direction(img):
    """
    青い三角形を検出して頂点方向を返す
    Returns: "up"/"down"/"left"/"right" or None
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 2.5:
            continue
        ret, triangle = cv2.minEnclosingTriangle(cnt)
        if triangle is None:
            continue
        tri_area = cv2.contourArea(triangle.reshape(-1, 2).astype(np.float32))
        if tri_area <= 0:
            continue
        triangularity = area / tri_area
        if triangularity > best_score:
            best_score = triangularity
            best = (cnt, triangle)

    if best is None or best_score < 0.6:
        return None

    cnt, triangle = best
    pts = triangle.reshape(-1, 2)
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    dists = [np.hypot(p[0]-cx, p[1]-cy) for p in pts]
    apex = pts[int(np.argmax(dists))]
    dx = apex[0] - cx
    dy = apex[1] - cy
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"

def get_transforms_from_triangle(direction):
    """
    青三角の頂点方向から、ラベルグリッドを標準向きに変換する関数リストを返す
    基準: 頂点=right(先手=左) → normal（学習データと同じ）

    【物理的な意味】
    スマホが横向き（先手が左）: 頂点=right → normal
    スマホが横向き（先手が右）: 頂点=left  → rot180
    スマホが縦向き（先手が下）: 頂点=up    → rot90ccw
      ※先手を左に持ってくるには反時計90度回転
    スマホが縦向き（先手が上）: 頂点=down  → rot90cw
      ※先手を左に持ってくるには時計90度回転
    """
    if direction == "right":
        return [lambda g: g], "normal(頂点right)"
    elif direction == "left":
        return [rotate_grid_180], "rot180(頂点left)"
    elif direction == "up":
        return [rotate_grid_90ccw], "rot90ccw(頂点up→先手が下)"
    elif direction == "down":
        return [rotate_grid_90cw], "rot90cw(頂点down→先手が上)"
    else:
        return [lambda g: g], "unknown→normal"

# -------------------------------------------------------
# 画像の向き検出（0手目の初期配置から4方向を判定）
# -------------------------------------------------------
# 標準の初期配置（画像座標: row0=9筋(上), col0=1段目(左/先手側)）
# extract_cells.pyで確定した座標系と同じ
INITIAL_BOARD_STD = [
    ["sente_kyo","empty","sente_fu","empty","empty","empty","gote_fu","empty","gote_kyo"],
    ["sente_kei","sente_kaku","sente_fu","empty","empty","empty","gote_fu","gote_hi","gote_kei"],
    ["sente_gin","empty","sente_fu","empty","empty","empty","gote_fu","empty","gote_gin"],
    ["sente_kin","empty","sente_fu","empty","empty","empty","gote_fu","empty","gote_kin"],
    ["sente_ou","empty","sente_fu","empty","empty","empty","gote_fu","empty","gote_ou"],
    ["sente_kin","empty","sente_fu","empty","empty","empty","gote_fu","empty","gote_kin"],
    ["sente_gin","empty","sente_fu","empty","empty","empty","gote_fu","empty","gote_gin"],
    ["sente_kei","sente_hi","sente_fu","empty","empty","empty","gote_fu","gote_kaku","gote_kei"],
    ["sente_kyo","empty","sente_fu","empty","empty","empty","gote_fu","empty","gote_kyo"],
]

def rotate_image_for_direction(img, direction):
    """
    青三角の頂点方向に応じて画像を回転させる。
    学習データは横向き（先手が左=right）基準なので、
    縦向き撮影（up/down）の場合は画像ごと回転して横向きに直す。
    これにより切り出したマス画像の駒の向きが学習データと一致する。
      right → そのまま
      left  → 180度回転
      up    → 反時計90度（先手が下→先手を左へ）
      down  → 時計90度（先手が上→先手を左へ）
    """
    import cv2 as _cv2
    if direction == "right" or direction is None:
        return img
    elif direction == "left":
        return _cv2.rotate(img, _cv2.ROTATE_180)
    elif direction == "up":
        return _cv2.rotate(img, _cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == "down":
        return _cv2.rotate(img, _cv2.ROTATE_90_CLOCKWISE)
    return img

def rotate_calib_for_direction(calib, direction):
    """
    画像回転に合わせてキャリブレーションの座標を変換する。
    calibは回転前の画像座標で保存されているので、
    画像を回転した場合はcalib座標も同じ変換を適用する必要がある。

    original_size: [width, height]
    up   (反時計90度): (x,y) → (y, W-1-x)  新サイズ: (H, W)
    down (時計90度):   (x,y) → (H-1-y, x)  新サイズ: (H, W)
    left (180度):      (x,y) → (W-1-x, H-1-y)
    right: そのまま
    """
    import numpy as np
    import cv2
    if direction is None or direction == "right":
        return calib

    W, H = calib["original_size"]  # [width, height]
    corners = np.array(calib["corner_points"], dtype=np.float32)

    if direction == "up":  # ROTATE_90_COUNTERCLOCKWISE
        # (x,y) → (y, W-1-x), 新サイズ: (H, W)
        new_corners = np.column_stack([corners[:, 1], W - 1 - corners[:, 0]])
        new_size = [H, W]
    elif direction == "down":  # ROTATE_90_CLOCKWISE
        # (x,y) → (H-1-y, x), 新サイズ: (H, W)
        new_corners = np.column_stack([H - 1 - corners[:, 1], corners[:, 0]])
        new_size = [H, W]
    elif direction == "left":  # ROTATE_180
        # (x,y) → (W-1-x, H-1-y)
        new_corners = np.column_stack([W - 1 - corners[:, 0], H - 1 - corners[:, 1]])
        new_size = [W, H]
    else:
        return calib

    # 新しいcalibを作成
    new_calib = dict(calib)
    new_calib = {k: v for k, v in calib.items()}
    new_calib["original_size"] = new_size
    new_calib["corner_points"] = new_corners.tolist()

    # 透視変換行列を再計算
    cell_px = calib.get("cell_px", 100)
    grid_size = calib.get("grid_size", 9)
    side = cell_px * grid_size
    dst_pts = np.array([
        [0, 0], [side, 0], [side, side], [0, side]
    ], dtype=np.float32)

    # corner_pointsの順序: [左上, 右上, 右下, 左下]
    src = new_corners.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst_pts)
    M_inv = np.linalg.inv(M)

    new_calib["perspective_matrix"] = M.tolist()
    new_calib["perspective_matrix_inv"] = M_inv.tolist()

    # cell_centers_originalも再計算
    centers = []
    for r in range(grid_size):
        for c in range(grid_size):
            px = (c + 0.5) * cell_px
            py = (r + 0.5) * cell_px
            pt = np.array([[[px, py]]], dtype=np.float32)
            orig = cv2.perspectiveTransform(pt, M_inv)
            centers.append(orig[0][0].tolist())
    new_calib["cell_centers_original"] = centers

    return new_calib

def rotate_grid_180(grid):
    """グリッドを180度回転（駒の色は変えない、位置だけ回転）"""
    return [[grid[8-r][8-c] for c in range(9)] for r in range(9)]

def rotate_grid_90cw(grid):
    """グリッドを時計回り90度回転"""
    return [[grid[8-c][r] for c in range(9)] for r in range(9)]

def rotate_grid_90ccw(grid):
    """グリッドを反時計回り90度回転"""
    return [[grid[c][8-r] for c in range(9)] for r in range(9)]

def flip_grid_vertical(grid):
    """上下反転（row→8-row、col保持、色は変えない）"""
    return [[grid[8-r][c] for c in range(9)] for r in range(9)]

def flip_grid_horizontal(grid):
    """左右反転（col→8-col、row保持、色は変えない）"""
    return [[grid[r][8-c] for c in range(9)] for r in range(9)]

def swap_colors(grid):
    """先手後手を入れ替える"""
    out = []
    for row in grid:
        new_row = []
        for cell in row:
            if cell == "empty":
                new_row.append("empty")
            elif cell.startswith("sente_"):
                new_row.append("gote_" + cell[6:])
            elif cell.startswith("gote_"):
                new_row.append("sente_" + cell[5:])
            else:
                new_row.append(cell)
        out.append(new_row)
    return out

def grid_similarity(grid_a, grid_b):
    """2つのグリッドの一致率（0.0-1.0）"""
    match = 0
    for r in range(9):
        for c in range(9):
            if grid_a[r][c] == grid_b[r][c]:
                match += 1
    return match / 81.0

def detect_orientation_from_initial(labels_0):
    """
    0手目のラベルグリッドから盤の向きを判定する。
    認識結果を4方向に回転させ、標準初期配置に最も一致する変換を返す。
    Returns: 変換関数のリスト（labels に順に適用すると標準向きになる）
    """
    # 候補となる変換（回転 + 色入れ替えの組み合わせ）
    candidates = []

    def identity(g): return g

    # 0度
    candidates.append(("normal", [identity]))
    # 180度回転
    candidates.append(("rot180", [rotate_grid_180]))
    # 180度回転 + 色入替（先手後手が逆向きのカメラ）
    candidates.append(("rot180_swap", [rotate_grid_180, swap_colors]))
    # 色入替のみ
    candidates.append(("swap", [swap_colors]))

    best_name = "normal"
    best_transforms = [identity]
    best_score = -1.0

    for name, transforms_list in candidates:
        g = labels_0
        for t in transforms_list:
            g = t(g)
        score = grid_similarity(g, INITIAL_BOARD_STD)
        print(f"  向き候補 {name}: 一致率 {score:.1%}")
        if score > best_score:
            best_score = score
            best_name = name
            best_transforms = transforms_list

    print(f"  → 判定: {best_name} (一致率 {best_score:.1%})")
    return best_transforms, best_name, best_score

def apply_transforms(labels, transforms_list):
    """変換リストをラベルグリッドに順に適用"""
    if labels is None:
        return None
    g = labels
    for t in transforms_list:
        g = t(g)
    return g

# -------------------------------------------------------
# 画像座標 ↔ shogi座標変換
# -------------------------------------------------------
def rc_to_square(row, col):
    """画像(row,col) → shogi square番号"""
    # row=0=9筋 col=0=9段（後手側）
    # square = (8-col)*9 + row
    return (8 - col) * 9 + row

def square_to_rc(sq):
    """shogi square → 画像(row, col)"""
    col = 8 - (sq // 9)
    row = sq % 9
    return row, col

def label_to_shogi_piece(label):
    """ラベル文字列 → (piece_type, color)"""
    if label == "empty":
        return None, None
    if label.startswith("sente_"):
        piece_name = label[6:]
        color = shogi.BLACK
    else:
        piece_name = label[5:]
        color = shogi.WHITE
    piece_type = LABEL_TO_PIECE.get(piece_name)
    return piece_type, color

# -------------------------------------------------------
# 差分検出・指し手推定
# -------------------------------------------------------
def detect_changes(labels_prev, labels_curr):
    """
    2つのラベルグリッドを比較して変化したマスを返す

    変化パターン:
      A→empty : 駒が消えた（移動元 or 取られた）
      empty→B : 駒が現れた（移動先 or 打ち駒）
      A→B     : 駒種/色が変わった（成り or 相手駒を取って置き換え）

    Returns:
        src_squares: [(row,col,label)] 移動元候補（駒が消えたor色が変わった）
        dst_squares: [(row,col,label)] 移動先候補（駒が現れたor色が変わった）
    """
    src_squares = []  # 移動元（駒が消えた or 別の駒に置き換わった）
    dst_squares = []  # 移動先（駒が現れた or 別の駒に置き換わった）

    for row in range(9):
        for col in range(9):
            prev = labels_prev[row][col]
            curr = labels_curr[row][col]
            if prev == curr:
                continue

            if prev != "empty" and curr == "empty":
                # 駒が消えた → 移動元
                src_squares.append((row, col, prev))
            elif prev == "empty" and curr != "empty":
                # 駒が現れた → 移動先（打ち駒含む）
                dst_squares.append((row, col, curr))
            else:
                # A→B: 駒が置き換わった
                # prevの色とcurrの色が違う → 取り駒（移動先）
                # prevの色とcurrの色が同じ → 成り（移動先、旧位置が移動元）
                prev_color = "sente" if prev.startswith("sente_") else "gote"
                curr_color = "sente" if curr.startswith("sente_") else "gote"
                if prev_color != curr_color:
                    # 相手の駒を取った → このマスは移動先
                    # prevは取られた駒（消失）、currは移動してきた駒
                    dst_squares.append((row, col, curr))
                    # prevは取られた側なのでsrc_squaresには入れない
                else:
                    # 同色で駒種が変わった → 成り（このマスが移動先）
                    src_squares.append((row, col, prev))
                    dst_squares.append((row, col, curr))

    return src_squares, dst_squares

def infer_move(src_squares, dst_squares, board, move_number):
    """
    差分情報から指し手をUSI形式で推定
    src_squares: 移動元候補
    dst_squares: 移動先候補
    Returns: list of candidate USI move strings
    """
    candidates = []

    # ケース1: 通常移動（src=1, dst=1）
    if len(src_squares) == 1 and len(dst_squares) == 1:
        src_row, src_col, src_label = src_squares[0]
        dst_row, dst_col, dst_label = dst_squares[0]

        src_sq = rc_to_square(src_row, src_col)
        dst_sq = rc_to_square(dst_row, dst_col)
        src_piece, src_color = label_to_shogi_piece(src_label)
        dst_piece, dst_color = label_to_shogi_piece(dst_label)

        if src_piece and dst_piece:
            src_name = src_label.split("_", 1)[1]
            dst_name = dst_label.split("_", 1)[1]
            promoted = (dst_name in PROMOTED_TO_BASE and
                        PROMOTED_TO_BASE.get(dst_name) == src_name)
            usi = f"{shogi.SQUARE_NAMES[src_sq]}{shogi.SQUARE_NAMES[dst_sq]}"
            if promoted:
                candidates.append(usi + "+")
            else:
                candidates.append(usi)
                if src_piece in [shogi.PAWN, shogi.LANCE, shogi.KNIGHT,
                                  shogi.SILVER, shogi.BISHOP, shogi.ROOK]:
                    candidates.append(usi + "+")

    # ケース2: 取り駒（src=1, dst=1 だが A→B の変換で処理済み）
    # detect_changesで取り駒マスはdst_squaresに入る
    # src=1（移動元）+ dst=1（取った先）の組み合わせはケース1で処理
    # → 追加でsrc=1, dst=1の取り駒も同様に処理されている

    # ケース3: 打ち駒（src=0, dst=1）
    elif len(src_squares) == 0 and len(dst_squares) == 1:
        dst_row, dst_col, dst_label = dst_squares[0]
        dst_sq = rc_to_square(dst_row, dst_col)
        dst_name = dst_label.split("_", 1)[1]
        piece_type = LABEL_TO_PIECE.get(dst_name)
        if piece_type:
            piece_sym = shogi.PIECE_SYMBOLS[piece_type].upper()
            usi = f"{piece_sym}*{shogi.SQUARE_NAMES[dst_sq]}"
            candidates.append(usi)

    # ケース4: 複数変化（認識誤りや複雑なケース）→ 合法手から絞り込み
    elif len(src_squares) >= 1 and len(dst_squares) >= 1:
        # 全組み合わせを試す
        for src_row, src_col, src_label in src_squares:
            src_sq = rc_to_square(src_row, src_col)
            src_piece, src_color = label_to_shogi_piece(src_label)
            if not src_piece:
                continue
            for dst_row, dst_col, dst_label in dst_squares:
                dst_sq = rc_to_square(dst_row, dst_col)
                dst_piece, dst_color = label_to_shogi_piece(dst_label)
                if not dst_piece:
                    continue
                if src_color != dst_color:
                    continue  # 同色のみ
                src_name = src_label.split("_", 1)[1]
                dst_name = dst_label.split("_", 1)[1]
                promoted = (dst_name in PROMOTED_TO_BASE and
                            PROMOTED_TO_BASE.get(dst_name) == src_name)
                usi = f"{shogi.SQUARE_NAMES[src_sq]}{shogi.SQUARE_NAMES[dst_sq]}"
                if promoted:
                    candidates.append(usi + "+")
                else:
                    candidates.append(usi)
                    if src_piece in [shogi.PAWN, shogi.LANCE, shogi.KNIGHT,
                                      shogi.SILVER, shogi.BISHOP, shogi.ROOK]:
                        candidates.append(usi + "+")

    return candidates

def find_legal_move(candidates, board):
    """候補手から合法手を選んで返す"""
    legal_moves = list(board.legal_moves)
    legal_usi   = [m.usi() for m in legal_moves]

    for cand in candidates:
        if cand in legal_usi:
            return cand

    return None

# -------------------------------------------------------
# KIF生成
# -------------------------------------------------------
def moves_to_kif(moves_usi, out_path):
    """USI形式の手リストをKIF形式で保存"""
    PIECE_JA = {
        'p': '歩', 'l': '香', 'n': '桂', 's': '銀', 'g': '金',
        'b': '角', 'r': '飛', 'k': '玉',
        '+p': 'と', '+l': '成香', '+n': '成桂', '+s': '成銀',
        '+b': '馬', '+r': '龍',
    }
    FILE_JA = ['', '１', '２', '３', '４', '５', '６', '７', '８', '９']
    RANK_JA = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九']

    def sq_name_to_ja(sq_name):
        """'7f' → '７六'"""
        file_map = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9}
        file_num = int(sq_name[0])
        rank_num = file_map.get(sq_name[1], 0)
        return f"{FILE_JA[file_num]}{RANK_JA[rank_num]}"

    lines = ["手数----指手---------消費時間--"]
    board = shogi.Board()

    for i, usi in enumerate(moves_usi):
        move_num = i + 1
        try:
            # 一時boardでmoveオブジェクトを取得
            _b = shogi.Board()
            _b.set_sfen(board.sfen())
            _b.push_usi(usi)
            move = list(_b.move_stack)[-1]
            # 移動先
            dst_sq   = move.to_square
            dst_name = shogi.SQUARE_NAMES[dst_sq]
            dst_ja   = sq_name_to_ja(dst_name)

            if move.drop_piece_type:
                # 打ち駒
                piece_sym = shogi.PIECE_SYMBOLS[move.drop_piece_type]
                piece_ja  = PIECE_JA.get(piece_sym, '?')
                move_ja   = f"{dst_ja}{piece_ja}打"
            else:
                src_sq   = move.from_square
                src_name = shogi.SQUARE_NAMES[src_sq]
                # KIFの移動元表記は半角数字2桁（筋+段）。例: "27" = 2筋7段。
                # sq_name_to_ja()は全角筋+漢数字段を返すので使わず、USI名から直接数字を作る。
                file_map = {'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9}
                src_ascii = f"{int(src_name[0])}{file_map.get(src_name[1], 0)}"
                piece    = board.piece_at(src_sq)
                piece_sym = shogi.PIECE_SYMBOLS[piece.piece_type] if piece else '?'
                piece_ja  = PIECE_JA.get(piece_sym, '?')
                promoted_str = "成" if move.promotion else ""
                move_ja   = f"{dst_ja}{piece_ja}{promoted_str}({src_ascii})"

            lines.append(f"{move_num:4d} {move_ja} (00:00/00:00:00)")
            board.push(move)
        except Exception as e:
            lines.append(f"{move_num:4d} (ERROR: {e})")

    lines.append(f"{len(moves_usi)+1:4d} 投了 (00:00/00:00:00)")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"KIF保存: {out_path}")

# -------------------------------------------------------
# メイン処理
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="差分検出・指し手推定")
    parser.add_argument("--images", required=True, help="対局画像フォルダ")
    parser.add_argument("--calib",  required=True, help="キャリブレーションJSONパス（0手目用）")
    parser.add_argument("--model",  required=True, help="モデルフォルダ")
    parser.add_argument("--out",    required=True, help="出力KIFパス")
    parser.add_argument("--pattern",default="IMG_", help="画像ファイルパターン")
    parser.add_argument("--preview",action="store_true", help="各手の差分をプレビュー表示")
    parser.add_argument("--force-direction", choices=["right","left","up","down"],
                        help="画像回転方向を手動指定（青三角検出を上書き）。"
                             "right=横向き先手左(回転なし) / up=縦向き先手手前(反時計90度) / "
                             "down=縦向き先手奥(時計90度) / left=横向き先手右(180度)")
    args = parser.parse_args()

    # 画像一覧（ファイル名順）
    images_folder = Path(args.images)
    image_files = sorted(set(
        list(images_folder.glob(f"{args.pattern}*.jpg")) +
        list(images_folder.glob(f"{args.pattern}*.JPG"))
    ))
    if not image_files:
        print(f"[ERROR] 画像が見つかりません: {images_folder}")
        sys.exit(1)
    print(f"対象画像: {len(image_files)} 枚")

    # キャリブレーション読み込み
    with open(args.calib, "r", encoding="utf-8") as f:
        calib = json.load(f)

    # モデル読み込み
    print("モデル読み込み中...")
    model = load_model(args.model)

    # ===== Step1: 画像回転方向を決定 =====
    # 【方式A】画像をこの方向に回転して横向き（先手左）に直す。
    #   right=回転なし / up=反時計90度 / down=時計90度 / left=180度
    rot_desc = {
        "right": "回転なし（横向き・先手左）",
        "left":  "180度回転（横向き・先手右）",
        "up":    "反時計90度（縦向き・先手手前→左へ）",
        "down":  "時計90度（縦向き・先手奥→左へ）",
    }
    if args.force_direction:
        img_dir = args.force_direction
        print(f"\n画像回転方向: {img_dir}（手動指定）→ {rot_desc.get(img_dir)}")
    else:
        print("\n向き判定中（0手目の青三角を検出）...")
        first_img = load_image(image_files[0])
        triangle_dir = detect_blue_triangle_direction(first_img) if first_img is not None else None
        if triangle_dir:
            img_dir = triangle_dir
            print(f"  青三角の頂点方向: {triangle_dir} → {rot_desc.get(triangle_dir, '回転なし')}")
        else:
            img_dir = "right"
            print("  [WARNING] 青三角を検出できませんでした → 回転なし(right)扱い")

    # ===== Step2: 全画像を認識 =====
    # 【方式A：画像回転1系統】
    #   各画像を img_dir 方向に回転（縦→横）してから、回転後画像で取得済みのcalibを適用。
    #   calibは「正しい向き（横向き・先手左）に回転済みの画像」で取得されている前提。
    #   → calib座標回転もグリッド変換も不要。マス内の駒が必ず学習データと同じ向きになる。
    print("盤面認識中...")
    all_labels = []
    all_confs  = []
    all_imgs   = []
    for i, img_path in enumerate(image_files):
        img = load_image(img_path)
        if img is None:
            print(f"  [ERROR] 読み込み失敗: {img_path.name}")
            all_labels.append(None)
            all_confs.append(None)
            all_imgs.append(None)
            continue
        # 画像を正しい向きに回転（縦向き撮影を横向きに直す）
        img_rotated = rotate_image_for_direction(img, img_dir)
        warped, cell_px, grid_size = warp_image(img_rotated, calib)
        labels, confs = predict_board(model, warped, cell_px, grid_size)
        all_labels.append(labels)
        all_confs.append(confs)
        all_imgs.append(warped)
        print(f"  [{i+1}/{len(image_files)}] {img_path.name}")

    # グリッド変換は行わない（画像回転で吸収済み = normal固定）

    # 差分検出・指し手推定
    print("\n差分検出・指し手推定中...")
    board = shogi.Board()
    moves_usi = []
    errors = []

    for i in range(1, len(image_files)):
        if all_labels[i-1] is None or all_labels[i] is None:
            print(f"  手{i}: [SKIP] 画像読み込みエラー")
            continue

        src_squares, dst_squares = detect_changes(all_labels[i-1], all_labels[i])
        candidates = infer_move(src_squares, dst_squares, board, i)
        legal_move = find_legal_move(candidates, board)

        if legal_move:
            moves_usi.append(legal_move)
            board.push_usi(legal_move)
            print(f"  move {i:3d}: {legal_move} OK")
        else:
            if not src_squares and not dst_squares:
                print(f"  move {i:3d}: [no change] skip")
            else:
                msg = f"move {i}: no legal move (candidates={candidates}, src={src_squares}, dst={dst_squares})"
                print(f"  [ERROR] {msg}")
                errors.append(msg)
                # error: reset current labels to previous to keep board in sync
                all_labels[i] = all_labels[i-1]

        if args.preview:
            # 差分を視覚化
            warped_prev = cv2.warpPerspective(
                load_image(image_files[i-1]),
                np.array(calib["perspective_matrix"], dtype="float32"),
                (900, 900)
            )
            warped_curr = cv2.warpPerspective(
                load_image(image_files[i]),
                np.array(calib["perspective_matrix"], dtype="float32"),
                (900, 900)
            )
            vis = np.hstack([warped_prev, warped_curr])
            cell_px = 100
            for row, col, label in src_squares:
                x1 = col*cell_px; y1 = row*cell_px
                cv2.rectangle(vis, (x1,y1), (x1+cell_px,y1+cell_px), (0,0,255), 3)
            for row, col, label in dst_squares:
                x1 = col*cell_px+900; y1 = row*cell_px
                cv2.rectangle(vis, (x1,y1), (x1+cell_px,y1+cell_px), (0,255,0), 3)
            cv2.namedWindow("Diff", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Diff", 1200, 600)
            title = f"手{i}: {legal_move if legal_move else 'ERROR'} | 赤=消失 緑=出現 | Enter:次 ESC:終了"
            cv2.setWindowTitle("Diff", title)
            cv2.imshow("Diff", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                args.preview = False
                cv2.destroyAllWindows()

    if args.preview:
        cv2.destroyAllWindows()

    # 結果サマリー
    print(f"\n{'='*50}")
    print(f"推定手数: {len(moves_usi)} 手")
    print(f"エラー数: {len(errors)}")
    if errors:
        print("エラー詳細:")
        for e in errors:
            print(f"  {e}")

    # KIF出力
    if moves_usi:
        moves_to_kif(moves_usi, args.out)
    else:
        print("[WARNING] 推定できた手が0でした")

if __name__ == "__main__":
    main()
