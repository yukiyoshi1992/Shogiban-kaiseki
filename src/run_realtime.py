"""
本番運用スクリプト: ファイル監視 → リアルタイムKIF生成

0手目（対局準備）の処理フロー（2026-06-19 chat.txtでの指示に基づく確定版）:
  1. 青三角で向き判定 → エラー(検出不可)なら3へ
  2. 赤丸でキャリブレーション → エラー(4個検出できない)なら3へ
  3. 認識結果を将棋の初期配置と照合 → 不一致なら3へ
     (上記いずれかでエラーが出た場合): 人間が盤の向き指定+四隅クリックで
     キャリブレーションし、その結果を無条件に採用して次へ進む。
     「赤丸が見える状態で撮り直してください」のような自動リトライループは行わない
     —— エラーが出たら即座に人間が直す、それだけ。
  4. 準備OK表示 → 対局開始（Enterで進む。画面UI実装時は対局開始ボタンに相当）
- 各手は直前の差分ではなく確定済みboardと比較し、合法手を総当たりで探して判定（classify_frame）
- 対局終了（一定時間 新規画像なし or 手動終了）でKIF確定出力

使い方:
  python src/run_realtime.py \
    --watch  "//YukiYoshiNAS/Shogiban-kaiseki-tool/runtime/input/" \
    --model  "//YukiYoshiNAS/Shogiban-kaiseki-tool/src/models/" \
    --out    "//YukiYoshiNAS/Shogiban-kaiseki-tool/runtime/result/" \
    --idle-timeout 600

オプション:
  --idle-timeout 秒  最後の画像からこの秒数 新規がなければ対局終了とみなす（デフォルト600=10分）
  --poll-interval 秒 フォルダをチェックする間隔（デフォルト5秒）
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
import time
from pathlib import Path
from datetime import datetime
import shogi

# detect_move.py のロジックを再利用するため同じ関数群を内包
# （本番では import detect_move でもよいが、単体で動くよう自己完結させる）

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

LABEL_TO_PIECE = {
    "fu":shogi.PAWN,"kyo":shogi.LANCE,"kei":shogi.KNIGHT,"gin":shogi.SILVER,
    "kin":shogi.GOLD,"kaku":shogi.BISHOP,"hi":shogi.ROOK,"ou":shogi.KING,
    "tokin":shogi.PROM_PAWN,"nari_kyo":shogi.PROM_LANCE,"nari_kei":shogi.PROM_KNIGHT,
    "nari_gin":shogi.PROM_SILVER,"uma":shogi.PROM_BISHOP,"ryu":shogi.PROM_ROOK,
}
# shogi駒情報 → ラベル名（LABEL_TO_PIECEの逆引き）
PIECE_TO_LABEL = {v: k for k, v in LABEL_TO_PIECE.items()}

# classify_frame()のしきい値（暫定値。新テストデータでの検証結果次第で調整する）
MOVE_DIFF_THRESHOLD = 2  # この差分数以下なら「この合法手(列)が起きた」と認める

# ===== モデル =====
def load_model(model_folder):
    model_path = Path(model_folder) / "best_model.pth"
    model = models.mobilenet_v2(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(in_f,256), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(256,NUM_CLASSES))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def predict_board(model, warped, cell_px=100, grid_size=9):
    labels = [[None]*grid_size for _ in range(grid_size)]
    for row in range(grid_size):
        for col in range(grid_size):
            x1,y1 = col*cell_px, row*cell_px
            cell = warped[y1:y1+cell_px, x1:x1+cell_px]
            pil = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            t = transform(pil).unsqueeze(0)
            with torch.no_grad():
                out = model(t); pred = out.max(1)[1]
            labels[row][col] = ALL_LABELS[pred.item()]
    return labels

# ===== 画像処理 =====
def load_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

# ===== 赤丸キャリブレーション =====
RED_LOWER1=np.array([0,120,80]); RED_UPPER1=np.array([10,255,255])
RED_LOWER2=np.array([165,120,80]); RED_UPPER2=np.array([180,255,255])
CIRC_MIN=0.55; BOARD_RADIUS_RATIO=0.006

def detect_red_circles(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.bitwise_or(cv2.inRange(hsv,RED_LOWER1,RED_UPPER1),
                        cv2.inRange(hsv,RED_LOWER2,RED_UPPER2))
    k=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    circles=[]
    for c in cnts:
        a=cv2.contourArea(c)
        if a<200: continue
        peri=cv2.arcLength(c,True)
        if peri==0: continue
        circ=4*np.pi*a/(peri*peri)
        (x,y),r=cv2.minEnclosingCircle(c)
        if 8<=r<=80:
            circles.append((float(x),float(y),float(r),float(circ)))
    return circles

def select_board_corners(circles, shape):
    h,w=shape[:2]; diag=(w**2+h**2)**0.5
    filt=[c for c in circles if c[3]>=CIRC_MIN]
    if len(filt)<4: filt=[c for c in circles if c[3]>=0.45]
    if len(filt)<4: return None
    rth=diag*BOARD_RADIUS_RATIO
    cand=[c for c in filt if c[2]>=rth]
    if len(cand)<4: cand=sorted(filt,key=lambda c:-c[2])[:max(4,len(cand))]
    corners=[(0,0),(w,0),(w,h),(0,h)]; sel=[]; used=set()
    for icx,icy in corners:
        best=-1; bd=1e18
        for i,c in enumerate(cand):
            if i in used: continue
            d=(c[0]-icx)**2+(c[1]-icy)**2
            if d<bd: bd=d; best=i
        if best<0: return None
        used.add(best); sel.append((cand[best][0],cand[best][1]))
    return sel

def order_points(pts):
    pts=np.array(pts,dtype="float32"); s=pts.sum(1); d=np.diff(pts,axis=1)
    o=np.zeros((4,2),dtype="float32")
    o[0]=pts[np.argmin(s)]; o[1]=pts[np.argmin(d)]
    o[2]=pts[np.argmax(s)]; o[3]=pts[np.argmax(d)]
    return o

def compute_calib(corners, grid_size=9, cell_px=100):
    side=cell_px*grid_size
    dst=np.array([[0,0],[side,0],[side,side],[0,side]],dtype="float32")
    M=cv2.getPerspectiveTransform(corners,dst)
    return M

def calibrate_from_image(img):
    circles=detect_red_circles(img)
    if len(circles)<4: return None
    corners=select_board_corners(circles,img.shape)
    if corners is None: return None
    return compute_calib(order_points(corners))

# ===== 青三角の向き判定 =====
BLUE_LOWER=np.array([90,60,40]); BLUE_UPPER=np.array([130,255,255])

def detect_triangle_direction(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,BLUE_LOWER,BLUE_UPPER)
    k=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best=None; bs=-1
    for c in cnts:
        a=cv2.contourArea(c)
        if a<500: continue
        x,y,bw,bh=cv2.boundingRect(c)
        if max(bw,bh)/max(min(bw,bh),1)>2.5: continue
        ret,tri=cv2.minEnclosingTriangle(c)
        if tri is None: continue
        ta=cv2.contourArea(tri.reshape(-1,2).astype(np.float32))
        if ta<=0: continue
        t=a/ta
        if t>bs: bs=t; best=(c,tri)
    if best is None or bs<0.6: return None
    c,tri=best; pts=tri.reshape(-1,2)
    M=cv2.moments(c)
    if M["m00"]==0: return None
    cx,cy=M["m10"]/M["m00"],M["m01"]/M["m00"]
    apex=pts[int(np.argmax([np.hypot(p[0]-cx,p[1]-cy) for p in pts]))]
    dx,dy=apex[0]-cx,apex[1]-cy
    return ("right" if dx>0 else "left") if abs(dx)>abs(dy) else ("down" if dy>0 else "up")

# ===== 標準初期配置（先手が左・row0=9筋）=====
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

# 駒ラベル → 表示用の短い記号
DISP = {
    "empty":" ・ ",
    "sente_fu":"歩","sente_kyo":"香","sente_kei":"桂","sente_gin":"銀","sente_kin":"金",
    "sente_kaku":"角","sente_hi":"飛","sente_ou":"玉",
    "sente_tokin":"と","sente_nari_kyo":"杏","sente_nari_kei":"圭","sente_nari_gin":"全",
    "sente_uma":"馬","sente_ryu":"龍",
    "gote_fu":"歩","gote_kyo":"香","gote_kei":"桂","gote_gin":"銀","gote_kin":"金",
    "gote_kaku":"角","gote_hi":"飛","gote_ou":"玉",
    "gote_tokin":"と","gote_nari_kyo":"杏","gote_nari_kei":"圭","gote_nari_gin":"全",
    "gote_uma":"馬","gote_ryu":"龍",
}

def print_board_compare(recognized, reference):
    """認識結果と初期配置を並べて表示し、不一致マスを返す"""
    mismatches = []
    print("\n  認識した盤面（先頭S=先手, G=後手）:")
    print("     列0  列1  列2  列3  列4  列5  列6  列7  列8")
    for r in range(9):
        cells = []
        for c in range(9):
            lab = recognized[r][c]
            if lab == "empty":
                cells.append(" ・ ")
            else:
                pre = "S" if lab.startswith("sente_") else "G"
                cells.append(f"{pre}{DISP[lab]}")
            if lab != reference[r][c]:
                mismatches.append((r, c, reference[r][c], lab))
        print(f"  行{r} " + " ".join(f"{x:>3}" for x in cells))
    return mismatches

def save_grid_overlay(img, M, out_path, grid_size=9, cell_px=100):
    """キャリブレーション結果にグリッドを重ねた画像を保存（人間の目視用）"""
    side = cell_px * grid_size
    warped = cv2.warpPerspective(img, M, (side, side))
    vis = warped.copy()
    for i in range(grid_size+1):
        cv2.line(vis, (0,i*cell_px), (side,i*cell_px), (0,200,0), 2)
        cv2.line(vis, (i*cell_px,0), (i*cell_px,side), (0,200,0), 2)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", vis)[1].tofile(str(out_path))

def manual_calibration_flow(img, out_dir):
    """
    0手目の自動判定（向き判定／赤丸キャリブレーション／初期配置照合）のいずれかで
    エラーになった場合に、人間が盤の向きを指定し、四隅をクリックしてキャリブレーションする。
    この経路に入ったら撮り直しは要求せず、確定した結果をそのまま採用して⑥準備OKへ進む
    （自動検出に依存しない確実な救済手段として、calibrate.pyと同じ4隅クリック方式を
    本スクリプトに内包する）。
    Returns: (direction, M) 確定した画像回転方向と透視変換行列
    """
    print(f"\n  盤の向きを指定してください（画像回転方向）:")
    print(f"   [1] right  横向き・先手が左（回転なし）")
    print(f"   [2] left   横向き・先手が右（180度回転）")
    print(f"   [3] up     縦向き・先手が下（反時計90度）")
    print(f"   [4] down   縦向き・先手が上（時計90度）")
    dir_map = {"1": "right", "2": "left", "3": "up", "4": "down"}
    direction = None
    while direction is None:
        ori_input = input("  入力 [1-4]: ").strip()
        direction = dir_map.get(ori_input)
        if direction is None:
            print("  [WARNING] 1〜4で入力してください")

    img_rotated = rotate_image_for_direction(img, direction)

    h, w = img_rotated.shape[:2]
    max_disp = 1200
    scale = min(max_disp / w, max_disp / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)
    base_disp = cv2.resize(img_rotated, (disp_w, disp_h))
    display_img = base_disp.copy()
    clicked_points = []

    def _mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))
            cv2.circle(display_img, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(display_img, str(len(clicked_points)), (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Manual Calibration", display_img)
            print(f"  点{len(clicked_points)}: ({x},{y})")

    print(f"\n  画像ウィンドウが開きます。盤面の4隅を以下の順でクリックしてください：")
    print(f"   1: 左上  2: 右上  3: 右下  4: 左下（盤の外枠ではなく、マス目=赤丸の角）")
    print(f"   q: やり直し  Enter: 確定")
    cv2.namedWindow("Manual Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Manual Calibration", disp_w, disp_h)
    cv2.setMouseCallback("Manual Calibration", _mouse_cb)
    cv2.imshow("Manual Calibration", display_img)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            clicked_points.clear()
            display_img[:] = base_disp
            cv2.imshow("Manual Calibration", display_img)
            print("  [リセット] もう一度4隅をクリックしてください")
        elif key == 13:  # Enter
            if len(clicked_points) == 4:
                break
            print(f"  [WARNING] まだ{len(clicked_points)}点しか選択されていません（4点必要）")
    cv2.destroyAllWindows()

    src_points_orig = [(x/scale, y/scale) for (x, y) in clicked_points]
    ordered = order_points(src_points_orig)
    M = compute_calib(ordered)

    overlay_path = out_dir / "calib_check_manual.png"
    save_grid_overlay(img_rotated, M, overlay_path)
    print(f"\n  手動キャリブレーション完了。確認画像: {overlay_path}")

    return direction, M

# ===== グリッド変換 =====
def rot180(g): return [[g[8-r][8-c] for c in range(9)] for r in range(9)]
def rot90cw(g): return [[g[8-c][r] for c in range(9)] for r in range(9)]
def rot90ccw(g): return [[g[c][8-r] for c in range(9)] for r in range(9)]

def get_transforms(direction):
    # 頂点=up → 先手が下（縦向き） → 反時計90度で先手を左へ
    # 頂点=down → 先手が上（縦向き） → 時計90度で先手を左へ
    return {"right":[lambda g:g],"left":[rot180],
            "up":[rot90ccw],"down":[rot90cw]}.get(direction,[lambda g:g])

def rotate_image_for_direction(img, direction):
    """青三角の頂点方向に応じて画像を回転する。
    学習データは横向き（先手が左=right）基準。
    縦向き撮影は画像ごと回転して横向きに直す。
      right → そのまま
      left  → 180度回転
      up    → 反時計90度（先手が下→先手を左へ）
      down  → 時計90度（先手が上→先手を左へ）
    """
    if direction is None or direction == "right":
        return img
    elif direction == "left":
        return cv2.rotate(img, cv2.ROTATE_180)
    elif direction == "up":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == "down":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def apply_transforms(g, tl):
    if g is None: return None
    for t in tl: g=t(g)
    return g

# ===== 座標・差分・指し手 =====
def rc_to_square(row,col): return (8-col)*9+row

def square_to_rc(sq):
    col = 8 - (sq // 9)
    row = sq % 9
    return row, col

def board_to_label_grid(board):
    """shogi.Boardを9x9ラベルグリッドに変換する（detect_move.pyの同名関数と同じロジック）"""
    grid = [["empty"] * 9 for _ in range(9)]
    for sq in shogi.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        row, col = square_to_rc(sq)
        piece_name = PIECE_TO_LABEL.get(piece.piece_type)
        if piece_name is None:
            continue
        color_prefix = "sente" if piece.color == shogi.BLACK else "gote"
        grid[row][col] = f"{color_prefix}_{piece_name}"
    return grid

def diff_count(grid_a, grid_b):
    n = 0
    for r in range(9):
        for c in range(9):
            if grid_a[r][c] != grid_b[r][c]:
                n += 1
    return n

def _best_legal_sequences(board, recognized, depth):
    """boardからdepth手先まですべての合法手列を試し、recognizedとの差分が最小の手列を返す。
    Returns: (best_diff, [手列(shogi.Moveのリスト), ...])  手列はdepth個のMoveを含む。
    """
    if depth == 1:
        best_diff = None
        best_seqs = []
        for m1 in board.legal_moves:
            b1 = shogi.Board()
            b1.set_sfen(board.sfen())
            b1.push(m1)
            d = diff_count(board_to_label_grid(b1), recognized)
            if best_diff is None or d < best_diff:
                best_diff, best_seqs = d, [[m1]]
            elif d == best_diff:
                best_seqs.append([m1])
        return best_diff, best_seqs

    # depth == 2
    best_diff = None
    best_seqs = []
    for m1 in board.legal_moves:
        b1 = shogi.Board()
        b1.set_sfen(board.sfen())
        b1.push(m1)
        for m2 in b1.legal_moves:
            b2 = shogi.Board()
            b2.set_sfen(b1.sfen())
            b2.push(m2)
            d = diff_count(board_to_label_grid(b2), recognized)
            if best_diff is None or d < best_diff:
                best_diff, best_seqs = d, [[m1, m2]]
            elif d == best_diff:
                best_seqs.append([m1, m2])
    return best_diff, best_seqs

def classify_frame(board, recognized, move_threshold=MOVE_DIFF_THRESHOLD):
    """
    直前の写真ではなく、現在確定しているboardと今回の認識結果を比較し、
    それを説明できる合法手を合法手全体から総当たりで探す（detect_move.pyと同じロジック）。
    「指し手ではない見え方の変化」と「誤読された本当の指し手」は差分の大きさだけでは区別
    できないため、diff0が小さいことを理由に指し手なしと断定することはしない（=安全側に倒し、
    写真がboardと完全一致(diff0==0)の場合のみ指し手なしとする）。1手で説明できなければ、
    直前のフレームが認識ミスで取りこぼされた可能性を考慮して2手先まで総当たりする。

    Returns: (status, payload)
      status="move":     payload=採用したUSI文字列のリスト（1〜2手）
      status="nochange": payload=0（写真がboardと完全一致）
      status="error":    payload=デバッグ情報dict（現状の認識ミス扱い）
    """
    diff0 = diff_count(board_to_label_grid(board), recognized)
    if diff0 == 0:
        return "nochange", diff0

    for depth in (1, 2):
        best_diff, best_seqs = _best_legal_sequences(board, recognized, depth)
        if best_diff is not None and best_diff <= move_threshold:
            if len(best_seqs) == 1:
                return "move", [m.usi() for m in best_seqs[0]]
            return "error", {
                "reason": "ambiguous", "depth": depth, "diff0": diff0,
                "best_diff": best_diff,
                "candidates": [[m.usi() for m in seq] for seq in best_seqs],
            }

    return "error", {"reason": "no_match", "diff0": diff0}

# ===== KIF出力 =====
def save_kif(moves_usi, out_path):
    FILE_JA=['','１','２','３','４','５','６','７','８','９']
    RANK_JA=['','一','二','三','四','五','六','七','八','九']
    PIECE_JA={'p':'歩','l':'香','n':'桂','s':'銀','g':'金','b':'角','r':'飛','k':'玉',
              '+p':'と','+l':'成香','+n':'成桂','+s':'成銀','+b':'馬','+r':'龍'}
    fmap={'a':1,'b':2,'c':3,'d':4,'e':5,'f':6,'g':7,'h':8,'i':9}
    def ja(sqn):
        return f"{FILE_JA[int(sqn[0])]}{RANK_JA[fmap[sqn[1]]]}"
    lines=["手数----指手---------消費時間--"]
    board=shogi.Board()
    for i,usi in enumerate(moves_usi):
        try:
            b=shogi.Board(); b.set_sfen(board.sfen()); b.push_usi(usi)
            move=list(b.move_stack)[-1]
            dst_ja=ja(shogi.SQUARE_NAMES[move.to_square])
            if move.drop_piece_type:
                pj=PIECE_JA.get(shogi.PIECE_SYMBOLS[move.drop_piece_type],'?')
                mj=f"{dst_ja}{pj}打"
            else:
                src=shogi.SQUARE_NAMES[move.from_square]
                pc=board.piece_at(move.from_square)
                pj=PIECE_JA.get(shogi.PIECE_SYMBOLS[pc.piece_type] if pc else '?','?')
                pr="成" if move.promotion else ""
                mj=f"{dst_ja}{pj}{pr}({src[0]}{fmap[src[1]]})"
            lines.append(f"{i+1:4d} {mj} (00:00/00:00:00)")
            board.push(move)
        except Exception as e:
            lines.append(f"{i+1:4d} (ERROR: {e})")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))

# ===== メインループ =====
def main():
    ap=argparse.ArgumentParser(description="本番: 監視→リアルタイムKIF生成")
    ap.add_argument("--watch",required=True,help="監視フォルダ")
    ap.add_argument("--model",required=True)
    ap.add_argument("--out",required=True,help="出力フォルダ")
    ap.add_argument("--idle-timeout",type=int,default=600,help="無更新で終了とみなす秒数")
    ap.add_argument("--poll-interval",type=int,default=5,help="チェック間隔秒")
    ap.add_argument("--pattern",default="",help="対象ファイル接頭辞（空なら全jpg）")
    args=ap.parse_args()

    watch=Path(args.watch); out=Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("モデル読み込み中...")
    model=load_model(args.model)

    print(f"監視開始: {watch}")
    print(f"（{args.idle_timeout}秒 新規画像がなければ対局終了とみなします）")

    processed=set()
    calib_M=None
    img_direction_confirmed="right"  # 確定した画像回転方向（デフォルト横向き）
    board=shogi.Board()
    moves_usi=[]
    last_new=time.time()
    move_count=0

    def list_images():
        files=sorted(set(list(watch.glob(f"{args.pattern}*.jpg"))+
                         list(watch.glob(f"{args.pattern}*.JPG"))))
        return files

    while True:
        files=list_images()
        new_files=[f for f in files if f.name not in processed]

        if new_files:
            last_new=time.time()
            for img_path in new_files:
                img=load_image(img_path)
                if img is None:
                    print(f"  [SKIP] 読込失敗: {img_path.name}")
                    processed.add(img_path.name); continue

                # 0手目（対局準備）: 向き判定→赤丸キャリブレーション→初期配置照合を自動で試す。
                # いずれかでエラーが出たら、その場で人間が向き指定+四隅クリックを行い、
                # 結果を無条件に採用して準備OKへ進む（「撮り直してください」という自動リトライ
                # ループは行わない方針 — エラーになったら即・人間が直す、の一回のみ）。
                if calib_M is None:
                    error_reason = None
                    img_direction = None
                    img_rotated = None
                    cand_M = None

                    # Step3: 青三角で向き判定
                    tri = detect_triangle_direction(img)
                    if tri:
                        img_direction = tri
                        print(f"  向き判定: 青三角頂点={tri}")
                    else:
                        error_reason = "向き判定エラー（青三角を検出できませんでした）"

                    # Step4: 赤丸でキャリブレーション
                    if error_reason is None:
                        img_rotated = rotate_image_for_direction(img, img_direction)
                        rot_desc = {
                            "right": "回転なし",
                            "left":  "180度回転",
                            "up":    "反時計90度回転（先手が下→左へ）",
                            "down":  "時計90度回転（先手が上→左へ）",
                        }.get(img_direction, "回転なし")
                        print(f"  画像回転: {rot_desc}")

                        cand_M = calibrate_from_image(img_rotated)
                        if cand_M is None:
                            error_reason = "キャリブレーションエラー（赤丸4個を検出できませんでした）"

                    # Step5: 初期配置と照合
                    if error_reason is None:
                        overlay_path = out / "calib_check.png"
                        save_grid_overlay(img_rotated, cand_M, overlay_path)
                        print(f"  キャリブレーション確認画像: {overlay_path}")

                        warped = cv2.warpPerspective(img_rotated, cand_M, (900,900))
                        labels = predict_board(model, warped)
                        mismatches = print_board_compare(labels, INITIAL_BOARD_STD)
                        if mismatches:
                            print(f"\n  [!] 初期配置と異なるマスが {len(mismatches)} 箇所あります:")
                            for r, c, expected, got in mismatches:
                                print(f"      行{r}列{c}: 期待={expected} / 認識={got}")
                            error_reason = "初期配置照合エラー（認識結果が将棋の初期配置と一致しませんでした）"

                    if error_reason is not None:
                        print(f"\n  [エラー] {error_reason}")
                        print(f"  -> 人間が盤の向き指定と四隅キャリブレーションを行います。")
                        img_direction, cand_M = manual_calibration_flow(img, out)
                    else:
                        print(f"\n  [OK] 自動判定すべて成功（向き判定・赤丸キャリブレーション・初期配置照合）。")

                    # Step6: 準備OK（自動成功でも手動キャリブレーションでも、ここで無条件に確定する）
                    calib_M = cand_M
                    img_direction_confirmed = img_direction
                    processed.add(img_path.name)
                    print(f"\n{'='*55}")
                    print(f"  [準備OK] 初期局面の準備が整いました。")
                    print(f"{'='*55}")
                    input("  対局を開始する準備ができたらEnterを押してください（対局開始ボタンの代わり）: ")
                    print(f"\n  [対局開始] 以降の手を記録します。\n")
                    continue

                # 各手: 画像を同じ方向に回転してから認識し、boardと比較して指し手を判定
                # （直前の写真ではなくboard=確定済み盤面と比較するため、駒のずれ直しや
                #  照明変化のようなノイズが乗っても、合法手としての説明力が最良なら採用できる）
                img_rotated = rotate_image_for_direction(img, img_direction_confirmed)
                warped=cv2.warpPerspective(img_rotated,calib_M,(900,900))
                labels=predict_board(model,warped)
                status, payload = classify_frame(board, labels)
                if status=="move":
                    usi_seq=payload
                    for mv in usi_seq:
                        moves_usi.append(mv); board.push_usi(mv)
                        move_count+=1
                    tag = usi_seq[-1] if len(usi_seq)==1 else f"{usi_seq} (2手分まとめて復元)"
                    print(f"  [{move_count}手目] {tag}  ({datetime.now():%H:%M:%S})")
                    # KIF逐次更新
                    save_kif(moves_usi, out/"live.kif")
                elif status=="nochange":
                    print(f"  [補正] 駒のずれ/照明変化と判断、手は記録しません (diff={payload})")
                else:
                    print(f"  [認識ミス] 手を確定できず（保留）: {payload}")
                processed.add(img_path.name)

        # 対局終了判定
        idle=time.time()-last_new
        if calib_M is not None and idle>args.idle_timeout and moves_usi:
            ts=datetime.now().strftime("%Y%m%d_%H%M%S")
            final=out/f"game_{ts}.kif"
            save_kif(moves_usi, final)
            print(f"\n対局終了（{args.idle_timeout}秒 無更新）")
            print(f"確定KIF: {final}（{len(moves_usi)}手）")
            break

        time.sleep(args.poll_interval)

if __name__=="__main__":
    main()
