"""
KIFファイル + 対局画像フォルダから学習データ(cells)を自動生成するツール

考え方:
  1. KIFを読み込み、各手を進めながら shogi.Board を更新していく
  2. 対局画像フォルダの画像を時刻順に並べ、既存モデルで認識して差分から「これが何手目の後の画像か」を推定する
     （detect_move.pyと同じロジックで画像同士の差分を取り、KIFの指し手と一致するかを検証）
  3. KIFの指し手と画像の差分が一致した画像だけを「正解ラベル付き」として採用し、
     その手数の後の正しい盤面(KIFから計算した正解)を9x9ラベルグリッドに変換してcellsに保存する
  4. 不一致の画像はスキップしてログに残す（人間が後で確認できるように）

前提:
  - 画像は学習データと同じ向き（先手が左/normal）に回転済みであること
    （本ツールでは画像回転は行わない。事前にユーザー側で回転しておく）
  - calibは対局フォルダごとに1つ（カメラを動かしていない前提）。手動キャリブレーションを推奨

使い方:
  python label_from_kif.py \
    --kif    "//.../001.kif" \
    --images "//.../001/" \
    --calib  "//.../001_calib.json" \
    --model  "//.../train/models/" \
    --out    "//.../train/data/cells/" \
    --pattern "IMG_"

出力:
  --out フォルダ配下の各ラベルディレクトリに、
  "{KIF名}_move{N}_r{row}c{col}.jpg" という名前でマス画像を保存する
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import shogi
import shogi.KIF


LABEL_TO_PIECE = {
    "fu": shogi.PAWN, "kyo": shogi.LANCE, "kei": shogi.KNIGHT,
    "gin": shogi.SILVER, "kin": shogi.GOLD, "kaku": shogi.BISHOP,
    "hi": shogi.ROOK, "ou": shogi.KING,
    "tokin": shogi.PROM_PAWN, "nari_kyo": shogi.PROM_LANCE,
    "nari_kei": shogi.PROM_KNIGHT, "nari_gin": shogi.PROM_SILVER,
    "uma": shogi.PROM_BISHOP, "ryu": shogi.PROM_ROOK,
}
PIECE_TO_LABEL = {v: k for k, v in LABEL_TO_PIECE.items()}


# -------------------------------------------------------
# 画像・キャリブレーション関連（detect_move.pyと同じロジック）
# -------------------------------------------------------
def load_image(path):
    """日本語パス対応の画像読み込み"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def load_calib(calib_path):
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    return calib


def warp_image(img, calib):
    """calibの透視変換行列を使って画像を正方形(9*cell_px角)に変換する"""
    M = np.array(calib["perspective_matrix"], dtype=np.float64)
    cell_px = calib.get("cell_px", 100)
    grid_size = calib.get("grid_size", 9)
    side = cell_px * grid_size
    warped = cv2.warpPerspective(img, M, (side, side))
    return warped, cell_px, grid_size


def load_model(model_folder):
    """既存モデルの読み込み（train_model.pyと同じMobileNetV2構造。predict_cell.pyと揃える）"""
    import torch
    import torch.nn as nn
    import torchvision.models as models

    model_folder = Path(model_folder)
    meta_path = model_folder / "model_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    classes = meta["labels"]

    net = models.mobilenet_v2(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, len(classes)),
    )
    state = torch.load(model_folder / "best_model.pth", map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    return net, classes


def predict_board(model_tuple, warped, cell_px=100, grid_size=9):
    """切り出した9x9マスをモデルで認識してラベルグリッドを返す（簡易版・確信度なし）"""
    import torch
    from torchvision import transforms
    from PIL import Image

    net, classes = model_tuple
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    grid = [["empty"] * grid_size for _ in range(grid_size)]
    for r in range(grid_size):
        for c in range(grid_size):
            cell = warped[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px]
            cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cell_rgb)
            tensor = tfm(pil_img).unsqueeze(0)
            with torch.no_grad():
                out = net(tensor)
                pred = out.argmax(dim=1).item()
            grid[r][c] = classes[pred]
    return grid


# -------------------------------------------------------
# shogi.Board <-> 9x9ラベルグリッド変換
# -------------------------------------------------------
def rc_to_square(row, col):
    """(row, col) [row=0が上(後手側=9筋), col=0が左(先手側1段目)] -> shogiのsquare index"""
    return (8 - col) * 9 + row


def square_to_rc(sq):
    col = 8 - (sq // 9)
    row = sq % 9
    return row, col


def board_to_label_grid(board):
    """shogi.Boardオブジェクトを9x9ラベルグリッドに変換する"""
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


def grid_to_set(grid):
    """ラベルグリッドを {(row,col): label} の集合に変換（emptyは除く）"""
    result = {}
    for r in range(9):
        for c in range(9):
            if grid[r][c] != "empty":
                result[(r, c)] = grid[r][c]
    return result


def diff_count(grid_a, grid_b):
    """2つのラベルグリッドの相違マス数を数える"""
    n = 0
    for r in range(9):
        for c in range(9):
            if grid_a[r][c] != grid_b[r][c]:
                n += 1
    return n


# -------------------------------------------------------
# KIF読み込み・全局面の事前計算
# -------------------------------------------------------

# 漢数字 -> 半角数字（KIFの指し手表記用）
KANJI_NUM = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}

PIECE_KANJI_TO_LABEL = {
    "歩":"fu","香":"kyo","桂":"kei","銀":"gin","金":"kin","角":"kaku","飛":"hi","玉":"ou","王":"ou",
    "と":"tokin","成香":"nari_kyo","成桂":"nari_kei","成銀":"nari_gin","馬":"uma","龍":"ryu","竜":"ryu",
}


def parse_kif_moves_fallback(kif_text):
    """
    shogi.KIF.Parserが使えない場合の簡易フォールバックパーサー。
    「N 漢数字漢数字駒(元の位置) ...」または「N 同　駒(元の位置) ...」または「N 駒打 ...」の行から
    指し手を抽出し、USI形式に変換しようとする。
    ただし完全な変換は複雑なため、ここでは (to_row, to_col, piece_label, promote, from_sq_or_None, is_drop) の
    中間形式を返す。mainの処理側でshogi.Boardに対してこの情報から合法手を探索して適用する。
    """
    moves = []
    last_to = None
    for line in kif_text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+\s+(.*)$", line)
        if not m:
            continue
        token = m.group(1)
        # 末尾の時間情報 "(00:00/00:00:00)" や "(0000000000)" を除去
        token = re.sub(r"\s*\([0-9:/]+\)\s*$", "", token).strip()
        if not token:
            continue
        if token in ("投了", "中断", "千日手", "持将棋"):
            break

        # 移動元座標の括弧 "(77)" を除いた「駒名表記部分」でpromote/is_dropを判定する
        # （末尾の括弧が残っていると "同　角成(77)" のように成の後に括弧が来て誤判定するため）
        name_part = re.sub(r"\([0-9]+\)$", "", token).strip()
        promote = name_part.endswith("成") and not name_part.startswith("成")
        is_drop = name_part.endswith("打")

        # 移動先座標
        if token.startswith("同"):
            if last_to is None:
                break
            to_col, to_row = last_to
            rest = token[1:].replace("　", "").strip()
        else:
            zm = re.match(r"^([１-９1-9])([一二三四五六七八九])", token)
            if not zm:
                break
            col_ch, row_kanji = zm.group(1), zm.group(2)
            zen_to_han = {"１":1,"２":2,"３":3,"４":4,"５":5,"６":6,"７":7,"８":8,"９":9}
            to_col = zen_to_han.get(col_ch, None)
            if to_col is None:
                try:
                    to_col = int(col_ch)
                except ValueError:
                    break
            to_row = KANJI_NUM.get(row_kanji)
            rest = token[2:]
            last_to = (to_col, to_row)

        # 駒名抽出（2文字駒名を先にチェック）
        piece_label = None
        for kanji, label in sorted(PIECE_KANJI_TO_LABEL.items(), key=lambda x: -len(x[0])):
            if rest.startswith(kanji):
                piece_label = label
                break
        if piece_label is None:
            break

        # 移動元 (from_col, from_row) または None（打つ手の場合）
        from_sq = None
        fm = re.search(r"\((\d)(\d)\)", token)
        if fm and not is_drop:
            from_col, from_row = int(fm.group(1)), int(fm.group(2))
            from_sq = (from_col, from_row)

        moves.append({
            "to": (to_col, to_row),
            "from": from_sq,
            "piece_label": piece_label,
            "promote": promote,
            "is_drop": is_drop,
        })

    return moves


def kifcoord_to_square(col, row):
    """KIF座標(col=1-9筋, row=1-9段[一=1...九=9]) -> shogi square index"""
    # board_to_label_gridのrc_to_square系に合わせる: row(0-8, 0が上=後手側) col(0-8, 0が左=先手側)
    # KIFのcol(筋)は9筋が左(先手から見て一番左)ではなく1筋が右。将棋の筋は右から1,2,...9。
    # 本ツールのcol=0は先手側(画像の左)。先手から見て9筋が一番左なので col=0 <-> 筋9, col=8 <-> 筋1
    grid_col = 9 - col
    grid_row = row - 1
    return rc_to_square(grid_row, grid_col)


def apply_fallback_move(board, mv):
    """parse_kif_moves_fallbackで得た中間形式の手をboardに適用する。合法手を総当たりで探す。"""
    to_sq = kifcoord_to_square(*mv["to"])
    piece_type_base = LABEL_TO_PIECE.get(mv["piece_label"]) if not mv["promote"] else None

    for legal in board.legal_moves:
        if legal.to_square != to_sq:
            continue
        if mv["is_drop"]:
            if legal.from_square is not None:
                continue
            if legal.drop_piece_type != LABEL_TO_PIECE.get(mv["piece_label"]):
                continue
        else:
            if legal.from_square is None:
                continue
            if mv["from"] is not None:
                from_sq = kifcoord_to_square(*mv["from"])
                if legal.from_square != from_sq:
                    continue
            if mv["promote"] != bool(legal.promotion):
                continue
        board.push(legal)
        return True
    return False


def load_kif_boards_fallback(kif_text):
    moves = parse_kif_moves_fallback(kif_text)
    board = shogi.Board()
    grids = [board_to_label_grid(board)]
    for mv in moves:
        ok = apply_fallback_move(board, mv)
        if not ok:
            print(f"  [WARNING] フォールバックパーサーで手を適用できませんでした: {mv} -> ここで打ち切り")
            break
        grids.append(board_to_label_grid(board))
    return grids


def load_kif_boards(kif_path):
    """
    KIFファイルを読み込み、0手目(初期局面)から最終手まで、
    各手数ごとのラベルグリッドのリストを返す。
    まずshogi.KIF.Parserを試し、失敗したら自前の簡易パーサーにフォールバックする。
    Returns: [grid_0(初期局面), grid_1(1手目後), grid_2(2手目後), ...]
    """
    with open(kif_path, "r", encoding="utf-8") as f:
        kif_text = f.read()

    try:
        parsed = shogi.KIF.Parser.parse_str(kif_text)[0]
        moves_usi = parsed["moves"]
        board = shogi.Board()
        grids = [board_to_label_grid(board)]
        for usi_move in moves_usi:
            move = shogi.Move.from_usi(usi_move)
            if move not in board.legal_moves:
                print(f"  [WARNING] KIF中の手が非合法と判定されました: {usi_move}（ここで打ち切り）")
                break
            board.push(move)
            grids.append(board_to_label_grid(board))
        if len(grids) > 1:
            print("  shogi.KIF.Parser で読み込みました")
            return grids
        raise ValueError("0手しか読めなかったためフォールバックします")
    except Exception as e:
        print(f"  [INFO] shogi.KIF.Parser失敗（{e}）。簡易フォールバックパーサーを使用します")
        return load_kif_boards_fallback(kif_text)


# -------------------------------------------------------
# メイン処理
# -------------------------------------------------------
def natural_key(path):
    """ファイル名中の数字で自然順ソート"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def main():
    parser = argparse.ArgumentParser(description="KIF+対局画像から学習データを自動生成")
    parser.add_argument("--kif", required=True, help="KIFファイルパス")
    parser.add_argument("--images", required=True, help="対局画像フォルダ")
    parser.add_argument("--calib", required=True, help="キャリブレーションJSON（1個、カメラ固定前提）")
    parser.add_argument("--model", required=True, help="モデルフォルダ")
    parser.add_argument("--out", required=True, help="出力先cellsフォルダ")
    parser.add_argument("--pattern", default="", help="画像ファイル名の接頭辞フィルタ")
    parser.add_argument("--max-diff-accept", type=int, default=2,
                         help="認識結果とKIF正解の差分マス数がこの値以下なら採用（デフォルト2）")
    parser.add_argument("--dry-run", action="store_true", help="実際には保存せず、対応関係の確認だけ行う")
    args = parser.parse_args()

    print(f"KIF読み込み中: {args.kif}")
    kif_grids = load_kif_boards(args.kif)
    print(f"  KIFから {len(kif_grids)} 局面（0手目〜{len(kif_grids)-1}手目）を再現しました")

    print(f"モデル読み込み中: {args.model}")
    model_tuple = load_model(args.model)

    calib = load_calib(args.calib)

    images_dir = Path(args.images)
    image_files = sorted(
        [p for p in images_dir.glob("*.jpg") if p.name.startswith(args.pattern)],
        key=natural_key,
    )
    print(f"対象画像: {len(image_files)} 枚")

    out_dir = Path(args.out)
    kif_stem = Path(args.kif).stem

    # 各画像を認識し、KIFのどの局面に最も近いかを探索する。
    # 画像は時系列順なので、KIF局面インデックスも単調増加するという制約を使い、
    # 直前にマッチした局面インデックス以降のみを探索範囲とする（高速化と誤マッチ防止）。
    search_start = 0
    accepted = 0
    skipped = 0
    log_lines = []

    for i, img_path in enumerate(image_files):
        img = load_image(img_path)
        if img is None:
            print(f"  [{i+1}/{len(image_files)}] [ERROR] 読み込み失敗: {img_path.name}")
            continue

        warped, cell_px, grid_size = warp_image(img, calib)
        recognized = predict_board(model_tuple, warped, cell_px, grid_size)

        # search_start以降のKIF局面の中から最も差分の少ない局面を探す
        # （探索窓を広めに取り、誤検出を防ぐ）
        best_idx = None
        best_diff = None
        search_end = min(search_start + 15, len(kif_grids))  # 一度に進みすぎないよう窓を制限
        for idx in range(search_start, search_end):
            d = diff_count(recognized, kif_grids[idx])
            if best_diff is None or d < best_diff:
                best_diff = d
                best_idx = idx

        status = "採用" if best_diff is not None and best_diff <= args.max_diff_accept else "スキップ"
        print(f"  [{i+1}/{len(image_files)}] {img_path.name} -> KIF{best_idx}手目 (差分{best_diff}マス) [{status}]")
        log_lines.append(f"{img_path.name}\tKIF{best_idx}\tdiff={best_diff}\t{status}")

        if status == "採用":
            accepted += 1
            search_start = best_idx  # 次の探索はここから（時系列の単調性を利用）

            if not args.dry_run:
                correct_grid = kif_grids[best_idx]
                for r in range(grid_size):
                    for c in range(grid_size):
                        label = correct_grid[r][c]
                        cell_img = warped[r*cell_px:(r+1)*cell_px, c*cell_px:(c+1)*cell_px]
                        save_dir = out_dir / label
                        save_dir.mkdir(parents=True, exist_ok=True)
                        fname = f"{kif_stem}_move{best_idx}_r{r}c{c}.jpg"
                        save_path = save_dir / fname
                        cv2.imencode(".jpg", cell_img)[1].tofile(str(save_path))
        else:
            skipped += 1

    print(f"\n{'='*55}")
    print(f"完了: 採用 {accepted}枚 / スキップ {skipped}枚 / 全{len(image_files)}枚")
    if args.dry_run:
        print("（--dry-run のため実際の保存は行っていません）")
    print(f"{'='*55}")

    log_path = out_dir / f"{kif_stem}_label_from_kif_log.txt"
    if not args.dry_run:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))
        print(f"ログ保存: {log_path}")


if __name__ == "__main__":
    main()
