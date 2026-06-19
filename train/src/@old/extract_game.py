"""
対局画像 + KIF から全手の盤面を学習データに切り出すスクリプト

使い方:
  python extract_game.py \
    --images  "//YukiYoshiNAS/.../train/data/対局v2_rotated/001/" \
    --kif     "//YukiYoshiNAS/.../train/data/対局v2/001/001.kif" \
    --calib   "//YukiYoshiNAS/.../train/src/対局v2_calib/001_calib.json" \
    --out     "//YukiYoshiNAS/.../train/data/cells/"

前提:
  - 画像は「横向き（先手が左）」に回転済みであること
  - calibは回転後の画像で取得済みであること
  - 画像はファイル名順に 0手目・1手目・2手目… と対応している
  - KIFは shogi ライブラリで読める形式であること

注意:
  - 0手目（初期配置）は extract_initial.py でも切り出せる
  - このスクリプトは0手目も含めて全手を処理する
"""

import cv2
import numpy as np
import json
import argparse
import sys
import re
from pathlib import Path

import shogi
import shogi.KIF

# -------------------------------------------------------
# ラベル変換
# -------------------------------------------------------
PIECE_LABEL = {
    (shogi.PAWN,         shogi.BLACK): "sente_fu",
    (shogi.LANCE,        shogi.BLACK): "sente_kyo",
    (shogi.KNIGHT,       shogi.BLACK): "sente_kei",
    (shogi.SILVER,       shogi.BLACK): "sente_gin",
    (shogi.GOLD,         shogi.BLACK): "sente_kin",
    (shogi.BISHOP,       shogi.BLACK): "sente_kaku",
    (shogi.ROOK,         shogi.BLACK): "sente_hi",
    (shogi.KING,         shogi.BLACK): "sente_ou",
    (shogi.PROM_PAWN,    shogi.BLACK): "sente_tokin",
    (shogi.PROM_LANCE,   shogi.BLACK): "sente_nari_kyo",
    (shogi.PROM_KNIGHT,  shogi.BLACK): "sente_nari_kei",
    (shogi.PROM_SILVER,  shogi.BLACK): "sente_nari_gin",
    (shogi.PROM_BISHOP,  shogi.BLACK): "sente_uma",
    (shogi.PROM_ROOK,    shogi.BLACK): "sente_ryu",
    (shogi.PAWN,         shogi.WHITE): "gote_fu",
    (shogi.LANCE,        shogi.WHITE): "gote_kyo",
    (shogi.KNIGHT,       shogi.WHITE): "gote_kei",
    (shogi.SILVER,       shogi.WHITE): "gote_gin",
    (shogi.GOLD,         shogi.WHITE): "gote_kin",
    (shogi.BISHOP,       shogi.WHITE): "gote_kaku",
    (shogi.ROOK,         shogi.WHITE): "gote_hi",
    (shogi.KING,         shogi.WHITE): "gote_ou",
    (shogi.PROM_PAWN,    shogi.WHITE): "gote_tokin",
    (shogi.PROM_LANCE,   shogi.WHITE): "gote_nari_kyo",
    (shogi.PROM_KNIGHT,  shogi.WHITE): "gote_nari_kei",
    (shogi.PROM_SILVER,  shogi.WHITE): "gote_nari_gin",
    (shogi.PROM_BISHOP,  shogi.WHITE): "gote_uma",
    (shogi.PROM_ROOK,    shogi.WHITE): "gote_ryu",
}

def board_to_label_grid(board):
    """
    shogi.Board → 9x9 ラベルグリッド
    座標系: row=0が左端(先手1段目=9筋側)、col=0が上(9段目)
    square = (8-col)*9 + row
    """
    grid = [["empty"] * 9 for _ in range(9)]
    for col in range(9):       # col: 0=上(9段目) 〜 8=下(1段目)
        for row in range(9):   # row: 0=左(先手1段目) 〜 8=右(後手1段目)
            sq = (8 - col) * 9 + row
            piece = board.piece_at(sq)
            if piece is not None:
                label = PIECE_LABEL.get((piece.piece_type, piece.color), "empty")
                grid[row][col] = label
    return grid

def load_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

def load_calibration(calib_path):
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
    M = np.array(calib["perspective_matrix"], dtype=np.float64)
    grid_size = calib.get("grid_size", 9)
    cell_px   = calib.get("cell_px", 100)
    return M, grid_size, cell_px

def extract_cells(img, M, grid_size=9, cell_px=100):
    side = cell_px * grid_size
    warped = cv2.warpPerspective(img, M, (side, side))
    cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * cell_px
            y1 = row * cell_px
            cell = warped[y1:y1+cell_px, x1:x1+cell_px]
            cells.append((row, col, cell))
    return warped, cells

def read_kif_moves(kif_path):
    """KIFを読んでUSI形式の手リストを返す"""
    # cp932で読み込み
    with open(kif_path, "r", encoding="cp932", errors="replace") as f:
        content = f.read()
    kif_data = shogi.KIF.Parser.parse_str(content)
    return kif_data["moves"]  # USI文字列のリスト

def main():
    parser = argparse.ArgumentParser(description="対局画像+KIFから全手の盤面を学習データに切り出す")
    parser.add_argument("--images",  required=True, help="対局画像フォルダ（回転済み）")
    parser.add_argument("--kif",     required=True, help="KIFファイルパス")
    parser.add_argument("--calib",   required=True, help="キャリブレーションJSONパス（回転後画像で取得済み）")
    parser.add_argument("--out",     required=True, help="出力先 cells/ フォルダ")
    parser.add_argument("--pattern", default="IMG_", help="画像ファイルパターン（デフォルト=IMG_）")
    parser.add_argument("--prefix",  default="", help="出力ファイル名のプレフィックス（例: game001_）")
    args = parser.parse_args()

    images_folder = Path(args.images)
    out_folder    = Path(args.out)

    # 画像一覧（ファイル名順）
    image_files = sorted(set(
        list(images_folder.glob(f"{args.pattern}*.jpg")) +
        list(images_folder.glob(f"{args.pattern}*.JPG"))
    ))
    if not image_files:
        print(f"[ERROR] 画像が見つかりません: {images_folder}")
        sys.exit(1)

    # KIF読み込み
    print(f"KIF読み込み: {args.kif}")
    try:
        moves_usi = read_kif_moves(args.kif)
    except Exception as e:
        print(f"[ERROR] KIF読み込み失敗: {e}")
        sys.exit(1)
    print(f"  手数: {len(moves_usi)} 手")

    # キャリブレーション読み込み
    M, grid_size, cell_px = load_calibration(args.calib)

    # 画像枚数と手数の確認
    # 画像: 0手目〜N手目 = N+1枚
    # KIF: N手分の指し手
    expected_images = len(moves_usi) + 1
    print(f"画像: {len(image_files)} 枚 / 期待値: {expected_images} 枚（{len(moves_usi)}手+0手目）")
    if len(image_files) < expected_images:
        print(f"  [WARNING] 画像が少ない。{len(image_files)}枚まで処理します。")
    elif len(image_files) > expected_images:
        print(f"  [WARNING] 画像が多い。最初の{expected_images}枚のみ処理します。")
        image_files = image_files[:expected_images]

    print()

    # 盤面を追跡しながら画像を切り出す
    board = shogi.Board()
    total = 0
    skipped = 0
    prefix = args.prefix

    for move_num, img_path in enumerate(image_files):
        # 現在の盤面をラベルグリッドに変換
        label_grid = board_to_label_grid(board)

        # 画像読み込み
        img = load_image(img_path)
        if img is None:
            print(f"  [{move_num}手目] [SKIP] 読み込み失敗: {img_path.name}")
            skipped += 1
            # 手を進める
            if move_num < len(moves_usi):
                try:
                    board.push_usi(moves_usi[move_num])
                except Exception:
                    pass
            continue

        # マス切り出し
        _, cells = extract_cells(img, M, grid_size, cell_px)
        saved = 0
        for row, col, cell_img in cells:
            label = label_grid[row][col]
            filename = f"{prefix}move{move_num:03d}_{img_path.stem}_r{row}c{col}.jpg"
            save_path = out_folder / label / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".jpg", cell_img)[1].tofile(str(save_path))
            saved += 1

        total += saved
        # 非emptyのマス数を表示
        non_empty = sum(1 for row in label_grid for lbl in row if lbl != "empty")
        print(f"  [{move_num:3d}手目] {img_path.name}: {saved}マス保存（駒{non_empty}個）")

        # 次の手を進める
        if move_num < len(moves_usi):
            try:
                board.push_usi(moves_usi[move_num])
            except Exception as e:
                print(f"    [WARNING] 手を進められません: {moves_usi[move_num]} ({e})")

    print()
    print(f"完了: {total} マス保存（スキップ {skipped} 枚）")

    # クラス別サマリ
    print()
    print("cells/ 全体のクラス別サンプル数:")
    counts = {}
    for d in sorted(out_folder.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.jpg")))
            counts[d.name] = n
    for k, v in sorted(counts.items()):
        if k != "empty":
            print(f"  {k}: {v}")
    print(f"  empty: {counts.get('empty', 0)}")
    print(f"  合計: {sum(counts.values())} / {len(counts)} クラス")

if __name__ == "__main__":
    main()
