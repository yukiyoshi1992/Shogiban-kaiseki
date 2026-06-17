"""
終盤局面マス切り出しツール
- data2～data8のKIFを読んで最終盤面を復元
- 対応する画像 + _calib.json でマスを切り出し
- train/data/cells/{ラベル}/ に保存（data1のcellsに追加）

使い方:
  python extract_endgame.py \
    --images "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/" \
    --calibs "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/" \
    --out    "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/cells/"

  # 特定のdataだけ処理する場合
  python extract_endgame.py ... --pattern "data2"
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
import shogi
import shogi.KIF

# -------------------------------------------------------
# 駒の定数 → ラベル文字列への変換
# python-shogiの駒定数:
#   PAWN=1 LANCE=2 KNIGHT=3 SILVER=4 GOLD=5 BISHOP=6 ROOK=7 KING=8
#   成り駒は +8: PPAWN=9 PLANCE=10 PKNIGHT=11 PSILVER=12 PBISHOP=14 PROOK=15
# -------------------------------------------------------
PIECE_TO_LABEL = {
    shogi.PAWN:        "fu",
    shogi.LANCE:       "kyo",
    shogi.KNIGHT:      "kei",
    shogi.SILVER:      "gin",
    shogi.GOLD:        "kin",
    shogi.BISHOP:      "kaku",
    shogi.ROOK:        "hi",
    shogi.KING:        "ou",
    shogi.PROM_PAWN:   "tokin",
    shogi.PROM_LANCE:  "nari_kyo",
    shogi.PROM_KNIGHT: "nari_kei",
    shogi.PROM_SILVER: "nari_gin",
    shogi.PROM_BISHOP: "uma",
    shogi.PROM_ROOK:   "ryu",
}

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

def board_to_label_grid(board):
    """
    python-shogiのBoardオブジェクトから9x9ラベルグリッドを生成

    python-shogiの座標系:
      square = file * 9 + rank  (file: 9～1筋=0～8, rank: 一～九段=0～8)
      board.piece_at(square) -> (piece_type, color) or None

    画像の座標系:
      col: 0=左=先手1段目(rank=0) ～ 8=右=後手9段目(rank=8)
      row: 0=上=9筋(file=0)       ～ 8=下=1筋(file=8)

    対応:
      col = rank  (段)
      row = file  (筋、ただしpython-shogiはfile=0が9筋)
    """
    grid = [["empty"] * 9 for _ in range(9)]

    # 座標対応（実画像との照合で確定）:
    #   先手の馬(9筋1段)=square0 → 画像の右上(row0,col8)
    #   先手の玉(6筋9段)=square75 → 画像の左・上から4行目(row3,col0)
    # SQUARE_NAMES: '9a'=sq0, '1a'=sq8, '9i'=sq72, '1i'=sq80
    #   row = square % 9        (筋: row0=9筋 ... row8=1筋)
    #   col = 8 - (square // 9) (段: col0=9段 ... col8=1段)
    for sq in range(81):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        row = sq % 9
        col = 8 - (sq // 9)

        piece_type = piece.piece_type
        color = piece.color  # shogi.BLACK=0=先手, shogi.WHITE=1=後手

        piece_label = PIECE_TO_LABEL.get(piece_type)
        if piece_label is None:
            print(f"  [WARNING] unknown piece type: {piece_type} at sq={sq}")
            continue

        side = "sente" if color == shogi.BLACK else "gote"
        grid[row][col] = f"{side}_{piece_label}"

    return grid

def load_kif(kif_path):
    """KIFを読んで最終盤面のBoardを返す"""
    kif = shogi.KIF.Parser.parse_file(str(kif_path))
    # parse_fileはリストを返す。kif[0]が対局データ
    game = kif[0]
    board = shogi.Board()
    for move_str in game['moves']:
        board.push_usi(move_str)
    return board

def load_calibration(calib_path):
    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    M = np.array(data["perspective_matrix"], dtype="float32")
    grid_size = data.get("grid_size", 9)
    cell_px   = data.get("cell_px", 100)
    return M, grid_size, cell_px

def load_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

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

def find_calib(calibs_folder, img_stem):
    calib_path = Path(calibs_folder) / f"{img_stem}_calib.json"
    return calib_path if calib_path.exists() else None

def find_kif(images_folder, data_name):
    """data2.jpg → data2.kif を探す"""
    kif_path = Path(images_folder) / f"{data_name}.kif"
    return kif_path if kif_path.exists() else None

def get_data_name(img_stem):
    """'data4 (2)' → 'data4'"""
    return img_stem.split(" ")[0].strip()

def main():
    parser = argparse.ArgumentParser(description="終盤局面マス切り出し（KIF対応）")
    parser.add_argument("--images",  required=True, help="画像・KIFフォルダ")
    parser.add_argument("--calibs",  required=True, help="_calib.jsonフォルダ")
    parser.add_argument("--out",     required=True, help="出力cells/フォルダ")
    parser.add_argument("--pattern", default="data", help="対象パターン（デフォルト: data）")
    parser.add_argument("--preview", action="store_true", help="プレビュー表示")
    parser.add_argument("--skip-data0", action="store_true", default=True,
                        help="data0・data1をスキップ（デフォルトON）")
    args = parser.parse_args()

    images_folder = Path(args.images)
    out_folder    = Path(args.out)

    # 出力フォルダ作成
    for label in ALL_LABELS:
        (out_folder / label).mkdir(parents=True, exist_ok=True)

    # 対象画像一覧（data2～data8）
    image_files = sorted(set(
        list(images_folder.glob(f"{args.pattern}*.jpg")) +
        list(images_folder.glob(f"{args.pattern}*.JPG"))
    ))

    # data0・data1はスキップ
    if args.skip_data0:
        image_files = [f for f in image_files
                       if not (f.stem.startswith("data0") or f.stem.startswith("data1"))]

    if not image_files:
        print(f"[ERROR] 対象画像が見つかりません")
        sys.exit(1)

    print(f"対象画像: {len(image_files)} 枚")

    total_saved = 0
    errors = []
    kif_cache = {}  # KIFは複数画像で共有するのでキャッシュ

    for img_path in image_files:
        data_name = get_data_name(img_path.stem)
        print(f"\n処理中: {img_path.name}  (KIF: {data_name}.kif)")

        # KIF読み込み（キャッシュ）
        if data_name not in kif_cache:
            kif_path = find_kif(images_folder, data_name)
            if kif_path is None:
                msg = f"  [SKIP] KIF not found: {data_name}.kif"
                print(msg); errors.append(msg); continue
            try:
                board = load_kif(kif_path)
                label_grid = board_to_label_grid(board)
                kif_cache[data_name] = label_grid
                print(f"  KIF読み込み完了: {kif_path.name}")
            except Exception as e:
                msg = f"  [SKIP] KIF読み込みエラー: {e}"
                print(msg); errors.append(msg); continue
        else:
            label_grid = kif_cache[data_name]

        # キャリブレーション読み込み
        calib_path = find_calib(args.calibs, img_path.stem)
        if calib_path is None:
            msg = f"  [SKIP] calib not found: {img_path.stem}_calib.json"
            print(msg); errors.append(msg); continue

        M, grid_size, cell_px = load_calibration(calib_path)

        # 画像読み込み
        img = load_image(img_path)
        if img is None:
            msg = f"  [SKIP] load failed: {img_path.name}"
            print(msg); errors.append(msg); continue

        # マス切り出し・保存
        warped, cells = extract_cells(img, M, grid_size, cell_px)
        saved = 0
        for row, col, cell_img in cells:
            label = label_grid[row][col]
            filename = f"{img_path.stem}_r{row}c{col}.jpg"
            save_path = out_folder / label / filename
            cv2.imencode(".jpg", cell_img)[1].tofile(str(save_path))
            saved += 1
        total_saved += saved
        print(f"  -> {saved} マス保存完了")

        # プレビュー
        if args.preview:
            preview = warped.copy()
            for row in range(grid_size):
                for col in range(grid_size):
                    label = label_grid[row][col]
                    x1 = col * cell_px
                    y1 = row * cell_px
                    color = (0, 200, 0) if label == "empty" else (0, 0, 200)
                    cv2.rectangle(preview, (x1, y1), (x1+cell_px, y1+cell_px), (0,180,0), 1)
                    short = label.replace("sente_","S:").replace("gote_","G:").replace("empty",".")
                    cv2.putText(preview, short, (x1+2, y1+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Preview", 720, 720)
            cv2.setWindowTitle("Preview",
                f"{img_path.name} | S:=先手 G:=後手 .=空 | Enter:次へ ESC:終了")
            cv2.imshow("Preview", preview)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                args.preview = False
                cv2.destroyAllWindows()

    if args.preview:
        cv2.destroyAllWindows()

    # サマリー
    print()
    print("=" * 55)
    print(f"完了: {total_saved} マス画像を保存しました")
    print()
    print("ラベル別保存数（cells/全体）:")
    for label in ALL_LABELS:
        count = len(list((out_folder / label).glob("*.jpg")))
        if count > 0:
            print(f"  {label:<25}: {count} 枚")

    if errors:
        print("\n警告:")
        for e in errors:
            print(f"  {e}")

if __name__ == "__main__":
    main()
