"""
駒認識用マス切り出しツール（修正版）
- data1の画像 + _calib.json を使って81マスを切り出し
- 将棋の初期配置に基づいてラベルを自動付与
- train/data/cells/{ラベル}/ に保存

座標系（画像の向きに合わせた定義）:
  col: 0=左=先手1段目  ～  8=右=後手9段目  (画像左→右)
  row: 0=上=9筋        ～  8=下=1筋        (画像上→下)

使い方:
  python extract_cells.py \
    --images "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/" \
    --calibs "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/" \
    --out    "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/cells/" \
    --pattern "data1"
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path

# -------------------------------------------------------
# 将棋の初期配置
#   board[row][col]
#   row: 0=9筋(上) ～ 8=1筋(下)
#   col: 0=先手1段目(左) ～ 8=後手9段目(右)
# -------------------------------------------------------
INITIAL_BOARD = [
    # col: 0          1            2           3       4       5       6          7            8
    ["sente_kyo",  "empty",     "sente_fu", "empty","empty","empty","gote_fu", "empty",     "gote_kyo" ],  # row0: 9筋
    ["sente_kei",  "sente_kaku","sente_fu", "empty","empty","empty","gote_fu", "gote_hi",   "gote_kei" ],  # row1: 8筋 左=先手角 右=後手飛
    ["sente_gin",  "empty",     "sente_fu", "empty","empty","empty","gote_fu", "empty",     "gote_gin" ],  # row2: 7筋
    ["sente_kin",  "empty",     "sente_fu", "empty","empty","empty","gote_fu", "empty",     "gote_kin" ],  # row3: 6筋
    ["sente_ou",   "empty",     "sente_fu", "empty","empty","empty","gote_fu", "empty",     "gote_ou"  ],  # row4: 5筋 <- 王
    ["sente_kin",  "empty",     "sente_fu", "empty","empty","empty","gote_fu", "empty",     "gote_kin" ],  # row5: 4筋
    ["sente_gin",  "empty",     "sente_fu", "empty","empty","empty","gote_fu", "empty",     "gote_gin" ],  # row6: 3筋
    ["sente_kei",  "sente_hi",  "sente_fu", "empty","empty","empty","gote_fu", "gote_kaku", "gote_kei" ],  # row7: 2筋 左=先手飛 右=後手角
    ["sente_kyo",  "empty",     "sente_fu", "empty","empty","empty","gote_fu", "empty",     "gote_kyo" ],  # row8: 1筋
]

ALL_LABELS = [
    "empty",
    "sente_fu","sente_kyo","sente_kei","sente_gin","sente_kin","sente_kaku","sente_hi","sente_ou",
    "gote_fu", "gote_kyo", "gote_kei", "gote_gin", "gote_kin", "gote_kaku", "gote_hi", "gote_ou",
]

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

def main():
    parser = argparse.ArgumentParser(description="駒認識用マス切り出し")
    parser.add_argument("--images",  required=True)
    parser.add_argument("--calibs",  required=True)
    parser.add_argument("--out",     required=True)
    parser.add_argument("--pattern", default="data1")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    images_folder = Path(args.images)
    out_folder    = Path(args.out)

    for label in ALL_LABELS:
        (out_folder / label).mkdir(parents=True, exist_ok=True)

    image_files = sorted(set(
        list(images_folder.glob(f"{args.pattern}*.jpg")) +
        list(images_folder.glob(f"{args.pattern}*.JPG"))
    ))

    if not image_files:
        print(f"[ERROR] 画像が見つかりません: {args.pattern}*.jpg in {images_folder}")
        sys.exit(1)

    print(f"対象画像: {len(image_files)} 枚 / 出力先: {out_folder}")

    total_saved = 0
    errors = []

    for img_path in image_files:
        print(f"処理中: {img_path.name}")

        calib_path = find_calib(args.calibs, img_path.stem)
        if calib_path is None:
            msg = f"  [SKIP] calib not found: {img_path.stem}_calib.json"
            print(msg); errors.append(msg); continue

        M, grid_size, cell_px = load_calibration(calib_path)
        img = load_image(img_path)
        if img is None:
            msg = f"  [SKIP] load failed: {img_path.name}"
            print(msg); errors.append(msg); continue

        warped, cells = extract_cells(img, M, grid_size, cell_px)

        for row, col, cell_img in cells:
            label = INITIAL_BOARD[row][col]
            filename = f"{img_path.stem}_r{row}c{col}.jpg"
            save_path = out_folder / label / filename
            cv2.imencode(".jpg", cell_img)[1].tofile(str(save_path))
            total_saved += 1

        print(f"  -> 81マス保存完了")

        if args.preview:
            preview = warped.copy()
            for row in range(grid_size):
                for col in range(grid_size):
                    label = INITIAL_BOARD[row][col]
                    x1 = col * cell_px
                    y1 = row * cell_px
                    cv2.rectangle(preview, (x1, y1), (x1+cell_px, y1+cell_px), (0,200,0), 1)
                    # ラベルをアルファベットのみで表示
                    short = label.replace("sente_","S:").replace("gote_","G:").replace("empty",".")
                    cv2.putText(preview, short, (x1+3, y1+18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,0,200), 1)
            # 王のマスを強調
            for row in range(grid_size):
                for col in range(grid_size):
                    if "ou" in INITIAL_BOARD[row][col]:
                        x1 = col * cell_px
                        y1 = row * cell_px
                        cv2.rectangle(preview, (x1, y1), (x1+cell_px, y1+cell_px), (0,0,255), 3)

            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Preview", 720, 720)
            cv2.setWindowTitle("Preview",
                f"{img_path.name} | 赤枠=王  S:=先手  G:=後手  .=空  Enter:次へ  ESC:終了")
            cv2.imshow("Preview", preview)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                args.preview = False
                cv2.destroyAllWindows()

    if args.preview:
        cv2.destroyAllWindows()

    print()
    print("=" * 50)
    print(f"完了: {total_saved} マス画像を保存")
    print()
    print("ラベル別保存数:")
    for label in ALL_LABELS:
        count = len(list((out_folder / label).glob("*.jpg")))
        if count > 0:
            print(f"  {label:<20}: {count} 枚")

    if errors:
        print("\n警告:")
        for e in errors:
            print(f"  {e}")

if __name__ == "__main__":
    main()