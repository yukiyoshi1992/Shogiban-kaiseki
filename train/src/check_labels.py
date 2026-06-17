"""
学習データ確認ツール
- cells/フォルダの各ラベルからランダムに画像を抽出して表示
- 画像とラベル名が一致しているか目視確認

使い方:
  python check_labels.py \
    --cells "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/cells/"

操作:
  Enter / →  : 次のラベルへ
  ←          : 前のラベルへ
  ESC / q    : 終了
"""

import cv2
import numpy as np
import argparse
import random
from pathlib import Path

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

def make_grid(images_with_names, label, max_cols=12, cell_size=70):
    """画像をグリッド状に並べてラベルを上部に大きく表示"""
    n = len(images_with_names)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    grid_h = cell_size * rows + 70  # 70px はラベル表示領域（文字を大きくしたため拡張）
    grid_w = max(cell_size * cols, 560)
    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    # ラベル名を上部に大きく表示
    cv2.putText(grid, label, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 180), 3)

    for idx, (img, fname) in enumerate(images_with_names):
        r = idx // cols
        c = idx % cols
        x1 = c * cell_size
        y1 = r * cell_size + 70
        resized = cv2.resize(img, (cell_size, cell_size))
        grid[y1:y1+cell_size, x1:x1+cell_size] = resized
        # 枠線
        cv2.rectangle(grid, (x1, y1), (x1+cell_size-1, y1+cell_size-1), (180,180,180), 1)
        # ファイル名（短縮）を下部に表示
        short = fname.replace("data", "d").replace("shift_", "sh_").replace(".jpg", "")
        short = short.replace(" ", "").replace("_r", " r")
        cv2.putText(grid, short, (x1+1, y1+cell_size-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, (200,0,0), 1)

    return grid

def main():
    parser = argparse.ArgumentParser(description="学習データラベル確認")
    parser.add_argument("--cells",   required=True, help="cells/フォルダのパス")
    parser.add_argument("--samples", type=int, default=60, help="ラベルごとの表示枚数（デフォルト60）")
    parser.add_argument("--seed",    type=int, default=42, help="ランダムシード")
    args = parser.parse_args()

    random.seed(args.seed)
    cells_folder = Path(args.cells)

    # 存在するラベルだけ対象に
    valid_labels = []
    for label in ALL_LABELS:
        d = cells_folder / label
        if d.exists() and any(d.glob("*.jpg")):
            valid_labels.append(label)

    if not valid_labels:
        print(f"[ERROR] 画像が見つかりません: {cells_folder}")
        return

    print(f"確認対象ラベル数: {len(valid_labels)}")
    print("操作: Enter/→=次 / ←=前 / q/ESC=終了")

    cv2.namedWindow("Label Check", cv2.WINDOW_NORMAL)
    idx = 0

    while True:
        label = valid_labels[idx]
        label_dir = cells_folder / label
        all_files = list(label_dir.glob("*.jpg"))
        samples = random.sample(all_files, min(args.samples, len(all_files)))

        # 画像読み込み
        images_with_names = []
        for f in samples:
            img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                images_with_names.append((img, f.name))

        grid = make_grid(images_with_names, f"[{idx+1}/{len(valid_labels)}] {label}  (全{len(all_files)}枚中{len(images_with_names)}枚表示)", cell_size=160)

        # ウィンドウサイズ調整
        h, w = grid.shape[:2]
        max_w = 1400
        scale = min(max_w / w, 1.0)
        disp = cv2.resize(grid, (int(w*scale), int(h*scale)))
        cv2.setWindowTitle("Label Check",
            f"[{idx+1}/{len(valid_labels)}] {label} | Enter/d=次  a=前  q=終了")
        cv2.resizeWindow("Label Check", int(w*scale), int(h*scale))
        cv2.imshow("Label Check", disp)

        key = cv2.waitKey(0) & 0xFF
        if key in [27, ord('q')]:
            break
        elif key in [13, ord('d'), 83]:   # Enter / d / →
            idx = (idx + 1) % len(valid_labels)
        elif key in [ord('a'), 81]:        # a / ←
            idx = (idx - 1) % len(valid_labels)

    cv2.destroyAllWindows()
    print("確認完了")

if __name__ == "__main__":
    main()
