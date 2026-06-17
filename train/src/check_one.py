"""
単一画像 + 対応calib 確認ツール
- 指定した画像とその _calib.json を使ってグリッドとマス番号を重ねて表示
- extract_endgameと同じ座標系で確認できる

使い方:
  python check_one.py \
    --image "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/data5 (1).jpg" \
    --calib "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/data5 (1)_calib.json"
"""
import cv2
import numpy as np
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--calib", required=True)
    args = parser.parse_args()

    # calib読み込み
    with open(args.calib, "r", encoding="utf-8") as f:
        calib = json.load(f)
    M = np.array(calib["perspective_matrix"], dtype="float32")
    cell_px = calib.get("cell_px", 100)
    grid_size = calib.get("grid_size", 9)

    # 画像読み込み
    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] 画像読み込み失敗: {args.image}")
        return

    # パース補正
    side = cell_px * grid_size
    warped = cv2.warpPerspective(img, M, (side, side))

    # グリッド線とマス座標を描画
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * cell_px
            y1 = row * cell_px
            cv2.rectangle(warped, (x1, y1), (x1+cell_px, y1+cell_px), (0, 200, 0), 1)
            cv2.putText(warped, f"r{row}c{col}", (x1+3, y1+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    print(f"画像: {Path(args.image).name}")
    print(f"calib: {Path(args.calib).name}")
    print("補正後の盤面を表示します。r0c0(左上)が先手側の最上段に来ているか確認。")
    print("ESCで閉じる")

    cv2.namedWindow("Check One", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Check One", 720, 720)
    cv2.imshow("Check One", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()