"""
キャリブレーション確認ツール
- calibration.jsonを使って複数画像にグリッドを重ねて表示
- キーボードで次/前の画像に移動
- ズレ具合を目視確認

使い方:
  python check_calibration.py \
    --calib "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/calibration.json" \
    --images "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/" \
    --pattern "data0"

操作:
  → or d : 次の画像
  ← or a : 前の画像
  g       : グリッド線の表示/非表示切り替え
  p       : 中心点の表示/非表示切り替え
  w       : パース補正後画像の表示/非表示切り替え
  s       : 現在の画像をPNG保存（確認用）
  ESC/q   : 終了
"""

import cv2
import numpy as np
import json
import argparse
import sys
import glob
import os
from pathlib import Path

def load_calibration(calib_path):
    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    M = np.array(data["perspective_matrix"], dtype="float32")
    M_inv = np.array(data["perspective_matrix_inv"], dtype="float32")
    corners = np.array(data["corner_points"], dtype="float32")
    centers = data["cell_centers_original"]
    cell_px = data.get("cell_px", 100)
    grid_size = data.get("grid_size", 9)
    return M, M_inv, corners, centers, cell_px, grid_size

def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img

def draw_overlay(img, corners, centers, grid_size, show_grid, show_points):
    out = img.copy()

    if show_grid:
        # 盤面外枠
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 255), thickness=2)

        # グリッド線（透視変換を使って正確に引く）
        # 横線・縦線を9×9に分割して描画
        for i in range(grid_size + 1):
            t = i / grid_size
            # 上辺→下辺の補間（縦線）
            top = corners[0] + t * (corners[1] - corners[0])
            bot = corners[3] + t * (corners[2] - corners[3])
            cv2.line(out, tuple(top.astype(int)), tuple(bot.astype(int)),
                     (0, 200, 0), 1)
            # 左辺→右辺の補間（横線）
            left = corners[0] + t * (corners[3] - corners[0])
            right = corners[1] + t * (corners[2] - corners[1])
            cv2.line(out, tuple(left.astype(int)), tuple(right.astype(int)),
                     (0, 200, 0), 1)

    if show_points:
        for idx, (cx, cy) in enumerate(centers):
            row = idx // grid_size
            col = idx % grid_size
            cx_i, cy_i = int(cx), int(cy)
            cv2.circle(out, (cx_i, cy_i), 4, (0, 0, 255), -1)
            # 端のマスだけ座標ラベル
            if row == 0 and col == 0:
                cv2.putText(out, "0,0", (cx_i+3, cy_i-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    return out

def draw_warped(img, M, cell_px, grid_size):
    side = cell_px * grid_size
    warped = cv2.warpPerspective(img, M, (side, side))
    # グリッド線を引く
    for i in range(grid_size + 1):
        pos = i * cell_px
        cv2.line(warped, (pos, 0), (pos, side), (0, 200, 0), 1)
        cv2.line(warped, (0, pos), (side, pos), (0, 200, 0), 1)
    return warped

def compute_display_scale(img, max_w=1200, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return scale

def main():
    parser = argparse.ArgumentParser(description="キャリブレーション確認ツール")
    parser.add_argument("--calib", required=True, help="calibration.jsonのパス")
    parser.add_argument("--images", required=True, help="画像フォルダのパス")
    parser.add_argument("--pattern", default="data0", help="対象ファイルのパターン（デフォルト: data0）")
    args = parser.parse_args()

    # キャリブレーション読み込み
    try:
        M, M_inv, corners, centers, cell_px, grid_size = load_calibration(args.calib)
    except Exception as e:
        print(f"[ERROR] calibration.json読み込み失敗: {e}")
        sys.exit(1)

    # 画像一覧取得
    folder = Path(args.images)
    all_files = sorted(folder.glob(f"{args.pattern}*.jpg")) + \
                sorted(folder.glob(f"{args.pattern}*.JPG")) + \
                sorted(folder.glob(f"{args.pattern}*.jpeg"))
    # 重複除去・ソート
    all_files = sorted(set(all_files))

    if not all_files:
        print(f"[ERROR] パターン '{args.pattern}*.jpg' に一致する画像が見つかりません: {folder}")
        sys.exit(1)

    print(f"対象画像: {len(all_files)} 枚")
    for i, f in enumerate(all_files):
        print(f"  [{i}] {f.name}")
    print()
    print("操作: →/d=次  ←/a=前  g=グリッド切替  p=点切替  w=補正画像切替  s=保存  ESC/q=終了")

    # 状態
    idx = 0
    show_grid = True
    show_points = True
    show_warped = False

    cv2.namedWindow("Calibration Check", cv2.WINDOW_NORMAL)

    while True:
        img_path = str(all_files[idx])
        img = load_image(img_path)
        if img is None:
            print(f"[WARNING] 読み込み失敗: {img_path}")
            idx = (idx + 1) % len(all_files)
            continue

        # オーバーレイ描画
        overlay = draw_overlay(img, corners, centers, grid_size, show_grid, show_points)

        # ウィンドウタイトル
        title = f"[{idx+1}/{len(all_files)}] {all_files[idx].name}  |  g=グリッド({('ON' if show_grid else 'OFF')})  p=点({('ON' if show_points else 'OFF')})  w=補正({('ON' if show_warped else 'OFF')})"
        cv2.setWindowTitle("Calibration Check", title)

        scale = compute_display_scale(overlay)
        dw = int(overlay.shape[1] * scale)
        dh = int(overlay.shape[0] * scale)
        cv2.resizeWindow("Calibration Check", dw, dh)
        disp = cv2.resize(overlay, (dw, dh))
        cv2.imshow("Calibration Check", disp)

        # パース補正ウィンドウ
        if show_warped:
            warped = draw_warped(img, M, cell_px, grid_size)
            warped_disp = cv2.resize(warped, (600, 600))
            cv2.imshow("Warped Board", warped_disp)
        else:
            try:
                cv2.destroyWindow("Warped Board")
            except:
                pass

        key = cv2.waitKey(30) & 0xFF

        if key in [27, ord('q')]:  # ESC or q
            break
        elif key in [83, ord('d'), 0]:  # → or d
            idx = (idx + 1) % len(all_files)
        elif key in [81, ord('a'), 0]:  # ← or a
            idx = (idx - 1) % len(all_files)
        elif key == ord('g'):
            show_grid = not show_grid
        elif key == ord('p'):
            show_points = not show_points
        elif key == ord('w'):
            show_warped = not show_warped
        elif key == ord('s'):
            save_name = f"check_{all_files[idx].stem}.png"
            cv2.imwrite(save_name, overlay)
            print(f"[保存] {save_name}")

    cv2.destroyAllWindows()
    print("終了しました")

if __name__ == "__main__":
    main()