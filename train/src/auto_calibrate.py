"""
赤丸自動検出キャリブレーションツール
- 画像内の赤丸を自動検出
- 盤面の4隅を特定してcalibration.jsonを生成
- 手動クリック不要で高速キャリブレーション

使い方:
  # 単一画像
  python auto_calibrate.py \
    --image "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/data0-akamaru (1).jpg" \
    --out   "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/auto_calib.json"

  # フォルダ内の全画像を一括処理
  python auto_calibrate.py \
    --images "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/対局/" \
    --out    "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/"

オプション:
  --preview   検出結果をプレビュー表示
  --pattern   対象ファイルパターン（デフォルト: IMG_）
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path

# -------------------------------------------------------
# 赤丸検出パラメータ
# -------------------------------------------------------
# HSV色空間での赤の範囲（赤は0度と180度付近の2範囲）
RED_LOWER1 = np.array([0,   120, 80])
RED_UPPER1 = np.array([10,  255, 255])
RED_LOWER2 = np.array([165, 120, 80])
RED_UPPER2 = np.array([180, 255, 255])

# 盤の赤丸は意図的に大きく描いてある（駒台の丸より大きい）。
# 画像の短辺に対する比率で半径しきい値を決める（撮影距離が一定なら安定）。
# 例: 短辺2160pxで盤の赤丸 半径30-35 → 比率 約0.014-0.016
BOARD_RADIUS_RATIO_MIN = 0.011   # 盤赤丸の最小半径（短辺比）
BOARD_RADIUS_RATIO_MAX = 0.025   # 盤赤丸の最大半径（短辺比）
MIN_CIRCULARITY = 0.45           # 円形度の下限（本・駒袋の歪んだ赤を除外）
MIN_AREA = 400                   # 最小面積(px^2)

def detect_red_circles(img):
    """
    画像から盤の赤丸の中心座標リストを返す
    盤の赤丸は大きめなので、サイズと円形度でフィルタして
    駒台の小さい丸・本や駒袋の赤を除外する
    Returns: list of (x, y, radius, circularity)
    """
    h, w = img.shape[:2]
    short_side = min(h, w)
    r_min = short_side * BOARD_RADIUS_RATIO_MIN
    r_max = short_side * BOARD_RADIUS_RATIO_MAX

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤マスク（2つの範囲を合成）
    mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
    mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)

    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 輪郭検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circ = 4 * np.pi * area / (peri * peri)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        # 盤の赤丸サイズ + 円形度でフィルタ
        if r_min <= radius <= r_max and circ >= MIN_CIRCULARITY:
            circles.append((float(x), float(y), float(radius), float(circ)))

    return circles

def select_board_corners(circles, img_shape):
    """
    検出された赤丸から盤面の4隅を選択する

    戦略: 盤の4隅は画像の四隅に最も近い赤丸
      - 左上隅  → 画像左上(0,0)に最も近い点
      - 右上隅  → 画像右上(w,0)に最も近い点
      - 右下隅  → 画像右下(w,h)に最も近い点
      - 左下隅  → 画像左下(0,h)に最も近い点
    """
    if len(circles) < 4:
        return None

    h, w = img_shape[:2]

    # 画像の4隅座標
    image_corners = [
        (0, 0),      # 左上
        (w, 0),      # 右上
        (w, h),      # 右下
        (0, h),      # 左下
    ]

    selected = []
    used = set()

    for ic_x, ic_y in image_corners:
        best_dist = float("inf")
        best_idx = -1
        for i, c in enumerate(circles):
            if i in used:
                continue
            x, y = c[0], c[1]
            dist = (x - ic_x)**2 + (y - ic_y)**2
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx == -1:
            return None
        used.add(best_idx)
        x, y = circles[best_idx][0], circles[best_idx][1]
        selected.append((x, y))

    # 左上・右上・右下・左下の順
    return [selected[0], selected[1], selected[2], selected[3]]

def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[2] = pts[np.argmax(s)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def compute_grid(src_pts, grid_size=9, cell_px=100):
    side = cell_px * grid_size
    dst_pts = np.array([
        [0, 0], [side, 0], [side, side], [0, side]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    centers_orig = []
    for row in range(grid_size):
        for col in range(grid_size):
            px = col * cell_px + cell_px // 2
            py = row * cell_px + cell_px // 2
            pt = np.array([[[float(px), float(py)]]], dtype="float32")
            orig = cv2.perspectiveTransform(pt, M_inv)
            centers_orig.append((float(orig[0][0][0]), float(orig[0][0][1])))

    return M, M_inv, centers_orig

def save_json(out_path, img_path, img_shape, corners, M, M_inv, centers, all_circles):
    h, w = img_shape[:2]
    data = {
        "image_path": str(img_path),
        "original_size": [w, h],
        "corner_points": corners.tolist(),
        "perspective_matrix": M.tolist(),
        "perspective_matrix_inv": M_inv.tolist(),
        "cell_centers_original": centers,
        "grid_size": 9,
        "cell_px": 100,
        "all_red_circles": [(c[0], c[1], c[2]) for c in all_circles],
        "note": "auto-calibrated from red circles"
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_image(img_path, out_path, preview=False):
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [ERROR] 読み込み失敗: {img_path}")
        return False

    h, w = img.shape[:2]

    # 赤丸検出
    circles = detect_red_circles(img)
    print(f"  赤丸検出数: {len(circles)}")

    if len(circles) < 4:
        print(f"  [ERROR] 赤丸が4個未満 ({len(circles)}個) → スキップ")
        return False

    # 盤面4隅を選択
    corner_pts = select_board_corners(circles, img.shape)
    if corner_pts is None:
        print(f"  [ERROR] 4隅の特定に失敗 → スキップ")
        return False

    # 点を正規順序に
    ordered = order_points(corner_pts)

    # グリッド計算
    M, M_inv, centers = compute_grid(ordered)

    # JSON保存
    save_json(out_path, img_path, img.shape, ordered, M, M_inv, centers, circles)
    print(f"  [OK] 保存: {out_path.name}")
    print(f"  4隅: {ordered.tolist()}")

    # プレビュー
    if preview:
        vis = img.copy()
        # 全赤丸を表示
        for c in circles:
            x, y, r = c[0], c[1], c[2]
            cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 255), 2)
            cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 255), -1)

        # 盤面4隅を強調
        colors = [(0,0,255),(0,128,255),(0,255,0),(255,0,0)]
        labels = ["TL","TR","BR","BL"]
        for i, (px, py) in enumerate(ordered):
            cv2.circle(vis, (int(px), int(py)), 15, colors[i], 3)
            cv2.putText(vis, labels[i], (int(px)+15, int(py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors[i], 2)

        # グリッド線
        for i in range(10):
            t = i / 9
            top  = ordered[0] + t * (ordered[1] - ordered[0])
            bot  = ordered[3] + t * (ordered[2] - ordered[3])
            left = ordered[0] + t * (ordered[3] - ordered[0])
            right= ordered[1] + t * (ordered[2] - ordered[1])
            cv2.line(vis, tuple(top.astype(int)), tuple(bot.astype(int)), (0,200,0), 1)
            cv2.line(vis, tuple(left.astype(int)), tuple(right.astype(int)), (0,200,0), 1)

        scale = min(1200/w, 900/h, 1.0)
        disp = cv2.resize(vis, (int(w*scale), int(h*scale)))
        cv2.namedWindow("Auto Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Auto Calibration", int(w*scale), int(h*scale))
        cv2.setWindowTitle("Auto Calibration",
            f"{img_path.name} | 黄=全赤丸 色付き=盤4隅 | Enter:次へ ESC:終了")
        cv2.imshow("Auto Calibration", disp)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return "abort"

    return True

def main():
    parser = argparse.ArgumentParser(description="赤丸自動検出キャリブレーション")
    parser.add_argument("--image",   help="単一画像パス")
    parser.add_argument("--images",  help="画像フォルダ")
    parser.add_argument("--out",     required=True, help="出力パス（単一画像時はJSONパス、フォルダ時は出力フォルダ）")
    parser.add_argument("--pattern", default="IMG_", help="対象ファイルパターン（デフォルト: IMG_）")
    parser.add_argument("--preview", action="store_true", help="検出結果をプレビュー表示")
    parser.add_argument("--resume",  action="store_true", help="既存JSONをスキップ")
    args = parser.parse_args()

    # 単一画像モード
    if args.image:
        img_path = Path(args.image)
        out_path = Path(args.out)
        print(f"処理: {img_path.name}")
        process_image(img_path, out_path, preview=args.preview)
        if args.preview:
            cv2.destroyAllWindows()
        return

    # フォルダモード
    if args.images:
        images_folder = Path(args.images)
        out_folder    = Path(args.out)
        out_folder.mkdir(parents=True, exist_ok=True)

        image_files = sorted(set(
            list(images_folder.glob(f"{args.pattern}*.jpg")) +
            list(images_folder.glob(f"{args.pattern}*.JPG")) +
            list(images_folder.glob(f"*.jpg")) +
            list(images_folder.glob(f"*.JPG"))
        ))
        # パターンでフィルタ
        if args.pattern != "*":
            image_files = [f for f in image_files if f.name.startswith(args.pattern)]

        if not image_files:
            print(f"[ERROR] 画像が見つかりません: {images_folder}")
            sys.exit(1)

        print(f"対象画像: {len(image_files)} 枚")
        ok, ng, skip = 0, 0, 0

        for img_path in image_files:
            out_path = out_folder / f"{img_path.stem}_calib.json"

            if args.resume and out_path.exists():
                print(f"[SKIP] {img_path.name}")
                skip += 1
                continue

            print(f"処理: {img_path.name}")
            result = process_image(img_path, out_path, preview=args.preview)
            if result == "abort":
                print("中断しました")
                break
            elif result:
                ok += 1
            else:
                ng += 1

        if args.preview:
            cv2.destroyAllWindows()

        print(f"\n完了: 成功={ok} 失敗={ng} スキップ={skip}")
        return

    print("[ERROR] --image または --images が必要です")
    sys.exit(1)

if __name__ == "__main__":
    main()