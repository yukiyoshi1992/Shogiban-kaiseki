"""
青三角検出・向き判定ツール
- 青色の三角形を検出
- 三角形の頂点方向を判定（頂点側=後手、底辺側=先手）
- 4パターンの光条件でテスト

使い方:
  # 単一画像
  python detect_triangle.py \
    --image "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/青三角データ/data0-aosankaku (1).jpg"

  # フォルダ一括テスト
  python detect_triangle.py \
    --images "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/青三角データ/"
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

# 青色のHSV範囲
BLUE_LOWER = np.array([90,  60,  40])
BLUE_UPPER = np.array([130, 255, 255])

def detect_blue_triangle(img):
    """
    青い三角形を検出して頂点方向を返す
    複数の青領域から「最も三角形らしい」ものを選ぶ（布などの誤検出を除外）
    Returns: (頂点方向, 重心, 三角形頂点リスト, マスク, 面積) or None
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 各青領域を「三角形らしさ」で評価
    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # 小さすぎる領域は無視
            continue

        # bounding boxのアスペクト比（細長い布を除外）
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 2.5:  # 細長すぎ = 布など
            continue

        # 三角形への近似
        ret, triangle = cv2.minEnclosingTriangle(cnt)
        if triangle is None:
            continue
        tri_area = cv2.contourArea(triangle.reshape(-1, 2).astype(np.float32))
        if tri_area <= 0:
            continue

        # 三角形らしさ = 輪郭面積 / 最小外接三角形の面積
        # 完全な三角形なら 1.0 に近い
        triangularity = area / tri_area

        # スコア = 三角形らしさ（布や四角は低くなる）
        score = triangularity
        if score > best_score:
            best_score = score
            best = (cnt, area, triangle)

    if best is None or best_score < 0.6:  # 三角形らしさが低い
        return None

    cnt, area, triangle = best
    pts = triangle.reshape(-1, 2)

    # 重心
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    # 頂点方向: 3頂点のうち重心から最も遠い点
    dists = [np.hypot(p[0]-cx, p[1]-cy) for p in pts]
    apex = pts[int(np.argmax(dists))]
    dx = apex[0] - cx
    dy = apex[1] - cy
    if abs(dx) > abs(dy):
        direction = "right" if dx > 0 else "left"
    else:
        direction = "down" if dy > 0 else "up"

    return direction, (cx, cy), pts, mask, area

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="単一画像")
    parser.add_argument("--images", help="画像フォルダ")
    parser.add_argument("--save", action="store_true", help="検出結果画像を保存")
    args = parser.parse_args()

    if args.image:
        files = [Path(args.image)]
    elif args.images:
        folder = Path(args.images)
        files = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.JPG"))
    else:
        print("--image または --images が必要")
        return

    for img_path in files:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ERROR] 読み込み失敗: {img_path.name}")
            continue

        result = detect_blue_triangle(img)
        if result is None:
            print(f"{img_path.name}: 青三角 検出できず")
            continue

        direction, centroid, pts, mask, area = result
        # 頂点方向 → 先手方向（頂点の反対が先手）
        sente_dir = {"up":"下", "down":"上", "left":"右", "right":"左"}[direction]
        print(f"{img_path.name}: 頂点={direction} 面積={area:.0f} 重心=({centroid[0]:.0f},{centroid[1]:.0f}) → 先手は画像の[{sente_dir}]側")

        # 常に検出結果を保存（原因調査用）
        vis = img.copy()
        # マスクを赤く重ねる
        mask_color = np.zeros_like(vis)
        mask_color[mask > 0] = (0, 0, 255)
        vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)
        # 三角頂点と重心
        for p in pts:
            cv2.circle(vis, (int(p[0]), int(p[1])), 20, (0,255,0), -1)
        cv2.circle(vis, (int(centroid[0]), int(centroid[1])), 15, (255,0,255), -1)
        out_path = img_path.parent / f"tri_{img_path.stem}.png"
        small = cv2.resize(vis, (vis.shape[1]//4, vis.shape[0]//4))
        cv2.imencode(".png", small)[1].tofile(str(out_path))
        print(f"  保存: {out_path.name}")

if __name__ == "__main__":
    main()
