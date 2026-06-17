"""
赤丸検出デバッグツール
全候補に番号を振って画像に描画 → ファイル保存
どの点が盤の四隅か目視で確認する
"""
import cv2
import numpy as np
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True, help="出力画像パス(png)")
    parser.add_argument("--circ-min", type=float, default=0.0, help="円形度の最小値")
    args = parser.parse_args()

    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    print(f"画像サイズ: {w} x {h}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,120,80]), np.array([10,255,255]))
    mask2 = cv2.inRange(hsv, np.array([165,120,80]), np.array([180,255,255]))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = img.copy()
    idx = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue
        circ = 4 * 3.14159 * area / (peri * peri)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if not (600 <= area <= 4000 and 15 <= r <= 40):
            continue
        if circ < args.circ_min:
            continue
        idx += 1
        # 円と番号を描画
        cv2.circle(vis, (int(x), int(y)), int(r)+5, (0, 255, 255), 3)
        cv2.putText(vis, f"{idx}", (int(x)+10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        print(f"  No.{idx}: 位置=({x:.0f},{y:.0f}) 円形度={circ:.2f} 半径={r:.1f} 面積={area:.0f}")

    # 画像四隅にマーカー
    cv2.circle(vis, (0,0), 30, (255,0,0), -1)
    cv2.circle(vis, (w,0), 30, (0,255,0), -1)
    cv2.circle(vis, (w,h), 30, (0,0,255), -1)
    cv2.circle(vis, (0,h), 30, (255,255,0), -1)

    out_path = Path(args.out)
    cv2.imencode(".png", vis)[1].tofile(str(out_path))
    print(f"\n保存: {out_path}")
    print("画像を開いて、盤の四隅にあたる番号を確認してください")

if __name__ == "__main__":
    main()
