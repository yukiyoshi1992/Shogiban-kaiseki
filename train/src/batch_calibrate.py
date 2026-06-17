"""
バッチキャリブレーションツール
- data0～data8の全画像を順番に表示
- 4隅クリック → Enter確定 → 自動的に {画像名}_calib.json で保存
- スキップ・やり直し・中断再開に対応

使い方:
  python batch_calibrate.py \
    --images "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/" \
    --out    "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/"

操作:
  クリック×4 : 4隅を指定（左上→右上→右下→左下）
  Enter       : 確定して次の画像へ
  q           : 現在の画像をやり直し
  s           : 現在の画像をスキップ（後で再実行可）
  ESC         : 中断（途中まで保存済みのJSONは残る）
"""

import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path

# ---- グローバル（クリックコールバック用） ----
clicked_points = []
display_img = None
original_img = None
scale = 1.0

def mouse_callback(event, x, y, flags, param):
    global clicked_points, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            cv2.circle(display_img, (x, y), 8, (0, 0, 255), -1)
            label = ["左上", "右上", "右下", "左下"][len(clicked_points) - 1]
            cv2.putText(display_img, f"{len(clicked_points)}:{label}",
                        (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            cv2.imshow("Batch Calibration", display_img)
            print(f"  点{len(clicked_points)} ({label}): ({x}, {y})")
            if len(clicked_points) == 4:
                print("  → Enter で確定、q でやり直し")

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
            cx = col * cell_px + cell_px // 2
            cy = row * cell_px + cell_px // 2
            pt = np.array([[[float(cx), float(cy)]]], dtype="float32")
            orig = cv2.perspectiveTransform(pt, M_inv)
            ox, oy = orig[0][0]
            centers_orig.append((float(ox), float(oy)))

    return M, M_inv, centers_orig

def draw_confirm(img, corners, centers, grid_size=9):
    out = img.copy()
    # 外枠
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    # グリッド線
    for i in range(grid_size + 1):
        t = i / grid_size
        top  = corners[0] + t * (corners[1] - corners[0])
        bot  = corners[3] + t * (corners[2] - corners[3])
        left = corners[0] + t * (corners[3] - corners[0])
        right= corners[1] + t * (corners[2] - corners[1])
        cv2.line(out, tuple(top.astype(int)), tuple(bot.astype(int)), (0, 200, 0), 1)
        cv2.line(out, tuple(left.astype(int)), tuple(right.astype(int)), (0, 200, 0), 1)
    # 中心点
    for cx, cy in centers:
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return out

def save_json(out_path, img_path, img_shape, corners, M, M_inv, centers):
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
        "note": "row=0が上側(後手側)、col=0が右側(9筋)"
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_image_list(folder, pattern="data"):
    """指定パターンの画像をソートして返す（KIFは除外）
    pattern="" なら全jpgが対象"""
    folder = Path(folder)
    files = []
    for jpg in sorted(folder.glob(f"{pattern}*.jpg")) + sorted(folder.glob(f"{pattern}*.JPG")):
        if jpg.suffix.lower() == ".jpg":
            files.append(jpg)
    # 重複除去・ソート
    seen = set()
    result = []
    for f in files:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return sorted(result)

def main():
    global clicked_points, display_img, original_img, scale

    parser = argparse.ArgumentParser(description="バッチキャリブレーション")
    parser.add_argument("--images", required=True, help="画像フォルダのパス")
    parser.add_argument("--out",    required=True, help="JSON出力フォルダのパス")
    parser.add_argument("--resume", action="store_true", help="既存JSONがある画像をスキップして再開")
    parser.add_argument("--pattern", default="data", help="対象ファイル名の接頭辞（空文字で全jpg）")
    args = parser.parse_args()

    out_folder = Path(args.out)
    out_folder.mkdir(parents=True, exist_ok=True)

    image_files = get_image_list(args.images, args.pattern)
    if not image_files:
        print(f"[ERROR] 画像が見つかりません: {args.images}")
        sys.exit(1)

    print(f"対象画像: {len(image_files)} 枚")

    # --resume: 既存JSONがある画像をスキップ
    if args.resume:
        remaining = []
        for f in image_files:
            json_path = out_folder / f"{f.stem}_calib.json"
            if json_path.exists():
                print(f"  [スキップ] {f.name} (既存JSON あり)")
            else:
                remaining.append(f)
        image_files = remaining
        print(f"残り: {len(image_files)} 枚\n")

    if not image_files:
        print("すべての画像のキャリブレーションが完了しています。")
        sys.exit(0)

    cv2.namedWindow("Batch Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Batch Calibration", mouse_callback)

    total = len(image_files)
    i = 0

    while i < total:
        img_path = image_files[i]
        json_out = out_folder / f"{img_path.stem}_calib.json"

        # 画像読み込み
        original_img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if original_img is None:
            print(f"[WARNING] 読み込み失敗: {img_path.name} → スキップ")
            i += 1
            continue

        h, w = original_img.shape[:2]
        max_disp = 1200
        scale = min(max_disp / w, max_disp / h, 1.0)
        dw, dh = int(w * scale), int(h * scale)

        # リセット
        clicked_points = []
        display_img = cv2.resize(original_img, (dw, dh))

        print(f"\n{'='*55}")
        print(f"[{i+1}/{total}] {img_path.name}")
        print(f"  出力: {json_out.name}")
        print(f"  操作: クリック×4で4隅指定 → Enter確定 / q=やり直し / s=スキップ / ESC=中断")
        print(f"{'='*55}")

        # タイトルバー
        cv2.setWindowTitle("Batch Calibration",
            f"[{i+1}/{total}] {img_path.name}  |  クリック×4 → Enter確定  q=やり直し  s=スキップ  ESC=中断")
        cv2.resizeWindow("Batch Calibration", dw, dh)
        cv2.imshow("Batch Calibration", display_img)

        action = None  # "next" / "skip" / "abort"

        while action is None:
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                action = "abort"

            elif key == ord('s'):  # スキップ
                print(f"  [スキップ] {img_path.name}")
                action = "skip"

            elif key == ord('q'):  # やり直し
                clicked_points = []
                display_img = cv2.resize(original_img, (dw, dh))
                cv2.imshow("Batch Calibration", display_img)
                print("  [やり直し] もう一度4隅をクリックしてください")

            elif key == 13:  # Enter
                if len(clicked_points) < 4:
                    print(f"  [WARNING] まだ{len(clicked_points)}点です（4点必要）")
                else:
                    # 元画像座標に変換
                    pts_orig = [(int(x / scale), int(y / scale)) for x, y in clicked_points]
                    ordered = order_points(pts_orig)
                    M, M_inv, centers = compute_grid(ordered)

                    # 確認表示
                    confirm_img = draw_confirm(original_img, ordered, centers)
                    confirm_disp = cv2.resize(confirm_img, (dw, dh))
                    cv2.setWindowTitle("Batch Calibration",
                        f"[{i+1}/{total}] 確認: {img_path.name}  |  Enter=保存して次へ  q=やり直し")
                    cv2.imshow("Batch Calibration", confirm_disp)
                    print("  グリッド確認中... Enter=保存して次へ / q=やり直し")

                    while True:
                        k2 = cv2.waitKey(20) & 0xFF
                        if k2 == 13:  # Enter → 保存
                            save_json(json_out, img_path, original_img.shape,
                                      ordered, M, M_inv, centers)
                            print(f"  [保存] {json_out.name}")
                            action = "next"
                            break
                        elif k2 == ord('q'):  # やり直し
                            clicked_points = []
                            display_img = cv2.resize(original_img, (dw, dh))
                            cv2.setWindowTitle("Batch Calibration",
                                f"[{i+1}/{total}] {img_path.name}  |  クリック×4 → Enter確定")
                            cv2.imshow("Batch Calibration", display_img)
                            print("  [やり直し]")
                            break

        if action == "abort":
            print(f"\n[中断] {i}/{total} 枚完了")
            print("再開するには --resume オプションをつけて実行してください")
            break
        elif action in ("next", "skip"):
            i += 1

    cv2.destroyAllWindows()

    # 完了サマリー
    done = [f for f in get_image_list(args.images)
            if (out_folder / f"{f.stem}_calib.json").exists()]
    print(f"\n完了: {len(done)}/{len(get_image_list(args.images))} 枚のJSONが保存されています")

if __name__ == "__main__":
    main()
