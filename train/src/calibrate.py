"""
将棋盤キャリブレーションツール
- data0の画像を開いて盤面4隅をクリック指定
- 81マスの座標をJSONに保存
- 確認用に分割結果を表示

使い方:
  python calibrate.py --image "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/data0 (1).jpg"
  python calibrate.py --image "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/data0 (1).jpg" --out "//YukiYoshiNAS/Shogiban-kaiseki-tool/src/calibration.json"
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

def mouse_callback(event, x, y, flags, param):
    global clicked_points, display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            # クリック点を描画
            cv2.circle(display_img, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(display_img, str(len(clicked_points)),
                        (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
            cv2.imshow("Calibration", display_img)
            print(f"  点{len(clicked_points)}: ({x}, {y})")

def order_points(pts):
    """4点を 左上・右上・右下・左下 の順に並び替え"""
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]    # 左上
    ordered[1] = pts[np.argmin(diff)] # 右上
    ordered[2] = pts[np.argmax(s)]    # 右下
    ordered[3] = pts[np.argmax(diff)] # 左下
    return ordered

def compute_grid(src_pts, grid_size=9):
    """
    4隅の座標から81マスの中心座標と変換行列を計算
    src_pts: 左上・右上・右下・左下 の順の座標
    """
    # 変換後の正方形サイズ
    side = 900  # 100px × 9マス
    dst_pts = np.array([
        [0, 0],
        [side, 0],
        [side, side],
        [0, side]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

    cell = side // grid_size
    centers = []
    for row in range(grid_size):
        for col in range(grid_size):
            cx = col * cell + cell // 2
            cy = row * cell + cell // 2
            centers.append((cx, cy))

    # 変換後座標 → 元画像座標に戻す
    centers_orig = []
    for (cx, cy) in centers:
        pt = np.array([[[float(cx), float(cy)]]], dtype="float32")
        orig = cv2.perspectiveTransform(pt, M_inv)
        ox, oy = orig[0][0]
        centers_orig.append((float(ox), float(oy)))

    return M, M_inv, centers_orig

def draw_grid(img, centers_orig, grid_size=9, M=None, cell_px=100):
    """元画像にグリッドを描画して確認用に返す
    Mがあれば透視変換の逆行列でグリッド線も描画する（より見やすい）"""
    import numpy as np
    out = img.copy()

    if M is not None:
        # ワープ後のグリッド線を元画像に逆変換して描く
        M_inv = np.linalg.inv(M)
        side = cell_px * grid_size

        def warp_pt(px, py):
            """ワープ後座標 → 元画像座標"""
            pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(pt, M_inv)
            return (int(dst[0][0][0]), int(dst[0][0][1]))

        # 横線・縦線を描く
        for i in range(grid_size + 1):
            # 横線
            p1 = warp_pt(0, i * cell_px)
            p2 = warp_pt(side, i * cell_px)
            cv2.line(out, p1, p2, (0, 255, 0), 2)
            # 縦線
            p3 = warp_pt(i * cell_px, 0)
            p4 = warp_pt(i * cell_px, side)
            cv2.line(out, p3, p4, (0, 255, 0), 2)

        # マスの中央に点を打つ（位置の確認用）
        for row in range(grid_size):
            for col in range(grid_size):
                cx = warp_pt(col * cell_px + cell_px // 2, row * cell_px + cell_px // 2)
                cv2.circle(out, cx, 3, (0, 200, 255), -1)
    else:
        # 従来の点のみ表示
        for idx, (cx, cy) in enumerate(centers_orig):
            row = idx // grid_size
            col = idx % grid_size
            cx, cy = int(cx), int(cy)
            cv2.circle(out, (cx, cy), 4, (0, 255, 0), -1)
            if col == 0 or row == 0:
                cv2.putText(out, f"{row},{col}", (cx+2, cy-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
    return out

def extract_cells(img, M, grid_size=9, cell_px=100):
    """パース補正後の画像から各マスを切り出してリストで返す"""
    side = cell_px * grid_size
    dst_pts = np.array([
        [0, 0], [side, 0], [side, side], [0, side]
    ], dtype="float32")
    # Mが元画像→正方形の変換行列であることを前提
    warped = cv2.warpPerspective(img, M, (side, side))
    cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = col * cell_px
            y1 = row * cell_px
            cell = warped[y1:y1+cell_px, x1:x1+cell_px]
            cells.append(cell)
    return warped, cells

def main():
    global clicked_points, display_img

    parser = argparse.ArgumentParser(description="将棋盤キャリブレーション")
    parser.add_argument("--image", required=True, help="空盤面画像パス (data0系)")
    parser.add_argument("--out", default="calibration.json", help="出力JSONパス")
    parser.add_argument("--preview-cells", action="store_true", help="マス切り出しプレビューも表示")
    args = parser.parse_args()

    # 画像読み込み（日本語・UNCパス対応）
    img_path = args.image
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] 画像を読み込めません: {img_path}")
        sys.exit(1)

    # 表示用にリサイズ（4MBのJPEGはそのままだと大きすぎる場合あり）
    h, w = img.shape[:2]
    max_disp = 1200
    scale = min(max_disp / w, max_disp / h, 1.0)
    disp_w, disp_h = int(w * scale), int(h * scale)
    display_img = cv2.resize(img, (disp_w, disp_h))

    print("=" * 50)
    print("将棋盤キャリブレーション")
    print("=" * 50)
    print("盤面の4隅を以下の順でクリックしてください：")
    print("  1: 左上  2: 右上  3: 右下  4: 左下")
    print("  ※盤の外枠（木の部分）の内側の角を指定")
    print("  q: やり直し  Enter: 確定")
    print("=" * 50)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", disp_w, disp_h)
    cv2.setMouseCallback("Calibration", mouse_callback)
    cv2.imshow("Calibration", display_img)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            # やり直し
            clicked_points = []
            display_img = cv2.resize(img, (disp_w, disp_h))
            cv2.imshow("Calibration", display_img)
            print("[リセット] もう一度4隅をクリックしてください")
        elif key == 13:  # Enter
            if len(clicked_points) == 4:
                break
            else:
                print(f"[WARNING] まだ{len(clicked_points)}点しか選択されていません（4点必要）")
        elif key == 27:  # ESC
            print("キャンセルしました")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # スケール戻し（元画像座標に変換）
    src_points_orig = [(int(x / scale), int(y / scale)) for (x, y) in clicked_points]
    print(f"\n元画像座標: {src_points_orig}")

    # 4点を正規順序に並べる
    ordered = order_points(src_points_orig)
    print(f"並べ替え後: {ordered.tolist()}")

    # グリッド計算
    M, M_inv, centers_orig = compute_grid(ordered)

    # 確認表示
    grid_img = draw_grid(img, centers_orig, M=M)
    grid_disp = cv2.resize(grid_img, (disp_w, disp_h))
    print("\nグリッド確認画像を表示します。良ければ Enter、やり直しは q")
    cv2.namedWindow("Grid Check", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grid Check", disp_w, disp_h)
    cv2.imshow("Grid Check", grid_disp)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 13:
            break
        elif key == ord('q'):
            print("やり直してください。スクリプトを再実行してください。")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # マス切り出しプレビュー
    if args.preview_cells:
        warped, cells = extract_cells(img, M)
        warped_disp = cv2.resize(warped, (900, 900))
        print("パース補正後の盤面を表示します（Enter で閉じる）")
        cv2.imshow("Warped Board", warped_disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # JSON保存
    calib_data = {
        "image_path": str(img_path),
        "original_size": [w, h],
        "corner_points": ordered.tolist(),
        "perspective_matrix": M.tolist(),
        "perspective_matrix_inv": M_inv.tolist(),
        "cell_centers_original": centers_orig,
        "grid_size": 9,
        "cell_px": 100,
        "note": "row=0が上側(後手側)、col=0が右側(9筋)"
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, ensure_ascii=False, indent=2)

    print(f"\n[完了] キャリブレーションデータを保存しました: {out_path}")
    print("次のステップ: Step 2 駒認識（data1の初期配置を使用）")

if __name__ == "__main__":
    main()
