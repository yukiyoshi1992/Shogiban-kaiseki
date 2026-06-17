"""
成り駒学習データ 自動ラベル付けツール
- placement_patterns.json の配置情報を使って、撮影画像から81マスを切り出し
- 各マスに正解ラベルを自動付与して cells/ に追加

前提:
- 撮影画像は「先手が左」の向き（学習データと同じ）
- 各画像のファイル名にパターン番号が含まれる、または順番で対応

使い方:
  # パターン番号を明示する場合（推奨）
  python label_placement.py \
    --image    "//.../train/data/成り駒/pattern1 (1).jpg" \
    --pattern  1 \
    --calib    "//.../train/src/成り駒_calib/pattern1 (1)_calib.json" \
    --patterns-json "//.../train/src/placement_patterns.json" \
    --out      "//.../train/data/cells/"

  # フォルダ一括（ファイル名が pattern{N} で始まる前提）
  python label_placement.py \
    --images   "//.../train/data/成り駒/" \
    --calibs   "//.../train/src/成り駒_calib/" \
    --patterns-json "//.../train/src/placement_patterns.json" \
    --out      "//.../train/data/cells/" \
    --preview
"""

import cv2
import numpy as np
import json
import argparse
import sys
import re
from pathlib import Path

# -------------------------------------------------------
# 駒記号 → ラベル変換
# -------------------------------------------------------
SYMBOL_TO_LABEL = {
    "S香":"sente_kyo","S桂":"sente_kei","S銀":"sente_gin","S金":"sente_kin",
    "S角":"sente_kaku","S飛":"sente_hi","S玉":"sente_ou","S歩":"sente_fu",
    "Sと":"sente_tokin","S杏":"sente_nari_kyo","S圭":"sente_nari_kei",
    "S全":"sente_nari_gin","S馬":"sente_uma","S龍":"sente_ryu",
    "G香":"gote_kyo","G桂":"gote_kei","G銀":"gote_gin","G金":"gote_kin",
    "G角":"gote_kaku","G飛":"gote_hi","G玉":"gote_ou","G歩":"gote_fu",
    "Gと":"gote_tokin","G杏":"gote_nari_kyo","G圭":"gote_nari_kei",
    "G全":"gote_nari_gin","G馬":"gote_uma","G龍":"gote_ryu",
    ".":"empty",
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

def load_calibration(calib_path):
    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    M = np.array(data["perspective_matrix"], dtype="float32")
    return M, data.get("grid_size", 9), data.get("cell_px", 100)

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
            cells.append((row, col, warped[y1:y1+cell_px, x1:x1+cell_px]))
    return warped, cells

def pattern_to_label_grid(pattern_grid):
    """配置記号グリッド → ラベルグリッド"""
    grid = []
    for row in pattern_grid:
        label_row = []
        for sym in row:
            label = SYMBOL_TO_LABEL.get(sym)
            if label is None:
                print(f"  [WARNING] 未知の駒記号: '{sym}' → empty扱い")
                label = "empty"
            label_row.append(label)
        grid.append(label_row)
    return grid

def detect_pattern_num(filename):
    """ファイル名からパターン番号を抽出（pattern3, p3, 3 などに対応）"""
    m = re.search(r'pattern\s*(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'p(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def detect_shift_pattern(filename):
    """
    ずらし撮影のファイル名を解析
    例: shift_baryu_S_row0.jpg → ("shift_baryu_S", 0)
    Returns: (shift_pattern_name, row) or (None, None)
    """
    m = re.search(r'(shift_[a-zA-Z0-9_]+?)_row(\d+)', filename, re.IGNORECASE)
    if m:
        return m.group(1), int(m.group(2))
    return None, None

def shift_pattern_to_label_grid(row_def, target_row):
    """
    1行ぶんの駒定義を指定行に配置した9x9ラベルグリッドを作る
    row_def: 9要素の駒記号リスト（1行ぶん）
    target_row: 0-8、この行に駒を置く
    """
    grid = [["empty"]*9 for _ in range(9)]
    for col, sym in enumerate(row_def):
        label = SYMBOL_TO_LABEL.get(sym, "empty")
        grid[target_row][col] = label
    return grid

def find_calib(calibs_folder, img_stem):
    p = Path(calibs_folder) / f"{img_stem}_calib.json"
    return p if p.exists() else None

def process_image(img_path, calib_path, label_grid, out_folder, preview=False):
    M, grid_size, cell_px = load_calibration(calib_path)
    img = load_image(img_path)
    if img is None:
        print(f"  [SKIP] 読み込み失敗: {img_path.name}")
        return 0

    warped, cells = extract_cells(img, M, grid_size, cell_px)
    saved = 0
    for row, col, cell_img in cells:
        label = label_grid[row][col]
        filename = f"{img_path.stem}_r{row}c{col}.jpg"
        save_path = out_folder / label / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".jpg", cell_img)[1].tofile(str(save_path))
        saved += 1

    if preview:
        prev = warped.copy()
        for row in range(grid_size):
            for col in range(grid_size):
                label = label_grid[row][col]
                x1, y1 = col*cell_px, row*cell_px
                cv2.rectangle(prev, (x1,y1), (x1+cell_px,y1+cell_px), (0,180,0), 1)
                short = label.replace("sente_","S:").replace("gote_","G:").replace("empty",".")
                cv2.putText(prev, short, (x1+2,y1+16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.26, (0,0,200), 1)
        cv2.namedWindow("Label Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Label Preview", 720, 720)
        cv2.setWindowTitle("Label Preview", f"{img_path.name} | Enter:次 ESC:プレビュー終了")
        cv2.imshow("Label Preview", prev)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return -saved  # プレビュー終了の合図
    return saved

def main():
    parser = argparse.ArgumentParser(description="成り駒学習データ 自動ラベル付け")
    parser.add_argument("--image", help="単一画像")
    parser.add_argument("--images", help="画像フォルダ")
    parser.add_argument("--pattern", type=int, help="パターン番号（単一画像時）")
    parser.add_argument("--calib", help="calibration JSON（単一画像時）")
    parser.add_argument("--calibs", help="calibrationフォルダ")
    parser.add_argument("--patterns-json", required=True, help="placement_patterns.json")
    parser.add_argument("--shift-json", help="shift_patterns.json（ずらし撮影用）")
    parser.add_argument("--out", required=True, help="出力cells/フォルダ")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    # 配置パターン読み込み
    with open(args.patterns_json, "r", encoding="utf-8") as f:
        patterns = json.load(f)

    # ずらしパターン読み込み（あれば）
    shift_patterns = {}
    if args.shift_json:
        with open(args.shift_json, "r", encoding="utf-8") as f:
            shift_patterns = json.load(f)

    out_folder = Path(args.out)
    for label in ALL_LABELS:
        (out_folder / label).mkdir(parents=True, exist_ok=True)

    total = 0

    # 単一画像モード
    if args.image:
        if args.pattern is None or not args.calib:
            print("[ERROR] --pattern と --calib が必要です")
            sys.exit(1)
        pkey = f"pattern{args.pattern}"
        if pkey not in patterns:
            print(f"[ERROR] {pkey} が patterns-json にありません")
            sys.exit(1)
        label_grid = pattern_to_label_grid(patterns[pkey])
        n = process_image(Path(args.image), Path(args.calib), label_grid,
                          out_folder, args.preview)
        total += abs(n)
        print(f"保存: {abs(n)} マス")

    # フォルダ一括モード
    elif args.images:
        images_folder = Path(args.images)
        calibs_folder = Path(args.calibs) if args.calibs else images_folder
        image_files = sorted(set(
            list(images_folder.glob("*.jpg")) + list(images_folder.glob("*.JPG"))
        ))
        if not image_files:
            print(f"[ERROR] 画像が見つかりません: {images_folder}")
            sys.exit(1)

        print(f"対象画像: {len(image_files)} 枚")
        preview = args.preview
        for img_path in image_files:
            # まずずらし撮影か判定
            shift_name, shift_row = detect_shift_pattern(img_path.name)
            if shift_name is not None:
                if shift_name not in shift_patterns:
                    print(f"  [SKIP] ずらしパターン '{shift_name}' が定義にありません: {img_path.name}")
                    continue
                calib_path = find_calib(calibs_folder, img_path.stem)
                if calib_path is None:
                    print(f"  [SKIP] calib なし: {img_path.stem}_calib.json")
                    continue
                label_grid = shift_pattern_to_label_grid(shift_patterns[shift_name], shift_row)
                n = process_image(img_path, calib_path, label_grid, out_folder, preview)
                if n < 0:
                    preview = False; n = -n
                total += n
                print(f"  {img_path.name} (ずらし {shift_name} 行{shift_row}): {n} マス保存")
                continue

            # 通常パターン
            pnum = detect_pattern_num(img_path.name)
            if pnum is None:
                print(f"  [SKIP] パターン番号を特定できません: {img_path.name}")
                continue
            pkey = f"pattern{pnum}"
            if pkey not in patterns:
                print(f"  [SKIP] {pkey} が定義にありません: {img_path.name}")
                continue
            calib_path = find_calib(calibs_folder, img_path.stem)
            if calib_path is None:
                print(f"  [SKIP] calib なし: {img_path.stem}_calib.json")
                continue

            label_grid = pattern_to_label_grid(patterns[pkey])
            n = process_image(img_path, calib_path, label_grid, out_folder, preview)
            if n < 0:
                preview = False
                n = -n
            total += n
            print(f"  {img_path.name} (パターン{pnum}): {n} マス保存")

        if preview:
            cv2.destroyAllWindows()

    else:
        print("[ERROR] --image または --images が必要です")
        sys.exit(1)

    # サマリー
    print(f"\n{'='*50}")
    print(f"合計 {total} マス保存")
    print("\nラベル別 合計枚数（cells全体）:")
    for label in ALL_LABELS:
        cnt = len(list((out_folder / label).glob("*.jpg")))
        if cnt > 0:
            print(f"  {label:<22}: {cnt} 枚")

if __name__ == "__main__":
    main()
