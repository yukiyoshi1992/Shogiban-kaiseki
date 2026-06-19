"""
駒認識推論・検証スクリプト
- 学習済みモデルで盤面画像を認識
- KIFの正解と照合して精度を確認
- 誤認識マスを視覚的に表示

使い方:
  # 単一画像の推論（KIFなし）
  python predict_cell.py \
    --model  "//YukiYoshiNAS/Shogiban-kaiseki-tool/src/models/" \
    --image  "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/data2.jpg" \
    --calib  "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/data2_calib.json"

  # KIFと照合して精度確認
  python predict_cell.py \
    --model  "//YukiYoshiNAS/Shogiban-kaiseki-tool/src/models/" \
    --image  "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/data2.jpg" \
    --calib  "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/data2_calib.json" \
    --kif    "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/data2.kif"

  # フォルダ内の全画像を一括検証
  python predict_cell.py \
    --model   "//YukiYoshiNAS/Shogiban-kaiseki-tool/src/models/" \
    --images  "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/data/" \
    --calibs  "//YukiYoshiNAS/Shogiban-kaiseki-tool/train/src/" \
    --pattern "data2"
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import json
import argparse
import sys
from pathlib import Path
import shogi
import shogi.KIF

# -------------------------------------------------------
# ラベル定義
# -------------------------------------------------------
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
NUM_CLASSES = len(ALL_LABELS)

PIECE_TO_LABEL = {
    shogi.PAWN:        "fu",
    shogi.LANCE:       "kyo",
    shogi.KNIGHT:      "kei",
    shogi.SILVER:      "gin",
    shogi.GOLD:        "kin",
    shogi.BISHOP:      "kaku",
    shogi.ROOK:        "hi",
    shogi.KING:        "ou",
    shogi.PROM_PAWN:   "tokin",
    shogi.PROM_LANCE:  "nari_kyo",
    shogi.PROM_KNIGHT: "nari_kei",
    shogi.PROM_SILVER: "nari_gin",
    shogi.PROM_BISHOP: "uma",
    shogi.PROM_ROOK:   "ryu",
}

# -------------------------------------------------------
# モデル読み込み
# -------------------------------------------------------
def load_model(model_folder):
    model_path = Path(model_folder) / "best_model.pth"
    meta_path  = Path(model_folder) / "model_meta.json"

    if not model_path.exists():
        print(f"[ERROR] モデルが見つかりません: {model_path}")
        sys.exit(1)

    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"モデル読み込み完了: {model_path}")
    return model

# -------------------------------------------------------
# 推論
# -------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_cells(model, cells):
    """cells: list of (row, col, img_bgr)"""
    from PIL import Image
    results = []
    for row, col, cell_img in cells:
        img_rgb = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = probs.max(1)
        label = ALL_LABELS[pred.item()]
        confidence = conf.item()
        results.append((row, col, label, confidence))
    return results

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

def load_calibration(calib_path):
    with open(calib_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    M = np.array(data["perspective_matrix"], dtype="float32")
    return M, data.get("grid_size", 9), data.get("cell_px", 100)

def load_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)

# -------------------------------------------------------
# KIFから正解グリッドを生成
# -------------------------------------------------------
def kif_to_label_grid(kif_path):
    kif = shogi.KIF.Parser.parse_file(str(kif_path))
    board = shogi.Board()
    for move_str in kif[0]['moves']:
        board.push_usi(move_str)

    grid = [["empty"] * 9 for _ in range(9)]
    for sq in range(81):
        piece = board.piece_at(sq)
        if piece is None:
            continue
        row = sq % 9
        col = 8 - (sq // 9)
        piece_label = PIECE_TO_LABEL.get(piece.piece_type)
        if piece_label is None:
            continue
        side = "sente" if piece.color == shogi.BLACK else "gote"
        grid[row][col] = f"{side}_{piece_label}"
    return grid

# -------------------------------------------------------
# 結果の可視化
# -------------------------------------------------------
def visualize_result(warped, results, correct_grid=None, cell_px=100, grid_size=9):
    """
    補正済み盤面に認識結果を重ねて表示
    正解グリッドがある場合: 正解=緑枠、誤認識=赤枠
    """
    vis = warped.copy()
    errors = []

    for row, col, pred_label, conf in results:
        x1 = col * cell_px
        y1 = row * cell_px

        if correct_grid is not None:
            correct = correct_grid[row][col]
            is_correct = (pred_label == correct)
            color = (0, 200, 0) if is_correct else (0, 0, 255)
            thickness = 1 if is_correct else 2
            if not is_correct:
                errors.append((row, col, pred_label, correct, conf))
        else:
            color = (0, 200, 0)
            thickness = 1

        cv2.rectangle(vis, (x1, y1), (x1+cell_px, y1+cell_px), color, thickness)

        # ラベル表示（短縮形）
        short = pred_label.replace("sente_","S:").replace("gote_","G:").replace("empty",".")
        cv2.putText(vis, short, (x1+2, y1+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

        # 信頼度（低い場合のみ表示）
        if conf < 0.9:
            cv2.putText(vis, f"{conf:.0%}", (x1+2, y1+28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 100, 200), 1)

    return vis, errors

# -------------------------------------------------------
# 単一画像の処理
# -------------------------------------------------------
def process_one(model, img_path, calib_path, kif_path=None, show=True):
    img = load_image(img_path)
    if img is None:
        print(f"[ERROR] 画像読み込み失敗: {img_path}")
        return None

    M, grid_size, cell_px = load_calibration(calib_path)
    warped, cells = extract_cells(img, M, grid_size, cell_px)
    results = predict_cells(model, cells)

    correct_grid = None
    if kif_path and Path(kif_path).exists():
        correct_grid = kif_to_label_grid(kif_path)

    vis, errors = visualize_result(warped, results, correct_grid, cell_px, grid_size)

    # 精度計算
    if correct_grid is not None:
        total = grid_size * grid_size
        correct_count = total - len(errors)
        acc = correct_count / total * 100
        print(f"\n認識精度: {correct_count}/{total} = {acc:.1f}%")
        if errors:
            print(f"誤認識 {len(errors)} マス:")
            for row, col, pred, correct, conf in errors:
                print(f"  r{row}c{col}: 予測={pred} 正解={correct} 信頼度={conf:.1%}")
        else:
            print("全マス正解!")
    else:
        print(f"\n推論完了: {Path(img_path).name}")
        for row, col, label, conf in results:
            if label != "empty":
                print(f"  r{row}c{col}: {label} ({conf:.1%})")

    if show:
        cv2.namedWindow("Prediction", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Prediction", 720, 720)
        title = Path(img_path).name
        if correct_grid:
            title += f" | 赤=誤認識 緑=正解 | ESCで次へ"
        cv2.setWindowTitle("Prediction", title)
        cv2.imshow("Prediction", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results, errors if correct_grid else None

# -------------------------------------------------------
# メイン
# -------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="駒認識推論・検証")
    parser.add_argument("--model",   required=True, help="モデルフォルダ")
    # 単一画像モード
    parser.add_argument("--image",  help="画像パス")
    parser.add_argument("--calib",  help="calibration JSONパス")
    parser.add_argument("--kif",    help="KIFパス（照合用）")
    # バッチモード
    parser.add_argument("--images",  help="画像フォルダ")
    parser.add_argument("--calibs",  help="calibフォルダ")
    parser.add_argument("--pattern", default="data2", help="対象パターン")
    parser.add_argument("--no-show", action="store_true", help="表示しない")
    args = parser.parse_args()

    model = load_model(args.model)

    # 単一画像モード
    if args.image:
        if not args.calib:
            print("[ERROR] --calib が必要です")
            sys.exit(1)
        process_one(model, args.image, args.calib, args.kif, show=not args.no_show)
        return

    # バッチモード
    if args.images:
        images_folder = Path(args.images)
        calibs_folder = Path(args.calibs) if args.calibs else images_folder

        image_files = sorted(set(
            list(images_folder.glob(f"{args.pattern}*.jpg")) +
            list(images_folder.glob(f"{args.pattern}*.JPG"))
        ))
        # data0,data1はスキップ
        image_files = [f for f in image_files
                       if not f.stem.startswith("data0")
                       and not f.stem.startswith("data1")]

        if not image_files:
            print(f"[ERROR] 対象画像なし: {args.pattern}")
            sys.exit(1)

        total_correct = 0
        total_cells = 0
        all_errors = []

        for img_path in image_files:
            calib_path = calibs_folder / f"{img_path.stem}_calib.json"
            if not calib_path.exists():
                print(f"[SKIP] calib not found: {calib_path.name}")
                continue

            # KIFを探す
            data_name = img_path.stem.split(" ")[0].strip()
            kif_path = images_folder / f"{data_name}.kif"
            kif_path = str(kif_path) if kif_path.exists() else None

            print(f"\n--- {img_path.name} ---")
            ret = process_one(model, img_path, calib_path, kif_path,
                              show=not args.no_show)
            if ret and ret[1] is not None:
                results, errors = ret
                total_cells += 81
                total_correct += 81 - len(errors)
                all_errors.extend([(img_path.name, *e) for e in errors])

        if total_cells > 0:
            print(f"\n{'='*55}")
            print(f"全体精度: {total_correct}/{total_cells} = {total_correct/total_cells*100:.1f}%")
            if all_errors:
                print(f"誤認識合計: {len(all_errors)} マス")
                # 誤認識パターンの集計
                from collections import Counter
                patterns = Counter([(e[2], e[3]) for e in all_errors])
                print("多い誤認識パターン（予測→正解）:")
                for (pred, correct), cnt in patterns.most_common(10):
                    print(f"  {pred} → {correct}: {cnt}回")

if __name__ == "__main__":
    main()