"""
モデル認識結果とKIF正解を全フレーム・全マスで比較し、個別の誤読を網羅的に洗い出す診断ツール。

detect_move.pyのclassify_frameは「直前の差分から1〜2手先まで合法手探索」という設計のため、
一度合法手が見つからず失敗すると盤面が凍結し、以降のフレームは正しい認識をしていても
「正解との差分が雪だるま式に増える」だけの無意味な比較になる（カスケード）。
本スクリプトはそれとは無関係に、各フレームをKIFを再生して得た「その時点の正解局面」と
直接比較するため、カスケードの影響を受けずに「本当にモデルが読み間違えたマス」だけを
洗い出せる。

使い方:
  python audit_recognition.py \
    --kif    "...001.kif" \
    --images "...001 ※画像が90度回転して入ってきている場合のパターン/" \
    --calib  "...001raw_calib.json" \
    --model  "...train/models/" \
    --out-pkl "...audit_001raw.pkl"
"""

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from label_from_kif import load_kif_boards
from detect_move import (
    load_model, predict_board, load_image, warp_image,
    detect_blue_triangle_direction, rotate_image_for_direction,
)


def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def main():
    parser = argparse.ArgumentParser(description="フレーム単位の認識精度監査（KIF正解との直接比較）")
    parser.add_argument("--kif", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pattern", default="IMG_")
    parser.add_argument("--force-direction", choices=["right", "left", "up", "down"])
    parser.add_argument("--out-pkl", help="結果をpickleで保存（後で再分析用）")
    args = parser.parse_args()

    print(f"KIF読み込み中: {args.kif}")
    kif_grids = load_kif_boards(args.kif)
    print(f"  {len(kif_grids)} 局面（0手目〜{len(kif_grids)-1}手目）")

    print("モデル読み込み中...")
    model = load_model(args.model)

    with open(args.calib, "r", encoding="utf-8") as f:
        calib = json.load(f)

    images_dir = Path(args.images)
    image_files = sorted(set(
        list(images_dir.glob(f"{args.pattern}*.jpg")) +
        list(images_dir.glob(f"{args.pattern}*.JPG"))
    ), key=natural_key)
    print(f"対象画像: {len(image_files)} 枚")

    if args.force_direction:
        img_dir = args.force_direction
        print(f"画像回転方向: {img_dir}（手動指定）")
    else:
        first_img = load_image(image_files[0])
        img_dir = detect_blue_triangle_direction(first_img) if first_img is not None else None
        if img_dir is None:
            img_dir = "right"
            print("[WARNING] 青三角を検出できませんでした → 回転なし(right)扱い")
        else:
            print(f"画像回転方向: {img_dir}（青三角検出）")

    n = min(len(image_files), len(kif_grids))
    if len(image_files) != len(kif_grids):
        print(f"[WARNING] 画像枚数({len(image_files)})とKIF局面数({len(kif_grids)})が一致しません。"
              f"先頭{n}件のみ比較します。")

    results = []
    correct_confs = []
    wrong_confs = []

    for i in range(n):
        img = load_image(image_files[i])
        if img is None:
            print(f"  [{i}/{n-1}] [ERROR] 読み込み失敗: {image_files[i].name}")
            continue
        img_r = rotate_image_for_direction(img, img_dir)
        warped, cell_px, grid_size = warp_image(img_r, calib)
        labels, confs = predict_board(model, warped, cell_px, grid_size)
        truth = kif_grids[i]

        mismatches = []
        for r in range(9):
            for c in range(9):
                conf = confs[r][c]
                if labels[r][c] == truth[r][c]:
                    correct_confs.append(conf)
                else:
                    wrong_confs.append(conf)
                    mismatches.append((r, c, truth[r][c], labels[r][c], conf))

        results.append({
            "index": i,
            "file": image_files[i].name,
            "diff": len(mismatches),
            "mismatches": mismatches,
        })
        marker = "" if len(mismatches) == 0 else f"  <-- {len(mismatches)}マス不一致"
        print(f"  [{i:3d}/{n-1}] {image_files[i].name}: diff={len(mismatches)}{marker}")

    if args.out_pkl:
        with open(args.out_pkl, "wb") as f:
            pickle.dump(results, f)
        print(f"\n結果をpickle保存: {args.out_pkl}")

    # ---- サマリー ----
    print(f"\n{'='*60}")
    total_mismatch = sum(r["diff"] for r in results)
    perfect_frames = sum(1 for r in results if r["diff"] == 0)
    print(f"完全一致フレーム: {perfect_frames}/{len(results)}")
    print(f"延べ不一致マス数: {total_mismatch}")

    cell_counter = Counter()
    for r in results:
        for (row, col, t, p, conf) in r["mismatches"]:
            cell_counter[(row, col)] += 1
    print(f"\n=== 不一致頻度が高いマス(上位20 / 全{len(results)}フレーム中) ===")
    for (row, col), cnt in cell_counter.most_common(20):
        print(f"  row{row},col{col}: {cnt}回")

    if correct_confs:
        print(f"\n正解時の平均確信度: {sum(correct_confs)/len(correct_confs):.4f} (n={len(correct_confs)})")
    if wrong_confs:
        print(f"誤読時の平均確信度: {sum(wrong_confs)/len(wrong_confs):.4f} (n={len(wrong_confs)})")
        low_conf_wrong = sum(1 for c in wrong_confs if c < 0.9)
        print(f"  誤読のうち確信度<0.9: {low_conf_wrong}/{len(wrong_confs)} "
              f"({low_conf_wrong/len(wrong_confs):.1%})")
        high_conf_wrong = sum(1 for c in wrong_confs if c >= 0.99)
        print(f"  誤読のうち確信度>=0.99（自信満々の誤読）: {high_conf_wrong}/{len(wrong_confs)} "
              f"({high_conf_wrong/len(wrong_confs):.1%})")


if __name__ == "__main__":
    main()
