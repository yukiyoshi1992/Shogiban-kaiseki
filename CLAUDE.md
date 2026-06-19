# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A tool that converts a sequence of smartphone photos (one photo per move of a shogi board) into a KIF-format game record. Board recognition uses a MobileNetV2 CNN (29-class classification per square: empty + 14 sente piece types + 14 gote piece types). Board alignment uses red-circle markers (auto-calibration); board orientation uses a blue-triangle marker.

All code lives under `train/src/`; the top-level `src/` and `.webaxs/` directories are empty/unused. There is no requirements.txt — dependencies observed in the code: `torch`, `torchvision`, `opencv-python` (`cv2`), `numpy`, `Pillow`, `python-shogi` (imported as `shogi`), `scikit-learn` (`train_model.py` only), `openpyxl` (for reading/writing the `docs/01 要件定義/*.xlsx` requirement docs), `streamlit` + `streamlit-image-coordinates` (installed 2026-06-18 for the planned simple operator UI — see `docs/handover.md` §7; no app code written yet).

For full project history, requirements, and in-progress work, see `docs/引き継ぎ資料.md` (Japanese handover doc) — it is the authoritative source of truth for project status and should be consulted/updated alongside this file.

## Commands

All scripts are run directly with `python`, no build step. Paths in this repo are a Windows network share (`\\YukiYoshiNAS\Shogiban-kaiseki-tool\`); when invoked from a POSIX shell (bash tool) use `//YukiYoshiNAS/Shogiban-kaiseki-tool/...` instead of backslashes.

```bash
# Train the CNN (MobileNetV2 transfer learning) from labeled cell images
python train/src/train_model.py --cells "train/data/cells/" --out "train/models/" [--epochs 20] [--batch 16] [--val 0.2]

# Manual 4-corner calibration of a board photo
python train/src/calibrate.py --image <photo.jpg> --out <calib.json>

# Automatic calibration via red-circle markers
python train/src/auto_calibrate.py --image <photo.jpg> --out <calib.json>

# Batch manual calibration
python train/src/batch_calibrate.py --pattern <name> [--resume]

# Visually verify a calibration's grid alignment against photos
python train/src/check_calibration.py --calib <calib.json> --images <folder> --pattern <name>

# Detect blue-triangle orientation marker
python train/src/detect_triangle.py --image <photo.jpg>

# Debug: show 81-square recognition result for one image
python train/src/predict_cell.py --image <photo.jpg> --calib <calib.json> --model "train/models/"

# Label training data from a known piece-placement pattern (one calib may cover many photos of the same pattern)
python train/src/label_placement.py --images <folder> --calibs <calib_folder> --patterns-json train/src/placement_patterns.json --shift-json train/src/shift_patterns.json --out "train/data/cells/"

# Label training data from a KIF + game photos (reconstructs the correct board at each move and cross-checks against model recognition)
python train/src/label_from_kif.py --kif <game.kif> --images <folder> --calib <calib.json> --model "train/models/" --out "train/data/cells/" [--pattern <filename_prefix>] [--max-diff-accept 2] [--dry-run]

# Visually mass-review labeled cells before training
python train/src/check_labels.py --cells "train/data/cells/" [--samples N]

# Batch move-detection / KIF generation from a folder of sequential photos (testing/offline)
python train/src/detect_move.py --images <folder> --calib <calib.json> --model "train/models/" --out <output.kif>

# Production: watch a folder and generate KIF in real time as photos arrive
python train/src/run_realtime.py --watch "runtime/input/" --model "train/models/" --out "runtime/result/" [--idle-timeout 600] [--poll-interval 5]
```

There is no automated test suite; verification is done via `check_calibration.py`, `predict_cell.py`, `check_labels.py`, and by running `detect_move.py`/`run_realtime.py` against known games and comparing the resulting KIF to the ground truth.

## Architecture

### Coordinate system (must not be changed without updating every consumer)
- `row=0` is the top (9th file/筋), `row=8` is the bottom (1st file)
- `col=0` is the left (sente's first rank), `col=8` is the right (gote's side)
- `square = (8 - col) * 9 + row` — this is the mapping used everywhere code converts between the grid representation and `python-shogi`'s `Square` indices (see `rc_to_square`/`square_to_rc` in `label_from_kif.py` and the equivalent logic in `detect_move.py`).
- Training data orientation baseline: sente on the left, board shot landscape, sente pieces point right (▶), gote pieces point left (◀).

### The single most important invariant: rotate the image once, nothing else
`detect_move.py` and `run_realtime.py` determine orientation from the blue-triangle marker, then **rotate the source image itself** to match the landscape/sente-left baseline before any other processing. Calibration is then computed (or re-applied) against that already-rotated image. Grid coordinates and calibration matrices are never themselves rotated/transformed — only the raw image is. (An earlier version of `detect_move.py` had a "triple rotation" bug — rotating the image, the calib coordinates, *and* the grid labels — which silently broke recognition. Do not reintroduce per-orientation coordinate or grid transforms; the fix is documented in `docs/引き継ぎ資料.md` §4-1.)

Blue-triangle apex direction → required rotation to reach the landscape baseline:
| apex direction | board orientation | rotation applied |
|---|---|---|
| right | sente left, landscape (baseline) | none |
| up | sente nearest camera, portrait | rotate 90° CCW |
| down | sente far from camera, portrait | rotate 90° CW |
| left | sente right, landscape | rotate 180° |

### Production flow (`run_realtime.py`)
1. Watch folder for new images.
2. On the first (move-0) image: detect orientation via blue triangle → rotate if needed → auto-calibrate via red circles → compare recognized board to the standard starting position and ask the operator to confirm (`y`/`n`). On `n`, a human manually specifies orientation and performs manual 4-corner calibration — this fallback is intentional (automatic detection is the default path, manual is the safety net, not the other way around).
3. Every subsequent image is rotated using the orientation locked in at move 0, calibration is applied, and the new 81-square recognition is diffed against the previous board state to infer the move, which is validated against `python-shogi` legal moves and appended to the KIF.

### Piece classes and the promotion/demotion constraint
29 classes total: `empty` + 14 sente + 14 gote (`fu, kyo, kei, gin, kin, kaku, hi, ou, tokin, nari_kyo, nari_kei, nari_gin, uma, ryu`). Promoted pieces share their physical token with the unpromoted piece, so the two share a hard count cap when building/validating a placement pattern: lance-family (kyo+nari_kyo) ≤4, knight-family ≤4, silver-family ≤4, gold ≤4, bishop-family (kaku+uma) ≤2, rook-family (hi+ryu) ≤2, king ≤2, pawn-family (fu+tokin) ≤18 — and per-side (sente-only or gote-only) bishop/rook/king/uma/ryu are capped at 1 each. Any new placement-pattern definition (in `placement_patterns.json`/`shift_patterns.json`) must respect these caps or the labeled data will be physically impossible.

### label_placement.py: validate new pattern data before trusting it
When a new entry is added to `placement_patterns.json` or `shift_patterns.json`, `label_placement.py` will silently mislabel cells as `empty` for any symbol it doesn't recognize or any structure it doesn't expect — it does not fail loudly. Two concrete cases already hit this:
- The king can be written as either `王` or `玉` depending on who placed the physical piece; `SYMBOL_TO_LABEL` must contain both (`"S王"`/`"S玉"` → `sente_ou`, `"G王"`/`"G玉"` → `gote_ou`) or king squares silently become `empty`.
- `shift_patterns.json` entries come in two shapes: a flat 9-element list (one row of pieces, placed at `target_row`) for older patterns, and a dict `{"rows": N, "row0": [...], "row1": [...]}` for newer ones (`shift_all_S/G`, `shift_nari_S/G`) that photograph multiple board rows shifted together in one shot — `row{k}` is placed at `target_row + k`. `shift_pattern_to_label_grid()` must branch on `isinstance(row_def, dict)`; treating the dict as a flat iterable silently iterates over its keys instead of piece symbols and mislabels the entire row as `empty`.

Before trusting a label_placement.py run on a newly-added pattern, scan for unrecognized symbols across every pattern (compare every symbol in `placement_patterns.json`/`shift_patterns.json` against `SYMBOL_TO_LABEL.keys()`) rather than relying on the `[WARNING]` lines in console output — console output on this Windows setup is cp932-encoded and can be cut off/garbled when captured through a redirected file or background task buffer, which makes it easy to miss a warning that occurred earlier in a long batch run.

A third case (found 2026-06-18, visually): unlike a wrong/unrecognized symbol, a **wrong row count in a pattern definition produces no warning at all** — `label_placement.py` has no way to know the table's row layout doesn't match the photo. `pattern11` in `placement_patterns.json` had a spurious extra blank separator row (an editing mistake — the photo only has one blank row between the two `と` rows, not a blank row before *every* piece group like other patterns in the same batch), which silently shifted every row from there on by one: real 香/桂 cells got labeled `empty`, real 銀/金 cells got labeled 香/桂, real 王/角/飛 cells got labeled 銀/金, and real *empty* cells got labeled 王/角/飛 (`sente_ou`/`gote_ou`/etc.). This is only catchable by warping a sample photo with its calib, overlaying row/col indices (see the diagnostic technique below), and comparing row-by-row against the table — not by anything label_placement.py prints. When adding a new pattern, render one annotated/warped sample and manually diff it against the table before trusting any of its output, especially when patterns in the same batch don't all share the same blank-row convention.

Diagnostic snippet for verifying a pattern/calib pair's row alignment (used to find the pattern11 bug):
```python
import cv2
warped, cell_px, grid_size = warp_image(img, calib)  # or extract_cells() from label_placement.py
for r in range(grid_size):
    for c in range(grid_size):
        cv2.putText(warped, f'{r},{c}', (c*cell_px+3, r*cell_px+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
# save and view warped, compare row-by-row against the pattern table
```

### Calibration file convention
Calibration JSONs (`*_calib.json`) hold a `perspective_matrix`, `grid_size` (9), and `cell_px` (100), and are looked up by exact image stem (`{image_stem}_calib.json`) first. `label_placement.py`'s `find_calib()` falls back to a shared pattern-name calib (e.g. `pattern10a_calib.json`) when no per-image calib exists — this supports the "one calibration shared across many photos of the same fixed-camera pattern" shooting style used for production-room data (see `train/src/本番部屋_calib/`). Calib folders are organized by data batch (`追加学習_calib/`, `本番部屋_calib/`, `対局v2_calib/`, `対局_calib/`) — don't assume a single calib applies globally.

### label_from_kif.py: verified working (2026-06-18), with one data caveat
`load_model()`/`predict_board()` previously used the wrong classifier head (`meta["classes"]` instead of `meta["labels"]`, a plain `Linear` layer instead of the `Dropout→Linear(256)→ReLU→Dropout→Linear` head, 96px resize instead of 224px) — it didn't match `train_model.py`/`predict_cell.py`'s actual architecture, so `state_dict` loading or inference would have silently produced garbage. Now fixed and matches `predict_cell.py` exactly. `shogi.KIF.Parser` (the primary path, not the fallback parser) verified working end-to-end against the old 対局2 data (72/72 frames matched the KIF, 0 diff every time) and 002 (112/112, diff 0–1). Single calib-per-folder is sufficient even when only some images in a folder have a same-stem calib file (old 対局2 data: 60/72 images had a dedicated calib; the other 12 still resolved fine using the one shared calib passed via `--calib`).

001 (`学習データ（対局画像＋対局kif）3/001 ※001の画像を正しい方向に90度回転加工済のデータ/`) is the one exception: despite the folder name claiming the images are already rotated to the correct baseline orientation, they need an *additional* 180° rotation to align with `対局v2_calib/001_calib.json` (confirmed by warping move-0 with vs. without an extra 180° — diff 35 vs. diff 0). This looks like the calib was created with corners clicked in the opposite order/orientation than the images were saved in. A corrected copy was generated at `学習データ（対局画像＋対局kif）3/001_180fix/` (gitignored, like all of `train/data`) — use that folder, not the original, for any future 001 work. Even with the rotation fixed, only 3/79 frames pass the diff≤2 acceptance threshold for 001 — the rest are skipped because the old model's recognition accuracy is genuinely poor in that room/lighting (this is the known, already-documented reason a retrain was needed; it is not a bug in label_from_kif.py itself).

### Visual data review must cover every source batch, not just the recently-added one
When asked to visually verify `cells/` labels, it's easy to only sample the batch you were just working on and miss contamination elsewhere. `cells/` filenames carry a source prefix that identifies which batch produced them: `001`/`002` (対局v2), `対局` (old game data), `data1`–`data8` (old `data1`-style + `学習データ（最終場面画像＋対局kif）` endgame extraction), `pattern1`–`pattern11` (both placement-pattern batches), and `shift_*` (`shift_all/nari/baryu/kahigyoku/mix1/mix2/narifull`, each `_S`/`_G` and `_row{N}` suffixed). A 2026-06-18 review only checked `pattern9`–`11` and `shift_all/nari` (the batch just modified) and reported "no issues found" — `pattern1`–`8` and all of `data1`–`8` and the other `shift_*` patterns were silently never looked at. Before declaring a review complete, group `cells/<label>/*.jpg` by source prefix and confirm every prefix that exists for that label actually appears in the sample, not just a random draw that happens to be dominated by one or two batches.

### komadai (piece-stand) corner red dots are intentionally unused
Some training photos show red dots not just at the board's 4 corners but also on a small wooden block (駒台/piece stand) elsewhere in frame. Both `auto_calibrate.py` (`BOARD_RADIUS_RATIO_MIN/MAX`, `MIN_CIRCULARITY`) and `run_realtime.py`'s copy of `detect_red_circles`/`select_board_corners` deliberately filter circles by radius specifically so the board's (intentionally larger) red dots are picked and the komadai's smaller ones are excluded as noise. The komadai dots are not a calibration input anywhere — they can be physically removed from the prop without affecting any code.

### Model training takes hours on this CPU-only setup
`train_model.py` has no GPU available here (`torch.cuda.is_available()` is False), and unfreezes the full MobileNetV2 backbone for fine-tuning after epoch 10 (`if epoch == 9: ... param.requires_grad = True`), which roughly triples per-epoch time from there. A 20-epoch run on the current ~33k-image `cells/` took **~11h40m** wall-clock (epochs 1–10 ~30–38 min each, epochs 11–20 ~45–57 min each). `print()` output is fully OS-buffered when the process is backgrounded/redirected on this Windows setup — nothing appears in the log file until the process exits, so don't expect to tail progress; instead poll `train/models/best_model.pth`'s mtime (saved every time validation accuracy improves, not every epoch) or just wait for the background-task completion notification. `train/models/` (unlike `train/data/`) is **not** gitignored — `best_model.pth`/`model_meta.json` are tracked in git, so the previous model is always recoverable from git history. Still, before retraining it's convenient to also copy the current `train/models/{best_model.pth,model_meta.json}` to an untracked dated folder (e.g. `train/models_backup_YYYYMMDD/`) for a fast local rollback without git commands, and commit the newly trained weights once validated.

### Japanese-path / encoding handling
- Image I/O must go through `cv2.imdecode(np.fromfile(path, dtype=np.uint8), ...)` / `cv2.imencode(...)[1].tofile(path)` instead of `cv2.imread`/`cv2.imwrite`, because the latter don't handle non-ASCII (Japanese) paths on Windows.
- Console/log output is sometimes cp932-encoded; mojibake in captured stdout (e.g. background task logs) usually just needs re-decoding, not re-running.
- Rotated copies of source images may get written back into the same folder as `*_rot.jpg` during processing — don't let these leak into a later batch run as if they were new source photos.

### detect_move.py: end-to-end retest after 3rd retrain (2026-06-19), and a KIF-output bug found/fixed
Re-ran the full production pipeline (`detect_move.py`: blue-triangle orientation detect → rotate → calibrate → per-frame recognition → diff → legal-move match → KIF) against two `学習データ（対局画像＋対局kif）3` folders, using the 3rd-retrained model (99.94% val accuracy):
- `002/` (already contributed 112 cells to training): **111/111 moves correct, 0 errors** — triangle direction auto-detected as `right` (no rotation needed), confirmed by independently re-parsing the generated KIF with `shogi.KIF.Parser` and diffing against the ground-truth KIF move list (exact match).
- `001 ※画像が90度回転して入ってきている場合のパターン/` (the **raw**, never-used-for-training copy of the 001 game — distinct from `001 ※001の画像を正しい方向に90度回転加工済のデータ/`, which is the *same* KIF/game but a manually pre-rotated copy that did contribute a few cells previously): **55/78 moves correct, 23 errors**. Triangle auto-detected as `up` (rotate 90° CCW) — confirmed correct by visually checking the warped move-0 board against the standard initial position. Errors cluster in two places: 3 isolated failures right at the start (moves 1, 3, 4 — an early single-frame recognition glitch that self-corrects), then a long clean run (moves 5–52), then a dense failure block from move 53 onward (53–58, 60–63, 65–70, 72–75) that the diff-based algorithm cannot self-recover from once it starts (a failed frame freezes the tracked board state, so the next comparison must explain *two* real moves at once, which usually has no legal match either — failures cascade). This is a genuine model/lighting accuracy gap specific to this room's data, not a pipeline bug — consistent with the already-documented poor recognition for the 001 game's room/lighting (see below). In a real production run (`run_realtime.py`), this game would have triggered the "recognition error → auto-abort" path at the very first failure, not run to completion.
- `auto_calibrate.py`'s red-circle detector failed (only found 2/4 corners) on the rotated 001 move-0 image — two corners had circularity scores (0.32–0.34) below `MIN_CIRCULARITY` (0.45), likely from partial occlusion/glare, even though their radius was in-range. Worked around by manually reading the 4 circle centers from the unfiltered contour list (`detect_red_circles`'s pre-filter output) and building the calib directly via `auto_calibrate.compute_grid()`/`save_json()` — `auto_calibrate.py` itself was not modified. If this recurs often, the fix would be loosening `MIN_CIRCULARITY` or detecting circles independent of the strict radius+circularity AND, but a single one-off manual override was simplest here.
- Found and fixed a **KIF-output bug** in `detect_move.py`'s `moves_to_kif()`: the move-source square in parentheses (e.g. `(27)`) was being built from `sq_name_to_ja()`'s *Japanese* (full-width file digit + kanji rank) representation reverse-engineered back into a number, producing a mixed full-width-file + half-width-rank string (e.g. `(４1)`) instead of the correct all-half-width `(41)`. This isn't just cosmetic: `shogi.KIF.Parser` re-parsing such a file silently stops at the first move it can't match (no exception raised) — confirmed on the 002 output, where only 24/111 moves parsed back before silent truncation. Fixed by computing the source square's file/rank as plain ASCII digits directly from the USI square name instead of round-tripping through the kanji string. Verified fix: re-parsed 002's regenerated KIF matches the 111-move ground truth exactly. Any KIF file produced by `detect_move.py` *before* this fix should be treated as suspect/unverified if it was meant to be re-parsed by another tool.

### Git on this network share
The repo lives on a Windows network share (`\\YukiYoshiNAS\...`); git may report "detected dubious ownership" depending on the session's user context. Do not run `git config --global --add safe.directory` from an agent session — this changes global git config, which is against this repo's working agreement. Ask the user to run it themselves if needed.
