# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A tool that converts a sequence of smartphone photos (one photo per move of a shogi board) into a KIF-format game record. Board recognition uses a MobileNetV2 CNN (29-class classification per square: empty + 14 sente piece types + 14 gote piece types). Board alignment uses red-circle markers (auto-calibration); board orientation uses a blue-triangle marker.

All code lives under `train/src/`; the top-level `src/` and `.webaxs/` directories are empty/unused. There is no requirements.txt — dependencies observed in the code: `torch`, `torchvision`, `opencv-python` (`cv2`), `numpy`, `Pillow`, `python-shogi` (imported as `shogi`), `scikit-learn` (`train_model.py` only).

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

### Calibration file convention
Calibration JSONs (`*_calib.json`) hold a `perspective_matrix`, `grid_size` (9), and `cell_px` (100), and are looked up by exact image stem (`{image_stem}_calib.json`) first. `label_placement.py`'s `find_calib()` falls back to a shared pattern-name calib (e.g. `pattern10a_calib.json`) when no per-image calib exists — this supports the "one calibration shared across many photos of the same fixed-camera pattern" shooting style used for production-room data (see `train/src/本番部屋_calib/`). Calib folders are organized by data batch (`追加学習_calib/`, `本番部屋_calib/`, `対局v2_calib/`, `対局_calib/`) — don't assume a single calib applies globally.

### Japanese-path / encoding handling
- Image I/O must go through `cv2.imdecode(np.fromfile(path, dtype=np.uint8), ...)` / `cv2.imencode(...)[1].tofile(path)` instead of `cv2.imread`/`cv2.imwrite`, because the latter don't handle non-ASCII (Japanese) paths on Windows.
- Console/log output is sometimes cp932-encoded; mojibake in captured stdout (e.g. background task logs) usually just needs re-decoding, not re-running.
- Rotated copies of source images may get written back into the same folder as `*_rot.jpg` during processing — don't let these leak into a later batch run as if they were new source photos.

### Git on this network share
The repo lives on a Windows network share (`\\YukiYoshiNAS\...`); git may report "detected dubious ownership" depending on the session's user context. Do not run `git config --global --add safe.directory` from an agent session — this changes global git config, which is against this repo's working agreement. Ask the user to run it themselves if needed.
