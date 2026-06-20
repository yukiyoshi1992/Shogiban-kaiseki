"""
本番運用画面（画面要件.xlsx準拠）: 4ボタン（対局準備/対局開始/対局終了/対局中止）+メッセージwindow

判定ロジック（向き判定・赤丸キャリブレーション・classify_frame・KIF出力など）は
run_realtime.py の関数をそのまま import して再利用する（ロジックの二重管理を避けるため）。
run_realtime.py自体はCLI単体動作用として維持し、こちらは画面のみを担当する。

起動方法:
  streamlit run src/app_streamlit.py

KIFファイル名の状態遷移（画面要件.xlsx準拠。コロンはWindowsのファイル名に使えないため
仕様の "yyyyMMdd_hh:mm" を "yyyyMMdd_hh-mm" に変更している）:
  対局開始 → 【対局中】yyyyMMdd_hh-mm.kif
  対局終了 → 【対局完了】...
  対局中止 → 【対局中止】...
  認識エラーで続行不能（画面要件No.6） → 【対局エラー中止】...
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import shogi
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_realtime as rr  # noqa: E402  (sys.path調整後にimportする必要があるため)

REPO_ROOT = Path(__file__).resolve().parents[1]
WATCH_DIR = REPO_ROOT / "runtime" / "input"
OUT_DIR = REPO_ROOT / "runtime" / "result"
MODEL_DIR = REPO_ROOT / "src" / "models"
POLL_INTERVAL_SEC = 5

DIRECTION_LABELS = {
    "right": "right: 横向き・先手が左（回転なし）",
    "left": "left: 横向き・先手が右（180度回転）",
    "up": "up: 縦向き・先手が下（反時計90度）",
    "down": "down: 縦向き・先手が上（時計90度）",
}

st.set_page_config(page_title="将棋盤画像解析", layout="wide")


# ===== モデル（プロセス内で1回だけ読み込む） =====
@st.cache_resource
def get_model():
    return rr.load_model(str(MODEL_DIR))


# ===== session_state初期化 =====
def init_state():
    defaults = dict(
        phase="idle",  # idle -> ready -> playing -> idle(終了/中止/エラー中止後)
        message="「対局準備」ボタンを押してください。",
        calib_M=None,
        direction=None,
        board=None,
        moves_usi=[],
        kif_path=None,
        processed=set(),
        pending_finish_prefix=None,  # "finishing"中に保留している対局終了後のプレフィックス
        manual_step=None,       # None / "direction" / "corners"
        manual_direction=None,
        manual_raw_img=None,
        manual_points=[],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_game_state():
    st.session_state.phase = "idle"
    st.session_state.calib_M = None
    st.session_state.direction = None
    st.session_state.board = None
    st.session_state.moves_usi = []
    st.session_state.kif_path = None
    st.session_state.processed = set()
    st.session_state.pending_finish_prefix = None
    st.session_state.manual_step = None
    st.session_state.manual_direction = None
    st.session_state.manual_raw_img = None
    st.session_state.manual_points = []


# ===== 対局準備: 向き判定→赤丸キャリブレーション→初期配置照合を自動実施 =====
def find_latest_image(folder: Path):
    files = list(folder.glob("*.jpg")) + list(folder.glob("*.JPG"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def run_preparation():
    reset_game_state()

    if not WATCH_DIR.exists():
        st.session_state.message = f"監視フォルダが存在しません: {WATCH_DIR}"
        return

    img_path = find_latest_image(WATCH_DIR)
    if img_path is None:
        st.session_state.message = f"{WATCH_DIR} に画像が見つかりません。"
        return

    img = rr.load_image(img_path)
    if img is None:
        st.session_state.message = f"画像の読み込みに失敗しました: {img_path.name}"
        return

    model = get_model()
    error_reason = None
    direction = None
    img_rotated = None
    cand_M = None

    tri = rr.detect_triangle_direction(img)
    if tri:
        direction = tri
    else:
        error_reason = "向き判定エラー（青三角を検出できませんでした）"

    if error_reason is None:
        img_rotated = rr.rotate_image_for_direction(img, direction)
        cand_M = rr.calibrate_from_image(img_rotated)
        if cand_M is None:
            error_reason = "キャリブレーションエラー（赤丸4個を検出できませんでした）"

    mismatches = []
    if error_reason is None:
        warped = cv2.warpPerspective(img_rotated, cand_M, (900, 900))
        labels = rr.predict_board(model, warped)
        for r in range(9):
            for c in range(9):
                if labels[r][c] != rr.INITIAL_BOARD_STD[r][c]:
                    mismatches.append((r, c, rr.INITIAL_BOARD_STD[r][c], labels[r][c]))
        if mismatches:
            error_reason = f"初期配置照合エラー（認識結果が初期配置と不一致: {len(mismatches)}箇所）"

    if error_reason is not None:
        st.session_state.message = (
            f"[使用画像] {img_path.name}\n[エラー] {error_reason}\n"
            f"→ 下のフォームで向きを指定し、四隅をクリックして手動キャリブレーションしてください。"
        )
        st.session_state.phase = "manual_calib"
        st.session_state.manual_step = "direction"
        st.session_state.manual_raw_img = img
        st.session_state.manual_points = []
    else:
        st.session_state.calib_M = cand_M
        st.session_state.direction = direction
        st.session_state.phase = "ready"
        st.session_state.message = (
            f"[使用画像] {img_path.name}\n準備OK（向き判定・赤丸キャリブレーション・初期配置照合すべて成功）"
        )


# ===== 手動キャリブレーション（streamlit-image-coordinatesで四隅クリック） =====
def render_manual_calibration():
    st.subheader("手動キャリブレーション")

    if st.session_state.manual_step == "direction":
        choice = st.radio(
            "盤の向きを選択してください（画像の回転方向）",
            list(DIRECTION_LABELS.keys()),
            format_func=lambda k: DIRECTION_LABELS[k],
            key="manual_direction_radio",
        )
        if st.button("向きを確定して次へ"):
            st.session_state.manual_direction = choice
            st.session_state.manual_step = "corners"
            st.session_state.manual_points = []
            st.rerun()
        return

    if st.session_state.manual_step == "corners":
        img_rotated = rr.rotate_image_for_direction(
            st.session_state.manual_raw_img, st.session_state.manual_direction
        )
        h, w = img_rotated.shape[:2]
        max_disp = 900
        scale = min(max_disp / w, max_disp / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        disp_img = cv2.resize(img_rotated, (disp_w, disp_h))

        vis = disp_img.copy()
        for i, (x, y) in enumerate(st.session_state.manual_points):
            cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(vis, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        pil_img = rr.Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        st.write("盤面の4隅を順にクリックしてください: 1.左上 → 2.右上 → 3.右下 → 4.左下"
                  "（マス目=赤丸の角。盤の外枠ではない）")
        coords = streamlit_image_coordinates(
            pil_img, key=f"calib_click_{len(st.session_state.manual_points)}"
        )
        if coords is not None and len(st.session_state.manual_points) < 4:
            pt = (coords["x"], coords["y"])
            if pt not in st.session_state.manual_points:
                st.session_state.manual_points.append(pt)
                st.rerun()

        st.caption(f"{len(st.session_state.manual_points)}/4 点を選択済み")
        col_a, col_b = st.columns(2)
        if col_a.button("やり直し"):
            st.session_state.manual_points = []
            st.rerun()
        if len(st.session_state.manual_points) == 4:
            if col_b.button("この4点で確定 → 準備OK"):
                src_orig = [(x / scale, y / scale) for x, y in st.session_state.manual_points]
                ordered = rr.order_points(src_orig)
                M = rr.compute_calib(ordered)
                st.session_state.calib_M = M
                st.session_state.direction = st.session_state.manual_direction
                st.session_state.phase = "ready"
                st.session_state.message = "手動キャリブレーション完了。準備OK"
                st.session_state.manual_step = None
                st.rerun()


# ===== KIFファイル名の状態遷移 =====
def kif_filename_for_start(now: datetime) -> str:
    # 仕様: 【対局中】yyyyMMdd_hh:mm.kif だが ":" はWindowsのファイル名に使えないため "-" に変更
    return f"【対局中】{now:%Y%m%d_%H-%M}.kif"


def rename_kif_prefix(path: Path, new_prefix: str) -> Path:
    new_name = re.sub(r"^【[^】]*】", new_prefix, path.name)
    new_path = path.parent / new_name
    if path.exists():
        path.rename(new_path)
    return new_path


def start_game():
    now = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    st.session_state.kif_path = OUT_DIR / kif_filename_for_start(now)
    st.session_state.board = shogi.Board()
    st.session_state.moves_usi = []
    rr.save_kif(st.session_state.moves_usi, st.session_state.kif_path)  # 0手目時点でファイルを作成しておく
    # 対局開始ボタン押下後に新しく入ってくる画像のみを対象にする
    # （押下時点で既に存在する画像=対局準備で使った0手目画像などは対象外）
    existing = {p.name for p in WATCH_DIR.glob("*.jpg")} | {p.name for p in WATCH_DIR.glob("*.JPG")}
    st.session_state.processed = existing
    st.session_state.phase = "playing"
    st.session_state.message = f"対局開始。{WATCH_DIR} を監視中...\nKIF: {st.session_state.kif_path.name}"


def finish_game(new_prefix: str, extra_message: str = ""):
    if st.session_state.kif_path is not None:
        new_path = rename_kif_prefix(st.session_state.kif_path, new_prefix)
        st.session_state.kif_path = new_path
        n = len(st.session_state.moves_usi)
        st.session_state.message = f"{extra_message}{new_prefix} {n}手まで記録。ファイル: {new_path.name}"
    else:
        st.session_state.message = f"{extra_message}{new_prefix}（記録なし）"
    reset_game_state()


def unprocessed_files():
    files = list(WATCH_DIR.glob("*.jpg")) + list(WATCH_DIR.glob("*.JPG"))
    return [f for f in files if f.name not in st.session_state.processed]


def request_finish(new_prefix: str):
    # 対局終了/対局中止ボタン押下時、まだ解析していない画像が残っていれば即終了せず、
    # watch_fragmentに残りを処理させてから自動的にfinish_gameする（課題①対応）。
    if not unprocessed_files():
        finish_game(new_prefix)
        return
    st.session_state.phase = "finishing"
    st.session_state.pending_finish_prefix = new_prefix
    st.session_state.message = "分析対応中...お待ちください（残り画像を解析しています）"


# ===== 対局中: 新規画像の監視・解析（st.fragmentで定期実行） =====
@st.fragment(run_every=POLL_INTERVAL_SEC)
def watch_fragment():
    if st.session_state.phase not in ("playing", "finishing"):
        return

    new_files = sorted(unprocessed_files(), key=lambda p: p.stat().st_mtime)

    model = get_model()
    aborted = False
    for img_path in new_files:
        st.session_state.processed.add(img_path.name)
        img = rr.load_image(img_path)
        if img is None:
            continue

        img_rotated = rr.rotate_image_for_direction(img, st.session_state.direction)
        warped = cv2.warpPerspective(img_rotated, st.session_state.calib_M, (900, 900))
        labels = rr.predict_board(model, warped)
        status, payload = rr.classify_frame(st.session_state.board, labels)

        if status == "move":
            for usi in payload:
                st.session_state.moves_usi.append(usi)
                st.session_state.board.push_usi(usi)
            rr.save_kif(st.session_state.moves_usi, st.session_state.kif_path)
            st.session_state.message = (
                f"{len(st.session_state.moves_usi)}手目まで記録 "
                f"({img_path.name}, {datetime.now():%H:%M:%S})"
            )
        elif status == "nochange":
            st.session_state.message = (
                f"{img_path.name}: 駒のずれ/照明変化と判断、手としては記録しません"
            )
        else:
            # 画面要件No.6: 認識エラーで続行不能 → 自動中止
            n = len(st.session_state.moves_usi)
            finish_game(
                "【対局エラー中止】",
                extra_message=f"[認識エラーのため自動中止]（{img_path.name}, {n}手目まで記録, 詳細: {payload}）\n",
            )
            aborted = True
            break

    if aborted:
        st.rerun()  # ボタンの有効/無効状態を即時反映するため全体を再実行
        return

    if st.session_state.phase == "finishing" and not unprocessed_files():
        finish_game(st.session_state.pending_finish_prefix)
        st.rerun()  # ボタンの有効/無効状態を即時反映するため全体を再実行
        return

    st.text_area("メッセージ", st.session_state.message, height=200, disabled=True,
                 key=f"msg_playing_{len(st.session_state.processed)}")


# ===== 画面本体 =====
def main():
    init_state()
    st.title("将棋盤画像解析 — 対局記録ツール")

    cols = st.columns(4)
    prep_disabled = st.session_state.phase not in ("idle", "ready")
    start_disabled = st.session_state.phase != "ready"
    end_disabled = st.session_state.phase != "playing"
    abort_disabled = st.session_state.phase != "playing"

    prep_clicked = cols[0].button("対局準備", disabled=prep_disabled, use_container_width=True)
    start_clicked = cols[1].button("対局開始", disabled=start_disabled, use_container_width=True)
    end_clicked = cols[2].button("対局終了", disabled=end_disabled, use_container_width=True)
    abort_clicked = cols[3].button("対局中止", disabled=abort_disabled, use_container_width=True)

    if prep_clicked:
        run_preparation()
        st.rerun()
    if start_clicked:
        start_game()
        st.rerun()
    if end_clicked:
        request_finish("【対局完了】")
        st.rerun()
    if abort_clicked:
        request_finish("【対局中止】")
        st.rerun()

    if st.session_state.phase == "manual_calib":
        st.text_area("メッセージ", st.session_state.message, height=150, disabled=True)
        render_manual_calibration()
    elif st.session_state.phase in ("playing", "finishing"):
        watch_fragment()
    else:
        st.text_area("メッセージ", st.session_state.message, height=200, disabled=True)


if __name__ == "__main__":
    main()
