"""Stop / Notificationイベント発火時にDiscordボット経由でPCを見てほしいタイミングを通知する。
Bot tokenは docs/discord_token.txt から読む（gitignore対象、settings.local.jsonには置かない）。
送信先チャンネルIDは DISCORD_CHANNEL_ID 環境変数（settings.local.jsonのenvで設定）。
いずれか欠けている場合は何もせず終了する。

- Stop: Claudeが応答を完全に終えたタイミング。直近の応答テキストを「🔔 PCを見てください」付きで転送する。
- Notification: 権限確認(permission_prompt)・アイドル待ち(idle_prompt)などClaude側がユーザーに
  通知を出すタイミング。短い呼びかけメッセージを送る（stdinの正確なフィールド名は未確定のため、
  複数候補を試した上でフォールバックする防御的な実装にしている。実際の通知が来たらdiscord_hook_debug.logで
  ペイロードを確認し、必要なら本文抽出ロジックを調整すること）。
"""
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

DISCORD_CONTENT_LIMIT = 1900
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOKEN_FILE = _REPO_ROOT / "docs" / "discord_token.txt"

# 一時デバッグ用: Stopフックが本当に自動発火しているか確認するためのログ
# (gitignore対象のtrain/test_runs/配下に書く。問題解決後は削除する)
_DEBUG_LOG = Path(__file__).resolve().parents[2] / "train" / "data" / "test_runs" / "discord_hook_debug.log"


def _debug(msg):
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} {msg}\n")
    except Exception:
        pass


def extract_last_assistant_text(transcript_path):
    last_text_blocks = None
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "assistant":
                    continue
                content = entry.get("message", {}).get("content")
                if not isinstance(content, list):
                    continue
                texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                if texts:
                    last_text_blocks = texts
    except OSError:
        return None
    if not last_text_blocks:
        return None
    return "\n".join(last_text_blocks)


def post_to_discord(channel_id, bot_token, text):
    if len(text) > DISCORD_CONTENT_LIMIT:
        text = text[:DISCORD_CONTENT_LIMIT] + "\n...(truncated)"

    payload_path = None
    try:
        fd, payload_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"content": text}, f, ensure_ascii=False)
        result = subprocess.run(
            [
                "curl", "-s", "-w", "%{http_code}", "-o", os.devnull, "-X", "POST",
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                "-H", f"Authorization: Bot {bot_token}",
                "-H", "Content-Type: application/json",
                "--data-binary", "@" + payload_path,
            ],
            timeout=15,
            capture_output=True,
            text=True,
        )
        _debug(f"curl done, returncode={result.returncode}, stdout={result.stdout!r}, stderr={result.stderr!r}")
    except Exception as e:
        _debug(f"curl raised exception: {e!r}")
    finally:
        if payload_path and os.path.exists(payload_path):
            os.remove(payload_path)


def build_stop_message(hook_input):
    transcript_path = hook_input.get("transcript_path") or hook_input.get("transcriptPath")
    if not transcript_path:
        _debug(f"no transcript_path in hook_input: {hook_input}")
        return None
    text = extract_last_assistant_text(transcript_path)
    if not text:
        _debug(f"no assistant text extracted from {transcript_path}")
        return None
    return f"\U0001f514 PCを見てください\n\n{text}"


def build_notification_message(hook_input):
    # Notificationイベントのstdinフィールド名はドキュメント上未確定のため、複数の
    # 候補キーを試す防御的な実装。最終的にどれも見つからなければ種別だけでも伝える。
    notif_type = (
        hook_input.get("notification_type")
        or hook_input.get("notificationType")
        or hook_input.get("type")
        or "unknown"
    )
    detail = (
        hook_input.get("message")
        or hook_input.get("notification")
        or hook_input.get("tool_name")
        or hook_input.get("toolName")
        or ""
    )
    label = {
        "permission_prompt": "権限確認待ち",
        "idle_prompt": "応答待ち（アイドル）",
    }.get(notif_type, notif_type)
    text = f"\U0001f514 PCを見てください（{label}）"
    if detail:
        text += f"\n{detail}"
    return text


def main():
    _debug(f"invoked, cwd={os.getcwd()}, argv={sys.argv}")
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        _debug(f"stdin JSON decode failed: {e}")
        hook_input = {}

    channel_id = os.environ.get("DISCORD_CHANNEL_ID")
    if not channel_id:
        _debug("DISCORD_CHANNEL_ID not set in env, aborting")
        return

    try:
        bot_token = _TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError as e:
        _debug(f"failed to read token file {_TOKEN_FILE}: {e}")
        return
    if not bot_token:
        _debug(f"token file {_TOKEN_FILE} is empty, aborting")
        return

    event_name = hook_input.get("hook_event_name") or hook_input.get("hookEventName")
    _debug(f"hook_event_name={event_name}, raw_input={hook_input}")

    if event_name == "Notification":
        text = build_notification_message(hook_input)
    else:
        text = build_stop_message(hook_input)

    if not text:
        return
    _debug(f"posting to discord: {text[:200]!r}")
    post_to_discord(channel_id, bot_token, text)


if __name__ == "__main__":
    main()
