"""Stopイベント発火時にClaudeの直近の応答テキストをDiscord Webhookへ転送する。
DISCORD_WEBHOOK_URL環境変数が未設定の場合は何もせず終了する（settings.local.jsonのenvで設定）。
"""
import json
import os
import subprocess
import sys
import tempfile

DISCORD_CONTENT_LIMIT = 1900


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


def main():
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        hook_input = {}

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return

    transcript_path = hook_input.get("transcript_path") or hook_input.get("transcriptPath")
    if not transcript_path:
        return

    text = extract_last_assistant_text(transcript_path)
    if not text:
        return

    if len(text) > DISCORD_CONTENT_LIMIT:
        text = text[:DISCORD_CONTENT_LIMIT] + "\n...(truncated)"

    payload_path = None
    try:
        fd, payload_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"content": text}, f, ensure_ascii=False)
        subprocess.run(
            [
                "curl", "-s", "-o", os.devnull, "-X", "POST", webhook_url,
                "-H", "Content-Type: application/json",
                "--data-binary", "@" + payload_path,
            ],
            timeout=15,
        )
    except Exception:
        pass
    finally:
        if payload_path and os.path.exists(payload_path):
            os.remove(payload_path)


if __name__ == "__main__":
    main()
