#!/usr/bin/env python3
"""
测试 _call_claude 的事件解析逻辑。
用 "你好" 作为 prompt，打印所有事件类型，验证实时输出是否正常。
"""
import json
import subprocess
import sys

PROMPT = "你好，请回复一句话。"
SYSTEM = "你是一个助手。"  # 改成 SKILL.md 内容可复现真实场景
EFFORT = "high"           # 对应 config.py 的 CLAUDE_EFFORT

cmd = [
    "claude", "--print",
    "--permission-mode", "auto",
    "--output-format", "stream-json",
    "--include-partial-messages",
    "--verbose",
    "--system-prompt", SYSTEM,
    "--model", "claude-sonnet-4-6",
    "--effort", EFFORT,
    PROMPT,
]

print("CMD:", " ".join(cmd[:6]), "...")
print("=" * 50)

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

full_text_parts = []
seen_types = set()

for raw_line in proc.stdout:
    raw_line = raw_line.strip()
    if not raw_line:
        continue

    try:
        event = json.loads(raw_line)
    except json.JSONDecodeError:
        print(f"[raw] {raw_line[:120]}")
        sys.stdout.flush()
        continue

    etype = event.get("type", "")
    seen_types.add(etype)

    # --- 按 skill_runner.py 的逻辑处理 ---

    # skill_runner 的分支 1：stream_event 包裹的 content_block_delta
    if etype == "stream_event":
        inner = event.get("event", {})
        inner_type = inner.get("type", "")
        if inner_type == "content_block_delta":
            text = inner.get("delta", {}).get("text", "")
            sys.stdout.write(f"[stream_event/delta] {text}")
            sys.stdout.flush()
        else:
            print(f"[stream_event/{inner_type}]")

    # skill_runner 的分支 2：完整 assistant 消息
    elif etype == "assistant":
        msg = event.get("message", {})
        for block in msg.get("content", []):
            if block.get("type") == "text":
                full_text_parts.append(block.get("text", ""))
        print(f"[assistant] content blocks: {len(msg.get('content', []))}")

    # skill_runner 的分支 3：tool_use
    elif etype == "tool_use":
        name = event.get("name", "")
        print(f"[tool_use] {name}")

    # 其他事件：打印出来帮助诊断
    else:
        print(f"[{etype}] {json.dumps(event, ensure_ascii=False)[:120]}")

    sys.stdout.flush()

proc.wait(timeout=60)
print()
print("=" * 50)
print("退出码:", proc.returncode)
print("见到的事件类型:", sorted(seen_types))
print("full_text_parts 长度:", len("".join(full_text_parts)))
if full_text_parts:
    print("完整输出:", "".join(full_text_parts)[:300])
if proc.returncode != 0:
    print("stderr:", proc.stderr.read()[:500])
