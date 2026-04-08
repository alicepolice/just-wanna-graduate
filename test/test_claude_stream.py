#!/usr/bin/env python3
"""最简测试：用 stream-json 调用 claude，实时打印输出"""
import builtins
import json
import subprocess
import sys

# deim 环境的 sitecustomize.py 劫持了 print，不支持 flush 参数，绕过它
_print = builtins.__dict__.get("_builtin_print", builtins.print)

def p(*args, **kwargs):
    kwargs.pop("flush", None)
    _print(*args, **kwargs)
    sys.stdout.flush()

cmd = [
    "claude", "--print", "--verbose",
    "--output-format", "stream-json",
    "--include-partial-messages",
    "你好",
]

p("CMD:", " ".join(cmd))
p("=" * 40)

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

for line in proc.stdout:
    line = line.strip()
    if not line:
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        p("[raw]", line)
        continue

    etype = event.get("type", "")
    p(f"[event:{etype}]", end=" ")

    if etype == "content_block_delta":
        text = event.get("delta", {}).get("text", "")
        sys.stdout.write(text)
        sys.stdout.flush()
    elif etype == "result":
        p("\n[result]", event.get("result", "")[:200])
    else:
        p(json.dumps(event, ensure_ascii=False)[:120])

proc.wait()
p("\n退出码:", proc.returncode)
if proc.returncode != 0:
    p("stderr:", proc.stderr.read())
