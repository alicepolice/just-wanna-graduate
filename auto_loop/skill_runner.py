"""
Skill 调用器 — 构造 prompt，调用 claude-agent-sdk 执行学术裁缝 skill，解析输出。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import textwrap
import threading
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import AssistantMessage, ResultMessage, StreamEvent, TextBlock

try:
    from .config import (
        CLAUDE_EFFORT,
        CLAUDE_MODEL,
        DEIM_ROOT,
        RECORD_DIR,
        SKILL_FILE,
    )
except ImportError:
    from config import (
        CLAUDE_EFFORT,
        CLAUDE_MODEL,
        DEIM_ROOT,
        RECORD_DIR,
        SKILL_FILE,
    )

logger = logging.getLogger(__name__)

# claude 输出中 JSON 摘要的标记
_JSON_MARKER = "AUTO_LOOP_RESULT"

# 累计 cost 追踪（跨轮次累加）
_total_cost_usd: float = 0.0


def _build_user_prompt(state: dict, next_version: str) -> str:
    best = state.get("best_model") or {}
    tried = state.get("tried_strategies", [])
    history = state.get("history", [])

    history_lines = "\n".join(
        f"  - {h['version']}: AP={h['ap']:.4f} ({h['delta']}) {'✓' if h['kept'] else '✗'} [{h.get('strategy', '')}]"
        for h in history[-10:]
    ) or "  （暂无历史）"

    tried_str = "、".join(tried) if tried else "（暂无）"

    best_yaml = best.get("yaml", "（未知）")
    best_yml = best.get("yml", "（未知）")
    best_ap = best.get("ap", "N/A")
    best_ap50 = best.get("ap50", "N/A")
    next_yaml_path, next_yml_path = _infer_next_artifact_paths(best, next_version)

    # 计算记录文件的规范路径（相对于 DEIM_ROOT）
    record_dir_rel = RECORD_DIR.relative_to(DEIM_ROOT)
    # 从 next_yaml_path 推导模型名（如 dfine-n-v2）
    next_model_stem = Path(next_yaml_path).stem
    record_file_path = f"{record_dir_rel}/{next_model_stem}.md"

    return textwrap.dedent(f"""
        ## 自动迭代任务

        你正在执行第 {next_version} 轮自动化实验迭代。请完整执行学术裁缝 Skill 的 Phase 1~6。

        ### 当前最优模型
        - 模型 YAML：`{best_yaml}`
        - 训练 YML：`{best_yml}`
        - AP (0.50:0.95)：{best_ap}
        - AP50：{best_ap50}

        ### 已尝试的改进策略（请避免重复）
        {tried_str}

        ### 历史迭代记录（最近 10 轮）
        {history_lines}

        ### 本轮任务
        1. 基于当前最优模型，选择一个**尚未尝试**的改进方向
        2. 完整执行 Phase 1~6（分析→方案→YAML→验证→记录）
        3. 基于当前 `state.json` 的最优模型路径，本轮新 YAML 应写到 `{next_yaml_path}`，训练 YML 应写到 `{next_yml_path}`
        4. **实验记录必须保存到 `{record_file_path}`**（命名规范：与模型 YAML 同名，如 `{next_model_stem}.md`）
        5. 完成后，在回复**最末尾**输出以下 JSON 块（不要省略，不要修改格式）：

        ```json
        {{{_JSON_MARKER}}}
        {{
          "version": "{next_version}",
          "yaml_path": "<相对于 DEIM_ROOT 的 yaml 路径>",
          "yml_path": "<相对于 DEIM_ROOT 的 yml 路径>",
          "strategy_name": "<本轮改进策略名称，一句话>",
          "record_path": "{record_file_path}",
          "get_info_passed": true
        }}
        ```
    """).strip()


def _infer_next_artifact_paths(best: dict, next_version: str) -> tuple[str, str]:
    current_version = str(best.get("version") or "").strip()
    yaml_path = _bump_versioned_path(
        best.get("yaml"),
        current_version=current_version,
        next_version=next_version,
        fallback="configs_lab/test/models",
        fallback_prefix="model",
        fallback_suffix=".yaml",
    )
    yml_path = _bump_versioned_path(
        best.get("yml"),
        current_version=current_version,
        next_version=next_version,
        fallback="configs_lab/test/train",
        fallback_prefix="train",
        fallback_suffix=".yml",
    )
    return yaml_path, yml_path


def _bump_versioned_path(
    raw_path: str | None,
    *,
    current_version: str,
    next_version: str,
    fallback: str,
    fallback_prefix: str,
    fallback_suffix: str,
) -> str:
    if not raw_path:
        return f"{fallback}/{fallback_prefix}-{next_version}{fallback_suffix}"

    path = Path(raw_path)
    stem = path.stem

    if current_version:
        current_pattern = re.escape(current_version)
        match = re.search(rf"(?P<prefix>.*?)(?P<sep>[-_]?){current_pattern}$", stem)
        if match:
            prefix = match.group("prefix")
            sep = match.group("sep")
            if prefix:
                new_stem = f"{prefix}{sep}{next_version}"
            else:
                new_stem = next_version
            return str(path.with_name(f"{new_stem}{path.suffix}"))

    match = re.search(r"(?P<prefix>.*?)(?P<sep>[-_])v\d+$", stem)
    if match:
        prefix = match.group("prefix")
        sep = match.group("sep")
        return str(path.with_name(f"{prefix}{sep}{next_version}{path.suffix}"))

    return str(path.with_name(f"{stem}-{next_version}{path.suffix}"))


def run(state: dict, next_version: str, max_retries: int = 2) -> dict:
    """
    调用 claude -p 执行学术裁缝 skill。
    返回解析后的 JSON 摘要 dict，包含 yaml_path、yml_path、strategy_name 等。
    失败时抛出 RuntimeError。
    """
    if not SKILL_FILE.exists():
        raise FileNotFoundError(f"Skill 文件不存在: {SKILL_FILE}")

    skill_content = SKILL_FILE.read_text(encoding="utf-8")
    user_prompt = _build_user_prompt(state, next_version)

    for attempt in range(max_retries + 1):
        logger.info("调用 claude -p (第 %d 次)...", attempt + 1)
        result = _call_claude(skill_content, user_prompt)
        if result is not None:
            return result
        if attempt < max_retries:
            logger.warning("解析失败，重试 (%d/%d)...", attempt + 1, max_retries)

    raise RuntimeError(f"claude -p 调用失败或输出无法解析（已重试 {max_retries} 次）")


def _call_claude(system_prompt: str, user_prompt: str) -> dict | None:
    """调用 claude-agent-sdk，返回解析后的 JSON 摘要，失败返回 None。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_call_claude_async(system_prompt, user_prompt))

    result: dict | None = None
    error: Exception | None = None

    def _runner() -> None:
        nonlocal result, error
        try:
            result = asyncio.run(_call_claude_async(system_prompt, user_prompt))
        except Exception as exc:
            error = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return result


async def _call_claude_async(system_prompt: str, user_prompt: str) -> dict | None:
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        cwd=str(DEIM_ROOT),
        permission_mode="auto",
        model=CLAUDE_MODEL or None,
        effort=CLAUDE_EFFORT or None,
        include_partial_messages=True,
    )

    full_text_parts: list[str] = []
    stream_text_parts: list[str] = []
    got_first_token = False
    pending_tools: dict[str, dict] = {}
    need_separator = False  # 在 tool 调用后、下一段文本前插入分隔线

    try:
        async for msg in query(prompt=user_prompt, options=options):
            if isinstance(msg, StreamEvent):
                event = msg.event
                etype = event.get("type", "")

                if etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text and not got_first_token:
                            got_first_token = True
                            logger.info("首个 token 到达")
                        # 在 tool 调用结束后的首段文本前插入时间戳分隔线
                        if text and need_separator:
                            _stream_write_separator()
                            need_separator = False
                        stream_text_parts.append(text)
                        _stream_write_text(text)
                    elif delta.get("type") == "input_json_delta":
                        idx = str(event.get("index", ""))
                        if idx in pending_tools:
                            pending_tools[idx]["input_buf"] += delta.get("partial_json", "")

                elif etype == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        idx = str(event.get("index", ""))
                        pending_tools[idx] = {"name": block.get("name", ""), "input_buf": ""}

                elif etype == "content_block_stop":
                    idx = str(event.get("index", ""))
                    if idx in pending_tools:
                        info = pending_tools.pop(idx)
                        name = info["name"]
                        try:
                            inp = json.loads(info["input_buf"]) if info["input_buf"] else {}
                        except json.JSONDecodeError:
                            inp = {}
                        detail = _format_tool_detail(name, inp)
                        _stream_write_tool(name, detail)
                        need_separator = True

            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        full_text_parts.append(block.text)

            elif isinstance(msg, ResultMessage):
                cost = msg.total_cost_usd or 0
                global _total_cost_usd
                _total_cost_usd += cost
                logger.info(
                    "claude 完成: turns=%d  本次=$%.4f  累计=$%.4f",
                    msg.num_turns, cost, _total_cost_usd,
                )

    except Exception as exc:
        logger.error("claude-agent-sdk 调用失败: %s", exc)
        return None

    _stream_write("\n")

    output = "\n".join(full_text_parts).strip() or "".join(stream_text_parts).strip()
    logger.debug("claude 输出长度: %d 字符", len(output))
    return _parse_output(output)


def _parse_output(output: str) -> dict | None:
    """从 claude 输出中提取 JSON 摘要块。"""
    pattern = re.compile(
        r"\{" + _JSON_MARKER + r"\}\s*(\{.*?\})",
        re.DOTALL,
    )
    match = pattern.search(output)
    if not match:
        json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", output, re.DOTALL)
        if json_blocks:
            raw = json_blocks[-1]
        else:
            logger.warning("未找到 JSON 摘要块，输出末尾:\n%s", output[-500:])
            return None
    else:
        raw = match.group(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("JSON 解析失败: %s\n原始: %s", exc, raw[:300])
        return None

    required = {"yaml_path", "yml_path", "strategy_name"}
    missing = required - data.keys()
    if missing:
        logger.warning("JSON 缺少字段: %s", missing)
        return None

    for key in ("yaml_path", "yml_path", "record_path"):
        if key in data and data[key]:
            p = Path(data[key])
            if not p.is_absolute():
                data[key] = str(DEIM_ROOT / p)

    logger.info("解析成功: strategy=%s, yaml=%s", data["strategy_name"], data["yaml_path"])
    return data


def _format_tool_detail(name: str, inp: dict) -> str:
    """格式化 tool 调用的摘要信息（完整，不截断关键字段）。"""
    if name == "Read":
        path = inp.get("file_path", inp.get("path", ""))
        offset = inp.get("offset")
        limit = inp.get("limit")
        if offset is not None or limit is not None:
            start = offset or 1
            end = (start + limit - 1) if limit else "EOF"
            return f"{path}  [{start}~{end} 行]"
        return f"{path}  [整个文件]"
    elif name in ("Edit", "Write"):
        return inp.get("file_path", inp.get("path", ""))
    elif name == "Bash":
        return inp.get("command", "")  # 完整命令，不截断
    elif name == "Glob":
        return inp.get("pattern", "")
    elif name == "Grep":
        path = inp.get("path", "")
        pattern = inp.get("pattern", "")
        return f"{pattern!r} in {path}" if path else pattern
    elif name == "TodoWrite":
        todos = inp.get("todos", [])
        parts = [f"({len(todos)} items)"]
        for t in todos:
            status = t.get("status", "?")
            icon = {"completed": "✓", "in_progress": "→", "pending": "○"}.get(status, "?")
            parts.append(f"  {icon} {t.get('content', '')}")
        return "\n".join(parts)
    elif name == "Agent":
        desc = inp.get("description", "")
        prompt = inp.get("prompt", "")
        # 显示 description + prompt 前 200 字
        snippet = prompt[:200] + ("…" if len(prompt) > 200 else "")
        return f"{desc}\n  {snippet}" if desc else snippet
    elif name == "Skill":
        return inp.get("skill", str(inp))
    else:
        # 其他 tool：完整 JSON，但对超长 value 做截断
        return str(inp)


# ── ANSI 颜色（终端流式输出用，无缓冲、无 markup 解析问题）────────
# rich.Console.print 有内部缓冲，且会把 [...] 当 markup 解析，
# 不适合逐 token 的流式场景；这里直接用 ANSI 转义码。
_ANSI_RESET  = "\033[0m"
_ANSI_BOLD   = "\033[1m"
_ANSI_DIM    = "\033[2m"

# 正文（claude 思考内容）：亮白色
_ANSI_TEXT   = "\033[97m"

# 分隔线：青色
_ANSI_SEP    = "\033[36m"
_ANSI_SEP_TS = "\033[1;36m"   # 时间戳加粗

# tool 名 → ANSI 前景色
_TOOL_ANSI: dict[str, str] = {
    "Read":      "\033[96m",    # 亮青
    "Write":     "\033[92m",    # 亮绿
    "Edit":      "\033[93m",    # 亮黄
    "Bash":      "\033[95m",    # 亮品红
    "Glob":      "\033[94m",    # 亮蓝
    "Grep":      "\033[94m",    # 亮蓝
    "TodoWrite": "\033[33m",    # 橙黄
    "Agent":     "\033[91m",    # 亮红
    "Skill":     "\033[35m",    # 品红
}
_ANSI_TOOL_DEFAULT = "\033[90m"  # 暗灰（未知 tool）


def _ansi(code: str, text: str) -> str:
    """用 ANSI 代码包裹文本，末尾自动 reset。"""
    return f"{code}{text}{_ANSI_RESET}"


def _stream_write(text: str) -> None:
    """写到终端（纯文本）+ loop.log，用于结构化输出（末尾换行等）。"""
    sys.stdout.write(text)
    try:
        sys.stdout.flush()
    except Exception:
        pass
    _log_file_write(text)


def _stream_write_text(text: str) -> None:
    """将 claude 流式文字逐 token 写到终端（亮白色）+ loop.log（纯文本）。
    必须用 sys.stdout.write + flush，不能用 rich.Console.print（有缓冲 + markup 解析）。
    """
    _log_file_write(text)
    if not text:
        return
    sys.stdout.write(f"{_ANSI_TEXT}{text}{_ANSI_RESET}")
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _stream_write_separator() -> None:
    """输出带时间戳的分隔线（终端彩色 + log 纯文本）。"""
    now = datetime.now().strftime("%H:%M:%S")
    sep_plain = f"\n{'─' * 20} [{now}] {'─' * 20}\n"
    _log_file_write(sep_plain)
    sep_color = (
        f"\n"
        f"{_ANSI_SEP}{'─' * 20}{_ANSI_RESET}"
        f" {_ANSI_SEP_TS}[{now}]{_ANSI_RESET} "
        f"{_ANSI_SEP}{'─' * 20}{_ANSI_RESET}"
        f"\n"
    )
    sys.stdout.write(sep_color)
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _stream_write_tool(name: str, detail: str) -> None:
    """将 tool 调用行以彩色格式写到终端（ANSI），纯文本写到 log。"""
    plain = f"\n[tool:{name}] {detail}\n"
    _log_file_write(plain)

    color = _TOOL_ANSI.get(name, _ANSI_TOOL_DEFAULT)
    detail_lines = detail.splitlines()
    first_line = detail_lines[0] if detail_lines else ""
    rest_lines = detail_lines[1:]

    out_parts = [
        "\n",
        f"{_ANSI_BOLD}{color}[tool:{name}]{_ANSI_RESET}",
        " ",
        f"{color}{first_line}{_ANSI_RESET}",
    ]
    for extra in rest_lines:
        out_parts.append(f"\n{_ANSI_DIM}{color}  {extra}{_ANSI_RESET}")
    out_parts.append("\n")

    sys.stdout.write("".join(out_parts))
    try:
        sys.stdout.flush()
    except Exception:
        pass


def _log_file_write(text: str) -> None:
    """追加写入 loop.log（与 logging FileHandler 共享同一文件）。"""
    try:
        from .config import LOOP_LOG
    except ImportError:
        from config import LOOP_LOG
    try:
        with open(LOOP_LOG, "a", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


def _timestamp_separator() -> str:
    """生成带时间戳的纯文本分隔线（供 log 使用）。"""
    now = datetime.now().strftime("%H:%M:%S")
    return f"\n{'─' * 20} [{now}] {'─' * 20}\n"
