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
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import AssistantMessage, ResultMessage, StreamEvent, TextBlock

try:
    from .config import (
        CLAUDE_EFFORT,
        CLAUDE_MODEL,
        DEIM_ROOT,
        SKILL_FILE,
    )
except ImportError:
    from config import (
        CLAUDE_EFFORT,
        CLAUDE_MODEL,
        DEIM_ROOT,
        SKILL_FILE,
    )

logger = logging.getLogger(__name__)

# claude 输出中 JSON 摘要的标记
_JSON_MARKER = "AUTO_LOOP_RESULT"


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
        4. 完成后，在回复**最末尾**输出以下 JSON 块（不要省略，不要修改格式）：

        ```json
        {{{_JSON_MARKER}}}
        {{
          "version": "{next_version}",
          "yaml_path": "<相对于 DEIM_ROOT 的 yaml 路径>",
          "yml_path": "<相对于 DEIM_ROOT 的 yml 路径>",
          "strategy_name": "<本轮改进策略名称，一句话>",
          "record_path": "<实验记录 md 文件路径>",
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

    for attempt in range(1, max_retries + 2):
        logger.info("调用 claude -p (第 %d 次)...", attempt)
        result = _call_claude(skill_content, user_prompt)
        if result is not None:
            return result
        if attempt <= max_retries:
            logger.warning("解析失败，重试 (%d/%d)...", attempt, max_retries)

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
                        stream_text_parts.append(text)
                        _stream_write(text)
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
                        if name in ("Read", "Edit", "Write"):
                            detail = inp.get("file_path", inp.get("path", ""))
                        elif name == "Bash":
                            detail = inp.get("command", "")[:120]
                        elif name == "Glob":
                            detail = inp.get("pattern", "")
                        elif name == "Grep":
                            detail = f"{inp.get('pattern', '')} in {inp.get('path', '')}"
                        else:
                            detail = str(inp)[:120]
                        _stream_write(f"\n[tool:{name}] {detail}\n")

            elif isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        full_text_parts.append(block.text)

            elif isinstance(msg, ResultMessage):
                logger.info("claude 完成: turns=%d cost=$%.4f", msg.num_turns, msg.total_cost_usd or 0)

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


def _stream_write(text: str) -> None:
    sys.stdout.write(text)
    flush = getattr(sys.stdout, "flush", None)
    if flush is None:
        return
    try:
        flush()
    except Exception:
        logger.debug("stdout flush 失败，已忽略", exc_info=True)
