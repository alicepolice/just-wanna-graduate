"""
Skill 调用器 — 构造 prompt，调用 claude -p 执行学术裁缝 skill，解析输出。

claude -p 调用约定：
  - 工作目录：DEIM_ROOT（让 claude 能直接读写项目文件）
  - system prompt：学术裁缝 skill 文件内容
  - user prompt：包含当前状态的结构化指令
  - 输出格式：要求 claude 在最后输出一个 JSON 摘要块，便于解析
"""
import json
import logging
import re
import subprocess
import textwrap
from pathlib import Path

from config import DEIM_ROOT, SKILL_FILE

logger = logging.getLogger(__name__)

# claude 输出中 JSON 摘要的标记
_JSON_MARKER = "AUTO_LOOP_RESULT"


def _build_user_prompt(state: dict, next_version: str) -> str:
    best = state.get("best_model") or {}
    tried = state.get("tried_strategies", [])
    history = state.get("history", [])

    history_lines = "\n".join(
        f"  - {h['version']}: AP={h['ap']:.4f} ({h['delta']}) {'✓' if h['kept'] else '✗'} [{h.get('strategy','')}]"
        for h in history[-10:]  # 只显示最近 10 条
    ) or "  （暂无历史）"

    tried_str = "、".join(tried) if tried else "（暂无）"

    best_yaml = best.get("yaml", "（未知）")
    best_yml  = best.get("yml",  "（未知）")
    best_ap   = best.get("ap",   "N/A")
    best_ap50 = best.get("ap50", "N/A")

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
        3. 新 YAML 命名为 `dfine-n-mm-{next_version}.yaml`，训练 YML 命名为 `deim_dfine_n_mm_FD_{next_version}.yml`
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
    """调用 claude -p，返回解析后的 JSON 摘要，失败返回 None。"""
    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--system-prompt", system_prompt,
        user_prompt,
    ]
    logger.debug("CMD: claude --print --dangerously-skip-permissions --system-prompt <skill> <prompt>")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(DEIM_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # 30 分钟超时
        )
    except subprocess.TimeoutExpired:
        logger.error("claude -p 超时（30 分钟）")
        return None

    if proc.returncode != 0:
        logger.error("claude -p 退出码 %d:\n%s", proc.returncode, proc.stderr[-1000:])
        return None

    output = proc.stdout
    logger.debug("claude 输出长度: %d 字符", len(output))
    return _parse_output(output)


def _parse_output(output: str) -> dict | None:
    """从 claude 输出中提取 JSON 摘要块。"""
    # 找到 AUTO_LOOP_RESULT 标记后的 JSON
    pattern = re.compile(
        r'\{' + _JSON_MARKER + r'\}\s*(\{.*?\})',
        re.DOTALL,
    )
    match = pattern.search(output)
    if not match:
        # 降级：尝试找最后一个 JSON 块
        json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
        if json_blocks:
            raw = json_blocks[-1]
        else:
            logger.warning("未找到 JSON 摘要块，输出末尾:\n%s", output[-500:])
            return None
    else:
        raw = match.group(1)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("JSON 解析失败: %s\n原始: %s", e, raw[:300])
        return None

    required = {"yaml_path", "yml_path", "strategy_name"}
    missing = required - data.keys()
    if missing:
        logger.warning("JSON 缺少字段: %s", missing)
        return None

    # 转为绝对路径
    for key in ("yaml_path", "yml_path", "record_path"):
        if key in data and data[key]:
            p = Path(data[key])
            if not p.is_absolute():
                data[key] = str(DEIM_ROOT / p)

    logger.info("解析成功: strategy=%s, yaml=%s", data["strategy_name"], data["yaml_path"])
    return data
