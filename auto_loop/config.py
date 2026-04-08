"""
路径配置 — 所有硬编码路径集中在这里。
修改实验项目时只需改这个文件。
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# ── 实验项目根目录 ─────────────────────────────────────────────
DEIM_ROOT = Path(os.environ.get("DEIM_ROOT", "/home/exp/DEIM-MOD-V2"))

# ── 关键脚本 ───────────────────────────────────────────────────
TRAIN_SH       = DEIM_ROOT / "scripts" / "train.sh"
GET_INFO_PY    = DEIM_ROOT / "tools" / "benchmark" / "get_info.py"
COMPARE_PY     = DEIM_ROOT / "scripts" / "compare_models.py"
TEST_SH        = DEIM_ROOT / "scripts" / "test.sh"

# ── Skill 文件 ─────────────────────────────────────────────────
SKILL_FILE     = DEIM_ROOT / ".claude" / "skills" / "缝合任务" / "SKILL.md"

# ── 实验记录目录 ───────────────────────────────────────────────
RECORD_DIR     = DEIM_ROOT / "configs_lab" / "test" / "record"

# ── 训练输出根目录 ─────────────────────────────────────────────
OUTPUTS_ROOT   = DEIM_ROOT / "outputs"

# ── 本工具目录 ─────────────────────────────────────────────────
TOOL_ROOT      = Path(__file__).resolve().parent.parent
STATE_FILE     = TOOL_ROOT / "state" / "state.json"
LOOP_LOG       = TOOL_ROOT / "loop.log"

# ── 迭代行为配置（原 loop_config.yml）────────────────────────────
MAX_ITERATIONS     = 50       # 最大迭代轮数（防止无限循环）
MAX_NO_IMPROVE     = 5        # 连续多少轮未提升 AP 则停止
AUTO_APPROVE       = True     # 是否自动批准每轮方案（False = 每轮需用户确认）
GPU                = "0"      # 训练使用的 GPU（单卡: "0"，多卡: "0,1"）
POLL_INTERVAL_SEC  = 10       # 训练完成检测轮询间隔（秒）
TIMEOUT_HOURS      = 24.0     # 训练超时时间（小时），超时后强制终止
AUTO_EVAL          = True     # 训练完成后是否自动运行 eval

# ── conda 环境名（用于激活 deim 环境运行 get_info）─────────────
CONDA_ENV      = "deim"

# ── Claude 调用配置 ────────────────────────────────────────────
# 模型 ID，留空则使用 claude 默认值
# 可选: "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
CLAUDE_MODEL   = "claude-sonnet-4-6"
# thinking effort: "low" | "medium" | "high" | "" (留空则不传)
CLAUDE_EFFORT  = "high"


def resolve_deim_path(path_like: str | Path) -> Path:
    """将相对 DEIM_ROOT 的路径转为绝对路径。"""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (DEIM_ROOT / path).resolve()


def resolve_output_dir(yml_path: str | Path) -> Path:
    """从训练 yml 读取 output_dir，并按 DEIM_ROOT 解析为绝对路径。"""
    yml_file = resolve_deim_path(yml_path)
    with yml_file.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YML 顶层不是映射类型: {yml_file}")

    output_dir = data.get("output_dir")
    if not output_dir:
        raise KeyError(f"YML 缺少 output_dir: {yml_file}")

    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    return (DEIM_ROOT / output_path).resolve()
