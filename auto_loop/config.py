"""
路径配置 — 所有硬编码路径集中在这里。
修改实验项目时只需改这个文件。
"""
from pathlib import Path

# ── 实验项目根目录 ─────────────────────────────────────────────
DEIM_ROOT = Path("/home/exp/DEIM-MOD-V2")

# ── 关键脚本 ───────────────────────────────────────────────────
TRAIN_SH       = DEIM_ROOT / "scripts" / "train.sh"
GET_INFO_PY    = DEIM_ROOT / "tools" / "benchmark" / "get_info.py"
COMPARE_PY     = DEIM_ROOT / "scripts" / "compare_models.py"
TEST_SH        = DEIM_ROOT / "scripts" / "test.sh"

# ── Skill 文件 ─────────────────────────────────────────────────
SKILL_FILE     = DEIM_ROOT / "prompt" / "skill-only-net-full" / "缝合任务.md"

# ── 实验记录目录 ───────────────────────────────────────────────
RECORD_DIR     = DEIM_ROOT / "prompt" / "record"

# ── 训练输出根目录 ─────────────────────────────────────────────
OUTPUTS_ROOT   = DEIM_ROOT / "outputs"

# ── 本工具目录 ─────────────────────────────────────────────────
TOOL_ROOT      = Path("/home/exp/just-wanna-graduate")
STATE_FILE     = TOOL_ROOT / "config" / "state.json"
LOOP_CONFIG    = TOOL_ROOT / "config" / "loop_config.yml"
LOOP_LOG       = TOOL_ROOT / "loop.log"

# ── conda 环境名（用于激活 deim 环境运行 get_info）─────────────
CONDA_ENV      = "deim"
