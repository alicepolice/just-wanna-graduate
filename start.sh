#!/usr/bin/env bash
# start.sh — 启动 Auto-Experiment-Loop
# 用法:
#   ./start.sh                  # 无限循环（受 loop_config.yml 中 max_iterations 限制）
#   ./start.sh --max-iter 3     # 最多跑 3 轮
#   ./start.sh --dry-run        # 只生成 YAML，不训练
#   ./start.sh --init           # 初始化 state.json（交互式）
#   ./start.sh --gpu 0,1        # 指定 GPU

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_LOOP_DIR="$SCRIPT_DIR/auto_loop"

# ── 检查依赖 ──────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] 未找到 python3，请先安装或激活对应 conda 环境" >&2
    exit 1
fi

if ! command -v tmux &>/dev/null; then
    echo "[ERROR] 未找到 tmux，训练监控依赖 tmux" >&2
    exit 1
fi

if ! command -v claude &>/dev/null; then
    echo "[ERROR] 未找到 claude CLI，请先安装 Claude Code" >&2
    exit 1
fi

# ── 检查 DEIM 项目 ────────────────────────────────────────────
DEIM_ROOT="/home/exp/DEIM-MOD-V2"
if [ ! -d "$DEIM_ROOT" ]; then
    echo "[ERROR] DEIM 项目目录不存在: $DEIM_ROOT" >&2
    echo "        请修改 auto_loop/config.py 中的 DEIM_ROOT" >&2
    exit 1
fi

# ── 启动 ──────────────────────────────────────────────────────
echo "======================================================"
echo "  Auto-Experiment-Loop"
echo "  工作目录: $SCRIPT_DIR"
echo "  DEIM 项目: $DEIM_ROOT"
echo "  日志文件: $SCRIPT_DIR/loop.log"
echo "======================================================"

cd "$AUTO_LOOP_DIR"
exec python3 auto_loop.py "$@"
