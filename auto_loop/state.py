"""
状态管理 — 读写 state.json，维护迭代状态。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from .config import STATE_FILE
except ImportError:
    from config import STATE_FILE

logger = logging.getLogger(__name__)


def load() -> dict:
    """读取 state.json，不存在则返回空状态。"""
    if not STATE_FILE.exists():
        return _empty_state()
    with open(STATE_FILE, encoding="utf-8") as f:
        return json.load(f)


def save(state: dict) -> None:
    """写入 state.json。"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = STATE_FILE.with_suffix(f"{STATE_FILE.suffix}.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    Path(tmp_file).replace(STATE_FILE)
    logger.debug("state saved → %s", STATE_FILE)


def _empty_state() -> dict:
    return {
        "iteration": 0,
        "best_model": None,
        "tried_strategies": [],
        "history": [],
        "current_experiment": None,
    }


def next_version(state: dict) -> str:
    """返回下一个版本号字符串，如 'v14'。"""
    return f"v{state['iteration'] + 1}"


def reserve_iteration(state: dict, version: str) -> dict:
    """预留已生成文件使用的版本号，避免失败重跑时覆盖同名产物。"""
    state["iteration"] = max(state["iteration"], _version_number(version))
    return state


def record_experiment_start(state: dict, version: str, yaml_path: str, yml_path: str) -> dict:
    """记录实验开始，写入 current_experiment。"""
    state["current_experiment"] = {
        "version": version,
        "yaml": yaml_path,
        "yml": yml_path,
        "started_at": datetime.now().isoformat(),
        "status": "training",
    }
    return state


def record_experiment_result(
    state: dict,
    ap: float,
    ap50: float,
    strategy_name: str,
    kept: bool,
    rationale: str = "",
) -> dict:
    """训练完成后更新状态：保留或回退。"""
    exp = state["current_experiment"]
    if exp is None:
        raise ValueError("current_experiment 为空，无法写入实验结果")

    exp["ap"] = ap
    exp["ap50"] = ap50
    exp["kept"] = kept
    exp["finished_at"] = datetime.now().isoformat()
    exp["status"] = "kept" if kept else "discarded"

    best = state["best_model"]
    delta = f"{ap - best['ap']:+.4f}" if best else "N/A"
    exp["delta"] = delta

    history_entry: dict = {
        "version": exp["version"],
        "ap": ap,
        "ap50": ap50,
        "delta": delta,
        "kept": kept,
        "strategy": strategy_name,
    }
    if rationale:
        history_entry["rationale"] = rationale
    state["history"].append(history_entry)

    if kept:
        state["best_model"] = {
            "yaml": exp["yaml"],
            "yml": exp["yml"],
            "ap": ap,
            "ap50": ap50,
            "version": exp["version"],
        }

    if strategy_name and strategy_name not in state["tried_strategies"]:
        state["tried_strategies"].append(strategy_name)

    state["current_experiment"] = None
    return state


def get_best_ap(state: dict) -> Optional[float]:
    if state["best_model"]:
        return state["best_model"]["ap"]
    return None


def _version_number(version: str) -> int:
    if not version.startswith("v"):
        raise ValueError(f"非法版本号: {version}")
    return int(version[1:])
