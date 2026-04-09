#!/usr/bin/env python3
"""
Auto-Experiment-Loop — 自动化实验迭代主控制器

用法:
    python auto_loop.py                    # 无限循环，直到手动停止
    python auto_loop.py --max-iter 3       # 最多跑 3 轮
    python auto_loop.py --dry-run          # 只生成 YAML，不训练
    python auto_loop.py --init             # 初始化 state.json（交互式）
    python auto_loop.py --gpu 0,1          # 指定 GPU
"""
import argparse
import logging
import sys
import time
from pathlib import Path

try:
    from . import state as state_mod
    from . import trainer, evaluator, skill_runner
    from .config import (
        DEIM_ROOT, GET_INFO_PY, LOOP_LOG, STATE_FILE,
        MAX_ITERATIONS, MAX_NO_IMPROVE, AUTO_APPROVE, GPU,
        POLL_INTERVAL_SEC, TIMEOUT_HOURS, AUTO_EVAL,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    import state as state_mod
    import trainer
    import evaluator
    import skill_runner
    from config import (
        DEIM_ROOT, GET_INFO_PY, LOOP_LOG, STATE_FILE,
        MAX_ITERATIONS, MAX_NO_IMPROVE, AUTO_APPROVE, GPU,
        POLL_INTERVAL_SEC, TIMEOUT_HOURS, AUTO_EVAL,
    )

# ── 日志配置 ──────────────────────────────────────────────────
def _setup_logging(log_file: Path) -> None:
    from rich.console import Console
    from rich.logging import RichHandler

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    # 终端：rich 彩色输出
    console = Console(highlight=True)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.INFO)

    # 文件：纯文本
    file_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(file_fmt)
    file_handler.setLevel(logging.INFO)

    root.addHandler(rich_handler)
    root.addHandler(file_handler)


# ── 配置加载 ──────────────────────────────────────────────────
def _load_config() -> dict:
    return {
        "max_iterations": MAX_ITERATIONS,
        "max_no_improve": MAX_NO_IMPROVE,
        "auto_approve": AUTO_APPROVE,
        "gpu": GPU,
        "poll_interval_sec": POLL_INTERVAL_SEC,
        "timeout_hours": TIMEOUT_HOURS,
        "auto_eval": AUTO_EVAL,
    }


# ── 验证 YAML ─────────────────────────────────────────────────
def _validate_yml(yml_path: str) -> bool:
    import subprocess
    cmd = [
        sys.executable, str(GET_INFO_PY),
        "-c", yml_path,
    ]
    result = subprocess.run(cmd, cwd=str(DEIM_ROOT), capture_output=True, text=True)
    if result.returncode == 0:
        logging.getLogger(__name__).info("get_info 验证通过")
        return True
    logging.getLogger(__name__).error("get_info 验证失败:\n%s", result.stderr[-1000:])
    return False


# ── 用户确认 ──────────────────────────────────────────────────
def _ask_approve(skill_result: dict) -> bool:
    print("\n" + "="*60)
    print(f"[待确认] 本轮改进方案")
    print(f"  策略: {skill_result['strategy_name']}")
    print(f"  YAML: {skill_result['yaml_path']}")
    print(f"  YML:  {skill_result['yml_path']}")
    print("="*60)
    ans = input("是否启动训练？[Y/n] ").strip().lower()
    return ans in ("", "y", "yes")


# ── 初始化 state.json ─────────────────────────────────────────
def _init_state() -> None:
    print("初始化 state.json")
    iteration = int(input("当前迭代版本号（如 13）: ").strip())
    yaml_path = input("当前最优模型 YAML 路径（相对 DEIM_ROOT）: ").strip()
    yml_path  = input("当前最优训练 YML 路径（相对 DEIM_ROOT）: ").strip()
    ap        = float(input("当前最优 AP (0.50:0.95): ").strip())
    ap50      = float(input("当前最优 AP50: ").strip())
    tried_raw = input("已尝试策略（逗号分隔，可留空）: ").strip()
    tried     = [s.strip() for s in tried_raw.split(",") if s.strip()]

    s = state_mod._empty_state()
    s["iteration"] = iteration
    s["best_model"] = {
        "yaml": yaml_path,
        "yml": yml_path,
        "ap": ap,
        "ap50": ap50,
        "version": f"v{iteration}",
    }
    s["tried_strategies"] = tried
    state_mod.save(s)
    print(f"state.json 已写入: {STATE_FILE}")


# ── 单轮迭代 ──────────────────────────────────────────────────
def _run_one_iteration(cfg: dict, dry_run: bool) -> str:
    """
    执行一轮完整迭代。
    返回: 'improved' | 'no_improve' | 'failed' | 'skipped'
    """
    log = logging.getLogger(__name__)
    s = state_mod.load()
    version = state_mod.next_version(s)
    log.info("===== 开始第 %s 轮迭代 =====", version)

    # ① 调用 skill 生成 YAML
    log.info("调用学术裁缝 skill...")
    try:
        skill_result = skill_runner.run(s, version)
    except Exception as e:
        log.error("skill_runner 失败: %s", e)
        return "failed"

    yml_path  = skill_result["yml_path"]
    yaml_path = skill_result["yaml_path"]
    strategy  = skill_result["strategy_name"]
    version = skill_result.get("version", version)

    s = state_mod.load()
    s = state_mod.reserve_iteration(s, version)
    state_mod.save(s)

    # ② 验证 YAML（skill 内部已验证，这里做二次确认）
    if not skill_result.get("get_info_passed"):
        log.info("skill 未报告 get_info 通过，重新验证...")
        if not _validate_yml(yml_path):
            log.error("YAML 验证失败，跳过本轮")
            return "failed"

    # ③ 用户确认（auto_approve=False 时）
    if not cfg["auto_approve"]:
        if not _ask_approve(skill_result):
            log.info("用户取消，跳过本轮")
            return "skipped"

    if dry_run:
        log.info("[dry-run] 跳过训练，YAML=%s", yaml_path)
        return "skipped"

    # ④ 记录实验开始
    s = state_mod.load()
    s = state_mod.record_experiment_start(s, version, yaml_path, yml_path)
    state_mod.save(s)

    # ⑤ 启动训练
    session = trainer.start(yml_path, gpu=cfg["gpu"])

    # ⑥ 等待训练完成
    status = trainer.wait_until_done(
        session,
        poll_interval=cfg["poll_interval_sec"],
        timeout_hours=cfg["timeout_hours"],
    )
    if status == "timeout":
        log.error("训练超时，跳过评估")
        s = state_mod.load()
        s["current_experiment"] = None
        state_mod.save(s)
        return "failed"

    # ⑦ 检查训练是否成功
    if not trainer.check_train_success(yml_path):
        log.error("训练失败（崩溃或无输出），跳过评估")
        s = state_mod.load()
        s["current_experiment"] = None
        state_mod.save(s)
        return "failed"

    # ⑧ 提取指标
    try:
        ap, ap50 = evaluator.get_ap(yml_path, gpu=cfg["gpu"], auto_eval=cfg["auto_eval"])
    except Exception as e:
        log.error("指标提取失败: %s", e)
        s = state_mod.load()
        s["current_experiment"] = None
        state_mod.save(s)
        return "failed"

    # ⑨ 比较并更新状态
    s = state_mod.load()
    best_ap = state_mod.get_best_ap(s) or 0.0
    kept = ap > best_ap
    s = state_mod.record_experiment_result(s, ap, ap50, strategy, kept)
    state_mod.save(s)

    if kept:
        log.info("AP 提升: %.4f → %.4f (+%.4f)，保留", best_ap, ap, ap - best_ap)
        return "improved"
    else:
        log.info("AP 未提升: %.4f vs best=%.4f，回退", ap, best_ap)
        return "no_improve"


# ── 主循环 ────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-Experiment-Loop")
    parser.add_argument("--max-iter",  type=int, default=None, help="最大迭代轮数")
    parser.add_argument("--dry-run",   action="store_true",    help="只生成 YAML，不训练")
    parser.add_argument("--init",      action="store_true",    help="初始化 state.json")
    parser.add_argument("--gpu",       type=str, default=None, help="GPU 编号，如 '0' 或 '0,1'")
    args = parser.parse_args()

    _setup_logging(LOOP_LOG)
    log = logging.getLogger(__name__)

    if args.init:
        _init_state()
        return

    cfg = _load_config()
    if args.gpu:
        cfg["gpu"] = args.gpu
    max_iter = args.max_iter or cfg["max_iterations"]

    log.info("Auto-Experiment-Loop 启动 (max_iter=%d, gpu=%s, dry_run=%s)",
             max_iter, cfg["gpu"], args.dry_run)

    no_improve_count = 0
    for i in range(max_iter):
        log.info("── 轮次 %d/%d ──", i + 1, max_iter)
        outcome = _run_one_iteration(cfg, dry_run=args.dry_run)

        if outcome == "improved":
            no_improve_count = 0
        elif outcome in ("no_improve", "failed"):
            no_improve_count += 1
        # skipped 不计入

        if no_improve_count >= cfg["max_no_improve"]:
            log.warning("连续 %d 轮未提升，停止迭代", cfg["max_no_improve"])
            break

        if args.dry_run:
            log.info("[dry-run] 单轮完成，退出")
            break

        # 轮次间短暂等待，避免立即触发下一轮
        time.sleep(5)

    log.info("Auto-Experiment-Loop 结束")


if __name__ == "__main__":
    main()
