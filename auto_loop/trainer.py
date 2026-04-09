"""
训练调度 — 调用 train.sh 启动训练，通过 tmux 监控完成状态。
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path

try:
    from .config import DEIM_ROOT, TRAIN_SH, resolve_output_dir
except ImportError:
    from config import DEIM_ROOT, TRAIN_SH, resolve_output_dir

logger = logging.getLogger(__name__)

# tmux session 名 = yml 文件名（不含路径和 .yml 后缀），与 train.sh 逻辑一致
def _session_name(yml_path: str) -> str:
    return Path(yml_path).stem


def _pick_session_name(base_session: str, existing_sessions: set[str]) -> str:
    """与 train.sh 保持一致：冲突时追加 -2, -3, ...。"""
    if base_session not in existing_sessions:
        return base_session

    suffix = 2
    while f"{base_session}-{suffix}" in existing_sessions:
        suffix += 1
    return f"{base_session}-{suffix}"


def start(yml_path: str, gpu: str = "0") -> str:
    """
    启动训练，返回 tmux session 名。
    由 trainer 自己创建 detached tmux session，在里面运行 train.sh。
    train.sh 检测到已在 tmux 中会跳过自己的 tmux 创建，直接执行 torchrun。
    训练完命令退出，session 自动消失。
    """
    base_session = _session_name(yml_path)
    session = _pick_session_name(base_session, _list_sessions())
    cmd = [
        "tmux", "new-session", "-d", "-s", session, "--",
        "bash", str(TRAIN_SH), yml_path, gpu,
    ]
    logger.info("启动训练: %s (GPU=%s, session=%s)", yml_path, gpu, session)
    subprocess.run(cmd, cwd=str(DEIM_ROOT), check=True)
    logger.info("tmux session '%s' 已创建", session)
    return session


def _session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
    )
    return result.returncode == 0


def _list_sessions() -> set[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#S"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _capture_ep_info(session: str) -> str:
    """从 tmux session 的最后几行中提取 ep 信息（如 ep102-240）。"""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p", "-S", "-5"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return ""
        lines = result.stdout.strip().splitlines()
        # 从后往前找包含 ep 信息的行
        ep_pattern = re.compile(r"ep\d+[-/]\d+")
        for line in reversed(lines):
            match = ep_pattern.search(line)
            if match:
                return match.group()
        return ""
    except Exception:
        return ""


def wait_until_done(session: str, poll_interval: int = 300, timeout_hours: float = 24.0) -> str:
    """
    阻塞等待训练完成。
    返回 'done' | 'timeout' | 'not_found'。
    poll_interval: 轮询间隔（秒），默认 5 分钟
    timeout_hours: 超时时间（小时）
    """
    max_polls = int(timeout_hours * 3600 / poll_interval)
    logger.info("等待训练完成 (session=%s, 超时=%.1fh)...", session, timeout_hours)

    # 启动后稍等 3 秒再做首次快照，给 torchrun 时间输出第一行
    time.sleep(3)
    ep_info = _capture_ep_info(session)
    if ep_info:
        logger.info("训练已启动 [%s]", ep_info)

    for i in range(max_polls):
        time.sleep(poll_interval)
        if not _session_exists(session):
            logger.info("tmux session '%s' 已消失，训练结束", session)
            return "done"
        elapsed_h = (i + 1) * poll_interval / 3600
        ep_info = _capture_ep_info(session)
        if ep_info:
            logger.info("训练进行中... (%.1fh / %.1fh) [%s]", elapsed_h, timeout_hours, ep_info)
        else:
            logger.info("训练进行中... (%.1fh / %.1fh)", elapsed_h, timeout_hours)

    logger.warning("训练超时 (%.1fh)，强制终止 session '%s'", timeout_hours, session)
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    return "timeout"


def check_train_success(yml_path: str) -> bool:
    """
    检查训练是否正常完成。
    判定顺序：
      1. best_stg*.pth 存在 → 成功（有最优模型保存）
      2. last.pth 存在 + log 中有完成关键词 → 成功（训练完整跑完但 AP=0 没保存 best）
      3. log 中有 Error/Traceback → 失败
      4. 其余 → 警告并假设失败
    """
    try:
        output_dir = resolve_output_dir(yml_path)
    except Exception as exc:
        logger.warning("解析输出目录失败: %s", exc)
        return False

    if not output_dir.is_dir():
        logger.warning("找不到输出目录: %s", output_dir)
        return False

    # 1. best_stg*.pth 优先
    best_ptns = list(output_dir.glob("best_stg*.pth"))
    if best_ptns:
        logger.info("训练成功，找到: %s", best_ptns[0].name)
        return True

    # 2. last.pth + log 关键词
    last_pth = output_dir / "last.pth"
    for log_file in (output_dir / "train.log", output_dir / "log.txt"):
        if not log_file.exists():
            continue
        tail = log_file.read_text(encoding="utf-8", errors="ignore")[-5000:]
        has_error = "Error" in tail or "Traceback" in tail
        if has_error:
            logger.error("训练 log 中发现错误:\n%s", tail[-800:])
            return False
        # 识别多种"训练结束"信号
        completed = any(kw in tail for kw in (
            "Training completed",
            "Training time",     # DEIM 训练结束时输出 "Training time 0:xx:xx"
            "best_ap",
        ))
        if completed:
            if last_pth.exists():
                logger.info("训练成功（AP 未超基线，未保存 best，但 last.pth 存在）")
            else:
                logger.info("训练成功（log 有完成标志）")
            return True

    # 3. last.pth 存在但 log 没有明确标志，也算成功（保守）
    if last_pth.exists():
        logger.info("训练成功（last.pth 存在）")
        return True

    logger.warning("无法确认训练状态，假设失败")
    return False
