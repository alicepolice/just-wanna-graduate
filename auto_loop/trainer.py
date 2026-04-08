"""
训练调度 — 调用 train.sh 启动训练，通过 tmux 监控完成状态。
"""
from __future__ import annotations

import logging
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
    train.sh 会自动创建 tmux session，这里只需调用它。
    """
    base_session = _session_name(yml_path)
    before = _list_sessions()
    cmd = ["bash", str(TRAIN_SH), yml_path, gpu]
    logger.info("启动训练: %s (GPU=%s)", yml_path, gpu)
    # train.sh 内部会 detach 到 tmux，等它自身退出即可
    proc = subprocess.Popen(
        cmd,
        cwd=str(DEIM_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # 等待 tmux session 出现（最多 30 秒），通过 diff 实际检测新 session
    session = None
    for _ in range(30):
        time.sleep(1)
        after = _list_sessions()
        new_sessions = after - before
        # 优先匹配以 base_session 开头的新 session
        for s in sorted(new_sessions):
            if s == base_session or s.startswith(f"{base_session}-"):
                session = s
                break
        if session:
            break
    # 回收 train.sh 自身的进程（它 detach 后应很快退出）
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("train.sh 进程未退出，强制终止")
        proc.kill()
        proc.wait()
    if session:
        logger.info("tmux session '%s' 已创建", session)
        return session
    # fallback: 用预测名
    fallback = _pick_session_name(base_session, _list_sessions())
    logger.warning("未检测到新 tmux session，使用预测名 '%s'", fallback)
    return fallback


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


def wait_until_done(session: str, poll_interval: int = 300, timeout_hours: float = 24.0) -> str:
    """
    阻塞等待训练完成。
    返回 'done' | 'timeout' | 'not_found'。
    poll_interval: 轮询间隔（秒），默认 5 分钟
    timeout_hours: 超时时间（小时）
    """
    max_polls = int(timeout_hours * 3600 / poll_interval)
    logger.info("等待训练完成 (session=%s, 超时=%.1fh)...", session, timeout_hours)

    for i in range(max_polls):
        time.sleep(poll_interval)
        if not _session_exists(session):
            logger.info("tmux session '%s' 已消失，训练结束", session)
            return "done"
        elapsed_h = (i + 1) * poll_interval / 3600
        logger.info("训练进行中... (%.1fh / %.1fh)", elapsed_h, timeout_hours)

    logger.warning("训练超时 (%.1fh)，强制终止 session '%s'", timeout_hours, session)
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    return "timeout"


def check_train_success(yml_path: str) -> bool:
    """
    检查训练是否正常完成（log.txt 中有 'Training completed' 或存在 best_stg*.pth）。
    """
    try:
        output_dir = resolve_output_dir(yml_path)
    except Exception as exc:
        logger.warning("解析输出目录失败: %s", exc)
        return False

    if not output_dir.is_dir():
        logger.warning("找不到输出目录: %s", output_dir)
        return False

    # 检查 best_stg*.pth 是否存在
    best_ptns = list(output_dir.glob("best_stg*.pth"))
    if best_ptns:
        logger.info("训练成功，找到: %s", best_ptns[0].name)
        return True

    # 兼容 train.sh 产出的 train.log 和旧版 log.txt
    for log_file in (output_dir / "train.log", output_dir / "log.txt"):
        if not log_file.exists():
            continue
        tail = log_file.read_text(encoding="utf-8", errors="ignore")[-3000:]
        if "Training completed" in tail or "best_ap" in tail.lower():
            return True
        if "Error" in tail or "Traceback" in tail:
            logger.error("训练 log 中发现错误:\n%s", tail[-500:])
            return False

    logger.warning("无法确认训练状态，假设失败")
    return False
