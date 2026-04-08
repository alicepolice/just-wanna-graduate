"""
训练调度 — 调用 train.sh 启动训练，通过 tmux 监控完成状态。
"""
import logging
import subprocess
import time
from pathlib import Path

from config import DEIM_ROOT, TRAIN_SH

logger = logging.getLogger(__name__)

# tmux session 名 = yml 文件名（不含路径和 .yml 后缀），与 train.sh 逻辑一致
def _session_name(yml_path: str) -> str:
    return Path(yml_path).stem


def start(yml_path: str, gpu: str = "0") -> str:
    """
    启动训练，返回 tmux session 名。
    train.sh 会自动创建 tmux session，这里只需调用它。
    """
    session = _session_name(yml_path)
    cmd = ["bash", str(TRAIN_SH), yml_path, gpu]
    logger.info("启动训练: %s (GPU=%s, session=%s)", yml_path, gpu, session)
    # 用 Popen 非阻塞启动；train.sh 内部会 detach 到 tmux
    subprocess.Popen(
        cmd,
        cwd=str(DEIM_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # 等待 tmux session 出现（最多 30 秒）
    for _ in range(30):
        time.sleep(1)
        if _session_exists(session):
            logger.info("tmux session '%s' 已创建", session)
            return session
    logger.warning("等待 tmux session '%s' 超时，可能已快速完成或失败", session)
    return session


def _session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
    )
    return result.returncode == 0


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
    from config import OUTPUTS_ROOT
    name = Path(yml_path).stem
    # 找输出目录
    output_dir = None
    for candidate in OUTPUTS_ROOT.rglob(name):
        if candidate.is_dir():
            output_dir = candidate
            break
    if output_dir is None:
        logger.warning("找不到输出目录: %s", name)
        return False

    # 检查 best_stg*.pth 是否存在
    best_ptns = list(output_dir.glob("best_stg*.pth"))
    if best_ptns:
        logger.info("训练成功，找到: %s", best_ptns[0].name)
        return True

    # 检查 log.txt
    log_file = output_dir / "log.txt"
    if log_file.exists():
        tail = log_file.read_text(encoding="utf-8", errors="ignore")[-3000:]
        if "Training completed" in tail or "best_ap" in tail.lower():
            return True
        if "Error" in tail or "Traceback" in tail:
            logger.error("训练 log 中发现错误:\n%s", tail[-500:])
            return False

    logger.warning("无法确认训练状态，假设失败")
    return False
