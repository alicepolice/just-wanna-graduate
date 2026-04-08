"""
结果提取 — 从训练输出目录读取 eval.pth，提取 COCO 指标。
若 eval.pth 不存在，尝试运行 test.sh 生成。
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import torch

try:
    from .config import DEIM_ROOT, OUTPUTS_ROOT, TEST_SH, resolve_output_dir
except ImportError:
    from config import DEIM_ROOT, OUTPUTS_ROOT, TEST_SH, resolve_output_dir

logger = logging.getLogger(__name__)


def find_output_dir(yml_path: str) -> Path:
    """从训练 yml 文件名推断 outputs/ 下的目录名。"""
    output_dir = resolve_output_dir(yml_path)
    if output_dir.is_dir():
        return output_dir
    raise FileNotFoundError(f"找不到训练输出目录: {output_dir}，已搜索 {OUTPUTS_ROOT}")


def get_eval_pth(output_dir: Path, gpu: str = "0", auto_eval: bool = True) -> Path:
    """返回 eval.pth 路径，不存在时可自动运行 test.sh。"""
    eval_pth = output_dir / "eval" / "eval.pth"
    if eval_pth.exists():
        return eval_pth

    if not auto_eval:
        raise FileNotFoundError(f"eval.pth 不存在: {eval_pth}")

    logger.info("eval.pth 不存在，运行 test.sh 生成...")
    best_pth = output_dir / "best_stg2.pth"
    if not best_pth.exists():
        best_pth = output_dir / "best_stg1.pth"
    if not best_pth.exists():
        raise FileNotFoundError(f"找不到 best_stg*.pth in {output_dir}")

    # 从 args.json 读取训练 yml 路径
    args_json = output_dir / "args.json"
    if not args_json.exists():
        raise FileNotFoundError(f"找不到 args.json in {output_dir}")

    with open(args_json) as f:
        args = json.load(f)
    yml_path = args.get("config", "")

    cmd = [
        "bash", str(TEST_SH),
        yml_path,
        gpu,
        best_pth.name,
    ]
    logger.info("运行: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(DEIM_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("test.sh 失败:\n%s", result.stderr[-2000:])
        raise RuntimeError("test.sh 运行失败")

    if not eval_pth.exists():
        raise FileNotFoundError(f"test.sh 运行后仍找不到 eval.pth: {eval_pth}")
    return eval_pth


def extract_metrics(eval_pth: Path) -> dict:
    """从 eval.pth 提取 COCO 指标（摘自 DEIM-MOD-V2/scripts/compare_models.py）。"""
    data      = torch.load(str(eval_pth), weights_only=False)
    precision = data["precision"]   # [T, R, K, A, M]
    recall    = data["recall"]      # [T, K, A, M]
    params    = data["params"]
    iou_thrs  = params.iouThrs
    max_dets  = params.maxDets

    def _ap(iou_lo, iou_hi, area_idx, md_idx):
        t = np.where((iou_thrs >= iou_lo) & (iou_thrs <= iou_hi))[0]
        if not len(t): return -1.0
        s = precision[t, :, :, area_idx, md_idx]
        s = s[s > -1]
        return float(np.mean(s)) if s.size else -1.0

    def _ar(iou_lo, iou_hi, area_idx, md_idx):
        t = np.where((iou_thrs >= iou_lo) & (iou_thrs <= iou_hi))[0]
        if not len(t): return -1.0
        s = recall[t, :, area_idx, md_idx]
        s = s[s > -1]
        return float(np.mean(s)) if s.size else -1.0

    md = len(max_dets) - 1
    return {
        "AP":    _ap(0.50, 0.95, 0, md),
        "AP50":  _ap(0.50, 0.50, 0, md),
        "AP75":  _ap(0.75, 0.75, 0, md),
        "APs":   _ap(0.50, 0.95, 1, md),
        "APm":   _ap(0.50, 0.95, 2, md),
        "APl":   _ap(0.50, 0.95, 3, md),
        "AR1":   _ar(0.50, 0.95, 0, 0),
        "AR10":  _ar(0.50, 0.95, 0, 1),
        "AR100": _ar(0.50, 0.95, 0, md),
        "ARs":   _ar(0.50, 0.95, 1, md),
        "ARm":   _ar(0.50, 0.95, 2, md),
        "ARl":   _ar(0.50, 0.95, 3, md),
        "date":  data.get("date", "N/A"),
    }


def get_ap(yml_path: str, gpu: str = "0", auto_eval: bool = True) -> tuple[float, float]:
    """
    主入口：给定训练 yml 路径，返回 (AP, AP50)。
    """
    output_dir = find_output_dir(yml_path)
    logger.info("输出目录: %s", output_dir)
    eval_pth = get_eval_pth(output_dir, gpu=gpu, auto_eval=auto_eval)
    metrics = extract_metrics(eval_pth)
    ap   = metrics.get("AP",   -1.0)
    ap50 = metrics.get("AP50", -1.0)
    logger.info("AP=%.4f  AP50=%.4f", ap, ap50)
    return ap, ap50
