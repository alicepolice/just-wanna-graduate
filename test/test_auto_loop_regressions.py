from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from auto_loop import config, evaluator, state, trainer


@pytest.mark.parametrize(
    "module_name",
    [
        "auto_loop.skill_runner",
        "auto_loop.trainer",
        "auto_loop.evaluator",
        "auto_loop.state",
        "auto_loop.auto_loop",
    ],
)
def test_package_imports_work(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_reserve_iteration_prevents_version_reuse() -> None:
    data = state._empty_state()

    state.reserve_iteration(data, "v2")
    assert data["iteration"] == 2
    assert state.next_version(data) == "v3"

    state.record_experiment_start(data, "v2", "demo.yaml", "demo.yml")
    state.record_experiment_result(data, ap=0.2, ap50=0.4, strategy_name="demo", kept=True)

    assert data["iteration"] == 2
    assert state.next_version(data) == "v3"
    assert data["best_model"]["version"] == "v2"


def test_pick_session_name_avoids_conflicts() -> None:
    assert trainer._pick_session_name("train-job", {"train-job", "train-job-2"}) == "train-job-3"


def test_check_train_success_uses_output_dir_from_yml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = tmp_path / "deim"
    output_dir = project_root / "outputs" / "runs" / "demo"
    yml_path = project_root / "configs" / "demo.yml"

    output_dir.mkdir(parents=True)
    yml_path.parent.mkdir(parents=True)
    yml_path.write_text("output_dir: ./outputs/runs/demo\n", encoding="utf-8")
    (output_dir / "train.log").write_text("... Training completed ...", encoding="utf-8")

    monkeypatch.setattr(config, "DEIM_ROOT", project_root)

    assert trainer.check_train_success(str(yml_path)) is True


def test_get_eval_pth_passes_gpu_before_model_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "outputs" / "demo"
    output_dir.mkdir(parents=True)
    (output_dir / "best_stg2.pth").write_text("weights", encoding="utf-8")
    (output_dir / "args.json").write_text(
        json.dumps({"config": "/tmp/demo.yml"}),
        encoding="utf-8",
    )

    captured: list[str] = []

    def fake_run(cmd: list[str], cwd: str, capture_output: bool, text: bool):
        captured[:] = cmd
        eval_dir = output_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "eval.pth").write_text("ok", encoding="utf-8")

        class Result:
            returncode = 0
            stderr = ""

        return Result()

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    eval_path = evaluator.get_eval_pth(output_dir, gpu="3", auto_eval=True)

    assert eval_path == output_dir / "eval" / "eval.pth"
    assert captured == ["bash", str(config.TEST_SH), "/tmp/demo.yml", "3", "best_stg2.pth"]
