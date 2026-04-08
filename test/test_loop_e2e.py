#!/usr/bin/env python3
"""
端到端快速测试 — mock 掉 skill / trainer / evaluator，
让 auto_loop 主流程在几秒内跑完，验证状态流转是否正确。

包含一个真实调用 Claude 的集成测试（用极简 skill 替代复杂的学术裁缝 skill）。

用法:
    pytest test/test_loop_e2e.py -v                    # 跑全部（含 Claude 调用）
    pytest test/test_loop_e2e.py -v -k "not claude"    # 只跑 mock 测试
    pytest test/test_loop_e2e.py -v -k "claude"        # 只跑 Claude 集成测试
    python test/test_loop_e2e.py                       # 直接运行
"""
from __future__ import annotations

import json
import logging
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# 确保项目根目录在 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auto_loop import state as state_mod, config
from auto_loop.auto_loop import _run_one_iteration, _load_config


# ── Fake Skill Runner ────────────────────────────────────────────
# 不调用 Claude，直接返回一个假方案（相当于 "算 1+1=2" 级别的 skill）
_FAKE_STRATEGIES = [
    "把学习率从 0.001 调到 0.0005",
    "在 backbone 后面加一层 dropout(0.1)",
    "把 batch_size 从 16 改成 32",
]


def _make_fake_skill_runner(tmp_deim: Path, call_count: list[int]):
    """返回一个 fake skill_runner.run，每次调用生成不同策略。"""

    def fake_run(state: dict, next_version: str, max_retries: int = 2) -> dict:
        idx = call_count[0] % len(_FAKE_STRATEGIES)
        call_count[0] += 1
        strategy = _FAKE_STRATEGIES[idx]

        # 创建假的 yaml / yml 文件
        yaml_dir = tmp_deim / "configs_lab" / "test" / "models"
        yml_dir = tmp_deim / "configs_lab" / "test" / "train"
        yaml_dir.mkdir(parents=True, exist_ok=True)
        yml_dir.mkdir(parents=True, exist_ok=True)

        yaml_path = yaml_dir / f"model-{next_version}.yaml"
        yml_path = yml_dir / f"train-{next_version}.yml"

        # 写一个最简 yaml/yml，包含 output_dir 让 evaluator 能解析
        output_dir = tmp_deim / "outputs" / f"run-{next_version}"
        yaml_path.write_text(f"# fake model config {next_version}\nmodel: fake\n")
        yml_path.write_text(f"output_dir: {output_dir}\nepochs: 1\n")

        return {
            "version": next_version,
            "yaml_path": str(yaml_path),
            "yml_path": str(yml_path),
            "strategy_name": strategy,
            "record_path": "",
            "get_info_passed": True,
        }

    return fake_run


# ── Fake Trainer ─────────────────────────────────────────────────
# 不启动 tmux，不等待，直接"训练完成"
def _make_fake_trainer(tmp_deim: Path):
    """返回 mock 过的 trainer 函数集。"""

    def fake_start(yml_path: str, gpu: str = "0") -> str:
        # 模拟训练产出：创建 output_dir 和 best_stg2.pth
        import yaml as _yaml
        with open(yml_path) as f:
            data = _yaml.safe_load(f)
        out = Path(data["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "best_stg2.pth").write_text("fake_weights")
        (out / "train.log").write_text("Training completed successfully\n")
        return f"fake-session-{Path(yml_path).stem}"

    def fake_wait(session: str, poll_interval: int = 300, timeout_hours: float = 24.0) -> str:
        return "done"

    def fake_check(yml_path: str) -> bool:
        return True

    return fake_start, fake_wait, fake_check


# ── Fake Evaluator ───────────────────────────────────────────────
# 返回递增的 AP，模拟"每轮都有提升"或"偶尔不提升"
def _make_fake_evaluator(ap_sequence: list[tuple[float, float]]):
    """ap_sequence: [(ap, ap50), ...] 按轮次返回。"""
    idx = [0]

    def fake_get_ap(yml_path: str, gpu: str = "0", auto_eval: bool = True) -> tuple[float, float]:
        i = min(idx[0], len(ap_sequence) - 1)
        idx[0] += 1
        return ap_sequence[i]

    return fake_get_ap


# ── Fixtures ─────────────────────────────────────────────────────
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """创建一个临时沙箱环境，重定向所有路径。"""
    tmp_deim = tmp_path / "DEIM-FAKE"
    tmp_deim.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # 重定向 config 中的路径
    monkeypatch.setattr(config, "DEIM_ROOT", tmp_deim)
    monkeypatch.setattr(config, "STATE_FILE", config_dir / "state.json")
    monkeypatch.setattr(config, "OUTPUTS_ROOT", tmp_deim / "outputs")

    # 初始化 state
    initial_state = state_mod._empty_state()
    initial_state["iteration"] = 1
    initial_state["best_model"] = {
        "yaml": "configs_lab/test/models/model-v1.yaml",
        "yml": "configs_lab/test/train/train-v1.yml",
        "ap": 0.30,
        "ap50": 0.50,
        "version": "v1",
    }
    state_mod.save(initial_state)

    return tmp_path, tmp_deim


# ── Tests ────────────────────────────────────────────────────────
class TestLoopE2E:
    """端到端测试：mock 所有外部依赖，快速跑完整个迭代流程。"""

    def test_single_iteration_improved(self, sandbox):
        """单轮迭代，AP 提升 → state 更新为新的 best。"""
        tmp_path, tmp_deim = sandbox
        call_count = [0]

        fake_start, fake_wait, fake_check = _make_fake_trainer(tmp_deim)
        fake_get_ap = _make_fake_evaluator([(0.35, 0.55)])  # 比 0.30 高

        cfg = _load_config()
        cfg["auto_approve"] = True

        with patch("auto_loop.auto_loop.skill_runner") as mock_skill, \
             patch("auto_loop.auto_loop.trainer") as mock_trainer, \
             patch("auto_loop.auto_loop.evaluator") as mock_eval:

            mock_skill.run = _make_fake_skill_runner(tmp_deim, call_count)
            mock_trainer.start = fake_start
            mock_trainer.wait_until_done = fake_wait
            mock_trainer.check_train_success = fake_check
            mock_eval.get_ap = fake_get_ap

            result = _run_one_iteration(cfg, dry_run=False)

        assert result == "improved"

        s = state_mod.load()
        assert s["best_model"]["ap"] == 0.35
        assert s["best_model"]["version"] == "v2"
        assert len(s["history"]) == 1
        assert s["history"][0]["kept"] is True
        assert s["current_experiment"] is None

    def test_single_iteration_no_improve(self, sandbox):
        """单轮迭代，AP 没提升 → state 保持旧 best。"""
        tmp_path, tmp_deim = sandbox
        call_count = [0]

        fake_start, fake_wait, fake_check = _make_fake_trainer(tmp_deim)
        fake_get_ap = _make_fake_evaluator([(0.28, 0.45)])  # 比 0.30 低

        cfg = _load_config()
        cfg["auto_approve"] = True

        with patch("auto_loop.auto_loop.skill_runner") as mock_skill, \
             patch("auto_loop.auto_loop.trainer") as mock_trainer, \
             patch("auto_loop.auto_loop.evaluator") as mock_eval:

            mock_skill.run = _make_fake_skill_runner(tmp_deim, call_count)
            mock_trainer.start = fake_start
            mock_trainer.wait_until_done = fake_wait
            mock_trainer.check_train_success = fake_check
            mock_eval.get_ap = fake_get_ap

            result = _run_one_iteration(cfg, dry_run=False)

        assert result == "no_improve"

        s = state_mod.load()
        assert s["best_model"]["ap"] == 0.30  # 没变
        assert s["best_model"]["version"] == "v1"
        assert len(s["history"]) == 1
        assert s["history"][0]["kept"] is False

    def test_dry_run_skips_training(self, sandbox):
        """dry-run 模式：只调用 skill，不训练。"""
        tmp_path, tmp_deim = sandbox
        call_count = [0]

        cfg = _load_config()
        cfg["auto_approve"] = True

        with patch("auto_loop.auto_loop.skill_runner") as mock_skill, \
             patch("auto_loop.auto_loop.trainer") as mock_trainer, \
             patch("auto_loop.auto_loop.evaluator") as mock_eval:

            mock_skill.run = _make_fake_skill_runner(tmp_deim, call_count)

            result = _run_one_iteration(cfg, dry_run=True)

        assert result == "skipped"
        # trainer 和 evaluator 不应被调用
        mock_trainer.start.assert_not_called()
        mock_eval.get_ap.assert_not_called()

        # state 不应有 history 记录
        s = state_mod.load()
        assert len(s["history"]) == 0

    def test_multi_iteration_state_accumulates(self, sandbox):
        """连续 3 轮迭代，验证 state 正确累积。"""
        tmp_path, tmp_deim = sandbox
        call_count = [0]

        fake_start, fake_wait, fake_check = _make_fake_trainer(tmp_deim)
        # 第1轮提升，第2轮不提升，第3轮提升
        ap_seq = [(0.35, 0.55), (0.33, 0.52), (0.40, 0.60)]
        fake_get_ap = _make_fake_evaluator(ap_seq)

        cfg = _load_config()
        cfg["auto_approve"] = True

        results = []
        for _ in range(3):
            with patch("auto_loop.auto_loop.skill_runner") as mock_skill, \
                 patch("auto_loop.auto_loop.trainer") as mock_trainer, \
                 patch("auto_loop.auto_loop.evaluator") as mock_eval:

                mock_skill.run = _make_fake_skill_runner(tmp_deim, call_count)
                mock_trainer.start = fake_start
                mock_trainer.wait_until_done = fake_wait
                mock_trainer.check_train_success = fake_check
                mock_eval.get_ap = fake_get_ap

                r = _run_one_iteration(cfg, dry_run=False)
                results.append(r)

        assert results == ["improved", "no_improve", "improved"]

        s = state_mod.load()
        assert len(s["history"]) == 3
        assert s["best_model"]["ap"] == 0.40
        assert s["best_model"]["version"] == "v4"  # v1 初始, v2 提升, v3 不提升, v4 提升
        assert len(s["tried_strategies"]) == 3

    def test_skill_failure_returns_failed(self, sandbox):
        """skill_runner 抛异常 → 返回 failed，state 不变。"""
        tmp_path, tmp_deim = sandbox

        cfg = _load_config()
        cfg["auto_approve"] = True

        with patch("auto_loop.auto_loop.skill_runner") as mock_skill:
            mock_skill.run = MagicMock(side_effect=RuntimeError("Claude API 炸了"))

            result = _run_one_iteration(cfg, dry_run=False)

        assert result == "failed"

        s = state_mod.load()
        assert len(s["history"]) == 0
        assert s["best_model"]["ap"] == 0.30


# ══════════════════════════════════════════════════════════════════
# 真实 Claude 集成测试 — 用极简 skill 走完 skill_runner 全链路
# ══════════════════════════════════════════════════════════════════

# 极简 skill：让 Claude 在 tmp 目录创建一个文件，然后输出约定格式的 JSON
_TRIVIAL_SKILL = textwrap.dedent("""\
    你是一个测试用的极简 skill。你的任务非常简单：

    1. 在当前工作目录下创建一个文件 `test_artifact.txt`，内容写 `1+1=2`
    2. 然后在回复的**最末尾**，严格按以下格式输出 JSON（不要修改格式）：

    ```json
    {AUTO_LOOP_RESULT}
    {
      "version": "<用户提供的版本号>",
      "yaml_path": "configs_lab/test/models/fake-model.yaml",
      "yml_path": "configs_lab/test/train/fake-train.yml",
      "strategy_name": "测试策略-算个1加1",
      "record_path": "",
      "get_info_passed": true
    }
    ```

    注意：version 字段请使用用户消息中提到的版本号。
""")


class TestClaudeIntegration:
    """真实调用 Claude 的集成测试，验证 skill_runner 的完整链路。"""

    def test_claude_skill_runner_e2e(self, tmp_path, monkeypatch):
        """
        用极简 skill 替代学术裁缝 skill，真正走一遍：
        skill_runner.run() → _call_claude() → claude-agent-sdk → _parse_output()

        验证：
        - Claude 被成功调用
        - 输出被正确解析为包含 yaml_path/yml_path/strategy_name 的 dict
        - Claude 确实创建了 test_artifact.txt 文件
        """
        from auto_loop import skill_runner

        tmp_deim = tmp_path / "DEIM-FAKE"
        tmp_deim.mkdir()

        # 写一个假的 SKILL 文件
        skill_dir = tmp_deim / ".claude" / "skills" / "缝合任务"
        skill_dir.mkdir(parents=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(_TRIVIAL_SKILL, encoding="utf-8")

        # 重定向 config
        monkeypatch.setattr(skill_runner, "SKILL_FILE", skill_file)
        monkeypatch.setattr(skill_runner, "DEIM_ROOT", tmp_deim)

        # 构造一个最简 state
        state = {
            "iteration": 1,
            "best_model": {
                "yaml": "configs_lab/test/models/model-v1.yaml",
                "yml": "configs_lab/test/train/train-v1.yml",
                "ap": 0.30,
                "ap50": 0.50,
                "version": "v1",
            },
            "tried_strategies": [],
            "history": [],
            "current_experiment": None,
        }

        # 真正调用 Claude！
        result = skill_runner.run(state, "v2", max_retries=1)

        # 验证返回的 dict 包含必要字段
        assert result is not None, "skill_runner.run() 返回 None，Claude 输出解析失败"
        assert "yaml_path" in result
        assert "yml_path" in result
        assert "strategy_name" in result
        assert result["version"] == "v2"

        print(f"\n[Claude 返回] strategy={result['strategy_name']}")
        print(f"  yaml_path={result['yaml_path']}")
        print(f"  yml_path={result['yml_path']}")

        # 验证 Claude 确实创建了文件
        artifact = tmp_deim / "test_artifact.txt"
        assert artifact.exists(), "Claude 没有创建 test_artifact.txt"
        content = artifact.read_text().strip()
        assert "1" in content and "2" in content, f"文件内容不对: {content}"
        print(f"  artifact 内容: {content}")


# ── 直接运行入口 ─────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
