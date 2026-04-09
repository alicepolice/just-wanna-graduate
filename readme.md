<p align="center">
  <img src="logo.svg" alt="I Just Wanna Graduate" width="250"/>
</p>

<h1 align="center">🧵 Just Wanna Graduate</h1>

<p align="center">
  <em>学术裁缝 Agent —— 让 AI 替你缝，你只管毕业</em>
</p>

<p align="center">
  专为 <strong>DETR / D-FINE / DEIM</strong> 系列目标检测框架设计的自动化实验迭代 Agent
</p>

<p align="center">
  <code>claude-agent-sdk</code> + <code>tmux</code> + <code>COCO eval</code> 驱动的全自动闭环
</p>

> **WIP** — 项目仍在迭代中，暂不建议直接使用。计划暑假前完成稳定版本。

---

## 为什么学术裁缝可以被 AI 替代？

DETR 系列（DETR → DAB-DETR → DN-DETR → DINO → D-FINE → DEIM）是当前目标检测的主流方向，结构清晰、模块化程度高，非常适合"搭积木式"改进。大部分实验改进的真实工作流：

```
搭好实验平台（基准模型 + 模块库 + 网络配置）
→ 换个 Backbone → 不行
→ 加个注意力模块 → 不行
→ 看篇新 Paper，抄个模块过来（也就百来行代码）
→ 缝到网络不同位置，每个都试一遍
→ 跑实验看指标
→ 涨了？留下。没涨？换个模块继续
→ 指标够了 → 编个故事应付审稿人
```

整个过程没有真正的创新，只是排列组合和不断试错。核心模块代码通常就几十到一百多行，剩下全是复制粘贴改配置。

这不就是 AI Agent 最擅长的事吗？

## 这个 Agent 做什么

本 Agent **只负责实验改进的自动化迭代部分**。搭建可复现基准模型的实验平台和论文写作需要你自行完成。

```
分析当前网络结构
  → 选择改进策略（避免重复已尝试的方向）
  → 生成新的模型 YAML + 训练 YML
  → 验证配置合法性
  → 自动训练 + 评估
  → AP 提升 → 保留；未提升 → 回退换路
  → 持续迭代，直到达标或耗尽策略
```

## 项目结构

```
just-wanna-graduate/
├── start.sh              # 入口脚本：激活 deim 环境，检查依赖后启动 auto_loop
├── loop.log              # 运行日志（rich 终端输出 + 文件日志）
├── auto_loop/            # 核心代码
│   ├── auto_loop.py      # 主控制器：参数解析、迭代主循环、YML 验证、停止条件
│   ├── config.py         # 集中配置：DEIM_ROOT、脚本路径、迭代参数、Claude 模型
│   ├── state.py          # 状态管理：原子写入 state.json，维护 best/history/current_experiment
│   ├── skill_runner.py   # 调用 claude-agent-sdk 执行 Skill，流式输出并解析结果 JSON
│   ├── trainer.py        # 训练调度：tmux 启动 train.sh，前缀追踪 session，超时终止
│   └── evaluator.py      # 结果提取：读取 eval.pth；缺失时自动跑 test.sh
├── state/
│   ├── state.json        # 主状态文件（自动生成）
│   └── state_backup.json # 状态备份
└── logo.svg
```

## 工作原理

### 单轮迭代流程

```
① skill_runner    读取当前 best_model、tried_strategies、最近 10 轮 history
                   调用 claude-agent-sdk 执行学术裁缝 Skill
                   要求 Skill 在回复末尾输出 AUTO_LOOP_RESULT JSON：
                   version / yaml_path / yml_path / strategy_name / record_path
                         ↓
② auto_loop       预留 iteration 版本号，避免失败重跑覆盖同名产物
                   调用 get_info.py 校验训练 YML：
                   python tools/benchmark/get_info.py -c <yml_path>
                   校验失败 → 本轮直接 failed
                         ↓
③ auto_loop       若 AUTO_APPROVE=False，则展示策略名、YAML、YML 等待用户确认
                   dry-run 模式下到这里直接退出，不启动训练
                         ↓
④ trainer         记录 current_experiment
                   在 detached tmux session 中执行：
                   bash scripts/train.sh <yml_path> <gpu>
                   session 名默认取 yml 文件名；若冲突则自动追加 -2、-3...
                   若训练期间 session 被 rename 成 xxx_ep5-9，也会继续按前缀追踪
                         ↓
⑤ trainer         轮询 tmux session 是否仍存在
                   超过 TIMEOUT_HOURS → kill-session 并记为 failed
                   正常结束后检查输出目录：
                   best_stg*.pth / last.pth / train.log / log.txt
                         ↓
⑥ evaluator       从 output_dir/eval/eval.pth 提取 COCO 指标
                   若 eval.pth 不存在且 AUTO_EVAL=True：
                   自动执行 bash scripts/test.sh <yml_path> <gpu> <best_pth_name>
                   提取 AP、AP50、AP75、APs、APm、APl、AR1、AR10、AR100、ARs、ARm、ARl
                         ↓
⑦ state           将本轮结果写入 history
                   若 AP > 当前 best_ap → 更新 best_model，status=kept
                   否则 status=discarded
                   strategy_name 会加入 tried_strategies
```

### 状态持久化

所有迭代状态保存在 `state/state.json`，写入方式是原子替换（先写 `.tmp` 再 replace），当前结构如下：

| 字段 | 说明 |
|------|------|
| `iteration` | 当前已预留的最大版本号，用来避免失败后重跑覆盖同名产物 |
| `best_model` | 当前最优模型的 yaml/yml 路径、AP、AP50、版本号 |
| `tried_strategies` | 已尝试的改进策略名列表，传给 Skill 用于去重 |
| `history` | 每轮实验的 version、AP、AP50、delta、kept、strategy 记录 |
| `current_experiment` | 当前正在训练中的实验，开始训练时写入，结束后清空 |

补充说明：

- `next_version()` 基于 `iteration + 1` 生成版本号，例如 `v14`
- `reserve_iteration()` 会在 YAML 生成后立刻预留版本号
- 只有 `AP > best_ap` 才会更新 `best_model`，相等不保留
- `skipped` 不计入连续未提升次数；`failed` 和 `no_improve` 会计入

## 快速开始

### 前置依赖

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` 命令可用)
- `claude-agent-sdk` Python 包
- `tmux`
- `rich`
- PyTorch + DEIM 项目环境
- conda 环境名为 `deim`（`start.sh` 当前直接执行 `conda activate deim`）

安装 Python 依赖：

```bash
pip install claude-agent-sdk rich
```

### 外部 DEIM 工程要求

本仓库并不是独立训练框架，默认依赖外部 DEIM 工程提供这些文件：

- `scripts/train.sh`
- `scripts/test.sh`
- `tools/benchmark/get_info.py`
- `.claude/skills/缝合任务/SKILL.md`

默认 DEIM 根目录配置在 `auto_loop/config.py`：

```python
DEIM_ROOT = Path(os.environ.get("DEIM_ROOT", "/home/exp/DEIM-MOD-V2"))
```

但要注意：`start.sh` 里对 DEIM 路径的检查目前仍然写死成 `/home/exp/DEIM-MOD-V2`，所以如果你只改环境变量、不改 `start.sh`，启动前检查仍可能失败。

### 使用

```bash
# 首次使用：交互式初始化当前最优模型信息
./start.sh --init

# 启动自动迭代
./start.sh

# 只生成 YAML 方案并执行 get_info 校验，不实际训练
./start.sh --dry-run

# 指定 GPU
./start.sh --gpu 0
./start.sh --gpu 0,1

# 限制迭代轮数
./start.sh --max-iter 5
```

初始化时会要求输入：

- 当前迭代版本号
- 当前最优模型 YAML 路径
- 当前最优训练 YML 路径
- 当前最优 AP
- 当前最优 AP50
- 已尝试策略列表

### 配置项

编辑 `auto_loop/config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEIM_ROOT` | `/home/exp/DEIM-MOD-V2` | DEIM 实验项目根目录 |
| `MAX_ITERATIONS` | `50` | 最大迭代轮数 |
| `MAX_NO_IMPROVE` | `5` | 连续多少轮 `failed` / `no_improve` 后停止 |
| `AUTO_APPROVE` | `True` | 是否自动批准每轮方案 |
| `GPU` | `"0"` | 默认训练 GPU 编号 |
| `POLL_INTERVAL_SEC` | `10` | 训练状态轮询间隔（秒） |
| `TIMEOUT_HOURS` | `24.0` | 单轮训练超时（小时） |
| `AUTO_EVAL` | `True` | 缺少 `eval.pth` 时是否自动跑 `test.sh` |
| `CONDA_ENV` | `"deim"` | 配置文件中的环境名；当前 `start.sh` 仍写死为 `deim` |
| `CLAUDE_MODEL` | `"claude-sonnet-4-6"` | 调用的 Claude 模型 ID |
| `CLAUDE_EFFORT` | `"medium"` | thinking effort |

日志同时输出到终端和 `loop.log`，Skill 的流式输出也会同步写入日志文件。

## 计划特性

- [ ] 用小型数据集验证完整工具流程
- [ ] 迭代优化 Skill Prompt，提升模块替换策略的灵活性
- [ ] 增强日志与实验记录
- [ ] 多卡多方案并行搜索树

---

<p align="center">
  <em>如果一个研究流程可以被 AI 完全自动化，那它本身就不该被称为"研究"</em>
</p>

---

## Related Work

- [academic-research-skills](https://github.com/Imbad0202/academic-research-skills) — Academic Research Skills for Claude Code
- [Awesome-Deep-Research](https://github.com/DavidZWZ/Awesome-Deep-Research) — Awesome Agentic Deep Research Resources
- [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) — Workshop-Level Automated Scientific Discovery via Agentic Tree Search
- [AI-Researcher](https://github.com/HKUDS/AI-Researcher) — Autonomous Scientific Innovation
- [AgentLaboratory](https://github.com/SamuelSchmidgall/AgentLaboratory) — End-to-end Autonomous Research Workflow
- [De-Anthropocentric-Research-Engine](https://github.com/Pthahnix/De-Anthropocentric-Research-Engine) — AI-powered Academic Research Automation
- [AIDE](https://github.com/WecoAI/aideml) — AI-Driven Exploration in the Space of Code
- [DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze) — Agentic LLM for Autonomous Data Science
- [OpenHands](https://github.com/OpenHands/OpenHands) — AI-Driven Development
- [Aider](https://github.com/Aider-AI/aider) — AI Pair Programming in Your Terminal
- [mcp-agent](https://github.com/lastmile-ai/mcp-agent) — Build Agents Using Model Context Protocol
