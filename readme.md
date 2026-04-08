
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

> **WIP** — 项目仍在迭代中，暂不建议直接使用。

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
├── start.sh              # 入口脚本，检查依赖后启动主循环
├── loop.log              # 运行日志
├── auto_loop/            # 核心代码
│   ├── auto_loop.py      # 主控制器，驱动整个迭代循环
│   ├── config.py         # 集中配置（路径、迭代行为、Claude 模型等）
│   ├── state.py          # 状态管理，读写 state.json 持久化迭代进度
│   ├── skill_runner.py   # 通过 claude-agent-sdk 调用学术裁缝 Skill，解析输出
│   ├── trainer.py        # 训练调度，通过 train.sh + tmux 启动并监控训练
│   └── evaluator.py      # 结果提取，从 eval.pth 读取 COCO 指标（AP / AP50）
├── state/
│   └── state.json        # 迭代状态持久化（自动生成）
└── template/             # 需求文档与参考资料
```

## 工作原理

### 单轮迭代流程

```
① skill_runner    调用 claude-agent-sdk + 学术裁缝 Skill
                   输入：当前最优模型、已尝试策略、历史记录
                   输出：新 YAML 配置 + 训练 YML + 改进策略名
                         ↓
② auto_loop       验证 YAML（调用 get_info.py 做合法性检查）
                   可选：等待用户确认（auto_approve: false 时）
                         ↓
③ trainer          调用 train.sh，在 tmux session 中启动训练
                   每 5 分钟轮询 tmux session 是否存活
                   session 消失 → 训练结束；超时 → 强制终止
                         ↓
④ evaluator        从输出目录找 eval.pth
                   若不存在，自动运行 test.sh 生成
                   提取 COCO 指标（AP / AP50 等 12 项）
                         ↓
⑤ state            比较新 AP 与历史最优
                   提升 → 保留为新 best_model
                   未提升 → 记录为 discarded，下轮换策略
                   连续 N 轮未提升 → 自动停止
```

### 状态持久化

所有迭代状态保存在 `state/state.json`，包含：

| 字段 | 说明 |
|------|------|
| `best_model` | 当前最优模型的 YAML/YML 路径、AP、AP50、版本号 |
| `tried_strategies` | 已尝试的改进策略列表（传给 Skill 避免重复） |
| `history` | 每轮实验的 AP、delta、是否保留的完整记录 |
| `current_experiment` | 当前正在进行的实验（用于异常恢复） |

进程意外中断后重启，会从 `state.json` 恢复，继续下一轮迭代。

## 快速开始

### 前置依赖

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` 命令可用)
- `claude-agent-sdk` Python 包
- `tmux`
- PyTorch + DEIM 项目环境（conda 环境 `deim`）

### 使用

```bash
# 1. 修改 auto_loop/config.py 中的 DEIM_ROOT 指向你的 DEIM 项目
# 2. 确保 DEIM 项目中存在学术裁缝 Skill 文件（.claude/skills/缝合任务/SKILL.md）

# 首次使用：交互式初始化当前最优模型信息
./start.sh --init

# 启动自动迭代（受 config.py 控制）
./start.sh

# 只生成 YAML 方案，不实际训练（用于调试 Skill 输出）
./start.sh --dry-run

# 指定 GPU 并限制轮数
./start.sh --gpu 0,1 --max-iter 5
```

### 配置项

编辑 `auto_loop/config.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MAX_ITERATIONS` | 50 | 最大迭代轮数 |
| `MAX_NO_IMPROVE` | 5 | 连续未提升轮数上限，达到后自动停止 |
| `AUTO_APPROVE` | True | 是否自动批准方案（False 时每轮需手动确认） |
| `GPU` | "0" | GPU 编号，多卡用逗号分隔 |
| `POLL_INTERVAL_SEC` | 10 | 训练完成检测轮询间隔（秒） |
| `TIMEOUT_HOURS` | 24.0 | 单次训练超时时间（小时） |
| `AUTO_EVAL` | True | 训练后自动运行评估 |

日志输出到 `loop.log`。

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