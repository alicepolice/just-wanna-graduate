
<p align="center">
  <img src="logo.svg" alt="I Just Wanna Graduate" width="250"/>
</p>

<h1 align="center">🧵 I Just Wanna Graduate</h1>

<p align="center">
  <em>学术裁缝 Agent —— 让 AI 替你缝，你只管毕业</em>
</p>

<p align="center">
  🚧 计划于暑假开发，优先兼容"目标检测"项目，先占个坑
</p>

## 为什么学术裁缝可以被 AI 替代？

大部分实验改进的真实工作流：

```
学习基础 → 达到前置条件，搭好实验平台（基准模型 + 模块库 + 网络配置），可参考 YOLO Ultra
→ 理解基准模型 → 换个 Backbone → 不行
→ 加个注意力模块 → 不行
→ 看篇新 Paper，抄个模块过来（也就百来行代码）
→ 缝到网络不同位置，每个都试一遍
→ 跑实验看指标
→ 涨了？留下。没涨？换个模块继续
→ 指标够了 → 编个故事应付审稿人
```

整个过程没有真正的创新，只是排列组合和不断试错。核心模块代码通常就几十到一百多行，剩下全是复制粘贴改配置。

这不就是 AI Agent 最擅长的事吗？

## 这个 Agent 要做什么

本 Agent 只负责实验改进的自动化迭代部分。搭建可复现基准模型的实验平台和论文编故事需要你自行完成

```
设定目标指标
  → 分析当前网络结构
  → 生成改进思路
  → 检索 Paper / 创建新模块
  → 搭积木式组装到网络不同位置
  → 自动训练 + 评估
  → 涨了保留，没涨回退换路
  → 无限迭代，直到达标
```

## 计划特性

- 🔄 全自动闭环：分析 → 改进 → 实验 → 评估
- 🧱 模块库：几百个可插拔模块，像乐高一样搭网络
- 📄 Paper-Aware：自动检索论文，提取核心模块
- 🌳 多卡多方案搜索树：并行探索多条改进路径

## 当前状态

> 项目仍在迭代中，暂不建议直接使用。

**近期计划：**

- [ ] 用小型数据集验证完整工具流程
- [ ] 迭代优化 skill.md，提升模块替换策略的灵活性（如参数调整、通道裁剪等）
- [ ] 增强日志记录
- [ ] 重新设计指标评估方案


## 工作原理

`auto_loop/` 目录实现了完整的自动化迭代闭环，由以下模块组成：

```
auto_loop/
├── auto_loop.py     # 主控制器，驱动整个迭代循环
├── config.py        # 路径配置（DEIM_ROOT、脚本路径、输出目录等）
├── state.py         # 状态管理，读写 config/state.json 持久化迭代进度
├── skill_runner.py  # 调用 claude -p 执行学术裁缝 Skill，解析输出
├── trainer.py       # 训练调度，通过 train.sh + tmux 启动并监控训练
└── evaluator.py     # 结果提取，从 eval.pth 读取 COCO 指标

config/
├── loop_config.yml  # 迭代行为配置（轮数、GPU、超时等）
└── state.json       # 迭代状态持久化（自动生成）
```

### 单轮迭代流程

```
① skill_runner  →  调用 claude -p + 学术裁缝 Skill
                    输入：当前最优模型、已尝试策略、历史记录
                    输出：新 YAML 配置 + 训练 YML + 改进策略名

② auto_loop     →  验证 YAML（调用 get_info.py 做合法性检查）
                    可选：等待用户确认（auto_approve=false 时）

③ trainer       →  调用 train.sh，在 tmux session 中启动训练
                    每 5 分钟轮询一次 tmux session 是否存活
                    session 消失 → 训练结束；超时 → 强制终止

④ evaluator     →  从输出目录找 eval.pth
                    若不存在，自动运行 test.sh 生成
                    调用 compare_models.py 提取 AP / AP50

⑤ state         →  比较新 AP 与历史最优
                    提升 → 保留，更新 best_model，iteration+1
                    未提升 → 记录为 discarded，下轮换策略
                    连续 max_no_improve 轮未提升 → 停止
```

### 状态持久化

所有迭代状态保存在 `config/state.json`，包含：
- `best_model`：当前最优模型的路径和指标
- `tried_strategies`：已尝试的改进策略列表（传给 Skill 避免重复）
- `history`：每轮实验的 AP、delta、是否保留的完整记录
- `current_experiment`：当前正在进行的实验（用于异常恢复）

进程意外中断后重启，会从 `state.json` 恢复，继续下一轮迭代。

### 快速开始

```bash
# 首次使用：初始化当前最优模型信息
./start.sh --init

# 启动自动迭代（受 loop_config.yml 控制）
./start.sh

# 只生成 YAML 方案，不实际训练（用于调试）
./start.sh --dry-run

# 指定 GPU 并限制轮数
./start.sh --gpu 0,1 --max-iter 5
```

配置项见 `config/loop_config.yml`，日志输出到 `loop.log`。

## 爆杀结尾

<p align="center">
  <em>如果一个研究流程可以被 AI 完全自动化，那它本身就不该被称为"研究"</em>
</p>

## Related Work
[GitHub - Imbad0202/academic-research-skills: Academic Research Skills for Claude Code: research → write → review → revise → finalize](https://github.com/Imbad0202/academic-research-skills)

[GitHub - DavidZWZ/Awesome-Deep-Research: \[Up-to-date\] Awesome Agentic Deep Research Resources](https://github.com/DavidZWZ/Awesome-Deep-Research)

[GitHub - oboard/claude-code-rev: Runnable ClaudeCode source code](https://github.com/oboard/claude-code-rev)

[GitHub - ruc-datalab/DeepAnalyze: DeepAnalyze is the first agentic LLM for autonomous data science. 🎈你的AI数据分析师，自动分析大量数据，一键生成专业分析报告！](https://github.com/ruc-datalab/DeepAnalyze)

[GitHub - HKUDS/AI-Researcher: \[NeurIPS2025\] "AI-Researcher: Autonomous Scientific Innovation" -- A production-ready version: https://novix.science/chat](https://github.com/HKUDS/AI-Researcher)

[GitHub - lastmile-ai/mcp-agent: Build effective agents using Model Context Protocol and simple workflow patterns](https://github.com/lastmile-ai/mcp-agent)


[GitHub - OpenHands/OpenHands: 🙌 OpenHands: AI-Driven Development](https://github.com/OpenHands/OpenHands)

[GitHub - Aider-AI/aider: aider is AI pair programming in your terminal](https://github.com/Aider-AI/aider)


[GitHub - WecoAI/aideml: AIDE: AI-Driven Exploration in the Space of Code. The machine Learning engineering agent that automates AI R&D.](https://github.com/WecoAI/aideml)

[GitHub - SamuelSchmidgall/AgentLaboratory: Agent Laboratory is an end-to-end autonomous research workflow meant to assist you as the human researcher toward implementing your research ideas](https://github.com/SamuelSchmidgall/AgentLaboratory)

[GitHub - SakanaAI/AI-Scientist-v2: The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](https://github.com/SakanaAI/AI-Scientist-v2)


[GitHub - Pthahnix/De-Anthropocentric-Research-Engine: De-Anthropocentric Research Engine — AI-powered academic research automation with deep literature survey, gap analysis, idea generation, experiment design & execution. Combines iterative deep research, adversarial debate, evolutionary...](https://github.com/Pthahnix/De-Anthropocentric-Research-Engine)