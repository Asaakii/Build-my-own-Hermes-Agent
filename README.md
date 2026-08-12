# HermesLite：从零构建受控 Hermes Agent

HermesLite 是一个面向学习的本地 Agent 项目。它不尝试复刻 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 的全部能力，而是以较小、可测试的 Python 实现，说明一个工具型 Agent 最重要的边界如何协作：模型、会话、工具、工作区、确认、持久化、审计与 CLI。

> 定位：学习项目与个人本地实验工具，不是生产级安全沙箱，也不应直接处理高敏感数据或不受信任代码。

## 已实现能力

- OpenAI 兼容模型客户端，支持 DeepSeek 等服务；配置与错误输出不会显示 API Key。
- 文本 Agent 与结构化工具调用 Agent Loop；工具结果会回写到会话，再由模型继续决策。
- 受限工作区：列目录、读文件、检索、创建文本、精确替换、运行 Pytest。
- 工具注册表和 JSON 参数模式；模型伪造工具、参数或路径时由执行层拒绝。
- 工具风险分级：读取可直接执行；写入和执行测试必须由同一交互进程、同一会话中的一次性确认令牌确认。
- SQLite 会话、工具观察、任务、报告、经授权长期记忆和脱敏审计事件。
- Prompt Builder、历史摘要接口、显式技能加载、有限重试、结构化运行日志。
- `hermeslite` CLI：配置、诊断、聊天、会话、记忆和技能管理。
- 离线端到端毕业任务：验证读取、失败测试、确认补丁、复测、恢复与记忆边界。

## 架构

```text
CLI / Gateway / Telegram / Scheduler
              │
          ChatRuntime
              │
          ToolAgent
     ┌────────┼─────────┐
模型客户端   工具注册表   SQLite 状态库
              │
         受限工作区
```

渠道和定时任务只负责输入、输出或投递；模型决策、工具权限、确认、会话和存储规则始终复用同一套核心服务。

## 环境要求与安装

- Python 3.12+
- 一个 OpenAI 兼容的模型 API（可选：离线测试与帮助命令不需要模型调用）

```bash
git clone https://github.com/Asaakii/Build-my-own-Hermes-Agent.git
cd Build-my-own-Hermes-Agent
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

在 `.env` 中填写模型配置；`.env` 已被 Git 忽略，不能提交真实密钥。

```dotenv
LLM_PROVIDER=deepseek
LLM_MODEL=你的模型名
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.deepseek.com
LLM_TIMEOUT_SECONDS=30
AGENT_WORKSPACE_PATH=sandbox_workspace
AGENT_STATE_DB_PATH=data/hermes_lite.sqlite3
```

先检查配置而不请求模型：

```bash
hermeslite config
hermeslite doctor
```

只有显式传入 `--check-model` 才会发出模型请求：

```bash
hermeslite doctor --check-model
```

## 基本使用

单次聊天或交互聊天：

```bash
hermeslite chat "用一句话解释 Python 虚拟环境"
hermeslite chat --session-id learning
```

高风险工具调用会停止并要求在同一交互进程输入：

```text
/confirm <一次性令牌>
```

令牌不会写入会话、任务、审计或日志；退出聊天后，未确认操作失效。

管理本地状态：

```bash
hermeslite sessions show local-default
hermeslite memory list
hermeslite memory search Python
hermeslite skills list
```

长期记忆只能通过聊天中的明确 `/remember <内容>` 写入；模型不能自行决定保存内容。

## 可选扩展：Gateway、Telegram 与计划任务

### 本机 Gateway

Gateway 只监听 `127.0.0.1`，请求需要 Bearer Token。将以下值写入本地 `.env`：

```dotenv
HERMES_GATEWAY_TOKEN=至少16位的本地随机令牌
HERMES_GATEWAY_HOST=127.0.0.1
HERMES_GATEWAY_PORT=18791
```

启动：

```bash
hermeslite gateway run
```

接口为受令牌保护的 `GET /health` 与 `POST /v1/messages`。消息请求体：

```json
{"session_id":"local:api","text":"你好"}
```

Gateway 会使用与 CLI 相同的 `ChatRuntime`。外部渠道不能提交 `/confirm`，高风险操作会返回“请在本地交互会话确认”，不会泄露令牌。

### Telegram 私聊渠道

先在 BotFather 创建 Bot，向 Bot 发送 `/start`，再从 `getUpdates` 返回数据中确认自己的数字用户 ID。将下列真实值仅写入 `.env`：

```dotenv
TELEGRAM_BOT_TOKEN=你的Bot令牌
TELEGRAM_ALLOWED_USER_ID=你的数字用户ID
TELEGRAM_POLL_TIMEOUT_SECONDS=20
```

启动长轮询：

```bash
hermeslite telegram run
```

只有白名单用户的**私聊文本**会进入 Agent；群聊、其他用户、非文本更新都会忽略。Telegram 与 CLI 通过 `telegram:<用户ID>` 使用稳定但独立的会话。由于确认令牌只在本地交互进程有效，渠道不能确认写入或执行操作。

### 计划提醒与候选复盘

计划任务保存于本地 SQLite，执行后只创建候选记忆，绝不自动写入长期记忆：

```bash
hermeslite schedule create --session-id local-default 60 "复习工作区安全边界"
hermeslite schedule list
hermeslite schedule run
hermeslite review list
hermeslite review approve-memory <candidate-id>
```

`run` 只投递已经到期的提醒并生成待审批候选；`approve-memory` 才会走既有的长期记忆校验与保存逻辑。当前候选复盘是确定性的本地摘要，不调用模型，也不自动创建或修改技能。将候选扩展为可审批技能是下一轮产品化工作，不应让后台任务静默改写权限声明。

## 工具与工作区

Agent 只能访问项目根目录下 `AGENT_WORKSPACE_PATH` 指向的目录。当前内置工具包括：

| 工具 | 风险 | 作用 |
| --- | --- | --- |
| `list_files`、`read_file`、`search_text` | 只读 | 观察受限工作区 |
| `create_text_file`、`replace_text_once` | 写入 | 创建或精确修改 UTF-8 文件 |
| `run_pytest` | 执行 | 用固定 `python -m pytest -q` 运行工作区已有测试 |

写入与测试都需要确认。测试工具在执行前清理工作区内旧 `.pyc`，防止同秒且等长的源代码修改被旧字节码缓存误判。

## 安全边界与限制

已确认的事实：

- 工具参数、工具名称、路径与工作区边界在模型输出之后仍会验证。
- 审计记录只保存操作元数据和参数类型摘要，不保存原始参数值、工具输出或确认令牌。
- SQLite 数据库仅允许放在项目内 `data/` 目录；长期记忆需要显式授权。

不能因此推出的结论：

- 这不是操作系统级沙箱。受控测试仍会运行工作区内的 Python 测试代码，必须只用于自有、可审查的练习项目。
- 模型可能提出错误补丁、错误计划或不完整解释；确认和测试只能减少风险，不能保证正确性。
- 基础敏感信息过滤不是完整的数据防泄漏系统。

## 与官方 Hermes 的差异

HermesLite 是一个面向学习的独立实现，不是官方 [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) 的分支、替代品或兼容实现。两者的差别首先来自目标不同：本项目用较小、可测试的 Python 代码说明 Agent 的模型调用、工具循环、状态持久化与权限边界；官方 Hermes 是面向实际使用的完整 Agent 平台。

| 方面 | HermesLite | 官方 Hermes |
| --- | --- | --- |
| 定位 | 单机学习与个人实验工具 | 可长期使用、持续演进的完整 Agent 平台 |
| 模型与渠道 | OpenAI 兼容模型接口、CLI、简化本机 Gateway 与 Telegram 私聊 | 多模型供应商，以及 CLI/TUI、Telegram、Discord、Slack、WhatsApp、Signal 等渠道 |
| 工具体系 | 受限工作区中的读、写、检索、精确替换与 Pytest | 大规模工具集、工具集管理、MCP 与多种终端后端 |
| 学习与记忆 | 显式技能加载、SQLite 记忆、待审批复盘候选 | 跨会话搜索、用户模型、自动创建和改进技能等闭环能力 |
| 调度与并发 | 本地延时提醒与人工审批记忆候选 | 面向渠道的 cron 自动化、子 Agent 与并行工作流 |
| 安全与可靠性 | 路径限制、参数校验、一次性本地确认、审计 | 更完整的命令审批、配对、隔离后端、安装升级与多平台运行支持 |

因此，HermesLite 的价值不在于复刻功能数量，而在于保留一条可读、可改、可测试的核心链路：模型提出动作，Agent Loop 编排，工具执行层验证，副作用经过确认和审计，状态写入 SQLite。官方 Hermes 已覆盖更广的产品能力；本项目没有实现其多渠道生态、完整 MCP、TUI、自动技能改进、子 Agent、远程运行后端或生产级隔离。

## 与 MyClaw（此前 OpenClaw 学习项目）的差异

两个项目都具备模型、会话、长期记忆、工具、Skills、SQLite、Gateway 和 Telegram 的学习版本，但架构重心不同。

| 方面 | MyClaw / OpenClaw 学习项目 | HermesLite |
| --- | --- | --- |
| 核心目标 | 个人聊天 Agent 与消息渠道服务 | 可控的工具型、编码任务 Agent |
| Gateway 地位 | 常驻 Gateway 是唯一协调者；CLI 与 Telegram 都只请求 Gateway | Gateway 是可选入口；CLI、Gateway、Telegram 与调度器复用核心运行时，但不要求所有功能都经过同一常驻服务 |
| 内置工具 | 时间、计算、天气、受限笔记、记忆与技能 | 受限工作区的读取、检索、创建、精确修改与运行测试 |
| 主要安全问题 | 渠道白名单、Gateway Token、工具白名单、提醒投递与记忆授权 | 文件写入、代码修改与测试执行必须经过同一交互进程内的一次性确认 |
| 定时能力 | Gateway 管理可恢复的 Telegram 提醒与投递状态 | 本地计划任务生成提醒和待审批记忆候选，不自动写入长期记忆 |
| 适合继续学习的方向 | 渠道适配、常驻服务、消息路由与个人助手产品化 | 工具调用、工作区边界、确认机制、测试反馈循环与编码 Agent |

实际调用拓扑也不同：

```text
MyClaw：
CLI / Telegram 适配器
        ↓
常驻 Gateway（唯一协调者）
        ↓
Agent、工具策略、SQLite、提醒服务

HermesLite：
CLI / Gateway / Telegram / Scheduler
        ↓
共享的 Agent Runtime
        ↓
模型、工具注册表、SQLite、受限工作区
```

这不是高低之分，而是职责取舍。MyClaw 更接近“消息平台上的个人助手服务”；HermesLite 更接近“在受控工作区内完成读代码、修改、测试与恢复的 Agent 实验室”。两者共同构成了从聊天型 Agent 到工具型 Agent 的两种典型架构。

## 测试与开发

```bash
python -m pytest -q
python -m pytest -q tests/test_graduation_demo.py
python -m pip check
git diff --check
```

测试按领域模型、工具边界、SQLite、模型模拟、CLI 和端到端轨迹分层。完整说明见 [docs/testing-and-regression.md](docs/testing-and-regression.md)，架构与安全取舍见 [docs/architecture-and-safety.md](docs/architecture-and-safety.md)。

## 文档与学习记录

- [完整学习计划](Hermes-Agent-学习开发计划.md)
- [公开学习日志](learning-log.md)
- [端到端毕业任务演示](docs/graduation-demo.md)
- 本地 `实际指导过程复盘.md`：用于个人复盘，已被 Git 忽略。

## 参考与致谢

本项目的学习方向参考：

- [learn-hermes-agent](https://github.com/longyunfeigu/learn-hermes-agent)
- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)

HermesLite 的代码和设计取舍由学习目标决定；它并非官方实现的分支、替代品或兼容实现。
