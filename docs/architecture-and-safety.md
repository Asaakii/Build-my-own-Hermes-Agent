# HermesLite 架构与安全边界

## 一次编码任务的数据流

用户任务
→ 会话保存 Message
→ Agent Loop 组装上下文并请求模型
→ 模型返回 ToolCall 或最终回答
→ 工具层验证 ToolCall
→ 工具执行后返回 ToolResult
→ ToolResult 写回会话，供下一轮模型决策
→ TaskState 记录任务进度和最终状态

## 数据归属

| 数据 | 所属模块 | 当前是否持久化 |
| --- | --- | --- |
| Message | 会话 | 否，阶段 3 再写入 SQLite |
| ToolCall | Agent Loop / 工具层 | 否 |
| ToolResult | 工具层 / 会话 | 否 |
| TaskState | Agent Loop | 否 |
| 长期记忆 | 记忆层 | 否，阶段 3 实现 |
| 技能 | 技能层 | 否，阶段 4 实现 |

## 第一版安全策略

| 操作 | 策略 |
| --- | --- |
| 列目录、读文件、搜索文本 | 只允许在 sandbox_workspace 内 |
| 创建或修改文件 | 仅允许在 sandbox_workspace 内，阶段 2 实现 |
| 运行测试 | 只允许白名单测试命令，阶段 2 实现 |
| 删除文件、任意 shell 命令、网络命令 | 默认拒绝 |
| 写入长期记忆、修改配置或安全策略 | 必须明确确认 |
| 工作区外路径、绝对路径、路径穿越 | 永远拒绝 |

## 初始限制

- 默认工作区：sandbox_workspace/
- 单次任务最大工具轮数：8
- 单次文件写入上限：100000 字节
- 模型请求超时：30 秒