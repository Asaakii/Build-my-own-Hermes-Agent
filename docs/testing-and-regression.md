# HermesLite 测试分层与回归基线

本项目的安全边界和状态逻辑优先使用确定性测试验证。真实模型只用于手动体验，不是自动化测试的前提。

## 测试分层

| 层次 | 主要关注点 | 对应测试 |
| --- | --- | --- |
| 领域单元测试 | 消息、任务、配置、重试、确认策略的数据契约 | `test_domain.py`、`test_config.py`、`test_retry_policy.py`、`test_confirmation_policy.py` |
| 工具边界测试 | 工具声明、参数模式、受限工作区、测试命令与技能权限 | `test_tool_registry.py`、`test_workspace.py`、`test_workspace_tools.py`、`test_test_runner.py`、`test_skill_loader.py` |
| SQLite 集成测试 | 会话、任务、记忆、审计、模式版本与损坏记录恢复 | `test_sqlite_state_store.py`、`test_sqlite_state_records.py`、`test_memory_store.py`、`test_audit_log.py` |
| 模拟模型运行测试 | 模型响应解析、Agent 循环、工具观察、上下文、确认与持久化聊天 | `test_model_client.py`、`test_tool_agent_loop.py`、`test_coding_agent.py`、`test_chat_runtime.py`、`test_confirmation_runtime.py` |
| CLI 与最终回归 | 命令出口、doctor、包安装，以及模型异常工具调用仍受限 | `test_cli.py`、`test_doctor.py`、`test_package.py`、`test_regression_baseline.py` |

## 每次提交前的基线

在已激活的项目虚拟环境中运行：

```bash
python -m pytest -q
python -m pip check
git diff --check
```

三个检查分别回答不同问题：功能是否回归、依赖是否冲突、提交内容是否包含常见格式问题。三者都通过才表示本地回归基线通过。

## 新虚拟环境安装验证

在需要验证安装可复现性时，创建一个临时虚拟环境，在其中安装项目与开发依赖，再执行：

```bash
python -m pip check
python -m pytest -q
hermeslite --help
```

该验证不需要模型请求。若依赖需要下载，可能需要网络；它验证的是安装路径，而不是模型连通性。

## 必须保持的拒绝场景

- 未登记工具不能执行。
- 已登记工具的未声明参数不能触达处理函数。
- 工作区路径不能逃离项目受限目录。
- 高风险工具必须在同一进程、同一会话内确认。
- 确认令牌、API Key、原始工具参数和输出不能写入运行日志或诊断输出。

`test_regression_baseline.py` 将“模型为已登记工具伪造额外参数”的场景串联 Agent、注册表和工具处理函数。期望结果不是模型聪明地纠正错误，而是系统即使面对错误模型决策也拒绝实际执行。
