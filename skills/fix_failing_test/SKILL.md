---
{"name":"fix_failing_test","description":"在受限工作区内定位并修复一个已有失败测试。","allowed_tools":["list_files","read_file","search_text","replace_text_once","run_pytest"]}
---
# 修复失败测试

1. 阅读任务说明并列出相关文件。
2. 先运行已有测试，确认失败现象。
3. 只做最小的精确修改，不扩大工作区范围。
4. 再次运行同一测试目标，并报告验证结果与剩余风险。
