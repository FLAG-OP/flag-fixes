# 验证记录

这里存放后续平台和 bundle 的小型结构化验收摘要与证据链接。记录精确源码版本、硬件/软件环境、命令、日期、判定标准、实际结果和已知限制。

本次仓库初始化未执行 NPU/XPU 测试。三方报告仍是来源仓库的历史证据，不能改写为本仓库新验收。大日志/模型/制品使用发布附件，不提交机器绝对路径、凭据或备份。

现有记录：

- [thead-zw810e-problem29](thead-zw810e-problem29/README.md)（2026-11-12 首轮 / 2026-11-13 两轮修订，issue #5）：平头哥 PPU-ZW810E 单卡平台基线，判定"需优化 + gems_full 训练被平台阻断"（FlagGems `29e14822` 全量 enable 18/18 微基准慢于 native，归因=host 侧 per-call enqueue 固定成本；selective 有效但无有益算子）；gelu 族 126 用例 v6 逐位置判据 72 PASS/39 DIFF/15 FAIL（硬失败=gelu_backward 小梯度区超差，逐例违例四元组归档）；平台阻断 F-1/F-2（sdpa 训练形状挂起、matmul backward 挂起）登记待平台恢复；生产判定 comparator.py + 负例自测 20/20 + 注入自测 exit 1。

管理工具和来源检查的现有结果见 [当前结果汇总](../docs/current-status.md)，全部文档入口见 [文档导航](../docs/index.md)。
