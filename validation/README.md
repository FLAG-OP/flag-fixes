# 验证记录

这里存放后续平台和 bundle 的小型结构化验收摘要与证据链接。记录精确源码版本、硬件/软件环境、命令、日期、判定标准、实际结果和已知限制。

本次仓库初始化未执行 NPU/XPU 测试。三方报告仍是来源仓库的历史证据，不能改写为本仓库新验收。大日志/模型/制品使用发布附件，不提交机器绝对路径、凭据或备份。

现有记录：

- [thead-zw810e-problem29](thead-zw810e-problem29/README.md)（2026-11-12 首轮 / 2026-11-13 两轮修订，issue #5）：平头哥 PPU-ZW810E 单卡平台基线，判定"需优化 + gems_full 训练被平台阻断"（FlagGems `29e14822` 全量 enable 18/18 微基准慢于 native，归因=host 侧 per-call enqueue 固定成本；selective 有效但无有益算子）；gelu 族 126 用例 v6 逐位置判据 72 PASS/39 DIFF/15 FAIL（硬失败=gelu_backward 小梯度区超差，逐例违例四元组归档）；平台阻断 F-1/F-2（sdpa 训练形状挂起、matmul backward 挂起）登记待平台恢复；生产判定 comparator.py + 负例自测 20/20 + 注入自测 exit 1。
- [kunlunxin-p800-real-bge-m3](kunlunxin-p800-real-bge-m3/README.md)（2026-11-17，issue #9）：昆仑芯 P800 **真实 BGE-M3 入口**（真权重 `BAAI/bge-m3` + 官方 THUIR/T2Ranking 重建 triplet-15 + 原始 FlagScale `examples/retrieval/`，batch 10×accum 4）基线与消融：native 3,572.4 ms/步（kernel 3,240.8），GELU-only **-14.1%**（GELU 类目 1,169.7→616.0 ms，-47.3%，厂商 `*_highprecision_nv` 全部切到 FlagGems kernel）、GEMM env **-2.3%**（GEMM 类目 -139 ms，输出逐位一致）、glue（fc_fusion=0）**+0.6% 净负收益**复现、all-fixes **-16.8%**（kernel 2,535.3，吞吐 56.0→67.3 texts/s）；数值：native 跨进程逐位可复现，配置间 loss 差 0.63–1.24%（bf16 舍入量级）、无非有限值；原始 problem29 的 kernel 次数口径全部重现（cast_te 4,800、transpose_021_sdnn 1,936、cluster_cast_kl3 12,460/12,399——"单卡复现不出"的归因缺口关闭）。新发现并移交：FlagGems gelu 变长 shape 开销 4.8×（616 vs ~129 ms/步）与冷缓存逐 shape 编译停顿；**FSDP2 2 卡平台阻断 F-P800-1**（XCCL all-gather 停在任意 SeqNum，7 触发 0 成功，单卡 DDP 全轮可用）。全部 fix 维持 `draft / deployable=false / hardware_status=unverified`；`-16.8%` 不写"原始问题已解决"。

管理工具和来源检查的现有结果见 [当前结果汇总](../docs/current-status.md)，全部文档入口见 [文档导航](../docs/index.md)。
