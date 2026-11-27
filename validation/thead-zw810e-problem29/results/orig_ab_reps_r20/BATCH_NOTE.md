# r20 批次说明（review 9fe0a05 项 1/2）

## manifest 归属
`manifest.jsonl` **属于 r20 批次**（12 条记录 ts=2026-11-19T11:51–11:56Z，全部 exit0、script_sha=a66f9aee…）。r19 批次早于 manifest 机制，**无清单**，其溯源为 round-19 blob 原样内容（与逐次日志配对已验证）。本清单为 v1 格式（无 env/wall_ms/result_sha256 字段）；v2 格式见 runner——读旧记录会判重跑，不冒充复用。

## native_stock_r1 样本替换说明
r20 目录 12 份 JSON 中 11 份与提交 2bce556 字节相同；`native_stock_r1` 被一次**未登记的孤儿补跑**替换：
- 时间线：21:27:26 修改后 runner 的干启动验证（无有效清单记录 → 判重跑）启动 native_stock r1；driver 于 ~2s 后被测试方终止，**python 子进程存活**并于 21:27:47 完成落盘（JSON wall=2133.62ms + 逐次日志）
- 总日志仅有 21:27 启动记录、**无完成记录**（driver 已死，manifest 未写）——如实注明
- 该补跑与批内其他运行同配置（native/stock 显式 env）、同脚本 SHA、同 seed；退出码未捕获（进程孤儿化），JSON 通过 strict validator
- 被替换样本 = 提交 2bce556 的 `orig_ab_reps_r20/native_stock_r1.json`（wall 2135.54ms）
- 组内三份 stock JSON 中位不受单样本替换影响（wall 中位 2138.11ms 维持）；clip+optimizer 中位按当前三份为 **20.26 ms**
