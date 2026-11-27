## 第十二/十三/十四/十五轮修正完成（PR #8 HEAD `b1b261d`）

### 已完成（按复审顺序，全部落地）

1. **判定完整性**（round-12 P1）：`verdict()` 生产判定函数含缺字段检查 + 快照完整性校验 + 快照数量校验 + 步数集合重复检测；`--selftest` 经真实 `aggregate()` 注入（缺记录/空列表/仅参数超差/仅状态非有限/step 不符/全过控制）全部通过；JSON verdict 与退出码同源。
2. **容差 floor 单一来源**（round-12 P1）：`floor = 0.01 * max|ref|` 为唯一定义；JSON `tolerances` 字段保存实际使用的公式与系数；文档引用同一来源。
3. **P1 probe11 mask 参数**（round-13）：`timed_in_context` 接受 `attn_mask` 参数并在所有调用中传入对应 mask 或 `None`；后端上下文覆盖 warmup + 计时；masked/unmasked 分支的 backend 一致。
4. **P1 AB losses_run/grad_norms_run 初始化移到循环外**（round-13）：seed 42 恢复（round-13 编辑误删，已恢复）；losses 跨步累积；JSON 含 12 个 loss 值（非末步 4 个）；同配置跨进程 loss 序列可复现。
5. **P2 probe11 后端上下文计时**（round-13）：所有后端计时均在 `sdpa_kernel(be)` 内执行（含 warmup）；结果确认 bool mask 下 flash 不可用 → efficient/math 仅有的 masked 路径。
6. **P2 probe10 预转换**（round-13）：fp32 转换移到计时外预创建。
7. **P2 容差元数据统一**（round-12/13）：JSON `tolerances` 字段保存实际使用的公式与系数；README 引用同一来源。

### 已确认的进展与性能边界

- fused AdamW：原对象整步中性（-0.80%）
- selective(gelu)：+1.23% 回归
- GEMM 单 shape ~110 TFLOP/s（bf16），tile/layout/累加精度消融无收益
- masked attention：bool mask 下 flash 不可用 → efficient/math 仅有的 masked 路径

### 当前 open 项（issue #7 跟踪）

- AdamW 稳定参考验收（fp64/稳定参考 vs fused）
- FlagScale 集成
- CUDA Graph 训练段
- GELU 剩余差异（tanh bwd 3,190 / fwd 26）
- F-1/F-2 复验
- 整步/多卡
- masked attention 等长分桶

### 不变项

现有条目 `draft / deployable=false / hardware_status=unverified`；`_FUSED_ADAMW_VALIDATED_VENDORS` 不扩；不外推昆仑芯 -15.3%；局部收益不直接相加；归因完成 ≠ 修复完成 ≠ 验收通过。Ref #7（保持 open）。
