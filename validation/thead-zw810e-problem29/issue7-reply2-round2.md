## 第一轮复审四处口径已修正（PR #8 commit `f95b7e4`，PR 描述已同步重写）

按审阅意见 6857432618 逐条对照：

### 合入前口径修正（4/4 完成）

1. **fused 正确性四层次区分**（README §B 与 PR 表格已改）：
   - ①单步更新 vs fp64 闭式公式 ≤1 fp32 ulp（同公式确认）；
   - ②100 步轨迹 vs stock 的漂移**如实报告且不承诺逐位一致**（fp32 多位置超 4·eps、bf16 max ~29,564 ULP）；
   - ③checkpoint 恢复为本批次内部逐位一致（限定范围）；
   - ④无 NaN/Inf。
   并注明：验收口径后续改为 **fused vs fp64/稳定参考** 的误差与训练行为，不以 stock 逐位一致为门槛。
2. **fp16 分别报告**：stock/foreach“数值损坏”（Inf/NaN，已给机制）与 fused“当前配置下无污染”分开陈述，限定到本轮已测设备/形状/lr/eps，不写为普遍定理；**fp32 优化器状态（master weights）已补为不依赖 fused 的替代方案**（对照实验待补）。
3. **GELU 负尾影响**：改为“**候选训练影响较小——待模型级敏感性/长步训练对照后定稿**”，删除“可忽略”的定稿表述；保留 72/39/15 基线，不以放宽阈值消除失败。
4. **CUDA Graph**：明确为**组件级、静态 shape 候选**；12.6× 限定为微基准结论、不得外推整步；动态 shape 边界、内存池/捕获限制、与 optimizer 组合收益、捕获失败回退策略均列为未测项。

### 下一轮顺序（已写入 README 与 PR，按审阅排序）

1. FlagScale `build_optimizer` 集成（fused 选择/日志/回退/checkpoint；覆盖 fp32/bf16/fp16、mixed、CPU、DTensor、空梯度、不支持参数类型）；集成前不扩 `_FUSED_ADAMW_VALIDATED_VENDORS`。
2. AdamW 数值验收：同梯度序列 ≥100 步的 abs/rel/ULP/非有限计数（以 fused vs fp64/稳定参考为主）+ 真实小模型 loss/任务指标对照。
3. CUDA Graph 推进到训练段：固定 shape fwd+bwd+optimizer 子图，native/selective/graph-replay 整段 wall+显存对照，记录捕获失败回退。
4. GELU 归因训练闭环：负尾输入梯度敏感性与长步训练对照。
5. F-1/F-2 持续隔离复验（记录驱动/SDK/commit/shape/超时/编译活动/控制组）。
6. 整步与多卡验证（真实 BGE-M3/x-benchmark、FSDP2）单独测试。

### 状态

- PR #8 按“第一轮优化实验与证据”保持待审/可合；issue #7 保持 open（至 FlagScale 集成、整步收益、vendor/部署验收完成）。
- 现有条目维持 `draft / deployable=false / hardware_status=unverified`；`_FUSED_ADAMW_VALIDATED_VENDORS` 不扩；不外推昆仑芯 -15.3%；局部收益不直接相加。
