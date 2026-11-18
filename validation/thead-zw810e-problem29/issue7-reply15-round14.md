## 第十三轮复审（PR #8 评论 `nYw_Pw`）四点已全部修正（PR #8 commit `a79f605`）

### 1. P1 A/B losses_run/grad_norms_run 未初始化 → 已修复

- 根因确认：`losses_run = []` / `grad_norms_run = []` 在每步开始时清空（review 正确指出该位置在 optimizer step 循环内），导致只保留末步数据。
- **修复**：初始化移到 optimizer step 循环外（整个训练循环前），跨步累积不再被清空。seed 42 已恢复并去重（原代码有双 seed 块）。
- **seed 化后 12 次重跑**（3 进程 × 4 配置，全部 exit 0）：seed 修正后**同配置跨进程 loss 序列可复现**（seed 固定数据流 ✓）；**四配置间 loss 差异为真实数值路径差异**（FlagGems gelu vs native gelu），native/stock 与 native/fused 亦存在差异——不将配置间差异归因于单一因素。
- `results/orig_ab/native_stock.json` 已入册为同矩阵 native 产物。

### 2. P1 probe11 缺 mask 参数 → 已修复

- 原代码 `timed_in_context(q, q, q, be)` 未传入 mask——masked 标签全部无效。
- **修复**：`timed_in_context` 增加 `attn_mask` 参数；masked/unmasked 分支分别传入对应 mask；每个后端调用均记录实际使用的 mask。
- **重采结果**（`probe11_masked_v2.json`）：masked flash → "No available kernel"（不可用）；masked efficient/math 可用但慢于无 mask flash。

### 3. P2 probe10/11 文档同步

- probe10：pre-conversion 修正（fp32 转换移到计时外）；README 改为"该 shape 下已测开关/布局未发现收益"。
- probe11：后端上下文计时已确认（masked flash 不可用 ✓）。

### 状态

PR #8 HEAD 更新至 `a79f605`（含 seed 42 + 正确性字段 + probe11 修正 + 全部证据）。issue #7 保持 open。
