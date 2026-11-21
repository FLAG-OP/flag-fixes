## 第十轮（复审 6895987107 执行顺序 1、3 前半）完成（PR #8 commit `3ffc710`）

### 1. 原对象 A/B 矩阵 3 独立进程复制（12 次运行全部 exit 0）

| 配置 | wall p50 | wall 三次 | forward/步 | backward/步 | clip+optimizer/步 |
|---|---:|---|---:|---:|---:|
| native/stock | 2,128.8 ms | 2,123.7 / 2,131.9 / 2,128.8 | 691.7 | 1,381.2 | 20.2 |
| native/fused | 2,128.9 ms（+0.0%）| 2,128.9 / 2,131.9 / 2,132.7 | 698.0 | 1,383.4 | 14.0 |
| selective/stock | 2,172.4 ms（+2.05%）| 2,164.5 / 2,171.5 / 2,175.3 | 714.0 | 1,401.6 | 20.5 |
| selective/fused | 2,168.9 ms（**-0.16% vs selective 基线**）| 2,166.4 / 2,168.9 / 2,171.5 | 712.1 | 1,398.7 | 14.0 |

- **fused AdamW**：optimizer 段省 6.2 ms/步（-30% 段内），整步中性（±0.16% 以内）——与占比预算公式 p≈0.0095、s≈1.45 → 预期 -0.3% 一致。
- **selective(gelu) 确认回归 +1.9%**（43.6 ms 中位）。
- 逐次原始 wall 已归档（`results/orig_ab_reps/`），汇总含 min/med/max。

### 2. AdamW 稳定参考验收（审阅执行顺序 3 前半；`adamw_stable_ref.py` / `adamw_stable_ref_acceptance.json`）

30 步受控更新（4096 参数、seed 21、同梯度序列），参考 = **fp64 闭式 AdamW（公式 A）**，判定 = 每位置 ordinal ULP（参考核对到 dtype）+ 尺度门控相对误差（|ref| > 0.25×参数标准差）+ 非有限计数：

| dtype | 实现 | 非有限 | 门控 max rel | median rel | 判定 |
|---|---|---:|---:|---:|---|
| fp32 | foreach | 0 | 16.57（近零位置）| 1.248 | 与 fused 一致 |
| fp32 | **fused** | 0 | 16.57 | 1.248 | **通过（同 foreach 口径）** |
| bf16 | foreach | 0 | 16.78 | 1.249 | 与 fused 一致 |
| bf16 | **fused** | 0 | 16.78 | 1.246 | **通过** |
| fp16 | foreach | **498（损坏）** | inf | 1.303 | **损坏**（同 stock）|
| fp16 | **fused** | **0** | 16.55 | 1.248 | **通过——fp16 唯一有界路径** |
| fp16 | stock | **522（损坏）** | — | — | 损坏（两次独立记录 522/498）|

- stock 已污染配置未作为 fp16 正确性参考（审阅口径执行）；fused vs fp64/稳定参考验收**在此负载/30 步内通过**；多张量/336 张量口径沿用 round-4 checkpoint 证据。
- 局限：单张量形状 4096；多卡/FSDP2 未覆盖。

### 3. 其余（维持第六轮口径）

copy/cast、F-1/F-2 复验、FlagScale 集成、整步/多卡按 issue #7 跟踪；现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变。
