# ZW810E problem29 首轮测试与评估结果（issue #5）

日期：2026-11-12 · 平台：平头哥 PPU-ZW810E（单卡，PPU 6）· 执行：flag-fixes 平台验证
目标树：FlagGems `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`v5.3.5-46-g29e14822`）
完整环境：`results/env_record.json` · 原始数据：`results/` · 复现命令见文末

## 结论（首轮）

**平台判定：需优化（FlagGems 侧在 ZW810E 当前为净负收益）；训练接入策略 = 不启用或仅选择性启用有益算子（本轮未发现有益算子）。**

1. **平台文档"FlagGems 13 算子 12 慢"口径与本轮实测方向一致**：本机可复现基线下，`flag_gems.enable()` 全量注册后在 18 项微基准上 **17 项慢于厂商 native aten（1.3–6.4×）**，无一项更快。
2. **归因**：FlagGems 在 ZW810E 存在 **~60–80 µs 的每算子固定开销**（1.2M 元素 gelu：native 13.6 µs vs gems 75.9 µs；尺寸增大到 98M 后差距收敛到 ≈1.0–1.2×），叠加稳态吞吐略慢（98M bf16 gelu 0.577 vs 0.489 ms）。训练步内算子多、单算子尺寸中小，固定开销是主要税负。mm（GEMM）回归最重（3.21×）。
3. **选择性注册机制在 ZW810E 有效**：`only_enable(include=[gelu 族])` 下未包含算子（add/cast/findmax/mul_scalar/mm/sdpa）全部保持 native 性能与 native kernel（profiler 证明）；但被包含的 gelu 族本身仍慢于 native（1.6–3.7×），故**本轮没有发现值得注册的算子**。
4. **精度**：gelu 族 126 用例矩阵（none/tanh 前向+反向、gelu_and_mul × fp32/bf16/fp16 × 连续/转置/切片 × 空/小/典型/大/边界）**0 硬失败**；84 PASS / 42 DIFF——DIFF 全部为次正规（FTZ）输出区 FlagGems 与 native 的 ±1–2 ulp 位级差异及 ±0 符号差异（NaN/Inf 掩码、非次正规区相对 fp64 oracle 精度、梯度均达标）。边界输入按 issue 要求覆盖 fp16 `0x03FF`、bf16 `0x007F`、fp32 `0x007FFFFF`。
5. **AdamW**：`torch.optim.AdamW(fused=True)` 在 ZW810E **可用**（无回退异常）；5 步轨迹与 stock 一致（fp32 末参差 ≤2.4e-6 = fused 求和顺序差；bf16 差 1 ulp；loss 无差异）。按 PR3 护栏，FlagScale 侧非昆仑芯平台默认仍应 stock——本数据支持后续把 ZW810E 加入 `_FUSED_ADAMW_VALIDATED_VENDORS` 的评估，但按仓库流程需另立条目与 owner 审核。
6. **小模型（固定种子 gated MLP + 2 层 attention 块，fp32）**：native / gems_full / gems_selective 三组 **loss 轨迹、15/15 参数梯度与 8 步后参数全部逐位一致**（sha256 相同）；稳态步时 native 13.4–13.8 ms、full 13.8–14.7 ms、selective 13.4–13.6 ms（3 独立进程复跑）——**该负载下启用 FlagGems 无收益也无正确性风险**。

## 微基准（p50 ms，3 独立进程取中位；p95 为三进程 p95 的中位）

| 算子 | native p50 | native p95 | gems_full p50 | gems_full p95 | full/native | selective p50 | selective p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cast_to/bfloat16 | 0.010 | 0.011 | 0.010 | 0.011 | 1.01x | 0.010 | 0.012 |
| cast_to/float32 | 0.026 | 0.027 | 0.083 | 0.089 | 3.14x | 0.026 | 0.028 |
| findmax/fp32 | 0.021 | 0.023 | 0.114 | 0.271 | 5.30x | 0.021 | 0.022 |
| gelu_and_mul/bf16（gelu+mul 两 kernel） | 0.024 | 0.025 | 0.155 | 0.191 | 6.37x | 0.082 | 0.086 |
| gelu_and_mul_fused/bf16（flag_gems API） | 0.024 | 0.025 | 0.087 | 0.099 | 3.58x | 0.087 | 0.100 |
| gelu_bwd_none/bfloat16 | 0.070 | 0.078 | 0.255 | 0.263 | 3.63x | 0.258 | 0.298 |
| gelu_bwd_none/float32 | 0.125 | 0.126 | 0.257 | 0.312 | 2.07x | 0.258 | 0.303 |
| gelu_fwd_none/bfloat16 | 0.028 | 0.028 | 0.081 | 0.086 | 2.94x | 0.082 | 0.102 |
| gelu_fwd_none/float32 | 0.051 | 0.057 | 0.084 | 0.127 | 1.65x | 0.083 | 0.122 |
| gelu_fwd_tanh/bfloat16 | 0.026 | 0.027 | 0.083 | 0.103 | 3.14x | 0.083 | 0.095 |
| gelu_fwd_tanh/float32 | 0.051 | 0.052 | 0.083 | 0.094 | 1.61x | 0.083 | 0.098 |
| mm_transposed/fp32 | 0.684 | 0.684 | 2.194 | 2.205 | 3.21x | 0.684 | 0.684 |
| mul_scalar/bfloat16 | 0.014 | 0.015 | 0.081 | 0.090 | 5.61x | 0.014 | 0.015 |
| mul_scalar/float32 | 0.048 | 0.049 | 0.090 | 0.098 | 1.87x | 0.048 | 0.049 |
| pointwise_add/bfloat16 | 0.014 | 0.015 | 0.086 | 0.097 | 6.12x | 0.014 | 0.015 |
| pointwise_add/float32 | 0.049 | 0.050 | 0.094 | 0.116 | 1.94x | 0.049 | 0.050 |
| sdpa_fwd/bf16 | 0.131 | 0.132 | 0.174 | 0.200 | 1.32x | 0.131 | 0.132 |
| transpose_copy/fp32 | 0.040 | 0.041 | 0.102 | 0.118 | 2.56x | 0.042 | 0.043 |

形状：行向量 [12288000]（12.25M，PR3 归因口径）；gelu_and_mul (16384,2560)；findmax (4096,1280)；mm [4096,1280]@[1280,2048]ᵀ；sdpa (8,16,512,64)。warmup 5，30 iters，CUDA event 同步计时；每配置 3 独立进程。

## 尺寸扫描（gelu fp32 前向，ms）

| 尺寸 | native | gems_full | 倍率 |
|---|---:|---:|---:|
| 1.2M | 0.0136 | 0.0759 | 5.58x |
| 12.25M | 0.0453 | 0.0900 | 1.99x |
| 98M | 0.3847 | 0.3858 | 1.00x |

→ 固定每算子开销 ≈ 60–80 µs；98M 后持平（bf16 仍有 0.577 vs 0.489 = 1.18x 稳态差）。

## 精度矩阵判定（126 用例）

- **PASS 84**：与 native 逐位相等，或相对 fp64 oracle 满足 `rel ≤ max(4·eps(dtype), 1.5·native_rel)`（gems 多数更准：内部 fp32 计算 vs aten 分步舍入）。
- **DIFF 42（记录为已知差异，非失败）**：
  - fp16 gelu/ gelu_and_mul：次正规输出区 FlagGems 与 native ±1–2 ulp 位级差（ulp_out=1–2，NaN/Inf 掩码一致）；
  - fp32 large：**梯度**次正规区逐位不一致（grad_ftz 位置数已记录）；
  - bf16/fp16 boundary gelu_and_mul：±0 符号差异（9 例，IEEE 0==−0）。
- **FAIL 0**：无 NaN/Inf 掩码错、无梯度缺失、无超出门槛的数值偏差。
- 边界输入：fp16 `0x03FF/0x0400/0x0001`、bf16 `0x007F/0x0080/0x0001`、fp32 `0x007FFFFF/0x00800000/0x00000001`、±Inf/NaN/±0 均覆盖（`accuracy_matrix.json` 每用例含 oracle 相对误差与 FTZ 位级计数）。

## 覆盖与未覆盖（按 issue 最小执行方案逐条）

- [x] 设备/驱动/固件/torch/厂商扩展/编译器/树 hash/环境变量：`results/env_record.json`
- [x] device/dispatch/注册入口确认 + profiler 命中证明（含 selective 排除证明）：`results/probe1_dispatch.json`、`results/probe3_scaling_hits.json`
- [x] native / 当前 FlagGems / 平台候选（selective enable、AdamW fused）三组同输入对照：gelu none/tanh 前后向、gelu_and_mul、AdamW stock/fused、full/selective enable 均覆盖
- [x] 正确性 fp32/bf16/fp16、连续/转置/切片、空/小/典型、NaN/Inf/±0、次正规边界（0x03FF/0x007F/0x007FFFFF）
- [x] 固定种子小模型 forward/loss/梯度/5–8 步更新（无外部权重）
- [x] 性能：固定形状/dtype、warmup+同步、编译冷启动与稳态分离（step1 单列）、每配置 ≥3 独立进程、p50/p95、显存、profiler 命中
- [ ] **pending：多卡/通信整步**（本机单卡可见性分配 PPUS=6,7，无真实多卡训练环境，按 issue 记 pending，不算验收）
- [ ] **pending：problem29 真实 BGE-M3 runner / x-benchmark 整模型**（本机无该容器与 runner，用 12.25M 口径微基准 + 小模型替代，已注明差异）
- [ ] **pending：`_FUSED_ADAMW_VALIDATED_VENDORS` 加入 ZW810E**（需独立 fix 条目 + owner 审核，本轮只提供数据）

## 复现命令

```bash
export PYTHONPATH=/path/to/FlagGems/src   # 29e14822
export CUDA_VISIBLE_DEVICES=0
python3 scripts/accuracy_matrix.py --out results/accuracy_matrix.json
python3 scripts/adamw_check.py --out results/adamw_check.json
for cfg in native gems_full gems_selective; do
  for rep in 1 2 3; do
    python3 scripts/perf_matrix.py --config $cfg --out results/perf_${cfg}_r${rep}.json
  done
done
python3 scripts/probe3_scaling_hits.py
for cfg in native gems_full gems_selective; do
  python3 scripts/small_model.py --config $cfg --out results/small_model_${cfg}.json
done
python3 scripts/probe4_small_model.py   # 3 进程复跑 + 逐位一致性
```

## 对三平台状态矩阵的建议更新

| 平台 | 本轮后状态 |
|---|---|
| 平头哥 ZW810E | **需优化**：FlagGems（29e14822，triton 3.6.0/PPU SDK 2.1.0）全量 enable 在实测全部热点为净负收益（1.3–6.4×），根因=每算子固定开销+稳态略慢；接入策略=不启用或 selective（本轮无有益算子）；gelu/cast/findmax/pointwise/transpose/attention 热点应走厂商 kernel 或厂商融合（对应 PR3 移交项）；精度可用（0 硬失败，FTZ 区位级差异已记录） |
