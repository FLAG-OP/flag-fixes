# ZW810E problem29 平台基线（issue #5，第二轮修订）

日期：2026-11-12 首轮 · 2026-11-13 按审阅意见修订 · 平台：平头哥 PPU-ZW810E（单卡，PPU 6）
目标树：FlagGems `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`v5.3.5-46-g29e14822`）
环境与版本：`results/env_record.json` · 原始数据：`results/`（含逐迭代样本）· 复现命令见文末

**修订说明（回应 2026-11-13 审阅）**：本轮按"最小收尾顺序"补交——证据随 PR 落地远端；精度判据收敛（DIFF 三分类接受策略 + 比较器负例自测 + oracle/阈值出处）；性能归因按定向探索重做（截距/斜率拟合 + host enqueue 分解 + 直调/注册入口对照），固定开销由"假设"升级为"拟合证据"，且归因结论从"每算子固定开销"**修正为"host 侧 per-call enqueue 开销"**；全部结论限定适用范围。

## 结论（第二轮修订口径）

**平台判定：需优化（首轮不变）。适用范围：本 FlagGems 版本（29e14822）/ 本尺寸集（0.3M–98M，重点 12.25M PR3 口径）/ 单卡 / 本轮微基准与小模型负载；不外推至 ZW810E 全部场景，不代表优化不可行。**

1. **"FlagGems 13 算子 12 慢"口径与实测方向一致**：18 项微基准下 gems_full（全量 enable）17 项慢于 native（1.3–6.4×），1 项持平（cast bf16 1.01×）。**这是接入配置对照（native vs 全量 enable vs selective enable），不是三种优化实现的对照——本轮未实现任何平台优化候选**，"需优化"即"FlagGems 侧收益为负，需另行优化后才值得启用"。
2. **归因（定向探索后修正）**：尺寸扫描线性拟合显示 **FlagGems 截距 60.4/70.0 µs vs native 4.7/4.6 µs（gelu/add fp32），斜率基本相同**（0.0033–0.0033 vs 0.0038–0.0039 ns/元素）——慢在**每调用固定成本**，不在稳态带宽。host enqueue 分解进一步定位：**native enqueue ≈ 5 µs/次，FlagGems ≈ 57–69 µs/次**（flag_gems python 分发 + libentry + triton launch 链路的 host 开销；空闲 GPU 上该延迟落入 event 窗口）。**直调入口 vs aten 注册入口调用同一 Triton kernel**（profiler 核验 `gelu_none_kernel_rank_1`，enqueue 差仅 ~5 µs）→ 注册/调度层只贡献 ~5 µs，主体在 flag_gems 调用链自身。**优化优先级应先考察 per-call host 开销（调度/launch 路径），而非直接推断"必须厂商融合"或"无需新 fix"**——原首轮该两句推断撤回。
3. **selective 机制有效但无收益**：`only_enable` 未包含算子保持 native 性能与 native kernel（profiler 证明）；被包含 gelu 族仍带同一 ~60 µs 固定成本 → 本轮无值得注册的算子。
4. **精度（126 用例，v5 判据）**：0 硬失败（84 PASS + 42 DIFF + 0 FAIL）。42 DIFF = 33 例 FTZ 位级不一致 + 9 例仅 ±0 符号差；NaN/Inf 掩码与 **±Inf 符号**全部一致；非次正规区相对 fp64 oracle 达标。**该结果限定于本矩阵覆盖的 op×dtype×layout×值类，不等同于平台全部算子无正确性问题**。
5. **AdamW**：正确性 5 步轨迹一致（fp32 ≤2.4e-6、bf16 1 ulp，`adamw_check.json`）；补充性能证据：336 张量配置 stock 26.8 ms/步（2,688 launch）vs fused 11.7 ms/步（39 launch）= 2.3×/launch -98.6%，单参数两者持平（0.128 vs 0.132 ms）；显存峰值同量级（`probe6_adamw_perf.json`）。**不据此放开 vendor 白名单**；stock/fused 耗时数据仅供后续平台 fix 条目评估。
6. **小模型**（固定种子 gated MLP + 2 层 attention，fp32）：三组接入配置 loss/梯度/参数 8 步后逐位一致（sha256 15/15）。**该结论仅限此模型与测试范围**，不概括为平台"无正确性风险"。`gelu_and_mul` 无 autograd 路径，梯度 N/A 维持。

## 精度判据（事先声明，非按本批结果反推）

判定门在脚本 docstring 中先于执行声明（`scripts/accuracy_matrix.py` v5，与 v3 判定门一致）：

- **PASS**：NaN/Inf 掩码与 ±Inf 符号等于 native；grad 存在性一致；`rel_err(gems, oracle) ≤ max(4·eps(dtype), 1.5·rel_err(native, oracle))`（非次正规区；gems 允许更准，但不得劣于 native 的 1.5 倍）；FTZ 区逐位一致。
- **DIFF（记录性接受，不算失败也不静默通过）**：仅次正规（FTZ）输出区与 native 位级不一致（±1–2 ulp），或仅 ±0 符号差（IEEE 0==−0；符号差单独计数，不因处于次正规区而自动忽略——本轮 9 例 ±0 符号差均在 gelu_and_mul 边界用例的乘法链路上，逐例位置与计数已归档）。
- **FAIL**：NaN/Inf/±Inf 符号掩码不一致、grad 缺失、超容忍、异常。
- **oracle**：aten fp64 精确公式（none：`x·Φ(x)`，Φ 用 `erf`；tanh：`x·½(1+tanh(√(2/π)(x+0.044715x³)))`；grad 为对应解析式），与被测同输入同设备计算；输入为脚本固定种子确定性构造（典型/大：cuda Generator crc32 种子；boundary：按 issue 位型拼装；来源与适用范围见脚本 `make_value_class`）。
- **阈值出处**：`4·eps` 与 `1.5×native_rel` 在 v3 起即固定于脚本 docstring，本轮未改门、只增证据字段。
- **边界位型确认**：fp16 `0x03FF`、bf16 `0x007F`、fp32 `0x007FFFFF` 解码后均验证为各自 dtype 的最大次正规数（`test_comparator_negative.py` 前 3 项断言）。
- **FTZ 定义**：oracle 输出 `|y| < dtype 最小正规数`（fp16 6.103515625e-05，bf16/fp32 1.1754943508222875e-38）的位置；该区要求 native 与 gems 逐位一致，不一致记 DIFF 并归档 `ftz_diff_count`/最差位置/双方位型。
- **比较器负例自测（设备无关，10/10 通过，exit 0）**：`scripts/test_comparator_negative.py` 证明分类会拒绝——NaN 出现、±Inf 符号翻转（v4 缺陷，v5 修复后回归）、Inf 出现、grad 缺失、越界误差、gems 劣于 native；并验证 FTZ-only 与 ±0-only 正确落入 DIFF 而非 PASS。
- **批次关系**：v1（直调 API 无 autograd）、v2（ULP 步进按 dtype 升位虚高 + NaN 入 diff）两批判定作废未归档；归档的 `accuracy_matrix.json` 为 **v5 全量重跑**（判定门与 v3 相同 + 增加最差位置/双方位型/±0 计数/FTZ 定义/特殊值计数等证据字段），非对旧批次的重评分；输入由固定种子决定，两批逐位同源。

## 微基准（p50 ms；每配置 3 独立进程，表中为"三进程 p50 的中位"；p95 为三进程 p95 的中位；分位数均为**进程内 30 次迭代统计**，不跨进程合并；逐迭代原始样本在 `results/perf_*_r*.json` 的 `samples_ms`）

| 算子 | native p50 | native p95 | gems_full p50 | gems_full p95 | full/native | selective p50 | selective p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cast_to/bfloat16 | 0.010 | 0.011 | 0.010 | 0.011 | 0.99x | 0.010 | 0.012 |
| cast_to/float32 | 0.026 | 0.028 | 0.083 | 0.092 | 3.18x | 0.026 | 0.027 |
| findmax/fp32 | 0.022 | 0.022 | 0.113 | 0.144 | 5.21x | 0.021 | 0.022 |
| gelu_and_mul/bf16 | 0.024 | 0.025 | 0.154 | 0.193 | 6.30x | 0.081 | 0.087 |
| gelu_and_mul_fused/bf16 | 0.024 | 0.025 | 0.087 | 0.095 | 3.60x | 0.086 | 0.101 |
| gelu_bwd_none/bfloat16 | 0.072 | 0.078 | 0.261 | 0.269 | 3.63x | 0.269 | 0.291 |
| gelu_bwd_none/float32 | 0.124 | 0.126 | 0.263 | 0.312 | 2.11x | 0.265 | 0.302 |
| gelu_fwd_none/bfloat16 | 0.028 | 0.028 | 0.081 | 0.085 | 2.95x | 0.085 | 0.145 |
| gelu_fwd_none/float32 | 0.051 | 0.057 | 0.084 | 0.127 | 1.64x | 0.084 | 0.127 |
| gelu_fwd_tanh/bfloat16 | 0.026 | 0.027 | 0.083 | 0.095 | 3.13x | 0.083 | 0.091 |
| gelu_fwd_tanh/float32 | 0.051 | 0.052 | 0.082 | 0.095 | 1.59x | 0.083 | 0.094 |
| mm_transposed/fp32 | 0.684 | 0.684 | 2.182 | 2.189 | 3.19x | 0.684 | 0.684 |
| mul_scalar/bfloat16 | 0.015 | 0.015 | 0.081 | 0.092 | 5.57x | 0.015 | 0.016 |
| mul_scalar/float32 | 0.048 | 0.049 | 0.090 | 0.101 | 1.88x | 0.048 | 0.050 |
| pointwise_add/bfloat16 | 0.014 | 0.015 | 0.085 | 0.096 | 6.09x | 0.014 | 0.017 |
| pointwise_add/float32 | 0.049 | 0.050 | 0.096 | 0.116 | 1.96x | 0.049 | 0.050 |
| sdpa_fwd/bf16 | 0.132 | 0.132 | 0.169 | 0.188 | 1.29x | 0.131 | 0.132 |
| transpose_copy/fp32 | 0.040 | 0.041 | 0.103 | 0.119 | 2.57x | 0.043 | 0.043 |

形状：行向量 [12288000]（12.25M，PR3 归因口径）；gelu_and_mul (16384,2560)；findmax (4096,1280)；mm [4096,1280]@[1280,2048]ᵀ；sdpa (8,16,512,64)。warmup 5 次（排除 JIT/首编译）后 30 次 CUDA event 同步计时。`gelu_and_mul/bf16` 为建模代码常见写法（gelu+mul 两 kernel），`gelu_and_mul_fused/bf16` 为 flag_gems API。

## 归因证据（probe5，方法注记见 JSON `method_notes`）

**尺寸扫描 + 线性拟合**（进程严格隔离：native 进程不 import flag_gems；`results/probe5_overhead_attribution.json`）：

| op | native 截距 | gems 截距 | native 斜率 | gems 斜率 |
|---|---:|---:|---:|---:|
| gelu fp32 | 4.73 µs | 60.43 µs | 0.0039 ns/元素 | 0.0033 ns/元素 |
| add fp32 | 4.60 µs | 70.02 µs | 0.0038 ns/元素 | 0.0030 ns/元素 |

范围 0.3M–98M 共 6 点。**首版 probe5 曾在同一进程内 enable 后标"native"，标签失真已作废**；归档为 v2 进程隔离版。

**host enqueue 分解**（空闲 GPU 上 event 窗口含 enqueue 延迟；enqueue = 无同步批量提交 50 次的 host 墙钟均值，5 批取最小）：

| 配置 | gelu event (1.2M) | gelu enqueue | add enqueue | native 对照 enqueue |
|---|---:|---:|---:|---:|
| native 进程 | 0.0135 ms | — | — | **4.9–5.5 µs** |
| gems 进程 | 0.0739 ms | 57.0 µs | 68.5 µs | — |

12.25M 时 gems gelu enqueue 56.7 µs（event 0.079 ms）vs native enqueue 5.0 µs（event 0.0448 ms）→ **固定成本主体 = host 侧 flag_gems 调用链（python 分发 + libentry + triton launch），而非 GPU kernel 本体或注册层**。

**直调 vs aten 注册入口**（同一 gems 进程）：profiler 核验两者命中同一 `gelu_none_kernel_rank_1`；enqueue 55.6–60.7 µs（直调低 ~5 µs）→ 注册层贡献 ~5 µs。

**profiler 计时注记**：profiler 会放大 device_time（同一 1.2M gelu 报 0.58 ms vs event 0.07 ms），故 profiler 仅用于命中/kernel 名证据，不用于计时声明。

## AdamW（新增性能证据）

| 配置 | stock | fused | Δ |
|---|---:|---:|---|
| 单参数 [1024,1024] | 0.128 ms/步，8 launch | 0.132 ms/步，2 launch | 持平；首步 447.8→15.4 ms（首调编译） |
| 336 张量（BGE-M3 口径） | 26.8 ms/步，2,688 launch，峰值 10.63 GB | **11.7 ms/步，39 launch**，峰值 10.57 GB | **2.3×，launch -98.6%** |

正确性见 `adamw_check.json`（5 步轨迹一致）。vendor 白名单维持不变。

## 覆盖与 pending

- ✅ 环境/dispatch/注册入口/命中证明：`env_record.json`、`probe1_dispatch.json`、`probe3_scaling_hits.json`
- ✅ 三接入配置对照（native / gems_full / gems_selective）×3 进程逐迭代样本：`perf_*_r{1..3}.json`
- ✅ 精度 126 用例 v5 判据 + 逐例证据 + 负例自测：`accuracy_matrix.json`、`test_comparator_negative.log`
- ✅ AdamW 正确性 + 性能：`adamw_check.json`、`probe6_adamw_perf.json`
- ✅ 归因（截距/斜率、enqueue 分解、入口对照）：`probe5_overhead_attribution.json`
- ⏳ pending（不计通过，拆分为后续任务）：真实 BGE-M3 / x-benchmark 整模型；多卡/通信整步；ZW810E 平台 fix 条目（若立项，优先评估 per-call host 开销路径）；`_FUSED_ADAMW_VALIDATED_VENDORS` 加入（需 owner 审核）
- 本轮为**平台基线记录**，不构成任何 fix 的硬件验收；现有条目 `draft / deployable=false / hardware_status=unverified` 维持不变；本轮不代表 PR2 性能瓶颈已修复。

## 复现命令

```bash
export PYTHONPATH=/path/to/FlagGems/src   # 29e14822
export CUDA_VISIBLE_DEVICES=0
python3 scripts/accuracy_matrix.py --out results/accuracy_matrix.json   # v5 判据
python3 scripts/test_comparator_negative.py                             # 设备无关，exit 0
python3 scripts/adamw_check.py --out results/adamw_check.json
python3 scripts/probe6_adamw_perf.py
for cfg in native gems_full gems_selective; do
  for rep in 1 2 3; do
    python3 scripts/perf_matrix.py --config $cfg --out results/perf_${cfg}_r${rep}.json
  done
done
python3 scripts/probe5_overhead_attribution.py   # 进程隔离版（v2）
python3 scripts/probe3_scaling_hits.py
for cfg in native gems_full gems_selective; do
  python3 scripts/small_model.py --config $cfg --out results/small_model_${cfg}.json
done
python3 scripts/probe4_small_model.py
```

## 修订记录

- 2026-11-12 首轮（RESULTS.md 原文，见 git 历史或 issue 评论）：判定"需优化"；"每算子固定开销 ~60–80 µs"为假设；"无需新 fix / 厂商融合优先"为过度推断；小模型逐位一致表述过宽。
- 2026-11-13 第二轮（本文件）：撤回过度推断；固定开销升级为拟合证据并修正为 host enqueue 归因；判据三分类 + 负例自测；批次关系披露；结论限定适用范围；AdamW 性能证据补充；native/full/selective 明确为接入配置而非优化实现。
