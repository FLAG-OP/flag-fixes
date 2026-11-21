# ZW810E problem29 平台基线（issue #5，第三轮修订 v6）

日期：2026-11-12 首轮 · 2026-11-13 第二轮（PR #6 证据落地）· 2026-11-13 第三轮（按复审 6842724980 修订）
平台：平头哥 PPU-ZW810E（单卡）· 目标树：FlagGems `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`v5.3.5-46-g29e14822`）
环境：`results/env_record.json` · 原始数据：`results/`（逐迭代样本、逐例判定、注入自测）· 复现命令见文末

**第三轮修订（回应复审五点）**：①小模型显式迁移到 PPU 并补命中证据，v1 CPU 结果作废；②精度判定改逐位置（消除跨位置稀释，审阅反例进负例自测）；③ULP 有序映射修复（无符号位型），正常区/FTZ 区差异分开报告；④生产判定抽取为共享模块 `comparator.py`，自测直接调用生产判定，±0 按位型比较；⑤失败计数与退出码可靠化（硬失败→exit 1），注入失败自测验证生产入口。

## 结论（第三轮口径）

**平台判定：需优化 + gems_full 训练路径当前被平台阻断（F-1/F-2 挂起）。适用范围：FlagGems `29e14822` / 0.3M–98M 尺寸集 / 单卡 / 本轮负载；不外推。**

1. **性能（微基准 batch-2，标签与口径已修正）**：18 项微基准 gems_full **18/18 慢于 native（1.29–6.45×）**，无一项更快。归因维持第二轮的拟合证据：**FlagGems 截距 60.4/70.0 µs vs native 4.7/4.6 µs，斜率与 native 基本相同**；host enqueue 分解 **native ≈5 µs/次 vs FlagGems ≈57–69 µs/次**；直调入口与 aten 注册入口命中同一 Triton kernel（差 ~5 µs）。**该证据支持"优先探索调用提交（enqueue/launch）路径"的方向假设，但尚未拆分 python/libentry/triton-launch 各环节贡献，也未测得纯设备执行成本差——不作为最终根因结论。**
2. **精度（v6 判据，逐位置门）**：126 用例 = **72 PASS / 39 DIFF / 15 FAIL（硬失败，exit 1）**。相比 v5 的"0 硬失败"，修复跨位置稀释后暴露了真实偏差：
   - **15 FAIL**：`gelu_backward`（gelu_none fp32/bf16、gelu_tanh fp32）在**近抵消小梯度区**（|grad|≈1e-8–1e-6，仍属 normal zone）逐位置相对误差超过 `max(4·eps, 1.5×native同位置)`——gems 最差达 native 误差的 ~2×（例：oracle -2.90e-08、native -2.98e-08、gems -5.96e-08，即 gems 100% 偏差、native 2.7%）；gelu_tanh fp32 前向在平滑区也有 ~2.5× native 相对误差的位置（abs ~1e-7 量级）。位置数：grad 7–74 万/2.6M（large/typical），fwd 2.6–4.8K。**绝对幅度小（≤1e-6 量级），按声明的逐位置判据构成 FAIL；对训练的实际影响评估与上游 kernel 修正建议已列入移交项。** 逐例 top-5 违例四元组（位置/native/gems/oracle）已归档于 JSON。
   - **39 DIFF（记录性差异，非失败）**：FTZ 区位型差 33 例（合计 91,135 个次正规位置，最大绝对差 1.19e-07）、±0 符号差 9 例、grad FTZ 9 例、grad ±0 3 例。**正常区位级差另列报告**：26 例存在合计 6,683,858 个 normal-zone 位级差异（±1–2 ULP，逐位置精度门内），**不再表述为"全部差异仅在 FTZ 区"**。
   - 判据与实现：生产判定模块 `scripts/comparator.py`（accuracy_matrix 与负例自测共用）；阈值 `max(4·eps, 1.5×native同位置)` 事先声明；FTZ 定义=oracle |y| < dtype 最小正规数，按**原始位型**比较；±0 按**原始位型**比较（torch.equal 视 +0==-0，不可用）。oracle=aten fp64 精确公式——**注明边界**：fp64 解析参考在负尾（|x| 大时 gelu 下溢为 0）同样受限，非无限精度真值；比较只在有限精度意义下成立。
   - 负例自测（设备无关）：`test_comparator_negative.py` 直接调用生产判定，**22/22 通过**（12 项 comparator 单元断言 + 10 项生产路径负例）（含审阅反例：FTZ 位置 native 误差不得放宽正常位——`oracle=[1e-40,1.0], native=[0,1], gems=[0,2]` 现判 FAIL）；`accuracy_matrix.py --inject-fail` 生产入口注入自测：污染用例判 FAIL 且**进程 exit 1**（`results/accuracy_matrix_inject.json`）。ULP 断言：±0 距离 1、相邻值 1、±min_normal 跨零距离 2049（fp16）、min_sub→0 距离 1。
   - 边界位型：fp16 `0x03FF`、bf16 `0x007F`、fp32 `0x007FFFFF` 断言为最大次正规。
3. **小模型（v2.1，显式 PPU 设备，可训练 embedding）**：embedding 在 optimizer 构造**前**创建并**纳入参数列表**（16 参数，embedding 梯度证据：cuda/finite/max_abs 3.7e-07）；blocks/head/embedding/token 全部 `.to("cuda")`，JSON 记录断言 `params_all_cuda_before_optim=true`、`grads_all_on_cuda=true`、`embedding_trainable_in_params=true`；profiler 命中证据：selective 配置前/反向命中 `gelu_none_kernel_rank_1`/`gelu_backward_none_kernel_rank_1`，native 配置命中 `GeluCUDAKernelImpl`/`GeluBackwardCUDAKe…`。**替代说明**：attention 用显式 matmul+softmax（因 F-1 sdpa 挂起）；结构仍为固定种子 gated MLP + 2 层 attention 块，无外部权重。**结果（native vs selective，8 步）**：逐步 loss 相对差 ≤1.2e-07（fp32 累积量级；embedding 梯度 max_abs 8 位有效数字一致）；16/16 参数 grad sha256 不同（两套 gelu kernel 数值路径不同，与精度矩阵 grad 差异一致）——**本负载只能证明"当前固定小模型可运行、数值偏差在 fp32 累积量级"，不能表述为逐位一致**；每配置 3 独立进程损失序列一致（配置内确定性 ✓；非逐位一致，fp32 舍入差异存在）；稳态步时 native ~2.25 ms / selective ~2.85 ms。**v1 的"三组逐位一致"结论作废**（v1 未上设备，实为 CPU aten 自身一致性）。
   - **gems_full 训练在本机当前被阻断**（F-2：matmul backward 挂起），无法提供 8 步对照——见下。
4. **平台阻断发现（F-1/F-2，本轮新发现）**：
   - **F-1**：FlagGems sdpa（`_flash_attention_forward` 注册路径）在训练形状 fp32 `[1,8,128,256]` **前向即挂起**（>300s 无返回、无编译活动）；推理形状 bf16 `[8,16,512,64]` 正常（微基准 0.17ms）。
   - **F-2**：FlagGems **matmul backward** 挂起（fp32 `[8,128,256]@[256,512]`）；同形状 forward 与大 forward 正常（微基准 2.18ms）。
   - 处置：按 issue 设备异常规则**单次隔离触发、不重试**（`probe7_platform_blockers.py`，exit 1）；**来源注记**：同一 FlagGems 树 + 小模型负载在 2026-11-12 晚间曾完整跑通（v1 产物，step 10.5–15.3 ms）——挂起与设备/驱动状态相关（11-12 21:30 至 11-13 11:30 间环境有变化，非本仓库可控）；待平台恢复后隔离复验。**F-1/F-2 未据此确认算子实现新增错误**（复审口径），作为平台阻断登记。
5. **AdamW**：正确性（5 步轨迹一致，fp32 ≤2.4e-6 / bf16 1 ulp）与性能证据保留：336 张量 stock 26.8 ms/步（2,688 launch）vs fused 11.7 ms/步（39 launch）= **2.3×**。**为单次探针初步结果**，后续优化立项时再补独立重复与代表性参数配置；vendor 白名单维持不变。

## 微基准（batch-2；p50 ms，3 独立进程取中位，分位数=进程内 30 次迭代统计；逐迭代样本在 `samples_ms`）

| 算子 | native p50 | native p95 | gems_full p50 | gems_full p95 | full/native | selective p50 | selective p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cast_to/bfloat16 | 0.108 | 0.109 | 0.142 | 0.159 | 1.32x | 0.108 | 0.109 |
| cast_to/float32 | 0.026 | 0.027 | 0.083 | 0.088 | 3.13x | 0.026 | 0.028 |
| findmax/fp32 | 0.021 | 0.022 | 0.113 | 0.136 | 5.32x | 0.021 | 0.023 |
| gelu_and_mul/bf16 | 0.024 | 0.025 | 0.155 | 0.180 | 6.45x | 0.082 | 0.086 |
| gelu_and_mul_fused/bf16 | 0.024 | 0.025 | 0.086 | 0.093 | 3.58x | 0.087 | 0.095 |
| gelu_fwd_bwd_none/bfloat16 | 0.071 | 0.078 | 0.264 | 0.270 | 3.73x | 0.264 | 0.284 |
| gelu_fwd_bwd_none/float32 | 0.125 | 0.126 | 0.266 | 0.306 | 2.13x | 0.263 | 0.290 |
| gelu_fwd_none/bfloat16 | 0.028 | 0.028 | 0.080 | 0.086 | 2.88x | 0.081 | 0.106 |
| gelu_fwd_none/float32 | 0.051 | 0.057 | 0.082 | 0.130 | 1.60x | 0.083 | 0.123 |
| gelu_fwd_tanh/bfloat16 | 0.027 | 0.027 | 0.082 | 0.098 | 3.08x | 0.082 | 0.088 |
| gelu_fwd_tanh/float32 | 0.051 | 0.052 | 0.082 | 0.095 | 1.58x | 0.081 | 0.087 |
| mm_transposed/fp32 | 0.684 | 0.685 | 2.180 | 2.189 | 3.19x | 0.684 | 0.684 |
| mul_scalar/bfloat16 | 0.014 | 0.014 | 0.080 | 0.088 | 5.73x | 0.014 | 0.015 |
| mul_scalar/float32 | 0.048 | 0.049 | 0.089 | 0.100 | 1.85x | 0.048 | 0.050 |
| pointwise_add/bfloat16 | 0.013 | 0.014 | 0.084 | 0.095 | 6.24x | 0.013 | 0.014 |
| pointwise_add/float32 | 0.049 | 0.050 | 0.094 | 0.114 | 1.93x | 0.049 | 0.050 |
| sdpa_fwd/bf16 | 0.131 | 0.132 | 0.170 | 0.180 | 1.29x | 0.131 | 0.132 |
| transpose_copy/fp32 | 0.040 | 0.041 | 0.101 | 0.118 | 2.52x | 0.043 | 0.044 |

口径修正：`gelu_fwd_bwd_none` = **前向+反向合计**计时（原 `gelu_bwd_none` 标签不准确）；`cast_to/bfloat16` = bf16→fp32→bf16 往返（原 bf16→bf16 为 no-op，已修正）；`gelu_and_mul` 实测形状 **(4096,1280)**（第二轮 README 误写 (16384,2560)，以本表为准）；mm `[4096,1280]@[1280,2048]ᵀ`；sdpa `[8,16,512,64]` bf16 推理。warmup 5（排除 JIT/首编译），30 次 CUDA event 同步计时。batch-1 旧数据存 `results/perf_batch1_superseded/`（数字与 batch-2 一致性已核对，方向不变）。

## 归因证据（probe5 v2，进程隔离）

| op | native 截距 | gems 截距 | native 斜率 | gems 斜率 |
|---|---:|---:|---:|---:|
| gelu fp32 | 4.73 µs | 60.43 µs | 0.0039 ns/元素 | 0.0033 ns/元素 |
| add fp32 | 4.60 µs | 70.02 µs | 0.0038 ns/元素 | 0.0030 ns/元素 |

host enqueue：native 4.9–5.5 µs/次 vs FlagGems 56.7–68.7 µs/次（gelu/add，1.2M/12.25M）；直调 vs aten 注册入口同一 kernel、enqueue 差 ~5 µs。**注**：python/libentry/launch 各环节贡献未拆分，"纯设备执行成本无差"未单独测量——方向假设非最终根因。

## 精度矩阵判定汇总（v6，逐位置门）

| 判定 | 数量 | 说明 |
|---|---:|---|
| PASS | 72 | 与 native 位级一致，或逐位置满足 `rel ≤ max(4·eps, 1.5×native同位置)` 且 FTZ 区位型一致 |
| DIFF | 39 | FTZ 区位型差（33 例 / 91,135 位置，max abs 1.19e-07）；±0 符号差（9 例）；grad FTZ（9）/grad ±0（3）——记录性接受 |
| FAIL | 15 | 逐位置门违例（见结论 §2，top-5 违例四元组逐例归档）；**进程 exit 1** |

## 覆盖与 pending

- ✅ 环境/dispatch/命中证据：`env_record.json`、`probe1_dispatch.json`、`probe3_scaling_hits.json`、小模型逐配置 kernel 命中（v2 JSON 内）
- ✅ 三接入配置微基准（batch-2）×3 进程逐迭代样本 + 错误聚合退出码
- ✅ 精度 v6（生产判定 + 注入自测 + 负例 22/22 + 逐例证据 + top-5 违例四元组）
- ✅ 小模型 native/selective 上设备对照（8 步、device 断言、命中证据、3 进程确定性）
- ✅ 平台阻断 F-1/F-2 单次触发出证（`probe7_platform_blockers.json`）
- ⏳ pending（拆分为后续任务，不计通过；按复审 6845369130 列明）：
  1. **15 个 hard FAIL 的位置级归因**：分别复现 native/Gems kernel 输出、fp64 oracle 与梯度，区分 GELU kernel 数值误差、反向近抵消放大与参考路径误差，并给出训练 loss/参数轨迹影响；
  2. **F-1/F-2 平台恢复后隔离复验**：记录驱动、SDK、FlagGems commit、shape、超时与编译活动；复验完成前保持 full-enable training blocked；
  3. **host enqueue 开销拆分**：Python / libentry / Triton launch / 设备执行时间分别测量，之后再决定是否需要新的优化 fix；
  4. **真实 BGE-M3 / 整模型与多卡测试**；AdamW 2.3× 保持 preliminary，不加入 vendor whitelist（需独立重复 + owner 审核）；`_FUSED_ADAMW_VALIDATED_VENDORS` 加入需独立条目 + owner 审核。
- 本轮为平台基线记录，不构成任何 fix 的硬件验收；现有条目 `draft / deployable=false / hardware_status=unverified` 维持不变；不代表 PR2 性能瓶颈已修复；`-15.3%` 不外推三平台。

## 复现命令

```bash
export PYTHONPATH=/path/to/FlagGems/src   # 29e14822
export CUDA_VISIBLE_DEVICES=0
python3 scripts/comparator.py                     # 生产判定单元自测, exit 0
python3 scripts/test_comparator_negative.py       # 负例 22/22, exit 0
PYTHONPATH=$PYTHONPATH python3 scripts/accuracy_matrix.py --out results/accuracy_matrix_inject.json --inject-fail
                                                  # 注入自测: FAIL + exit 1（预期非零）
python3 scripts/accuracy_matrix.py --out results/accuracy_matrix.json
                                                  # v6 全量: 72/39/15, exit 1（存在硬失败, 预期非零）
python3 scripts/adamw_check.py --out results/adamw_check.json
python3 scripts/probe6_adamw_perf.py
for cfg in native gems_full gems_selective; do
  for rep in 1 2 3; do
    python3 scripts/perf_matrix.py --config $cfg --out results/perf_${cfg}_r${rep}.json
  done
done
python3 scripts/probe5_overhead_attribution.py
python3 scripts/probe3_scaling_hits.py
for cfg in native gems_selective; do              # gems_full 训练被 F-2 阻断，暂不可跑
  python3 scripts/small_model.py --config $cfg --out results/small_model_v2/small_${cfg}.json
done
python3 scripts/probe7_platform_blockers.py       # F-1/F-2 出证, exit 1（预期非零）
```

## 状态口径（复审 6845369130 确认）

**基线完成、需优化**——不作为平台已验收或可部署结论；不据此把相关 fix 标记为 `deployable`。

## 修订记录

- 2026-11-12 首轮：判定"需优化"；固定开销为假设；小模型未上设备（不知情缺陷）；"0 硬失败"基于稀释门。
- 2026-11-13 第二轮（PR #6 f288df6）：证据落地；判据三分类 + 负例自测（发现并修 ±Inf 符号缺口）；归因修正为 host enqueue；结论限定范围。
- 2026-11-13 第三轮 v3.1（本文件，回应复审 6845369130 两处口径）：负例自测改 22/22；embedding 改为 optimizer 前创建并纳入参数（16 参数 + 梯度证据）；小模型口径改"逐步差 ≤1.2e-07、不能表述为逐位一致"；补列四项跟进任务与状态口径。第三轮原文：v6 判据逐位置化（审阅反例入自测）、ULP 修复、正常区/FTZ 区分开报告、**15 硬 FAIL 如实报告**；小模型上设备重跑（v1 作废），native/selective 训练数值差异量化；发现并登记 F-1/F-2 平台阻断（gems_full 训练阻断、sdpa 挂起、matmul backward 挂起）；perf batch-2 重跑（标签/口径修正）；失败计数与退出码可靠化。

---

## issue #7 第一轮（2026-11-13）：fused AdamW 与调用提交开销

面向 issue #7（A–E）。新脚本：`scripts/{adamw_deep,probe8_enqueue_candidates,fail_attribution,small_model,probe7_platform_blockers}.py`；新数据：`results/{adamw_deep,probe8_enqueue_candidates,fail_attribution_gelu_bwd,probe7_recheck_2026-11-13}.json`、`results/small_model_e/`。

### B. fused AdamW（torch.optim 层完整验证）

| 配置 | dtype | stock | foreach | fused | 收益 | 正确性 |
|---|---|---:|---:|---:|---:|---|
| single [1024,1024] | fp32 | 0.190 ms | 0.186 | 0.106 | 1.8× | step1 ≤2 ulp@scale ✓；100 步发散 2.4e-07（有界）|
| single | bf16 | 0.192 | 0.189 | 0.109 | 1.8× | 同上（100 步 4.2e-03 = 线性 ulp 累积）|
| multi (8) | fp32 | 0.212 | 0.207 | 0.121 | 1.8× | ✓ |
| multi (8) | mixed | 0.294 | 0.287 | 0.142 | 2.1× | ✓ |
| bge336 | fp32 | 29.67 | 29.50 | **10.54** | **2.8×** | ✓；launch 2,688→39 |
| bge336 | bf16 | 16.49 | 16.44 | **8.94** | **1.84×** | ✓ |
| bge336 | mixed | 30.50 | 30.70 | **10.48** | **2.9×** | ✓ |

- **foreach ≡ stock（逐位一致）且无收益**——不是候选。
- **fused 正确性按四个独立层次区分（不得混用）**【第三轮补充：状态比较声明修正——v2 脚本的 state_compare 因 `id(p)` 索引错位实为空集，v3 已修复并重跑（5,520 项状态对比）：foreach 状态与 stock 逐位一致；fused 的 exp_avg/exp_avg_sq 相对 stock 存在声明容差内外的偏离（求和顺序 + 低精度内部 fp32 计算），逐项记录于 `adamw_deep_v3.json`。旧 `adamw_deep.json`（v2）superseded】：
  1. **单步更新 vs fp64 闭式公式**：≤1 fp32 ulp（同公式确认）；
  2. **100 步轨迹 vs stock**：存在可观漂移——fp32 100 步大量位置超过 4·eps（max abs 2.7e-07）、bf16 max ~29,564 ULP——独立舍入路径的预期累积，**fused 不承诺与 stock 长期逐位一致**；
  3. **checkpoint 恢复**：本批次内部逐位一致（50+50 步）；
  4. **无 NaN/Inf**（fused 全配置）。
  验收口径以后续 fused vs fp64/稳定参考的误差与训练行为为主，不以 stock 逐位一致为门槛（待补）。
- **新发现（fp16，限定本轮已测配置：本设备/已测形状/lr=1e-3/eps=1e-8）**：stock 与 foreach 的 fp16 AdamW **数值损坏**（Inf/NaN：single 45,378 位置、bge336 25,588,726 位置；机制：fp16 下 `v_hat=g²` 下溢为 0 且 `eps=1e-8` 下溢为 0 → 0 除）；**fused 在当前配置下无污染**（0 NaN/Inf）。二者分别报告、不合并为"所有 fp16 AdamW 配置"的普遍定理；**fp32 优化器状态（master weights）为不依赖 fused 的替代方案**，其对照待补。
- checkpoint 保存/恢复（fused bge336, 50+50 步）：**参数与状态逐位一致**。
- 收益跨实验稳定：probe6 2.3×、adamw_deep 2.8×、小模型 optimizer 段 1.76–1.9×/整步 -5.6~-11%（§E）。FlagScale `build_optimizer` 集成路径验证 **pending**（本机无 FlagScale 运行环境）；vendor 白名单维持不变，待集成验证 + owner 审核。

### C. 提交开销：CUDA Graph 捕获/重放在 ZW810E 可用（候选 1 成立）

gelu fp32 图重放（100 次调用捕获、重放均值）vs 单次 event：

| 尺寸 | event | graph replay/iter | 提升 |
|---|---:|---:|---:|
| 1.2M | 0.0731 ms | **0.0058 ms** | **12.6×**（enqueue 完全消除）|
| 12.25M | 0.0868 | 0.0519 | 1.67× |
| 98M | 0.3823 | 0.3875 | 持平（纯设备时间）|

→ 证实提交开销假设（重放后 1.2M 5.8 µs ≈ 纯设备执行），**图捕获/重放为组件级、静态 shape 候选**；12.6× 为微基准结论，**不得外推为整步收益**（尚未接入训练整步 fwd+bwd+optimizer 子图，动态 shape 边界、内存池/捕获限制、与 optimizer 组合收益均未测，捕获失败回退策略未定）；组件级拆分（python/libentry/launch）与 raw-kernel 直调入口未完成（生产 wrapper introspection 未暴露稳定入口，如实记录）。thead 无独立 gelu 覆写（回退共享 `ops/gelu.py`；pow 出现 7 处但 none-path 主项为 erf/exp，非昆仑芯 pow 热点同型——不套用昆仑芯根因）。

### D. 15 hard FAIL 归因（gelu_backward 负尾）+ F-1/F-2 复验

- 违例位置 84% 在 |x|>3.9 负尾（p50 |x|=7.1）：**FlagGems fp32 gelu_backward 在 |x|≳8 输出精确 0.0**（native/真值 ~1e-14–1e-36），5.5≤|x|≤8 幅度偏差至 1.6×，|x|<5 正常；绝对误差 ≤1.14e-07。fp64 双公式模拟确认 gems 公式数学等价——**负尾有限精度伪影（erf 抵消 + exp 项损失），非算法错误**；单点梯度 ≤1.14e-07，**候选训练影响较小——但"可忽略"的定稿需模型级敏感性/长步训练对照支持，当前为推断、待训练验证**。第四轮归因修正与扩展（复审 6876765307；v1/v2 归因批次 superseded，现行 `fail_attribution_stats_v3.json`）：①v1 参考公式错误已撤回（正确导数 `Phi(x)+x·phi(x)`）；②参考改 erfc 稳定 fp64 形式；③violation 掩码用生产 comparator；④**全 15 FAIL 类别三级匹配统计**（违例子集内，对 TTIR 顺序 fp32 模拟）：
- **gelu_none 9 类（分母 4,012,714）：unexplained=0**——exact = ulp≤2 = **3,967,306（98.87%）**（两者计数相同，不存在"ulp≤2 补齐其余"）；容差 4,012,714（**100%**）覆盖剩余位置；**求和顺序量化：有证据范围限制的归因假设**（编译后 TTIR：`pdf + 0.5·erf` 在 0.5 量级舍入、fp32 ulp=2^-24≈5.96e-08，再 `+0.5` → 负尾梯度被量化到 2^-24 栅格或精确 0）；
- **gelu_tanh backward 6 类：unexplained 3,190/430,500**（每类 426–628 位置未解释——模拟公式与编译内核细节差异，独立调查）；**与 gelu_tanh forward 26/22,242 分别报告，不合并为一个比例**；
- **gelu_tanh forward 6 类：前向模拟公式 x²→x³ 修正后，unexplained 26/22,242**（其余 99.88% 被修正后模拟解释；剩余 26 位置独立调查）。
- 栅格命中按存档样本逐个判定：**none 类 25/26（96.2%）、tanh 类 1/19（5.3%）、合计 26/45**（v2 报告的 46 为分母缺陷，已修正）。三级匹配总量（最终 JSON）：exact 4,192,640（94.36%）、ulp≤2 4,290,428（96.57%）、容差 4,440,024（99.93%）——**容差匹配非位级证明**。
- 容差匹配口径声明（统一常量后）：`|sim-device| ≤ 6e-8 + |device|·2×4.77e-07` 下全张量匹配 **39,304,592/39,321,600（99.957%）**——该容差非位级精确证明（6e-8 接近 2^-24 量化间隔本身）。
- TTIR 证据归档：`results/ttir_evidence/{gelu_backward_none_kernel_rank_2,gelu_tanh_kernel_rank_1,gelu_tanh_kernel_rank_2}.ttir` + SHA256SUMS；**适用范围注明**：gelu_backward_none 存档 TTIR 为 bf16/rank-2 编译产物（none backward fp32 类的量化结论部分依赖同源源码公式，非全部布局的直接编译证据）；tanh 存档 TTIR 为 fp16/bf16 内核，不视作 fp32 tanh 失败路径的直接编译证据；相关内核证据不外推到其他失败路径。
- F-1/F-2 复验（2026-11-13 第二次，单次触发）：**仍挂起**（控制组正常）——full-enable training 保持 blocked；等平台/设备恢复后按记录字段复验。

### E. 小模型 × fused optimizer（可训练 embedding，8 步，全部 finite）

| 配置 | optimizer 段 | 整步 | loss 相对差 vs native/stock |
|---|---:|---:|---:|
| native/stock | 0.422 ms | 2.27 ms | — |
| native/fused | 0.219 ms（1.9×）| 2.01 ms（-11%）| ≤4.7e-07 |
| selective/stock | 0.394 ms | 2.84 ms | ≤1.2e-07 |
| selective/fused | 0.224 ms（1.76×）| 2.68 ms（-5.6%）| ≤4.7e-07 |

### 结论（issue #7 第一轮）

| 候选 | 状态 |
|---|---|
| fused AdamW（fp32/bf16/mixed、bge336 级负载） | **有效且精度通过**：2.8–2.9×（optimizer 段）、整步 -5.6~-11%（小模型）；checkpoint 批次内逐位一致；fp16 仅在本轮已测配置下 fused 无非有限值（非普遍定理）。候选 fix 定位：FlagScale `adamw-fused` 集成验证 pending + fp16 建议改为"当前配置下 fused 为已验证可用路径，fp32-state 替代待对照" |
| CUDA graph 捕获/重放（静态 shape） | **可行候选**（1.2M 12.6×，组件级微基准）；未接入训练整步，不得外推整步收益；组件拆分、动态边界、回退策略待下一轮 |
| selective enable | 维持（无变化）|
| full-enable training | blocked（F-1/F-2）|

### 第一轮复审口径修订（2026-11-13，意见 6857432618）

四处口径已按审阅修正：①fused 正确性四层次区分（单步闭式 ≤1 ulp / 100 步轨迹漂移不承诺逐位 / checkpoint 内部一致 / 无 NaN/Inf）；②fp16 分别报告 stock 损坏与 fused 当前配置无污染，限定范围并补 fp32-state 替代方案；③GELU 负尾影响改"候选较小、待训练验证"；④CUDA graph 明确组件级静态 shape 候选、12.6× 不外推整步。下一轮顺序按审阅执行：FlagScale 集成 → AdamW 数值验收（vs fp64/稳定参考）→ CUDA Graph 训练段接入 → GELU 归因训练闭环 → F-1/F-2 持续复验 → 整步/多卡。

### 第三轮实施与 provenance（复审 6876765307）

- `adamw_deep.py` v3：状态按参数稳定索引对齐（修复 `id(clone)` 错位）、shape/dtype 校验、缺状态/空对比/数量不符=硬失败、mixed 逐张量 dtype、checkpoint 失败进入退出码、错误 schema 统一（非有限 step1 含全字段）、`--selftest` 入口级负例 3/3。**重跑 12 组合**（`adamw_deep_v3.json`，exit 1——状态容差违例如实入账）：5,520 项状态对比（foreach 逐位一致；fused exp_avg/exp_avg_sq 偏离记录）；fp16 stock/foreach nan/inf 证据保留；checkpoint 逐位一致。`adamw_deep.json`（v2，state_compare 空集）**superseded**。
- `fail_attribution_v2.py`/`fail_attribution_v2.json`：公式修正 + erfc 稳定参考 + 生产 comparator 掩码 + 全 15 类覆盖 + TTIR 顺序模拟（见上）。`fail_attribution_gelu_bwd.json`（v1）**superseded**（其 aten 参考公式有误）。
- `probe8_graph_output_verify.py`/`...json`：图重放输出对照（新输入写入 → 重放 → 与 eager 逐位一致）。
- 设备未重跑项：微基准 batch-2、accuracy v6、小模型 v2.1 均维持既有产物（复审确认可保留）。

### 第四轮实施与 provenance（复审 6876765307 下一轮顺序第 1–3 步）

- `adamw_deep.py` v4：**唯一生产聚合函数 `aggregate()`**（summary 与 exit code 同源；v3 的双聚合列表已删）；`--selftest` 注入五类用例经真实 `aggregate()`（仅 checkpoint 失败/仅状态缺失/仅状态超差/仅 step 不匹配 → 全部映射非零；全过对照 → 空+exit 0）6/6。**重跑 12 组合**（`adamw_deep_v4.json`，exit 1 与 hard_deviations 2,702 项同源）：状态验证已执行且存在超差/非有限失败（如实入账）；v1–v3 批次 superseded（v1/v2 state_compare 静默空、v3 双聚合列表）。
- `fail_attribution_stats.py`/`fail_attribution_stats_v3.json`：违例子集三级匹配（exact 位级 / ulp≤2 / 声明容差）+ tanh 前向独立归因 + 未解释位置示例（见上第四轮归因段）。exit 1（unexplained>0，如实）。
- `probe8_graph_output_verify.py` 修正捕获次数显式化（捕获循环/计时分母/元数据同源），重跑：每调用摊销 0.0053 ms（修正此前 100 倍低估）、输出对照逐位一致。
- 未重跑项：微基准 batch-2、accuracy v6、小模型 v2.1、F-1/F-2（保持 blocked）。

### 第五轮实施（复审 6879678135）

- **tanh 前向模拟公式修正**（x²→x³；CPU 反例 x=2：1.94005→1.95460 与解析值一致）+ 前向公式自测（0/±2/尾部 5 点，含复审反例）；修正后重跑前向归因：unexplained **26/22,242**（99.88% 被修正后模拟解释；此前 22,242 全未解释系公式错误）。
- **三级匹配统计修正**：bf16/fp16 的 exact 比较改在目标 dtype 位表示（此前 fp32→int16 拆分致 exact 双计——bf16 503,890 而非 1,007,780）；TOL_REL 统一为单一常量、判据只用一次；`explained = exact ∪ ulp≤2 ∪ tol`、`unexplained` = 补集（口径显式）；不变量自测 `0≤count≤gate_violations` 内置 assert。最终：exact 94.36%、ulp≤2 96.57%、容差 99.93%、unexplained 3,216/4,443,214（backward 3,190 + forward 26）。
- **AdamW 聚合补齐**：compare_state fp32 分支记录 nan/inf/nonfinite 计数；`aggregate()` 将状态 NaN/Inf 列为硬失败；自测新增**仅状态 NaN / 仅状态 Inf**（经真实 `aggregate()`，8/8 通过）。v5 聚合语义应用于 v4 设备数据 → 2,702 项硬偏差（与归档一致），v4 及更早聚合版本 superseded。
- 文档数字全部由最终 JSON 生成：none 9 类（非 12）、tanh backward 3,190/430,500、tanh forward 26/22,242、三级匹配 94.36%/96.57%/99.93%。
- 未重跑项：微基准 batch-2、accuracy v6、小模型 v2.1、F-1/F-2（blocked）。AdamW 状态 NaN/Inf 新增计数无需设备重跑（利用 v4 已有数据；新计数路径由自测覆盖）。

### 第六轮实施（复审 6886949698 三项收尾）

- **exit_rule 持久化**：先构造完整 out（含 backward/forward 未解释数与唯一退出判定）→ 写文件 → 序列化后回读断言 saved==verdict → 打印 → 退出；batch 标识更新为本轮语义（v3.1）。
- **摘要分母修正（auto_summary 字段由脚本自动生成，README/PR 按其引用）**：none 分组（9 类，分母 4,012,714）：exact = ulp≤2 = 3,967,306（98.87%）、容差 100%（4,012,714）、unexplained 0；backward 全类别 3,190/4,443,214（仅 tanh 3,190/430,500）；forward 26/22,242——总体与子集分组分开报告，不合并比例。
- **不变量断言移至全部赋值之后**，并新增 `explained_union + unexplained == gate_violations` 恒等检查（对现有 JSON 通过；脚本修正保障未来产物）。
- **compare_state → aggregate 全链路 CPU 反例已补**（torch CPU 环境）：NaN/Inf 状态张量经真实 compare_state 后进入 aggregate → state_nan_inf 硬失败（selftest 9/9）。旧设备 JSON 中 1,848 条 fp32 状态记录缺 Inf/nonfinite 计数——确认为历史数据边界，不追溯证明其无 Inf；未来正式验收用新比较器重采完整字段。

---

## 第八轮（复审 6887760539 + 6888593048）：回到 problem29 原对象——核查清单与 native 基线

按审阅要求回到 problem29 原问题仓库中的测试对象、训练入口与配置。完整清单见 `results/orig_entry/`（含原入口验证文件）与本节；本节仅列事实。

### 来源（真实入口，全部已验证）

- 原问题仓库：https://github.com/wxj630/FlagScale/tree/feat/retrieval-flagos（problem.md 首行给定；分支 HEAD `6862134`）
- FlagScale 基线 commit：flagos-ai/FlagScale @ `89f342e690053e07dda845f1365089ec00464e09`（"docs(retrieval): record the real 3-epoch wall clock, and add a three-platform watcher"）
- 训练入口：`flagscale/train/train_retrieval.py`（1,325 行）——**sha256 = 5ee5757…11df，与 fixes/adamw-fused/fix.yaml `before_sha256` 完全一致**（adamw-fused 的目标文件即此文件）
- 配置：`examples/retrieval/conf/train/bge_m3.yaml`（batch 10、gradient_accumulation 4、bf16、max_length 512、lr 2e-5、T²-Ranking triplet-15）
- profiling 脚本：`examples/retrieval/profile_step_segments.py`、`profile_op_micro.py`（problem.md §二/§三的工具）
- 原始命令：`python examples/retrieval/profile_step_segments.py --conf examples/retrieval/conf/train/bge_m3.yaml --steps 3 --skip 2 --prof-steps 4 --device cuda`
- 权重：BAAI/bge-m3（hf-mirror 下载 `pytorch_model.bin` 2,269,966,336 B + `sparse_linear.pt`/`colbert_linear.pt`/tokenizer）
- 数据：sentence-transformers/t2ranking `triplet-15/`（11 parquet，1.1 GB，schema anchor/positive/negative_1..15；与 FlagScale 文档"11 个文件, 340,111 行"一致；conf 数据路径同名同 schema）
- x-benchmark 容器入口：`/home/wuyiqiang/x-benchmark` + `run_model.py`——**不在本机**（见阻碍②）

### 对象与差异

原对象 = BGE-M3（XLM-R large 24 层/hidden 1024/vocab 250002，bf16，CLS pooling + sparse/colbert 头，对比损失 temperature 0.05，AdamW lr 2e-5，真实 T²-Ranking triplet-15 数据，length-sort 窗口 256）。与自建小模型（2 层/D256/fp32/随机 token/MSE）和昆仑芯重建负载（XLM-R large 随机权重、非预训练）的差异已按审阅表格要求列于 `results/original_object_checklist.md` §二。

### 已执行（native 基线，exit 0）

- 权重 + 数据下载（hf-mirror，3.4 GB 落盘）；入口文件 sha256 与 fix.yaml before 完全一致
- **整步 wall = 2,131.9 ms**（batch 10 × accum 4 = 200 texts/step，3 计时步中位，skip 2）
- 分段：forward 707.5 ms（33.2%）/ backward 1,380.5 ms（64.8%）/ clip+optimizer 20.3 ms（1.0%）；device compute = 2,029.5 ms（95.2%）
- **aten 归因（每步，175 条目仅取 aten:: 层——遵循 problem.md §5.2 PPU 双层重复计数警告）**：

| 类别 | ms/step | calls/step | %aten |
|---|---:|---:|---:|
| GEMM（aten::mm 675.3 + addmm 337.2）| **1,012.5** | mm **5,776** | **53.1%** |
| attention（efficient fwd 225.8 + bwd 347.8）| **573.5** | 480/960 | 30.1% |
| pointwise | 91.2 | 11,095 | 4.8% |
| copy/cast（aten::copy_ 72.5）| 72.5 | 9,820 | 3.8% |
| layer_norm | 39.9 | 1,960 | 2.1% |
| GELU | **32.6** | 960 | 1.7% |

- **aten::mm 5,776 次/step 与 problem.md 原始 PPU 归因逐次一致**；GEMM/attention/copy-cast/GELU 排序与 problem.md §4.3（GEMM 44.3% 主导、GELU 0.9× 非瓶颈）方向一致——原对象 native 基线在本机复现了原始归因。
- attention 走 `_efficient_attention` 后端的原因已由 `probe9b_sdpa_masked.json` 定位：**bool padding mask 下 flash "No available kernel"** → 回落 efficient（4.5× 慢于无掩码 flash）/math（9.5×）。

### 阻碍（逐项，含证据与影响）

| # | 阻碍 | 证据 | 影响 | 动作 |
|---|---|---|---|---|
| ① | 磁盘 df=100% | 实测仍可写（1.1 GB 瞬时成功；本轮落盘 3.4 GB）| 大规模数据/多 checkpoint | 必要下载已完成；扩容需平台 |
| ② | x-benchmark 容器不在本机 | `/home/wuyiqiang`、`/data/ppu-benchmark` 为另一训练机路径（zeros-7181 problem.md）| x-benchmark 整模型口径 | 平台提供容器/入口 |
| ③ | T²-Ranking 官方原始集未下载 | 已用 conf 同名镜像 triplet-15（schema 对齐文档）| 官方 parquet 逐字节同源性 | 后续按 FlagScale 流程重建 |
| ④ | FlagGems full-enable blocked | F-1/F-2 两次单次触发出证（见第六轮）| full-enable 原对象训练 | 隔离复验待设备状态（平台+后续）|
| ⑤ | FSDP2 多卡口径 | 本机单卡分配 | 多卡验收 | 多卡环境就绪后按 §5.5 单列通信（后续）|

### 未尝试（明确标注）

官方 T²-Ranking parquet 逐字节同源重建；真实 3-epoch 收敛训练（仅 3 计时步基线）；FSDP2 多卡；x-benchmark 容器运行；FlagGems full-enable 在原对象上的运行（blocked）。

### 定位声明

PR #8 的 AdamW/Graph/小模型内容按第六轮口径维持"候选观测/组件级"；本节为 problem29 原对象在 ZW810E 的**首轮 native 基线**——下一步按审阅排序在原对象上做优化前后对照（GEMM 真实 shape 调优、SDPA 掩码路径、copy/cast），full-enable 复验通过后再补 FlagGems 配置。

### 第九轮实施（复审 6895987107 执行顺序 1–3）

#### 1. 原对象 A/B 矩阵（3 独立进程 x 4 配置；**seed 42 固定**——四配置同一 batch 流；每进程 3 计时步中位，fwd/bwd/clip 为 micro-batch 段 x accum 4 = 每优化器步；**JSON 含 loss/grad-norm/非有限/峰值显存字段**）

| 配置 | wall 中位（**round-16 种子化批次**） | 相对 native/stock | clip+optimizer/步 |
|---|---:|---:|---:|
| native/stock | 2,136.835 ms | 基线 | 20.523 |
| native/fused | 2,136.163 ms | **-0.0314%**（噪声内） | 13.940 |
| selective(gelu)/stock | 2,172.336 ms | **+1.6614%**（回归） | 20.396 |
| selective(gelu)/fused | 2,165.293 ms | +1.3318% | 13.937 |

历史批次（round-9，4 loss + 1 grad-norm 口径，**已 superseded，仅存档**）：

| 配置 | wall 中位 | 相对 |
|---|---:|---:|
| native/stock | 2,141.4 ms | 基线 |
| native/fused | 2,124.3 ms | -0.80% |
| selective(gelu)/stock | 2,167.8 ms | +1.23% |
| selective(gelu)/fused | 2,170.9 ms | +1.38% |

结论（对应审阅执行顺序 1、6）：
- fused AdamW：optimizer 段省 ~6.6 ms/步（20.5→13.9）；整步 **-0.0314%**（round-16 种子化批次，20 loss + 3 grad-norm 口径；与历史批次 -0.80% 均在噪声/波动带内，不能声称可靠整步加速）；selective(gelu) 回归 **+1.6614%**。全部配置无非有限值、峰值显存 24.3 GiB。
- **峰值显存 24.3 GiB（JSON peak_mem_bytes = 26,092,513,280 bytes）**——显存列留空并以运行日志为准；列入工具改进。
- 四配置 loss 末步中位 0.7208-0.7263（数值路径差异所致，方向性一致）。**seed 42 的意图是控制数据流/模型初始化/dropout；现有 JSON 未记录逐步输入指纹，不能据其证明跨进程输入或 loss 序列实测一致**——实测一致性与固定 seed 意图分开表述，前者需输入指纹证据（列入后续批次）。


#### 1b. AB r19/r20 批次（strict validator + manifest 溯源后全量重跑）

| 配置 | wall 中位（r19） | 相对 | clip+optimizer |
|---|---:|---:|---:|
| native/stock | 2,130.629 ms | 基线 | 20.257 |
| native/fused | 2,136.082 ms | **+0.2560%**（噪声内） | 13.896 |
| selective(gelu)/stock | 2,169.832 ms | **+1.8400%**（回归） | 20.574 |
| selective(gelu)/fused | 2,164.760 ms | +1.6020% | 13.967 |

**r20 批次**（strict validator + manifest 溯源，全量重跑，`results/orig_ab_reps_r20/`，manifest 绑定 script SHA + wall_ms + 显式 env）：

| 配置 | wall 中位 | 相对 | clip+optimizer (clip_step) |
|---|---:|---:|---:|
| native/stock | 2,138.114 ms | 基线 | **20.26** |
| native/fused | 2,159.890 ms | **+1.0184%** | 15.041 |
| selective(gelu)/stock | 2,173.268 ms | **+1.6441%** | 20.370 |
| selective(gelu)/fused | 2,173.903 ms | +1.6738% | 14.238 |

本轮 fused 局部省 ~5.2 ms（20.26→15.04），整步中位反而 +21.8 ms——跨批次方向不稳定再次确认。

**样本与清单说明**（详见 `results/orig_ab_reps_r20/BATCH_NOTE.md`）：12 份 r20 JSON 中 11 份与 2bce556 字节相同；`native_stock_r1`（wall 2135.54→2133.62）系 21:27 一次未登记孤儿补跑替换（runner 干启动验证产生，driver 被终止后子进程存活完成，退出码未捕获、总日志无完成记录——如实注明）；组内 wall 中位 2138.11 不受影响。`manifest.jsonl` 属 r20 批次（v1 格式，无 env/wall/hash 字段）；r19 批次早于 manifest 机制无清单（溯源=round-19 blob）。v2 runner 记录 config_sha256/result_sha256/env，skip 经 `check_reuse.py` 全字段核验（CPU 反例 5/5：改任一 loss→拒、改配置→拒、wall 漂移→拒、exit≠0→拒）。

#### 1c. problem29 平头哥阶段结论（如实）

**截至本轮，在已提交的真实 BGE-M3 端到端测试中，尚未取得可靠、可复现的整步性能提升。** fused AdamW 相对 native/stock 的整步变化跨批次为 -0.80% → -0.03% → +0.26%（r19）→ **+1.02%（r20）**（方向不稳定且各批有已注明的测量/来源边界）；selective(gelu) 各批均回退约 +1.2%–1.8%。局部 optimizer 省时（~6.4 ms/步 ≈ 整步 0.30%）、Graph 组件微基准与瓶颈归因属有效探索，尚未转化为整步收益。基线为 native/stock；full-enable 受 F-1/F-2 阻断，不能宣称相对原始 full-enable FlagGems 的加速验收；无 trace/设备计数器依据，不认定硬件已到极限。下一优化选择以自然训练 trace 关键路径证据为准（见 round-20 计划）。

#### 2. GEMM 消融（真实 shape M2990 K1024 N1024 bf16，probe10）

**批次口径**：下表为 probe10 **历史批次**（round-9/19，含乘积错配时期的布局数字——已被下方 r20 同乘积重测取代）；r20 当前有效批次见本节末尾"r20 重测"表。

| 消融项 | 结果 |
|---|---|
| allow_bf16_reduced_precision_reduction True/False | 0.0567 vs 0.0564 ms（110.6/111.2 TFLOP/s）——**无收益** |
| 输入非连续（.t().contiguous().t() 布局） | 0.0611 vs 0.0566 ms（+8%）——**contiguous 更优** |
| fp32 matmul | 0.2256 ms（4× 慢）——**维持 bf16** |
| 结论 | bf16 GEMM 在该 shape 已 110 TFLOP/s（probe9 一致），该 shape 下已测开关/布局（reduced-precision-reduction、contiguous）未发现收益；GEMM 差距属厂商 kernel/硬件层（运行时层移交项），**其他形状/实现的应用层空间待验证** |

#### 3. masked attention：等长分桶候选量化（probe11，真实 shape B8 H16 S512 D64 bf16）

| 路径 | ms |
|---|---:|
| 带 bool padding mask（现路径，efficient 后端） | **0.5866** |
| 等长无 mask（flash 后端） | **0.1313**（4.46×）|

**适用边界（round-21 撤回外推）**：4.46× 来自 B8/S512、无 dropout、**仅前向**的合成探针，不能外推到原 BGE 训练（B10、动态长度、带 mask、含反向——原对象 attention ~573.5 ms 中约 ~348 ms 在反向）。**撤回"预测整步 -20% wall"的表述**（前向倍率不覆盖反向路径）；**撤回"梯度正确性由构造保证"**（反向未测，需输出+dQ/dK/dV 对照）。等长分桶后只要 batch 内仍有 padding，**仍须保留 mask**；"无 mask 即等长"仅当分桶真正做到批内等长才成立，属训练侧改造决策。**未实施**，登记为候选方向（训练侧 + 运行时层），收益评估需含反向的整层对照。

#### 3b. 真实线性层试点（issue #7 探索方案 7011219007 第 1/2 项，round-25 修订）

**捕获**（`capture_linear_layers.py` r25 版：profiler 步前恢复原 forward + **断言 wrapper 计数在 profiler 步不变**；逐捕获步独立保存 calls/fwd_ms/形状集；`linear_capture_r25.json`，3 捕获步验证 calls=[20,20,20]）：

| 类别 | 模块数 | 权重形状 | 输入形状 | 调用/步 | fwd ms/步（逐步）|
|---|---:|---|---|---|---:|
| attention QKV+output | 96 | [1024,1024] | [10,S,1024] S=12–512 变长 | 20 | 135.1/144.7/159.4（均 146.4）|
| FFN intermediate | 24 | [4096,1024] | 同上 | 20 | 101.1/112.9/126.5（均 113.5）|
| FFN output | 24 | [1024,4096] | [10,S,4096] | 20 | 98.7/108.2/119.5（均 108.8）|
| pooler | 1 | [1024,1024] | [10,1024] | 20 | 0.4 |

线性层前向合计 ≈ **369 ms/步**（逐步 335/366/406）。r23（有缺陷口径）与 r24（两步）均被本表取代。每优化器步 4 micro × 5 组 → 每层 20 次；bias 全 True。**均值口径澄清（复审 06:12）**：表中逐步类均值为该类模块的均值；跨 145 模块的步均值为 **2.55ms**（r26 日志中 4.856ms 为最慢模块的步均值）。

**有效 token 统计（round-28 修正口径：attention_mask 二值掩码统计。撤回 round-26"组内零 padding"结论——系方法伪影：先 padding=True 再 sum(len) 使分子恒等于补齐后总位置数，任何批次都会得到 0%）**：

| 捕获步 | 有效 token（mask.sum） | padded 位置（mask.numel） | padding 比 | 线性 fwd ms |
|---|---:|---:|---:|---:|
| step 0 | 27,052 | 54,160 | **50.0%** | 335 |
| step 1 | 31,454 | 61,430 | 48.8% | 366 |
| step 2 | 35,506 | 69,810 | 49.1% | 406 |

**padding 集中在负例组**（capture_step_0 micro1 示例：negative_1 49.6%、negative_3 50.9%、negative_2 36.6%、anchor 30.6%、positive 仅 7.1%；部分 micro 的 negative 组达 57–68%）。**含义（修正）**：负例组内长度方差大、padding 浪费显著（整步约一半计算位置是 padding），按相似长度分桶/合并负例组的候选**仍有实际空间**（探索方案第 4 项维持）；attention mask 必要性需按 `prepare_embedding_inputs` 的 mask 传入单独核实（设备待办）。步间耗时与 padded 位置总量正相关（1.09/1.11 vs 1.13/1.14）✓。

**注册命中计数**（1 profiled step）：`aten::mm` **5776**、`aten::addmm` **2900**、`aten::linear` 4840、`aten::matmul` **16**——训练热点是 mm/addmm；候选注册 `aten::matmul` 无法命中训练路径（复审判断证实）。

**GEMM 热点表（round-29 按模块类修正，`linear_hotspots_r29.json`；`build_hotspot_table.py` CPU 重建）**

round-28 表以 input shape 单键聚合，把 QKV 与 FFN 升维（同输入 [10,512,1024]、不同权重）合并错归——**作废**。修正后按模块类 × 形状 × stride × dtype 分组（权重形状入键），对账断言通过：per-class calls 和 8700 == 模块 calls 和（2900/步 × 3 步）✓；ms 和 368.26 ≈ 模块 368.34（逐项舍入差 <0.1%）✓。维度按 Linear 语义推导（M=B×S 展平）；时间为 Linear 前向 event 累计。

Top 类（S=512 主导行，按 fwd ms/步）：

| 模块类 | 前向 (M,K,N) | 调用/步 | ms/步 | 模块数 |
|---|---|---:|---:|---:|
| FFN 升维（intermediate）| (5120,1024,4096) | 120 | 43.875 | 24 |
| FFN 降维（output）| (5120,4096,1024) | 120 | 41.007 | 24 |
| Attention QKV | (5120,1024,1024) | 360 | 38.934 | 72 |
| Attention 输出投影 | (5120,1024,1024) | 120 | 11.593 | 24 |

S=512 为负例组截断上限（max_length=512 truncation），主导线性前向。**dtype 区分**：wrapper 记录的输入 dtype 为 autocast 前的 fp32；GEMM 操作数 dtype 为 autocast 后 bf16（权重原生 bf16）。**dX/dW 为派生形状**（dX=(M,N)×(N,K)；dW=(N,M)×(M,K)）；实际 `mm/addmm` 操作数形状/转置/stride/dtype 直接 dump 列入设备待办。此表用于选择厂商 GEMM 对照形状（issue #7 路线 1）。

**操作数阶段证据（round-31 修正，`gemm_operands_r31.json`，`dump_gemm_operands.py`；round-30 的三项采集问题已修复并设备复测）**：

1. **匹配修复 + known-input 校验**（复审 P1-1）：TorchDispatchMode 匹配改用 `torch.ops.aten.mm.default`/`aten.addmm.default`（旧 `'aten::mm'` 字符串从未匹配点号格式——**round-30 的零计数系匹配伪影，撤回"XMLIR dispatch-key 拦截"结论**）。已知输入探针在 PPU 实测：mode 收到 `aten.mm.default`×1、`aten.addmm.default`×1 及张量 shape/stride/dtype 元数据 ✓
2. **真实路径操作数直接测量**（60 条采样，`real_path_operand_sample`）：**全部 bf16**；每条 addmm 模式 = bias[N]（1 维）+ input [M,K] 行主序连续（stride [K,1]）+ **weight.t() 列主序 stride [1,K]（转置视图，实测）** ✓——不再依赖语义推导
3. **连续完整步窗口**（复审 P2-3）：单一 profiler 窗口采完 4 个 micro，`record_function` 标记分段；全程实测 **mm 5776 + addmm 2900**（round-30 的 ×4 外推标签移除）；分段一致：每 micro 前向 addmm 725 + mm 2（loss 计算）、反向 mm 1442；**每 micro 间反复启停导致丢事件的旧实现作废**；反向与 2×2900=5800 差 32 的逐模块归因仍待查（诚实保留）
4. raw events 的 input_shapes 已存（`raw_shape_samples`：如 FFN 升维 addmm [4096],[190,1024],[1024,4096]）

**单层回放**（`replay_linear_r25.json`；**边界：代表性形状合成回放**（随机 x/w/gy、固定 S），非自然训练张量 replay；bf16 固定非 autocast；溯源：FlagOS-OP pr35 commit `4df994d`、triton_level/_shape sha256、dtype/stride/layout 入 JSON；vendor 判定经本地 API shim 返回真实值 thead）。

**统一计时口径（round-25 修正）**：native = 前向单独计时 + `autograd.grad(dX,dW,db)` 单独计时（图在计时外预建，**计时内无重复前向**）；candidate = (matmul + bias add) + (backward impl + gy.sum db)——两侧合计均含 bias 梯度：

| S | native fwd+bwd ms | cand fwd+bwd ms | 整层比 | y/dX/dW 违例 |
|---:|---:|---:|---:|---|
| 512 | 0.358+0.881=1.239 | 5.425+4.588=10.014 | **8.08× 慢**（fwd 15.1×、bwd 5.2×）| 105,393 / 0 / 154,183 |
| 256 | 0.196+0.453=0.648 | 2.709+2.422=5.131 | 7.92× 慢 | 52,784 / 0 / 123,161 |
| 128 | 0.095+0.239=0.334 | 1.369+1.325=2.694 | 8.06× 慢 | 25,040 / 0 / 153,521 |

旧口径倍率（6.44/6.35/6.56×，native 计时含重复前向、两侧 bias 口径不一致）**作废，以本表为准**。

**正确性**（声明容差：每位置 ≤4·ulp_bf16(max(|ref|,floor))、floor=0.01·max|ref|）：**dX 0 违例 ✓；y ~0.5%、dW ~3.7% 违例**——候选在 ZW810E 不满足该层 bf16 容差（与候选文档"其他厂商须分别验证"一致）。bias 梯度 ~0.026ms、布局预转换 ~0.043ms——非主要差异。

**结论（保留负结果）**：
- **已确认**：候选整层 ~8× 慢且 y/dW 超 bf16 容差；`aten::matmul` 注册命中不了训练热点
- **估算（非实测）**：按代表性回放比例外推替换 FFN intermediate 类为整步 +数百 ms——**代表性回放外推，非自然训练整步实测**；整步 AB 未做（单层决定性）
- **归因边界**：差异主要在前向（~15×）次在反向（~5×）；额外拷贝可忽略。"32×32×32 tile 未对 PPU 调优"为**待厂商/设备计数器确认的假设**——当前证据只支持"候选更慢且精度不满足"
- 方向：厂商 GEMM 库调优或 PPU 专属 kernel 重调（需设备计数器/编译报告支持）；SDPA 维持 native efficient 基线

#### 3c. 自然训练设备 trace（round-32 初采；**r32/r33 分析已被 r36 验收窗口取代，标记 superseded**，仅存档）

**采集**（`capture_training_trace.py`：原入口 native/stock、bf16、batch 10、accum 4、mask/dropout 保留；seed 42；预热 2 步 → 单一连续 profiler 窗口 3 步（`record_function` 标记 rf_s{step}_{phase}_m{micro}）→ `export_chrome_trace` gzip；无 profiler 复跑同窗口对照）：

- trace：`training_trace_r32_trace.json.gz`（25.6MB，131,493 kernel 事件）；meta `training_trace_r32_meta.json`（命令/开销/计时）

- **归因 100%**：ac2g flow id==kernel correlation；flow 时间戳为设备执行时刻，反向 kernel 异步排空在 marker 窗口外——按最近前置 marker（提交相位）归因，跨步排空单列

**相位清单（round-33 修正：每步归一 + fwd/bwd 分列，`training_trace_analysis_r33.json`；round-32 表的"每步"实际是 3 步累计且相位被 m0-m3 合并——作废）**：

| 类别/相位 | calls/步 | device ms/步 |
|---|---:|---:|
| GEMM backward | 5,768 | **772.1** |
| SDPA/attention backward | 1,440 | **375.5** |
| GEMM forward | 2,908 | **330.3** |
| SDPA/attention forward | 480 | **204.7** |
| elementwise backward | 17,514 | 125.9 |
| elementwise forward | 6,384 | 58.3 |
| reduction backward | 3,384 | 51.9 |
| gelu backward/forward | 480/480 | 21.3/13.8 |
| other clip（裁剪+优化器）| 256 | 17.6 |
| layernorm backward/forward | 980/980 | 16.1/15.0 |

（与复审复算一致：GEMM fwd 330.3 / bwd 772.1；attention fwd 204.7 / bwd 375.5 ✓）

**采集开销修正（round-33，`samewindow_noprofiler_r33.json`）**：round-32 的"无 profiler 复跑"在同一迭代器上继续读 loader（非同窗口）——作废。修正后用**独立模型 + 独立 loader（同 seed 42 同样本窗口）**复测步 2–4：**1895.6 / 2128.6 / 2364.3ms，中位 2128.6ms**；同窗口采集开销 = 2265.1/2128.6 − 1 ≈ **6.4%**（批长度摘要已存 JSON 供断言一致）。

**GEMM 同窗口分形状效率（round-33，`gemm_efficiency_r33.json`，`measure_gemm_efficiency.py`——同一步窗内每 mm/addmm 事件对实测；round-32 的 73% 为跨窗口估算（r28 形状 × r32 时间，单位为 TFLOP/步非 PFLOP），降级为估算值）**：

| 相位/类 | (M,K,N) | calls/步 | device ms/步 | TFLOP/s（同窗实测）|
|---|---|---:|---:|---:|
| fwd FFN 升维 addmm | (5120,1024,4096) | 120 | 34.28 | **120.3** |
| fwd FFN 降维 addmm | (5120,4096,1024) | 120 | 32.77 | **125.8** |
| fwd QKV/attn-out addmm | (5120,1024,1024) | 384 | 37.73 | **109.3** |
| bwd FFN-up dX | (5120,1024,4096) | 96 | 34.57 | **119.3** |
| bwd FFN-down dX | (5120,4096,1024) | 96 | 31.90 | **129.3** |
| bwd FFN-up dW | (4096,5120,1024) | 96 | 46.69 | **88.3** |
| bwd FFN-down dW | (1024,5120,4096) | 96 | 46.67 | **88.3** |
| bwd QKV dX | (5120,1024,1024) | 384 | 37.78 | **109.1** |
| bwd QKV/attn-out dW | (1024,5120,1024) | 384 | 42.36 | **97.3** |

主热点（S=512 组）**88–129 TFLOP/s（63–93% 峰值 139.2；参照 135.7）**——**dW（转置 operand）明显慢于 fwd/dX（88 vs 119–129）**。短形状（S≈431–493 组）~86–88 TFLOP/s。

#### 3c-2. 最终汇总表（round-35，由 `results/final_table_r35.json` 程序化生成——**全部基于 r35 同窗 trace 与同窗 kernel 效率测量**，round-32 trace 复解析口径作废）

窗口：seed 42、batch 10、accum 4、native/stock bf16、3 步连续（`training_trace_r35_trace.json.gz`）；脚本 commit 见各文件 blob。

| 阶段 | 相位 | calls/步 | device ms/步 | FLOPs (TFLOP) | TFLOP/s | HBM | 来源 |
|---|---|---:|---:|---:|---:|---|---|
| FFN 升维 | fwd | 96 | 34.3 | 4.12 | **120.1** | 估算* | kernel_efficiency_r35 |
| FFN 降维 | fwd | 96 | 32.8 | 4.12 | **125.9** | 估算* | 同上 |
| QKV | fwd | 288 | 28.3 | 3.09 | **109.3** | 估算* | 同上 |
| attn 输出投影 | fwd | 96 | 9.3 | 1.03 | **111.3** | 估算* | 同上 |
| FFN 升维 | dX | 96 | 34.6 | 2.75 | **119.3** | 估算* | 同上 |
| FFN 降维 | dX | 96 | 31.9 | 2.75 | **129.3** | 估算* | 同上 |
| FFN 升/降维 | dW | 96+96 | 46.8/46.7 | 2.76 | **88.0/88.3** | 估算* | 同上 |
| QKV | dX/dW | 288/288 | 28.4/31.8 | 2.02 | **109.0/97.4** | 估算* | 同上（拓扑推断 1:3，计数断言）|
| attn 输出投影 | dX/dW | 96+96 | 9.5/10.6 | 0.85 | **109.1/97.3** | 估算* | 同上 |
| SDPA/attention | fwd | 480 | 204.7 | **0.105**（单步均值；r35 指纹逐步 0.085/0.103/0.129）| **0.52** | 估算* | sdpa_efficiency_r35（公式 4BH S²D，60 组逐行；round-35 的 1.55 系 3 步 FLOPs 对单步 ms——已修正）|
| SDPA/attention | bwd | 1,440 | 375.5 | **0.158**（0.127/0.154/0.193）| **0.42** | 估算* | 同上（6BH S²D；重算未知）|

小 S 形状（140–180）TFLOP/s 6–24（latency 主导）；中 S 随 M 上升。前向 FFN 全部低于 139.2 峰值（round-34 的 >峰值数字系单 M 聚合 bug，已修）。

**关键路径（r35 trace，含 memcpy/memset 的 busy union）**：busy 85.9/89.0/88.8%；窗口内 **HtoD 120 条 0.27ms、DtoD 60 条、DtoH 60 条、memset 7,842 条 1.99ms**——此前"H2D=0"系只查 kernel 类别所致，已更正。

**严格配对（round-36，text-hash 真实指纹）**：trace 与 no-profiler 独立模型+loader（seed 42），双侧保存 **loader_index + sha256 文本哈希（text_hash16）** 并强制比较（长度摘要仅辅助）：**逐 micro 断言 12/12 hash+token 一致、0 mismatch**；no-profiler 中位 **2112.0ms**、p95=样本最大 **2335.6ms**（3 样本，非稳定 p95）、有效 tokens/s **14415–15202**；**严格配对采集开销 = 2264.9/2112.0−1 ≈ 7.2%**。kernel-efficiency 测量窗亦通过指纹断言（step_2，20/20 hash 匹配——见 `kernel_efficiency_r36.json.fingerprint_verification`）。

**HBM 状态：BLOCKED**——Asight v2.1.0 在位；探测命令/失败输出/dmesg 根因（PCM 被 pg-exporter 占用）见 `logs/hbm_probe_r34.log`；复测命令已记录；计数器读回前 HBM 字节/带宽/利用率全部为估算，不列入已完成测量。

**估算标注**："dW 提升 ~35ms/步"、"S512 达 135.7 ~150–200ms/步"、"逐元素融合 100–200ms/步"均为估算（依据同窗实测差值），非实测结论。ac2g 时间戳统一口径为主机提交时刻。

#### 4. F-2 最小 shape 隔离（单次触发）#### 4. F-2 最小 shape 隔离（单次触发）

最小 shape（M8 K256 N512，fp32，enable 下 mm backward）单次触发**仍挂起**（300s 超时，控制组通过）——F-2 与 shape 无关的普遍挂起，继续 blocked（`probe7_recheck_2026-11-13.json` 口径维持；本轮未重复触发已挂起路径之外的组合）。

#### 4b. AdamW 稳定参考验收（round-9 执行顺序 3 前半；`adamw_stable_ref.py` / `adamw_stable_ref_acceptance.json`）

30 步受控更新（4096 参数、seed 21、同梯度序列），参考 = **fp64 闭式 AdamW（公式 A）**；判定口径 = 每位置 ordinal ULP（参考核对到 dtype）+ 尺度门控相对误差 + 非有限计数：

| dtype | 实现 | 非有限 | 尺度门控 max rel | 典型位置 median rel | 判定 |
|---|---|---:|---:|---:|---|
| fp32 | foreach | 0 | 16.57（近零位置）| 1.248 | 与 fused 一致 |
| fp32 | **fused** | 0 | 16.57 | 1.248 | **通过（同 foreach 口径）** |
| bf16 | foreach | 0 | 16.78 | 1.249 | 与 fused 一致 |
| bf16 | **fused** | 0 | 16.78 | 1.246 | **通过** |
| fp16 | foreach | **498（损坏）** | inf | 1.303 | **损坏**（同 stock）|
| fp16 | **fused** | **0** | 16.55 | 1.248 | **通过——fp16 唯一有界路径** |
| fp16 | stock | **522（损坏）** | — | — | 损坏（两次独立记录）|

- 尺度门控 max rel 16.6× 位于 |ref|≈1e-8 的权重衰减收缩位置（30 步独立舍入路径累积）；typical 位置 median rel ≈1.25；**fused 与 foreach 同界，fp16 下 fused 为唯一有界路径**（stock/foreach fp16 均损坏，两次独立记录 522/498 位置）。
- stock 已污染配置未作为 fp16 正确性参考（审阅口径执行）。
- **撤回（复审 6918783681 确认项 + round-12 复核）**："fused vs fp64/稳定参考验收在此通过"的表述**不成立**——v2/v3 脚本比较了不一致的输入（不同初值、参考用 fp32 梯度 vs 实现用 dtype 转换梯度），且无可执行判定；**round-12 重写（同 p0_dt/grads_dt 克隆、同值 fp64 参考、唯一判定函数、verdict=exit 同源、负例自测）后按声明的每位置容差判据重跑：30 步轨迹上所有实现（fp32/bf16 的 stock/foreach/fused 与 fp16 全部）均 FAIL**——fp32 fused worst p err 2.96e-08（近零位置超 4eps 门）、fp16 fused worst 1.52e-02。旧 JSON（v1–v3）superseded（比较输入不一致、不可用于验收）。
- 局限：单张量形状 4096；多卡/FSDP2 未覆盖。

#### 5. 措辞修正（复审 A 项）
#### 5. 措辞修正（复审 A 项）

- README 归因对照表措辞改为"**方向性复现**（GEMM/attention/GELU 排序一致；copy/cast 248.8 vs 72.5 待同口径核对——kernel 层 vs aten 层统计口径差异）"。
**r20 重测（round-21 修复后：全分支同乘积 x@w.t()，CPU 非方阵核对；旧 layout 结论作废）**：

| 分支 | p50 ms | 说明 |
|---|---:|---|
| weight transposed-view（基线） | 0.0615 | x @ w.t() |
| weight precontig | 0.0721 | **更慢 ~17%**——旧"precontig ~1.6% 收益"系乘积错配（wc.t() 实算 x@w），撤回 |
| input noncontig | 0.0832 | **~35% 罚**（真实 layout 效应） |
| fp32 reference | 0.2011 | 同对象同运算 |
| bf16 vs fp32 max reldiff | 0.27% | 实际 bf16 输出 vs fp32 参考（非自比较） |

- 复现文档保留**原始命令 + 本机覆盖项**（model_path/data.path 两处覆盖已显式标注于 conf/run/bge_m3_ppu.yaml 与清单 §一）。

### 第十二轮（AdamW 稳定参考判定器重写，`adamw_stable_ref_v2.py` / `adamw_stable_ref_v4.json`）

复审 6879678135 确认项 + 6918783681 的 AdamW 稳定参考问题在 round-12 判定器重写后**仍未通过**：修复三个输入缺陷（不同初值、参考梯度 dtype 不一致、无可执行判定）后，按声明的每位置容差判据（fp32: 8*eps32*max(|ref|,floor)；bf16/fp16: 4*ulp(max(|ref|,floor))；floor=0.01*max|ref|（verdict() 单一来源；JSON tolerances 字段同式））重跑 30 步：

| dtype/实现 | 首步 | 中步 | 末步 | 判定 |
|---|---|---|---|---|
| fp32 stock/foreach/fused | PASS | **FAIL**（worst p err 1.19e-02）| **FAIL**（1.74e-02）| FAIL |
| bf16 stock/foreach/fused | PASS | **FAIL**（m worst 4.44e-04）| FAIL | FAIL |
| fp16 stock/foreach | FAIL（非有限 498/522） | FAIL | FAIL | FAIL |
| fp16 **fused** | PASS | **FAIL**（worst p err 1.52e-02） | FAIL | FAIL |

**无一实现通过 30 步稳定参考验收**——fp32 中/末步 ALL 实现一致地 FAIL（近零位置超 8*eps32*floor 门，属 fp32 累积舍入到达门限的诚实结果）；fp16 stock/foreach 非有限损坏（真实）；fp16 fused 中步 worst p err 1.52e-02（真实 fp16 优化器漂移）。**有效观测**：三实现相互一致（fp32 逐位；bf16/fp16 差异 ≤ dtype 量化）；fused 不劣于 stock/foreach。**AdamW 稳定参考验收未完成**——撤回此前"已完成/通过"表述；后续需 fp32 master states 或 steps-scaled 容差变体，属后续工作。
