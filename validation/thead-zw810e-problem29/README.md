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

#### 3c-2. 最终汇总表（round-36，由 `final_table_r36.json` 经 `build_final_table.py` + `render_readme_3c2.py` 程序化生成）

窗口：`training_trace_r36_trace.json.gz`；{'batch_rows': 10, 'accum': 4, 'dtype': 'bf16', 'env': {'FL_FLAGGEMS': '', 'FL_ADAMW_FUSED': '0'}, 'seed': 42}；脚本/数据 commit 见各文件 blob。r32/r35 trace 及其分析**标记 superseded**（仅存档）。

| 阶段 | 相位 | MKN | calls/步 | device ms/步 | FLOPs (TFLOP) | TFLOP/s | HBM | 来源 |
|---|---|---|---:|---:|---:|---:|---|---|
| GEMM ffn_up dW bwd | — | [[4096, 5120], [5120, 1024]] | 96 | 46.8 | 4.123 | **88.1** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_down dW bwd | — | [[1024, 5120], [5120, 4096]] | 96 | 46.71 | 4.123 | **88.3** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_down dX bwd | — | [[5120, 1024], [1024, 4096]] | 96 | 34.56 | 4.123 | **119.3** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_up fwd | — | [5120, 1024, 4096] | 96 | 34.28 | 4.123 | **120.3** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_down fwd | — | [5120, 4096, 1024] | 96 | 32.75 | 4.123 | **125.9** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_up dX bwd | — | [[5120, 4096], [4096, 1024]] | 96 | 31.9 | 4.123 | **129.3** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM attn_qkv dW bwd | — | [[1024, 5120], [5120, 1024]] | 288 | 31.77 | 3.092 | **97.3** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json (topology-inferred 1:3) |
| GEMM attn_qkv dX bwd | — | [[5120, 1024], [1024, 1024]] | 288 | 28.37 | 3.092 | **109.0** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json (topology-inferred 1:3) |
| GEMM attn_qkv fwd | — | [5120, 1024, 1024] | 288 | 28.36 | 3.092 | **109.0** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM attn_out dW bwd | — | [[1024, 5120], [5120, 1024]] | 96 | 10.6 | 1.031 | **97.3** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json (topology-inferred 1:3) |
| GEMM ffn_up dW bwd | — | [[4096, 4430], [4430, 1024]] | 24 | 10.32 | 0.892 | **86.4** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_down dW bwd | — | [[1024, 4430], [4430, 4096]] | 24 | 10.25 | 0.892 | **87.0** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM attn_proj dX bwd | — | [[4430, 1024], [1024, 1024]] | 96 | 10.17 | 0.892 | **87.7** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM attn_out fwd | — | [5120, 1024, 1024] | 96 | 9.26 | 1.031 | **111.3** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_up fwd | — | [4430, 1024, 4096] | 24 | 7.25 | 0.892 | **123.0** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_down fwd | — | [4430, 4096, 1024] | 24 | 7.25 | 0.892 | **123.1** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_up fwd | — | [4310, 1024, 4096] | 24 | 7.18 | 0.868 | **120.9** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_down fwd | — | [4310, 4096, 1024] | 24 | 6.84 | 0.868 | **126.9** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM attn_qkv fwd | — | [160, 1024, 1024] | 144 | 6.65 | 0.048 | **7.3** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_up fwd | — | [3740, 1024, 4096] | 24 | 6.46 | 0.753 | **116.6** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM ffn_down fwd | — | [3740, 4096, 1024] | 24 | 6.25 | 0.753 | **120.6** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM attn_qkv fwd | — | [4430, 1024, 1024] | 72 | 6.16 | 0.669 | **108.5** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM attn_qkv fwd | — | [4310, 1024, 1024] | 72 | 5.89 | 0.651 | **110.5** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| GEMM attn_qkv fwd | — | [1740, 1024, 1024] | 72 | 5.87 | 0.263 | **44.8** | ESTIMATE (PCM blocked) | kernel_efficiency_r36.json |
| SDPA/attention | fwd | — | 480 | 204.64 | 2.531 | **12.37** | ESTIMATE | sdpa_efficiency_r36（4BH S²D，60 组逐行）|
| SDPA/attention | bwd | — | 1,440 | 375.44 | 5.061 | **13.48** | ESTIMATE | 同上（6BH S²D；重算未知）|

小 S 形状（140–180）TFLOP/s 6–24（latency 主导）。**全部效率行 ≤139.2 峰值（离线校验通过）**。

**关键路径（r36 trace，busy union 含 kernels+gpu_memcpy+gpu_memset）**：busy **87.9/89.9/91.5%**、暴露空闲 **249.5/229.7/211.2ms/步**；
per-step memcpy（首步）：HtoD 40 条 0.08ms; DtoH 20 条 0.01ms; DtoD 20 条 0.07ms; memset 2544 条 0.63ms（逐步拆分 + 守恒断言 8082==8082 通过；此前"H2D=0"系只查 kernel 类别——已更正）。

**严格配对（text-hash 真实指纹）**：双侧 loader_index + sha256 text_hash16；逐 micro 断言 **12/12 hash+token 一致、0 mismatch**；no-profiler 中位 **2123.3ms**、p95=样本最大 **2350.5ms**（3 样本，非稳定 p95）、有效 tokens/s **14283.0–15106.0**；**严格配对采集开销 ≈ 6.7%**（profiled 2264.9ms）。kernel-efficiency 窗指纹断言 step_2 20/20 匹配。

**HBM：BLOCKED**——Asight v2.1.0 在位；PCM 被 pg-exporter 占用（dmesg 证据 `logs/hbm_probe_r34.log`）；复测命令已记录；**HBM 字节/带宽/利用率均为估算，不列入已完成测量**。SDPA dropout 配置 0.1（模块级激活）、SDPA 内部 dropout 标志未知、反向重算未知。

**估算标注**：dW 提升/135.7 参照/逐元素融合等收益数字全部为估算（依据同窗实测差值），非实测结论。ac2g 时间戳=主机提交时刻（统一口径）。
#### 3c-4. 命中数证实 + 中序列 dW + 配对整步（round-38b）——**历史记录，整步数字与"方差带内"表述已 superseded，当前口径见 3c-6/3c-8**

**命中数诊断**（`candidate_hit_counts_r38.json`，dispatch probe 固定匹配）：候选（FlagGems mm selective）**替换反向全部 mm 5,768 次/步**——dW_ffn 960、dX_ffn 480、dW_attention_proj 1,920、dX_attention_proj 1,920、other 488；**前向 addmm 2,908 未替换**（候选只注册 aten.mm）。

**中序列 dW 对照**（`ffn_dw_midseq_r38.json`，top-10 形状覆盖 87.6/156.5ms 中序列 dW 时间）：全部形状 FlagGems mm 更快，逐步节省 **0.94–2.71ms/形状**（top10 合计 **~18.1ms**；+S512 15.6ms → dW 候选总潜力 **~33.7ms/步 ≈ 1.6%**）。

**SDPA mask 修正后对照**（`sdpa_training_compare_r38.json`）：mask 语义修正（True=参与，padded=False）；**padded-V 独立性 0.0**（有效区域输出不受 padding V 影响 ✓）；native p=0 fwd 0.70/fwd+bwd **1.21ms**（graph 外计时修正，旧 1.93 含图内重建）、p=0.1 时 1.38ms（**dropout 执行的间接证据**；实际 p UNKNOWN）；FlagGems SDPA 编译失败（完整异常/签名/版本已存 JSON）。

**整步配对（方差如实）**：native 两 pass 中位 2112.0/2123.3（差 11.3ms）；mm 候选中位 2117.1 落在方差带内——dW 局部 15.6ms（+中序列 18.1ms）**未复现为可分辨的整步收益**；候选不采纳。**[superseded："分发开销抵消"为当时的推测，复审明确不能据此宣称；"库级接入"亦未验证，均不再是当前结论]**

#### 3c-5. 三组交错对照（round-39）——**历史记录：其中 dW-only 零命中 bug 导致的"0ms 保留"结论已撤回；2117ms 候选测量已 RETIRED；当前口径见 3c-6/3c-8**

真实入口内三组（各自独立进程、同 seed 同窗、3 轮交错记录）：

| 组 | 中位 ms | rounds ms | loss(首 micro) | 结论 |
|---|---:|---|---|---|
| native | 2,152.1 | 1900.9/2152.1/2359.0 | 0.9434/0.4700/0.6885 | 基线 |
| wrapped-stock（bmm 逃逸）| 2,140.1 | 1909.9/2140.1/2366.6 | 0.9490/0.4723/0.6942 | **包装本身 ≈ 无开销**（−12ms 在方差内；bmm/mm 内在差含入）|
| wrapped-FlagGems mm | **19,600.7** | 1955.0/21415.6/19600.7 | 0.9446/0.4788/0.6818 | **灾难性**：Triton JIT/autotune 风暴 + 逐调用开销 |

**[superseded——该 dW-only 组实为零命中 bug，结论撤回]** **对复审问题的回答**：dW 局部 33.69ms 潜力经 aten-override 接入 **0ms 保留到整步**——FlagGems Triton mm 路径在训练规模不可用（新形状 autotune 风暴 + 逐调用开销）。**call 级增益（103–110 vs 88 TFLOP/s）仍成立**，但兑现需要库/厂商级 dW kernel 接入（无逐调用分发），已按形状目标（4096×5120×1024 类）列入后续移交。

**先前 2117ms"候选"测量 RETIRED**：同 only_enable 调用跨运行行为不一致（一次近 native 空转、一次触发 Triton 路径），根因未定——该结果不可靠，不再引用。wrapped-stock 对照初次实现 RecursionError（override 内重入 aten.mm），改 bmm 逃逸（bmm/mm 内在差已计入并注明）。

**命中分类修正**（`candidate_hit_counts_r38.json`）：前向 = addmm（未替换）+ loss mm（候选替换，8 次）；反向 FFN dX 两类合计 **960** ✓（此前 480 归入 other）；dW_ffn 960；attn_proj dX/dW 各 1,920。

#### 3c-6. 四组整步对照（round-42 修正口径；早期"精度通过/保留 80% 局部收益/唯一可用接入路径"的表述**已标记 superseded**，见 3c-8）

**当前可证结论**：
- **整步**：dW-only 相对 native 的三步配对差为 −6.8/+7.3/+1.8 ms（中位 +1.8，`wholestep_perf_r42.json`）；各步负载不同导致轮间 sd≈225ms，故仅配对差有效——**当前三步窗口未分辨出稳定整步收益**，不能据此宣称收益已被某项开销抵消，也不能宣称有收益
- **单算子预算与整步差值同口径**：预算按本三步窗口实际命中加权约 **25.09 ms/步**（复审口径）；旧 step-2 的 33.69 ms 与三步中位数不可直接相除——**已停止该比较**
- **精度**：dW-only 同状态梯度判定 **FAIL**（阈值 2%，最差 12.1%）；但 3c-8 证明该偏差处于 **native 自身非确定性基线内**（native-vs-native 同状态最大 23–43%），且冻结 dW 输出 stock 与 FlagGems **逐位一致**——FAIL 归因于设备非确定性而非候选 kernel；**不作为部署候选**（结论不变，归因修正）
- fg_all 整步较 native 慢约 75 ms（r42）；首跑显著变慢（冷缓存 warmup0 22.2 s vs 热 6.8 s），该差额混有编译/加载/执行/同步，**不再称为"冷编译时间"**（见 3c-9-C）

#### 3c-7. SDPA 训练态重测（round-41，`sdpa_training_compare_r41.json`，复审 5b8fb29 项 5）

- **mask 语义（S 轴）**：v2[:,:,-32:,:] 扰动被屏蔽 key 位置；**with-mask 有效区域 diff 0.0**、**无 mask 对照 6.75**（扰动真实存在）✓
- **计时（统一 clone/建图/sync 边界）**：p=0 fwd **0.7018** / bwd **1.1251** / 完整 **1.8723** ms；p=0.1 fwd 0.7840 / bwd 1.2967 / 完整 2.1289（dropout 活动证据 +0.17ms bwd）；dropout 配置 0.1（模块级）、随机性=seeded torch generator
- **native vs FP32**（同输入同上游梯度、dropout=0）：out/dQ/dK/dV reldiff **0.35–0.48%**（bf16 级）✓
- **候选经模型入口（only_enable→F.sdpa）**：**NO-OP PASSTHROUGH VERIFIED**——profiler 证实执行 kernel 为 **fmha_cutlassF_bf16_aligned_k64x128x64_sm80（AttentionKernelFA2，与 native 相同）**；该 FlagGems 构建在 PPU 上未提供经 aten 入口的替代 SDPA kernel
- **直接调用候选（flag_gems.scaled_dot_product_attention）**：编译失败（GROUP_HEAD 断言，BLOCK_N≤HEAD_DIM tile 约束）——最小复现/实现版本（FlagGems-base @ 29e14822）/受阻形状归档
- **结论**：本栈无可用 SDPA 候选替代（入口透传 + 直调编译失败）；native（cutlassFA2 kernel）为唯一可用路径

#### 3c-8. 精度闭环 + 冷/热重放 + SDPA profiler 证据（round-42，复审 fe65d8f）

**精度闭环**（`acc_closure_r42.json`；`accuracy_closure.py`，三组同状态对照）：
- **同状态前提全部成立**：init 参数哈希 `74fb309b`、CPU RNG `59294f27`、CUDA RNG `aefdce03`、4 个 micro 输入哈希——三组（native/wrapped-stock/dW-only）**完全一致**，排除 RNG/输入/初始化差异
- **native-vs-native（两次独立运行）非确定性基线**：ffn_up max 23.2%/median 3.7%；ffn_down max 33.8%/median 11.1%；attn_output max **43.1%**/median 15.8%
- **dW-only vs native**：ffn_up 14.9%/5.6%；ffn_down 67.6%/9.5%；attn_output 43.1%/14.8%——**落在 native 自身非确定性基线量级内**
- **wrapped-stock（mm.out，同一 stock kernel）vs native**：14.9%/67.6%/29.9%——**仅改用 mm.out 即产生同量级偏差**，证明偏差来自执行路径非确定性而非候选 kernel
- **[superseded]** ~~冻结 dW 对照：stock 与 FlagGems dW 输出逐位一致（36 形状）~~——**已撤回**：该轮只比较了前 **4** 项，且以四舍五入后的相对误差 0.0 推出"36 形状逐位一致"；正确的 12 形状原始位模式对照见 **3c-9-A**
- **[superseded，见 3c-9]** 上一轮"12.1% FAIL 已排除候选影响"的归因**不再作为结论**：该轮精度脚本用了比性能组更宽的维度条件分发，attention 权重梯度也进入了候选，两组选中的调用不同；且冻结 dW 只覆盖前 4 项、用四舍五入后的相对误差 0.0 声称"36 形状逐位一致"——**该表述已撤回**。当前可证结论以 3c-9 为准

**冷/热重放**（`coldwarm_cold_r42.json` / `coldwarm_warm_r42.json`，固定 seed 42 init/批次窗口/RNG，Triton cache 目录隔离）：

| | COLD（空缓存）| WARM（预填充缓存）|
|---|---:|---:|
| warmup0 | **21,901 ms** | 5,900 ms |
| timed 0/1/2 | 2500/3661/2939 | 2511/2712/2936 |
| 首次调用总耗时（编译+加载+执行+同步，不可分）| **20,081 ms** | 1,578 ms |
| 缓存新增条目（miss 代理）| 31 | **0** |
| 单形状首次调用 median/max | 1.43 / **1,472** ms | 1.42 / 934 ms |
| 稳态 ms/调用（含每次 sync）| 0.1685 | 0.1684 |

- **[已撤回]** 此前"约 20 秒慢步 = 冷编译伪影、发生在 warmup0、旧入口无独立 warmup"的表述**全部撤回**：旧 `threeway_mm_compare.py` 入口**已有两个预热步**，旧的 ~20 s 出现在**预热之后的计时步**；本表只是冷/热缓存对**首跑时间**的实测差异（warmup0 22 s vs 5.9 s），**不能解释旧计时步**。旧慢步的统一口径见 3c-9-C：**未复现，确切原因未确定**
- **[已撤回]** 冷/热首次调用耗时之差（20,081−1,578 ms）混有编译、加载、执行与同步（诊断脚本每次 MM 都同步），**不称为编译时间**
- **每个计时步仍引入新形状**（303→369→435，逐步序列长度不同）。这是**已观察到的事实**，与 fg_all 的整步变慢**相关**，但**未通过对照实验证实为根因**（未做"固定形状 vs 动态形状"的单变量对照）——不写成因果结论
- 边界说明：`first_ms` 含编译+autotune+执行+同步，未设 TRITON_KERNEL_DUMP 故无法再细分；`later_ms` 为稳态执行+同步；compiler log 未捕获（已注明）

**SDPA profiler 证据**（`sdpa_training_compare_r42.json` + `sdpa_training_compare_r42_kernel_evidence.json`，脚本内置 profiler 自动生成 kernel 字段）：
- **模型入口 only_enable 后 = NO-OP PASSTHROUGH**：同 dropout(p=0) 下 kernel 集合与 native **完全一致**（`fmha_cutlassF_bf16_aligned_k64x128x64_sm80` AttentionKernelFA2）；**only_enable 返回 "No op to register"**（决定性证据——该 FlagGems 构建在 PPU 无 aten 级 SDPA kernel）
- 修正：`_dropout_sm80` 后缀 kernel 是 **native p=0.1 变体**（证实 dropout 执行），非候选——上轮以此判"engaged"为误判，已撤回
- **直调候选 = COMPILE FAILED（受阻）**：`tl.static_assert(BLOCK_N <= HEAD_DIM)`，tile 参数（GROUP_HEAD/HEAD_DIM/HEAD_DIM_ACTUAL/BLOCK_M/BLOCK_N）已记录；FlagGems SHA `29e14822`；受阻形状 S512_pad32 / S214_pad16；最小复现已存
- 真实变长 mask：S512(pad32) fwd 0.784ms、S214(pad16) 0.205ms（p=0.1，含 dropout 变体 kernel）
- S 轴扰动独立性 0.0 / 无 mask 对照 6.75；统一边界 fwd/bwd/full = 0.7018/1.1251/1.8723 ms(p0)；native vs FP32 0.35–0.48%

**HBM/计算单元利用率：BLOCKED**（`logs/hbm_probe_r42.log`）——acu 真实退出码 **101**（本轮直接捕获，解决此前 0/101 矛盾）；dmesg `PPU0000 PCM is in used by app[pg-exporter]`；L2 指标同受 PCM 锁阻塞；平台反馈渠道本环境不可达（已记录请求）。

#### 3c-9. round-43 修正：同路由精度归因 + 12 形状逐位对照 + 实际吞吐 + 原入口慢步重放 + SDPA 逐形状（复审 df851e0）

**A. 精度归因（与性能组同一 12 形状白名单，`acc43_analysis.json`）**

同状态前提全部成立（init `74fb309b`、CPU RNG `59294f27`、CUDA RNG `aefdce03`、4 个 micro 输入哈希）。逐形状命中：S512 对各 408 次、中序列 8 形状各 24 次，共 **1,056**（与性能组一致，attention 权重梯度**未**进入候选）。

| 对照（首 warmup 步目标梯度，floored rel）| ffn_up max / median | ffn_down max / median | attn_output max / median |
|---|---|---|---|
| A native vs native（两次独立运行）| 12.1% / 4.3% | **33.8%** / 9.8% | 40.3% / 15.3% |
| B wrapped-stock vs native | 14.9% / 3.4% | 31.3% / 8.5% | 43.1% / 12.7% |
| C dW-only vs native | 14.9% / 3.9% | **31.3%** / 11.8% | 29.1% / 15.7% |

- 上一轮"候选 67.57% vs 原生 33.78%"的差距**已消除**：改用同一白名单后，候选对原生的最大偏差（31.3%）**不高于**原生自比（33.8%），三组的中位数同量级
- **冻结 dW 逐位对照（12/12 形状，`acc43_analysis.json.frozen_dW_12shapes`）**：对真实反向捕获的同一对 gy/x，stock（`mm.out`）与候选（`flag_gems.mm`，设备 kernel 为 Triton `mm_kernel_general`，与 stock 的 `gemm_ktype0…` **确为不同 kernel**）的输出**按原始 int16 位模式比较：50,331,648 个元素中不等元素 0 个，未舍入最大绝对误差 0.0，非有限 0**；每形状保留 dtype、stride、输入哈希
- **判定边界（已被 A2 取代）**：上表（首预热步快照）中候选与原生自比的偏差同量级，当时无法据此判定候选。**A2 的同快照命中步验收已解决该问题**：在受控同状态下三组对照全部逐位一致，dW-only 在 2% 门限下 **PASS**。上表的 12–40% 偏差来自跨轨迹分叉（各运行从零训练的历史不同），不是步内非确定性

**A2. 候选实际命中步的精度验收（`acc_hit_step_r44.json`，复审 bqPbopQ 项 1）**

此前梯度快照取自首个预热步（白名单命中 0）。本轮改为：运行至 **step 2 之前**保存模型/优化器/批次/CPU+CUDA RNG 快照，分别恢复后运行 native、wrapped-stock、dW-only、native 自比（A/B 两阶段独立子进程，native 组不注册任何 override）。**本步 dW-only 命中 432 次**（候选确实执行）；目标命中为 0 则 exit 4。

| 对照（2% floored rel 门限）| FFN 升维 | FFN 降维 | attention 输出投影 | 非有限 |
|---|---|---|---|---|
| A native vs native | 0.0000 PASS | 0.0000 PASS | 0.0000 PASS | 0 |
| B wrapped-stock vs native | 0.0000 PASS | 0.0000 PASS | 0.0000 PASS | 0 |
| **C dW-only vs native** | **0.0000 PASS** | **0.0000 PASS** | **0.0000 PASS** | 0 |

更新后参数逐位相同（absdiff 0.0），四次运行 grad_norm 均为 10.5625。已独立核验各组张量为**不同对象**（data_ptr 不同）而内容逐位一致。

- **判定：在候选确实执行的步上，dW-only 通过 2% 梯度门限（PASS）**
- **重要更正**：从同一快照恢复后重跑同一步，**设备是确定性的**（native 自比逐位相同）。此前观察到的 native-vs-native 12–40% 偏差，来自各次运行**从零训练到该步的累积轨迹差异**，而非步内随机性——之前"设备非确定性"的表述据此收窄为"跨轨迹分叉"
- 结合 12 形状冻结 dW 逐位一致（A 节）：候选 kernel 在这些形状上的输出与 stock 逐位相同，故整条反向链亦逐位一致

**B. 有效吞吐与配对整步（`wholestep_perf_r43_paired.json`，两次独立重复、组序轮换）**

有效 token 从**实际消费的批次**经 tokenizer attention_mask 统计（不再使用存储常数）；各组输入哈希逐步一致。两步窗口 token 数 27,052 / 31,454 / 35,506。

| 重复（组序）| native ms | dW-only ms | dW−native 配对差 | dW−wrapped-stock 配对差 |
|---|---|---|---|---|
| rep1（native 先）| 1897.8/2132.8/2356.9 | 1909.8/2140.3/2349.4 | +12.0 / +7.5 / −7.5（中位 +7.5）| −9.3 / −10.3 / −32.6 |
| rep2（dW-only 先）| 1928.4/2135.6/2394.3 | 1893.9/2124.5/2337.6 | −34.5 / −11.1 / −56.7（中位 −34.5）| −40.2 / −25.7 / −37.6 |

native 有效 tokens/s：14,255 / 14,748 / 15,065。**dW-only 相对 native 的配对差随重复变号**（rep1 偏正、rep2 偏负，合并 6 个范围 −56.7～+12.0）→ **原生训练整步尚无稳定加速，也未分辨出稳定回退**；相对同包装 stock 的 6 个配对差全为负（−9.3～−40.2）→ 同包装下改善一致成立。

**C. 旧慢步原入口重放（`orig_replay_cold_r43.json` / `orig_replay_warm_r43.json`）**

用**原计时入口**（无逐 MM 事件/同步，仅整数首遇计数），固定 seed 42 init/批次窗口/RNG，Triton cache 目录隔离：

| 步 | COLD ms（本步新形状 / 缓存条目）| WARM ms（本步新形状 / 缓存条目）|
|---|---|---|
| warmup0 | 22,151（123 / 29）| 6,786（123 / 32）|
| warmup1 | 3,638（102 / 31）| 2,204（102 / 32）|
| timed0 | 1,944（78 / 31）| 2,096（78 / 32）|
| timed1 | **3,135**（66 / 32）| 2,229（66 / 32）|
| timed2 | 2,439（66 / 32）| 2,419（66 / 32）|

- **机制确认**：每个计时步都引入新的序列长度 → 新 mm 形状（435 个唯一形状，分布于 5 步：123/102/78/66/66）；冷缓存下 timed1 3,135ms 明显慢于热缓存 2,229ms
- **缓存条目仅 31→32**，远少于 435 个形状——Triton kernel 跨形状共享；且**热缓存下新形状首遇成本仍在**（warm0 6,786ms，缓存新增 0）→ 每个新形状的成本不全是磁盘缓存缺失，含进程内 autotune/首次启动
- **旧 ~20 秒计时步：未复现，原因未确定**。本轮冷缓存重放预热后最慢计时步为 **3,135 ms**（非 20 s）。更正：旧 `threeway_mm_compare.py` 入口**已有两个预热步**（此前"无独立 warmup"的说法有误），其慢步发生在预热之后的计时步——与"仅首跑冷启动"的解释不完全吻合；旧运行经 `only_enable` 注册而非直调、缓存状态亦不同。**确切原因列为未确定**，不以冷编译单独归因
- 编译/加载/执行/同步**未能分离**：未启用 TRITON_KERNEL_DUMP / MLIR 时间戳；以每步缓存条目增量与新形状数作为替代证据

**D. SDPA 逐形状（`sdpa_training_compare_r43.json`，模型实际调用签名）**

从真实模型捕获（`sdpa_actual_call_r43.json`）：**dropout_p=0.1**、mask 为 **float32 加性 `[B,1,S,S]`（padding 位置 −inf）**、q 布局 `[B,H,S,D]` 转置视图（非连续）、`is_causal=False`；在 autocast bf16 下执行。顺序：先采同 dropout 的 native kernel，再 `only_enable` 后采对照。

| 形状 | native kernel（p=0 / p=0.1）| only_enable 入口 | 直调候选 |
|---|---|---|---|
| S512_pad32 | `…k64x128x64_sm80` / `…_dropout_sm80` | **NO-OP PASSTHROUGH**（kernel 集合相同）| **COMPILE FAILED**：`tl.static_assert(BLOCK_N <= HEAD_DIM)` |
| S214_pad16 | 同上 | **NO-OP PASSTHROUGH** | **可运行**：fwd **0.1062ms** vs native 0.1776ms |

- `only_enable` 返回 `No op to register`（两种形状均如此）→ 模型入口不切换 kernel
- **直调候选的可用性依赖形状**：S=512 受阻，**S=214 可运行且前向更快**——此前"本栈无 SDPA 候选"的笼统结论**修正为按形状区分**
- 其余：S 轴扰动独立性 **0.0**、无 mask 对照 6.72 / 8.31；fwd/bwd/完整（p0）S512 0.693/1.128/1.868ms、S214 0.178/0.341/0.537ms；native vs FP32 0.29–0.55%
- **[已完成，见 3c-9-E]** 真实训练态 S=228（模型实际调用捕获）对照已完成：候选**前向可运行且更快，但反向与 dropout 均不支持**

**E. 真实训练态 SDPA（round-45 修正，`sdpa_review_r45_summary.json`）——候选不可用，"前向更快"已撤回**

**目标批次已找回**（`sdpa_S214_target_r45_manifest.json`）：step 2 / micro 0 / negative_2，指纹 `4d84ea53dba8924d`，10 条序列、**1,356 个有效 token**、**S=214**——与复审 trace 记录完全一致。此前名为"S214"的文件实为 S=228，现以文件名区分（`S214_*` / `S228_*`）。捕获 Q/K/V、逐样本加性 mask 及**真实反传梯度 dO**：

| 张量 | shape | dtype | stride | 连续 |
|---|---|---|---|---|
| Q / K / V | [10,16,214,64] | bf16 | [219136,64,1024,1] | 否（转置视图）|
| mask | [10,1,214,214] | float32 | [45796,45796,214,1] | 逐样本，`-3.4e38`，有效 63.4% |
| **dO** | [10,16,214,64] | bf16 | [219136,64,1024,1] | **否**（与 Q 同布局）|

**先比前向，再固定数值只改 dO 布局独立测反向**（`sdpa_dO_S214_*.json` / `sdpa_dO_S228_*.json`）：

| | S=214 目标批次 | S=228 |
|---|---|---|
| **native 前向 vs FP32** | **0.18%** | **0.16%** |
| **候选 前向 vs FP32** | **79.3%**（max_abs 6.6 / 输出量级 8.4）| **78.8%** |
| native 反向 dQ/dK/dV（原始 dO）| 2.9e-3 / 3.3e-3 / 2.8e-3 | 4.9e-3 / 3.6e-3 / 2.7e-3 |
| native 反向耗时：**仅反向** / clone+前向+反向 | **0.341** / 0.528 ms | **0.347** / 0.538 ms |
| native 反向（dO 连续化）| **与原始完全相同** | 同 |
| 候选反向 · 原始 dO | 失败 `attention.py:975 assert do.is_contiguous()` | 同 |
| 候选反向 · dO 连续化 | **仍失败 → `:977 query.is_contiguous()`** | 同 |
| 候选反向 · Q/K/V+dO 全连续 | 可跑但**数值错误**：dQ 5.35 / dK 7.77 | dQ 3.95e3 / dK 1.58e3 |

- **反向限制已定位为两条独立断言**（`:975` dO、`:977` query）——真实训练中 dO **和** Q 都是非连续转置视图，仅连续化 dO 不够
- **候选前向在真实输入上数值就是错的（~79%）**，全连续化后反向能跑也是继承错误的前向 → **此前"前向快 1.8×"是在错误数值下测得，已撤回**
- **模型实际 dropout=0.1**（`sdpa_S214_p01_*.json`）：native 完整前反向 **0.599 ms**（kernel `…_dropout_sm80`）；候选命中 `Currently only support dropout_p=0.0`（两形状一致）
- `p=0` 的前向速度差**单列、不计入训练吞吐**（0.106 vs 0.178 ms，数值错误下测得，无采用意义）
- **计时边界（复审 bqR2JZg 更正）**：此前标为 `bwd_ms` 的数值实为"输入 clone + 前向 + 反向"的合计；现拆为 `clone_fwd_bwd_ms` 与 **`bwd_only_ms`（图在计时外预建，仅计 autograd.grad）**，并在 JSON 的 `timing_boundaries` 中注明
- **判定：FlagGems SDPA 在真实训练态不可用**（前向数值错误 + Q/dO 连续性约束 + 不支持 dropout）；训练继续使用 native cutlassFA2。S=512 编译失败证据沿用

**F. dW 命中步精度的独立复算（`acc_hit_step_r44_recomputed.json`，`recompute_dw_accuracy.py`，复审 bqQqCnw 项 2）**

从归档 `.records.pkl` 用独立脚本重算（不复用原比较逻辑），明确梯度采样于 `clip_grad_norm_` **之后**、`optimizer.step` **之前**：

| 对照（native_run1 为参考）| FFN 升维 | FFN 降维 | attn 输出投影 | 更新后参数位比较 |
|---|---|---|---|---|
| A native_run2 | 0.000e+00 PASS | 0.000e+00 PASS | 0.000e+00 PASS | 0 / 100,663,296 · 0 / 100,663,296 · 0 / 25,165,824 |
| B wrapped-stock | 0.000e+00 PASS | 0.000e+00 PASS | 0.000e+00 PASS | 同上，不一致元素 **0** |
| **C dW-only** | **0.000e+00 PASS** | **0.000e+00 PASS** | **0.000e+00 PASS** | **不一致元素 0** |

- 未舍入误差全部**精确为零**（非四舍五入）；各组 NaN/Inf = 0；退出码 0
- 退出码已覆盖：缺失目标 → 2、零命中 → 3、非有限 → 4、门限失败 → 5
- **判定：dW-only 在命中步通过 2% 门限（PASS），更新后目标参数按 int32 位表示逐位相同**
- **复算脚本已按复审修正**（`recompute_dw_accuracy.py`）：改为**逐元素** `|x−ref| / max(|ref|, 0.01·max|ref|, eps=1e-12)` 再取最大（旧的张量级 floor 对 `ref=[1,1e-4]`、`x=[1.001,1e-4]` 给 10%，正确值 **0.1%**）；比较前先对**独立预期键集**（模型结构：24 层 × 3 类 = **72**，非从记录推导）校验键、形状、元素数、有限性；**原生自比有效性纳入整体验收**（exit 7）；`--selftest` **五个**回归场景全过：同值 → 0、删空 `post_params` → 2、公式 → 0.1000%、**四组缩到 3 目标 → 2**、**native 自比梯度翻倍 → 7**。真实记录重算：72/72 覆盖、自比有效、候选 PASS、exit 0
- **目标集对应配置**：72 项预期键集来自**当前 BGE-M3 配置**（XLM-RoBERTa-large，24 层编码器 × 3 类目标权重）；换模型配置需重设 `N_LAYERS`/键模板
- **逐目标输出**：复算 JSON 现保留 **72 个目标的未舍入误差、NaN/Inf、参数位差**（`per_target_*` 字段），不再只有组级最大值——取得 records 后审阅者可逐层独立重算
- **records 可得性（如实）**：`records.pkl` 为 **6.9 GB**（72 目标层 × 4 组，float32），超出 git 限制且本环境无 release/artifact 权限。已入库 **576 个逐张量 sha256**（`acc_hit_step_r44_tensor_hashes.json`）供完整性核对
- **"从快照重建同一记录"不成立**（`acc_hit_step_r46_rebuild_check.json`）：新进程从 seed 42 重建后，与归档记录**逐位匹配 0/72**（grad_norm 10.5625 → 12.1250）——**同进程内**快照恢复是确定性的，**跨进程**重建不是（模型初始化或前两步存在非确定性来源）。可复现的是**判定**（每次运行内四组仍逐位一致、PASS、命中 432），不是绝对张量值
- **需要协助**：请维护方将 `acc_hit_step_r44.json.records.pkl`（sha256 `7d00baa3…`）作为 release artifact 附加，或授予 artifact 上传权限
- **待证实项**：旧运行 12–40% 的 native-vs-native 偏差只在同快照重放中消失，其确切来源**仍待证实**

#### 3c-10. 路由假设的最小可证伪实验（round-48，复审 bqiozcA 项 1，`route_isolated_r48.json`）——**本轮取代 round-47 的叠加注册结果**

**[已更正]** round-47 的 "+303 ms/步 host 开销"与"整步中性由路由解释"的结论**降级为相关性证据**：该数据是 profiler 开启时的 CPU 事件时间（非墙钟），且当时三种模式在**同一进程内叠加注册** `aten::mm` override（命中计数在 1056/2112 间互换即为症状），不能作为精确分解。

**本轮：全新进程隔离**（`route_isolated.py`，每模式一个进程，仅一次注册）。**发现并修复一个库句柄 bug**：`torch.library.Library(...)` 返回对象必须持有，否则被 GC 后注册**静默失效**（首次运行所有模式 hits=0、`aten::mm` 事件数相同）。

| 模式 | 白名单 dW 计算 | 12 形状命中 | host `aten::mm` 事件 | 诊断 kernel |
|---|---|---:|---:|---|
| native | 原生 `mm` | — | 5,776 | `gemm_ktype0…tile128x128` 3,216 |
| py_route | Triton `flag_gems.mm` | **1,584（12/12）** | 11,024 | **`mm_kernel_general` 528** + 原生 2,688 |
| native_contig | 原生 `mm` + `.contiguous()` | 1,584（12/12）| 11,552 | 原生 3,216 |

其余 kernel 计数三模式完全相同 → **两条路由路径在同一调用集上做相同计算**（528 = 白名单 dW 每步次数；1,584 = 528 × 3 步）。非白名单回退两条路由路径一致（`mm.out` + 预分配输出）。

**整步配对差**（profiler 关，3 个真实步 27,052/31,454/35,506 tokens，输入哈希三模式一致，两次重复组序轮换）：

| 重复（组序）| py_route − native | native_contig − native |
|---|---|---|
| rep1（native → py_route → contig）| **−18.8 / −3.3 / −7.9** | +35.4 / +10.3 / +30.7 |
| rep2（contig → py_route → native）| **−3.2 / +0.1 / −11.1** | +41.2 / +22.7 / +34.0 |
| **6 步合计** | 中位 **−5.6 ms，5/6 为负** | 中位 **+32.4 ms，0/6 为负** |

native 有效 tokens/s：14,416 / 14,895 / 15,197（rep1）。

**结论（回答"dW kernel 的局部收益能否穿过调用边界兑现到整步"）**：
- **能，但很小**：干净单次注册下，dW Python 路由在 6 个真实步中 **5 步快于原生**（中位 −5.6 ms，范围 −18.8～+0.1），约占 ~2.13 s 整步的 **0.26%**
- **host 分发开销是真实的**（`aten::mm` 事件 5,776 → 11,024），但**略小于** kernel 收益（所测形状约 15–33 ms/步）——这是首个**方向基本一致**的结果；此前所有轮次的变号来自注册叠加
- **原生 + `.contiguous()` 方案关闭**：6/6 一致更慢（复制成本高于转置视图惩罚）
- **原生转置视图专用路径不存在**：同乘积的 4 种原生写法（交换操作数 0.519、内部复制 0.637、matmul 0.520、往返 0.519 ms）均不快于 `gy.t()@x`（0.485 ms）
- **库级 / C++ hook 本环境不可用**（FlagGems `mm` 是 Python 函数）——剩余 host 开销无法在此消除

**对训练方案的影响**：dW Python 路由精度逐位一致（命中步重算 PASS），整步有小幅、方向基本一致的收益（中位 ~5.6 ms ≈ 0.26%），**但 6 步样本、幅度在步间波动内**——**不作为可部署项采纳**（`hardware_status=unverified`），记为**唯一方向为正且可复现的候选**。**[round-57：该 −5.6 ms 测于 r36 采集循环、每模式一个进程，低于本机进程间噪声；原始入口循环上的进程内配对复测见 3c-12（−9.0 ms，CI [−17.5, −0.6]，仍为候选）]**

#### 3c-11. 软件 roofline + 分词预取（round-55 修正版，复审 BqpXfeg；`software_roofline_r49.json` / `gap_attribution_r53.json`（gap_v3）/ `tokenize_overlap_r55.json` / 一页结论 `software_roofline_r49_oneline.md`）**——其中「分词预取稳定整步收益 / 候选可采纳」与「前向空隙 ~99% UNKNOWN」已被 3c-12 撤回 / 归因**

**口径**：模型 FLOPs 与**逻辑字节率**推导，**非 HBM 实测**；`hardware_status=unverified`，HBM 字节/带宽利用率/矩阵单元 activity **BLOCKED**。

**本轮修正**：① 分词预取修复共享 FIFO 样本错配（按组名取 future + 回归用例 `test_prefetch_order.py`）；② gap attribution v3：host 事件**先裁剪到 gap 再取并集**（s4 2.600 ms gap：覆盖 0.109 / UNKNOWN 2.491，与复审样例一致）；③ headroom 按**实际 token 维**分组（dW 取 K、dX 取 M；旧版 `max(M,K)` 把 FFN 特征维 4096 误计为序列长）；④ 单步 718.73 ms 与三步均值 772.12 ms 分列，"93.1% 覆盖率"与"53.39 ms 未测"清除。

| 组 | 设备 ms/步 | 有效 TFLOP/s | vs 139.2 | 逻辑字节率 |
|---|---:|---:|---:|---:|
| 反向 GEMM | 772.12（38.0%）| 88–129（6 热点）| 63%–93% | 124–183 GB/s |
| 反向 attention | 375.44（18.5%）| 13.48（阶段口径）| 不可比 | — |
| S=512 dW | 46.80 | 88.1 | 63.3% | **124.75 GB/s** |

**headroom 按实际 token 维分组**（参考时间差 = `device_ms − FLOPs/139.2e12`，非端到端上限）：

| 组 | 行数 | 调用/步 | 设备 ms（单步）| headroom ms | 占比 |
|---|---:|---:|---:|---:|---:|
| **中档（1400–4000）** | 60 | 2,880 | 324.79 | **113.91** | **46%** |
| **长档（≥4000）** | 18 | 1,344 | 311.11 | **87.16** | **35%** |
| 短档（<1400）| 22 | 1,544 | 82.83 | 47.65 | 19% |

**修正**：**中档是最大 headroom（46%）**，长档次之——与上轮"长序列 58%"相反（上轮把 FFN 特征维 4096 误计为序列长度）。优先形状：S=512 两个 dW 各 ~17 ms（长档）；dX_attn_proj (1670–3740) 与 dW_ffn (3740)（中档）。

**空隙归因（gap_v3）**：forward 最大 19.24 ms gap host 覆盖 1.504 / **UNKNOWN 17.737**；典型 3–5 ms gap 裁剪后 host 覆盖 **~0.05 ms（~99% UNKNOWN）**；无 tokenize cpu_op、无 H2D 相交。保留 UNKNOWN。 **[3c-12：加显式标记后，典型前向空隙的 94–98% 为 tokenizer；19.24 ms 空隙未复现]**

**分词预取（修复后）——【已撤回，见 3c-12】** 本段原结论为「12/12 步全部更快、中位 −25.2 ms ≈ 1.2% 整步、候选可采纳」。**round-57 撤回**：该收益是测量路径伪影——诊断脚本在 `compute()` 内加入了原入口没有的实验专用 `torch.cuda.synchronize()`，它使 inline 模式每组 launch 后阻塞主机。在原始入口循环上复测（`tokenize_overlap_native_r57.json`）：忠实实现 **12/12 更慢，中位 +27.7 ms，95% CI [+21.8, +41.0]**。此外 r56 的预取实现把 loss 移到 autocast 之外重写，**与原入口数值不等价**（`train_state_r57.json`：loss 0.927814 vs 0.917383、386/386 稠密梯度不同），当时的「输入哈希一致 + loss 偏差 0.0096 在波动带内」不足以发现。**候选关闭。** 原段落文字见 git 历史（`9bf250be`）。

**下一步**：【已由 3c-12 取代】原计划「分词预取进入 FlagScale 主路径（整步 −25 ms）」取消；中档 GEMM（114 ms）> 长档（87 ms）的 headroom 排序不变；forward 空隙的 host 埋点已完成（3c-12）。

#### 3c-12. round-57：原始入口循环复测 + 训练状态一致性 + 前向空隙显式标记（复审 7166338004；全部数字由 `scripts/build_summary_r57.py` 从各结果文件汇入 `summary_r57.json`）

**先更正本 PR 自己的两条结论**

1. **round-56 的「分词预取 −25.2 ms、12/12 更快、候选可采纳」撤回。** 原因是测量路径：诊断脚本在 `compute()` 内加入了原入口没有的实验专用 `torch.cuda.synchronize()`，它让 inline 模式每组 launch 后阻塞主机。在**原始入口循环**上（无实验同步、profiler 关）复测：忠实实现（同一个 `loss_for_batch`，仅把 tokenization 挪到工作线程）**12/12 更慢，中位 +27.7 ms，均值 +31.4 ms，95% CI [+21.8, +41.0]，范围 [+9.3, +57.9]**，与运行顺序无关。**候选关闭。**
2. **round-56 的预取实现在数值上不等价于原入口。** 它把 normalize / logits 矩阵乘 / cross_entropy 挪到 autocast 之外重写，logits 矩阵乘因此是 fp32（原入口在 autocast 内是 bf16）。同进程同快照对照（native 走原入口路径）：loss **0.927814 vs 0.917383**（差 0.0104）、grad norm 相等（9.5）、**386/386** 稠密梯度张量不同（max_abs 0.0088），超过声明的 2% 门限。当时的「输入哈希一致 + loss 偏差 0.0096 在波动带内」发现不了这一点——两模式在不同进程，基线本身就不同。

**循环保真度更正**：第 ≤56 轮的整步计时都用 r36 采集循环（无 lookahead、无 `isfinite(loss)` 同步）。原始入口（`train_retrieval.py:646-673`）每个 micro 在 forward 后有 `if not torch.isfinite(raw_loss)` 主机同步，并有跨步的一个 batch lookahead，比采集循环慢 **+17.2 / +14.6 / +10.1 ms**（三步）。r57 的全部计时改用原始入口循环；此前的整步绝对值因此偏小约 0.4–0.9%，配对差方向不受影响，但小于 ~10 ms 的效应在此前口径下无法分辨。

**1 · 训练状态一致性（复审项 1；`train_state_r57.json` + `train_state_r57.<candidate>.json`；`train_state_consistency.py`）**

同一进程：2 步预热后取快照（模型、优化器状态、CPU+CUDA RNG、4 个 micro-batch），native×3 → 安装候选 → 候选×2，每次从同一快照跑一步原始入口循环（autocast 内的原始 `loss_for_batch`、`isfinite` 同步、backward、`clip_grad_norm_`、`optimizer.step`）。阈值事先声明：梯度与优化器状态 2% 相对误差（floor = 张量最大绝对值的 1%）、参数绝对误差 1e-3、非有限值 0；判定区分「逐位相同」与「门限内」。

| 候选（对比同进程 native） | 配对数 | 稠密梯度（386 张量） | 更新后参数 | 优化器状态 | loss / grad norm | 判定 |
|---|---:|---|---|---|---|---|
| overlap = `prefetch_faithful` | 7 | 逐位相同（max_abs 0） | 逐位相同 | 逐位相同 | 相等（micro0 loss 0.917383 / 9.5） | **PASS（逐位）** |
| `h2d_nonblocking` | 7 | 逐位相同 | 逐位相同 | 逐位相同 | 相等 | **PASS（逐位）** |
| dW Python 路由 | 7 | 逐位相同 | 逐位相同 | 逐位相同 | 相等 | **PASS（逐位）** |
| r56 / run-1 的预取实现（autocast 外调用） | 7 | 386/386 不同（max_abs 0.0088） | 296 张量不同（max_abs 6.1e-05） | max_abs 9.2e-05 | loss 不等（0.927814 vs 0.917383）/ grad norm 相等 | **FAIL（是另一个计算）** |

dropout / 随机数状态 / 数据顺序：每次运行前从快照恢复 CPU 与 CUDA RNG 状态（dropout mask 相同）；4 个 micro-batch 深拷贝在快照内（数据顺序固定）；分词无随机性；预取工作线程不消耗 torch RNG。稠密参数 SHA-256 在各候选配对中相等。3 张 embedding 表（word / position / token_type）的梯度来自索引累加的反向，求和顺序在本设备上不可复现：**native-vs-native 的 3 个配对也都在且只在这 3 张表上不同**（max_abs 1.1e-4–3.1e-4），单列并排除在候选判定之外。跨进程对照无效：r57 的第一次尝试把候选放在另一进程，得到梯度 max_rel 215、loss0 0.95186 vs 0.94985——那是跨进程重建的模型本身不同（`acc_hit_step_r46_rebuild_check` 0/72），已弃用。r57 首个提交（`fc2a466e`）里 `prefetch_loss_reimpl_run1` 的对照另有一处脚本 bug：该进程的 native 基线被跑成了无 autocast（参数哈希 `509df944…`，与其它进程的 `b133cb9f…` 不同），由此得到的「loss 0.928161 vs 0.927814、grad norm 9.9375 vs 9.5、max_abs 0.6875」作废；本节数字来自修正后的运行（native 走原入口路径）。

**2 · 整步收益不是测量路径伪影（复审项 2；`tokenize_overlap_native_r57.json` + `_run1.json` + `_run2/*.json`；`tokenize_overlap_native.py`）**

原始入口循环、profiler 关、loss 路径内无实验同步（唯一新增同步是闭合计时窗的步末 `synchronize`，两模式相同）。每个 rep 一个 inline 进程 + 一个候选进程（seed 42、相同 batch 顺序），各 2 步预热 + 3 计时步；A→B / B→A 交错；配对 batch 文本哈希 0 不匹配。

| 模式（候选 − inline，ms） | 更慢 | 中位 | 均值 | 均值 95% CI | 范围 | 有效 tokens/s（均值） |
|---|---:|---:|---:|---|---|---:|
| `prefetch_faithful`（run 2，12 对） | **12/12** | **+27.7** | +31.4 | [+21.8, +41.0] | [+9.3, +57.9] | inline 14,692 → 14,479 |
| `prefetch_loss_reimpl`（run 1，12 对） | 10/12 | +16.6 | +16.9 | [+8.6, +25.3] | [−3.9, +48.4] | 未记录 |
| `h2d_nb`（run 2，12 对） | 5/12 | −1.8 | −1.0 | [−10.1, +8.1] | [−30.6, +23.4] | 14,698 |

每 rep 中位（`prefetch_faithful`）：+56.4 / +31.6 / +21.3 / +22.3。run-1 的文件曾写「11/12」，数据是 10/12，已记 erratum。**分辨率下限**：`h2d_nb`（训练状态与 inline 逐位相同的改动）配对差 sd 14.3 ms、范围 −30.6…+23.4——本机整步计时的进程间噪声即此量级。

**为什么预取在原入口上没有可隐藏的时间**（无 profiler 的主机阶段计时，`host_stage_timeline_r57.entry_12steps.json`，12 步均值）：

| 主机阶段 | ms/步 | 说明 |
|---|---:|---|
| 分词 `prepare_embedding_inputs`×20 | **65.7（3.0%）** | 纯 CPU |
| H2D `move_features` | 252.4 | **分页内存 `.to(device)` = 整条 stream 同步**，主机在此等设备 |
| 前向 launch | 268.2 | 每组 ≈13.4 ms |
| `isfinite(loss)` 同步（入口自带） | 118.8 | 等前向完成 |
| backward launch | 1425.7 | 主机被 launch 队列反压（设备受限） |
| next_batch / clip / step / 步末等待 | 2.5 / 2.6 / 4.2 / 13.4 | |

微测试：已入队 119.6 ms kernel 时，分页 H2D 阻塞主机 **117.9 ms**；`non_blocking=True` 仅 **0.02 ms**。逐组数据：主机先分词下一组（此时设备仍在执行上一组前向），随后在 H2D 里等上一组前向完成（约 6–36 ms）——**分词在原入口里本来就已与前一组前向重叠**。（预取为何更慢未进一步归因；关闭该候选不依赖这一归因。）

**H2D 非阻塞化（pinned + non_blocking，r57 新候选）**：H2D 主机等待 244.8 → 1.9 ms/步，但前向 launch 268.7 → 507.7 ms/步（主机改在 launch 调用里等，即队列已满）；三步 1891.4 / 2129.6 / 2350.8 vs 1898.1 / 2127.1 / 2351.5；配对 **中位 −1.8 ms，CI [−10.1, +8.1]，无显著差异**；训练状态逐位 PASS。→ **原入口的前向阶段是设备受限的**；不采纳，作为证据保留。

**3 · 前向空隙的显式标记（复审项 3；`gap_markers_r57.entry.json` / `gap_markers_r57.capture.json` + 各自 trace；`gap_markers.py`）**

在 trace 时间轴上加显式 `record_function` 标记：`MK_tokenizer`（分词器调用本身）、`MK_prepare_embedding_inputs`、`MK_h2d`、`MK_next_batch`（入口循环）、`MK_loss_sync`（入口循环）、`MK_python_gc`；并读取同一 trace 的 CUDA runtime 调用：`launch_submit`（cudaLaunchKernel* / cuLaunchKernel* 调用时长）、`blocking_runtime`（*Synchronize / cudaMemcpy / cudaMalloc…）、`other_runtime`、`cpu_op`。对每个 ≥2 ms 的设备空隙：各类别区间**先裁剪到空隙、再取并集、再按优先级做成互斥分割**（相加即覆盖时间），余下为 UNKNOWN；同时报告空隙后的**下一 kernel** 及其 launch 调用相对空隙起点的时刻。不做邻近推断。2 步预热 + 3 步采集（steps 2–4，与 r36 同窗口）；r36 的 trace 无法回溯加标记，采集循环运行重复其循环与窗口。

| 循环 | 设备 busy | 前向窗口空隙 ≥2 ms（3 步合计） | | | | 反向窗口空隙 ≥2 ms |
|---|---:|---:|---:|---:|---:|---|
| | | 空隙数 / 空闲 | tokenizer | 其它已解释 | **UNKNOWN** | 空闲 / UNKNOWN |
| 原始入口 | 89.07% | 28 / 94.91 ms | 83.73 ms | 10.42 | **0.76 ms（0.8%）** | 10 个 / 32.44 ms，其中 UNKNOWN 32.05 |
| r36 采集循环 | 89.99% | 27 / 96.20 ms | 89.46 ms | 4.24 | **2.50 ms（2.6%）** | 无 |

- 逐空隙：入口循环 28 个前向空隙中 **25 个由 tokenizer 主导**（占空隙的中位 96.5%，最高 98.0%）；其余 3 个为 2 个由 `cpu_op`（aten op 分发，2.05 / 2.59 ms）覆盖的空隙和 1 个纯 H2D 停顿（3.01 ms）。单个空隙内 UNKNOWN 最大 0.124 ms。空隙后的下一 kernel 在 26/28 个空隙里是 `cudaMemcpyAsync`（H2D），其 launch 在分词结束后 0.05–0.19 ms 发出。采集循环 27 个前向空隙中 26 个由 tokenizer 主导，余下 1 个（2.68 ms）含 1.45 ms UNKNOWN。
- 「其它已解释」主要是偶发的 ~2.0 ms 空隙，完整落在 `move_features` 内（分页 H2D + `cudaStreamSynchronize` 1.93–2.00 ms）。
- **r36 的 19.24 ms 最大空隙未复现**（本次采集循环最大前向空隙 5.44 ms）。
- **剩余 UNKNOWN**（原始入口循环）：反向窗口里几乎每个 micro 一次 2.0–10.2 ms 的停顿，前后都是 `vectorized_elementwise_kernel`，内部没有任何 cpu_op / runtime / GC 事件，下一 kernel 的 launch 恰在空隙末尾发出。已检验并排除 Python GC；其余**保留 UNKNOWN**，且仅出现在带 profiler 的 trace 中。
- 全部空隙（含 <2 ms）3 步合计 749 ms，其中 `cpu_op` 410 ms + `launch_submit` 78.5 ms——profiler 抬高每次 launch 的主机开销所致；**带 profiler 的空隙毫秒数不外推到无 profiler 的整步**。

**无 profiler 的设备利用率**（`smi_utilization_r57.json`，`nvidia-smi` 100 ms 周期采样，不向负载添加任何 instrumentation；12 步窗口 259 个样本）：`utilization.gpu` 最小 88 / p10 94 / **中位 96** / 均值 96.1 / p90 98 / 最大 100。该指标语义由厂商定义、本环境无文档，**仅报告原始分布，不当作忙碌时间测量**，更不替代 HBM / 矩阵单元计数器。

**GC 假设检验（否定）**：`gc.callbacks` 统计每步 5 次回收、合计 0.35–0.44 ms、最大单次 0.18 ms；`gc.disable()` 后整步 1892.8 / 2123.5 / 2359.8 vs 默认 1890.7 / 2123.4 / 2348.7（不变）。带标记的 trace 中 `MK_python_gc` 对 ≥2 ms 空隙的覆盖为 0.00 ms。

**4 · dW Python 路由在原始入口循环上的复测（`route_entry_ab_r57.json` + `route_entry_ab_r57/*.json` + `route_entry_ab_r57_batch1_console/*.json`；`route_entry_ab.py`）**

r48 的 −5.6 ms（5/6）测于采集循环、每模式一个进程，低于上文量到的进程间噪声。r57 改为**进程内配对 + 差中差**：每进程阶段 1（native；新数据迭代器，2 预热 + 3 计时步）→ [注册 dW 路由] → 阶段 2（同 seed 新迭代器，batch 文本哈希逐项相同）；`--second native` 为对照（不注册，阶段差 = 纯顺序效应）。第二批 16 进程交错（8 py_route + 8 native）；第一批 12 进程仅有控制台记录（文件命名 bug 致数据文件丢失，已如实转录并标注来源）。

| | 配对步数 | 阶段 2 − 阶段 1（ms） |
|---|---:|---|
| py_route（合并） | 42 | 中位 **−4.9**，42 步中 32 步更快 |
| native 对照（合并） | 42 | 中位 +1.4，17/42 更快 |
| **差中差（合并 28 进程）** | | **−9.04 ms，95% CI [−17.52, −0.55]** |
| 按批 | | 第一批 −13.97 [−24.49, −3.46]；第二批 −5.33 [−18.74, +8.07] |

进程级均值：py_route 14 个进程中 13 个为负，native 对照 14 个中 6 个（Fisher 双侧 p = 0.0128）；命中 12/12 形状（每进程阶段 2 共 1056 次）。**读法**：方向一致的**小幅**效应，点估计 ≈ 0.4% 整步，CI 上缘贴近 0，数据完整的第二批单独看并不显著，进程级离群值达 +40…+72 ms。训练状态逐位 PASS。**维持「候选、不可部署」**，不作为「稳定降低整步时间」的改动。

**5 · 硬件效率与 records（复审项 4、5）**：无变化。PCM 仍被 `pg-exporter` 占用（`hbm_blocked_r46.md`），HBM 读写字节 / 带宽利用率 / 矩阵单元活跃度**未回答**，`hardware_status=unverified`；逻辑字节率是模型推导量。6.9 GB `records.pkl`（sha256 `7d00baa3…`）仍需仓库维护方托管。

**对复审三个问题的当前回答**

1. **哪个改动能稳定降低整步时间、降低多少**：**没有可部署的改动**。分词预取 +27.7 ms（更慢，关闭）；H2D 非阻塞 −1.8 ms（无显著差异）；dW Python 路由 −9.0 ms [−17.5, −0.6]（≈0.4%，训练状态逐位相同，候选）。
2. **各部分占用多少、哪些仍未知**：反向 GEMM 772 ms/步（38%）、反向 attention 375 ms（18.5%）、前向 GEMM 330 ms（16.2%）、前向 attention 205 ms（10.1%）（r36 kernel 时间）；主机分词 65.7 ms/步（3.0%）；前向空隙（带 profiler）≈31.6 ms/步，其中 tokenizer ≈27.9 已归因，前向 UNKNOWN 0.25，反向 UNKNOWN ≈10.8（仅带 profiler）；分词预取不省时间。仍未知：无 profiler 运行中 ~4% 未利用时间的构成。
3. **哪些硬件效率结论有计数器证据**：**没有**。HBM 带宽利用率与矩阵单元活跃度均受平台测量条件限制。

`draft / deployable=false / hardware_status=unverified` 维持；vendor 白名单维持。

#### 4. F-2 最小 shape 隔离（单次触发）

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
