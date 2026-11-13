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
3. **小模型（v2.1，显式 PPU 设备，可训练 embedding）**：embedding 在 optimizer 构造**前**创建并**纳入参数列表**（16 参数，embedding 梯度证据：cuda/finite/max_abs 3.7e-07）；blocks/head/embedding/token 全部 `.to("cuda")`，JSON 记录断言 `params_all_cuda_before_optim=true`、`grads_all_on_cuda=true`、`embedding_trainable_in_params=true`；profiler 命中证据：selective 配置前/反向命中 `gelu_none_kernel_rank_1`/`gelu_backward_none_kernel_rank_1`，native 配置命中 `GeluCUDAKernelImpl`/`GeluBackwardCUDAKe…`。**替代说明**：attention 用显式 matmul+softmax（因 F-1 sdpa 挂起）；结构仍为固定种子 gated MLP + 2 层 attention 块，无外部权重。**结果（native vs selective，8 步）**：逐步 loss 相对差 ≤1.2e-07（fp32 累积量级；embedding 梯度 max_abs 8 位有效数字一致）；16/16 参数 grad sha256 不同（两套 gelu kernel 数值路径不同，与精度矩阵 grad 差异一致）——**本负载只能证明"当前固定小模型可运行、数值偏差在 fp32 累积量级"，不能表述为逐位一致**；每配置 3 独立进程损失序列逐位可复现（配置内确定性）；稳态步时 native ~2.25 ms / selective ~2.85 ms。**v1 的"三组逐位一致"结论作废**（v1 未上设备，实为 CPU aten 自身一致性）。
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
- **fused 正确性按四个独立层次区分（不得混用）**：
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

- 违例位置 84% 在 |x|>3.9 负尾（p50 |x|=7.1）：**FlagGems fp32 gelu_backward 在 |x|≳8 输出精确 0.0**（native/真值 ~1e-14–1e-36），5.5≤|x|≤8 幅度偏差至 1.6×，|x|<5 正常；绝对误差 ≤1.14e-07。fp64 双公式模拟确认 gems 公式数学等价——**负尾有限精度伪影（erf 抵消 + exp 项损失），非算法错误**；单点梯度 ≤1.14e-07，**候选训练影响较小——但"可忽略"的定稿需模型级敏感性/长步训练对照支持，当前为推断、待训练验证**。证据：`fail_attribution_gelu_bwd.json`（含 fp64 双公式复算）。
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
| fused AdamW（fp32/bf16/mixed、bge336 级负载） | **有效且精度通过**：2.8–2.9×（optimizer 段）、整步 -5.6~-11%（小模型）；checkpoint 逐位一致；fp16 场景 fused 为唯一可靠路径。候选 fix 定位：FlagScale `adamw-fused` 白名单扩 thead（集成验证 pending）+ fp16 训练强制 fused 建议 |
| CUDA graph 捕获/重放（静态 shape） | **可行候选**（1.2M 12.6×，组件级微基准）；未接入训练整步，不得外推整步收益；组件拆分、动态边界、回退策略待下一轮 |
| selective enable | 维持（无变化）|
| full-enable training | blocked（F-1/F-2）|

### 第一轮复审口径修订（2026-11-13，意见 6857432618）

四处口径已按审阅修正：①fused 正确性四层次区分（单步闭式 ≤1 ulp / 100 步轨迹漂移不承诺逐位 / checkpoint 内部一致 / 无 NaN/Inf）；②fp16 分别报告 stock 损坏与 fused 当前配置无污染，限定范围并补 fp32-state 替代方案；③GELU 负尾影响改"候选较小、待训练验证"；④CUDA graph 明确组件级静态 shape 候选、12.6× 不外推整步。下一轮顺序按审阅执行：FlagScale 集成 → AdamW 数值验收（vs fp64/稳定参考）→ CUDA Graph 训练段接入 → GELU 归因训练闭环 → F-1/F-2 持续复验 → 整步/多卡。
