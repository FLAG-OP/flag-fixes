# 昆仑芯 P800 真实 BGE-M3 入口基线与消融（issue #9，problem29 真实对象验收）

日期：2026-11-17 · 平台：昆仑芯 P800 OAM（8 卡节点；本轮单卡跑可见卡 1，双卡 FSDP2 用卡 1、2）·
目标树：FlagGems `4e08b44e9b9ad68c2c27b4d25d04c6b76c30ff7a`（v5.0.0，目标文件 hash 与
`gelu-kunlunxin-pow` before/after 逐一核对）+ FlagScale `wxj630/FlagScale@729ae45`
（feat/retrieval-flagos，`flagscale/train/utils/flaggems.py` = selective-enable fix 的
before hash `cff9890b…`；`train_retrieval.py` HEAD hash `53a5b8a8…`，adamw patch 以
+29 行 offset 干净应用）· 环境：`results/env_record.json`（torch 2.9.0+cu129 / triton
3.0.0 / XPU-SMI 驱动 5.0.21.47 / XPU-RT 5.13.0 / Python 3.10.18）·
原始数据：`results/`（逐配置 3 独立进程 JSON、kernel 表、数值轨迹、GEMM sweep、trainer 日志）·
复现命令见文末。

**回答 issue #9 的核心问题**：现有 `-15.3%` 来自 XLM-R 重建负载，不能当作原始
problem29 的结论。本轮在**真实 BGE-M3 权重 + 真实 T2Ranking triplet-15 数据 +
真实 FlagScale 入口**（`examples/retrieval/`，batch 10 × accum 4 = 200 texts/步）
上重建了基线与消融，完成 GELU / GEMM / glue 的同批证据。

## 结论（真实对象口径，3 独立进程，wall 取中位数；收益不可相加）

| 配置 | wall/step | Δwall | kernel/step | 吞吐 | 组成 |
|---|---:|---:|---:|---:|---|
| native（stock） | 3,572.4 ms | — | 3,240.8 ms | 56.0 texts/s | 厂商 kernel 路径 |
| +GEMM env（D） | 3,488.8 ms | **-2.3%** | 3,113.5 ms | 57.3 | `XMLIR_MATMUL_FAST_MODE/FAST_TUNE`（固定形状 sweep 逐位一致；真实变长 shape 轨迹有 ~0.5% loss 偏移，见数值节） |
| +GELU only（B） | 3,070.2 ms | **-14.1%** | 2,687.1 ms | 65.1 | gelu-kunlunxin-pow + flaggems-selective-enable（AdamW 保持 stock） |
| +glue only（C） | 3,594.7 ms | **+0.6%** | 3,348.1 ms | 55.6 | `XMLIR_ENABLE_LINEAR_FC_FUSION=0`（净负收益，见 §C） |
| all-fixes（E） | **2,972.8 ms** | **-16.8%** | **2,535.3 ms** | **67.3** | GELU + adamw-fused + GEMM env（FlagScale 集成路径） |

配置内稳定性：3 进程极差 ≤0.7%（如 native 3,580.4/3,563.4/3,572.4）；GELU kernel
类目在三个复点逐位一致（616.0 ms）。trainer 真入口（torchrun 单卡，DDP 封装）交叉
核对：native 两独立进程 p50=3,676.8/3,673.4 ms（比 harness 高 ~100 ms，DDP 包裹与
日志开销），all 3,002.6 ms（-18.3%），方向与量级一致。

## 与原始 problem29 对账（归因口径逐项对上）

| 指标 | 原始 problem29 | 本轮（重建数据集） | 偏差 |
|---|---|---|---|
| 整步 kernel | 2,923 ms（A100 1,419，2.06×） | 3,240.8 ms | 绝对值受数据长度分布影响（见下），不可直接互换 |
| GELU 占 kernel | 1,028.9 ms（35.2%） | 1,169.7 ms（**36.1%**） | +13.6%（负例更长） |
| cast_te | 66.2 ms、4,800 次/步 | 71.9 ms、**4,800 次/步** | 次数完全一致 |
| findmax1d | 40.5 ms、4,800 次/步 | 42.9 ms、4,800 次/步 | 一致 |
| transpose_021_sdnn | 1,936 次/步 | 1,936 次/步、38.2 ms | **次数完全一致** |
| cluster_cast_kl3 | 12,399 次/步 | **12,460 次/步、112.9 ms** | 次数一致；PR3 曾报"单卡重建负载复现不出"，真实入口**单卡即可复现**，归因缺口关闭 |
| 胶水合计 | 309.0 ms（20.6% 差距） | 306.5 ms | -0.8% |
| device 计算占比 | 96.0% | 94.9%（tokenize 104.5 ms、dataloader 0.87 ms/步） | 一致：数据管线不是瓶颈 |

数据差异声明：原始 `triplet-15` 制品不在本机（内部处理产物）。本轮按官方
THUIR/T2Ranking 发布件重建：positive=qrels 最高相关（平局取小 pid，rel≥1），
negative=BM25 top 序排除该 query 全部 qrels 判定（防假负例），不足 15 个时用
`blake2b(qid, counter)` 确定性补采样（仅 9/200,372 行触发）；200,372 行 × 3 parquet。
重建数据负例长度分布比原制品长（BM25 top 负例中位 ~400 token vs 原 padded ~299），
因此**绝对毫秒数不可与原始报告互换**；配置间对比、归因与 kernel 次数同批自洽。
BGE-M3 权重为本体 `BAAI/bge-m3`（pytorch_model.bin sha256 `b5e0ce34…aad38`，完整值见
`results/model_sha256.txt`）；入口/配置/tokenizer/损失/优化器全部为原始 FlagScale
代码——`profile_step_segments.py` 逐字节未改，训练入口 `train_retrieval.py` 仅打
`adamw-fused` patch（stock 树另存，native/gemm/glue 配置走未打补丁的
FlagScale-stock，flaggems.py 在两棵树分别等于 fix 的 after/before hash）。

## 逐任务证据（issue #9 A–E）

### A. 真实对象和基线 — ✅ 完成

- 环境与版本：`results/env_record.json`；模型加载冒烟：`results/model-smoke.json`
  （567.8M 参数、391 可训练张量，与 adamw-fused 记录的 389+2 一致量级）。
- 分段表（native，3 进程各取 micro 中位数 ×accum 后再取进程中位）：
  data(next) 0.87 ms、forward 1,341.3 ms、backward 2,140.1 ms、clip+optimizer
  62.1 ms；tokenize 单独 104.5 ms——**94.9% 在 device 计算**，数据管线不是瓶颈。
  （首轮 README 误写 forward=1,530.5/backward=1,838.2，来自被并发污染的
  native_r1 复点，已按本批干净数据更正并由 `scripts/aggregate.py` 生成。）
- profiler 交叉检查：分段 device 计算 3,375.3 ms vs profiler kernel 合计
  3,240.8 ms，差 ~4%（profiler 统计不到的同步空隙与发射开销），两个独立口径互证。
- kernel 级归因表（112 项、含裸 kernel 名/次数/时长）已存于
  `results/segments/native_r*.json` 的 `ops` 字段；汇总见 `results/summary.json`
  与 `results/aggregate-stdout.txt`（由 `scripts/aggregate.py` 从本批 JSON 生成）。

### B. GELU（P0）— ✅ 优化生效 + 路径切换确认；数值为六步轨迹观察（形式验收 pending）；tanh 候选重新打开

1. **路径切换证据（真实入口）**：native 的 480 次/步 `gelu_highprecision_nv` +
   480 次/步 `gelu_bwd_highprecision_nv` 在 gelu 配置下**完全消失**，替换为
   FlagGems `gelu_none_kernel_rank_1` / `gelu_backward_none_kernel_rank_1`
   （`results/segments/gelu_r*.json` ops 表；微基准级探针
   `results/gelu-refresh/gelu-path-{native,gems}.json`）。
2. **同批前后对照（ms/步）**：GELU 类目 1,169.7 → 616.0（**-47.3%**）；逐调用
   前向 888.6→550.5 µs（1.61×）、反向 1,548.3→732.9 µs（2.11×）。其他 kernel
   类目零变化（GEMM 911.0→911.1、attention 323.2→323.1、cast_te 71.9→71.8）——
   **修复只动了该动的地方**。整步 wall -14.1%。
3. **GELU 真实调用清单（复审后新增，`results/gelu_call_inventory.json`）**：
   训练态 hook 记录 2,880 次前向调用（6 步）：**全部为 FFN 中间维 (10, L, 4096)、
   bf16、contiguous、autocast 开启、approximate="none"**（复审指出的 4096 中间维
   得到确认）；长度直方图高度可变——**64% 调用在满 padded (10,512,4096)**，
   短 anchor 批 (10,12–16,4096) 占 ~18%，其余为各种中间长度。
4. **路径对比与 tanh 候选（复审后重做，v1 探针撤回）**：v1 探针存在两处缺陷
   （用扁平 (2990,1024) 而非真实 (10,L,4096)；bwd 计时含前向且未清梯度），
   其"tanh 中性/关闭决策项"结论**撤回**。修正版（真实直方图、前向/反向分开
   计时、`retain_graph` 复用图并逐步清梯度，`results/gelu_none_vs_tanh_v2.json`）：
   **none 638.8 vs tanh 284.4 ms/步**——在真实 (10,L,4096) 调用上 tanh 快 ~55%，
   若替换可再省约 -11% step wall。**tanh 候选重新打开**（作为观察与候选记录）：
   它改变数值语义，须先过同输入/模型轨迹检查才能进入端到端候选；本表不据此
   排除也不据此采用。
5. **数值轨迹（六步观察，精度验收 pending）**：数值运行与性能配置共用定义
   （`scripts/ablation_configs.sh`）：all_fixes_v2 数值运行含与性能 all 完全一致的
   env（XMLIR fast GEMM + adamw auto + gelu include）。固定 seed 6 步、相同输入
   顺序（全批输入指纹一致）与 RNG：native 三独立进程（r1/r2/r3）逐步**逐位一致**
   （loss 序列与 param/grad sha256 全同，过程内噪声下限=0）。跨配置**量化对比**
   （`results/numerics/compare_*.json`，张量级 numeric-equal 占比/max_abs/
   L2-rel（逐张量最大值，ref 范数非零者）/grad L2 差范数）：

   | 对比 | loss 逐步最大相对差 | param L2 rel（逐张量最大值） | param max_abs | grad 异常 |
   |---|---:|---:|---:|---|
   | gelu_only_v2 vs native_r3 | 1.24% | 0.60% | 3.66e-4 | 小范数 LayerNorm 张量（‖g‖≈0.01）上 L2-rel 至 2.8×，绝对量级 1e-2 |
   | all_fixes_v2 vs native_r3 | 0.72% | 0.62% | 3.66e-4 | 同上量级 |
   | gemm_env_v2 vs native_r3 | 0.43% | 0.50% | 3.66e-4 | — |

   **GEMM env 在真实变长 shape 上改变数值**（loss ~0.4–0.5%）：固定形状 sweep 的
   "逐位一致"只在 M=2990 成立，不可外推到变长训练——此点为本轮修正。
   6 步轨迹内无非有限参数，峰值显存 26.7 GB 一致。预声明的暂定判据
   （`scripts/numerics_compare.py` docstring，声明于运行前）中 loss/param L2/
   非有限满足，**逐位占比判据不满足**（6 步后约 2/3 参数元素有 ±ULP 级差异，
   param max_abs 3.66e-4 ≈ 数个 bf16 ULP）、grad L2 差范数 0.2–0.5（超出 2e-3
   参考线）——**不据此判定算法错误，也不据此宣布通过；形式精度验收 pending**
   （判据需 owner 对最终配置签核，当前仅为六步轨迹观察）。量化对比所用张量
   快照（每配置 ~4.6 GB .pt）不入库，
    以逐位 sha256 与上述量化 JSON 代之，复跑用 `numerics_check.py --save-final`
    重建。边界输入证据沿用 gelu-kunlunxin-pow 的
   `verify_gelu_boundary.py`/`test_gelu_boundary.py` 与 2026-11-11 边界对照
   （乘法替换仅 fp32 tanh 反向 @±3 相邻值 1 ulp，bf16/fp16 逐位一致）。
6. **GELU 残余耗时（修正解释，撤回"4.8× shape 开销"强归因）**：修正对象后的
   微基准（真实直方图 (10,L,4096)）得到 none 估算 **638.8 ms/步 ≈ 实测
   616.0 ms（+3.7%）**——kernel 随元素数近似线性扩展（(10,512,4096)=20.97M
   元素前向 0.756 ms / 反向 1.007 ms），**不存在此前宣称的"4.8 倍变长 shape
   处理开销"**（该结论是比较错误对象所致，撤回）；616 ms 就是真实工作负载在
   FlagGems none 路径上的实际成本，进一步优化方向是 none→tanh（上条）与
   Linear epilogue 融合（厂商项），不再是"shape bucket 化 autotune"假设。
   保留的独立事实：冷 Triton 缓存下每个新 padded 长度首遇触发 10–15 s 编译
   （短跑日志 `logs/trainer/gelu_rr1.log` 可见），完整 epoch 内摊销、共享缓存
   后续进程不再出现；接入脚本应预热或预共享缓存。

### C. cast/transpose 胶水（P0）— ✅ 真入口归因完成 + 局部开关复测为净负收益

1. **逐调用归因（native，单卡真入口）**：cast_te 71.9 ms/4,800 次、findmax1d
   42.9 ms/4,800 次（与 cast_te 严格成对，PR3 的 F.linear fc_fusion 前处理税
   结论在真实入口成立）、cluster_cast_kl3 112.9 ms/12,460 次（**原报告称单卡
   复现不出，真实入口单卡即出现，次数 12,460 vs 12,399**）、transpose_021_sdnn
   38.2 ms/1,936 次 + transpose_0213_cluster 40.5 ms/2,976 次、copy/pad 1.4 ms。
   胶水合计 306.5 ms/步（8.5% kernel、占对 A100 差距口径下与原 20.6% 同量级）。
2. **fc_fusion 开关（唯一本地可切胶水开关）**：`XMLIR_ENABLE_LINEAR_FC_FUSION=0`
   使 cast_te/findmax 归零（-114.8 ms），但 GEMM 类目 911.0→1,165.6
   （**+254.6 ms**）、tensor_calc +13.5，整步 wall **+0.6%（净负收益）**——
   PR3 重建负载结论（税 34 µs/次 < fast-GEMM 收益）在真实入口定量复现。
   "权重侧 amax/scale 缓存"仍是正确的厂商请求（潜力 ~-49 ms/步，未变）。
3. **多卡边界**：cluster_cast_kl3 的 12,460 次/步在单卡即存在，其归因不再
   依赖真实多卡节点；多卡下的通信等待与带宽分离仍需运行时层（见 §E）。

### D. GEMM（P1）— ✅ env 有小收益（wall -2.3%）；固定形状无 headroom；数值边界见 §B

1. **真入口 A/B**：GEMM 类目 911.0 → 772.0 ms/步（**-139.0 ms，-15.3%**），整步
   wall -2.3%。收益来自变长 M 分布（真入口每 batch M=10×padded，1,400–5,120
   变动），厂商 fast algo 对该分布整体提速。**数值边界**：该 env 在真实变长
   shape 上改变训练数值（6 步 loss 最大相对差 ~0.5%，见 §B 数值表）——固定
   形状 sweep 的"输出逐位一致"只在 M=2990 成立，不外推到变长训练。
2. **固定形状微 sweep（7 env × 真实 N/K）**：base 42.9/153.8/141.7 TFLOPS
   （qkv/ffn_up/ffn_down @M=2990，bf16）；`XMLIR_MATMUL_FAST_*`、
   `KLX_USE_AUTOTUNE`、`XMLIR_ENABLE_XBLAS_ADDMM`、`XMLIR_FC_AUTOTUNE_BOOL`、
   `XMLIR_BMM_DISPATCH_VALUE` 在固定形状上全部 ≈ base（±噪声，0.144–0.156 ms），
   且**输出与 base 逐位一致**（max_abs_diff=0.0，`results-clean/gemm/`）——
   固定形状无 env 可挖，变长分布才有 -139 ms/步，与 2 互证。
3. **覆盖原差距结论**：原报告 GEMM 占差距 10.2%（1.2×慢）。本轮 -139 ms 约占
   本轮 GELU+GEMM 可优化空间的 20%；GEMM 主要部分（~770 ms）为厂商算法层，
   本地无进一步开关，维持"优先级低/移交"结论不变。

### E. 组合与多卡 — ✅ 组合完成；多卡登记平台阻断（F-P800-1）

1. **消融矩阵**：见顶部表。非相加性验证：B（-14.1%）+ D（-2.3%）简单相加
   -16.4%，实测 all **-16.8%**（AdamW 贡献 clip+optimizer 62.1→38.8 ms，
   -23.3 ms/步，与局部收益叠加方向一致但不等长）；glue 单独 +0.6%，
   未并入 all 推荐配置。
2. **稳定性**：每配置 3 独立进程，报告中位数与逐 rep 值（`results/summary.json`
   `wall_ms_range`）；3 进程样本不构造步级 p95（首轮误用 `wall_ms_p95`，与 p50
   相同，已改为中位数+范围口径）。trainer 口径 12 个稳态步 p50 与 min/max 见
   §顶部交叉核对行；
   `results/numerics/` 记录 loss/参数/梯度轨迹与显存。
3. **多卡（FSDP2 2 卡）— 平台阻断 F-P800-1（本轮新发现，未解决）**：在安静
   窗口（host 仅本任务、load≈3）按单次隔离触发协议共触发 **4 次**（真实
   trainer native/all + 单步 profiler harness 各两配置），**全部失败**，失败点
   均为 XCCL 集合通信 watchdog 600 s 超时（`_ALLGATHER_BASE`，各次 SeqNum=
   7/29/59/76——集合通信能推进数十次后**停在任意一次**，与 payload 大小无关：
   6.3M 与 132.7M 元素都停）；同日早些时候的 3 次带载触发（与单卡矩阵并发）
   另出现 `cudaErrorLaunchTimeout`、厂商 fc_fusion `findmax failed code 2`、
   SeqNum=4 all-gather 超时三种签名。合计 7 次触发、2 种代码路径（trainer/
   harness）、2 种 master dtype（fp32/bf16）、fc_fusion 开/关两态，无一次完成
   首个 optimizer step；而**单卡 DDP 全轮可用**（harness 与 trainer 全部
   rc=0）。FlagScale 侧已内置"coalesced allgather shim（per-tensor 回退）"
   仍复现，指向厂商 XCCL/FlagCX 栈的 FSDP2 兼容性问题，移交运行时层。
   日志：`logs/fsdp2/{profile-native,native,profile-all,all}.log`。
   因此 §5.5 要求的"通信与计算分列/按 rank 归集"本轮**无法出数**，登记为
   pending；多卡结论不外推，保持 PR3 平台矩阵"多卡未验收"状态。
4. **状态保持**：按 issue #9 要求，在 owner 审核、平台 x-benchmark 与完整
   FSDP2/多卡验收完成前，全部 fix 维持 `draft / deployable=false /
   hardware_status=unverified`；本轮是"真实入口可复现 + 根因闭环"的证据层，
   不改变部署状态，`-16.8%` 不得写入"原始问题已解决"。

## 已知边界（不装完美）

- **数据集为确定性重建**（规则见上），非原始内部 triplet-15 制品；绝对毫秒数
  只在本轮配置间可比。已尽最大对账：kernel 次数（cast_te 4,800、findmax 4,800、
  transpose_021_sdnn 1,936、cluster_cast_kl3 12,460/12,399）与类别占比
  （GELU 35.2%→36.1%）与原始报告一致。
- **本轮 kernel 差距对 A100 无同批对照**（机器上无 A100）：占差距比例沿用
  原始 problem29 的 A100 数字做参照，不做新倍数声明。
- **FlagGems gelu 变长直方图成本与编译停顿**：真实 (10,L,4096) 直方图上
  none 路径 616–639 ms/步（kernel 线性扩展，无异常 shape 开销）；tanh 候选
  284 ms/步待轨迹验证；冷缓存逐 shape 编译停顿（10–15 s）需预热/共享缓存。
  均未在本轮修复，作为 FlagGems/运行时层候选与接入边界登记。
- **GEMM env 的数值边界**：固定形状 sweep 逐位一致只在 M=2990 成立；真实
  变长 shape 上 env 使 6 步 loss 偏移 ~0.5%（见 §B 数值表）——部署该 env 需
  配套精度验收，不能引用固定形状的逐位一致声明。
- **FSDP2 多卡**：平台阻断 F-P800-1（XCCL 集合通信在 FSDP2 下停在任意一次
  all-gather，7 次触发 0 成功；单卡 DDP 全轮可用）——多卡通信分解 pending，
  待运行时层解决后按 `scripts/run_fsdp2.sh` 复测；多卡结论不外推。
- 微基准 tanh/none 对比（v2）使用真实调用直方图（`gelu_call_inventory.json`，
  6 步 2,880 调用）逐 shape 加权，前向/反向分开计时；v1 探针及其结论已撤回
  （见 §B）。
- 共享主机测量纪律（PR3 历史）：本轮全部性能数据在安静窗口顺序采集（host
  load ≈3 由本任务自身构成）；首轮曾发生与卡 2 任务并发导致的 wall 污染，
  该批数据已废弃不用（保留在 `results/`（旧）目录外的 evidence 工作区，
  不进入本交付）。

## 复现命令

### 0) 外部树与数据准备（一次性）

```bash
# 源码树（provenance）
#   FlagGems   4e08b44e9b9ad68c2c27b4d25d04c6b76c30ff7a (v5.0.0)  → /workspace/FlagGems
#   FlagGems patched：同树 + gelu-kunlunxin-pow patch，after hash 核对 → /workspace/FlagGems-fixed
#   FlagScale  wxj630/FlagScale feat/retrieval-flagos @729ae45    → /workspace/FlagScale
#     （sparse checkout 后需补齐 flagscale/train/utils、flagscale/platforms；
#      再应用 selective-enable 与 adamw-fused patch → /workspace/FlagScale 为集成树）
#   FlagScale-stock：同 commit 未打补丁副本                        → /workspace/FlagScale-stock

# 数据（官方 THUIR/T2Ranking，重建规则见 scripts/build_triplet_step1.sh 头注）
bash scripts/build_triplet_step1.sh /workspace/data-t2ranking
python3 scripts/assemble_triplet15.py /workspace/data-t2ranking \
        /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15
#   数据指纹：queries.train.tsv 10,210,389 B / qrels.train.tsv 29,881,875 B /
#   collection.tsv 3,659,243,528 B / train.bm25.tsv 3,650,744,848 B（字节数=下载校验）
#   产物：200,372 行 × 3 parquet（columns anchor,positive,negative_1..15）

# 模型：BAAI/bge-m3 → /flagos-retrieval-models/bge-m3/bge-m3
#   pytorch_model.bin sha256 b5e0ce34…aad38（results/model_sha256.txt，
#   HF_ENDPOINT=https://hf-mirror.com）
```

### 1) 单卡消融矩阵（每配置 3 独立进程；~2.5 h，安静窗口串行）

```bash
bash scripts/run_ablation_matrix.sh     # 失败聚合：任一 rep 失败或 JSON 缺字段 → 非零退出
```

### 2) 真实 trainer 交叉核对（torchrun 单卡，MAX_SAMPLES=600 → 15 optimizer steps）

```bash
MAX_SAMPLES=600 FLAGSCALE_SAVE_EVERY_EPOCH=false FS_TREE=/workspace/FlagScale-stock \
  bash scripts/train_entry.sh native r1 results/trainer
```

### 3) GEMM sweep（每 env 独立进程）

```bash
bash scripts/run_gemm_sweep.sh
```

### 4) 数值轨迹（与性能配置共用定义 scripts/ablation_configs.sh）

```bash
bash scripts/run_numerics_v2.sh revision   # native_r3/all_fixes_v2/gemm_env_v2 + 量化对比
python3 scripts/numerics_compare.py <ref.json> <ref.pt> <tgt.json> <tgt.pt> out.json
```

### 5) FSDP2 2 卡（安静窗口；single-trigger blocker 协议）

```bash
bash scripts/run_fsdp2.sh profile-native && bash scripts/run_fsdp2.sh native
bash scripts/run_fsdp2.sh profile-all    && bash scripts/run_fsdp2.sh all
```

### 6) GELU 调用清单 / 路径探针 / tanh-none 对比

```bash
python3 scripts/gelu_call_inventory.py        # 训练态逐调用 shape/dtype/stride（复审 P1/P2）
python3 scripts/probe_gelu_path.py native out.json   # + gems 变体（见 scripts 头注）
python3 scripts/gelu_none_vs_tanh.py          # v2：真实直方图、fwd/bwd 分开计时
```

环境变量基线：`KLX_USE_AUTOTUNE=0`、`XPU_ENABLE_PROFILER_TRACING=1`、
`XPU_CUPTI_ENABLE_DEVICE=0`（昆仑芯 torch.profiler 必需，problem29 §3.2）、
`TRITON_CACHE_DIR`/`FLAGGEMS_CACHE_DIR` 指向数据盘、`HF_HUB_OFFLINE=1`。
脚本内的 `/workspace/evidence` 绝对路径为本机运行时布局；仓库内 scripts/ 为
同一脚本的归档副本，复跑前把 `EV=` 变量改为本地路径即可（各脚本头部注明）。
汇总表由 `scripts/aggregate.py` 从 results-clean/ 的 JSON 生成
（`results/summary.json`、`results/aggregate-stdout.txt` 为本批产物）。
