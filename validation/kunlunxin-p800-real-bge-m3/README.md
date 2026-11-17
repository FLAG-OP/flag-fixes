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
| +GEMM env（D） | 3,488.8 ms | **-2.3%** | 3,113.5 ms | 57.3 | `XMLIR_MATMUL_FAST_MODE/FAST_TUNE`，输出逐位一致 |
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
- 分段表（native，median）：data(next) 0.87 ms、forward(含 tokenize+H2D)
  1,530.5 ms、backward 1,838.2 ms、clip+optimizer 62.1 ms；tokenize 单独 104.5 ms、
  H2D 2.3 ms——**94.9% 在 device 计算**，数据管线不是瓶颈（直接排除"优化数据加载"）。
- profiler 交叉检查：分段 device 计算 3,375.3 ms vs profiler kernel 合计
  3,240.8 ms，差 ~4%（profiler 统计不到的同步空隙与发射开销），两个独立口径互证。
- kernel 级归因表（112 项、含裸 kernel 名/次数/时长）已存于
  `results/segments/native_r*.json` 的 `ops` 字段；类别映射脚本与口径同 PR3。

### B. GELU（P0）— ✅ 优化生效 + 路径确认 + 数值通过；残余为 shape 处理开销

1. **路径切换证据（真实入口）**：native 的 480 次/步 `gelu_highprecision_nv` +
   480 次/步 `gelu_bwd_highprecision_nv` 在 gelu 配置下**完全消失**，替换为
   FlagGems `gelu_none_kernel_rank_1` / `gelu_backward_none_kernel_rank_1`
   （`results/segments/gelu_r*.json` ops 表；微基准级探针
   `results/gelu-refresh/gelu-path-{native,gems}.json`）。
2. **同批前后对照（ms/步）**：GELU 类目 1,169.7 → 616.0（**-47.3%**）；逐调用
   前向 888.6→550.5 µs（1.61×）、反向 1,548.3→732.9 µs（2.11×）。其他 kernel
   类目零变化（GEMM 911.0→911.1、attention 323.2→323.1、cast_te 71.9→71.8）——
   **修复只动了该动的地方**。整步 wall -14.1%。
3. **路径对比与 tanh 决策项**：厂商 fp32 high-precision（native 口径）vs FlagGems
   bf16 none（现配置）vs FlagGems tanh 候选——真形状分布（10 行 × padded
   140–512）微测：tanh 估 289.0 vs none 294.6 ms/步，**差异 ~2%，tanh 在
   FlagGems 路径上不再是有效杠杆**（PR3 时代"tanh 再省 ~100 ms"是厂商 fp32
   路径的结论，本表将其关闭：无需改训练配方、无语义变更）。
   数据：`results/gelu_none_vs_tanh_realshapes.json`。
4. **数值验收**：固定 seed 6 步轨迹（`results/numerics/`）——native 两独立进程
   **逐步逐位一致**（loss 序列与 param/grad sha256 全同，过程内噪声下限=0）；
   gelu/all/gemm 相对 native 的逐步 loss 最大相对差 0.63–1.24%（bf16 双路径
   舍入量级），最终 loss 0.8036–0.8138 同带，**无非有限参数**，峰值显存
   26.714 GB 三配置一致。边界输入证据沿用 gelu-kunlunxin-pow 的
   `verify_gelu_boundary.py`/`test_gelu_boundary.py` 与 2026-11-11 边界对照
   （乘法替换仅 fp32 tanh 反向 @±3 相邻值 1 ulp，bf16/fp16 逐位一致）。
5. **残余差距（新发现，移交运行时/FlagGems 层）**：FlagGems gelu 在真入口的
   变长 shape 下 616 ms/步，而同 kernel 在固定 (2990,1024) 微基准为
   ~129 ms/步（480×(115+154) µs）——**4.8 倍 shape 处理开销**（按 shape
   autotune 的 config 选择 + 变长 M 对齐）。这是 GELU 未达 problem29
   "<150 ms/步"目标的全部残余；方向是把 autotune config 对变长训练做泛化
   （shape bucket 化），不是换 kernel。
6. **变长 shape 首遇编译停顿（新发现，接入边界）**：冷 Triton 缓存下，每个
   新 padded 长度首次命中触发 10–15 s 编译（trainer 15 步短跑中 8 步超 4 s，
   最大 15.0 s）；shape 集合有限（≤512 种），完整 epoch 内摊销，共享
   `TRITON_CACHE_DIR` 后续进程不再出现（all 配置即无停顿）。接入脚本应预热
   或预共享缓存，日志与数据见 `logs/trainer/gelu_rr1.log`。

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

### D. GEMM（P1）— ✅ env 有小收益、逐位一致；算法层无更多 headroom

1. **真入口 A/B**：GEMM 类目 911.0 → 772.0 ms/步（**-139.0 ms，-15.3%**，env
   逐位一致），整步 wall -2.3%。收益来自变长 M 分布（真入口每 batch M=10×padded，
   1,400–5,120 变动），厂商 fast algo 对该分布整体提速。
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
2. **稳定性**：每配置 3 独立进程，p50 报告、逐 rep 值在 `results/summary.json`；
   trainer 口径 12 个稳态步 p50/p95 与 min/max 见 §顶部交叉核对行；
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
- **FlagGems 变长 shape 开销**（GELU 残余 4.8×、首遇编译停顿）是接入层新
  边界，未在本轮修复，作为运行时/FlagGems 移交项登记。
- **FSDP2 多卡**：平台阻断 F-P800-1（XCCL 集合通信在 FSDP2 下停在任意一次
  all-gather，7 次触发 0 成功；单卡 DDP 全轮可用）——多卡通信分解 pending，
  待运行时层解决后按 `scripts/run_fsdp2.sh` 复测；多卡结论不外推。
- 微基准 tanh/none 对比使用固定分组近似真实长度分布（p25/p50/p90/max 混合），
  非逐 batch 采样。
- 共享主机测量纪律（PR3 历史）：本轮全部性能数据在安静窗口顺序采集（host
  load ≈3 由本任务自身构成）；首轮曾发生与卡 2 任务并发导致的 wall 污染，
  该批数据已废弃不用（保留在 `results/`（旧）目录外的 evidence 工作区，
  不进入本交付）。

## 复现命令

```bash
# 0) 数据与模型（一次性）
bash scripts/build_triplet_step1.sh /workspace/data-t2ranking   # qrels→pos、bm25→neg15（含确定性规则）
python3 scripts/assemble_triplet15.py /workspace/data-t2ranking \
        /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15
# 模型：BAAI/bge-m3 → /flagos-retrieval-models/bge-m3/bge-m3（HF_ENDPOINT=https://hf-mirror.com）

# 1) 单卡消融矩阵（每配置 3 独立进程；~2.5 h）
bash scripts/run_ablation_matrix.sh

# 2) 真实 trainer 交叉核对（torchrun 单卡，MAX_SAMPLES=600 → 15 optimizer steps）
MAX_SAMPLES=600 FS_TREE=/workspace/FlagScale-stock bash scripts/train_entry.sh native r1 results/trainer

# 3) GEMM sweep（每 env 独立进程）
bash scripts/run_gemm_sweep.sh

# 4) 数值轨迹（seed 42，6 步，跨配置对照）
bash scripts/run_numerics.sh

# 5) FSDP2 2 卡（安静窗口；single-trigger blocker 协议）
bash scripts/run_fsdp2.sh profile-native && bash scripts/run_fsdp2.sh native
bash scripts/run_fsdp2.sh profile-all    && bash scripts/run_fsdp2.sh all

# 6) GELU 路径探针与 tanh/none 对比、FlagGems 树微基准
python3 scripts/probe_gelu_path.py native out.json   # + gems 变体（见 scripts 头注）
python3 scripts/gelu_none_vs_tanh.py
```

环境变量基线：`KLX_USE_AUTOTUNE=0`、`XPU_ENABLE_PROFILER_TRACING=1`、
`XPU_CUPTI_ENABLE_DEVICE=0`（昆仑芯 torch.profiler 必需，problem29 §3.2）、
`TRITON_CACHE_DIR`/`FLAGGEMS_CACHE_DIR` 指向数据盘、`HF_HUB_OFFLINE=1`。
