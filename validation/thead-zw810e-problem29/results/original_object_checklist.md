# 原对象核查清单与 native 基线（issue #7 round-8，复审 6887760539）

日期：2026-11-15 · 平台：平头哥 ZW810E 单卡 · HEAD：PR #8 分支

## 一、来源（真实入口，非本地路径）

| 项 | 值 | 验证方式 |
|---|---|---|
| 原问题仓库 | https://github.com/wxj630/FlagScale/tree/feat/retrieval-flagos（problem.md 首行给定）| 分支 HEAD `6862134` |
| FlagScale 基线 commit | flagos-ai/FlagScale @ `89f342e690053e07dda845f1365089ec00464e09`（"docs(retrieval): record the real 3-epoch wall clock, and add a three-platform watcher"，2026-11-04）| `gh api commits/89f342e6` |
| 训练入口 | `flagscale/train/train_retrieval.py`（1,325 行）| 已下载；**sha256 = 5ee5757…11df，与 fixes/adamw-fused/fix.yaml `before_sha256` 完全一致**——adamw-fused 的目标文件即此文件 |
| 问题描述 | problem29（本目录 problem.md）：三层 profiling 方法 + PPU 归因（GEMM 44.3% / attention 20.4% / copy-cast 14.7% / GELU 0.9× 更快）| 已核对，本机复现见下 |
| profiling 脚本 | `examples/retrieval/profile_step_segments.py`（阶段拆解）、`profile_op_micro.py`（算子微基准）| 已读取并**实际执行**前者 |
| 配置 | `examples/retrieval/conf/train/bge_m3.yaml`（batch 10、accum 4、bf16、max_length 512、lr 2e-5、T²-Ranking triplet-15）| 已拷贝为本机覆盖版 `conf/run/bge_m3_ppu.yaml`（仅改 model_path/data.path 为本机下载路径）|
| 模型权重 | BAAI/bge-m3（hf-mirror 下载）| `pytorch_model.bin` 2,269,966,336 B + `sparse_linear.pt` + `colbert_linear.pt` + tokenizer；加载成功 |
| 训练数据 | sentence-transformers/t2ranking `triplet-15/`（**11 个 parquet，1.1 GB，schema = anchor/positive/negative_1..15，首 shard 30,920 行**——与 FlagScale 文档"11 个文件, 340,111 行"一致）| 已下载全部 11 个分片；pyarrow 读取验证 |
| 原始运行命令 | `python examples/retrieval/profile_step_segments.py --conf examples/retrieval/conf/train/bge_m3.yaml --steps 3 --skip 2 --prof-steps 4 --device cuda`（problem.md §七）| 已执行（本机覆盖 conf）|
| x-benchmark 容器入口 | `/home/wuyiqiang/x-benchmark` + `run_model.py`（zeros-7181 problem.md 引用）| **不在本机**（属另一训练机），见"阻碍 ②" |

## 二、对象结构（BGE-M3 训练对象）与差异

| 维度 | 原对象（本机已运行） | 自建小模型（PR #8 阶段探针） | 昆仑芯重建负载（PR #3） |
|---|---|---|---|
| 骨干 | BGE-M3 = XLM-R large（24 层、hidden 1024、vocab 250002、bf16）| gated MLP + 2 层 attention（D256/H512，fp32）| XLM-R large 随机权重 |
| 头 | CLS pooling（dense）+ sparse_linear + colbert_linear | 无 | BGE-M3 重建配方 |
| 数据 | T²-Ranking **triplet-15 真实 parquet**（length-sort 窗口 256，真实 padding 分布）| 随机 token（标注为诊断）| 其自建数据 |
| batch | 10 × accum 4 = 200 texts/step | B8 S128 ×8 步 | 其重建 batch |
| loss | 对比损失（temperature 0.05，loss_for_batch）| MSE | 其重建 loss |
| optimizer | AdamW（build_optimizer，lr 2e-5、wd 0.01、cosine）| AdamW stock/fused | AdamW fused |
| 并行 | 原验收口径含 FSDP2 多卡 | 单卡 | 单卡重建 |

差异要点：本机本轮**首次**以原对象结构 + 原数据 + 原入口脚本在 ZW810E 运行；与昆仑芯重建负载相比使用**预训练权重**（其用随机权重）与同一 triplet-15 数据源。

## 三、检查与尝试（事实清单）

**已读取**：`train_retrieval.py`（init_distributed/build_optimizer/autocast_context）、`models/retrieval/bge_m3.py`（from_pretrained = AutoModel local_files_only）、`train/datasets/retrieval.py`（parquet + hash 分桶）、`docs/retrieval/01-bge-m3.md`（配方与数据可得性）、`docs/retrieval/09-ppu.md`、`ENVIRONMENTS.md`、`profile_step_segments.py`（计时/分层两阶段设计）。

**已执行（ZW810E，CUDA_VISIBLE_DEVICES=0，transformers 4.57.0 / pyarrow 20.0 / omegaconf 2.3.0）**：
1. 入口文件 sha256 验证（= fix.yaml before）✓
2. 权重下载 2,269,966,336 B（hf-mirror）✓；triplet-15 全 11 分片 1.1 GB ✓
3. 冒烟（--steps 1 --no-profile）：真实数据加载 ✓、step0 = 11,445 ms（含 JIT/首编译）✓
4. **正式 native 基线**（--steps 3 --skip 2 --prof-steps 3，exit 0）：
   - **整步 wall = 2,131.9 ms**（3 步中位；200 texts/step）
   - forward 707.5 ms（33.2%）/ backward 1,380.5 ms（64.8%）/ clip+optimizer 20.3 ms（1.0%）；device compute (fwd+bwd) = 2,029.5 ms（95.2%）；kernel（aten 层）= 1,906.4 ms（89.4% of wall）
   - **aten 归因（每步）**：GEMM **1,012.5 ms（53.1%）**（aten::mm 675.3 × **5,776 次**——与 problem.md"PPU mm 调用次数与 A100 完全相同（5,776 次）"**逐次一致**；addmm 337.2）；attention **573.5 ms（30.1%）**（`_efficient_attention_forward/backward`——带掩码路径）；pointwise 91.2；copy 72.5；layer_norm 39.9；**GELU 32.6 ms（1.7%）**
   - 归档：`results/ppu_bge_m3_segments_native.json`、`results/ppu_bge_m3_aten_classes.json`（原始 profiler 全 175 条目保留）

**与 problem29 原始 PPU 归因对照（§4.3）**：

| 类别 | problem.md 原始 PPU | 本机（本次） | 判定 |
|---|---|---|---|
| GEMM | 1,158.6 ms，占差距 44.3% | 1,012.5 ms（53.1% of aten）；mm 5,776 次一致 | **一致复现** |
| attention | 553.5 ms，20.4%，"SDPA 落 math/无融合内核" | 573.5 ms（带掩码→efficient 后端；probe9b 证实 **bool mask 下 flash "No available kernel"**）| **机制已定位** |
| copy/cast | 248.8 ms，14.7% | 72.5 ms copy_ + pointwise 部分（合计 ~5%）| 本机更低（布局/数据差异），非主要项 |
| GELU | 36.8 ms，0.9× 更快 | 32.6 ms（1.7%）| **一致：非瓶颈**（本机 gelu 微基准与 AdamW/Graph 工作对 PPU problem29 非优先项）|

**未尝试（明确标注）**：T²-Ranking 官方原始数据集（THUIR/T2Ranking）下载（triplet-15 已由 sentence-transformers 镜像提供，同 schema）；真实 3-epoch 收敛训练（仅 3 计时步基线）；FSDP2 多卡口径（本机单卡分配）；x-benchmark 容器（不在本机）；FlagGems full-enable 在原对象上的运行（F-2 matmul backward 挂起使 bf16 训练主路径被阻断，两次单次触发出证）。

## 四、具体阻碍（逐项）

| # | 阻碍 | 实际证据 | 影响配置 | 解决动作 / 责任 |
|---|---|---|---|---|
| ① | 磁盘 df=100%（0 avail）| 实测仍可写（1.1 GB 瞬时成功；本轮已落盘 3.4 GB 权重+数据）| 大规模数据/多 checkpoint | 已完成必要下载；后续大规模数据需平台扩容（平台）|
| ② | x-benchmark 容器与 `/data/ppu-benchmark` 不在本机 | 路径属于另一训练机（zeros-7181 problem.md）；本机无 run_model.py | x-benchmark 整模型口径 | 需平台提供容器/入口或等效环境（平台）|
| ③ | T²-Ranking 官方原始集未下载 | 已用 sentence-transformers/t2ranking triplet-15（conf 同名同 schema，文档引用同一来源）| 仅影响"官方原始 parquet"与镜像 parquet 的逐字节同源性（schema/行数已对齐文档）| 如需逐字节同源：下载官方集并按 FlagScale 流程重建（后续）|
| ④ | FlagGems full-enable blocked | F-1（SDPA 训练形状挂起）/F-2（matmul backward 挂起）两次单次触发出证 | full-enable 原对象训练 | 隔离复验待设备/驱动状态恢复（平台 + 本仓库后续）|
| ⑤ | FlagScale FSDP2 多卡 | 本机单卡分配（PPUS=6,7）| 多卡口径 | 多卡环境就绪后按 §5.5 口径单列通信（后续）|

## 五、结论

原对象 native 基线已在 ZW810E 建立（真实入口/配置/数据/权重），归因与 problem29 原始 PPU 归因一致：**GEMM 主导（53.1%，mm 5,776 次逐次一致）、attention 带掩码路径（30.1%，flash 因 bool mask 不可用 → efficient/math，机制已由 probe9b 证实）、copy/cast 次要、GELU 非瓶颈**。下一步按审阅排序：真实 shape GEMM 调优（tile/workspace/layout/autotune/累加精度）→ SDPA 掩码路径（等长分桶去 mask 或融合内核）→ copy/cast；native/stock、native/fused、FlagGems 配置的优化前后对照在原对象上继续。
