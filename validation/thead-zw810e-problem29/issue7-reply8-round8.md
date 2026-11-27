## 第八轮复审回应：原对象核查清单 + 原入口首轮 native 基线已交付（PR #8 commit `28de472`）

### 交付一：原对象核查清单（`validation/thead-zw810e-problem29/original_object_checklist.md`）

按审阅四项逐项补齐（完整表格见清单文档），要点：

- **来源**：原问题仓库 https://github.com/wxj630/FlagScale/tree/feat/retrieval-flagos（problem.md 首行给定，HEAD `6862134`）；FlagScale 基线 commit flagos-ai/FlagScale @ `89f342e6`（已验证提交存在）；训练入口 `flagscale/train/train_retrieval.py`——**已下载并验证 sha256 = 5ee5757…11df 与 fixes/adamw-fused/fix.yaml `before_sha256` 完全一致**；配置 `examples/retrieval/conf/train/bge_m3.yaml`；profiling 脚本 `profile_step_segments.py`/`profile_op_micro.py`；权重 BAAI/bge-m3（hf-mirror 下载 2,269,966,336 B）；数据 sentence-transformers/t2ranking `triplet-15/`（11 parquet 1.1 GB，schema/行数与 FlagScale 文档一致）；原始命令已执行（本机覆盖 conf：仅 model_path/data.path 指向本机下载件）。x-benchmark 容器入口 `/home/wuyiqiang/x-benchmark` 不在本机（阻碍②）。
- **对象**：BGE-M3 = XLM-R large（24 层/hidden 1024/vocab 250002，bf16，max_length 512）+ CLS/sparse/colbert 头 + 对比损失（temperature 0.05）+ AdamW lr 2e-5 + 真实 T²-Ranking triplet-15（batch 10 × accum 4 = 200 texts/step）。与自建小模型（2 层/fp32/随机 token/MSE）及昆仑芯重建负载（XLM-R large **随机权重**）的差异已按表格列明——本机本轮用**预训练权重 + 原数据**，差异表中同时列明未下载 T²-Ranking 官方原始 parquet（用 conf 同名镜像 triplet-15，schema/行数对齐文档）。
- **检查与尝试**：已读入口/模型/数据/配置/文档各文件；实际执行 sha 验证、权重与数据下载、冒烟（step0 11,445 ms 含 JIT）、正式基线两遍（可复现）；未尝试项逐条标注（官方 parquet 逐字节同源、3-epoch 收敛训练、FSDP2 多卡、x-benchmark 容器、FlagGems full-enable=blocked）。
- **具体阻碍**：①磁盘 df=100%（实测可写，已落盘 3.4 GB——不阻塞本轮）；②x-benchmark 容器不在本机（影响整模型口径；需平台提供）；③官方原始 parquet 未下载（影响逐字节同源；schema 已对齐）；④FlagGems full-enable blocked（F-1/F-2 两次单次触发出证，影响 FlagGems 配置）；⑤FSDP2 多卡（本机单卡分配）。每项附影响配置与动作归属。

### 交付二：原对象首轮 native 基线（原入口脚本实际执行，exit 0，两遍可复现）

命令（本机覆盖 conf：model_path=/tmp/bge-m3-hf、data.path=/tmp/t2ranking/t2ranking/triplet-15，其余与原配置一致）：
`python examples/retrieval/profile_step_segments.py --conf examples/retrieval/conf/run/bge_m3_ppu.yaml --steps 3 --skip 2 --prof-steps 3 --device cuda`

| 项 | 第一遍 | 第二遍（复现） |
|---|---:|---:|
| 整步 wall | 2,131.9 ms | 2,123.7 ms |
| device compute (fwd+bwd) | 2,029.5 ms（95.2%）| 2,000.2 ms（94.2%）|
| kernel（aten 层）| 1,906.4 ms（89.4%）| 1,908.7 ms（89.9%）|
| **GEMM（aten::mm+addmm）** | **1,012.5 ms（53.1%）**，mm **5,776 次** | **1,013.8 ms（53.1%）** |
| attention（efficient fwd+bwd） | 573.5 ms（30.1%）| 574.1 ms（30.1%）|
| GELU | 32.6 ms（1.7%）| 32.6 ms（1.7%）|

- **aten::mm 5,776 次/step 与 problem29 原始 PPU 归因逐次一致**；GEMM 主导、attention 次之、GELU 非瓶颈——原对象 native 基线在本机复现了原始归因（problem.md §4.3）。
- attention 走 efficient 后端的**机制已由 probe9b 定位**：bool padding mask 下 flash "No available kernel" → efficient（4.5× 慢于无掩码 flash）/math（9.5×）。
- GEMM 三级测量（probe9）：bf16 真实 shape 89–124 TFLOP/s（健康）、fp32 ~31 TF——下一步 GEMM 调优方向（tile/workspace/layout/autotune/累加精度）按审阅排序在原对象上继续。

### 与既有工作的关系

PR #8 既有内容（AdamW/Graph/小模型/归因工具）按第六轮口径维持"候选观测/组件级"，不与本节原对象数字混用；F-1/F-2 保持 blocked；72/39/15 精度基线与 `draft / deployable=false / hardware_status=unverified` 不变。GEMM/attention/copy-cast 的优化实施按审阅顺序推进，本基线为其提供同对象对照起点。
