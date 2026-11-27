# reranker-last-token-logits

- ID：reranker-last-token-logits
- 状态：draft（deployable=false；单卡复现、回归、dispatch、benchmark 已跑，2 卡平台形态下
  基线失败已自然复现，修复后完整跑通未验证）
- 负责人：待填写（owner 保持 null，不虚构）
- 上游 issue/PR：无（FlagScale 本地修复，尚未向上游 `wxj630/FlagScale` 提 PR；问题来源是 problem26）
- 受影响算子、软件层、后端：`Qwen3RerankerModel.forward`（FlagScale **模型适配层**，不是算子），
  文件 `flagscale/models/retrieval/qwen3_reranker.py`，目标包 flagscale @
  `89f342e690053e07dda845f1365089ec00464e09`（`feat/retrieval-flagos`；该文件到分支头 `555a7d6a` 未变）；
  代码与厂商无关，设备证据只来自 KunlunXin P800（后端 `kunlunxin`）
- 问题与修复后行为：Qwen3-Reranker-8B 在昆仑芯 P800 上以 FSDP2 + `batch_size=32` 训练时，
  每步向分配器申请一整块 **11.57 GiB** 的连续显存；显存碎片化后申请失败
  （`Tried to allocate 11.57 GiB ... 20.90 GiB free`）。这块显存是适配器让因果 LM
  对**每个位置**算出的词表 logits（64 对 × 640 token × 151,669 词表 × bf16），而适配器
  每条序列只用最后 1 行。修复后 LM head 只投影被打分的那一行：
  输出 11.57 GiB → 18.5 MiB，同一位置、同一权重，分数与梯度与原公式**逐位相同**。

## 一句话版本

**`self.backbone(**features)` 会算出整条序列的词表 logits 再只取一位；改成先跑
`self.backbone.model(...)`、取末位 hidden、再过 `self.backbone.lm_head(...)`，
那块 11.57 GiB 的申请就没了。**

```
修复前: Tried to allocate 11.57 GiB. GPU 0 has a total capacity of 96.00 GiB of which 20.63 GiB is free.
        Of the allocated memory 73.51 GiB is allocated by PyTorch ...   (results/repro-base-*.json)
修复后: 同一个碎片堆上同一步跑通，前向 + 反向峰值 +25.36 GiB          (results/repro-fixed-*.json)
```

## 问题是怎么回事

平台记录（`examples/retrieval/conf/train/qwen3_reranker_p800.yaml`）把 11.57 GiB 归因于
"lm_head 的 all-gather"，并把 P800 的 batch 从 32 降到 16 规避。对账结果是：

| 量 | 值 |
|---|---|
| 报错里的请求 | 11.57 GiB |
| 整序列 logits | 64 × 640 × 151,669 × 2 B = **11.5714 GiB**（实测 `outputs.logits` 形状） |
| lm_head 的 all-gather | 621,236,224 × 2 B = **1.157 GiB**（trainer 日志 `gathers 1185 MiB`） |
| 报错时 PyTorch 已分配 | 71.68 GiB = step 1 后常驻 61.02 GiB + 当时存活的检查点激活 |

容量并不缺（71.68 + 11.57 < 96 GiB）；失败是因为 P800 没有 expandable segments
（`expandable_segments not supported on this platform`），空闲的 20.9 GiB 拼不出一整块。
同一配置（2 卡 FSDP2、真实权重、batch 32）在本机自然复现：基线第 3 步反向报
`Tried to allocate 11.57 GiB … 20.77 GiB is free … 71.66 GiB is allocated by PyTorch`，
失败点是 lm_head 反向物化整序列 logits 的梯度。完整推理见 [诊断报告](reports/diagnosis.md)。

## 修复思路

```diff
-        outputs = self.backbone(**features, return_dict=True, **kwargs)
-        logits = outputs.logits
-        positions = ...
-        final_logits = logits[rows, positions]
+        outputs = self.backbone.model(**features, return_dict=True, **kwargs)
+        final_hidden = last_token_pool(outputs.last_hidden_state, features["attention_mask"])
+        # Call the module rather than F.linear so FSDP's hooks on lm_head run.
+        final_logits = self.backbone.lm_head(final_hidden)
```

- `last_token_pool` 是 `transformer_encoder.py` 里已有的函数（embedding 模型在用），
  取每条序列最后一个非 pad 位置，与原来的下标逐行相同。
- lm_head 在 FSDP2 下是单独分片的模块（`iter_large_leaf_modules` 的 "oversized leaf"），
  调用模块本身保证它的 all-gather / reshard hook 照常触发。
- 只动非 score-head 分支；score-head 分支、Megatron 路径、训练循环都不变。
- 代码路径与厂商无关；设备证据只来自 KunlunXin P800。

完整 diff 见 `patches/reranker-last-token-logits.patch`。目标文件 hash：

| 文件 | before sha256 | after sha256 |
| --- | --- | --- |
| `flagscale/models/retrieval/qwen3_reranker.py` | `50147024…` | `dd34c0dd…` |

## 怎么用：单项补丁安装

前置：FlagScale `feat/retrieval-flagos` @ `89f342e690053e07dda845f1365089ec00464e09`，
目标文件 hash 与上表一致。验证硬件是昆仑芯 P800 OAM（96 GiB）；本机环境与实际执行过的命令见
[验证记录](reports/validation.md)。

本目录提供独立的 patch 和测试脚本，**不是独立 pip 包**。`draft` 表示验收材料尚未完整，
`deployable=false` 表示统一工具的 `apply` 暂不开放；这不妨碍在匹配基线的测试工作树中手工
应用补丁。PR 合并不会自动改变这两个状态。

FlagScale 仓库布局是根下 `flagscale/` 包（没有顶层 `src/`），所以 `PYTHONPATH` 指向仓库根。
以下均为参数化示例，不是某台机器的实际路径：

```bash
MANAGER=/path/to/flag-fixes   # 本仓库根目录（含 catalog/、fixes/、src/）
TARGET=/path/to/FlagScale     # FlagScale 源码根（含 flagscale/ 包）

# 0) 核对目标仍是修复前内容
sha256sum "$TARGET/flagscale/models/retrieval/qwen3_reranker.py"   # 上表 before hash

# 1) 预演：确认 patch 能干净应用（不落盘）
PYTHONPATH="$MANAGER/src" python3 -m flag_fixes --root "$MANAGER" \
  check reranker-last-token-logits --target "$TARGET"

# 2) 手工应用最小 patch（本修复未获部署资格，flag-fixes apply 会按设计拒绝）
git -C "$TARGET" apply "$MANAGER/fixes/reranker-last-token-logits/patches/reranker-last-token-logits.patch"
sha256sum "$TARGET/flagscale/models/retrieval/qwen3_reranker.py"   # 上表 after hash

# 3) 单项验证：退出码 0 才算通过；缺设备时退出码 2，不算通过
#    P800 上设备脚本需要与平台启动脚本相同的取值（XBLAS 否则会选到一个无法 JIT 的 bmm kernel）
export XMLIR_BMM_DISPATCH_VALUE=0
cd "$MANAGER/fixes/reranker-last-token-logits"
PYTHONPATH="$TARGET" python3 tests/test_reranker_lmhead_dispatch.py            # CPU
PYTHONPATH="$TARGET" python3 tests/test_reranker_fsdp2_step.py                 # CPU（gloo）
PYTHONPATH="$TARGET" python3 tests/repro_reranker_logits_alloc.py              # 需一张 ≥ 90 GiB 的空闲卡
PYTHONPATH="$TARGET" python3 tests/test_reranker_scores_equivalence.py         # 需设备
PYTHONPATH="$TARGET" python3 tests/bench_reranker_step.py --iters 5           # 需设备
# 端到端另需真实 checkpoint 与 T2Ranking triplet-15 parquet：
RERANKER_MODEL=/path/to/Qwen3-Reranker-8B RERANKER_DATA=/path/to/triplet-15 \
  PYTHONPATH="$TARGET" python3 tests/e2e_reranker_train.py \
  --nproc 1 --parallel ddp --master-dtype bf16 --batch 16 --min-steps 5 --max-grad-norm 0 --no-save

# 或由管理工具一次跑完 6 个声明命令（要求目标已是修复后内容；同样需要上面两个环境变量）
PYTHONPATH="$MANAGER/src" python3 -m flag_fixes --root "$MANAGER" \
  verify reranker-last-token-logits --target "$TARGET" --python python3

# 4) 撤销手工补丁：先检查反向应用是否可行。手工安装没有管理工具的备份与安装记录，
#    不能用 flag-fixes rollback 撤销
git -C "$TARGET" apply --reverse --check "$MANAGER/fixes/reranker-last-token-logits/patches/reranker-last-token-logits.patch" && \
  git -C "$TARGET" apply --reverse "$MANAGER/fixes/reranker-last-token-logits/patches/reranker-last-token-logits.patch"
```

不要对同一目标重复执行手工 `git apply` 和 `flag-fixes apply`。

## 目录结构

```
├── README.md                          # 本文
├── fix.yaml                           # 清单（schema_version 1）
├── WORKFLOW.md                        # 8 步执行清单及证据
├── patches/reranker-last-token-logits.patch   # 最小 diff（1 文件 +15/−6），git apply 用
├── src/flagscale/models/retrieval/qwen3_reranker.py   # 修复后完整文件快照，与 patch 结果逐字节一致
├── tests/
│   ├── _reranker_common.py            #   共享工具（随机权重、桩分词器、lm_head 探针）
│   ├── test_reranker_lmhead_dispatch.py        #   lm_head 只收到 [rows, hidden]（level: dispatch，CPU）
│   ├── test_reranker_fsdp2_step.py + _fsdp2_worker.py   # 真实 apply_fsdp2 下的分片、钩子、梯度（level: dispatch）
│   ├── repro_reranker_logits_alloc.py          #   61 GiB 常驻 + 碎片堆上的一步（level: repro，基线必 OOM）
│   ├── test_reranker_scores_equivalence.py     #   分数与梯度对原公式（level: regression）
│   ├── bench_reranker_step.py                  #   前后耗时与峰值显存（level: benchmark）
│   ├── e2e_reranker_train.py + _e2e_trainer_entry.py   # 原始 trainer（level: e2e，需真实权重与数据）
│   ├── build_long_parquet.py          #   构造长文档子集（2 卡自然复现用）
│   ├── probe_p800_allocator.py        #   诊断：分配器性质（非验证命令）
│   ├── probe_xccl_timing.py           #   诊断：集合通信卡死的时序探索（非验证命令）
│   └── fixtures/qwen3_reranker_8b_config.json   # 发布版 config.json 原文件
├── benchmarks/README.md               # 前后耗时、峰值显存与数值精度
├── reports/
│   ├── diagnosis.md                   #   根因、方案、影响范围
│   └── validation.md                  #   环境、命令、前后结果、2 卡尝试、not-run 清单
└── results/                           # 逐命令 JSON / 日志、verify 原始输出
```

## 验证结果摘要

设备数据 2026-11-26/27 采于本机 P800 OAM（torch 2.9.0+cu129，transformers 4.57.6），
逐命令的精确命令行与证据文件见 [验证记录](reports/validation.md)。

| 项目 | 修复前 | 修复后 |
| --- | --- | --- |
| `test_reranker_lmhead_dispatch.py`（CPU，25 例） | exit 1，6 例失败：lm_head 收到 `[rows, seq, hidden]` | exit 0，25/25 |
| `test_reranker_fsdp2_step.py`（真实 `apply_fsdp2`，CPU/gloo） | exit 1：lm_head 输出 `[16, 258, 151669]` | exit 0：`[16, 151669]`，分片与钩子正常 |
| `repro_reranker_logits_alloc.py`（单卡碎片堆上的一步） | exit 1：`Tried to allocate 11.57 GiB … 20.63 GiB is free` | exit 0，峰值 +25.36 GiB |
| `test_reranker_scores_equivalence.py`（真实架构，bf16 / fp32） | exit 0（基线即参考） | exit 0，5 例 max\|diff\| = 0.0 |
| batch 32 前向 / 反向峰值 allocated（权重 15.25 GiB） | 40.80 / 54.99 GiB | 29.94 / 40.61 GiB |
| batch 32 前向 / 反向耗时 | 3.459 / 10.129 s | 3.247 / 9.760 s |
| 真实 trainer 单卡 bf16 batch 16，reserved 峰值 | 92.76 GiB | 79.64 GiB |
| 同上，5 步 loss / validation / test loss | 0.539403 / 0.461096 / 0.329392 | 逐位相同 |
| 真实 trainer 2 卡 FSDP2、batch 32、长文档 | 第 3 步反向 OOM（与平台记录逐项吻合） | 第 4 次尝试走完 3 步，**没有一次干净退出**（XCCL 卡死） |
| `flag-fixes validate` / `check` / `plan` | — | exit 0 |
| `flag-fixes verify`（修复树，6 个声明命令） | — | passed，6/6 rc=0（干净提交 `c2aef972`） |

判定与覆盖边界：脚本只有退出码 0 才算通过，缺设备或缺 checkpoint/数据一律退出码 2。
普通 CI（无设备 runner）只执行 `flag-fixes validate`、`python -m pytest -q` 和本地 fix 脚本的
语法检查（`python -m compileall -q fixes`），**不**执行这些设备脚本；
语法检查通过 ≠ 入口执行通过 ≠ 硬件验证通过。全部测试由 AI 代理自动执行，没有人工复核。

## 已知边界（不装完美）

- **平台验收未做**：`hardware_status` 保持 `unverified`，`deployable=false`、`owner=null`，
  `flag-fixes apply` 会按设计拒绝。
- **基线失败已在平台形态下自然复现**：2 卡 FSDP2 + fp32 master + batch 32 + 真实权重，数据换成
  每对都截断到 640 的长文档子集，基线第 3 步反向在两个 rank 同时报
  `Tried to allocate 11.57 GiB … 20.77 GiB is free … 71.66 GiB is allocated by PyTorch`
  （平台记录 11.57 / 20.90 / 71.68）。
- **修复树在同一配置下没有一次干净退出**：本机 XCCL 集合通信间歇性卡死（已登记的 F-P800-1）。
  长文档子集上 4 次尝试：3 次分别停在第 2、3、1 步，第 4 次走完 3 步——包括基线 OOM 的那一步——
  后停在 validation；走完的 step 与基线 loss 逐位相同。常规数据上走完 3 步 + validation 后卡在
  收尾的集合通信。修复对照的干净结果来自单卡构造堆（repro：基线 OOM → 修复跑通）与真实
  `apply_fsdp2` 的 2 卡单元测试（修复树通过一次）。
- **平台配置的 batch 16 / GA 1 是为这次 OOM 设的**：修复后是否有余量回到 32 或 accum=2，
  需在平台 2 卡环境实测后再改配置；平台注释里"11.57 GiB 是 lm_head all-gather"的归因待更正。
- **只改了适配层一处分支**：厂商 `clip_grad_norm_` 对 bf16 梯度会物化约 35 GiB 临时张量，
  AdamW step 有 15.25 GiB 临时张量——这两项与本修复无关，只记录未修。A100 / PPU / 昇腾上的
  显存收益、评测入口 `evaluate_retrieval.py` 的推理显存均未测。
- **证据文件的来源与可追溯性**：逐命令的 JSON / 日志采集时修复目录尚未提交；当时的脚本与提交
  版本逻辑相同，只多两个后来删掉的未使用辅助函数（复现脚本另有一处说明文字后来改过）。
  `results/verify-2026-11-27.json` 是干净提交 `c2aef972` 上的重跑结果（`source_dirty=false`），
  其后的提交只改文档与结果索引。工具原始输出含本机绝对路径，未做脱敏或改写。
