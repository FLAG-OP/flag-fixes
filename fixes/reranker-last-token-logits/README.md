# reranker-last-token-logits

- ID：reranker-last-token-logits
- 状态：draft（未具备部署资格）
- 负责人：待定
- 上游 issue/PR：无（FlagScale 本地修复，尚未提交上游）
- 受影响对象：`flagscale/models/retrieval/qwen3_reranker.py::Qwen3RerankerModel.forward`，
  目标包 flagscale @ `89f342e690053e07dda845f1365089ec00464e09`（`feat/retrieval-flagos` 分支；
  该文件到分支头 `555a7d6a` 内容未变）
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

## 修复

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

## 验证入口（6 个命令，见 fix.yaml `validation.commands`）

```
tests/test_reranker_lmhead_dispatch.py   # CPU，2 层随机 Qwen3：lm_head 只收到 [rows, hidden]（基线 6/25 失败）
tests/test_reranker_fsdp2_step.py        # 真实 apply_fsdp2（CPU/gloo）：lm_head 分片、hook、梯度、optimizer（基线失败）
tests/repro_reranker_logits_alloc.py     # 单卡真实形状：61 GiB 常驻 + 碎片堆上的一步（基线 OOM 11.57 GiB）
# 2 卡 FSDP2 平台形态的自然复现（基线第 3 步 OOM）与修复树的尝试记录：results/e2e-fsdp2-2card-*longdoc*
tests/test_reranker_scores_equivalence.py  # 真实架构 bf16/fp32：分数与梯度对基线公式逐位相同
tests/bench_reranker_step.py             # batch 16/32 前向反向耗时与峰值显存
tests/e2e_reranker_train.py              # 原始 trainer + 真实 checkpoint + T2Ranking（需 $RERANKER_MODEL/$RERANKER_DATA）
```

关键数字（P800 单卡，详见 [验证记录](reports/validation.md)）：

| 对照 | 基线 | 修复 |
|---|---:|---:|
| LM head 单次输出，batch 32 | 11.5714 GiB | 0.0181 GiB |
| 前向 / 反向峰值 allocated，batch 32（权重 15.25 GiB） | 40.80 / 54.99 GiB | 29.94 / 40.61 GiB |
| 前向 / 反向耗时，batch 32 | 3.459 / 10.129 s | 3.247 / 9.760 s |
| 原始 trainer 单卡 bf16，batch 16，5 步 reserved 峰值 | 92.76 GiB | 79.64 GiB |
| 分数 / 梯度对基线公式（bf16 与 fp32，5 例） | — | max\|Δ\| = 0.0 |
| 真实 trainer 5 步 loss、validation、test loss | 0.539403 / 0.461096 / 0.329392 | 逐位相同 |
| `flag-fixes verify`（修复树，6 个声明命令，干净提交 `57e2c756`） | — | passed，6/6 rc=0 |

## 边界

- `deployable=false`、`hardware_status=unverified`、`owner=null`。
- **基线失败已在平台形态下自然复现**：2 卡 FSDP2 + fp32 master + batch 32 + 真实权重，数据换成
  每对都截断到 640 的长文档子集，基线第 3 步反向在两个 rank 同时报
  `Tried to allocate 11.57 GiB … 20.77 GiB is free … 71.66 GiB is allocated by PyTorch`
  （平台记录 11.57 / 20.90 / 71.68）。
- **修复树在同一配置下没有完整跑通**：本机 XCCL 集合通信间歇性卡死（已登记的 F-P800-1）。
  长文档子集上 3 次尝试分别停在第 2、3、1 步（走完的 step 与基线 loss 逐位相同）；
  常规数据上走完 3 步 + validation 后卡在收尾的集合通信。修复对照的干净结果来自单卡构造堆
  （repro：基线 OOM → 修复跑通）与真实 `apply_fsdp2` 的 2 卡单元测试（修复树通过一次）。
- 平台配置的 batch 16 / GA 1 是为这次 OOM 设的；修复后是否有余量回到 32 或 accum=2，
  需在平台 2 卡环境实测后再改配置。
- 本机还观察到与 logits 无关的两项显存开销（未修）：厂商 `clip_grad_norm_` 对 bf16 梯度
  物化约 35 GiB 临时张量；AdamW step 有 15.25 GiB 临时张量。
