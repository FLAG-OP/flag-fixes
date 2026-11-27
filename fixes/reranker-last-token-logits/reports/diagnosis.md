# 诊断报告：reranker-last-token-logits

## 1. 原始问题与预期行为

problem26：

> 【检索】qwen3-reranker-8b 在昆仑芯上运行，会因为显存碎片化无法分配连续显存，导致显存爆炸
> （来源 `wxj630/FlagScale` 分支 `feat/retrieval-flagos`）

平台在 `examples/retrieval/conf/train/qwen3_reranker_p800.yaml` 里留下的现场记录：

```
# Why 16 (measured 2026-11-20): FSDP2's all-gather of lm_head asks for a single
# 11.57 GiB CONTIGUOUS block. The vendor allocator does not support
# expandable_segments, so batch 32 died with
#   Tried to allocate 11.57 GiB. 96.00 GiB total, 20.90 GiB free,
#   71.68 GiB already allocated by PyTorch
# -- 20.9 GiB free but not as one block. That is fragmentation, not capacity,
# a larger card would not help.
```

运行形态（`run-p800-fsdp2-2card.sh` + `qwen3_reranker.yaml`）：Qwen3-Reranker-8B 全参微调，
2 卡 FSDP2、fp32 master、bf16 autocast、梯度检查点、`max_length=640`、`num_negatives=1`，
共享配置 `batch_size=32`。平台规避办法是把 P800 的 batch 降到 16，显存仍贴着
95/96 GB（`docs/retrieval/24-week-summary-2026-11-13-to-20.md`）。

预期行为：batch 32 这一步不再提出 11.57 GiB 的连续请求；同一位置、同一权重给出同样的分数。

## 2. 数字对账

| 量 | 取值 | 来源 |
|---|---|---|
| 报错里的请求 | 11.57 GiB | 平台记录 |
| batch 32 展开后的序列数 | 64 对（32 正 + 32 负） | `reranker_loss` 的展开规则 |
| 整序列 logits | 64 × 640 × 151,669 × 2 B = **11.5714 GiB** | 实测 `outputs.logits` shape `(64, 640, 151669)` bf16（`results/repro-base-*.json` part1） |
| lm_head 的 all-gather | 621,236,224 × 2 B = **1.157 GiB** | trainer 日志 `oversized leaf: backbone.lm_head gathers 1185 MiB in torch.bfloat16` |
| 报错时 PyTorch 已分配 | 71.68 GiB | 平台记录 |
| step 1 后的常驻显存 | 61.02 GiB（fp32 master 15.25 + fp32 梯度分片 15.25 + AdamW 两阶矩 30.5） | 实跑 2 卡 FSDP2（`results/e2e-fsdp2-2card-*.json`） |

三点结论：

1. **11.57 GiB 是整序列 logits，不是 all-gather。** lm_head 的 all-gather 只有 1.157 GiB；
   根组（embed_tokens + lm_head）合计 2.31 GiB。11.57 恰是 1.157 的 10 倍，
   且与 `64 × 640 × vocab × 2 B` 逐字节相等。
2. **71.68 GiB 对得上真实常驻量。** 61.02 GiB 常驻，加上 LM head 执行时仍存活的
   梯度检查点激活约 10.7 GiB，正是 71.68。
3. **容量并不缺。** 71.68 + 11.57 = 83.25 GiB < 96 GiB；请求失败是因为空闲的 20.9 GiB
   不连续，这正是问题标题所说的"碎片化"。

## 3. 根因与证据

### 3.1 适配器让因果 LM 对每个位置算词表投影，却只用一行

`flagscale/models/retrieval/qwen3_reranker.py::Qwen3RerankerModel.forward`（基线）：

```python
outputs = self.backbone(**features, return_dict=True, **kwargs)   # logits: [64, 640, 151669]
logits = outputs.logits
...
final_logits = logits[rows, positions]                            # 只留每条序列的最后一个非 pad 位置
scores = final_logits[:, self.true_token_id] - final_logits[:, self.false_token_id]
```

40,960 行里用到的只有 64 行（0.16%），其余全部丢弃。反向传播时 advanced indexing 的
梯度是 `zeros_like(logits)` 再 scatter，于是反向又出现一个同形状的 11.57 GiB 张量。

单卡、真实架构、bf16 autocast + 梯度检查点、长度 640 的实测（`results/bench-*-*.json`，
权重 15.25 GiB）：

| batch（对数） | 树 | 前向峰值（超出权重） | 反向峰值（超出权重） | LM head 单次输出 |
|---|---|---:|---:|---:|
| 32（64） | 基线 | 25.54 GiB | 39.74 GiB | 11.5714 GiB |
| 32（64） | 修复 | 14.69 GiB | 25.36 GiB | 0.0181 GiB |
| 16（32） | 基线 | 13.35 GiB | 21.61 GiB | 5.7857 GiB |
| 16（32） | 修复 | 7.42 GiB | 19.69 GiB | 0.0090 GiB |

### 3.2 P800 分配器：无 expandable segments，碎片后拼不出大块

- `expandable_segments:True` 在本机不生效：经 `PYTORCH_CUDA_ALLOC_CONF` 设置会触发
  `UserWarning: expandable_segments not supported on this platform`；经 `PYTORCH_ALLOC_CONF`
  设置则没有任何告警，行为与默认分配器相同（碎片堆上同样 OOM）。原因在构建层：这个厂商 torch
  的 `libc10_cuda.so` 里，PyTorch 用于虚拟内存管理的驱动接口（`cuMemCreate`、`cuMemAddressReserve`、
  `cuMemMap` 等）全是只会断言的占位实现，PyTorch 因此把该选项强制关掉；实际生效的是厂商的
  `xpu native` 分配器后端（[环境记录](../../../validation/kunlunxin-p800-reranker-fsdp2/results/env_record.json)）。
- 分配器探针（同一张卡、默认分配器）：先铺 221 个 64 MiB–1 GiB 的不等大块，释放其中 2/3
  并交还设备后，**设备空闲 72.84 GiB、PyTorch 缓存空闲 0 字节**，此时申请 11.57 GiB：

  ```
  CUDA out of memory. Tried to allocate 11.60 GiB. GPU 0 has a total capacity of 96.00 GiB
  of which 72.84 GiB is free. Of the allocated memory 23.12 GiB is allocated by PyTorch,
  and 0 bytes is reserved by PyTorch but unallocated.
  ```

  空卡上同样的请求（对照 A）以及"1 GiB 等大空洞"（对照 B）都能成功。也就是说，
  这张卡上一次大请求能否成功取决于空闲段的形状，PyTorch 常规的解法（expandable
  segments）又不可用。

### 3.3 真实 trainer 上的自然复现（2 卡 FSDP2、真实权重、batch 32）

平台记录的是 2 卡 FSDP2 + fp32 master + batch 32。用同一配置、真实 checkpoint，并把数据换成
T2Ranking 里正负样本都超过 640 token 预算的 454 行（每对都被截断到 `max_length`，即报错时的
padded 形状），基线树在本机得到：

```
[e2e-mem] after optimizer step 1: allocated=61.01 GiB reserved=92.06 GiB
[e2e-mem] after optimizer step 2: allocated=61.01 GiB reserved=92.27 GiB step_peak_allocated=87.02 GiB
  File ".../train_retrieval.py", line 631, in train_epoch
    loss.backward()
  File "hydrax/xaccelerator/linear/_linear.py", line 140, in LinearFunction.backward   # lm_head 的反向
  File "hydrax/xaccelerator/linear/_utils.py", line 96, in linear_bwd_dgrad
  File "hydrax/xaccelerator/fc_fusion.py", line 365, in fc_fusion
  File "hydrax/xaccelerator/stateful.py", line 85, in TensorViewState.value
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 11.57 GiB. GPU 0 has a total capacity
of 96.00 GiB of which 20.77 GiB is free. Of the allocated memory 71.66 GiB is allocated by PyTorch,
and 2.98 GiB is reserved by PyTorch but unallocated.
```

| | 平台记录 | 本机自然复现 |
|---|---|---|
| 请求 | 11.57 GiB | 11.57 GiB |
| 设备空闲 | 20.90 GiB | 20.77 GiB |
| PyTorch 已分配 | 71.68 GiB | 71.66 GiB（= 常驻 61.01 + 存活激活 10.65） |

两个 rank 同时在第 3 步的反向失败，失败点是 lm_head 反向里对**整序列 logits 的梯度**
（`[64, 640, 151669]` bf16）的物化；那一刻已分配的 71.66 GiB 里没有任何 11.57 GiB 的张量
（= 常驻 61.01 + 存活激活 10.65），所以是这一请求本身拿不到整块。

第 1、2 步的同一请求都成功了（step 2 峰值 allocated 87.02 GiB 里含这块），为什么第 3 步不行
取决于分配器的缓存历史，本机没有逐事件追踪，**原因未确认**。一个与现象一致的解释是：
一步之内"前向 logits 用完 → 反向梯度复用同一整块"可以自洽，而每个 optimizer step 的
AdamW 状态（step 1 起 15.25 GiB 常驻）与临时量（每步 15.25 GiB）会重新切分缓存，几个 step
之后就不再有 ≥ 11.57 GiB 的整块，只能向设备要——设备空闲 20.77 GiB 却是零散的（§3.2）。
平台那次落在哪一步不可知，数字则与本机一致。

`results/e2e-fsdp2-2card-base-b32-longdoc-2026-11-26.{json,train.log}`。长文档子集由
`tests/build_long_parquet.py` 从 T2Ranking shard 0 确定性地选出（sha256 与行数见 validation.md）。

### 3.4 单卡构造堆上的复现

2 卡实跑依赖 XCCL，本机的集合通信又间歇性卡死（§7）。为了得到不依赖多卡的稳定复现，
`tests/repro_reranker_logits_alloc.py` 在单卡上构造同样的堆：bf16 权重加每参数 3 个替身
（梯度 + AdamW 两阶矩）= 61.0 GiB 常驻，与上面实跑的 61.01 GiB 相同；余下 34 GiB 切成
9 块（最大 6 GiB）后释放。随后的一步：

```
基线  CUDA out of memory. Tried to allocate 11.57 GiB. GPU 0 has a total capacity of 96.00 GiB
      of which 20.63 GiB is free. Of the allocated memory 73.51 GiB is allocated by PyTorch,
      and 1.57 GiB is reserved by PyTorch but unallocated.
修复  同一步跑通，峰值 +25.36 GiB
```

这是**构造**的碎片历史，只用来在同一张卡上稳定复现"大请求 + 碎片"这一组合，自然复现以
§3.3 为准。

## 4. 修复

```diff
-        outputs = self.backbone(**features, return_dict=True, **kwargs)
-        logits = outputs.logits
-        positions = features["attention_mask"].to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
-        rows = torch.arange(logits.shape[0], device=logits.device)
-        final_logits = logits[rows, positions]
+        outputs = self.backbone.model(**features, return_dict=True, **kwargs)
+        final_hidden = last_token_pool(outputs.last_hidden_state, features["attention_mask"])
+        # Call the module rather than F.linear so FSDP's hooks on lm_head run.
+        final_logits = self.backbone.lm_head(final_hidden)
```

- `self.backbone.model(...)` 只跑解码器；`last_token_pool`（`transformer_encoder.py` 已有、
  embedding 模型在用）取每条序列最后一个非 pad 位置的 hidden；`self.backbone.lm_head`
  只投影这 64 行。位置选择与基线逐行相同（同一个 `attention_mask.sum(-1) - 1`）。
- LM head 的输入从 `[64, 640, 4096]` 变成 `[64, 4096]`，输出从 11.57 GiB 变成 18.5 MiB。
- **FSDP2 兼容**：`parallel.iter_large_leaf_modules` 把 lm_head 单独分片（"oversized leaf"），
  它有自己的 pre/post-forward hook 负责 all-gather 与 reshard。这里仍然调用模块本身
  （而不是 `F.linear` 作用在 weight 上），hook 照常触发；`use_score_head` 分支本来
  就是 `self.backbone.model(...)` 的两段式写法。

## 5. 备选方案与取舍

| 方案 | 结论 |
|---|---|
| `self.backbone(..., logits_to_keep=1)` | 取的是序列最后一个位置；本适配器右 padding，短序列会取到 pad 位，语义变了。不采用 |
| 对 lm_head 分块计算 | 峰值变小但仍对全部位置做词表投影，算力照旧浪费。不采用 |
| `expandable_segments` / `empty_cache` | 平台不支持 / 只是缓解，根因（无用的 11.57 GiB 请求）仍在 |
| batch 降到 16（平台现状） | 规避而非修复，显存仍 95/96 GB |

## 6. 影响范围

- 只改 `flagscale/models/retrieval/qwen3_reranker.py` 的非 score-head 分支；score-head
  分支、Megatron 路径（`qwen3_reranker_megatron.py`）、训练循环都不动。
- 数值：同位置、同权重、同 GEMM，bf16 autocast 与 fp32 下分数和梯度与基线公式**逐位相同**
  （`results/equivalence-fixed-*.json`，5 例 max diff 均为 0.0）；真实 trainer 上两棵树
  5 步 loss、validation、test loss 逐位相同。
- 代码路径与厂商无关；设备证据只来自 KunlunXin P800（见 `validation.md`）。评测入口
  `evaluate_retrieval.py` 走同一个 forward，推理侧同样少一次全序列投影（未实测）。

## 7. 与其他修复的关系

| 修复 | 关系 |
|---|---|
| `adamw-fused`（flagscale，`train/train_retrieval.py`） | 无共享文件、无 requires/conflicts。`flag-fixes plan reranker-last-token-logits adamw-fused --target <基线树>` 两个一起预演 exit 0；两份 patch 叠加后的设备验证没有跑 |
| `flaggems-selective-enable`（flagscale，`train/utils/flaggems.py`） | 无共享文件。本次验证全程 `use_flaggems=false`，两者叠加没有验证 |
| `gelu-kunlunxin-pow`（FlagGems） | 不同包，无关 |
| FlagScale 里既有的 `iter_large_leaf_modules`（`parallel.py`） | 同一个平台现象的两次不同处理：那次把 lm_head / embed_tokens 拆成单独的 FSDP 组，使根组的 all-gather 从 2.31 GiB 降到 1.16 GiB；本修复去掉的是另一处 11.57 GiB 的申请。两者正交，可以并存 |

相似根因不自动构成依赖：本修复只改适配层一个分支，不依赖其他修复，也不被它们依赖。

## 8. 已知限制与后续

- **修复树在平台形态下没有完整跑通**。基线失败已自然复现（§3.3）；同一数据上修复树的 4 次
  2 卡尝试里，前三次分别停在第 2、3、1 步，第 4 次走完了 3 步——包括基线 OOM 的那一步，
  step 3 峰值 allocated 84.41 GiB——然后停在 validation。卡住的不是 OOM，而是本机的 XCCL
  集合通信（已登记的 F-P800-1：`Watchdog caught collective operation timeout`，终止点与
  payload 无关）。常规数据上修复树走完 3 步 + validation（step 3 峰值 allocated 83.20 GiB）
  后同样卡在收尾。因此"修复后 batch 32 在 2 卡上完整跑通"**未验证**；修复对照的干净结果
  来自单卡构造堆（§3.4）与真实 `apply_fsdp2` 的 2 卡单元测试。
- 本机 2 卡主干触发 14 次（trainer 10 次、FSDP2 测试 4 次），两棵树都中招：3 次按计划跑完
  （基线自然 OOM 1 次、FSDP2 测试 2 次），修复树 2 次走完 3 个 optimizer step 后卡住，其余
  9 次被 watchdog 或设备异常提前终止。这是平台阻断，不是本修复引入的；完整清单与两项
  探索（步骤边界同步、主机时序）见 `validation.md`。
- 单卡 bf16 DDP 下，厂商的 `clip_grad_norm_` 对 bf16 梯度会物化约 35 GiB 临时张量
  （allocated 30.5 GiB 时 clip 峰值 65.66 GiB），AdamW step 另有 15.25 GiB 临时张量。
  这两项与 logits 无关、本修复不覆盖，e2e 的单卡命令以 `max_grad_norm=0` 绕开前者。
- 平台配置的 batch 16 / GA 1 是为这次 OOM 设的；修复后是否有余量回到 32 或 accum=2，
  需在平台 2 卡环境实测后再改配置。
- 上游：未向 `wxj630/FlagScale` 提 PR；平台配置注释里"11.57 GiB 是 lm_head all-gather"
  的归因可在上游一并更正。
