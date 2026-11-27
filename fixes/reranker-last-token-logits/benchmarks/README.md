# 性能与精度

这是显存修复，附带少量耗时收益；以下为本机单卡 P800 OAM（96 GiB）数据，
**不是**整模型吞吐，也不代表平台验收。

## 口径

| 项目 | 值 |
| --- | --- |
| 脚本 | `../tests/bench_reranker_step.py --iters 5` |
| 设备 | 昆仑芯 P800 OAM，torch 2.9.0+cu129（torch_xmlir shim），transformers 4.57.6 |
| 模型 | Qwen3-Reranker-8B 架构（36 层、词表 151,669），随机 bf16 权重 15.25 GiB，梯度检查点开 |
| 输入 | 每条序列截断到 `max_length` 640；batch 16 = 32 对，batch 32 = 64 对 |
| 计算 | bf16 autocast，`reranker` 的 BCE loss；无优化器状态 |
| 计时 | warmup 2；5 次取中位数；每阶段前后 `torch.cuda.synchronize()`。event 中位数与墙钟一致（差 < 1 ms） |
| 显存 | `torch.cuda.max_memory_allocated()`，含权重；峰值 reserved 为整轮最大缓存 |
| 环境变量 | `XMLIR_BMM_DISPATCH_VALUE=0`（设备命令统一设置，平台 P800 启动脚本同值）；其余未设置 |

## 前后对照

| batch（对数） | 树 | 前向 | 反向 | 前向峰值 allocated | 反向峰值 allocated | 峰值 reserved | LM head 单次输出 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16（32） | 基线 | 1.767 s | 5.146 s | 28.60 GiB | 36.86 GiB | 38.26 GiB | 5.7857 GiB |
| 16（32） | 修复 | 1.663 s | 4.964 s | 22.67 GiB | 34.94 GiB | 39.19 GiB | 0.0090 GiB |
| 32（64） | 基线 | 3.459 s | 10.129 s | 40.80 GiB | 54.99 GiB | 59.68 GiB | 11.5714 GiB |
| 32（64） | 修复 | 3.247 s | 9.760 s | 29.94 GiB | 40.61 GiB | 49.65 GiB | 0.0181 GiB |

（反向的 5 次样本两棵树内部各自的极差 ≤ 2 ms，如基线 batch 32：
10.129/10.129/10.130/10.130/10.128 s。）

读法：

- **显存**是修复目的：batch 32 前向峰值 −10.86 GiB、反向峰值 −14.38 GiB；
  batch 16 前向峰值 −5.93 GiB。batch 16 的反向峰值只差 1.92 GiB，因为该尺寸下
  反向峰值由"已累积的梯度 + 重算临时量"决定，5.79 GiB 的 `grad_logits` 落在峰值之下。
- **耗时**是副产品：词表投影从 40,960 行降到 64 行（batch 32）或 512 行降到 32 行
  （batch 16），前向 −6%，反向 −3.5%。整步其余部分不变，所以占比不大。
- 峰值 reserved 取决于分配器缓存历史，受同进程前一轮影响，只作参考。

## 数值精度

`../tests/test_reranker_scores_equivalence.py`（同架构、同一 tokenized batch，适配器 forward 对"整序列
logits 取末位"的基线公式；每一例记录权重 dtype 与是否 autocast）：

| 用例 | 权重 / 计算 | 分数 max\|Δ\| | 梯度 max\|Δ\| / max\|g\| |
| --- | --- | ---: | ---: |
| bf16 autocast，混合长度 16 行（32 对，pad 到 640） | bf16 | 0.0 | — |
| bf16 autocast，全长 640，32 行（64 对） | bf16 | 0.0 | — |
| 无 autocast，混合长度 4 行 | bf16 | 0.0 | — |
| bf16 autocast，BCE 反向，16 行（lm_head/embedding/norm/首尾块 9 个预期梯度张量） | bf16 | — | 0.0 |
| 无 autocast，BCE 反向，2 行 | bf16 | — | 0.0 |
| 无 autocast，混合长度 4 行 | **fp32 权重** | 2.4e-7（阈值 1e-4） | — |
| 无 autocast，BCE 反向，2 行 | **fp32 权重** | — | 2.0e-4（阈值 1e-3） |

读法：

- bf16 下在本机上**逐位相同**：同一 GEMM 在 `[64, 640, 4096]` 与 `[64, 4096]` 上被选中的 64 行给出同样的结果。
- fp32 权重下不是逐位相同：分数差 1 个 fp32 ulp 量级，梯度相对差 2.0e-4，都在声明的阈值以内。差异只能来自
  lm_head 在两种输入形状下的 fp32 舍入，具体 kernel 选择没有追踪。训练形态是 bf16 权重 + bf16 autocast，
  属于上一条。
- 梯度比较先验结构（两侧各 9 个预期梯度）并逐张量检查有限性，再算误差；同一脚本里的 7 个 `comparator_*`
  用例确认 NaN 梯度、两侧同时缺一个等情况会被拒绝。
- 原始数据 `../results/equivalence-{base,fixed}-2026-11-27.json`（审阅修订后的 14 例）。11-26 的旧版记录
  （5 例）把 bf16 关 autocast 的两例标成了 fp32，已由本表取代。

真实 trainer、真实权重下：两棵树 5 个 optimizer step 的 `running_loss/window_loss`、`train_loss=0.539403`、
`validation_loss=0.461096`、`test_loss=0.329392` 在日志记录精度（6 位小数）下一致
（`../results/e2e-1card-*-2026-11-27.json`）。

## 运行方式

`$TREE` 为打了补丁或未打补丁的 FlagScale 仓库根：

```bash
XMLIR_BMM_DISPATCH_VALUE=0 PYTHONPATH="$TREE" python3 tests/bench_reranker_step.py --iters 5 --json out.json
```
