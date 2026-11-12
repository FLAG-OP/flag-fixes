# 诊断报告：adamw-fused

## 1. 现象

平台文档（FlagScale `docs/retrieval/08-kunlunxin.md` §按语义归类）的差距分解表里，
`other` 类别为昆仑芯 104.3 ms/步 vs A100 1.8 ms（58.1×），从未被解释。
同时文档记录 A100 的 optimizer 是 `_foreach_*` 融合 kernel（18.6 ms），
而昆仑芯该类别为 0——说明昆仑芯没有走 foreach 路径。

## 2. 归因（详见 /workspace/problem29/ATTRIBUTION.md §二）

原始 trace 未留存、模型/数据集不在本机，故用重建负载归因
（XLM-R large 架构精确 + 平台文档记载的负载配方：每 micro-batch 5 次 forward
× batch 10 × accum 4；kernel 级与文档对照偏差 ≤10%）。
逐 kernel 分类后 "other" ≈ 104 ms 由三部分组成：

| 成分 | ms/步 | launch/步 |
|---|---:|---:|
| **AdamW 逐参数更新** | **41.8** | **3,890** |
| clip_grad_norm_ | 9.8 | 2,099 |
| copy/pad 类（SDPA mask 预处理为主） | ~50 | 6,231 |

AdamW 的 8 种 kernel × 389 参数：`addcdiv`(389) / `sqrt`(389) /
`addcmul`(389) / `lerp`(389) / `tensor_calc_scalar`×3(1,167) /
`constant`(778) 等。A100 上同一计算是少量 `_foreach_*` kernel。

## 3. 根因

| 实现 | 墙钟/步 | launch/步 | device/步 |
|---|---:|---:|---:|
| stock AdamW | 41.8 ms | 3,890 | — |
| `foreach=True` | 41.0 ms | ~同 stock | — |
| **`fused=True`** | **17.3 ms** | 少量 | — |

两个事实：

1. **`foreach=True` 在厂商栈上无融合收益**（`results/bench-base-*.json`：
   stock 3,024 launches vs foreach 2,688，device 26.9 vs 25.7 ms——
   只省了逐参数那一次 Python 侧派发，kernel 仍逐参数）。因此这不是
   "忘开 foreach" 能解决的。
2. **`fused=True` 才走真正的多张量单支 kernel**（672 launches、device 8.9 ms）。

`torch.optim.AdamW` 默认在 CUDA 上自动选 foreach，永远不会自动选 fused，
这就是差距的结构性来源。

## 4. 方案

`flagscale/train/train_retrieval.py::build_optimizer` 的 AdamW 分支改用
`adamw_impl_kwargs(parameters)` 决定实现 flag：

- **auto（默认）**：参数全部为 CUDA fp32/bf16/fp16 且类型不是
  `torch.distributed.*`（DTensor/FSDP2）→ `fused=True`；否则 stock。
- **env 覆盖**：`FLAGSCALE_ADAMW_FUSED=0/false/off` 强制 stock；
  `=1/true/on` 表达 fused 意愿（DTensor/CPU 护栏仍然生效并记 log）。
- **构造期回退**：torch 构造函数拒绝 `fused` kwarg 时捕获
  RuntimeError/TypeError/ValueError 回退 stock。

选择 fused 而不是 foreach 的理由：foreach 已被实测无收益（§3）；
fused 要求全部参数同 device 同 dtype 且不支持 DTensor——FSDP2 分片参数
必须留在 stock 路径，护栏保证这一点。

## 5. 影响范围

- 仅 `flagscale/train/train_retrieval.py` 一个文件的 AdamW 分支；
- Adafactor 分支与 `build_lr_scheduler` 不动；
- 不触碰任何 kernel、dtype、lr/wd 语义；逐参数→单支只改变实现路径，
  数值差在 bf16 1 ulp 量级（`results/numerics-fixed-*.log`：bf16 3.1e-5、
  fp32 3.0e-8，5 步同梯度序列，allclose 通过）。
- 非 AdamW 用户（Adafactor）与 CPU/offload 参数路径零变化。

## 6. 与其他修复的关系

- 与 `gelu-kunlunxin-pow`（FlagGems 层）无依赖/冲突：不同包、不同层。
- 与"向厂商要 weight-scale 缓存"（problem29/ATTRIBUTION.md §三）互补：
  本修复解决 "other" 里的 optimizer 部分，后者解决 cast 胶水部分。
- clip_grad_norm_ 的逐参数 cast+reduce（9.8 ms）本修复未覆盖：
  torch 2.9 的 `clip_grad_norm_(foreach=True)` 实测只省 ~2 ms，
  需要厂商 multi-tensor reduce，另立条目。
