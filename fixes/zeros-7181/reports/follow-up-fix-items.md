# 另立修复项（第四轮全量组暴露问题的跟进清单）

评审要求（comment `6766490204` §推进顺序 4）：全量组暴露的问题另立修复项，
优先 `mse_loss_backward` 标量 grad_output 广播（静默错梯度），其次 complex
pointwise/sgn/fill 与 FP64 编译；每项保留最小复现、原生对照与修复前后证据；
预存归因须关联同路径基线或隔离对照，不得仅以"文件不在本 patch 内"作为全部
因果归属证明。本页逐项给出复现脚本、两树隔离对照结果与当前状态
（均未实现修复，属后续独立修复包的工作范围）。

## FIX-ITEM-1（优先）：`mse_loss_backward` 不广播 0 维标量 grad_output

- **症状**：全量注册下对标量 loss 反向传播时，`aten::mse_loss_backward` 收到
  autograd 传入的 0 维标量上游梯度；FlagGems kernel 按元素索引读取该 1 元素
  张量，第 0 元素精确、其余 39/40 元素读到越界垃圾 → **静默错梯度**
  （小模型 B 在 f_all 下 190/192 项超差、rel_l2≈1.0，训练发散；前向精确）。
- **最小复现**：`tests/repro_mse_scalar_grad.py`（8×5 张量，三种 reduction 对照）。
  - 全量张量 grad_output + mean：**精确**（max_abs=0）
  - 标量 grad_output + mean：**39/40 错**（第 0 元素精确）
  - reduction none/sum：精确
- **原生对照 / 隔离对照（预存归因）**：
  - 修复树（`flagfix-fixed`）：39/40 错 → `matrix-rescored/repro-mse-scalar-fixed.log`（四轮证据 `mse-scalar-repro.log` 同）
  - **修复前对照树（`flagfix-e2ebase`）：39/40 错，逐位同形态** → `matrix-rescored/repro-mse-scalar-btree.log` —— 同路径基线复现，预存性由隔离对照直接证明
  - `ops/mse_loss_backward.py` 两树逐字节相同（`diff` 为空）
- **影响面**：任何使用标量 loss（mean/sum reduction MSE）+ 全量注册的训练反向。
- **状态**：proposed（未实现）。建议修法：wrapper 内对 0 维 grad_output
  `expand_as/expand` 至 self 形状后再进 kernel（或 kernel 内 stride-0 广播读）。
  修复前后证据在独立修复包中补齐。

## FIX-ITEM-2：complex 组装张量的 pointwise/fill 路径（empty_like / conj / sgn / zeros / to_copy）+ FP64 编译

- **症状**：triton `canonicalize_dtype` 无 complex 条目，经 `fill_scalar_`
  的路径对 complex64/128 直接 KeyError；触发面包括：
  - `torch.empty_like`、`conj()`（`_conj.py:69`）、`sgn`（`sgn.py:149`）作用于
    **`torch.complex(p, q)` 组装张量**（cast 张量 `randn().to(c64)` 不触发）
  - `torch.zeros(complex, ...)` 的尺寸相关分支（bench 逐用例记录：8×8 探针 OK、
    32×32 及以上 KeyError）
  - `.to(complex)`（`to_copy` → fill）在注册后执行会中止进程（bench 探针张量
    因此改为注册前构建）
  - fp64/c128：triton CompilationError（loop-carried accumulator fp32→fp64）
- **最小复现**：`tests/repro_complex_fill.py`（5 个探针）。
  - 修复树：5/5 KeyError → `matrix-rescored/repro-complex-fill-fixed.log`
  - **修复前对照树：5/5 KeyError（`abs(z)` 前向亦崩，即修复前该路径完全不可用）**
    → `matrix-rescored/repro-complex-fill-btree.log` —— 同路径基线复现
- **与四轮 f_all-A 崩溃的关系**：矩阵中 f_all-A 的 traceback（`sgn.py:149` →
  `fill.py:103`）与本复现同节点；abs 反向经 sgn→empty_like 触发，即
  `abs(sum(z)) backward` 探针（两树均崩）。
- **影响面**：全量注册下任何经组装 complex 张量的 pointwise/创建路径；
  f_all 模型 A（复数核）整体不可跑。
- **状态**：proposed（未实现）。建议方向：pointwise_dynamic/fill 增加 complex
  显式拒绝并回退原生（同 zeros-7181 对 zeros/zero_ 的处理模式），或补齐 triton
  complex 指针支持；FP64 需 triton 侧累加器 dtype 修正。修复前后证据在独立
  修复包中补齐。

## 跟进原则

- 两项都不在本 patch 的 8 个目标文件内；本页仅登记问题与预存证据，不据此
  扩大 zeros-7181 的声明范围。
- 实际修复时各自独立建包（fix.yaml + patch + 回归 + 修复前后日志），并沿用
  "错树前置 / 缺依赖 exit 2 / 证据单独提交" 的既有流程。
- 原始运行证据（四轮矩阵/bench 与 traceback）保留在
  `results/small-model-2026-11-10/`，不改写。
