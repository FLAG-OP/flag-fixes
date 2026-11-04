# 根因与修复方案

## 原始问题与预期行为

- 上游 issue：https://github.com/flagos-ai/FlagGems/issues/5379
  （`[BUG] nn.Linear fails on non-contiguous activations: view() before contiguous() in linear forward/backward`，open）
- 本仓库问题描述：`viewFix/problem.md`——全连接 view 崩溃
  `view size is not compatible with ... stride`；补丁为 `view → reshape`，修复后 18 个视觉模型
  36/36 翻转 Success，最小复现模型 `google/siglip-base-patch16-224`。
- 复现条件：开启 flag_gems 后，`linear` 的输入是非连续张量（`transpose` / `permute` / 切片 /
  expand 出来的激活），且 `input.dim() > 1` 且压平维度数与原 stride 不兼容。
- 原始错误（本机实测，基线 `29e14822`）：

```
input (2, 16, 64) stride=(1024, 1, 16) contiguous=False
  File ".../src/flag_gems/ops/linear.py", line 158, in linear
    input_flat = input.view(M, K)
RuntimeError: view size is not compatible with input tensor's size and stride
(at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

- 预期行为：`linear` 对任意布局的输入都应给出与原生 torch 一致的结果；连续输入的现有行为
  （数值与性能）不允许变化。

## 根因与证据

肇事代码是压平步骤（基线 `src/flag_gems/ops/linear.py:158`，`linear()` 内）：

```python
input_flat = input.view(M, K)   # M = 前置 batch 维乘积, K = in_features
```

`view()` 要求"目标形状与现有 stride 兼容"，即不产生跨越两个连续子空间的重解释。视觉/注意力
流水线里的激活经常不满足：实测三类输入全部触发同一错误（`tests/test_linear_noncontig.py`）：

| 输入构造 | 形状 | stride | 连续 |
| --- | --- | --- | --- |
| `randn(2, 64, 16).transpose(1, 2)` | (2,16,64) | (1024, 1, 16) | 否 |
| `randn(4, 16, 128)[..., :64]` | (4,16,64) | (2048, 128, 1) | 否 |
| `randn(1, 16, 64).expand(2, 16, 64)` | (2,16,64) | (0, 64, 1) | 否 |

对照实验（同一份基线代码，仅改变输入布局与 dtype）：

- 连续 2-D / 1-D 输入：`view` 与 `reshape` 等价，修复前后结果逐位相同（max_diff=0）；
- `M=1` 的输入：即使维度数 >1 也往往能兼容，修复前后都通过——说明崩不崩只取决于 stride
  兼容性，而不是形状本身；
- 只把 `view` 换成 `reshape` 后，同一批用例全部通过，错误消失（`reports/validation.md`）。

错误同时从两条路径抛出：Torch dispatch（`F.linear` 在 `flag_gems.enable()` 下）与直接调用
`flag_gems.linear`，说明问题在算子实现本身，不在注册或 dispatch 层。

## 修复与影响

```diff
--- a/src/flag_gems/ops/linear.py
+++ b/src/flag_gems/ops/linear.py
@@ -155,7 +155,7 @@ def linear(input, weight, bias=None):
     N = weight.shape[0]  # out_features
 
     # Flatten input: (*, K) -> (M, K)
-    input_flat = input.view(M, K)
+    input_flat = input.reshape(M, K).contiguous()
```

- 机制：`reshape` = 能视图就视图、不能就拷贝，正好对应"外部传入、布局不可控"的输入；
  `.contiguous()` 对已经连续的结果是空操作，因此连续路径**不会**多一次拷贝，非连续路径本来
  就必须有一次物化。
- 与 kernel 的关系：`linear_kernel` / `linear_gemv_kernel` 全部通过显式 stride 读写
  （`stride_im`、`stride_ik` …），因此压平后是视图还是拷贝块都正确；`_linear_gemv`、
  `_linear_dot` 两条分支均实测通过。
- 作用层级：FlagGems 算子层，厂商无关文件 `src/flag_gems/ops/linear.py`；本机 vendor 解析为
  `thead`（`device_name='cuda'`、`device_query_cmd='ppu-smi'`），基线版本 `_thead/ops/` 下没有
  linear 覆写，因此本机执行的就是该文件（traceback 可证）。
- 依赖/替代关系：`requires`/`conflicts` 均为空；单文件单行，可独立安装与回滚；
  `git apply --check` 干净，patch 结果与 `src/flag_gems/ops/linear.py` 快照逐字节一致
  （`flag-fixes validate` / `check` / `plan` 通过）。
- 数值影响：fp32 与原生 torch 的 mse 在 1e-11 ~ 1e-14 量级；连续对照组修复前后完全一致。

## 已知限制与上游跟踪

1. **平台验收未做**：本环境没有 `/data/ppu-benchmark/FlagGems` 与 x-benchmark 运行器，
   problem.md 要求的容器侧 `run_model.py`（独立 `--result-root`）回归未执行；
   `validation.hardware_status` 保持 `unverified`，`deployable=false`、`owner=null`，
   待平台负责人验收后才可能提升。
2. **同源问题的其他文件未覆盖**（按"不同后端分条"的约定单独修，跟踪在 issue #5379）：
   - `src/flag_gems/runtime/backend/_enflame/gcu300/ops/linear.py:158`（issue 原文引用）
   - `src/flag_gems/runtime/backend/_enflame/gcu400/ops/linear.py:158`
   - `src/flag_gems/runtime/backend/_iluvatar/ops/linear.py:135`
   - `src/flag_gems/ops/linear_backward.py:73-74`、`_thead`/`_iluvatar` 的 linear_backward
     （`.view(...)` 在 `.contiguous()` 之前，同样会先抛）。本机按 issue 的 backward 复现
     形状未触发（梯度到达时已是连续），故未纳入本次验证矩阵。
3. **验证中发现的既有缺陷（与本修复无关）**：本机 `N=32, K=64` 时 fp16/bfloat16 在
   `M ∈ {5,6,7,8}` 输出完全错误（maxdiff ≈ 2.0e1~3.4e1），`M ∈ {1,2,3,4,9,12,16,32,64}`
   完全正确；连续与非连续输入、修复前后同样复现，说明是该 M 段的 kernel/调优既有问题。
   回归矩阵因此只在 fp32 上断言该 M 段（GEMV 路径），并在测试内注明原因，不把已知失败
   伪装成通过，也不让它掩盖本修复的判定。建议上游单独开 issue。
4. **无性能基准**：崩溃修复，连续路径逐位不变，非连续路径修复前无法完成计算，无法给出
   "修复前耗时"对照；`benchmarks/` 保持模板状态。
5. **上游 PR 未提交**：跟踪 issue #5379；合入 commit 与发行版本待补。
