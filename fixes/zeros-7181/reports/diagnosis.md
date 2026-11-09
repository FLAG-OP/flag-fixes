# 根因与修复方案

## 原始问题与预期行为

- 上游 issue：https://github.com/flagos-ai/FlagGems/issues/7181
  （`Complex zeros and abs reach unsupported Triton dtype handling on Hygon`，open）。
  复现形态：`flag_gems.zeros((2,), dtype=torch.complex64/128)` 与 `flag_gems.abs(x)`
  在 complex dtype 下抛 `KeyError: 'complex64'/'complex128'`。
- 本仓库问题描述：`complexFix/problem.md`——m2-bert 容器整模型跑
  `togethercomputer/m2-bert-80M-8k-retrieval` 时出现
  `KeyError: 'complex64'`，live 堆栈为
  `triton/_utils.py:137 canonicalize_ptr_dtype → :133 canonicalize_dtype →
  type_canonicalisation_dict[dtype_str] → KeyError: 'complex64'`。
- 复现条件：`flag_gems.enable()` 下跑该模型的训练反向；问题只出现在**反向**，
  前向对该模型是安全的（实测前向只经过 `aten.div.Tensor`、`aten.mul.Tensor`、
  `aten._fft_c2r`，均为 complex 安全路径；baseline + linear-5379 的全量配置前向
  `FORWARD OK`）。
- 原始错误（本机实测，基线 `29e14822`，修复前 e2e）：

```
  File ".../src/flag_gems/ops/sum.py", line 385, in sum_dim
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype)
  File ".../src/flag_gems/ops/sum.py", line 306, in sum_dim_comm
    sum_dim_kernel_non_inner[grid](...)
  File ".../triton/_utils.py", line 133, in canonicalize_dtype
    return type_canonicalisation_dict[dtype_str]
KeyError: 'complex64'
```

- 预期行为：问题 issue #7181 口径——complex 的 `zeros` 保持 dtype 并正确清零、
  `abs` 返回实部模长（f32/f64），或在无法支持时给出清晰的前置拒绝；
  对本仓库 problem.md 口径——m2-bert 反向不再以 `KeyError` 中断，训练步完整跑完，
  前向输出与原生 torch 一致。

## 根因与证据

### 1. 主根因：complex dtype 进入 triton 后无法 mangle（#7181 类本体）

triton 3.6.0 的 `type_canonicalisation_dict`（`triton/_utils.py` L90–135）只登记了
实数 dtype，没有 `complex64/complex128`。FlagGems 基线里 `sum`、`zeros/zero_/zeros_like`、
`abs/absolute`（in-place 与部分入口）把 complex 张量直接送进为实数设计的 Triton kernel，
binder 在 `mangle_type` 阶段取 `type_canonicalisation_dict['complex64']` 即抛 KeyError。
同一机制同时解释了 `zeros.py`、`zero.py`、`zeros_like.py`、`sum.py`、`abs.py` 的崩溃，
以及 issue #7181 在 Hygon 上的同签名报错。

对照实验（全部双树执行，日志见 `results/`）：

| 实验 | 修复前（基线） | 修复后 |
| --- | --- | --- |
| `repro_complex64.py`（issue #7181 直接调用形态 + torch 入口） | exit 1，`KeyError: 'complex64'` | exit 0 |
| `test_complex_ops.py`（139 用例矩阵，2026-11-08 版） | `TOTAL: 18 passed, 121 failed, 139 total`（107 条 complex KeyError，其余为 strided 断言 / 交叉 dtype 静默分歧等修复缺失行为） | `TOTAL: 139 passed, 0 failed, 139 total` |
| 真实模型反向（`test_m2_bert_backward.py`，见下 3） | `ops/sum.py:385` KeyError 中断 | 反向完成 |

> 2026-11-07 的旧日志（`results/*-2026-11-07.log`）中 regression 的
> `TOTAL: 54 passed, 38 failed` 一行有**计数器缺陷**：打印把 passed+failed 都写成了
> passed（`TOTAL: {passed + failed} passed, {failed} failed`）。旧日志原文保留不改，
> 实际旧口径为 **16 passed / 38 failed / 54 total**（38 条失败全部为 complex KeyError，
> 未受影响）。计数器已在 2026-11-08 修复为 `passed/failed/total` 三个独立数字。

m2-bert 反向里首个被触发的 complex 求和来自混合卷积（hyena filter）梯度路径中的
`aten.sum.dim_IntList(complex64)`，注册表走 `("sum.dim_IntList", sum_dim) →
sum_dim_comm → sum_dim_kernel_non_inner`，因此容器堆栈停在 `sum.py`。
`probe` 与回归同时确认 `zeros` 族与 `abs` 是同一签名下的并行缺口（#7181 将
zeros 与 abs 并列报告的原因）。

### 2. 同一反向链路上的两个预存前置阻塞（不含则反向到不了/完不成 complex 验证）

**（a）`conv2d.py` 的 constexpr 下标崩溃。** `ops/conv2d.py:288` 写的是
`(ci_point_value * input_c_stride)[:, None]` 中的 `input_c_stride[None]`——当
`input_c_stride` 是 Python 标量且**取值为 1** 时，triton 3.6.0 的 `mangle_type(1,
specialize=True)` 会把整型参数特化成 `constexpr`，对 constexpr 做 `[None]`
（`__getitem__`）直接 `AttributeError: 'NoneType' object has no attribute 'type'`。
m2-bert 的 conv1d 输入来自 `rearrange(b l d -> b d l)` 转置（非连续），其
`stride(1) == 1` 稳定触发。最小复现：`nn.Conv1d(C,C,3,groups=C,padding=2)` +
非连续输入，基线崩、修复后过；语义等价性由 `[:, None]` 对标量与 1 元素张量
的广播行为验证，且该形态在文件内仅此一处。

**（b）`_thead/ops/linear_backward.py:170/171` 的 view 崩溃。**
`input.view(batch_size, in_features)` 与 `grad_output.view(batch_size, out_features)`
对非连续输入抛 `RuntimeError: view size is not compatible ...`（与 linear-5379
同签名、不同文件；上游 master 至 `a4e2556d` 未修）。证据：仅基线 + linear-5379
的 e2e 在此处中断（`results/e2e-linearblocker-baseline-2026-11-07.log`）。
修复为 `reshape(...).contiguous()`，与 linear-5379 同口径。

两个阻塞的 e2e 归属链（同一模型、同一配置，逐层剥洋葱）：

```
基线 + linear-5379                      -> 反向崩在 linear_backward.py:170 view RuntimeError
基线 + linear-5379 + (a)(b)             -> 反向崩在 sum.py:385 KeyError: 'complex64'（problem.md 签名）
基线 + linear-5379 + (a)(b) + complex 修复 -> 反向完成，E2E OK
```

### 3. 修复后仍然存在、但不属于本 patch 的预存缺陷（如实记录）

以下现象在**完全不含 complex 修复**的树上以 `flag_gems.enable(unused=[...])` 排除
`sum/zeros/abs/fill` 系后（反向可以完成）与修复树上**同形态**出现，因此判定为
预存上游问题、超出 #7181 类修复范围（原始日志
`results/gradgap-probe-noncomplex-2026-11-07.log` 与
`results/gradgap-probe-fixed-2026-11-07.log`）：

- **filter_fn 梯度缺口**：flag_gems 全包没有任何 `register_autograd` 调用
  （`grep -rn register_autograd src/flag_gems/` 0 命中），m2-bert 反向经过
  `aten::pad` 与 `aten::conv1d` 时触发
  `"an autograd kernel was not registered ... may lead to silently incorrect behavior"`，
  hyena `filter_fn.*` 子树拿不到梯度。精确集合已固化为 fixture
  `tests/known_gap_m2.txt`（12L 默认配置 **384** 个 = 32 个 layer 模板 × 12 层，
  从实测运行逐条记录），e2e 双向断言：fixture 之外缺梯度 → 失败；fixture 内
  条目反而有梯度（fixture 过期）→ 同样失败。1L 配置同形态 32 个（+2 个
  未使用 pooler 不产生 native 梯度，不在 `g_native` 中故不计入）。
- **梯度数值分歧**：native 与 flag_gems 的梯度在若干参数上量级分歧巨大
  （近驻点参数 native ~1e-9、flag_gems ~0.5 量级，反向亦有 native 有值、
  flag_gems 为 0 的项），形态与集合在无 complex 修复树/修复树上一致。
- **loss 张量存储被覆写（间歇）**：部分运行中 `loss_gems.backward()` 之后
  loss 标量张量回读值变成 ~1e-6/0（另一些运行正常），属间歇性内存异常。
  **归因未定**：修复前的树反向根本跑不完（同一路径从未完成过，无法 A/B），
  修复树上亦时有时无——既未证明是预存、也未证明由本修复引入，不站队。
  e2e 因此在 backward **之前** 把 loss 快照成 Python float 再比较，
  注释见 `tests/test_m2_bert_backward.py`。
- `mean`/`full`/`ones`、`eq`/`where` 等其余 complex 缺口（`ComplexMode.NONE`
  的 pointwise 或未覆写入口）未修：m2-bert 反向链路不需要它们，
  `torch.equal` 一类请走 `torch.view_as_real` 对比（回归脚本即如此）。

## 修复与影响

一个实例、8 个文件（完整 diff 见 `patches/zeros-7181.patch`，366 行 / +195 −23）。
按评审口径分**三类改动**（同一次反向链路上的三个层次，故并入一个 fix；每类有独立验收）：

| 类 | 文件 | 验收口径 |
| --- | --- | --- |
| **A. complex dtype 支持（#7181 本体）** | `sum.py`、`zeros.py`、`zero.py`、`zeros_like.py`、`abs.py`、`absolute.py` | KeyError 消失 + native oracle 数值一致（含 strided / 交叉 dtype out 矩阵 / 模长误差界） |
| **B. 同链路前置阻塞（不含则反向到不了/完不成 A 类验证）** | `conv2d.py`（constexpr 下标）、`_thead/ops/linear_backward.py`（view） | 最小复现 + e2e 剥洋葱归属链（见上 2） |
| **C. 后端/依赖边界** | `ops/linear.py` **不在本 patch**（归 linear-5379，`requires: [linear-5379]` + `e2e_precondition` 哈希钉死）；`linear_backward` 仅 `_thead` 覆写 | e2e 起跑前 9 文件 sha256 前置检查，不匹配 exit 2 |

### A 类逐文件机制

| 文件 | 改动与机制 |
| --- | --- |
| `ops/sum.py` | `_COMPLEX_TO_REAL` 映射 + `_real_sum`/`_sum_complex`：complex 输入经 `torch.view_as_real` 降到实数视图复用现有实数 kernel，结果按映射 dtype 转回；`sum/sum_out/sum_dim/sum_dim_out` 四入口前置分支。**交叉 dtype out 语义对齐 ATen**：`sum_dim_out(dtype=None)` 采用 `out.dtype`（实数与 complex 均如此；bool out 先按 i64 计算），显式 `dtype != out.dtype` 抛与 ATen 逐字一致的报错；`sum_out` 全量形式不采用、只校验；`sum`/`sum_out`/`sum_dim_comm` 先把输入 cast 到 dtype 再归约（ATen cast-first，修复 f32→i64 ±1 级分歧）。实数路径逐位不变（分支在 `is_complex()` 处短路）；实数 in + complex out 走 `_sum_complex`。 |
| `ops/zeros.py` | `zeros()` 对 complex 分配后 `zero_(out)`；complex `zero_()` = **`x.copy_(0); return x`**——直接 ATen redispatch 到 native 实现，**stride 正确**（切片/视图/转置张量的稀疏存储也被清零）、保持 dtype 与"返回入参张量"语义。（评审修正：旧实现经 `view_as_real` 递归会把连续视图的存储当稠密 buffer 清零，strided 张量非全零区间/原张量被污染。） |
| `ops/zero.py` | `_launch_zero_kernel` 同口径：complex 一律 `tensor.copy_(0); return tensor`，`zero`/`zero.out`/`zero_` 三入口覆盖。 |
| `ops/zeros_like.py` | complex 分支调 `zero_(out)` 继承 strides/dtype/布局（`zeros_like` 语义天然正确）。 |
| `ops/abs.py` | `abs()` complex 委托给已带 `_absolute_complex_kernel` 的 `absolute()`；`abs_()` complex 抛 `RuntimeError("In-place abs is not supported for complex tensors.")`，**与 native 逐字一致**（回归逐字比对）。 |
| `ops/absolute.py` | `_absolute_complex_kernel` 模长改用**缩放 hypot**：`m=max(|re|,|im|)`，`m_safe=where(m>0,m,1)`，`y=m*sqrt((re/m)^2+(im/m)^2)`，`y=where(is_inf,inf,y)`——inf 优先于 NaN、NaN 正常传播、避免大数中间量溢出（评审指正：旧 `sqrt(re^2+im^2)` 在 ~1e20 级溢出为 inf、NaN 形态不一致）；`absolute_()` complex 前置拒绝同 `abs_`。 |
| `ops/conv2d.py:288` | `input_c_stride[None]` → 标量直接参与广播（消除 constexpr `[None]` 崩溃），仅此一处。 |
| `_thead/ops/linear_backward.py:170/171` | `view` → `reshape`（非连续输入的机械修复）。 |

- 作用层级：全部是 FlagGems 算子实现层；未改任何注册表/autograd 注册
  （`__init__.py` 零改动），因此不可能改变 dispatch 与梯度注册结构。
- 依赖关系：`requires: [linear-5379]`（schema 字符串 ID）；patch 在基线
  `git apply --check` 干净；patch 应用结果与 `src/` 快照逐字节一致（8 文件 cmp 通过，
  `flagfix-fixed` 工作树按 `fix.yaml` after sha256 逐文件复核一致）。
- **`ops/linear.py` 不在本 patch**：line 158 归 linear-5379（已合入本仓库），
  其 after sha `ddec3383…` 钉入 `e2e_precondition`。本仓库 e2e 树构造 =
  基线 + `linear-5379.patch` + `zeros-7181.patch`。
- 数值影响：complex 求和/清零/模长与 native oracle 逐项一致（回归 139 用例：
  模长 rtol=1e-6 或 1e-10、atol=0、equal_nan；交叉 dtype out 矩阵 ok/err 双向
  + ATen 报错逐字；autograd 链 parity rtol=1e-4/atol=1e-6，限实数标量结尾链）；
  m2-bert 前向 loss 与 native **完全一致**（rel=0.000e+00）。

## 已知限制与上游跟踪

- 上游跟踪：issue #7181（同类）；本修复 PR 在本仓库提交，未向 flagos-ai/FlagGems
  提交上游 PR（记为 not-run，见 `reports/validation.md`）。
- 预存缺陷见上文 3：filter_fn 梯度缺口、梯度数值分歧、loss 存储间歇覆写，
  均有双树对照证据、均不属于本 patch 引入；`mean/full/ones/eq/where` 等 complex
  缺口未修。
- 容器侧 x-benchmark 回归（problem.md 指定的 `run_model.py` + 独立
  `--result-root`）依赖目标容器，本环境无 dockerexec/x-benchmark，未执行；
  `hardware_status: unverified`、`deployable=false` 不变。
