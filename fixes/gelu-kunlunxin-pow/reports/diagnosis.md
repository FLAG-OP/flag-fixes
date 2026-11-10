# 根因与修复方案

## 原始问题与预期行为

问题来源：`problem29/problem.md`（“gelu / cast_te / findmax1d / tensor_calc_basic /
transpose 等算子严重拖慢训练吞吐”），证据链在 FlagScale `docs/retrieval/13-bottleneck-ops.md`
与 `docs/retrieval/08-kunlunxin.md`。

- 平台：昆仑芯 P800 OAM（XPU3）。`08-kunlunxin.md` 的逐 kernel 表显示 activation（GELU）
  一项从 A100 的 40.5 ms/步涨到 1,028.9 ms/步，占整步 kernel 时间 **65.8%**，
  相对 A100 **25.4x**；`gelu_bwd_highprecision_nv` 654.2 ms + `gelu_highprecision_nv`
  374.7 ms。
- `13-bottleneck-ops.md` 的结论是 FlagGems 在 P800 上 10 个算子 9 个更慢（GELU 0.65x），
  不能整体替换厂商路径；同时把 `cast_te` / `findmax1d` / `tensor_calc_basic` /
  `transpose` 归类为“布局/dtype 转换与 kernel 未融合”的胶水 kernel，需要厂商融合或图级
  优化——FlagGems 单算子层无法消除它们。
- 本修复只处理其中可由 FlagGems 修改、且能给出同口径前后数据的一项：FlagGems
  KunlunXIN 后端的 GELU 家族。预期行为：与原生 torch 数值一致（同 dtype 容差内），
  前反向耗时显著下降，不改变 dispatch、dtype 提升规则和 API。

## 根因与证据

根因：KunlunXIN 后端把“求平方”写成 `tl_extra_shim.pow(y, 2.0)`。在 XPU3 上
`pow` 不是一条乘法指令，而是 libdevice 调用（`__nv_powf` 之类），单元素成本远高于
`y * y`，并且支配了整个 kernel。涉及位置（基线 `v5.0.0`）：

```python
# src/flag_gems/runtime/backend/_kunlunxin/ops/gelu.py
gelu_tanh:          pow(x.to(tl.float32), 2.0)
gelu_backward_none: pow(scale1 * x_fp32, 2.0)
gelu_backward_tanh: pow(x_fp32, 2.0)、pow(tanh_out, 2.0)

# src/flag_gems/runtime/backend/_kunlunxin/fused/gelu_and_mul.py
gelu_tanh_and_mul:  pow(x_fp32.to(tl.float32), 2.0)
```

对照实验（不是猜测）：同一 bf16 `(2990, 1024)`、同一 `pointwise_dynamic` 默认配置，
只改 kernel 内部求平方方式：

| 反向 kernel 内容 | 耗时 |
| --- | ---: |
| 基线：`0.5*erf(s1*x) + s2*x*exp(-pow(s1*x,2)) + 0.5` | 2.538 ms |
| 仅把 `pow(s1*x, 2)` 换成 `t*t`（`t = s1*x`） | 0.162 ms |
| 只保留 `exp(-x*x)` 项 | 0.060 ms |
| 只保留 `0.5*erf(s1*x)+0.5` 项 | 0.121 ms |

前向 tanh 路径同样：`pow` 版 0.3124 ms → 乘法版 0.053 ms（单 kernel 实验 0.053 ms，
交付后含 dispatch 为 0.0542 ms）。说明 `pow` 本身就是主要成本，而不是 GELU 的
`erf`/`tanh`/`exp` 数学。

旁证（编译层）：同样写法在 `_kunlunxin/fused/geglu.py` 中会让 `xpu3-elfconv` 直接
失败（`subprocess.CalledProcessError: xpu3-elfconv ... returned non-zero exit status 1`）；
把 `pow` 换成乘法后可编译。这进一步说明 XPU3 对 `pow` 的处理是特殊路径而非普通算子。

## 修复与影响

修改文件（2 个，均为 KunlunXIN 厂商后端，不涉及通用 `src/flag_gems/ops/gelu.py`）：

| 文件 | 改动 |
| --- | --- |
| `_kunlunxin/ops/gelu.py` | `gelu_tanh`、`gelu_backward_none`、`gelu_backward_tanh` 中 4 处 `pow(·, 2.0)` → 乘法；删除不再使用的 `pow = tl_extra_shim.pow` |
| `_kunlunxin/fused/gelu_and_mul.py` | `gelu_tanh_and_mul_kernel` 中 1 处 `pow(x_fp32, 2.0)` → 乘法；删除 `pow` 导入 |

作用机制：`pow(y, 2.0)` 与 `y*y` 在 fp32 上数学等价；后者编译成单条乘法，避免
libdevice 调用。必要时缓存 `x_fp32` 与 `x_sq` 以避免重复类型转换和重复乘法。

影响层：只影响 KunlunXIN 后端 `gelu` / `gelu_backward` / `gelu_and_mul` 的 kernel 实现。
`aten::gelu`、`aten::gelu_backward` 的 schema、dtype promotion 和 dispatch 不变；直接
调用 `flag_gems.gelu*`（测试路径）与 `flag_gems.enable()` 后的 `F.gelu` / `nn.GELU`
走同一实现，已验证一致（`tests/test_gelu_family_dispatch.py`）。

依赖/替代关系：无 `requires`、无 `conflicts`。与厂商 aten 的
`gelu_bwd_highprecision_nv` 路径互不影响——本修复不能让厂商 kernel 变快，只是让
“启用 FlagGems GELU”从负收益（0.119x native）变成正收益（1.96x native）。

## 已知限制与上游跟踪

- **未覆盖 `_kunlunxin/fused/geglu.py`**：它的 `pow` 会导致 `xpu3-elfconv` 编译失败，
  去掉 `pow` 后能编译，但该 kernel 当前配置实测约 21 ms（原生 0.31 ms），是独立的
  tiling/配置问题，不应混入“乘方改乘法”这个最小 patch。
- **未覆盖其他厂商后端**：`_cambricon` / `_enflame` / `_hygon` / `_arm` 的 GELU 也有
  类似 `pow` 写法，但慢的根因是 XPU3 libdevice，未在这些平台验证前不做批量修改。
- **未覆盖胶水 kernel**：`cast_te` / `findmax1d` / `tensor_calc_basic` / `transpose`
  是厂商布局/dtype 转换 kernel，FlagGems 单算子层无法消除（见
  `13-bottleneck-ops.md`）；本轮不宣称解决。
- **平台整模型验收 not-run**：没有运行 problem 要求的 x-benchmark `run_model.py` on 趟，
  `hardware_status` 保持 `unverified`，`deployable=false`。
- **上游跟踪**：未提交 FlagGems PR；无合入 commit、无发行版本。上游 issue 待创建。
