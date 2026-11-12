# KunlunXIN GELU 家族 `pow(x, 2)` 性能修复（乘方改乘法）

- ID：gelu-kunlunxin-pow
- 状态：draft（deployable=false；本机 P800 微基准/精度/回归已跑，平台整模型验收未做）
- 负责人：待填写
- 上游 issue/PR：待填写（问题来源见 `problem29/problem.md` 与 FlagScale `docs/retrieval/13-bottleneck-ops.md`）
- 受影响算子、软件层、后端：KunlunXIN `gelu` / `gelu_backward`（none 与 tanh 两条路径）、
  fused `gelu_and_mul`（tanh 路径）；FlagGems 厂商后端算子层，后端 `kunlunxin`
- 问题与修复后行为：GELU 家族用 `tl_extra_shim.pow(x, 2.0)` 求平方，XPU3 把整数指数 2 的
  `pow` 降低成慢速 libdevice 调用，前向/反向都被它支配（反向 2.54 ms，比原生 torch 慢 8.4x）。
  改成 `x * x` 后数学等价、误差不增，前反向反而快于原生 torch。

## 一句话版本

**GELU 里所有 `pow(x, 2.0)` 都只是求平方；在 XPU3 上它是 libdevice 调用而不是乘法。
换成 `x * x` 后，`gelu_backward` 从 2.54 ms 降到 0.154 ms（16.5x），tanh 前向从
0.312 ms 降到 0.054 ms（5.8x），fused `gelu_and_mul` 从 1.30 ms 降到 0.22 ms（5.9x）。**

```
基线（P800, bf16, 2990x1024）        修复后
gelu_backward            2.539 ms  -> 0.154 ms   (16.5x, 1.96x native)
gelu(tanh) forward       0.312 ms  -> 0.054 ms   ( 5.8x, 2.40x native)
gelu_backward(tanh)      0.574 ms  -> 0.064 ms   ( 9.0x, 2.19x native)
gelu_and_mul(tanh)       1.297 ms  -> 0.220 ms   ( 5.9x, 2.48x native)
gelu(none) forward       0.115 ms  -> 0.117 ms   (不经过 pow，逐位不变)
```

## 问题是怎么回事

`docs/retrieval/13-bottleneck-ops.md` 的结论是昆仑芯 P800 上 GELU 占整步 kernel 时间
35.2%，比 A100 慢 25.4x，其中 `gelu_bwd_highprecision_nv` 一项 654 ms/步。该文档同时指出
FlagGems 的 GELU 在 P800 上“实测持平/更慢”，不能直接替换厂商 aten 路径。

本修复针对 FlagGems 自己的 KunlunXIN GELU 实现。基线代码里三处平方都写成 libdevice `pow`：

```python
# _kunlunxin/ops/gelu.py（基线）
gelu_tanh:          pow(x.to(tl.float32), 2.0)
gelu_backward_none: pow(scale1 * x_fp32, 2.0)
gelu_backward_tanh: pow(x_fp32, 2.0) / pow(tanh_out, 2.0)
# _kunlunxin/fused/gelu_and_mul.py（基线）
gelu_tanh_and_mul:  pow(x_fp32.to(tl.float32), 2.0)
```

`pow` 在 XPU3 上不是一条乘法指令：它走 libdevice 调用，单元素成本远高于 `x*x`。实测同一
kernel 只改这一处（bf16, 2990x1024）：

| 表达式 | 耗时 |
| --- | ---: |
| `exp(-pow(s1*x, 2)) + erf(...)`（基线反向） | 2.538 ms |
| `exp(-t*t) + erf(...)`（仅换平方） | 0.162 ms |
| 只保留 `exp` | 0.060 ms |
| 只保留 `erf` | 0.121 ms |

`pow` 还会让部分手工 kernel 无法编译：把 `geglu` 的 `pow` 换成乘法前，`xpu3-elfconv`
直接返回非零；换掉后可编译（该 kernel 自身另有 21 ms 的配置问题，未纳入本 patch）。

## 修复

只把求平方改成乘法，必要时缓存 `x_fp32` 和平方值，其余表达式、dtype、dispatch 均不变：

```diff
-def gelu_tanh(x):
-    output = 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * pow(x.to(tl.float32), 2.0))))
+def gelu_tanh(x):
+    x_fp32 = x.to(tl.float32)
+    x_sq = x_fp32 * x_fp32
+    output = 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * x_sq)))

-    dydx = scale2 * x_fp32 * tl.exp(-pow(scale1 * x_fp32, 2.0)) + 0.5 * erf(scale1 * x_fp32) + 0.5
+    scaled = scale1 * x_fp32
+    dydx = scale2 * x_fp32 * tl.exp(-scaled * scaled) + 0.5 * erf(scaled) + 0.5
```

数值上 `pow(y, 2.0)` 与 `y*y` 对 fp32 是同一运算；实测误差没有变大（见下表），
`gelu(tanh)` bf16 的 max_abs_err 反而从 `1.5e-5` 降到 `3.8e-6`（`pow` 的中间精度更低）。

完整 diff 见 `patches/gelu-kunlunxin-pow.patch`。目标文件 hash：

| 文件 | before sha256 | after sha256 |
| --- | --- | --- |
| `_kunlunxin/ops/gelu.py` | `6c53c738…` | `e9c72e11…` |
| `_kunlunxin/fused/gelu_and_mul.py` | `a3192f8d…` | `2669825c…` |

## 验证结果（P800 OAM，2026-11-10）

环境：8× P800 OAM（本机用可见设备 0），XPU-SMI 驱动 5.0.21.47 / XPU-RT 5.13.0，
torch 2.9.0+cu129（torch_xmlir），triton 3.0.0，`GEMS_VENDOR=kunlunxin`，
`KLX_USE_AUTOTUNE=0`；输入 bf16 `(2990, 1024)`，同设备/同输入/同计时口径（CUDA event，
warmup 20，100 iter × 5 轮取最优）。

| 项目 | 修复前 | 修复后 | 倍数 | 修复后 / native |
| --- | ---: | ---: | ---: | ---: |
| `gelu_backward`（none） | 2.539 ms | **0.154 ms** | **16.5x** | 1.96x |
| `gelu(tanh)` forward | 0.312 ms | **0.054 ms** | **5.8x** | 2.40x |
| `gelu_backward`（tanh） | 0.574 ms | **0.064 ms** | **9.0x** | 2.19x |
| `gelu_and_mul(tanh)` | 1.297 ms | **0.220 ms** | **5.9x** | 2.48x |
| `gelu(none)` forward | 0.115 ms | 0.117 ms | 1.0x | 1.46x |
| `gelu_and_mul(none)` | 0.466 ms | 0.465 ms | 1.0x | 1.47x |

精度（max_abs_err / max_rel_err，修复前 → 修复后）：

| 用例 | 修复前 | 修复后 |
| --- | --- | --- |
| `gelu_backward` none | 7.8e-3 / 1.5e-3 | **2.0e-3 / 3.9e-4** |
| `gelu(tanh)` forward | 1.5e-5 / 3.1e-6 | **3.8e-6 / 7.7e-7** |
| `gelu_backward` tanh | 1.6e-2 / 3.1e-3 | 1.6e-2 / 3.1e-3 |
| `gelu_and_mul(tanh)` | 6.3e-2 / 4.5e-3 | 6.3e-2 / 4.5e-3 |

`tests/repro_gelu_pow.py`：修复前 exit 1（backward 8.41x native、tanh 前向 2.41x native，
阈值 2.0x），修复后 exit 0（0.51x / 0.42x）。

覆盖：`tests/test_gelu_accuracy.py`（3 dtype × 4 形状 × 连续/转置/切片/permute ×
none/tanh 前反向 + fused 前后向）与 `tests/test_gelu_family_dispatch.py`
（`enable()` 后 `F.gelu` / `nn.GELU` / autograd / 非连续输入 / gated MLP）均为 0 failed。
`flag-fixes validate` / `check` / `plan` / `verify` 通过，`apply` 按 `deployable=false` 正确拒绝。

## 目录结构

```
├── README.md                     # 本文
├── fix.yaml                      # 清单（schema_version 1）
├── WORKFLOW.md                   # 8 步执行清单及证据
├── patches/gelu-kunlunxin-pow.patch
├── src/.../gelu.py               # 修复后完整文件快照（与 patch 结果逐字节一致）
├── src/.../gelu_and_mul.py
├── tests/
│   ├── repro_gelu_pow.py          # level: repro（基线必失败）
│   ├── test_gelu_accuracy.py      # level: regression
│   ├── test_gelu_family_dispatch.py # level: dispatch
│   └── bench_gelu_family.py       # level: benchmark
├── benchmarks/README.md           # 性能/精度数据与口径
├── reports/{diagnosis,validation}.md
└── results/                       # 原始 JSON / 日志
```

## 怎么用

前置：FlagGems `v5.0.0`（`4e08b44e9b9ad68c2c27b4d25d04c6b76c30ff7a`），两个目标文件
hash 与上表一致。测试与应用必须导入打了补丁的源码树（例如 `PYTHONPATH="$TARGET/src"`）。

```bash
MANAGER=/path/to/flag-fixes
TARGET=/path/to/FlagGems
PYTHONPATH="$MANAGER/src" python3 -m flag_fixes --root "$MANAGER" \
  check gelu-kunlunxin-pow --target "$TARGET"
git -C "$TARGET" apply "$MANAGER/fixes/gelu-kunlunxin-pow/patches/gelu-kunlunxin-pow.patch"
PYTHONPATH="$TARGET/src" python3 "$MANAGER/fixes/gelu-kunlunxin-pow/tests/repro_gelu_pow.py"
```

## 已知边界（不装完美）

- **只覆盖 FlagGems 路径**：训练里的厂商 aten `gelu_highprecision_nv` /
  `gelu_bwd_highprecision_nv` 在厂商二进制里，本 patch 改不到；需要厂商提供 bf16 原生
  kernel（`docs/retrieval/13` 的结论）。本修复让“启用 FlagGems”这条路从负收益变成正收益。
- **只改 `_kunlunxin` 后端**：其他厂商后端的 `ops/gelu.py` 有同类写法，但慢的根因是
  XPU3 libdevice `pow`，不应在未验证的平台上批量改。
- **`_kunlunxin/fused/geglu.py` 未包含**：`pow` 会让其 `xpu3-elfconv` 编译失败；换乘法后
  可编译，但该 kernel 当前配置实测 21 ms（原生 0.31 ms），属于独立问题，不随本 patch 交付。
- **平台整模型验收 not-run**：problem 里的 x-benchmark `run_model.py` on 趟未在本环境执行；
  `hardware_status` 保持 `unverified`，`deployable=false`、`owner=null`，`apply` 会拒绝。
- **测量噪声**：本机为共享 P800（可见设备 0 当时有其它负载），表中为 best-of-5 的均值；
  趋势与倍数稳定，但绝对值会随卡负载变化。原始 JSON 未脱敏。
