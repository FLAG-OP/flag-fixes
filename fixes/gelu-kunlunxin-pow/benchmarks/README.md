# 性能与精度

这是性能修复，必须给同设备、同栈、同输入、同计时口径的前后对照。以下是本机
（单卡 P800 OAM / XPU3）数据，**不是**整模型吞吐，也不代表平台验收。

## 口径

| 项目 | 值 |
| --- | --- |
| 脚本 | `../tests/bench_gelu_family.py` |
| 设备 | 昆仑芯 P800 OAM（XPU3），`cuda:0`（本进程 `CUDA_VISIBLE_DEVICES=1,2`，用可见设备 0） |
| 栈 | torch 2.9.0+cu129（torch_xmlir）、triton 3.0.0、`GEMS_VENDOR=kunlunxin`、`KLX_USE_AUTOTUNE=0` |
| dtype | bf16（`--dtype bfloat16`） |
| 输入 | `(2990, 1024)`，`torch.randn(seed=0)`；fused `gelu_and_mul` 用 `(2990, 4096)` |
| warmup / 计时 | warmup 20 次；每轮 100 iter，CUDA event 计时，5 轮取最优轮均值 |
| native 参考 | 在 `flag_gems.enable()` 之前用原生 torch 计算并计时 |
| 同步 | 每轮 `torch.cuda.synchronize()`；进程内前后对照 |
| 精度门限 | fp32 `atol=rtol=1e-4`；bf16/fp16 `atol=5e-2, rtol=3e-2`（相对参考幅值） |
| 退出码 | 0 成功；1 精度/基准失败；2 无设备（跳过不算通过） |
| 原始数据 | `../results/bench-{base,fixed}-2026-11-10.json` |

运行方式（`$TREE` 为打了补丁或未打补丁的 FlagGems 源码树）：

```bash
PYTHONPATH="$TREE/src" python3 tests/bench_gelu_family.py --iters 100 \
  --json results/bench-$(basename "$TREE")-2026-11-10.json
```

## 前后对照（bf16, 2026-11-10）

| 用例 | 修复前 | 修复后 | 加速 | native | 修复后/native | max_abs_err 前→后 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `gelu_fwd_none` | 0.1149 ms | 0.1167 ms | 0.99x | 0.170 ms | 1.46x | 1.53e-5 → 1.53e-5 |
| `gelu_fwd_tanh` | 0.3124 ms | **0.0542 ms** | **5.77x** | 0.130 ms | 2.40x | 1.53e-5 → 3.81e-6 |
| `gelu_bwd_none` | 2.5392 ms | **0.1538 ms** | **16.51x** | 0.302 ms | 1.96x | 7.81e-3 → 1.95e-3 |
| `gelu_bwd_tanh` | 0.5743 ms | **0.0639 ms** | **8.98x** | 0.139 ms | 2.19x | 1.56e-2 → 1.56e-2 |
| `gelu_and_mul_none` | 0.4665 ms | 0.4650 ms | 1.00x | 0.685 ms | 1.48x | 6.25e-2 → 6.25e-2 |
| `gelu_and_mul_tanh` | 1.2972 ms | **0.2197 ms** | **5.90x** | 0.546 ms | 2.48x | 6.25e-2 → 6.25e-2 |

读法：

- 走 `pow` 的四个用例（tanh 前向、两条反向、fused tanh）全部大幅下降，且修复后
  快于原生 torch（1.96x–2.48x）。
- 不经过 `pow` 的 `gelu(none)` 前向和 `gelu_and_mul(none)` 逐位不变、耗时不变。
- bf16 的 `max_abs_err` 在 1e-2 量级是 bf16 输出舍入（参考值幅值 ~4），
  `max_rel_err` 为 4.5e-3，在 bf16 容差内；fp32 路径误差 ~1e-7，见
  `../results/accuracy-*.log`（`tests/test_gelu_accuracy.py` 通过 fp32/bf16/fp16 全矩阵）。

## 机制对照（单 kernel）

同一 bf16 `(2990, 1024)`、同一 pointwise 配置，只改反向 kernel 内求平方方式：

| kernel 内容 | 耗时 |
| --- | ---: |
| 基线 `exp(-pow(s1*x, 2)) + 0.5*erf(s1*x) + 0.5` | 2.538 ms |
| 仅 `pow` → `t*t` | 0.162 ms |
| 只保留 `exp` 项 | 0.060 ms |
| 只保留 `erf` 项 | 0.121 ms |

说明成本由 `pow` 支配，不是 GELU 的超越函数本身。

## 精度矩阵

`../tests/test_gelu_accuracy.py`：

- dtype：fp32 / bf16 / fp16
- 形状：`(1, 64)`、`(17, 33)`、`(2990, 1024)`、`(3, 128, 512)`
- 布局：连续、转置、切片（末维 `::2`）、permute（3D）
- 算子：`gelu` / `gelu_backward` × {none, tanh}，fused `gelu_and_mul` 前向 + 两个输入梯度
- 结果：修复前后均 0 failed；修复后 tanh 前向与 none 反向误差反而更小。

`../tests/test_gelu_family_dispatch.py`：`flag_gems.enable()` 后 `F.gelu`、`nn.GELU(tanh)`、
`aten.gelu_backward`、`autograd.grad`、非连续输入、gated MLP，修复前后均 0 failed。

## 限制

- 共享卡：测量时物理 1 号卡有其他负载（xpu-smi 显示 70% util），绝对值会漂移；
  倍数为同进程前后对照，趋势稳定。
- 未测整模型；微基准倍数不能直接换算为训练吞吐提升。
- 未覆盖 `geglu`（pow 换成乘法后能编译，但 kernel 配置本身约 21 ms）与厂商 aten
  highprecision GELU 路径。
