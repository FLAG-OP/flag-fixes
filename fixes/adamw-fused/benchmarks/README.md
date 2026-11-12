# 性能与精度

这是性能修复，必须给同设备、同栈、同参数集、同计时口径的前后对照。以下是本机
（单卡 P800 OAM / XPU3）数据，**不是**整模型吞吐，也不代表平台验收。

## 口径

| 项目 | 值 |
| --- | --- |
| 脚本 | `../tests/bench_adamw_impls.py` |
| 设备 | 昆仑芯 P800 OAM（XPU3），torch 2.9.0+cu129（torch_xmlir shim） |
| 参数集 | 336 张量 bf16，BGE-M3 transformer 块形状（qkvo 1024²、ff 4096×1024/1024×4096、bias/LN 向量）；归因报告的 389 张量真实参数集另见下表 |
| 梯度 | 每次计时前 `randn_like × 0.01`（同 seed 流） |
| warmup / 计时 | warmup 5；30 iter median，每步 `torch.cuda.synchronize()` |
| launch 统计 | `torch.profiler`（CUDA activity）单步 |

## 前后对照（build_optimizer 的选择）

| 树 | 选中实现 | 墙钟/步 | launches | device/步 |
| --- | --- | ---: | ---: | ---: |
| 基线 | foreach（厂商分解为逐参数） | 27.6 ms | 2,688 | 25.7 ms |
| 修复 | **fused** | **11.5 ms** | **672** | **8.9 ms** |

**2.4× 墙钟、4× launch、2.9× device。** foreach 与 stock 差距只有
~336 次 launch（每参数一次的派发开销），kernel 仍逐参数——这是
"foreach 在厂商栈上无融合收益"的直接证据（`bench-base-*.json`）。

## 归因口径（389 张量真实参数集，`probe_optim.py`）

| 实现 | 墙钟/步 | device/步 | launches |
| --- | ---: | ---: | ---: |
| stock | 41.8 ms | — | 3,890 |
| foreach=True | 41.0 ms | — | ~3,890 |
| **fused=True** | **17.3 ms** | — | 少量 |

## 数值

5 步同梯度序列，fused vs stock（`results/numerics-fixed-2026-11-10.log`）：

| dtype | max abs diff | allclose |
| --- | ---: | --- |
| bf16 | 3.05e-05（≈1 ulp @ 参数幅值） | ✅ |
| fp32 | 2.98e-08 | ✅ |

## 换算到真实负载（预估，非平台实测）

归因口径下 AdamW kernel 41.8→17.3 ms；平台文档"clip+optimizer 62 ms 墙钟"
→ 预期 ~37 ms（clip 的 9.8 ms 本修复不覆盖）。折合整步（kernel 2,923 ms）
约 **-0.8%**。`hardware_status=unverified`，此数字不作为验收依据。

原始数据：`../results/bench-{base,fixed}-2026-11-10.json`。

运行方式（`$TREE` 为打了补丁或未打补丁的 FlagScale 仓库根）：

```bash
PYTHONPATH="$TREE" python3 tests/bench_adamw_impls.py --iters 30 --out out.json
```
