# 验证入口

所有脚本需把**目标源码树**（flagscale 仓库根，含 `flagscale/` 包）放在
`PYTHONPATH` 上：

```bash
export PYTHONPATH=/path/to/FlagScale          # 仓库根，不是 src/
python3 tests/<script>.py
```

注意：FlagScale 仓库布局是根下 `flagscale/` 包（无顶层 `src/`）；若用
`flag-fixes verify`，工具预置的 `PYTHONPATH=<target>/src` 不存在但无害，
fix.yaml 各命令的 `env.PYTHONPATH={target}` 负责真正生效的路径。

依赖：torch（CUDA）、omegaconf（`build_optimizer` 的配置入口）。无设备时
脚本打印 `SKIP: no CUDA device; exit 2` 并返回 **2**（跳过不算通过）。

设备：昆仑芯 P800 OAM（XPU3），torch 2.9.0+cu129（torch_xmlir shim），
profiler 开关 `XPU_ENABLE_PROFILER_TRACING=1`、`XPU_CUPTI_ENABLE_DEVICE=0`
（bench 的 launch 统计需要；不开也能跑，只是没有 kernel 明细）。

| 脚本 | 层级 | 覆盖 | 修复前预期 | 修复后预期 | 超时 |
| --- | --- | --- | --- | --- | --- |
| `repro_adamw_unfused.py` | repro | 336 张量 BGE-M3 形状参数集；`build_optimizer` 选择的实现与 stock AdamW 单步对照 + `defaults['fused']` | **exit 1**（fused=False，31.5 ms/步） | exit 0（fused=True，11.0 ms，ratio<0.75） | 900 s |
| `test_adamw_numerics.py` | regression | 5 步同梯度序列，bf16 + fp32，fused vs stock 逐步参数对照（allclose） | exit 1（基线不选 fused） | exit 0（bf16 3.1e-5 / fp32 3.0e-8） | 1800 s |
| `test_adamw_selection.py` | dispatch | 选择矩阵：auto/0/1/垃圾值 × CUDA/CPU/DTensor 替身/空列表 + build_optimizer 集成 | exit 1（新函数不存在） | exit 0 | 1800 s |
| `bench_adamw_impls.py` | benchmark | stock/foreach/build_optimizer 三实现：墙钟、launch 数、device 时间、top kernels（`--out FILE`） | exit 0（foreach 无融合收益） | exit 0（fused：launch 4× 减少） | 1800 s |

说明：

- `test_adamw_selection.py` 对 DTensor 的护栏用替身对象覆盖（构造真实
  DTensor 需要 dist 初始化）；`type(param).__module__` 前缀判断与实现一致。
- bench 的 launch 统计用 `torch.profiler`（CUDA activity），单次 step；
  墙钟 median-of-30。
- 基线树与修复树的构建见 `../reports/validation.md`；两棵树只有
  `flagscale/train/train_retrieval.py` 一个文件不同。
- 清单 `fix.yaml` 的 4 个 `validation.commands` 与上表一一对应。
