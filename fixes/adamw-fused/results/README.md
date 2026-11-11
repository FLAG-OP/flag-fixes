# 结果

本目录保存小型 JSON / 日志，均为 2026-11-10 在本机（P800 OAM / XPU3）实测。
没有凭据、机器备份或模型缓存。全部文件合计 < 60 KB。

## 文件清单

| 文件 | 内容 |
| --- | --- |
| `bench-base-2026-11-10.json` / `.log` | 基线树 `tests/bench_adamw_impls.py --iters 30`：stock 31.2 ms/3,024 launches、foreach 27.5 ms/2,688、build_optimizer（foreach）27.6 ms/2,688 |
| `bench-fixed-2026-11-10.json` / `.log` | 修复树同口径：build_optimizer（fused）**11.5 ms / 672 launches**、device 8.9 ms |
| `repro-base-2026-11-10.log` | `tests/repro_adamw_unfused.py` 在基线：**exit 1**（fused=False，31.5 ms/步） |
| `repro-fixed-2026-11-10.log` | 同一脚本修复后：**exit 0**（fused=True，11.0 ms，ratio 0.354） |
| `numerics-base-2026-11-10.log` | `tests/test_adamw_numerics.py` 在基线：exit 1（基线不选 fused） |
| `numerics-fixed-2026-11-10.log` | 修复后：exit 0（bf16 max\|diff\| 3.05e-05、fp32 2.98e-08，allclose 全通过） |
| `selection-base-2026-11-10.log` | `tests/test_adamw_selection.py` 在基线：exit 1（`adamw_impl_kwargs` 不存在） |
| `selection-fixed-2026-11-10.log` | 修复后：exit 0（9 例选择矩阵全对，含 DTensor/CPU/env 护栏） |
| `verify-2026-11-10.json` / `.stderr.log` | 首次 `flag_fixes verify`（fixed 树）：4/4 rc=0，`source_dirty=true`（fix 尚未提交；stderr 空文件） |
| `verify-2026-11-10-clean.json` / `.stderr.log` | 干净提交树 `25afaebc5d89a1a30f7bafcbad518fdfa3eaf39c` 上复跑：4/4 rc=0，`source_dirty=false`（stderr 空文件） |

## 判定

**测量口径（PR #3 评审后修订）**：共享主机的 launch 路径（cudaLaunchKernel
CPU 时间）在一天内出现过 7 µs → 0.9–2.5 ms/次的漂移，会把 launch 重的
wall 对照反转。因此 repro 的性能判定改用 **torch.profiler 的 device time**
（需要 XPU_ENABLE_PROFILER_TRACING=1；CUPTI 不可用时 exit 2=skip，不算
通过）。

| 项目 | 判定 |
| --- | --- |
| 复现（exit 1 → exit 0） | ✅ |
| 数值回归（bf16/fp32 allclose） | ✅（fixed exit 0；基线 exit 1 是"未选 fused"的预期失败） |
| 选择矩阵 | ✅（fixed exit 0） |
| 性能前后对照（同参数集/同梯度/同计时） | ✅ 2.4× 墙钟、4× launch |
| `flag_fixes verify`（fixed 树，4 个声明命令） | ✅ 4/4 rc=0；干净树复跑 `source_dirty=false` |
| 平台整模型验收 | ❌ not-run（无平台 runner/模型/数据），`hardware_status=unverified` |

## 口径说明

- 参数集为 **336 张量的 BGE-M3 transformer 块形状**（不含 250002×1024 embedding，
  它对逐参数/融合的差异没有贡献）；归因报告里的 41.8→17.3 ms 是 389 张量
  真实参数集（含 embedding），两套数字都保留并注明。
- `--iters 30`、median-of-N、每步 `torch.cuda.synchronize()`。
- 日志保留了厂商栈的环境告警（`SYMBOL_REWRITE`、XCCL 加载），grep 过滤后
  的关键行见 WORKFLOW.md 表格。
