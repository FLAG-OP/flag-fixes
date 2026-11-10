# 结果

本目录保存小型 JSON / 日志，均为 2026-11-10 在本机（P800 OAM / XPU3）实测。
没有凭据、机器备份或模型缓存。大文件不需要外链：全部文件合计 < 60 KB。

## 文件清单

| 文件 | 内容 |
| --- | --- |
| `bench-base-2026-11-10.json` | 基线（未打补丁）GELU 家族耗时 + 精度，`tests/bench_gelu_family.py --iters 100` |
| `bench-fixed-2026-11-10.json` | 修复后同口径数据（与 base 同输入、同 seed、同进程结构） |
| `bench-base-2026-11-10.stdout.log` / `.stderr.log` | 基线 benchmark 原始 stdout/stderr（环境警告保留） |
| `bench-fixed-2026-11-10.stdout.log` / `.stderr.log` | 修复后同上 |
| `repro-base-2026-11-10.log` | `tests/repro_gelu_pow.py` 在基线：**exit 1**（8.41x / 2.41x native） |
| `repro-fixed-2026-11-10.log` | 同一脚本修复后：**exit 0**（0.51x / 0.43x native） |
| `accuracy-base-2026-11-10.log` / `accuracy-fixed-2026-11-10.log` | `tests/test_gelu_accuracy.py`，修复前后均 0 failed |
| `dispatch-base-2026-11-10.log` / `dispatch-fixed-2026-11-10.log` | `tests/test_gelu_family_dispatch.py`，修复前后均 0 failed |
| `verify-2026-11-10.json` | 首次 `flag_fixes verify`：4/4 rc=0，`source_dirty=true`（fix 尚未提交） |
| `verify-2026-11-10.stderr.log` | 首次 verify 的 stderr（空文件） |
| `verify-2026-11-10-clean.json` | 最终树 `d57e825472f66809ac1224dd7aca8c6e4bec47a3`（fix 提交 `3e95f021` + 文档同步）的干净树上复跑：4/4 rc=0，`source_dirty=false` |
| `verify-2026-11-10-clean.stderr.log` | 干净树 verify 的 stderr（空文件） |

## 判定

| 项目 | 判定 |
| --- | --- |
| 最小复现 | 基线失败 / 修复后通过（exit 1 → 0） |
| 精度回归 | 通过（0 failed，3 dtype × 4 形状 × 4 布局 × none/tanh + fused） |
| dispatch 回归 | 通过（0 failed） |
| 性能 | tanh 前向 5.77x、none 反向 16.51x、tanh 反向 8.98x、fused tanh 5.90x；none 路径不变 |
| 工具链 | `validate` / `check` / `plan` / `verify` 通过；`apply` 按 `deployable=false` 拒绝 |
| 平台整模型 | **not-run**（无 runner/模型缓存），`hardware_status=unverified` |
| 上游 PR | **not-run** |

原始环境：torch 2.9.0+cu129（torch_xmlir）、triton 3.0.0、`GEMS_VENDOR=kunlunxin`、
XPU-SMI 5.0.21.47 / XPU-RT 5.13.0、8× P800 OAM。详见 `../reports/validation.md`。
