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

## 边界语义证据（PR3 三轮评审，dumper v3）

| 文件 | 内容 |
| --- | --- |
| `boundary-base-2026-11-11.json/.log` | FlagGems @4e08b44e9（未打补丁）在 P800/XPU3 的边界输出（v3 dumper，含设备输入位型/FTZ），作为已提交基线 |
| `boundary-fixed-2026-11-11.json/.log` | 打补丁树同口径输出（verify_gelu_boundary.py 捕获，带 provenance） |
| `boundary-compare-2026-11-11.json` | compare_gelu_boundary.py 对比报告：结构/NaN 掩码/Inf 符号/±0 符号位/位型 + 冻结容差（fp32 gelu_tanh_bwd atol=2e-7）全部通过 |
| ⚠️ **上三行与 verify-2026-11-11-combo 为旧批次**：fp16 邻界输入误用 0x007F（bf16 邻界位型）；**0x03FF 正确邻界批次的 base/fixed dump 与 5 命令 verify pending**（base 树重跑触发 error 721 驱动异常并楔死设备，见 history/） |
| `boundary-fp64ref-2026-11-11.json` | 独立 fp64 参考对照（±3：base/fixed 绝对误差完全相同，pow→x*x 未引入额外误差） |
| `history/` | 旧 schema（无输入位型）的早期文件，已标记 superseded，仅存档 |

可重复入口：`tests/verify_gelu_boundary.py`（wrapper，5 命令之一，
`PYTHONPATH=<FlagGems-fixed 树>`），旧 schema/缺字段返回非零。

### fp16 邻界覆盖状态（2026-11-12 更新，如实记录）

- 评审指出 fp16 邻界位型应为 **0x03FF**（此前误用 bf16 的 0x007F）；
  `nm1_bits` 已改为按 dtype 选择并加邻界性断言（nm1+1bit == finfo.tiny）；
- **但 0x03FF 输入的 base/fixed dump 均未产出**：重跑时未打补丁树的
  pow kernel 触发**驱动级 kernel exception（error 721）**并楔死设备
  （见 `history/boundary-base-hang-2026-11-12.log`），fp16 邻界输入的
  base 对照因此挂起；
- 当前已提交基线的 fp16 用例使用 0x007F（实为 bf16 邻界位型，对 fp16
  是普通次正规数）——patch 前后对照仍有效（同输入双树），但**不构成
  fp16 邻界覆盖**；
- 设备当前楔死（新进程 `device='cuda'` 即挂起），需平台侧重置/协调；
  恢复后补跑：base/fixed 以 0x03FF 输入重生成 → comparator → 刷新本节。
