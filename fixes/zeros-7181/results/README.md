# 结果

2026-11-07 本机（PPU / thead / torch 2.10.0 / triton 3.6.0）实测，工具与脚本原文、
未脱敏。判定与命令见 `reports/validation.md`。

| 文件 | 内容 |
| --- | --- |
| `repro-baseline-2026-11-07.log` | 修复前 `repro_complex64.py`：exit 1，`ISSUE REPRODUCED` + `KeyError: 'complex64'` |
| `regression-baseline-2026-11-07.log` | 修复前 `test_complex_ops.py`：`TOTAL: 54 passed, 38 failed` |
| `regression-fixed-2026-11-07.log` | 修复后 `test_complex_ops.py`：`TOTAL: 54 passed, 0 failed` |
| `e2e-linearblocker-baseline-2026-11-07.log` | e2e 阶段一（基线 + linear-5379）：`linear_backward.py:170` view RuntimeError |
| `e2e-problem-path-noncomplex-2026-11-07.log` | e2e 阶段二（无 complex 修复）：反向 `sum.py:385 KeyError: 'complex64'` = problem.md 签名 |
| `e2e-fixed-2026-11-07.log` | e2e 修复后：`E2E OK` loss rel=0.000e+00，161 梯度存在且有限，384 跳过 |
| `e2e-missing-dep-2026-11-07.log` | 卸掉 `transformers`：`MISSING DEPENDENCY -> exit 2`（skip 不算通过） |
| `gradgap-probe-noncomplex-2026-11-07.log` | 无 complex 修复树的梯度缺口对照（`missing=34`、数值分歧形态） |
| `gradgap-probe-fixed-2026-11-07.log` | 修复树同探针：与上一行集合/形态一致 → 缺口属预存、与本 patch 无关 |
| `verify-2026-11-07-dirty.json` | `flag-fixes verify`（HEAD `f159ce07b199`，`source_dirty: true`）：`passed: true`，3/3 |
| `verify-2026-11-07.json` | `flag-fixes verify` 在已提交干净工作树上的重跑 |

不提交凭据、机器备份、模型缓存或无出处的测量数据；大日志若超仓库体量应改放发布附件并链接。
