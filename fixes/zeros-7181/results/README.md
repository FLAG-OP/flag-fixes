# 结果

本机（PPU / thead / torch 2.10.0 / triton 3.6.0）实测，工具与脚本原文、未脱敏。
**权威日志：`regression-*-2026-11-10.log` 与 `verify-2026-11-10.json` 为评审第三轮
（测试与文档小修后，当前代码）；`*-2026-11-09.log` 为评审第二轮（11-10 回归日志
的前身，测试样例修订前，其余命令仍为当前）**；`*-2026-11-08.log` 为评审第一轮
（139 用例口径）；`*-2026-11-07.log` 为初版历史原文，不改写。
判定与命令见 `reports/validation.md`。

| 文件 | 内容 |
| --- | --- |
| `regression-baseline-2026-11-10.log` | 修复前 `test_complex_ops.py`（第三轮测试，含非连续 conj 样例）：`TOTAL: 33 passed, 177 failed, 210 total`，exit 1 |
| `regression-fixed-2026-11-10.log` | 修复后 `test_complex_ops.py`（同上）：`TOTAL: 210 passed, 0 failed, 210 total`，exit 0；20 个 conj 用例（含非连续 direct/dispatch）全 PASS |
| `verify-2026-11-10.json` | 第三轮提交（`1fa248d`）后的干净 `flag-fixes verify`（3/3，`source_dirty: false`） |
| `repro-baseline-2026-11-09.log` | 修复前 `repro_complex64.py`：exit 1，`KeyError: 'complex64'` |
| `repro-fixed-2026-11-09.log` | 修复后 `repro_complex64.py`：exit 0 |
| `regression-baseline-2026-11-09.log` | （第二轮历史）修复前 `test_complex_ops.py`（样例修订前）：`TOTAL: 33 passed, 177 failed, 210 total` |
| `regression-fixed-2026-11-09.log` | （第二轮历史）修复后 `test_complex_ops.py`（样例修订前）：`TOTAL: 210 passed, 0 failed, 210 total` |
| `e2e-fixed-2026-11-09.log` | e2e 修复后：precondition OK（9 文件）、revision pin、`E2E OK` loss rel=0.000e+00、161 梯度 present+finite、384 缺口与 fixture 双向断言 |
| `e2e-precondition-baseline-2026-11-09.log` | e2e 跑在基线树：模型工作前 exit 2 `MISSING PRECONDITION`（linear.py 失配） |
| `e2e-precondition-linearonly-2026-11-09.log` | e2e 跑在仅 linear-5379 树：exit 2 `MISSING PRECONDITION`（sum.py 等逐条失配） |
| `e2e-missing-dep-2026-11-09.log` | 隔离 `transformers` import：`MISSING DEPENDENCY -> exit 2`（skip 不算通过） |
| `nocuda-regression-2026-11-09.log` | 强制无设备（`CUDA_VISIBLE_DEVICES=`）：exit 2 |
| `cli-validate-2026-11-09.log` 等 `cli-*.log` | validate/check/plan/apply/status 各退出码原文；`cli-check-requiredfail-2026-11-09.log` = 漏依赖 ID exit 2 `include required fixes: ['linear-5379']` |
| `verify-2026-11-09.json` | 第二轮提交后的干净 `flag-fixes verify`（3/3） |
| `repro-*-2026-11-08.log`、`regression-*-2026-11-08.log`、`e2e-*-2026-11-08.log`、`cli-*-2026-11-08.log`、`nocuda-*-2026-11-08.log` | （第一轮历史）评审 must-fix 后的139 用例口径（18/121/139 → 139/0/139），对应提交 `085e199`/`9fa84cf`；当前代码复跑见 -11-09 |
| `verify-2026-11-08.json` | （第一轮历史）评审修正轮提交后的干净 `flag-fixes verify`（3/3，commit `085e199`） |
| `e2e-linearblocker-baseline-2026-11-07.log` | （历史）e2e 阶段一：`linear_backward.py:170` view RuntimeError |
| `e2e-problem-path-noncomplex-2026-11-07.log` | （历史）e2e 阶段二：反向 `sum.py:385 KeyError: 'complex64'` = problem.md 签名 |
| `e2e-fixed-2026-11-07.log` / `e2e-missing-dep-2026-11-07.log` | （历史）初版修复后 e2e 与缺依赖日志 |
| `regression-*-2026-11-07.log` | （历史）**注意 TOTAL 行计数器缺陷**：printed N = passed+failed，旧口径实际 16 passed / 38 failed / 54 total（38 条失败全为 complex KeyError）；计数器 11-08 已修正，原文不改写 |
| `repro-baseline-2026-11-07.log` | （历史）初版修复前复现 |
| `gradgap-probe-noncomplex-2026-11-07.log` | 无 complex 修复树的梯度缺口对照（`missing=34`、数值分歧形态） |
| `gradgap-probe-fixed-2026-11-07.log` | 修复树同探针：与上一行集合/形态一致 → 缺口属预存、与本 patch 无关 |
| `verify-2026-11-07-dirty.json` / `verify-2026-11-07.json` | 初版 `flag-fixes verify`：`passed: true`，3/3（dirty / 已提交干净树） |

不提交凭据、机器备份、模型缓存或无出处的测量数据；大日志若超仓库体量应改放发布附件并链接。
