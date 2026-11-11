# 结果

本机（PPU / thead / torch 2.10.0 / triton 3.6.0）实测，工具与脚本原文、未脱敏。
**权威日志：`regression-*-2026-11-10.log` 与 `verify-2026-11-10.json` 为评审第三轮
（测试与文档小修后，当前代码）；`*-2026-11-09.log` 为评审第二轮（11-10 回归日志
的前身，测试样例修订前，其余命令仍为当前）**；`*-2026-11-08.log` 为评审第一轮
（139 用例口径）；`*-2026-11-07.log` 为初版历史原文，不改写。
第四轮（评审意见 6756872311）的小模型评估证据整体放在
`small-model-2026-11-10/`（矩阵跑在干净提交 `968d624`、bench 跑在 `d236d5a`，
**该两提交间**（968d624..d236d5a）被测 tests 文件逐字节一致；**注意**：后续
fac401c 对 `tests/test_small_models.py` 加固比较器（226 行差异）并对已存
dumps 重评分——设备运行用的是当时 harness，见 run-commits.json 与第六轮节），
判定见 `reports/small-model-evaluation.md`。

| 文件 | 内容 |
| --- | --- |
| `small-model-2026-11-10/bench-r2/zeros-hitprobe-{f_target,native}.json` | 第六轮：zeros 族 profiler **全事件清单**（分类、计数、设备时间；dtype/shape/调用入口/同步方式记录）——f_target 下 `zero_` 仅 Memcpy HtoD、`zeros` 为 native FillFunctor+HtoD、无 Triton kernel；native 下 `zero_` 为 3µs memset |
| `small-model-2026-11-10/bench-r2/zero-staging-attribution.log` | 第六轮根因归因（原生无注册；**round-8 更新**：设备标量构造分列单次/50 次平均）：`copy_(0)` 0.82–0.83ms ≈ f_target 0.90ms（ATen complex-标量拷贝 host 中转固有）；native `zero_` 0.004–0.006ms；**预建 device-0dim copy 净成本 0.0071/0.0069ms**（旧"0.02ms"系混入 `.cuda()` 构造，已 superseded）→ FIX-ITEM-3 |
| `small-model-2026-11-10/bench-r2/perf-section-auto.md` + `run-commits.json` | 第六轮：性能节由 gen_perf_section.py 从 bench2-summary.json 自动生成（旗标 115、比值、撤回清单）；机器可读 run provenance 清单 |
| `small-model-2026-11-10/matrix-rescored/` | 第五轮：比较器结构加固（shape/dtype/键集合/step ID/参考无效判 fail）后用已存 .pt 重评分 20 对+噪声对（f_target 9/9+noise PASS 维持，原始 compare 不改写）；不依赖设备负例自测 10/10；mse 与 complex-fill 复现在修复树与修复前对照树的双树日志（预存归因隔离对照） |
| `small-model-2026-11-10/bench-r2/` | 第五轮性能（跑在干净提交 `dd5ebea`）：按族×组独立进程注册（f_target 期望键断言）+ profiler kernel 名命中验证 + calls_per_s 公式修正 + 块均值 P95 + 显存 baseline/peak/delta + 候选旗标计数 115（自动生成）+ 正确性失败配置标 diagnostic；`bench2-summary.json`/`bench2-table.md` |
| `small-model-2026-11-10/matrix/` | 第四轮准确性：冻结 `thresholds.json`、`lr.json`（0.1/0.1）、`fd_*.json`（PASS）、`summary.json`（29 run / 20 对比 / 噪声门 max_abs=0.0）、逐 run `.pt`+`.meta.json`（确定性/注册范围/布局前提/异常全记录）、全部 compare JSON；f_target 9/9 PASS（fp32≤4.8e-7、fp64≤8.9e-16），b 组崩溃形态与 f_all 首分歧节点见 meta |
| `small-model-2026-11-10/bench/` | 第四轮性能（**historical / superseded by bench-r2**：其 zeros/zero_ 结论与 58 旗标计数已撤回，当前口径见 bench-r2 与第六轮/第七轮节）：`bench-{n,f_target,f_all,b,n_rep}.json`（5 块×100 迭代原始块计时、首调单独记、峰值显存）、`bench-summary.json`（58 项 >10% 回归旗标）、`bench-table.md` |
| `small-model-2026-11-10/mse-scalar-repro.log` | f_all-B 首分歧最小复现输出：`mse_loss_backward` 标量 grad_output 不广播（39/40 错、第 0 元素精确、全量张量精确、none/sum 精确）；脚本 `tests/repro_mse_scalar_grad.py` |
| `small-model-2026-11-10/provenance.json` | 两组运行的 commit、分组定义、阈值冻结与校准、bench 协议 |
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
