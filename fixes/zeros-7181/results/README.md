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
| `small-model-2026-11-10/bench-r2/zero-staging-attribution.log`（round-8 探针批）+ `zero-staging-attribution-r7-batch.log`（round-7 批归档） | 第九轮批次口径，根因归因（原生无注册）：`copy_(0)` 0.82–0.83ms ≈ f_target 0.90ms（ATen complex-标量拷贝 host 中转固有）；native `zero_` 0.004–0.006ms；**预建 device-0dim copy 净成本按批标注：round-7 批 0.0071/0.0069ms、round-8 批 0.0089/0.0067ms**（批间计时波动，非功能差异；单次创建 0.336ms 含冷启动；旧"0.02ms"系混入 `.cuda()` 构造，已 superseded）→ FIX-ITEM-3 |
| `small-model-2026-11-10/bench-r2/perf-section-auto.md` + `run-commits.json` | 第六轮：性能节由 gen_perf_section.py 从 bench2-summary.json 自动生成（旗标 115、比值、撤回清单）；机器可读 run provenance 清单 |
| `small-model-2026-11-10/baseline/` | 第十轮基线冻结（baseline_freeze.py 三组进程）：zero_/zero_ 五布局 × c64/c128 + conj/empty/0-dim + sum/abs 数值边缘（NaN/Inf/1e38/1e-38/empty/conj 位级对比）——fix vs native ~~52/58 位级一致~~（**superseded by round-12 严格比较器：50 numeric_equal + 8 numeric_diff，1–4 ULP**）、prefix **58/58 全崩**逐例记录、perf 三布局；旧 JSON 以 .r10-superseded 后缀归档 |
| `small-model-2026-11-10/bench-r2/zero-candidates.{json,log}` | 第十轮候选筛选（zero_candidates.py，原生 harness）：正确性矩阵 **60/60 全过**（6 布局 × c64/c128 × 当前/A/B/C/native，含未选中 storage 位级保持）；2^20 性能——B(预建设备标量) 0.0050–0.0135 ms 全布局最优、A(uint8-view memset) 0.0071–0.0135、C(real-imag) 0.0154–0.0241 vs 当前 0.67–1.89 ms |
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
| `small-model-2026-11-10/bench-r2/zero-candidates-r11-{native,registered}.{json,log}` | 第十一轮：候选矩阵扩展至 260 格（zeros 创建/empty/0-dim/组合布局/精确 alias，原生 260/260 全过）+ **注册环境复测**（A/B 事件全 native-aten 无 HtoD、0.0051–0.0075 ms；C 被淘汰 → FIX-ITEM-5）+ 复杂度（guard µs 级、B-only 建议首选） |
| `small-model-2026-11-10/bench-r2/u8sum-repro-{fix,prefix}.log` + `benchmarks/u8sum_repro.py` | 第十一轮 FIX-ITEM-4 专项复现：按评审清单（1d/多维/显式 dtype/>255/empty/reduction 维/非连续）直取值+dtype 双树对照——修复树与 prefix **同型 6/8 失败**（预存归因闭环；4 元素小例侥幸正确） |
| `small-model-2026-11-10/bench-r2/strided-zero-{fix,prefix}.log` + `benchmarks/strided_zero_probe.py` | 第十一轮 FIX-ITEM-5 最小复现：注册 zero_ 对实数跨步视图稠密写（未选中 storage 污染），双树同型（预存归因闭环） |
| `small-model-2026-11-10/bench-r2/zero-candidates-r12-{native,registered}-final.{json,log}` 等 | 第十二轮：harness 修正（dtype 贯穿——此前 c128 条目实际 float32；u8sum 显式 uint8）后的干净提交 fc71b82 重跑——候选原生 260/260、注册 224+36C（FIX-ITEM-5）；u8sum 7/8 双树同型失败（4 元素小例撤回）；严格比较器负例 8 项 |
| `small-model-2026-11-10/baseline/`（round-12 重跑）+ `diff-details.json` | 第十二轮：严格比较器（全字节 sha256/异常与缺字段独立分类/grad 显式 N/A）定论 **50 numeric_equal + 8 numeric_diff**；8 例 ULP/误差量化（conj ≤2 ULP、scaled-hypot 1 ULP、NaN payload 位不保证传播正确）；round-10/11 的 52/58 口径标注 superseded（r12-superseded-manifest.md） |
| `small-model-2026-11-10/baseline/diff-details.json`（round-14 重写）+ `ulp-mapping-selftest.json` | 第十四轮：位级 NaN payload 比较（同源 NaN+NaN·j 构造，in_bits_equal=True → out_checksum 差异=输出 NaN payload 位型）、分类完备指标（类不一致=失败；inf 符号/±0 位型单列 flag）、有序 ULP 映射（±0 策略）+ 精确断言自测 17/17——8 项 numeric_diff 定稿 conj ≤4 f64 ULP / abs 1–2 f32 ULP 等 |
