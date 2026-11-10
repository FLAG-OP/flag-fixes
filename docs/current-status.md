# 当前结果汇总

[返回文档导航](index.md) · [仓库首页](../README.md) · [最新 CI 记录](https://github.com/FLAG-OP/flag-fixes/actions/workflows/ci.yml)

## 验证基线与结果

本页汇总初始化提交 [`2275e66`](https://github.com/FLAG-OP/flag-fixes/commit/2275e66f5457e40c1d2d651af50baf4572ca93a9) 的验证结果。后续最新运行请查看上方 CI 链接；表中的通过状态仅针对该基线。

| 项目 | 当前记录 | 证据 / 详情 |
| --- | --- | --- |
| 来源接入 | 8 个固定 commit 的 Git submodule，9 项修复 | [三方维护说明](../third_party/README.md)、[修复清单](../catalog/fixes/) |
| 目录校验 | 通过：来源指针、patch hash、快照 hash 和 schema | [初始化 CI](https://github.com/FLAG-OP/flag-fixes/actions/runs/41778171519)、[清单约定](manifest-spec.md) |
| 管理工具测试 | 26 项通过；Python 3.10 / 3.12 CI 均通过 | [初始化 CI](https://github.com/FLAG-OP/flag-fixes/actions/runs/41778171519)、[测试代码](../tests/) |
| 包构建 | sdist 和 wheel 构建通过 | [初始化 CI 的构建步骤](https://github.com/FLAG-OP/flag-fixes/actions/runs/41778171519)、[包配置](../pyproject.toml) |
| 来源交付一致性 | 5 项 source-checked，4 项 needs-adaptation | [检查方法、逐项 patch/源码链接和差异](source-audit.md) |
| 本仓库硬件复验 | 尚未执行 NPU/XPU 测试，9 项均 unverified | [验证记录说明](../validation/README.md) |
| 可部署条目 | 0 项；导入条目均为草稿，apply 会拒绝 | [安装行为及限制](installation.md) |
| 平台 bundle / 离线发布 | 尚未交付 | [bundle 规划](../bundles/README.md)、[发布流程](release-process.md) |

目录和安装工具的测试通过，不等于算子已通过设备验证。下表的历史报告是原仓库记录，不是本仓库复跑结果。

## 修复逐项索引

fix 名称链接到本仓库清单；历史说明固定到当前 submodule commit，避免上游更新后旧结果与新源码混用。源码与 patch 的逐项链接见 [来源交付检查](source-audit.md)。

| fix / 清单 | 后端范围 | 来源检查 | 固定版本的历史说明与结果入口 |
| --- | --- | --- | --- |
| [arith-5732](../catalog/fixes/arith-5732.yaml) | ascend, spacemit | source-checked | [arithFIX README](https://github.com/FLAG-OP/arithFIX/blob/b73fbf173e1ea1e8ac7ed08316b5f96205ef02a9/README.md) |
| [cann-pow-promotion](../catalog/fixes/cann-pow-promotion.yaml) | ascend | needs-adaptation | [geluFIX README](https://github.com/FLAG-OP/geluFIX/blob/f07f22206a0961c3713c7ceea9f86bb2750e3dcd/README.md) |
| [copy-kunlunxin](../catalog/fixes/copy-kunlunxin.yaml) | kunlunxin | source-checked | [copyFIX README](https://github.com/FLAG-OP/copyFIX/blob/2b348a37fc1bc67c9226eb5371fd5431659f8e83/README.md) |
| [gather-5746](../catalog/fixes/gather-5746.yaml) | ascend | needs-adaptation | [gatherFIX README](https://github.com/FLAG-OP/gatherFIX/blob/dbb6b6d06f93fa5ee280b0ded3273086bbc2915a/README.md) |
| [gelu-5867](../catalog/fixes/gelu-5867.yaml) | ascend | source-checked | [geluFIX README](https://github.com/FLAG-OP/geluFIX/blob/f07f22206a0961c3713c7ceea9f86bb2750e3dcd/README.md) |
| [lift-fresh-5743](../catalog/fixes/lift-fresh-5743.yaml) | ascend | needs-adaptation | [liftfreshFIX README](https://github.com/FLAG-OP/liftfreshFIX/blob/233323cde6ce9c4cb34b284a1319aa243ff24be7/README.md) |
| [mm-bf16-kunlunxin](../catalog/fixes/mm-bf16-kunlunxin.yaml) | kunlunxin | source-checked | [mmBF16FIX README](https://github.com/FLAG-OP/mmBF16FIX/blob/09967ec546277c8175fe4d1021daeb3e3b3678cd/README.md) |
| [one-hot-5802](../catalog/fixes/one-hot-5802.yaml) | ascend | needs-adaptation | [onehotFIX README](https://github.com/FLAG-OP/onehotFIX/blob/3cef1d720fc5ce42978be4b7fc1c500eb516278f/README.md) |
| [pointwise-5739](../catalog/fixes/pointwise-5739.yaml) | ascend | source-checked | [pointwiseFIX README](https://github.com/FLAG-OP/pointwiseFIX/blob/a6b538daa4b61f8da5e368e92847a45ae79ba856/README.md) |

上表只覆盖 9 项外部来源修复；本仓库另有 2 项本地维护修复，见下节，合计 11 项 fix 索引。

所有条目的 `deployable=false`、`hardware_status=unverified`。gather、lift_fresh、one_hot 的来源差异是 patch 与完整源码的注释/署名不一致；CANN pow 条目需要补丁格式和目标基线适配。

## 本次变更记录（不改写上节基线）

以下条目晚于上节 `2275e66` 基线，基线表中的历史数字保持原样；目录校验与管理工具测试的通过状态以对应 CI 运行为准。

| 变更 | 内容 | 证据 |
| --- | --- | --- |
| 2026-11-05 新增本地 fix `linear-5379` | 本仓库第 1 项本地修复（fix 索引合计 10 项 = 9 项外部 + 1 项本地）：FlagGems `linear` 对非连续输入调用 `view` 崩溃，改为 `reshape(...).contiguous()`；`deployable=false`、`hardware_status=unverified`，`apply` 会拒绝 | [fix 目录](../fixes/linear-5379/)、[诊断](../fixes/linear-5379/reports/diagnosis.md)、[验证记录](../fixes/linear-5379/reports/validation.md)、[verify 结果](../fixes/linear-5379/results/)、PR #1 |
| 2026-11-07 新增本地 fix `zeros-7181` | 本仓库第 2 项本地修复（fix 索引合计 11 项 = 9 项外部 + 2 项本地）：m2-bert 反向 `KeyError: 'complex64'`（triton 无 complex mangle 条目），8 文件一次修复（`sum`/`zeros` 族/`abs`/`conv2d` constexpr/`_thead` `linear_backward` view），e2e 三树剥洋葱定位；`deployable=false`、`hardware_status=unverified`，`apply` 会拒绝 | [fix 目录](../fixes/zeros-7181/)、[诊断](../fixes/zeros-7181/reports/diagnosis.md)、[验证记录](../fixes/zeros-7181/reports/validation.md)、[verify 结果](../fixes/zeros-7181/results/) |
| 2026-11-08 `zeros-7181` 评审修正轮（PR #2） | 按评审 must-fix 修订并全量复跑：complex `zero_` 改原生 `copy_(0)`（stride 正确）、`abs` 模长改缩放 hypot（inf 优先 NaN）、regression 计数器修正并扩至 139 用例（修复前 18/121/139、修复后 139/0/139；旧 11-07 日志 TOTAL 行注明缺陷、实际 16/38/54）、`requires: [linear-5379]` + `e2e_precondition` 9 文件前置哈希（错树 exit 2 实测）、`known_gap_m2.txt` 缺口 fixture 双向断言、e2e revision pin；`validate`/`check`/`plan`/`verify` 复验通过 | [验证记录](../fixes/zeros-7181/reports/validation.md)、`results/*-2026-11-08.log`、PR #2 |
| 2026-11-09 `zeros-7181` 评审第二轮（PR #2） | 确认第一轮四项落地后，按新评审修两处 P2 + conj 跟进：`sum_dim_out` 改为一律采用 `out.dtype`（bool out 可用，曾强制 i64 拒绝）、complex→bool 改任一分量非零规则（`sum([0+1j], dtype=bool)`=True，曾取实部得 False）、`abs`/`sum` 入口 `resolve_conj()`（conj bit 物化）、连带修空输入 `ZeroDivisionError` 与 float→bool cast-first 截断分歧（`ne(0)` 非零规则）；测试扩至 210 用例（bool out 矩阵/complex→bool/空输入/conj 直调+dispatch），设备复跑 regression 33/177/210→**210/0/210**、e2e `E2E OK`（161/384、loss rel=0）、repro 1→0、`validate`/`check`/`plan`/`pytest`(26)/`compileall` 通过；patch/`fix.yaml`/快照 hash 一致 | [验证记录](../fixes/zeros-7181/reports/validation.md)、`results/*-2026-11-09.log`、PR #2 |
| 2026-11-10 `zeros-7181` 评审第四轮：随机权重小模型评估（PR #2，意见 6756872311） | 自建 A（复数 sum/abs 反向）/B（conv2d+NHWC 非连续 linear 反向）两模型 × 4 组进程隔离（n/f_target/f_all/b）× 种子 0/1/2（A 另 fp64）：阈值按评审表先冻结、原生噪声门 max_abs=0.0、FD 抽检 PASS、lr 校准 0.1；**f_target 9/9 全 PASS**（fp32 max_abs≤4.8e-7、fp64≤8.9e-16）；**b 组在修复目标路径崩溃**（complex sum_dim KeyError、NHWC linear_backward view RuntimeError）证明修复真实命中；**f_all 两失败均为预存且不在 patch 8 目标文件内**（A：sgn/fill triton complex64 KeyError + fp64 CompilationError；B：`mse_loss_backward` 不广播标量 grad_output 静默错梯度，最小复现 repro 脚本）；scoped 性能 75 用例只报告不作加速声明（complex sum/abs 慢于原生 1.06–21×、zeros/zero_ 0.83–1.00×、58 项 >10% 回归打旗标、崩溃 N/A）；矩阵跑在干净提交 `968d624`、bench 跑在 `d236d5a`；`hardware_status`/`deployable`/`owner` 维持不变 | [小模型评估报告](../fixes/zeros-7181/reports/small-model-evaluation.md)、`results/small-model-2026-11-10/`、PR #2 |
| 2026-11-10 `zeros-7181` 评审第五轮：小模型评估修正（PR #2，意见 6766490204） | 四项修正全部落地：(1) **P1** bench 按族×组独立进程注册（a-sum-abs/b-linear/zeros 各自 include，f_target 期望键断言缺失即败）+ torch.profiler kernel 名非计时命中验证（registered≠called；`zero_` 命中原生 fill 系修复刻意走原生 `copy_(0)`）→ **modelB f_target 重测 fwd 2.42×/fwd+bwd 2.93×/step 2.52×**（四轮 1.07× 系未注册目标算子致误归因，作废），bench 重跑在干净提交 `dd5ebea` → `results/small-model-2026-11-10/bench-r2/`；(2) 比较器结构加固（shape/dtype 严格相等、键集合、step ID 序列 0..n-1、native 参考无效判 fail 非 skip）+ 不依赖设备负例自测 10/10（`tests/test_comparator_negative.py`，提交 `fac401c`）+ 已存 .pt 重评分 20 对+噪声对 **f_target 9/9 维持** → `matrix-rescored/`；(3) `calls_per_s=1000/ms` 公式修正（旧 samples_per_s 多乘 100 撤回）、P95 改名块均值分位、显存 baseline/peak/delta、旗标改候选语义并从数据生成 **115**（旧 58 作废）、f_all/b 正确性失败配置标 diagnostic；(4) 报告三态重生成（目标隔离 PASS / 全量 FAIL / 性能已复验），bench 版本统一；**另立修复项** `reports/follow-up-fix-items.md`（FIX-ITEM-1 mse 标量广播优先、FIX-ITEM-2 complex pointwise/fill/zeros 尺寸分支/to_copy+FP64 编译），预存归因补齐**同路径隔离对照**：mse 39/40 与 complex fill 5/5（含 abs 前向）在修复前对照树逐位复现（触发条件：`torch.complex(p,q)` 组装张量，cast 不触发）；原始 fix 与验收口径维持不变 | [小模型评估报告](../fixes/zeros-7181/reports/small-model-evaluation.md)、`results/small-model-2026-11-10/{matrix-rescored,bench-r2}/`、`reports/follow-up-fix-items.md`、PR #2 |

本地 fix 不进入上节 catalog 表。普通 CI 对本地 fix 脚本只做语法检查（`python -m compileall -q fixes`）、`flag-fixes validate` 的 schema、各声明文件的 hash 和脚本存在性检查与管理工具 pytest。`validate` 不应用补丁，也不证明基线经 patch 后等于修复快照；该关系由针对目标树的 `check`/`plan` 单独验证。三个硬件脚本只在带设备的环境运行，语法检查通过不等于硬件验证通过。

## 后续工作

1. 协调四项来源交付适配，固定新 commit 并更新清单和 hash。
2. 明确负责人、上游基线及目标环境，补齐真实设备验证入口和结果。
3. 完成逐项部署审核后，再验证组合并发布 bundle。

更新源码指针、清单状态或测试基线时，同一个 PR 更新本页和 [来源检查记录](source-audit.md)，保留旧 CI/硬件记录的 commit 关联。未执行的检查继续标为未验证。
