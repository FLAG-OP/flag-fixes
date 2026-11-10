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

上表只覆盖 9 项外部来源修复；本仓库另有 2 项本地维护修复（[linear-5379](../fixes/linear-5379/)、[gelu-kunlunxin-pow](../fixes/gelu-kunlunxin-pow/)），见下节，合计 11 项 fix 索引。

所有条目的 `deployable=false`、`hardware_status=unverified`。gather、lift_fresh、one_hot 的来源差异是 patch 与完整源码的注释/署名不一致；CANN pow 条目需要补丁格式和目标基线适配。

## 本次变更记录（不改写上节基线）

以下条目晚于上节 `2275e66` 基线，基线表中的历史数字保持原样；目录校验与管理工具测试的通过状态以对应 CI 运行为准。

| 变更 | 内容 | 证据 |
| --- | --- | --- |
| 2026-11-05 新增本地 fix `linear-5379` | 本仓库第 1 项本地修复（fix 索引合计 10 项 = 9 项外部 + 1 项本地）：FlagGems `linear` 对非连续输入调用 `view` 崩溃，改为 `reshape(...).contiguous()`；`deployable=false`、`hardware_status=unverified`，`apply` 会拒绝 | [fix 目录](../fixes/linear-5379/)、[诊断](../fixes/linear-5379/reports/diagnosis.md)、[验证记录](../fixes/linear-5379/reports/validation.md)、[verify 结果](../fixes/linear-5379/results/)、PR #1 |
| 2026-11-10 新增本地 fix `gelu-kunlunxin-pow` | 本仓库第 2 项本地修复（fix 索引合计 11 项 = 9 项外部 + 2 项本地）：KunlunXIN GELU 家族把 `pow(x, 2.0)` 换成乘法，P800/XPU3 上 tanh 前向 5.8x、none 反向 16.5x、fused tanh 5.9x，精度不降；`deployable=false`、`hardware_status=unverified`（本机微基准通过，平台整模型验收未做），`apply` 会拒绝 | [fix 目录](../fixes/gelu-kunlunxin-pow/)、[诊断](../fixes/gelu-kunlunxin-pow/reports/diagnosis.md)、[验证记录](../fixes/gelu-kunlunxin-pow/reports/validation.md)、[性能数据](../fixes/gelu-kunlunxin-pow/benchmarks/README.md)、[verify 结果](../fixes/gelu-kunlunxin-pow/results/) |

本地 fix 不进入上节 catalog 表。普通 CI 对本地 fix 脚本只做语法检查（`python -m compileall -q fixes`）、`flag-fixes validate` 的 schema、各声明文件的 hash 和脚本存在性检查与管理工具 pytest。`validate` 不应用补丁，也不证明基线经 patch 后等于修复快照；该关系由针对目标树的 `check`/`plan` 单独验证。三个硬件脚本只在带设备的环境运行，语法检查通过不等于硬件验证通过。

## 后续工作

1. 协调四项来源交付适配，固定新 commit 并更新清单和 hash。
2. 明确负责人、上游基线及目标环境，补齐真实设备验证入口和结果。
3. 完成逐项部署审核后，再验证组合并发布 bundle。

更新源码指针、清单状态或测试基线时，同一个 PR 更新本页和 [来源检查记录](source-audit.md)，保留旧 CI/硬件记录的 commit 关联。未执行的检查继续标为未验证。
