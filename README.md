# flag-fixes

FLAG-OP 的修复目录、开发模板和安装管理工具。现有 8 个单算子修复仓库通过 `third_party/` 下固定 commit 的 Git submodule 接入；新的修复直接维护在本仓库的 `fixes/<fix-id>/`。

**[文档与结果导航](docs/index.md)** · **[当前结果汇总](docs/current-status.md)** · [最新 CI](https://github.com/FLAG-OP/flag-fixes/actions/workflows/ci.yml)

## 快速开始

需要 Python 3.10+、Git 和 Linux（安装事务使用文件锁）。

```bash
git clone --recurse-submodules https://github.com/FLAG-OP/flag-fixes.git
cd flag-fixes
python -m pip install -e .
flag-fixes --root . list
flag-fixes --root . validate
flag-fixes --root . show gather-5746
```

已有普通 clone 可执行 `git submodule update --init --recursive`。CLI 从工作树读取清单和修复制品，安装工具后请保留该工作树，跨目录使用时显式传入 `--root /path/to/flag-fixes`。

## 当前交付范围

- 8 个历史修复源仓库的固定 submodule 和 9 项 fix 索引（GELU 来源拆出 Triton 层条目）。
- 5 项本仓库维护的本地修复 [linear-5379](fixes/linear-5379/)、[zeros-7181](fixes/zeros-7181/)、2 项 FlagGems/FlagScale 性能修复 [gelu-kunlunxin-pow](fixes/gelu-kunlunxin-pow/)、[adamw-fused](fixes/adamw-fused/)，以及 1 项接入层修复 [flaggems-selective-enable](fixes/flaggems-selective-enable/)（FlagScale 层，flaggems_include 选择性注册）（flag_fixes fix 索引合计 14 项 = 9 项外部 + 5 项本地）。
- 新修复模板和 `new` 创建入口。
- 清单校验、基线检查、安装预演、显式安装、状态、回滚与验证命令。
- 安装器使用临时目标树进行功能测试。

导入条目初始均为 `deployable: false`、`hardware_status: unverified`。原仓库中的测试报告保留为历史证据；本仓库尚未完成当前平台复验，`apply` 会拒绝这些草稿。清单通过、安装机制测试通过、设备验收通过是不同状态。当前 5 项标记 source-checked，4 项 needs-adaptation：gather/lift_fresh/one_hot 的 patch 结果与源码快照在注释/署名上不同，cann-pow 补丁格式需适配。详见 [源码接入审计](docs/source-audit.md)。source-checked 仅表示指定来源的 patch/快照字节一致，不是设备或版本兼容性声明。平台 profile、经过整组硬件验证的 bundle 与离线发布工具属于后续工作。

## 检查与部署

首版 `--target` 指向 **FlagGems 源码根目录**，目标文件通常位于其 `src/flag_gems/`。不自动寻找或覆盖 site-packages。

```bash
flag-fixes --root . check pointwise-5739 --target /path/to/FlagGems
flag-fixes --root . plan pointwise-5739 --target /path/to/FlagGems
# 仅适用于已完成审核、标为可部署的条目
flag-fixes --root . apply <fix-id> --target /path/to/FlagGems
flag-fixes --root . status --target /path/to/FlagGems
flag-fixes --root . verify <fix-id> --target /path/to/FlagGems --python /path/to/device-env/bin/python
flag-fixes --root . rollback --target /path/to/FlagGems
```

`check` 和 `plan` 是同一精确基线暂存预演，要求目标仍是修复前内容；已安装内容不能用它们作验收。前述三项快照不一致条目的预演会失败。v1 拒绝任何共享目标文件的组合，也没有 bundle 安装入口。

`verify` 只输出 JSON 测试结果，不修改安装状态中的 `verification: not-run`，也不提升 catalog 的 hardware_status；需另存输出并提交验收证据。

`pip install` 只安装工具。实际部署必须显式 `apply`；请停止使用目标环境的进程，修改后以新进程重新加载模块。安装按精确文件 hash 检查基线、保存备份，回滚前检查文件是否被用户再次修改。多文件事务依靠日志恢复，不承诺整个文件系统原子替换。

## 开发新修复

```bash
flag-fixes --root . new my-fix --title '修复问题的简短说明'
```

新目录位于 `fixes/my-fix/`。依次完成登记、基线、复现、诊断、patch、验证、接入和上游跟踪。详见 [接入指南](docs/adding-a-fix.md)、[模板](templates/fix/README.md) 和 [贡献说明](CONTRIBUTING.md)。

## 内容分工

| 目录 | 职责 |
| --- | --- |
| `third_party/` | 现有外部 FIX 仓库，固定 submodule commit |
| `fixes/` | 此后新修复的源码、patch、测试及源描述 |
| `templates/fix/` | 修复步骤和内容规范 |
| `catalog/fixes/` | 外部仓库的接入清单，不重复保存本地 fix.yaml |
| `src/flag_fixes/` | 查询、检查、安装、状态、回滚和验证工具 |
| `profiles/`、`bundles/`、`validation/` | 后续环境描述、组合锁定及验收证据 |

## 更新外部仓库

submodule 不会自动追踪最新代码。修改在原仓库进行，使用 [三方库维护流程](third_party/README.md) 选择新 commit，更新清单和证据，经 PR 检查后提交指针。旧发布仍固定旧版本。

设计与限制见 [设计文档](docs/design.md)、[清单约定](docs/manifest-spec.md)、[安装说明](docs/installation.md) 和 [发布流程](docs/release-process.md)。本仓库工具的许可证见 LICENSE；三方源码沿用各来源许可及版权声明，接入不代表重新授权。
