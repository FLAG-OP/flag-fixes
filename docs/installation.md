# 安装与恢复

通过 `git clone --recurse-submodules` 获取管理仓库和固定三方源码，在仓库内运行 `python -m pip install -e .`。CLI 需要可读的管理工作树和可写的目标源码树。

`check`/`plan` 接受 fix ID 列表与 `--target`，目标是包含 `src/flag_gems/` 的 FlagGems 根目录。两个命令执行同一流程：首先检查精确的修复前文件 hash 和 patch，在临时树预演，并检查修复后 hash，不写入目标模块。目标已是修复后内容时会基线不匹配；请用 verify 验收已修改目标。草稿可用于检查，但不允许 apply。当前 gather/lift_fresh/one_hot 的 patch 与快照不一致，因此预演会在结果 hash 检查失败；pointwise-5739 是通过来源一致性检查的示例，但仍未获部署资格。见 [来源审计](source-audit.md)。v1 拒绝所有共享目标文件的修复组合，即使设想有 bundle 也不放行。

`apply` 只允许可部署条目；停止使用目标环境的应用后执行。锁定、复核 hash、记录恢复日志和备份后替换文件。状态目录在目标根的 `.flag-fixes-state/`，不要手工删除其备份。`status` 只读取记录。

`rollback` 按事务逆序恢复。若目标文件已被其他人修改，会拒绝覆盖，需要先协调这些修改。安装后启动新进程；不要依赖运行中已导入模块热更新。

`verify <id> --target <root> --python <device-python> [--timeout <seconds>]` 显式运行清单声明的验证。先核对目标文件为精确修复后内容，再执行声明的命令。未声明命令或缺少设备不算通过。结果只打印 JSON，不回写安装事务的 verification 字段（仍为 not-run），也不修改目录 hardware_status；可将 stdout 保存为验收材料。验证可能需要源仓库额外的 Torch、Triton、模型等依赖，应先阅读该条目说明。

首版不自动发现 site-packages、不安装厂商软件栈、不提供经过硬件验收的组合包；默认导入清单处于不可部署状态。

## 草稿、合并和单项安装的区别

`status: draft` 表示修复条目的验收材料尚未完整，`deployable: false` 表示管理工具拒绝自动安装；
这不是 GitHub 的 Draft PR 状态，也不表示补丁必然无效。条目可以先以草稿状态合入，后续补齐
负责人、兼容性和平台验证，再通过清单变更开放自动安装。PR 合并不会自动授予部署资格。

单项修复可提供手工 patch 验证路径，例如 [linear-5379 的单项安装说明](../fixes/linear-5379/README.md)。
这种方式用于匹配基线的测试工作树，不生成管理工具安装记录，回退时使用该补丁的反向应用，
不能依赖 `flag-fixes rollback`。单项目录不是独立 pip 包；算子运行仍依赖匹配的 FlagGems 和设备软件栈。
