# 新修复接入

新的 fix 直接归本仓库管理：运行 `flag-fixes --root . new <id> --title '<标题>'` 后编辑 `fixes/<id>/`。不要再创建一个单算子仓库，也不要在 catalog 为本地 fix 手工复制一份清单。扫描器直接发现 `fixes/*/fix.yaml`。

ID 使用小写英文、数字与连字符，保持稳定。按 WORKFLOW.md 完成每一步，把 patch 放在 patches/，按目标源码根记录 targets 相对路径和前后 SHA256。patch 必须能够通过 Git 干净应用；首版安装器不支持新增、删除、重命名、二进制和符号链接目标。

一个 fix 可以修改多个紧密关联文件；不同包、后端或可单独安装的修复应分条。src/ 是可选源码快照，保留目标相对路径，并校验与 patch 结果一致；基线以 commit 追溯，无需复制 .orig。

本地 source.kind=local，source.path=fixes/<id>，source.commit 保持 null：修复来源由管理仓库的发布 commit 固定，不要求清单自引用。所有变更通过 PR。补齐验证前维持 deployable=false。部署资格与硬件验证由平台验收和维护者审核决定。

外部修复继续通过 submodule 引用，见 [三方库约定](../third_party/README.md)。
