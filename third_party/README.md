# 历史修复源仓库

8 个已有仓库以 Git submodule 接入：arithFIX、copyFIX、gatherFIX、geluFIX、liftfreshFIX、mmBF16FIX、onehotFIX、pointwiseFIX。

克隆使用 `git clone --recurse-submodules`；已有工作树执行 `git submodule update --init --recursive`。父仓库提交固定 Git 指针，不会自动跟随远端分支。后续新增 fix 直接放在 fixes/。

更新流程：

1. 在原仓库完成修复并提交，取得需要接入的完整 commit。
2. 在对应 submodule 执行 `git fetch origin`，然后 `git checkout <完整commit>`。
3. 更新对应 catalog 的 source.commit、patch SHA256、目标 hash 和验证证据；源 patch 变化后重新评估基线和兼容性。
4. 执行目录校验、安装预演和适用设备测试，PR 中说明来源变化和验证范围。
5. 一起提交 submodule 指针和清单。不要使用不经审核的批量 `update --remote` 作为发布流程。

保留各来源版权和许可证。本仓库工具的许可不会覆盖三方源码；分发前需检查各来源许可，不明确时不能默认重新授权。若某来源不再维护，可经单独迁移 PR 将其转入 fixes/，保留来源和历史，避免同时维护两份。

当前接入 8 个 submodule、9 项 fix。来源一致性结果及固定 commit 证据见 [源码接入审计](../docs/source-audit.md)。三方来源未发现独立 LICENSE 文件；保留现有源文件版权/许可声明，不自行补写来源仓库许可。
