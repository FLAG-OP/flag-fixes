# 贡献说明

新修复在 `fixes/<fix-id>/` 维护；历史修复先在对应 `third_party` 来源仓库完成变更，再在此提交固定指针更新。不要在 submodule 中留下未提交修改，也不要自动将所有指针升级到远端最新版本。

1. 使用 `flag-fixes --root . new <fix-id> --title '<标题>'` 创建骨架。
2. 完成 WORKFLOW.md，提供最小复现、精确基线、最小 patch、真实验证结果。
3. 填写 fix.yaml；未知项保持 null，未验证维持 draft/deployable=false。
4. 执行 `flag-fixes --root . validate` 及项目测试；有设备时运行对应复现和回归。
5. PR 说明修改原因、目标文件、基线、影响范围、运行过的命令及未验证事项。

fix 负责人负责源码和单项测试，工具维护者负责 schema/安装器，平台负责人负责实际设备和组合验收，发布负责人负责固定制品。小团队可以兼任，但每项修复需要具体 owner。不要填写未存在的团队名或把自动检查描述为人工核验。

审核部署资格时需核对 patch、来源 commit、前后 hash、适用环境、测试入口和证据。普通 CI 不代替 NPU/XPU 验证。上游合入后记录 commit 和包含修复的发行版本，再调整适用范围。
