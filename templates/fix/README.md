# FIX_TITLE

- ID：FIX_ID
- 状态：draft（未具备部署资格）
- 负责人：待填写
- 上游 issue/PR：待填写
- 受影响算子、软件层、后端：待填写
- 问题与修复后行为：待填写

这是新修复模板，使用 `flag-fixes --root . new <id> --title '<标题>'` 生成 fixes/<id>/。按 WORKFLOW.md 完成步骤，编辑 fix.yaml、最小 patch、复现和真实验证报告。模板本身不参与修复扫描，也不包含假装通过的测试。

安装、备份和回滚使用管理仓库统一工具。未完成基线及设备验收时保持 deployable=false；不要在本目录另写覆盖 site-packages 的安装器。
