# 文档与结果导航

[仓库首页](../README.md) · **[当前结果汇总](current-status.md)** · [最新 CI](https://github.com/FLAG-OP/flag-fixes/actions/workflows/ci.yml)

## 按任务查找

| 要了解的内容 | 页面 |
| --- | --- |
| 当前做完了什么、测试是否通过、哪些修复还不能安装 | [当前结果汇总](current-status.md) |
| 每项修复的源码、patch、来源检查结果及已知差异 | [来源交付检查](source-audit.md) |
| 本仓库本地维护的 fix（源码、patch、测试、验证记录） | [fixes/](../fixes/)（当前 4 项：[linear-5379](../fixes/linear-5379/)、[gelu-kunlunxin-pow](../fixes/gelu-kunlunxin-pow/)、[adamw-fused](../fixes/adamw-fused/)、[flaggems-selective-enable](../fixes/flaggems-selective-enable/)）；昆仑芯 GELU/胶水 kernel 归因与探索结论见 [kunlunxin-gelu-attribution](kunlunxin-gelu-attribution.md) |
| 整体目录、内容归属和分阶段计划 | [仓库设计](design.md) |
| 如何检查、安装、验证和回滚 | [安装说明](installation.md) |
| 如何新增一个由本仓库维护的 fix | [接入指南](adding-a-fix.md)、[fix 模板](../templates/fix/README.md) |
| 一项 fix 必须经过哪些步骤、提交哪些材料 | [八步工作流](../templates/fix/WORKFLOW.md)、[验收报告模板](../templates/fix/reports/validation.md) |
| 清单字段、状态和验证命令如何填写 | [清单约定](manifest-spec.md)、[JSON Schema](../catalog/schemas/fix.schema.json) |
| 如何跟进原单算子仓库后续更新 | [三方仓库维护](../third_party/README.md) |
| 如何贡献、更改由谁维护 | [贡献说明](../CONTRIBUTING.md)、[CODEOWNERS](../.github/CODEOWNERS) |
| 如何发布、组合验证计划是什么 | [发布流程](release-process.md)、[bundle 说明](../bundles/README.md) |
| 历次功能变化 | [变更记录](../CHANGELOG.md) |

## 结果与证据入口

- [逐项 fix 清单](../catalog/fixes/)：来源 commit、patch/快照 hash、状态和历史说明链接。
- [当前结果汇总](current-status.md)：管理工具测试、来源一致性、硬件复验与部署资格分开呈现。
- [GitHub Actions](https://github.com/FLAG-OP/flag-fixes/actions/workflows/ci.yml)：每次提交对应的目录校验、测试和构建日志。
- [硬件与组合验证记录](../validation/README.md)：尚无本仓库设备复验结果；新增结果按规范记录。
- [平台定义](../profiles/README.md)、[组合包](../bundles/README.md)：当前是规划入口，不代表已有可安装发布包。

后续新增文档应加入本导航；新增结果必须关联实际测试的源码 commit、环境、命令及证据。
