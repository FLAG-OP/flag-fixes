# 修复执行清单

| 步骤 | 状态 | 内容与完成条件 | 证据/原因 |
| --- | --- | --- | --- |
| 1 登记 | done | 稳定 ID、owner、issue、预期行为、边界 | `negative-length-regroup`；owner 待分配；PR8 第 58 轮后继任务；只影响文本 embedding 的 hard-negative 分组。 |
| 2 基线 | done | 上游 commit、软件栈、目标文件 hash，可重建环境 | FlagScale `89f342e690053e07dda845f1365089ec00464e09`；`flagscale/train/losses/retrieval.py` before `cd84e26f077dd6f5b292423f7a15a459240ad692bcb923da694b375ffe7bc3e1`。 |
| 3 复现 | done | 最小脚本、命令、原始错误及预期失败 | `tests/test_negative_length_regroup.py` 显示缺省路径调用 4 次 tokenizer、排序路径调用 3 次且 loss 不变。 |
| 4 诊断 | done | 根因、方案、影响层和依赖关系 | PR8 第 58 轮指出当前批次 padding 比例 49.3%，负例按槽位分组放大 padding；方案是在 loss 层做可逆重分组。 |
| 5 实现 | done | 最小 patch、基线及 patch/源码一致性 | patch 只改 `flagscale/train/losses/retrieval.py`；缺省关闭，`attention_mask` 缺失或负例长度不一致时回退。 |
| 6 验证 | not-run | 原复现、边界、回归、实际调用方；必要的性能/精度 | 本地语义回归已通过，但 ZW810E 整步 A/B、显存、梯度/loss 长程对照未跑，因此本步骤整体仍为 not-run。 |
| 7 接入 | not-run | fix.yaml、安装/回滚检查、平台验收与 PR | fix.yaml 已登记为 draft；`apply` 仍应被拒绝，等待设备验收。 |
| 8 上游 | not-run | 贡献 PR、合入 commit、包含修复的发行版本 | 尚未向上游 FlagScale 发起 PR。 |
