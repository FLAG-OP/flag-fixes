# 修复执行清单

每步填写 done / not-run / not-applicable；后两者解释原因并注明证据，不把未跑测试记为通过。

| 步骤 | 状态 | 内容与完成条件 | 证据/原因 |
| --- | --- | --- | --- |
| 1 登记 | done | ID `linear-5379`（沿用 `gelu-5867`/`gather-5746` 的算子-issue 号命名）；owner 保持 `null`（无既有负责人，不虚构）；上游 issue #5379；预期行为：`linear` 对任意布局输入与原生 torch 一致；边界：连续输入结果与性能不变、只改厂商无关通用文件 | `fix.yaml`、`README.md` |
| 2 基线 | done | FlagGems commit `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`git describe` `v5.3.5-46-g29e14822`，2026-09-01，与 problem.md live 复现同日）；目标 `src/flag_gems/ops/linear.py` sha256 `4d46bae0…`，line 158 即肇事行；软件栈 torch 2.10.0 / triton 3.6.0 / Python 3.12.3 / PPU 驱动 1.5.1-1d747a；两个 worktree 可按 `reports/validation.md` 重建 | `reports/validation.md` 环境表 |
| 3 复现 | done | 最小脚本 `tests/repro_linear_view.py`（非连续 (2,16,64) 输入走 `F.linear`）；命令 `PYTHONPATH=$TARGET_BASE/src python3 tests/repro_linear_view.py`（`$TARGET_BASE` 见 `reports/validation.md`，本机为 `/tmp/opencode/FlagGems-base`）；修复前 exit 1，traceback 指向 `ops/linear.py:158 input_flat = input.view(M, K)`，错误文本与 problem.md 完全一致 | `reports/validation.md` traceback 原文、`results/repro-prefix-2026-11-05.log`（未脱敏原文） |
| 4 诊断 | done | 根因：`view` 要求目标形状与 stride 兼容，视觉/注意力激活常为转置/切片布局；方案：`reshape(M, K).contiguous()`（能视图就视图，否则拷贝，`contiguous()` 对连续结果空操作）；影响层：FlagGems 算子层通用文件，dispatch 与直接调用两条路径；依赖：无 `requires`/`conflicts` | `reports/diagnosis.md` |
| 5 实现 | done | 单文件单行 patch `patches/linear-5379.patch`（sha256 `7055d632…`），基线/修复后文件 hash `4d46bae0…` → `ddec3383…`；`flag-fixes validate` exit 0；`check`/`plan` exit 0（`git apply --check` 干净、patch 结果与 `src/flag_gems/ops/linear.py` 快照逐字节一致） | `fix.yaml`、`results/verify-2026-11-04.json` 之外的 CLI 输出已摘录进 `reports/validation.md` |
| 6 验证 | done | 原复现：修复前 exit 1 → 修复后 exit 0（mse 2.67e-11）；边界：3 dtype × 9 布局 × dispatch/direct，修复前 20/50 通过 → 修复后 50/50；回归：连续对照组修复前后 max_diff=0；实际调用方：NHWC MLP、seq-first MLP、随机初始化 SigLIP 视觉塔 3/3（mse ≤ 8.1e-13）；精度：oracle 在注册前由原生 torch 计算、比较在 CPU 上进行；e2e 缺 `transformers` 时退出码 2（`MISSING DEPENDENCY`，skip 不算通过）；性能：not-applicable（崩溃修复，连续路径无额外拷贝，无法给出同口径修复前耗时，`benchmarks/` 不伪造数据） | `reports/validation.md` 结果表、`results/verify-2026-11-04.json`（历史，dirty=true）、`results/verify-2026-11-05.json`（已提交干净源码重跑）、`results/e2e-missing-dep-2026-11-05.log` |
| 7 接入 | not-run | 本地部分已完成：`fix.yaml` 通过 schema 与 hash 校验、`check`/`plan` 预演通过、`verify` `passed: true`、`apply` 按 `deployable=false` 正确拒绝、`status` 干净；PR 已提交（flag-fixes#1），审阅意见的本地修改已落地（缺依赖退出码 2、命令工作目录、导航、CI 语法检查、证据重跑）。**未完成**：平台验收（problem.md 要求的容器 `run_model.py` on 趟 + 独立 `--result-root`，本环境无该容器与运行器）与 owner 审核 | 原因与证据：`reports/validation.md`「未执行」清单；`hardware_status` 保持 `unverified`、`deployable=false` |
| 8 上游 | not-run | 未提交上游 PR，无合入 commit 与发行版本；跟踪 issue https://github.com/flagos-ai/FlagGems/issues/5379（open），同源的 `_enflame/*`、`_iluvatar` linear 覆写与 linear_backward 待按后端分条另修 | 原因：本修复尚未获部署资格，未在上游开 PR |

性能修复必须给同口径前后数据和数值精度。dispatch/e2e 按层级选择；编译器修复可使用编译复现及调用方回归，解释不适用项目。设备不足时可提交草稿。
