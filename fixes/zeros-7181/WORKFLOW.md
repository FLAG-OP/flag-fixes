# 修复执行清单

每步填写 done / not-run / not-applicable；后两者解释原因并注明证据，不把未跑测试记为通过。

| 步骤 | 状态 | 内容与完成条件 | 证据/原因 |
| --- | --- | --- | --- |
| 1 登记 | done | ID `zeros-7181`（沿用 算子-issue 号命名）；owner 保持 `null`（无既有负责人，不虚构）；上游 issue #7181；预期行为：complex `zeros` 保 dtype 清零、`abs` 返回模长或前置拒绝、m2-bert 反向不再 `KeyError` 中断；边界：实数路径逐位不变、8 文件（`ops/linear.py` 归 linear-5379） | `fix.yaml`、`README.md` |
| 2 基线 | done | FlagGems commit `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`git describe` `v5.3.5-46-g29e14822`）；8 个目标文件 before sha256 记录于 `fix.yaml`/README；软件栈 torch 2.10.0 / triton 3.6.0 / Python 3.12.3 / transformers 4.57.0 / PPU-ZW810E thead；四个工作树可按 `reports/validation.md` 的构造命令重建（重建后 `diff -rq` 逐字节一致） | `reports/validation.md` 环境表、目标树构造 |
| 3 复现 | done | `tests/repro_complex64.py`（issue #7181 直接调用形态 + torch 入口）；修复前 exit 1 `ISSUE REPRODUCED` + `KeyError: 'complex64'`，错误自 `triton/_utils.py type_canonicalisation_dict`；真实模型反向（阶段二树）复现 problem.md 同款 `sum.py:385` 堆栈 | `results/repro-baseline-2026-11-08.log`、`results/e2e-problem-path-noncomplex-2026-11-07.log`（原文未脱敏） |
| 4 诊断 | done | 根因：triton 无 complex mangle 条目，`sum/zeros 族/abs` 把 complex 送进实数 kernel；同链路两个预存阻塞：conv2d `input_c_stride[None]` constexpr 崩溃（标量 1 特化）、thead linear_backward `view` 非连续崩溃；方案与影响见 diagnosis（三类改动 A/B/C 及各自验收）；预存梯度缺口（0 个 `register_autograd`）双树同形态，不属于本 patch | `reports/diagnosis.md`、`results/gradgap-probe-*.log` |
| 5 实现 | done | 8 文件 patch `patches/zeros-7181.patch`（sha256 `60acb22f1800f2f2c9b2aacf3f0bcd254465e40b2e1babcc3cc48e1a2387a6ea`，494 行 / +267 −46；第二轮评审后重建），before/after hash 逐文件记录；`flag-fixes validate` exit 0；`check`/`plan` exit 0（须同时给 `linear-5379 zeros-7181` 两个 ID；`git apply --check` 干净、patch 结果与 `src/` 快照逐字节一致）；漏依赖 ID 时 exit 2 实测 | `fix.yaml`、`reports/validation.md` 管理工具表、`results/cli-*-2026-11-09.log` |
| 6 验证 | done | 复现：exit 1 → exit 0；回归（210 用例）：33/177/210 → **210/0/210**（含 strided zero_、模长量级、交叉 dtype out 矩阵、autograd parity、第二轮新增 bool out 采用 / complex→bool any-component / 空输入归约 / conj abs+sum；第一轮 139 口径 18/121/139 → 139/0/139 见 `*-2026-11-08.log`；旧 11-07 日志 TOTAL 行计数器缺陷注明、实际 16/38/54）；e2e：阶段一 view 阶段二 complex KeyError → `E2E OK`（前置 9 文件哈希 OK、revision pin、fixture 双向断言 384、loss rel=0、161 梯度存在且有限）；错树 e2e 前置哈希 exit 2 实测；缺依赖/缺设备全部 exit 2（skip 不算通过）；梯度缺口双树对照一致（预存）；`verify` `passed: true` 3/3；性能 not-applicable（崩溃修复无同口径修复前耗时，不伪造） | `reports/validation.md` 结果表、`results/*-2026-11-09.log`、`results/verify-2026-11-09.json`、`results/*-2026-11-0{7,8}*.json` |
| 7 接入 | not-run | 本地部分已完成：`validate`/`check`/`plan` 通过（含 requires 强制与前置哈希）、`verify` passed、`apply` 按 `deployable=false` 正确拒绝 exit 2、`status` 干净。**未完成**：平台验收（problem.md 容器 `run_model.py` on 趟 + 独立 `--result-root`，本环境无该容器）与 owner 审核 | 原因与证据：`reports/validation.md`「未执行」清单；`hardware_status` 保持 `unverified`、`deployable=false` |
| 8 上游 | not-run | 未提交上游 PR，无合入 commit；跟踪 issue https://github.com/flagos-ai/FlagGems/issues/7181（open）；`mean/full/ones/eq` 等其余 complex 缺口与非 thead 后端同源 view 待按范围另修 | 原因：本修复尚未获部署资格，未在上游开 PR |

性能修复必须给同口径前后数据和数值精度。dispatch/e2e 按层级选择；编译器修复可使用编译复现及调用方回归，解释不适用项。设备不足时可提交草稿。
