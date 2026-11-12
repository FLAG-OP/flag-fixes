# 修复执行清单

每步填写 done / not-run / not-applicable；后两者解释原因并注明证据，不把未跑测试记为通过。

| 步骤 | 状态 | 内容与完成条件 | 证据/原因 |
| --- | --- | --- | --- |
| 1 登记 | done | ID `adamw-fused`（优化器实现 + KunlunXIN 证据）；owner 保持 `null`（不虚构）；来源为 `problem29/ATTRIBUTION.md` §二（"other" 104.3 ms 拆解）与 FlagScale `docs/retrieval/14-glue-kernel-origin.md`；预期行为：AdamW 在普通 CUDA 参数上用单支 fused kernel，DTensor/FSDP2 与 CPU 参数维持 stock；边界：只改 `build_optimizer` 的 AdamW 分支，Adafactor 不动，不改变学习率语义 | `fix.yaml`、`README.md`、`reports/diagnosis.md` |
| 2 基线 | done | FlagScale 本地 checkout `89f342e690053e07dda845f1365089ec00464e09`（`docs/retrieval` 之外无未提交代码改动）；目标 `flagscale/train/train_retrieval.py` before sha256 `5ee57573…`；环境 P800 OAM ×8、torch 2.9.0+cu129（torch_xmlir shim）、XPU-SMI 5.0.21.47；两棵 worktree `/tmp/opencode/FlagScale-adamw-{base,fixed}` 可按 `reports/validation.md` 重建 | `reports/validation.md` 环境表、`results/bench-base-2026-11-10.json` |
| 3 复现 | done | `tests/repro_adamw_unfused.py`：336 张量 BGE-M3 形状参数集，比较 `build_optimizer` 选择的实现与 stock AdamW 的单步耗时与 `defaults['fused']`。基线 **exit 1**（fused=False，31.5 ms/步），修复后 **exit 0**（fused=True，11.0 ms，ratio 0.354） | `results/repro-base-2026-11-10.log`、`results/repro-fixed-2026-11-10.log` |
| 4 诊断 | done | 根因：厂商 torch 对 `foreach=True` 也逐参数分解（2688 launches vs stock 3024，device time 几乎不变），只有 `fused=True` 走单支 kernel（672 launches，device 26.9→8.9 ms）；方案：`adamw_impl_kwargs` 自动选择 + env 覆盖 + DTensor/CPU 护栏 + 构造期回退；影响层：FlagScale 优化器构造层，不影响任何 kernel/dtype 语义 | `reports/diagnosis.md`、`results/bench-{base,fixed}-2026-11-10.json` |
| 5 实现 | done | 单 patch 改 1 个文件 51+/1-：`patches/adamw-fused.patch`（sha256 `dbd6e8d8…`）；before/after hash 见 `fix.yaml`；`flag-fixes validate` exit 0、`check`/`plan` exit 0（基线 hash 命中、`git apply --check` 干净、patch 结果与 `src/` 快照一致） | `fix.yaml`、`reports/validation.md` 命令表 |
| 6 验证 | done | 复现：exit 1 → 0；数值回归：5 步同梯度序列 bf16 max\|diff\|=3.1e-5（1 ulp）、fp32 3.0e-8，全部 allclose，exit 1 → 0；选择矩阵 9 例含 DTensor/CPU/env/回退，fixed exit 0；性能：build_optimizer 27.6→11.5 ms/步（2.4×）、launch 2688→672；`verify` passed=true（4 个命令 rc=0，干净树 `25afaebc…`、`source_dirty=false`）；**平台整模型 e2e not-run** | `results/{repro,numerics,selection}-{base,fixed}-2026-11-10.log`、`results/bench-{base,fixed}-2026-11-10.json`、`results/verify-2026-11-10{,-clean}.json` |
| 7 接入 | not-run | 本地部分完成：schema/hash 校验、`check`/`plan` 预演、`verify` 通过（fixed 树）、`apply` 按 `deployable=false` 拒绝、`status` 干净。**未完成**：平台整模型训练（含 FSDP2 reranker）验证未执行；owner 审核未做 | 原因：本环境没有平台训练 runner/数据集（`/flagos-retrieval-*` 不存在）；`hardware_status` 保持 `unverified`、`deployable=false` |
| 8 上游 | not-run | 未提交 FlagScale 上游 PR，无合入 commit/发行版本；FlagScale 本地 checkout 保持未打补丁状态（修复以 patch+worktree 形式交付） | 原因：未获部署资格，未在上游开 PR |

## 机制对照（步骤 4 证据）

同一 336 张量 bf16 参数集、同一梯度流（`results/bench-*.json`）：

| 实现 | step (墙钟) | launches | device 时间 |
| --- | ---: | ---: | ---: |
| stock（`fused=False`） | 35.4 ms | 3,024 | 26.8 ms |
| foreach（`foreach=True`） | 29.3 ms | 2,688 | 25.7 ms |
| **build_optimizer（fused）** | **11.5 ms** | **672** | **8.9 ms** |

关键事实：**foreach 在厂商栈上没有融合收益**（launches 只少 336，device 时间
25.7 vs 26.9 ms）——厂商把 foreach 也分解成逐参数 kernel；只有 `fused=True`
真正合成单支。所以修的不是"忘开 foreach"，而是 foreach 不可用。

性能修复必须给同口径前后数据和数值精度。dispatch/e2e 按层级选择；设备不足时可提交草稿。
