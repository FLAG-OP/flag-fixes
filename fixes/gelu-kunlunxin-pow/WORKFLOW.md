# 修复执行清单

每步填写 done / not-run / not-applicable；后两者解释原因并注明证据，不把未跑测试记为通过。

| 步骤 | 状态 | 内容与完成条件 | 证据/原因 |
| --- | --- | --- | --- |
| 1 登记 | done | ID `gelu-kunlunxin-pow`（算子 + 后端 + 根因）；owner 保持 `null`（无既有负责人，不虚构）；来源为 `problem29/problem.md` 与 FlagScale `docs/retrieval/13-bottleneck-ops.md`（GELU 占 P800 整步 35.2%）；预期行为：GELU 家族与原生 torch 数值一致，前反向不再被 `pow` 拖慢；边界：只改 KunlunXIN 后端、不改变 dtype/dispatch | `fix.yaml`、`README.md`、`reports/diagnosis.md` |
| 2 基线 | done | FlagGems `v5.0.0` = `4e08b44e9b9ad68c2c27b4d25d04c6b76c30ff7a`；目标 `_kunlunxin/ops/gelu.py` sha256 `6c53c738…`、`_kunlunxin/fused/gelu_and_mul.py` sha256 `a3192f8d…`；环境 P800 OAM ×8、XPU-SMI 5.0.21.47、torch 2.9.0+cu129、triton 3.0.0、`GEMS_VENDOR=kunlunxin`；两棵 worktree 可按 `reports/validation.md` 重建 | `reports/validation.md` 环境表、`results/bench-base-2026-11-10.json` |
| 3 复现 | done | `tests/repro_gelu_pow.py`：bf16 (2990,1024)，比较 `flag_gems.gelu_backward` / `gelu(tanh)` 与原生 torch；阈值 2x native。基线 **exit 1**（8.41x / 2.41x），修复后 exit 0（0.51x / 0.42x） | `results/repro-base-2026-11-10.log`、`results/repro-fixed-2026-11-10.log` |
| 4 诊断 | done | 根因：`tl_extra_shim.pow(x, 2.0)` 在 XPU3 降低为 libdevice 调用；同一 kernel 只把平方换成乘法，反向 2.538→0.162 ms，`pow` 还会让部分 kernel 的 `xpu3-elfconv` 失败；方案：`x*x`（必要时缓存 `x_sq`），表达式其余不变；影响层：FlagGems KunlunXIN 算子层，dispatch 与直接调用同一 kernel | `reports/diagnosis.md`、下方“机制对照” |
| 5 实现 | done | 单 patch 改 2 个文件：`patches/gelu-kunlunxin-pow.patch`（sha256 `65ffc3c7…`）；before/after hash 见 `fix.yaml`；`flag-fixes validate` exit 0（11 fixes）、`check`/`plan` exit 0（基线 hash 命中、`git apply --check` 干净、patch 结果与 `src/` 快照一致） | `fix.yaml`、`reports/validation.md` 命令表 |
| 6 验证 | done | 原复现：exit 1 → 0；回归：3 dtype × 4 形状 × 4 布局 × none/tanh 前反向 + fused 前后向，0 failed；dispatch：`F.gelu` / `nn.GELU` / autograd / 非连续 / gated MLP，0 failed；性能：backward 16.4x、tanh 前向 5.7x、fused tanh 5.9x，none 前向逐位不变；精度：误差不增（bwd none 7.8e-3→2.0e-3）；`verify` passed=true（4 个命令 rc=0）；平台整模型 e2e **not-run** | `results/accuracy-*.log`、`results/dispatch-*.log`、`results/bench-*.json`、`results/verify-2026-11-10.json` |
| 7 接入 | not-run | 本地部分完成：schema/hash 校验、`check`/`plan` 预演、`verify` 通过、`apply` 按 `deployable=false` 拒绝、`status` 干净。**未完成**：problem 要求的 x-benchmark 整模型 on 趟（独立 `--result-root`）未执行；owner 审核与 PR 未做 | 原因：本环境没有 x-benchmark runner/模型缓存；`hardware_status` 保持 `unverified`、`deployable=false` |
| 8 上游 | not-run | 未提交 FlagGems 上游 PR，无合入 commit/发行版本；同源 `pow(x,2)` 仍在 `_kunlunxin/fused/geglu.py` 与其他厂商后端，需按“不同后端分条”另修 | 原因：本修复尚未获部署资格，未在上游开 PR |

## 机制对照（步骤 4 证据）

同一输入 bf16 (2990, 1024)、同一 pointwise 配置，只改 kernel 内求平方方式：

| 反向 kernel 内容 | 耗时 |
| --- | ---: |
| 基线：`exp(-pow(s1*x,2)) + 0.5*erf(s1*x) + 0.5` | 2.538 ms |
| 仅 `pow` → `t*t` | 0.162 ms |
| 仅 `exp` | 0.060 ms |
| 仅 `erf` | 0.121 ms |

前向 tanh 路径同样：`pow` 版 0.313 ms → 乘法版 0.053 ms。`geglu` 的 `pow` 版本更是让
`xpu3-elfconv` 编译失败（`CalledProcessError`），换乘法后可编译。

性能修复必须给同口径前后数据和数值精度。dispatch/e2e 按层级选择；编译器修复可使用
编译复现及调用方回归，解释不适用项目。设备不足时可提交草稿。
