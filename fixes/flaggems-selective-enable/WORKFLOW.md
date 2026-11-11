# 修复执行清单

每步填写 done / not-run / not-applicable；后两者解释原因并注明证据，不把未跑测试记为通过。

| 步骤 | 状态 | 内容与完成条件 | 证据/原因 |
| --- | --- | --- | --- |
| 1 登记 | done | ID `flaggems-selective-enable`（接入层 + KunlunXIN 证据）；owner 保持 `null`；来源为 problem29 整步 A/B 矩阵与 problem29/ATTRIBUTION.md（"other"/全量 enable 的净负收益量化）；预期行为：`flaggems_include` 设置时只注册列出的算子，未设置时与基线一致；边界：只改 enable_flaggems 注册路径，不改 exclude 语义、不改注册时机 | `fix.yaml`、`README.md`、`reports/diagnosis.md` |
| 2 基线 | done | FlagScale 本地 checkout `89f342e690053e07dda845f1365089ec00464e09`；目标 `flagscale/train/utils/flaggems.py` before sha256 `cff9890b…`；环境 P800 OAM、torch 2.9.0+cu129（torch_xmlir）、flag_gems 5.0.0（site 与 FlagGems-fixed 两种）；验证树 `/tmp/opencode/FlagScale-adamw-fixed` = 基线 + adamw-fused + 本修复 | `reports/validation.md` 环境表 |
| 3 复现 | done | `tests/repro_flaggems_full_enable.py`：include=[gelu, gelu_backward] 时统计实际注册算子。基线 **exit 1（349 个 op，含 layer_norm）**，修复后 **exit 0（2 个）** | `results/repro-{base,fixed}-2026-11-10.log` |
| 4 诊断 | done | 根因：enable_flaggems 只有 `enable(unused=...)` 全量路径；昆仑芯 FlagGems 10 算子 9 慢 → 全量注册净负收益；且 `unused` 对 cpp_patched_ops（gelu）不生效，排除也不可用。方案：include 路径走 `only_enable`；影响层：接入层一处分支，未设置 include 时行为不变（回归第 2/3 例验证） | `reports/diagnosis.md` |
| 5 实现 | done | 单 patch 1 文件 26+/3-：`patches/flaggems-selective-enable.patch`（sha256 `bba9c4f3…`）；before/after hash 见 `fix.yaml`；validate/check/plan exit 0 | `fix.yaml`、`reports/validation.md` |
| 6 验证 | done | 复现 exit 1→0；语义回归 7 例（子进程隔离）fixed exit 0、基线 3 例失败即差异（含 include+exclude 优先级用例）；dispatch：include 模式下 gelu 前反向派发 `gelu_none_kernel_rank_1`/`gelu_backward_none_kernel_rank_1`、无厂商 highprecision；bench：gelu pair vendor 1.86→1.17 ms（1.59x，需配合 gelu-kunlunxin-pow）；整步链路（E 变体）2,665.9→2,291.2 ms（-14.0%）；`verify` 4/4 rc=0；平台整模型 e2e **not-run** | `results/*-2026-11-10.*`、`results/step-E-final.json` |
| 7 接入 | not-run | 平台整模型（含 FSDP2）验收未跑：本机无 runner/模型/数据；owner 审核未做。`hardware_status=unverified`、`deployable=false` | 原因：`/flagos-retrieval-*` 不在本机 |
| 8 上游 | not-run | 未提交 FlagScale 上游 PR | 原因：未获部署资格 |

## 与其他修复的关系

- **必须**与 `gelu-kunlunxin-pow`（FlagGems 层）同装：include 只决定注册谁，
  kernel 快慢由 FlagGems 补丁决定。未打补丁时 include=[gelu, gelu_backward]
  会吃到含 libdevice pow 的 `gelu_backward_none`（12.25M 元素反向 ~10.3 ms/次，
  比厂商 kernel 慢 5.6x）—— bench 的 10.8 ms 行就是这么来的。
- 与 `adamw-fused` 独立（不同文件），验证树同时含两者。
- FlagGems 侧遗留问题（未修，已记录）：`unused` 过滤对 cpp_patched_ops
  （gelu 在名单内）不生效，基线/修复一致。

性能修复必须给同口径前后数据和数值精度；接入层修复给行为对照（注册集合 +
实际派发 kernel）。
