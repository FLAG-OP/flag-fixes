# 结果（2026-11-10，P800/XPU3 本机实测）

| 文件 | 内容 |
| --- | --- |
| `repro-base-2026-11-10.log` | 基线：include 被忽略，**349 个 op** 全量注册，exit 1 |
| `repro-fixed-2026-11-10.log` | 修复后：2 个 op，exit 0 |
| `semantics-base-2026-11-10.log` | （历史，7 例口径前）语义矩阵基线运行：include 忽略相关用例失败，exit 1；最新对照见 `semantics-base-2026-11-11`（本轮 regen 见 `verify-2026-11-11-clean.json` stdout） |
| `semantics-fixed-2026-11-10.log` | 修复后 6 例全对，exit 0 |
| `dispatch-base-2026-11-10.log` | 基线 dispatch（全量注册同样重定向 gelu），exit 0 |
| `dispatch-fixed-2026-11-10.log` | include 模式派发 FlagGems kernel，exit 0 |
| `bench-{base,fixed}-2026-11-10.{json,log}` | gelu pair：base 10.78 ms（未打补丁 pow）/ fixed 1.17 ms；vendor 对照 1.9 ms |
| `step-E-final.{json,log}` | 整链路整步验证（AdamW fused + 集成 include + FlagGems 补丁）：2,291.2 ms/step |

判定：复现 1→0、语义 fixed exit 0、dispatch 机制成立、整步 -14.0%。
平台整模型验收 not-run（hardware_status=unverified）。

| `verify-2026-11-11-clean.json` | 干净树复跑，`source_dirty=false`；JSON `source_commit`=`c3c555b0`（verify 运行时 flag-fixes HEAD，机器可读 provenance），patch/source 主体提交=`4052ef37`，生成时间 2026-11-11。adamw 含 vendor nvidia/thead/kunlunxin、显式 forced=1、混合 dtype 断言；flaggems 含 7 例语义矩阵与 WARNING 断言。 |
