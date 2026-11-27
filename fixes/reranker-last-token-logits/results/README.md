# 结果

本目录保存小型 JSON / 日志，均为 2026-11-26/27 在本机（KunlunXin P800 OAM 96 GiB，
torch 2.9.0+cu129，transformers 4.57.6）实测。没有凭据、机器备份或模型缓存。
`base` = FlagScale `89f342e6` 干净快照，`fixed` = 基线 + 本 patch。环境、hash、逐命令的精确命令行与
证据采集时间见 `../reports/validation.md`；下面每条记录的输入、判定标准与结果以对应 JSON 为准。

## 文件清单

| 文件 | 内容 |
| --- | --- |
| `dispatch-{base,fixed}-2026-11-26.{json,log}` | `tests/test_reranker_lmhead_dispatch.py`（CPU）：基线 6/25 例失败（lm_head 输入 `[rows, seq, hidden]`）exit 1；修复 25/25 exit 0 |
| `fsdp2-cpu-{base,fixed}-2026-11-26.{json,log}` | `tests/test_reranker_fsdp2_step.py --device cpu`（真实 `apply_fsdp2`，gloo 单 rank）：基线 exit 1，修复 exit 0；两棵树 loss 在记录精度（6 位小数）下一致 |
| `fsdp2-cuda-2card-base-2026-11-26.{json,log}` | 同上 `--device cuda --nproc 2`（XCCL）：基线跑完，仅 lm_head 投影判据失败，exit 1 |
| `fsdp2-cuda-2card-fixed-2026-11-26.{json,log}` | 修复树第 3 次尝试：两 rank 全过，exit 0 |
| `fsdp2-cuda-2card-fixed-attempt{1,2}-blocked-2026-11-26.{json,log}` | 修复树前两次尝试：XCCL watchdog（`SeqNum=3`、`SeqNum=22`），exit 2 |
| `repro-{base,fixed}-2026-11-26.{json,log}` | `tests/repro_reranker_logits_alloc.py`（单卡，61 GiB 常驻 + 34 GiB 碎片堆）：基线 `Tried to allocate 11.57 GiB … 20.63 GiB is free` exit 1；修复 exit 0 |
| `equivalence-{base,fixed}-2026-11-26.{json,log}` | `tests/test_reranker_scores_equivalence.py` 的旧版（5 例）：分数与梯度对基线公式 max\|diff\| = 0.0（两棵树 exit 0）。其中标为 `fp32_no_autocast` 的两例实际是 bf16 权重关 autocast，已由下一行的 14 例版本取代 |
| `equivalence-{base,fixed}-2026-11-27.{json,log}` | 同一脚本审阅修订后的版本（14 例，两棵树 exit 0）：bf16 权重（autocast 开/关）5 例 max\|diff\| = 0.0；fp32 权重 2 例分数 2.4e-7、梯度相对差 2.0e-4；7 个 `comparator_*` 自检用例 |
| `fsdp2-cuda-2card-fixed-rerun-2026-11-27.{json,log}` | 审阅修订后经新 launcher 在 2 卡上重跑 `test_reranker_fsdp2_step.py --device cuda --nproc 2`：26.4 s，两 rank 全过，exit 0 |
| `bench-{base,fixed}-2026-11-26.{json,log}` | `tests/bench_reranker_step.py`：batch 16/32 前向/反向耗时与峰值显存 |
| `e2e-1card-{base,fixed}-2026-11-27.{json,log,train.log}` | 真实 trainer + 真实权重 + T2Ranking，单卡 bf16 DDP，batch 16，5 步：逐 step allocated/reserved 与 loss（JSON 的 `loss_log`），两棵树均 exit 0 |
| `e2e-fsdp2-2card-base-b32-longdoc-2026-11-26.{json,log,train.log}` | **平台形态的自然复现**：2 卡 FSDP2、fp32 master、batch 32、真实权重、长文档子集。基线第 3 步反向在两个 rank 同时 OOM（`Tried to allocate 11.57 GiB … 20.77 GiB is free … 71.66 GiB is allocated by PyTorch`） |
| `e2e-fsdp2-2card-fixed-b32-longdoc-attempt1-blocked-2026-11-26.{json,log,train.log}`、`…attempt{2,3}-blocked-2026-11-27.{json,log,train.log}` | 修复树在同一数据上的前 3 次尝试：分别停在第 2、3、1 步（XCCL watchdog / launcher 超时）。attempt 3 的 `train.log` 省略了收尾阶段重复的 4500 余行 TCPStore/HeartbeatMonitor 回溯，并注明 |
| `e2e-fsdp2-2card-fixed-b32-longdoc-plain4-2026-11-27.{json,log,train.log}` | 修复树第 4 次尝试：**step 1–3 全部完成**（含基线 OOM 的那一步），停在 validation 后由 launcher 超时终止（`training_steps_complete=true`、`passed=false`） |
| `e2e-fsdp2-2card-fixed-b32-longdoc-idlegap-2026-11-27.train.log` | 带标注的诊断运行：每次前向前所有 rank 睡 3 s。step 1–2 完成后仍停在第 3 步，launcher 被手动终止，没有 JSON |
| `f-p800-1-sync-ab-2026-11-27.json`、`f-p800-1-timing-probe-2026-11-27.json` | 对平台阻断 F-P800-1 的两项探索（2 rank FSDP2 小模型，非本修复范围）：步骤边界主机同步的交错 A/B；主机时序 4 变体 × 3 轮。只有"所有 rank 前向前睡 3 s"3/3 干净退出 |
| `e2e-fsdp2-2card-{base,fixed}-2026-11-26.{json,log,train.log}` | 常规数据、真实权重、batch 32：基线走完 2 步、修复树走完 3 步 + validation，均被 XCCL watchdog 终止 |
| `e2e-fsdp2-2card-{base,fixed}-randominit-2026-11-26.{json,train.log}`、`e2e-fsdp2-2card-fixed-b16-randominit-2026-11-26.{json,train.log}` | 常规数据、随机权重的早期尝试（基线停在第 2 步、修复停在第 3 步；batch 16 第 1 步遇设备 kernel exception）。这几次没有单独保存 launcher 输出，摘要都在 JSON 里 |
| `allocator-probe-2026-11-27.{json,log}` | `tests/probe_p800_allocator.py`：不等大空洞交还设备后 11.57 GiB 请求在 72.84 GiB 空闲时失败、1.25 GiB 成功；`expandable_segments` 在两种环境变量拼写下都不生效 |
| `verify-2026-11-27.json`、`verify-2026-11-27.stderr.log` | 干净提交 `c2aef972` 上的 `flag-fixes verify`（修复树，6 个声明命令，passed、`source_dirty=false`；stderr 为空）。对应审阅修订前的脚本 |
| `verify-2026-11-27-r2.json`、`verify-2026-11-27-r2.stderr.log` | 审阅修订后的干净提交上的 `flag-fixes verify`（修复树，7 个声明命令，passed、`source_dirty=false`；提交哈希在文件的 `source_commit`） |
| `launcher-selftest-scoped-2026-11-27.{json,log}`、`…-{hang,normal}.log` | `tests/test_reranker_launchers.py`（默认 scoped）：两个共用同一 worker 脚本路径的 torchrun 作业，17/17 exit 0。超时那组的 2 个 rank 与 2 个子进程在 launcher 返回时全部结束；另一组未被信号，60 s 超时时刻活着、91.9 s 正常结束。`hang.log` / `normal.log` 是两个作业的 torchrun 输出（含 worker 的 `[job-worker]` 公告行） |
| `launcher-selftest-legacy-2026-11-27.{json,log}`、`…-{hang,normal}.log` | 同一自测用 `--cleanup legacy`（`c6f73493` 的清理方式，`pkill -KILL -f <worker 脚本>`）：另一组 60.2 s 被杀（返回码 -9），超时那组留下 2 个子进程，3/14 项失败，exit 1——预期结果，用来说明自测能区分两种行为 |
| `launcher-visible-devices-before-c6f73493-2026-11-27.log` | `c6f73493` 的 e2e launcher 在两种环境下的原始输出（以 `###` 分隔行标注环境与命令；文件由 shell 拼接）：默认可见、仅 `XPU_VISIBLE_DEVICES=1,2` 都是 `KeyError: 'CUDA_VISIBLE_DEVICES'`，exit 1，trainer 未启动 |
| `e2e-1card-fixed-{defaultvis,explicitvis,xpuonly}-2026-11-27.{json,log}` | 新 launcher 在三种可见设备环境下启动真实 trainer（修复树，单卡 bf16 batch 16）：默认可见 exit 0（5 步）；显式 `CUDA_VISIBLE_DEVICES=1,2` exit 0（5 步）；仅 `XPU_VISIBLE_DEVICES=1,2` launcher 透传并启动，trainer exit 1（0 步）。JSON 的 `visible_devices` 记录处理前后的两个变量，`job_id` / `timeout_cleanup` 记录作业身份 |
| `e2e-1card-fixed-xpuonly-2026-11-27.train.log` | 仅 XPU 列表那次的 trainer 日志：`import torch` 后第一次 CUDA 调用报 `device >= 0 && device < num_gpus INTERNAL ASSERT FAILED … device=2, num_gpus=2`（厂商栈行为） |

e2e 的 `.log` 是 launcher 输出（JSON 摘要），`.train.log` 是 trainer 的 `train.log`
（只把解释器与脚本的本机前缀改写为相对路径，内容未删减，attempt 3 除外）；被 watchdog
终止的运行没有 per-rank memory 记录，只有日志。

## 判定

| 项目 | 判定 |
| --- | --- |
| 平台形态基线失败的自然复现（2 卡 FSDP2，batch 32，长文档） | ✅ 第 3 步反向 OOM，请求/空闲/已分配与平台记录逐项吻合 |
| 构造碎片堆上的复现与修复对照（单卡） | ✅ 基线 exit 1 → 修复 exit 0 |
| 数值回归（分数与梯度对基线公式） | ✅ fixed exit 0，14/14：bf16 权重 5 例逐位相同；fp32 权重 2 例在 1e-4 / 1e-3 以内（不是逐位相同）；7 个比较器自检用例按预期 |
| dispatch（CPU、FSDP2 CPU、FSDP2 2 卡） | ✅ 基线 exit 1 → 修复 exit 0 |
| 真实 trainer 单卡 bf16 batch 16 | ✅ 两棵树 exit 0；loss 在日志记录精度（6 位小数）下一致；reserved 峰值 92.76 → 79.64 GiB |
| 真实 trainer 2 卡 FSDP2：修复版越过基线 OOM 的那一步（长文档 step 3） | ✅ 第 4 次尝试 step 1–3 完成（越过基线 OOM 点，step 3 峰值 allocated 84.41 GiB）。可比的 loss 只有前两步，在日志记录精度下一致；基线没有 step 3 的记录 |
| launcher 超时清理只作用于本作业（自测，两个共用脚本路径的并发作业） | ✅ scoped 17/17 exit 0；旧清理方式下同一自测 3/14 失败（exit 1，预期） |
| launcher 在默认可见 / 显式选卡 / 仅 XPU 列表下启动 trainer | ✅ 前两种各跑完 5 步；⚠️ 仅 XPU 列表：launcher 不再崩溃并启动了 trainer，但 trainer 在本机厂商栈 import 阶段失败（不在本修复范围，不计为通过） |
| 真实 trainer 2 卡 FSDP2 修复后完整跑通（干净退出） | ❌ 未得到：本机 XCCL 集合通信间歇性卡死（F-P800-1），修复树 7 次 trainer 尝试里 2 次走完 3 个 optimizer step，均未干净退出 |
| `flag-fixes verify`（修复树，7 个声明命令） | ✅ passed，7/7 rc=0（审阅修订后的干净提交，`source_dirty=false`；`verify-2026-11-27-r2.json`） |
| 平台整模型验收 | ❌ not-run，`hardware_status=unverified` |

## 口径说明

- 随机权重的脚本（repro / equivalence / bench / FSDP2 测试）按发布版 `config.json` 在设备上
  初始化，形状与真实模型一致；loss 数值本身无意义，两棵树在日志记录精度下一致才是检验点（loss 只保留 6 位小数）。
- repro 的碎片堆是**构造**的（常驻 61.0 GiB + 9 块不等大碎片），用于稳定复现"大请求 + 碎片"；
  自然复现以 2 卡长文档那次为准。
- bench 与 e2e 单卡的峰值 reserved 受分配器缓存历史影响，同配置重跑会有 ±数 GiB 的浮动；
  allocated 峰值稳定。
