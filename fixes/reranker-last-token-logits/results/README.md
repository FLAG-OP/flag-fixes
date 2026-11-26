# 结果

本目录保存小型 JSON / 日志，均为 2026-11-26/27 在本机（KunlunXin P800 OAM 96 GiB，
torch 2.9.0+cu129，transformers 4.57.6）实测。没有凭据、机器备份或模型缓存。
`base` = FlagScale `89f342e6` 干净快照，`fixed` = 基线 + 本 patch。

## 文件清单

| 文件 | 内容 |
| --- | --- |
| `dispatch-{base,fixed}-2026-11-26.{json,log}` | `tests/test_reranker_lmhead_dispatch.py`（CPU）：基线 6/25 例失败（lm_head 输入 `[rows, seq, hidden]`）exit 1；修复 25/25 exit 0 |
| `fsdp2-cpu-{base,fixed}-2026-11-26.{json,log}` | `tests/test_reranker_fsdp2_step.py --device cpu`（真实 `apply_fsdp2`，gloo 单 rank）：基线 exit 1，修复 exit 0；两棵树 loss 逐位相同 |
| `fsdp2-cuda-2card-base-2026-11-26.{json,log}` | 同上 `--device cuda --nproc 2`（XCCL）：基线跑完，仅 lm_head 投影判据失败，exit 1 |
| `fsdp2-cuda-2card-fixed-2026-11-26.{json,log}` | 修复树第 3 次尝试：两 rank 全过，exit 0 |
| `fsdp2-cuda-2card-fixed-attempt{1,2}-blocked-2026-11-26.{json,log}` | 修复树前两次尝试：XCCL watchdog（`SeqNum=3`、`SeqNum=22`），exit 2 |
| `repro-{base,fixed}-2026-11-26.{json,log}` | `tests/repro_reranker_logits_alloc.py`（单卡，61 GiB 常驻 + 34 GiB 碎片堆）：基线 `Tried to allocate 11.57 GiB … 20.63 GiB is free` exit 1；修复 exit 0 |
| `equivalence-{base,fixed}-2026-11-26.{json,log}` | `tests/test_reranker_scores_equivalence.py`：分数与梯度对基线公式 max\|diff\| = 0.0（两棵树 exit 0） |
| `bench-{base,fixed}-2026-11-26.{json,log}` | `tests/bench_reranker_step.py`：batch 16/32 前向/反向耗时与峰值显存 |
| `e2e-1card-{base,fixed}-2026-11-27.{json,log,train.log}` | 真实 trainer + 真实权重 + T2Ranking，单卡 bf16 DDP，batch 16，5 步：逐 step allocated/reserved 与 loss（JSON 的 `loss_log`），两棵树均 exit 0 |
| `e2e-fsdp2-2card-base-b32-longdoc-2026-11-26.{json,log,train.log}` | **平台形态的自然复现**：2 卡 FSDP2、fp32 master、batch 32、真实权重、长文档子集。基线第 3 步反向在两个 rank 同时 OOM（`Tried to allocate 11.57 GiB … 20.77 GiB is free … 71.66 GiB is allocated by PyTorch`） |
| `e2e-fsdp2-2card-fixed-b32-longdoc-attempt1-blocked-2026-11-26.{json,log,train.log}`、`…attempt{2,3}-blocked-2026-11-27.{json,log,train.log}` | 修复树在同一数据上的 3 次尝试：分别停在第 2、3、1 步（XCCL watchdog / launcher 超时）。attempt 3 的 `train.log` 省略了收尾阶段重复的 4500 余行 TCPStore/HeartbeatMonitor 回溯，并注明 |
| `e2e-fsdp2-2card-{base,fixed}-2026-11-26.{json,log,train.log}` | 常规数据、真实权重、batch 32：基线走完 2 步、修复树走完 3 步 + validation，均被 XCCL watchdog 终止 |
| `e2e-fsdp2-2card-{base,fixed}-randominit-2026-11-26.{json,train.log}`、`e2e-fsdp2-2card-fixed-b16-randominit-2026-11-26.{json,train.log}` | 常规数据、随机权重的早期尝试（基线停在第 2 步、修复停在第 3 步；batch 16 第 1 步遇设备 kernel exception）。这几次没有单独保存 launcher 输出，摘要都在 JSON 里 |
| `allocator-probe-2026-11-27.{json,log}` | `tests/probe_p800_allocator.py`：不等大空洞交还设备后 11.57 GiB 请求在 72.84 GiB 空闲时失败、1.25 GiB 成功；`expandable_segments` 在两种环境变量拼写下都不生效 |
| `verify-2026-11-27.json`、`verify-2026-11-27.stderr.log` | 干净提交 `57e2c756` 上的 `flag-fixes verify`（修复树，6 个声明命令，passed、`source_dirty=false`；stderr 为空） |

e2e 的 `.log` 是 launcher 输出（JSON 摘要），`.train.log` 是 trainer 的 `train.log`
（只把解释器与脚本的本机前缀改写为相对路径，内容未删减，attempt 3 除外）；被 watchdog
终止的运行没有 per-rank memory 记录，只有日志。

## 判定

| 项目 | 判定 |
| --- | --- |
| 平台形态基线失败的自然复现（2 卡 FSDP2，batch 32，长文档） | ✅ 第 3 步反向 OOM，请求/空闲/已分配与平台记录逐项吻合 |
| 构造碎片堆上的复现与修复对照（单卡） | ✅ 基线 exit 1 → 修复 exit 0 |
| 数值回归（分数与梯度对基线公式，bf16 / fp32） | ✅ 逐位相同（fixed exit 0） |
| dispatch（CPU、FSDP2 CPU、FSDP2 2 卡） | ✅ 基线 exit 1 → 修复 exit 0 |
| 真实 trainer 单卡 bf16 batch 16 | ✅ 两棵树 exit 0；loss 逐位相同；reserved 峰值 92.76 → 79.64 GiB |
| 真实 trainer 2 卡 FSDP2 修复后完整跑通 | ❌ 未得到：本机 XCCL 集合通信间歇性卡死（F-P800-1），修复树 6 次 trainer 尝试中 1 次走完 3 步 + validation 后卡在收尾 |
| `flag-fixes verify`（修复树，6 个声明命令） | ✅ passed，6/6 rc=0（干净提交，`source_dirty=false`） |
| 平台整模型验收 | ❌ not-run，`hardware_status=unverified` |

## 口径说明

- 随机权重的脚本（repro / equivalence / bench / FSDP2 测试）按发布版 `config.json` 在设备上
  初始化，形状与真实模型一致；loss 数值本身无意义，两棵树逐位相同才是检验点。
- repro 的碎片堆是**构造**的（常驻 61.0 GiB + 9 块不等大碎片），用于稳定复现"大请求 + 碎片"；
  自然复现以 2 卡长文档那次为准。
- bench 与 e2e 单卡的峰值 reserved 受分配器缓存历史影响，同配置重跑会有 ±数 GiB 的浮动；
  allocated 峰值稳定。
