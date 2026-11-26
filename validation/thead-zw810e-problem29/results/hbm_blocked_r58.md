# HBM / 矩阵单元计数器：BLOCKED 状态更新（round-58，复审 7182886976 项 4）

## 状态（未变）
**BLOCKED** —— 仍未取得任何 HBM 读写字节、HBM 带宽利用率或矩阵单元活跃度的计数器；`hardware_status=unverified`。本轮能交付的设备侧数据只有 SMI 利用率采样（r57：12 步窗口 259 个样本，`utilization.gpu` 中位 96 / 均值 96.1），它是设备利用率采样，不是上述三项指标的计数器测量。

## 本轮复测证据（可复核）
- 时间：2026-11-26T04:52:41Z（UTC）
- 工具：`/usr/local/PPU_SDK/asight/bin/acu`（Asight，PPU SDK 2.1.0-a5f865）
- 命令：`acu --metrics ppu__time_duration.sum,kvd__bytes_load_pipe_l2.sum,kvd__bytes_store_pipe_l2.sum --launch-count 3 python3 /tmp/tiny_gemm.py`
- 退出码：**101**（直接捕获）；日志：`logs/hbm_probe_r58.log`
- 原始输出（末 6 行）：
```
[Asight]: Target application pid: 3513189, pgrp:3513189
[Asight]: ==WARNING== The application running on the device[0], may affect the metrics results of profiling.
                      Running application list : 
                      	1323324:/usr/local/PPU_SDK/asight/bin/traced_device
[Asight]: Abort current profiling(PID:3513189): Device is not ready for profiling. Please run "dmesg --level=err,crit,alert,emerg" to identify which application is using the device.
[Asight]: All target application created processes are terminated.
```
- `dmesg --level=err,crit,alert,emerg` 对应行：
```
[1319822.898031] [alixpu] PPU0000 PCM is in used by app[pg-exporter]!
[1319822.918557] [alixpu] PPU0000 PCM is in used by app[pg-exporter]!
[1319822.939078] [alixpu] PPU0000 PCM is in used by app[pg-exporter]!
```
- `asys status`（采集环境自检，不占用 PCM）：`Linux perf_event_open syscall available: OK`、`PPU SDK Installed: OK(2.1.0-a5f865)`、`Profiling Environment Check: OK`（`logs/asys_status_r58.log`）——环境本身可用，唯独 PCM 被占。

## 负责方 / 已采取的动作 / 具体依赖
| 事项 | 负责方 | 已采取的动作 | 具体依赖 |
|---|---|---|---|
| PCM 采集窗口 | **ZW810E 平台维护方**（管理 `pg-exporter` 的一方） | r42/r46/r58 三次用同一命令探测，均 exit 101；r46 起随每轮 PR/issue 回复同步请求（`hbm_blocked_r46.md`） | 在 PPU0000 上临时释放或分时调度被 `pg-exporter` 占用的 PCM，给出一个 **≥ 15 分钟**的可用时间段并通知本任务 |
| 或等价计数器 | 平台维护方 / 厂商 | 已确认 `acu`/`asys` 在位；指标名需 PCM 可用后由 `acu --query-metrics` 确认 | 提供不依赖独占 PCM 的 HBM 读写 / 矩阵单元活动采集工具或 API |
| 指标层级确认 | 厂商 | 已把候选指标 `kvd__bytes_load/store_pipe_l2.sum` 写入探测命令 | 确认其为 L2 级还是 HBM 级，并给出 HBM 级与矩阵单元活跃度的指标名、单位 |
| 原始 records 托管 | 仓库维护方 | **本轮已完成**：7.25 GB 文件按 1900 MiB 分 4 卷，上传到草稿 release（见 `records_delivery_r58.json`） | 无（待维护方决定是否公开发布草稿 release） |

本环境没有平台联系渠道（无工单系统、无 IM）；以上 PCM 请求只能由能联系平台方的人转达。

## 窗口可用后的测量计划（就绪，复用 r36 基线窗口）
同一负载与窗口（seed 42，steps 2–4，有效 tokens 27,052 / 31,454 / 35,506）：
1. `acu --query-metrics` 确认 HBM 级读写字节与矩阵单元活跃度的指标名和单位，写入结果 JSON 的 `metric_definitions`
2. 对 3 个训练步采集逐 kernel 指标：按 kernel 名归入反向 GEMM / 反向 attention（分类规则沿用 `kernel_efficiency_r36` 的 phase×kind 划分），报每类的设备时间、调用数、HBM 读字节、写字节、读写字节合计、带宽 = 字节合计 / 该类设备时间、矩阵单元活跃度百分比及其分母定义
3. 采样窗口、原始输出（`acu` 原始文本）、命令、工具版本、workload hash（steps 2–4 的输入文本哈希 `d2bfed5625354170 / c5896ff2bece725f / 6228631cc16cc2dd`）一并提交
4. 判读规则见 `hbm_blocked_r46.md`
