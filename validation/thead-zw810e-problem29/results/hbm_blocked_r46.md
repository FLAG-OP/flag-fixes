# HBM / 计算单元利用率：BLOCKED 状态与所需协助（round-46，复审 bqR2JZg 项 2）

## 状态
**BLOCKED** —— 未取得任何 HBM 读写字节或矩阵单元活动计数器；**不下算力/带宽/提交空隙任一受限结论**。

## 阻塞证据（可复核）
- 工具在位：`/usr/local/PPU_SDK/asight/bin/{asys,acu}`，Asight v2.1.0_20260416-3bb05b6
- 探测命令：`acu --metrics ppu__time_duration.sum,kvd__bytes_load_pipe_l2.sum,kvd__bytes_store_pipe_l2.sum --launch-count 3 python3 /tmp/tiny_gemm.py`
- **真实退出码 101**（直接捕获，非推断）：`Device is not ready for profiling`
- dmesg：`[alixpu] PPU0000 PCM is in used by app[pg-exporter]!`
- 日志：`logs/hbm_probe_r42.log`（r38 为早期版本，退出码记录有误，以 r42 为准）
- L2 级指标（`kvd__bytes_load/store_pipe_l2`）同受 PCM 锁阻塞；HBM 级与矩阵单元活动指标名需在 PCM 可用后由 `acu --query-metrics` 确认

## 需要谁提供什么
| 需求 | 提供方 | 具体协助 |
|---|---|---|
| PCM 采集窗口 | **ZW810E 平台维护方** | 临时释放或分时调度 `pg-exporter` 占用的 PPU0000 PCM，给出可用时间段 |
| 或等价计数器 | 平台维护方 / 厂商 | 提供不依赖独占 PCM 的 HBM 读写与矩阵单元活动采集工具或 API |
| 确认指标层级 | 厂商 | 确认 `kvd__bytes_*` 为 L2 级还是 HBM 级，并给出 HBM 级指标名与单位 |
| 原始 records 托管 | 仓库维护方 | 将 6.9 GB `acc_hit_step_r44.json.records.pkl` 作为 release artifact，或授予上传权限 |

本环境**无平台联系渠道**（无工单系统、无 IM、无 release 权限）；以上请求已随 PR #8 / issue #7 回复同步。

## 窗口可用后的测量计划（已就绪，复用 r36 基线）
沿用同一训练窗口（seed 42，step 2–4，tokens 27,052/31,454/35,506）：
1. **反向 GEMM**（772.12 ms/步，r36）：按热点形状报设备时间、调用数、HBM 读写字节、带宽 = 读写总字节 / 同窗口设备时间、矩阵单元活动百分比及其分母
2. **attention 前向/反向**（204.7 / 375.4 ms/步）：同上，注明 softmax / dropout / 重算的覆盖范围
3. 用时间线核对关键路径空隙与重叠（区间并集，不用 kernel 时长求和）
4. 判读规则：矩阵单元接近同卡可达上限 → 结合占整步时间估算 MM 调优上限；HBM 接近上限 → 查 trace 中的转置/连续化/中间张量读写；两者都低 → 核对小形状、短 kernel 与调用间空隙
