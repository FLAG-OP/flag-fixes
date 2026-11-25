# 软件 roofline 一页结论（round-57 修订版，复审 7166338004；round-55 版中的分词预取结论已撤回）

**口径**：模型 FLOPs 与**逻辑字节率**推导，**非 HBM 实测**。`hardware_status=unverified`；HBM 字节/带宽利用率/矩阵单元 activity **BLOCKED**。本页全部数字来自 `summary_r57.json`（由 `scripts/build_summary_r57.py` 从各结果文件生成）与 `software_roofline_r49.json`。

## 当前结论（一句话）

**在保持训练语义与精度的前提下，尚无可部署的整步改动**：分词预取在原始入口上**更慢**（12/12，中位 +27.7 ms，候选关闭）；H2D 非阻塞化无显著差异（−1.8 ms，CI [−10.1, +8.1]）；dW Python 路由 −9.0 ms [−17.5, −0.6]（≈0.4%，训练状态逐位相同，**候选**）。

## round-57 对本 PR 此前结论的更正

| 此前说法（round-55/56） | 现状 | 证据 |
|---|---|---|
| 分词预取 −25.2 ms、12/12 更快、候选可采纳 | **撤回**：测量路径伪影（诊断脚本在 `compute()` 内的实验专用 `synchronize`） | `tokenize_overlap_native_r57.json`：原始入口循环上 12/12 更慢，中位 +27.7 ms，95% CI [+21.8, +41.0] |
| 预取"保持训练语义与精度" | **撤回**：r56 的实现在 autocast 外重写 loss（logits 矩阵乘 fp32），与原入口不等价 | `train_state_r57.json`：loss 0.927814 vs 0.917383，386/386 稠密梯度不同（max_abs 0.0088） |
| 前向空隙 ~99% UNKNOWN | **已归因**：原始入口 28 个前向空隙中 25 个由 tokenizer 主导（占空隙中位 96.5%）| `gap_markers_r57.{entry,capture}.json` |
| ≤56 轮的整步计时 | **循环保真度更正**：用的是 r36 采集循环（无 lookahead、无 `isfinite(loss)` 同步），比原始入口快 10–17 ms/步 | `host_stage_timeline_r57.{entry,capture}.json` |

## headroom 按实际 token 维分组（round-55，未变）

| 组 | 行数 | 调用/步 | 设备 ms（单步）| headroom ms | 占 headroom |
|---|---:|---:|---:|---:|---:|
| **中档（1400–4000）** | 60 | 2,880 | 324.79 | **113.91** | **46%** |
| **长档（≥4000）** | 18 | 1,344 | 311.11 | **87.16** | **35%** |
| 短档（<1400）| 22 | 1,544 | 82.83 | 47.65 | 19% |

参考时间差 = `device_ms − FLOPs/139.2e12`，**非端到端上限**；实际可兑现的收益以整步配对为准。优先形状：dW_ffn_up/down (S=512) 各 ~17 ms；dX_attn_proj (1670–3740) 与 dW_ffn_down (3740)。

| 组 | 设备 ms/步 | 有效 TFLOP/s | vs 139.2 | 逻辑字节率 |
|---|---:|---:|---:|---:|
| 反向 GEMM | 772.12（38.0%）| 88–129（6 热点）| 63%–93% | 124–183 GB/s |
| 反向 attention | 375.44（18.5%）| 13.48（阶段口径）| 不可比 | — |
| S=512 dW | 46.80 | 88.1 | 63.3% | **124.75 GB/s** |

## 整步计时在原始入口循环上的结果（round-57）

原始入口循环 = `train_retrieval.py:646-673`（一个 batch 的 lookahead、autocast 内的 loss、`isfinite(loss)` 主机同步、backward、clip+step）；profiler 关；loss 路径内无实验同步。

| 模式（候选 − inline，ms） | 更慢 | 中位 | 均值 95% CI | 范围 | 有效 tokens/s（均值）| 训练状态 |
|---|---:|---:|---|---|---:|---|
| prefetch_faithful（12 对）| **12/12** | **+27.7** | [+21.8, +41.0] | [+9.3, +57.9] | 14,692 → 14,479 | 逐位 PASS |
| h2d_nb：pinned + non_blocking（12 对）| 5/12 | −1.8 | [−10.1, +8.1] | [−30.6, +23.4] | 14,698 | 逐位 PASS |
| dW Python 路由（差中差，28 进程 42+42 配对步）| — | −4.9（阶段差）| **[−17.5, −0.6]**，点估计 **−9.0** | — | — | 逐位 PASS |

进程间噪声下限：h2d_nb 配对差 sd 14.3 ms——小于 ~10 ms 的整步效应在本机需要更多配对才能分辨。

## 原始入口的主机时间去向（无 profiler，12 步均值，`host_stage_timeline_r57.entry_12steps.json`）

| 主机阶段 | ms/步 | 说明 |
|---|---:|---|
| 分词 `prepare_embedding_inputs`×20 | **65.7（3.0%）** | 纯 CPU |
| H2D `move_features` | 252.4 | 分页内存 `.to(device)` = 整条 stream 同步：主机在此**等设备** |
| 前向 launch | 268.2 | 每组 ≈13.4 ms |
| `isfinite(loss)` 同步 | 118.8 | 等前向完成 |
| backward launch | 1425.7 | 主机被 launch 队列反压（设备受限） |

微测试：已入队 119.6 ms kernel 时，分页 H2D 阻塞主机 117.9 ms，`non_blocking=True` 仅 0.02 ms。**分词与前一组前向本来就已重叠**；去掉 20 次/步的 H2D 同步后主机改在 launch 调用里等，整步不变 → **前向阶段是设备受限的**。

## 前向空隙的显式标记（2 预热 + 3 采集步，steps 2–4）

每个 ≥2 ms 的设备空隙：各类别区间先裁剪、再取并集、再做成互斥分割；余下为 UNKNOWN；并报告空隙后的下一 kernel 与其 launch 时刻。

| 循环 | 设备 busy | 前向空隙 ≥2 ms（3 步）| tokenizer | 其它已解释 | **UNKNOWN** | 反向空隙 ≥2 ms |
|---|---:|---:|---:|---:|---:|---|
| 原始入口 | 89.07% | 28 个 / 94.91 ms | 83.73 ms | 10.42 | **0.76 ms（0.8%）** | 10 个 / 32.44 ms，其中 UNKNOWN 32.05 |
| r36 采集循环 | 89.99% | 27 个 / 96.20 ms | 89.46 ms | 4.24 | **2.50 ms（2.6%）** | 无 |

- 原始入口 28 个前向空隙中 25 个由 tokenizer 主导（占空隙的中位 96.5%，最高 98.0%）；其余 3 个为 2 个由 `cpu_op` 覆盖的空隙、1 个纯 H2D 停顿；单空隙 UNKNOWN 最大 0.124 ms。下一 kernel 多为 `cudaMemcpyAsync`，其 launch 在分词结束后 0.05–0.19 ms 发出。
- r36 的 19.24 ms 最大空隙**未复现**（本次最大前向空隙 5.44 ms）。
- 反向窗口的 UNKNOWN：每个 micro 一次 2.0–10.2 ms 停顿，内部无任何 cpu_op / runtime / GC 事件；已排除 Python GC（每步 5 次、0.4 ms；`gc.disable()` 整步不变）；仅出现在带 profiler 的 trace 中，**保留 UNKNOWN**。
- 无 profiler 的设备利用率（`nvidia-smi` 100 ms 采样，259 样本）：中位 96%、p10 94、p90 98；语义由厂商定义，仅报告原始分布。

## 下一步（待协助项不变）

1. 平台方：PCM 或等价计数器窗口 → HBM 读写字节、带宽利用率、矩阵单元活跃度
2. 仓库维护方：6.9 GB `records.pkl`（sha256 `7d00baa3…`）托管
3. 若继续压整步：dW Python 路由需在更低噪声环境（或更多配对）下确认 −9 ms 是否稳定；主机侧重叠类改动已由两项干预排除

## 边界

**可提交**：有效 TFLOP/s、逻辑字节率、分组 headroom、阶段占比、原始入口循环上的整步配对与训练状态对照、带显式标记的空隙分割、无 profiler 的主机阶段计时。
**BLOCKED**：真实 HBM 字节、带宽利用率、矩阵单元 activity。
