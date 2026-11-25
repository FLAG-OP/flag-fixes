# 软件 roofline 一页结论（round-53 修正版，复审 BqnWy6w）

**口径**：模型 FLOPs 与**逻辑字节率**推导，**非 HBM 实测**。`hardware_status=unverified`；HBM 字节/带宽利用率/矩阵单元 activity **BLOCKED**。`roofline_v1`，单位 ms。

## 本轮修正

1. **旧字符串清除**：`1.2e5–2.3e5 GB/s` 与"强计算受限"已从 JSON/README 删除；统一为逻辑字节率（S=512 dW 124.75 GB/s），计算受限改为**待验证假设**。
2. **窗口口径**：100 行逐形状数据来自**单个训练步**（step 2）；772.12 ms 是**三步均值**。718.73/772.12 是**不同窗口的比值**，不写成同窗口覆盖率。
3. **headroom 按真实序列维度分组**（见下）；逐形状 `device_ms − FLOPs/peak` 是 139.2 参考假设下的**理想设备时间差，不是端到端上限**。
4. **gap attribution 修正**（`gap_attribution_r53.json`，`gap_v2`）：只用 `cat==user_annotation` CPU marker（GPU marker 分开保存）；旧版因同名覆盖把 backward 窗口写成 0 ms（真实值如 `rf_s2_backward_m0` = 317.08 ms），**已 superseded**。

## 四组核心结果

| 组 | 调用/步 | 设备 ms/步 | 有效 TFLOP/s | vs 139.2 | 逻辑字节率 |
|---|---:|---:|---:|---:|---:|
| 反向 GEMM | 5,768 | **772.12**（38.0%）| 88–129（6 热点）| 63%–93% | 124–183 GB/s |
| 反向 attention | 1,440 | **375.44**（18.5%）| 13.48¹ | 不可比 | — |
| S=512 dW | 96 | 46.80 | 88.1 | 63.3% | **124.75 GB/s** |
| 中序列 dW | 24 | 10.32 | 86.4 | 62.1% | 125.0 GB/s |

¹ 模型矩阵乘 FLOPs ÷ 整个阶段设备时间（含 softmax/dropout/mask），不可比 GEMM 峰值（定义保留）。

## headroom 按序列维度分组（第 1 项修正）

| 组 | 行数 | 调用/步 | 设备 ms | headroom ms | 占 headroom |
|---|---:|---:|---:|---:|---:|
| **长序列（token ≥ 4000）** | 46 | 2,400 | 470.85 | **145.14** | **58%** |
| 中序列（1400–4000）| 40 | 2,400 | 215.76 | 75.18 | 30% |
| 短序列（< 1400）| 14 | 968 | 32.12 | 28.42 | 11% |

**修正**：headroom **主要在长序列**（58%），而非上轮所说的"77% 在中短形状"。短序列 headroom **比例**高（28.42/32.12 = 88%，如 `[160,1024,1024]` 理想 0.46 / 实际 6.72 ms），但**绝对量仅 11%**。6 个热点行仅 57.19 ms（23%）。

## 空隙归因（第 3 项修正，`gap_attribution_r53.json`）

CPU 窗口修正后：**backward 窗口几乎无大空隙**（最大 0.08–2.52 ms）；空隙集中在**forward 窗口**（29–55 ms/微批次）。

| gap | host 事件覆盖 | tokenize 覆盖 | **UNKNOWN** |
|---|---:|---:|---:|
| **19.24 ms**（s2 最大）| 1.52 | **0.0** | **17.72** |
| 典型 3–5 ms | ~0.2 | **0.0** | **~95%** |

**关键修正**：上轮"host tokenize 阻塞设备"的归因**不成立**——
- trace 中**无可识别的 tokenize cpu_op** 相交空隙（tokenize 覆盖 0.0）
- 空隙内**无 H2D memcpy 相交**（上轮的"next HtoD"是空隙**之后**的事件）
- host 事件（cpu_op ∪ runtime ∪ sync）只覆盖 0.2–1.5 ms，**其余为 UNKNOWN**（可能为 Python host 代码/GIL/不发射 aten op 的 tokenizer，但 trace 无证据）
- **不得**以"有任意 CPU 事件"断言整个 gap host-bound

## 训练决策：分词重叠假设（第 2 项，重做）

**上轮实验（round-52）作废**：`sorted()` 把 `negative_1` 当 positive、`zero_grad` 每 micro 调用、两模式 host 顺序相同（无真实重叠）。

**重做**（`tokenize_overlap.py`，原训练入口；显式 `anchor`/`positive`/`negative_1..3`；每步一次 `zero_grad`、4 micro 累积；loss 与 `embedding_loss` 一致）：
- **语义验证**：inline loss0 = **0.95/0.47/0.69** ✓（真实训练值；作废版本为 2.86/1568）；输入哈希与 token 数（27,052/31,454/35,506）两模式一致
- **执行顺序确实不同**：inline 实测重叠 **0.0 ms**；overlap **34–39 ms**（tokenize 被设备计算隐藏，时间戳可查）
- **整步配对差**（3 次重复、组序轮换、profiler 关）：**+10.8/+1.5/−3.4 | −10.6/−7.1/−31.0 | +1.6/+9.1/−11.6 → 中位 −3.4 ms，5/9 为负**——持平
- **正确性缺陷（新发现）**：overlap 模式在 rep1（1 个）与 rep3（3 个）出现 **loss 异常值 2.5–3.2**（正常带 0.43–0.95），inline 三次均无 → **tokenizer/prepare_embedding_inputs 非线程安全**，并发分词会破坏输入

**判定**：假设在性能上**不成立**（有 34–39 ms 真实重叠却无整步收益 → 被隐藏的 tokenize 大部分不在关键路径），且实现在线程下**不正确**。**不采纳**。安全版本需基于进程的预取（dataloader workers），但鉴于持平结果预期无收益。

## 下一步（按 headroom）

1. **长序列 GEMM**（145 ms headroom，58%）：厂商侧 kernel 调优；转置视图 dW 专用路径
2. **中序列**（75 ms）
3. 短序列固定开销（28 ms，比例最高但绝对量小）
4. **forward 窗口空隙**（29–55 ms/微批次）：**~95% UNKNOWN**，需 host 侧显式埋点（tokenizer/prepare）才能归因——下一步可做

## 边界

**可提交**：有效 TFLOP/s、逻辑字节率、算术强度、分组 headroom、阶段占比、关键路径空隙及其 covered/UNKNOWN 拆分。
**BLOCKED**：真实 HBM 字节、带宽利用率、矩阵单元 activity。
**所需协助**：平台 PCM 窗口；6.9 GB records 托管（sha256 `7d00baa3…`）；72 项目标集仅对应 BGE-M3 24 层 × 3 类，换配置需重设。
