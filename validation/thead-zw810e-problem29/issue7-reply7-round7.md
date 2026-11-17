## 第七轮（性能推进意见）回应：两项非阻塞收尾完成，PR #8 已具备"阶段性验证成果"合入条件

### 本阶段合入判断（按复审 6887416894）

- 两项非阻塞收尾完成：
  1. **9/9 自测日志已归档**：`logs/adamw_selftest_v5_9of9.log`（round-5/6 版完整 9 项：含仅状态 NaN/Inf 注入与 compare_state→aggregate 全链路两类；旧 6 项日志标注为历史版本，脚本版本见 provenance adamw_deep v5）。
  2. **PR 开头措辞已改**："GELU 归因完成" → **"完成本阶段 GELU 归因分析"**；tanh backward 3,190、forward 26 个剩余未解释位置在 PR 描述与 README 中保持显式计数，不被掩盖。
- 复审确认三项收尾（exit_rule 构造顺序/摘要重算一致/不变量后置）均已落地且无合入阻塞 → PR #8 可按"阶段性验证成果"合入；性能路线图（§A–E）**不作为合入新增前置**。

### 性能摘要口径（已按审阅写入 PR 描述与 README）

- **基线/候选标识固定**（native/stock、native/fused、selective/stock、selective/fused、full-enable=blocked），负载/shape/dtype、计时层级、批次、精度状态、耗时比与吞吐比明确分母：
  - selective/stock（2.84 ms）→ selective/fused（2.68 ms）：**整步 -5.6%、吞吐约 +6%**——可明确回答的"保留 FlagGems 接入后的整步收益"；
  - -11% 系 **native/stock→native/fused**，不记在 FlagGems 基线；
  - selective/fused 2.68 ms 较 native/stock 2.27 ms 仍慢约 18%——不称 FlagGems 相对原生正收益；
  - full-enable blocked，不以 selective 结果代替其前后对照。
- Graph 12.6× = 0.0731→0.0058 ms/调用组件微基准（同批）；输出验证批次 0.0053 ms/调用单独标注、不拼接；非 kernel 算法优化、非整步收益。

### §2 整步收益小于昆仑芯的讨论（认可并采纳口径）

昆仑芯是 GELU 修复+selective+fused+GEMM 的**组合优化**与 BGE-M3 重建负载；平头哥当前仅 AdamW 单项 + 小模型，且 full-enable 训练路径被 F-1/F-2 阻断——两者不可直接比较为平台上限。占比口径采纳：selective 下 optimizer 段 0.394/2.84≈14%，段内 -0.17 ms 对应约 6 个百分点（与实测 -5.6% 相符）；该 14% 为分段计时的说明性上限估计，非设备效率上限。已测 enqueue 57–69 µs vs native ~5 µs 支持继续调查提交路径/融合/Graph，**不以提交差×调用数当作可回收 wall 时间**（提交与设备执行可能重叠）。

### §A–E 实施方案（已在 PR 描述"下一轮顺序"按序登记，由 issue #7 跟踪）

A 固定代表性负载与对照矩阵（含 eager/replay 配对、full-enable 待 F-1/F-2 复验后加入）→ B wall 关键路径与提交开销证据（无 profiler 整步 wall + 短采样区分 CPU/kernel/设备忙碌/空闲）→ C 收益预算 `1/((1-p)+p/s)` 预测与"预测 ms/实测 ms/差异原因"对照 → D 正确性与计时同批收敛（3 进程 × 100–300 稳态步）→ F-1/F-2 复验、整步/多卡。组合收益实测、不拼接局部倍率；不宣称接近硬件理论上限。

### 状态

PR #8（HEAD d429f3c）按"阶段性验证成果"待合入；issue #7 保持 open 跟踪 §A–E 与 F-1/F-2、真实 BGE-M3/多卡。现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变。
