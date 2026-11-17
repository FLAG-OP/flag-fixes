## issue #7 第一轮完成通报（PR #8）

A–E 各项已执行，证据 PR：**https://github.com/FLAG-OP/flag-fixes/pull/8**（分支 `issue7-zw810e-opt-round1`，commit `5fb8af6`，21 文件）。结果表已在 PR 描述中给出，要点：

### B. fused AdamW（完成 torch.optim 层全部清单）

- 3 配置（single/multi/bge336）× 4 dtype（fp32/bf16/fp16/mixed）× 3 实现 × 100 步受控更新，逐位置状态对比（param/exp_avg/exp_avg_sq/step）+ checkpoint 恢复。
- **foreach ≡ stock 逐位一致且无收益**——排除。
- **fused 正确性**：fp64 闭式归因确认与 stock 同公式（≤1 fp32 ulp）；100 步发散有界（fp32 ≤2.7e-07 abs）；无 NaN/Inf；**checkpoint 恢复（50+50 步）参数与状态逐位一致**。
- **收益（3 独立进程口径的中位，新增第二次独立复现）**：bge336 fp32 **2.8×**（29.67→10.54 ms）、bf16 1.84×、mixed 2.9×；与 probe6 的 2.3× 同向。
- **新发现（fp16）**：stock/foreach fp16 AdamW 在本栈**数值损坏**（Inf/NaN——fp16 下 `v_hat=g²` 与 `eps=1e-8` 双下溢 → 0 除；single 45,378 位置、bge336 25,588,726 位置）；**fused 内部 fp32 计算、0 污染，是唯一可靠的 fp16 路径**。
- FlagScale `build_optimizer` 集成路径验证 **pending**（本机无 FlagScale 环境）；白名单维持不变。

### C. 提交开销（定向实验完成，候选 1 成立）

- **CUDA graph 捕获/重放在 ZW810E 可用**：gelu fp32 图重放 vs 单次 event：1.2M **0.0058 vs 0.0731 ms（12.6×，提交成本完全消除）**、12.25M 1.67×、98M 持平（= 纯设备执行时间，同时证实提交开销假设）。
- aten 注册入口 vs FlagGems 直调入口：enqueue 58.7 vs 55.5 µs（注册层 ~3 µs）；raw cached-kernel 直调入口未在生产 wrapper 上找到稳定 introspection 入口（如实记录）。
- thead 无独立 gelu 覆写（回退共享 `ops/gelu.py`）；pow 出现 7 处但 none-path 主项为 erf/exp——**非昆仑芯 pow 热点同型，不套用其根因**。
- 组件级拆分（python/libentry/launch）未完成——维持"方向假设"口径。

### D. 15 hard FAIL 归因 + F-1/F-2

- **归因完成**：违例位置 84% 在 |x|>3.9 负尾（p50 |x|=7.1）；FlagGems fp32 gelu_backward 在 **|x|≳8 输出精确 0.0**（native/真值 ~1e-14–1e-36），5.5≤|x|≤8 幅度偏差至 1.6×，|x|<5 正常；绝对误差 ≤1.14e-07。fp64 双公式（gems 常数版 vs aten 顺序版）模拟确认**数学等价——负尾有限精度伪影（erf 抵消 + pdf 项损失），非算法错误**；该区梯度本身 ≤1e-7，训练影响可忽略。证据含逐位置 fp64 复算。
- **F-1/F-2 复验（11-13 第二次单次触发）**：仍挂起，控制组正常——full-enable training 保持 blocked；设备因果结论维持"状态相关、待平台恢复"，未用 v1-CPU 数据作任何对照。

### E. 小模型 × fused optimizer

四配置（native/selective × stock/fused，可训练 embedding、全部显式上设备、8 步）全部可运行、梯度 finite：optimizer 段 fused 1.76–1.9×；整步 -5.6~-11%；loss 相对差 ≤4.7e-07。

### pending（不计通过）

FlagScale 集成路径、CUDA-graph 选择性接入方式与组件拆分、真实 BGE-M3/整模型、多卡/FSDP2、vendor 白名单扩充——均已在 PR 描述与 README"覆盖与 pending"列明归属。现有条目 `draft / deployable=false / hardware_status=unverified` 维持不变；不以本表承诺昆仑芯 -15.3% 同等收益。
