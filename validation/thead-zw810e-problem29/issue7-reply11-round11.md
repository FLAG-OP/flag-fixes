## 第十轮复审（PR #8 评论 mw4Dfg）五点逐项回应 + round-11 提交（commit `c6fc79d`）

### 1. A/B 重复与 native_stock 同矩阵归档

- 复审基于 `1893497`（round-9），**round-10（`3ffc710`，14:34 推送）已含 12 次运行 × 3 独立进程**（orig_ab_reps/，native_stock_r1..r3 在列）；本轮 round-11 在此基础上 **seed 42 固定后全部重跑**（12 runs，exit 0 ×12），并新增 `results/orig_ab/native_stock.json`（同矩阵 native 产物，r1 内容）。每配置 3 独立进程 × 3 计时步；汇总含 p50/min/max（`orig_ab_reps_summary.json`）。
- **seed 修正后的数字（wall p50）**：native/stock 2,141.4 / native/fused 2,124.3（-0.80%）/ sel_stock 2,167.8（+1.23%）/ sel_fused 2,170.9（+1.38%）——方向与复审确认的口径一致（fused 整步中性、selective 回归）。

### 2. probe10/probe11 脚本入册

- `scripts/probe10_gemm_ablation.py`（本轮已在册）与 `scripts/probe11_mask_vs_equal_length.py`（**本轮新增**，含 shape/mask 定义/iters/warmup/backend 元数据与 raw samples）已提交；probe11 重跑归档 raw p50/min/max 与 per-backend 可用性（bool mask 下 flash "No available kernel"，efficient/math 4.5×/9.5×）。

### 3. 正确性字段与文档一致

- `profile_step_segments_ab.py` 已补 **seed 42 + losses/grad-norms/非有限/峰值显存字段**；12 组重跑全部 exit 0，losses 全 finite（0.70–0.75）、grad norms ~10–11、无非有限值。
- **seed 化后跨配置对比有效**（同一 batch 流）；配置间 loss 数值差异（0.70–0.75）为数值路径差异（FlagGems gelu vs native gelu），如实呈现。
- **同配置跨进程 loss 序列可复现**（seed 生效）；README "losses/grad/非有限" 表述与 JSON 字段现已一致。
- 遗留如实记录：**峰值显存捕获为 0**（reset 时机不当，工具缺口）——显存列以运行日志为准，列入工具改进项。

### 4. 结论范围收紧

- GEMM"应用层无改善空间"限定为**该 shape（M2990 K1024 N1024 bf16）/该实现**（reduced-reduction 与布局消融无收益、fp32 4× 慢）；不概括全部形状。
- masked attention **-20% 仍为预测**（等长分桶未实施）；0.5866/0.1313 为局部比例，不外推整步。
- 复审确认的阶段性结论维持：fused AdamW 原对象整步收益不可见（-0.80% 本次批次，噪声内）、selective GELU 回归 +1.23%、masked attention 为候选方向。

### 5. 文档同步

- PR 描述已重写（HEAD `c6fc79d`、round-11 数字、修订索引至 r11）；README 残留单元格（"682.0×4=…见 JSON"）已替换为完整表格；HEAD 引用更新。

### 下一轮顺序（按审阅维持）

AdamW 稳定参考验收（本负载完成）→ FlagScale `build_optimizer` 集成 + 固定种子对照 → CUDA Graph 静态训练段 → GELU 剩余差异独立调查 → F-1/F-2 复验 → 整步/多卡。

### 状态

PR #8 按"阶段性验证成果"待评估合入；issue #7 保持 open。现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变。
