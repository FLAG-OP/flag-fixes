## 第十二轮（AdamW 稳定参考判定器重写）完成（PR #8 commit `18c3a48`）

### P1 AdamW 稳定参考验收——v2/v3 脚本缺陷修复后重跑（诚实结果：无一实现通过）

三个缺陷修复（同 p0_dt/grads_dt 克隆、fp64 参考同值提升、唯一判定函数 + verdict=exit 同源 + 负例自测 6/6）后，按声明的每位置容差判据（fp32: 8*eps32*max(|ref|,floor)、bf16/fp16: 4*ulp(max(|ref|,floor))、floor=0.25×参数标准差）重跑 30 步：

| dtype/实现 | 首步 | 中步 | 末步 | 判定 |
|---|---|---|---|---|
| fp32 stock/foreach/**fused** | PASS | **FAIL** | **FAIL** | **FAIL ×3** |
| bf16 stock/foreach/**fused** | PASS | **FAIL** | **FAIL** | **FAIL ×3** |
| fp16 stock/foreach | **FAIL**（非有限 498/522） | FAIL | FAIL | **FAIL ×3** |
| fp16 **fused** | PASS | **FAIL**（worst p err 1.52e-02） | FAIL | **FAIL** |

- **无一实现通过 30 步稳定参考验收**——撤回此前"已完成/通过"表述，README 已同步撤回。
- **fp16 stock/foreach 数值损坏是真实发现**（498/522 非有限位置；Inf 参数；机制=fp16 下 v_hat 与 eps 双下溢 → 0 除）；fused fp16 是唯一有界路径但仍超容差。
- **有效观测（非验收通过）**：三实现相互一致（fp32 逐位；bf16/fp16 差异 ≤ dtype 量化）；fused 不劣于 stock/foreach。
- 后续候选：fp32 master states 变体或声明 steps-scaled 容差变体的数值验收；FlagScale 集成。

### 实施顺序 1–3 的其余项

- **P1-2 正确性字段**（AB 变体）：seed 42 固定（同 batch 流跨配置）；losses/grad-norms/非有限/峰值显存入 JSON；**同配置跨进程 loss 序列可复现**（seed 生效）。原对象 A/B 数字（seed 后）：native/stock 2,141.4 / native/fused 2,124.3（-0.80%）/ sel_stock 2,167.8（+1.23%）/ sel_fused 2,170.9（+1.38%）。
- **P2 probe10/11**：probe10 脚本已提交（本轮已入册）；probe11 脚本已提交 + 重跑（raw samples + per-backend 可用性 + bool mask 下 flash "No available kernel"）。README 残留单元格已修。
- **P2 文档**：PR 描述重写至 `18c3a48`；README 第十二轮节新增（含撤回声明）；批次 superseded 标注。

### 状态

issue #7 保持 open。PR #8 含 round-12 判定器重写（`18c3a48`）——**审阅确认后可按"阶段性验证成果"评估合入**。AdamW 稳定参考验收**未完成**（如实）；fused AdamW 性能收益观测 + fp16 损坏发现维持候选观测；FlagScale 集成、CUDA Graph 训练段、F-1/F-2 复验、整步/多卡由 issue #7 跟踪。现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变。
