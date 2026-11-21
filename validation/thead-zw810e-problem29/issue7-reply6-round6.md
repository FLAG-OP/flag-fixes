## 第六轮复审三项收尾已全部完成（PR #8 commit `9736fc5`）

### 1. exit_rule 真正写入 JSON + 批次标识同步

- 修复顺序缺陷：**先构造完整 `out`**（含 backward/forward 未解释数与唯一退出判定 `exit_verdict`）→ `json.dump` → **序列化后回读断言 `saved == verdict`** → 打印 → 退出。本轮归档 `fail_attribution_stats_v3.json` 已含 `exit_rule: {backward_unexplained: 3190, forward_unexplained: 26, exit: 1, rule: "..."}`（回读校验通过）。
- batch 标识更新为本轮语义（`v3.1 (round-6) ... exit_rule persisted in JSON`）。

### 2. 摘要分母与归因措辞（auto_summary 自动生成，README/PR 按其引用）

| 分组 | 分母 | exact | ulp≤2 | 容差 | unexplained |
|---|---:|---:|---:|---:|---:|
| gelu_none backward（9 类） | 4,012,714 | 3,967,306（**98.87%**）| 3,967,306（**98.87%**，与 exact 相同——不存在"ulp≤2 补齐其余"）| 4,012,714（**100%**）| **0** |
| gelu_tanh backward（6 类） | 430,500 | — | — | — | **3,190/430,500** |
| gelu_tanh forward（6 类） | 22,242 | — | — | — | **26/22,242** |

- 总体三级（分母 4,443,214）：exact 94.36% / ulp≤2 96.57% / 容差 99.93%——**仅作总体参考，不与分组口径混用**；backward 与 forward 分别报告。
- 措辞统一为"**有证据范围限制的归因假设**"（README"机制成立"已改）；**TTIR 适用范围注明**：none backward 存档 TTIR 为 bf16/rank-2 编译产物（fp32 类量化结论部分依赖同源源码公式）、tanh 存档为 fp16/bf16 内核——不视作全部 fp32 失败路径的直接编译证据，不外推其他失败路径。

### 3. 不变量断言位置（已修正，现有 JSON 验证通过）

- 断言移至**全部统计赋值之后**（此前在 tol/unexplained 仍为 0 时检查）；新增恒等检查 **`explained_union + unexplained == gate_violations`**。对现有 JSON：全部通过（如 bf16 contig large：503,890+0=503,890）。
- **compare_state → aggregate 全链路 CPU 反例已补**（torch CPU 环境具备）：NaN/Inf 状态张量经真实 `compare_state` 进入 `aggregate()` → `state_nan_inf` 硬失败；selftest 现 **9/9**（含新增全链路）。
- 历史 fp32 状态记录（1,848 条缺 Inf/nonfinite 计数）确认为数据边界——不追溯证明其无 Inf；未来正式验收用新比较器重采完整字段。

### 状态

PR #8 按审阅口径以"阶段性验证成果"待评估合入；issue #7 保持 open（稳定参考验收、FlagScale 集成、CUDA Graph 训练段、GELU 归因训练闭环、F-1/F-2 复验、整步/多卡由其跟踪）。现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变；归因完成 ≠ 修复完成 ≠ 验收通过；性能收益仍为候选观测。
