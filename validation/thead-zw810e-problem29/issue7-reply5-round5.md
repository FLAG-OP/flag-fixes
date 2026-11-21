## 第五轮复审四点已全部修正（PR #8 commit `02c866f`，数字由最终 JSON 自动生成）

### 1. tanh 前向模拟公式（已修复并重跑归因）

- `x + 0.044715*x**2` → `x + 0.044715*x**3`；前向公式自测（0/+2/−2/尾部 5 点，含复审反例）内置脚本入口，失败即 exit 2——x=2 现输出 **1.95460**（错误公式为 1.94005，已入自测断言）。
- **重跑前向归因：unexplained 26/22,242**（99.88% 被修正后模拟解释；此前"22,242 全部未解释"系公式错误，已撤回）。

### 2. exact 位级计数与容差实现（已修复，重生成统计）

- **exact 改在目标 dtype 位表示比较**（模拟先舍入到输出 dtype 再比 raw bits；一个原始位置只计一次）——bf16 exact 由 1,007,780（fp32→int16 拆分双计）修正为 **503,890**；新增不变量断言 `0 ≤ count ≤ gate_violations`。
- **TOL_REL 统一**为单一常量 `2×4.768e-07`，判据只使用一次（此前 `2×` 后又乘 `2` 实为 4×）；`explained = exact ∪ ulp≤2 ∪ tol`（并集口径显式），`unexplained` = 补集。
- 重生成后总量（fail_attribution_stats 最终 JSON）：**exact 4,192,640（94.36%）/ ulp≤2 4,290,428（96.57%）/ 容差 4,440,024（99.93%）/ unexplained 3,216（backward 3,190 + forward 26）/ 总门违例 4,443,214**。
- "机制已证实"按要求降级：**gelu_none 负尾量化为"有证据范围限制的归因假设"**（该 9 类 unexplained=0），tanh backward/forward 剩余未解释位置继续独立调查。

### 3. AdamW 状态 NaN/Inf 与仅前向未解释的漏报路径（已修复）

- `compare_state` fp32 分支**记录 nan_a/nan_b/inf_a/inf_b/nonfinite_total**（不再静默过滤非有限位置）；`aggregate()` 将状态 NaN/Inf 列入硬失败。
- `--selftest` 新增**仅状态 NaN / 仅状态 Inf** 两类注入（经真实 `aggregate()`），现 **8/8 通过**；fail_attribution_stats 退出码改为 **backward ∪ forward unexplained 任一非零 → exit 1**（`exit_rule` 字段入 JSON），仅前向未解释不再返回 0。
- v5 聚合语义应用于 v4 设备数据 → **2,702 项硬偏差（与归档一致）**，聚合版本 provenance 已注明；无需为新计数重跑设备。

### 4. 文档/描述/JSON 一致性（自动生成摘要）

全部数字从最终 JSON 提取并写入 README/PR（本轮回复同）：**none 9 类（非 12）、4,012,714 违例、unexplained=0；tanh backward 3,190/430,500；tanh forward 26/22,242；三级匹配 94.36%/96.57%/99.93%；栅格 26/45**。TTIR 证据注明对应内核/dtype/前后向/布局适用范围。旧批次（adamw v1–v4 聚合版、fail_attribution v1/v2）superseded 并注明缺陷。

### 状态

PR #8 按本轮实际完成阶段评估合入；issue #7 保持 open（稳定参考验收、FlagScale 集成、CUDA Graph 训练段、GELU 归因训练闭环、F-1/F-2 复验、整步/多卡由其跟踪）。现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变；性能收益仍为候选观测，不升级为数值验收通过或可部署结论。
