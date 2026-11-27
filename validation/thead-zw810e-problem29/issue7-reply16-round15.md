## 第十二轮复审收尾（PR #8 commit `d67fb0f`）+ 第十三轮修正（a79f605..d67fb0f）

按复审意见四点逐项修复并推送。以下按实际落地内容汇报：

### 1. P1 AB losses_run/grad_norms_run 初始化（已修复 + 已重跑）

- `losses_run = []` / `grad_norms_run = []` 已移到 optimizer step 循环外初始化，跨步累积不再清空。
- seed 42 已恢复（`random.seed(42)` + `torch.manual_seed(42)`），loader seed=42 也保留。
- seed 化后 12 次（3 进程 × 4 配置）全部 exit 0；losses 全 finite。
- JSON 现含 **12 个 loss 值 + 3 个梯度范数**（非末步 4+1），覆盖全部计时步。

### 2. P1 probe11 mask 参数缺失（已修复）

- `timed_in_context` 函数增加 `attn_mask` 参数；masked/unmasked 分支分别传入对应 mask 或 `None`；每个后端调用均记录实际使用的 mask 或 `None`。
- 重采结果：**masked flash 不可用**（"No available kernel"）；**masked efficient/math 可用**（0.248/0.987 ms）；unmasked flash 0.1315 ms。
- mask 的 shape/dtype/有效位置已入 JSON；masked/unmasked 标签与实际调用一致。

### 3. P2 probe10 fp32 预转换 + 源码

- probe10 源码已提交（含 shape 元数据 + raw samples）。
- fp32 转换（`x.float()` / `w.t().float()`）移到计时外预创建。
- fp32 GEMM 比较口径修正为"已预转换的纯 GEMM 计时"，不混入转换成本。

### 4. P2 文档同步

- peak mem **24.3 GiB**（26,092,513,280 bytes）——非"捕获为 0"，JSON 已正确记录；README 列已更新。
- fp16 stock 非有限计数口径：round-12 批次为 **585**（跨快照累计非有限参数计数）；旧批 498/522 分别标注批次。
- "逐位可复现"声明**已撤回**（复审确认 r1/r2/r3 loss 不同）；改为"seed 固定数据流 ✓ + 配置内确定性 ✓"。
- "已完成/通过"声明**已撤回**（AdamW 稳定参考验收未完成）；README/PR 中不再有无证据的"通过"声明。
- README 重复副本（344 行）已删除。
- 容差 floor 统一为**单一来源**（`0.01 * max|ref|`），JSON `tolerances` 字段保存实际使用的公式与系数。

### 5. AdamW 稳定参考验收诚实结果（v2 判定器，同 p0_dt/grads_dt）

| dtype/实现 | 首步 | 中步 | 末步 | 判定 |
|---|---|---|---|---|
| fp32 stock/foreach/fused | PASS | **FAIL** | **FAIL** | **FAIL ×3** |
| bf16 stock/foreach/fused | PASS | **FAIL** | **FAIL** | **FAIL ×3** |
| fp16 stock/foreach | **FAIL**（非有限 498/522） | FAIL | FAIL | **FAIL ×3** |
| fp16 fused | PASS | **FAIL**（worst 1.52e-02） | FAIL | **FAIL** |

- fp32 中/末步 FAIL = 近零位置超 `8*eps32*floor` 门（fp32 累积舍入），非算法错误。
- fp16 stock/foreach 非有限损坏 = fp16 精度不足（真实发现）。
- fp16 fused 有界但仍超 fp16 容差。
- **AdamW 稳定参考验收未完成**——后续需 fp32 master states 或 steps-scaled 容差变体。

### 6. 判定完整性

- `verdict()` 生产函数含缺字段检查（`p/m/v/step` 任一缺失→FAIL with `missing` 列表）；`None` 张量→FAIL。
- 聚合层增加快照数量校验（`len(snaps) != len(SNAP_AT)` → FAIL）+ 步数集合重复检测 + 缺中步/末步→FAIL。
- `--selftest` 经真实 `verdict()` 注入：空快照列表→FAIL；步数重复→FAIL；缺记录→FAIL（**8/8**）。
- `fail_attribution_stats.py` 退出码包含 backward + forward 未解释项（`exit_rule` 字段）。

### 7. 状态

PR #8 按"阶段性验证成果"待评估合入；issue #7 保持 open。现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变。
