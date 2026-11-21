## 第十二轮复审四点已全部修正（PR #8 commit `6aab451`）

### 1. P1 AdamW 运行崩溃（snaps 未定义）→ 已恢复 run_impl 调用

- **根因**：round-11 完整性检查编辑误删了 `snaps = run_impl(kind, dtype, p0_dt, grads_dt)` 行——`snaps` 未定义即被读取，NameError 导致流程在首个配置崩溃。
- **修复**：恢复 `run_impl` 调用；完整性检查在 `run_impl` 返回后执行（先获取快照，再验证数量/步数集合/重复）。
- **重跑**：exit 0，9/9 impl verdict FAIL 如实入账（`results/adamw_stable_ref_v4.json` 为本轮生成）；v4 旧 JSON superseded。
- **日志已归档**：`logs/adamw_stable_ref_v2_final.log` 含 selftest 6/6 + impl verdicts + 完整性检查输出。

### 2. P1 自测覆盖生产判定 → 已修复

- `verdict()` 函数已增加**缺字段早退检查**（`p/m/v/step` 任一缺失→FAIL with `missing` 列表）和 `None` 张量→FAIL。
- `--selftest` 通过真实 `verdict()` 函数注入：缺 m 字段→FAIL；空快照列表→FAIL。
- 存档 `adamw_stable_ref_v4.json` 27 个快照经 `hard_fail()` 复核：9/9 配置 FAIL 一致。

### 3. P1 A/B 跨步记录 + seed → 已恢复

- seed 42 恢复（round-11 编辑误删）；losses_run/grad_norms_run **跨步累积**（4 micro-batch × accum 4 步内累积）。
- 12 次重跑全部 exit 0；losses 全 finite；四配置间 loss 差异为真实数值路径差异。
- `orig_ab/native_stock.json` 已恢复为同矩阵 native/stock r1 内容（959 字节，JSON 有效）。

### 4. P2 probe11 后端上下文计时

- probe11 脚本（`probe11_sdpa_context.py`）修正：**ALL warmup + timing inside `sdpa_kernel(be)` context**；结果：
  - **unmasked flash**: 0.1315 ms（正常）
  - **masked flash: FAIL**（bool mask 下 "No available kernel"→ 仅 efficient/math 可用）
  - **masked efficient**: 0.2479 ms
  - **masked math**: 0.9867 ms
  - 结论不变：**bool mask 下 flash 不可用 → efficient/math 为唯一路径**；非"各强制后端独立性能"（仅记录强制后端的计时标签）
- probe10 源码已提交（`scripts/probe10_gemm_ablation.py`，含 shape 元数据）；raw samples 未入——listed as tool gap。

### 文档与批次

- peak mem 24.3 GiB 已修正（原"捕获为 0"已删除）；fp16 stock/foreach corruption count 585 保留并标注为本轮批次的跨快照累计。
- README/PR 描述已更新为当前准确口径（无旧批残留）；批次 superseded 标注维持。
- 已有可复核的历史性能数据（微基准 batch-2、accuracy v6、AdamW 稳定参考 FAIL）维持不重跑。
