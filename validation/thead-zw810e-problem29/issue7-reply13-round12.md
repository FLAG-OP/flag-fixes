## 第十二轮复审（PR #8 评论 `nXCXKw`）四点已全部修正（PR #8 commit `4cfd840`）

### 1. P1 判定完整性（缺失快照/空列表/重复步/缺字段）

- `verdict()` 生产判定函数开头增加**缺字段检查**（`p/m/v/step` 任一缺失→FAIL with `missing` 列表）；`None` 张量→FAIL。
- 聚合层增加**快照完整性检查**：快照数量 = `len(SNAP_AT)`；步数集合无重复且与 `SNAP_AT` 一致；不满足→FAIL（不走后续判定）。
- `--selftest` 通过真实 `aggregate()` 注入：空快照列表 → FAIL；步数重复 → FAIL。
- 存档 JSON `adamw_stable_ref_v4.json` 27 个快照经 `hard_fail()` 复核：9/9 配置 FAIL 一致。

### 2. P1 容差 floor 三套口径统一

- **唯一来源**：代码中 `floor = 0.01 * max(|ref|)`（即 0.01 × 该参考张量的最大绝对值）；JSON `tolerances` 字段保存实际使用的公式文本；README 引用同一来源。
- fp32 追加近零绝对门 `8*eps32*max(|ref|, 1e-6)`（与低精度公式的 `4*ulp(max(|ref|,floor))` 对齐）；bf16/fp16 使用 `4*ulp_dtype(max(|ref|,floor))`。
- v2/v3 旧 JSON superseded；当前 `adamw_stable_ref_v4.json` 由 round-12 判定器生成，27 个快照经 `hard_fail()` 复核一致。

### 3. P1/P2 原对象正确性 + probe10/11

- **AB 变体改为跨步累积**：`losses_run`/`grad_norms_run` 跨步累积（不再每步重置）；losses/grad-norms finite 检查 + `nonfinite_param_count` + `peak_mem_bytes` 全部入 JSON；**peak_mem 24.3 GiB（非 0）**——README 已删"捕获为 0"过时描述。
- **probe11 后端上下文计时修正**：warmup + 计时均在 `sdpa_kernel(be)` 内执行；结果确认 bool mask 下 flash 不可用 → efficient/math 4.5×/9.5×。
- **probe10 源码补交**：`scripts/probe10_gemm_ablation.py`（含 shape 元数据与 raw samples）已入册。
- **措辞修正**：README "tile…均无改善空间" → "该 shape 下已测开关/布局未发现收益"；GEMM 应用层空间限定"其他形状/实现待验证"。

### 4. 原对象正确性记录范围

- AB 变体 JSON 含：per-step losses（3 计时步 × accum 4 = 12 值）、grad_norms、nonfinite_param_count、peak_mem_bytes——**scope = "末步+全程 loss 轨迹，参数/状态验收待稳定参考完成后"**。
- 四配置间 loss 差异（0.70–0.75）为真实数值路径差异（FlagGems gelu vs native gelu）；**不将配置间 loss 差异归因于 FlagGems GELU 单一因素——native/stock 与 native/fused 也存在差异（fused optimizer 数值路径差异），已分别标注**。
- **fp16 stock 非有限计数口径**：round-12 记录为 **585**（12 JSON 中 fp16 stock 的跨快照累计非有限参数计数），非旧批 498/522——已按批次分别标注。

### 后续工作（由 issue #7 跟踪）

等长分桶训练侧实现、其他 GEMM 热点 shape、CUDA Graph 训练段、F-1/F-2 复验、整步/多卡——均不追加为本 PR 合入前置条件。

### 状态

PR #8 按"阶段性验证成果"待评估合入；issue #7 保持 open。现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变。
