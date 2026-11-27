## 第三轮复审四点已全部修正（PR #8 commit `659b257`）

按复审意见 6876765307 的"下一轮最小实施顺序"第 1–2 步逐项对照（第 3–5 步为后续任务，见文末）：

### 1. P1 AdamW 状态比较被跳过 → 已修复并重跑

- **根因确认**：v2 用 `id(p)` 保存状态、用 `id(p.detach().clone())` 查询——克隆张量 id 不同 → `state_compare` 静默为空。归档 `adamw_deep.json`（v2）**superseded**（README 已标注；补充 provenance：其 summary 由 v2 脚本生成，与 v3 不可混用）。
- **v3 修复**（`adamw_deep_v3.json`，12 组合重跑，exit 1 如实）：状态按**参数稳定索引**对齐；shape/dtype 校验；缺状态/空对比/数量不符 = **硬失败**（不再静默 `{}`）；mixed 逐张量实际 dtype 对比；错误 schema 统一（非有限 step1 分支含全字段，聚合不再 KeyError）；**checkpoint 失败进入非零退出码**；`--selftest` 入口级负例 3/3（非有限 step1 schema、缺状态、checkpoint 参数不一致）。
- **真实状态对比结果（5,520 项）**：foreach 状态与 stock **逐位一致**；fused 的 `exp_avg`/`exp_avg_sq` 相对 stock 存在偏离（部分超声明容差——求和顺序 + 低精度内部 fp32 计算），逐项入账并归入"fused 不承诺与 stock 长期逐位一致"层次；fp16 stock/foreach 的 nan/inf 证据保留（45K–2,559 万位置）。
- 注入负例：聚合函数对非有限 step1/缺状态/checkpoint 不一致三类均可检出并计入 hard（自测 exit 0 证明不崩、有检出）。

### 2. P1 GELU 归因参考公式错误 → 撤回并重建

- **撤回声明**：v1 的 `formula_aten_fp64` 误用前向项 `0.5·x·(1+erf)`（正确导数首项为 `Φ(x)`），该错误公式不能支撑"数学等价、非算法错误、归因闭环"的结论——**v1 归因结论与产物 superseded**（`fail_attribution_gelu_bwd.json` 标注）。
- **v2 重建**（`fail_attribution_v2.py` / `fail_attribution_v2.json`）：
  - 正确导数 `Φ(x)+x·φ(x)`；负尾参考改 **erfc 稳定 fp64 形式**；设备无关解析检查（x=0→0.5、x=2→1.0852318、x=−2→−0.0852318）通过；
  - violation 掩码改用**生产 comparator**（与 v6 基线一致：如 bf16 contig large 复现 503,890 位置）；
  - **覆盖全部 15 个 FAIL 类别**（gelu_none fp32/bf16、gelu_tanh fp32，typical/large × 三布局）——不再以单类概括；
  - **根因从"推测"升级为"已证实机制"**：编译后 TTIR（`gelu_backward_none_kernel_rank_2.ttir`，已归档引用）显示求和顺序为 `pdf + 0.5·erf` → `+0.5`——负尾 pdf 项在 **0.5 量级栅格（fp32 ulp=2^-24≈5.96e-08）**上被量化或舍入为精确 0；TTIR 顺序 fp32 模拟在全部 15 类复现设备输出 **39,303,952/39,321,600 位置（gelu_none 100%）**；46/46 违例样本值精确落在 2^-24 栅格。
  - 影响口径维持"候选训练影响较小、待模型级敏感性/长步训练对照后定稿"；72/39/15 基线保留，不以放宽阈值消除。

### 3. P2 错误聚合崩溃 + checkpoint 退出码 → 已修复

- 聚合改用健壮 schema（`.get` 全字段），非有限分支不再引发 KeyError；`ck_ok/state_ok` **纳入硬失败**（checkpoint 参数/状态不一致 → 非零退出）；入口级负例验证三类注入均可检出。
- **provenance 补充**：`adamw_deep_v3.json` 头部含脚本版本说明；v2 产物标注 superseded；本评论所引全部数据出自 v3（`results/adamw_deep_v3.json`，commit `659b257` 版脚本）。

### 4. P2 文档与 CUDA Graph 边界 → 已同步

- README 结论表："有效且精度通过" → **"已有性能收益观测；数值验收与 FlagScale 集成 pending"**；"唯一可靠路径/强制 fused 建议" → **"本轮已测配置下无非有限值（非普遍定理）+ fp32-state 替代方案"**；长期漂移定性改为"待 fused vs fp64/稳定参考验收"，不先行判为验收通过。
- **graph 输出验证已补**（`probe8_graph_output_verify.py/json`，exit 0）：向已捕获输入缓冲写入新值 → 重放 → 输出与 eager **逐位一致（max diff 0.0）**，证明重放读取当前缓冲内容；输出 sha256 归档。
- 口径修正：12.6× 改为 **"100 次调用捕获后图重放的每调用摊销耗时"**（含图调度与输入缓冲复用）——不再写"enqueue 完全消除/纯设备执行时间"。

### 重跑范围与批次标注

按审阅"只重跑受影响项"执行：重跑 = AdamW 状态验证（v3 全 12 组合）+ FAIL 归因（v2 全 15 类）+ graph 输出对照；**未重跑**微基准 batch-2 / accuracy v6 / 小模型 v2.1（复审确认保留）。脚本版本、命令、退出码均随产物归档；v1/v2 旧产物逐个标注 superseded。

### 后续（不变，按审阅顺序）

FlagScale `build_optimizer` 真实集成（fused 选择/回退/checkpoint，覆盖 fp32/bf16/fp16、mixed、CPU、DTensor、空梯度、不支持类型）→ 集成前不扩 thead 白名单；CUDA Graph 固定 shape 训练段（eager/replay wall+显存+数值一致性，无收益也如实记录）；F-1/F-2 保持 blocked、设备健康确认后隔离复验（不自动重试）；真实 BGE-M3/多卡/FSDP2 独立跟踪。现有条目 `draft / deployable=false / hardware_status=unverified` 不变。
