## 第四轮复审四点已全部修正（PR #8 commit `933ba35`）

### 1. P1 唯一生产聚合/退出码（已修复，无需设备重跑）

- 删除 v3 双聚合列表，抽取唯一生产聚合函数 `aggregate()`；summary 与 `sys.exit` **同源**（`sys.exit(1 if final else 0)`，v3 的 `hard_bad` 旧列表已删除——复审注入的反例（仅 checkpoint 失败 → summary 有记录但 exit 0）不再可能）。
- `--selftest` 改为**调用真实 `aggregate()`**：五类注入（仅 checkpoint 失败 / 仅状态缺失 / 仅状态超差 / 仅 step 不匹配 / 全过对照）**6/6 通过**——失败类全部映射非零、对照映射 0（`logs/adamw_selftest_v4.log`）。
- **重跑 12 组合**（`adamw_deep_v4.json`，exit 1 与 summary.hard_deviations 2,702 项同源）：fused exp_avg/exp_avg_sq 状态超差逐项入账、fp16 stock/foreach 污染证据保留、checkpoint 逐位一致。v1–v3 批次 superseded 并注明各自缺陷（v1/v2 状态对比静默空、v3 双聚合列表）。

### 2. P1 Graph 计时除数（已修复并重跑）

`CAPTURE_CALLS=100` 显式变量，同时用于捕获循环、计时分母与 JSON 元数据；重跑结果：**每调用摊销 0.0053 ms**（此前 0.0001 系 100 倍低估，已修正）；输出对照不变（重放读取当前缓冲、与 eager 逐位一致，max diff 0.0）。

### 3. P2 归因口径（数字同步 + 三级匹配 + tanh 前向独立归因 + TTIR 归档）

- **容差匹配口径声明**：`|sim-device| ≤ 6e-8 + 2·|device|·4.77e-07` 下全张量匹配 **39,304,592 / 39,321,600（99.957%）**——明确注明 6e-8 接近 2^-24 量化间隔本身，**非位级精确证明**（v2 报告的 39,303,952 已按 JSON 求和同步）。
- **栅格命中分母修正**：按存档样本逐个判定——**none 类 25/26（96.2%）、tanh 类 1/19（5.3%）、合计 26/45**（v2 的 46 系分母缺陷——grid24 按 top-5 gidx 计、examples 长度不足 5——已修正）。
- **三级匹配统计**（违例子集，`fail_attribution_stats_v3.json`，exit 1 因 unexplained>0 如实）：
  - **gelu_none 全 12 类：unexplained=0**（exact 位级 95–100%、ulp≤2 覆盖其余、容差 100%）——求和顺序量化机制在 none 负尾**证实**；
  - **gelu_tanh backward 6 类：836/444,667 未解释**（每类 118–290 位置，示例已归档——模拟公式与编译内核细节差异，独立调查）；
  - **gelu_tanh forward 6 类：22,242 个前向违例位置全部未解释**——前向 TTIR 顺序模拟与设备不符，**独立调查、不称已归因**。
- **TTIR 证据已归档**：`results/ttir_evidence/{gelu_backward_none_kernel_rank_2, gelu_tanh_kernel_rank_1, gelu_tanh_kernel_rank_2}.ttir` + `SHA256SUMS`（none: `35095f7f…`、tanh: `a9feb850…`/`5e15dfb4…`）；归因适用范围注明为"已编译内核与已测布局"，不以单内核概括其他路径。
- 未解释位置的三元组（x/sim/device/fp64_ref）示例已随 JSON 归档。

### 4. P2 README/PR 描述同步

- PR 描述整体重写为当前口径（本条上方"当前准确口径"段）；README 微基准邻近段落的"enqueue 完全消除/纯设备执行"历史表述由显式替代句接管（"由本条替代"标注）；旧解释仅在修订记录/历史标注区保留。
- 批次说明：adamw_deep v1–v3 superseded、fail_attribution v1/v2 superseded，均注明具体缺陷；脚本版本与产物对应关系已写入 provenance。

### 状态

issue #7 保持 open；PR #8 待审。FlagScale 集成、稳定参考验收、CUDA Graph 训练段、GELU 归因训练闭环、F-1/F-2 复验、整步/多卡按审阅顺序推进；现有条目 `draft / deployable=false / hardware_status=unverified` 不变。
