## 首轮复审回应：证据已落地 PR #6，判据与归因已按意见修订

复审三点已逐项处理，证据 PR：**https://github.com/FLAG-OP/flag-fixes/pull/6**（分支 `issue5-zw810e-evidence`，commit `f288df6`，55 文件；CI 会在 PR 上运行）。按审阅的"最小收尾顺序"逐条对照：

### 1. 证据落地（原阻塞项）

- `validation/thead-zw810e-problem29/`：README（第二轮修订版）+ scripts 10 个 + results 20 个原始 JSON（含逐迭代性能样本 `samples_ms`）+ logs 17 个（含退出码），共 ~300 KB 文本。
- `docs/current-status.md`（硬件复验行、平台范围说明、变更记录行）与 `bundles/kunlunxin-bge-perf/README.md`（平台矩阵 ZW810E 行）同步更新，消除"已同步"与远端不一致。
- 设备侧数据**未重跑**（按审阅"不必因证据未推送而重跑设备实验"）；重跑的只有证据补强项（见下）。issue 完成通报口径相应收敛为："设备侧首轮运行完成，证据以 PR #6 提交审阅"。

### 2. 精度判据收敛

- **DIFF 三分类接受策略**（README"精度判据"节）：PASS = NaN/Inf 掩码与 **±Inf 符号**一致 + grad 存在性一致 + `rel_err(gems,oracle) ≤ max(4·eps, 1.5·native_rel)`（非次正规区）+ FTZ 区逐位一致；**DIFF（记录性接受，不静默通过）** = 仅 FTZ 区位级不一致（±1–2 ulp，33 例）或仅 ±0 符号差（9 例，逐例位置计数归档，**不因处于次正规区自动忽略**）；FAIL = 特殊值语义/越界/grad 缺失。最终：84 PASS + 42 DIFF + **0 FAIL**。
- **逐例证据**：`accuracy_matrix.json` v5 每用例含 dtype/shape/layout、最差位置（flat index）与双方输出位型（hex）、NaN/Inf/±0 掩码与计数、signed-zero 差异计数、FTZ 定义（oracle `|y| < dtype 最小正规数`）、oracle 相对误差（gems 与 native 双侧）。
- **边界位型确认**：0x03FF / 0x007F / 0x007FFFFF 断言为各自 dtype 最大次正规（负例自测前 3 项）。
- **oracle**：aten fp64 精确公式（none：`x·Φ(x)` erf 精确；tanh：精确解析式；grad 为对应解析导数），与被测同输入同设备；输入来源=脚本固定种子确定性构造（`make_value_class`）。
- **阈值出处**：`4·eps` 与 `1.5×native_rel` 自 v3 起固定于脚本 docstring（先于执行声明），本轮判定门 v3→v5 未变、只增证据字段——无按本批误差反推。
- **批次关系**：v1（直调 API 无 autograd）、v2（ULP 按 dtype 升位虚高 + NaN 入 diff）两批**判定作废、未归档**；归档 JSON 为 **v5 全量重跑**（同固定种子输入），非旧批重评分；无遗留未重跑的原始数据。
- **比较器负例自测（设备无关，exit 0）**：`scripts/test_comparator_negative.py` 10/10——NaN 出现、**±Inf 符号翻转**、Inf 出现、grad 缺失、越界误差、gems 劣于 native 均被拒绝；FTZ-only 与 ±0-only 正确落入 DIFF 而非 PASS。**±Inf 符号缺口正是负例自测发现的真实缺陷（v4 的 isinf 掩码不区分符号），已在 v5 修复并加回归**。
- **范围限定**：小模型逐位一致仅限该模型与测试范围，不再表述为平台"无正确性风险"；`gelu_and_mul` 梯度 N/A 保留。

### 3. 性能：定向探索已完成，归因修正

- **原始样本**：`results/perf_{native,gems_full,gems_selective}_r{1..3}.json` 每算子含 30 次逐迭代样本（`samples_ms`）；分位数口径已写明——**进程内统计**，跨进程报告中位，不合并样本。
- **接入配置 vs 优化实现**：已在 README 明确 native/full/selective 为**三种接入配置对照**，本轮未实现任何平台优化候选，不写成"三种实现对照"。
- **"60–80 µs 固定开销"从假设升级为拟合证据，且归因修正**（`probe5_overhead_attribution.json`，方法注记在 JSON 内）：
  - 线性拟合（0.3M–98M 六点，**进程严格隔离**，native 进程不 import flag_gems——首版同进程标签失真已作废并披露）：FlagGems 截距 **60.4/70.0 µs** vs native **4.7/4.6 µs**（gelu/add fp32），**斜率与 native 基本相同**（0.0030–0.0033 vs 0.0038–0.0039 ns/元素）→ 慢在每调用固定成本，不在稳态带宽。
  - host wall / 设备 event 分解 + enqueue 法（无同步批量提交的 host 墙钟）：native enqueue **≈5 µs/次**，FlagGems **≈57–69 µs/次** → 固定成本主体是 **host 侧 flag_gems 调用链（python 分发 + libentry + triton launch）**，空闲 GPU 上该延迟落入 event 窗口。
  - 直调入口 vs aten 注册入口：profiler 核验**命中同一 `gelu_none_kernel_rank_1`**，enqueue 差仅 ~5 µs → 注册/调度层贡献小。
  - profiler 计时注记：profiler 放大 device_time（同 1.2M gelu 报 0.58 ms vs event 0.07 ms），故仅用于命中证据不作计时声明。
  - **结论修正**：优化优先级应先考察 per-call host 开销（调度/launch 路径）；首轮"必须厂商融合""无需新 fix"两句**撤回**，是否立项平台 fix 留待按此方向探索后再定。
- **结论范围**：首轮"需优化"限定为 FlagGems `29e14822` / 本尺寸集 / 单卡 / 本轮负载，不外推 ZW810E 全部场景，不代表优化不可行。
- **AdamW**：5 步正确性结论保留；新增性能证据 `probe6_adamw_perf.json`（336 张量 stock 26.8 ms/步 / 2,688 launch vs fused 11.7 ms/步 / 39 launch = 2.3×；单参数持平；显存峰值同量级）。**本轮不放开 vendor 白名单**，stock/fused 耗时数据供后续平台 fix 条目评估。

### 后续任务拆分（pending 不计通过）

真实 BGE-M3 / x-benchmark 整模型、多卡/通信整步、ZW810E 平台 fix 立项（若立项优先探索 per-call host 开销路径）、`_FUSED_ADAMW_VALIDATED_VENDORS` 加入（需 owner 审核）——均已列为后续任务，不阻塞首轮基线闭环。

现有 fix 条目维持 `draft / deployable=false / hardware_status=unverified` 不变；本 PR 为平台基线记录，不构成任何 fix 的硬件验收，也不代表 PR2 性能瓶颈已修复。若 PR #6 证据复核通过，首轮平台基线任务即完成。
