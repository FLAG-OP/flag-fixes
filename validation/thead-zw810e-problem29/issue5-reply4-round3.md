## 复审五点已全部处理（PR #6 新增 commit `0fd26e3`）

按"最小收尾顺序"逐条对照（详细证据见 PR #6 第三轮 commit 与 README v3 修订记录）：

### 1. P1 小模型设备放置 → 已修正并重跑

- 根因确认：v1 脚本未 `.to(device)`，模型实际跑在 CPU——**v1 的"三组逐位一致"是 CPU aten 自身一致性，不构成平台证据，已作废**（产物移入 `results/small_model_v1_superseded/`）。
- v2：模型/head/embedding/token 全部显式 `.to("cuda")`，**optimizer 构造前完成**；JSON 记录断言 `params_all_cuda_before_optim=true`、`grads_all_on_cuda=true`；逐配置 profiler 命中证据（selective=`gelu_none_kernel_rank_1`/`gelu_backward_none_kernel_rank_1`，native=`GeluCUDAKernelImpl`/`GeluBackwardCUDA…`）。
- 统一 8 步 + 完整证据：逐步 loss、15/15 参数 grad sha256/max_abs/device、param sha。**native vs selective（PPU 上）**：loss 前 7 步逐位一致、第 8 步相对差 9.15e-08；梯度 max_abs 4 位有效数字一致；3 独立进程逐位可复现。稳态步时 native ~2.2 / selective ~2.85 ms。
- **gems_full 训练被平台阻断**（见第 3 点 F-1/F-2），8 步对照无法提供——按 issue 口径登记为被阻断，不虚构数据。

### 2. P1 精度门槛稀释 → 判据逐位置化，暴露 15 硬失败

- `comparator.py` 抽为**生产判定共享模块**（矩阵与负例自测同源调用）；normal zone（|oracle| ≥ min_normal）**逐位置**判 `rel ≤ max(4·eps, 1.5×native同位置)`，次正规位置的 native 误差不再放宽其他位置；**审阅反例进负例自测并判 FAIL**。
- **v6 结果（如实报告）：72 PASS / 39 DIFF / 15 FAIL，进程 exit 1**。15 硬失败 = `gelu_backward`（fp32/bf16 none、fp32 tanh）在近抵消小梯度区（|grad|≈1e-8–1e-6）逐位置超差（gems 最差达 native ~2×；7–74 万位置/2.6M）+ gelu_tanh fp32 平滑区前向 ~2.5× native 相对误差（abs ~1e-7 量级）。**top-5 违例四元组（位置/native/gems/oracle）逐例归档**；绝对幅度小，对训练实际影响的评估列入移交项。
- normal-zone 与 FTZ-zone **分开报告**：26 例存在合计 6,683,858 个正常区位级差异（±1–2 ULP，门内记录）；FTZ 位型差 33 例（91,135 位置，max abs 1.19e-07）。不再表述"全部差异仅在 FTZ 区"。
- ±0 符号差 9 例 → 按声明规则判 DIFF（**原始位型**比较，`torch.equal` 的 +0==-0 缺陷已规避，`bits_equal_out` 字段单列）；oracle 为 aten fp64 精确公式并**注明负尾下溢边界（非无限精度真值）**。

### 3. P2 平台阻断（新发现，已单次触发出证）

- **F-1**：FlagGems sdpa（`_flash_attention_forward` 注册路径）训练形状 fp32 `[1,8,128,256]` **前向挂起**（>300s、无编译活动）；推理 bf16 `[8,16,512,64]` 正常（控制组 ✓）。
- **F-2**：FlagGems **matmul backward** 挂起（fp32 `[8,128,256]@[256,512]`；forward 控制组 ✓）。
- 按 issue 设备异常规则**单次隔离触发、不重试**（`probe7_platform_blockers.py`，exit 1）；来源注记：同树负载 11-12 晚曾完整跑通——挂起与设备/驱动状态变化相关，**待平台恢复后隔离复验**；未据此确认算子实现新增错误。

### 4. P2 ULP/自测/计数退出码 → 全部修复

- ULP 有序映射改为**无符号位型**标准映射：±0 距离 1、相邻值 1、±min_normal 跨零 2049（fp16）、min_sub→0 距离 1——精确断言入 `comparator.selftest()`；v5 的 32767/2147483647 系符号位剥离错误，已修正并承认该度量曾夸大 `ulp_grad`。
- 失败计数改为 `passed/diff/hard_failed` 分列（summary 不再把 DIFF 计入 failed）；`accuracy_matrix.py` 聚合硬失败→**非零退出**，`--inject-fail` 生产入口注入自测：污染用例 FAIL + exit 1（`accuracy_matrix_inject.json`）；perf_matrix 错误聚合→非零退出。
- 自测改为**直接调用生产判定**（20/20，含 NaN/±Inf 符号/越界/grad 缺失拒绝、FTZ-only 与 ±0-only 落 DIFF）。

### 5. 标签与措辞

- perf batch-2 重跑：`gelu_fwd_bwd_none`（实为前向+反向）、`cast_to/bfloat16` 改 bf16→fp32→bf16 往返（原 no-op）、`gelu_and_mul` 形状统一为实测 **(4096,1280)**；batch-1 归档 `perf_batch1_superseded/`（结论方向不变，18/18 慢 1.29–6.45×）。
- probe5 措辞收敛：支持"优先探索调用提交开销"方向假设，**python/libentry/launch 环节拆分未做、纯设备执行成本未单独测，不作最终根因**；AdamW 2.3× 标注单次探针初步结果，白名单不放开。

### 状态

- PR #6：commit `f288df6` → `0fd26e3`（89 blob 更新 + 3 删除 + 3 文档）；issue 保持 open。
- 微基准结论按复审意见**保留未重跑方向**；重跑仅限受影响项（精度矩阵 v6、小模型 v2、perf batch-2 标签修正）。
- 现有条目维持 `draft / deployable=false / hardware_status=unverified`；F-1/F-2、真实 BGE-M3、多卡、平台 fix 立项均为后续任务。
