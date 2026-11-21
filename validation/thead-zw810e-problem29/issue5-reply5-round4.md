## 复审两处口径已修正（PR #6 commit `317600e`），首轮基线按"基线完成、需优化"定稿

### 1. 负例自测计数：20/20 → **22/22**

实际日志 22 项 PASS（12 项 comparator 单元断言 + 10 项生产路径负例），README 已改为 22/22，不再写错计数。

### 2. embedding 放置：改为 optimizer 前创建并纳入参数（可训练）

- `small_model.py` v2.1：embedding 在 optimizer 构造**前**创建、`.to("cuda")` 后**加入参数列表**（现共 **16 参数**）；新增断言 `embedding_trainable_in_params=true`；梯度证据：embedding weight grad 在 cuda、finite、max_abs 3.7177e-07（native）vs 3.7177e-07（selective，8 位有效数字一致）。
- 修正后的 native vs selective（8 步，3 独立进程/配置，配置内损失序列逐位可复现）：**逐步 loss 相对差 ≤1.2e-07**（fp32 累积量级，embeding 参与后自第 2 步出现）；**16/16 参数 grad sha256 不同**（两套 gelu kernel 数值路径不同，与 v6 精度矩阵的 grad 差异自洽）。
- README 口径按审阅改为：**该负载只证明"当前固定小模型可在本平台运行、数值偏差在 fp32 累积量级"，不能表述为逐位一致**。旧 5 步（15 参数冻结 embedding）产物已由 16 参数新批替代并归档。

### 状态口径（按复审原文）

- PR #6 = 首轮基线证据：**基线完成、需优化**；不作为平台已验收或可部署结论；不据此把相关 fix 标记为 `deployable`；issue #5 保持 open（状态建议"基线完成、问题待优化"）。
- 数据口径维持复审确认的版本：精度矩阵 **126 = 72 PASS / 39 DIFF / 15 hard FAIL**（15 硬失败待位置级归因，不沿用"0 hard failure"）；微基准 **18/18 慢 1.29–6.45×**，selective 无有益算子；enqueue 方向为假设非最终根因；小模型**非逐位一致**；`gems_full` 训练受 F-2 阻断。

### 跟进任务（已写入 README"覆盖与 pending"，拆分如下）

1. 15 个 hard FAIL 位置级归因（native/Gems kernel 输出 × fp64 oracle × 梯度三分量 + 训练轨迹影响）；
2. F-1/F-2 平台恢复后隔离复验（记录驱动/SDK/commit/shape/超时/编译活动；复验前 full-enable training 保持 blocked）；
3. host enqueue 拆分 Python/libentry/Triton launch/设备执行时间，之后再决定优化 fix；
4. 真实 BGE-M3/整模型与多卡测试；AdamW 2.3× 保持 preliminary，不加入 vendor whitelist（需独立重复 + owner 审核）。
