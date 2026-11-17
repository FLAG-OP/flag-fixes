## 第九轮执行顺序 1–3 完成（PR #8 commit `1893497`）：原对象 A/B 矩阵 + GEMM 消融 + masked-attention 候选量化

### 1. 原对象 A/B 矩阵（native/selective × stock/fused，3 计时步中位，skip 2，exit 0 ×4）

| 配置 | 整步 wall | forward/步（×4）| backward/步（×4）| clip+optimizer/步 | 非有限值 |
|---|---:|---:|---:|---:|---|
| native/stock | 2,123.7 ms | 681.7 | 1,378.8 | 20.2 | 无 |
| native/fused | 2,132.7 ms | 689.2 | 1,385.7 | 13.9 | 无 |
| selective(gelu)/stock | 2,164.5 ms | 682.0 | 1,396.0 | 20.2 | 无 |
| selective(gelu)/fused | 2,171.5 ms | 726.8 | 1,404.8 | 14.3 | 无 |

- **fused AdamW 在原对象上仅省 clip+optimizer 6.3 ms/步（20.2→13.9）**；该段占整步 ~0.95%——整步差 -0.3%（运行噪声内）。与审阅占比公式一致：p≈0.0095、s≈1.45 → 预期 ≈-0.3%。**原对象（BGE-M3，389 张量、fp32 state）上 optimizer 段占比远小于昆仑芯配方口径，fused 整步收益可忽略**。
- **FlagGems selective(gelu) 回归 +40.8 ms/步（+1.9%）**：gelu 由 native（problem.md：PPU gelu 0.9× 快于 A100）换 FlagGems triton 后变慢——**印证 problem.md"FlagGems 不是解法"，selective 注册 gelu 族在 PPU 原对象为负收益**。
- 非 finite：全部配置无 NaN/Inf。
- 局限（如实）：loss/显存未入 JSON（原脚本不含该字段）；单次运行 × 4 配置——**≥3 独立进程复制标为首轮待补**（本表为首轮 A/B）。

### 2. GEMM 消融（真实 shape M2990 K1024 N1024 bf16，probe10）

| 消融 | 结果 |
|---|---|
| allow_bf16_reduced_precision_reduction True/False | 0.0567 / 0.0564 ms（110.6 / 111.2 TFLOP/s）——无收益 |
| 输入非连续布局 | +8%（0.0611 vs 0.0566 ms）——contiguous 更优 |
| fp32 matmul | 4× 慢（0.2256 ms）——维持 bf16 |

结论：bf16 GEMM 在真实 shape 已 110 TFLOP/s（与 probe9 一致），tile/layout/累加精度的应用层消融无改善空间——**GEMM 差距属厂商 kernel/硬件层，运行时层移交项维持**。

### 3. masked attention：等长分桶候选量化（probe11，真实 shape B8 H16 S512 D64 bf16）

| 路径 | ms |
|---|---:|
| 带 bool padding mask（现路径 → efficient 后端） | **0.5866** |
| 等长无 mask（flash 后端） | **0.1313**（4.46×）|

按审阅公式（p=attention 占 wall 26.9%、s=4.46）：预测整步 ≈ **-20% wall**。**前提与边界**：等长分桶需训练侧改造（loader 已 length-sort 窗口 256，但批内仍 padding 到批内最长）；等长批次下无 padding 即无 mask，mask 语义天然满足、梯度正确性由构造保证；**未实施，登记为候选方向（训练侧 + 运行时层）**。

### 4. F-2 最小 shape 隔离（单次触发）

最小 shape（M8 K256 N512 fp32，enable 下 mm backward）单次触发**仍挂起**（300s 超时；native 控制通过）——F-2 与 shape 无关，继续 blocked；未重复触发其他组合。

### 5. 措辞修正（复审 A 项，README 已改）

归因对照改为"**方向性复现**（GEMM/attention/GELU 排序一致；copy/cast 248.8 vs 72.5 待同口径核对——kernel 层 vs aten 层统计口径差异）"；复现文档保留原始命令 + 本机覆盖项（conf/run/bge_m3_ppu.yaml 两处覆盖显式标注）。

### 状态

PR #8 HEAD：`1893497`（11 文件：A/B JSON ×3、probe10/11、AB 变体脚本、日志）。issue #7 保持 open；现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变；`-5.6%`（小模型）与 12.6×（组件）维持探索标注；原对象上未产生优化收益（A/B 实测），**masked-attention 等长分桶（预测 ~-20% wall）与厂商 GEMM/attention 内核移交项为下一步候选**。
