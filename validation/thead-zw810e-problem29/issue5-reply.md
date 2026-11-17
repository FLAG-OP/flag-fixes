ZW810E 首轮测试与评估已完成（单卡，2026-11-12），结论：**需优化——FlagGems 在 ZW810E 当前为净负收益；接入策略 = 不启用或 selective enable（本轮未发现有益算子）**。证据归档于 `validation/thead-zw810e-problem29/`（脚本、原始 JSON/log、退出码、环境记录），结果表如下。

## 环境（完整记录：results/env_record.json）

- 设备：PPU-ZW810E（Beethoven，cc 8.0，96 GB，64 SM，PCIe Gen5），本容器分配单卡
- 驱动/固件：driver `1.5.1-1d747a`，HGGC 13.0，PPU SDK `2.1.0-a5f865`
- 软件：torch 2.10.0 / triton 3.6.0 / Python 3.12.3；目标树 FlagGems `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`v5.3.5-46-g29e14822`）
- device/dispatch：`PPU_SDK` env → vendor=`thead`（`_thead` 后端），dispatch key `cuda`；`enable()` 覆写 aten CUDA 注册，selective 走 `only_enable`
- 命中证明（profiler）：enable 后 `aten::gelu → gelu_none_kernel_rank_1`、tanh 前后向 `gelu_tanh_kernel_rank_1`+`gelu_backward_tanh_kernel_rank_2`、`F.gelu(a)*b → gelu_none_and_mul…`、`max → max_kernel`、`add → add_func_kernel_rank_1`；**参考路径一律在 enable 前运行**（脚本 P1 阶段），确保参考未被替换

## 结果表（微基准：p50 ms，3 独立进程取中位；12.25M PR3 口径；warmup+同步计时）

| 配置 | 命中路径 | 精度判定 | p50 | p95 | 相对 native | 显存峰值 |
|---|---|---|---:|---:|---:|---|
| native（厂商 aten） | GeluCUDAKernelImpl 等 | 基准（fp64 oracle 达标） | — | — | 1.00× | 各配置已记录 |
| gems_full（enable 全量） | Triton kernel（已验证命中） | 0 硬失败；FTZ 区 DIFF | gelu_fwd bf16 0.081 vs native 0.028 | 0.086 | **17/18 项慢 1.3–6.4×，无一项更快** | 见 JSON |
| gems_selective（only_enable=gelu 族） | 包含项 Triton / 其余 native | 同上 | 未包含算子 = native（findmax 0.021 vs full 0.114） | — | 包含项仍 1.6–3.7× | 见 JSON |

关键项：mm 3.21×、findmax 5.30×、pointwise_add bf16 6.12×、mul_scalar bf16 5.61×、cast fp32 3.14×、gelu_and_mul 6.37×（fused API 3.58×）、gelu fwd/bwd 1.6–3.6×、sdpa 1.32×、transpose_copy 2.56×。
**归因**：尺寸扫描（1.2M→98M）显示 FlagGems 有 **每算子固定开销 ~60–80 µs**（1.2M 5.58× → 98M 1.00× 收敛，bf16 稳态仍 1.18×）。训练步内中小算子密集，固定开销为主要税负——与平台文档"FlagGems 13 算子 12 慢"口径方向一致。

## 精度与训练对照

- **gelu 族 126 用例矩阵**（none/tanh 前向+反向、gelu_and_mul × fp32/bf16/fp16 × 连续/转置/切片 × 空/小/典型/大/边界）：**0 硬失败**（84 PASS + 42 DIFF）。DIFF 全部为次正规（FTZ）输出区 FlagGems 与 native 的 ±1–2 ulp 位级差异（9 例另有 ±0 符号差）；NaN/Inf 掩码全部一致；非次正规区相对 fp64 oracle 满足 `rel ≤ max(4·eps, 1.5·native_rel)`（FlagGems 多数更准）。边界输入含 **fp16 `0x03FF`、bf16 `0x007F`、fp32 `0x007FFFFF`** 及 ±Inf/NaN/±0，FTZ 位级计数已逐用例归档。
- **AdamW stock vs fused**：`fused=True` 在 ZW810E 可用（无回退异常）；5 步轨迹与 stock 一致（fp32 末参差 ≤2.4e-6，bf16 1 ulp，loss 无差异）。按 PR3 护栏 FlagScale 侧非昆仑芯仍默认 stock；数据已具备，加入 `_FUSED_ADAMW_VALIDATED_VENDORS` 需另立 fix 条目 + owner 审核。
- **小模型**（固定种子 gated MLP + 2 层 attention，fp32，无外部权重）：native / gems_full / gems_selective 三组 loss 轨迹、15/15 参数梯度与 8 步后参数 **逐位一致（sha256 相同）**；稳态步时 13.4–13.8 / 13.8–14.7 / 13.4–13.6 ms（3 进程复跑）——该负载下启用无收益也无正确性风险。

## pending（不计通过）

- 多卡/通信整步：本环境单卡分配，无真实多卡训练环境。
- problem29 真实 BGE-M3 runner / x-benchmark 整模型：本机无该容器与 runner；以 12.25M 口径微基准 + 小模型替代（差异已注明）。
- ZW810E 平台专属 fix 条目：首轮结论为"无需新 fix，优先厂商 kernel/融合"（对应 PR3 移交项）；如需把 gelu/cast/findmax 热点做成厂商侧优化请求，按仓库流程另立条目。

原始数据：`validation/thead-zw810e-problem29/results/*.json`（accuracy_matrix / perf_{native,gems_full,gems_selective}_r{1..3} / probe3_scaling_hits / small_model_3rep / adamw_check / env_record），复现命令见该目录 README。合并 PR3 不等于部署授权：现有条目仍为 `draft / deployable=false / hardware_status=unverified`（本轮为平台基线记录，不改变任何 fix 的验收状态）。
