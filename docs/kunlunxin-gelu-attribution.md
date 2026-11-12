# 14 · 胶水 kernel 的真正来源：cast_te/findmax1d 是 `F.linear` 的，不是算子内部的

**结论**：此前把 `cast_te` + `findmax1d` 当作"某算子内部 cast→fp32→max→cast"
（[11](11-vendor-questions.md) §1.2 的假设），并估计"fused softmax 能消掉它"
（[13](13-bottleneck-ops.md) 上一轮讨论）。**两个都错了。**

这对 kernel 是 **`F.linear` 本身**发的：厂商 torch 把 `F.linear` 替换成了
`hydrax.xaccelerator` 的 `fc_fusion` 路径，**每次调用固定发
2× `cast_te` + 2× `findmax1d`（合计 ~34 µs）**。全步
6 linear/层 × 24 层 × 20 forward = 2,880 次 → **~98 ms/步**，
与实测的 cast_te 63.6 + findmax1d 40.5 = 104 ms 对上（数量守恒）。

**证据链**（每个实验独立可跑，脚本见下）：

| 实验 | 观察 | 排除/锁定 |
|---|---|---|
| 整步归因 | 每层 forward 恰 12 对 = **6 个 Linear × 2** | 排除 embedding/loss |
| sdpa vs eager | 都是 288 对/forward，分毫不差 | 与 attention 实现无关 |
| 有 mask vs 无 mask | 完全一致 | 与 mask/padding 无关 |
| 单算子微测 | layer_norm/softmax/gelu/add/dropout/`x@w` 都只发 1 kernel | 排除"算子内部" |
| `F.linear` 微测 | **2×cast_te + 2×findmax1d 恒定出现**（2D/3D、有无 bias、1024/4096 全一样） | **锁定** |

供应商库符号：`hydrax.fc_fusion`、`cast_bf16_to_fp16*`（5 变体）、
`scaled_fc_fusion(_castte)`、`CachedCastBf16ToFp16`、
`TensorState(scale/scale_row/scale_col)`——即 **bf16→fp16 带 scale 的
GEMM 输入前处理**（amax + cast）。内部意图是推断（.so 无源码）。

**能不能关**：不能。`XMLIR_ENABLE_LINEAR_FC_FUSION=0` 会让这对 kernel 消失
（端到端 -104 ms），但 FFN 形状 GEMM 变慢（0.162→0.277 ms/次），
整步 kernel 时间 **2,649 → 2,747 ms（+98 ms）**。
税 34 µs < fast-GEMM 收益 115 µs，厂商默认是划算的。

**该向厂商要的**：
1. **weight 侧 amax/cast 缓存**——权重每步只更新一次，却被重复 cast 20 次/步
   （每 micro-forward 一次）。`CachedCastBf16ToFp16`/`TensorState(is_param=True)`
   部件已在库里，缺缓存命中。命中则每 linear 省 ~17 µs → **~49 ms/步**。
2. `scaled_fc_fusion` 的显式 scale 入口（训练侧自管 scale）。

**顺带把 "other" 104.3 ms（58×）拆开了**：

| 内容 | ms/步 | 次数 | 处置 |
|---|---|---|---|
| AdamW 逐参数 | 41.8 | 3,890 | **`fused=True` → 17.3 ms（2.4×）**，数值 bf16 等价（max diff 7.6e-5） |
| clip_grad_norm_ | 9.8 | 2,099 | foreach 只省 ~2 ms；上限需 multi-tensor reduce |
| copy/pad | ~50 | 6,231 | ~0.63 ms/forward 是厂商 SDPA 的 mask 预处理，反向一份 |
| misc | 0.85 | 125 | radixsort/unique 等 |

A100 上这两块是 `_foreach_*`（18.6 ms）+1.8 ms——**58× 的真相是"逐参数 vs 融合"，
不是 kernel 慢**。

**修正记录**：
- 11 §1.2 的假设证伪；
- "机器上没有控制精度的环境变量"不成立——`strings` 提取到 50+ 个厂商开关
  （`XMLIR_ENABLE_LINEAR_FC_FUSION`、`XMLIR_MATMUL_FAST_MODE`、
  `XMLIR_FA_GEMM_TYPE`、`XMLIR_FC_AUTOTUNE_*`、`HYDRA_CASTTE_TYPE`、
  `HYDRA_FINDMAX_TYPE`、`XPUAPI_BF16_FP16_MODE`…），建议先问语义再逐个测；
- "fused softmax 消掉 cast+max（~36 ms）"的估计作废。

**方法与复现**：原始 trace 未留存、模型/数据集不在本机，用重建负载
（XLM-R large 随机权重 + 同款 5-forward×batch10×accum4 配方 + 同款对比损失），
kernel 级与文档对照偏差 ≤10%（总时间 2,649 vs 2,923；cast_te 63.6 vs 66.2；
tensor_calc_basic 87.9 vs 90.1；dropout 54.5 vs 57.4；clip+opt 墙钟 60.5 vs 62）。
重建无 FSDP/DDP，通信侧 `cluster_cast_kl3`（12,399 次/步）仍无法归因，
需在真实训练节点跑 `profile_step_segments.py`。

完整报告与脚本：`problem29/ATTRIBUTION.md`、`problem29/probes/`。

## 对账：与《profiler 方法与移交清单》（problem29/reason.md，飞书 wiki 镜像）

reason.md 是方法论文档（三层归因 + 局限 + 移交清单），数字与 docs/retrieval
08/13 同源，无冲突。对账结论：

| reason.md 的表述 | 本文档（14 + problem29）的关系 |
|---|---|
| §5.1 "我们的方法只能发现，不能定位根因"（"分析 kernel 内部为什么慢"列在移交侧） | 本文 §二/§三 正是这一层：cast_te/findmax1d 定位到 `F.linear`(fc_fusion)，"other" 拆到 AdamW/clip/copy——根因定位已在本地完成，移交内容应据此更新 |
| §5.6 移交表"实现融合、向量化、精度路径修正" | GELU 已本地交付（`gelu-kunlunxin-pow`），优化器侧交付 `adamw-fused`（flagscale 层）；reason.md §6.1 目标"GELU 1,028.9 → <150 ms/步"按修复后单核耗时推算 ≈129 ms，可达标（待平台验收） |
| §5.5 多卡边界：2 卡 FSDP2 实测通信占 device 56.3%（A100，AllGather 308.6 ms）；"通信与计算分列、按 rank 归集、区分等待与带宽、同并行度对比"列为运行时层需求 | 前两条 torch.profiler 可做，采纳进归因工具包；后两条（等待 vs 带宽、跨并行度）确需运行时层支持，不在本工具范围。`cluster_cast_kl3` 12,399 次/步的归因必须在此类真实多卡环境做，单卡重建复现不出 |
| §5.2/5.3 昆仑芯 profiler 两个开关、event key 是裸 kernel 名 | 本文所有归类脚本按裸名匹配，与该口径一致；ascend 910C 只有微基准，本文不涉及 |

对移交清单（§6 / 11-vendor-questions）的修订建议：

1. 11 §1.2 的"cast→fp32→max→cast 来自某算子内部"假设已证伪并替代
   （见本文修正记录）；给厂商的问题应改为"weight 侧 amax/cast 缓存"
   与"scaled_fc_fusion 显式 scale 入口"（本文 §三）。
2. 新增一问：`XMLIR_ENABLE_LINEAR_FC_FUSION` 等环境变量的官方语义
   （本文 §六清单）——运行时/编译层不应各自依赖未文档化的开关。
3. AdamW/clip 的逐参数分解（"other" 里 42+10 ms）不依赖厂商新 kernel：
   flagscale 侧已修（`adamw-fused`，fix 索引第 12 项），移交清单该项可从
   "运行时层"移除，改为运行时确认 foreach 分解行为是否为预期。

---

## 归属与维护说明（2026-11-10）

本文档的维护位置已迁移：**性能优化的全部文档统一在 flag-fixes 仓库维护**，
不再在 FlagScale 仓库的 docs/retrieval 下另立一套。上游 `docs/retrieval/00-13`
（飞书 wiki 镜像）保持原样。

- 完整归因报告：`/workspace/problem29/ATTRIBUTION.md`（cast_te/findmax1d =
  F.linear fc_fusion 前处理；"other" 104 ms 拆解；重建负载方法论）
- 整步 A/B 矩阵：`/workspace/problem29/RESULT.md` 第五阶段（-14.0%）
- GELU kernel 深挖（ALU 上限/查表死路/两个 vendor 编译器 bug）：
  `/workspace/problem29/RESULT.md` 第六阶段
- profiler 方法对账（reason.md）：见本文上方各节

对应的修复索引（均 draft / deployable=false / hardware_status=unverified）：

| fix | 层 | 作用 |
|---|---|---|
| gelu-kunlunxin-pow | FlagGems | gelu 家族去 libdevice pow |
| adamw-fused | FlagScale | AdamW 单内核 |
| flaggems-selective-enable | FlagScale | 只注册打补丁的 gelu 家族 |

三者同装：BGE-M3 重建负载整步 2,665.9 → 2,291.2 ms（**-14.0%**），全链路走
FlagScale 集成路径（`enable_flaggems` include 模式）。

遗留决策项/移交项：

1. `approximate="tanh"`：单核再省 ~100 ms/步，**训练配方决策**（bf16 输出
   差异 ~1 ulp 量级），需 loss 曲线签核；
2. 厂商三项：libdevice erf/exp 质量（~40 ns/元素）、F.linear fc_fusion 的
   weight 侧 amax/cast 缓存（权重被重复 cast 20 次/步）、
   `XMLIR_ENABLE_LINEAR_FC_FUSION` 等 50+ 环境变量语义；
3. 平台节点复跑 A/B 矩阵（命令见 problem29/RESULT.md 第五阶段）；
4. 真实多卡 FSDP2 归因 `cluster_cast_kl3`（12,399 次/步，单卡复现不出）。

## 附：最终运行配置与总账（2026-11-10，整步实测）

在全部代码修复之上，厂商环境变量扫描发现一对零代码收益
（`problem29/RESULT.md` 第八阶段，wall 3 步中位数，数值逐位不变）：

```bash
export XMLIR_MATMUL_FAST_MODE=1      # GEMM 换 fast algo（精度逐位一致）
export XMLIR_MATMUL_FAST_TUNE=1      # 配套 tune
```

**总账**（BGE-M3 重建负载，reason.md 口径）：

| 指标 | 基线 | 最终 | 提升 |
|---|---:|---:|---|
| step wall | 2,665.9 ms | 2,258.3 ms | **-15.3%**（吞吐 +18.1%） |
| 对 A100（1,419 ms） | 1.86× | 1.59× | 差距收敛 35.4% |

分解：GELU -350 ms（gelu-kunlunxin-pow + flaggems-selective-enable）+
AdamW -24 ms（adamw-fused）+ GEMM algo -33 ms（上述 env，零代码）。

无效/负收益项（已测，勿重试）：`KLX_USE_AUTOTUNE=1`（0）、
`XMLIR_ENABLE_XBLAS_ADDMM=1`（0）、`XMLIR_FC_AUTOTUNE_BOOL=1`（+12.7 ms 更差）。

训练侧决策项：`approximate="tanh"`（单核再省 ~100 ms/步，需 loss 签核）。
厂商移交项：libdevice erf/exp 质量、fc_fusion weight-scale 缓存、env 语义、
两个 Triton 编译器崩溃 bug（nan splat / 循环标量 load，最小复现在
problem29/probes/）。
