# Bundle: kunlunxin-bge-perf（problem29 答复包）

**面向问题**：problem29——"gelu / cast_te / findmax1d / tensor_calc_basic /
transpose 等算子严重拖慢昆仑芯 P800 训练吞吐（BGE-M3 对 A100 kernel 2.06×），
需要进一步优化"。

本 bundle 是该问题的整体答复包：3 个 fix + 1 组运行配置，按下方顺序应用。
组合验收已完成（整步实测，见"组合证据"）——单项 fix 的独立验证见各自
`fixes/<id>/`。

## 组成与应用顺序

| 序 | fix ID | 目标包 | 目标文件 | 内容 |
|---|---|---|---|---|
| 1 | `gelu-kunlunxin-pow` | flag_gems | `_kunlunxin/ops/gelu.py`、`_kunlunxin/fused/gelu_and_mul.py` | gelu 家族 libdevice `pow` → 乘法 |
| 2 | `adamw-fused` | flagscale | `train/train_retrieval.py` | AdamW 按参数类型自动选 `fused=True` |
| 3 | `flaggems-selective-enable` | flagscale | `train/utils/flaggems.py` | `flaggems_include` 选择性注册（必须与 1 同装） |
| 4 | 运行 env | 运行脚本 | `XMLIR_MATMUL_FAST_MODE=1`、`XMLIR_MATMUL_FAST_TUNE=1` | GEMM fast algo（精度逐位不变） |

1 与 2/3 顺序无关；3 必须在 1 之后（include 注册的 kernel 来自打补丁的
FlagGems）。所有 fix 均 draft / deployable=false / hardware_status=unverified。

## 逐项答复记分卡（problem29 点名的 5 项）

| 原问题项 | 基线 ms/步 | 现在 | Δ | 答复 |
|---|---:|---:|---:|---|
| **gelu** | 851.8 | 501.9 | **-349.9** | ✅ **已解决**（fix 1+3）；残余在本栈 ALU 下限，tanh 可再省 ~100 ms（训练侧决策） |
| **cast_te** | ~66 | 不变 | 0 | 🔍 **根因定位**：F.linear fc_fusion 前处理税（34 µs/次，全步 ~104 ms）；厂商项 = weight-scale 缓存（潜力 ~-49 ms/步） |
| **findmax1d** | ~40 | 不变 | 0 | 🔍 同上（与 cast_te 严格成对，同一厂商请求覆盖） |
| **tensor_calc_basic** | ~166（类目） | ~166 | **-11.5**（间接受益） | ⚠️ fused AdamW 消掉逐参数 scalar kernel；主体为厂商兜底 pointwise，无自收益空间 |
| **transpose** | ~150 | 不变 | 0 | ❌ 厂商 flash/SDPA 包装内部 layout，需厂商带原生 mask 的融合 attention |

清单外新增解决：**optimizer/AdamW -27.1 ms**（fix 2）、**GEMM algo -33 ms**
（env）、**FlagGems 接入方式**（fix 3，前提条件）。

## 组合证据（整步实测，重建负载 BGE-M3，reason.md 口径）

| 配置 | wall/step（3 步中位） | Δ |
|---|---:|---:|
| 基线 | 2,665.9 ms | — |
| + fix 2 | 2,641.7 | -0.9% |
| + fix 1+3（直接 remap） | 2,313.8 | -13.2% |
| + fix 1+2+3（直接 remap，D） | 2,296.6 | -13.9% |
| **+ fix 1+2+3（FlagScale 集成路径，E）** | **2,291.0** | **-14.0%** |
| **E + env（最终推荐配置）** | **2,258.3** | **-15.3%（吞吐 +18.1%）** |

对 A100（kernel 1,419 ms）：1.86× → **1.59×**（剩余差距收敛 35.4%）。
kernel 归因逐类目对照：gelu -349.9、optimizer -27.1、elementwise -11.5、
gemm(env) -32.7，其余类目零变化——**修复只动了该动的地方**。

组合级数据：`problem29/step-experiment/`（A–F + env-E* 原始 JSON/log）。

## 已知边界与遗留

- 全部 `hardware_status=unverified`：平台整模型 x-benchmark 未跑（本机无
  runner/模型/数据）；A/B 矩阵复跑命令见 problem29/RESULT.md 第五阶段。
- 训练侧决策项：`approximate="tanh"`（gelu 再省 ~100 ms/步，bf16 输出差异
  ~1 ulp 量级，需 loss 曲线签核）。
- 厂商移交项：①fc_fusion weight 侧 amax/cast 缓存（税不缓存，每步冗余
  20 次，~49 ms/步）；②gelu/gelu_backward 做 fc_fusion activation epilogue
  （cuBLASLt EPILOGUE_GELU 先例，预估 200–300 ms/步）；③libdevice erf/exp
  质量（40 ns/元素）；④50+ XMLIR/HYDRA env 语义；⑤两个 Triton 编译器
  崩溃 bug（最小复现在 problem29/probes/）。
- 真实多卡 FSDP2：`cluster_cast_kl3`（12,399 次/步）单卡复现不出，需真实
  节点归因。
- 边界验证遗留（PR3 四轮评审）：fp16 邻界输入（0x03FF）的 base/fixed
  dump **pending**——未打补丁 pow 路径触发 error 721 驱动异常并楔死
  设备（现场：gelu-kunlunxin-pow/results/history/），设备恢复后按
  `tests/verify_gelu_boundary.py` 流程补跑并刷新对比报告；
  当前已提交基线的 fp16 用例为 0x007F（bf16 邻界位型，对 fp16 是普通
  次正规数），patch 前后同输入对照仍有效，但不构成 fp16 邻界覆盖。

## 平台状态矩阵（PR #3 评审补充）

problem29 覆盖三个平台；**本 bundle 当前的证据只覆盖昆仑芯单卡**，
`-15.3%` 不得外推为三平台结论。昇腾/平头哥需要独立建立 fix 条目
（torch/npu 注册路径、dispatch、性能基线与数值门槛各不相同）。

| 平台 | 当前状态 | 后续动作 |
|---|---|---|
| 昆仑芯 P800/XPU3 | 本 PR 有单卡重建负载证据（整步 A/B 矩阵 + 组合验收）；整模型平台验收未完成 | 完成 x-benchmark、owner 审核与多卡/FSDP2 验收 |
| 昇腾 910C | 未实现、未验证（仅微基线数据存在于平台文档） | 单独建立 backend/fix 条目；确认 torch/npu 与算子注册路径 |
| 平头哥 ZW810E | 2026-11-12/13 首轮平台基线完成（issue #5，非 fix 验收）：判定"需优化"——FlagGems `29e14822` 全量 enable 18 项微基准 17 项慢于 native（1.3–6.4×），归因=host 侧 per-call enqueue 固定成本 ~55–69 µs（截距拟合 60.4/70.0 vs 4.7/4.6 µs，斜率与 native 相同）；gelu 族 126 用例精度矩阵 0 硬失败（42 DIFF=FTZ ±1–2 ulp + ±0 符号差）；"平台文档 13 算子 12 慢"口径与实测方向一致 | 证据：[validation/thead-zw810e-problem29](../../validation/thead-zw810e-problem29/README.md)；后续：优先评估 per-call host 开销路径与厂商 kernel/融合移交项，fix 条目与验收另立 |

`adamw-fused` 的 fused 选择已按评审加入 vendor 护栏：非 KunlunXIN 平台
默认 stock（显式请求也回退并记录原因），后续平台验证完成后再加入
`_FUSED_ADAMW_VALIDATED_VENDORS` 集合。
