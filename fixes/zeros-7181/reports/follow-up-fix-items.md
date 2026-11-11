# 另立修复项（第四轮全量组暴露问题的跟进清单）

评审要求（comment `6766490204` §推进顺序 4）：全量组暴露的问题另立修复项，
优先 `mse_loss_backward` 标量 grad_output 广播（静默错梯度），其次 complex
pointwise/sgn/fill 与 FP64 编译；每项保留最小复现、原生对照与修复前后证据；
预存归因须关联同路径基线或隔离对照，不得仅以"文件不在本 patch 内"作为全部
因果归属证明。本页逐项给出复现脚本、两树隔离对照结果与当前状态
（均未实现修复，属后续独立修复包的工作范围）。

## FIX-ITEM-1（优先）：`mse_loss_backward` 不广播 0 维标量 grad_output

- **症状**：全量注册下对标量 loss 反向传播时，`aten::mse_loss_backward` 收到
  autograd 传入的 0 维标量上游梯度；FlagGems kernel 按元素索引读取该 1 元素
  张量，第 0 元素精确、其余 39/40 元素读到越界垃圾 → **静默错梯度**
  （小模型 B 在 f_all 下 190/192 项超差、rel_l2≈1.0，训练发散；前向精确）。
- **最小复现**：`tests/repro_mse_scalar_grad.py`（8×5 张量，三种 reduction 对照）。
  - 全量张量 grad_output + mean：**精确**（max_abs=0）
  - 标量 grad_output + mean：**39/40 错**（第 0 元素精确）
  - reduction none/sum：精确
- **原生对照 / 隔离对照（预存归因）**：
  - 修复树（`flagfix-fixed`）：39/40 错 → `matrix-rescored/repro-mse-scalar-fixed.log`（四轮证据 `mse-scalar-repro.log` 同）
  - **修复前对照树（`flagfix-e2ebase`）：39/40 错，逐位同形态** → `matrix-rescored/repro-mse-scalar-btree.log` —— 同路径基线复现，预存性由隔离对照直接证明
  - `ops/mse_loss_backward.py` 两树逐字节相同（`diff` 为空）
- **影响面**：任何使用标量 loss（mean/sum reduction MSE）+ 全量注册的训练反向。
- **状态**：proposed（未实现）。建议修法：wrapper 内对 0 维 grad_output
  `expand_as/expand` 至 self 形状后再进 kernel（或 kernel 内 stride-0 广播读）。
  修复前后证据在独立修复包中补齐。

## FIX-ITEM-2：complex 组装张量的 pointwise/fill 路径（empty_like / conj / sgn / zeros / to_copy）+ FP64 编译

- **症状**：triton `canonicalize_dtype` 无 complex 条目，经 `fill_scalar_`
  的路径对 complex64/128 直接 KeyError；触发面包括：
  - `torch.empty_like`、`conj()`（`_conj.py:69`）、`sgn`（`sgn.py:149`）作用于
    **`torch.complex(p, q)` 组装张量**（cast 张量 `randn().to(c64)` 不触发）
  - `torch.zeros(complex, ...)` 的尺寸相关分支（bench 逐用例记录：8×8 探针 OK、
    32×32 及以上 KeyError）
  - `.to(complex)`（`to_copy` → fill）在注册后执行会中止进程（bench 探针张量
    因此改为注册前构建）
  - fp64/c128：triton CompilationError（loop-carried accumulator fp32→fp64）
- **最小复现**：`tests/repro_complex_fill.py`（5 个探针）。
  - 修复树：5/5 KeyError → `matrix-rescored/repro-complex-fill-fixed.log`
  - **修复前对照树：5/5 KeyError（`abs(z)` 前向亦崩，即修复前该路径完全不可用）**
    → `matrix-rescored/repro-complex-fill-btree.log` —— 同路径基线复现
- **与四轮 f_all-A 崩溃的关系**：矩阵中 f_all-A 的 traceback（`sgn.py:149` →
  `fill.py:103`）与本复现同节点；abs 反向经 sgn→empty_like 触发，即
  `abs(sum(z)) backward` 探针（两树均崩）。
- **影响面**：全量注册下任何经组装 complex 张量的 pointwise/创建路径；
  f_all 模型 A（复数核）整体不可跑。
- **状态**：proposed（未实现）。建议方向：pointwise_dynamic/fill 增加 complex
  显式拒绝并回退原生（同 zeros-7181 对 zeros/zero_ 的处理模式），或补齐 triton
  complex 指针支持；FP64 需 triton 侧累加器 dtype 修正。修复前后证据在独立
  修复包中补齐。

## FIX-ITEM-3：complex zeros/zero_ 重派生路径的性能（round-6 新立，round-7 对照加固）

- **症状**：f_target 下 complex `zero_`/`zeros` 开销随张量字节数线性增长
  （~9–14 GB/s 主机中转特征；2^20 c64/c128 达 **233×/290×**（`zero_`）、
  **97×/175×**（`zeros`）；2^10 仅 2.7–4.7×）——bench-r2 实测，第六轮评审指出
  旧"接近原生"结论与之矛盾，已撤回并修正报告。
- **机制（已根因定位，round-7 修正探针后确认）**：修复为保 stride 正确性将
  complex `zero_` 重派生到 `x.copy_(0)`、`zeros` 到 `empty + zero_`；而
  **native ATen 的 complex-标量 copy 本身经全量张量可分页 HtoD 中转**。
  归因实验（`benchmarks/zero_staging_probe.py`，原生无注册，2^20；
  `zero-staging-attribution.log`；round-7 按评审要求把设备标量**预建在计时
  之外**，并单列构造 / 拷贝净成本 / memset / 同步边界）：

  | 路径（2^20，原生无注册） | c64 (8MB) | c128 (16MB) |
  |---|---:|---:|
  | `copy_(0)` python 标量（修复继承路径） | 0.8276 ms | 1.4955 ms |
  | `copy_(cpu 0-dim)` | 0.8150 ms | - |
  | **`copy_(预建 device 0-dim)`（拷贝净成本）** | **0.0071 ms（round-7 批）/ 0.0089 ms（round-8 批）** | **0.0069 ms（round-7 批）/ 0.0067 ms（round-8 批）** |
  | device 标量构造 | 单次 0.3358 ms（含冷启动）/ 50 次平均 0.0145 ms | - |
  | `zero_()` 设备 memset | 0.0042/0.0043 ms | 0.0055/0.0057 ms |
  | `torch.zeros(c64)` 创建参照 | 0.0079 ms | - |

  **round-9 批次口径**：两批 copy-only（round-7 探针 0.0071/0.0069，round-8
  探针 0.0089/0.0067）来自不同运行批次，属计时波动与复测差异，非功能差异；
  批/设备/计时协议 provenance 见
  `zero-staging-attribution-r7-batch.log`（round-7 归档）与
  `zero-staging-attribution.log`（round-8），汇总引用时须标明批次来源。

  round-5/6 探针的 0.02 ms 混入了每次 `.cuda()` 构造成本，已按修正探针更正；
  host staging 与设备路径净成本差距 ≈ **117×（c64）**。profiler 全事件
  （`benchmarks/zeros_hitprobe.py` → `zeros-hitprobe-{f_target,native}.json`）：
  f_target 下 `zero_` 仅 `Memcpy HtoD (Pageable)`、`zeros` 为 native
  FillFunctor + HtoD，**无 Triton kernel**（round-5 回复中"zeros 命中 Triton"
  系子串误判，已更正）；native 下 `zero_` 仅 3 µs memset、无 memcpy。
- **结论**：开销主体是被 `copy_(0)` 重派生继承的 **ATen complex-标量拷贝
  host 中转路径**，非 FlagGems 额外计算；正确性不受影响（stride 正确是
  round-2 选择该路径的原因）。定位方向获评审确认（"native complex scalar
  copy 存在 host staging"），具体数值以 round-7 修正探针为准。
- **三条候选路径（round-7 评审框定，独立跟踪）**：
  1. `zero_()` 原生设备 memset（native 基准）；
  2. 预存设备 0 维 complex 标量的 `copy_`（round-7 修正探针净成本 0.007 ms 级）；
  3. candidate real-view/逐元素设备 kernel。
  三者实现时都必须通过同一**正确性矩阵**：contiguous、slice/transpose 非连续
  视图、storage offset、complex64/complex128，并检查**未选中 storage 不被
  污染**；候选实现只有在正确性回归、非连续视图与大尺寸性能全部通过后，才
  考虑独立建优化 fix。
- **候选评估协议（round-8 评审补充）**：每个候选方案同时记录——正确性、
  非连续视图行为、大尺寸耗时、内存变化。
  ~~优化实现完成后，分别在**昆仑芯、昇腾、平头哥**三类平台复跑同一矩阵，
  避免把 PPU/thead 上的归因直接外推到其他后端。~~【**round-9 评审更正并撤回**：
  三平台复验属 PR3 problem29 性能 bundle 范围，**PR2 不要求**；FIX-ITEM-3
  若后续扩大到其它后端，应另建平台适配/优化条目再分别定义验证范围】
- **round-10 候选筛选已执行**（`benchmarks/zero_candidates.py`，原生 harness、
  不改动本 patch）：正确性矩阵 **60/60 全过**（6 布局[contig/slice-stride/
  narrow/transpose/storage-offset/conj] × c64/c128 × 5 实现[当前/A/B/C/
  native]，含未选中 storage 位级保持与 native 逐位一致检查；候选 A 的 conj
  回退路径在首轮暴露并修正）。2^20 性能（ms/iter，c64 contig）：

  | 实现 | c64 contig | c64 slice | c128 contig | 结论 |
  |---|---:|---:|---:|---|
  | 当前（copy_(0)） | 0.6674 | 1.1078 | 1.7097 | host staging |
  | **A uint8-view memset** | 0.0083 | 0.0071 | 0.0082 | 快，仅限全 storage 连续 |
  | **B 预建设备标量 copy** | **0.0050** | **0.0056** | 0.0116 | **最优，全布局** |
  | C real/imag 双通道 | 0.0154 | 0.0163 | 0.0151 | 正确但最慢（两遍） |
  | native zero_ 基准 | 0.0042 | 0.0054 | 0.0053 | — |

  筛选结论：**候选 B 全布局最优且矩阵全过**；A 可作 contiguous 全 storage
  快路径；C 正确但无双遍优势。证据 `bench-r2/zero-candidates.{json,log}`。
  独立优化 fix 的实现建议按评审顺序 A→B→C 落地（可组合：A 快路径 + B 兜底），
  通过第"验收门槛"节全部门槛后另行提交；实现时每候选仍须按 round-8 协议
  记录正确性、非连续视图行为、大尺寸耗时与显存变化。
- **round-11 扩展与注册环境复测（评审 6788210724 §1/§2/§4）**：
  - 矩阵扩展后 **260/260 全过**（原生 harness）：新增 zeros 创建路径、
    empty/0-dim、组合布局（offset×transpose×slice×conj 五种组合）、精确
    storage 索引映射的 alias/未选中检查；负 stride 框架不支持，登记 N/A。
    候选 A 新增两条实现约束（0 维与 conj 必须 fallback B）。
  - **注册环境复测**（`--registered`，zeros 族 only_enable）：A/B 的
    profiler 事件均为 **native-aten、无 HtoD-staging**，耗时与原生 harness
    一致（0.0051–0.0075 ms）——A 未被 FlagGems fill 污染、B 未重引
    host staging ✓。
  - **候选 C 在注册环境被淘汰**：`select(-1,i).zero_()` 命中注册的
    FlagGems `zero_`，对**实数跨步视图按稠密写**——矩阵 C 全部失败
    （all_zero=False、未选中区被污染），根因即 **FIX-ITEM-5**（round-2 弃用
    view_as_real 的同一缺陷模式在实数路径复现）。C 仅在原生环境正确。
  - **B-only vs A+B 组合**（评审 §4）：A 即"A 快路径 + B 兜底"组合。
    实测：注册环境下 **B 占优或持平全部已测布局**（contig c64 0.0051 vs
    A 0.0063、c128 0.0051 vs A 0.0075；slice 两者持平 ~0.0056）；原生
    harness 中仅 c128 contig A 略优（0.0082 vs 0.0116）；guard 分支为纯
    host 判断（µs 级，已实测）。**B-only 已接近 native（0.005–0.007 ms）
    且复杂度最低——建议独立优化 fix 首选 B-only**，A 快路径作为可选增强
    （收益仅 c128 contiguous 一处、量级 ~3µs）。
  - "B 全布局最优"表述已按评审要求限定为"**在本轮已测试布局中最优**"。
  - 证据：`bench-r2/zero-candidates-r11-{native,registered}.{json,log}`。
- **状态**：proposed（未实现）。修复前后证据在独立修复包轮次补齐。

## FIX-ITEM-4：FlagGems `sum` 对 uint8 的累加器回绕与输出 dtype 错误（round-10 探索中发现）

- **症状**：注册 FlagGems `sum` 后，`uint8` 张量求和返回 dtype **uint8**
  （native 为 **int64**）且数值按 256 回绕：`[200,100,50,7].sum()` = **101**
  （期望 357，= 357 mod 256）；万级元素求和同样回绕。
- **最小复现**：`u8 = torch.tensor([200,100,50,7], dtype=torch.uint8,
  device='cuda'); u8.sum()` → 注册后 `(101, torch.uint8)` vs native
  `(357, torch.int64)`。
- **发现路径**：round-10 基线冻结的 checksum 辅助函数在注册组出现
  byte_sum 回绕（28173→255），定位为 uint8 sum 累加器问题（探针 artifact
  已改用 `sum(dtype=torch.int64)` 规避）。
- **预存归因（round-12 修正探针后闭环）**：专项复现脚本
  `benchmarks/u8sum_repro.py` 按评审清单（1d/多维、默认/显式 dtype、>255
  累加、empty、不同 reduction 维、连续/非连续）双树对照。**round-12 修正**：
  原脚本 4 元素样例未指定 dtype（实为 int64），"4 元素小例侥幸正确"与尺寸
  分支归因**撤回**；非连续用例改为先建 uint8 base 再切片。修正后（真
  uint8）：**修复树 7/8 失败、修复前对照树同型 7/8 失败**（仅显式
  dtype=int64 一例正确；连 4 元素小例也回绕 357→101），失败集合逐项一致 →
  预存归因由同路径基线直接证明。日志
  `bench-r2/u8sum-repro-{fix,prefix}-final.log`。checksum 回绕仅作发现
  线索（已按评审要求不以之为归因证据）。输入 dtype/stride 与输出值/dtype
  已按评审要求记录（results JSON `_inputs` 行）。
- **影响面**：所有经 FlagGems sum 的 uint8 归约（大值/多元素必回绕；
  4 元素小例路径侥幸正确）。
- **状态**：proposed（未实现）。修复方向：uint8/int8 等小整型归约按 ATen
  提升为 int64 累加与输出。

## FIX-ITEM-5：FlagGems `zero_` 对实数跨步视图稠密写（round-11 候选复测中发现）

- **症状**：注册 FlagGems `zero_` 后，对**实数（float）跨步视图**调用
  `zero_` 按**稠密**清零——选中元素与未选中元素全部变 0，未选中 storage 被
  污染：`buf[:, ::2].zero_()` 把 8 列全清零（应只清 4 列）。native ATen
  stride 正确。
- **最小复现**：`benchmarks/strided_zero_probe.py`（4×8 float、`[:, ::2]`
  视图，选中/未选中断言）——日志 `bench-r2/strided-zero-{fix,prefix}.log`。
- **预存归因**：修复前对照树**同型污染**（逐位一致）——预存，非本 patch
  引入。与 round-2 在 complex 路径弃用 view_as_real 的缺陷模式同源
  （稠密写不识别 stride），但此处为**实数路径**，属 FlagGems `zero_`
  自身 kernel 的跨步处理缺陷。
- **关联**：这正是 round-11 候选 C 在注册环境被淘汰的根因（C 经
  `view_as_real().select(-1,i).zero_()` 会命中此缺陷）；也意味着任何在
  注册环境下对跨步实数张量的 `zero_` 都会损坏数据（影响面大）。
- **状态**：proposed（未实现）。修复方向：FlagGems `zero_` kernel 按
  stride 寻址（或跨步输入回退原生）。

## round-10 探索计划登记（评审 §三：sum/abs 性能探索）

作为独立后续工作（不在本 patch 范围）：
- `sum`：real/imag 分开累加、减少 complex 拆装与中间张量；inner/outer/full/
  strided 四类分别分析；forward/backward/fwd+bwd 分开测；
- `abs`：保留 scaled-hypot，测试寄存器压力/向量化/不同 shape 吞吐；
- 数值检查扩展：NaN、Inf、极大/极小、空输入、conj 输入（round-10 基线
  `baseline/` 已覆盖首trap：见 `baseline-freeze-*.json` 的 edge_numeric 节）；
- 每候选记录 max_abs、相对误差、梯度误差与 kernel 数量。

## round-10 基线冻结（评审 §一）已完成

- 产物：`results/small-model-2026-11-10/baseline/`
  （`baseline-{n,fix,prefix}.json`、`baseline-summary.json`、
  `baseline-table.md`；脚本 `benchmarks/baseline_freeze.py`）。
- 覆盖：zeros/zero_ × {contig, slice-stride, narrow, transpose,
  storage-offset} × c64/c128 × 三路径；conj/empty/0-dim 边缘；
  sum/abs 的 NaN/Inf/±Inf/1e38/1e-38/empty/conj 数值边缘（位级 checksum
  交叉对比）。
- 结果：fix 与 native 在 52/58 用例**位级一致**；6 例为既定声明内的 1ulp 级
  差异（conj 归约求和顺序 ×4、abs scaled-hypot 舍入 ×2——abs 值差异见
  `abs(1e-38)` 1.11803383 vs 1.11803397，第 8 位有效数字）；**修复前对照树
  58/58 全崩**（零路径与边缘数值全部不可执行）；prefix 崩溃清单逐例记录。
- provenance：每 JSON 含 group/tree/torch/device/determinism/注册键样本；
  运行提交见 run-commits.json（round-10 追加）。

## 验收门槛（评审 §五，优化候选转正式 fix 前必须全过）

1. 当前 210 用例回归；
2. zero view/storage 矩阵（zero_candidates.py 60 项）;
3. complex64/complex128 数值和梯度对照；
4. 小模型前向、反向和短训练步；
5. 目标后端完整模型容器验收；
6. 性能重复测量和 provenance 检查。
随机权重小模型只证明局部数值一致，不替代完整模型验证；PR2 维持
`draft / deployable=false / hardware_status=unverified`。

## 跟进原则

- 两项都不在本 patch 的 8 个目标文件内；本页仅登记问题与预存证据，不据此
  扩大 zeros-7181 的声明范围。
- 实际修复时各自独立建包（fix.yaml + patch + 回归 + 修复前后日志），并沿用
  "错树前置 / 缺依赖 exit 2 / 证据单独提交" 的既有流程。
- 原始运行证据（四轮矩阵/bench 与 traceback）保留在
  `results/small-model-2026-11-10/`，不改写。


## round-13 补充（评审 6794976151）

- FIX-ITEM-3 的 ULP 量化按 dtype 修正后定稿：conj sum/abs ≤4 float64 ULP
  （c128）、≤2 float32 ULP（c64）；abs scaled-hypot 1 ULP 级（c64/c128）；
  NaN 数值全同、payload 位不保证。详见 `baseline/diff-details.json` 与
  `baseline/ulp-mapping-selftest.json`。
- B-only 首选结论不变；"本轮已测试布局中最优"限定语维持。


## round-14 补充（评审 6797761554）

- 差异明细工具 `diff_details.py` 重写：NaN 用例与基线构造同源（NaN+NaN·j）、
  位级 payload 比较、分类完备（类不一致=失败）、±0 策略显式、有序 ULP 映射
  精确断言。FIX-ITEM-3 相关的 8 项 numeric_diff 定稿：conj ≤4 float64 ULP、
  abs ≤2 float32 ULP / ≤2 float64 ULP、abs extreme 1 ULP、abs/c128/nan =
  输出 NaN payload 位型差异（输入位级一致，掩码一致，策略不要求 payload）。
- FIX-ITEM-4/5 不受本工具修正影响，维持独立跟踪。
