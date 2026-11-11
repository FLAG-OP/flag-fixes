# complex dtype 反向崩溃修复（#7181 类：sum / zeros 族 / abs + conv2d constexpr + linear_backward view）

- ID：zeros-7181
- 状态：draft（deployable=false，未完成平台验收与 owner 审核）
- 负责人：待填写（owner 保持 null，不虚构）
- 上游 issue/PR：https://github.com/flagos-ai/FlagGems/issues/7181 （PR 未提交）
- 受影响算子、软件层、后端：`sum`、`zeros`/`zero_`/`zeros_like`、`abs`/`absolute`（算子层
  `src/flag_gems/ops/`）、`conv2d`（同层 constexpr 下标修复）、`linear_backward`
  （`src/flag_gems/runtime/backend/_thead/ops/`，thead 后端覆写）；本机以 cuda dispatch key 验证
  （FlagGems vendor 解析为 thead / ppu-smi）
- 问题与修复后行为：`flag_gems.enable()` 下跑 m2-bert 反向必抛
  `KeyError: 'complex64'`（triton `type_canonicalisation_dict` 无 complex 条目），
  issue #7181 的 `flag_gems.zeros(complex)` / `flag_gems.abs` 同签名；修复后反向完整跑完、
  前向 loss 与原生逐位一致（rel=0，单次模型前向、有日志支持）；除 complex 支持外，
  本 patch 亦修正 sum 的 dtype/out 推断、bool 转换与空输入语义——已列出的实数回归
  用例通过，未覆盖输入与后端不作逐位一致保证（第三轮评审口径）。
- 第四轮小模型评估（2026-11-10，评审意见 6756872311）：随机权重小模型四组对照
  （n / f_target / f_all / b）。仅注册目标算子的 f_target 9/9 全 PASS
  （fp32 max_abs≤4.8e-7、fp64≤8.9e-16，阈值先冻结、原生噪声门位级一致、FD 抽检
  PASS）；修复前对照树在修复目标路径崩溃（complex sum_dim、NHWC 非连续
  linear_backward view）证明修复真实命中；全量注册的两个失败均为预存缺口且不在
  本 patch 8 目标文件内（A：sgn/fill triton complex64 KeyError 与 fp64
  CompilationError；B：`mse_loss_backward` 不广播标量 grad_output 致静默错梯度，
  含最小复现）；scoped 性能首轮只报告不作加速声明。详见
  `reports/small-model-evaluation.md` 与 `results/small-model-2026-11-10/`。
- 第五轮评审修正（2026-11-10，意见 6766490204）：(1) bench 按族拆分注册+期望键
  断言+kernel 名命中验证（registered≠called）——modelB f_target 重测 2.42–2.93×
  （四轮 1.07× 系未注册目标算子，作废）；(2) 比较器结构加固+负例自测 10/10+已存
  .pt 重评分 f_target 9/9 维持；(3) 吞吐公式/P95 命名/显存口径/候选旗标（计数
  115 自动生成，旧 58 作废）/正确性失败配置标 diagnostic；(4) 报告三态重生成，
  另立修复项 `reports/follow-up-fix-items.md`，两缺口在修复前对照树逐位复现
  （预存归因有隔离对照支撑）。状态与验收口径不变。
- 第六轮评审修正（2026-11-10/11，意见 6784122575）：(1) 撤回旧清零性能结论
  （bench-r1 0.83–1.00×），性能节改由脚本从 bench2-summary.json 自动生成——
  实测 complex zeros/zero_ 2^20 达 97–290× 且随字节数线性增长；根因：complex
  `zero_`→`copy_(0)` 重派生**继承了 ATen complex-标量拷贝的全量张量可分页
  HtoD host 中转**（native `copy_(0)` 0.74–0.85ms ≈ f_target 0.90ms；native
  `zero_` 0.005ms 设备 memset）→ 登记 FIX-ITEM-3（候选方向：设备端 0 维源/
  real 视图清零，不承诺优化比例）；sum "大尺寸 1.1–2×" 同步撤回（实测
  3.95–22.92×）。(2) 命中说明三分法（registered/probe-hit/timed-case）并附
  profiler 全事件清单——zeros 族无 Triton kernel（此前"命中 Triton"系子串
  误判，更正）。(3) provenance 更正：test_small_models.py 在
  968d624..fac401c 间 226 行差异，设备运行在旧 harness、比较器随后加固并重
  评分已存 dumps；run-commits.json 机器可读清单。状态与验收口径不变。

## 一句话版本

**complex 张量被送进为实数设计的 Triton kernel，triton mangle 阶段取
`type_canonicalisation_dict['complex64']` 直接 KeyError；同一条反向链路还埋着两个预存
阻塞（conv2d 标量 `[None]` 的 constexpr 崩溃、thead linear_backward 的 view 崩溃）。
8 个文件一次修完：complex 求和走实数视图复用现有 kernel、complex `zero_` 一律
`x.copy_(0)` 原生 redispatch（stride 正确）、abs 模长用缩放 hypot（inf 优先 NaN、
防大数溢出），另两个是机械修复。**

```
修复前: KeyError: 'complex64' (triton/_utils.py type_canonicalisation_dict)
        -> 反向首崩于 src/flag_gems/ops/sum.py:385（problem.md live 同款）
修复后: 反向完成，loss rel=0.000e+00，回归 210/210，复现脚本 exit 0
```

## 问题是怎么回事

m2-bert 容器整模型（problem.md）在**反向**时，混合卷积（hyena filter）梯度触发
`aten.sum.dim_IntList(complex64)`，注册表路由 `("sum.dim_IntList", sum_dim) → sum_dim_comm
→ sum_dim_kernel_non_inner`，把 complex64 送进 Triton：

```
  File ".../src/flag_gems/ops/sum.py", line 385, in sum_dim
  File ".../src/flag_gems/ops/sum.py", line 306, in sum_dim_comm
    sum_dim_kernel_non_inner[grid](...)
  File ".../triton/_utils.py", line 133, in canonicalize_dtype
    return type_canonicalisation_dict[dtype_str]
KeyError: 'complex64'
```

triton 3.6.0 的 `type_canonicalisation_dict` 只登记实数 dtype；`zeros` 族与 `abs`（部分
入口/原地形式）是同一签名下的并行缺口——上游 issue #7181 把 zeros 与 abs 并列报告的原因。
前向对该模型是安全的（只经过 `div`/`mul`/`_fft_c2r`），所以故障只在反向暴露。

同一反向链路还埋着两个**预存前置阻塞**，不修则反向到不了/完不成 complex 验证：

1. `ops/conv2d.py:288`：`input_c_stride[None]`——标量 1 被 triton 特化成 `constexpr` 后
   做 `[None]` 直接 `AttributeError`；m2-bert 的 conv1d 输入来自 `rearrange` 转置
   （非连续，`stride(1)==1`），稳定触发。最小复现：`Conv1d(groups=C)` + 非连续输入。
2. `_thead/ops/linear_backward.py:170/171`：`view(M, K)` 对非连续激活抛
   `RuntimeError: view size is not compatible …`（与 linear-5379 同签名、不同文件）。

剥洋葱归属链（e2e 三树，同一模型同配置）：

```
基线 + linear-5379                -> linear_backward.py:170 view RuntimeError
+ conv2d/linear_backward 修复      -> sum.py:385 KeyError: 'complex64'（problem.md 签名）
+ complex 修复                     -> E2E OK
```

## 修复思路

一个实例、8 个文件，`patches/zeros-7181.patch`（494 行 / +267 −46）。按评审口径分
**三类改动**（同一反向链路的三个层次，故并入一个 fix；每类独立验收，详见
`reports/diagnosis.md`）：A 类 = complex dtype 支持本体（sum/zeros 族/abs，
验收 = KeyError 消失 + native oracle 一致）；B 类 = 同链路前置阻塞（conv2d、
linear_backward，验收 = 最小复现 + e2e 剥洋葱链）；C 类 = 后端/依赖边界
（linear.py 归 linear-5379，`requires` + `e2e_precondition` 钉死，验收 = e2e
起跑前 9 文件哈希不匹配 exit 2）。

| 文件 | 机制 |
| --- | --- |
| `ops/sum.py` | `_COMPLEX_TO_REAL` + `torch.view_as_real` 降维复用现有实数 kernel，结果按映射 dtype 转回；4 个入口前置分支；除 complex 支持外亦修正实数语义（cast-first、float→bool 非零转换、out dtype 推断、空输入归约——已列实数回归用例通过，未覆盖输入与后端不作逐位一致保证）；交叉 dtype out 语义对齐 ATen（`sum_dim_out(dtype=None)` 采用 `out.dtype`、显式不符逐字抛错、`sum` 系 cast-first） |
| `ops/zeros.py` / `ops/zero.py` | complex 一律 `x.copy_(0); return x`——ATen 原生 redispatch，**stride 正确**（切片/视图/转置的稀疏存储也被清零），保持 dtype 与"返回同一张量"语义 |
| `ops/zeros_like.py` | complex 分支调 `zero_(out)` 继承 strides/dtype/布局（`zeros_like` 语义天然正确） |
| `ops/abs.py` / `ops/absolute.py` | `abs()` complex 委托已有 `_absolute_complex_kernel` 的 `absolute()`；模长用**缩放 hypot**（`m=max(\|re\|,\|im\|)` 归一后开方，inf 优先于 NaN、NaN 传播、防大数溢出）；`abs_()`/`absolute_()` 按 native 原样抛 `RuntimeError`（逐字比对） |
| `ops/conv2d.py:288` | 标量直接参与广播，消除 constexpr `[None]`；文件内仅此一处 |
| `_thead/ops/linear_backward.py:170/171` | `view` → `reshape`（同 linear-5379 口径） |

**`ops/linear.py` 不在本 patch**（line 158 归已合入的 linear-5379）；`requires:
[linear-5379]` + `e2e_precondition`（9 文件 sha256）把依赖钉进 e2e 起跑前检查。
e2e 树 = 基线 + `linear-5379.patch` + 本 patch。

完整 diff 见 `patches/zeros-7181.patch`，前后文件 hash（sha256）：

| 文件 | before | after |
| --- | --- | --- |
| `src/flag_gems/ops/sum.py` | `5259b9f60445ef4a99e85387bc1e4c7a2264681170c695c6fc9b94378fa6348b` | `827500417ac340aa3b3d02e3d14a7307e15444bf668a3e9c3c6ccfd768d4bcdd` |
| `src/flag_gems/ops/zeros.py` | `ef6ee0ed2217edf2e49441ebd03ede9221b59d0a95e8ca2f7884dd28b64ae543` | `391893014d200a1bca9b167903517ea24e69a1187d3a0a223d9fb0ae9990f9f7` |
| `src/flag_gems/ops/zero.py` | `8f8d83c541e003274466f95f8817b46eb8df4d450b2068a51386a3e6bb278b10` | `1baf813d998fe4d4eaf522846bc862e21822014369a9eed347948b381a79d7fd` |
| `src/flag_gems/ops/zeros_like.py` | `14bc478e3a474ae5872503717704f2569d4ef2dc2f65a4aaf54a7dd64817fc2d` | `754892efc4dec014628890d4389e0b705ff8f04cc20f6bea4642b31684e09fb9` |
| `src/flag_gems/ops/abs.py` | `b7cf2a18414e159822c0c98c41aad61434184765aebc625b8f016eeee15fb930` | `9d473703db901f6ccc1cda134b7a95229b6fbec12656bd680eae48f14149ae0c` |
| `src/flag_gems/ops/absolute.py` | `fc3a44054ba94313dae0755b0d340db731a9feb3cb08ca0a05c6495735ab5b54` | `b9a9ce1981862e244817c16ef91e828e9c43beeddd5ace619d019f19afa441c2` |
| `src/flag_gems/ops/conv2d.py` | `290a77432d169331d594a59696d61e6b020ed65ce0e1d1a18ece3b669ff391e7` | `778519c06d988a99dae05075e5d15ed5556f9ce0f8540f635811666a7c35de24` |
| `src/flag_gems/runtime/backend/_thead/ops/linear_backward.py` | `f47b39ea0d62911eeb8601c613e2348e87852685119d0390ece26b437b915267` | `3fd7813424813de45ae0bf0e9aa8f0fb825601cb1b02948183e6dd48f348f413` |

## 怎么用：单项补丁安装

前置：FlagGems 基线 `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`git describe` =
`v5.3.5-46-g29e14822`），8 个目标文件 hash 与上表 before 一致。验证硬件为 PPU，vendor
解析为 `thead`，设备/dispatch 标识为 `cuda`：`cuda` 只是本环境设备标识，不代表已验证
所有 CUDA 厂商后端。本机环境与实际执行过的命令见 `reports/validation.md`。

本目录提供独立的 patch 和测试脚本，**不是独立 pip 包**。`draft` 表示验收材料尚未完整，
`deployable=false` 表示统一工具的 `apply` 暂不开放（实测 exit 2 正确拒绝）；这不妨碍在
匹配基线的测试工作树中手工应用补丁。

```bash
FIX_DIR=/path/to/flag-fixes/fixes/zeros-7181
TARGET=/path/to/FlagGems          # 通常还需先应用 linear-5379.patch（见下）
sha256sum "$TARGET/src/flag_gems/ops/sum.py"   # 核对上表 before hash
git -C "$TARGET" apply --check "$FIX_DIR/patches/zeros-7181.patch" && \
  git -C "$TARGET" apply "$FIX_DIR/patches/zeros-7181.patch"
```

注意：本机 m2-bert e2e 树 = 基线 + `fixes/linear-5379/patches/linear-5379.patch` +
本 patch（两个 patch 改动文件不相交，顺序任意）。验证必须导入修改后的目标源码树：
`PYTHONPATH="$TARGET/src"` 并启动新进程；若仍导入 site-packages 中的 FlagGems，本补丁不生效。

管理工具完整预演（`--root` 永远指向管理仓库根）。**注意 `requires: [linear-5379]`
非空后，select 类命令必须同时给出两个 ID**（只给 zeros-7181 会 exit 2
`include required fixes`）：

```bash
MANAGER=/path/to/flag-fixes
TARGET=/path/to/FlagGems   # 纯基线

PYTHONPATH="$MANAGER/src" python3 -m flag_fixes --root "$MANAGER" \
  check linear-5379 zeros-7181 --target "$TARGET"   # exit 0：before hash + git apply --check
PYTHONPATH="$MANAGER/src" python3 -m flag_fixes --root "$MANAGER" \
  plan linear-5379 zeros-7181 --target "$TARGET"    # exit 0：预演不落盘
# apply 会因 deployable=false 返回 exit 2（按设计），需手工 git apply
PYTHONPATH="$TARGET/src" python3 "$MANAGER/fixes/zeros-7181/tests/repro_complex64.py"
PYTHONPATH="$TARGET/src" python3 "$MANAGER/fixes/zeros-7181/tests/test_complex_ops.py"
PYTHONPATH="$TARGET/src" python3 "$MANAGER/fixes/zeros-7181/tests/test_m2_bert_backward.py"  # 需 transformers
```

三个脚本只有退出码 0 才算通过；缺设备或缺 `transformers` 时退出码 2（skip 不算通过）。
GPU 脚本需 `CUDA_VISIBLE_DEVICES=0`（本环境不设时设备枚举挂起，见 validation.md）。
或由管理工具跑（要求目标已是修复后内容；e2e 内另有 9 文件前置哈希检查）：

```bash
PYTHONPATH="$MANAGER/src" python3 -m flag_fixes --root "$MANAGER" \
  verify zeros-7181 --target "$TARGET" --python python3   # 输出 JSON 见 results/
```

撤销手工补丁：

```bash
git -C "$TARGET" apply --reverse --check "$MANAGER/fixes/zeros-7181/patches/zeros-7181.patch" && \
  git -C "$TARGET" apply --reverse "$MANAGER/fixes/zeros-7181/patches/zeros-7181.patch"
```

容器侧整模型回归（problem.md 要求的独立 result-root，避免命中已有 on-Success 缓存）。
以下路径是目标容器内的固定路径，按 problem.md 原样给出，不是本机路径：

```bash
cd /home/wuyiqiang/x-benchmark && HF_HOME=/home/wuyiqiang/hf_cache \
PYTHONPATH=/data/ppu-benchmark/FlagGems/src \
python3 run_model.py --models "togethercomputer/m2-bert-80M-8k-retrieval" \
  --models-json assets/models_info_20260813_073927_top515_new317.json \
  --envs-dir /data/ppu-benchmark/venvs --device cuda:0 \
  --result-root /data/ppu-benchmark/results_top515_repro
```
（该步依赖目标容器，本环境未执行，见 `reports/validation.md` 的未执行清单。）

## 目录结构

```
├── README.md                     # 本文
├── fix.yaml                      # 清单（schema_version 1，8 目标 + requires/
│                                 #   e2e_precondition 9 文件 + 3 硬件命令）
├── WORKFLOW.md                   # 8 步执行清单及证据
├── patches/zeros-7181.patch      # 最小 diff（8 文件，494 行 / +267 −46），git apply 用
├── src/flag_gems/...             # 修复后完整文件快照（8 个），与 patch 结果逐字节一致
├── tests/
│   ├── repro_complex64.py        #   issue #7181 / problem.md 最小复现（level: repro）
│   ├── test_complex_ops.py       #   回归 210 用例：基础矩阵 + strided + 模长 +
│   │                             #   交叉 dtype out 矩阵 + autograd parity +
│   │                             #   bool/complex→bool/空输入/conj（level: regression）
│   ├── test_m2_bert_backward.py  #   m2-bert 前向+反向（level: e2e，需 transformers，
│   │                             #   含 9 文件前置哈希 + revision pin + 缺口 fixture 断言）
│   ├── known_gap_m2.txt          #   filter_fn 已知缺口精确集合（32 模板 × 12 = 384）
│   └── _stubs/opt_einsum.py      #   缺依赖时的 import 兜底（仅当宿主无 opt_einsum 生效）
├── reports/
│   ├── diagnosis.md              #   根因、方案、影响范围、预存缺陷双树证据
│   └── validation.md             #   环境、命令、前后结果、未执行清单
└── results/                      # 权威日志为 *-2026-11-09.log（评审第二轮）：
    ├── repro-{baseline,fixed}-2026-11-09.log       # 复现 exit 1 / exit 0
    ├── regression-{baseline,fixed}-2026-11-09.log  # 33/177/210 → 210/0/210
    ├── e2e-fixed-2026-11-09.log                    # E2E OK（precondition/fixture/revision）
    ├── e2e-precondition-{baseline,linearonly}-2026-11-09.log  # 前置哈希 exit 2 实测
    ├── e2e-missing-dep-2026-11-09.log / nocuda-regression-2026-11-09.log  # exit 2
    ├── cli-{validate,check,plan,apply,status,check-requiredfail}-2026-11-09.log
    ├── verify-2026-11-09.json                      # 第二轮提交后干净 verify
    ├── *-2026-11-08.log / verify-2026-11-08.json   # 第一轮历史（139 用例口径，
    │                                               #   提交 085e199/9fa84cf，不改写）
    ├── *-2026-11-07.log                            # 初版历史原文（不改写；
    │                                               #   其 regression TOTAL 行有计数器
    │                                               #   缺陷，实际 16/38/54，见
    │                                               #   validation.md 注）
    ├── e2e-{linearblocker-baseline,problem-path-noncomplex}-2026-11-07.log
    ├── gradgap-probe-{noncomplex,fixed}-2026-11-07.log  # 预存梯度缺口双树对照
    └── verify-2026-11-07{,-dirty}.json              # 初版 verify（passed 3/3）
```

## 验证结果摘要

（2026-11-10 评审第三轮：回归日志见 `results/regression-*-2026-11-10.log`、干净
verify 见 `results/verify-2026-11-10.json`；其余命令日志为第二轮
`results/*-2026-11-09.log`；第一轮 139 用例口径见
`results/*-2026-11-08.log`）

| 项目 | 修复前 | 修复后 |
| --- | --- | --- |
| `repro_complex64.py` | exit 1，`ISSUE REPRODUCED` + `KeyError: 'complex64'` | exit 0 |
| `test_complex_ops.py`（210 用例） | 33 passed / 177 failed / 210 total | 210 passed / 0 failed / 210 total |
| `test_m2_bert_backward.py`（e2e） | 阶段一：linear_backward `view` RuntimeError；阶段二：反向 `sum.py:385 KeyError: 'complex64'`（problem.md 签名） | exit 0，`E2E OK`，precondition 9 文件 OK，loss rel=0.000e+00，161 梯度存在且有限，384 缺口与 fixture 双向断言一致 |
| e2e 前置哈希（基线树 / 仅 linear 树） | — | 均 exit 2 `MISSING PRECONDITION`（模型工作之前） |
| 前向对照 | （阶段二）loss rel=0.0 | loss rel=0.0（与原生逐位一致） |
| 实数路径回归 | 通过 | 通过（已列用例；未覆盖输入与后端不作逐位一致保证） |
| `flag-fixes validate/check/plan` | — | exit 0（9 文件 hash、patch 应用、快照一致；须同时给两个 ID） |
| `flag-fixes check zeros-7181`（漏依赖） | — | exit 2 `include required fixes: ['linear-5379']` |
| `flag-fixes apply` | — | exit 2（`deployable=false` 按设计拒绝） |
| `flag-fixes verify` | — | `passed: true`，3/3 命令 exit 0（11-07 两份 JSON；11-08 提交后重跑见 results/） |
| 缺 transformers / 缺设备 | — | 退出码 2（`MISSING …`，skip 不算通过） |

判定与覆盖边界：三个硬件脚本只有退出码 0 才算通过。普通 CI（无设备 runner）执行
`flag-fixes validate`（schema、patch hash、快照 hash、脚本存在性）、`python -m pytest -q`
和 `python -m compileall -q fixes`，**不**执行这三个硬件脚本；语法检查通过 ≠ 入口执行
通过 ≠ 硬件验证通过。

## 已知边界（不装完美）

- **平台验收未做**：`hardware_status` 保持 `unverified`，problem.md 要求的 x-benchmark
  容器 on 趟回归（独立 `--result-root`）未在本环境执行；`deployable=false`、
  `owner=null`，`flag-fixes apply` 会按设计拒绝。
- **预存缺陷（双树同形态，非本 patch 引入，见 reports/diagnosis.md 与
  results/gradgap-probe-*.log）**：flag_gems 全包 0 个 `register_autograd`，
  `aten::pad`/`aten::conv1d` 回退导致 hyena `filter_fn.*` 梯度缺失（12L 精确集合
  384 条固化为 `tests/known_gap_m2.txt`，e2e 双向断言；1L 同形态 32+2）；
  近驻点参数 native-vs-flag_gems 梯度数值分歧；个别 run 中 loss 张量存储被间歇
  覆写——**归因未定**（既未证明预存、也未证明本修复引入；e2e 因此在反向前快照 loss）。
- **未修的 complex 缺口**：`mean`/`full`/`ones`、`eq`/`where` 等
  `ComplexMode=NONE`/未覆写入口不在本 patch（m2-bert 反向不需要）；complex 比较请用
  `torch.view_as_real`。
- 只修 `_thead` 后端的 linear_backward 覆写（本机实测文件）；其他厂商后端的同源
  `view` 按「不同后端分条」约定未包含，跟踪在上游 issue #5379。
- 没有性能基准：崩溃修复，反向修复前跑不完、无同口径耗时；`benchmarks/` 保持模板状态。
- **e2e 依赖**：`test_m2_bert_backward.py` 需要 `transformers`，缺失时退出码 2 并打印
  `MISSING DEPENDENCY`，不会因为前两个脚本通过而返回 0；`opt_einsum` 缺失时使用
  `tests/_stubs/opt_einsum.py` 兜底（host 侧 import 需要，非 FlagGems 依赖），运行时
  打印 `opt_einsum=bundled-stub|real`；模型代码 revision 打印并比对 pin `90e0b28f…`。
