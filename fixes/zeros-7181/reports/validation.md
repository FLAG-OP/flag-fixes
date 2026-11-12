# 验证记录

- 修复 ID：`zeros-7181`；验证日期：2026-11-07（初版）→ **2026-11-08（评审修正轮复跑）**，
  本机 PPU 环境实测。下表凡 2026-11-08 的日志为**当前代码**（评审修正后）的权威证据；
  `-2026-11-07` 日志为历史原文，不改写（其 regression `TOTAL` 行有计数器缺陷，见下注）。
- `hardware_status: unverified`：problem.md 指定的 x-benchmark 容器整模型 on 趟回归未在
  本环境执行（见文末「未执行」），下述全部为本地双树对照证据。
- 三个硬件脚本只有退出码 0 才算通过；缺设备 / 缺 `transformers` 一律退出码 2，
  **skip 不算通过**。

## 环境与源码

| 项 | 值 |
| --- | --- |
| FlagGems 基线 | commit `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`git describe` = `v5.3.5-46-g29e14822`） |
| e2e 模型代码 revision | `90e0b28f3382c289cc6c1e92ef53d7dc5a3ec14b`（`trust_remote_code` 动态加载，e2e 打印 `model code: commit=…` 并与 pin 比对） |
| 硬件 / vendor | PPU-ZW810E，FlagGems vendor 解析为 `thead`，设备/dispatch 标识为 `cuda`（`cuda` 只是本环境设备标识，不代表已验证 CUDA 厂商后端） |
| 软件栈 | torch 2.10.0 / triton 3.6.0 / Python 3.12.3 / transformers 4.57.0 / PyYAML 6.0.2 / 驱动 `ppu-smi` |
| `opt_einsum` | 宿主缺失；e2e 用 `tests/_stubs/opt_einsum.py` 兜底并打印 `opt_einsum=bundled-stub`（非 flag_gems 依赖，仅 host import） |
| 模型 | `togethercomputer/m2-bert-80M-8k-retrieval`（`trust_remote_code`，随机初始化，不下载权重，`HF_ENDPOINT=https://hf-mirror.com`） |
| 管理仓库 | `/root/flag-fixes`，分支 `zeros-7181`；11-08 复跑时 HEAD 待提交（验证记录随后以 verify JSON 的 `commit` 字段钉死） |
| 运行条件 | 所有 GPU 脚本必须 `CUDA_VISIBLE_DEVICES=0`：本环境不设置该变量时 `torch.cuda.is_available()` 枚举挂起（已知怪癖，`device_count()` 不受影响）；统一 `timeout -k 10` 兜底 |

肇事堆栈（problem.md live，修复前实测同款）：

```
  File ".../src/flag_gems/ops/sum.py", line 385, in sum_dim
    return sum_dim_comm(inp, dim, keepdim, dtype=dtype)
  File ".../src/flag_gems/ops/sum.py", line 306, in sum_dim_comm
    sum_dim_kernel_non_inner[grid](...)
  File ".../triton/_utils.py", line 133, in canonicalize_dtype
    return type_canonicalisation_dict[dtype_str]
KeyError: 'complex64'
```

## 目标树与构造方法

四个工作树，均为干净基线的 `cp -a` 副本（`git status` 差异文件数 = 下表改动数）：

| 树 | 内容 | 用途 |
| --- | --- | --- |
| `/tmp/opencode/FlagGems-base` | 纯基线 `29e14822` | `check`/`plan` 的 hash 预演目标 |
| `/tmp/opencode/flagfix-e2ebase` | 基线 + `linear-5379.patch`（1 文件） | e2e「修复前-阶段一」 |
| `/tmp/opencode/flagfix-noncplx` | e2ebase + 本 patch 的 conv2d + linear_backward 两个 hunk（回退 6 个 complex 文件，3 文件） | e2e「修复前-阶段二」= problem.md 签名树 |
| `/tmp/opencode/flagfix-fixed` | e2ebase + `zeros-7181.patch` 全量（9 文件含 linear.py 依赖） | e2e 修复后（11-09 用**第二轮评审最终 patch** 重建） |

```bash
BASE=/tmp/opencode/FlagGems-base
FIX=/root/flag-fixes/fixes/zeros-7181
LIN=/root/flag-fixes/fixes/linear-5379

cp -a $BASE /tmp/opencode/flagfix-e2ebase
git -C /tmp/opencode/flagfix-e2ebase apply $LIN/patches/linear-5379.patch

cp -a /tmp/opencode/flagfix-e2ebase /tmp/opencode/flagfix-noncplx
git -C /tmp/opencode/flagfix-noncplx apply $FIX/patches/zeros-7181.patch
git -C /tmp/opencode/flagfix-noncplx checkout -- \
  src/flag_gems/ops/sum.py src/flag_gems/ops/zeros.py src/flag_gems/ops/zero.py \
  src/flag_gems/ops/zeros_like.py src/flag_gems/ops/abs.py src/flag_gems/ops/absolute.py

cp -a /tmp/opencode/flagfix-e2ebase /tmp/opencode/flagfix-fixed
git -C /tmp/opencode/flagfix-fixed apply $FIX/patches/zeros-7181.patch
```

按上式重建的两棵树与在用树 `diff -rq` 仅 `.git` 元数据不同、工作区文件逐字节一致。
`flagfix-fixed` 的 8 个目标文件 sha256 与 `fix.yaml` after hash 逐文件复核一致，
且与 `fixes/zeros-7181/src/` 快照逐字节 `cmp` 通过；`ops/linear.py` 亦等于
linear-5379 的 after hash `ddec3383…`（9 文件与 `e2e_precondition` 一致）。

## 命令与结果

下文 `$FIX` = `/root/flag-fixes/fixes/zeros-7181`，`$T` 为对应树根；
硬件脚本统一形如 `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$T/src python3 $FIX/tests/<script>`。
所有日志为工具原文、未脱敏，已存 `results/`（文件名见各 log 内首行命令与下表）。

| 命令 / 脚本 | 修复前（`flagfix-noncplx` / 基线） | 修复后（`flagfix-fixed` / 基线） |
| --- | --- | --- |
| `tests/repro_complex64.py`（issue #7181 形态 + torch 入口） | **exit 1**：`ISSUE REPRODUCED`，`KeyError: 'complex64'` → `results/repro-baseline-2026-11-09.log` | **exit 0** → `results/repro-fixed-2026-11-09.log` |
| `tests/test_complex_ops.py`（**210 用例**矩阵，第二轮扩含 bool/complex→bool/空输入/conj；第三轮修订 strided conj 样例） | **exit 1**：`TOTAL: 33 passed, 177 failed, 210 total`（137 条 complex KeyError；其余为 strided `zero_` 断言、交叉 dtype 静默分歧、bool out 拒绝、conj 直调等修复缺失行为）→ `results/regression-baseline-2026-11-10.log` | **exit 0**：`TOTAL: 210 passed, 0 failed, 210 total` → `results/regression-fixed-2026-11-10.log`（第二轮样例修订前原文 `results/regression-*-2026-11-09.log` 同口径 33/177/210 → 210/0/210；第一轮 139 口径 `results/regression-*-2026-11-08.log`：18/121/139 → 139/0/139） |
| `tests/test_m2_bert_backward.py`（12L/768/12/3072/8192，问题原配置） | **阶段一** `flagfix-e2ebase` exit 1：`_thead/ops/linear_backward.py:170` view `RuntimeError` → `results/e2e-linearblocker-baseline-2026-11-07.log`；**阶段二** `flagfix-noncplx` exit 1：前向 `FORWARD OK` loss rel=0.0，反向 `ops/sum.py:385 → KeyError: 'complex64'`（= problem.md 签名）→ `results/e2e-problem-path-noncomplex-2026-11-07.log` | **exit 0**：`E2E OK: loss_native=1.000000000e+00 loss_gems=1.000000000e+00 rel=0.000e+00 grads_present_finite=161 grads_missing_known_gap=384` → `results/e2e-fixed-2026-11-09.log` |
| 同脚本，强制无设备（`CUDA_VISIBLE_DEVICES=`） | exit 2（缺设备） | exit 2（缺设备）→ `results/nocuda-regression-2026-11-09.log` |
| 同脚本，隔离 `transformers` import 后 | exit 2：`MISSING DEPENDENCY: transformers -> exit 2` | exit 2 → `results/e2e-missing-dep-2026-11-09.log` |
| 同脚本，跑在**基线树**（前置哈希检查） | — | **exit 2**（模型工作前）：`MISSING PRECONDITION: src/flag_gems/ops/linear.py sha256 4d46bae0… != expected ddec3383…` → `results/e2e-precondition-baseline-2026-11-09.log` |
| 同脚本，跑在**仅 linear-5379 树**（缺 zeros-7181） | — | **exit 2**（模型工作前）：首个失配 `sum.py 5259b9f6… != 82750041…`（连同 zeros/zero/zeros_like/abs 等逐条列出）→ `results/e2e-precondition-linearonly-2026-11-09.log` |
| e2e 判据（修复后日志逐条满足） | — | 见下「e2e 判据与证据链」 |

> **计数器缺陷注（旧日志）**：`results/*-2026-11-07.log` 中 regression 的
> `TOTAL: {N} passed, {M} failed` 行，`{N}` 实为 passed+failed（`TOTAL: {passed + failed}
> passed, {failed} failed`）。旧日志原文保留；旧口径实际为 **16 passed / 38 failed /
> 54 total**（38 条失败全部为 complex KeyError）。2026-11-08 起打印改为
> `passed/failed/total` 三个独立数字，新日志无此问题。

### e2e 判据与证据链（`results/e2e-fixed-2026-11-08.log`）

执行顺序与逐条判据：

1. **前置哈希（模型工作之前）**：`e2e_precondition.files` 9 条 sha256 逐文件核对
   （8 个目标 after hash + `ops/linear.py` `ddec3383…`），任一失配打印
   `MISSING PRECONDITION: … -> exit 2`（上面两行日志即其实测）；同时做**双源校验**
   ——`e2e_precondition` 的 sha 必须与 `targets[].after_sha256` 一致，防清单漂移。
   通过时打印 `precondition OK: 9 files at declared sha256 (baseline 29e1482 + linear-5379 + zeros-7181)`。
2. **模型代码 pin**：打印 `model code: commit=90e0b28f…` 与 `revision` 比对。
3. **依赖**：`opt_einsum=bundled-stub`（或 `real`）打印；`transformers` 缺失 exit 2。
4. **前向**：`FORWARD OK`，loss 快照在 backward **之前** 为 Python float（防 loss
   存储间歇覆写），`rel=0.000e+00`（≤1e-3 阈值，实际逐位一致）。
5. **反向完成**；原生产出的梯度全部存在且有限：`grads_present_finite=161`；
   缺失集合与 `tests/known_gap_m2.txt`（32 模板 × 12 = **384** 条）**双向断言**
   ——fixture 外缺梯度 → fail；fixture 内反而有梯度 → fail。全部
   `missinggrad:` 行原样打印作证据。
6. native 侧同配置梯度数 `native grads=545` 打印（对照参考，不作阈值判据）。
7. 退出码 0 才算 `E2E OK`。

### oracle 口径

native 参照在 `flag_gems.enable()` **之前**捕获（enable 后 call-site 的
`torch.*` 已被 dispatch 进 flag_gems，不能当 oracle）；complex 比较一律经
`torch.view_as_real`（flag_gems `eq`/`nonzero` 不支持 complex，属已记录的预存缺口）。

回归 210 用例分层判据（2026-11-09 版 `test_complex_ops.py`，第二轮评审后）：

| 套件 | 判据 |
| --- | --- |
| 原 54 用例（sum/zeros 族/abs 基础矩阵） | rtol=1e-5 / atol=1e-5（逐项与 native 比） |
| **strided `zero_`**（`x[::2]`、`x[1::2]`、view 列、narrow、转置、0 维 × `gops.zero_`/`gops.zero`/`aten.zero_`） | 被操作子与 native 同形：仅目标区间归零、原存储不被污染、返回值身份/dtype/stride 正确 |
| **模长量级**（`abs`/`absolute`，大数/亚正规/inf/NaN 形态） | rtol=1e-6 或 1e-10（量级分档）、atol=0、`equal_nan=True`；inf 优先于 NaN 与 native 一致 |
| **交叉 dtype out 矩阵**（`sum`/`sum_dim` out × real/complex/bool in × dtype 显式/省略，含空输入） | ok 用例数值一致；err 用例双向：本实现不抛而 native 抛 → fail，反之亦然；报错**逐字**与 ATen 一致（如 `expected scalar type but found ComplexFloat`） |
| **bool out 采用**（第二轮新增）：dim 形 `dtype=None` 采用 `out.dtype`（bool/real/complex × 空/非空）、全量 `sum.out` 不采用只校验、显式 `dtype != out.dtype` 报错逐字 | 与 native 双向 oracle（ok 值+dtype+shape、err 类型+消息逐字）；`sum([T,T,T], dim=0, out=bool)` 等计数语义与 ATen 一致 |
| **complex→bool any-component**（第二轮新增）：纯虚 `[0+1j]`、全零、混合、c64/c128 × 全量/dim × 显式 `dtype=bool` 与 bool out | c10 TypeCast 规则：任一分量非零为 True（`sum([0+1j], dtype=bool)` = True）；报错逐字 |
| **空输入归约**（第二轮新增）：bool/f32/complex × dim0/dim1/keepdim/全量 × 各 out dtype | 与 native 一致的零结果（不进 kernel、无除零）；空 + bool out 亦不抛 |
| **conj 张量**（第二轮新增）：预 enable 构造带 conj bit 的 contiguous/strided 张量 × `abs`/`sum` × 直调 `gops.*` 与 `torch.*` dispatch | 与 native oracle 一致（`abs(conj)` 模长、`sum(conj)` 共轭求和、strided conj 视图） |
| **cast-first 非零规则**（第二轮新增）：f32 抵消样例 `[[0.5,…],[−0.5,…]]` 与小数 complex → `dtype=bool` | ATen 先按非零转 bool 再归约（0.5→True），非求和后截断（0.0→False）；直调与 dispatch 一致 |
| **autograd parity**（`abs().sum()`、`sum().abs()`、`(x*2).abs().sum()` × c64/c128） | 梯度+前向 rtol=1e-4 / atol=1e-6；仅**实数标量结尾**链入判据（`pow(2)` 链与 complex 中间标量风险已排除，不纳入） |

### 梯度缺口对照（不属于本 patch 的预存缺陷）

用 `flag_gems.enable(unused=[...])` 排除 `sum/zeros/abs/fill` 系后反向可以在两棵树上跑完，
逐参数对照（日志：`results/gradgap-probe-noncomplex-2026-11-07.log`、
`results/gradgap-probe-fixed-2026-11-07.log`，1L 配置）：

| 观察项 | 无 complex 修复树 | 修复树 | 结论 |
| --- | --- | --- | --- |
| 缺失梯度集合 | `missing=34`（32× `filter_fn.*` + 2× 未使用 pooler） | 完全相同 `missing=34` | flag_gems 全包 0 个 `register_autograd`，`aten::pad`/`aten::conv1d` 走 autograd-not-implemented 回退 → 预存、与本 patch 无关；12L 默认配置精确集合固化为 `tests/known_gap_m2.txt`（384 条）并被 e2e 双向断言 |
| 梯度数值分歧 | native ~1e-9 vs flag_gems ~0.5 量级（近驻点参数），另有 native 有值 / flag_gems 为 0 项 | 同形态、同参数集合 | 预存、与本 patch 无关 |
| loss 张量回读（个别 run） | 不适用（反向跑不完） | 部分 run `backward()` 后 loss 存储被覆写（间歇） | **归因未定**（既未证明预存、也未证明本修复引入，见 diagnosis.md §3）；e2e 因此在反向**前**快照 loss 为 Python float |

## 管理工具执行结果

均在 `/root/flag-fixes` 下、`PYTHONPATH=src` 执行。**注意**：`requires:
[linear-5379]` 非空后，select 类命令（`check`/`plan`/`apply`/`status`）必须同时给出
两个 ID：`… linear-5379 zeros-7181`（catalog 会做 requires 子集检查）。

| 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `flag_fixes validate` | 0 | `{"valid": true, "fixes": 11, "deployable": 0, "hardware_verified": 0}`（2026-11-09，第二轮 hash 更新后复验）→ `results/cli-validate-2026-11-09.log` |
| `flag_fixes check linear-5379 zeros-7181 --target $BASE` | 0 | 9 目标（8+linear.py）before hash 全中、patch `git apply --check` 干净 → `results/cli-check-2026-11-09.log` |
| `flag_fixes plan linear-5379 zeros-7181 --target $BASE` | 0 | 同上（预演不落盘）→ `results/cli-plan-2026-11-09.log` |
| `flag_fixes check zeros-7181 --target $BASE`（**漏依赖 ID**） | 2 | `error: zeros-7181: include required fixes: ['linear-5379']`（requires 子集检查实测）→ `results/cli-check-requiredfail-2026-11-09.log` |
| `flag_fixes apply linear-5379 zeros-7181 --target $BASE` | 2 | `error: Selected fixes are not approved for deployment`（`deployable=false`，**按设计拒绝**）→ `results/cli-apply-2026-11-09.log` |
| `flag_fixes status --target /tmp/opencode/flagfix-fixed` | 0 | `{"transactions": [], "drift": [], "recovery_required": false}` → `results/cli-status-2026-11-09.log` |
| `flag_fixes verify zeros-7181 --target /tmp/opencode/flagfix-fixed --python python3` | 0 | 三轮均 `passed: true` 3/3（11-07 初版 `verify-2026-11-07{,-dirty}.json`、11-08 第一轮 `verify-2026-11-08.json`、11-09 第二轮提交后干净跑 → `results/verify-2026-11-09.json`，本表提交步骤完成后执行） |

## 第二轮评审修正（2026-11-09，基于 `9fa84cf`）

评审确认第一轮四项 must-fix 全部落地后，新增两处 P2 正确性缺陷与 conj 跟进项，本轮
全部修复并复验（源码改动仅 `ops/sum.py`、`ops/absolute.py`；测试扩至 210 用例）：

| 评审项 | 修复 | 验证 |
| --- | --- | --- |
| P2：`sum_dim_out` 拒绝合法 bool out（`dtype=None` 时曾强制按 i64 计算再判不匹配） | 删除 bool→i64 特殊分支：dim 形一律采用 `out.dtype`（bool/real/complex、空/非空），显式 `dtype != out.dtype` 仍逐字报错；全量 `sum.out` 保持不采用只校验 | bool out 采用矩阵 + 报错逐字双向 oracle（`regression-fixed-2026-11-09.log` 全绿）；native 对照见测试 pre-enable oracle |
| P2：complex→bool 丢虚部（`_sum_complex` 曾对所有实目标取 `inp.real`，`sum([0+1j], dtype=bool)` 得 False、native True） | bool 目标改用完整张量 any-component 规则（`real.ne(0) \| imag.ne(0)`，对齐 c10 TypeCast），其余实目标保持实部语义 | 纯虚/全零/混合 × c64/c128 × 全量/dim × `dtype=bool`/bool out 全部与 native 一致 |
| conj 跟进：`view_as_real` 不接受未物化 conj bit（`.contiguous()` 不清 conj 位） | `absolute.py` 入口 `resolve_conj().contiguous()`；`_sum_complex` 入口 `resolve_conj()`（`.imag` 访问器在 enable 后返回未取负的裸存储视图，dispatch 路径靠 ATen 物化掩盖、直调路径暴露） | 预 enable 构造 conj 张量 × `abs`/`sum` × 直调 `gops.*` 与 `torch.*` dispatch 全部与 native 一致 |
| 连带发现并修复：空输入 dim-out `ZeroDivisionError`（`sum_dim_comm` 除以零长度维；全量 `sum.out` 空输入同样） | `sum_dim`/`sum_dim_out` 空分支共用 `_empty_dim_result`（按 dtype 造约简形状零张量，不进 kernel）；`sum_out` 补 `M == 0` 分支 | bool/f32/complex 空输入 × dim/全量 × 各 out dtype 与 native 一致 |
| 连带发现并修复：float→bool cast-first 截断分歧（flag_gems cast kernel `0.5→False`，ATen 非零规则 `0.5→True`）；scalar `dim` 直调 `sum_dim_comm` 未归一化 | `_cast_before_sum`：bool 目标用 `inp.ne(0)` 建掩码（`sum`/`sum_out`/`sum_dim_comm` 三处），complex bool 支用 real/imag 比较；`sum_dim_comm` 入口把标量 `dim` 归一化为列表 | 抵消样例 `[[0.5,…],[−0.5,…]]` 与小数 complex 的 bool 归约与 native 一致；`gops.sum_dim(x, 0)` 直调可用 |

同轮复验（全部 exit code 达标、日志见 `results/*-2026-11-09.log`）：repro 1→0、
regression 33/177/210→210/0/210、e2e `E2E OK`（前置 9 文件新哈希、161/384、
loss rel=0）、错树前置 exit 2、缺依赖/缺设备 exit 2、CLI 六项退出码与第一轮一致、
`compileall`/`pytest`(26) 通过；patch/快照/`fix.yaml` 三者 hash 一致（`validate`+`check`/`plan`）。

## 第三轮评审小修（2026-11-10，基于 `b5a2cc8`）

评审结论：无阻塞性问题，可作为 `draft / deployable=false / hardware_status=unverified`
条目合入；剩两项非阻塞小修，本轮完成（仅测试与文档，源码零改动）：

| 建议项 | 处理 | 验证 |
| --- | --- | --- |
| strided conj 样例实际仍连续（`_cbase[::2]` 首维长度为 1） | 改为 `_cbase[:, ::2].conj()`（stride `(2,2)`，真正非连续），enable 前加 `assert _pc.is_conj()` / `assert not _pcs.is_contiguous()` / `assert _pcs.is_conj()` 断言前提；direct + torch dispatch 双入口对照保留 | 干净提交 `1fa248d` 上设备重跑：20 个 conj 用例（`abs`/`abs_strided`/`sum`/`sum_dim0`/`sum_strided` × c64/c128 × direct/dispatch）全 PASS，`results/regression-fixed-2026-11-10.log`；基线同套件 KeyError 失败形态不变（33/177/210） |
| “实数路径逐位不变”表述过强 | README（首段/改动表/验证表）、`WORKFLOW.md` 登记行、`reports/diagnosis.md`、PR 描述统一改为：分支在 `is_complex()` 处短路但含实数语义修正（cast-first、float→bool 非零、out dtype 推断、空输入），已列实数回归用例通过、**未覆盖输入与后端不作逐位一致保证**；单次模型前向 loss rel=0 结论保留（有日志支持），不推广 | 文档 diff 自查；历史原始日志不改写 |

复验（测试脚本变更，评审要求在干净提交上重跑并关联 commit）：`flag-fixes verify`
在 `1fa248d` 干净树上 `passed: true` 3/3、`source_dirty: false`、regression
210/0/210 → `results/verify-2026-11-10.json`（单独提交）；文档调整不重复硬件运行。

## 第四轮评审：随机权重小模型数值一致性与性能评估（2026-11-10，评审意见 comment `6756872311`）

按评审方案自建两个小模型（A：复数 sum/abs 全反向路径；B：conv2d + NHWC 非连续
linear 反向路径），四组进程隔离（n 原生 / f_target 仅注册目标算子 / f_all 全量注册 /
b 修复前对照树 = base+linear-5379），种子 0/1/2、fp32（A 另有 fp64/c128）、确定性设置
逐 run 记录、阈值按评审表先冻结（原生噪声门 max_abs=0.0 位级一致）、有限差分抽检
PASS、lr 校准 0.1（候选 1e-1 为按评审"调任务再校准"条款新增，理由记录在案）。

| 结论 | 数据 |
| --- | --- |
| **f_target（修复声明范围）9/9 全 PASS** | fp32 max_abs 1.19e-7–4.77e-7（阈值 1e-5）；fp64/c128 max_abs 4.4e-16–8.9e-16（阈值 1e-10）；单步中间量/全部梯度/SGD 更新后参数/20 步训练逐 checkpoint 通过，梯度存在性硬校验无白名单 |
| **b（修复前对照）两模型均在修复目标路径崩溃** | A：`sum_dim_comm`→`fill_scalar_` `KeyError: 'complex64'`（正是本修复目标）；B：旧 `linear_backward.py:170` NHWC 非连续 `.view()` RuntimeError（修复树已改 `reshape`，f_target 通过）→ 证明修复在真实模型路径生效且被实际命中 |
| **f_all（全量注册）失败，首个分歧节点已定位（均为预存、不在本 patch 8 目标文件内）** | A fp32：abs 反向 `sgn.py:149` `empty_like`(complex64)→triton `canonicalize_dtype` KeyError；fp64：triton CompilationError（累加器 fp32/fp64）；B：**静默错梯度**——`mse_loss_backward` 不广播 0 维标量 grad_output（autograd 对标量 loss 的传参），最小复现 `tests/repro_mse_scalar_grad.py`：39/40 元素错、第 0 元素精确、全量张量 grad_output 时精确、none/sum reduction 精确；该文件与修复前树逐字节相同（预存） |
| **性能（首轮只报告，不作加速声明）** | 75 用例（sum 全/内/外、abs、zeros/zero_，c64/c128 × 2^10/2^16/2^20 × 连续/跨步 + fwd+bwd；模型级 fwd/fwd+bwd/整训练步）：本构建上 f_target 的 complex sum/abs **慢于**原生（sum 3–21×、abs 1.06–1.7×），zeros/zero_ 持平到略快（0.83–0.96×）；模型级 f_target 1.07–2.36×；f_all/b 可跑路径 10.5–12.7× 慢；>10% 稳定回归 58 项已打旗标；崩溃路径一律记 N/A 不计加速 **[historical / superseded by bench-r2（第六轮）：zeros/zero_ 结论、58 旗标计数、模型级 1.07× 归因均已撤回，当前有效口径见第六轮/第七轮节]** |

产物：`results/small-model-2026-11-10/`（`matrix/`=阈值+校准+29 run+20 对比+fd+summary，
`bench/`=5 组原始块计时+汇总表，`mse-scalar-repro.log`，`provenance.json`）；
矩阵跑在干净提交 `968d624`，bench 跑在 `d236d5a`（两者 `tests/small_models.py`/
`tests/test_small_models.py` 逐字节一致，`git diff` 为空【该表述仅限
968d624..d236d5a 区间；后续 fac401c 对该文件加固比较器 226 行差异并对已存
dumps 重评分，见第六轮节】）；完整报告
`reports/small-model-evaluation.md`。结论限定于这两个模型与当前后端构建。

## 第五轮评审修正（2026-11-10，意见 comment `6766490204`，基于 `5e02b3b`）

评审确认隔离组 9/9 证据有效；四项修正全部完成（代码 `fac401c`/`dd5ebea`，证据见
`results/small-model-2026-11-10/{matrix-rescored,bench-r2}/`，报告已重生成替换）：

| 意见 | 处理 | 验证 |
| --- | --- | --- |
| P1 性能 f_target 注册范围错（固定 model="a"，B/清零未注册目标算子） | bench 重写为按族×组独立进程（a-sum-abs / b-linear / zeros 各自 include），f_target 期望注册键**断言**缺失即失败；`registered_target_keys` 改名并在注册之外用 torch.profiler 捕获 CUDA kernel 名做**非计时命中验证**（sum/abs/conv2d/linear 命中 triton kernel；`zero_` 命中原生 fill = 修复刻意走原生 `copy_(0)`；b 树探针崩溃逐条记录） | bench-r2 15 组全 rc=0；`bench2-{family}-{group}.json` provenance 含注册键断言与 kernel 名；**modelB f_target 重测 fwd 2.42×/fwd+bwd 2.93×/step 2.52×**（四轮的 1.07× 系未注册目标算子所致，作废） |
| P2 比较器缺结构检查 | `compare_dumps` 纯函数化；先查 shape/dtype 严格相等（禁广播、禁静默升 dtype）、inter/grads/params 键集合、训练步数+step ID 序列（0..n-1、无重复、两端一致）；native 参考梯度缺失/非有限 → **判参考实验无效（fail）而非 skip** | 负例自测 `tests/test_comparator_negative.py` 10/10（对照过、错 dtype/可广播错 shape/缺尾 checkpoint/重复/错序 step/缺梯度键/参考非有限/修复侧非有限/超阈值 全部按预期失败）；20 对 + 噪声对用已存 .pt 重评分：**f_target 9/9 + noise PASS 维持**，原始运行证据未改写 |
| P2 吞吐公式多乘 100 / P95 命名 / 显存口径 | `calls_per_s = 1000/ms_per_iter`（模型级另加 `samples_per_s = batch×calls_per_s`，旧 `samples_per_s` 字段撤回）；P95 改名 `block_mean_p95_ms_per_iter` 并注明"5 块均值的分位估计，非单请求尾延迟"；显存记录 case 前 baseline、窗口内 peak（含编译+预热）与 delta（共享驻留下界） | bench-r2 JSON 字段与协议说明同步；报告措辞替换 |
| P2 "稳定退化"与计数 58；f_all-B 定性 | 旗标改名 `*_candidate_regression_gt10pct`（仅中位比值判据，非稳定性结论），**计数从数据自动生成 = 115**（四轮报告的 58 作废）；bench 驱动读 accuracy summary，正确性失败配置（f_all/b 模型级）标 `diagnostic only`，不计入有效性能 | `bench2-summary.json` 字段 `candidate_regression_gt10pct_count=115`；表格含 correctness 标注 |
| 推进顺序 3/4：报告三态、bench commit 统一、另立修复项 | 报告重生成：目标隔离组通过 / 全量组失败 / 性能已复验三态；bench 版本统一到实际证据 `dd5ebea`（四轮 bench 目录保留为历史）；**另立修复项** `reports/follow-up-fix-items.md`（FIX-ITEM-1 mse 标量广播优先、FIX-ITEM-2 complex pointwise/fill/zeros 尺寸分支/to_copy + FP64 编译），并补齐**同路径隔离对照**：mse 39/40 错与 complex 5/5 KeyError 在修复前对照树逐位复现（`matrix-rescored/repro-{mse-scalar,complex-fill}-{btree,fixed}.log`）——预存归因不依赖"文件不在 patch 内"单独成立 | 复现脚本入库（`tests/repro_complex_fill.py` 扩展组装张量探针；触发条件 = `torch.complex(p,q)` 组装张量，cast 张量不触发）；四轮原始证据不改写 |

未执行口径不变：原始 fix 维持 `draft / deployable=false / hardware_status=unverified /
owner=null`；小模型结论不替代大模型完整梯度与容器平台验收；两项新缺口各自独立建包，
修复前后证据在其自身轮次补齐。

## 第六轮评审修正（2026-11-10/11，意见 comment `6784122575`，基于 `fb13007`）

| 意见 | 处理 | 验证 |
| --- | --- | --- |
| P1 报告沿用旧清零结论（0.89–0.96×/0.83–1.00×）与 bench-r2 实测（2^20 达 97–290×）矛盾 | 旧表述**撤回**；性能节改为由 `benchmarks/gen_perf_section.py` 从 `bench2-summary.json` **自动生成**（`bench-r2/perf-section-auto.md`，报告内嵌），旗标计数 115、比值与舍入全部出自数据；sum "大尺寸仅 1.1–2×" 同步撤回（实测 forward 3.95–22.92×、fwd+bwd ≈3×） | 自动生成文件 + 报告内嵌一致；机器可复核 |
| P1 后续定位：zero_ 的 copy_(0) 追踪 | 根因完成：**native ATen complex-标量 copy 本身经全量张量可分页 HtoD 中转**——native `copy_(0)` 0.74–0.85 ms ≈ f_target 0.90 ms（同路径）；native `zero_()` 0.005 ms 设备 memset；设备端 0 维源 0.017–0.020 ms **[historical / superseded：该值混入每次 `.cuda()` 构造，第七轮预建后拷贝净成本 0.0071/0.0069 ms，见第七轮节]**。profiler 全事件：f_target `zero_` 仅 Memcpy HtoD、`zeros` 为 native FillFunctor+HtoD、无 Triton kernel。→ **FIX-ITEM-3**（`reports/follow-up-fix-items.md`），候选方向（设备端 0 维源/real 视图清零）已列，不承诺优化比例 | `benchmarks/zero_staging_probe.py` → `bench-r2/zero-staging-attribution.log`；`benchmarks/zeros_hitprobe.py` → `bench-r2/zeros-hitprobe-{f_target,native}.json`（全事件清单+分类+dtype/shape/入口/同步方式） |
| P2 profiler 命中说明与 JSON 不符 | 三分法表述：**registered（断言）/ probe-hit（kernel 事件）/ timed-case（尺寸标度）**；round-5 回复"zeros 命中 Triton"系 `vectorized_elementwise_kernel` 子串误判，**更正为 native FillFunctor+HtoD、无 Triton kernel**；无法识别的 kernel 名标 unknown、原始名逐字保留 | `zeros-hitprobe-*.json` 全事件清单；报告与回复同步更正 |
| P2 provenance 不准确（"tests 文件逐字节一致 968d624..fac401c"不实） | 更正：`test_small_models.py` 在该区间 **226 行差异（170+/56−，比较器加固）**；准确表述 = 设备运行在 `968d624`（当时 harness）、`fac401c` 加固比较器并对**已保存 dumps** 重评分（设备数据未重跑）；`tests/small_models.py` 确无变化。新增 `bench-r2/run-commits.json` 机器可读清单，报告 provenance 表自动对应 | `git diff --stat 968d624..fac401c` 记录于清单；报告 provenance 表重写 |

可保留结论（评审确认）：隔离组 9/9 + 比较器加固与负例自测；全量组两缺口（mse 标量
广播、complex pointwise/fill）继续作为独立修复项；性能定位为"修正归因后的观测报告"，
**清零路径大幅开销已如实量化并根因（FIX-ITEM-3）**，是否另立优化项交维护者决定。
状态口径不变：`draft / deployable=false / hardware_status=unverified / owner=null`。

## 第七轮评审修正（2026-11-11，意见 comment `6785197539`，基于 `236fbfd`）

两项 P2 全部落地：

| 意见 | 处理 | 验证 |
| --- | --- | --- |
| P2 设备端 0 维标量对照混入每次 `.cuda()` 构造成本 | `benchmarks/zero_staging_probe.py` 修正：设备标量**预建在计时之外**，单列 device-标量构造（0.0139 ms，**round-8 更正标签：此为 50 次平均；单次 CUDA event 实测 0.3358 ms**）/ 拷贝净成本 / 设备 memset / 同步边界；补 c128 确认 | 修正探针实测（2^20，round-7 批次）：`copy_(0)` 0.83/1.50 ms（继承路径）vs **`copy_(预建 device-0dim)` 0.0071/0.0069 ms（拷贝净成本，round-7 批；round-8 批复测 0.0089/0.0067 ms，批间为计时波动）** vs `zero_()` 0.0042/0.0055 ms——host staging 与设备路径差距 ≈117×，原 0.02 ms 系混入构造成本、已更正；报告与 FIX-ITEM-3 同步 |
| P2 历史性能段落未标 superseded | 四处旧口径加 **[historical / superseded by bench-r2]** 标记（原日志不改写，仅加标记）：`fix.yaml` 第四轮 bullet、`reports/validation.md` 第四轮性能行、`results/README.md` bench/ 行、（报告 §4 的撤回清单本已内嵌）；`README` 的 withdraw 表述本身无需改 | grep 复核：旧数字仅存于历史节/撤回说明/带标记处 |

**当前有效口径（机器读取以此为准；观测仅限当前后端与计时协议，不外推、不构成加速承诺）**：
- complex `zeros/zero_` bench-r2 2^20 实测约为 native 的 **97–290×**（随字节数线性，host staging 机制）；
- complex `sum` forward 约 **3.95–22.92×**、fwd+bwd 约 **3×**；
- modelB f_target 重测 fwd **2.42×** / fwd+bwd **2.93×** / step **2.52×**（第五轮重测口径）。

**round-9 口径细化（copy-only 双批次）**：预建设备标量 copy-only 净成本按探针批次标注——
**round-7 批 0.0071/0.0069 ms、round-8 批 0.0089/0.0067 ms**（c64/c128；批间为计时
波动与复测差异，非功能差异；批/设备/协议 provenance 见
`zero-staging-attribution-r7-batch.log` 与 `zero-staging-attribution.log`）；不再宣称
单一"唯一口径"。单次创建 0.3358 ms 一律保留**含冷启动**说明，与稳态拷贝成本区分。

FIX-ITEM-3 按评审框定**三条候选路径**独立跟踪（native memset 基准 / 预存设备 0 维
标量 copy / real-view 或逐元素设备 kernel），三者均须通过同一正确性矩阵
（contiguous、slice/transpose、storage offset、c64/c128、未选中 storage 不污染）；
候选实现只有在正确性回归、非连续视图与大尺寸性能全部通过后才考虑独立建优化 fix。

## 第八轮评审修正（2026-11-11，意见 comment `6786286552`，基于 `4b71e9a`）

评审确认无新功能回归；两项 P2 收尾（均系文档/标签一致性）：

| 意见 | 处理 | 验证 |
| --- | --- | --- |
| P2 "一次性创建成本"标签不准（`bench()` 为 warmup+50 次平均） | 探针改为**单次 CUDA event 测量 + 明确标注 50 次平均**，两个数值都报告 | 单次 0.3358 ms（含冷启动）/ 50 次平均 0.0145 ms / copy-only 净成本 0.0089 ms（本次运行）——`bench-r2/zero-staging-attribution.log` 已更新；FIX-ITEM-3 表格加 round-8 更正说明 |
| P2 旧证据未完全同步（results/README"逐字节一致"与 0.02ms、validation 第六轮表 0.017–0.020 ms、报告旧 0.02 ms 描述） | 全部统一到当前口径：results/README 头注改为"968d624..d236d5a 区间一致 + fac401c 后续 226 行加固"并更新 0.02ms 行；validation 第六轮表 0.017–0.020 ms 加 **[historical / superseded]**；报告 §4 机制段替换为 round-7/8 修正措辞（生成器 `gen_perf_section.py` 同步修改并重生成 `perf-section-auto.md`）；Round-7/8 correction 段加 historical 标记 | grep 复核：旧值仅存于带标记处与原始日志；`copy_(预建 device-0dim)` 0.0071/0.0069 ms、native `zero_()` 0.0042/0.0055 ms 为唯一当前口径 |

评审的**后续实施建议**已登记：FIX-ITEM-3 候选评估协议（每候选记录正确性/非连续
视图行为/大尺寸耗时/内存变化）~~与多平台复验要求（优化实现完成后在昆仑芯/昇腾/
平头哥复跑同一矩阵，不将 PPU/thead 归因外推）~~【**round-9 评审更正：三平台复验
属 PR3 problem29 性能 bundle 范围，PR2 不要求，该条撤回**；FIX-ITEM-3 若扩展到
其它后端应另建平台适配/优化条目再分别定义验证范围】已写入 `follow-up-fix-items.md`。

## 第九轮评审收尾（2026-11-11，意见 comment `6787003142` + 更正 `6787034891`，基于 `78bece4`）

评审确认无新功能回归；一项 P2 收尾 + 一处范围更正：

| 项 | 处理 |
| --- | --- |
| P2 copy-only 双批数值并存易混淆 | **批次口径**：round-7 探针批 0.0071/0.0069 ms、round-8 探针批 0.0089/0.0067 ms（c64/c128）——批间为计时波动与复测差异，非功能差异；汇总引用须标明批次来源，不再宣称单一"唯一口径"。round-7 批日志自 git 历史归档为 `zero-staging-attribution-r7-batch.log`，round-8 批为 `zero-staging-attribution.log`；生成器 `gen_perf_section.py` 同步改写并重生成 `perf-section-auto.md`，报告/validation/results·README/FIX-ITEM-3 全部按批标注 |
| 单次创建 0.3358 ms | 生成段与各文档一律保留**含冷启动**说明，与稳态拷贝成本区分 |
| 多平台复验范围更正（撤回） | 按评审更正：**昆仑芯/昇腾/平头哥三平台复验属 PR3 problem29 性能 bundle，PR2 不要求**；validation 第八轮表该句划线撤回、FIX-ITEM-3 同步；FIX-ITEM-3 若扩展到其它后端另建平台适配/优化条目 |

收尾确认（评审原文）：CI 通过、探针标签与 provenance 闭环、无新功能阻塞；剩余仅
可选的历史数值标注清理（本节完成）。功能 fix 维持 `draft / deployable=false /
hardware_status=unverified`；9/9 与比较器负例保留；FIX-ITEM-1/2/3 独立推进
（FIX-ITEM-3 正确性矩阵与每候选记录要求不变，仅多平台项撤回）。

## 第十轮：后续探索与实施方案落地（2026-11-11，意见 comment `6787395475`，基于 `2a0753a`）

本轮为方案执行（非缺陷修复）：按评审 §一/§二 完成基线冻结与候选筛选，
§三 登记探索计划，§五 登记验收门槛。

| 评审节 | 落地 | 结果 |
| --- | --- | --- |
| §一 基线冻结 | 新增 `benchmarks/baseline_freeze.py`（三组进程：native / fix[注册 zeros+sum+abs 族] / prefix[修复前对照树]）；zeros/zero_ × {contig, slice-stride, narrow, transpose, storage-offset} × c64/c128 + conj/empty/0-dim 边缘 + sum/abs 数值边缘（NaN/Inf/±Inf/1e38/1e-38/empty/conj，位级 checksum 交叉对比）；provenance（tree/torch/device/determinism/注册键）逐 JSON 记录 | `results/small-model-2026-11-10/baseline/`：~~fix vs native 52/58 位级一致；6 例 1ulp 级差异~~ **[historical / superseded by round-12 严格比较器（截断 checksum 与双 exception 判等缺陷）：r12 重跑 50 numeric_equal + 8 numeric_diff，ULP/误差已量化（conj ≤2 ULP、scaled-hypot 1 ULP 级、NaN payload 位不保证），见第十二轮节]**；**prefix 58/58 全崩**逐例记录；探针过程中发现 FIX-ITEM-4（见下） |
| §二 zero_/zeros 三候选 | 新增 `benchmarks/zero_candidates.py`（原生 harness，不改动本 patch）：正确性矩阵 6 布局 × c64/c128 × 5 实现（含未选中 storage 位级保持、native 逐位一致）+ 2^20 三布局性能（首调/5×100/events/显存） | **矩阵 60/60 全过**（候选 A conj 回退首轮暴露即修正）；2^20 c64 contig：当前 0.6674 ms → **A 0.0083 / B 0.0050（最优全布局）/ C 0.0154 / native 0.0042 ms**；独立优化 fix 建议按 A→B→C 落地（可组合 A 快路径 + B 兜底），过验收门槛后另行提交 |
| §三 sum/abs 探索 | 登记为探索计划（`follow-up-fix-items.md` round-10 节）：real/imag 分累加、四类 reduction 分别分析、fwd/bwd 分测、数值边缘扩展（基线已含首批）、每候选记录 max_abs/相对误差/梯度误差/kernel 数 | 探索过程中**发现 FIX-ITEM-4**：FlagGems `sum` 对 uint8 返回 uint8 dtype 且按 256 回绕（`[200,100,50,7].sum()`=101 vs native 357/int64）——最小复现与归因要求（对照树实测）已登记 |
| §四 独立跟踪 | FIX-ITEM-1/2/3 维持独立；新增 FIX-ITEM-4 | `follow-up-fix-items.md` |
| §五 验收门槛 | 六项门槛（210 回归 / view-storage 矩阵 / c64-c128 数值梯度对照 / 小模型三步 / 容器验收 / 性能重复+provenance）写入 `follow-up-fix-items.md`"验收门槛"节 | 小模型只证明局部一致，不替代完整模型验证 |

功能 fix 维持 `draft / deployable=false / hardware_status=unverified`。

**round-10 确认批**（干净提交 `59301a1` 重跑，`run-commits.json` round10_additions）：
候选矩阵 **60/60**、基线 ~~52/58 位级一致~~（**superseded by round-12**）+
prefix 58/58 全崩——与首轮（工作树批）完全一致；性能中位数批间偏差 ≤~3%
（首轮数值为引用口径，确认批 JSON 以 `-59301a1-confirmation` 后缀归档）。

## 第十一轮：候选矩阵扩展、注册环境复测与两项新缺口（2026-11-11，意见 comment `6788210724`，基于 `9723734`）

| 评审项 | 落地 | 结果 |
| --- | --- | --- |
| §1 矩阵扩展 | `zero_candidates.py` 扩展：zeros **创建路径**、empty/0-dim、组合布局（offset×transpose×slice×conj 五种）、精确 storage 索引映射的 alias/未选中检查；负 stride 框架不支持登记 N/A | 原生 harness **260/260 全过**；候选 A 新增两条实现约束（0 维/conj 必须 fallback B） |
| §2 注册环境复测 | `--registered` 模式（zeros 族 only_enable）：A/B/C/native_ref 逐候选取 profiler 事件 + 耗时 | **A/B 过验证**（事件全为 native-aten、无 HtoD-staging、0.0051–0.0075 ms 与原生 harness 一致）；**候选 C 被淘汰**——`select(-1,i).zero_()` 命中注册 zero_ 对实数跨步视图**稠密写**，矩阵 C 全失败，根因 = 新 **FIX-ITEM-5**（双树同型污染归因闭环）；native_ref 在注册下即当前路径（HtoD-staging 0.69–0.73 ms，自洽） |
| §3 FIX-ITEM-4 独立验证 | 新增 `benchmarks/u8sum_repro.py` 按评审清单直取**值+dtype** 双树对照（1d/多维、默认/显式 dtype、>255 累加、empty、reduction 维、连续/非连续） | **修复树 6/8 失败、修复前对照树同型 6/8 失败**（大累加/2d/非连续回绕+dtype 错；4 元素小例侥幸正确）——预存归因闭环，checksum 回绕仅作发现线索 |
| §4 复杂度评估 | A 即"A 快路径+B 兜底"组合；注册实测逐布局对比 + guard 分支 µs 级实测 | **B-only 建议为首选**（注册下全布局占优或持平、0.005–0.007 ms 接近 native、复杂度最低）；A 快路径仅 c128 contiguous 一处收益 ~3µs，作为可选增强 |

"B 全布局最优"已限定为"**在本轮已测试布局中最优**"。功能 fix 维持
`draft / deployable=false / hardware_status=unverified`。

## 第十二轮：harness 缺陷修正与严格比较器定稿（2026-11-11，意见 comment `6793416880`，基于 `4eb4cb6`）

三项 P2 全部落地（harness 修正提交 `51642d8`/`fc71b82`，干净提交重跑为最终批）：

| 评审项 | 处理 | 最终结果（干净提交 `fc71b82` 重跑） |
| --- | --- | --- |
| P2-c dtype 贯穿 | `zero_candidates.py` 与 `baseline_freeze.py` 全部构造路径按传入 dtype 组装（c64→float32、c128→float64）；执行前断言并记录 actual_dtype/conj/stride/offset；设备标量按 dtype 匹配 | 原生候选矩阵 **260/260**（真实 c64/c128）；注册 **224 过 + 36 C 失败**（全部为 FIX-ITEM-5 稠密写，含新建 contiguous 的 C 内部跨步 select 视图）；c128 性能/路径为真实 float64 数据 |
| P2-b uint8 小样例 | `u8sum_repro.py` 显式 `dtype=torch.uint8` 并断言；非连续用例先建 uint8 base 再切片断言非连续 | **修复树 7/8 失败、prefix 同型 7/8**（真 uint8 下连 4 元素小例也回绕 357→101；仅显式 int64 正确）——"4 元素小例侥幸正确"与尺寸分支归因**撤回**；仅非连续用例按评审改为 uint8 base 直切片 |
| P2-a 严格比较器 | `compare_case` 重写：numeric 比较要求双方 status=ok + 全部适用 checksum 字段（grad 按 grad_applicable 显式 N/A）齐全；双 exception/单 exception/缺参考/缺字段为独立类别；checksum 改**全字节 sha256** + dtype/shape 检查；abs 前向改用梯度克隆（修正 10 行 spurious exception）；负例自测 8 项（双 exception/缺字段/仅第 32 字节后差异/dtype 不符等均不得判过）随 --matrix 前置强制执行 | 基线重跑定论：**50 numeric_equal + 8 numeric_diff**；8 例全部量化（`baseline/diff-details.json`，实际输出计算）：conj sum/abs ≤**2 ULP**（归约顺序）、abs scaled-hypot **1 ULP 级**（`abs/c64/extreme_low` max_abs=1.4e-45、`abs/c128/extreme_high` rel=1.69e-16）、NaN 传播正确但 **payload 位不保证**（sha256 位级不同、数值均为 NaN）——全部属 round-2/3 已声明的"逐位不保证"舍入类，无功能问题 |

~~评审此前对"52/58 位级一致"的认可~~ 由评审自行更正撤回；该口径作废，替换为上行定论。registered 模式 `native_ref` 更名说明为 **registered-path reference（当前修复路径）**（基线的 native 参考本就来自独立未注册进程，命名问题不否定其正确性检查）。历史原始 JSON/日志保留，`bench-r2/r12-superseded-manifest.md` 标注新旧批次对应。B-only 仍为优先候选；真实 c64/c128 布局/0 维/空/conj/alias/未选中 storage 检查已按本批补齐后，再评估是否单独提交优化 fix；性能区分标量构造成本（单次 0.336 ms 含冷启动 / 50 次平均 0.0145 ms）、稳态拷贝净成本（0.0071/0.0069 与 0.0089/0.0067 双批）与整体调用成本（0.005–0.009 ms），注册路径事件已验证。三平台要求属 PR3。功能 fix 维持 `draft / deployable=false / hardware_status=unverified`。

## 第十三轮：ULP dtype-aware 修正与证据同步收尾（2026-11-11，意见 comment `6794976151`，基于 `4c588b3`）

两项 P2 落地（diff_details ULP 修正提交 `32ee6a9`，干净提交重跑）：

| 评审项 | 处理 | 结果 |
| --- | --- | --- |
| P2-b ULP 映射按 dtype | `_ulp_distance(n, f, float64)`：c64 恒用 float32 位型、c128 恒用 float64 位型（旧按数值大小选择使 c128 落 float32 范围即误报 0 ULP）；NaN/Inf 不入 ULP；**ULP 映射自测**：c128 亚 float32-ULP 差异 → 1 float64 ULP（旧映射报 0）、f32 1-ULP 差异按 f64 映射 → 536870912（演示两映射不可混用） | `baseline/diff-details.json` 重生成：c64 extreme_low **1 ULP**（float32）、c128 extreme_low **1 ULP** / extreme_high **1 ULP**（float64）、c128 conj sum **4 ULP** / abs **2 ULP**（float64）、c64 conj **2 ULP**；NaN 数值全同、payload 位不保证 |
| P2-a 结果同步 | 严格比较最终批产物拷入 `baseline/`（50 numeric_equal + 8 numeric_diff 的 summary/table/三组 JSON），round-10 旧产物以 `.r10-superseded` 后缀归档；README 两处旧"52/58"加 superseded 标注并替换为当前口径；`run-commits.json` round13_additions 记录 commit↔文件↔批次映射 | grep 复核：旧 52/58 仅存于带 superseded 标记处与归档文件；`baseline-freeze-final.log` ↔ `baseline/*`(strict) ↔ `diff-details.json` ↔ run-commits 一一对应 |

评审前轮认可撤回事项已按第七/十二轮处理。B-only 仍为 FIX-ITEM-3 优先候选；
PR2 范围维持 PPU/thead；三平台属 PR3。功能 fix 维持 `draft / deployable=false /
hardware_status=unverified`。

## 第十四轮：NaN 对照同输入化、分类完备指标与 ULP 有序映射修正（2026-11-11，意见 comment `6797761554`，基于 `ff673ce`；harness 修正提交 `7a691c8`）

两项 P2 全部落地：

| 评审项 | 处理 | 结果 |
| --- | --- | --- |
| P2-a NaN 差异解释用不同输入 + 分类不完备 | `diff_details.py` 重写：NaN 用例复用**基线构造**（NaN + NaN·j，经 full(nan)/full(nan·0.5)）；逐分量**分类完备**（nan/inf+/inf−/zero±/finite），类不一致（含 Inf 符号、NaN vs 非 NaN）计为**失败**而非跳过；NaN payload 用**实际位型**比较（repr 不可分辨）；输入/输出位型与 in/out_bits_equal 全记录；修复合并探针暴露的分类 bug（exp_mask 未按 dtype、mant 误含指数位 → inf 曾误判 nan） | **自测 17/17 全过**（含评审的两个假阴性复现：`1.0 vs inf`、`(1+nanj) vs (1+0j)` 现均判 class_mismatch；±inf 符号不匹配判失败；NaN payload 位差异经 bits 标记）。`diff-details.json` 重生成：**8 例全部解释**——7 例 1–4 ULP 舍入类（conj c128 4/c64 2、abs extreme 1、abs conj 2）；**abs/c128/nan = 输出 NaN payload 位型差异**（scaled-hypot vs native 生成；**输入位级一致 in_bits_equal=True、双方均 NaN、掩码一致**——按策略 payload 不作为通过条件）。"待确认"项已解释完成 |
| P2-b ULP 有序映射跨符号/±0 | `_ulp_distance` 改为**标准有序位型映射**（正数过中点、负数镜像），显式 **±0 策略**（数值 ULP 同零点距离 0，符号位差异单列 flag）；selftest 补**精确预期断言**：f32/f64 的 ±0、±min_sub、相邻值（含负数相邻）、跨零最小次正规数（距离=3），不再只断言 ≥1；并修正 selftest 标签错配（f64 相邻对被标成 float32 1-ULP，改用真正 f32 相邻对断言 ==1，另保留旧对 f32 映射=0 的检查） | 精确断言全过：f32 `+0.0→0x80000000`、`−0.0→0x7FFFFFFF`、`±min_sub` 距离=2（跨零=3）、f64 同构；同号有限值 1–4 ULP 结论维持（c128 conj 4 ULP 为 float64 位型真实值） |

历史 `diff-details`（round-11/12 批）标注 superseded（`r12-superseded-manifest.md` 追加）；8 项 numeric_diff 的定性由"1 ULP 级"细化为**按 dtype 位型的 1–4 ULP**。功能 fix 维持 `draft / deployable=false / hardware_status=unverified`。

## 第十五轮：分类负例、位级归档与退出码修正（2026-11-12，意见 comment `6811732879`，基于 `bf71431`；工具重写提交 `967e1f8`）

三项 P2 全部落地：

| 评审项 | 处理 | 结果 |
| --- | --- | --- |
| P2-a −inf 误判 NaN | `classify()` 改为**符号/指数/尾数独立提取**（尾数掩码 = `sign−1−exp_mask`，排除符号位；此前 `bits & ~exp_mask` 保留符号位 → −inf 尾数非零被误判 NaN） | 自测覆盖 f32/f64 ±Inf、±NaN（符号位/载荷）、−Inf↔NaN、±Inf 符号翻转——**全部精确分类**；`-inf vs NaN` 判 class_mismatch 而非 payload flag |
| P2-b 验证失败不聚合、退出码恒 0 | `main()` 聚合 hard failures（子进程 rc≠0/缺结果、in_bits 不一致、长度不符、分类不一致、缺字段），写诊断后**非零退出**；`run_child` 检查 returncode；`aggregate_failures` 纯函数 + **入口级负例**（注入分类不一致 → 聚合非零；干净用例 → 0）`--exit-code-negative-test` PASS | 实跑 `hard_failures: 0`、rc=0（干净基线）；负例注入验证通过 |
| P2-c 原始位型未归档、payload 经 repr | 逐分量比较**直接从子进程返回的 raw bits** 进行（NaN payload 不经 repr）；每用例归档 `raw_in_bits_{native,fix}`、`raw_out_bits_{native,fix}`、dtype/shape/byteorder/输出 sha256/raw-hex 前缀；`abs/c128/nan` 与 **round-10 提交 checksum 交叉核对**（该用例输入为确定性 full 构造） | abs/c64/extreme_low **双树均复现 r10 位级输出**（确定性 full 输入）；conj 用例输入依赖 RNG 状态（构造代码相同、生成状态不同）——**同 run 内 native vs fix 对比（in_bits_equal=True）为有效检验**，与 r10 checksum 的差异属新批次输入差异，已在 JSON `batch_reproduction_note` 说明 |

结论口径更新：8 项 numeric_diff 定稿为 **conj sum/abs ≤4 float64 ULP（c128）/ ≤2 float32 ULP（c64）、abs extreme 1 ULP、abs/c128/nan = 输出 NaN payload 位型差异（输入位级一致、双方均 NaN、掩码一致——按策略 payload 不作为通过条件）**；"8 例全部解释完成"表述替换为上述准确状态。功能 fix 维持 `draft / deployable=false / hardware_status=unverified`。

## 第十六轮收尾：诊断必写、入口负例与基线说明对齐（2026-11-12，意见 comment `6813485321`，基于 `ed1b7d0`；工具提交见后续 `git log`）

两项 P2 收尾落地：

| 评审项 | 处理 | 验证 |
| --- | --- | --- |
| P2-a 异常输入下诊断可能不写/KeyError | `ulp_metrics` **统一 schema**：所有返回路径（长度不符/缺字段）均带全字段默认值；**必需字段集中校验**（out_bits/in_bits/values/out_is_complex 缺失 → 聚合 hard failure 而非抛异常）；`process_case` 抽取为函数并置于 try/except——**诊断 JSON 必写**、失败聚合后**非零退出** | 入口级负例 `--exit-code-negative-test` 扩展为三场景：**长度错误 / 缺字段 / 子进程失败**全部聚合非零且诊断 JSON 写出——PASS；selftest 16 项继续全过 |
| P2-b 基线说明对齐当前 strict baseline | `batch_reproduction_note` 与相关文档改为：**确定性用例匹配 strict round-12/13 baseline**（abs/c64/extreme_low 双树位级复现）；**conj 用例不与旧批次 hash 直接比较**（RNG 生成状态不同），有效检验为**同批 native/fix 输入位级一致**（in_bits_equal=True）；c128 r10 条目系 float32 误标不可比 | `diff-details.json` 重生成（结构未变，仅注记），strict 复现条目确认 |

收尾后：hard_failures=0、raw bits 与 sha256 复核通过；功能 fix 维持 `draft /
deployable=false / hardware_status=unverified`。工具收尾后按评审建议的顺序，
后续在 PPU/thead 范围做针对性正确性与性能复测（complex strided `zero_`、
稳定 `abs`、dtype/out 语义，小模型 p50/p95 固定输入多次重复）；B-only 仍为
FIX-ITEM-3 优先候选；三平台属 PR3/problem29。

## 第十七轮：批次对照切换至当前 strict baseline（2026-11-12，意见 comment `6816060628`，基于 `867c4b3`；提交 `b26e84f` + 清理 `cd5c2d6`/`7e39f36`）

一项 P2 口径同步落地（按评审方案 1）：`diff_details.py` 的批次对照从
`.r10-superseded` 归档切换为**当前 strict baseline**（`baseline-n.json` /
`baseline-fix.json`），字段更名为 `batch_reproduction_vs_strict`
（`native_matches_strict` / `fix_matches_strict`），比较口径为 **sha256**
（strict baseline 的 checksum 即 round-12 版本的全字节 sha256）。`batch_reproduction_note` 同步改写。

| 用例 | native 匹配 strict | fix 匹配 strict |
| --- | --- | --- |
| abs/c64/extreme_low（确定性 full 输入） | ✓ | ✓ |
| abs/c128/extreme_low（真实 float64） | ✓ | ✓ |
| abs/c128/nan | ✓ | ✓ |
| abs/c128/extreme_high | ✓ | ✓ |
| sum+abs conj（RNG 依赖输入） | ✗（新批次输入） | ✗（新批次输入） |

- 4 例确定性用例**双树 sha256 与 strict baseline 完全一致**（位级复现）；
- conj 用例构造代码相同但 RNG 生成状态不同——不与任何存储批次 hash 直接
  比较，有效检验为**同批 native vs fix 输入位级一致**（in_bits_equal=True）；
- c128 的 r10 条目系 float32 误标（round-12 修正），其 checksum 与真实
  c128 输出不可比——已在 note 说明；
- `.r10-superseded` 归档仅作历史 raw 证据保留（manifest 已标注）。

**流程披露**：round-17 提交 `b26e84f` 因 `git add -A` 误将工作树中其他并行
修复包（adamw-fused / flaggems-selective-enable / gelu-kunlunxin-pow /
kunlunxin bundle/docs 等 121 文件）带入分支；后续提交 `cd5c2d6`/`7e39f36`
已将其从分支移除（文件保留在磁盘未跟踪，其他工作不丢失），共享文件
（CHANGELOG/README/catalog/docs）恢复至误提交前状态。当前分支相对 `867c4b3`
仅含 `benchmarks/diff_details.py` 与 `baseline/diff-details.json` 两个预期
文件。功能 fix 维持 `draft / deployable=false / hardware_status=unverified`。

## 未执行（如实登记，不算通过）

- **problem.md 容器整模型回归**：`dockerexec xbench-ppu … run_model.py --models
  "togethercomputer/m2-bert-80M-8k-retrieval" … --result-root …` 需要目标容器
  （x-benchmark、`/home/wuyiqiang/x-benchmark`、dockerexec），本环境没有；因此
  `hardware_status` 保持 `unverified`、`deployable=false`、`owner=null`，
  `flag-fixes apply` 按设计拒绝。
- **上游 PR**：未向 flagos-ai/FlagGems 提交 PR（跟踪 issue #7181）。
- **性能基准**：崩溃修复，无同口径修复前耗时（修复前该路径根本跑不完），
  `benchmarks/` 不伪造数据。（第四轮已完成 scoped 的小模型口径性能评估，见上节；
  容器整模型同口径仍属未执行。）
- **未修的 complex 缺口**：`mean`/`full`/`ones` 及 `ComplexMode=NONE` 的 pointwise
  （`eq`/`where` 等）不在本 patch（m2-bert 反向不需要），见 `reports/diagnosis.md`。
