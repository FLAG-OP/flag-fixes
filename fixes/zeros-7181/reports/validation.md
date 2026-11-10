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
| **性能（首轮只报告，不作加速声明）** | 75 用例（sum 全/内/外、abs、zeros/zero_，c64/c128 × 2^10/2^16/2^20 × 连续/跨步 + fwd+bwd；模型级 fwd/fwd+bwd/整训练步）：本构建上 f_target 的 complex sum/abs **慢于**原生（sum 3–21×、abs 1.06–1.7×），zeros/zero_ 持平到略快（0.83–0.96×）；模型级 f_target 1.07–2.36×；f_all/b 可跑路径 10.5–12.7× 慢；>10% 稳定回归 58 项已打旗标；崩溃路径一律记 N/A 不计加速 |

产物：`results/small-model-2026-11-10/`（`matrix/`=阈值+校准+29 run+20 对比+fd+summary，
`bench/`=5 组原始块计时+汇总表，`mse-scalar-repro.log`，`provenance.json`）；
矩阵跑在干净提交 `968d624`，bench 跑在 `d236d5a`（两者 `tests/small_models.py`/
`tests/test_small_models.py` 逐字节一致，`git diff` 为空）；完整报告
`reports/small-model-evaluation.md`。结论限定于这两个模型与当前后端构建。

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
