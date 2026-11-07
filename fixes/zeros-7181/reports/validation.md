# 验证记录

- 修复 ID：`zeros-7181`；验证日期：2026-11-07（本机 PPU 环境实测）。
- `hardware_status: unverified`：problem.md 指定的 x-benchmark 容器整模型 on 趟回归未在
  本环境执行（见文末「未执行」），下述全部为本地双树对照证据。
- 三个硬件脚本只有退出码 0 才算通过；缺设备 / 缺 `transformers` 一律退出码 2，
  **skip 不是通过**。

## 环境与源码

| 项 | 值 |
| --- | --- |
| FlagGems 基线 | commit `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`git describe` = `v5.3.5-46-g29e14822`） |
| 硬件 / vendor | PPU-ZW810E，FlagGems vendor 解析为 `thead`，设备/dispatch 标识为 `cuda`（`cuda` 只是本环境设备标识，不代表已验证 CUDA 厂商后端） |
| 软件栈 | torch 2.10.0 / triton 3.6.0 / Python 3.12.3 / transformers 4.57.0 / 驱动 `ppu-smi` |
| 模型 | `togethercomputer/m2-bert-80M-8k-retrieval`（`trust_remote_code`，随机初始化，不下载权重，`HF_ENDPOINT=https://hf-mirror.com`） |
| 管理仓库 | `/root/flag-fixes`，分支 `zeros-7181`；dirty 验证时 HEAD `f159ce07b199`（`source_dirty: true`） |

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
| `/tmp/opencode/flagfix-fixed` | e2ebase + `zeros-7181.patch` 全量（9 文件） | e2e 修复后 |

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
且与 `fixes/zeros-7181/src/` 快照逐字节 `cmp` 通过。

## 命令与结果

下文 `$FIX` = `/root/flag-fixes/fixes/zeros-7181`，`$T` 为对应树根；
硬件脚本统一形如 `PYTHONPATH=$T/src python3 $FIX/tests/<script>`。
所有日志为工具原文、未脱敏，已存 `results/`（文件名见各 log 内首行命令与下表）。

| 命令 / 脚本 | 修复前（`flagfix-noncplx` / 基线） | 修复后（`flagfix-fixed` / 基线） |
| --- | --- | --- |
| `tests/repro_complex64.py`（issue #7181 形态 + torch 入口） | **exit 1**：`ISSUE REPRODUCED`，`KeyError: 'complex64'` → `results/repro-baseline-2026-11-07.log` | **exit 0** |
| `tests/test_complex_ops.py`（54 用例矩阵） | **exit 1**：`TOTAL: 54 passed, 38 failed`（38 条全是 complex KeyError）→ `results/regression-baseline-2026-11-07.log` | **exit 0**：`TOTAL: 54 passed, 0 failed` → `results/regression-fixed-2026-11-07.log` |
| `tests/test_m2_bert_backward.py`（12L/768/12/3072/8192，问题原配置） | **阶段一** `flagfix-e2ebase` exit 1：`_thead/ops/linear_backward.py:170` view `RuntimeError` → `results/e2e-linearblocker-baseline-2026-11-07.log`；**阶段二** `flagfix-noncplx` exit 1：前向 `FORWARD OK` loss rel=0.0，反向 `ops/sum.py:385 → KeyError: 'complex64'`（= problem.md 签名）→ `results/e2e-problem-path-noncomplex-2026-11-07.log` | **exit 0**：`E2E OK: loss_native=1.000000000e+00 loss_gems=1.000000000e+00 rel=0.000e+00 grads_present_finite=161 grads_skipped_known_gap=384` → `results/e2e-fixed-2026-11-07.log` |
| 同脚本，强制无设备（`CUDA_VISIBLE_DEVICES=`） | exit 2（缺设备） | exit 2（缺设备） |
| 同脚本，卸掉 `transformers` 后 | exit 2：`MISSING DEPENDENCY: transformers -> exit 2` → `results/e2e-missing-dep-2026-11-07.log` | exit 2 |
| e2e 判据（修复后日志逐条满足） | — | loss 快照 rel ≤ 1e-3（实际 0）；反向完成；除文档化 `filter_fn.*` 缺口外原生产出的梯度全部存在且有限（161 个），无 flag_gems 独有梯度；梯度数值仅打印不判据 |

oracle 口径：native 参照在 `flag_gems.enable()` **之前**捕获（enable 后 call-site 的
`torch.*` 已被 dispatch 进 flag_gems，不能当 oracle）；complex 比较一律经
`torch.view_as_real`（flag_gems `eq`/`nonzero` 不支持 complex，属已记录的预存缺口）。

### 梯度缺口对照（不属于本 patch 的预存缺陷）

用 `flag_gems.enable(unused=[...])` 排除 `sum/zeros/abs/fill` 系后反向可以在两棵树上跑完，
逐参数对照（日志：`results/gradgap-probe-noncomplex-2026-11-07.log`、
`results/gradgap-probe-fixed-2026-11-07.log`，1L 配置）：

| 观察项 | 无 complex 修复树 | 修复树 | 结论 |
| --- | --- | --- | --- |
| 缺失梯度集合 | `missing=34`（32× `filter_fn.*` + 2× 未使用 pooler） | 完全相同 `missing=34` | flag_gems 全包 0 个 `register_autograd`，`aten::pad`/`aten::conv1d` 走 autograd-not-implemented 回退 → 预存、与本 patch 无关 |
| 梯度数值分歧 | native ~1e-9 vs flag_gems ~0.5 量级（近驻点参数），另有 native 有值 / flag_gems 为 0 项 | 同形态、同参数集合 | 预存、与本 patch 无关 |
| loss 张量回读（个别 run） | 不适用（反向跑不完） | 部分 run `backward()` 后 loss 存储被覆写（间歇） | 预存内存异常，无法修复前 A/B；e2e 因此在反向**前**快照 loss 为 Python float |

## 管理工具执行结果

均在 `/root/flag-fixes` 下、`PYTHONPATH=src` 执行，2026-11-07：

| 命令 | 退出码 | 结果 |
| --- | --- | --- |
| `flag_fixes validate` | 0 | `{"valid": true, "fixes": 11, "deployable": 0, "hardware_verified": 0}` |
| `flag_fixes check zeros-7181 --target $BASE` | 0 | 8 目标 before hash 全中、patch `git apply --check` 干净 |
| `flag_fixes plan zeros-7181 --target $BASE` | 0 | 同上（预演不落盘） |
| `flag_fixes apply zeros-7181 --target $BASE` | 2 | `error: Selected fixes are not approved for deployment`（`deployable=false`，**按设计拒绝**） |
| `flag_fixes status --target /tmp/opencode/flagfix-fixed` | 0 | `{"transactions": [], "drift": [], "recovery_required": false}` |
| `flag_fixes verify zeros-7181 --target /tmp/opencode/flagfix-fixed --python python3` | 0 | `passed: true`，3/3 命令 `returncode=0`（repro / regression / e2e），耗时 20m01s，`source_dirty: true`，`source_commit: f159ce07b199` → `results/verify-2026-11-07-dirty.json` |

`verify` 已在**已提交、干净**工作树上重跑的结果见 `results/verify-2026-11-07.json`。

## 未执行（如实登记，不算通过）

- **problem.md 容器整模型回归**：`dockerexec xbench-ppu … run_model.py --models
  "togethercomputer/m2-bert-80M-8k-retrieval" … --result-root …` 需要目标容器
  （x-benchmark、`/home/wuyiqiang/x-benchmark`、dockerexec），本环境没有；因此
  `hardware_status` 保持 `unverified`、`deployable=false`、`owner=null`，
  `flag-fixes apply` 按设计拒绝。
- **上游 PR**：未向 flagos-ai/FlagGems 提交 PR（跟踪 issue #7181）。
- **性能基准**：崩溃修复，无同口径修复前耗时（修复前该路径根本跑不完），
  `benchmarks/` 不伪造数据。
- **未修的 complex 缺口**：`mean`/`full`/`ones` 及 `ComplexMode=NONE` 的 pointwise
  （`eq`/`where` 等）不在本 patch（m2-bert 反向不需要），见 `reports/diagnosis.md`。
