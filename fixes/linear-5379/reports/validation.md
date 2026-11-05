# 验证记录

当前：**done（本机设备三脚本全部通过；平台容器回归 not-run，见文末）**。
执行日期 2026-11-04（UTC 14:08–14:16，初版三脚本验证）与 2026-11-05（已提交、干净工作树上的重跑），
执行者：本仓库维护者（自动化会话，命令与原始输出逐条可复跑）。表中未标注日期的"修复前/修复后"数值
均来自 2026-11-04；2026-11-05 的重跑结果在对应行与"verify 结果文件"一节单独标注。

## 环境与源码

| 项 | 值 |
| --- | --- |
| 设备 | PPU-ZW810E ×3（用 GPU 0），`nvidia-smi` 驱动 1.5.1-1d747a，`torch.cuda.is_available()=True` |
| 软件栈 | Python 3.12.3，torch 2.10.0，triton 3.6.0，transformers 4.57.0（仅 e2e 的 SigLIP 用例） |
| FlagGems 基线 | commit `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`git describe` = `v5.3.5-46-g29e14822`） |
| 基线选取依据 | 该 commit 时间 2026-09-01，与 problem.md live 复现同日；`src/flag_gems/ops/linear.py` 在 `41ace9c3..e5a396b4` 区间内容不变，line 158 即 `input_flat = input.view(M, K)`，与容器内目标文件一致 |
| 目标文件 | `src/flag_gems/ops/linear.py`，before `4d46bae0804ac32b698582d010d5f0e02d3570c6703efff9d1d666e2bf43fd94`，after `ddec3383c1d6ca4b2f53cf6ea9550c2d5fead492a2562e0864e723b958ee951d` |
| patch | `patches/linear-5379.patch` sha256 `7055d63218f73529fc0de4012def51275235c2e6c0e5ed49da6253f56f053773` |
| 管理仓库（2026-11-04 首跑） | flag-fixes `984891ebf6449c398329ab8eddd4a0f9283130be`，本 fix 目录当时**未提交**，`verify` 记录 `source_dirty=true` → `results/verify-2026-11-04.json`（历史记录，原样保留、未改写 SHA） |
| 管理仓库（2026-11-05 重跑） | 本 fix 已提交、工作树干净，`verify` 记录 `source_dirty=false`、`source_commit` 为当时包含本 fix 的提交（具体哈希见该 JSON）→ `results/verify-2026-11-05.json`，由后续提交保存 |
| 执行入口 | 三个测试脚本不依赖工作目录；2026-11-04 首跑用 `cwd = fixes/linear-5379`，2026-11-05 重跑用脚本绝对路径。`PYTHONPATH=<target>/src`（`verify` 由工具注入同名变量）；`--root` 始终指向管理仓库根 |
| 运行时 vendor | FlagGems 解析 `vendor_name='thead'`, `device_name='cuda'`, `device_query_cmd='ppu-smi'`。**`cuda` 是本 PPU 环境的设备/dispatch 标识，不代表已验证所有 CUDA 厂商后端**；硬件为 PPU-ZW810E |

目标树构造（可重建）。下表与下文命令用 `$TARGET_BASE` / `$TARGET_FIXED` 指代目标树，
其取值是本机实际路径，换机器时改成自己的目录即可：

```bash
MANAGER=/root/flag-fixes               # 本仓库根（换机器时改成本地 clone 路径）
TARGET_BASE=$MANAGER/../FlagGems-base  # 修复前树；本机实际为 /tmp/opencode/FlagGems-base
TARGET_FIXED=$MANAGER/../FlagGems-fixed# 修复后树；本机实际为 /tmp/opencode/FlagGems-fixed
git -C <FlagGems> worktree add "$TARGET_BASE"  29e14822f6dfabf990973c41f7c2df7811a6f60f  # 修复前
git -C <FlagGems> worktree add "$TARGET_FIXED" 29e14822f6dfabf990973c41f7c2df7811a6f60f  # 修复前基线
git -C "$TARGET_FIXED" apply "$MANAGER/fixes/linear-5379/patches/linear-5379.patch"      # 打上 patch
```

## 命令与结果

| 项目 | 精确命令 | 判定标准 | 修复前 | 修复后 | 证据 |
| --- | --- | --- | --- | --- | --- |
| 最小复现 | `PYTHONPATH=$TARGET_{BASE,FIXED}/src python3 tests/repro_linear_view.py` | 退出码 0 且打印 `OK`；修复前必须以 `linear.py:158` RuntimeError 非零退出 | **exit 1**：`File ".../ops/linear.py", line 158, in linear / input_flat = input.view(M, K)` → `RuntimeError: view size is not compatible with input tensor's size and stride ...`；输入 `(2,16,64) stride=(1024,1,16)` | **exit 0**：`out (2,16,32) mse=2.666e-11 max_diff=2.861e-05`，打印 `OK` | `tests/repro_linear_view.py`；traceback 原文见下 |
| 回归/边界 | `PYTHONPATH=$TARGET_BASE/src python3 tests/test_linear_noncontig.py` | `TOTAL: ... 0 failed`（3 dtype × 9 布局 × dispatch/direct） | **exit 1**：`TOTAL: 20 passed, 30 failed`，30 条全部是同一 `view size is not compatible` （transpose/slice/permute/expand/4d/无 bias/gemv 布局 × dispatch 与 direct） | **exit 0**：`TOTAL: 50 passed, 0 failed` | 脚本内逐用例打印 mse/max_diff |
| 调用方 / e2e | `PYTHONPATH=$TARGET_BASE/src python3 tests/test_linear_callers.py` | `TOTAL: 0 failed`（NHWC MLP、seq-first MLP、随机初始化 SigLIP 视觉塔）；缺依赖必须非零退出 | **exit 1**：第一个用例即在 `ops/linear.py:158` 抛同一 RuntimeError | **exit 0**：`TOTAL: 3 passed, 0 failed`，mse 依次 8.838e-15 / 4.177e-14 / 8.106e-13 | `tests/test_linear_callers.py`（SigLIP 随机初始化，不下载权重） |
| e2e 缺依赖（skip ≠ pass） | 同上，`PYTHONPATH` 前置一个只抛 `ImportError` 的同名 `transformers` 模块模拟缺依赖 | 必须非零（**exit 2**）且打印 `MISSING DEPENDENCY`，不得因前两项通过返回 0 | — | **exit 2**：`TOTAL: 2 passed, 0 failed`，stderr `MISSING DEPENDENCY: transformers - SigLIP caller case not run; a skip is not a pass (exit 2)` | 原始输出 `results/e2e-missing-dep-2026-11-05.log` |
| 精度 | 同上（脚本内 oracle 均在注册前用原生 torch 计算，比较在 CPU 上进行） | fp32 `allclose(rtol=1e-3, atol=1e-3)`；连续对照组修复前后逐位相同 | 连续对照组 max_diff=0（通过） | fp32 dispatch/direct 全绿；连续对照组 max_diff=0，即修复未改变连续路径 | 修复后 repro mse 2.67e-11；`flag_gems.linear` vs 原生 |
| 性能 | 不适用 | 非性能修复：连续路径无额外拷贝（`reshape` 返回视图 + `contiguous()` 空操作），非连续路径修复前根本无法完成计算，无法给出同口径前后耗时 | not-applicable | not-applicable | `benchmarks/` 保持模板状态，不伪造耗时 |
| 清单与基线校验 | `PYTHONPATH=src python3 -m flag_fixes --root . validate` | exit 0，`valid: true` | — | **exit 0**：`{"valid": true, "fixes": 10, "deployable": 0, "hardware_verified": 0}` | 管理仓库 CLI |
| 安装预演 | `PYTHONPATH=src python3 -m flag_fixes --root . check linear-5379 --target $TARGET_BASE`（`plan` 同结果） | exit 0：基线 hash 命中、`git apply --check` 干净、patch 后 hash 与声明一致 | — | **exit 0**，输出 `before_sha256=4d46bae0…` / `after_sha256=ddec3383…` / `deployable: false` | `check`、`plan` 两次运行输出一致 |
| 部署拒绝检查 | `... apply linear-5379 --target $TARGET_BASE` | 必须拒绝（deployable=false） | — | **exit 2**：`error: Selected fixes are not approved for deployment` | 未产生任何安装状态 |
| 状态/回滚 | `... status --target $TARGET_BASE` | 无安装记录、无漂移 | — | **exit 0**：`{"transactions": [], "drift": [], "recovery_required": false}` | 目标树保持修复前内容 |
| 工具级验收 | `PYTHONPATH=src python3 -m flag_fixes --root . verify linear-5379 --target $TARGET_FIXED --python python3` | JSON `passed: true`，三个命令 `returncode=0` | — | **exit 0**，`passed: true`；repro/regression/e2e 分别 rc=0（两次运行均如此） | 原始输出 `results/verify-2026-11-04.json`（历史，dirty=true）与 `results/verify-2026-11-05.json`（干净重跑，见下节） |
| 容器整模型回归（x-benchmark） | problem.md 的 `run_model.py --models google/siglip-base-patch16-224 ... --result-root /data/ppu-benchmark/results_top515_repro` | on 趟 Success（且必须用独立 `--result-root` 绕过已有 on-Success 缓存） | not-run（本环境无该容器与运行器） | not-run（同左） | 见下方 not-run 说明 |

### 修复前 traceback（`tests/repro_linear_view.py`，基线树）

2026-11-05 在 `$TARGET_BASE` 上复跑，**以下为未脱敏、未省略的原文**（完整日志：
`results/repro-prefix-2026-11-05.log`；2026-11-04 首跑的同一失败在表中"最小复现"行记录，
失败点与错误文本一致）：

```
ISSUE #5379 REPRODUCED: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
input (2, 16, 64) stride=(1024, 1, 16) contiguous=False
flag_gems from: /tmp/opencode/FlagGems-base/src/flag_gems/__init__.py
Traceback (most recent call last):
  File "/root/flag-fixes/fixes/linear-5379/tests/repro_linear_view.py", line 36, in <module>
    out = torch.nn.functional.linear(x, w, b)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/opencode/FlagGems-base/src/flag_gems/ops/linear.py", line 158, in linear
    input_flat = input.view(M, K)
                 ^^^^^^^^^^^^^^^^
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
```

## verify 结果文件的来源与追溯

| 文件 | 运行时的管理仓库状态 | 说明 |
| --- | --- | --- |
| `results/verify-2026-11-04.json` | HEAD `984891ebf6449c398329ab8eddd4a0f9283130be`，本 fix 目录未提交，`source_dirty=true` | 2026-11-04 的**历史记录，原样保留**：其中的 `source_commit` 不含当时未提交的 fix 文件，单凭该 commit 无法还原运行内容，不能作为"已提交源码"的证据；按审阅意见不改写其中任何字段 |
| `results/verify-2026-11-05.json` | 本 fix 已提交、工作树干净，`source_commit` 为当时 HEAD（含本 fix），`source_dirty=false` | 2026-11-05 在**已提交、干净**源码上重跑，作为"可复跑"证据；结果文件随后续提交入库 |

两份 JSON 都是 `flag-fixes verify` 的原始输出，含本机绝对路径，**未做脱敏、未做任何修改**。

## 普通 CI 的覆盖边界

普通 CI（ubuntu runner，无设备）对本 fix 覆盖：`flag-fixes validate`（schema、patch hash、
快照 hash、脚本存在性）+ `python -m pytest -q` + `python -m compileall -q fixes`（本地 fix 脚本
语法检查）。**三个硬件脚本不进普通 CI**，只在带 PPU/CUDA 设备的环境运行；语法检查通过
≠ 入口执行通过 ≠ 硬件验证通过，`hardware_status` 保持 `unverified`。

## 未执行、不适用与已知限制

- **not-run：容器侧整模型回归。** 本环境没有 `/data/ppu-benchmark/FlagGems`、
  `/home/wuyiqiang/x-benchmark` 与模型缓存，problem.md 的 `run_model.py` on 趟无法执行。
  该步是平台验收依据，`validation.hardware_status` 因此保持 `unverified`；
  problem.md 记录的 18 视觉模型 36/36 翻转（2026-09-01 live）作为容器侧证据写入 `fix.yaml.evidence`，
  但不冒充本环境的实测结果。
- **not-run：`flag-fixes apply` 与真实回滚事务。** `deployable=false`（无 owner、未过平台验收），
  apply 被工具按设计拒绝；已验证拒绝行为与 `status` 干净，未在任何目标上写入安装状态。
- **not-applicable：性能基准。** 理由见上表"性能"行。
- **失败中发现的既有缺陷（非本修复引入）**：fp16/bfloat16 在 `N=32, K=64`、`M ∈ {5,6,7,8}`
  输出完全错误（maxdiff ≈ 2.0e1~3.4e1），修复前后、连续与非连续均同样复现；回归矩阵只在
  fp32 上断言该 M 段，脚本内有注释，`reports/diagnosis.md` 有 M 扫描数据。
- **覆盖面**：只验证了通用 `src/flag_gems/ops/linear.py`；`_enflame/*`、`_iluvatar` 的 linear
  覆写与各处 linear_backward 未包含、未验证（见 diagnosis 已知限制）。
- 单项验收不代表组合通过：本次未验证与其他 fix 的 bundle 组合。
- **原始输出未脱敏**：本页引用的日志与 JSON 均为原文（含 `/root/flag-fixes`、`/tmp/opencode/...`
  等本机绝对路径）；上文 traceback 与命令表中的 `$TARGET_*` 只是把可变路径参数化，未改动任何
  运行结果或字段。
