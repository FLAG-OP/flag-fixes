# linear 非连续输入 view 崩溃修复（view→reshape）

- ID：linear-5379
- 状态：draft（deployable=false，未完成平台验收与 owner 审核）
- 负责人：待填写
- 上游 issue/PR：https://github.com/flagos-ai/FlagGems/issues/5379 （PR 未提交）
- 受影响算子、软件层、后端：`linear`（FlagGems 算子层 `src/flag_gems/ops/linear.py`）；厂商无关通用文件，本机以 cuda dispatch key 验证（FlagGems vendor 解析为 thead / ppu-smi）
- 问题与修复后行为：非连续输入（转置/切片/permute 出来的激活）走 `flag_gems.linear` 必然抛
  `RuntimeError: view size is not compatible with input tensor's size and stride`；修复后与原生
  torch 逐项一致（fp32 mse ≈ 1e-11 量级），连续输入路径逐位不变。

## 一句话版本

**`linear()` 把用户布局未知的激活 `view(M, K)` 压平，非连续张量直接崩；改成 `reshape(M, K).contiguous()`，
需要拷贝时才拷贝，其余情况仍是零拷贝视图。**

```
修复前: RuntimeError: view size is not compatible with input tensor's size and stride
        (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        -> 位于 src/flag_gems/ops/linear.py:158
修复后: 输出与原生 torch 一致，50/50 回归用例通过（连续对照组逐位相等）
```

## 问题是怎么回事

`linear()` 需要把 `(*, K)` 的输入压成 `(M, K)`：

```python
M = input_flat.shape[0]
input_flat = input.view(M, K)   # 基线 src/flag_gems/ops/linear.py:158
```

`view()` 的前提是新形状与原 stride 兼容（"at least one dimension spans across two
contiguous subspaces" 就是它在拒绝这种布局）。视觉模型里的激活经常不是连续的：

- attention 出入口的 `transpose` / `permute`：`(B, S, D) -> (S, B, D)`
- 卷积特征图按 NHWC 使用：`(B, C, H, W).permute(0, 2, 3, 1)`
- 对最后一维做切片 `x[..., :K]`：最后一维 stride 不再是 1

实测肇事输入（issue #5379 的最小复现形状）：

```
input (2, 16, 64) stride=(1024, 1, 16) contiguous=False
  File ".../src/flag_gems/ops/linear.py", line 158, in linear
    input_flat = input.view(M, K)
RuntimeError: view size is not compatible with input tensor's size and stride ...
```

这条路径同时影响 Torch dispatch（`F.linear` / `nn.Linear` 在 `flag_gems.enable()` 下）与直接调用
`flag_gems.linear`，两条都在 `tests/repro_linear_view.py` 里复现：修复前退出码 1，修复后退出码 0。

容器侧（problem.md，2026-09-01 live）：还原成 `view` 后视觉模型 on 趟失败，
`view→reshape` 补丁后 18 个视觉模型 36/36 翻转 Success，最小复现模型 `google/siglip-base-patch16-224`。

## 修复思路

```diff
-    input_flat = input.view(M, K)
+    input_flat = input.reshape(M, K).contiguous()
```

- `reshape` 语义 = 尽量返回视图，做不到就拷贝，正好适配"外部传入、布局不可控"的输入；
- 结果本来就连续时 `.contiguous()` 是空操作，**不会**给连续路径多加一次拷贝；
- kernel 消费的是显式 stride（`stride_im` / `stride_ik`），所以连续与非连续两条路都正确；
- 单文件、单行改动，`git apply --check` 干净，可直接回滚。

完整 diff 见 `patches/linear-5379.patch`（1 行替换），前后文件 hash：

| | sha256 |
| --- | --- |
| 基线 `src/flag_gems/ops/linear.py` | `4d46bae0804ac32b698582d010d5f0e02d3570c6703efff9d1d666e2bf43fd94` |
| 修复后 | `ddec3383c1d6ca4b2f53cf6ea9550c2d5fead492a2562e0864e723b958ee951d` |

## 怎么用

前置：FlagGems 基线 `29e14822f6dfabf990973c41f7c2df7811a6f60f`（`git describe` = `v5.3.5-46-g29e14822`），
目标文件 hash 与上表一致。验证硬件为 PPU，FlagGems vendor 解析为 `thead`，设备/dispatch 标识为 `cuda`：
`cuda` 只是本环境的设备标识，不代表已验证所有 CUDA 厂商后端。本机环境与实际执行过的命令见
`reports/validation.md`。

三个测试脚本不依赖工作目录，但 `--root` 和 patch 路径依赖：`git -C` 会先切换工作目录、
`--root .` 取决于当前目录，所以先保存两个绝对路径，后续命令全程使用它们（下文均为参数化示例，
不是某台机器的实际路径）：

```bash
MANAGER=/path/to/flag-fixes   # 本仓库根目录（含 catalog/、fixes/、src/）
TARGET=/path/to/FlagGems      # FlagGems 源码根（含 src/flag_gems/...）

# 1) 预演：确认目标仍是修复前内容，patch 能干净应用（不落盘）
PYTHONPATH="$MANAGER/src" python3 -m flag_fixes --root "$MANAGER" \
  check linear-5379 --target "$TARGET"

# 2) 手工应用最小 patch（本修复未获部署资格，flag-fixes apply 会按设计拒绝）
git -C "$TARGET" apply "$MANAGER/fixes/linear-5379/patches/linear-5379.patch"

# 3) 验证：三个脚本都必须退出码 0；缺设备或缺 transformers 时退出码 2，不算通过
PYTHONPATH="$TARGET/src" python3 "$MANAGER/fixes/linear-5379/tests/repro_linear_view.py"
PYTHONPATH="$TARGET/src" python3 "$MANAGER/fixes/linear-5379/tests/test_linear_noncontig.py"
PYTHONPATH="$TARGET/src" python3 "$MANAGER/fixes/linear-5379/tests/test_linear_callers.py"  # 需要 transformers

# 或由管理工具跑（要求目标已是修复后内容；--root 永远指向管理仓库根，输出 JSON 见 results/）
PYTHONPATH="$MANAGER/src" python3 -m flag_fixes --root "$MANAGER" \
  verify linear-5379 --target "$TARGET" --python python3

# 4) 回滚
git -C "$TARGET" checkout -- src/flag_gems/ops/linear.py
```

容器侧整模型回归（problem.md 要求的独立 result-root，避免命中已有 on-Success 缓存）。
以下路径是目标容器内的固定路径，按 problem.md 原样给出，不是本机路径：

```bash
cd /home/wuyiqiang/x-benchmark && HF_HOME=/home/wuyiqiang/hf_cache \
PYTHONPATH=/data/ppu-benchmark/FlagGems/src \
python3 run_model.py --models "google/siglip-base-patch16-224" \
  --models-json assets/models_info_20260813_073927_top515_new317.json \
  --envs-dir /data/ppu-benchmark/venvs --device cuda:0 \
  --result-root /data/ppu-benchmark/results_top515_repro
```
（该步依赖目标容器，本环境未执行，见 `reports/validation.md` 的 not-run 清单。）

## 目录结构

```
├── README.md                     # 本文
├── fix.yaml                      # 清单（schema_version 1）
├── WORKFLOW.md                   # 8 步执行清单及证据
├── patches/linear-5379.patch     # 最小 diff（1 行），git apply 用
├── src/flag_gems/ops/linear.py   # 修复后完整文件快照，与 patch 结果逐字节一致
├── tests/
│   ├── repro_linear_view.py      #   issue 最小复现 + 修复验收（level: repro）
│   ├── test_linear_noncontig.py  #   布局 × dtype × 路径矩阵，50 用例（level: regression）
│   └── test_linear_callers.py    #   NHWC/seq-first 调用方 + 随机初始化 SigLIP 视觉塔（level: e2e，需 transformers）
├── reports/
│   ├── diagnosis.md              #   根因、方案、影响范围
│   └── validation.md             #   环境、命令、前后结果、not-run 清单
└── results/
    ├── verify-2026-11-04.json    # 历史记录：修复目录未提交时（HEAD 984891e，dirty=true）
    └── verify-2026-11-05.json    # 已提交、干净工作树上的重跑结果
```

## 验证结果摘要

| 项目 | 修复前 | 修复后 |
| --- | --- | --- |
| `repro_linear_view.py` | exit 1，`linear.py:158` RuntimeError | exit 0，mse=2.67e-11 |
| `test_linear_noncontig.py` | 20 passed / 30 failed | 50 passed / 0 failed |
| `test_linear_callers.py` | exit 1（NHWC MLP 即崩） | 3 passed / 0 failed（含 SigLIP 视觉塔 mse=8.1e-13） |
| 连续对照组 | 通过 | 通过，且与修复前逐位相同（max_diff=0） |
| `flag-fixes check/plan` | — | exit 0（基线 hash、patch 应用、快照一致） |
| 缺 transformers 时的 e2e | — | 退出码 2（`MISSING DEPENDENCY`，skip 不算通过） |

判定与覆盖边界：三个硬件脚本只有退出码 0 才算通过，缺设备或缺 `transformers` 一律退出码 2。
普通 CI（无设备 runner）执行 `flag-fixes validate`（schema、patch hash、快照 hash、脚本存在性）、
`python -m pytest -q` 和本地 fix 脚本语法检查（`python -m compileall -q fixes`），**不**执行这三个
硬件脚本；语法检查通过 ≠ 入口执行通过 ≠ 硬件验证通过。

## 已知边界（不装完美）

- **平台验收未做**：`hardware_status` 保持 `unverified`，problem.md 要求的 x-benchmark 容器
  on 趟回归（独立 `--result-root`）未在本环境执行；`deployable=false`、`owner=null`，
  `flag-fixes apply` 会按设计拒绝。
- **只改通用文件**：同源 `view(M, K)` 还在 `_enflame/gcu300`、`_enflame/gcu400`、
  `_iluvatar` 的 linear 覆写，以及 `ops/linear_backward.py:73-74`、`_thead`/`_iluvatar`
  的 linear_backward 中；按"不同后端分条"的约定未包含在本 patch，跟踪在上游 issue #5379。
- **发现一个与本修复无关的既有缺陷**：本机 fp16/bfloat16 在 `M ∈ [5, 8]`（N=32, K=64）时
  输出完全错误（maxdiff ≈ 2.0e1~3.4e1），连续与非连续输入、修复前后完全一致地复现，
  与 view/reshape 无关；回归矩阵因此只在 fp32 上断言该 M 段，详见 `reports/diagnosis.md`。
- 没有性能基准：这是崩溃修复，连续路径逐位不变、非连续路径原本根本跑不完；
  `benchmarks/` 保持模板状态，未伪造前后耗时。
- **证据文件的来源与可追溯性**：`results/verify-2026-11-04.json` 是修复目录尚未提交时
  （管理仓库 HEAD `984891e`、`source_dirty=true`）跑出的历史记录，按审阅意见**原样保留**，
  其中的 SHA 未改写，也不能单凭该 commit 还原当时未提交的脚本；`results/verify-2026-11-05.json`
  是在已提交、干净的工作树上重跑的结果（`source_dirty=false`）。两份都是工具原始输出，
  含本机绝对路径，未做脱敏或改写；`reports/validation.md` 中的摘录若为原文会明确标注。
- **e2e 依赖**：`tests/test_linear_callers.py` 第 3 项需要 `transformers`（随机初始化、不下载权重），
  缺失时退出码 2 并打印 `MISSING DEPENDENCY`，不会因为前两项通过而返回 0。
