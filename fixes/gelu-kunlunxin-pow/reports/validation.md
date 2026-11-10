# 验证记录

当前：**本机（P800/XPU3）微基准、精度、dispatch 回归与 flag-fixes 工具链验证已执行；
平台整模型 x-benchmark 验收 not-run**。因此 `hardware_status` 仍为 `unverified`，
`deployable=false`，`apply` 按设计拒绝。

## 环境与源码

| 项目 | 值 |
| --- | --- |
| 机器 | 8× 昆仑芯 P800 OAM（XPU3），AMD EPYC 9K84，1.5 TB 内存 |
| 本次进程可见设备 | `CUDA_VISIBLE_DEVICES=1,2`；脚本使用 `cuda:0`（物理 1 号卡，当时有其他负载） |
| 驱动 / 运行时 | XPU-SMI 5.0.21.47，XPU-RT 5.13.0 |
| Python | 3.10.18 |
| PyTorch | 2.9.0+cu129（`torch_xmlir`，设备名报告为 `GPU`，capability 8.6，96 GiB） |
| Triton | 3.0.0（`triton/backends/xpu`，`xpu3-elfconv` 工具链） |
| FlagGems | `v5.0.0`，基线 commit `4e08b44e9b9ad68c2c27b4d25d04c6b76c30ff7a` |
| 关键环境变量 | `GEMS_VENDOR=kunlunxin`，`KLX_USE_AUTOTUNE=0`，`USE_FLAGGEMS=1`，`XPU_FORCE_SHARED_DEVICE_CONTEXT=1` |
| 管理仓库 | FLAG-OP/flag-fixes，本地 fix `fixes/gelu-kunlunxin-pow/`；首个 verify 时 `source_commit=f159ce07b1995fe6c313b5f7845bf4f46821b876`、`source_dirty=true`（fix 未提交）；fix 提交为 `3e95f021e93ed105d37842a6366034a86857b267`，在最终树 `d57e825472f66809ac1224dd7aca8c6e4bec47a3` 上复跑得 `source_dirty=false` |
| 目标文件 hash | `_kunlunxin/ops/gelu.py` `6c53c738…` → `e9c72e11…`；`_kunlunxin/fused/gelu_and_mul.py` `a3192f8d…` → `2669825c…` |
| 执行日期 | 2026-11-10 |
| 执行者 | AI agent（无人类 owner；`owner=null`） |

复现用的两棵 worktree 由同一次 `git worktree add --detach` 从
`/workspace/FlagGems` 的 `4e08b44e9…` 创建：

```bash
git -C /workspace/FlagGems worktree add --detach /tmp/opencode/FlagGems-gelu-base  4e08b44e9b9ad68c2c27b4d25d04c6b76c30ff7a
git -C /workspace/FlagGems worktree add --detach /tmp/opencode/FlagGems-gelu-fixed 4e08b44e9b9ad68c2c27b4d25d04c6b76c30ff7a
git -C /tmp/opencode/FlagGems-gelu-fixed apply \
  /workspace/flag-fixes/fixes/gelu-kunlunxin-pow/patches/gelu-kunlunxin-pow.patch
```

## 命令与结果

所有测试都以目标源码树优先导入：
`PYTHONPATH=/tmp/opencode/FlagGems-gelu-{base,fixed}/src python3 <script>`。
“修复前”用 base 树，“修复后”用 fixed 树，输入、随机种子、计时口径相同。

| 项目 | 精确命令 | 判定标准 | 修复前 | 修复后 | 证据 |
| --- | --- | --- | --- | --- | --- |
| 清单校验 | `PYTHONPATH=src python3 -m flag_fixes --root . validate`（在 flag-fixes 内） | exit 0，schema/hash 全部有效 | — | exit 0（11 fixes，0 deployable，0 hardware_verified） | 见下方“工具链输出” |
| 基线/补丁预演 | `… check gelu-kunlunxin-pow --target /tmp/opencode/FlagGems-gelu-base`；`… plan … --target …base` | 基线 hash 命中、`git apply --check` 干净、patch 结果与 `src/` 快照一致 | — | 两项均 exit 0 | 终端输出（`check` 打印 before/after hash 与 patch 列表） |
| 安装拒绝 | `… apply gelu-kunlunxin-pow --target /tmp/opencode/FlagGems-gelu-base` | `deployable=false` 时必须拒绝 | — | 正确拒绝（exit 非 0，未改目标树） | 终端输出 |
| 最小复现 | `python3 tests/repro_gelu_pow.py` | `gelu_backward` 与 `gelu(tanh)` 前向都 ≤ 2x native，且 max_abs_err ≤ 0.05 | **exit 1**：bwd 8.41x、tanh 前向 2.41x native | **exit 0**：bwd 0.51x、tanh 前向 0.43x native | `results/repro-{base,fixed}-2026-11-10.log` |
| 回归/边界 | `python3 tests/test_gelu_accuracy.py` | 3 dtype（fp32/bf16/fp16）× 4 形状（含 (1,64)、(17,33)、(2990,1024)、(3,128,512)）× 连续/转置/切片/permute × none/tanh 前反向 + fused `gelu_and_mul` 前后向，误差在 atol/rtol 内 | exit 0，0 failed | exit 0，0 failed | `results/accuracy-{base,fixed}-2026-11-10.log` |
| 调用方/dispatch | `python3 tests/test_gelu_family_dispatch.py` | `enable()` 后 `F.gelu`、`nn.GELU(tanh)`、`aten.gelu_backward`、`autograd.grad`、非连续输入、gated MLP 与 native 参考一致 | exit 0，0 failed | exit 0，0 failed | `results/dispatch-{base,fixed}-2026-11-10.log` |
| 性能/精度 | `python3 tests/bench_gelu_family.py --iters 100 --json results/bench-<tree>-2026-11-10.json` | 同设备/同输入/同计时；同时校验精度门限 | 见下方性能表 | 见下方性能表 | `results/bench-{base,fixed}-2026-11-10.{json,stdout.log,stderr.log}` |
| 工具链 verify | `PYTHONPATH=src python3 -m flag_fixes --root . verify gelu-kunlunxin-pow --target /tmp/opencode/FlagGems-gelu-fixed --python python3` | 4 个 declared 命令全部 rc=0 | — | `passed=true`，4/4 rc=0；首个结果 `source_dirty=true`，在最终树 `d57e8254`（只比 `3e95f021` 多文档/标题改动）上复跑为 `source_dirty=false` | `results/verify-2026-11-10.json`、`results/verify-2026-11-10-clean.json` |
| 平台整模型 e2e | problem 所述 x-benchmark `run_model.py`（on 趟，独立 `--result-root`） | 整模型吞吐提升且不劣化 | **not-run** | **not-run** | 原因：本环境没有 x-benchmark runner / 模型权重缓存；未伪造结果 |
| 安装/回滚演练 | `… apply` / `… rollback` | 安装事务成功、回滚 hash 还原 | **not-run** | **not-run** | 原因：`deployable=false`，`apply` 按规范拒绝；未获得部署资格前不做目标树安装 |
| 上游 PR | FlagGems upstream | PR 合入 | **not-run** | **not-run** | 原因：草稿尚未通过平台验收与 owner 审核 |

## 性能数据（bf16, (2990, 1024), best-of-5 × 100 iters）

| 用例 | 修复前 | 修复后 | 加速 | native 修复前 | native 修复后 | 修复后/native | max_abs_err 前→后 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `gelu_fwd_none` | 0.1149 ms | 0.1167 ms | 0.99x | 0.1704 ms | 0.1701 ms | 1.46x | 1.53e-5 → 1.53e-5 |
| `gelu_fwd_tanh` | 0.3124 ms | **0.0542 ms** | **5.77x** | 0.1301 ms | 0.1302 ms | 2.40x | 1.53e-5 → **3.81e-6** |
| `gelu_bwd_none` | 2.5392 ms | **0.1538 ms** | **16.51x** | 0.3023 ms | 0.3019 ms | 1.96x | 7.81e-3 → **1.95e-3** |
| `gelu_bwd_tanh` | 0.5743 ms | **0.0639 ms** | **8.98x** | 0.1390 ms | 0.1399 ms | 2.19x | 1.56e-2 → 1.56e-2 |
| `gelu_and_mul_none` | 0.4665 ms | 0.4650 ms | 1.00x | 0.6852 ms | 0.6858 ms | 1.48x | 6.25e-2 → 6.25e-2 |
| `gelu_and_mul_tanh` | 1.2972 ms | **0.2197 ms** | **5.90x** | 0.5464 ms | 0.5460 ms | 2.48x | 6.25e-2 → 6.25e-2 |

口径：`tests/bench_gelu_family.py`，CUDA event 计时，warmup 20，100 iter × 5 轮取最优
均值；native 参考在 `flag_gems.enable()` 之前测量；每个 FlagGems 结果都在同一轮里
与 native 参考比较，精度门限 fp32 `atol=1e-4/rtol=1e-4`、fp16/bf16
`atol=5e-2/rtol=3e-2`，两个 log 中 `ACCURACY FAIL` 计数均为 0。
`gelu(none)` 与 `gelu_and_mul(none)` 不经过 `pow`，改动后耗时不降（符合预期，且
`gelu(none)` 前向 1.46x、`gelu_and_mul(none)` 1.48x native，本来就不慢）。

## 工具链输出

```text
$ PYTHONPATH=src python3 -m flag_fixes --root . validate
{
  "valid": true,
  "fixes": 11,
  "deployable": 0,
  "hardware_verified": 0
}

$ python3 -m flag_fixes --root . verify gelu-kunlunxin-pow \
    --target /tmp/opencode/FlagGems-gelu-fixed --python python3
{ "fix": "gelu-kunlunxin-pow", "passed": true,
  "scope": "declared commands only; does not promote catalog hardware status",
  "source_commit": "d57e825472f66809ac1224dd7aca8c6e4bec47a3",
  "source_dirty": false }
# gelu-pow-repro             rc=0  PASS: GELU kernels are within the native-torch budget
# gelu-family-accuracy       rc=0  TOTAL: 0 failed (GELU family matches native torch)
# gelu-family-dispatch       rc=0  TOTAL: 0 failed (dispatch / autograd / gated MLP)
# gelu-family-benchmark      rc=0  (JSON 输出，无 ACCURACY FAIL)
```

`verify` 的作用域是“声明的命令是否通过”，**不会**把 `hardware_status` 提升为
verified；平台验收仍需单独执行。

## 未执行、不适用与已知限制

- **平台整模型 x-benchmark：not-run**。缺 runner/模型缓存；`hardware_status=unverified`，
  `deployable=false`。微基准的 16.5x 不能直接换算成整模型吞吐提升。
- **安装/回滚：not-run**。`apply` 在 `deployable=false` 时按设计拒绝，未做目标树安装
  演练；回滚机制由管理仓库统一工具负责，本 fix 未新增安装器。
- **厂商 aten highprecision GELU：not-applicable**。`gelu_highprecision_nv` /
  `gelu_bwd_highprecision_nv` 在厂商二进制里，FlagGems patch 改不到；本修复只让
  启用 FlagGems GELU 的路径变快。
- **`_kunlunxin/fused/geglu.py`：not-applicable（本 patch）**。`pow` 会导致
  `xpu3-elfconv` 失败；去掉后可编译，但 kernel 当前配置约 21 ms，单独跟踪。
- **其他厂商后端：not-run**。未在 Cambricon/Enflame/Hygon/ARM 验证，不批量修改。
- **测量噪声**：物理 1 号卡当时有其他负载（xpu-smi 显示 70% util、45 GB 占用），
  绝对值会漂移；本报告用同进程、同输入的前后对照和 best-of-5 降低噪声，倍数是
  稳定的，但不应当当作绝对性能上限。
- **环境异常（与修复无关）**：FlagGems 的 `copy_` 对非连续张量会 redispatch 到
  `CompositeExplicitAutograd` 并报 `CUDA error: invalid device function`，未启用
  FlagGems 时同样复现；因此所有检查函数先 `.cpu()` 再归约，测试不依赖该路径。
