# 验证报告：adamw-fused

## 环境

| 项 | 值 |
|---|---|
| 机器 | KunlunXIN P800 OAM ×8（本验证用 1 卡），XPU-SMI 5.0.21.47 / XPU-RT 5.13.0 |
| torch | 2.9.0+cu129（torch_xmlir shim；`torch.cuda.get_device_name` = "GPU"） |
| FlagScale 基线 | 本地 checkout `89f342e690053e07dda845f1365089ec00464e09` |
| profiler 开关 | `XPU_ENABLE_PROFILER_TRACING=1`、`XPU_CUPTI_ENABLE_DEVICE=0` |
| 目标树 | `/tmp/opencode/FlagScale-adamw-base`（基线 worktree）、`/tmp/opencode/FlagScale-adamw-fixed`（基线 + patch） |
| 重建 | `git -C /tmp/opencode/FlagScale worktree add <path> 89f342e…`；fixed 树 `git apply adamw-fused.patch` |

hash（fix.yaml 记录同值）：

```
before  5ee57573698129e1a2e3d5f7cdb6b5e8768fe9fc60c59369c95cc52eb51127df  flagscale/train/train_retrieval.py
after   40f1e4c4dc42c60e1d53c2fa9b74c5b38fba08e616bed26849675567b54ce394  flagscale/train/train_retrieval.py
patch   774473b5acf5aa642060816107d90fc6aa2d2dc458f4d95af4e3ea415c8c1498  patches/adamw-fused.patch
```

## 命令与结果（2026-11-10）

统一方式：`PYTHONPATH=<tree> python tests/<script>`（cwd = fixes/adamw-fused）。

| 脚本 | 基线树 | 修复树 | 判定 |
|---|---|---|---|
| `tests/repro_adamw_unfused.py` | exit **1**（fused=False，31.48 ms/步） | exit **0**（fused=True，10.999 ms，ratio 0.354） | 符合预期 |
| `tests/test_adamw_numerics.py` | exit **1**（基线无 fused 选择） | exit **0**（bf16 max\|diff\| 3.05e-05、fp32 2.98e-08，全 allclose） | 符合预期 |
| `tests/test_adamw_selection.py` | exit **1**（`adamw_impl_kwargs` 不存在） | exit **0**（9 例选择矩阵全对） | 符合预期 |
| `tests/bench_adamw_impls.py --iters 30` | exit 0（build_optimizer=foreach：27.6 ms / 2,688 launches） | exit 0（**11.5 ms / 672 launches**，device 8.9 ms） | 性能证据 |

原始日志与 JSON：`results/*-2026-11-10.{log,json}`。

## 关键数据（bench，336 张量 bf16 参数集）

| 实现 | 基线树 step | 修复树 step | launch（fixed） | device（fixed） |
|---|---:|---:|---:|---:|
| stock | 31.2 ms | 35.4 ms | 3,024 | 26.8 ms |
| foreach | 27.5 ms | 29.3 ms | 2,688 | 25.7 ms |
| **build_optimizer** | **27.6 ms** | **11.5 ms** | **672** | **8.9 ms** |

即修复把 build_optimizer 的选择从"无融合收益的 foreach"换成"真正的单支
fused"：**2.4× 墙钟、4× launch、2.9× device**。
换算到 BGE-M3 真实负载（389 张量、每步 1 次 step）：
ATTRIBUTION 实测 41.8→17.3 ms kernel，对应平台文档 "clip+optimizer 62 ms"
→ 预期 ~37 ms 墙钟（预估，未在平台验证）。

## 评审修订（PR #3，2026-11-10）

**测量口径变更**：共享主机的 launch 路径（cudaLaunchKernel CPU）当日出现
7 µs → 0.9–2.5 ms/次 的漂移，曾使 wall 对照在两次运行间反转（stock 30↔9,965 ms）。
repro 的性能判定已改为 torch.profiler **device time**（fused 8.97 vs stock 27.0 ms，
ratio 0.332，确定性通过）；CUPTI 关闭时 exit 2=skip。历史 wall 数字（11.0 ms 等）
为快速窗口期测得，保留作参考。

- 新增 vendor/platform 护栏：`_detect_vendor()`（FLAGSCALE_VENDOR →
  GEMS_VENDOR → torch_xmlir shim 探测）+ `_FUSED_ADAMW_VALIDATED_VENDORS`
  = {kunlunxin}；非 KunlunXIN 平台 auto 与显式请求均回退 stock 并记录原因；
- 选择矩阵扩至 13 例：新增 GEMS_VENDOR=nvidia/thead/kunlunxin、显式覆盖、
  混合 dtype（均受支持 → fused）、torch_xmlir 探测等断言
  （`results/selection-fixed-2026-11-10.log`，13 ok / exit 0）；
- repro/numerics 在更新后的树上复跑全过（日志 results/ 下 *-fixed2 已覆盖
  归档为 *-fixed）。

## 工具链校验

- `flag-fixes --root . validate`：valid（13 fixes = 9 external + 4 local）
- `flag-fixes check adamw-fused --target <base 树>` / `plan`：exit 0
  （基线 hash 命中、`git apply --check` 干净、patch 结果与 `src/` 快照一致）
- `flag-fixes apply adamw-fused --target <树>`：exit 2 拒绝（`deployable=false`，符合预期）
- `flag-fixes status --target <fixed 树>`：clean
- `flag-fixes verify adamw-fused --target <fixed 树> --python <本机 python>`：
  首次 4/4 rc=0（`results/verify-2026-11-10.json`，`source_dirty=true`，
  fix 未提交）；提交后干净树复跑 4/4 rc=0
  （`results/verify-2026-11-10-clean.json`，source commit `25afaebc…`、
  `source_dirty=false`）
- 仓库测试 `python -m pytest -q`：26 passed, 3 subtests passed

## not-run（未验证清单，不记为通过）

| 项 | 原因 |
|---|---|
| 平台 BGE-M3 / reranker（FSDP2）整模型训练验收 | 本机无 `/flagos-retrieval-models`、数据集与平台 runner；`hardware_status=unverified`、`deployable=false` |
| FSDP2/DTensor 下的真实护栏验证 | 无多卡 dist 环境初始化；护栏以单测替身（`type.__module__` 检查）覆盖，真实 DTensor 行为未测 |
| 真实 step 端到端收益（含 clip 62 ms 墙钟） | 同上；只给出 kernel 级实测 + 墙钟预估 |
| `HYDRA_*`/foreach 在其他厂商后端（PPU/Ascend/CUDA）的行为 | 无设备 |
| `clip_grad_norm_` 的 multi-tensor reduce 优化 | torch 2.9 foreach 实测只省 ~2 ms，需厂商支持，未立项 |

## 二轮评审修订复验（2026-11-11）

最新 clean verify 绑定提交 `4052ef37`
（`results/verify-2026-11-11-clean.json`，source_dirty=false，4/4 rc=0）；
选择矩阵扩至 18 例（含 vendor nvidia/thead/kunlunxin、显式 forced=1 回退日志断言、
混合 dtype）。性能判定口径为 torch.profiler device time。