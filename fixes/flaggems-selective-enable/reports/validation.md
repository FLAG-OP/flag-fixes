# 验证报告：flaggems-selective-enable

## 环境

| 项 | 值 |
|---|---|
| 机器 | KunlunXIN P800 OAM ×8（本验证用 1 卡），XPU-SMI 5.0.21.47 |
| torch | 2.9.0+cu129（torch_xmlir shim） |
| flag_gems | 5.0.0（site，未打补丁）与 FlagGems-fixed worktree（gelu-kunlunxin-pow）两种 |
| FlagScale 基线 | `89f342e690053e07dda845f1365089ec00464e09` |
| 验证树 | `/tmp/opencode/FlagScale-adamw-base`（基线）、`/tmp/opencode/FlagScale-adamw-fixed`（基线 + adamw-fused + 本修复） |

hash（fix.yaml 记录同值）：

```
before  cff9890b022a8ec548619a5302f0e38a9ddd4a949665bffa82dd4702b20a1cd2  flagscale/train/utils/flaggems.py
after   725d26f9af01ccd70316ed3fa00289f541719b43e932e6fdb5ece23f4ef47e13  flagscale/train/utils/flaggems.py
patch   2e3ad7ab6abcfdd1e31c06937db59cb2cdb592e50ed7501a91cf7bae61d0e92f  patches/flaggems-selective-enable.patch
```

## 命令与结果（2026-11-10）

| 脚本 | 基线树 | 修复树 | 判定 |
|---|---|---|---|
| `tests/repro_flaggems_full_enable.py` | exit **1**（include 被忽略，**349 个 op** 全量注册） | exit **0**（2 个 op：gelu, gelu_backward） | 差异所在 |
| `tests/test_flaggems_semantics.py` | exit 1（include 两例失败） | exit **0**（6 例全对；exclude 的 FlagGems cpp-patch 怪癖两侧一致） | 行为对照 |
| `tests/test_flaggems_dispatch.py` | exit 0（全量注册同样重定向 gelu） | exit **0**（include 模式下派发 `gelu_none_kernel_rank_1`/`gelu_backward_none_kernel_rank_1`，无 `highprecision`） | 机制验证（不区分基线/修复，见 README） |
| `tests/bench_flaggems_enable.py --iters 30` | vendor 1.917 ms / include(**未打补丁**) 10.778 ms | vendor 1.860 ms / **include 1.168 ms** | 性能证据 |

## 关键事实

1. **未打补丁的 FlagGems `gelu_backward_none` 内部是
   `tl.exp(-pow(scale1*x, 2.0))`**（libdevice pow）：12.25M 元素上反向单核
   ~10.3 ms，是厂商 kernel（1.36 ms）的 7.6×、打补丁后（0.65 ms）的 15.8×。
   所以三件套必须同装——这同时定量解释了平台"use_flaggems=true 真实训练
   跑不完/更慢"的现象。
2. include 模式整步（E 变体，AdamW fused + 本修复 + FlagGems 补丁，全链路走
   `enable_flaggems` 集成路径）：**2,665.9 → 2,291.2 ms/step（-14.0%）**，
   与直接 `only_enable`（D 变体 2,296.6）一致。

## 评审修订（PR #3，2026-11-10）

- `fix.yaml` 声明 `requires: [gelu-kunlunxin-pow]`：select() 强制与 FlagGems
  补丁同装（check/plan 需同时选择两个 ID）；未打补丁时 include 注册的
  gelu_backward 含 libdevice pow（pair 10.8 ms）；
- include 与 exclude 同时设置：include 优先并打印 WARNING
  （与 flag_gems.use_gems 语义一致），语义测试新增该用例
  （`results/semantics-fixed-2026-11-10.log`，7 例 / exit 0；基线 4 例失败
  = 差异所在）。

## 工具链校验

- `flag-fixes validate`：valid（13 fixes = 9 external + 4 local）
- `check`/`plan`（base 树）：exit 0
- `apply`：exit 2 拒绝（deployable=false）
- `verify`（combined fixed 树）：4/4 rc=0，干净提交 `source_dirty=false`
  （`results/verify-2026-11-10{,-clean}.json`）
- 仓库 pytest：26 passed

## not-run（不记为通过）

| 项 | 原因 |
|---|---|
| 平台整模型验收 | 本机无 runner/模型/数据；hardware_status=unverified |
| 多卡 FSDP2 下注册时机 | 本机单卡；注册时机代码未改，风险低但未实测 |
| enable_flaggems 的 import 失败分支 | 需要破坏性环境；逻辑未改动，不在修复范围 |
