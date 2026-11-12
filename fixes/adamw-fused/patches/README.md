# 最小 patch

`adamw-fused.patch` 在 FlagScale 本地 checkout
（`89f342e690053e07dda845f1365089ec00464e09`）的干净基线上生成：

```bash
git -C /tmp/opencode/FlagScale-adamw-fixed diff -- \
  flagscale/train/train_retrieval.py \
  > patches/adamw-fused.patch
```

- 只含文本修改（1 文件，51+/1-），无新增/删除/重命名/二进制/符号链接文件。
- 只新增 `adamw_impl_kwargs()` 并在 `build_optimizer` 的 AdamW 分支接入
  （含构造期回退）；无格式化、无机器路径。
- SHA256：`dbd6e8d8db9f78b8bffe7161a62c130d7a1ed3569eda4b81ce8a64a047e2b58d`
  （记录在 `../fix.yaml` 的 `patches[0].sha256`）。
- 应用前先 `git apply --check`；应用结果与 `../src/` 快照逐字节一致
  （`check` / `plan` 会验证 before/after hash）。

注意：目标包是 flagscale，仓库布局为根下 `flagscale/` 包（无 `src/` 顶层），
patch 路径与仓库路径一致。
