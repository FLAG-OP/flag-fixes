# 最小 patch

`gelu-kunlunxin-pow.patch` 在 FlagGems `v5.0.0`
（`4e08b44e9b9ad68c2c27b4d25d04c6b76c30ff7a`）的干净基线上生成：

```bash
git -C "$TARGET" diff -- \
  src/flag_gems/runtime/backend/_kunlunxin/ops/gelu.py \
  src/flag_gems/runtime/backend/_kunlunxin/fused/gelu_and_mul.py \
  > patches/gelu-kunlunxin-pow.patch
```

- 只含文本修改，无新增/删除/重命名/二进制/符号链接文件，首版安装器可应用。
- 只把 `pow(·, 2.0)` 换成乘法（必要时缓存 `x_fp32` / `x_sq`），并删除因此不再使用的
  `pow = tl_extra_shim.pow`；无格式化、无机器路径。
- SHA256：`65ffc3c7191d6d8382e51fb2ffb1f5a02a5485090efabb5ab560efe489d047e0`
  （记录在 `../fix.yaml` 的 `patches[0].sha256`）。
- 应用前先 `git apply --check`；应用结果与 `../src/` 快照逐字节一致（`check` / `plan`
  会验证 before/after hash）。

```bash
git -C "$TARGET" apply --check patches/gelu-kunlunxin-pow.patch
git -C "$TARGET" apply         patches/gelu-kunlunxin-pow.patch
```

不在目标树提交，也不在目标树留下编译缓存；安装和回滚由 flag-fixes 安装器统一处理。
