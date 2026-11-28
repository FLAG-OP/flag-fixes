# 最小 patch

`reranker-last-token-logits.patch` 在 FlagScale `feat/retrieval-flagos` 分支的
`89f342e690053e07dda845f1365089ec00464e09` 干净基线上生成（与 `adamw-fused`、
`flaggems-selective-enable` 同一基线）。该文件到分支头 `555a7d6a10dcef514e02fdfbeff60df89783f6af`
（2026-11-23）内容未变，git blob 始终是 `e5abccd6e516b85406e88bde36159a524c88c3e9`：

```bash
git diff -- flagscale/models/retrieval/qwen3_reranker.py \
  > patches/reranker-last-token-logits.patch
```

- 只含文本修改（1 文件，+15/−6），无新增/删除/重命名/二进制/符号链接文件。
- 只改 `Qwen3RerankerModel.forward` 的非 score-head 分支，并从 `transformer_encoder`
  多导入一个已有的 `last_token_pool`；无格式化、无机器路径。
- SHA256：`ee7ded27c09fd2ae7b15f499ce1f14a79891ecf20b428f30fa812848197537d4`
  （记录在 `../fix.yaml` 的 `patches[0].sha256`）。
- 应用前先 `git apply --check`；应用结果与 `../src/` 快照逐字节一致
  （after sha256 `dd34c0dd0459f0469280968496ed7a42d86e46fb3feac6ada822d221ccbfcd89`，
  `check` / `plan` 验证 before/after hash）。

注意：目标包是 flagscale，仓库布局为根下 `flagscale/` 包（无顶层 `src/`），
patch 路径与仓库路径一致。

补丁不新增配置开关，也没有关闭路径：它不改变打分语义（同一位置、同一权重，
见 `../reports/validation.md` 的数值对照：bf16 下逐位相同，fp32 下在 1e-4 / 1e-3 以内），因此没有保留"整序列 logits"的必要。
