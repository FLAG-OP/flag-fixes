# 最小 patch

固定上游基线 commit，在干净基线生成 `git diff --binary` 前请确认只有必要的文本修改；首版安装器不支持二进制、新增、删除、重命名和符号链接。推荐 `git diff -- src/flag_gems/... > fix.patch`。

在 fix.yaml 的 patches 中记录相对此 source.path 的路径和 SHA256；targets 记录相对目标源码根的路径及前后 SHA256。先用 `git apply --check`，再确认应用结果与可选 src/ 快照一致。不要包含机器路径、无关格式化或生成缓存。
