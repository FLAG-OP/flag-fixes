# 可选源码快照

按目标源码根保留修复后的完整文件，用于证明 patch 应用结果与快照一致：

| 快照 | 对应 target | after sha256 |
| --- | --- | --- |
| `src/flagscale/train/train_retrieval.py` | 同路径 | `96fc52a9c5f93205479c0026817f6cbf989aa958aabdf5d1cf1b43b34d5c84bf` |

原版由 FlagScale 本地 checkout commit
`89f342e690053e07dda845f1365089ec00464e09` 追溯（before sha256
`5ee57573698129e1a2e3d5f7cdb6b5e8768fe9fc60c59369c95cc52eb51127df`），不保存 `.orig`。
文件保留 FlagScale 原有版权与许可证（Apache-2.0）。
