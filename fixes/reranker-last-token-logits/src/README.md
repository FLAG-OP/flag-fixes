# 可选源码快照

按目标源码根保留修复后的完整文件，用于证明 patch 应用结果与快照一致：

| 快照 | 对应 target | after sha256 |
| --- | --- | --- |
| `src/flagscale/models/retrieval/qwen3_reranker.py` | 同路径 | `dd34c0dd0459f0469280968496ed7a42d86e46fb3feac6ada822d221ccbfcd89` |

原版由 FlagScale `feat/retrieval-flagos` 的 commit
`89f342e690053e07dda845f1365089ec00464e09` 追溯（before sha256
`501470249c00c934268bd3a02bb29cad0884dd4a9336635c1d2f85a70781f7fc`，
git blob `e5abccd6e516b85406e88bde36159a524c88c3e9`），不保存 `.orig`。
文件保留 FlagScale 原有版权与许可证（Apache-2.0）。
