# 可选源码快照

按目标源码根保留修复后的完整文件，用于证明 patch 应用结果与快照一致：

| 快照 | 对应 target | after sha256 |
| --- | --- | --- |
| `src/flag_gems/runtime/backend/_kunlunxin/ops/gelu.py` | 同路径 | `e9c72e110834d1f040b82541e6f50b3793eb809a5db3af06e3d11b0e3f286048` |
| `src/flag_gems/runtime/backend/_kunlunxin/fused/gelu_and_mul.py` | 同路径 | `2669825cfda5ccacb86b4826f989b1d8b3e8fbe5c8a4bd5a4608d7523d8501e0` |

原版由上游 commit `4e08b44e9b9ad68c2c27b4d25d04c6b76c30ff7a` 追溯，不保存 `.orig`。
文件保留 FlagGems 原有版权与许可证（Apache-2.0）。
