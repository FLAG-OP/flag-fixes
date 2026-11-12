# 性能对照（gelu pair，10,299,4096 bf16）

| 模式 | pair (median/30) | 说明 |
| --- | ---: | --- |
| vendor（无 FlagGems） | 1.860 ms | 厂商 gelu_highprecision 前反向 |
| include（本修复 + FlagGems 补丁） | **1.168 ms**（1.59x） | `enable_flaggems` include 路径 |

口径：median-of-30，每步 synchronize；GB/s 数字按 3-tensor traffic
（x + dy + dx）计。原始数据 `../results/bench-*.json`。

**配套要求**：include 注册的 gelu kernel 必须来自打过 `gelu-kunlunxin-pow`
补丁的 FlagGems——未打补丁时反向内含 libdevice pow，pair ~10.8 ms
（`../results/bench-base-*.json` 即此状态）。

整步验证：BGE-M3 重建负载（AdamW fused + 本修复 + FlagGems 补丁，
全链路走 enable_flaggems 集成路径）2,665.9 → 2,291.2 ms/step（-14.0%），
见 `../results/step-E-final.json` 与 `problem29/RESULT.md` 第五阶段。
