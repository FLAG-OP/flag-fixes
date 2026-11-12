# r12 superseded manifest（round-12 评审 6793416880）

| 旧产物/结论 | 状态 | 替代 |
| --- | --- | --- |
| `baseline/baseline-{n,fix,prefix}.json`、`baseline-summary.json`、`baseline-table.md`（round-10 初批 + round-10/11 引用的"52/58 位级一致"） | superseded（截断 checksum、双 exception 判等、abs 梯度输入缺陷） | `baseline-freeze-final.log` + `baseline/` 内 round-12 重跑产物（本目录 `baseline-summary.json` 已更新为 strict comparator 结果 50 equal + 8 numeric_diff） |
| `zero-candidates-{json,log}`（round-10 初批）、`zero-candidates-r11-*`（round-11） | superseded（c128 条目实际 float32 构造） | `zero-candidates-r12-native{,-final}.json/log`、`zero-candidates-r12-registered{,-final}.json/log`（真实 c64/c128） |
| FIX-ITEM-4 的"4 元素小例侥幸正确"与尺寸分支归因 | withdrawn（旧样例实为 int64） | `u8sum-repro-{fix,prefix}-final.log`（真 uint8：7/8 失败，双树同型） |
| "c128 contiguous 约 3µs 收益"（候选 A 选型依据） | withdrawn（数据来自 c64 误标） | r12 registered PATH 数据（B 占优或持平全部已测布局） |

历史文件保留原文不改写；以上替代关系供机器/人工检索。

| `baseline/diff-details.json`（round-11/12 批：c128 ULP 误报 0、bit_list 复数 numel 未翻倍/缺 cpu/conj resolve） | superseded | round-14 重跑（提交 7a691c8）：dtype-aware ULP（c128 float64 位型 1–4 ULP）、NaN case 用基线构造位级解释（out_checksum 差异 = 输出 NaN payload 位型、输入一致）——见 `baseline/diff-details.json`、`ulp-mapping-selftest.json` |
