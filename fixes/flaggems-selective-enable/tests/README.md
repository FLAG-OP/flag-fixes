# 验证入口

```bash
export PYTHONPATH=/path/to/FlagScale          # 仓库根
python3 tests/<script>.py
```

| 脚本 | 层级 | 覆盖 | 基线预期 | 修复预期 | 超时 |
| --- | --- | --- | --- | --- | --- |
| `repro_flaggems_full_enable.py` | repro | include=[gelu, gelu_backward] 时实际注册集合 | **exit 1**（349 op 全量） | exit 0（2 op） | 900 s |
| `test_flaggems_semantics.py` | regression | 6 例语义矩阵（子进程隔离）：include 列表/缺省全量/"none"/ListConfig/exclude/off | exit 1（include 两例） | exit 0 | 1800 s |
| `test_flaggems_dispatch.py` | dispatch | include 模式下 gelu 前反向实际派发 kernel（profiler 断言） | exit 0（全量注册同样重定向） | exit 0（FlagGems kernel，无 highprecision） | 900 s |
| `bench_flaggems_enable.py` | benchmark | gelu pair：vendor vs include 模式（`--out FILE`） | exit 0 | exit 0（1.17 ms vs 1.86 ms） | 900 s |

- 语义测试通过 `FLAGSCALE_TREE` 环境变量把目标树传给子进程；其余脚本读
  `PYTHONPATH`。无设备时 exit 2（跳过不算通过）。
- exclude 对 cpp_patched_ops（gelu）不生效是 FlagGems 侧行为，测试按实际
  行为断言并注明（基线/修复一致）。
