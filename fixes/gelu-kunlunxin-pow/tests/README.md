# 验证入口

所有脚本在仓库根目录或本目录执行皆可，需把**目标源码树**放在 `PYTHONPATH` 前面
（脚本会打印实际导入的 `flag_gems.__file__` / kernel 文件路径，便于确认没有误用
site-packages 里的 FlagGems）：

```bash
export PYTHONPATH=/path/to/FlagGems/src
python3 tests/<script>.py
```

设备：昆仑芯 P800 OAM（XPU3），torch 2.9.0+cu129（torch_xmlir），triton 3.0.0，
`GEMS_VENDOR=kunlunxin`。无设备时脚本打印 `SKIP: no accelerator available; not a pass`
并返回 **2**（跳过不算通过）。

| 脚本 | 层级 | 覆盖 | 修复前预期 | 修复后预期 | 超时 |
| --- | --- | --- | --- | --- | --- |
| `repro_gelu_pow.py` | repro | bf16 `(2990,1024)`，`gelu_backward` + `gelu(tanh)` 前向 vs native，阈值 2x native | **exit 1**（8.41x / 2.41x） | exit 0（0.51x / 0.43x） | 900 s |
| `test_gelu_accuracy.py` | regression | fp32/bf16/fp16 × 4 形状 × 连续/转置/切片/permute × none/tanh 前反向 + fused `gelu_and_mul` 前后向 | exit 0 | exit 0（误差不增） | 1800 s |
| `test_gelu_family_dispatch.py` | dispatch | `enable()` 后 `F.gelu`、`nn.GELU(tanh)`、`aten.gelu_backward`、`autograd.grad`、非连续输入、gated MLP | exit 0 | exit 0 | 1800 s |
| `bench_gelu_family.py` | benchmark | 6 个入口的耗时 + native 对照 + 精度门限，输出 JSON（`--json FILE`） | exit 0（慢） | exit 0（快 5.8x–16.5x） | 1800 s |

说明：

- 精度检查在 CPU 上做归约（先 `.cpu()`）。本环境 FlagGems `copy_` 对非连续张量的
  redispatch 会报 `CUDA error: invalid device function`（未启用 FlagGems 时同样复现），
  属环境问题，测试不依赖该路径。
- 各脚本自行把 native 参考在启用 FlagGems 之前算好，避免参考也被 patch 影响。
- 基准脚本的判定同时包含精度；出现 `ACCURACY FAIL` 时 exit 1，不会把“跑完”当通过。
- 清单 `fix.yaml` 的 4 个 `validation.commands` 与上表一一对应。

| `test_gelu_boundary.py` | boundary (supplemental) | 边界语义 dumper（v3：±0/±Inf/NaN/次正规数位级构造/±1e30 大值 × 3 dtype × none/tanh × 前反向；设备输入位型与 FTZ 存档） | — | — | 600 s |
| `compare_gelu_boundary.py` | boundary comparator | 结构检查 + NaN 掩码/Inf 符号/±0 符号位/位型分层 + 冻结容差（fp32 gelu_tanh_bwd atol=2e-7）；`--self-test` 六个负例（含容差分支接受） | — | — | 300 s |
| `verify_gelu_boundary.py` | boundary wrapper | 已注册为 verify 第 5 命令：目标树 dump → 对比已提交基线（FlagGems @4e08b44e9 未打补丁、同版本 dumper）；旧 schema/缺字段/不一致返回非零 | — | — | 600 s |

边界双树用法：分别以 FlagGems-base 与 FlagGems-fixed 的 `src` 为 PYTHONPATH 运行 dumper，再用 comparator 对比（NaN 掩码一致 ≠ payload 一致，payload 已以位型存档支持位级检查；±0 以符号位断言）。历史结论标记：早期版本曾有“±3 相邻值 1 ULP”“±0 相等”“4 个 commands”的描述，已由本轮 v3 口径（5 命令、NaN 掩码比较、±0 符号位断言）替代；旧批次 fp32 gelu_tanh_bwd @−3 曾出现 170 ULP（abs 1.583e-7）差异，**在新批次（v3 dumper）中未复现**（base/fixed @±3 逐位一致），属单批次现象，原因未定位。
