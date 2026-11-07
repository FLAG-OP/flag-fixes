# 验证入口

三个硬件脚本不依赖工作目录；`PYTHONPATH` 必须指向待测树的 `<FlagGems>/src`（每脚本首行
`Run:` 给出模板）。退出码：0 = 通过，1 = 失败/问题复现，2 = 缺设备或缺依赖——**skip 不是
通过**。`os._exit()` 收尾，避免 CUDA teardown 挂起。oracle 一律在 `flag_gems.enable()`
之前由原生 torch 捕获，complex 比较经 `torch.view_as_real`。

| 脚本 | 层级 | 依赖 / 设备 | 修复前预期 | 修复后预期 | 超时建议 |
| --- | --- | --- | --- | --- | --- |
| `repro_complex64.py` | repro | CUDA 必需；无 transformers 依赖 | exit 1，`ISSUE REPRODUCED` + `KeyError: 'complex64'` | exit 0 | 900s |
| `test_complex_ops.py` | regression | CUDA 必需 | exit 1，`TOTAL: 54 passed, 38 failed` | exit 0，`TOTAL: 54 passed, 0 failed` | 1800s |
| `test_m2_bert_backward.py` | e2e | CUDA + `transformers`（随机初始化，不下载权重；缺 `opt_einsum` 时用 `_stubs/` 兜底） | exit 1：阶段一 `linear_backward:170` view RuntimeError，阶段二 `sum.py:385 KeyError: 'complex64'` | exit 0，`E2E OK`（loss rel ≤ 1e-3、反向完成、除 `filter_fn.*` 预存缺口外梯度齐全且有限） | 1800s |

- `_stubs/opt_einsum.py`：仅当宿主未安装 `opt_einsum` 时由 e2e 脚本注入 import 兜底
  （宿主 transformers 的 import 需要，非 FlagGems 侧依赖）；不改变任何被测行为。
- 判据细节（梯度存在性+有限为判据、数值仅打印；loss 在反向前快照为 Python float）见
  脚本 docstring 与 `reports/validation.md`。
- 直接调用 kernel 通过不等于 Torch dispatch 或端到端通过：本修复三个层级各自有脚本。
- 清单只引用实际存在的脚本，不创建默认 PASS 占位文件。
