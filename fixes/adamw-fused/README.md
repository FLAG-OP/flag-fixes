# adamw-fused

- ID：adamw-fused
- 状态：draft（未具备部署资格）
- 负责人：待定
- 上游 issue/PR：无（FlagScale 本地修复，尚未提交上游）
- 受影响对象：`flagscale/train/train_retrieval.py::build_optimizer`（AdamW 构造），
  目标包 flagscale @ 89f342e690053e07dda845f1365089ec00464e09；
  影响所有用 AdamW 的 native 训练路径，收益实测在 KunlunXIN（P800/XPU3）
- 问题与修复后行为：厂商 torch 把 **foreach AdamW 也分解成逐参数 kernel**
  （BGE-M3 的 389 个参数张量 = 每步 2,688–3,890 次 launch、~26–42 ms），
  A100 上同样的更新是 `_foreach_*` 融合 kernel（18.6 ms）——这是平台文档
  差距分解表里 "other 104.3 ms / 58.1×" 的主要成分之一。本修复让
  `build_optimizer` 在参数为普通 CUDA fp32/bf16/fp16 张量时改用
  **`fused=True`**（单支多张量 kernel）：BGE-M3 形状 336 张量集上
  **27.6 → 11.5 ms/步（2.4×），launch 2,688 → 672**，数值与 stock 逐位量级一致
  （bf16 max|diff| 3.1e-5 = 1 ulp，fp32 3.0e-8）。

选择规则（`adamw_impl_kwargs`）：

| 条件 | 选择 |
|---|---|
| `FLAGSCALE_ADAMW_FUSED=0/false/off` | stock（强制关闭） |
| **vendor 非 KunlunXIN**（NVIDIA/thead/ascend/未知） | **stock + 记录原因**（auto 与 `=1` 均回退；fused 未在这些平台验证——PR #3 评审修订） |
| `=1/true/on`（KunlunXIN + CUDA fp32/bf16/fp16 + 非 DTensor） | fused（意愿表达，护栏仍生效） |
| 未设置 / `auto`（默认，KunlunXIN） | 参数全部是 CUDA fp32/bf16/fp16 且非 DTensor → fused；否则 stock |
| 混合 dtype（bf16+fp32 等，均在支持集合内） | fused（torch 按 dtype 分组调用） |
| DTensor / CPU / 空 / 不支持的 dtype | stock（记 log） |
| 构造时 torch 拒绝 `fused` kwarg | 捕获后回退 stock（记 log） |

vendor 检测顺序：`FLAGSCALE_VENDOR` → `GEMS_VENDOR` → `torch_xmlir` shim
存在性（`_detect_vendor()`）；已验证集合 `_FUSED_ADAMW_VALIDATED_VENDORS`
当前仅含 `kunlunxin`，其它平台完成独立验证后在集合中加入即可。

选择矩阵共 **18 例**（PR #3 二轮评审后含 vendor nvidia/thead/kunlunxin、显式 forced=1 回退日志断言、
混合 dtype、torch_xmlir 探测等断言）。

FSDP2/DTensor 分片参数强制走 stock（fused 不感知 DTensor），因此
reranker 8B 的 FSDP2 路径不受影响、不冒险。

验证入口（4 个命令，见 fix.yaml `validation.commands`）：

```
tests/repro_adamw_unfused.py      # 基线 exit 1（stock 31.5ms），修复后 exit 0（11.0ms）
tests/test_adamw_numerics.py      # 5 步同梯度序列，bf16/fp32 vs stock 全部 allclose
tests/test_adamw_selection.py     # 选择矩阵：env 覆盖 / DTensor 护栏 / CPU 护栏 / 回退
tests/bench_adamw_impls.py        # stock/foreach/build_optimizer 时间 + launch 数
```

原始数据在 `results/`（2026-11-10，P800/XPU3 软件栈，torch 2.9.0+cu129）。
归因背景：`/workspace/problem29/ATTRIBUTION.md` §二；
平台文档侧记录：FlagScale `docs/retrieval/14-glue-kernel-origin.md`。

**边界**：`deployable=false`、`hardware_status=unverified`——未在平台
整模型训练里跑过验收；收益数字来自重建负载（XLM-R large 随机权重 +
同款 5-forward×batch10×accum4 配方），与平台文档的 kernel 级数字交叉对齐
（偏差 ≤10%），但不替代真实 step 验收。Adafactor 路径未动。
