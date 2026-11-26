# 验证入口

所有脚本把**目标源码树**（FlagScale 仓库根，含 `flagscale/` 包）放在 `PYTHONPATH` 上：

```bash
export PYTHONPATH=/path/to/FlagScale          # 仓库根，不是 src/
python3 tests/<script>.py --json out.json     # 打印一行 RESULT_JSON 并写 out.json
```

FlagScale 仓库布局是根下 `flagscale/` 包（无顶层 `src/`）；`flag-fixes verify` 预置的
`PYTHONPATH=<target>/src` 不存在但无害，fix.yaml 各命令的 `env.PYTHONPATH={target}`
负责真正生效的路径。`--json` 给相对路径时按当前目录解析。

依赖：torch、transformers、omegaconf（`flagscale.models.retrieval` 的导入链）。两个
launcher（`test_reranker_fsdp2_step.py`、`e2e_reranker_train.py`）通过
`torch.distributed.run` 起子进程；e2e 另需 pyarrow/PIL 和真实资产。缺依赖或缺设备时
脚本打印 `SKIP: ...; exit 2` 并返回 **2**（跳过不算通过）。

设备：昆仑芯 P800 OAM（96 GiB），torch 2.9.0+cu129（torch_xmlir shim），
transformers 4.57.6。设备命令沿用平台启动脚本的 `XMLIR_BMM_DISPATCH_VALUE=0`。

## 自包含设计

除 e2e 外的脚本都不需要 16 GB 权重：显存行为由 shape 决定，与权重取值无关。

- 架构取发布版 `config.json`（`fixtures/qwen3_reranker_8b_config.json`，即
  Qwen/Qwen3-Reranker-8B revision `77d193c7` 的原文件），权重在设备上按种子随机初始化为 bf16；
- 分词用 `_reranker_common.StubTokenizer`（每字符一个 id、右 padding、确定性），只决定 shape 与 mask；
- 数值对照在同一次运行、同一份权重上比较"适配器 forward"与"基线公式"，所以随机权重
  也是有效检验；真实权重下的对照由 e2e 提供。

`_reranker_common.py` 是共享工具；`_fsdp2_worker.py` / `_e2e_trainer_entry.py` 只被
launcher 启动，不单独作为验证入口。

## 脚本

| 脚本 | 层级 | 设备 | 覆盖 | 修复前预期 | 修复后预期 | 超时 |
| --- | --- | --- | --- | --- | --- | --- |
| `test_reranker_lmhead_dispatch.py` | dispatch | CPU | 2 层随机 Qwen3 + 真实适配器：混合/等长/截断/单条/32 行共 5 组 batch 的分数对基线公式（fp32，≤1e-4）、`lm_head` 被调用一次且输入是 `[rows, hidden]`、梯度到达 lm_head/embedding/norm/块、梯度检查点、kwargs、score-head 分支 | **exit 1**（lm_head 输入 `[rows, seq, hidden]`，6/25 例失败） | exit 0（25/25） | 900 s |
| `test_reranker_fsdp2_step.py` | dispatch | 默认 CPU（gloo、单 rank）；`--device cuda --nproc 2` 上设备 | 真实 `parallel_utils.apply_fsdp2`：lm_head 是独立分片的 FSDP 模块且带 FSDP hook；每步两个 micro-batch（先关梯度同步再同步，对应 `gradient_accumulation_steps=2`）；梯度存在且有限、optimizer 更新 lm_head 分片；分数对基线公式 | **exit 1**（lm_head 输入三维） | exit 0 | 1800 s |
| `test_reranker_scores_equivalence.py` | regression | XPU | 真实架构（36 层、词表 151,669）随机 bf16 权重：bf16 autocast 与 fp32 下分数对基线公式；BCE 反向后 lm_head/embedding/norm/首尾块梯度对基线公式 | exit 0（平凡：基线即参考） | exit 0（非平凡） | 3600 s |
| `repro_reranker_logits_alloc.py` | repro | XPU（≥90 GiB 单卡） | batch 32（64 对 × 640）：part 1 记录 `lm_head` 单次输出；part 2 在"61 GiB 常驻 + 34 GiB 不等大碎片"的堆上跑一步真实 `reranker_loss` 前向+反向 | **exit 1**（`Tried to allocate 11.57 GiB`） | exit 0（同一步跑通） | 1800 s |
| `bench_reranker_step.py` | benchmark | XPU | batch 16/32、长度 640 的前向/反向耗时（墙钟与 event 中位数）与峰值显存 | exit 0（记录） | exit 0（记录，对照两份 JSON） | 3600 s |
| `e2e_reranker_train.py` | e2e | XPU | 原始 trainer `flagscale.train.train_retrieval` + 真实 checkpoint + T2Ranking `triplet-15`：逐 optimizer step 的 allocated/reserved、loss；资产来自 `--model`/`$RERANKER_MODEL`、`--data`/`$RERANKER_DATA` | 视配置（2 卡 batch 32 在本机被 F-P800-1 阻断） | 视配置 | 3600 s |

判定口径：

- repro 的"碎片堆"是**构造**的：权重 15.25 GiB 加每参数 3 个 bf16 替身（梯度 + AdamW
  两阶矩）凑到 61.0 GiB 常驻——与真实 trainer step 1 之后实测的 61.03 GiB 相同——再把余下
  空闲显存按 5/2.5/3.75/2/4.5/3/6/2.25 GiB 铺满后释放。它不是 trainer 自然形成的碎片
  历史，只用来在同一张卡上稳定复现"大请求 + 碎片"这一组合。
- 随机初始化的模型 loss 数值本身没有意义；两棵树的 loss 逐位相同才是检验点。
- 直接调用 kernel 通过不等于端到端通过；e2e 之前先看 dispatch 与 repro。

## 诊断脚本（不作为验证命令）

`probe_p800_allocator.py`：记录本机分配器的两条平台性质——`expandable_segments` 不被支持、
不等大空洞交还设备后 11.57 GiB 请求在数十 GiB 空闲时失败而 1.25 GiB 请求成功。它只记录
行为、不判定修复，需要一张空闲的 ≥ 90 GiB 卡，结果在 `../results/allocator-probe-*.json`。

`build_long_parquet.py`：从 T2Ranking shard 0 里选出正、负样本 prompt 都超过 640 token 预算的行
（454 行），使每一对都被截断到 `max_length`——平台报错时的 padded 形状。2 卡自然复现
（`e2e_reranker_train.py --data <该 parquet>`）用它；输出是确定性的。
