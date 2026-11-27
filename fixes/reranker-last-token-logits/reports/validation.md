# 验证记录：reranker-last-token-logits

设备数据于 2026-11-26/27 在本机采集；命令、退出码与原始 JSON/日志均在 `../results/`。
平台形态（2 卡 FSDP2）的**基线失败已自然复现**；修复树在同一配置下被本机的 XCCL
集合通信卡死（F-P800-1）反复打断，没有一次干净退出，详见"2 卡 FSDP2"一节。
`hardware_status` 保持 `unverified`。

审阅修订（PR #11 评论 7210838015，2026-11-27）：两个 launcher、等价性脚本有改动并新增 launcher 自测，受影响的
脚本都已重跑；改动、理由与证据见文末"审阅修订"一节。

## 环境与源码

| 项 | 值 |
|---|---|
| 执行日期 | 首轮 2026-11-26 20:22 – 2026-11-27 08:44；审阅修订后的重跑 2026-11-27 13:50 起（本机时间） |
| 执行者 | AI 代理自动执行，没有人工复核；全部命令与原始输出都在 `../results/` |
| 机器 | KunlunXin P800 OAM ×8 节点，本验证用可见的 2 张卡（每卡 96 GiB）；同节点卡 3、4 上始终有其他进程的负载（非本验证） |
| 驱动 / 运行时 | XPU-SMI 5.0.21.47 / XPU-RT 5.13.0；分配器后端 `xpu native` |
| torch | 2.9.0+cu129（torch_xmlir shim：XMLIR dev+2026060515、XCCL 3.1.0.2；`torch.cuda.get_device_name` = "GPU"） |
| transformers | 4.57.6 |
| FlagGems / Triton | 未启用（`use_flaggems=false`），与本次验证无关 |
| FlagScale 基线 | `wxj630/FlagScale` `feat/retrieval-flagos` @ `89f342e690053e07dda845f1365089ec00464e09`（tarball 快照；目标文件 blob `e5abccd6` 与 GitHub API 一致，到分支头 `555a7d6a` 未变） |
| 目标树 | 基线 = 该 commit 的干净快照；修复 = 基线 + `patches/reranker-last-token-logits.patch`（`check`/`plan` 校验 after hash） |
| 模型（e2e） | `Qwen/Qwen3-Reranker-8B` revision `77d193c791ed757ca307ee72715aa132723da912`；5 个 safetensors 分片 sha256 与 hub LFS oid 逐一相同（`model-0000{1..5}`：`22cdfea4…`、`d2163b74…`、`a5038caa…`、`247f8553…`、`8ba41b93…`） |
| 数据（e2e） | `sentence-transformers/t2ranking` `triplet-15/train-00000-of-00011.parquet`（30,920 行，sha256 `4d92878e…`）；长文档子集由 `tests/build_long_parquet.py` 从中确定性选出 454 行（sha256 `0d3a0d41…`） |
| 环境变量 | 设备命令 `XMLIR_BMM_DISPATCH_VALUE=0`（平台 P800 启动脚本同值）；e2e 另设 `WANDB_DISABLED=true`、`HF_HUB_OFFLINE=1` |

hash（fix.yaml 记录同值）：

```
before  501470249c00c934268bd3a02bb29cad0884dd4a9336635c1d2f85a70781f7fc  flagscale/models/retrieval/qwen3_reranker.py
after   dd34c0dd0459f0469280968496ed7a42d86e46fb3feac6ada822d221ccbfcd89  flagscale/models/retrieval/qwen3_reranker.py
patch   ee7ded27c09fd2ae7b15f499ce1f14a79891ecf20b428f30fa812848197537d4  patches/reranker-last-token-logits.patch
```

依赖：torch、transformers、omegaconf（`flagscale.models.retrieval` 的导入链）；e2e 另需 pyarrow、PIL。
omegaconf 不在共享环境里，本机以 `pip install --target` 装到独立目录，并用一个解释器包装
把它喂给 `verify`；未修改共享 conda 环境。

## 命令与结果

以下命令均在 `fixes/reranker-last-token-logits/` 下执行；`<树>` 是基线或修复后的 FlagScale
仓库根（放在 `PYTHONPATH` 上），设备命令带 `XMLIR_BMM_DISPATCH_VALUE=0`。证据列的文件都在
`../results/`，基线与修复各一份。

| 项目 | 精确命令 | 判定标准 | 修复前 | 修复后 | 证据 |
| --- | --- | --- | --- | --- | --- |
| 最小复现（单卡碎片堆） | `PYTHONPATH=<树> python3 tests/repro_reranker_logits_alloc.py` | 修复前一步内 OOM；修复后同一步跑通且 lm_head 只输出 `[rows, vocab]` | exit **1**：`Tried to allocate 11.57 GiB … 20.63 GiB is free … 73.51 GiB is allocated by PyTorch` | exit **0**，峰值 +25.36 GiB | `repro-{base,fixed}-2026-11-26.{json,log}` |
| 最小复现（平台形态，2 卡） | `RERANKER_DATA=<长文档 parquet> PYTHONPATH=<树> python3 tests/e2e_reranker_train.py --nproc 2 --parallel fsdp2 --master-dtype fp32 --batch 32 --min-steps 3 --no-save` | 基线出现平台记录的报错；修复后走完 3 步 | exit 1，**step 3 反向 OOM**，与平台记录逐项吻合 | 4 次尝试：第 4 次越过基线 OOM 的第 3 步（step 1–3 走完，`training_steps_complete=true`），停在 validation，`passed=false` | `e2e-fsdp2-2card-{base,fixed}-b32-longdoc-*` |
| dispatch（CPU） | `PYTHONPATH=<树> python3 tests/test_reranker_lmhead_dispatch.py` | lm_head 只收到 `[rows, hidden]`，分数对基线公式 ≤ 1e-4，梯度到达各层 | exit **1**（6/25 例失败：lm_head 收到 `[rows, seq, hidden]`） | exit **0**（25/25） | `dispatch-{base,fixed}-2026-11-26.{json,log}` |
| dispatch（真实 FSDP2，CPU/gloo） | `PYTHONPATH=<树> python3 tests/test_reranker_fsdp2_step.py --device cpu --nproc 1 --steps 2` | lm_head 是分片的 FSDP 模块且钩子照常；每步两个 micro-batch；梯度与 optimizer 更新正常 | exit **1**（lm_head 输出 `[16, 258, 151669]`） | exit **0**（`[16, 151669]`） | `fsdp2-cpu-{base,fixed}-2026-11-26.{json,log}` |
| dispatch（真实 FSDP2，P800 ×2） | `PYTHONPATH=<树> python3 tests/test_reranker_fsdp2_step.py --device cuda --nproc 2 --steps 2` | 同上，bf16 autocast | exit **1**（328.6 s 跑完，仅 lm_head 投影判据失败） | exit **0**（26.3 s，两 rank 全过）；此前两次被 XCCL watchdog 阻断 | `fsdp2-cuda-2card-*` |
| 回归 / 边界 | `PYTHONPATH=<树> python3 tests/test_reranker_scores_equivalence.py` | 两侧各 9 个预期梯度、逐张量有限；分数在参考最大 logit 的 2 个 bf16 ulp 内，梯度相对差 ≤ 3%（bf16 权重）；fp32 权重下 1e-4 / 1e-3；7 个比较器自检用例全部按预期 | exit 0（基线即参考，平凡通过） | exit **0**（14/14：bf16 权重 5 例 max\|diff\| = 0.0；fp32 权重 2 例分数 2.4e-7、梯度相对差 2.0e-4；比较器 7 例） | `equivalence-{base,fixed}-2026-11-27.{json,log}`（审阅修订后重跑；11-26 的 5 例旧版记录保留） |
| 性能 / 精度 | `PYTHONPATH=<树> python3 tests/bench_reranker_step.py --iters 5` | 记录前后耗时与峰值显存（见 `../benchmarks/README.md`） | exit 0 | exit 0 | `bench-{base,fixed}-2026-11-26.{json,log}` |
| e2e（真实 trainer） | `RERANKER_MODEL=<checkpoint> RERANKER_DATA=<triplet-15> PYTHONPATH=<树> python3 tests/e2e_reranker_train.py --nproc 1 --parallel ddp --master-dtype bf16 --batch 16 --min-steps 5 --max-grad-norm 0 --no-save` | 两棵树都跑完 5 步；loss 一致 | exit 0 | exit 0；5 步 loss、validation、test loss 在日志记录精度（6 位小数）下一致 | `e2e-1card-{base,fixed}-2026-11-27.{json,log,train.log}` |
| launcher 自测（CPU） | `python3 tests/test_reranker_launchers.py` | 两个共用同一 worker 脚本路径的 torchrun 作业（各 2 rank）：超时那组自己的 rank 与子进程在 launcher 返回时全部退出；另一组未被信号，每个 rank 在超时前启动、超时后结束，exit 0。可见设备 7 种输入 | 旧清理方式（`--cleanup legacy`）：另一组在 60.2 s 被杀（返回码 -9），超时那组留下 2 个子进程；3/14 项失败，exit **1** | exit **0**（17/17） | `launcher-selftest-{scoped,legacy}-2026-11-27.{json,log}` |
| launcher 启动路径（真实 trainer，修复树） | `env -u CUDA_VISIBLE_DEVICES -u XPU_VISIBLE_DEVICES RERANKER_MODEL=<checkpoint> RERANKER_DATA=<triplet-15> PYTHONPATH=<树> python3 tests/e2e_reranker_train.py --nproc 1 --parallel ddp --master-dtype bf16 --batch 16 --min-steps 5 --max-grad-norm 0 --no-save`；另两种环境：`CUDA_VISIBLE_DEVICES=1,2`；仅 `XPU_VISIBLE_DEVICES=1,2` | launcher 不因可见设备变量崩溃，trainer 被启动 | `c6f73493` 的 launcher：默认可见与仅 XPU 列表都是 `KeyError: 'CUDA_VISIBLE_DEVICES'`，exit 1，trainer 未启动 | 默认可见：exit 0（5 步）；显式选卡：exit 0（5 步）；仅 XPU 列表：launcher 透传并启动，trainer 在厂商栈 import 阶段失败（exit 1，0 步） | `launcher-visible-devices-before-c6f73493-2026-11-27.log`、`e2e-1card-fixed-{defaultvis,explicitvis,xpuonly}-2026-11-27.{json,log}` |
| 安装 / 回滚 | `python3 -m flag_fixes --root <本仓库> check / plan / apply / status reranker-last-token-logits --target <树>` | 基线 hash 命中、patch 可干净应用、结果与快照一致；未获部署资格时 `apply` 拒绝 | — | `check`/`plan` exit 0；`apply` exit 2（`deployable=false`）；`status` clean | 本页"工具链校验"一节 |

判定：只有退出码 0 算通过；缺设备或缺 checkpoint/数据时脚本退出码 2（跳过，不算通过）。

仅 XPU 列表那次 trainer 的失败（`e2e-1card-fixed-xpuonly-2026-11-27.train.log`）：`import torch` 之后的第一次
CUDA 调用（FlagScale 注册平台时导入 flag_gems → `torch.cuda.get_device_name()`）报
`device >= 0 && device < num_gpus INTERNAL ASSERT FAILED … device=2, num_gpus=2`。同一环境下的独立探针：
`torch.cuda.device_count()` 返回 8，逐卡 `get_device_name` 到第 3 张失败——Python 侧仍按 8 张卡枚举，运行时侧
只给了 XPU 列表里的 2 张。这是本机 torch_xmlir 栈对"只设 `XPU_VISIBLE_DEVICES`"的处理，不在本修复范围；
launcher 的责任（不崩溃、不替用户发明 CUDA 列表、如实透传）已经满足，但这一种情况**不算跑通**。

### 证据的采集时间与脚本版本

- 11-26 的单卡证据（dispatch、fsdp2-cpu、repro、equivalence、bench）与 11-27 的 allocator 探针采集时
  修复目录尚未提交；脚本与提交版本逻辑相同，之后只删了两个未使用的辅助函数、改了复现脚本的一处
  说明文字。
- 三份 2 卡随机权重 trainer 结果（`e2e-fsdp2-2card-*-randominit-*`）和三份 2 卡 FSDP2 测试结果
  （`fsdp2-cuda-2card-base-*`、`…-fixed-attempt{1,2}-blocked-*`）用的是入库前的脚本副本，其
  `command` 字段指向 `/tmp/opencode/fixwork/`。与提交版本的差别是：trainer launcher 后来修正了
  输出目录为空的问题并补了 loss 记录；FSDP2 测试后来把"lm_head 梯度非零"的判据改为只对拥有
  "yes"/"no" 两行的分片成立。这两处都不影响这些文件记录的事实（报错文本、显存数字、卡死签名、
  lm_head 收到的形状）。
- `verify-2026-11-27.json` 是干净提交 `c2aef972` 上的重跑（`source_dirty=false`，6/6 rc=0），对应审阅修订前的脚本。
- 审阅修订（PR #11 评论 7210838015）改了两个 launcher（超时清理按作业限定、可见设备映射）、等价性脚本（按实际
  dtype 命名、补 fp32 权重对照、梯度比较器）并新增 launcher 自测。受影响的都在修订后重跑：
  `equivalence-{base,fixed}-2026-11-27.*`、`launcher-selftest-*`、`fsdp2-cuda-2card-fixed-rerun-*`、
  `e2e-1card-fixed-{defaultvis,explicitvis,xpuonly}-*`。`repro`、`bench`、CPU dispatch 与 FSDP2 worker 的判据没有
  改动，11-26 的记录仍对应现版脚本。
- `verify-2026-11-27-r2.json` 是修订提交 `217d46f` 上的重跑（7 个命令，`source_dirty=false`），取代上一条。
  重跑时 fix 目录没有未提交改动：verify 的输出要先写到仓库之外、跑完再拷进来，否则输出文件本身会被
  `git status` 计入 dirty（第一次就把输出重定向进了 `results/`，那份 7/7 的记录带着 `source_dirty=true`，已丢弃）。
- 工具与脚本的原始输出含本机绝对路径，未做脱敏或改写。

## 关键数据

### 复现（`repro-*-2026-11-26.json`）

| 项 | 基线 | 修复 |
|---|---|---|
| part 1：`lm_head` 单次输出 | `(64, 640, 151669)` = 11.5714 GiB | `(64, 151669)` = 0.0181 GiB |
| part 1：前向峰值（超出权重） | +25.54 GiB | +14.69 GiB |
| part 2：常驻（权重 + 每参数 3 个 bf16 替身） | 61.01 GiB | 61.01 GiB |
| part 2：碎片堆 | 9 块共 34.0 GiB，最大 6.0 GiB；设备空闲 0.13 GiB | 同 |
| part 2：一步前向 + 反向 | **OutOfMemoryError**：`Tried to allocate 11.57 GiB … of which 20.63 GiB is free. Of the allocated memory 73.51 GiB is allocated by PyTorch, and 1.57 GiB is reserved by PyTorch but unallocated` | 完成，loss 0.968551，峰值 +25.36 GiB（allocated），reserved 峰值 95.58 GiB |

### 真实 trainer，单卡 bf16，真实权重，batch 16，5 步（`e2e-1card-*-2026-11-27.json`）

| step | 基线 allocated / reserved（GiB） | 修复 allocated / reserved（GiB） | window_loss（两棵树在日志记录精度下一致） |
|---|---|---|---|
| 1 | 61.019 / 77.285 | 61.024 / 77.322 | 0.627536 |
| 2 | 61.012 / 84.217 | 61.012 / 77.326 | 0.466457 |
| 3 | 61.012 / 89.334 | 61.013 / 79.641 | 0.370828 |
| 4 | 61.014 / 92.762 | 61.015 / 79.641 | 0.639762 |
| 5 | 61.021 / 87.084 | 61.019 / 79.641 | 0.592430 |

train_loss 0.539403、validation_loss 0.461096、test_loss 0.329392 两棵树在日志记录精度下一致：trainer 的 loss 行
只打印 6 位小数，JSON 里保存的就是这些日志行，没有更高精度的值可比。
每步 allocated 峰值 76.27 GiB 两棵树在记录精度（0.001 GiB）下相同——它来自厂商 AdamW 的 15.25 GiB 临时张量
（61.02 + 15.25），高于 batch 16 的反向峰值，所以差异体现在 reserved 上。

## 2 卡 FSDP2（平台形态）

配置：`--nproc 2 --parallel fsdp2 --master-dtype fp32`，梯度检查点、`max_length` 640、
`num_negatives` 1，与 `run-p800-fsdp2-2card.sh` + `qwen3_reranker.yaml` 一致；
`RERANKER_E2E_NO_SAVE=1` 跳过 32 GB 的 checkpoint 写出。

### 自然复现（长文档子集、真实权重、batch 32）

子集里每对的正、负样本 prompt 都 ≥ 592 token，全部被截断到 640——即平台报错时的 padded 形状。

逐 step 对照。两边的数字各自实测，两个 rank 相同：基线来自 `memory.json.rank*` 与 trainer 日志；修复树的几次
尝试被 XCCL 终止、没有 per-rank 记录，数字来自 trainer 日志的 `[e2e-mem]` 行。

| step | 基线：step 后 allocated / reserved，step 峰值 allocated / reserved（GiB） | 修复：同（GiB） | loss（running / window，trainer 保留 6 位小数） |
|---|---|---|---|
| 1 | 61.01 / 92.06，76.26 / 92.06 | 61.01 / 92.20，76.26 / 92.20 | 基线 0.392380 / 0.392380；修复 0.392380 / 0.392380 |
| 2 | 61.01 / 92.27，87.02 / 95.13 | 61.01 / 90.34，84.41 / 95.02 | 基线 0.377111 / 0.361841；修复 0.377111 / 0.361841 |
| 3 | **反向 OOM**：请求 11.57 GiB 时设备空闲 20.77、PyTorch 已分配 71.66、reserved 未用 2.98；没有 step 3 的 loss，也没有 step 后的显存 | 61.01 / 90.34，84.41 / 94.32；loss 0.454597 / 0.609568 | 只有修复树有记录：**修复版越过了基线的 OOM 点**；基线没有可比的 step 3 |

- 可比的 loss 只有前两步，它们在日志记录精度下一致；第 3 步只有修复树的数字，不能写成"与基线相同"。
- 基线的失败点：`Tried to allocate 11.57 GiB … of which 20.77 GiB is free. Of the allocated memory 71.66 GiB is
  allocated by PyTorch, and 2.98 GiB is reserved by PyTorch but unallocated`；traceback 落在
  `train_epoch → loss.backward() → hydrax LinearFunction.backward → linear_bwd_dgrad → fc_fusion → TensorViewState.value`，
  即 lm_head 反向对整序列 logits 梯度的物化。
- 平台记录（11.57 GiB / 20.90 GiB free / 71.68 GiB allocated）与本机基线（11.57 / 20.77 / 71.66）逐项吻合。

修复树的 4 次尝试（同一数据、同一配置）：

| 尝试 | 走到 | 终止方式 | 文件 |
|---|---|---|---|
| 第 1 次 | step 1（61.01 / 92.20 GiB） | 停在第 2 步，XCCL watchdog | `e2e-fsdp2-2card-fixed-b32-longdoc-attempt1-blocked-2026-11-26.*` |
| 第 2 次 | step 1–2（step 2 峰值 allocated 84.41 / reserved 95.02） | 停在第 3 步——基线 OOM 的那一步——XCCL watchdog | `e2e-fsdp2-2card-fixed-b32-longdoc-attempt2-blocked-2026-11-27.*` |
| 第 3 次 | 0 步：模型加载与 FSDP2 分片完成后停在第 1 步的前向，900 s 内 watchdog 也未触发 | launcher 超时终止 | `e2e-fsdp2-2card-fixed-b32-longdoc-attempt3-blocked-2026-11-27.*` |
| 第 4 次 | **step 1–3 完成**（step 3：61.01 / 90.34，峰值 84.41 / 94.32）——越过基线 OOM 点；随后停在 validation 的首个集合通信 | 420 s 后 launcher 超时终止（`training_steps_complete=true`，`passed=false`） | `e2e-fsdp2-2card-fixed-b32-longdoc-plain4-2026-11-27.*` |

### 常规数据

| 权重 | 树 | batch | 完成 | 终止位置 | 文件 |
|---|---|---|---|---|---|
| 随机 | 基线 | 32 | step 1（61.02 / 92.24 GiB） | 第 2 步，`Watchdog caught collective operation timeout: WorkXCCL(SeqNum=190, OpType=_ALLGATHER_BASE, NumelIn=2048, NumelOut=4096)` | `e2e-fsdp2-2card-base-randominit-2026-11-26.*` |
| 随机 | 修复 | 32 | step 1–2（step 2 峰值 allocated 80.28 = 61.02 + 19.26，reserved 95.19） | 第 3 步，同样 watchdog，`SeqNum=379` | `e2e-fsdp2-2card-fixed-randominit-2026-11-26.*` |
| 随机 | 修复 | 16 | 0 | 第 1 步内 `kl3ChannelCheckErrors failed … A kernel exception has occurred`（status 702），20 分钟后 `xpu_free error code=-714` | `e2e-fsdp2-2card-fixed-b16-randominit-2026-11-26.*` |
| 真实 | 修复 | 32 | step 1–3 + validation（train_loss 0.450768，validation_loss 0.512823；step 3 峰值 allocated 83.20） | 训练循环结束后的收尾集合通信，watchdog `SeqNum=609` | `e2e-fsdp2-2card-fixed-2026-11-26.*` |
| 真实 | 基线 | 32 | step 1–2（loss 在日志记录精度下与修复树一致） | 第 3 步，watchdog | `e2e-fsdp2-2card-base-2026-11-26.*` |

常规数据下 batch 长度随窗口变化，前几步较短，基线没有触发 11.57 GiB 的请求；自然 OOM 要在
长 batch 上才出现（上一节）。

### FSDP2 测试（`test_reranker_fsdp2_step.py --device cuda --nproc 2`）

4 层、8 行、2 步，真实 `apply_fsdp2` + bf16 autocast。基线 1 次尝试跑完（328.6 s，仅 lm_head
投影判据失败）；修复树 3 次尝试——前两次分别停在 `SeqNum=3`（块 all-gather）与 `SeqNum=22`
（norm all-gather）的 XCCL watchdog，第三次 26.3 s 两 rank 全过：lm_head 是 `Shard(0)` DTensor
且带 FSDP pre/post hook，输入 `[16, 4096]`、输出 `[16, 151669]`（基线 `[16, 258, 151669]`，
1.25 GB/次），分数对整序列参考 max\|Δ\| = 0.0，两 rank 的 loss 与基线树在记录精度下一致
（`[0.805798, 17.0]`，worker 只保留 6 位小数）。审阅修订后经新 launcher 在 2 卡上重跑一次：26.4 s，两 rank 全过，
exit 0（`fsdp2-cuda-2card-fixed-rerun-2026-11-27.{json,log}`）。梯度：拥有 "yes"/"no" 两行的 rank 0 上 lm_head 梯度范数 20.37，
rank 1 上为 0——lm_head 沿词表分片，只有这两行有梯度，基线同样如此。

### 阻断统计

本机 2 卡的主干触发 **14 次**（trainer 10 次：基线 3 次、修复树 7 次；FSDP2 测试 4 次：基线 1 次、
修复树 3 次）。按各自目标跑完的 3 次——基线自然 OOM（长文档）、基线 FSDP2 测试、修复树 FSDP2
测试；修复树另有 2 次走完 3 个 optimizer step（常规数据那次含 validation，长文档 plain4 那次
正好走过基线 OOM 的第 3 步）后卡住；其余 9 次被 XCCL 集合通信超时或设备异常提前终止。
两棵树都中招。终止点与 payload 大小无关（SeqNum=3/22/190/379/609，`_ALLGATHER_BASE`），
与已登记的平台阻断 **F-P800-1**（`validation/kunlunxin-p800-real-bge-m3/README.md` §E.3）同一签名。
因此**"修复后 batch 32 在 2 卡 FSDP2 上完整跑通"没有拿到干净结果**；已得到的是：同一数据上
基线第 3 步 OOM，修复版越过了这一步。前两步两边都有 loss 与显存记录（loss 在日志记录精度下一致，显存各自列在
上面的逐 step 表里），第 3 步只有修复树有记录。

### 对阻断本身的探索（非本修复范围，供运行时层参考）

| 探针 | 做法 | 结果 | 文件 |
|---|---|---|---|
| 步骤边界主机同步 | 2 rank FSDP2 测试，交错 3+3 次：plain / 每次前向前与每个 optimizer step 后 `torch.cuda.synchronize()` | plain 1/3、sync 2/3 通过——不显著，开关已从 harness 移除 | `f-p800-1-sync-ab-2026-11-27.json` |
| 主机时序（`tests/probe_xccl_timing.py`） | 2 层小模型走真实 `apply_fsdp2`，4 步，交错 3 轮 × 4 变体 | plain **0/3**（都停在第 0 步前向）、rank 1 单边睡 3 s **0/3**（停在第 0/2/3 步）、`barrier` 1/3、**两 rank 都睡 3 s 3/3 干净退出** | `f-p800-1-timing-probe-2026-11-27.json` |
| 把 3 s 空闲搬进 trainer | 修复树、长文档、batch 32，每次前向前所有 rank 睡 3 s（带标注的诊断运行） | step 1–2 完成后仍停在第 3 步——没有带过来 | `e2e-fsdp2-2card-fixed-b32-longdoc-idlegap-2026-11-27.train.log` |

样本都很小，只作为观察记录：小模型上"所有 rank 前向前有一段空闲"与不卡死相关，但单边延迟和
barrier 都不行，而且这个现象在 8B trainer 上没有复现出可用的规避办法。两处试验用的 harness 开关
都已移除，仓库里的脚本只跑 plain 运行。

### 分配器探针（`probe_p800_allocator.py`，`allocator-probe-*.json`）

| 模式 | 堆状态 | 11.57 GiB 请求 |
|---|---|---|
| A 空卡 | 设备空闲 95.97 GiB | 成功 |
| B 82 × 1 GiB，隔一块释放一块并交还设备 | 设备空闲 54.97 GiB，PyTorch allocated 41.00 | 成功 |
| C 同 B，空洞留在 PyTorch 缓存 | 设备空闲 13.97 GiB，缓存空闲 41 GiB | 成功 |
| D 221 块（64 MiB–1 GiB）释放 2/3 并交还设备 | 设备空闲 **72.84 GiB**，PyTorch allocated 23.12，缓存空闲 0 | **OOM**：`Tried to allocate 11.60 GiB … of which 72.84 GiB is free … 0 bytes is reserved by PyTorch but unallocated`；随后 1.25 GiB 请求成功 |
| E 同 D，子进程设 `PYTORCH_ALLOC_CONF=expandable_segments:True` | 设备空闲 72.75 GiB | **同样的 OOM**，且没有任何告警——设置被静默忽略 |
| E 同 D，子进程设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 设备空闲 72.75 GiB | **同样的 OOM**，并告警 `UserWarning: expandable_segments not supported on this platform` |

结论：这张卡上一次大请求能否成功取决于空闲段的形状；PyTorch 常规的解法（expandable
segments）在两种拼写下都不生效。修复后的步骤最大单次请求在 1.25 GiB 这个量级（D 末行可满足）。

不生效的原因定位到构建层（记录在 `../../../validation/kunlunxin-p800-reranker-fsdp2/results/env_record.json`）：
PyTorch 的 expandable segments 建立在 CUDA 驱动的虚拟内存管理接口上；本机 `libc10_cuda.so` 里
`cuMemCreate`、`cuMemAddressReserve`、`cuMemMap` 等全是只会断言的占位实现，几个 torch 动态库对它们
没有任何外部符号引用，`CUDAAllocatorConfig.h` 在构建缺少 `PYTORCH_C10_DRIVER_API_SUPPORTED` 时把该
选项强制关掉并告警；实际生效的是厂商的 `xpu native` 分配器后端。

## 工具链校验

- `flag-fixes --root . validate`：valid（15 fixes = 9 external + 6 local）
- `flag-fixes --root . check reranker-last-token-logits --target <基线树>` / `plan`：exit 0
  （基线 hash 命中、`git apply --check` 干净、patch 结果与 `src/` 快照一致）
- `flag-fixes apply reranker-last-token-logits --target <树>`：exit 2 拒绝（`deployable=false`，符合预期）
- `flag-fixes status --target <基线树>`：clean；对已打补丁的树 `check` 报 `Baseline checksum differs`（exit 2）
- `flag-fixes verify reranker-last-token-logits --target <修复树>`：**passed=true，7/7 rc=0**，在修订提交 `217d46f` 上
  运行（fix 目录无未提交改动，`source_dirty=false`），2026-11-27；原始输出
  `../results/verify-2026-11-27-r2.json`（stderr 为空）。声明命令比首轮多一个 launcher 自测
- 首轮记录 `../results/verify-2026-11-27.json`（干净提交 `c2aef972`，6/6 rc=0）对应修订前的脚本，已被上一条取代；
  首轮的 6 个命令：
  - reranker-lmhead-dispatch（dispatch）：rc=0
  - reranker-fsdp2-step-cpu（dispatch）：rc=0
  - reranker-logits-alloc-repro（repro）：rc=0
  - reranker-scores-equivalence（regression）：rc=0
  - reranker-step-benchmark（benchmark）：rc=0
  - reranker-trainer-single-card（e2e）：rc=0
- 仓库测试 `python -m pytest -q`：26 passed, 3 subtests passed；`python -m compileall -q fixes`：exit 0

verify 只输出 JSON 测试结果，不提升 `hardware_status`；上面的设备结论仍以平台验收为准。

## 审阅修订（PR #11 评论 7210838015，2026-11-27）

审阅对象是 `c6f73493`。四项意见都成立，逐项修改如下；受影响的测试结果已并入上面对应的章节。

| # | 意见 | 修改 | 证据 |
|---|---|---|---|
| 1 | 两个 launcher 超时后按入口脚本路径 `pkill -KILL -f`，会终止同一 checkout 下并行作业的 worker | 新增 `tests/_torchrun_job.py`：启动前生成作业 id 放进 torchrun agent 的环境，worker 及其子进程继承；超时后结束 agent 的进程组，再只对环境里带该 id 的进程发 SIGKILL 并等待退出。launcher 的 JSON 记录 `job_id`、`timeout_cleanup`（被信号的进程、残留）；worker 在日志里公告自己的 pid 与作业 id。新增自测 `tests/test_reranker_launchers.py`，并声明为 verify 命令 | `launcher-selftest-scoped-2026-11-27.json`：17/17，exit 0——超时那组的 2 个 rank 与 2 个子进程在 launcher 返回时全部结束；另一组未被信号，60 s 超时时刻活着、91.9 s 正常结束。`launcher-selftest-legacy-2026-11-27.json`：旧清理方式下另一组 60.2 s 被杀（返回码 -9），超时那组留下 2 个子进程，3/14 项失败，exit 1（预期） |
| 2 | 模型始终用 bf16 权重构造，名为 `fp32_no_autocast` 的两例名不副实；梯度比较会漏过两侧同时缺梯度与 NaN 差值 | 用例按实际 dtype 命名（`bf16_weights:…` / `fp32_weights:…`），每例记录权重 dtype、是否 autocast、分数 dtype；补真正的 fp32 权重对照（分数、梯度各一例）。梯度比较器先验结构（两侧各 9 个预期梯度），再逐张量检查参考值、候选值、差值有限，最后才计算相对误差；新增 7 个 `comparator_*` 自检用例 | `equivalence-{base,fixed}-2026-11-27.json`：两棵树 14/14。修复树 bf16 权重 5 例 max\|diff\| = 0.0；fp32 权重分数 2.4e-7（阈值 1e-4）、梯度相对差 2.0e-4（阈值 1e-3）——**fp32 下不是逐位相同**，此前文档里的"fp32 下 max diff = 0.0"是 bf16 权重关 autocast 的结果。一次性并排核对（未入库）：旧比较器放过"候选含 NaN""参考含 NaN""两侧同时缺一个""差值为 NaN"四种情况，"单侧缺失"直接 KeyError；新比较器六种破坏全部拒绝 |
| 3 | `env.setdefault("XPU_VISIBLE_DEVICES", env["CUDA_VISIBLE_DEVICES"])` 在没设 CUDA 列表时 `KeyError` | 映射收进 `limit_visible_devices`：只在有显式 `CUDA_VISIBLE_DEVICES` 时截断并镜像；默认可见与仅 XPU 列表原样透传。自测覆盖 7 种输入；真实 trainer 在三种环境下各启动一次 | 修复前：`launcher-visible-devices-before-c6f73493-2026-11-27.log`（默认可见、仅 XPU 列表都是 KeyError）。修复后：`e2e-1card-fixed-defaultvis-*`、`e2e-1card-fixed-explicitvis-*` exit 0（各 5 步，loss 与首轮相同）；`e2e-1card-fixed-xpuonly-*` launcher 透传并启动，trainer 在厂商栈 import 阶段失败（见"命令与结果"里 launcher 启动路径一行之后的说明） |
| 4 | trainer 的 loss 只保存 6 位小数，不能说"逐位相同"；2 卡只有前两步有双方可比的 loss，第 3 步应写"修复版越过基线 OOM 点"并分别列两边实测的显存；`WORKFLOW.md` 里"AdamW 切碎缓存导致第 2 步失败"与实际第 3 步 OOM 冲突；缺模板第 8 行 | 凡是 loss 对照一律改为"日志记录精度下一致"（README、WORKFLOW、fix.yaml、两份报告、results 索引）；2 卡一节改为逐 step 对照表，两边显存分列；WORKFLOW 第 4 行改为"触发步次取决于分配器缓存历史，机制未逐事件确认"；补第 8 行"上游：not-run" | 本页"2 卡 FSDP2"一节；`WORKFLOW.md` 第 4、6、8 行 |

修订没有改动补丁本身（patch sha256 不变），也没有改变 `deployable=false` / `hardware_status=unverified`。

## not-run（未验证清单，不记为通过）

| 项 | 原因 |
|---|---|
| 平台 2 卡 FSDP2 整模型验收（修复前后 batch 16 / 32 / accum 2 的完整训练与吞吐） | 本机 XCCL 集合通信间歇性卡死（F-P800-1），单次尝试最多走完 3 个 optimizer step + validation |
| P800 配置回到 batch 32 或 accum=2 的余量评估 | 同上；需平台环境实测后再改 `qwen3_reranker_p800.yaml` |
| A100 / PPU / 昇腾上的显存收益 | 无设备 |
| 评测入口 `evaluate_retrieval.py` 的推理显存 | 未测 |
| 只设 `XPU_VISIBLE_DEVICES`（不设 CUDA 列表）时 trainer 的启动 | 本机 torch_xmlir 栈在 import 阶段失败（Python 侧 `device_count()` = 8、运行时侧 2 张卡），是厂商栈的行为；launcher 只保证不崩溃、如实透传。未在本修复处理 |
| 厂商 `clip_grad_norm_`（bf16 梯度约 35 GiB 临时张量）与 AdamW step（15.25 GiB 临时张量） | 不在本修复范围，仅记录 |
