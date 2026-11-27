# 验证记录：reranker-last-token-logits

设备数据于 2026-11-26/27 在本机采集；命令、退出码与原始 JSON/日志均在 `../results/`。
平台形态（2 卡 FSDP2）的**基线失败已自然复现**；修复树在同一配置下被本机的 XCCL
集合通信卡死（F-P800-1）反复打断，完整跑通的步数有限，详见"2 卡 FSDP2"一节。
`hardware_status` 保持 `unverified`。

## 环境

| 项 | 值 |
|---|---|
| 机器 | KunlunXin P800 OAM ×8 节点，本验证用可见的 2 张卡（每卡 96 GiB），XPU-SMI 5.0.21.47 / XPU-RT 5.13.0 |
| torch | 2.9.0+cu129（torch_xmlir shim：XMLIR dev+2026060515、XCCL 3.1.0.2；`torch.cuda.get_device_name` = "GPU"） |
| transformers | 4.57.6 |
| FlagScale 基线 | `wxj630/FlagScale` `feat/retrieval-flagos` @ `89f342e690053e07dda845f1365089ec00464e09`（tarball 快照；目标文件 blob `e5abccd6` 与 GitHub API 一致，到分支头 `555a7d6a` 未变） |
| 目标树 | 基线 = 该 commit 的干净快照；修复 = 基线 + `patches/reranker-last-token-logits.patch`（`check`/`plan` 校验 after hash） |
| 模型（e2e） | `Qwen/Qwen3-Reranker-8B` revision `77d193c791ed757ca307ee72715aa132723da912`；5 个 safetensors 分片 sha256 与 hub LFS oid 逐一相同（`model-0000{1..5}`：`22cdfea4…`、`d2163b74…`、`a5038caa…`、`247f8553…`、`8ba41b93…`） |
| 数据（e2e） | `sentence-transformers/t2ranking` `triplet-15/train-00000-of-00011.parquet`（30,920 行，sha256 `4d92878e…`）；长文档子集由 `tests/build_long_parquet.py` 从中确定性选出 454 行（sha256 `0d3a0d41…`） |
| 设备命令环境 | `XMLIR_BMM_DISPATCH_VALUE=0`（平台 P800 启动脚本同值）；其余脚本自备随机权重 |

hash（fix.yaml 记录同值）：

```
before  501470249c00c934268bd3a02bb29cad0884dd4a9336635c1d2f85a70781f7fc  flagscale/models/retrieval/qwen3_reranker.py
after   dd34c0dd0459f0469280968496ed7a42d86e46fb3feac6ada822d221ccbfcd89  flagscale/models/retrieval/qwen3_reranker.py
patch   ee7ded27c09fd2ae7b15f499ce1f14a79891ecf20b428f30fa812848197537d4  patches/reranker-last-token-logits.patch
```

omegaconf / hydra 不在共享环境里，本机以 `pip install --target` 装到独立目录并用解释器包装
喂给 `verify`；未修改共享 conda 环境。

## 命令与结果

统一方式：在 `fixes/reranker-last-token-logits/` 下，
`PYTHONPATH=<树> python3 tests/<script> [args] --json results/<name>-<tree>-<date>.json`。

| 脚本 | 设备 | 基线树 | 修复树 | 判定 |
|---|---|---|---|---|
| `test_reranker_lmhead_dispatch.py` | CPU | exit **1**（6/25 例失败：lm_head 输入 `[rows, seq, hidden]`） | exit **0**（25/25） | 符合预期 |
| `test_reranker_fsdp2_step.py --device cpu --nproc 1` | CPU（gloo） | exit **1**（lm_head 输出 `[16, 258, 151669]`，2.5 GB/次） | exit **0**（`[16, 151669]`，9.7 MB） | 符合预期 |
| `test_reranker_fsdp2_step.py --device cuda --nproc 2` | P800 ×2（XCCL） | exit **1**（328.6 s 跑完，仅 lm_head 投影判据失败） | exit **0**（26.3 s，两 rank 全过）；此前两次尝试被 XCCL watchdog 阻断 | 符合预期；2 卡不稳定 |
| `repro_reranker_logits_alloc.py` | P800 单卡 | exit **1**：`Tried to allocate 11.57 GiB … 20.63 GiB is free … 73.51 GiB is allocated by PyTorch` | exit **0**：同一步跑通，峰值 +25.36 GiB | 符合预期 |
| `test_reranker_scores_equivalence.py` | P800 单卡 | exit 0（平凡：基线即参考） | exit **0**（5 例 max\|diff\| = 0.0） | 数值不回归 |
| `bench_reranker_step.py --iters 5` | P800 单卡 | exit 0（记录） | exit 0（记录） | 见 `../benchmarks/README.md` |
| `e2e_reranker_train.py --nproc 1 --parallel ddp --master-dtype bf16 --batch 16 --min-steps 5 --max-grad-norm 0` | P800 单卡 | exit 0（5 步） | exit 0（5 步） | loss 逐位相同；reserved 峰值 92.76 → 79.64 GiB |

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

| step | 基线 allocated / reserved（GiB） | 修复 allocated / reserved（GiB） | window_loss（两棵树相同） |
|---|---|---|---|
| 1 | 61.019 / 77.285 | 61.024 / 77.322 | 0.627536 |
| 2 | 61.012 / 84.217 | 61.012 / 77.326 | 0.466457 |
| 3 | 61.012 / 89.334 | 61.013 / 79.641 | 0.370828 |
| 4 | 61.014 / 92.762 | 61.015 / 79.641 | 0.639762 |
| 5 | 61.021 / 87.084 | 61.019 / 79.641 | 0.592430 |

train_loss 0.539403、validation_loss 0.461096、test_loss 0.329392 两棵树逐位相同。
每步 allocated 峰值 76.27 GiB 两棵树相同——它来自厂商 AdamW 的 15.25 GiB 临时张量
（61.02 + 15.25），高于 batch 16 的反向峰值，所以差异体现在 reserved 上。

## 2 卡 FSDP2（平台形态）

配置：`--nproc 2 --parallel fsdp2 --master-dtype fp32`，梯度检查点、`max_length` 640、
`num_negatives` 1，与 `run-p800-fsdp2-2card.sh` + `qwen3_reranker.yaml` 一致；
`RERANKER_E2E_NO_SAVE=1` 跳过 32 GB 的 checkpoint 写出。

### 自然复现（长文档子集、真实权重、batch 32）

子集里每对的正、负样本 prompt 都 ≥ 592 token，全部被截断到 640——即平台报错时的 padded 形状。

| 树 | 结果 | 文件 |
|---|---|---|
| 基线 | step 1（allocated 61.01 / reserved 92.06 GiB）、step 2（step 峰值 allocated 87.02 / reserved 95.13）完成；**step 3 反向在两个 rank 同时 OOM**：`Tried to allocate 11.57 GiB … of which 20.77 GiB is free. Of the allocated memory 71.66 GiB is allocated by PyTorch, and 2.98 GiB is reserved by PyTorch but unallocated`；traceback 落在 `train_epoch → loss.backward() → hydrax LinearFunction.backward → linear_bwd_dgrad → fc_fusion → TensorViewState.value`，即 lm_head 反向对整序列 logits 梯度的物化 | `e2e-fsdp2-2card-base-b32-longdoc-2026-11-26.{json,log,train.log}` |
| 修复 第 1 次尝试 | step 1 完成（61.01 / 92.20 GiB，loss 0.392380 与基线相同）后停在第 2 步，XCCL watchdog | `e2e-fsdp2-2card-fixed-b32-longdoc-attempt1-blocked-2026-11-26.*` |
| 修复 第 2 次尝试 | step 1（allocated 61.01 / reserved 92.20 GiB）、step 2（step 峰值 allocated 84.41 / reserved 95.02）完成，loss 与基线相同（0.392380、0.377111 / 0.361841）；停在第 3 步——基线 OOM 的那一步——XCCL watchdog | `e2e-fsdp2-2card-fixed-b32-longdoc-attempt2-blocked-2026-11-27.*` |
| 修复 第 3 次尝试 | 模型加载与 FSDP2 分片完成后停在第 1 步的前向，900 s 内 watchdog 也未触发，由 launcher 超时终止（0 步） | `e2e-fsdp2-2card-fixed-b32-longdoc-attempt3-blocked-2026-11-27.*` |
| 修复 第 4 次尝试 | **step 1–3 全部完成**（step 3 峰值 allocated 84.41 / reserved 94.32 GiB，loss 0.454597 / 0.609568）——即基线 OOM 的那一步通过；随后停在 validation 的首个集合通信，420 s 后由 launcher 超时终止（`training_steps_complete=true`，`passed=false`） | `e2e-fsdp2-2card-fixed-b32-longdoc-plain4-2026-11-27.*` |

平台记录（11.57 GiB / 20.90 GiB free / 71.68 GiB allocated）与本机基线（11.57 / 20.77 / 71.66）逐项吻合。

### 常规数据

| 权重 | 树 | batch | 完成 | 终止位置 | 文件 |
|---|---|---|---|---|---|
| 随机 | 基线 | 32 | step 1（61.02 / 92.24 GiB） | 第 2 步，`Watchdog caught collective operation timeout: WorkXCCL(SeqNum=190, OpType=_ALLGATHER_BASE, NumelIn=2048, NumelOut=4096)` | `e2e-fsdp2-2card-base-randominit-2026-11-26.*` |
| 随机 | 修复 | 32 | step 1–2（step 2 峰值 allocated 80.28 = 61.02 + 19.26，reserved 95.19） | 第 3 步，同样 watchdog，`SeqNum=379` | `e2e-fsdp2-2card-fixed-randominit-2026-11-26.*` |
| 随机 | 修复 | 16 | 0 | 第 1 步内 `kl3ChannelCheckErrors failed … A kernel exception has occurred`（status 702），20 分钟后 `xpu_free error code=-714` | `e2e-fsdp2-2card-fixed-b16-randominit-2026-11-26.*` |
| 真实 | 修复 | 32 | step 1–3 + validation（train_loss 0.450768，validation_loss 0.512823；step 3 峰值 allocated 83.20） | 训练循环结束后的收尾集合通信，watchdog `SeqNum=609` | `e2e-fsdp2-2card-fixed-2026-11-26.*` |
| 真实 | 基线 | 32 | step 1–2（loss 与修复树相同） | 第 3 步，watchdog | `e2e-fsdp2-2card-base-2026-11-26.*` |

常规数据下 batch 长度随窗口变化，前几步较短，基线没有触发 11.57 GiB 的请求；自然 OOM 要在
长 batch 上才出现（上一节）。

### FSDP2 测试（`test_reranker_fsdp2_step.py --device cuda --nproc 2`）

4 层、8 行、2 步，真实 `apply_fsdp2` + bf16 autocast。基线 1 次尝试跑完（328.6 s，仅 lm_head
投影判据失败）；修复树 3 次尝试——前两次分别停在 `SeqNum=3`（块 all-gather）与 `SeqNum=22`
（norm all-gather）的 XCCL watchdog，第三次 26.3 s 两 rank 全过：lm_head 是 `Shard(0)` DTensor
且带 FSDP pre/post hook，输入 `[16, 4096]`、输出 `[16, 151669]`（基线 `[16, 258, 151669]`，
1.25 GB/次），分数对整序列参考 max\|Δ\| = 0.0，两 rank 的 loss 与基线树逐位相同
（`[0.805798, 17.0]`）。梯度：拥有 "yes"/"no" 两行的 rank 0 上 lm_head 梯度范数 20.37，
rank 1 上为 0——lm_head 沿词表分片，只有这两行有梯度，基线同样如此。

### 阻断统计

本机 2 卡的主干触发 **14 次**（trainer 10 次：基线 3 次、修复树 7 次；FSDP2 测试 4 次：基线 1 次、
修复树 3 次）。按各自目标跑完的 3 次——基线自然 OOM（长文档）、基线 FSDP2 测试、修复树 FSDP2
测试；修复树另有 2 次走完 3 个 optimizer step（常规数据那次含 validation，长文档 plain4 那次
正好走过基线 OOM 的第 3 步）后卡住；其余 9 次被 XCCL 集合通信超时或设备异常提前终止。
两棵树都中招。终止点与 payload 大小无关（SeqNum=3/22/190/379/609，`_ALLGATHER_BASE`），
与已登记的平台阻断 **F-P800-1**（`validation/kunlunxin-p800-real-bge-m3/README.md` §E.3）同一签名。
因此**"修复后 batch 32 在 2 卡 FSDP2 上完整跑通"没有拿到干净结果**；已得到的是：同一数据上
基线第 3 步 OOM，修复树走过了这一步，每一步的 allocated/reserved 与 loss 都有记录且与基线相同。

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

## 工具链校验

- `flag-fixes --root . validate`：valid（15 fixes = 9 external + 6 local）
- `flag-fixes --root . check reranker-last-token-logits --target <基线树>` / `plan`：exit 0
  （基线 hash 命中、`git apply --check` 干净、patch 结果与 `src/` 快照一致）
- `flag-fixes apply reranker-last-token-logits --target <树>`：exit 2 拒绝（`deployable=false`，符合预期）
- `flag-fixes status --target <基线树>`：clean；对已打补丁的树 `check` 报 `Baseline checksum differs`（exit 2）
- `flag-fixes verify reranker-last-token-logits --target <修复树>`：**passed=true，6/6 rc=0**，在干净提交
  `57e2c756`（`source_dirty=false`）上运行，2026-11-27；原始输出 `../results/verify-2026-11-27.json`（stderr 为空）
  - reranker-lmhead-dispatch（dispatch）：rc=0
  - reranker-fsdp2-step-cpu（dispatch）：rc=0
  - reranker-logits-alloc-repro（repro）：rc=0
  - reranker-scores-equivalence（regression）：rc=0
  - reranker-step-benchmark（benchmark）：rc=0
  - reranker-trainer-single-card（e2e）：rc=0
- 仓库测试 `python -m pytest -q`：26 passed, 3 subtests passed；`python -m compileall -q fixes`：exit 0

verify 只输出 JSON 测试结果，不提升 `hardware_status`；上面的设备结论仍以平台验收为准。

## not-run（未验证清单，不记为通过）

| 项 | 原因 |
|---|---|
| 平台 2 卡 FSDP2 整模型验收（修复前后 batch 16 / 32 / accum 2 的完整训练与吞吐） | 本机 XCCL 集合通信间歇性卡死（F-P800-1），单次尝试最多走完 3 个 optimizer step + validation |
| P800 配置回到 batch 32 或 accum=2 的余量评估 | 同上；需平台环境实测后再改 `qwen3_reranker_p800.yaml` |
| A100 / PPU / 昇腾上的显存收益 | 无设备 |
| 评测入口 `evaluate_retrieval.py` 的推理显存 | 未测 |
| 厂商 `clip_grad_norm_`（bf16 梯度约 35 GiB 临时张量）与 AdamW step（15.25 GiB 临时张量） | 不在本修复范围，仅记录 |
