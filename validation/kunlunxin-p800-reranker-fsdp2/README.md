# 昆仑芯 P800 上 Qwen3-Reranker-8B 的 2 卡 FSDP2 记录（problem26，多卡平台行为）

日期：2026-11-26/27 · 平台：昆仑芯 P800 OAM（8 卡节点；本验证用可见的 1、2 两张卡，每卡 96 GiB）·
目标树：FlagScale `wxj630/FlagScale@89f342e6`（`feat/retrieval-flagos`；修复树 = 基线 +
`reranker-last-token-logits` patch，前后 hash 见该 fix 的 `fix.yaml`）·
环境：`results/env_record.json`（torch 2.9.0+cu129 / transformers 4.57.6 / XPU-SMI 驱动 5.0.21.47 /
XPU-RT 5.13.0 / 分配器后端 `xpu native`）· 原始数据：[fix 的 results](../../fixes/reranker-last-token-logits/results/)
（逐次 JSON、trainer 日志、探针 JSON）· 执行者：AI 代理自动执行，没有人工复核 · 复现命令见文末。

**本记录不是任何 fix 的硬件验收。** fix 的状态、单卡证据与逐命令结果见
[fix 的验证记录](../../fixes/reranker-last-token-logits/reports/validation.md)。

## 结论（多卡口径）

- **平台记录的 OOM 已自然复现**：2 卡 FSDP2 + fp32 master + batch 32 + 真实权重，数据换成每一对都截断到
  640 的长文档子集后，基线第 3 步反向在两个 rank 同时报 `Tried to allocate 11.57 GiB … 20.77 GiB is free …
  71.66 GiB is allocated by PyTorch`（平台记录 11.57 / 20.90 / 71.68）。
- **修复树走过了那一步，但没有一次干净退出**：同一数据上第 4 次尝试 step 1–3 全部完成，随后停在
  validation。本机 2 卡主干共触发 14 次，两棵树都被集合通信卡死打断。
- **平台阻断 F-P800-1 再次触发**（[首次登记](../kunlunxin-p800-real-bge-m3/README.md) §E.3）：
  `Watchdog caught collective operation timeout`，`OpType=_ALLGATHER_BASE`，终止点 SeqNum=3/22/190/379/609，
  与 payload 大小无关。这次多了两条观察：不同树、不同数据下多次停在**相同的 SeqNum**（三次 379、
  两次 190），说明卡死点和训练循环里的特定位置有关，具体是哪个集合通信没有逐个对上；小模型上的
  时序探针显示，所有 rank 前向前有一段空闲时 3/3 不卡，单边延迟、barrier、以及把这个空闲搬进
  8B trainer 都不行。

## 与 problem26 对账

| 问题里的说法 | 对账结果 |
|---|---|
| "显存碎片化无法分配连续显存" | 属实。P800 这套 torch 构建里 expandable segments 不可用（见下节）；空闲显存零散后，11.57 GiB 的请求在设备空闲 72.84 GiB 时仍失败 |
| "导致显存爆炸" | 属实，但那块显存本来就不该申请：11.57 GiB 是适配器对每个位置都算的词表 logits，每条序列只用 1 行。修复后 LM head 单次输出 18.5 MiB |
| 平台注释："lm_head 的 all-gather 要一整块 11.57 GiB" | 不成立。all-gather 的是 lm_head 权重，621,236,224 × 2 B = 1.157 GiB（trainer 日志 `gathers 1185 MiB`）；11.57 GiB 与 `64 × 640 × 151,669 × 2 B` 逐字节相等 |
| 平台注释："vendor allocator 不支持 expandable_segments" | 属实，且能定位到构建层：`libc10_cuda.so` 里 PyTorch 用于虚拟内存管理的驱动接口（`cuMemCreate`、`cuMemAddressReserve`、`cuMemMap` 等）全部是只会断言的占位实现，几个 torch 动态库里对它们没有任何外部符号引用；`CUDAAllocatorConfig.h` 在没有 `PYTORCH_C10_DRIVER_API_SUPPORTED` 时把该选项强制关掉并告警 |

## 2 卡触发清单（主干 14 次）

| # | 树 | 权重 / 数据 | batch | 走到 | 终止方式 | 用时 |
|---|---|---|---|---|---|---:|
| 1 | 基线 | 随机 / 常规 | 32 | step 1 | watchdog，SeqNum=190 | 712 s |
| 2 | 修复 | 随机 / 常规 | 32 | step 1–2 | watchdog，SeqNum=379 | 721 s |
| 3 | 修复 | 随机 / 常规 | 16 | 0 | 设备 `kl3ChannelCheckErrors failed … A kernel exception has occurred`（status 702） | 1233 s |
| 4 | 修复 | 真实 / 常规 | 32 | step 1–3 + validation | watchdog，SeqNum=609（训练循环结束后的收尾） | 737 s |
| 5 | 基线 | 真实 / 常规 | 32 | step 1–2 | watchdog，SeqNum=379 | 724 s |
| 6 | 基线 | 真实 / 长文档 | 32 | step 1–2 | **step 3 反向 OOM**（两个 rank 同时；自然复现，干净退出） | 70 s |
| 7 | 修复 | 真实 / 长文档 | 32 | step 1 | watchdog，SeqNum=190 | 718 s |
| 8 | 修复 | 真实 / 长文档 | 32 | step 1–2 | watchdog，SeqNum=379 | 730 s |
| 9 | 修复 | 真实 / 长文档 | 32 | 0（第 1 步前向） | 900 s 内 watchdog 未触发，launcher 超时终止 | 900 s |
| 10 | 修复 | 真实 / 长文档 | 32 | **step 1–3** | 停在 validation 的首个集合通信，launcher 超时终止 | 420 s |
| 11 | 基线 | FSDP2 单元测试（4 层） | — | 跑完 | exit 1，仅 lm_head 投影判据失败（预期） | 329 s |
| 12 | 修复 | FSDP2 单元测试 | — | 第 1 步前向 | launcher 超时；SeqNum=3（块 all-gather） | 900 s |
| 13 | 修复 | FSDP2 单元测试 | — | 第 1 步前向 | watchdog，SeqNum=22（norm all-gather） | 689 s |
| 14 | 修复 | FSDP2 单元测试 | — | 跑完 | **exit 0**，两 rank 全过 | 26 s |

长文档子集：`tests/build_long_parquet.py` 从 T2Ranking `triplet-15` shard 0 选出正负样本都超过 592 token
预算的 454 行（414 / 13 / 27 落在 train / validation / test）。常规数据下前几步的 batch 较短，基线不会触发
11.57 GiB 的请求，所以自然 OOM 只在长文档上出现。

## 分配器

`tests/probe_p800_allocator.py`（[原始数据](../../fixes/reranker-last-token-logits/results/allocator-probe-2026-11-27.json)）：

| 模式 | 堆状态 | 11.57 GiB 请求 |
|---|---|---|
| A 空卡 | 设备空闲 95.97 GiB | 成功 |
| B 82 × 1 GiB，隔一块释放一块并交还设备 | 设备空闲 54.97 GiB | 成功 |
| C 同 B，空洞留在 PyTorch 缓存 | 设备空闲 13.97 GiB | 成功 |
| D 221 块（64 MiB–1 GiB）释放 2/3 并交还设备 | 设备空闲 **72.84 GiB**，PyTorch 缓存空闲 0 | **OOM**；随后 1.25 GiB 成功 |
| E 同 D，`PYTORCH_ALLOC_CONF=expandable_segments:True` | 设备空闲 72.75 GiB | **同样的 OOM**，没有告警 |
| E 同 D，`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 设备空闲 72.75 GiB | **同样的 OOM**，告警 `expandable_segments not supported on this platform` |

为什么不可用（`results/env_record.json` 的 `expandable_segments_build_support`）：PyTorch 的 expandable
segments 建立在 CUDA 驱动的虚拟内存管理接口上；本机构建的 `libc10_cuda.so` 里这些接口全是
"INTERNAL ASSERT FAILED … driver_api.cpp:23" 的占位实现，`libc10_cuda.so`、`libtorch_cuda.so`、
`libtorch_cpu.so` 对 `cuMemCreate` / `cuMemAddressReserve` / `cuMemMap` 等没有任何外部符号引用；
实际生效的是厂商的 `xpu native` 分配器后端。厂商运行时底层能否提供等价能力，需要厂商回答。

## 对阻断本身的探索（小样本，供运行时层参考）

| 探针 | 做法 | 结果 |
|---|---|---|
| 步骤边界主机同步 | 2 rank FSDP2 单元测试，交错 3+3 次：plain / 每次前向前与每个 optimizer step 后 `torch.cuda.synchronize()` | plain 1/3、sync 2/3 通过——不显著 |
| 主机时序（`tests/probe_xccl_timing.py`） | 2 层小模型走真实 `apply_fsdp2`，4 步，交错 3 轮 × 4 变体 | plain **0/3**（都停在第 0 步前向）、rank 1 单边睡 3 s **0/3**（停在第 0/2/3 步）、`barrier` 1/3、**两 rank 都睡 3 s 3/3 干净退出** |
| 把 3 s 空闲搬进 trainer | 修复树、长文档、batch 32，每次前向前所有 rank 睡 3 s | step 1–2 完成后仍停在第 3 步 |

原始数据：`f-p800-1-sync-ab-2026-11-27.json`、`f-p800-1-timing-probe-2026-11-27.json`（均在 fix 的 results）。
这些试验用的 harness 开关都已移除，仓库里的脚本只跑 plain 运行。

## 其他平台观察（非本修复范围，供参考）

只设 `XPU_VISIBLE_DEVICES=1,2`、不设 `CUDA_VISIBLE_DEVICES` 时，本机 torch_xmlir 栈的 device 枚举不一致：新进程里
`torch.cuda.device_count()` 返回 8，逐卡 `torch.cuda.get_device_name(i)` 到第 3 张报
`device >= 0 && device < num_gpus INTERNAL ASSERT FAILED … device=2, num_gpus=2`——Python 侧仍按 8 张卡枚举，
运行时侧只给了 XPU 列表里的 2 张。因此这样启动的 trainer 在 import 阶段（flag_gems 注册时）就失败。
显式 `CUDA_VISIBLE_DEVICES=1,2` 或两个变量都不设时没有这个问题。记录见
[fix 的 results](../../fixes/reranker-last-token-logits/results/)（`e2e-1card-fixed-xpuonly-2026-11-27.train.log`）。

## 已知边界（不装完美）

- **样本小**：每个变体 3 次；探针结论是观察，不是根因。
- **共享节点**：卡 3、4 上始终有其他进程的负载，主机时序类探针的绝对值会受影响；卡死是否与之相关没有验证。
- **干净退出只有 3 次**（#6、#11、#14），都不是"修复后完整训练"；修复后 2 卡完整跑通要在平台环境验收。
- **自动执行**：全部由 AI 代理执行，没有人工复核；日志含本机绝对路径，未脱敏。
- 与 #10 记录的差异：那次是 BGE-M3（bf16 master / fp32 master 各试过），这次是 8B reranker + fp32 master；
  两次用的是同一台机器、同一套 torch，卡死签名一致。

## 复现命令

```bash
MANAGER=/path/to/flag-fixes
TARGET=/path/to/FlagScale            # 基线或修复后的仓库根
FIX=$MANAGER/fixes/reranker-last-token-logits
export RERANKER_MODEL=/path/to/Qwen3-Reranker-8B RERANKER_E2E_OUT=/path/to/runs
cd "$FIX"

# 长文档子集（一次）
python3 tests/build_long_parquet.py --tokenizer-model "$RERANKER_MODEL" \
    --source /path/to/triplet-15/train-00000-of-00011.parquet --out /path/to/long/train-long-00000.parquet

# 2 卡平台形态：基线在第 3 步反向 OOM；修复树能否走完取决于集合通信是否卡死
RERANKER_DATA=/path/to/long PYTHONPATH="$TARGET" python3 tests/e2e_reranker_train.py \
    --nproc 2 --parallel fsdp2 --master-dtype fp32 --batch 32 --min-steps 3 --no-save --timeout 420

# FSDP2 单元测试（2 卡）
PYTHONPATH="$TARGET" python3 tests/test_reranker_fsdp2_step.py --device cuda --nproc 2 --steps 2 --timeout 300

# 分配器探针（一张空闲卡）
python3 tests/probe_p800_allocator.py

# 主机时序探针
for v in plain skew aligned barrier; do
  SKEW=$v PYTHONPATH="$TARGET" timeout 150 python3 -m torch.distributed.run \
      --standalone --nproc_per_node=2 tests/probe_xccl_timing.py
done
```
