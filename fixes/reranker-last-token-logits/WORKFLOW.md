# 修复执行清单

每步填写 done / not-run / not-applicable；后两者解释原因并注明证据，不把未跑测试记为通过。

| 步骤 | 状态 | 内容与完成条件 | 证据/原因 |
| --- | --- | --- | --- |
| 1 登记 | done | ID `reranker-last-token-logits`（模型适配层 + P800 碎片化 OOM）；owner 保持 `null`（不虚构）；来源为 problem26 与 FlagScale `feat/retrieval-flagos` 的平台记录（`examples/retrieval/conf/train/qwen3_reranker_p800.yaml`、`docs/retrieval/24-week-summary-2026-11-13-to-20.md`）；预期行为：batch 32 的一步不再申请 11.57 GiB 连续显存，打分位置与数值不变；边界：只改 `Qwen3RerankerModel.forward` 的非 score-head 分支，score-head 分支、Megatron 路径、训练循环不动 | `fix.yaml`、`README.md`、`reports/diagnosis.md` |
| 2 基线 | done | FlagScale `feat/retrieval-flagos` @ `89f342e690053e07dda845f1365089ec00464e09`（与 adamw-fused、flaggems-selective-enable 同基线；目标文件到分支头 `555a7d6a` 未变，blob `e5abccd6`）；目标 `flagscale/models/retrieval/qwen3_reranker.py` before sha256 `50147024…`；环境 P800 OAM 96 GiB、torch 2.9.0+cu129（torch_xmlir shim）、transformers 4.57.6、XPU-SMI 5.0.21.47 / XPU-RT 5.13.0；基线树与修复树的构建见 `reports/validation.md` | `reports/validation.md` 环境表 |
| 3 复现 | done | `tests/repro_reranker_logits_alloc.py`：真实架构、batch 32（64 对 × 640），在 61.0 GiB 常驻 + 34 GiB 碎片的堆上跑一步真实 `reranker_loss`。基线 **exit 1**（`Tried to allocate 11.57 GiB … 20.63 GiB is free … 73.51 GiB is allocated by PyTorch`），修复后 **exit 0**（同一步跑通）。part 1 同时记录 `lm_head` 单次输出：基线 `(64, 640, 151669)` = 11.5714 GiB，修复 `(64, 151669)` = 0.0181 GiB | `results/repro-{base,fixed}-2026-11-26.{json,log}` |
| 4 诊断 | done | 根因：适配器让因果 LM 对每个位置做词表投影、只用 1 行，11.57 GiB 是整序列 logits（all-gather 只有 1.157 GiB）；平台无 expandable segments，碎片后拼不出大块。具体在第几步触发取决于分配器的缓存历史：本机自然复现与平台记录都是第 3 步反向，第 1、2 步的同一请求成功——这一步差异的分配器机制**没有逐事件确认**（`reports/diagnosis.md` §3.3）。方案：解码器与 LM head 分两次调用，只投影打分行；影响层：模型适配层一处分支，不改数值语义 | `reports/diagnosis.md`、`results/bench-*-2026-11-26.json`、`results/allocator-probe-*.json` |
| 5 实现 | done | 单 patch 改 1 个文件 +15/−6：`patches/reranker-last-token-logits.patch`（sha256 `ee7ded27…`）；before/after hash 见 `fix.yaml`；`flag-fixes validate` exit 0、`check`/`plan` exit 0（基线 hash 命中、`git apply --check` 干净、patch 结果与 `src/` 快照一致） | `fix.yaml`、`reports/validation.md` 工具链校验 |
| 6 验证 | done | 复现：构造碎片堆 exit 1 → 0；平台形态（2 卡 FSDP2、fp32 master、batch 32、真实权重、长文档子集）自然复现基线失败——第 3 步反向 OOM，`Tried to allocate 11.57 GiB … 20.77 GiB is free … 71.66 GiB is allocated by PyTorch`，与平台记录逐项吻合。数值回归：真实架构、随机权重，bf16 权重（autocast 开/关）5 例分数与梯度对基线公式 max\|diff\| = 0.0，fp32 权重 2 例分数 2.4e-7、梯度相对差 2.0e-4（阈值 1e-4 / 1e-3），梯度比较先验结构（两侧各 9 个、逐张量有限）再算误差，另有 7 例比较器自检（14 例，两棵树 exit 0）；dispatch：CPU 2 层随机 Qwen3 25 例（基线 6 例失败 exit 1 → 修复 exit 0）、真实 `apply_fsdp2` CPU/gloo（基线 exit 1 → 修复 exit 0）与 P800 ×2 XCCL（基线 exit 1 → 修复 exit 0，另两次被 watchdog 阻断；审阅修订后经新 launcher 重跑一次 exit 0）；bench：batch 32 前向/反向峰值 40.80/54.99 → 29.94/40.61 GiB；e2e：真实 trainer + 真实 checkpoint 单卡 bf16 batch 16，两棵树 5 步 loss、validation、test loss 在日志记录精度（6 位小数）下一致，reserved 峰值 92.76 → 79.64 GiB。launcher（审阅修订后）：自测 17 项 exit 0——超时清理只作用于本作业、可见设备三种情况；真实 trainer 在默认可见与显式选卡下都跑完 5 步，只设 XPU 列表时 launcher 能启动 trainer，但 trainer 在厂商栈 import 阶段失败。**修复树的 2 卡 FSDP2 端到端未完整跑通**（XCCL，F-P800-1）；长文档上可比的 loss 只有前两步，第 3 步修复版越过基线 OOM 点、基线没有可比记录 | `results/*-2026-11-2{6,7}.*`；阻断记录 `results/e2e-fsdp2-2card-*` |
| 7 接入 | not-run | 本地部分完成：schema/hash 校验、`check`/`plan` 预演、`verify`（修复树）；`apply` 按 `deployable=false` 拒绝；`status` 干净。**未完成**：平台 2 卡整模型验收（修复后完整训练未跑通，基线失败已复现）；owner 审核未做 | 原因：本机 XCCL 集合通信间歇性卡死（F-P800-1），修复树 7 次 2 卡 trainer 尝试里 2 次走完 3 个 optimizer step（1 次含 validation），没有一次干净退出；`hardware_status` 保持 `unverified`、`deployable=false` |
| 8 上游 | not-run | 未向 `wxj630/FlagScale` 提 PR，没有合入 commit，也没有包含修复的发行版本。代码在 `feat/retrieval-flagos` 分支，本地修复先行；是否上游贡献、连同平台注释里 all-gather 归因的更正一起提，待仓库所有者决定 | 原因：本地修复尚未经平台 2 卡整模型验收（见第 7 行）；`upstream_issue: null` |

## 机制对照（步骤 4 证据）

同一张卡、同一权重（15.25 GiB）、长度 640、bf16 autocast + 梯度检查点，无优化器状态
（`results/bench-*-2026-11-26.json`）：

| batch（对数） | 树 | LM head 单次输出 | 前向峰值 allocated | 反向峰值 allocated |
| --- | --- | ---: | ---: | ---: |
| 32（64） | 基线 | 11.5714 GiB | 40.80 GiB | 54.99 GiB |
| 32（64） | 修复 | 0.0181 GiB | 29.94 GiB | 40.61 GiB |
| 16（32） | 基线 | 5.7857 GiB | 28.60 GiB | 36.86 GiB |
| 16（32） | 修复 | 0.0090 GiB | 22.67 GiB | 34.94 GiB |

性能/显存修复必须给同口径前后数据和数值精度；dispatch/e2e 按层级选择；设备不足时可提交草稿。
