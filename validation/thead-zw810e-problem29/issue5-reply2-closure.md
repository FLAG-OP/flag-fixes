## 任务完成通报（issue #5 首轮闭环）

上一条评论的结果表与数据已全部落地，本轮工作**已完成**。状态汇总：

### 已完成（对照 issue 最小执行方案）

| 项 | 状态 | 证据 |
|---|---|---|
| 环境/设备/驱动/树 hash 记录 | ✅ | `results/env_record.json`（driver 1.5.1-1d747a / SDK 2.1.0-a5f865 / torch 2.10.0 / FlagGems `29e14822`） |
| device/dispatch/注册入口 + 命中证明 | ✅ | profiler：enable 后 gelu 族/add/max 全部 Triton 命中；selective 排除项保持 native kernel |
| native / FlagGems / 平台候选三组对照 | ✅ | 18 项微基准 × 3 独立进程（p50/p95/显存）；full 17/18 慢 1.3–6.4×，无一项更快 |
| 精度矩阵（含 0x03FF/0x007F/0x007FFFFF 次正规边界、NaN/Inf/±0、三种布局） | ✅ | 126 用例：**0 硬失败**（84 PASS + 42 DIFF，DIFF 全部为 FTZ 区 ±1–2 ulp / ±0 符号，已逐用例归档位级计数） |
| AdamW stock/fused | ✅ | fused 可用，5 步轨迹一致（fp32 ≤2.4e-6、bf16 1 ulp） |
| 固定种子小模型（gated MLP+2 层 attention） | ✅ | 三组 loss/梯度/参数 8 步后**逐位一致**（sha256 15/15） |
| selective enable 消融 | ✅ | 机制有效（未包含算子 native 性能+native kernel）；被包含 gelu 族仍 1.6–3.7×，本轮无有益算子 |
| 归档与可复现命令 | ✅ | `validation/thead-zw810e-problem29/{README.md,scripts,results,logs}`（约 288 KB，含退出码） |

### 首轮结论

**ZW810E：需优化。** FlagGems（`29e14822`）在该平台为净负收益，根因 = 每算子固定开销 ~60–80 µs + 稳态略慢；接入策略 = 不启用或 selective；gelu/cast/findmax/pointwise/transpose/attention 热点建议走厂商 kernel/融合（对应 PR3 移交项 ①②）。"FlagGems 13 算子 12 慢"口径与本轮实测方向一致。本轮为**平台基线记录**，不构成任何 fix 的硬件验收；全部条目 `draft / deployable=false / hardware_status=unverified` 维持不变。

### 过程问题与修正（透明披露）

1. 执行初期 github.com 网络短暂不可达（TCP 阻断），经 codeload/raw.githubusercontent/jsdelivr/ungh 完成仓库与 PR 内容获取，任务中段恢复——不影响设备侧数据。
2. 精度 harness 三处缺陷在首轮内修复：FlagGems 直调 API 无 autograd（改走 aten 路径并由 profiler 复证命中）、NaN 参与 max-abs 统计（改有限掩码）、ULP 计数按 dtype 升位虚高（改 dtype 位型序数映射）；`gelu_and_mul` 无 autograd 路径，梯度记 N/A。
3. 小模型步时首轮单进程显示 gems_full 更快 ~2 ms/步，3 进程复跑证伪（系噪声，实际 full 略慢 0.4–0.9 ms）——以复跑数据为准，避免单次测量误判。

### pending（不计通过，按 issue 口径登记）

- 多卡/通信整步测试：本环境单卡分配（PPUS=6,7），无真实多卡训练环境。
- problem29 真实 BGE-M3 runner / x-benchmark 整模型：本机无该容器与 runner；以 12.25M 口径微基准 + 小模型替代，差异已注明。
- `_FUSED_ADAMW_VALIDATED_VENDORS` 加入 ZW810E、平台专属 fix 条目：按仓库流程需独立条目 + owner 审核，本轮仅提供数据。
- 驱动异常/设备健康：本轮全程无 error 721 类异常，设备状态正常，无需隔离复验。

后续动作建议：平台矩阵更新已同步至 `docs/current-status.md` 与 `bundles/kunlunxin-bge-perf/README.md` 对应口径（"需优化"）；昇腾 910C（issue #4）可复用本轮 harness（scripts/ 平台无关，仅需替换目标树与 dispatch 确认步骤）。
