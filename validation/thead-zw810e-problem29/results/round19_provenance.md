# Round-19 产物溯源（review b5521fb 要求：命令/SHA256/退出码/原始 JSON）

## 1. probe11（default 入口修复后重采）
- 命令：`python3 examples/retrieval/probe11_mask_vs_equal_length.py --out results/probe11_ctx_r19.json`（cwd=FlagScale-ppu；PYTHONPATH=FlagGems-base/src；CUDA_VISIBLE_DEVICES=0）
- 入口自检：`--cpu-check` → "CPU-CHECK OK: timed defined before call; PROGRAM_ERRORS table present" exit 0
- 设备运行 exit=0，`program_error=false`；masked_flash=UNSUPPORTED（"No available kernel"，预期不支持，非程序错误）
- 结果：masked_default 0.5843ms（自动选 efficient）、masked_efficient 0.5848ms、masked_math 1.2489ms、unmasked_flash 0.1314ms、unmasked_default 0.1314ms（自动选 flash）、unmasked_efficient 0.2476ms、unmasked_math 0.9893ms
- 关键证据：default 路径在 masked 下自动回落 efficient（0.5843≈0.5848）、unmasked 下自动选 flash（0.1314=0.1314）——SDPA 默认分发行为与强制后端一致
- 脚本 SHA256：3eb7b8cf2f727ff01175e9f025cda463141a61724d2dbbcd32b0b50aa372f45b（round-19 版，timed 定义先于调用+错误分类）
- 注：`backend_enforced:true` 为代码写入的请求标志；输出/梯度对照与完整 mask 元数据为下一步设备工作（issue #7 跟踪）

## 2. probe10（运算对象修复后重采）
- 命令：`python3 scripts/probe10_gemm_ablation.py --out results/probe10_gemm_ablation_r19.json`
- 修复：fp32 参考改用与 bf16 基线相同对象 `w.float()` + 相同运算 `x@w.t()`（旧版 wf32=w.t().float() 再 .t() 实算 x@w，方阵 K=N 下静默错配）
- 新增 layout 对照：weight transposed-view 0.0575ms vs precontig 0.0566ms（~1.6%）；input noncontig 0.0613 vs contig 0.0566（~8%）
- fp32 0.2012ms vs bf16 0.0571ms；bf16_vs_fp32_relerr 0.0029（fp32_vs_bf16_max_absdiff=0.0 校验相同运算对象）
- reduced_reduction True/False：109.82/110.59 TFLOP/s（无差异，维持既有结论）
- exit=0

## 3. AdamW stable-ref（round-20 更新：自测 gate 断言 + JSON 实际再生成）
- 自测 10/10 PASS（round-20：duplicate→duplicate_steps gate、missing-key→missing_keys gate 具体断言命中；其余经生产 aggregate/verdict）
- 实验 9/9 FAIL 诚实维持（exit 1 来自实验判定，非自测失败）
- **results/adamw_stable_ref_v4.json 为 round-20 实际运行再生成**（tolerances 字段=floor 0.01*max|ref| 新口径，非旧文件改字）
- 脚本 SHA256：见 git blob 对账；运行命令 python3 scripts/adamw_stable_ref_v2.py --selftest

## 3b.（round-19 原记录，已被 3. 取代）
- 命令：`python3 scripts/adamw_stable_ref_v2.py --selftest`（同一命令跑自测+30 步实验）
- 自测 10/10 PASS：5 个 verdict 级（control/param-over-tau/NaN/Inf/step-mismatch）+ 5 个 aggregate 级（empty-list/missing-snapshot/duplicate-steps/missing-key/all-pass-control）全部经生产 `aggregate()`+`verdict()` 真实判定（无手动 dict 改写）
- 实验 verdict：9/9 FAIL（fp32/bf16/fp16 × stock/foreach/fused）——诚实结果维持，与归档 v4 结论一致
- 容差单一来源：floor=0.01*max|ref|（verdict() t_scale）；JSON tolerances 字段与 README 同式（旧"1e-6"/"0.25*参数标准差"口径作废）
- JSON：results/adamw_stable_ref_v4.json（regenerated，round-19 代码路径）
- 脚本 SHA256：03074e3d26c06c64efc49389438c1613b4027177238dddd3fd142e3de1a2fdba+（round-19 版本，见 git blob）

## 4. AB r19 批次（run_ab_reps.sh 修复后）
- runner 修复：存在即跳过 → 仅当 JSON 通过校验才跳过（校验：≥20 losses、≥3 grad norms、全 finite、segments 非空——`all([])` 缺口修复）；子进程 rc≠0 或校验失败 → FAILED 计数，不打印 REPS_DONE，exit 1
- validator 单测 7/7 PASS（valid/4+1/inf/empty-grads/no-segments/missing-key/corrupt）
- 批次目录：results/orig_ab_reps_r19/（12 JSON，round-19 runner 生成）
