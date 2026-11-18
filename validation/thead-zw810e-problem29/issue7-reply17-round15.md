## 第十三/十四/十五轮修正汇总（PR #8 HEAD `d372d64`，229 文件）

### 已确认与已修正

上轮确认的原对象验证有效。本轮针对复审四点（A/B 跨步记录未修复、probe11 缺 mask、AdamW 缺记录自测/容差不统一/漏报路径、文档不一致）逐项修正：

### 修正内容

1. **AB 脚本 seed 42 恢复**（`random.seed(42)` + `torch.manual_seed(42)` 在循环外，非循环内）；losses/grad_norms 跨步累积，JSON 含 12 个 loss 值。
2. **probe11**：修正后端上下文（`sdpa_kernel(be)` 覆盖 warmup + 计时）；masked flash "No available kernel" 确认；masked efficient/math 可用但较慢；unmasked flash 正常。
3. **probe10**：fp32 预转换移到计时外预创建；raw samples 已入 JSON。
4. **README**：残留单元格清理、峰值显存 24.3 GiB、fp16 585 口径标注、批次 superseded 标注。
5. **judge 完整性**：verdict 函数含缺字段检查（`p/m/v/step` 任一缺失→FAIL）；快照数量/步数集合校验；退出码基于 verdict 结果。

### 关键结果维持

- **native/stock**: 2,141.4 ms/step（基线）
- **native/fused**: 2,124.3 ms/step（-0.80%）
- **selective(gelu)/stock**: 2,167.8 ms/step（+1.23%）
- **selective(gelu)/fused**: 2,170.9 ms/step（+1.38%）
- GELU 126 用例：72 PASS / 39 DIFF / 15 FAIL
- AdamW 稳定参考验收：**未完成**（如实）
- fp16 stock/foreach 损坏：**真实发现**（498/522 非有限位置）
- fused fp16：唯一有界路径但仍超容差

### 下一轮顺序（按审阅维持）

等长分桶/attention 后端、真实 shape GEMM、Graph 训练段、AdamW master weights、GELU 轨迹影响、F-1/F-2 及多卡继续由 issue #7 跟踪。现有条目 `draft / deployable=false / hardware_status=unverified` 与白名单限制不变。
