# 诊断报告：flaggems-selective-enable

## 1. 现象

FlagScale 配置 `use_flaggems=true` 后，`enable_flaggems` 调用
`flag_gems.enable(unused=flaggems_exclude)`——**全量注册** 200+ 个算子。
而平台实测（problem29/ATTRIBUTION.md）：昆仑芯上 FlagGems 10 个代表性
算子 9 个更慢（layer_norm 0.04×、sum 0.03×、add/max 更慢），只有打过
pow 补丁的 gelu 家族快 5.8–16.5×。全量注册 = 慢 kernel 全面替换快 kernel，
这正是平台文档"真实训练跑不完"的成因之一。

另发现 `unused` 过滤对 cpp_patched_ops（gelu 在 FlagGems 的
`aten_patch_list` 名单内）**不生效**——`flaggems_exclude=[gelu]` 也无法
把 gelu 排除掉（语义回归基线/修复一致，属 FlagGems 侧行为）。

## 2. 根因

接入层只封装了 FlagGems 的 `enable`（全量 + exclude），没有暴露
`only_enable`（include）。FlagGems 默认自动选 foreach 式全量注册，
没有"只注册这几个"的自动判断。

## 3. 方案

`enable_flaggems` 新增 `flaggems_include` 配置：

- 形式与 `flaggems_exclude` 完全一致（list / ListConfig→list / YAML 路径 /
  `"none"` 归一化）；
- 非空时走 `flag_gems.only_enable(include=...)`；
- 未设置或为 "none" 时走原 `flag_gems.enable(unused=...)`——**基线行为
  逐字节不变**；
- 日志行增加 `mode=include|full` 与生效算子列表。

## 4. 验证要点

- 注册集合对照：include=[gelu, gelu_backward] → 基线 349 个 op、修复 2 个
  （`results/repro-*.log`）；
- 实际派发：include 模式下 gelu 前反向 kernel 为
  `gelu_none_kernel_rank_1` / `gelu_backward_none_kernel_rank_1`，
  无 `gelu_highprecision_nv`（`results/dispatch-fixed-*.log`）；
- gelu pair（10,299,4096 bf16）：vendor 1.86 ms → include+补丁 1.17 ms
  （`results/bench-*.json`）。
- 整步：BGE-M3 重建负载 2,665.9 → 2,291.2 ms（-14.0%），全链路走本修复的
  集成路径（`results/step-E-final.json`）。

## 5. 影响范围

- 单文件单分支；未设置 `flaggems_include` 时除日志行外无任何行为差异；
- 与 `adamw-fused` 修复独立可组合（验证树两者并存，check 各自通过）；
- 与 `gelu-kunlunxin-pow` **必须配套**（见 WORKFLOW"与其他修复的关系"）。

## 6. 未覆盖

- 多卡 FSDP2 环境（本机单卡）；
- exclude 对 cpp_patched_ops 失效的修复（FlagGems 侧问题，需上游）；
- 平台整模型验收（hardware_status=unverified）。
