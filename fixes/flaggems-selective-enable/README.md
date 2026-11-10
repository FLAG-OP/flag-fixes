# flaggems-selective-enable

- ID：flaggems-selective-enable
- 状态：draft（未具备部署资格）
- 负责人：待定
- 上游 issue/PR：无（FlagScale 本地修复）
- 受影响对象：`flagscale/train/utils/flaggems.py::enable_flaggems`，目标包
  flagscale @ 89f342e690053e07dda845f1365089ec00464e09
- 问题与修复后行为：`use_flaggems=true` 走 `flag_gems.enable(unused=...)`
  **全量注册**。昆仑芯实测 10 个代表性算子 9 个更慢（layer_norm 0.04x、
  sum 0.03x），全量注册是净负收益；同时 `unused` 过滤对 cpp_patched_ops
  （含 gelu）不生效，排除也不可用。修复新增 **`flaggems_include`** 配置：
  设置时只注册列出的算子（`only_enable`），未设置时行为与基线逐字节一致。

配套关系（三件套，缺一不可）：

| 修复 | 层 | 作用 |
|---|---|---|
| `gelu-kunlunxin-pow` | FlagGems | gelu 家族去掉 libdevice pow（bwd 在 12.25M 元素上 10.3 ms/次 → 0.65 ms） |
| **本修复** | FlagScale | 只注册 gelu 家族，避免全量慢 kernel 进来 |
| `adamw-fused` | FlagScale | AdamW 单内核（-24 ms/step） |

三者同装，BGE-M3 重建负载整步 **2,665.9 → 2,291.2 ms（-14.0%）**，且全程走
FlagScale 真实集成路径（`enable_flaggems` → `only_enable`）。

## 使用

```yaml
# train conf
hardware:
  use_flaggems: true
  flaggems_include: [gelu, gelu_backward]   # 只注册打补丁的 gelu 家族
```

未设置 `flaggems_include` 时保持旧行为（全量 + `flaggems_exclude`），
不对既有用户产生任何变化。

## 边界

- `deployable=false`、`hardware_status=unverified`（平台整模型未跑）；
- exclude 对 cpp_patched_ops（gelu）不生效是 FlagGems 侧行为，已记录、未修；
- 多卡 FSDP2 未测（本机单卡）；注册时机（init_distributed 后、建模型前）
  与文档口径一致，未改。
