# 固定来源的安装交付检查

本次初始化对 catalog 中固定 commit 进行只读检查：8 个 submodule 共 9 项 fix，5 项 source-checked，4 项 needs-adaptation。全部 deployable=false、hardware_status=unverified；未执行 NPU/XPU 测试，也未确认任何完整软件栈版本兼容性。

检查方法：校验 submodule 指针和 patch/快照 SHA256，将声明 before_source 文件复制到临时树的 targets[].path，使用 git apply 预演，再与 after_source 的字节及 after_sha256 比较。比较不忽略注释、署名或空白，且不会修改来源仓库。

| fix | 检查状态 | 固定 commit 的 patch | 声明的修复后快照 |
| --- | --- | --- | --- |
| `arith-5732` | source-checked | [arith.patch](https://github.com/FLAG-OP/arithFIX/blob/b73fbf173e1ea1e8ac7ed08316b5f96205ef02a9/arith.patch) | [src/pow.py](https://github.com/FLAG-OP/arithFIX/blob/b73fbf173e1ea1e8ac7ed08316b5f96205ef02a9/src/pow.py), [src/argmin_spacemit.py](https://github.com/FLAG-OP/arithFIX/blob/b73fbf173e1ea1e8ac7ed08316b5f96205ef02a9/src/argmin_spacemit.py) |
| `cann-pow-promotion` | needs-adaptation | [libdevice_pow_promotion.patch](https://github.com/FLAG-OP/geluFIX/blob/f07f22206a0961c3713c7ceea9f86bb2750e3dcd/libdevice_pow_promotion.patch) | 尚未声明完整目标快照 |
| `copy-kunlunxin` | source-checked | [copy.patch](https://github.com/FLAG-OP/copyFIX/blob/2b348a37fc1bc67c9226eb5371fd5431659f8e83/copy.patch) | [src/copy.py](https://github.com/FLAG-OP/copyFIX/blob/2b348a37fc1bc67c9226eb5371fd5431659f8e83/src/copy.py) |
| `gather-5746` | needs-adaptation | [gather.patch](https://github.com/FLAG-OP/gatherFIX/blob/dbb6b6d06f93fa5ee280b0ded3273086bbc2915a/gather.patch) | [src/gather.py](https://github.com/FLAG-OP/gatherFIX/blob/dbb6b6d06f93fa5ee280b0ded3273086bbc2915a/src/gather.py) |
| `gelu-5867` | source-checked | [gelu.patch](https://github.com/FLAG-OP/geluFIX/blob/f07f22206a0961c3713c7ceea9f86bb2750e3dcd/gelu.patch) | [src/geglu.py](https://github.com/FLAG-OP/geluFIX/blob/f07f22206a0961c3713c7ceea9f86bb2750e3dcd/src/geglu.py), [src/gelu_and_mul.py](https://github.com/FLAG-OP/geluFIX/blob/f07f22206a0961c3713c7ceea9f86bb2750e3dcd/src/gelu_and_mul.py), [src/gelu.py](https://github.com/FLAG-OP/geluFIX/blob/f07f22206a0961c3713c7ceea9f86bb2750e3dcd/src/gelu.py) |
| `lift-fresh-5743` | needs-adaptation | [lift_fresh.patch](https://github.com/FLAG-OP/liftfreshFIX/blob/233323cde6ce9c4cb34b284a1319aa243ff24be7/lift_fresh.patch) | [src/lift_fresh.py](https://github.com/FLAG-OP/liftfreshFIX/blob/233323cde6ce9c4cb34b284a1319aa243ff24be7/src/lift_fresh.py), [src/lift_fresh_copy.py](https://github.com/FLAG-OP/liftfreshFIX/blob/233323cde6ce9c4cb34b284a1319aa243ff24be7/src/lift_fresh_copy.py) |
| `mm-bf16-kunlunxin` | source-checked | [mm.patch](https://github.com/FLAG-OP/mmBF16FIX/blob/09967ec546277c8175fe4d1021daeb3e3b3678cd/mm.patch) | [kernel/triton_level.py](https://github.com/FLAG-OP/mmBF16FIX/blob/09967ec546277c8175fe4d1021daeb3e3b3678cd/kernel/triton_level.py) |
| `one-hot-5802` | needs-adaptation | [one_hot.patch](https://github.com/FLAG-OP/onehotFIX/blob/3cef1d720fc5ce42978be4b7fc1c500eb516278f/one_hot.patch) | [src/one_hot.py](https://github.com/FLAG-OP/onehotFIX/blob/3cef1d720fc5ce42978be4b7fc1c500eb516278f/src/one_hot.py) |
| `pointwise-5739` | source-checked | [pointwise_dynamic.patch](https://github.com/FLAG-OP/pointwiseFIX/blob/a6b538daa4b61f8da5e368e92847a45ae79ba856/pointwise_dynamic.patch) | [src/pointwise_dynamic.py](https://github.com/FLAG-OP/pointwiseFIX/blob/a6b538daa4b61f8da5e368e92847a45ae79ba856/src/pointwise_dynamic.py) |

## 需要适配的四项

- **gather-5746**：patch 应用结果相对 src/gather.py 缺少若干说明注释和署名行，after_sha256 不符。
- **lift-fresh-5743**：两份修复后快照包含 patch 中未含的署名注释，after_sha256 不符。
- **one-hot-5802**：修复后快照在多个守卫附近多出署名注释，after_sha256 不符。
- **cann-pow-promotion**：原始 libdevice_pow_promotion.patch 为 normal diff，v1 仅支持统一 diff；仍需精确目标基线、前后 hash 及可应用 patch。目标包是 Triton，不能套用 FlagGems 根路径。

前三项是来源交付物的字节不一致发现。这里不声称其算法错误，也不根据注释差异推断其硬件或语义等价性。当前 check/plan 对原始快照基线预演后会拒绝不匹配结果，不能用 source-checked 条目的结果代替这些项的检查。

## 后续处理与限制

保持原 submodule 固定内容，不直接替来源改 patch。应在来源仓库协调 patch 和快照的一致性，取得新 commit，然后一起更新 submodule 指针、catalog hashes 和证据，经 PR 再检查。

source-checked 五项仍需真实设备和目标环境验收、owner、验证入口及部署审核。本检查使用源仓库声明的原版快照，没有独立证明它们对应某一上游版本，因此 baseline_version/commit 保留 null。历史 README 测量仅为来源报告，不是本仓库复跑。

三方仓库当前未发现独立 LICENSE 文件；来源代码中的已有版权/许可声明原样保留，不据此补写或推断整仓许可证。本仓库工具的许可不覆盖三方内容，正式制品分发前需进一步确认许可。
