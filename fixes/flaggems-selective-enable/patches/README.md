# 最小 patch

`flaggems-selective-enable.patch` 在 FlagScale `89f342e690053e07dda845f1365089ec00464e09`
干净基线上生成（`git -C /tmp/opencode/FlagScale-adamw-fixed diff -- flagscale/train/utils/flaggems.py`）。

- 1 文件，26+/3-：新增 `flaggems_include` 配置与非空时的 `only_enable` 路径；
  日志行加 `mode=`；未设置 include 时行为与基线一致。
- SHA256：`bba9c4f3859d6c952659060068520e487eb7a4cfbbd179558dca6817ad262c67`。
- 与 `adamw-fused` 的 patch 不同文件，可同时应用于同一棵树。
