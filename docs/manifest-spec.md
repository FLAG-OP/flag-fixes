# 清单约定（schema_version: 1）

权威格式是 [fix.schema.json](../catalog/schemas/fix.schema.json)，使用 JSON Schema 2020-12，各级对象拒绝未定义字段。外部条目在 `catalog/fixes/*.yaml`，本地条目在 `fixes/*/fix.yaml`，模板不参与发现。

## 字段

| 字段 | 类型与约束 |
| --- | --- |
| schema_version | 必填，固定 1 |
| template_version | 可选，整数 ≥1 |
| id | 必填，小写字母开头、英文数字和分段连字符 |
| title、owner | title 非空字符串；owner 为字符串或 null |
| status | draft / active / deprecated / retired |
| deployable | 布尔值，初始 false |
| source | 必填 kind、path、commit、repository；kind 为 submodule/local，commit 为 40 位小写 SHA 或 null，repository 为字符串或 null |
| target | 必填 package（非空）、baseline_version、baseline_commit（均为字符串或 null） |
| patches | 数组，每项必填 path（相对 source.path）及 64 位 sha256 |
| targets | 数组，每项必填 path、before_sha256、after_sha256；可选 before_source、after_source（相对 source.path 的快照路径） |
| backends、requires、conflicts | 去重的字符串数组 |
| validation | 必填 hardware_status、evidence、commands |
| notes | 可选字符串数组 |
| upstream_issue | 可选字符串或 null |
| installation_status | 可选 source-checked / needs-adaptation / draft |

submodule 固定来源完整 commit，local 的 source.commit=null，随本仓库发布 commit 固定来源，无自引用。target.path 不存在；每个文件的路径放在 targets[].path。

首版 --target 为显式源码根，FlagGems 的模块路径为 `src/flag_gems/...`，不是 Python 包目录本身。Triton 层 cann-pow 条目尚未适配，不能按 FlagGems 根使用。文件、patch、快照路径必须位于各自根下，hash 均按原始字节 SHA256 计算。

## 验证命令

validation.hardware_status 为 unverified / passed / failed，evidence 为证据字符串数组，commands 是对象数组：

```yaml
validation:
  hardware_status: unverified
  evidence: []
  commands:
    - name: direct-regression
      script: tests/test_example.py
      level: regression
      args: []
      env:
        EXAMPLE_TARGET: '{target}/src/flag_gems/ops/example.py'
      timeout_seconds: 600
```

name、script、level 必填；level 只允许 repro / regression / dispatch / e2e / benchmark；args 为字符串数组，env 为字符串值映射，timeout_seconds 为 1–86400 整数。

script 相对 source.path，命令工作目录为 source.path。CLI 用 --python 指定解释器，参数列表不经 shell 执行；仅 env 中替换 `{target}`，args 不做该替换。PYTHONPATH 前置目标的 src/。默认超时 600 秒，CLI --timeout 可覆盖。失败返回非零值；缺设备或跳过不构成通过。

verify 先检查目标为声明的 after_sha256，只输出 JSON 结果，不更新安装状态的 verification 字段，也不提升 catalog hardware_status。需保存结果作为后续审核材料。

## 部署资格与来源状态

schema 对 deployable=true 额外要求 owner 非空、status=active、至少一项 patch 和 target、hardware_status=passed、至少一条 evidence 和 command。来源 pin/hash、路径和可应用性还要由运行时与目录检查执行；schema 合法不代表可安装。

当前所有导入项均不可部署。source-checked 仅表示指定来源的 patch 应用结果与声明快照字节一致；needs-adaptation 表示当前不能满足该安装交付约定。二者均不是硬件测试结论。具体结果见 [源码接入审计](source-audit.md)。
