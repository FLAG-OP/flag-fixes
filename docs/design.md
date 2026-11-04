# FLAG-OP/flag-fixes 仓库设计草案

状态：设计与演进计划。当前以本仓库 README 和清单 schema 为实现边界；profile/bundle 与离线发布流程仍属后续计划。

## 1. 定位与边界

新仓库负责修复发现、兼容性检查、安装与回滚、组合验证、版本化分发。
现有 8 个 xxFIX 仓库通过 third_party/<repo>/ Git submodule 固定接入，其源码仍在来源仓库维护；新的修复直接归本仓库 fixes/<fix-id>/ 管理。
FlagOS-OP 继续负责算子开发模板、接入方法和验证框架，链接新仓库的安装入口。
flag-fixes 维护专门的缺陷修复模板 templates/fix/，规范从问题复现到发布接入的步骤；开发者通过 new 命令在本仓库 fixes/<fix-id>/ 创建修复。

新修复的源码、patch 和测试在 fixes/ 维护；外部来源不另行复制一份手工维护。发布制品可以包含固定 commit 的 patch、测试、源码快照及许可证；这些内容由构建流程生成，支持离线使用。
源码与 patch 同时存在时，源仓库须校验 patch 应用于声明基线后的内容与修复源码一致。

## 2. 建议目录

```text
flag-fixes/
├── README.md                       # 使用入口和支持范围
├── CONTRIBUTING.md                 # 新修复接入与变更流程
├── LICENSE                         # 管理工具许可证；制品另保留源许可证
├── pyproject.toml                   # 安装 CLI 工具，不自动修改目标环境
├── templates/
│   └── fix/                        # 修复源仓库样板，见第 3.1 节
├── fixes/<fix-id>/                 # 本仓库直接维护的新修复
├── third_party/<repo>/             # 已有外部仓库的固定 submodule
├── catalog/
│   ├── fixes/<fix-id>.yaml          # 外部来源的接入清单；本地 fix.yaml 直接发现
│   └── schemas/                    # manifest/profile/bundle 的格式约束
├── profiles/<profile-id>.yaml       # 精确的软硬件环境描述
├── bundles/<bundle-id>.yaml         # 锁定条目、来源、应用顺序与环境
├── src/flag_fixes/
│   ├── cli.py                      # 命令与输出
│   ├── catalog.py                  # 读取、查询、校验清单
│   ├── environment.py              # 定位目标解释器、包、源码根目录和版本
│   ├── planner.py                  # 基线、依赖、冲突及应用计划（初期可合并实现）
│   ├── artifacts.py                # 获取固定制品、缓存与校验
│   ├── patcher.py                  # 暂存、应用与失败恢复
│   ├── state.py                    # 环境锁、安装日志、备份与回滚
│   └── verification.py             # 运行选定验证并收集结果
├── tests/
│   ├── unit/                       # 清单、规划、状态等纯逻辑
│   └── integration/                # 临时目标树中的安装/恢复测试
├── validation/<bundle-id>/          # 结构化结果与证据链接
├── tools/                          # 清单检查、制品构建等维护命令
├── docs/
│   ├── architecture.md
│   ├── manifest-spec.md
│   ├── installation.md
│   ├── adding-a-fix.md
│   └── release-process.md
└── .github/
    ├── workflows/
    ├── ISSUE_TEMPLATE/
    └── CODEOWNERS
```

仓库只保存小型验证摘要；原始大日志和离线包作为 release/CI 制品分发。
实际环境的绝对路径、备份、缓存、安装日志写到本机环境专属状态目录，不提交到仓库。

## 3. 内容与责任分工

| 内容 | 权威维护位置 | 负责人 |
| --- | --- | --- |
| 修复源码、patch、复现和单项测试 | 新修复 fixes/；历史修复 third_party 来源仓库 | fix 负责人 |
| 根因分析和单项性能/精度证据 | 新修复 fixes/；历史修复 third_party 来源仓库 | fix 负责人 |
| fix 模板、步骤及交付要求 | flag-fixes/templates/fix | 目录维护者维护，fix/平台负责人共同评审 |
| 修复索引、兼容性、来源 commit/hash | flag-fixes/catalog | fix 负责人提供，目录维护者审核 |
| CLI、安装事务、状态与回滚 | flag-fixes/src | 工具负责人 |
| 软硬件 profile 和硬件验收 | flag-fixes/profiles、validation | 平台负责人 |
| 依赖、互斥、应用顺序和组合回归 | flag-fixes/bundles | 平台负责人和目录维护者 |
| release、离线制品、变更日志 | flag-fixes | 发布负责人 |
| 通用开发模板和接入方法 | FlagOS-OP | 框架维护者 |

小团队允许兼任角色，以 CODEOWNERS 和 PR 模板明确变更责任。
已有重复性能文档逐步改为链接单一维护位置；安装说明统一指向 flag-fixes。

### 3.1 fix 模板：目录与使用方式

templates/fix/ 提供以下标准骨架，适用于算子、共享工具层和编译器层修复：

```text
templates/fix/
├── README.md                       # 问题摘要、影响范围、状态、统一安装入口
├── WORKFLOW.md                     # 按步骤填写的执行清单
├── fix.yaml                        # 源描述：ID、owner、目标、基线、验证入口
├── patches/
│   └── README.md                   # 最小 patch 的生成、命名及基线要求
├── src/
│   └── README.md                   # 可选完整修复源码，保留目标包相对目录
├── tests/
│   └── README.md                   # repro/regression/dispatch/e2e 验证约定
├── benchmarks/
│   └── README.md                   # 修复前后性能和精度比较约定
├── reports/
│   ├── diagnosis.md               # 原始问题、根因、方案、影响及已知限制
│   └── validation.md              # 环境、基线/修复 commit、命令、结果、证据
└── results/
    └── README.md                   # 小型结构化结果格式与大制品存放约定
```

模板中的目录说明随实际实现逐步补全，不放运行后直接 PASS 的测试占位脚本。
fix.yaml 带 template_version/schema_version。未知字段保持 null，初始 status=draft、deployable=false。新修复的 source.kind=local、path=fixes/<id>、commit=null，通过管理仓库发布 commit 固定来源，避免自引用。扫描器直接发现本地 fix.yaml，不再复制到 catalog。
外部条目放在 catalog/fixes/，source.kind=submodule，固定来源完整 commit、patch SHA256 和目标 hash；更新需连同 submodule 指针经 PR 评审。部署资格在完成环境和验证审核后由维护者调整，不能仅凭文件齐全改为 true。
一个源仓库可以对应多个安装单元。旧 xxFIX 仓库按映射接入，保持目录结构；后续新增修复均放 fixes/。紧密关联、必须一起生效的多文件修复使用同一个条目。
src/ 为可选项，patch 为首版部署交付物；原始基线通过上游 commit 追溯，不再要求每个文件附带 .orig 备份。
保留目标源代码许可证及版权声明，生成制品时同时带入。安装、备份和回滚由统一工具负责，模板不提供一套独立安装器。

### 3.2 fix 模板：标准执行步骤

| 步骤 | 必须记录或提交的内容 | 完成条件 |
| --- | --- | --- |
| 1. 登记问题 | fix ID、负责人、issue/现象、受影响算子与软件层 | 确定修复边界和预期行为 |
| 2. 固定基线 | 上游 commit、包版本、设备及运行时、目标文件 hash | 能重建修复前环境 |
| 3. 最小复现 | 复现脚本、运行命令、修复前日志 | 观察到目标缺陷；若不能复现则明确原因 |
| 4. 根因与方案 | diagnosis.md、修改范围、依赖/替代关系 | 说明代码修改如何对应缺陷 |
| 5. 实现修复 | 最小 patch、可选源码、来源许可 | patch 可应用且与保留源码一致 |
| 6. 验证修复 | 原复现、边界/回归、实际 dispatch；必要时 e2e、性能/精度比较 | 记录实际结果和限制，不把直调测试当作集成验收 |
| 7. 接入分发 | 完整源描述、固定源 commit、目录 PR、安装/回滚验收 | 由管理仓库完成清单检查和适用环境验证 |
| 8. 跟踪上游 | 上游 PR、合入 commit、包含修复的版本 | 据实更新生命周期与适用范围 |

WORKFLOW.md 每一步支持 done/not-run/not-applicable 状态，后两者必须解释原因。未跑测试不记为通过；性能修复必须记录同口径修复前后数据及数值精度。
验证入口须声明修复前后各自的预期结果和退出码约定，验收脚本失败时返回非零值；因缺设备而跳过必须单独记录。dispatch/e2e 按修改层级选择：编译器层修复可采用编译复现和受影响调用方回归，不强制编造独立算子 dispatch 测试。
复现失败或硬件缺失的修复可保存为草稿，发布资格仍按对应 profile/bundle 的验证规则判断。
validation.md 与结构化结果记录测试命令、环境、基线/修复 commit、日期、判定标准、结果及证据路径；错误修复可以以异常消失为前后对照，不强求修复前能完成 benchmark。
模板仅规范单项修复的交付，bundle 的跨修复组合验证由管理仓库承担。

## 4. 元数据模型

区分三类对象：fix 描述一个可部署修复；profile 描述目标环境；bundle 锁定在该环境下验证过的一组修复。
一个源仓库可以发布多个 fix。同一个 fix 支持多个经过验证的基线时，为每个基线提供独立适配记录。

可部署 fix 的必要字段如下；目录草稿只要求 ID、来源、负责人和状态，缺失其他字段时必须标记 deployable=false，并由安装器拒绝部署：

- 稳定 ID、标题、负责人、来源仓库、上游 issue/PR。
- 外部来源完整 commit、本地来源的管理仓库发布 commit、patch 路径及 SHA256；保留来源和许可证信息。
- 目标包名和相对路径；上游基线 commit/版本、修改前和修改后文件 hash。
- 已验证环境的 profile 引用；测试日期、测试来源 commit 和结果链接。
- requires、conflicts、supersedes 等组合约束；相似根因不自动构成依赖。
- 验证入口、工作目录、所需设备、超时和验收条件。
- 生命周期 active/deprecated/retired；上游合入 commit 和包含修复的发行版本（已核实后填写）。

版本号只辅助说明，安装以基线和文件内容核验为准。未知兼容性标记 unknown，不从单一环境推断整个版本范围。
生命周期与验证状态分开记录：例如 active 条目可以仍处于 hardware-unverified。
bundle 固定 fix 内容摘要、源 commit、profile、应用顺序、组合后的文件 hash 以及组合测试证据。
工具版本与 bundle 版本独立，清单含 schema_version，bundle 记录最低兼容工具版本。

## 5. 现有仓库接入

| 来源 | 初始目录建模 |
| --- | --- |
| gatherFIX | gather 非连续索引修复 |
| liftfreshFIX | lift_fresh 家族修复，保留多文件原子应用 |
| onehotFIX | one_hot 分发及边界修复 |
| pointwiseFIX | 共享 pointwise 框架修复，验证影响面覆盖多个算子 |
| arithFIX | Ascend pow 与 Spacemit argmin 分开建条目，提取独立 patch 由源仓库维护 |
| geluFIX | FlagGems gelu 家族与 Triton libdevice 修复分开建条目 |
| copyFIX | KunlunXIN copy 修复 |
| mmBF16FIX | KunlunXIN mm 修复，记录 fast-mode 环境设置及精度条件 |

这些是拟建条目；不意味着所有条目已经具备发布所需基线和硬件证据。
gelu 的两层修复不默认同时安装，是否替代或可以叠加由具体验证记录决定。
mm 的环境变量影响及 Triton 层修改都必须显示在安装计划中。

## 6. 安装器行为

建议命令面：list、show、check、plan、apply、verify、status、rollback。
pip 安装只安装管理工具；修改目标环境通过显式 apply 执行。
首版只接受显式目标源码根目录 --target（包含 src/flag_gems/），不自动探测 site-packages。verify 通过 --python 指定设备解释器。

安装过程：

1. 定位目标，核对 profile、版本、文件 hash 和已有安装状态。
2. 生成计划，检查 requires/conflicts，获取并校验固定制品。
3. 在暂存树中完成整组预演，确认所有 patch 和最终文件 hash。
4. 锁定目标环境后重新核对全部目标文件 hash，确认与预演输入一致，再写备份与恢复日志，逐个原子替换目标文件。其他管理工具操作必须遵守同一把锁。
5. 完成安装状态记录，清理被修改模块对应的 Python 字节码缓存；需要重启使用者进程。
6. verify 在独立新进程运行验证并输出 JSON。当前不回写安装状态的 verification（保持 not-run），也不自动更新 catalog 的 hardware_status。

多文件替换不是整体文件系统原子事务：操作期间应停止使用目标环境，失败和异常中断依靠事务日志恢复。
首版不自动清空整个 Triton 缓存，也不在已运行的应用进程中热替换导入模块。
apply 默认不自动运行耗时硬件测试。安装成功与硬件验证通过分别展示，不能混为一个状态。

行为约束：

- 基线不匹配时停止，不使用模糊匹配或强制覆盖绕过检查。
- 重复 apply 为幂等操作；已有目标内容却没有安装日志时报告状态，不能伪造备份或安装记录。
- 同一文件被多个 fix 修改时，v1 一律拒绝，包括任何 bundle；共享文件组合与中间 hash 链留待后续设计。
- 首版按安装事务逆序回滚；不支持任意拆除被后续修复依赖的单项。
- 回滚前检查当前文件 hash；发现用户后续修改时保留现场，要求人工协调。
- 执行失败自动恢复已修改文件；验证失败通过输出 JSON 和退出码呈现，提供显式回滚入口；当前安装日志不持久化验证结果。

## 7. CI 和发布

普通 CI：schema、路径和来源摘要检查，patch 应用性，源码/patch 一致性，以及幂等安装、冲突拒绝、失败恢复、回滚和离线制品测试。
硬件 CI：在对应 NPU/XPU 环境执行单项复现、数值验收、公共 dispatch 和组合回归；性能门槛由各条目明确。
没有硬件 runner 时可接收平台负责人提供的结构化执行记录，但须关联精确环境、commit 和命令。
普通 CI 通过不代表硬件验证通过，单项通过不代表整组 bundle 通过。

接入过程：新修复提交 fixes/ PR；外部修复提交来源变更并更新 submodule/清单 PR → 普通 CI → 平台验证 → 后续锁定 bundle 与发布。
发布产物包含清单、固定 patch、必要验证脚本、来源信息、许可证和 checksums；验证脚本的额外依赖也须声明。
上游包含修复后，更新对应兼容范围和退役状态；旧基线需要的历史 bundle 保持可获取。

## 8. 分阶段实施与验收

MVP = P0 + P1；P0 单独交付时仅为目录预览版，P2 为后续扩展。

### P0：目录与约定

完成 README、schema、templates/fix/、8 个源仓库的条目映射、目标文件与基线盘点、负责人和生命周期字段。
提供 list/show 和清单校验。兼容性缺失可入目录，明确标记不可部署。
验收：每个条目能回答修什么、改哪个包、从哪个 commit 获取、由谁维护、目前在哪个环境验证。
模板验收：用一个现有 fix 填写完整实例，核对每步产物、源描述到 catalog 的映射以及所有验证入口；未填写模板本身不得被目录扫描器当作可部署修复。

### P1：最小安装闭环

选择 gather 和 lift_fresh 作为 Ascend 试点，实现 check/plan/apply/status/verify/rollback。
首版只支持已明确定位的可写源码树或纯 Python 包目录，依靠精确基线检查；不自动重装厂商环境。
验收：临时目标树中覆盖成功、重复安装、错误基线、第二项应用失败、回滚及回滚冲突；实际设备上完成两项单测和组合验收。

### P2：扩展与发布

接入剩余条目，补齐跨包、跨后端和共享文件策略；分别发布经验证的 Ascend、KunlunXIN bundle。
Spacemit 独立记录实机验证状态，不沿用 Ascend 结果。
生成兼容性表与离线包，旧仓库 README 指向统一安装入口。
验收：选定 bundle 在目标环境离线安装、验证与回滚可复现，来源、测试证据和制品对应一致。

首版暂不建设 Web UI、运行时 monkey patch、自动跨版本合并、复杂插件系统或完整 FlagGems fork。

## 9. 依据

本方案基于会话中对 FLAG-OP 公共仓库的只读检查；未复跑算子硬件测试，也未在本轮重新判断上游修复合入状态。

- 组织仓库：https://github.com/orgs/FLAG-OP/repositories
- 开发框架：https://github.com/FLAG-OP/FlagOS-OP
- 双层修复实例：https://github.com/FLAG-OP/geluFIX
- 多后端修复实例：https://github.com/FLAG-OP/arithFIX
- 手动安装实例：https://github.com/FLAG-OP/gatherFIX

## 10. 本次初始化状态

现有 8 个来源以固定 submodule 纳入，所有导入条目保持 deployable=false/hardware_status=unverified。本次未执行 NPU/XPU 复验；来源报告属于历史证据。工具提供目录与源码树安装管理入口，真实设备闭环及正式 profile/bundle 属于后续平台工作。CLI 与目录布局以当前代码和 README 为准；本设计中的拆分模块及离线发布工具可以后续演进。本文中的 arith/gelu 多条目拆分为演进目标；当前映射和可部署性以 catalog 实际条目为准。

### 初始化来源一致性结果

8 个固定 submodule 共建模 9 项 fix；5 项 source-checked，4 项 needs-adaptation。gather/lift_fresh/one_hot 在注释和署名处存在 patch/快照差异，严格预演失败；cann-pow 使用不支持的 normal diff，尚无完整目标 hash。全部 deployable=false、hardware_status=unverified。详见 [来源审计](source-audit.md)。这不构成算子数值、性能或硬件语义结论。
