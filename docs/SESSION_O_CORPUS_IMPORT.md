# Session O-FULL 原迹导入与语料状态

2026-11-20。本报告记录新上传 O-FULL 的不可变来源、实际书写审核和 Φ 主体尺寸诊断。
本次没有神经参数更新；原迹采集、裁剪、轮廓提取及几何校准均不称为神经网络训练。
整页渲染、符号替换和发布资格见已完成的[最终工程报告](SESSION_O_MATH_COMPLETION.md)。

## 数量与来源单位

| 层级 | 本次数量 | 含义与限制 |
| --- | ---: | --- |
| 原始文档 / 采集会话 | 1 / 1 | 同一份 O-FULL，不是 19 个独立作者来源 |
| 原始页面 | 19 | 同源 300 DPI RGB 渲染，未变形或清理笔迹 |
| 印刷提示行 | 123 | 每行有实际书写；提示与实际转写分字段保存 |
| 行裁剪视图 | 369 | 每行 3 种重叠视图，不能当作 369 个独立样本 |
| 列位置候选 region | 333 | 32 行产生的未标注几何候选，不能称为 333 个训练字形 |
| 未原子化的完整行 | 91 | 混合公式、优先字形所在行等均保留完整上下文 |
| 首批独立源审通过的孤立字形 | 30 | 12 种语义角色；只对应精确 source ID、原色 crop 和审批 |
| 新星号来源 | 3 | 精确 3 枚 `*`，已独立源审；最终 bank 共 33 枚、13 种角色 |

同一页面、行和字形的多 DPI 图、裁剪与轮廓不能跨独立 Train/Holdout 拆分。
本次来源已经用于开发和作者反馈，不是新的全系统 Holdout。未给 333 个 region
自动标注 Unicode、重复组号或训练 split，也没有把整行审核等同于隔离字形准入。

原文件是 `source_pdf/session-O_full_source.pdf`，SHA256：
`23a84ff4cf5dc4882b5fb76d5eb4cc3a32c59e9799ea7a351f545598401f4d43`。
实际 PDF 为 595.440002 × 841.679993 pt、Rotate 0；渲染是 2482 × 3507 RGB。
按实际 PDF/栅格尺寸存储双向变换，未用整数 A4 尺寸替代，也未从乱码 PDF 文字层猜标签。

## 不可变运行与审核入口

以下路径均相对仓库根目录，每个 run 都保存独立源码快照和 `run.json`；状态为
`prepared`，已通过 `verify_run`。状态表示来源证据准备完成，不代表生产发布。

| 用途 | 运行目录 | `run.json` SHA256 |
| --- | --- | --- |
| 原 PDF 与 19 页 RGB | `workspace/runs/20261120T133948143789Z-session-o-full-source-rgb-import-7e119738` | `4808657394d5da511ddae8ed6280759a412c6fa7047e06d9258a3f50a3990b1a` |
| 123 行来源索引 | `workspace/runs/20261120T134345155374Z-session-o-full-row-source-index-6886ffcf` | `0239b5b50d0f71cbed5a578f7000b2414707a50b589b0f2e0b00f93a99fea96d` |
| 333 未标注列候选 | `workspace/runs/20261120T134825480758Z-session-o-unlabelled-symbol-column-proposals-c1312600` | `6ae5d8ba9497defde8c41646a32cff1a5f45e5a79562ab00964db264e7cb2d38` |
| Φ 原迹主体测量 | `workspace/runs/20261120T135736179146Z-session-o-phi-body-source-audit-d762aaae` | `75275ac8389a57b190fed955960498034672ab9129b137bea9d5ce232c63cbc6` |

页面原件：第一个 run 的 `solutions/session-O_full_source.pdf` 和
`solutions/pages/page-01.png` 至 `page-19.png`；`reports/pages.json` 保存 PDF SHA、
PNG SHA、解码 RGB SHA、300 DPI 声明、实际页尺寸、裁切框及 PDF↔栅格矩阵。

行索引：第二个 run 的 `reports/rows.json`；每行都有原色 `context-rgb.png`、
`writing-band-rgb.png`、`lower-overlap-rgb.png`。完整页仍是权威。
上下文裁剪允许重叠下一行印刷提示，最后一行延伸到页底，避免丢失作者写到 guide
下方的第三遍，如 P01R07、P02R07、P13R01/R04 等。名义写区不能充当安全字形边界。
`comparisons/all-row-contexts.pdf` 与 19 张分页联系图供审核。

列候选：第三个 run 的 `reports/candidates.json`、`column-groups.json`、
`unatomized-rows.json`；原色候选位于 `solutions/column-candidates/`，32 张
`comparisons/native-columns/` 保留原生像素大小。每项保存原页和父行、矩形坐标、
坐标变换、SHA、候选序号、独立的 intended prompt；`actual_label=null`，
`training_ready=false`，不从印刷顺序生成真值。

所有 369 行裁剪和 333 列 region 均逐像素等于其原页矩形；变换往返最大记录误差
为 4.55e-13 px。123 个印刷行 ID 定位与模板的偏移均为 0 px，最低相关值
0.9957202077；该检查只定位模板，不识别或批准手写标签。灰色 guide 留在原图中。
列投影使用灰度 <180 和至少 15 个空白列定位，未对原色做擦除、补画或形态学清理。

投影有已知边界：P07R03 C12+C13 是最后一个 `:=` 的两部分；P08R01 C02+C03
是第一组平行双竖线。长箭头、点列也可能出现一字多 region，连通 span 也可能含多字。
它们必须回看父行后合并或保留，不得按序号批量认可标签。优先 bank 所在行未重切，
因此同排其他 Greek 仍有完整行证据；P11–12 的符号加公式混排行也没有强拆。

## 实际书写审核与标签纪律

独立 Agent 已目视全部 19 页、123 行，没有观察到整行空白。
详细记录见 [独立原迹审核说明](../workspace/analysis/20261120T133823816211Z-o-full-independent-review-887ce17a/README.md)
与同目录 `ROW_REVIEW.json`；其 SHA256 为
`bafcb23ff03053c12bcdea9ce9c1978cc078d1713055d8cff62054821364cc8f`。
`SEALED_REVIEW.json` SHA256：
`33bdbb19a66c8eec54232f3254b716282121b9b2898c13922da25f9527f42feb`。

14 行存在具体角色 hold，不表示整行无用。原提示、实际观察与不确定性同时保留：

| 行 | 实际观察及限制 |
| --- | --- |
| P01R07 | 后两次 RHS 为 v/ν 样，不能从提示自动认作 γ |
| P03R03 | θ/ϑ/Θ 槽位均近闭环横杠形；未证明独立 ϑ 变体 |
| P03R05 | 实写 `π π Π`；第二项不是自动成立的 ϖ |
| P08R02 | 多个度数符号是短痕或开口环，不能统一批准闭圆形 |
| P08R06 | dagger 位置为普通十字，不据提示宣称独立 dagger 形态 |
| P09R06 | 普通单线 N Z Q R C，不冒充独立双线集合符号 |
| P13R03 | 高右角括号接近圆括号；短角括号的独立批准不受此替代 |
| P15R02 | 最后一遍 widehat 主要覆盖 B，不能宣称覆盖整个 AB |
| P15R06 | 短 tilde 多为帽/弧形，不能自动用波浪符标签 |
| P15R07 | 多为普通 F/L/H，不能从提示自动建立花体类别 |
| P18R01 | 前两矩阵为 `[a b; c d]`；第三下左实际是 **b**，保存 `[a b; b d]`，不改成 c |
| P19R03 | 部分 i/j 在帽下仍有点，不擦点以制造 dotless 源 |
| P19R04 | 普通 E(x)、P(A)，不强加黑板体或 E[X] 标签 |
| P19R05 | 普通 N，三组写在不同高度，不自动匹配整行 token 顺序 |

此外 P03R04 只有两组不是缺失数据；表格原本要求各写一次，不能强制三遍。
连写、自然断开的根式连接和贴近上下标都保留原样，无法安全隔离时保留公式/完整 span。

首批 30 个精确裁剪在 `20261120T134401932769Z-session-o-priority-original-crops-b9cafef1`
通过独立来源审核：π、κ、Φ、ξ、η、∏、≈、ζ、ψ 各 3 枚，⟨、⟩、∮ 各 1 枚。
`PRIORITY_SOURCE_APPROVAL.json` SHA256 为
`ea972c3f7671a1ebfa44b0309850c1f5bc94d6d1a53f56509d39e0378db8beeb`。
大连乘角色由真实上下文支持，不声称其 Pi 形笔迹必须区别于 capital Pi。
后续冻结 bank 是
`workspace/runs/20261120T135929419362Z-session-o-reviewed-glyph-bank-b803206a/solutions/glyphs`。
其 `sources.json` 实际为 **33 个 agent-accepted 源 / 13 种语义角色**，包含新增
`O-FULL-P07-R02-U002A-V01` 至 `V03`。bank run SHA256 是
`5a58f990f4b062f36725015c25c7144fb93d2218113b5be31a1c243983e27fe6`；
manifest SHA256 是 `2ae93d8c9f2261abfae4811a82e87f275232e3231bc7c2fc6cf53c7fc9372f18`。
原 30 裁剪不变，Φ 仅添加 `body_reference` 几何元数据；三个 body 引用指向下面
独立测量 JSON。`reports/source-approval.json` 与 `reports/result.json` 保存来源回执。
这里不将源审或 33 成员数提前当作目标尺寸、邻接安全、作者风格认可或发布通过。

## Φ 的主体与长竖：来源事实

初次完整 20 页实验 `20261120T134632687261Z-session-o-reviewed-math-complete-20-adf8f1c8`
显示 Φ 相对 κ/π 偏细。原色复查发现：既有三枚 Φ 的长竖远超圆碗，整字外包框
不能直接当作普通大写字的主体高度。P02R07、P03R02 上下文亦作过目视候选定位，
未因此批准新来源；本诊断最终只测现有三枚已接受 P02R06 源和同组相邻 Ω。

| 既有 Φ 源 | 圆碗 bbox：原 crop 半开坐标 `[x0,y0,x1,y1)` | 全墨迹高 | 圆碗高 / 宽 | 相邻 Ω 高 | 全高 / 圆碗高 |
| --- | --- | ---: | --- | ---: | ---: |
| V01 | `[5,29,77,94)` | 118 px | 65 / 72 px | 64 px | 1.815 |
| V02 | `[5,30,69,103)` | 133 px | 73 / 64 px | 72 px | 1.822 |
| V03 | `[5,29,83,107)` | 142 px | 78 / 78 px | 76 px | 1.821 |

这些是 Agent 目视原色、按当前轮廓灰度阈值 L<160 核对的 body 外缘像素包络，
边界不确定度标为 ±1 px；不是笔画中心线，也不声称是物理轮廓的精确分解。
抗锯齿的连续轮廓可能越过整数边界小于一像素。body 只作几何锚点，绝不用于裁掉长竖。
圆碗/相邻实际 Ω 高度比为 1.016/1.014/1.026，三组原迹都支持相同的 body 对齐解释。

完整证据见 Φ run 的 `reports/body-measurements.json` 和
`comparisons/phi-body-and-omega.png` / `.pdf`；原色裁剪与注释副本分别保存。
测量 JSON SHA256 为 `de10fa5ffbb53b8e4c1f81f2967d3f8918d31aebc1a15f7d9b0445adcee35d90`。

可检验规则是：整个 contour 只用一个统一 scale，以 body 高度对应既有 MathText
cap 槽高度，body 中心对应槽中心，完整长竖保留并伸入经验证的周边空白。
body 若为 30 px，完整字高约 54.46/54.66/54.62 px；这是来源比例推导，
不是指定页面调参结果或自动准入。原变体选择、数学语义与布局解析保持。
超出旧墨迹框的竖笔和字宽需要实际邻接、页边与非目标像素门禁；不安全时明确保留旧项。
不通过加粗、擦细、补画或选更好看的 seed 处理问题。实际候选由主 Agent 独立渲染与验收。

## 源码、命令、环境与保护

导入实现为 [research/session_o_import.py](../research/session_o_import.py)。三个阶段
逐步形成独立源码快照，不能只凭同一 Git commit 假设脚本完全相同：

| 阶段 | `logs/source/session_o_import.py` SHA256 |
| --- | --- |
| RGB 页面 | `0d8cae019fbba22f7fb3679245564ef83c88be6456db5df2eb750d992d74a8b6` |
| 行索引 | `198414c7104806aabba0084c5e5f5378719b3f7dcfa7936d7d1d912de28936d2` |
| 列候选快照 | `5091f1c381331d6654e81a82a70e7ce59f484d5f2708179f5c92c96086768433` |

提交前仅统一 Python 缩进格式，实现提交 `ae9c9db`；当前 importer SHA256 为 `c10dfb403837a8e87aea09851d1bc5ba5dc785c5efd36ad8670467b9fb0ed9ed`；
以上运行保留各自实际使用的源码快照，不重写旧回执。

三次 run 记录的源码 commit 为 `9881dc744dc609643aca6923c0a7bfc022f049b4`，seed 为 0；
实际研究脚本还须按上表快照 SHA 定位。Φ 实现
[research/session_o_phi_audit.py](../research/session_o_phi_audit.py) SHA256 为
`c5bf9bdcadfa5872329fa43e140bf7a4e3f691e6f9779af269fe98f0f054fdd4`。

执行命令（每次调用创建新 ID，不覆盖已有 run）：

```bash
MPLCONFIGDIR=/tmp/hwsol-mpl .venv/bin/python -m research.session_o_import render
MPLCONFIGDIR=/tmp/hwsol-mpl .venv/bin/python -m research.session_o_import rows \
  --parent workspace/runs/20261120T133948143789Z-session-o-full-source-rgb-import-7e119738
MPLCONFIGDIR=/tmp/hwsol-mpl .venv/bin/python -m research.session_o_import columns \
  --parent workspace/runs/20261120T134345155374Z-session-o-full-row-source-index-6886ffcf
MPLCONFIGDIR=/tmp/hwsol-mpl .venv/bin/python -m research.session_o_phi_audit
.venv/bin/ruff check research/session_o_import.py research/session_o_phi_audit.py
```

运行环境为 Linux 6.8.0-110 x86_64、Python 3.12.16、32 逻辑 CPU；
Pillow 13.0.0、NumPy 2.5.5、OpenCV headless 5.0.0.93、pypdfium2 5.15.2、
contourpy 1.4.0。流程只用 CPU，未启动 CUDA、OCR 云服务或神经训练，也未给 CPU Runtime
新增 Torch 强制依赖。SHA、原页矩形一致性、坐标往返和 Ruff 是工程验证，不当作笔迹质量提升证据。

本次仅访问新授权 O-FULL、其空白模板、相应新 run 与审核记录；未打开历史数据库、
旧历史 PDF、封存 C 或旧 Holdout。没有覆盖原扫描、既有 Profile、原标签、训练数据、
正式 latest 或生产分支，也没有上传个人资料。上述访问边界只描述本次 O-FULL 工作，
不改写此前单独披露的历史目录访问事件。

目前 Go 的是来源保全、逐行实际书写审核及精确获批孤立源的候选资格。
333 几何候选仍须逐项标签/完整性审核，角色 hold 继续保留；不安全的连笔不强拆。
bank 已确定 33 枚；完整页面效果、零回退适用范围和发布 Go/No-Go 以主 Agent 最终报告为准。
