# M1 完整作业与数学覆盖验收

2026-11-20。**本轮完整作业基准已执行，候选继续保留为研究版本；v1.0 发布 No-Go。**
随后作者视觉验收指出长/高根号、斜线分数及部分 CPU 字形问题，当前 B 未通过风格验收。
首轮修订已获作者明显改善评价；[第二轮修订](M1_SECOND_AUTHOR_REVISION.md)的完整20页
已通过主/独立 Agent 复核，仍待作者判断。原始统计与一次性留出结果不改写。
按最新要求改用 [19 页完整数学采集册](COMPREHENSIVE_MATH_COLLECTION.md)；下文保留原始基准记录。
原始 24 题中，开发集 13 题出页、3 题按预期明确拒绝；一次性留出集 7 题出页、
1 题发生未预期的明确拒绝。所有成功页面均经过主 Agent 阅读与数学局部检查。
拒绝、来源缺口和未验收的风格问题均保留，不能把这轮结果称为“任意公式零回退”。

## 直接查看产物

| 内容 | 本地产物 |
| --- | --- |
| B：原始成功的 20 页完整数学作业 | [PDF](../workspace/runs/20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7/rendered/homework_solutions.pdf) · [逐页 PNG 目录](../workspace/runs/20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7/rendered/pages) |
| A：相同内容、英文、背景和布局条件 | [CPU 字体基线 PDF](../workspace/runs/20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7/comparisons/A-font-cpu.pdf) |
| 173 组公式/矩阵局部对照 | [PDF](../workspace/runs/20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7/comparisons/all-formula-regions.pdf) |
| 标注实际字体回退区域 | [20 页标注 PDF](../workspace/runs/20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7/comparisons/math-fallback-overlay.pdf) |
| 每个可见数学项目及变换 | [完整 Provenance JSON](../workspace/runs/20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7/reports/math-glyph-provenance.json) · [简表 CSV](../workspace/runs/20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7/reports/math-glyph-inventory.csv) |
| 全部成功与失败输入 | [可编辑解答](../workspace/runs/20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7/solutions) · [逐题状态](../workspace/runs/20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7/reports/cases.json) |
| 独立原题→解题→校核→实际 CPU 出稿 | [4 页 PDF](../workspace/runs/20261120T103556645202Z-m1-text-end-to-end-5127591d/rendered/homework_solutions.pdf) |
| 失败题的显式等价恢复 | [恢复后的候选 PDF](../workspace/runs/20261120T104853010542Z-homework-m1-render-7595af17/rendered/homework_solutions.pdf) · [原因、原式、逐字差异与验证](M1_LIMITATIONS.md) |
| 历史最小补采 O（现由 O-FULL 替代） | [打印 PDF](../workspace/runs/20261120T105153402369Z-m1-optional-math-session-o-137ebfeb/solutions/collection/session-O.pdf) · [PNG](../workspace/runs/20261120T105153402369Z-m1-optional-math-session-o-137ebfeb/solutions/collection/session-O.png) |

原始 20 页集合不夹入恢复页或重复页来提高通过率。恢复页与 4 道端到端题单独保存。
所有产物留在私有 `workspace/runs/`，未激活正式 latest。

## 假设、对照与冻结

本轮检验：原来只在七页中成功的 v0.10.26 个人数学 Resolver，能否用于新的完整
作业，在不改变英文墨迹和数学结构的条件下减少字体回退。A 使用已有 CPU 字体路径，
B 使用冻结的个人数学解析/替换路径；两者英文均为 Handnote CPU。
**这不是新的神经英文模型比较，也没有用旧题缓存词图冒充新内容生成。**
旧 v21 英文权重、v23 词距和 v26 七页的复现另见 [M0 报告](M0_FREEZE_REPORT.md)。

24 道题由 Agent 新编，覆盖线性代数、微积分、统计和物理，各 6 题；包含长英文、
多小问、密集计算、背景、矩阵与新数学符号。它们没有参与此前七页的调整，但不属于
从外部大学收集的独立作业，也不是新增作者笔迹。16 道开发、8 道一次性留出在执行前
分开；[协议](M1_BENCHMARK_PROTOCOL.md)和候选冻结于 Git `17d664d`：

- 冻结：`20261120T104305918542Z-m1-frozen-evaluation-candidate-19c55bd1`。
- 开发：`20261120T104306262492Z-homework-m1-render-cb8276e6`。
- 一次性留出：`20261120T104341625106Z-homework-m1-render-38ea743e`。
- 主 Agent 汇总：`20261120T104919541290Z-m1-reviewed-homework-evidence-f4b2cca7`。

冻结记录绑定源码、Profile、原始数学素材、解释器/依赖/字体哈希与协议。消费账本固定
在仓库 `workspace/registry/evaluation_uses/`，不能通过另换工作目录或候选绕过。
当前这些题都已曝光，未来修复只能作为回归，不能再次称为全新留出。

## 实际结果与边界

| 集合 | 实际结果 | 结论边界 |
| --- | --- | --- |
| 16 题开发 | 13 成功出页；`cases`、`underbrace`、`hbar` 三项明确拒绝 | 三项为预先声明的不支持路径，并未静默漏掉 |
| 8 题一次性留出 | 7 成功出页；1 项跨矩阵的大括号解析失败 | 保留原始 7/8，不能改报 8/8 |
| 4 题文本端到端 | 独立解答、代数/数值校核和实际 CPU 出页通过 | 只验证文本输入，不包含图片/PDF OCR；重放脚本是答案记录，不是新自动解题模型 |
| 暴露后的恢复 | 定义原矩阵列为 `v`，再写集合；语义等价、完整单页通过 | 源文本明确修订；没有修复通用嵌套矩阵解析器 |

原始 20 页共有 **2,269 个可见数学项目**，字体回退 **201 → 38 次**。
全部 173 组局部对照的数学项目身份、顺序和解析轨迹保持一致；英文区域像素差为零。
这些程序检查是工程证据，不能独自证明数学正确。主 Agent 另阅读题目、推导和全部页面，
结合独立精确计算，没有发现已接受输出中的漏字、错号、裁切或表达式碰撞。
高根式的个人竖笔仍细而略斜，其可辨认程度与是否像本人，需要作者视觉确认。

| 来源类别（20 页 B） | 项目数 | 实际含义 |
| --- | ---: | --- |
| personal-outline | 1,883 | 已有个人字体轮廓 |
| personal-outline-stretched | 71 | 个人轮廓按布局拉伸，不是新的原始笔迹 |
| personal-session-outline | 100 | M/N 原迹驱动，具体缩放/适配见逐项记录 |
| personal-scan-envelope | 92 | 原迹包络驱动的动态结构笔画，不是完整原字形复制 |
| personal-scan-outline | 66 | 历史作业原迹轮廓，具体变换可追溯 |
| procedural-geometry | 19 | 明确记录的程序结构笔画 |
| external-font | 37 | 已报告的外部字体回退 |
| inherited-font | 1 | 已报告的继承字体 |

不能将上述个人来源类别一概表述为“未经变换的原始轮廓”。原色源样本仍保留；出稿中的
缩放、结构拟合和根式分段变换必须与源样本区分。新增数学研究页的英文完全冻结在本轮
CPU 条件下，不能据此宣称神经英文全新内容已验收。

## 缺口与最小补采

下表按实际出现次数排列；结构算子虽然频率低，视觉显著性仍高。不为消灭指标而冒险
把不等价字形强行拼成新符号。

| 字符 | 原始 20 页回退次数 | 原因/下一步 |
| --- | ---: | --- |
| κ、‖ | 各 6 | κ 缺少可靠已采素材；‖ 可评估双竖线结构组合，也可补采 |
| Φ | 5 | 未发现可用已审核原迹 |
| ξ | 4 | 同上 |
| η、⟨、⟩ | 各 3 | 同上；角括号成对采集以保持习惯 |
| ∏ | 2 | 缺可靠个人乘积算子，多尺寸需要结构适配验收 |
| ζ、ψ、≈、∈、∮、∬ | 各 1 | 缺已审核素材；双积分可另评估结构组合 |
| ℝ | 原始 20 页无；恢复页新出现 | 恢复页明确字体回退，加入补采表 |

已搜索已接受的历史/M/N 符号清单、当前 Profile 和已审核语料标签；不宣称穷尽了所有
尚未定位的原始 PDF 区域。OCR 中的可疑命中不能直接晋升为真值。
Session O 是可选的快速补足路径，前六行覆盖 **13 个缺失字符**：
`Φ κ ζ ξ η ψ ∏ ∮ ≈ ∈ ℝ ⟨ ⟩`；第七行 `‖ ∬` 可选。
无需重写已提供的根号、0、绝对值和 M/N 已接受符号。
打印后按个人习惯在提示下方书写，原色扫描保存为 `source_pdf/session-O_source.pdf`。
空白表本身不要混入原始笔迹库。提示不是确认标签，导入时仍按实际笔迹、原 PDF、
300 DPI 原色裁剪与坐标变换核验。此项叫字形采集，不叫神经网络训练。

## 失败原因与已执行修复

1. 跨矩阵的 `\left\{ ... \right\}`：Handnote 按矩阵拆平公式，左右括号分组被
   拆断；Resolver 尚未调用，因而补写花括号或继续训练都不能解决。保留明确拒绝，
   本轮仅给出经独立代数校核的显式等价写法。未来通用修复需支持平衡嵌套结构。
2. 端到端物理题多行显示公式：原始失败保存于
   `20261120T103527461026Z-m1-text-end-to-end-e6103b9e`。将同一 LaTeX 公式整理为
   单行后通过；未改变数学内容。
3. 历史导出 Job 的模式是作者模板 `combined`，实际运行配置为 `baseline`。
   原档不改写；新入口保留未执行模板，另导出准确配置/Profile pin 的执行 Job。
   16 个模板逐字保留，一道已曝光题重放 A/B 均为 0 像素差。详见 [限制报告](M1_LIMITATIONS.md)。

实际命令、源码快照、种子、Profile、依赖与运行设备均保存在各 run；主要入口：

```bash
.venv/bin/python -m research.project_state verify
PYTHONPATH=.:src MPLCONFIGDIR=/tmp/hwsol-m1-mpl .venv/bin/python -m research.homework_benchmark.render --help
.venv/bin/python -m research.homework_benchmark.report
.venv/bin/python -m research.homework_benchmark.collection
```

`report` 汇总固定的已验收证据；`collection` 每次生成新的模板 run。一般新作业使用
`render --manifest <job.json>`。不能重跑已消费的留出集。旧记录中的作者模板不是精确
重放入口，须使用归档运行配置/源码或限制报告中的准确执行 Job。

## Go / No-Go 与下一入口

**Go：** 保留 v26 为研究基线；使用新通用 manifest 入口继续明确报告回退的 CPU/个人数学
实验；保留完整页面、原始失败、等价恢复与工程保护。继续公开安装包、语料注册和机制诊断。

**No-Go：** 直接发布 v1.0、替换默认 Profile、宣称任意数学零回退、宣称新神经英文泛化，
或将本轮页面当作作者风格盲评成功。全局产品验收还缺个人数学正式候选整合、更多不支持
结构的处理、macOS 实机验证和作者对实际页面的视觉验收。本轮数学路径没有新增神经训练。
作者已有“B 更好”的评价只覆盖此前 v24 对照，不能自动扩展为这轮候选发布授权。
