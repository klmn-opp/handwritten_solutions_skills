# 手写真实性对照与个人校准

本文前半部分记录最初的合成 Profile 与替代样本实验；“尚无本人参考”和原版 Baseline 字节一致等描述对应当时归档产物。此后已有真实个人训练及共享渲染修正，当前 baseline 不等于原 v0.3.0。最新状态、数据和实验边界见 [当前实现审计](CURRENT_STATE_AUDIT.md)。

这次源码升级有两个独立开关。布局开关处理行和公式的位置；笔画开关处理同一字符的真实变体和字母倾斜。可在作业 JSON 顶层设置 `realism`，或用 `hwsol render --realism combined` 临时指定。交付 manifest 保存最终模式。已有 manifest 缺省使用 baseline。旧 v0.3.0 发行包可作为独立对照。[原版代码诊断和上游调研](REALISM_DIAGNOSIS.md)

## 五题对照

同一份 mixed 解答、同一 demo@v2 快照、A4 白纸、150 DPI、13 pt、seed 42。demo@v2 是已有两轮合成采集的工程 Profile，仅 x 有多变体个人数据。四份文件各五页，题号依次为 2、4、6、7、9；各通过五个数值等式和逐页 PDF→PNG 像素核验。完整 manifest、PNG、解答和报告位于本工作区的 `out/realism/comparison/<模式>/`。每份 PDF 另从其归档 manifest 重放，文件摘要一致。[可视对照](evidence/realism/comparison.png) · [持久化测量记录](evidence/realism/five-problems.json)

| PDF | 观测 |
|---|---|
| [Baseline](evidence/realism/baseline.pdf) | 与修改前在 e4e6349 实际生成的五题 PDF 摘要一致，SHA-256 为 08c02f8d24f285fc8c373ca18f03463847435683be6fc69e28a3552cea0afe5e。 |
| [Layout-enhanced](evidence/realism/layout-enhanced.pdf) | 第 6 题方差公式左上角由 (545, 262) 移至 (238, 271) px。分式内的字形和线条保留相对位置。 |
| [Stroke-enhanced](evidence/realism/stroke-enhanced.pdf) | 第 6 题的两个公式保持原位置。连续正文呈现相同的轻微倾斜与笔宽；同一表达式多次使用带变体的字形时可重新选样。 |
| [Combined](evidence/realism/combined.pdf) | 同时采用行规划与笔画处理。各题公式组的局部几何摘要、独立矩阵的单元内容／尺寸及运算符回退记录与基线逐页核对。 |

图中数学内容与解答逐项对应，包括均值 4、总体方差 8/3、行列式 7 和定积分 8。作业核验中的等式是输入声明的数值检查。局部几何摘要来自实际放置的公式字形与规则线的身份及变换，独立矩阵另记行列、单元内容与尺寸。像素核验检查 PDF 与 PNG 的一致性。三个检查各自的范围不同。

**字迹相似度**。本人尚未提供参考，暂不能评分。参考 GNHK 图 7，真实英文可有连续的连笔、字母高低差和更长的下伸笔画。工程样本的 x 由两条交叉线构成，视觉上容易与乘号混淆。当前工程 Profile 的英文仍由单字轮廓组成，尤其是重复的 e 和相邻小写字母；增加页内一致的倾斜与笔宽后，这个差距仍然可见。[GNHK 论文图 7](https://cdn.amazon.science/38/fe/4c3105fb43129bf59cc0aadb5d78/gnhk-a-dataset-for-english-handwriting-in-the-wild.pdf#page=8)

**整页自然度**。第 6 题的独立公式现在更靠近推导行，多个行首和词距出现可见变化。五道短解答仍集中在纸张上部；保留原推导和一题一页，没有填充空白。公开笔记图 1 呈现不同的缩进和留白，这些差异值得用本人作业校准。尚无盲评自然度分数。[GNHK 论文图 1](https://cdn.amazon.science/38/fe/4c3105fb43129bf59cc0aadb5d78/gnhk-a-dataset-for-english-handwriting-in-the-wild.pdf#page=4)

### 真实替代样本的留出实验

从 Lee、Chung 与 Lee 的 GNHK 论文第 4 页图 1 左图，取嵌入图像 (0, 0, 339, 452) 范围内的 7 行手写购物清单。用前 5 行建立单独的 reference-layout@v1，仍以 demo@v2 作为字形基底，后 2 行只用于比较。按标注框测量墨迹投影并人工核对行序，估计 em 为 28 像素。拍照倾斜、页面裁边、笔粗和题材差异均影响测量。公开样本未混入五题对照或用户 Profile。[GNHK 作者与数据许可 CC BY 4.0](https://github.com/GoodNotes/GNHK-dataset) · [扫描区域](evidence/realism/scan-regions.json)

两行留出文本为 macaroni、spaghetti。参照图中左边界为照片宽度的 22.42% 和 23.60%。在同一已发布快照上以四种模式渲染这两个词，PDF 按 150 DPI 栅格化，取灰度小于 140 的首个墨迹列，按页宽归一化，排除题号。两行平均绝对误差分别为 Baseline **14.39**、Layout-enhanced **9.95**、Stroke-enhanced **14.43**、Combined **10.00** 个页宽百分点。改善集中在布局开关；即使 Combined 也明显低估了这位作者后两行的缩进。该实验只测一张照片中的两个留出行起点，数学作业自然度和字迹保真度仍需本人样本与读者评价。[实验记录](evidence/realism/heldout-layout.json)

## CPU 实现

- 行规划在 Handnote 分词、字体度量与公式盒之上工作。用一个页面样本的行首、右侧留白和基线间隔决定行宽与密度，使用实际词间距及字体成对度量决定断行。对同一行的五个基线控制点做平滑插值。没有校准页时采用有界、相邻控制点相关的工程默认值；验证报告记录所选样本与规划行。
- 数学表达式沿用原 LaTeX 解析器。分子、分母、根号、上下标和线条随整个行内或独立公式移动。规划器只在需要换行时拆分无嵌套、无命令的简单算术连等式，保留所有等号。带结构的长公式保留原块，超出一页时报出具体题号。显式对齐／缩进沿用 Handnote，带吸附的横线、方格和点阵沿用纸张定位。
- 笔画开关只从同一码点的实际 GSUB 槽选择候选，保持测量好的轮廓范围和公式布局矩阵。正文采用同页一致的字母倾斜，可通过向下笔段的斜率校准。正式作业四种模式均只填充字形轮廓，正文、推导与最终结果使用同一用笔规则。原始压感可保存在 Profile 中。前文历史对照 PDF 使用当时版本的笔宽设置。外部回退字符沿用原轮廓。每页给出码点、具体字体、基底／个人／回退、变体编号与次数；几何笔画及无法逐字识别的栅格片段单列。
- 所有模式在最终墨迹变换后检查页数、边界及背景可见性。作业锁定字体文件以避免同名字体冲突，清除 MathText 未纳入字体的解析缓存。训练模块和可选研究模型保持在作业渲染之外。

Linux x86_64 的 Python 3.12、3.13 各通过 70 项测试，包含兼容后端、同进程字体切换、校准发布／回滚、扫描测量、复杂公式和自定义纸张。源码包与 wheel 已在本机离线构建。Windows 和 macOS 本轮未原生执行。[Python 3.12 记录](evidence/realism/tests-py312.xml) · [Python 3.13 记录](evidence/realism/tests-py313.xml)

## 在自己的机器上生成四份对照

选定一个已发布的 Profile。对照脚本先保存 Baseline 的来源、背景和 Profile 摘要，另外三组从该归档 manifest 读取相同输入。以下命令使用当前选定的 Profile，结果写入新的目录。

```bash
uv run --frozen python scripts/compare_realism.py --manifest examples/phase2/mixed/job.json \
  --output out/my-comparison --page 3
uv run --frozen hwsol verify --job out/my-comparison/combined --pixels
```

若要重建本次合成夹具，使用 `scripts/phase1_demo.py --output <新目录>`，将 `HWSOL_DATA_DIR` 设为生成的 studio-data 并执行 `hwsol profiles activate demo@v2`。新建快照的时间、样本 ID 和摘要会变化。按字节重放历史输出需要对应的渲染器版本、冻结的 Profile 和归档 manifest。

### 首轮实拍字格发现的公式间距问题

MathText 用首选字形计算相邻符号的位置。真实字格中的同一个数字可以有明显不同的宽度。用一枚较宽的 `2` 替换首选轮廓后，示例第 6 题的 `2+4` 在纸面上出现粘连，PDF 和 PNG 仍能通过像素一致性检查。源码版在数学字形越过首选字形横向边界时，按比例缩放或平移该轮廓，保留解析器计算的运算符位置。这项修正覆盖源码版的四种模式与 Handnote 探针。需要对照原版行为时，可使用 v0.3.0 wheel 渲染相同的 Profile 和解答。

## 本人样本与下一轮

可在对话中附文件，或放在本地 `samples/realism-inbox/`。建议用日常书写速度、常用的笔和纸，先交三页。首次只有一页也可开始。

1. 一页英文，8–12 行，含普通单词、标点和自然换行。
2. 一页数学推导，6–10 个表达式，包括分数、根号、上下标、积分、矩阵。若要专门补足字形变体，可再打印 [两套采集格](../examples/realism/math-symbols.pdf)，按页序回传，保留[配套模板 JSON](../examples/realism/math-symbols.json)。乘点请按格内符号填写。
3. 一页独立验收。按平时写作业的方式解“求 2、4、6 的均值和总体方差”，从题号写到答案。该页留到最后对照。

照片或扫描可用 PNG、JPG 或 PDF，保留纸张四边，附纸张尺寸和简短转写。平板可同时附原始笔迹，优先保留 x、y、时间、压力和抬笔。收到后用前两页构建候选 Profile，第三页用于四组盲读，分别记录字迹相似度、整页自然度、数字和运算符转写。当前结果支持先进行个人校准，尚无投入神经训练所需的人评证据。
