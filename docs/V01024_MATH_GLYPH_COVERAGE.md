# Math Glyph Coverage Report · v0.10.24

2026-11-20。本轮从 [PROJECT_AUDIT_2026-11-20](../PROJECT_AUDIT_2026-11-20.md) 的审计提交
`41f760d` 继续，在 `experiment/personal-math-glyph-v01024` 创建私有实验。正文基线为 v0.10.23
四题混排，渲染使用 `personal@v1`、300 DPI、白纸、版面 seed 1909、冻结英文 seed 2012。
解析器只在实验分支的研究模块显式启用。

## 产物

| 内容 | 链接 |
| --- | --- |
| 纸面 A/B | [A 完整 PDF](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/rendered/A-homework.pdf) · [B 完整 PDF](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/rendered/B-homework.pdf) · [按题交替对照](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/comparisons/A-B-homework.pdf) |
| PNG | [A 逐页](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/rendered/pages/A) · [B 逐页](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/rendered/pages/B) |
| 公式局部 | [32 组局部对照 PDF](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/comparisons/formula-regions.pdf) · [局部 PNG](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/comparisons) |
| 回退区 | [原四页标注 PDF](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/comparisons/math-fallback-overlay.pdf) · [谱分解 PNG](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/comparisons/fallback-page-01.png) · [参数积分 PNG](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/comparisons/fallback-page-02.png) · [回归 PNG](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/comparisons/fallback-page-03.png) · [RLC PNG](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/comparisons/fallback-page-04.png) |
| 逐项清单 | [四页来源与优先级](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/reports/math-glyph-coverage.json) · [六页 A/B Glyph Provenance](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/reports/glyph-provenance.json) · [固定基准原始逐项清单](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/reports/benchmark-baseline-inventory.json) |
| 历史原件 | [裁剪、父区域、标签与坐标清单](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/solutions/historical-glyphs/sources.json) · [300 DPI 原色裁剪](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/solutions/historical-glyphs/crops) |
| 补写 | [M 优先补写表](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/collection/session-M.pdf) · [N 可选补写表](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/collection/session-N.pdf) |
| 验收 | [数学语义](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/reports/semantic-verification.json) · [重放回执](../workspace/runs/20261120T051702972180Z-v01024-math-glyph-verification-bd71dc27/reports/result.json) · [Go/No-Go](V01024_GO_NO_GO.md) |

## 原四页来源清点

分母始终为 761 个数学可见项。逐项保留页、绘制索引、Unicode、实际字体或字形 ID、公式原文、角色、
像素坐标与 Profile 摘要。基线清单保留原始字段，并标明候选阶段识别出的结构为矩阵括号、极限关系箭头
或帽号。U+2192 独立关系箭头与 U+20D7 字母上方向量箭头分别审计。

| 来源 | A | B |
| --- | ---: | ---: |
| 个人字体轮廓 | 636 | 636 |
| 拉伸个人字体轮廓 | 19 | 33 |
| 历史原迹提取轮廓 | 0 | 33 |
| 个人扫描横线包络 | 0 | 25 |
| 程序结构 | 52 | 13 |
| Profile 继承字体 | 22 | 2 |
| 外部字体 | 32 | 19 |
| 合计 | 761 | 761 |

程序项中的 11 根横线在 B 按个人根号主体的笔宽接合，仍归为程序结构；另外 2 根原高根号横线保留。
字体回退合计由 54 降到 21。固定 Benchmark v1 的历史 60 例也逐项核查，共 1,596 个数学可见项，
含外部字体 37、继承字体 42，以及程序结构 12。该历史控制使用 `personal@v3`，含 80 个已测量的扫描横线项。
执行本轮 CPU 控制得到 54 例出页、6 例预期拒绝。B 的对照范围为本报告六页。
[固定套件控制记录](../workspace/runs/v01024-cpu-benchmark-control/reports/benchmark.json)

## 发生过的缺口与替换结果

优先级按四页实际次数 × 视觉显著性 ÷ 替换风险计算，后二者各取 1–3。分数线与带上下限的求和需要
保障宽度，帽号、向量标记、长根号和长括号需要保障几何附着。完整分值及每项原因见覆盖 JSON。

| 项目 | A 中的缺口 | 原因与 B 的结果 |
| --- | ---: | --- |
| λ | 10 继承 | Profile 缺采集字形。用物理作业的原迹替换 10 处。 |
| 分数线 | 25 程序 | v1 没有已测量笔画，v3 有分数线包络。复用该来源，在原矩形内替换 25 处。 |
| ∞ | 6 继承 | Profile 缺采集字形。用公式页原迹替换 6 处。 |
| ω | 7 外部 | 现有电路笔记写法接近拉丁 `w`。样本留待作者确认，7 处沿用原回退。 |
| 根号横线 | 13 程序 | 必须保持被开方项的范围和间隙。11 处按现有个人根号主体接合，在原联合墨迹框内落笔。2 处高根式保留原横线。 |
| 矩阵圆括号 | 14 程序 | v1 有真实括号轮廓；Handnote 的内联矩阵原来走程序笔画。B 按原左右边框复用第二个个人变体。 |
| ∂ | 2 外部 | Profile 缺采集字形。用偏导公式原迹替换 2 处。 |
| ′ | 5 外部 | Profile 缺采集字形。用导数原迹替换 5 处，双 prime 保留两个独立项。 |
| 向量上箭头 | 5 外部 | 已审材料缺足够清楚的字母上方箭头样本。独立右箭头有单独的来源和用途。 |
| β | 3 外部 | Profile 缺采集字形。用概率公式中的原迹替换 3 处。帽号保持原角色。 |
| ∑ | 3 继承 | Profile 缺采集字形。用带上下限的历史求和原迹替换 3 处。 |
| ± | 2 继承 | 已审材料缺可确认的原迹。2 处保留。 |
| 帽号 | 3 外部 | 已审材料缺可确认且能安全分割的原迹。3 处保留。 |
| → | 3 外部 | Profile 缺采集字形。用物理作业独立右箭头原迹替换 3 处。 |
| 高根号主体 | 2 外部 | Profile 已有短根号，拉伸比超出现有门限。2 处保持原回退和被开方结构。 |
| 冒号 | 1 继承 | Profile 缺采集字形。用本人原稿标题中的冒号替换。 |
| 长 `[` 和 `]` | 各 1 外部 | 已找到短方括号，长括号的直接拟合会缩短围合高度。RLC 长括号保留，短括号在扩展页获准使用。 |

扩展页还保留长方括号矩阵的程序笔画、个别高或嵌套根式，以及 μ、ω、θ、φ、帽号、向量箭头、
±、大括号和大于号各自的基线来源，逐项原因在 Glyph Provenance。六页合计使用了全部 18 类获准轮廓，冒号出现于原回归页。

## 历史原始墨迹

查看了原已导入作业的目标页，以及 Formula Sheet 已审核的 39 个数学区域。候选定位复用已有 UniMERNet
建议和 Agent 审核记录；图像与公式上下文用于裁剪级核验。已有 OCR 在 CAL3 漏掉 prime、在 PHY2 的根式中
混淆符号，均保留为候选记录。**18 类为本轮 Agent 核验的可隔离原迹，另有 1 个角频率候选暂存。**
已有人工确认字段与本轮 Agent 判定各自保留。

| 原件与页码 | 通过隔离审核的符号 |
| --- | --- |
| Formula Sheet I，2 页 | ∑、∞、[、]、∩、: |
| Formula Sheet I，5 页 | ∂、β、α、ρ、≥ |
| PHY1，17 / 102 页 | λ、τ |
| PHY2，54 / 55 页 | →、π |
| CAL3，124 页 | ′ |
| CAL2，68 页 | ε、δ |

保存原 PDF 摘要、原始转写、父裁剪、300 DPI RGB 裁剪、像素框、PDF 点坐标、双向矩阵和裁剪原点变换。
既有已审核父裁剪与 PDFium 重放像素逐项比对。对原始细笔仅提取亮度等值轮廓作为渲染视图。
合并参照公式在父区域内查阅读法，印刷参考与标准答案均未参与裁剪标签改写。角频率反例单独保留，
其父区域按纸面写法记录为 `w`，未进入 B。封存 Session C 留在原归档。

## 语义和视觉边界

- 保留 Handnote 原排版和 MathText 的字形顺序、公式源码、矩阵行列、上下限与上下标的槽位。
  局部替换只改对应绘制项。SymPy 独立核对谱分解、参数积分与微分、回归残差与方差、
  RLC 初值与微分方程，以及两页扩展中的数值和解析结果。
- 原四页 A 的像素与 v0.10.23 交付一致。B 的 526 个英文神经词、原位回退词组和 477 个词距观测
  保留相同的墨迹与坐标。新增两页两臂均共用 Handnote 英文。实验页脚的 B 标签随版本标记变化。
- 六页裁切和边界通过实际 PDF 光栅核对。修改像素落在对应数学槽位内，正文字区像素零变化。
  11 组原根号接缝采用已验证的实验逻辑，并通过原联合框限制。局部对照使用相同缩放检查符号、间距与接缝。
- 12 张 A/B 页 PNG、4 张回退标注 PNG 及逐项来源报告经独立运行重放一致。19 份原色裁剪及各自父区域
  重新从原 PDF 提取并核对坐标。作者自然度与无提示独立转写仍待核验。
  [运行记录](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/run.json) · [独立重放](../workspace/runs/20261120T051613620006Z-v01024-personal-math-glyph-completion-71122bac/run.json)

## 补写清单

优先填 **Session M 的 6 行**。依次写角频率 ω、带上箭头的向量、帽号、±、高根号与嵌套根号、
围住分数的长方括号。每个提示用常用笔在下方空白行自然写一遍，保留页面和原色，单独存成一份 PDF。
可选 Session N 收集较少见希腊字母、∇、绝对值、运算关系、重音及矩阵长括号。
[Session M](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/collection/session-M.pdf) · [Session N](../workspace/runs/20261120T051613583545Z-v01024-personal-math-glyph-completion-e7b8c8e5/collection/session-N.pdf)
