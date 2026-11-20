# M1 作者反馈修订

2026-11-20。原始 20 页 B 的作者视觉验收**未通过**。首轮完整修订及数学 d/f/k/r、
历史 ∈/双积分和范数竖线已集成。作者反馈「前面说的问题有明显改善」，同时指出
部分 2/π 偏粗、分数横线与长根号横线像印刷体、f 仍偏小偏矮偏淡。第二轮针对这些
明确问题进行有界审计和修订。第二轮已交付完整20页及主/独立 Agent 验收，
最新产物请读[第二轮报告](M1_SECOND_AUTHOR_REVISION.md)，下文保留首轮证据。
**不是神经训练，尚非发布候选。**
原报告、原 PDF、原始 7/8 留出结果均保留，修订不算新的独立泛化证据。

作者原话和输入 PDF SHA 保存在
[反馈记录](../workspace/analysis/20261120-master-execution/author-feedback-m1.json)。
第二轮原话和所评 PDF SHA 另存
[最新反馈](../workspace/analysis/20261120-master-execution/author-feedback-m1-round2.json)。

## 首轮最终交付与作者回评

运行 `20261120T115012607677Z-author-feedback-final-20-pages-50910a3a`：
[完整 20 页 PDF](../workspace/runs/20261120T115012607677Z-author-feedback-final-20-pages-50910a3a/rendered/homework_solutions.pdf) ·
[逐页 PNG/PDF](../workspace/runs/20261120T115012607677Z-author-feedback-final-20-pages-50910a3a/rendered/pages) ·
[根号、分数与符号局部](../workspace/runs/20261120T115012607677Z-author-feedback-final-20-pages-50910a3a/comparisons/roots-fractions-symbols/formula-regions.pdf) ·
[英文字母与数学变量局部](../workspace/runs/20261120T115012607677Z-author-feedback-final-20-pages-50910a3a/comparisons/english-and-math-letters/formula-regions.pdf) ·
[回退区域标注 PDF](../workspace/runs/20261120T115012607677Z-author-feedback-final-20-pages-50910a3a/comparisons/fallback-overlay/math-fallback-overlay.pdf) ·
[逐项来源、原因、语义和边界报告](../workspace/runs/20261120T115012607677Z-author-feedback-final-20-pages-50910a3a/reports)。

最终 2,265 个可见数学项，字体回退 **38→30**；源分数改写后少了 4 个不再需要的
分组定界符，未通过删公式降低计数。剩余 κ6、Φ5、ξ4、η3、⟨3、⟩3、∏2、≈1、ζ1、
ψ1、∮1。≈ 是继承字体，其余 29 项为外部字体。原因均为尚无审核通过且可适用于此
位置的个人轮廓；独立恢复案例中的 ℝ 不混入这 20 页统计。

来源细分类：personal-outline 1837、personal-outline-stretched 71、
personal-session-outline 100、personal-outline-geometric-calibration 7、
personal-scan-envelope 129、personal-component-composition 6、
personal-scan-outline 66、procedural-geometry 19、external-font 29、inherited-font 1。
变体拟合、根式接合、结构包络等还保留逐项 transformation，不能将这些计数都说成
「原样真实轮廓」。后续作者对横笔的意见也说明有来源并不等于风格已经还原。

主 Agent 已检查前序全部 20 页、全部 72 个数学 d/f/k/r 局部，以及最后再生成的
第 2/6/18 页。其余 17 页 PDF/PNG 与已审核数学 Latin 父运行逐字节相等。
数学 Latin 之后至最终版英文像素不变；原始 B 到最终版因明确字形校准和排版变动
不声称像素不变。作者回评认可改善，但不是全页风格验收通过。

按最新要求，新补采改用 [19 页完整数学采集册](COMPREHENSIVE_MATH_COLLECTION.md)，
不再为了缩减书写量继续最小化清单。旧 Session O/M/N 和原始扫描全部保留。

## 已完成的第一份完整修订

运行 `20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1`：
[完整 20 页 PDF](../workspace/runs/20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1/rendered/homework_solutions.pdf) ·
[逐页 PNG/PDF](../workspace/runs/20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1/rendered/pages) ·
[修复前原 B](../workspace/runs/20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1/comparisons/original-B20.pdf) ·
[逐页并排](../workspace/runs/20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1/comparisons/original-to-combined) ·
[公式局部 PDF](../workspace/runs/20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1/comparisons/all-original-formulas/formula-regions.pdf) ·
[逐项来源及检查](../workspace/runs/20261120T112636836466Z-author-feedback-combined-20-pages-e19104e1/reports)。

| 作者反馈 | 查明的原因 | 已执行的修订 |
| --- | --- | --- |
| 长、高根号左侧空白 | MathText 原槽与已替换真实轮廓的可见边缘不一致 | 按同高度实际笔墨间距收紧 7 个根式，目标约 5.52 px（11 pt / 300 DPI）；内部所有项整体平移、顶横缩短，轮廓及相对结构保留 |
| 分数应上下写 | 原答案部分使用斜杠，并非已有 `\\frac` 渲染成斜杠 | 37 处显式改写；逐项保存原片段、分子分母、坐标、SHA 及符号等价检查；单位斜线保留 |
| 标题等 0 粗重 | 两枚重变体原来仅在数学中被过滤 | 在正文/标题选择同一字体较轻的真实 0，共 20 项；不人工描细原稿 |
| 正文 r 间距 | 继承左边距偏大 | 487 项左边距与 advance 同减 .021 em，轮廓、空格策略不变；后续文字可能正常重排 |
| d 肚子偏高 | d0 的下部高于已有 d1 | 130 项选真实 d1 并拟合原框 |
| k 肚子偏高 | 两变体下臂均偏高 | 19 项输出几何单调下调，保留上下端点、原始字形和每项变换 |
| f 小、淡 | f1 更窄、更细，已有更饱满 f0 | 72 项选择真实 f0 原 em 比例；不膨胀、补画实际笔迹；逐一检查无新增邻字接触 |

14 个根式修正是数学专项 **20 页加独立压力页** 的计数；主 20 页实际为 7 个。
单数字根号沿用此前接受的处理。根号 stem 的最终风格仍需作者判断，不把更小间距当成
风格已被认可的证据。f 的横臂会影响全图“笔宽分位数”，不能用该数值宣称真实笔压改善。

## 真实来源与实现边界

新题英文采用 CPU 字体；旧七页 D 的前四页冻结神经英文不在本次重训或替换范围。
0/d/f 使用已有真实字形，k 使用有记录的输出几何校准；均不改原始 JPG/PDF、旧标签、
字体/Profile 或权重。compiled-font-lineage 将所用轮廓追溯到原 sample 字体；早期 Latin
表的注册坐标不是原 JPG 坐标，报告没有伪造 300 DPI 原色新裁剪。

实现：`research/math_feedback_candidate.py`、`research/cpu_glyph_feedback.py`、
`research/author_feedback_revision.py`。分数改写有精确输入 SHA 门禁，只适用于已经审核的
20 份源文，不是可对任意斜杠安全工作的通用解析器。后续答案应直接遵守已记入 AGENTS.md
的上下分数偏好。新修订使用独立运行，不改原 Benchmark 或已消费的 Holdout。

原 B 和数学单项候选先逐页精确重放，然后才比较集成结果。英文身份顺序保留；非目标
字形的来源/变体/轮廓保留，最大数值差约 4.55e-13 px；正文修订不改变数学字形来源或
轮廓，最大数值差约 2.27e-13 px。分数增高、r advance 修正会改变位置和页面像素，
本轮**不声称英文像素完全不变**。全 20 页经根 Agent 目视，后 10 页另经独立 Agent 目视；
未发现新增漏项、越界或裁切。检查范围和具体图片见
[根审核](../workspace/analysis/20261120-author-feedback-root-review/review.md) 与
[独立审核](../workspace/analysis/20261120-combined-independent-pages11-20/README.md)。

## 补齐数学变量与符号

第一份完整候选仍有 38 次字体回退。历史有界搜索实际审核 20 个原迹区域，找到两枚
∈ 和一对真实相邻积分号；原色图、标签、原 PDF、300 DPI 坐标变换均保留。
复核发现右侧 ∈ 裁剪边缘有墨，保持 held；最终只接收完整的左侧 ∈。
双积分邻近 A 是独立连通域，只允许明确分割，不允许擦改原图。
见 [历史搜索及来源报告](M1_MISSING_MATH_REVIEW.md)。其余符号维持明确回退，不用
印刷体、OCR 错标或普通 R 冒充个人 Φ、κ、ζ 或 ℝ。

再次检查发现数学变量有同源遗漏：d0 18 项、f1 2 项、k 7 项尚未应用正文校准。
数学 r 还存在另一原因：MathText 用 r0 的槽排版，窄 r1 仍占该槽，10 项可能多留约
157/1000 em 的尾部空白（当前约 7.22 px）。不能把正文度量补丁机械套用数学。
这些变量已通过独立候选验证：d18/f2/k7/r10 共 37 项修订。数学使用 100 units/em，
正文使用 1000；第一次机械套用造成 9 项被安全门禁拒绝，失败运行保留。新 f/r 从
既有 MathText 方法取真实原生轮廓，k 在已拟合轮廓上变换，d 拟合原框。
第 12 页 rad/s 的 r→a 间距从 9.06 到 1.84 px，a 与单位空间保持原位置。

历史 ∈ 与双积分共替换 2 项回退；双积分保留原 `\\int\\int` 标签，明确记录映射到
U+222C。原 A 邻字及原色图没有擦改，仅从派生轮廓中排除独立 A 连通域。
第 2 页 6 个范数定界符由已审核 Session N 单竖笔两次放置组成，明确记录为结构组合，
不冒充新采集的完整双竖符号。全部 8 项增量只改变目标槽内像素。

实际源码：`research/math_latin_feedback.py`、`research/historical_math_candidate.py`、
`research/math_paired_bars.py`、`research/author_feedback_delivery.py`。
每个 run 保存精确脚本、Profile、种子、配置、数据来源、旧版对照和失败说明。

## 当前准入结论

**Go：** 首轮修订作为后续隔离实验和作者对照基线；作者已确认明显改善。
完整采集册通过主/独立 Agent 打印与结构审核，可以交给作者填写。

**No-Go：** 宣称作者认可、数学零回退或通用符号覆盖；生产发布、默认 Profile 替换；
把本次字体/排版修正称为神经训练。第二轮必须检查粗 2/π、横笔形态和 f 的源尺度，
并以实际全页判断；不能通过增加测试或训练次数替代质量。正式 main/latest 保持不变。
