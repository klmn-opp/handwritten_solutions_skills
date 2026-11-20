# v0.10.25 Session M/N 个人数学字形校准

作者对 v0.10.24 同页 A/B 的反馈为「明显B的更好」。本轮以该版 B 为对照，新增 C 为 M/N 校准稿。
之前的整体偏好原样归档。C 的独立作者评审待进行。
[作者反馈记录](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/reports/author-feedback.json)

## 成品

- [新版 C 完整七页 PDF](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/rendered/C-homework.pdf) · [C 逐页 PNG](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/rendered/pages/C)
- [B/C 按题交替对照](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/comparisons/B-C-homework.pdf) · [26 组局部对照](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/comparisons/formula-regions.pdf)
- [逐项 Glyph Provenance](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/reports/glyph-provenance.json) · [数学语义和计算核对](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/reports/semantic-verification.json)

## 样本与适配

两份 Session 原件各一页。核对 65 个手写区域，58 个可隔离裁剪覆盖 34 种样本标签，7 个上下文区域保留待用。
原始 PDF、300 DPI RGB 裁剪、相邻父区域、标签、逐页像素摘要和 PDF 点坐标一并归档。打印提示只提供定位，
符号取自手写作答。没有改写导入原件，也没有人工重描字形。
[原色裁剪、原标签与坐标清单](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/solutions/session-glyphs/sources.json)

- M01 按作答语境识别 ω。登记两处接近手写 `w` 的原始写法，实际输出选用 M-001 并保留 MathText 的 ω 语义。
  M02、M03、M04 提供向量箭头、帽号、±，M06 提供高方括号。已在分数括号和方括号矩阵中使用。
- N 的自由补充区提供 +、−、×、÷、=、<、>、≤、≥、→。第七页用明确计算式及不等式验收这些变体。
  另外展示希腊字母、绝对值、上横线、双点和 ∇。原四页沿用已有的个人加减笔形和正文。
- 测量 N 的 7 个独立根号，原始宽度约 75–355 像素，高度约 132–234 像素。
  两个相连样本分别覆盖短横和长横，按目标宽高独立适配到 MathText 的根号主体与横线槽位。
  其余多轮廓原迹保留供后续比较。本轮用相连原迹适配了 7 处，包含原回归页两个高根号、含分数和内层根号的外部根号，以及扩展页长根式。
  已有 B 的个人短根号延续使用。
  [根号实测与接缝位置](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/reports/radical-calibration.json)

## 完整页验收

原四页 761 项中，19 处外部字体和 2 处继承字体回退均替换，两个高根号的程序横线也改用会话笔迹。
11 个先前接合的程序根号横线仍有单独来源。原参数积分页已具备个人覆盖，B/C 正文逐像素相同。
七页扩展里，两个集合的左右花括号沿用基线字体。当前原迹左右手形过于接近，故保留原字形以保证可辨。

旧 B 的六页精确重放。原四页 526 个神经英文词、477 对词距观测及回退文本组沿用原始资产和坐标。
公式的 LaTeX、矩阵行列、上下标、绘制项顺序和局部像素范围通过核验。B/C 的 14 张逐页 PNG、
对照 PDF 的 14 页以及公式对照 PNG 通过独立重放。65 个原迹和父区域可从两份 PDF 重裁得相同像素。
[运行摘要](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/reports/result.json) · [重放回执](../workspace/runs/20261120T055114599868Z-v01025-session-mn-replay-verification-00058cbe/reports/result.json)

## 发布决定

Go 用于本地作业对照，正式发布仍保持实验状态。当前数据用于原迹轮廓和几何校准，CPU 渲染进程未加载 Torch。
C 的作者风格及更广题库验证另行记录。正式 `latest` 仍指向 `v06-five-final`，55 项保护摘要核验通过。

源码提交 `e1c8126`。实验保存 Profile 副本、配置、布局与英文种子、校准样本和源码。
可使用归档的 `solutions/session-selection.json` 运行 `python -m research.corpus.math_session_completion --selection <path>`。
[主运行](../workspace/runs/20261120T055031910274Z-v01025-session-mn-math-calibration-6dbf2ef4/run.json) · [复现运行](../workspace/runs/20261120T055031907292Z-v01025-session-mn-math-calibration-60287f4e/run.json)
