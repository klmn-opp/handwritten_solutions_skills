# v0.10.26 根号内距与数学笔宽修复

以上一版七页 C 为基线。D 只修改本轮反馈的数学项，三种具体操作为字形选换、局部几何拟合和根式组内移动。

| 反馈 | 修复结果 |
| --- | --- |
| 小根号与数字之间空白大 | 15 处单数字根式左右共同收紧，根号到数字的测量间距从 35.4–39.0 px 收至 5.52 px，约 0.47 毫米。数字保留原轮廓和纵向位置，横线保留与根号的搭接及数字右侧留边。 |
| 部分 0 偏重 | 22 处改选同一 Profile 中较轻的个人零字形，保留数学符号身份。 |
| 长括号、绝对值和高根号竖笔异常粗 | 限制拉伸后的主干笔宽，沿原来笔画中心收窄。实际修复 14 处长括号、两条绝对值线和 7 个高根号的左笔。绝对值竖笔的截面宽度第 90 百分位由 4.75 px 收至 2.53 px，高根号最粗的一处由 4.68 px 收至 2.56 px。 |
| 花括号退回印刷字体 | 七页中两对花括号恢复成 Session N 的左右原迹，来源为 N-035 和 N-036。另以含分数的高花括号检查全高度适配。 |

[七页 D PDF](../workspace/runs/20261120T064909731073Z-v01026-session-math-polish-b9371278/rendered/D-homework.pdf) · [完整 C/D PDF](../workspace/runs/20261120T064909731073Z-v01026-session-math-polish-b9371278/comparisons/C-D-homework.pdf) · [27 组公式局部](../workspace/runs/20261120T064909731073Z-v01026-session-math-polish-b9371278/comparisons/formula-regions.pdf) · [额外高花括号与绝对值对照](../workspace/runs/20261120T064955145918Z-v01026-session-math-polish-verification-35e9532e/comparisons/tall-braces-before-after.png)

## 验收

- 逐项比较公式源码、数学符号身份、绘制顺序和页面边界。修复后七页的字体回退记录为空，既有程序结构沿用各自的来源记录。
- 七页 C 精确重放。D 中 526 个英文神经词的墨迹与落位、477 处词距均保留，正文字区像素相同。15 个移动后的根号内数字对齐原位置比较，单像素最大灰度差为 3/255，差异位于 PDF 栅格化的抗锯齿边缘。
- 七页 C/D、对照 PDF 的 14 页和局部 PNG 均经第二次独立运行逐像素重放。56 项保护摘要核验通过，正式 `latest` 仍为 `v06-five-final`。D 作为本地实验保留。

[逐项来源与度量](../workspace/runs/20261120T064909731073Z-v01026-session-math-polish-b9371278/reports/glyph-provenance.json) · [根号间距数据](../workspace/runs/20261120T064955145918Z-v01026-session-math-polish-verification-35e9532e/reports/compact-roots.json) · [重放回执](../workspace/runs/20261120T064955145918Z-v01026-session-math-polish-verification-35e9532e/reports/result.json)

代码提交 `17808be`。运行记录保存源码、种子、Profile 副本以及旧 C 与 M/N 原件摘要。
