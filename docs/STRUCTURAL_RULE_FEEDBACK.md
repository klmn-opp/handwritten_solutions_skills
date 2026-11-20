# Structural horizontal ink — bounded author-feedback candidate

2026-11-20。作者指出分数横线及长根号横线过于像打印结构。本轮检查真实来源、布局接口和
全部实际横笔，生成有界候选；没有神经训练、原始笔迹编辑、正式 Profile 更新或 latest 变更。

最终已通过根 Agent 隔离视觉审核的运行：
[20261120T121043171023Z-structural-rule-provenance-correction-657bc90f](../workspace/runs/20261120T121043171023Z-structural-rule-provenance-correction-657bc90f/run.json)。
[完整二十页 PDF](../workspace/runs/20261120T121043171023Z-structural-rule-provenance-correction-657bc90f/rendered/homework_solutions.pdf) ·
[全部 150 横笔紧凑对照 PDF](../workspace/runs/20261120T121043171023Z-structural-rule-provenance-correction-657bc90f/comparisons/compact-baseline/all-rule-contacts.pdf) ·
[完整根式局部对照](../workspace/runs/20261120T121043171023Z-structural-rule-provenance-correction-657bc90f/comparisons/full-roots) ·
[单独暴露的结构压力页](../workspace/runs/20261120T121043171023Z-structural-rule-provenance-correction-657bc90f/rendered/exposed-stress.pdf)。

基线为 `20261120T115012607677Z-author-feedback-final-20-pages-50910a3a`，通过根 Agent 提供的
`research.author_feedback_round2.replay_page` 验证每页像素、来源及 parser trace 完全重放，
然后才开始增量处理。源码仅新增 `research/structural_rule_feedback.py`，未修改旧 Resolver、
Session polish、MathText 或版式器。封存采集册完全不动。

## 已确认根因

基线二十页共有 **129 条分数线、21 条根号横笔**。其中 19 条根号横笔最终是程序矩形，
2 条仍来自 M/N 根号轮廓。问题发生在结构笔墨映射，并非缺少原始扫描或训练不足。

- `research/corpus/math_glyph_resolver.py:86` 的 `rule_outline` 将真实中心线高差压到
  parser 矩形高度的 16%，厚度压成 `height * (0.56 + 0.24 * thickness / max)`。
  原始粗细比例和起收笔被收窄，即便来源标为个人包络，也容易呈现几乎等厚直线。
- `research/corpus/readability.py:147` 的 `join_radicals` 以五顶点、固定厚度矩形创建 roof。
  `resolve_radicals` 接受该结构后，原本横笔的起伏、倾斜与终端不会自动恢复。
- 现有 Session 根式拟合将源 roof 的整体竖向范围压入 parser 横线窄槽；后续根号净空修正
  调整 x 范围，通常保留这条已经压窄的横笔。不是再增加随机噪声就能解决的问题。

旧 v3 rule 包络来自已使用的 `c35e…` 历史 heldout-mean-variance 素材。本轮只检查已保存的
Profile 数值以定位压缩原因，没有打开该旧 heldout 原图选样或调参，也未读取 sealed C。

## 新候选的真实依据与角色边界

仅使用已审核 Session N 的六个来源，原 PDF SHA：
`d455f36021e9a0e4c7d9f1c319dd091c3b1f9da04edd4cbe28412e1a52f8f6f8`。
原 PDF、300 DPI RGB 裁剪、父区域、原标签、采集行、crop→PDF 变换及来源 SHA 均保存于
`solutions/structural-sources`，另有 `measurement-manifest.json`、选中轮廓 NPZ 和源图对照。

| 来源 | 实际原标签 | 允许用途 |
| --- | --- | --- |
| N-031 | overline，上划线 | 完整起收笔用于分数结构横笔 |
| N-042、N-043 | minus，减号 | 完整起收笔用于较短分数结构横笔 |
| N-063、N-064、N-065 | radical，根号 | 仅选已记录 seam 右侧真实横笔区域用于 roof |

没有把减号重标为作者写下的分数线，也没有把从根号选择的 roof 宣称为新的独立字符真值。
这是明确的结构角色转用。分数线只可使用完整水平原迹，不能把根号 seam 切口冒充自由起笔。
六份来源不是六个新增笔迹样本，更不是新训练数据增长。

原轮廓作稠密取样后，分开测量每个 x 的真实中心位置和笔厚。按当前笔宽归一化后的长度
匹配选择来源，没有随机种子搜索、噪声或逐页挑选最好看的结果。端点和粗细来自真实轮廓，
目标 x 左右范围不变；中心线按长度缩放，笔厚按当前结构笔宽缩放。如果空白不足，只能
在预先计算的安全包络内限制斜率，或明确保持原输出。没有移动分子、分母、被开方内容。

来源标记准确区分三个概念：

- `personal_collected=True`：使用的来源是已审核真人笔迹。
- `measured_source=True`：保存源轮廓、中心/笔厚测量及映射依据。
- `procedural_geometry=True`：最终结构经过中心/笔厚分离映射、尺度限制，部分 roof 还与
  原接缝片段组合，**不是未经变换的原迹**。

因此统一使用 `source_type="personal-scan-structural-rule"`，每条另附
`structural_rule_feedback`，不因存在真实源就隐藏程序几何。

## 安全边界与实际结果

`allowed_bounds_px` 在变更前由所有原邻居测量，并限制在原中心上下最多 0.18 em 范围。
邻近分子、分母及其他符号进一步缩小该范围；它不是把生成后的 bbox 再当安全证明。
每个 roof 最左至多 1 px 原接缝几何保留，复核原 stem 接触不丢失。所有规则还检查新的
邻居墨迹接触；嵌套结构不会因为先替换一条线而静默放大另一条的空间。

| 集合 | 总横笔 | 应用 | held |
| --- | ---: | ---: | ---: |
| 原二十页 | 150 | 143 = 129 分数线 + 14 roof | 7 条极短 roof |
| 独立暴露的短/长/高/嵌套结构压力页 | 17 | 16 | 1 条极短 roof |

原页 held 条目为 P3:375、P16:89、P17:168/259/260/277/338。其 x 范围只有约 4–6 px，
宽度小于当前笔厚的 2.5 倍，无法安全展示自然端点而不变成墨团，因此保留原形并记录
`held-short-range`。没有为了“全部修复”挤动内容或让 held 项伪装成个人轮廓。

原二十页选源分布：N-031 59 次、N-043 51 次、N-042 19 次、N-064 12 次、N-063 2 次。
N-065 在压力页使用一次。此为有限真实来源的确定性几何匹配，不是学习到的新随机变化。

实际检查：原二十页均严格重放；所有非横笔 item、轮廓、坐标、英文像素及 parser/identity
保持；安全包络外改变像素为 0；没有新增非配对笔墨接触，原 root 接缝接触均保留，无页边界
裁切。Parser trace 相同只证明增量未改结构记录，不替代独立解答正确性验证。

完整 150 组 A/B 可在五张 `comparisons/compact-baseline/contact-01..05.png` 检查。
根号原生局部包括第 8/12/14/17 页全部目标，完整 stem、roof 与被开方内容同框；不是只截
横线而隐藏其接缝。压力页独立标为 exposed diagnostic，不当作新 Holdout 成功。

## 保留的初版与来源披露修订

1. `20261120T120543820623Z-structural-rule-source-envelope-candidate-87a0bbd8`：工程通过，
   但自审发现选源允许 seam 截断的 roof 去当分数线，其起笔不是真正自由端点。该来源选择
   不接受，后续将分数线限定为完整 overline/minus。旧 run 与证据保留。
2. `20261120T120823655982Z-structural-rule-source-envelope-candidate-1c421622`：修正上述
   角色限制并补齐紧凑/完整根式对照。根 Agent 审查又指出 `procedural_geometry=False`
   披露不充分；动态结构应同时标记真实来源与程序几何。
3. 当前 `657bc90f` 是独立 metadata 修订：完整 PDF、PNG、轮廓 NPZ、字体全部逐字节继承
   `1c421622`，基线记录及全部映射数值保持，仅补正程序几何/测量来源字段和源码快照。
   详见 `reports/metadata-correction.json`。没有重复渲染来伪造新视觉收益。

## 准入状态和集成入口

当前 **Go：来源、工程门禁及根 Agent 视觉审核通过，获准进入隔离组合**。
根 Agent 已看全部五张 150 横笔对照、压力页 17 横笔、N031/N063 原 RGB→测量证据，以及
第 8/12 页长根式和多个短根式，未见新结构或边界缺陷，认为横笔变化比旧等厚线更协调。
这不代表作者个人风格认可、生产准入或神经模型训练完成。随后已完成
[第二轮完整组合及主/独立整页检查](M1_SECOND_AUTHOR_REVISION.md)，143 横笔接入、
14 个原 root 接触保留，未发现新增非配对接触。作者个人风格复核仍待进行。

冻结记录：run SHA `13c581128741b45e487d07ffcffbc973d57c1fc4ad2091824c2f39abcf1585fe`；
源码 SHA `aa817ae199889ec94ea445a5608c0438dde9c0aab09ad872d76c2a927f629560`；
来源清单 SHA `ec32bfbf7984162fd1dc91954097cfe1f83311e29e6db126c6f492d8fd12773e`。
批准后不再改动该映射或封存产物。

```python
from research.structural_rule_feedback import StructuralRuleFeedback

treatment = StructuralRuleFeedback(candidate_run / "solutions/structural-sources")
records = treatment.apply(page, renderer, records)
# treatment.decisions contains every applied or held horizontal item.
```

实际执行：

```sh
.venv/bin/ruff check research/structural_rule_feedback.py
MPLCONFIGDIR=/tmp/hwsol-rule-mpl .venv/bin/python -m research.structural_rule_feedback
MPLCONFIGDIR=/tmp/hwsol-rule-mpl .venv/bin/python -m research.structural_rule_feedback --metadata-correction workspace/runs/20261120T120823655982Z-structural-rule-source-envelope-candidate-1c421622
```

源码 Ruff 与最终 run 完整性验证通过。最终组合时仍应独立复核其他新处理与本模块叠加后的
接触、字符像素和完整页面；本轮隔离通过不能替代组合后的验证。
