# M1 第二轮作者修订：2、结构横笔与 f

2026-11-20。作者确认首轮修订明显改善，但指出部分 2/π 偏粗、分数和长根号横笔像打印线、
f 偏小偏矮偏淡。本轮完成真实来源审计、有界候选、主/独立 Agent 视觉复核及完整 20 页组合。
**Go：交给作者对照；No-Go：生产发布、宣称全部问题解决或新神经模型提升。**
交付后作者认可改善和 f 的形状，但指出所有 f 趋同、π 仍偏粗，续见
[第三轮工作](M1_THIRD_AUTHOR_REVISION.md)。其前轮反馈中的“分号使用的横”按上下文
解释为分数线，并已告知作者。

## 当前可检查交付

最终运行 `20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6`：

- [完整 20 页 PDF](../workspace/runs/20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6/rendered/homework_solutions.pdf)
- [逐页 PNG / PDF](../workspace/runs/20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6/rendered/pages)
- [24 组重点局部对照](../workspace/runs/20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6/comparisons/focused/formula-regions.pdf)
- [全部 173 组公式对照](../workspace/runs/20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6/comparisons/all-formulas/formula-regions.pdf)
- [前版完整 PDF](../workspace/runs/20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6/comparisons/previous-reviewed20.pdf)及[逐页并排 PNG](../workspace/runs/20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6/comparisons/pages)
- [字体回退位置标注 PDF](../workspace/runs/20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6/comparisons/fallback-overlay/math-fallback-overlay.pdf)
- [逐项来源、变更、拒绝原因和边界报告](../workspace/runs/20261120T121552890180Z-author-feedback-round2-complete-20-956bbad6/reports)

本轮沿用首轮最终 `50910a3a` 的全部题目、答案、布局、背景、Profile 和 seed 1909。
每页先要求像素、来源、parser trace 精确重放，再依次应用 2、f、横笔。全部修改限定在
350 个既有 item；没有新增或删除文字、数学项目、公式或页面。20 页英文使用 CPU 字形，
不是新的神经英文生成；旧 v21/v23 冻结神经英文页保持不变。

## 根因、实际修复和保留范围

| 项目 | 查明的原因 | 本轮结果 |
| --- | --- | --- |
| 2 | 真实 v2/v3 原稿高度较小，归一化到统一字体高度时相对笔厚被放大 | 203 处 2 中，92 处换用已核验的真实 v0；111 处原本较轻的 v0/v1 保持 |
| π | 20 处均来自同一已核验历史原迹，其横笔和两腿本来就较重；实际横纵缩放比仅约 1.034–1.049 | 20 处原样保留；未找到可安全替代的已核验轻笔来源，不擅自擦细原笔迹 |
| 分数横笔 | 旧代码将真实中心线与粗细变化压入很窄的近等厚包络 | 129/129 处按 Session N 真实 overline/minus 的中心线、笔厚及完整端点重新映射 |
| 根号横笔 | 21 处中 19 处被旧接缝流程改为固定厚度矩形，另 2 处源笔迹也受窄槽压缩 | 14 处使用真实根号 roof，保留原 stem 接缝；7 处极短横笔明确保留 |
| f | 字库编译独立拟合每个字形高度，压缩了原 f 相对 h/o 的长杆、下伸和笔厚；横臂位置也被抬高 | 从同一来源 h/l/d/a/c/e/o 的共同坐标恢复 f，115/145 处接受，30 处保留 |

数字替换保持旧排版槽、方向、剪切与邻字位置。7 处旧槽较宽，较轻原迹居中后每侧留白
增加约 7.64–8.49 px，已逐处目视；没有为了填满旧槽而再次横向拉粗。
原始采集图、配准变换、原字体和来源记录随父运行复制到最终 run 的 `solutions/evidence`。
π 来源为历史物理作业第 55 页 `U03C0-p55`，原裁剪/父区域/坐标完整保留。

横笔来自 N-031、N-042、N-043、N-063、N-064、N-065 的已审核原迹。
分数线只用有完整起收笔的 overline/minus；根号 roof 只用于根号，不把切下的接缝当自由起笔。
中心线与笔厚分别按结构范围映射，受分子/分母/邻居净空约束；这属于
`personal-scan-structural-rule`，同时记录 `measured_source=true` 和
`procedural_geometry=true`，不冒称未经变换的原笔迹。详见[横笔专项](STRUCTURAL_RULE_FEEDBACK.md)。

f 的共同来源拟合给出统一缩放 1.444028 倍，顶端接近 h、恢复下伸；没有膨胀笔画或新增墨迹。
拟合使用 14 个真实参考端点，残差约 19.80 字体单位，不能视为精确个人书写真值。
准入预算预先限定最大尺度、上下左右扩展、至少 0.60 px 新邻字净距和 3 px 页边界距离，
另查上下行及同词 ff。最终最紧接受净距约 0.654 px。未移动邻字以强行接纳放大。

初版 D 有 123 处可单字通过，但出现 8 组“一大一小”的 ff，主/独立审核拒绝集成。
最终从实际 Markdown 的文本顺序识别 14 组真实同词 ff：6 组共同通过，8 组整组保留。
跨词的 `of factors` 等不误绑。30 处保留由 22 处原安全拒绝及 8 处组内共同保留组成，
包括 3 处数学 f 和 27 处正文/标题 f。第 19 页部分 first/final 仍小，属于明确未修复范围。
此轮不再扩大参数搜索；后续需要可信上下文/布局策略，不能只继续加粗放大。

## 数学来源与验收边界

2,265 个数学可见项目的来源分布如下；这是此 20 页的覆盖，不是任意公式覆盖。

| 来源 | 数量 |
| --- | ---: |
| personal-outline | 1,835 |
| personal-outline-stretched | 71 |
| personal-session-outline | 98 |
| personal-outline-geometric-calibration | 9 |
| personal-scan-structural-rule | 143 |
| personal-component-composition | 6 |
| personal-scan-outline | 66 |
| procedural-geometry | 7 |
| external-font | 29 |
| inherited-font | 1 |

字体回退仍为 **30**：κ×6、Φ×5、ξ×4、η×3、⟨×3、⟩×3、∏×2、≈×1、ζ×1、ψ×1、∮×1。
≈ 为继承字体，其余为外部字体；具体源字体与缺少可用个人映射的原因在
`fallback-reasons.json`。π 是真实原迹偏粗，不应误报成字体回退。
原作者不满意的四/七页以外符号缺口保持可见，本轮没有凭空增加新的真人样本。

组合验收检查了实际绘制轮廓、项目身份/顺序/数量、原 parser trace、页面尺寸及差分像素。
结果：非目标轮廓和落位保持；目标范围之外变动像素为 0；新增非配对接触为 0；
14 条修改 roof 的原 stem 接触全部保留。新门禁还用断开接缝的反例确认会拒绝该错误。
这证明增量结构保持，不代替解答重新独立验算，也不将旧 7/8 留出结果改写为新成功。

封存后的全字段审计覆盖 10,774 个绘制记录：10,387 个完全相同，350 个实际变更，
另外 37 个仅追加明确拒绝记录（30 f、7 roof），所有旧字段仍精确相同。
非目标来源记录差异为 0；π 的 20 个记录完全相同。
[审计脚本与结果](../workspace/analysis/20261120-round2-root-review/post-assembly-record-audit.json)。

主 Agent 已看所有 92 个数字变化、145 个 f、150 个原横笔及 17 个额外结构诊断，
最终组合看第 1–10 页及 12/14/17/20 页。独立 Agent 看最终 11–20 页及 10 组完整公式局部，
并从最终合集 PDF 以 300 DPI 重栅格，后 10 页均与逐页 PNG 像素一致。
未发现新增漏项、碰触、裁切、根号断接或混合尺度 ff。
第 17 页高根号主干相对横笔仍偏重，保留为风格限制，不以接缝检查代替作者判断。
[独立整页审核](../workspace/analysis/20261120-round2-final-independent-pages11-20/README.md)；
[主审核与保护记录](../workspace/analysis/20261120-round2-root-review)。

## 可复现记录与失败证据

来源提交：`55bacf3` 数字、`4250f2e` 横笔、`d9391dc` f；基线重放工具 `47c1330`。
本轮新组合工具为 `research/author_feedback_round2_delivery.py`，门禁为
`research/author_feedback_round2_checks.py`。六个实际文件 SHA、完整研究源码快照、
全仓库状态、父运行清单、配置、Profile 和真实来源均保存在 run。
`run.build.working_tree_dirty` 只描述其追踪的核心构建；完整实验源码身份应以
`reports/approval-receipt.json` 和 `logs/source` 为准，不能仅凭该布尔值声称整仓库干净。

三个获准父候选分别是数字 `95ce7292`、f `3728598e`、横笔 `657bc90f`，均为不可变的 held run。
主 Agent 先封存精确 SHA 准入回执，然后合成20页。保留数字初次 bbox 空白区误报警、
f 序列化/导入失败及混合 ff 候选、横笔初版错误角色选择及后续程序几何披露修订，
详见各父 run 报告；没有删除失败样例，也没有把元数据修正称为新的视觉改进。

环境为 Linux x86_64、Python 3.12.16、Handnote 2.0.0.dev0，32 个可见 CPU；本轮 CPU 无 Torch。
组合用时约 71 秒，seed 1909；原源文件和旧候选哈希、latest、锁文件及主计划原文复验保持。
本阶段没有神经参数更新、优化器或新增 checkpoint。

实际运行和检查：

```sh
MPLCONFIGDIR=/tmp/hwsol-round2-delivery-mpl .venv/bin/python -m research.author_feedback_round2_delivery --approval workspace/analysis/20261120-round2-root-review/integration-approval.json
.venv/bin/python workspace/analysis/20261120-round2-root-review/audit_records.py
.venv/bin/python -m research.project_state verify
.venv/bin/ruff check research/author_feedback_round2_checks.py research/author_feedback_round2_delivery.py research/f_size_feedback.py research/numeral_pi_feedback.py research/structural_rule_feedback.py research/comprehensive_math_collection.py
```

最终 run JSON SHA：`c6a2fbc94e384335271262828de7c7e1e1a329932527b42a2cf5b0e963e3323f`。
PDF SHA：`b6145cd3d115afde957b5dff85accc6d9fe847540b4be2a9993ed540b423e0f6`。
run 为 held，封存后视觉审核单列于 analysis，未反写产物清单。

## 下一步与作者采集

[完整数学采集册](COMPREHENSIVE_MATH_COLLECTION.md)已按最新偏好准备为 19 页、123 行，
包括全部 Greek/Latin、数字、运算/关系/集合/逻辑、积分求和、根式括号、上下标、向量、
分数、矩阵和常见物理结构。可直接按惯常笔迹填写，无需模仿打印字形，也不再缩减清单。
原色扫描存为 `source_pdf/session-O_full_source.pdf`；提示标签只有在审核实际笔迹后才可采信。

本阶段到达作者风格复核和新真实样本输入点。后续优先审核新 π 和缺项，再验证复杂公式、
保留 f 与高根号的局部风格；不将此时的字体工程改称神经训练。M2 的个人数学正式候选集成、
macOS 实机和完整发布门禁仍欠缺；M4 的当前混杂 residual 假设仍 No-Go。正式 main/latest、
默认 Profile、原始扫描/标签、历史权重与封存 C 均保持。
