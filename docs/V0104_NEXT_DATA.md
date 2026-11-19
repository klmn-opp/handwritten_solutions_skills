# 下一步需要补充的真实英文材料

本轮自主完成A/B/C三轮共10400次CUDA更新，全部新候选仍未通过内容与自然度门禁。
不是“只要多训就会变好”；对齐、输出表示、教师偏差和数据覆盖仍需改进。
目前需要新连续英文材料和从未使用的文档测试，避免继续只在已曝光的25条Val上试方案。

## 最少需要做什么

优先提供已有的真实英文解题叙述：建议3份独立书写/采集的PDF，每份约10条清楚完整英文行。
长短可自然变化，包含说明、推导衔接、数字/标点；不要只有公式、打印题面或零散词。
其中2份补训练，1份标记为“仅测试”，保持完整文档隔离。
这不是声称原7份PDF已被逐页穷尽：本次新增审查89未用页仅收12短语，没有新>=5词行。

若手边没有这样的作业，可使用下列可选模板，不用重抄字母表：

- [A：10句训练材料](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T070809304768Z-english-continuous-collection-request-87725bae/rendered/session-A.pdf)
- [B：10句训练材料](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T070809304768Z-english-continuous-collection-request-87725bae/rendered/session-B.pdf)
- [C：10句独立测试材料](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T070809304768Z-english-continuous-collection-request-87725bae/rendered/session-C.pdf)

共30行、每句6词。6词适配当前保守的完整词组视图；更长原始作业也可保存，但需之后采用适合长行的视图。
请自然书写，每句一次，用平常的笔，不描字体、不刻意统一大小。可直接写模板下方，也可写在空白纸并标A/B/C和题号。
三份在独立书写/采集会话完成并分别保存，不合并成一个PDF，不把同页裁剪分进Train与Holdout。
扫描约300DPI或清晰正拍，保留页面边缘和原颜色；发生涂改可另起一行重写，原痕迹保留。
保存到仓库`source_pdf/`，例如`english_session_A.pdf`、`english_session_B.pdf`、`english_holdout_C.pdf`。

模板上的文字只是预期提示，不会自动成为真实转写；实际写下的内容仍由保守核验确认，不确定就弃用。
30行是下一小试的补充建议，不是足以训练成熟英文生成器的保证。
收到后先导入/核验和文档级划分，封存C；再改进对齐与生成表示，候选冻结后才开C。

本轮失败候选已由模型盲检，无需你再给这些失败图打分。作者风格选择会等到内容门禁通过之后再做。
默认font-cpu、personal@v1和latest不变。
