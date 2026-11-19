# v0.10.5 数据版本与来源

新完整版本：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T082902471963Z-continuous-corpus-extension-f9e7198a`。338区域/106页/9文档，C未读未进入数据。
训练使用的英语版本和高分辨率视图见主run `/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T081423714624Z-new-continuous-corpus-v0105-970c87ac` 下 `reports/data-runs.json` 及最终参考视图 `/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T084542183420Z-context-word-aligned-views-71f26275`。
原件只读副本、300DPI原色图、纸张背景和墨迹图在主run的 `logs/originals/` 与 `logs/imported_corpus/`。
旧正式Corpus数据库未改动。

新准入：20六词英文行、3英文标题、39数学区域（41式）。模型标注与人工确认严格区分：
`review_status=model-accepted`、`model=gpt-6.2-sol`、`reasoning_effort=xhigh`、`human_review=false`；用户授权直接使用高置信模型清洗，未经人工复核不改称human-verified。
所有样本可通过source_hash/page_number/bbox/crop_sha256反查原PDF，重渲染和逐像素验证已通过。
标签是实际书写，不自动纠正到印刷参考。数学正确性另行判断；print/background/blank均不计真实训练。

A/B整词边界保存在 `word_boxes_source_crop` 与 `word_box_labels`，相对于原始行裁剪，120整词保持连笔、点和下伸。
只生成训练派生视图，不改原像素。词内与词间不是字符切割。视图记录 `source_box_coordinate_system`；
旧兼容字段名 `source_box_in_whole_view` 的数值在审核框路径实际为original-crop坐标，必须以显式coordinate_system为准。
训练对齐把完整词等比放入Handnote参考槽，词间节奏没有学习。

C只保存摘要、只读原件和页数；没有渲染、OCR、模型查看或训练。
以文档SHA分组而非裁剪随机划分。A/B/C是否不同实际书写时间未核实，不夸大为独立采集会话。
数学两处不确定候选保留rejected与原图；任何模型生成图都不回填为真实样本。
数学对照预览路径：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T082902552123Z-paired-math-transcription-evidence-c23fedfc/comparisons/math-review/index.html`。
