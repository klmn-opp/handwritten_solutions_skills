# v0.10.3 笔画浓度实验：加深有效，去灰影未通过

用户授权继续训练，分支 `experiment/context-stroke-opacity-v0103`。本轮从作者偏好的旧mass出发，
完成三臂各1000次CUDA参数更新。**新模型笔画更深，但灰影未减少，最强保护组粘连增多；
不替换mass研究起点，不发布默认模型，不进入扩大训练或RL。**

## 直接验收

主run：`workspace/runs/20261119T061055965963Z-context-stroke-opacity-v0103-08869a27/`。

- `rendered/homework_solutions.pdf`：10页A4速览。每页依次为本人原图、原Handnote、旧mass、旧distance、三组新训练。
  最后三行均为本轮真实训练结果；最后一行不是选出的最佳模型。
- `comparisons/all-clean-ink-comparisons.pdf`：39页完整对照（8 Train、25 Val、6新区域）。
- `rendered/pages/comparison-072.png`：同一句“the sample maximum.”的直接对照。
- `comparisons/author-review/index.html`：4组20图，位置均衡、模型名隐藏，可本地继续/导出；先核内容再评风格。
  本轮没有作者新评分；不需要重抄字母表。可选填写“变深是否值得增加的灰影/粘连”，JSON仍导出到`source_pdf/review_exports/`。
- `comparisons/training-curves.png`、`reports/training-runs.json`：曲线和真实权重路径。

这些是同内容研究对照PDF，不是新解题成果；新网络仍只生成英文完整区域，数学继续由既有Handnote/MathText承担。
新网络是整块确定性卷积生成，不是逐字选择，但没有随机风格潜变量、自由连笔轨迹或新数学生成能力。

## 实际训练与复现

模型仍为`aligned-content-ink/1`，144709参数、64×512墨迹场；不是继续训练旧孤字CVAE。
起点为v0102 mass `checkpoint-1500.pt`，数据未扩充：171准入真实Train、25 Val，每批3真实+1字体辅助，
seed1303、lr5e-5、新AdamW、固定1000步，不按新留出换checkpoint。

三臂保留旧SDF/BCE/Dice、距离自一致、墨量及冻结TrOCR内容损失；新增监督见[V0103_PROTOCOL](V0103_PROTOCOL.md)。
core对真实前景和远背景分别约束，前景允许1像素位置容差，背景保留3像素不确定带；
stroke再为小组件加权，点和下伸不腐蚀，不将组件当字符分割或修补真实墨迹。

|臂|目标|更新|实际秒数|峰值张量显存 GiB|参数绝对变化和|
|---|---|---:|---:|---:|---:|
|control|原mass目标续训|1000|110.79|2.58|208.21|
|core|前景/背景浓度|1000|111.52|2.74|227.47|
|stroke|core+组件保留|1000|111.90|2.74|251.01|

RTX5060 Laptop 8151MiB、driver580.126.09、Torch2.7.1+cu128；隔离`english-htr-v08`环境。
初始参数、500/1000步权重、优化器、RNG、源码快照、教师/数据/协议摘要、依赖和日志全部保存。
三臂冻结教师参数均未改变。训练实现提交`914d3df`，报告扩展`304df2a`；训练关键源码15项与快照一致。
全部候选冻结后才打开新留出，之后未重新训练。三个新模型的663张PNG同设备重放逐字节一致。
旧mass/distance各215张开发集输出也与旧封存图逐字节一致。

## 质量维度分开报告

25 Val，墨色指标在自身>.5核心计算，越高代表更深，不代表字符或风格正确。
灰墨指标为距自身核心>1.5像素且墨量>.1的质量和；越低不必然更好，删细笔也可能降低它。

|方法|核心墨色|外围灰墨|总墨量|Tesseract裁边CER|TrOCR裁边CER（训练教师）|
|---|---:|---:|---:|---:|---:|
|handnote|0.800|13.6|2071.6|13.98%|1.87%|
|context-mass|0.658|294.0|2076.6|16.86%|3.17%|
|context-distance|0.723|261.9|2523.2|19.74%|2.45%|
|context-control|0.685|290.9|2225.5|20.46%|4.47%|
|context-core|0.703|300.0|2356.8|25.07%|5.48%|
|context-stroke|0.740|302.1|2673.8|25.65%|5.33%|

stroke核心浓度.658→.740，但外围灰墨294.0→302.1（约+2.7%），不是去灰影进步。
CER/WER、替换/遗漏/插入、尾部遗漏等均保存于机器报告；它们是识别器观察，不是经作者核验的真实文本错误率。
固定L<230包围框+8px白边，保留原始和裁边两套输入/观察，共434图、868识别结果。
短词原始大白边使TrOCR全部方法在6新区域上Exact=0；裁边后Handnote5/6、旧mass4/6、core/stroke各3/6。
真实扫描裁边也只有1/6，说明识别器本身仍有明显局限；不能按教师分数发布模型。

Sol匿名内容观察：4 Val+6新区域×5方法=50图。未提供目标文本或模型名；同内容不同候选仍可能相互提示。
“完整且完全匹配”包含大小写和标点；不确定不算成功，以下不是人工真实性评分。

|方法|完整且匹配/10|部分辨读|灰影|重影|粘连|偏淡|
|---|---:|---:|---:|---:|---:|---:|
|handnote|8|1|0|0|0|0|
|context-mass|8|1|9|3|1|6|
|context-control|9|0|9|2|1|6|
|context-core|9|0|9|8|2|6|
|context-stroke|8|1|10|10|7|1|

共同未转出一条Val的末尾逗号，不能据此断言所有渲染器都漏标点；旧mass和stroke的`zero`出现r/y歧义。
Handnote的`zero`也被Sol判断边缘受损；原PDF裁图无墨迹触边，规范化漏掉的>.1淡墨仅1像素/总墨量0.11%，
不足以证实真实PDF裁切，保留为模型观察，不能借此宣称神经模型超过Handnote。边界审计与原图齐备。
所有1105生成PNG均非空、无>.5核心触边；这不是内容正确性的充分证据。

- 内容：大部分盲图可辨，仍有字形/标点疑问，尚未证明比mass更可靠。
- 浓度/细笔：变深真实可测；stroke偏淡观察减少，但粘连7/10、重影10/10，代价明显。
- 灰影：失败，未通过预定目标。
- 作者相似性/自然度：本轮尚无新作者评分，不能沿用上轮评分证明新模型更好。
- 页面/数学：只完成同内容A4对照，未训练新数学或改变整页布局策略。
- 工程：234自动测试通过、重放/15源码检查/72保护文件/14原件/原Corpus均通过。

## 数据与原因判断

本轮未新增训练样本。171准入Train只有8条>=5词（早期原始190条口径有14条），长句监督偏少。
165唯一字串；6个重复字串共12样本，同字串两次真实笔迹的二值IoU均值约.092。
这是相同确定性输入对应不同笔迹的实际冲突，但仅占少部分样本，**不能作为全部灰影的唯一解释**。
训练目标仍只按完整词槽对齐，词内笔画位置并不精确一致；64像素高的灰度重采样与多种密集像素损失
可能鼓励模糊折中。属于待消融机制假设，不等于转写标签不准确，也不能只归因于数据量少。

Sol在38个剩余页只选出6条高置信新区域（4词、2双词），避开涂改/遮挡，仍是旧phy1文档。
入选页曾在历史缩略总览出现，1条字串与旧区域重复但来源不同；无源页/crop重复。
`fresh_document=false`，此次只是探索性区域测试；现在已曝光，后续不能再称新独立测试。

## Go / No-Go 与下一步

Go：保留整块生成能力、可靠转写管线和真实训练证据，研究把墨迹形状与浓度分开表达。
No-Go：发布core/stroke、宣称去影成功、继续加大当前损失或只加步数、扩大长句训练、数学替换、RL。
旧mass仍作为研究参考，正式默认仍是font-cpu/personal@v1。

下一有限试验先用现有可靠短词做16–32例学习能力诊断，检查原图高分辨率与词内软对齐，
评估轮廓支撑与透明度分开的表示及窄带过渡；保护点/细笔，不能简单二值化吞掉淡墨。
在开发集同时通过去灰与内容门禁后，再封存新文档做最终测试，不能无限重复旧6区域。
若要提供新测试资料，建议2–3份从未用于研发的完整作业PDF（合计5–10页，原扫描/约300DPI，
含清楚英文解释与数学混排），保持原样，不需抄字母表；新增文档先封存不训练。暂不要求立即补材料。

## 精确输出与状态

- control权重/优化器：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T061350656982Z-context-stroke-opacity-training-cf40aa73/logs/checkpoint-1000.pt`。
- core权重/优化器：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T061548522301Z-context-stroke-opacity-training-f2bbf086/logs/checkpoint-1000.pt`。
- stroke权重/优化器：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T061747401310Z-context-stroke-opacity-training-71018e2e/logs/checkpoint-1000.pt`。
- freeze：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T061949810220Z-context-content-freeze-0b62d860`。
- evaluation：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T061951169433Z-context-content-evaluation-dca960fe`。
- comparison：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T062000302201Z-context-same-text-comparisons-6f4bf6bf`。
- recognition_views：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T062041195654Z-context-opacity-recognition-views-6f7dfafd`。
- recognition：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T062042294036Z-context-recognition-observations-bd22eec7`。

主run的`reports/run-index.json`链接全部11子run；`reports/protection-check.json`、`all-generation-verification.json`和`review-verification.json`记录保护与校验。
