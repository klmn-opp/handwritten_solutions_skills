# v0.10.4 自主连续训练报告：未达标，进入需要新真实材料的节点

分支`experiment/context-levelset-v0104`。按用户授权实际完成A/B/C三轮、6组实验、10400次CUDA参数更新，
其中A轮2组是小样本拟合诊断。**新候选全部No-Go，未达到内容可靠且自然协调的目标。**
保留作者偏好的旧mass研究基线，正式font-cpu/personal@v1/latest不变。

不是把识别、投影或模板制作称作训练：每组保留初始与定期模型/优化器/RNG、参数变化、源码/依赖/数据/教师摘要。
轮廓投影本身无可学习参数，其零更新结果单独展示，不能算神经学习收益。

## 可直接看到的证据

- A轮24页有限拟合对照：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T063806723010Z-context-levelset-v0104-94ac18e7/comparisons/capacity-diagnostic.pdf`。
- B轮6页速览/33页全对照：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T063806723010Z-context-levelset-v0104-94ac18e7/rendered/homework_solutions.pdf`、`comparisons/all-clean-ink-comparisons.pdf`。
- C轮6页速览：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T070111966640Z-context-levelset-ablation-delivery-bfaefdbb/rendered/homework_solutions.pdf`；同目录逐页PNG，完整33页在`comparisons/`。
  每页本人原图、原Handnote、旧mass、旧mass仅投影、B混合目标、C真实目标加教师、C真实目标无教师。
  最后两行是最后一轮新训练，明显失真仍如实保留，绝不作为最佳模型发布。
- B/C各保留4组20图匿名作者页，但目前不要求用户评失败图；[下一步材料清单](V0104_NEXT_DATA.md)才是需要的手动工作。

## 三轮学习与决策

|轮次/臂|数据/约束|更新|秒数|峰值张量显存GiB|参数变化和|最终Train Dice|
|---|---|---:|---:|---:|---:|---:|
|A/soft|24真实，有限记忆诊断|1200|8.83|0.16|713.80|0.814|
|A/projected|24真实，有限记忆诊断|1200|10.39|0.16|722.38|0.805|
|B/soft|171真实/25%字体辅助+冻结教师|2000|220.47|2.56|483.37|0.289|
|B/projected|171真实/25%字体辅助+冻结教师|2000|224.88|2.56|443.82|0.257|
|C/real-content|171真实，去字体输出目标|2000|224.44|2.56|430.42|0.286|
|C/real-shape|171真实，去字体输出目标|2000|19.08|0.25|970.32|0.513|

A轮从旧mass开始，同24真实样本，soft/projected各1200。目标为扫描轮廓距离栅格，Dice从.099到.814/.805，
达到预先>=.80且增加>=.20、无空白的容量门槛。它证明可记住有限样本，不证明新字串泛化。
B轮重新从旧mass开始，不使用A轮权重；171 Train、25 Val、3真实1字体，双视图冻结TrOCR内容约束，固定2000步。
soft仍灰影/粘连；projected清掉灰背景却保留黑双线和字形损坏，内容门禁未过。
C轮同起点/数据/seed/2000步，去掉随机字体输出目标，再分别保留/去掉内容教师。
去教师提高Train形状拟合，Val未改善，完整文字严重失真。不能以训练图更像真迹或去灰成功替代内容正确。

GPU RTX5060 Laptop约8GiB、Torch2.7.1+cu128；CPU Runtime未装Torch。
B/C教师均冻结检查通过；A和C real-shape无教师。比较训练步数相同的臂，计算量不完全相同。

## 验收：内容优先，灰影和风格分开

B轮Sol匿名观察：固定12 Val×5方法=60图，完整且精确匹配：

|方法|完整且匹配/12|部分辨读|主要观察|
|---|---:|---:|---|
|handnote|11|0|{"none": 12}|
|context-mass|11|0|{"gray-halo": 12, "faint-ink": 11, "ghosting": 10, "missing-dot": 1}|
|context-mass-projected|2|10|{"broken-strokes": 12, "damaged-strokes": 5, "faint-ink": 7, "missing-dot": 4, "joined-strokes": 1}|
|context-soft|10|1|{"gray-halo": 12, "ghosting": 12, "joined-strokes": 12}|
|context-projected|4|5|{"ghosting": 6, "joined-strokes": 12, "damaged-strokes": 7, "dark-double-line": 6, "broken-strokes": 2, "missing-dot": 2}|

C轮24图独立匿名观察：real-content 12/12只能部分辨读，real-shape 12/12整体不可辨，两个均0完整成功。
它们的灰背景虽然为0，但黑块、粘连与字形损坏明显。所有观察是模型目视判断，不是人工正确率；
内容已在开发使用，同文字版本可能相互提示，也不叫新独立测试。未对作者相似性赋分。

25 Val裁边识别观察（CER，不等于真实错误率）：

|方法|Tesseract|TrOCR（训练教师，非独立）|
|---|---:|---:|
|Handnote|13.98%|1.87%|
|旧mass|16.57%|3.17%|
|B soft|46.11%|4.90%|
|B projected|38.76%|11.82%|
|C real-content|76.22%|30.40%|
|C real-shape|83.72%|84.29%|

B原图/固定裁边共300输入、600双识别观察；C350输入、700观察。CER/WER/替换/插入/删除/尾部遗漏保留机器报告。
未把TrOCR教师分数或灰影指标作为最终checkpoint选择，全部固定最后一步。
本轮不再打开任何旧held作新测试；只有开发集验收，独立新文档泛化仍未知。

## 原因诊断：已确认与尚未确认

已确认：24样本有限拟合可行；完整数据时目标与条件仍难协调。
同一字体参考输入随机对应真实/字体两种目标，平均核心IoU仅.115、Dice.203，是实际存在的条件冲突。
C轮去掉它仍失败，说明它不是唯一原因；不能将所有问题归咎字体辅助或数据少。

轮廓投影可消除透明灰影，但不修复学错的位置/字形。直通梯度有偏，可能产生高频黑块；
识别教师对个人风格有域偏差，且内容损失不能可靠约束每根笔画。词内只按整词边框对齐、低分辨率、
真实连续英文不足也可能影响训练。以上机制仍需在新增可靠样本上用局部对齐/高分辨率/适当表示分别验证。
此刻继续加相同损失或训练步数没有成功依据，不制造“已经更自然”的结论。

## 真实语料继续积累

Sol新增审查89个未用页：微积分59页收12短语（9个2词、2个3词、1个4词），phy2/ICS30页收0。
纸面主要是公式、中文或印刷题面；4个候选框因不确定等弃用。未穷尽全部PDF，不夸大为全文语料审计。
新增12条无新字符类别、无>=5词行，全部300DPI原图/原框/摘要/模型审核可追溯。

新版本dataset：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T070232837773Z-continuous-context-dataset-fd330a28`；新视图：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T070256771182Z-context-word-aligned-views-9d34cf70`。
总276区域/101页/6文档；英文Train202、Val25；可准入对齐Train181（旧171+10），2新框因自然词间隙歧义排除。
准入Train仍只有8条>=5词，原始未筛对齐口径14条。新增批次没有偷偷加入B/C受控实验，保留下一阶段使用。
原Corpus、原文件、旧数据版本不改。

## 需要用户参与的边界

当前没有新的独立作业文档可验证泛化，现有可靠长英文训练样本很少，新增审查没有补到长行。
这次停止节点是需要真实连续英文与全新独立文档，不是模型达标，也不是未经允许发布。
[材料清单与三份可选模板](V0104_NEXT_DATA.md)：优先已有作业；没有则A/B/C各10句、每句6词，
分开书写/扫描保存，A/B补训练，C封存测试。模板只是提示，实际转写仍核验，不自动当真实数据。
补30行不保证成功；收到后先继续修正监督/对齐与模型表示，再评独立样本。

## 工程核验与剩余限制

236自动化测试通过；B/C四模型860PNG真实CUDA重放逐字节一致。
旧mass与历史221区域批量推理相比，只有最后7张图因末批8变7产生最多1灰阶浮点舍入差（739像素、2个阈值核心像素）；
不是权重或Profile变更，已记录数值检查，不宣称这些旧图全部字节相同。
81旧保护文件、14原始归档、原Corpus与生产源码/CPU依赖无变更。全部监督进程结束。
所有run有固定来源/配置/权重/依赖/种子，失败/弃用证据保留。没有新的数学模型、随机风格多样性、RL或默认发布。
完整训练路径见主run`reports/capacity-runs.json`、`full-training-runs.json`、`ablation-training-runs.json`；
所有子run见`reports/run-index.json`。原稿及个人资产仍不进入Git。
