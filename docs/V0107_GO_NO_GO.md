# v0.10.7：作者反馈后的词内对齐与条件风格学习

分支 `experiment/aligned-style-v0107`。**完成两组各1000步真实CUDA神经训练；通用生成和生产发布仍No-Go。**
该原型学习词内坐标变化，输出取样自既有个人字体；它不是自由新墨迹生成器，也不是端到端整句模型。
本轮没有新增真实样本或使用C，personal@v1、正式latest、生产CPU依赖和历史实验不变。

## 直接看成果

- [16页完整对照PDF](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T112427973080Z-aligned-style-corrected-presentation-ac340a5c/rendered/homework_solutions.pdf)：前4页整句、后12页开发词。
- [只看4页整句](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T112427973080Z-aligned-style-corrected-presentation-ac340a5c/comparisons/sentence-comparisons.pdf)：第1行本人原扫描，第2行Handnote，第3行新均值模型，第4–5行新随机模型两个种子。
- [四组匿名风格观察](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T112903510212Z-aligned-style-author-decision-b70b48f7/comparisons/author-review/index.html)：先看内容，再仅评自然程度/像本人程度，可同分。
- [配准失败与修正诊断](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T112427973080Z-aligned-style-corrected-presentation-ac340a5c/comparisons/registration-diagnostics.pdf)：看过目标扫描的oracle，不算生成成功。

整句为训练中已曝光的前4条新采集句子，预先固定，不按好坏挑选。网络按完整词生成形变后在Handnote原坐标中组合。
比较中字体和模型共用归一化/逆变换，真实扫描保留原样；原Handnote原生图另存于句子run。词间距和连线拓扑未学习。
所有逐页PNG在同run的`rendered/pages/`。主run也保存相同16页PDF。

## 已导入的作者意见

上一轮两组8候选全部被作者确认内容正确。Handnote与2度旋转对照自然度/本人相似性均4.0；新learned-A/B均2.5。
因此暂停v0106整词直接距离场CVAE，不继续扩大词表或增加相同损失步数。仅两种筛选后的词，不能推断统计显著性。
导出原件、图像身份与摘要绑定和分维度偏好保存在 `/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T110351375823Z-context-author-feedback-75ccc4a9`；未用于RL或回标为真实数据。

## 数据与对齐诊断

继续使用既有512已审核真实词：452Train、60已曝光开发；不拆连笔、不修补原图，不改变文档划分。
- 自由2D控制网格：452Train中404负Jacobian、15边界触墨，仅47通过预定准入，停止该训练分支。
  轮廓Dice由0.203升至0.656，却产生折叠和拉丝，说明高轮廓分数不能替代内容质量。
- 唯一一次机制修正：横向位置保持单调，纵向为正比例和局部平移。零负Jacobian；373Train/268词形通过相同门槛。
  34边界、45低Dice等问题仍排除，门槛没有放宽。整体Train配准Dice0.525，仍是看过目标的诊断数。

配准每对240次数值优化，**不计入神经生成模型训练**。监督是从真实词对提取的坐标场，不把配准后字体图冒充真实手写。
可追溯原数据 `/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T095310338134Z-whole-word-reviewed-training-data-1b5f966c`；配准版本 `/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T111242241295Z-aligned-style-registration-b26549d5`。本轮373是原452的子集，不是新增373份样本。

## 真实训练及限制

内容条件是完整ASCII字符序列和Handnote词图；随机分支含4维整词latent、条件prior和训练时posterior。
网络学习从内容和latent预测无折叠坐标场，取样原连续墨迹；不二值化，不加入负浓度修正。
它可调整词内宽度、高低和局部比例，但不能创造字体中不存在的笔画或连线，个人全局风格仅隐含在单作者训练数据中。

|模型|Train|真实更新|参数量|权重绝对变化和|峰值张量显存|训练及输出耗时|
|---|---:|---:|---:|---:|---:|---:|
|mean|373|1000|163203|3552.43|450.17MiB|20.17s|
|stochastic|373|1000|163203|5610.69|450.72MiB|58.98s|

耗时包含checkpoint/PNG输出，不能直接当纯训练速度；显存为进程张量峰值，不是整卡占用。
真实设备RTX5060 Laptop/8151MiB，驱动580.126.09，Torch2.7.1+cu128，独立环境完整依赖在主run。
两模型使用相同373Train、固定种子1707、相同1000步预算。只在冻结后评60开发+12Train，未据开发挑checkpoint。
优化器、模型、RNG、初始状态、曲线、源码/数据/权重摘要完整保存，独立核验optimizer均到1000步。

- mean：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T111545455227Z-aligned-style-train-c261558e`
- stochastic：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T111607709288Z-aligned-style-train-9cc79b67`
- 独立重放：`/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T111708851352Z-aligned-style-frozen-evaluation-94c45824`，360张原始prior PNG完全一致，推理未读真实目标。

另有一次mean尝试在500步保存checkpoint时因`numpy_rng.state`误作函数调用而失败；旧失败run完整保留。
这500步仅有执行日志、没有最终权重和优化器证明，单独报告，不计入2000个已完成且核验的更新。

## 多维验收

|维度|实际证据|判断|
|---|---|---|
|文字内容|固定12开发词：Handnote12/12、mean12/12、stoch1708 12/12、stoch1709 11/12；共48图47匹配|模型观察，非人工真值/整体正确率|
|完整句子|4个已见训练句×4方法，16/16模型转写匹配；无边界核心墨迹|仅组成测试，不能证明新文档泛化|
|断墨/裁边|随机一词局部淡笔不确定；288随机图中3图共8个边界核心像素|仍No-Go自动发布，无静默回退|
|字形变化|随机四种子成对像素MAE0.066–0.072，mean固定；全部最终场Jacobian为正|随机变量确实影响形变，不等于自然度提高|
|个人风格|本轮没有作者评分；上一轮作者更偏好字体|当前必须由作者看句子判断|
|记忆|60开发图到真实Train最近邻Dice约0.259–0.551，无整幅二值完全相同|字体像素本来就被复用，不能据此宣称自由新字形或排除局部记忆|
|形状分数|开发Handnote→真实Dice0.226；mean0.219，stochastic均值0.220|未优于字体；不能单据此评价风格|
|数学/整页|本轮未生成新神经数学或完整解题页，原生产数学路径不变|无新增数学正确性结论|
|工程|248自动测试、3项真实CUDA性质测试、360图重放、独立run及来源记录|工程通过，视觉目标尚未达成|

三张随机边界图：开发index123/seed1708、135/1709，Train index365/1711，原图保留。
阈值组件计数未上升，却仍有人工模型观察淡笔，说明组件代理不能替代逐图内容检查。
整句四张`boundary`末尾没有句点，实际目标本来无句点，不算模型漏标点。4张观察词距偏紧，尚可辨词界。

## OCR/HTR证据及独立性

436输入（60开发词×7方法，加16句子图）、872个本地识别观察；训练未用识别器教师，无外部上传。
识别器对真实孤立词本身也存在高误差，因此下面是**识别差异，不是真实文本错误率**。

|方法|60词Tesseract CER|60词TrOCR CER|
|---|---:|---:|
|Handnote|28.23%|13.17%|
|mean|36.29%|20.97%|
|stoch1708|36.56%|18.28%|
|stoch1709|34.68%|30.65%|
|stoch1710|29.30%|15.59%|
|stoch1711|35.48%|15.05%|
|真实词图|62.10%|25.81%|

4句TrOCR CER：Handnote1.68%、mean4.47%、stoch1708 2.23%、1709 3.91%。总体不支持新候选内容优于字体。
原始OCR/HTR及CER/WER/替换/插入/删除/重复/尾部观察在 `/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T112458691872Z-context-independent-recognition-cf7a4a88`。
旧汇总器只列train/validation/holdout，漏列新增exposed-train-composition统计；原始872观察完整。
已修通用汇总并新增测试；补齐结果写主run`reports/recognition-complete-summary.json`，旧sealed run未改写、识别未重跑。

## 失败保存与展示修正

最初展示PDF的Matplotlib默认色图把灰度PNG显示成伪彩色。原始PNG/权重一直正确；新版显式灰度重建16页，未改原像素。
旧PDF留存，最终请使用上述corrected-presentation入口。浏览器已实际测试图像加载、确认启用评分、下一组/返回和本地保存；
测试使用隔离会话，已清除测试状态并关闭，没有制造作者评分。

## Go / No-Go 与下一步

Go：词内对应关系提取的有限研究工具、保留清晰字体墨迹的学习式变化、4句作者风格观察。
No-Go：通用新墨迹生成、可靠新连笔、自动神经数学/生产替换、扩大相同训练、RL。C继续封存，未完成长期自然连续书写目标。

当前需要作者判断四句中的变化是否值得保留。暂不要求补写、上传PDF或输入LaTeX。
若自然度/本人相似性确有收益，下一阶段先修复输出边界与亚像素细笔保留，并与等幅度非学习对照比较，再决定是否开C；
若仍不优于Handnote，停止把该形变分支当主生成器，不追加步数。保留配准诊断用于后续新墨迹架构，而不是加强字体收缩。
作者评分仅用于评估/规划，不自动发布、不回标合成训练数据。

最终保护复核：121个历史/配置/权重文件摘要不变；14份归档原件、512个真实词裁剪及原Corpus数据库不变。
本轮全部子run终态封存并校验；最终Git与源码摘要在主run `reports/final-code-identity.json`，完整目录见`reports/run-index.json`。
