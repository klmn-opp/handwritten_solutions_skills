# v0.10.5 新真实语料与神经轮廓实验：待作者风格验收

分支 `experiment/new-continuous-corpus-v0105`。新增材料已实际导入、Sol核验与版本化；
实际完成4模型、6000次CUDA参数更新。**内容与清晰度可用作有限研究候选，但未证明连续个人风格学习优于Handnote，生产仍No-Go。**
正式font-cpu、personal@v1、latest不变。C原件继续封存，没有把已反复使用的开发集叫独立测试。

## 可直接看的证据

- [最终8页对照PDF](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T090230165969Z-context-handnote-flow-delivery-a49555bb/rendered/homework_solutions.pdf)，[49页全对照](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T090230165969Z-context-handnote-flow-delivery-a49555bb/comparisons/all-clean-ink-comparisons.pdf)。
  每页依次为本人真实扫描、正式Handnote、旧mass、零更新Handnote距离场、新实训Flow。
  最后一行才是本轮候选；第四行没有神经训练。所有对照标明Train/Validation。
- [作者匿名评审：4组×3候选](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T090230165969Z-context-handnote-flow-delivery-a49555bb/comparisons/author-review/index.html)。2条新Train行+2条旧开发片段，仅作内容与风格判断；C不在其中。
- [数学实际手写—LaTeX对照13页](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T082902552123Z-paired-math-transcription-evidence-c23fedfc/rendered/homework_solutions.pdf)，[离线公式页面](/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T082902552123Z-paired-math-transcription-evidence-c23fedfc/comparisons/math-review/index.html)。这是原始语料转写，没有神经生成数学。
- 逐页PNG与原始生成图、源码、权重/优化器/RNG和曲线均在相应run内；路径见主run `reports/all-training-runs.json`、`reports/run-index.json`。

## 新材料真实数量

4份PDF共9页：A/B各1页，C1页仅归档，数学6页中3页手写、2页印刷参考、1页空白。
新增62个可靠原像素区域：20完整英文行（120词、120个整词框）、3英文标题、39数学区域（41式）。
A03/B03实际未写句号，标签未按印刷提示补句号。浅灰模板线保留原图，不合成修补笔画。
数学按实际手写保留 `Y≥y` 与印刷 `Y=y`、手写 `dF(x)`、大小写等差异；准确转写不代表公式本身正确。
有歧义或不完整的两处数学候选弃用，证据保留。印刷区域不计入真实样本。

完整新数据集 `/home/klmn/handwritten_solutions_skills/workspace/runs/20261119T082902471963Z-continuous-corpus-extension-f9e7198a`：338区域、106页、9文档。
英文225 Train/25 Val；数学82 Train/6 Val。C未包含，书写会话时间独立性未核实，只称新文档封存。
受控英语实验使用含A/B的前一个数据版本，222英文Train/25Val中203/18通过对齐；所有25Val仍参与内容评估。
20条新完整行全部准入。可靠至少5词的Train由8增到28；3个数学页英文标题没有事后混入受控训练。
128x1024视图另让2条旧样本通过空隙规则，原版本与拒绝记录不改。

## 实际怎么训练

不再从旧CVAE加步数。此次小网络输入个人字体/Handnote真实字形轮廓，学习整区域的墨迹修形。
local只看局部，context增加沿行共享上下文，均1500步。Flow另外学习平滑二维位移（最多横8px/纵5px）与最多1px边缘变化。
所有样本的监督目标来自真实扫描，A/B采样概率35%、旧真实材料65%，没有把字体当真实目标；没有OCR教师参与新训练。
最终修正Flow只改变参考路径：直接走Handnote矢量渲染，保留每个字形ID/变体/语义来源，不修改任何Profile。

|候选|真实更新|参数变化绝对和|开发形状Dice（前→后）|峰值张量显存MiB|
|---|---:|---:|---:|---:|
|local|1500|120.36|0.22775 → 0.22635|608.8|
|context|1500|149.88|0.22775 → 0.22418|673.4|
|flow|1500|211.82|0.22775 → 0.22289|1563.6|
|handnote-flow|1500|168.02|0.20546 → 0.20473|1563.6|

最后一行的参考与对齐路径不同，绝对Dice不能跨参考路径排名，只比较各自前后。

RTX5060 Laptop，Torch2.7.1+cu128隔离环境。每500步保存权重/优化器/RNG，固定1500步最终候选，未按C选模型。
最终位移场未发现离散Jacobian≤0，最小值约0.814；这是局部折叠代理，不是全局内容正确性的数学证明。
新网络仍受字体拓扑与Handnote词间距强约束，没有随机风格潜变量、自由连字、真实在线轨迹或神经数学结构。
学习式位移确实有参数更新，但不能据此宣布学会自然连续书写。

## 验收与公平性修正

初轮匿名80图发现：Pillow默认0变体在零更新和新候选中被读成6；正式Handnote此位置采用variant2/glyph00870，读成0。
问题存在于输入参考，故统一到真正Handnote路径后重训。原失败/歧义证据保留，没有把参考修正称作学习收益。

最终候选25条开发Val+4条Train均由Sol实际目视完整转写且匹配标签，29/29；这是模型观察，存在已曝光文字提示，不是人工真实错误率。
另做16图同300DPI Handnote基线复核，16/16完整匹配。旧150DPI基线和其局部平截观察已归档为过时显示对照，不能据其模糊宣称神经收益。
最终候选29图中13图有局部细笔偏淡；未观察到旧mass那样的灰晕/重影，仍需作者判断本人风格。

同300DPI参考、固定识别裁边的25Val观察：

|方法|Tesseract CER|TrOCR CER|
|---|---:|---:|
|Handnote|14.27%|1.01%|
|零更新距离场|14.12%|1.87%|
|新Handnote-guided Flow|13.54%|1.87%|
|旧mass|16.86%|3.17%|

OCR分数不是实际错误率；不同识别器结果并不一致。新候选没有可靠胜过零更新对照，不能宣称神经训练带来明确内容收益。
局部灰影清除主要由明确的轮廓表示和参考路径提供；神经修形的额外价值尚未成立。
25Val中没有与字体距离场或整个Train区域二值完全相同的输出；这不排除局部记忆，也不证明新个人风格。
禁用上下文会改变输出，平均像素绝对差约0.0022；这证明模块参与计算，不能证明它有益。

## Go / No-Go 与需要用户做的事

Go：持续累积这类完整自然英文与配对公式；保留当前内容稳定的字体约束候选，开展小规模作者风格判断。
No-Go：声称已优于Handnote、扩大当前损失训练、自由连续神经书写、数学替换、生产发布或RL。
无需继续补材料或手输LaTeX。请在4组匿名页面中先确认内容，再只评“自然程度”和“像本人程度”；若差不多可同分并备注。
导出JSON至 `source_pdf/review_exports/`。评分不会改变默认Profile。

若作者也认为零更新/Handnote更好，下一阶段先做词内软对齐与多尺度内容表示的小型可学习性诊断，
不要继续加粗字体锚定损失或堆步数。只有开发内容和作者风格都值得继续，才冻结候选后打开C作一次真正文档级评估。

## 工程与未验证范围

最终实现239项CPU自动化测试通过；独立CUDA梯度/采样/位移边界检查通过。四候选共988区域在一致确定性设置下真实重放完全一致。
第一次context评估少了确定性设置，最大浮点差4.77e-7；旧证据保留，修正后重新运行，不宣称所有历史运行均逐字节一致。
13页数学PDF的39个KaTeX预览/原图均加载、无解析错误/横向溢出；作者UI内容门禁、翻组和保存已在独立本地浏览器检查。
全部原件与旧Profile/权重/正式latest受保护。生产源码与CPU依赖不变，无Torch强制依赖，未上传私人数据。
未验证：C泛化、作者相似性、随机自然多样性、完整新神经数学作业或CPU神经部署性能。
