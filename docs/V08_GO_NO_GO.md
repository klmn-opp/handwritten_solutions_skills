# v0.8 Evidence-Based Go/No-Go Report

日期：2026-11-18。分支：`experiment/english-evidence-v08`，从 `experiment/diffink-cuda-v07@21a582b` 分出；稳定main与v0.6.0标签未移动。当前CPU包仍为v0.6.0，本报告是研究阶段交付，不是新生产模型发布。

## 决策

**个人监督微调：目前所有路线均 No-Go。推荐 DiffusionPen 作为下一轮英文生成工程验证的主候选，DiffBrush保留为整行对照；生产继续font-cpu。**

DiffusionPen已经完成官方真实权重、五个本人扫描词条件的英文CUDA推理。在相同16条smoke、相同两个种子的比较中，它的TrOCR差异率低于DiffBrush，但仍有长词、数字、标点与词间尺度问题；没有覆盖完整120条或独立holdout，也没有作者盲评结果。其优势仅支持优先投入下一轮验证，**不支持启动训练或替换生产**。DiffBrush的主要内容问题已经存在于原始模型输出，单纯修补Adapter或增加优化步数缺乏依据。DiffInk当前发布词表与中文在线模型不满足任意英文输入。

本轮没有从零训练、监督微调、LoRA、D3PO/RL、旧bandit扩展或自动发布。五词条件推理、字体归一化与统计评测均不称为神经训练。

## 实际执行范围

| 路线 | 固定文本与执行 | 结果边界 |
|---|---|---|
| personal@v1 / v3 | 各120行，96 development +24 holdout，seed42 | 各120生成并验证；两Profile行图120/120逐像素相同 |
| DiffBrush 原始模型对照 | smoke16×seed42/43；官方与Handnote同权重、噪声、风格、内容 | 32组真实官方/Adapter对照；原图与PDF保真 |
| DiffBrush 全基准 | development96×2 + holdout24×2 | 240请求，236生成；4次明确拒绝 `% [ ] =`，未删改输入 |
| DiffBrush 参考消融 | 相同smoke16×2，raw与保全宽度fit | fit32生成；没有选fit重跑或调节holdout |
| DiffusionPen | 官方EMA、5本人真实扫描词，smoke16×2 | 32次真实行输出；每个词的原始模型图另存 |
| 真实本人扫描 | 16行历史样本与历史转写 | 用于识别器校准，不声称历史转写已再次人工核验 |
| 独立识别 | Tesseract + TrOCR handwritten，对上述588张图逐张执行 | 588/588 HTR完成，无输出token上限命中；全部原始转写保留 |
| 人工机制 | 72份匿名材料、五项评分、内容先行门禁、答卷导入 | 浏览器流程和导入校验已测试；真实人工答卷0份 |

字体与模型使用相同目标文本。DiffBrush用本人扫描整行，DiffusionPen从同一行取五个真实词，属于同作者、同源参考的可比较条件，并非形状相同的模型输入。输出分辨率不同，独立识别采用统一归一化；原生图全部保留。数学路径和一题一页正式作业系统没有改动。

## 内容指标：这是识别器差异，不是真实生成错误率

以下为同16条smoke的微平均。字体每条一个固定种子，模型每条两个种子；未根据成绩挑种子。Exact Match对大小写及标点敏感，只合并空白。

| 路线 | 图像数 | Tesseract CER / WER | TrOCR CER / WER | TrOCR Exact Match |
|---|---:|---:|---:|---:|
| personal v1 | 16 | 9.86% / 48.72% | 5.14% / 43.59% | 0% |
| personal v3 | 16 | 9.86% / 48.72% | 5.14% / 43.59% | 0% |
| DiffBrush raw | 32 | 55.90% / 111.54% | 32.78% / 72.22% | 0% |
| DiffBrush fit参考 | 32 | 60.14% / 117.52% | 32.78% / 74.36% | 0% |
| DiffusionPen | 32 | 54.17% / 104.27% | 22.57% / 58.12% | 0% |

WER可以超过100%，因为插入也计入分子。**0%严格Exact Match不等于所有图像内容都错了**：TrOCR常在句末标点前输出空格；Tesseract会拆开连笔或把字母认成数字。没有为改善分数而回写转写或删除标点。两识别器训练域不同，TrOCR与生成模型共享IAM数据域偏好；它们不是对真实内容的两个无偏裁判。

完整120行字体基线：Tesseract CER9.27%，TrOCR CER4.35%、WER37.51%、Exact Match1.67%；v1/v3相同。真实本人16行扫描：Tesseract CER33.10%，TrOCR CER12.64%、WER40.21%。这组校准结果表明识别误差不能被直接记为作者或模型写错。

DiffBrush development TrOCR CER29.76%、WER67.00%；独立holdout CER29.83%、WER63.22%。按原始输入长度分层，少于35字符CER44.72%，35–61字符14.76%，超过61字符50.14%。这与官方生成脚本选用35–61字符的条件一致，但仍只是分层观察，不构成训练长度因果证明。

逐字符/逐词对齐、替换/删除/插入、重复插入词、尾部删除、按类别/seed/split汇总均在 `v08-evaluation-final/reports/observations.json`；全部未获双识别器一致确认的项目进入失败复核队列。该队列不是人工已确认的失败集合。

## DiffBrush内容错误定位

1. **输入完整**：所有受支持文本的字符索引可完整还原，内容张量长度正确；安全Unifont转换与官方一致，没有查到Adapter静默截字。
2. **官方也出现异常**：32组原始官方输出与Handnote仅有很小数值差异，round后最大平均灰度差0.03493/255，不能将主要重复/漏写归因于Adapter。最短句 `Check the sign.` 的原图存在明显sign等笔迹重复，保留图例与识别原文；此为辅助视觉检查，不计作者评分。
3. **参考预处理问题确实存在**：1294×75参考按高64缩放后超1024宽，右侧会被裁；保全宽度fit消融未显示总体内容改善，不据此宣称解决根因。参考中的纸纹、横线等也可能造成域差异，尚无干净采集A/B证明其影响大小。
4. **导出未新增裁切**：生成器保留未trim原图，另存trim副本。所有生成PDF逐像素对回原图；末字问题已经在原图出现。模型固定画布边界的作用尚待独立人工标注。
5. **触边检测不可当真值**：生成图广泛触边，真实本人扫描也有12/16触边（纸线/裁剪等），所以只报警，不把它计成真实截断率。

详见 [源码与权重调查](V08_MODEL_INVESTIGATION.md)，包含上游commit、严格加载、许可和Reforge组件审计。

## 风格与个人相似性

72份盲评材料按八类的固定01号开发文本选取，包含两个CPU Profile、DiffBrush raw/fit、DiffusionPen的两个种子，以及8份本人扫描控制；没有按效果选图。相同内容先盲转写，再校核，内容完整才能评连笔、字母比例、间距节奏、本人相似性、清晰度，分数为1–5。真实本人参考另展示；后端、OCR结果与解盲表不出现在评审界面。

目前 **人工内容准确率、自然度、个人风格相似性全部未验证**。辅助图像检查可见DiffusionPen词内部连笔较连贯，但不同词的大小和基线差异明显，且长词存在缺失/变形。不能据此给出“真实性从60%提高到某个百分比”的结论。

## 部署与成本

本机RTX5060 Laptop，8151MiB，驱动580.126.09，PyTorch2.7.1+cu128。时间为本轮实际单行中位数，GPU项是最大已分配显存，不是整卡总占用；冷启动/依赖加载不包含在单行时间内。

| 路线 | 单行中位时间 | 峰值已分配显存 | 主要权重体积 | 参考需求 / 训练判断 |
|---|---:|---:|---:|---|
| font-cpu | 0.092s（含作业验证） | 0 | 既有个人字体 | 无GPU，无新增训练 |
| DiffBrush | 1.53s | 1.84GB | 1.166GB +共用VAE0.335GB | 本人扫描行；需先解决内容与边界 |
| DiffusionPen | 3.78s；实测1.61–8.73s | 1.13GB | EMA0.680GB +style0.011GB +VAE0.335GB | 恰好5个扫描词，无需在线轨迹；小样本条件推理不等于微调已可行 |
| DiffInk | v0.7中文20步0.76–0.85s；本轮未新增英文推理 | v0.7约1.05GB | DiT2.915GB +VAE0.179GB | 需真实/明确标注估计来源的在线轨迹；当前发布词表不足 |

8行对照页已实际拼版，字体、DiffBrush、DiffusionPen对应已测行耗时之和约1.39s、12.23s、29.43s，拼版另耗0.08–0.18s。**这些是分项测量，不是独立计时的整页模型端到端推理**；真实整页一次性神经生成尚未验证。DiffusionPen仍是词组合，DiffBrush是行输出。部署许可、IAM数据限制、SD VAE许可与Reforge缺少明确LICENSE的情况见调查报告；没有发布商业可用性承诺。

## 交付路径与版本

所有路径相对仓库根目录，实验均遵循 `workspace/runs/<run_id>/`；run.json固定代码、环境、Profile/参考摘要、权重、seed和硬件，日志含研究源码快照。每个实验ID不可重用，不更新正式latest。

| 内容 | 路径 |
|---|---|
| 字体基线PDF | `workspace/runs/v08-font-v1-frozen/rendered/homework_solutions.pdf`；v3同结构 |
| 官方DiffBrush原图/原始DDIM对照 | `workspace/runs/v08-diffbrush-official-smoke/rendered/pages/` 与 `comparisons/*-official*.png` |
| 全量DiffBrush / 留出 / fit | `workspace/runs/v08-diffbrush-development/`、`v08-diffbrush-holdout/`、`v08-diffbrush-fit-smoke/` |
| DiffusionPen原生词图、行图、PDF | `workspace/runs/v08-diffusionpen-smoke/comparisons/`、`rendered/` |
| 两识别器逐图证据 | `workspace/runs/v08-evaluation-final/reports/observations.json`、`htr.json`、`recognizer-provenance.json` |
| 四页同文本对比PDF | `workspace/runs/v08-evidence-analysis/rendered/homework_solutions.pdf` |
| 失败图例 / 全量复核队列 | `workspace/runs/v08-evidence-analysis/comparisons/failure-examples.png`；`reports/failure-review-queue.json` |
| 人工盲评 | `workspace/runs/v08-blind-review/comparisons/blind-review.html`；空答卷模板与私有解盲表在其reports目录 |
| 汇总 / 资产 / DiffInk缺字 | `workspace/runs/v08-evidence-analysis/reports/summary.json`、`new-model-assets.json`、`diffink-english-coverage.json` |
| 自动化测试 / 构建 | `workspace/runs/v08-final-validation/reports/`、`logs/`；构建在 `v08-validation/comparisons/dist/` |

有效阶段提交：语料`7d9e429`；渲染/官方对照`2423e83`；HTR/本人对照`0894324`；DiffusionPen`e7ff1dd`；完成状态保护`8052b71`；盲评`4c8f9bc`及`41c7dd7`；协议/依赖`6783254`；调查与汇总`602a8fd`。最终文档和验收收据另有交付提交。不是重建项目，也没有合入main。

## 已通过、失败与未验证

**已通过**：最终两版Python各141项全套测试通过，零失败、零跳过；Ruff检查/格式；独立wheel/sdist构建；模型严格权重加载；官方LFS摘要；PDF像素保真；盲评门禁实际浏览器测试。单元测试与UI夹具不计为模型推理或人工质量证据。

**保留的失败**：最初HTR依赖索引解析失败；冗余慢下载取消并保留partial；首次评测提前读取未完成run失败；no-isolation构建缺setuptools失败；uv freeze默认缓存只读失败。均保留日志，随后有独立成功重试；DiffBrush4次不支持字符拒绝仍是有效限制。模型识别差异、明显重复/缺失和词间比例问题没有被包装成已修复。

**仍未验证**：作者盲评与独立人工错误标注、DiffusionPen完整120/holdout、整页端到端神经推理、个人监督微调收益与资源需求、模型商业分发许可确认。没有把统计成绩或工程通过率当自然手写提升证据。

保护复核：初始清单5074项含审计自身run.json，排除封存自引用后 **5073个既有历史文件摘要全部不变**。Runtime源码、已安装CPU环境、默认personal@v1、正式latest=`v06-five-final`和用户.vscode未改；生产仍无torch。构建只更新了可再生 `src/handwritten_solutions.egg-info/SOURCES.txt` 清单，其原摘要及新文件已留证，不冒充源码改动；其他源文件保持原摘要。旧Profile、数据集、权重与CUDA运行没有删除。

## 下一里程碑

1. 完成盲评内容核验，至少建立独立双人复核子集；校准TrOCR标点空格和长行域偏差，保留严格原始指标，不直接根据同一OCR挑候选又验收。
2. 以DiffusionPen为主验证词间x-height/基线约束、标点及数字完整性；保留原始词图和逐项来源。不得用合成标点掩盖模型内容失败。完整开发集验证后再用未参与决策的独立holdout。
3. 若内容和作者风格门槛均通过，再确定个人监督微调试验：按采集页/会话划分真实扫描词/行数据，保留历史版本，避免近重复跨集合；先验证上游监督方法能否在现有GPU上稳定运行。五个参考词不是足够的训练数据集，不启动未经论证的LoRA或从零训练。

**最终判断：DiffusionPen获下一轮工程验证优先级；目前没有任何路线获个人化训练或生产准入。**
