# 当前状态：v0.10.2 去灰影训练完成，改善与细笔损失并存，待作者视觉选择

分支`experiment/context-clean-ink-v0102`；[报告和验收](V0102_GO_NO_GO.md) · [研发日志](V0102_DEVELOPMENT_LOG.md)。
以用户认可的上一轮aligned-content为起点，三臂各1500次真实CUDA更新，原171 Train/25 Val不变。

- 距离场修复使25 Val外围灰墨代理减少约59%；墨量约束总墨量约减52%，但短词局部缺笔增多。
- 旧模型固定对比度显示方案也明显减灰，明确零训练更新；所有新学习结果保留原始输出。
- 100图Sol盲转写，主要候选均18/20完整匹配，标点和大小写歧义保留；不把模型观察当人工真实错误率。
- 新16区域（15单词/1词组）冻结后读取，仍旧文档；短词HTR边距敏感另用固定12图校准，原指标保留。
- 10页速览、49页全对照、4组20候选作者UI；462图CUDA重放一致，219测试全部通过。
- 60受保护文件、14原件、原Corpus、正式CPU/Profile/latest不变。当前没有新作者评分、不自动发布。

主交付：`workspace/runs/20261119T035058113853Z-context-clean-ink-v0102-3d8446a7/`。
打开`comparisons/author-review/index.html`，只需核验内容、评价自然度/本人相似性；JSON导出到`source_pdf/review_exports/`。
Go：保守清晰化与作者验收；No-Go：直接发布墨量约束模型、扩大当前训练、数学替换或RL。

---

以下保留历史状态。

# 当前状态：v0.10.1 已完成对齐/内容实训，可辨读性改善，清晰度仍未达到Handnote

研究分支 `experiment/context-alignment-v0101`；[完整报告和验收路径](V0101_GO_NO_GO.md) · [研发日志](V0101_DEVELOPMENT_LOG.md)。
正式font-cpu、personal@v1、latest未变，全部旧证据保留。

- 新增120英文Train/14 Val；连续数据264区域/93页，英文190 Train/25 Val。三个真实训练臂均使用同一171可对齐Train。
- 小集诊断2000次、字体预热1500次、raw/aligned/aligned-content各2000次真实CUDA参数更新；新模型144709参数。
- 最终候选冻结后评估21新区域，仍非全新文档；5方法1180图，486图双识别器972观察。
- 独立Sol盲转写：Handnote20/21、纯对齐16/21、加内容约束20/21完整辨读且一致；新模型均有明显粘连，未通过生产质量门禁。
- TrOCR是冻结内容教师，其分数不能作为独立成功证据；本人风格/自然度尚待作者评分。
- 10页速览、54页全对照、6组24候选匿名UI；236张内容臂PNG真实CUDA重放完全一致；215测试通过。
- 两个错误教师标签训练run及一个浅色参考评估失败run全部保留；修正后从同一预热完成有效重跑，没有混用旧优化器。
- 41受保护文件、14原件和原Corpus不变；生产源码与依赖无改动，CPU环境无Torch。新网络未替换数学，保持held。

主交付：`workspace/runs/20261119T025248583848Z-context-alignment-v0101-01ce31a5/`。
先看`rendered/homework_solutions.pdf`（研究对照，不是新解题PDF）；最后三行是本轮真实训练候选。
Go：词内对齐/墨迹清晰度/真实风格条件的有限研究。No-Go：扩大当前整句模型、默认发布、数学替换及RL。

---

以下保留此前各轮状态，以顶部为当前结论。

# 当前状态：v0.10 已完成上下文原型实训，内容失败，继续保留 Handnote

分支 `experiment/context-handwriting-v010`；[完整证据与决策](V010_GO_NO_GO.md)。
正式 font-cpu、personal@v1、latest 不变，不恢复外部扩散模型研发。

- 冻结诊断：repair 更靠近字体、多样性和潜变量响应下降；无真实Train二值精确复制，不能断言记忆过拟合。
- 完成32片段/24页对照、4完整作业组/12页A4及PNG、37页本人同内容参考对照。
- 新连续区域130条（Train113/Val17），另20未用区域冻结后评估；所有7个PDF已在历史使用，无全新文档Holdout。
- 新整块区域上下文模型5臂×800=4000次真实CUDA更新，完整保存初始/定期/最终权重、优化器、配置与曲线。
- 主臂在独立6段盲转写中0/6完整辨读，Handnote/repair各6/6；5臂均未获内容/自然度验收。
- 261图双识别器522观察，识别差异与真实错误分开；1820生成图，主臂364图同种子重放一致。
- 211测试覆盖全部通过（210项原全套中1项socket限制后重跑，新增1项终态保护）、4CUDA架构检查、12浏览器行为检查。
- 结论：GO继续已有PDF的连续数据/对齐；NO-GO扩大当前模型、生产、SFT规模扩张或RL。作者风格盲评尚待完成。

主交付：`workspace/runs/20261119T012706161166Z-context-handwriting-v010-da36188a`，held，不推进正式latest。

---

以下保留历史状态，数量及下一步以顶部为准。

# 当前状态：v0.9 新增298真实Train，完成两轮续训与一次修正，生产仍No-Go

研究分支`experiment/expanded-corpus-two-rounds-v09`；[完整证据与路径](V09_EXPANDED_CORPUS_TRAINING.md)。
正式font-cpu、personal@v1、latest及原Studio数据未改变，主交付保持held。

- Sol xhigh清洗新增298张真实Train，41→339张、4→52类；178字体辅助/75类保持不变。23类仍无真实Train。
- 两轮8000+8000次真实CUDA更新，开发集发现断笔后额外4000次修正；全部75类继续联合训练。
  本轮共20000次，修正候选选中祖先累计9500步；保存best/last/优化器/原始图，不把OCR称作训练。
- 新Validation58/40类（合计71）参与开发；新Holdout57/38类仅在冻结全部候选后查看。
- 第二轮Holdout形状距离0.127473，上一版0.133715，几何0.128807；约4.7%代理改善，不等于自然度提升。
  600图断裂代理为旧6、R1 16、R2 20、修正11；修正轮8类变化偏小、14图二值同字体，存在退步。
- 375图Sol盲检每种方法75图：缺损标记Handnote1/旧7/R1 10/R2 7/修正2；新模型仍有5/J、√/T不匹配。
  孤立字符歧义和弃答不算真实文本错误，没有作者风格评分，不能声称超过Handnote。
- 全200测试通过，另3项Torch损失检查通过；600生成PNG同设备/权重/种子重放一致。23个子run封存验证通过。
- 14原件、审核快照、受保护Profile/latest/旧权重摘要不变；生产环境无Torch、源码与依赖未改。

交付：`workspace/runs/20261118T174726751468Z-expanded-corpus-two-rounds-58bbf5d2/`。
rendered/homework_solutions.pdf为新5页全75类对照，含Handnote/旧版/两轮/修正；逐页PNG与原600图齐备。
下一阶段补低覆盖真实类、解决身份/完整性与多样性的冲突；不继续盲目加步数、不发布默认模型、不做RL。

---

以下为前轮状态，保留历史；其中旧覆盖数与评估数不是当前值。

# 当前状态：v0.9 全75个人字符首版神经候选已完成，生产仍No-Go

研究分支`experiment/all-personal-glyphs-v09`，继续原项目，正式font-cpu、personal@v1、latest不变。
[完整全字符报告](V09_ALL_CHARACTER_TRAINING.md) · [上一轮数字1辅助训练](V09_HANDNOTE_TRAINING.md)。

- 178个人字体派生轮廓覆盖75类；真实Train41仅覆盖0/1/2/x，其他71类仍缺真实作业训练。
- 新参考条件小型CVAE真实CUDA三轮训练，共16000次更新；最终选中权重祖先累计4500步。
  579449参数，三轮GPU训练约341秒，峰值分配约428MiB；不是字形随机选择或几何抖动。
- 根据开发证据修正少量真实样本过度重复；全部类别持续联合学习，保留字体来源和完整下伸/标点。
- 固定600图：0空白/0越界，6断裂代理，11张二值轮廓同字体，C/m变化小；99.5%是模板代理，非OCR正确率。
- 新Validation30/25类、新Holdout30/30类由Sol保守清洗，模型冻结后才查看Holdout；从PDF重渲染到裁剪校验通过。
  最终Holdout形状距离0.1821，字体0.1869、几何0.1736，仍未胜几何；未据测试结果换模型。
- 全196测试通过；同权重/种子600张真实GPU生成PNG逐字节重放一致。单词连笔、作者风格和完整数学未验证。
- 冻结后Sol无标签盲检150图：120清晰、22歧义、8缺陷；125个非空转写中5个与目标不一致
  （√→T、5→J、o→σ）。它是模型判断，非人工错误率；内容门禁尚未通过。

主交付：`workspace/runs/20261118T170015253623Z-all-personal-glyph-delivery-3e580c3a/`。
`rendered/homework_solutions.pdf`为5页全字符研究对照，逐页PNG和全字表对照齐备；不是完整神经作业验收。
候选`v09-all75-reference-candidate-1`保持held，不激活默认模型；后续重点补真实覆盖和独立内容质量。

---

以下保留上一轮真实状态和历史证据。

# 当前状态：v0.9 Handnote辅助真实训练完成，数字1过淡改善，整体生成仍No-Go

研究分支`experiment/handnote-assisted-glyph-v09`，沿用原项目；正式font-cpu、personal@v1、latest不变。
[完整训练报告](V09_HANDNOTE_TRAINING.md) · [前次原因诊断](V09_GLYPH_DIAGNOSIS.md) · [语料操作](PERSONAL_HOMEWORK_CORPUS.md)。

- 两个随机种子，旧目标续训/修正先验/字体辅助后微调三路线；8个真实CUDA阶段，共6000次参数更新。
- 真实监督仍只有数字1的20 Train/5 Validation。原版本94条数据不变，Sol标注直接准入、不逐条重复识别。
- 从Handnote个人轮廓导出174张辅助图、73类；75个原个人字符中的逗号/句号4轮廓由既有度量规则排除。
  辅助图不是新增真实扫描，也不把v1/v3重复、v5尺度派生和基底回退混入真实样本数。
- Sol另封存12个数字1，来自phy1第16/26/46页；候选冻结后才评估。同一独立Holdout文档的新页面，
  不是新文档或人工真值；没有加入训练、修改Corpus或要求用户重复确认。
- 原试验64图中40图没有深色像素；修先验与辅助路线各128图中为0。新测试形状代理距离分别0.1027/0.1014，
  差距小且跨种子不一致；辅助路线有额外1200步预算，尚不能证明其稳定优于修先验或几何扰动。
- 字符1清晰度改善但仍有软边、断裂/钩形；0/2/x/a/b/g/y抽查自由生成仍失败。73类辅助训练不等于73类可用。
  **有限研究继续Go；生产、连续英文、数学整页、SFT/RL仍No-Go。**没有发布模型或改变默认Profile。
- 完整Python3.12测试187通过；补充评估模块7项通过。初次HTTP沙箱限制已在允许loopback后解决；
  无生产代码/依赖改动，原件、审核、旧权重和latest保护检查通过。未声称再次完成双Python版本验收。

主输出：`workspace/runs/20261118T161319497530Z-handnote-assisted-training-study-9901e784/`。
PDF为2页研究对照，逐页PNG在rendered/pages/；非目标字符失败图、训练曲线在comparisons/。
所有8个训练run、原始生成图、模型权重与完整评估路径见上述训练报告。

下一步优先扩充少数字符的真实覆盖并做联合训练，评估明确字形参考条件与类别保留；不投入RL。
Studio继续为http://127.0.0.1:8769，“Sol清洗已采纳”无需再次人工确认。已有四学科CPU作业基线保留。

---

以下v0.8及更早状态保留用于追溯，其模型路线建议已由v0.9决定暂停。

# 当前状态：v0.8证据评估完成，个人训练No-Go

当前研究分支 `experiment/english-evidence-v08`；稳定main、v0.6.0包、CPU Runtime、personal@v1和正式latest均未改变。[完整v0.8 Evidence-Based Go/No-Go](V08_GO_NO_GO.md) · [英文基准协议](ENGLISH_BENCHMARK_V1.md) · [模型调查](V08_MODEL_INVESTIGATION.md) · [研发日志](V08_DEVELOPMENT_LOG.md)。

- 固定120行英文：96 development、24独立holdout，含16行smoke。v1/v3各120行实际渲染，此条件下像素全部相同。
- DiffBrush官方32组真实对照；全基准240请求中236生成、4次明确不支持字符；参考fit消融32生成。输入完整，原图至PDF保真，主要内容异常已存在于原始模型。
- DiffusionPen官方权重+5本人扫描词，16文本×2种子真实CUDA推理成功。它仍按词生成后组合，未训练。
- Tesseract与真实TrOCR共同评测588张图；同smoke的TrOCR CER：字体5.14%、DiffBrush32.78%、DiffusionPen22.57%，不是真实人工错误率。16行本人扫描校准也显示识别偏差。
- 72份内容先行盲评材料已生成，真实人工评分0份。没有证据给自然度或作者相似性打新分数。
- 推荐DiffusionPen为下一轮工程验证主候选；所有路线当前均不准入个人监督微调或生产。DiffInk发布词表无法完整覆盖任何一条本轮英文文本，原中文在线实验保留。
- Python3.12/3.13各141项全套测试通过（零失败、零跳过）；构建、Ruff、PDF保真、模型摘要和浏览器门禁检查通过。

交付入口：`workspace/runs/v08-evidence-analysis/rendered/homework_solutions.pdf`、`workspace/runs/v08-blind-review/comparisons/blind-review.html`、`workspace/runs/v08-evaluation-final/reports/observations.json`。下阶段先内容人工校核与词间尺度/标点验证，达标后再论证监督微调；不做从零训练、LoRA或RL。

---

以下为保留的v0.7历史状态，不代表已完成v0.8个人训练。

# 当前状态：v0.6 已交付，v0.7 已完成真实 CUDA 实验

继续原仓库和 Agent Skill，没有重建项目。历史审计见 [CURRENT_STATE_AUDIT.md](CURRENT_STATE_AUDIT.md)；v0.4.0 的原证据保留。稳定 `main` 已合入 v0.6，当前研究位于 `experiment/diffink-cuda-v07`。CPU 发布标签 `v0.6.0 → 2bcb260352dd4fee177e9cc8ce9179201bf0b5f3`；研究源码提交和每次执行的来源详见 run.json。

| 阶段 | 实现 / 实测 | 状态与下一步 |
|---|---|---|
| Phase 0 | main、功能/实验分支；不可复用 run ID；代码/依赖/Profile/种子/硬件/权重摘要；旧路径兼容；原子 latest；2,213 旧文件保护；DVC 评估 | 已完成。干净 wheel 安装重放成功；未上传私有数据或部署 DVC |
| Phase 1 | 固定 60 例、两条评估路径；60/60 预期通过，54 页和 6 例明确拒绝；真实 Agent smoke 8；英文 OCR 10 | 基准实现完成；独立全量解题判卷、数学转写、作者自然度评分待完成 |
| Phase 2 | RTX 5060 Laptop 实测；固定 DiffInk 真权重推理、显式在线输入 Adapter、轨迹/PNG/SVG、同种子重放；另成功运行 Handnote DiffBrush 个人扫描参考英文生成 | 真实 CUDA 复现完成；DiffInk 不支持一般英文，DiffBrush 两候选内容未过关，均未准入生产 |
| Phase 3 | 区分扫描、估计轨迹、真实轨迹；Adapter 验证在线文本/字符边界 | 未训练。缺少个人真实在线数据，先完成英文基线独立评估，不启动监督微调或从零训练 |
| Phase 4 | 两个真实英文候选，独立 OCR 内容筛选；D3PO / Diffusion-DRO 方法核对 | 没有合格候选，没有神经偏好集；未做 RL，未延长旧 bandit |
| Phase 5 | font-cpu 默认；保留 Agent、Profile、每题一页、背景、数学来源、PDF/PNG；实验命令隔离 | 生产 diffink-experimental 仍明确拒绝；未发布模型、未改变默认 Profile |

CPU 的正式五题 PDF/PNG 与 v0.4.0 对应交付逐字节相同；本轮没有证据证明默认字迹自然度已提升。默认仍 `personal@v1`，研究对照显式使用 `personal@v3`。

## 实际质量与工程结果

- 全套测试在 Python 3.12、3.13 各 **134 passed**，无跳过；Ruff 检查与格式检查通过。v0.6 发行时各 125 项，新增 9 项为实验输入边界/保留末字/失败指针测试，不冒充真实推理证据。
- 固定 benchmark 的 10 个英文用例，实际 Tesseract CER **7.56%**、WER **42.71%**。60/60 是预期行为匹配，包括应拒绝的 6 例，不是 60 道题内容全对或自然度全合格。
- DiffInk 20 步真实推理约 **0.76–0.85 s**，峰值分配显存 **1.05 GB**；只复现了中文在线前缀续写，同接口重放轨迹/PNG 摘要完全一致。中文 OCR 对参考与生成都很差，不足以验收模型。
- DiffBrush 50 步、实际本人扫描参考，两个种子约 **1.62 s/行**、峰值分配显存 **1.84 GB**。同句 OCR CER：字体 **7.69%**、模型 **23.08% / 20.51%**；仅一个短句，不能推广为总体质量。两候选均 held，原图有边界墨迹和词尾异常待复核；同 seed42 重放 PNG 一致。
- 作者风格相似性、自然度、完整数学独立转写准确率仍未评分；不以测试数、OCR 或代理奖励替代这些指标。
- 2,213 个旧文件摘要未变；生产无 torch；latest 仍指向 `v06-five-final`。失败、模型和基准没有覆盖最近正式作业。

[Phase 0/1 验收与包对应关系](evidence/v06-2026-11-18/README.md) · [CUDA 实验验收及路径](evidence/v07-2026-11-18/README.md) · [DiffInk 复现](DIFFINK_CUDA_REPRODUCTION.md) · [DiffBrush 对照](DIFFBRUSH_CUDA_COMPARISON.md) · [工作区契约](WORKSPACE.md) · [基准协议](BENCHMARK_V1.md)。

下一里程碑：优先扩大英文生成的独立内容/边界评估、解决漏写重复，再做本人风格盲评；DiffInk 需先解决英文词表与真实在线输入需求。满足这些条件后才决定个人监督微调。v0.8 个人训练/偏好优化仍未交付，不能把已完成的 CUDA 实验称为完成训练。
