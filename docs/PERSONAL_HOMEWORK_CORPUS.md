# Personal Homework Corpus v1.1

本地研究模块 `research/corpus/`，不属于正式 CPU wheel 的必需依赖。
操作入口保持在现有仓库，所有实验均生成 `workspace/runs/<run_id>/`。
原件、裁剪、审核数据库和模型不进入 Git，不上传外部识别服务。

## 使用

```bash
cd /home/klmn/handwritten_solutions_skills
export PYTHONPATH=.:src

# 查看当前导入量和候选状态
.venv/bin/python -m research.corpus.cli status

# 本机审核界面；浏览器打开 http://127.0.0.1:8769
.venv/bin/python -m research.corpus.cli review

# 继续导入某文档的指定页（页码从 1 开始）；省略 --pages 导入全部页
.venv/bin/python -m research.corpus.cli import source_pdf/source_pdf_cal3.pdf \
  --batch document-985ed6df69cf0632 --course cal3 --pages 125,126 --propose-lines

# 每次生成新的不可覆盖数据版本，零合格样本也明确出具报告
.venv/bin/python -m research.corpus.cli export
```

新文档须指定真实采集会话；不清楚会话时按完整原文档分组。**不能为了改变留出结果
而换 batch**；重复原件的首次分组保留。跨文档同一会话应使用相同 batch。
同一原件已有 DPI/处理配方时拒绝改配方覆盖。本轮 PDF 试点为 150 DPI；
以后提高识别分辨率需要增加版本化派生图支持，不能覆盖现有页图。

已收到的7份 PDF 共831页均已归档；当前已渲染其中13页做试点，另有7张旧采集图。
这不表示所有831页已经清洗或转写。初始版本及失败运行保留。

## 审核与涂改处理

先选择页面，再点原图上的区域或右侧候选列表。逐项对照原图；点击识别器按钮只是
复制候选文本，**不会确认标签**。英文拼写、语法和数学错误均照实保留，不能按答案修正。
填写内容类型、书写者、墨迹质量、缺陷及隐私状态，勾选实际核验声明后才确认转写。
LaTeX 由本地 KaTeX 预览；预览正确不代表实际内容正确。

不熟悉LaTeX时，先看候选卡片的公式预览，点击“使用候选”填入草稿。公式候选会把
当前内容类型切到数学表达式；仍需作者确认书写者、质量、隐私和实际内容。重复候选合并显示，
原始解码和来源运行仍保留。“刷新候选”只更新识别列表，不覆盖当前转写或草稿。

没有合适候选时，点“识别当前区域”；“补全本页公式候选”只处理本页尚无非空UniMERNet
结果的待审核数学/未知区域，跳过已确认、已排除及已分类英文区域。后台本地CUDA推理时仍可
切换区域、继续审核；页面刷新后会恢复任务查询。任务失败会显示原因，已有候选仍可使用。
每个区域使用固定权重做一次贪心解码及两次固定种子采样（902/903/904，温度0.35/0.65），
最多得到3个不同候选；相同结果只显示一次。采样不是第二个识别器，候选一致也不证明正确。
本地模型或独立环境缺失时识别按钮禁用，不影响查看旧候选和原CPU流程。

常用公式输入辅助提供分数、根号、上下标、积分、求和、箭头和矩阵模板；选中文字后点按钮
可包入结构，需替换占位符并对照预览。中文、印刷混排、跨行大区域可能得到错误公式，优先框选
完整的单行或局部公式再识别，不能将识别器猜测当作原文。已确认标签不会被任何识别任务改写。

命令行原有recognize支持 `--candidates 3`（仅UniMERNet）；默认仍为1，旧命令兼容。

“确认转写并下一条”会在保存成功后切换下一个区域。缺少审核人、内容分类或核验声明时，
确认按钮旁会显示具体原因，不会提交。即使表单尚未填完也能切换区域：未提交内容先保存在
本机浏览器草稿中，切回来或刷新会恢复；切换本身不替你提交或确认。

| 实际区域 | 处理 | 生成训练 |
|---|---|---|
| 完整、清晰、本人笔迹，实际转写已核验 | `verified` + `clean`，无缺陷 | 满足其他门禁后可导出 |
| 有划改但能确定内容 | `verified` + `readable-correction`，保留缺陷 | 不进入干净字形训练；保留识别研究价值 |
| 遮挡/重写后字形无法确定 | `needs-review` 或 `rejected`，`unusable` | 排除，不猜标签、不补笔画 |
| 批注、印刷、照片、混合区域 | 标记来源，必要时拆分 | 非纯本人区域不导出 |
| 同页仅局部污染 | 框选完整的干净行/词/符号；保留整页原件 | 不必丢弃整页 |

自动像素检查仅提示彩色墨迹、边缘墨迹、长线、密集/过淡区域，**不是可靠的涂改检测器**。
长线可能是分数线、根号线或网格；红色也可能是本人正常书写。不得据此自动擦除。
背景估计、墨迹视图是可追溯派生图，不能当作已恢复的真实笔画；不做生成式补全。
无法分离的连笔保留为词/行，不能强切字符。复杂公式同理。

拖动框选新区域；“拆分当前区域”允许在父框内画多个子框，点完成后提交；合并需勾选
同页多个区域。操作原子执行，父区域保留为 rejected，子区域全部重新核验，不继承旧确认。
服务端每次修改有 revision，过期修改返回冲突。未提交草稿在本机浏览器保存；
“保存待复核”后写入 SQLite，可关闭服务后继续。清理浏览器数据会丢失尚未提交的草稿，
不会丢失已经保存到语料库的标注。

若仍在使用修复前的页面，请普通刷新一次，不要清除浏览器站点数据。旧版草稿键兼容保留；
如浏览器询问是否离开，可先复制正在编辑的转写作备份。不同修订的草稿不会静默覆盖新记录。

## 存储与格式

默认 `workspace/datasets/personal-homework/`：

- `corpus.sqlite`：事务/WAL 数据库，records + 追加式 events。
- `assets/<hash前缀>/<SHA256>.<扩展名>`：原件、彩色页、估计背景、处理墨迹、裁剪。
  临时写入、fsync、硬链接原子发布；已有相同摘要只校验复用，资产设为只读。
- 版本化导出位于 run 的 `reports/dataset/`，含 samples.jsonl、review-snapshot.json、
  pending.json、manifest.json、split-plan.json、near-duplicates.json、readiness.json。
  快照包含当时全部候选及识别结果，训练清单只包含符合门禁的条目。

`document_id=d-<原文件SHA256>`；page_id 来自原文件摘要和0基页号。
页记录另存实际像素摘要、尺寸、DPI、PDF MediaBox/CropBox/rotation、PDF↔栅格矩阵。
图片记录 EXIF 方向操作及原尺寸；未知扫描 DPI 为 null，绝不编造。
PDF 文字层只作为候选，扫描原件不假定具备真实在线轨迹。

样本包括：sample_id、document_id、page_id、source_hash、image_crop、crop_sha256、
bounding_box、content_type、candidate_text、verified_text/verified_latex、label_source、
recognizer_name/version（初始 null；实际结果在独立 recognition 记录）、confidence、
review_status、writer_provenance、preprocessing_transforms、dataset_split、notes、parents、
revision、privacy_checked、quality_status、defects、quality_triage、review_attestation。
坐标框是原始展示页栅格像素 `[x0,y0,x1,y1)`，不是经过缩放的识别器内部坐标。

识别结果独立保存 engine/version、精确输入摘要、候选转写、tokens/TSV 原输出、
模型摘要、来源运行、耗时和未校准置信度。自动识别不可写入 verified 字段。
LaTeX 规范化只移除两端空白；括号/环境平衡检查不求解、不改变上下标或符号语义。

## 数据划分与准入

初始按文档/会话的固定摘要顺序分配至少一个 Validation、一个 Holdout；剩余为 Train。
此后新文档按固定种子追加，旧组不重排。旧校准图只能进入 Train，不能冒充独立评估。
v1 的纯哈希桶碰巧把7份新文档全分到 Train，检测后在任何个人训练前冻结 v2 分组方案，
生成新导出 **v1.1**；旧 v1 和 No-Go 收据原样保留，未通过反复改种子选择成绩。
本轮 physics1 为 Holdout、applied statistics 为 Validation；完整映射见 split-plan.json。

完全重复文件按摘要去重；同页所有裁剪留在同组。不同文档近似页用 dHash 初筛并保守隔离，
存在误报，不能把相似字格自动判为同一真实样本。本轮有1对疑似重复旧字格，等待复核。
目前尚无界面内解除近重复隔离的流程，不能绕过隔离直接训练。

训练类别：English Words & Lines、Mathematical Symbols、Mathematical Expressions、
Page Layout、Verified Glyphs。导出同时统计已核验覆盖率和待复核比例。
所有样本还必须是 personal-real、非生成、纯本人、隐私已检查、干净无缺陷。
公开数据、基底字体、合成夹具、模型输出不会转换成真实个人来源。

## 识别与生成器环境

识别运行参数：

```bash
.venv/bin/python -m research.corpus.cli recognize --engine unimernet \
  --ids /path/to/sample-ids.json \
  --python workspace/environments/corpus-neural-v09/bin/python \
  --checkpoint workspace/checkpoints/unimernet-tiny-v09 \
  --source workspace/sources/v09/UniMERNet
```

ids 文件仅为 sample_id 数组。worker 只接收图像及摘要，不接收预期答案。
TrOCR 使用同一入口和 `--engine trocr`；Tesseract 不需要神经环境。
OCR 与生成训练环境分离，依赖分别固定在 `research/corpus/recognition.lock`、generator.lock。
CUDA PyTorch 使用官方 cu128 wheel 索引。UniMERNet 官方源码和本地 KaTeX 资源位于
workspace/sources/v09，来源 commit 和模型摘要见 `v09-recognition-preparation/reports/`。

训练协调器仍在 CPU 环境运行，先验证数据，准入后才启动单独的 CUDA worker：

```bash
.venv/bin/python -m research.corpus.train <run>/reports/dataset \
  --python workspace/environments/corpus-generator-v09/bin/python
```

当前最小原型是约40万参数的64×64条件 VAE，输入有限单字符、writer 条件、上下文和潜变量。
不是连续英文生成器，也不是在线笔顺恢复。门槛为每个字符20 Train / 5 Validation /
5 Holdout 清晰核验实例、至少3独立组；该门槛只是启动研究的下限，不保证效果。
上下文在没有可靠量测时明确为零。训练代码保存初始/最佳/最后权重、优化器、曲线、
种子和来源，最佳模型依据 Validation；仍必须独立检查身份、自然变化、记忆与风格。
数字1已达到上述研究门槛并实际完成GPU训练；输出偏淡模糊，生成质量仍No-Go。
其余字符、连续英文、完整数学公式尚未准入，不能以一次单类试验宣称完整个人模型已可用。


## 用户授权的Sol清洗直接准入

用户明确要求直接使用GPT-6.2-Sol xhigh的识别结果，不追加主Agent逐条复读或人工确认。
该策略只对显式授权的原始PDF及干净单符号生效；普通OCR/UniMERNet候选仍不能自动准入。

- `model-accepted`：Sol明确判定完整、本人、单符号、无隐私信息、无质量或歧义标记；
  文本存`model_text`，`verified_text/verified_latex`与人工核验声明保持空值。
- `model_review`记录模型/推理级别、证据文件摘要、逐条证据摘要、标签/裁剪绑定、授权策略摘要。
  策略保留用户原始指令及文档范围。`human_review=false`，不是人工真值。
- 多公式组合、涂改、遮挡、重写、裁切及不确定字符默认拒绝。不能借上下文猜测或按参考答案补字。
  本轮单符号入口不接收含上述风险的整体区域；如存在独立干净子区域，另存可追溯的新框。
- 入库仍自动检查原页、原件和裁剪摘要、父区域修订/坐标、重复物理墨迹和固定分组。
  这些是完整性检查，不是再次付费识别。不会自动填成人工审核通过。
- 普通审核API不能手工伪造model-accepted。用户仍可纠正或拒绝模型记录；旧模型证据留在事件历史。
  待核验队列不再显示已采纳模型条目，单独队列“Sol清洗已采纳”用于查看，无需重新确认。

新导出附`label-policy.json`及摘要；统计分别报告人工与模型标签数量。训练协调器与独立worker
均验证授权和证据，策略不自动扩展到其他作者、文档或生成图像。当前94条中25条人工、69条模型。

可靠符号准备入口（先生成候选，不用公式字符串位置猜字）：

```bash
.venv/bin/python -m research.corpus.cli prepare-symbols --parents parent-ids.json
.venv/bin/python -m research.corpus.cli publish-symbols --selection sol-proposals.json \
  --annotator gpt-6.2-sol-xhigh --model-evidence sol-proposals.json
```

首次准入前需通过`label_admission.authorize`保存用户的实际授权及文档清单；未授权时上面的
model-evidence路径会被拒绝。保留完整Sol证据，不以代码生成的占位标签冒充模型工作。

本次训练采用显式`--preprocessing dark-ink-v1`：原图不修改，64×64学习视图只抑制浅纸纹/格线，
保留深色笔画的软边。此模式不适用于浅色或彩色笔迹；默认仍raw-gray，不自动擦除结构线。

```bash
.venv/bin/python -m research.corpus.train <dataset-dir> \
  --run-id <new-unique-id> --epochs 120 --preprocessing dark-ink-v1
.venv/bin/python -m research.corpus.evaluate_generator <training-run> \
  --font profiles/realism-studio/profiles/personal/versions/v1/font.ttf
```

两条命令都由CPU协调器启动独立Torch worker，不向生产Runtime添加Torch或Handnote训练环境依赖。
目前评估命令仅支持本次数字1试验。16张固定种子采样、真实Train/Holdout、个人字体和非学习扰动
对照保存在新run；没有独立身份/风格评分前仍为held。字体对照为Pillow直接渲染personal@v1
字形，不冒充整页Handnote排版对照。像素不同不证明自然变化或未记忆。


## 可选Handnote辅助训练对照

新增独立模块font_auxiliary与neural_study；默认Runtime和原train命令不受影响。
辅助图通过Handnote轮廓与FontTools度量生成，标为personal-font-derived/auxiliary-train-only，
保存Profile/font/sources与逐图摘要；不进入真实扫描统计或Holdout。不得把字体标签冒充人工确认。

在CPU环境用`export_personal_font(profile_store().resolve("personal@v1"), new_directory)`
创建不可覆盖辅助数据。每个新实验由CPU协调器核验后启动现有隔离Torch环境：

```bash
PYTHONPATH=.:src .venv/bin/python -m research.corpus.neural_study real-prior \
  --dataset <frozen-dataset> --auxiliary <font-auxiliary> \
  --initial-run <previous-sealed-training-run> --seed 902
```

stage包括real-control、real-prior、font-pretrain、font-finetune。最后一项必须读取真实font-pretrain
封存权重。每阶段保存权重/优化器与协议，优化器显式重置；仅允许已有足够独立真实覆盖的数字1进入
本次真实微调。73类字体辅助输出不自动取得这些类别的真实训练或生产资格。

先封存候选再运行study_evaluation；模型摘要、测试原页/裁剪字节、Sol标注、边界及重复墨迹均校验。
评估只读取冻结权重，模型图像、字体派生图、非学习扰动和原始扫描始终区分。
本轮具体命令、数据、运行路径和结果见[V09_HANDNOTE_TRAINING](V09_HANDNOTE_TRAINING.md)。

## 全个人字符参考条件候选

用户进一步授权全部75个人字符联合实验。独立`reference_study`/`reference_generator`实现
空间字形条件的小型CVAE，`reference_data`保留178字体派生轮廓及41真实Train/13Validation，
不覆盖旧`train`与`neural_study`协议。稀疏类进入实验不改变旧生产资格门槛，也不把字体视图记成扫描。
新Sol验证/留出包各30条单字符，仅作评估；来源为PDF原页与原像素裁剪。Corpus记录保持原样。
`reference_test`拒绝未冻结的Holdout调用，并重渲染原PDF核对页/裁剪；`reference_evaluate`
保存600张原始神经图、逐字来源及5页对照，每页15类、每类字体参考+7个展示候选（全部8个PNG另存）。
全部环境、分阶段真实更新、失败样本和可复现命令见[V09_ALL_CHARACTER_TRAINING](V09_ALL_CHARACTER_TRAINING.md)。
候选仍held；不激活Profile、不改latest、不把模型图或Sol结论伪装成人工真实样本。


## 已封存清洗批增量数据版本

新增`research.corpus.dataset_extension`；接受Sol按已授权策略制作、封存并通过原PDF重渲染的Train/Validation包。
300DPI page view以独立ID和原PDF坐标保存，原Studio150DPI records及审核事件不变。
同页的不同DPI裁剪以PDF物理区域检测重叠，不能重复计为新增笔迹。原color crop不修改；学习视图另存。
输出完整dataset、views和extension证明，逐条保留parent标签/原记录/所有旧像素；Holdout包拒绝加入。
已在父数据里的同Validation可只读重放评估，不能再次准入。标签、分组、授权摘要均绑定manifest。

```bash
PYTHONPATH=.:src .venv/bin/python -m research.corpus.dataset_extension   PARENT_DATA --train-package SEALED_TRAIN_RUN --validation-package SEALED_VALIDATION_RUN
```

省略validation参数可只增Train。续训必须显式`reference_study train --allow-data-extension`，
checkpoint所依赖数据必须精确匹配extension父版本；不能替换旧数据后伪装同一次续训。
本轮最终339真实Train/52类+71Validation+178字体辅助/75类，model标签与human标签仍分开。
`prior_support_weight`默认0，实验可加入同类别完整Train距离场约束，绝不读取Validation/Holdout支持图。
详见[V09_EXPANDED_CORPUS_TRAINING](V09_EXPANDED_CORPUS_TRAINING.md)；模型输出依旧是offline image generation。


## v0.10.5 新配对材料与整词框

见[V0105_DATA](V0105_DATA.md)。新62真实区域保留原件/300DPI坐标/高置信Sol来源，C仅封存。
独立候选Corpus和版本化dataset不覆盖旧Studio数据库，印刷公式仅作辅助，不会自动修正实际书写标签。


## v0.10.6 整词派生监督

word_style_data从已审词边界反推原裁剪像素；524候选经Sol逐项复核弃12，512准入。保留原始PDF/hash/页面/父框/局部框/实际标签/split，不补墨、不拆字符。标准词图独立统一缩放512x128，不能据此声称学习绝对词间距。少词集58Train/6开发是原数据子集，不增加真实样本计数。


## v0.10.7 配准派生监督

512词原数据不变；373Train/268标签通过无折叠/边界/形状准入，坐标监督可反查原词及PDF。
配准场是derived-registration，配准后字体不是real-personal；两组神经模型只从Train坐标监督学习。
词级输入保留完整ASCII且拒绝静默截断；60旧开发和C均不进入训练。详见[V0107_GO_NO_GO](V0107_GO_NO_GO.md)。
