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

已收到的7份 PDF 共831页均已归档；只渲染其中10页做试点，另有7张旧采集图。
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
目前数据准入为 No-Go，完整个人训练循环与生成质量尚未实测。
