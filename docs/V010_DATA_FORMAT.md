# v0.10 连续语料与复现接口

全部私有run均位于 `workspace/runs/<不可重复run_id>/`，终态不能修改；实验不更新latest。
原始PDF/图像、私有标签和checkpoint不进入Git。CPU Runtime与研究Torch/HTR环境分开。

## 数据

`context_dataset.py`读取Sol保守清洗包的 `reports/proposals.json`，检查accepted/high-confidence/
完整裁剪/本人来源/无质量缺陷/300DPI/固定文档分组、PDF原件摘要与重新渲染原像素一致。
不确定、多公式无法归属、涂改遮挡直接拒绝。不强切字符，不修补缺笔。
缺失complete_crop时拒绝；可由同一清洗Agent在独立新run提供精确绑定父JSON摘要的完整性补充声明，
不得回写旧run。该声明不是人工验收。

导出 `reports/samples.jsonl`、manifest、audit；每行包括sample_id、document_id、page_number、
source_pdf/source_hash、crop/crop_sha256、bbox、dpi、content_type、normalized_label、
dataset_split、writer_provenance、label_source、model、reasoning_effort、human_verified、
source_package/source_package_sha256与原提案字段。不修改原SQLite。
英文仅去首尾空白；LaTeX不作语义改写，括号/环境平衡检查不代表正确转写。
数学与英文保持不同类型；扫描像素是offline来源，轮廓不是在线真实轨迹。

训练视图 `context_training_data.py` 保存原裁剪链接、灰度墨迹阈值、统一尺度/偏移、reference/
target/SDF数组摘要。原始300DPI颜色图保持不变。字体是辅助条件，不能记作新增真实样本。
Train/Validation按原文档分组；新区域Holdout来自历史已曝光phy1，清楚标注限制。

## 模型与评估

`context_model.py`是整块区域生成器；`context_train.py`独立GPU worker实际更新参数。
每臂固定800步，logs/checkpoint-0200/0400/0600/0800.pt含优化器/RNG，初始状态另存；
reports/request.json、environment.json、training.json、curve.json固定环境/数据/配置/实际更新证据。
主run reports/context-training-runs.json定位全部候选；没有任何模型获准生产。

命令入口（先设置`PYTHONPATH=.:src MPLCONFIGDIR=/tmp/hwsol-mpl`）：

```bash
.venv/bin/python -m research.corpus.context_render
.venv/bin/python -m research.corpus.context_dataset --help
.venv/bin/python -m research.corpus.context_training_data --help
.venv/bin/python -m research.corpus.context_train --help
.venv/bin/python -m research.corpus.context_evaluate --help
.venv/bin/python -m research.corpus.context_comparison <sealed-evaluation-run>
.venv/bin/python -m research.corpus.context_recognition <sealed-comparison-run>
.venv/bin/python -m research.corpus.context_report <running-main-run> <comparison-run> <diagnosis-run>
```

每次操作新run，不允许重用已封存目录。报告命令目标是尚在构建的主run。
TrOCR使用本机完整`trocr-base-handwritten-v08-xet`，请求只有id/image/hash，不能提供目标文本。
CER/WER为识别观察；空结果/弃答与真实错误分开。每个生成PNG有固定seed和模型摘要。
`context_review.build_review`生成离线匿名HTML：内容准确后才允许风格评分，浏览器本地保存、JSON导出/恢复。
匿名key只在私有reports，不嵌入HTML。导出评分不得当作真实手写训练样本。


## v0.10.1 扩展（不修改旧版本）

`context_aligned_data.prepare`保留原始300DPI裁剪及`raw_transform`，新增`alignment`/`alignment_rejected_reason`。
完整词对齐只生成派生视图；`arrays.npz`含reference/raw/aligned及对应SDF、split、admitted。
三个训练臂共用admitted Train，所有Val都保留；reference评估布局不使用真实目标坐标。
`context_content_train`保存content标签协议、教师/data/sample/warmup/resume摘要和每500步优化器/RNG。
恢复必须匹配标签协议，不允许将错误BOS训练的优化器混入修正实验。
`context_content_evaluate.freeze`先封候选再读取新区域；`context_evaluation_ink`的浅色回退仅用于评估显示，
target-transforms记录黑白点，原图及训练阈值不变。识别模块用`--teacher-used`显式区分TrOCR独立性。
`aligned-content-ink/1`是确定性整块SDF解码器，不声称在线轨迹、随机风格、自由词距或数学生成。


## v0.10.2 去灰影扩展

`ink_geometry.signed_distance`抽出为仅NumPy/OpenCV依赖的同一计算，原`reference_data`继续导出别名。
`clean_ink_train.train(control|distance|mass)`固定从认可checkpoint起始；自一致mask由当前预测派生，
不是新真实标签。所有原始/字体/派生正则来源在request中分开，教师保持冻结。
冻结接口支持显式protocol摘要；评估`contrast_views={name:3}`把零更新显示增益记入每个候选的provenance，
关闭时省略该项。它不修改模型权重、生产渲染、原始图或Profile。
作者JSON沿用`context-human-review/1`，匿名id与图片SHA绑定，导出存放`source_pdf/review_exports/`，
后续复核另建新run，禁止把评分写回已封存交付目录。
