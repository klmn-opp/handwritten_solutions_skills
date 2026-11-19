# English Handwriting Benchmark v1.0.0

本基准用于 v0.8 独立英文生成评测，不替代原60题数学作业基准，也不修改 CPU Runtime。源码在 `research/english/`，固定数据为 `benchmarks/v1/lines.json` 及 SHA256SUMS；120条原创文本按MIT许可提供。

## 固定数据与隔离

八类各15条：短句、长句、连续重复字母、困难连笔组合、数字标点大小写、超长单词/边界、易混字符、作业叙述。每类01–12为development，共96；13–15为holdout，共24。smoke是每类01–02，共16，是development子集，不是额外数据。所有文本保留原大小写、标点和拼写；不因后端不支持而替换字符。每条含稳定ID、类别、复杂度、来源、约束与禁止训练标志。

holdout只准评测，不供诊断、调参或训练。当前 loader 在 diagnostic 模式拒绝 holdout/all，语料摘要变化即拒绝。文件可读不等于物理权限隔离，未来训练管线仍须使用显式允许清单；本轮没有训练。基线固定后仅评测一次留出集，不由留出成绩挑选种子或参考预处理。DiffusionPen本轮仅做smoke，不声称覆盖holdout或全120条。

## 三层证据

1. 输入与导出：保存原目标、字符编码/完整性、资产摘要、原生PNG、模型/适配器对照及PDF逐像素复核。程序拒绝、图像生成、内容正确是三种不同状态。
2. 独立识别：Tesseract eng/psm7与真实 `microsoft/trocr-base-handwritten` CUDA推理。两识别器不接收期望文本。TrOCR固定beam4、不采样、最多256新token，记录token上限事件。Tesseract记录版本及eng traineddata摘要。
3. 人工校核：先隐藏期望文本盲转写，再揭示目标，判断完整/错误/不确定；只有完整且转写一致的样本可评风格。后端与识别器结果在页面隐藏，私有解盲表另存。没有人工答卷时，人类正确率和风格评分保持null。

识别前统一采用墨迹边界、等比64像素高、20像素白边；Tesseract再放大3倍。此步骤不删除内部笔画。触边墨迹只标记possible clipping，尾部删除只标记recognizer discrepancy，不能直接判为真实截断。真实本人16行历史扫描作为识别器校准对照；历史转写尚未独立复核。超长行被TrOCR压入其标准输入尺寸时会有域偏差；IAM域识别器也可能偏向IAM训练的模型，这些因素必须随成绩说明。

CER与WER使用确定性Levenshtein对齐、微平均；Exact Match区分大小写和标点，比较前仅合并空白。输出字符替换/删除/插入、重复插入词、尾部删除、逐字符/逐词对齐。汇总按后端、类别、split与seed分开；生成失败数另列，不能从分母悄悄消失。两识别器一致与目标一致只是待人工确认候选，不是真实准确率。

## 运行

生产解释器只负责编排、字体、指标、PDF；PyTorch仅在 `workspace/environments/english-htr-v08/` 中。该环境的实际固定依赖在 `research/english/requirements-evaluation.lock`，torch/torchvision来自官方cu128索引，模型资产需按run请求摘要提前准备；CPU安装包不包含研究代码或torch依赖。

```bash
# 仅在没有同名环境时创建；本次已有环境应直接复用，不重建。
# 不得向 .venv 安装这些神经依赖。
UV_CACHE_DIR=/tmp/hwsol-eval-cache uv venv --python 3.12 \
  workspace/environments/english-htr-v08
UV_CACHE_DIR=/tmp/hwsol-eval-cache uv pip install \
  --python workspace/environments/english-htr-v08/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match -r research/english/requirements-evaluation.lock
# 固定模型资产须另按请求清单准备。

.venv/bin/python -m research.english.cli font \
  --profile personal@v1 --split development --run-id NEW-UNIQUE-ID

.venv/bin/python -m research.english.cli diffbrush \
  --python workspace/environments/english-htr-v08/bin/python \
  --split smoke --official-pairs --seeds 42 43 --run-id NEW-UNIQUE-ID

.venv/bin/python -m research.english.diffusionpen \
  --seeds 42 43 --run-id NEW-UNIQUE-ID

.venv/bin/python -m research.english.evaluate \
  --runs workspace/runs/SEALED-RUN --run-id NEW-UNIQUE-ID

.venv/bin/python -m research.english.review packet \
  --evaluation workspace/runs/EVALUATION-RUN --run-id NEW-UNIQUE-ID

.venv/bin/python -m research.english.review ingest \
  --packet workspace/runs/REVIEW-RUN/reports/reviewer-packet.json \
  --answers /path/to/actual-human-review.json --run-id NEW-UNIQUE-ID
```

每条命令使用新的不可复用ID，全部写入 `workspace/runs/<run_id>/` 的标准目录；包含research源码快照。运行中的上游不可评测；失败不得更新正式latest。模型运行默认held，评测和人工导入也不自动发布。使用既有font-cpu即为回退；研究环境和实验模块可完全不安装。

## 输出

`reports/lines.json`记录原始生成；`reports/request.json`固定源码/权重/Profile或参考样本/seed；`logs/source/`保存实际研究代码；`reports/pdf-integrity.json`记录PDF像素一致性。评测的 `reports/observations.json` 包含两识别器逐项对齐和汇总，`reports/htr.json`保留原始转写/token/耗时/显存。盲评HTML及未填写答卷模板在独立review run，真实答卷导入另建新run，不修改封存历史。

正式结果、失败样例与训练决策见 [v0.8 Go/No-Go](V08_GO_NO_GO.md)。

## v0.10.3 区域训练评估补充（独立于固定120条基准）

[本轮报告](V0103_GO_NO_GO.md)使用历史真实作业的215开发区域及6新留出区域，
不是重新跑完本页120条英文基准。6区域仅来自旧文档，曾有缩略总览曝光；不称全新文档测试。
所有候选先冻结，再评估；已打开区域后续仅作历史对照，不复用为新测试。

原512×64模型图与固定L<230包围框+8px白边视图并列保留，434输入/868识别观察。
TrOCR已作为冻结内容教师，因此不独立；Tesseract和50图Sol匿名观察另列，均不冒充作者核验。
作者4组20图候选位置均衡、先内容后风格，本轮尚无新评分。
新增核心墨色、外围灰墨和细笔/重影观察，发现加深不等于去影；三臂结果未获发布资格。
全部原图、转写、失败案例、SHA绑定和多维报告在本轮run内，正式基准数据及CPU路径未改。


## v0.10.4 连续自主训练评估

本轮A容量诊断+B/C完整数据消融见[V0104_AUTONOMOUS_REPORT](V0104_AUTONOMOUS_REPORT.md)。
10400更新、B60/C24图Sol匿名观察、B600/C700识别观察。所有评价仅开发集，不声称重新跑固定120条基准或新文档测试。
窄带轮廓显示会让灰墨指标归零，但零更新投影严重丢淡笔；只有学习后内容/风格独立通过才算进步。
新增84图视觉观察与识别转写分开，C未通过内容门禁，作者风格暂无新评分。
旧mass末7图受批大小变化影响最多1灰阶，已审计；B/C四新模型在同215输入下860图重放全等。
后续C采集文档需物理分开、封存到候选冻结后；打印模板不是个人训练样本，实际转写必须核验。


## v0.10.5 私有连续材料研究边界

新增A/B20真实六词行仅作Train，不扩充或冒充原固定基准成绩。旧25Val均开发使用，最终模型观察29/29匹配非人工错误率。
C新文档封存，尚未进入评估或调参。修正Handnote基线至300DPI，与神经参考一致；旧150DPI观察不得作为新候选优势证据。
模型转写、HTR/Tesseract差异、作者风格分开报告，见[V0105_GO_NO_GO](V0105_GO_NO_GO.md)。


## v0.10.6 整词内容与表示诊断

见[V0106_GO_NO_GO](V0106_GO_NO_GO.md)。512清洗词，6个实训/10500更新；594图双识别与120图Sol观察。广词内容失败；6词小集固定48prior仅29模型转写匹配。作者页面是内容筛选的2词子集，不是无偏基准。C未开，无新泛化结论。
