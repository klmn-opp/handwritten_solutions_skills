# Benchmark v1 (1.0.0)

固定数据在 `src/hwsol/assets/benchmarks/v1/cases.json`，同目录 SHA256SUMS 验证版本。60 个原创 MIT 用例，smoke 8、core 24、stress 16、holdout 12。题号、类别、复杂度、来源、原题、输入类型、参考解答、数值核对、纸张、预期页数/结构/拒绝原因均独立于渲染后端。数学参考由解析推导和明确的有理数检查核对；这些检查不等于所有推导经过外部专家审核。

覆盖线性代数、微积分、统计、物理、英文、布局与错误路径。48 例开发集包含 42 个应渲染用例和 6 个应拒绝用例。unsupported cases/align、未知命令、缺字、纵向/横向溢出不算成功出页；只在错误类型/说明匹配时计作基准预期通过。不会修改难例使其变成简单例。

## 两条路径

```bash
hwsol benchmark list --suite all
hwsol benchmark run --suite development --profile personal@v3
hwsol benchmark run --suite holdout --profile personal@v3 --purpose evaluation
hwsol benchmark prepare-agent --suite smoke
# 读取返回 run 的 reports/questions.json，只读取其中的原题源，不读取参考解答。
hwsol benchmark run --suite smoke --profile personal@v3 \
  --packet workspace/runs/<packet_id> --submissions <agent-submissions.json>
hwsol benchmark ocr --run workspace/runs/<benchmark_id>
```

也可使用 `hwsol-benchmark`。每次实验及每个 case 都有独立 run ID，失败保留，重复运行不覆盖。每例在新 Python 进程中执行，记录实际总墙钟时间、渲染耗时和进程 RSS 高水位。RSS 包括解释器及依赖，不称为单个函数增量内存。基准合集 PDF 一题一页，不更新日常作业 latest.json。单例失败不会跳过后续 case。

render-only 从版本化参考解答生成；end-to-end 的 packet 只包含原题 text/image/PDF 和问题索引，不包含参考解答。Agent 独立提交后才开始渲染。缺少提交标为 not_run；不合格源摘要、空解答或失败自检标为 solver_failure，渲染器未执行。渲染错误单独标为 renderer_failure。未经独立判卷，提交不记作 solver 正确；工程通过不代替解题正确率。

提交格式：

```json
{"submissions":[{
  "case_id":"smoke-matrix-inverse",
  "source":"solutions/smoke-matrix-inverse/question.txt",
  "source_sha256":"<packet 中实际文件摘要>",
  "agent":"<实际执行者/模型及版本>",
  "solution_markdown":"<实际推导与结果>",
  "verification":{"status":"passed","evidence":"<实际复核过程>","checks":[]}
}]}
```

目前不嵌入另一个求解模型，也不会把 reference_solution 自动复制到 end-to-end 作为 Agent 成功证据。题库作者进行的演练不是独立盲测；完整解题准确率需要新的独立 Agent 会话/评审。

## 度量与独立性

- 英文：可执行已安装 Tesseract，直接识别实际 PNG 的正文裁剪；不向 OCR 传入提示词或预期转写。固定 psm=6、eng、3 倍图像，空白归一化，保留大小写及标点。报告 CER/WER 和漏/重复词数量差。仅英文用例，OCR 误差不等于作者自然度。
- 数学：可见字符及结构记录、应有构件、回退统计可直接计算；数学符号/结构准确率需要独立转写或人工核对，默认 null。不能把 MathText 自身输出当成独立识别结果。
- 字形：个人/继承/外部/程序几何分别计数，缺失项单列。
- 风格、自然度、可读性：独立评分接口，不生成虚构分数。`benchmark observe` 绑定输出 PDF SHA、评审者与盲评条件，终态原运行不修改。
- 布局：实际墨迹边界检查、页数、包围盒重叠候选。公式内部交叠排除；候选数量是代理量，不能冒充真实碰撞率。
- 资源：每例新进程 RSS、耗时；CUDA 试验另记录实际 GPU 指标。

独立观察格式为 `{"observations":[{"case_id":"...","artifact_sha256":"...","method":"human-blind","evaluator":"...","reference_visible":false,"english_transcription":"...","symbol_accuracy":0.0,"structure_accuracy":0.0,"style_similarity":0.0,"naturalness":0.0,"legibility":0.0}]}`。评分使用 [0,1]；未测量的字段省略，不填 0。观察接口验证归属和格式，不可能自动证明评审者真的盲评。

holdout 是固定、可审计的隔离集合，不是保密考试。CLI 拒绝用它执行 purpose=tuning；所有题库条目禁止 model_training。先冻结开发实现，再运行 holdout。若据 holdout 修复渲染器，该版本 holdout 就不再是新的泛化证据，必须另建版本而不能改写旧结果。


## v0.8 英文专项基准

新增独立的120行英文基准，96 development / 24 holdout，含16行smoke子集。两识别器和人工校核严格分开；详情见 [English Handwriting Benchmark v1](ENGLISH_BENCHMARK_V1.md) 与 [v0.8 决策报告](V08_GO_NO_GO.md)。原60题数据和结果保留。


## v0.9 历史作业识别与语料质量试点

固定60题仍用于渲染回归。本轮另在用户7份PDF共831页中选择10页，连同7张旧采集图建立
可追溯语料试点；两者不能混同。4学科CPU基线是reference replay，不是新Agent解题或神经生成成功。

识别记录共58条（TrOCR21、UniMERNet22、Tesseract15），不输入期望答案。
原始图/坐标/模型摘要/转写保留；人工真值0条，真实CER/WER/Exact Match保持null。
助手可指出负号遗漏、x/π替换和语法改写反例，但不能把这些观察冒充独立人工错误率。

语料按原文档/会话冻结划分，旧采集校准图不作独立Holdout；干净墨迹与可读涂改分离。
覆盖指标仅统计实际核验的数据，候选数与831页总数不代表可训练量。当前190候选、0合格训练样本。
试点未覆盖全部矩阵/求和/混合印刷情形，需扩充分层样本，不能把小样本模型结果外推到全库。
参见[数据规范](PERSONAL_HOMEWORK_CORPUS.md)、[v0.9报告](V09_GO_NO_GO.md)。

## v0.9 全个人字符生成实验（独立补充，不替换60题）

详见[V09_ALL_CHARACTER_TRAINING](V09_ALL_CHARACTER_TRAINING.md)。75个人码点×8固定采样，
记录空白、边界、额外分量、模板归类、二值同形与变化不足；这些是代理，不当成OCR CER/WER。
真实数据按原文档分组：新Validation30/25类，冻结后新Holdout30/30类；Sol xhigh转写不冒充人工真值。
原PDF重渲染与逐像素裁剪校验通过，同页/同文档相关数据不跨Train/Holdout。
模型、字体、几何8选1均在同一对齐条件比较；本轮神经未胜几何，不宣称自然度成功。
75类候选不是75类已有独立真实训练：71类只有字体辅助，数学结构和英文词行基准尚未接入。
600张同权重同种子GPU PNG可重放，不改固定60题与120行英文结果，也不以全196测试代替视觉质量。

本轮冻结后另做150图Sol xhigh盲转写：120清晰、22歧义、8缺陷；125个非空转写中5个不匹配目标。
它不是人工错误率，孤立符号弃答也不能统统算成错字。目标映射在独立评分前不提供给识别Agent，
原始转写/备选/置信度均保存；测试后没有再训练或更换候选。结果说明模板代理不能作为内容硬门禁。


## 扩充语料后的冻结评估

[V09_EXPANDED_CORPUS_TRAINING](V09_EXPANDED_CORPUS_TRAINING.md)补充四模型固定75×8采样，
新Validation58/40类、新Holdout57/38类；模型先冻结，测试后未继续训练或选择胜者。
两轮与修正轮同时报告形状、缺损、同形复制和变化不足，不能把0.127473的距离当成字符正确率。
375图独立Sol盲转写含Handnote/旧版/R1/R2/修正，每类方法sample0；两评审随机拆分，无重复评级一致性统计。
保留弃答、模型不匹配与人工真值区别。修正缺损标记2/75，Handnote1/75，不足以证明风格优越。
固定60题和120行原基准未改变，本次也没有重新执行其完整视觉评估；全200测试是工程回归。


## v0.10 Context Handwriting Benchmark / 1

`research/corpus/context_benchmark.py` 固定32片段及内容摘要；原创开发参考，不用于训练。
`context_homework.py`扩展4学科完整解答（线性消元、回归、反常积分、RL暂态）。
同来源/尺度/白纸/布局，Handnote、R2、repair三方法；后两者明确是逐字神经输出组合。
实际24页片段、12页完整作业，原始语义/几何/provenance可对比；37页真实同内容对照独立保存。
新上下文模型只对真实英文完整区域试验：70Train/11Val/10区域Holdout，5冻结模型各4样本。
29区域×9方法含真实参考共261图接受Tesseract/TrOCR，原输出及CER/WER/Exact/编辑类型可复核。
12整页与18英文另作Sol盲转写，人工作者相似度/自然度保持null。
不以识别器CER当真实文本错误，不以页数/形状距离/软件测试通过当自然度提高。
区域Holdout来自旧文档，fresh_document=false；不可宣称新文档级测试或在其上继续调参。
详见 [v0.10报告](V010_GO_NO_GO.md)。
