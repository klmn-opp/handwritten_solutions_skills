# v0.8 研发日志

## 初始审计与隔离

起点为 `experiment/diffink-cuda-v07@21a582b`。`v0.6.0` 实际提交是 `2bcb260352dd4fee177e9cc8ce9179201bf0b5f3`；main 为 `b2754e7`，已经合入该发布，且是 v0.7 分支祖先。新建 `experiment/english-evidence-v08`，保留 main、旧分支及标签。

已读取 CURRENT_STATE、ROADMAP、BENCHMARK_V1 与两份 CUDA 复现报告。旧 ROADMAP 含早期 v0.3 目标，本轮将增加当前决策，不抹去历史记录。未跟踪 `.vscode/` 原样保留。

`workspace/runs/v08-initial-audit/` 固定 5,074 个历史文件（个人资料、原输出、已有运行和权重）的摘要，另记录全部 src 源码、正式 latest、默认 Profile 与 .vscode 摘要。CPU Runtime 不修改，本轮代码仅放 research/english 与测试中。

## 基准冻结准备

English Handwriting Benchmark v1.0.0 包含 120 条原创英文行：8 类各15条，development 96、holdout 24，开发集内 smoke16。固定文本保留困难标点、短句、长句和边界，不因模型不支持而改写。holdout 禁止诊断调参及任何训练。字符对齐、词对齐、替换/删除/插入/重复及尾部缺失分别报告为识别器差异，真实内容错误需独立校核。

## 环境与源代码准备中的故障

HTR 首次依赖解析被 PyTorch 索引优先级阻断：所指定 NumPy 版本不在该索引。保留失败日志，改用兼容固定版本并允许在官方 PyTorch/PyPI 两源匹配；不改变生产依赖。准备记录在 `v08-model-preparation`，任务完成前不封存仍有日志写入的 run。

## 固定推理与独立评测协议

CPU 基线沿用 2423e83 的8pt、白纸、contextual，personal@v1/v3 各120条；本轮不调字体策略。DiffBrush 默认50步、float32、seed42/43、原始 Handnote 参考预处理，不从 holdout 选配置。fit 保全参考宽度仅作 smoke 消融，不据此修改正式策略。DiffusionPen 使用原始64×256单词模型、50步、五个本人扫描词、完整严格加载；词长超过官方40 token容量（含两个特殊token）明确拒绝。两个模型均不改数学渲染。

TrOCR 固定 beam4、max_new_tokens256、不采样、不接收目标文本；Tesseract eng/psm7，归一化后放大3倍。归一化只裁外部空白、等比调整并加白边。两识别器与目标一致仅为待人工确认候选，不能称为真实准确率。人类内容校核前不得开放风格评分。训练 Go 要求内容可靠性和作者盲评均有证据；本轮不设仅靠OCR即可通过的阈值。

首个综合评测过早读取未完成的 DiffBrush run 而失败，`v08-evaluation-baselines` 保留；入口已增加上游封存状态检查。官方 DiffBrush 初始化还下载了 torchvision ResNet18 初始权重，随后被完整checkpoint覆盖；后续缓存固定在 workspace/checkpoints/torch-v08，不将初始化下载当作训练。

## 实际推理、识别与工程验证

DiffBrush原始官方32次同输入对照完成；Handnote原生输出与PDF逐像素一致。development 192请求中188生成，4次明确拒绝不支持的标点；holdout 48/48生成。fit参考消融32/32生成。DiffusionPen官方EMA+完整Canine参数严格加载，五个真实扫描词条件，smoke16×seed42/43全部实际生成；无训练。

TrOCR与DiffusionPen权重均核对固定HF提交的官方LFS摘要。TrOCR原始转写保留，其标点前多空格会影响严格CER/WER，不能被当作真实漏字或生成错误。两个CPU Profile在此次120条固定英文条件下像素完全相同，不宣称v3改进英文风格。

真实浏览器验证盲评页面：初始评分关闭、锁定正确转写且确认完整后开放、改判内容错误后重新关闭并清空；不显示后端名或识别器结果。所有自动化答卷均为测试夹具，没有写入人工评分记录。

Python3.12与3.13各跑完整测试：138 passed、2个可选兼容测试因环境变量未设而skip；补设原有隔离解释器后，两版各2 passed，共140项均覆盖。Ruff检查及格式通过。首次no-isolation构建缺setuptools失败；改用临时隔离构建成功，生产解释器不新增依赖。首次uv freeze因默认缓存只读失败，改用/tmp独立缓存成功，固定依赖已补全提交。

初始5,074项保护清单意外包含正在创建的审计自身run.json；它在封存时自然变化。复核明确排除此自引用项，实际旧历史文件为5,073，不能虚报5,074全部未变。源码打包生成了egg-info元数据变化；PKG-INFO已从安装包中找到与原摘要完全一致的副本恢复，SOURCES.txt属于可再生构建清单，其新内容和原摘要分别保留并披露。真正Runtime源码、安装环境、默认Profile、正式latest与用户.vscode均保持不变。

## 最终证据与决策

588图双识别器实际完成，TrOCR没有token上限命中。同smoke CER：font5.14%、DiffBrush32.78%、DiffusionPen22.57%；均为识别差异，人工真值仍空。完整DiffBrush236生成图HTR CER29.77%；短于35字符44.72%、35–61字符14.76%、长于61字符50.14%，不能据此推断训练分布因果。

生成四页同文本对比PDF、完整失败复核队列、72份盲评（按固定类别ID选择，包含全部对应种子而非挑图）。推荐DiffusionPen进入下一轮工程验证；当前所有路线均No-Go个人训练与生产。默认Profile/latest/CPU源码不变。

最终审查补充封存保护：汇总工具拒绝修改sealed run或覆盖已有summary，盲评只读取封存评测；对应提交6fef239。新增回归后重新跑完整测试，Python3.12/3.13各141 passed，零失败零跳过。最终72项盲评页面含74张图全部加载、72组风格评分默认禁用、无后端名称显示。最终19个v0.8 run的封存摘要全部验证通过，失败运行仍保留failed状态。
