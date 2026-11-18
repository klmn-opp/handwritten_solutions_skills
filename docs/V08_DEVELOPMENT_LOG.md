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
