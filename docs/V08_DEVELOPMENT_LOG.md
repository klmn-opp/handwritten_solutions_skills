# v0.8 研发日志

## 初始审计与隔离

起点为 `experiment/diffink-cuda-v07@21a582b`。`v0.6.0` 实际提交是 `2bcb260352dd4fee177e9cc8ce9179201bf0b5f3`；main 为 `b2754e7`，已经合入该发布，且是 v0.7 分支祖先。新建 `experiment/english-evidence-v08`，保留 main、旧分支及标签。

已读取 CURRENT_STATE、ROADMAP、BENCHMARK_V1 与两份 CUDA 复现报告。旧 ROADMAP 含早期 v0.3 目标，本轮将增加当前决策，不抹去历史记录。未跟踪 `.vscode/` 原样保留。

`workspace/runs/v08-initial-audit/` 固定 5,074 个历史文件（个人资料、原输出、已有运行和权重）的摘要，另记录全部 src 源码、正式 latest、默认 Profile 与 .vscode 摘要。CPU Runtime 不修改，本轮代码仅放 research/english 与测试中。

## 基准冻结准备

English Handwriting Benchmark v1.0.0 包含 120 条原创英文行：8 类各15条，development 96、holdout 24，开发集内 smoke16。固定文本保留困难标点、短句、长句和边界，不因模型不支持而改写。holdout 禁止诊断调参及任何训练。字符对齐、词对齐、替换/删除/插入/重复及尾部缺失分别报告为识别器差异，真实内容错误需独立校核。

## 环境与源代码准备中的故障

HTR 首次依赖解析被 PyTorch 索引优先级阻断：所指定 NumPy 版本不在该索引。保留失败日志，改用兼容固定版本并允许在官方 PyTorch/PyPI 两源匹配；不改变生产依赖。准备记录在 `v08-model-preparation`，任务完成前不封存仍有日志写入的 run。
