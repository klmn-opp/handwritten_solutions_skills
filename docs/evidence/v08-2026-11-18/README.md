# v0.8验收证据索引

正式结论见 [Evidence-Based Go/No-Go Report](../../V08_GO_NO_GO.md)。研究分支 `experiment/english-evidence-v08`，CPU发布和默认Profile保持原状；本轮没有训练或发布新模型。

- [运行清单及封存摘要](runs.json)：包含真实推理、失败重试、识别、盲评、测试和分析；held不代表内容通过，manifest_verification只证明文件完整性。
- [机器可读汇总](summary.json)：588图两识别器观察；人工错误率和自然度保持null。
- [Go/No-Go](go-no-go.json)：DiffusionPen为下一轮工程验证主候选；个人训练和生产均No-Go。
- [保护复核](preservation.json)：5073历史文件、Runtime源码、默认Profile、latest、.vscode不变；审计自引用与可再生构建清单变化单独披露。
- [自动化验证](verification.json)：最终全量回归及原构建验证引用，不把单元测试当真实推理或人工评分。

真实个人图像、生成PDF/PNG和权重不复制进Git。以下路径相对仓库根：

- 对比PDF：`workspace/runs/v08-evidence-analysis/rendered/homework_solutions.pdf`
- 72份盲评：`workspace/runs/v08-blind-review/comparisons/blind-review.html`
- 原始OCR/HTR：`workspace/runs/v08-evaluation-final/reports/`
- DiffBrush原图：`workspace/runs/v08-diffbrush-development/rendered/pages/`，官方对照在 `v08-diffbrush-official-smoke/comparisons/`
- DiffusionPen原始词图：`workspace/runs/v08-diffusionpen-smoke/comparisons/`
- 默认CPU基线：`workspace/runs/v08-font-v1-frozen/rendered/homework_solutions.pdf`

复现协议：[英文基准](../../ENGLISH_BENCHMARK_V1.md)。每次执行必须新建run ID；不可复用或修改封存实验。源码、模型、参考样本及种子可在各run.json和reports/request.json追溯。生产latest仍指向v06-five-final。
