# v0.7 真实模型实验验收

研究分支：`experiment/diffink-cuda-v07`。DiffInk Adapter 初始实现 `fc9dd9a`，真实 DiffBrush worker 与比较工具 `3482c6d/ff40820`，实际 VAE 加载修复 `8586558`，输入拒绝来源与边界检查 `6f6e61f`。每个 run.json 保存其执行时的准确 commit、dirty 状态、源码/依赖/模型摘要和配置；本目录后续文档提交不改写原记录。

生产发行仍为已测试的 v0.6.0；没有将研究分支打成新的生产包。未激活任何新 Profile、训练神经模型、上传资产或发布实验模型。

机器证据：[18 个实验运行索引](experiments.json)、[英文对照指标](english-comparison.json)、[中文 OCR](chinese-ocr.json)、[历史保护与默认状态](preservation.json)、[最终 Python 3.12 日志](tests-final-py312.log)、[最终 Python 3.13 日志](tests-final-py313.log)、[静态检查](ruff.log)。对应 JUnit 同名 `.xml`；初次全套测试日志也保留。本目录文件摘要见 SHA256SUMS。

## 可直接查看的产物

- [个人字体 / DiffBrush 两候选英文对照 PDF](../../../workspace/runs/v07-english-model-comparison/rendered/homework_solutions.pdf)，[PNG](../../../workspace/runs/v07-english-model-comparison/rendered/pages/01.png)，[原始机器报告](../../../workspace/runs/v07-english-model-comparison/reports/comparison.json)。
- [DiffInk 完整在线输出预览 PDF](../../../workspace/runs/v07-diffink-adapter-original/rendered/homework_solutions.pdf)，[SVG](../../../workspace/runs/v07-diffink-adapter-original/rendered/strokes.svg)，[NPY](../../../workspace/runs/v07-diffink-adapter-original/solutions/generated-strokes.npy)。
- DiffInk 原始官方预览与真实参考在 `workspace/runs/v07-diffink-official-01/comparisons/official/`；显式目标原句、改句和重放各自为 `v07-diffink-adapter-original/changed/repeat`。
- DiffBrush 原始 64×1024 图在 `workspace/runs/v07-diffbrush-personal-42-retry/rendered/pages/01.png` 和 `v07-diffbrush-personal-43/rendered/pages/01.png`；同种子重放在 `v07-diffbrush-personal-42-repeat/`。
- [中文独立 OCR](../../../workspace/runs/v07-diffink-ocr/reports/ocr.json)。DiffInk 不支持固定英文的失败证据在 `v07-diffink-english-gate/`。
- 全套最终测试、静态检查与历史摘要核验在 `workspace/runs/v07-final-verification/`。本目录保存非私密摘要和测试日志；模型、个人输入、生成 PDF/PNG 均留在忽略的 workspace。

## 分开报告的验收维度

| 维度 | 已有证据 | 仍缺什么 |
|---|---|---|
| 内容正确性 | 60 例 renderer 预期行为；8 例 Agent 演练；真实 OCR；模型失败保留 | 独立全量 solver 判卷和数学转写；神经输出人工盲读 |
| 单词尺度协调性 | 旧 v0.4 contextual 度量保持；本次统一显示尺度对照 | 无新增独立自然度/协调性评分，不声称改善 |
| 数学覆盖与可读性 | 保留 CPU provenance，基准收集覆盖/回退；未让模型生成数学 | 更广的独立符号和结构转写 |
| 与作者的风格相似性 | DiffBrush 真正以本人扫描行作条件，CPU 用同作者 Profile | 本人盲评；不同参考模态不能直接作为等输入风格结论 |
| 整页自然度 | CPU 一题一页与裁切拒绝不退化 | 模型仅短文字行；整页自然度未评 |
| 工程一致性 | 双 Python 各134；模型严格加载；固定资产；DiffInk/DiffBrush 同种子重复一致；2,213旧文件未变 | CUDA训练复现未进行，跨GPU/操作系统未验 |

## 结论与下一步

DiffInk 与 DiffBrush 都有真实 CUDA 输出。DiffInk 官方 checkpoint 的词表/前缀条件不适用于现阶段一般英文；DiffBrush 有连续字行输出，但两个候选在当前内容筛选下均不合格。因此默认仍是 CPU 个人字体，没有以视觉连贯替代文本正确性。

Phase 0/1 已交付，Phase 2 核心 CUDA 复现已完成；一般英文 DiffInk Adapter、全面神经基准和风格验收未完成。Phase 3 个人训练、Phase 4 偏好优化训练、Phase 5 神经生产准入暂缓。下一步先扩展英文 development 集的独立内容评估与本人盲评，保留 holdout 隔离；不能用现有静态扫描冒充在线笔顺数据。

详细方法、命令、依赖及故障见 [DiffInk](../../DIFFINK_CUDA_REPRODUCTION.md) 与 [DiffBrush](../../DIFFBRUSH_CUDA_COMPARISON.md)。已保存首轮下载中断和 safetensors 导入失败；后续修复使用新运行 ID，没有覆盖失败证据。
