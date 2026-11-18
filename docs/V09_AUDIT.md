# v0.9 源码与数据审计

实验分支 `experiment/personal-corpus-v09` 起点为 v0.8 `24589ba`。稳定 main
`b2754e7`、v0.6.0 `2bcb260`、v0.7 实验分支 `21a582b` 均保留。原有
`.vscode/` 未纳入此次提交。审计证据：`workspace/runs/v09-initial-audit/`。

## 数据事实

审计冻结了 12,822 个历史文件的摘要。个人原始采集图片去重后为 **7 张**，另有
11 份重复副本；不是 18 份独立样本，也不是新历史作业 PDF。其中四张字格表、
两张布局页、一张历史验收页。没有真实在线笔画轨迹。历史验收页后来参与过校准，
不能重新声称是未见过的评估数据。

用户指定新入口 `source_pdf/`，目前尚未上传 PDF。该目录被 Git 忽略。原始文件
只读归档，所有新实验使用 `workspace/runs/<run_id>/`，不更新正式 latest。

| Profile | 状态 | 个人码点 / 登记形状 | 实际差异 |
|---|---|---|---|
| personal@v1 | 正式默认 | 75 / 178 | 原始个人字体 |
| personal@v3 | 已发布但未激活 | 75 / 178 | 75 个主字形轮廓与 v1 相同；增加布局及笔画校准 |
| personal@v5 | 未发布候选 | 75 / 178 | 加入字形度量；69 个主轮廓经过已有尺度处理 |

形状数不等于独立核验训练实例数。旧 `accepted` 表示旧字体流程接纳，不意味着
OCR 转写经过人工确认。字格模板记录“应该写什么”，不能替代“实际写下什么”。
新管线把这类标签保留为候选，禁止静默晋升为 verified。

## 集成边界

锁定 Handnote 的 Renderer 支持记录 path、stroke 和 image。现有 MathText 给出
数学字形及结构线条的位置，`jobs/handwriting.py` 在结构定位后替换个人轮廓。
未来生成墨迹可接在这一层，但须保持锚点、结构和来源记录；未通过内容验收仍用 CPU。
本阶段不改变正式 Runtime、依赖、Profile 或发布指针。

既有数学基准有 60 例（8/24/16/12），包含消元、特征值、积分、统计、RLC 等，
可复用作渲染回归。它不是历史混合扫描页识别基准，也不能证明 OCR 或新生成模型正确。

## 识别部署决策

本机 RTX 5060 Laptop 约 8GB。源码固定在 `workspace/sources/v09/`，独立环境为
`workspace/environments/corpus-neural-v09/`。OXR 官方完整部署要求至少约 24GB、
建议更高显存，因此仅审计其结构接口，不在此机器强行部署完整服务。
先试 TrOCR handwritten 与 UniMERNet tiny，Tesseract 提供独立对照。
PaddleOCR 作为本地检测及印刷区域候选；所有识别都不等于生成模型训练。

来源：[OXR](https://github.com/SoMarkAI/OXR)、
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、
[UniMERNet](https://github.com/opendatalab/UniMERNet)。实际运行与失败将另行记录。
