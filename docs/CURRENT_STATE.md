# 当前状态：v0.9 真实作业语料与审核试点已运行，个人训练仍 No-Go

研究分支 `experiment/personal-corpus-v09`，沿用既有项目/Agent Skill/Handnote。
正式 font-cpu、personal@v1、latest不变；DiffBrush/DiffusionPen/DiffInk研发暂停，旧证据保留。

- 新收到7份PDF，共831页；原件已只读归档，10页实际作业试点，另保留7张旧采集图。
- 190个可追溯候选区域；TrOCR/UniMERNet/Tesseract实际得到58条识别结果，人工核验0条。
- 本地Review Studio支持原图高亮、英文/LaTeX修正、公式预览、合并拆分、缺陷/作者标记、修订历史与恢复。
- 涂改、遮挡、重写及不确定来源不进入干净墨迹训练；不猜标签，不用补全图冒充真实笔迹。
- 数据集v1.1已冻结文档/会话分组并导出全部候选快照；合格监督样本为0，真实覆盖率仍待核验。
- 约40万参数条件VAE原型与独立训练环境已准备，CUDA合成夹具参数更新测试通过；**没有个人模型训练完成**。
- 四学科4页CPU基线PDF/PNG验证通过，神经字形使用数为0。完整个人生成、连续英文和SFT/RL仍未准入。
- Python3.12/3.13各159项全套测试通过；回归和历史保护结果见 [v0.9验收报告](V09_GO_NO_GO.md)。原12,822历史文件与7新PDF均未改变。

入口：[数据格式与审核操作](PERSONAL_HOMEWORK_CORPUS.md) · [研发日志](V09_DEVELOPMENT_LOG.md) · [最初审计](V09_AUDIT.md)。
本机运行 `.venv/bin/python -m research.corpus.cli review`（`PYTHONPATH=.:src`），打开 `http://127.0.0.1:8769`。

证据目录：`workspace/runs/v09-corpus-evidence-report/`、`v09-personal-corpus-v1.1/`、
`v09-four-subject-cpu-baseline-retry/`、`v09-final-gate-verification/`。
下一步是核验少量真实候选并扩大分层抽样，不要求先手工清洗所有831页或重新抄字母表。

Studio后续修复：未填完表单不再阻塞切换；错误显示在确认按钮旁；旧草稿和上次区域可恢复；
确认保存成功后自动进入下一条。使用修复前页面需普通刷新，保留浏览器站点数据。

---

以下v0.8及更早状态保留用于追溯，其模型路线建议已由v0.9决定暂停。

# 当前状态：v0.8证据评估完成，个人训练No-Go

当前研究分支 `experiment/english-evidence-v08`；稳定main、v0.6.0包、CPU Runtime、personal@v1和正式latest均未改变。[完整v0.8 Evidence-Based Go/No-Go](V08_GO_NO_GO.md) · [英文基准协议](ENGLISH_BENCHMARK_V1.md) · [模型调查](V08_MODEL_INVESTIGATION.md) · [研发日志](V08_DEVELOPMENT_LOG.md)。

- 固定120行英文：96 development、24独立holdout，含16行smoke。v1/v3各120行实际渲染，此条件下像素全部相同。
- DiffBrush官方32组真实对照；全基准240请求中236生成、4次明确不支持字符；参考fit消融32生成。输入完整，原图至PDF保真，主要内容异常已存在于原始模型。
- DiffusionPen官方权重+5本人扫描词，16文本×2种子真实CUDA推理成功。它仍按词生成后组合，未训练。
- Tesseract与真实TrOCR共同评测588张图；同smoke的TrOCR CER：字体5.14%、DiffBrush32.78%、DiffusionPen22.57%，不是真实人工错误率。16行本人扫描校准也显示识别偏差。
- 72份内容先行盲评材料已生成，真实人工评分0份。没有证据给自然度或作者相似性打新分数。
- 推荐DiffusionPen为下一轮工程验证主候选；所有路线当前均不准入个人监督微调或生产。DiffInk发布词表无法完整覆盖任何一条本轮英文文本，原中文在线实验保留。
- Python3.12/3.13各141项全套测试通过（零失败、零跳过）；构建、Ruff、PDF保真、模型摘要和浏览器门禁检查通过。

交付入口：`workspace/runs/v08-evidence-analysis/rendered/homework_solutions.pdf`、`workspace/runs/v08-blind-review/comparisons/blind-review.html`、`workspace/runs/v08-evaluation-final/reports/observations.json`。下阶段先内容人工校核与词间尺度/标点验证，达标后再论证监督微调；不做从零训练、LoRA或RL。

---

以下为保留的v0.7历史状态，不代表已完成v0.8个人训练。

# 当前状态：v0.6 已交付，v0.7 已完成真实 CUDA 实验

继续原仓库和 Agent Skill，没有重建项目。历史审计见 [CURRENT_STATE_AUDIT.md](CURRENT_STATE_AUDIT.md)；v0.4.0 的原证据保留。稳定 `main` 已合入 v0.6，当前研究位于 `experiment/diffink-cuda-v07`。CPU 发布标签 `v0.6.0 → 2bcb260352dd4fee177e9cc8ce9179201bf0b5f3`；研究源码提交和每次执行的来源详见 run.json。

| 阶段 | 实现 / 实测 | 状态与下一步 |
|---|---|---|
| Phase 0 | main、功能/实验分支；不可复用 run ID；代码/依赖/Profile/种子/硬件/权重摘要；旧路径兼容；原子 latest；2,213 旧文件保护；DVC 评估 | 已完成。干净 wheel 安装重放成功；未上传私有数据或部署 DVC |
| Phase 1 | 固定 60 例、两条评估路径；60/60 预期通过，54 页和 6 例明确拒绝；真实 Agent smoke 8；英文 OCR 10 | 基准实现完成；独立全量解题判卷、数学转写、作者自然度评分待完成 |
| Phase 2 | RTX 5060 Laptop 实测；固定 DiffInk 真权重推理、显式在线输入 Adapter、轨迹/PNG/SVG、同种子重放；另成功运行 Handnote DiffBrush 个人扫描参考英文生成 | 真实 CUDA 复现完成；DiffInk 不支持一般英文，DiffBrush 两候选内容未过关，均未准入生产 |
| Phase 3 | 区分扫描、估计轨迹、真实轨迹；Adapter 验证在线文本/字符边界 | 未训练。缺少个人真实在线数据，先完成英文基线独立评估，不启动监督微调或从零训练 |
| Phase 4 | 两个真实英文候选，独立 OCR 内容筛选；D3PO / Diffusion-DRO 方法核对 | 没有合格候选，没有神经偏好集；未做 RL，未延长旧 bandit |
| Phase 5 | font-cpu 默认；保留 Agent、Profile、每题一页、背景、数学来源、PDF/PNG；实验命令隔离 | 生产 diffink-experimental 仍明确拒绝；未发布模型、未改变默认 Profile |

CPU 的正式五题 PDF/PNG 与 v0.4.0 对应交付逐字节相同；本轮没有证据证明默认字迹自然度已提升。默认仍 `personal@v1`，研究对照显式使用 `personal@v3`。

## 实际质量与工程结果

- 全套测试在 Python 3.12、3.13 各 **134 passed**，无跳过；Ruff 检查与格式检查通过。v0.6 发行时各 125 项，新增 9 项为实验输入边界/保留末字/失败指针测试，不冒充真实推理证据。
- 固定 benchmark 的 10 个英文用例，实际 Tesseract CER **7.56%**、WER **42.71%**。60/60 是预期行为匹配，包括应拒绝的 6 例，不是 60 道题内容全对或自然度全合格。
- DiffInk 20 步真实推理约 **0.76–0.85 s**，峰值分配显存 **1.05 GB**；只复现了中文在线前缀续写，同接口重放轨迹/PNG 摘要完全一致。中文 OCR 对参考与生成都很差，不足以验收模型。
- DiffBrush 50 步、实际本人扫描参考，两个种子约 **1.62 s/行**、峰值分配显存 **1.84 GB**。同句 OCR CER：字体 **7.69%**、模型 **23.08% / 20.51%**；仅一个短句，不能推广为总体质量。两候选均 held，原图有边界墨迹和词尾异常待复核；同 seed42 重放 PNG 一致。
- 作者风格相似性、自然度、完整数学独立转写准确率仍未评分；不以测试数、OCR 或代理奖励替代这些指标。
- 2,213 个旧文件摘要未变；生产无 torch；latest 仍指向 `v06-five-final`。失败、模型和基准没有覆盖最近正式作业。

[Phase 0/1 验收与包对应关系](evidence/v06-2026-11-18/README.md) · [CUDA 实验验收及路径](evidence/v07-2026-11-18/README.md) · [DiffInk 复现](DIFFINK_CUDA_REPRODUCTION.md) · [DiffBrush 对照](DIFFBRUSH_CUDA_COMPARISON.md) · [工作区契约](WORKSPACE.md) · [基准协议](BENCHMARK_V1.md)。

下一里程碑：优先扩大英文生成的独立内容/边界评估、解决漏写重复，再做本人风格盲评；DiffInk 需先解决英文词表与真实在线输入需求。满足这些条件后才决定个人监督微调。v0.8 个人训练/偏好优化仍未交付，不能把已完成的 CUDA 实验称为完成训练。
