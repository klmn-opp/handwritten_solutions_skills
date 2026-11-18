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
