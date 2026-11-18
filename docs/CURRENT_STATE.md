# 当前状态：v0.6 CPU 交付，v0.7 CUDA 复现准备中

继续原仓库和 Agent Skill，没有重建项目。历史审计见 [CURRENT_STATE_AUDIT.md](CURRENT_STATE_AUDIT.md)，原 v0.4.0 证据保留。

| 阶段 | 实现 / 实测 | 状态与下一步 |
|---|---|---|
| Phase 0 | 稳定 main、功能/实验分支；不可复用 run ID，固定代码/依赖/Profile/种子/硬件/资产摘要；旧 Profile 路径兼容；原子 latest；2,213 旧文件保护；DVC 评估 | CPU 实现完成；d392235。未上传数据或部署 DVC |
| Phase 1 | 固定 60 例与两条评估路径；正式 60/60 预期通过，54 页、6 明确拒绝；8 例真实 Agent 流程；10 例真实 OCR | 5028b54 冻结后运行 holdout；独立全量解题判卷、数学转写和人类风格评估待完成 |
| Phase 2 | 沙箱外实际发现 RTX 5060 Laptop GPU / 8151 MiB / driver 580.126.09 / compute 12.0；DiffInk 固定 65055545fbef5e22fbee0811a1ef3855ca242c22 | 隔离 PyTorch CUDA 环境和官方资产准备中；尚未完成模型推理 |
| Phase 3 | 静态扫描、估计轨迹和真实轨迹的已有区分保留 | 先成功推理并核验真实在线样本，再决定监督微调；未训练 |
| Phase 4 | 未增加旧 bandit 步数 | 必须有真实神经基线、独立评测和偏好数据后再研究 |
| Phase 5 | font-cpu 默认，保留 Agent 解题、背景、每题一页、数学来源、PDF/PNG；diffink-experimental 明确隔离 | 实验模型未准入生产，未发布模型或更改默认 Profile |

Python 3.12、3.13 各 125 项通过。实际 Tesseract 4.1.1 对 10 个英文用例 CER 7.56%、WER 42.71%；不能称作人类自然度。新五题 PDF/PNG 与 v0.4.0 对应交付逐字节相同。默认仍 personal@v1。

[验收报告及精确产物路径](evidence/v06-2026-11-18/README.md) · [工作区契约](WORKSPACE.md) · [基准协议](BENCHMARK_V1.md)。v0.6.0 对应源码和发行文件摘要见本地 dist/v0.6.0/release.json 与 SHA256SUMS；发行后的干净安装验证另留证据。

GPU 更正：沙箱内 nvidia-smi 失败是设备可见性限制，不能外推为主机无 GPU。后续 CUDA 命令需获准在沙箱外执行；不修改驱动或生产 CPU 环境。
