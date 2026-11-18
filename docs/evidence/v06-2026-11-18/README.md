# v0.6 Phase 0 / Phase 1 验收

CPU 实现冻结于 `5028b541deb09ae8a1d4b528cfc4b8b554e37244`，随后才首次运行 holdout。无默认 Profile 切换、无神经训练、无远端发布。

| 项目 | 实测结果 |
|---|---|
| 版本化基准 | 60/60 符合预期；54 页成功，6 例明确拒绝；12 例 holdout 未参与调参 |
| 端到端 | 8 例原题 text/image/PDF 实际读取、另行推导、复核并出页；同会话作者演练，非独立解题正确率 |
| 独立英文 OCR | Tesseract 4.1.1，10 例；micro CER 7.5552%，WER 42.7083%；没有用提示词充当识别结果 |
| Python 3.12 / 3.13 | 各 125 passed，无跳过；3.13 测当前源码，发行 wheel 另在干净环境验证 |
| 历史保护 | 2,213 个旧文件摘要未变；默认 personal@v1 未变 |
| 五题兼容 | 新契约生成的 PDF 和 5 PNG 与 v0.4.0 contextual 交付逐字节相同 |
| 资源 | 每例独立子进程，渲染中位数约 0.969 秒，最高 RSS 179,249,152 字节；非受控独占吞吐测试 |

[完整基准摘要](benchmark-summary.json)、[OCR 摘要](ocr-summary.json)、[端到端摘要](end-to-end-summary.json)、[历史保护](preservation.json)、[五题回放](five-replay.json)。测试日志与 JUnit 为 tests-py312/313；首轮旧测试失败证据保留于 tests-initial-failed.log，不掩盖中途故障。

## 实际产物

- [54 页基准合集](../../../workspace/runs/v06-benchmark-v1-final/rendered/homework_solutions.pdf)，逐例指标与来源在 [benchmark.json](../../../workspace/runs/v06-benchmark-v1-final/reports/benchmark.json)。每个 case 的规范运行路径为 `workspace/runs/v06-benchmark-v1-final-<case_id>/`。
- [8 页端到端演练](../../../workspace/runs/v06-e2e-smoke/rendered/homework_solutions.pdf)。原题包 `workspace/runs/v06-agent-packet/`；实际另行作答 `workspace/runs/v06-agent-solutions/`。
- [英文真实 OCR](../../../workspace/runs/v06-ocr-final/reports/ocr.json)，原识别文本与裁剪图保留在该运行的 logs/ 和 comparisons/。
- [五题兼容 PDF](../../../workspace/runs/v06-five-final/rendered/homework_solutions.pdf)，`workspace/latest.json` 指向该验证交付，基准和失败运行不更新它。

个人字体和 PDF 都在被忽略的本地 workspace，以上产物链接仅本工作区可用。以前跟踪的 4 份 PDF 和比较 PNG 已从当前 Git 索引移出，本地原文件与 archive 副本仍在，未重写旧提交。

## 发现与修复

首轮开发集 `v06-development-01` 是 46/48，`\ge` 触发 MathText 解析回退，旧摘要计算不支持 Image。修复标准关系别名并拒绝不可审计的整式回退。旧测试随后发现合法矩阵 stroke 被过度拒绝；恢复矢量矩阵笔画后 125 项全通过。第二轮开发集 `v06-development-02` 是 48/48。两轮原始运行均保留。未据 holdout 更改渲染策略或题库。

## 限制和下一步

CER/WER 表明英文可读性仍有问题，不能据工程测试认定自然度提升。数学独立转写准确率、作者风格、自然度与真实碰撞率尚未测量；包围盒重叠只是候选。完整 60 例独立 Agent 判卷尚未运行，当前实际端到端覆盖是 smoke 8。

下一步：完成隔离 CUDA 实测和 DiffInk 官方权重复现，再决定个人在线数据和监督微调。独立模型未准入生产路径；不得把接口或随机初始化输出算成成功推理。

干净 wheel 安装已实测通过：[回放报告](wheel-replay.json)。安装环境不含 torch/pytest；60 例题库随包可用；五题 PDF/PNG 与源码逐字节一致。v0.6.0 标签固定在 2bcb260352dd4fee177e9cc8ce9179201bf0b5f3，后续证据提交不移动标签。
