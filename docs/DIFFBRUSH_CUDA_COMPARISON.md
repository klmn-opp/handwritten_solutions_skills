# DiffBrush 真实个人英文参考对照

本轮已经成功运行真实 DiffBrush CUDA 模型。此报告是 v0.7 实验状态；旧 [v0.4 失败报告](CONTINUOUS_WRITING_EXPERIMENT.md) 保留为当时证据，不改写历史结论。

使用已有 Handnote `114919f183817643c795e47e851ea25b96e3c393` 的 `linegen.generate.LineGenerator`，没有重新实现扫描识别或神经网络。它适配 DiffBrush `da9addc1140bdfec463b2d41a974a0a392f80798` 的真实英文权重。官方 README 的 Drive 权重确实可取得，并与锁定 Handnote 的大小和 SHA256 匹配。[DiffBrush 官方仓库](https://github.com/dailenson/DiffBrush)

隔离环境 `workspace/environments/diffbrush-inference/` 使用 Python 3.12.16、torch 2.7.1+cu128、torchvision 0.22.1+cu128、diffusers 0.32.2。完整依赖见 [lock](../research/diffbrush/requirements-inference.lock)。它与 DiffInk、生产 CPU 环境相互独立；没有训练环境或训练进程。

## 真实执行条件

- 正文固定：`The median is the middle ordered value.`
- 本人参考：既有 `out/next-stage-2026-11-18/line-experiment-final/reference-line.png`，是自然英文行扫描；不是在线笔顺。运行时复制到各 run 的 solutions/reference.png，并记录摘要。
- CPU 对照：同一作者的 `personal@v3`、contextual、黑墨、白纸。模型使用该作者的扫描行，CPU 使用已采集个人字形；参考模态不同，不能声称是严格等输入风格实验。
- 模型严格加载实际权重；50 步 DDIM，float32，种子 42 / 43；保留原始 64×1024 输出，不裁掉模型误写。另存上游常规裁剪图。
- 展示与 OCR：对各结果统一白背景、墨迹高度 64 px、等比例缩放、20 px 边；OCR 只看放大 4 倍的图，Tesseract 4.1.1 eng / psm7，不接收期望文字。

| 结果 | 推理时间 | 峰值分配显存 | 该句 OCR CER / WER |
|---|---:|---:|---:|
| 个人字体 CPU | 正式整页渲染 0.763 s；与单行 GPU 计时范围不同 | 无 CUDA | 7.69% / 28.57% |
| DiffBrush seed 42 | 1.6176 s | 1,838,090,240 B | 23.08% / 57.14% |
| DiffBrush seed 43 | 1.6152 s | 1,838,090,240 B | 20.51% / 42.86% |
| DiffInk 同英文 | 字表不支持，未生成 | 不适用 | 未测量，不能用中文结果替代 |

DiffBrush 模型加载约 2.41–2.45 s，整个 worker 约 5.45–5.48 s，主机峰值 RSS 约 3.79 GB，CUDA reserved 约 2.17 GB。以上仅短句数据，不能推导整页速度。

两条模型候选都未得到精确 OCR 转写，原始画布均有墨迹接触边界。可视检查也发现词尾异常 / 重复嫌疑。边界接触是待复核信号，不直接等于已经证明裁切。按“先内容精确和边界检查，再评价风格”的保守筛选，本次没有合格候选，不选择一个看起来更漂亮的错误答案。OCR 本身会误读个人字体，因此其 CER 不是最终人类正确率；仍需盲读。作者风格相似性和自然度未评分。

这证明模型可以使用现有个人扫描参考做真实整行生成，不证明比 CPU 更准确、更像本人或能稳定处理数学。模型只输出栅格，不输出真实在线轨迹；来源始终为 model-generated，不标记为个人采集。

## 路径与重放

- `workspace/runs/v07-font-english/`：CPU 原页与完整来源。
- `workspace/runs/v07-diffbrush-personal-42-retry/`、`v07-diffbrush-personal-43/`：真实模型原图、PDF、计时、显存、参考摘要及依赖。
- `workspace/runs/v07-diffbrush-personal-42-repeat/`：同种子重放。
- [同页英文对照 PDF](../workspace/runs/v07-english-model-comparison/rendered/homework_solutions.pdf)。对应 PNG 位于 rendered/pages/01.png。
- [机器可读对照](../workspace/runs/v07-english-model-comparison/reports/comparison.json)，原始 OCR 文本在 logs/。

```bash
.venv/bin/python research/diffbrush/run.py \
  --python workspace/environments/diffbrush-inference/bin/python \
  --cache workspace/checkpoints/handnote-diffbrush \
  --reference out/next-stage-2026-11-18/line-experiment-final/reference-line.png \
  --text 'The median is the middle ordered value.' \
  --seed 42 --steps 50 --run-id NEW-UNUSED-ID
```

再用 `research/diffbrush/compare.py --font-run ... --model-runs ... --run-id ...` 建立新对照。模型和对照都封存为 held，不更新 latest，不发布模型。

## 保留的故障与修复

1. `v07-diffbrush-preflight`：Handnote 下载在权重中途结束，大小/摘要检查拒绝。失败原日志保留。新 run 通过 HTTP Range 续传，仍以固定 SHA256 校验；旧历史部分下载未动。
2. `v07-diffbrush-preflight-retry`：完整权重、VAE、内容字形验证通过。资产路径在 `workspace/checkpoints/handnote-diffbrush/diffbrush/`。权重 SHA256 `ed0d6059a3b0a68c18476c02ba2410d607fc478eaead597faa2483406c0082fa`；VAE SHA256 `a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c`。
3. `v07-diffbrush-personal-42`：diffusers 0.32.2 访问未显式导入的 `safetensors.torch`，实际 VAE 加载失败。修复只是在隔离 worker 显式导入该模块，未改变模型、采样算法或上游代码。重试使用新 run ID。

DiffBrush 权重依据 IAM 研究条款；SD1.5 VAE 另有 CreativeML OpenRAIL-M 条款。代码、权重和数据的许可分别记录，不把权重随 MIT 项目发行。

下一步优先做更广英文 development 集、独立转写与本人盲评，并检查边界和重复字符。当前仅两候选的小实验不构成 Best-of-N 成功、偏好数据集或 RL 训练；个人微调继续暂缓。
