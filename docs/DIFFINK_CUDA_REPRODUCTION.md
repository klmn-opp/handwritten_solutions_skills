# v0.7 实验分支：真实 CUDA 复现与准入边界

本报告对应 `experiment/diffink-cuda-v07`，CPU 发布包仍为 `v0.6.0`。生产默认保持 `font-cpu`，默认 Profile 保持 `personal@v1`。本阶段没有个人模型训练、RL、模型发布或自动替换数学渲染。

## 已实际运行

设备为 NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB，驱动 580.126.09，compute capability 12.0。沙箱内看不到驱动，但获准的主机命令成功执行 `nvidia-smi` 和真实 CUDA 矩阵运算；不能把沙箱错误解释为机器无 GPU。

DiffInk 隔离环境：`workspace/environments/diffink-inference/`，Python 3.10.20、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128、torchaudio 2.7.1+cu128。完整版本见 [requirements-inference.lock](../research/diffink/requirements-inference.lock)。生产 `.venv` 未安装 torch。原版 Python 3.8 / torch 2.0+cu118 不适合该 Blackwell 设备；此处明确使用支持 Blackwell 的 CUDA 12.8 构建，未更换驱动。[PyTorch 2.7 官方说明](https://pytorch.org/blog/pytorch-2-7/)

上游固定为 [awei669/DiffInk@65055545](https://github.com/awei669/DiffInk/tree/65055545fbef5e22fbee0811a1ef3855ca242c22)。Git 网络克隆失败后改为下载该提交归档，逐文件摘要见 [upstream-files.json](../research/diffink/upstream-files.json)。源码位于 `workspace/sources/DiffInk/`，没有修改上游文件。

| 真实实验 run ID | 输入 / 结果 | 推理时间 | 峰值 CUDA 分配 |
|---|---|---:|---:|
| v07-diffink-official-01 | 官方验证集 line_2296 / writer_212；真实在线轨迹，20 步 DDIM | 0.8500 s | 1,049,719,808 B |
| v07-diffink-adapter-original | 显式在线参考，`要的应该是法律问题!` | 0.7691 s | 1,049,719,808 B |
| v07-diffink-adapter-repeat | 同参考、文本、seed=4206；轨迹和 PNG 摘要与上一行完全相同 | 0.7888 s | 1,049,719,808 B |
| v07-diffink-adapter-changed | 保留前缀，改为 `要的应该是数学问题!`；输出轨迹实际改变 | 0.7611 s | 1,049,719,808 B |
| v07-diffink-english-rejected | `The median is the middle ordered value.` | 未推理 | 字表缺空格和 v，明确拒绝 |

时间包含上游推理函数及其预览保存，不含模型加载和依赖导入。Adapter 原句 worker 整体约 5.80 s，峰值主机 RSS 2,908,426,240 B；这些短样本结果不是长段落吞吐测试。官方 DataLoader 与 Adapter 显式单批入口的随机数消费不同，因此第一行与第二行不作逐字节等价声明。固定 Adapter 入口的原句与重复运行确实逐字节一致。

模型真实严格加载 VAE 的 253 个张量和 DiT 的 270 个张量，再执行 CUDA VAE encode、DiT DDIM、VAE decode 和 GMM 采样。不存在随机初始化回退、mock 输出或用单元测试替代推理。输入 301 点，实际输出 304 点（含上游补齐 / 结束状态）。输出点数或结束事件数量不能直接当作正确字符数。

## 资产、许可与本地适配

官方 Drive 资产存放于 `workspace/checkpoints/diffink-65055545/`，映射、完整大小、Google Drive ID 和 SHA256 见 [official-assets.json](../research/diffink/official-assets.json)。

| 资产 | 大小 |
|---|---:|
| dit_epoch_1.pt | 2,914,621,632 B |
| vae_epoch_100.pt | 178,524,954 B |
| val.h5 | 80,310,344 B |
| All_zi.json | 44,043 B |
| selected_400_100.json | 11,163 B |

SHA256 为下载后本地计算，不冒充作者发布的摘要。代码 MIT 不代表训练数据或权重也是 MIT；上游明确要求权重遵循 CASIA 数据许可，相关材料限定研究用途。未把模型和公共/个人样本放进普通 Git 或发行包。[DiffInk 官方 README](https://github.com/awei669/DiffInk) · [CASIA 数据条款](https://nlpr.ia.ac.cn/databases/handwriting/Application_form.html)

适配代码集中于 [reproduce.py](../research/diffink/reproduce.py)，不是改写生产 Runtime：

- batch=1、workers=0，选一条真实短验证行。
- 使用 `weights_only=True, mmap=True` 在 CPU 安全载入，严格匹配模型参数后转 CUDA；拒绝缺权重。
- 辅助 writer classifier 维数从真实 checkpoint 读取，避免配置不一致。
- 捕获上游 sampler 的真实输出，保存原始 Nx5 在线序列。
- 可选显式参考替代数据集批次；保留上游 30% 前缀条件与等长限制。
- 完整校验上游源码、资产、参考和隔离依赖；记录本地 worker 摘要、配置、种子、硬件。

上游 `plot_line_cv2_new` 会裁到倒数第二个 character-end，并略去超过阈值的长线段。保留其原始预览作为证据，Adapter 的最终预览则显示所有连接笔画，另提供 SVG；不以裁剪美化内容检查。轨迹是绝对 x/y 与三种 pen state，没有伪造时间或压力。

## Adapter 的实际能力

`hwsol lab diffink` 输入文本、带字符边界的在线参考、种子、步数和 cfg。生成结果为原始 NPY、SVG、PNG、预览 PDF、来源记录、真实推理时间与显存。源码在 [diffink.py](../src/hwsol/lab/diffink.py)。

这个已复现 checkpoint 的字表只有 2648 字符，不支持一般英文。当前入口还要求目标与参考字符数相同，前 30% 字符一致；它是受限续写，不能称为任意文本风格迁移。数学被拒绝。`recorded-pen`、`dataset-recorded-online`、`estimated-trajectory` 明确区分；扫描轮廓不能冒充真实轨迹。模型输出始终标为 `model-generated`、`personally_collected=false`。

所有成功生成目前为 `held`，CLI 退出码 2 表示内容/风格待验收，并不表示模型未运行。`reports/inference.json:model_executed` 给出真实运行状态。失败和 held 都不更新 `workspace/latest.json`。

```bash
.venv/bin/hwsol lab diffink \
  --text '要的应该是法律问题!' \
  --reference workspace/runs/v07-diffink-official-01/solutions/reference.json \
  --python workspace/environments/diffink-inference/bin/python \
  --checkout workspace/sources/DiffInk \
  --assets-root workspace/checkpoints/diffink-65055545 \
  --research research/diffink \
  --seed 4206 --steps 20 --cfg-scale 1 \
  --run-id NEW-UNUSED-ID
```

新环境按隔离 lock 安装；官方资产根据 manifest 准备。首次没有已有 reference 时，可按 `v07-diffink-official-01/reports/request.json` 的字段创建新标准 run，调用 worker 自动从固定 val.h5 选出样本。任何 run ID 不能复用，不能直接写回上面的已封存结果。

## 独立评估结果与限制

中文 OCR 实际使用 Tesseract 4.1.1 和固定 `tessdata_fast@87416418657359cb625c412a48b6e1d6d41c29bd/chi_sim.traineddata`，模型 SHA256 为 `a5fcb6f0db1e1d6d8522f39db4e848f05984669172e584e8d76b6b3141e1f730`。OCR 仅看完整预览，不接收目标文本；按原字符保留标点，只去空白。

`workspace/runs/v07-diffink-ocr/reports/ocr.json` 中，真实参考的 CER=90%，两条生成图 CER=100%。这同时表明通用印刷体 OCR 对该真实潦草笔迹很差，不能据此独自判定作者可读性或给出可靠模型错误率；也不能隐去差结果、声称内容已经通过。中文未定义分词，因此 WER 为 null。人类转写、风格相似性、自然度均未评分，前缀条件部分也包含在整句 OCR 中。

同目标英文对照的 CPU 输出在 `workspace/runs/v07-font-english/`；DiffInk 对该英文的确拒绝，不能拿中文公开作者的输出冒充与个人英文等条件的比较。DiffBrush 的实际对照另见本轮验收证据。

## 个人训练和偏好优化的决定

目前不进行个人监督微调：现有个人资料是静态扫描字形和布局，没有带文字标注与字符对齐的真实在线行数据；DiffInk 官方权重又不覆盖目标英文。一次 CUDA 成功不足以满足个人训练数据和内容评估条件。没有从零训练 InkVAE/InkDiT，没有称字体构建、尺度校准或旧 bandit 为神经训练。

下一步先确定适合英文和现有参考形式的神经基线，在固定 development 集和隔离 holdout 上做独立转写，再决定采集在线样本或适配。训练与推理需各自独立环境；未来 dataset 必须保留原始记录、来源类型、文本/轨迹对齐、作者/会话分割、不可重用 holdout、版本化 checkpoint。候选评估后才能显式发布。

偏好优化仅完成方法核对，未训练。这里的 D3PO 指 CVPR 2024 的 Direct Preference for Denoising Diffusion Policy Optimization，不与后来的 Discrete Diffusion DPO 同名工作混淆。它以逐步去噪偏好替代独立奖励模型；Diffusion-DRO 将偏好排序转化为去噪目标，并结合专家数据和在线负样本。两者原实验主要是图像扩散，不能直接当作 DiffInk 插件；需要推导与验证对应噪声过程、log probability / score、参考模型及内容约束。[D3PO 论文](https://arxiv.org/abs/2311.13231) · [Diffusion-DRO 论文](https://proceedings.nips.cc/paper_files/paper/2025/hash/561dd71d04695c85e1c0f6a0fbe0bd85-Abstract-Conference.html)

可以先开展小规模候选生成和独立内容筛选，但不得以好看的字替代正确的字。当前没有可用神经偏好数据集、独立风格评分、复现训练流程；因此没有启动 D3PO、Diffusion-DRO 或旧六动作 bandit 的延长训练。
