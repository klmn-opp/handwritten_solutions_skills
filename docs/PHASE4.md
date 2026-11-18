# Phase 4 · 英文生成式笔迹实验

`hwsol lab` 用于本地英文词和行的研究。普通作业继续使用已发布的字形 Profile 和确定性数学排版。实验命令支持准备语料、调用独立环境、导入生成图和盲评。所有后端都处于研究状态。

| 后端与固定源码 | 已核实的接口及资产 | 当前支持 |
|---|---|---|
| [DiffusionPen](https://github.com/koninik/DiffusionPen/tree/059e2f2bcb7372231038719e955597917c1a8783) | 英文词，论文方法使用 5 个风格示例。train.py 的 single_sampling 写死 text 和 word，并随机选训练风格标号。上游还需要风格提取器、扩散权重和 VAE。 | 可导入独立实验产生的词图。原入口不提供通用 prompt 和个人样本参数。 |
| [DiffBrush](https://github.com/dailenson/DiffBrush/tree/da9addc1140bdfec463b2d41a974a0a392f80798) | generate.py 支持 --pretrained_model，IAM 配置使用 15 张风格图，采样器筛选 35–61 字符的行。使用 NCCL、CUDA、torch 1.13.1 和本地 VAE。 | 可调用固定版本的批采样入口，可从 fid_wri 的单一作者目录导入。 |
| [One-DM](https://github.com/dailenson/One-DM/tree/dde2205a70a2c70d1786503d198a795358c80ee4) | 单参考样本的词生成，test.py 接受 --one_dm 与 --generate_type oov_u。需要英文风格图、对应 Laplace 图、字形内容数据和 VAE。上游环境为 Python 3.8、torch 1.13.1、CUDA 11.7。 | 可调用固定版本批采样，从 oov_u 下的单一作者目录导入。 |

三份固定源码的 LICENSE 均为 MIT。IAM 官方下载要求注册并接受非商业研究条款。实验包只分发本项目的适配器和原创英文语料。具体 checkpoint、VAE、字形素材、风格样本、其衍生产物的授权需要分别记录。本项目未取得这些资产的再分发授权。[IAM 下载及许可](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)

## 单次实验

1. 运行 `hwsol lab models` 查看版本，再运行 `hwsol lab prepare --model one-dm --output out/trial`。固定语料为 12 个英文单词，DiffBrush 使用 8 行英文，保留大小写、空格和标点。每次试验选定同一位作者，提前确定种子、参考样本和权重版本。
2. 在独立研究环境准备有使用许可的代码、检查点、本地 VAE 和上游所需的预处理资源。指定独立工作目录；One-DM 从该目录的 `data/oov.common_words` 读语料，DiffBrush 从 `data/wikitext103.te` 读语料。将生成的 prompts.txt 原样复制到对应位置，保留上游要求的其他本地资源。源码 checkout 和现有数据目录可保持独立。
3. 对于 One-DM 和 DiffBrush，用下面的命令预检。去掉 `--dry-run` 执行同一命令。执行前检查实际 Git 提交、源码状态、语料、配置、权重和 VAE 位置。子进程使用独立 torchrun，本地记录 argv、提交、权重摘要、耗时、退出状态和日志。HF 离线环境变量传给上游，机器级网络隔离由实验环境提供。
4. 导入一个作者的所有输出。One-DM 的 source 指向 `raw/images/oov_u/<writer>`。DiffBrush 指向 `raw/images/fid_wri/<writer>` 并传 `--writer <writer>`。其他外部试验可按 prompts.txt 的顺序放入 `01.png` 等文件，使用 `--numbered`。空白、缺图或同一行有多个未区分的候选会中止导入。
5. 将 `batch/blind/` 和授权的风格参照提供给本地评阅者，保留 `key.json` 与 prompts.txt 在评阅者视野之外。评阅者逐图填写 review.json 的实际读数及 1–5 风格分，读不出来可填空字符串。1 为明显不似，3 为部分相似，5 为与参照高度一致。运行人填写 run.json 中实际设备、种子、checkpoint 的 SHA-256、风格参照文件相对路径及 SHA-256，以及记录各项授权依据的本地文本相对路径。
6. 运行 `hwsol lab evaluate --run out/trial` 获得 JSON，保存 stdout 即可归档。报告包含逐图字符编辑距离、CER、整词/整行完全匹配数、风格评分与文件摘要。格式错误和未达评估门槛均返回退出码 2。人评转写及许可材料由实验提供者负责；评分器验证文件一致性及声明完整性。

```bash
hwsol lab sample --run out/trial --checkout /research/One-DM --workdir /research/iam-local --torchrun /research/env/bin/torchrun --checkpoint /research/model.pt --vae /research/model_zoo --devices 1 --dry-run
hwsol lab collect --run out/trial --source out/trial/raw/images/oov_u/WRITER_ID
hwsol lab evaluate --run out/trial
```

## 研究门槛与实际验证

必须逐图完整转写，当前小语料要求零字符错误、输出图互异、平均风格分至少 4/5、单图至少 3/5，并有完整本地来源记录。通过只获得扩大实验资格。生产接入还需多作者的独立留出集、重复种子、人工复核误写率与风格、可用硬件的延时/显存、分发授权和确定性公式组合测试。

Linux 实测了三个适配器的文件导入、盲表、错误计数与门槛，以及两种采样器的参数预检和真实子进程失败记录。测试用合成图片，分数只用于检查计算。主机查询识别到 RTX 5060 Laptop GPU、8151 MiB、驱动 580.126.09。本次没有获授权且配套齐全的神经采样输入，故没有 GPU 推理、训练、风格相似度、实际误写率、峰值显存或吞吐结果。上游公开的 CUDA 路径也未验证 CPU 或 Apple MPS 兼容性。
