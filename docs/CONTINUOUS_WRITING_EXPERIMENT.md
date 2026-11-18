# Continuous writing experiment — unsuccessful inference

本轮没有成功运行生成模型，未产生可用于宣称自然度提升的模型输出。实际执行的是锁定 Handnote 的
DiffBrush CLI。首次退出于可选依赖检查；随后补齐独立 CPU 依赖并重试，因权重不完整仍未进入模型推理。不存在 mock 图替代模型结果。

锁定 Handnote `114919f183817643c795e47e851ea25b96e3c393` 的已安装源码确实包含
`linegen/generate.py::LineGenerator`、`generate_lines` 和 `handnote line generate/fetch/bench`。
其 `weights.py` 固定 DiffBrush checkpoint、VAE 与内容字形摘要，checkpoint 约 1.17 GB，VAE 约 335 MB。
接口验证来自本地固定版本源码，不是根据最新 README 猜测。

隔离环境位于 `/tmp/hwsol-next-stage-lineenv`，未把 torch 等研究依赖装入生产 Runtime。
输入是本人自然英文扫描行，正文固定为 `The median is the middle ordered value.`，seed=42，CPU/50 steps。
执行命令、参考图摘要、退出码和标准错误保存在本地 `out/next-stage-2026-11-18/line-experiment-final/`。
本机 `nvidia-smi` 无法连接驱动；未发现已就绪模型权重。首次入口报缺少 linegen 可选依赖。随后在独立环境安装 torch 2.6.0+cpu、torchvision 0.21.0+cpu、diffusers 0.32.2、einops 0.8.1、huggingface-hub 0.28.1、safetensors 0.5.3，available() 为 true。
真实权重下载持续推进到 295,698,432 / 1,165,546,469 字节后，本轮主动停止该可行性探查，保留部分文件；这不是网络失败，也不是模型成功。再次真实调用因权重不完整失败。生产环境未加入这些依赖。
进程耗时仅是失败检查耗时，不能称为推理速度。CER、WER、风格自然度、显存及推理时间均为未测量。

| 路线 | 本轮状态 | 后续所需证据 |
|---|---|---|
| Handnote / DiffBrush | 已验证接口、安装 CPU 依赖、获取部分真实权重并重试；未进入推理 | 独立模型环境、真实权重、相同参考/内容/输出条件、独立转写与资源实测 |
| DiffInk | 官方源码/推理路径复核，未运行 | 固定源码及权重，真实在线风格输入或明确标识的估计轨迹 |
| InkSight | 官方扫描到轨迹路线复核，未运行 | 小规模真实扫描推理；输出只能标为 estimated-trajectory，不能冒充实测笔顺 |
| CPU 完整词轨迹组合 | TextLineRenderer 实验实现；本人无词轨迹，无法做真实效果对比 | 足够的完整词/字母组合、真实抬笔边界，缺词回退到个人字体 |

DiffBrush 官方仓库提供完整英文行模型及推理入口；其输出为图像，仍需检查漏字和重复。
[DiffBrush 官方源码](https://github.com/dailenson/DiffBrush)。
DiffInk 官方实现提供文本到在线笔迹的推理，权重遵循数据来源的相关条款；本轮没有训练或下载运行。
[DiffInk 官方源码](https://github.com/awei669/DiffInk)。
InkSight 将扫描图转成估计数字墨迹，官方提供模型与示例，适合独立评估轨迹提取，不意味着恢复了真实时间/压力。
[InkSight 官方源码](https://github.com/google-research/inksight)。

`hwsol.lab.text_lines` 不被生产公式路径导入。请求拒绝 LaTeX、等式与多行字符串，结果状态为
`generated-unvalidated` 或 `composed-unvalidated`；独立转写通过前没有自动发布或生产切换。
CER/WER 检查使用独立转写，不把输入 prompt 复制成“识别结果”。普通 CPU 用户继续使用当前个人字体 Runtime。

本阶段决定：先交付可关闭的 M0–M3 CPU 改进和真实样张，暂不投入模型适配/训练；模型比较仍列为未完成，
不能以接口测试通过替代真实推理成功。
