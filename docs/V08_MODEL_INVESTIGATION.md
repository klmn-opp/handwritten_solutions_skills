# v0.8 英文模型调查与诊断边界

本文件记录源码、真实权重与运行机制；最终定量比较及训练准入结论见 [Go/No-Go](V08_GO_NO_GO.md)。所有检查在 `experiment/english-evidence-v08`，CPU Runtime 保持v0.6。

## 固定上游

| 项目 | 本轮固定版本 | 用途 |
|---|---|---|
| Handnote | 114919f183817643c795e47e851ea25b96e3c393 | 既有字体与DiffBrush接口 |
| DiffBrush | da9addc1140bdfec463b2d41a974a0a392f80798 | 原始UNet/DDIM同输入对照 |
| DiffusionPen | 059e2f2bcb7372231038719e955597917c1a8783 | 官方UNet、MobileNet风格编码器 |
| DiffusionPen发布权重 | HF 5636a68365001435ec76553114313c22c6f1ca07 | IAM EMA模型与风格编码器 |
| Reforge | e59126bd151c74db1f57970ec0f398134051cb7a | 只读组件审计，没有执行或移植其完整管线 |
| TrOCR handwritten | HF eaacaf452b06415df8f10bb6fad3a4c11e609406 | IAM手写行识别，独立观察器 |
| DiffInk | 65055545fbef5e22fbee0811a1ef3855ca242c22 | 保留v0.7真实中文在线实验，重查发布词表 |

源码快照与逐文件摘要在 `workspace/sources/v08/`，实际研究脚本还复制到每个run的 `logs/source/`。新权重与官方LFS摘要逐项一致：TrOCR 1,333,384,464B；DiffusionPen EMA 679,837,485B；两个风格编码器各10,880,397B。只实际使用 `iam_style_diffusionpen.pth`，triplet变体虽下载但未推理，不把它列为已比较模型。共用SD1.5 VAE 334,643,276B，来源及摘要保留；原有DiffBrush权重1,165,546,469B不变。

## DiffBrush：已定位与尚不能归因的部分

- 官方 `generate.py` 过滤35–61字符的生成文本。Handnote的 `write` 接口只检查字符集，直接调用不会自动采用CLI的文本分行规则。本基准刻意保留短句、长句与超长词；35–61是上游生成脚本条件，**不能误写成已经证明的训练集长度硬限制**。
- 对32次smoke对照，保存完整输入、编码索引、内容张量shape/SHA；从索引还原文本完全相等。Handnote安全转换的Unifont与官方原文件逐元素一致。未发现Adapter丢弃输入字符。
- 原始官方UNet严格加载相同权重，使用相同噪声、风格和内容张量执行50步DDIM。Handnote缓存内容编码以减少重复计算，官方逐步计算。官方图像floor转换与Handnote round转换分别保留；round后最大平均绝对灰度差0.03493/255，局部最大差41/255。它们不是逐像素相同，但整体差异很小，尚无证据把主要漏写/重复归因于Adapter。
- 本人参考原图1294×75，按高64缩放约1104宽，Handnote会右裁到1024；这是**参考风格图**被裁，不是目标文本被裁。raw与完整保留参考宽度的fit两组使用相同16文本及seed42/43。两组只是预处理消融，没有据其成绩调整留出集。
- 原图固定64×1024。未使用Handnote trim作为最终图；保留trim副本便于对照。每张最终PDF在72dpi逐像素对回原生PNG，通过则排除该导出步骤新增裁切。触边墨迹和OCR尾删仍是疑似模型画布/生成问题，必须人工复核，不能断言所有触边都是漏字。
- 视觉辅助检查发现短句 `Check the sign.` 的实际原图重复生成sign等笔迹；这不是正式作者评分。失败队列保留全部不一致项，而非仅挑选少数好看的图。
- 全120条×2种子中，`% [ ] =` 涉及的两条开发文本共4次请求明确拒绝。没有删除字符再伪装成功。

## DiffusionPen：真实英文、五个扫描词条件，不是训练

官方代码以IAM加载器和词级采样为主，本轮直接复用其UNet/feature_extractor定义，将目标文本和五张本人扫描词接入50步DDIM。五词来自已有同一作者扫描行，保留原图、裁剪框、转写与摘要，不借用外部字体或估计在线轨迹。

EMA包含全部Canine参数；从固定配置构建后，去掉DataParallel前缀，**所有模型键严格加载**。风格网络不下载随机代替权重；仅剔除原训练分类器两个不参与特征提取的键，其余严格加载。SD1.5 VAE和官方调度器配置均固定。官方 `max_length=40` 包含两个特殊token，本实现拒绝超过38字符的单词，不静默截断。此次16条smoke均在此容量内。

原生输出为64×256单词图；所有词及标点均由模型实际生成。合成文字行只裁水平方向外部空白，保留64像素完整垂直画布，用16像素空格连接。没有合成标点、没有重绘末字，没有强制相同字母高度。此最小组合会显露模型按词填充画布造成的比例/基线差异；它不是一个已经完成连续整行书写的系统。个人参考条件推理成功不等于个人化训练成功。

[官方代码](https://github.com/koninik/DiffusionPen/tree/059e2f2bcb7372231038719e955597917c1a8783)与[发布模型卡](https://huggingface.co/konnik/DiffusionPen/tree/5636a68365001435ec76553114313c22c6f1ca07)支持IAM英文与少样本风格条件路线；没有真实监督微调实验，无法承诺训练后的相似性提升。

## Reforge：可借鉴组件与复用阻碍

[固定源码](https://github.com/peterzat/reforge/tree/e59126bd151c74db1f57970ec0f398134051cb7a)是DiffusionPen工程封装，不是另一个已训练的新模型。可研究其词分割、x-height与基线估计、单次重采样、内容候选筛选、清单记录及人工标注测试样本。

本地审计发现：UNet加载使用strict=False；有长词拆分、收缩词拼接和程序化标点；使用OCR做候选选择。直接复用前须改成可核查的严格加载、逐笔来源标记，并使用独立于筛选器的最终识别器，否则会发生评价循环。合成标点不得被记为本人采集或神经模型输出。固定快照未找到LICENSE文件，故本轮没有复制其实现；README的效果声明与其mock测试不能作为本项目真实推理证据。

## DiffInk：当前发布模型不能直接生成任意英文

现有发布词表共2,648项。对本基准120条做只读覆盖核查，**0/120完全覆盖**；缺空格，并缺若干大小写字母（例如v、j、q、x、z、K、M、Q、X、Y、Z）及部分标点。完整缺字清单在 `v08-evidence-analysis/reports/diffink-english-coverage.json`。不能删除空格或替换未知字再称任意英文成功。

已下载发布权重使用CASIA-OLHWDB；论文/README提及IAM-OnDB不等于已经提供可用的IAM英文checkpoint。v0.7复现的是具有字符边界的真实在线笔迹前缀续写，接口还保留同字符数和同30%前缀约束；静态扫描不能直接充当其真实在线风格输入。

因此，**不改变当前权重/词表的情况下，答案是否定的**。未来可能通过作者发布英文权重、经过论证的词表扩展与监督适配解决；现有证据并不证明必须从零预训练，也不能保证少量个人数据微调足以解决。没有进行任何中文替代英文的“成功演示”。旧CUDA轨迹、权重、环境和全部运行保留。

## 许可证与部署条件

| 路线 | 代码 / 权重 / 数据的分别核查 |
|---|---|
| DiffBrush | [代码MIT](https://github.com/dailenson/DiffBrush/blob/da9addc1140bdfec463b2d41a974a0a392f80798/LICENSE)；固定下载checkpoint未发现独立充分权重许可声明，不据代码许可推断所有权利；IAM数据限非商业研究 |
| DiffusionPen | 代码MIT，HF权重模型卡标MIT；IAM数据使用条款另外存在；SD1.5 VAE使用CreativeML OpenRAIL-M，不能只看顶层MIT |
| DiffInk | [官方说明](https://github.com/awei669/DiffInk#license)区分代码MIT与CASIA来源权重；权重遵循CASIA原数据条件 |
| Reforge | 固定快照未找到LICENSE，暂不直接复制；其依赖DiffusionPen和SD VAE的限制仍需保留 |
| TrOCR | 模型卡MIT，IAM手写域微调；只作独立评测，未加入CPU Runtime |

[原始IAM条款](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)限定非商业研究；[CASIA条件](https://www.nlpr.ia.ac.cn/databases/handwriting/Application_form.html)区分学术与商业用途。本轮未下载新的IAM训练集，未上传本人样本，未打包分发模型权重；商业交付及额外数据使用许可尚未完成确认。
