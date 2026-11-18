# v0.3.0 手写渲染诊断

基线为 e4e6349。实际用原版命令渲染 mixed 的五题，固定 demo@v2、白纸、150 DPI、seed 42。该 Profile 是既有 CPU Studio 合成测试夹具；只有 x 有采集覆盖，其他字符继承 Patrick Hand。原件在 out/realism/release-baseline。

| 路径 | 代码和产物揭示的行为 |
|---|---|
| 正式作业 | jobs.runtime.build → jobs.layout.render_one → SinglePageRenderer.render_vector，委托固定版本 Handnote。合并向量 PDF 后以 PDFium 逐页生成 PNG。Agent 提供解答 Markdown。 |
| 普通文字 | Handnote 的 _tokenize 按单词决定能否折行，render_page 仍循环调用 draw_char。advance、行高和起点由字体度量和配置决定。没有英文单词连接模型或整行轨迹。 |
| 行内与独立公式 | _math_path 调 MathText 解析、取得字形索引、位置、大小和线条，再取实际轮廓。Profile 同时作为 rm/it/bf 的首选字体参与测量；上下标、分数线、根号、算子间距仍服从数学排版。相同 tex/字号复用缓存。独立公式默认水平居中。 |
| 矩阵 | 独立矩阵有固定的格宽和行高，单字符格直接走正文字体，其他格与前缀走数学路径。括号由几何画线生成；嵌入表达式另有合成路径。 |
| 字体回退 | 正文预检拦截缺字。公式由 MathText 实际选择字体，显式配置 STIX 回退。五题中第 4、9 题的 U+22C5 乘点、第 6 题的 U+0304 组合上横线及 U+03C3 小写 sigma 实际来自 STIXGeneral.ttf。第 2、7 题没有检测到外部数学字形。同一字体文件中的基底字和采集字也要区分。 |
| 随机性 | hwsol.adapters.handnote.upstream_config 关闭字符旋转、水平/垂直抖动、字号抖动和轮廓形变，墨色固定。存在 GSUB 变体时，正文与命中 Profile 的数学字形能抽取候选。上游矩阵括号另有带种子的绘制。实验无需增大独立字符噪声。 |
| 训练与探针 | Handnote 采集经栅格化、描摹、合并到字体。日常作业读取已发布字体；原始笔顺、时间、压力、词/行与页习惯均未进入基线。render-probe 是另一路径；MarkdownToHandwrite 仅在显式指定解释器时启用，正文与数学使用所选字体并配置 STIX。隔离 worker 关闭 SDT、第二层形变，含多变体的 Profile 无法在此兼容路径渲染。 |
| 原有报告盲区 | 作业报告列出实际外部数学码点，但不统计继承字/本人字/变体的使用次数；普通覆盖报告按 cmap 估算。几何笔画没有字体来源。 |

优先处理的两个可独立验证的问题是整行与整页的位置过于规则，以及同一轮廓在整页反复出现。保留 MathText 解析和测量，将完整公式或矩阵作为规划单位，并在最终变换后复查墨迹边界。

## 上游取舍

- Handnote 的采集、GSUB、字形记录器和向量 PDF 可以直接作为 Adapter 的底层。固定提交支持单字多组采集和数学组件组装。[Handnote 固定版本](https://github.com/zh271/handnote/tree/114919f183817643c795e47e851ea25b96e3c393)
- MarkdownToHandwrite 的 SDT 笔画路径与现有隔离环境可供对照；捆绑轨迹的授权和作者匹配尚未解决，直接启用会混入其他人的字形。[固定版本](https://github.com/lyc07/MarkdownToHandwrite/tree/634cf07987a9c84c59f808e03f879c9d93485374)
- My Text in Your Handwriting 以标注的真实笔迹学习字形、字距和笔画特征，支持把词间空白和行内关联作为校准量。[UCL 作者稿](https://discovery.ucl.ac.uk/id/eprint/10075697/)
- HMEG 将符号关系图映射到布局、掩码和公式图像。作者提供训练代码和权重链接。本项目可借鉴保留结构关系的思路；作业公式继续复用确定性解析器。[CVPR 论文](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Generating_Handwritten_Mathematical_Expressions_From_Symbol_Graphs_An_End-to-End_Pipeline_CVPR_2024_paper.html) · [作者仓库](https://github.com/AiArt-Gao/HMEG)
- DiffInk 直接生成完整行的在线轨迹。作者在 2026-03-21 公布代码和权重，README 指定 Python 3.8 和 CUDA 推理。代码为 MIT，权重沿用 CASIA 数据许可。针对本项目英文题解的 CER、作者相似度、显存和延迟没有实测；已有 lab 的独立转写和评分协议可供后续评估。论文报告不能代替本地英文测量。[作者 README](https://github.com/awei669/DiffInk) · [论文](https://arxiv.org/abs/2509.23624)

## 同进程切换 Profile 的复现

原始 e4e6349 在一个 Python 进程内先以 bundled-default@v1 渲染 `x=2+3`，再以 demo@v2 渲染相同公式。第二次记录的 x、=、2、+、3 全部来自 PatrickHand-Regular.ttf。Matplotlib 的 MathText 解析缓存键未包含自定义数学字体，按 family 搜索也无法区分同名快照。作业和 Handnote 探针现均按快照的字体文件绑定 rm、it、bf，并在持有渲染锁的每次渲染开始时清除该解析缓存。STIX 仍承担实际缺字。独立进程、同一固定 Profile 的五题 Baseline 另与原版逐字节比对。
