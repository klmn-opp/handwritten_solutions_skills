# M1 bounded historical missing-symbol review

2026-11-20。已完成本轮 **20 个原迹区域** 的有界复核。找到两枚可继续独立审核的真实
`∈` 原色裁剪；其他请求项没有得到可直接接入的可靠新字形。本轮不改 Profile、不训练、
不写回历史标签，也没有把任何新素材接入当前二十页候选。

运行：
[20261120T112651209228Z-missing-math-historical-review-88800d28](../workspace/runs/20261120T112651209228Z-missing-math-historical-review-88800d28/run.json)。
[候选原迹 PDF](../workspace/runs/20261120T112651209228Z-missing-math-historical-review-88800d28/comparisons/candidate-contact-sheets.pdf) ·
[逐项目视决定](../workspace/runs/20261120T112651209228Z-missing-math-historical-review-88800d28/reports/visual-review.json) ·
[新裁剪及来源变换](../workspace/runs/20261120T112651209228Z-missing-math-historical-review-88800d28/reports/reviewed-extractions.json)。

## 搜索范围与隔离

先读取 catalog 的文档 SHA 和样本祖先 ID，再过滤，之后才查询样本标签及 OCR 文本。
封存 C 的 SHA 为 `ad5e01fdca0920d999893ccdf71933e3303c48f95ebac99e4d7a86818ae99b7d`；
所有共享该来源的样本、其祖先链衍生和匹配 Session C 名称的路径都不得进入查询或图片读取。
额外排除了旧 `heldout-mean-variance.jpg` 及四个样本。没有打开 C 的原件、图像或衍生素材。

- 两个真实 SQLite catalog 在排除后覆盖 **16 个文档记录、27 个已导入页、255 个样本、
  206 条识别记录**。17 文档/28 页是排除旧 heldout 文档前的总数，不能当成已搜范围。
- 查询既有 UniMERNet/OCR 文本及 verified/candidate 标签。15 类目标精确命中
  **12 条识别记录、8 个去重区域**；其中 3 个 PDF 区域原先已被 rejected。
- 既有 candidate_text 又提供了同一照片中的成员关系线索。合并后共 **14 个原迹区域**，
  来自 4 份历史 PDF 和 1 张原生照片；再检查已导入 Formula Sheet I 的 6 页，合计 20。
- 联合注册表的 **84 条历史/M/N 数学观察**先按文档 SHA 过滤，再检查标签；没有目标标签
  命中。没有把其图像重复计成新素材。

没有新增 OCR、外部请求或云上传。检索覆盖的是已有 catalog、既有识别和指定六页公式表；
**不是穷尽全部历史 PDF 页数**。原始已导入 PDF 通常仅少数页有分割和识别记录。
完整 SQL 查询方式及来源清单保存在源码、search-scope.json 和 registry-search.json。

## 原迹判定

| 请求符号 | 实际发现 | 本轮决定 |
| --- | --- | --- |
| ∈ | Formula Sheet I 第 2 页两处清楚的手写 x∈R；第 5 页及旧照片还有成员关系 | 两个 PDF 原色裁剪 Agent 接受，等待独立审核；不自动接入 |
| ∬ | 公式表第 5 页作者实际写了两个相邻积分号 | 保留真实 `\int\int` 标签；邻近下限 A 侵入矩形裁剪，单独字形 No-Go |
| κ、ζ | OCR 在 Statistics 第 69 页长段输出多次命中 | 原图是 Poisson 推导中的 α 和 Latin k；OCR 误识别，原 rejected 状态保留 |
| ≈ | OCR 命中 Calculus 2 第 68 页和 Calculus 3 第 124 页等 | 实际为英文/中文与不等式，未见可靠 ≈；No-Go |
| Φ | 印刷公式页出现相关记号 | 印刷体不是个人笔迹；没有接受 |
| ℝ | 手写上下文使用普通 R | 不把普通 R 擅自标成黑板粗体 ℝ；No-Go |
| ξ、η、ψ、∏、∮、⟨、⟩、‖ | 本轮已搜范围无可靠个人原迹 | 保留明确回退及 Session O 补采 |

其他重要拒绝：ICS 第 17 页的 OCR ∈ 实际来自红色英文 `False`；两个照片区域的 ∈ 命中
其实是 ε/不等式。公式表第 1、4 页是印刷参考，不能因内容匹配而加入真人样本。第 6 页空白。
20 区域中 10 个被接受为“有相关真实手写内容的区域”，10 个拒绝；前者包含同一照片的八行，
**不是十枚新增可靠字形、十个独立来源或十个训练样本**。

## 两枚可审核的 ∈ 原迹

共同来源：`Formula Sheet I _source.pdf`，SHA
`c6fa4ab4f530863d0d3c207e4ef782bdc24930452f54f73998a145246c575de9`，第 2 页。
先由原始 PDF 以 300 DPI 渲染 RGB，再做矩形裁剪；原图像素未清理、补画、加粗或重标。

| 原迹 | 300 DPI 页坐标框 | 裁剪 |
| --- | --- | --- |
| 左侧 x∈R 中的 ∈ | [1060, 970, 1130, 1055] | [70×85 RGB](../workspace/runs/20261120T112651209228Z-missing-math-historical-review-88800d28/solutions/reviewed-extractions/membership-left.png) |
| 右侧 x∈R 中的 ∈ | [2117, 963, 2170, 1043] | [53×80 RGB](../workspace/runs/20261120T112651209228Z-missing-math-historical-review-88800d28/solutions/reviewed-extractions/membership-right.png) |

原 PDF 的字节副本、原页坐标、catalog→300 DPI 变换、PDF rotation/cropbox、
crop→PDF 矩阵、原色父区域及全部 SHA 均保留。旧照片保持原生 RGB 和 144 DPI 元数据，
没有上采样冒充新的 300 DPI 扫描。双积分
[上下文裁剪](../workspace/runs/20261120T112651209228Z-missing-math-historical-review-88800d28/solutions/reviewed-extractions/double-integral-context.png)
明确保留侵入的 A 笔画；没有擦除它来制造“干净样本”。

## Go / No-Go 与后续

**Go：** 将两枚 ∈ 裁剪提交给主 Agent 做独立标签、边界及轮廓提取评估。此为来源复核，
还没有构建 glyph、更换正式字体或证明完整页面质量。

**No-Go：** 自动集成任意新符号、把 OCR 候选当真值、把普通 R 改标签为 ℝ、把污染双积分
裁剪直接作为干净 glyph。其余目标保持 Session O 最小补采；∈ 项在独立提取和页面验收前
不取消。‖ 和 ∬ 可以另开受控结构布局任务，分别复用真实单竖线/积分号，但应记为组合结构，
不能声称找到了新的独立原迹。本轮没有实施该路线。

实际命令：

```sh
MPLCONFIGDIR=/tmp/hwsol-missing-math-mpl .venv/bin/python -m research.missing_math_review
MPLCONFIGDIR=/tmp/hwsol-missing-math-mpl .venv/bin/python -m research.missing_math_review --finish workspace/runs/20261120T112651209228Z-missing-math-historical-review-88800d28 --decisions /tmp/hwsol-missing-math-visual-decisions.json
.venv/bin/ruff check research/missing_math_review.py
```

prepare 和最终 review 的源码分别留在 logs/source.py 与 logs/final-review-source.py；
二十项决定保存在运行内，不依赖临时决定文件恢复。两 SQLite 的原 SHA 复核不变。
这份报告和原迹只在本地研发区，公开包 allowlist 不包含它们。
