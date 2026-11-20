# Session O-FULL — comprehensive university mathematics / physics collection

2026-11-20。按作者最新要求，停止为减少书写量而继续最小化符号清单，改为一次覆盖常用
大学数学与物理笔迹的 **19 页、123 行**采集册。已有符号可以重新采集，不要求作者模仿
打印字体；这是一份实用覆盖清单，不宣称穷尽数学 Unicode。

运行：
[20261120T115345363337Z-comprehensive-math-collection-o-full-6e3979c3](../workspace/runs/20261120T115345363337Z-comprehensive-math-collection-o-full-6e3979c3/run.json)。
[完整打印 PDF](../workspace/runs/20261120T115345363337Z-comprehensive-math-collection-o-full-6e3979c3/solutions/collection/session-O_full.pdf) ·
[单页 PDF / 300 DPI PNG](../workspace/runs/20261120T115345363337Z-comprehensive-math-collection-o-full-6e3979c3/solutions/collection/pages) ·
[精确目录 JSON](../workspace/runs/20261120T115345363337Z-comprehensive-math-collection-o-full-6e3979c3/solutions/collection/catalog.json) ·
[逐行说明](../workspace/runs/20261120T115345363337Z-comprehensive-math-collection-o-full-6e3979c3/solutions/collection/README.md)。

源码为 `research/comprehensive_math_collection.py`，与旧 `collection.py` 分开。
原 Session O revision2、M/N 及任何 source_pdf 原件均未覆盖，也未读取个人原 PDF 来生成本册。
每页/行采用独立 ID，例如 `O-FULL-P03-R01`，不会混入旧 O01 等标签。

## 范围和写法

| 页 | 内容 | 行数 |
| --- | --- | ---: |
| 1–3 | 全部 24 个希腊小写、24 个大写、常用变体及易混淆字形 | 21 |
| 4–6 | 数字、全部 Latin 大小写变量、科学记数、单位和真实短式 | 21 |
| 7–8 | 四则、正负、比较、近似、比例、几何、矩阵运算 | 14 |
| 9–10 | 集合、黑板字母 ℕℤℚℝℂ、逻辑、箭头 | 14 |
| 11–12 | 求和、乘积、积分及闭积分、微分、偏导、向量微分、物理记号 | 14 |
| 13–14 | 不同高度的括号、绝对值、范数、长/高/嵌套根式 | 10 |
| 15–16 | 向量、帽号、横线、点/双点、撇号、上下标、组合及左侧指标 | 14 |
| 17–18 | 分数、嵌套短式、二阶/三阶矩阵、行列式、分段函数 | 9 |
| 19 | 各种省略号、长箭头、基向量、期望/概率、正态分布、上下花括线 | 6 |

正常使用自己的笔与惯常字形，在下方空白区写一遍，不要描摹打印轮廓。独立字形之间留出
间隔；如习惯存在多个自然变体，可以在同一行空余处追加，不必强行写成陌生风格。
结构示例按实际含义保留上下标、积分限、分数、矩阵行列和分段关系。

完成后可按页序扫描成一个原色 PDF，保存为 **`source_pdf/session-O_full_source.pdf`**。
空白打印模板不要当作真人笔迹导入。原始扫描保持不变；后续导入必须保留原 PDF、
300 DPI RGB 裁剪、页/行 ID、坐标变换和实际写下的标签。提示只表示意图，不能成为
自动真值；写法不同、漏写或无法确认的内容需要独立审核。

## 打印与工程验收

原 M/N 的“提示＋下方空白线”形式保留，标题改为完整采集册。普通页每页 7 行，
括号/根式/分数每页 5 行，矩阵页 4 行。用途说明单独放在提示上方，避免旧两列布局中
长公式和说明挤在一起。空白区已按实际打印 bbox 测量：普通字形通常有约 43 pt 高，
算子上下限等最紧行仍有 33.48 pt；根式、分数、括号页至少 50.68 pt，矩阵页至少 54 pt。
普通提示为 14 pt，根式、括号与分数大结构页采用 18 pt，以便阅读嵌套层。

- 全部 123 行检查提示/说明互不覆盖、页内边界及空白区高度，未发生裁切。
- 采用本地 STIX 数学字体和 DejaVu 文本字体。记录 7 个实际数学字体文件 SHA，
  每个可见打印 glyph 的 Unicode、glyph ID、字体及位置；没有缺字或字体警告。
- 对 `\pmod` 与 `\land` / `\lor` 的 MathText 不支持项，保留原 intended LaTeX，
  另记录等价的打印表达式及原因，没有输出裸命令或改变数学含义。
- 独立审查发现本机 MathText 将 `\epsilon` 与 `\varepsilon` 映射到同一 open epsilon。
  提示因此显式使用 **ϵ U+03F5 / ε U+03B5**；`\phi` / `\varphi` 保留区分。
  bullet、perp、models、hbar 的实际 MathText 码位与目录意图不同之处也使用显式 Unicode
  印样，并在 `printed_mapping` 说明。空集两个同义宏没有被宣称为两个不同印刷变体。
- 矩阵、分段、overbrace / underbrace 由受控向量结构与原样单元格/标签表达式打印，
  intended LaTeX、结构数据和实际单元表达式同时保留。
- 24 大写希腊中与 Latin 同形的字母、以及小写 omicron，明确记录其 Greek 角色与
  常规同形打印方式，不宣称计算机字体能表达出不存在的独立轮廓区别。

源文件、配置、字体 SHA、逐行坐标、打印验证和全部输出均在新 run 内。
`verify_run` 与源码 Ruff 检查通过；旧采集表及 latest SHA 未变。此时实际新增笔迹为 **0**，
神经训练更新为 **0**。独立 Agent 已逐页目视首个完整版本的第 3、11–19 页，确认字形角色、结构和边界；
其发现嵌套分数提示偏小、上下限书写区较紧，已在本最终候选放大并扩展空白区。
最新版本的五个变化页已再次从实际 PDF 栅格逐页独立复核，印样、结构和书写空间可用，
结论 **Go：可供作者打印填写**。其余 14 页 PNG SHA 与已审版本完全一致。
主 Agent 也已目视最终版本全部五张概览、合计 19 页，确认页面整洁、结构及留白通过。
[独立审查及封存记录](../workspace/analysis/20261120-comprehensive-math-collection-independent/README.md)
包含[最终审核 JSON](../workspace/analysis/20261120-comprehensive-math-collection-independent/FINAL_REVIEW.json)
和 `SEALED_REVIEW.json`。最终状态为工程及 Agent 目视 Go；这只是采集表可用性验收，
不代表作者笔迹标签已确认、神经模型质量改善或生产 Profile 获批。

## 保留的生成失败

以下三个失败运行未删除或覆盖，均保留源码与 traceback：

- `20261120T115037460213Z-comprehensive-math-collection-o-full-a26a5b50`：本地 MathText
  glyph 元组含额外 glyph ID，不能按上游常见的 5 字段解包；改为明确支持 5/6 字段并记录 ID。
- `20261120T115052952459Z-comprehensive-math-collection-o-full-84979f6f`：42 pt 左边界
  浮点往返误差触发过严比较；加入 1e-6 pt 数值容差，没有放宽实际页边界。
- `20261120T115100793971Z-comprehensive-math-collection-o-full-b9921ebd`：16 pt 求和上下限
  使书写区低于预设 25 pt，门禁拒绝。普通提示改为清晰的 14 pt，保留完整内容和大空白区。

首个完整通过版本 `20261120T115127305190Z-comprehensive-math-collection-o-full-b52ad519`
亦保持不变，作为打印可读性改良前对照。最终版本仅将第 13/14/17 页提示提高到 18 pt，
并为第 11 页前两行、第 16 页末行各增加 12 pt 书写空间；其余内容、行数与结构不变。
对两个已封存版本作只读复核：123 行 ID、intended LaTeX 和 Unicode 角色完全一致；
仅第 11、13、14、16、17 页有像素差异，其余 14 页像素完全相同。

实际命令：

```sh
.venv/bin/ruff check research/comprehensive_math_collection.py
MPLCONFIGDIR=/tmp/hwsol-comprehensive-collection-mpl .venv/bin/python -m research.comprehensive_math_collection
```

本册未加入公开分发包；没有上传个人 PDF，也没有把新采集提示称为训练数据。
