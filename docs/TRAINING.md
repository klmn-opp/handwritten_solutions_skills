# CPU 字迹 Studio 使用指南

当前复用固定版本 Handnote 的模板、轨迹栅格化、轮廓提取和字体合并，FontTools 负责字体检查及
尺度/复合字形规范化。没有自研字体描摹器；没有神经微调、模型权重或云 API。
`hwsol-studio` 是独立入口，与 `hwsol profile` 共用实现。Runtime 不导入 `hwsol.training`，
不依赖原始样本。CPU Studio 随主包提供，不增加依赖；Handnote 核心仍含 OpenCV 等工具，
这不等同于把全部字体工具从安装包剥离。未来神经 Adapter 经 `TrainingBackend` 协议扩展，尚未实现可选安装组。

## 采集与输入

```bash
hwsol profile create --profile my-handwriting --base bundled-default@v1
hwsol profile template --output /path/to/letters.pdf --chars "ABCabc0123x" --sets 2
```

省略 create 时，第一次导入会派生 bundled-default。模板生成 PDF 和同名 JSON；保留 JSON，
扫描和轨迹构建依靠其中的格子编号、字符、页面及尺寸。每份模板 1–8 套，同一 Profile 的采集轮次不限。
字符最多 512 个且不重复。默认字符集来自上游；中文等扩展字形需另做质量评估。

| 输入 | 命令与当前支持 |
|---|---|
| Handnote 浏览器轨迹 JSON | `profile import --profile NAME --source capture.json --template letters.json`；支持原始 capture 对象或含 capture 字段的导出对象 |
| 打印模板扫描/照片 | `--source page1.jpg --source page2.jpg --template letters.json`；使用 Handnote 标记校正和扫描分割，保留页序，一轮最多八套 |
| 已有个人 TTF / glyf OTF | `--source personal.ttf`；累积合并真实字形及 rand 变体；可变字体不支持 |
| CFF OTF | 可独立作为一次字体导入并发布；与后续字体/样本合并明确失败，不转换或虚称增量训练成功 |
| 普通照片、PDF 笔记、任意无模板素材 | 原样归档为 archived，不参与训练；没有自动 OCR 标注/分割和 PDF 扫描提取 |

`--kind font/capture/scan/archive` 可显式选择。输入文件不足、格式错误给出具体原因；有效导入的
损坏字体/轨迹会保存原件并标为 quarantined。原件和模板按字节保存并校验哈希；不从图片伪造笔顺、速度或压感。
has_recorded_strokes 只表示合法轨迹输入，不表示压力/时间字段一定存在。扫描当前自动化使用合成模板，
真实照片的光照、透视、杂笔和风格相似度还需要实际样本验收。

轨迹格式示例（**合成工程输入，不是人的笔迹**）：

```json
{"version":1,"cells":[{"index":0,"set":0,"strokes":[[[0.2,0.2],[0.8,0.8]],[[0.2,0.8],[0.8,0.2]]]}]}
```

坐标相对模板格子归一化；index 对应 JSON 模板中的字符，set 范围 0–7。第三坐标可表示真实采集的压力。
浏览器 UI 没有内嵌到本项目；现阶段使用 Handnote 采集导出文件或扫描模板，核心流程均可从 CLI 调用。

## 累积构建与质量

```bash
hwsol profile import --profile my-handwriting --source /path/to/round1.json \
  --template /path/to/letters.json --quality 1 --weight 1 \
  --license-label LicenseRef-My-Own-Samples --license-file /path/to/permission.txt
hwsol profile samples --profile my-handwriting
hwsol profile train --profile my-handwriting --append
hwsol profile evaluate --profile my-handwriting --version v1
hwsol profile publish --profile my-handwriting --version v1
hwsol profiles activate my-handwriting@v1
```

`quality` 为用户标注的 0–1 值；`weight` 为 0–10。quality < 0.5 或 weight = 0 隔离保存，不参与构建。
没有自动风格评分模型。每次构建采用所有 accepted 历史样本，按质量、权重、时间排序决定首选；
同等质量旧样本优先。重复字符保留真实轮廓变体，不靠随机旋转模拟。基础字体的未采集字符继续保留，
不把它们算成个人采集覆盖。缺笔画的非空字形仍可能通过机器检查，应检查预览。

运行时使用上游固定种子均匀选取真实变体，关闭合成形变；**weight 目前影响首选优先级，不是采样概率**。
连续九轮同一字符的真实变体保留已测试。导出字体供其他软件使用时，rand 可保留全部变体，
仅支持 calt 的外部排版软件对后续新增变体的循环行为尚未保证。MTH Adapter 不支持本项目的真实变体选择承诺，显式拒绝这类 Profile。

再导入第二轮并执行 train/evaluate/publish 会产生 v2。所有原件保留；候选构建失败记录在 run.json，
已发布版本继续使用。没有样本删除、重标注或直接修改历史快照的 CLI；需修正质量时可重新导入并记录来源。

评估保存标准英文/数字/数学测试页，检查覆盖、空字形、变体元数据、渲染异常及 PDF/PNG 结构，并比较旧版本。
只有绑定当前候选摘要的通过报告及完整预览才允许发布。它不能判定所有笔画正确、公式语义和风格相似度；
报告明确 `manual_style_review=not_performed`。用户应查看预览再决定激活。publish 默认不激活，`--activate` 可显式开启。

## 回滚、迁移与权限

```bash
hwsol profiles list
hwsol profiles compare my-handwriting@v1 my-handwriting@v2
hwsol profiles rollback my-handwriting@v1
hwsol profiles export my-handwriting@v2 --output /path/to/my-handwriting-v2.zip
hwsol profiles import --source /path/to/my-handwriting-v2.zip
```

导出仅包含 Runtime 资产；迁移训练历史需备份整个数据目录。版本、路径及故障恢复见 [Profile 格式](PROFILE_FORMAT.md)。
个人字体/样本许可默认是未知的本地用途记录；不自动获得分发许可。基底 OFL 和用户提供的许可文本随版本携带，
合并字体更改 family 名称；`local-only` 是发布策略，不能取消字体本来的许可义务。
本项目不上传样本、作业或字体。获授权的真实作者笔迹仍待提供，bundled-default 继续明确标为测试字体。

## 单词、整行和整页布局校准

每次 `--kind calibration` 导入一页标注 JSON。坐标原点在左上，x 向右、y 向下；width、height、em、所有坐标统一用原始页像素或平板单位。em 取正文的参考字号尺度，应在同一作者各页使用一致的估计方式。每条基线按书写方向标出若干点，从首词左缘到末词右缘；words 列表可记录每个词的左右墨迹边界。text 与 math 分开标注。traces 支持 glyph、word、line、math，记录原文和分笔段 strokes；每个新段表示一次抬笔后重新落笔。点必有 x、y，time 与 0–1 的 pressure 仅在设备有记录时填写。[合成格式示例](../examples/realism/calibration.json)

对于照片或扫描，先选定每一行的紧致矩形和转写。regions.json 形如 `{"em":24,"lines":[{"kind":"text","text":"Two words","box":[20,40,200,75]}]}`，box 为校正 EXIF 方向后的左、上、右、下像素坐标。PDF 可先用 `hwsol read-page` 转成逐页 PNG。以下命令根据标注区域的前景投影估计基线和词间空白，再核对生成的 JSON。表格线、背景纹理、交叉行和相连的单词需要人工修正。扫描只产生几何观察量。

```bash
hwsol profile measure-page --source page.png --regions regions.json --output page-layout.json
hwsol profile import --profile my-handwriting --kind calibration --source page-layout.json
hwsol profile train --profile my-handwriting --append
hwsol profile evaluate --profile my-handwriting --version v1
hwsol profile publish --profile my-handwriting --version v1
```

原始 JSON 留在 samples 中。训练累积所有 accepted 轮次和基底的校准页，写入摘要绑定的 calibration.json，包括缩进、基线、词距、页面上下文及可用的笔压、下行笔段倾斜统计。校准评估额外生成 combined.pdf 供检查，发布前验证其摘要。自然段轨迹用于校准；生产文字仍由个人字体和多字形轮廓构成。单个数学字符的轮廓训练沿用原有字格模板。若希望把原始在线轨迹随快照迁移，在 train 时显式加 `--include-traces`。之后导出该版本也会导出原始轨迹、转写和相关模板。后续版本可独立选择该选项。

## 本人均值横线与数学横线

当前开发版支持从扫描中测量横线轮廓，经同一 `--kind calibration` 流程累积发布。
标注 JSON 的 `math_strokes` 可单独存在，或和 `lines` 共存；原页摘要必须与每条横线的
`source_sha256` 一致，裁剪框必须在页内。字段定义见 [Profile 格式](PROFILE_FORMAT.md)。
先人工确认裁剪区域并隔离横线墨迹，再取沿 x 方向的上下边界；`measure-page` 尚不自动识别这些横线。
照片不产生真实 time、pressure 或 pen-up 数据，`has_recorded_strokes` 仍为 false。

Runtime 在 MathText 完成公式布局后，使用 `bar` 轮廓替换 U+0304 均值横线，使用 `rule` 轮廓
替换水平数学矩形线段（包括分数线，以及根号、overline 的水平部分）。
它保持线段原有横向范围和中心高度，用相邻字形间隙限制笔画上下起伏，随后以填充轮廓输出。
这条路径在四种 realism 模式下均使用已发布的个人资产；没有样本的 Profile 保持原有行为。
报告的 `handwriting.math_strokes` 保存来源、替换前后范围和垂直压缩比例；
未替换的根号等回退仍会如实记录。这项校准不证明整体风格已达标，也不属于 RL 训练。

用于提取横线的页面应归入训练材料，之后以另一份从未参与校准的手写页做自然度验收。

## CPU 渲染策略强化学习

当前开发版可通过 `hwsol lab rl prepare/train` 学习真实字形选择、笔宽一致性和相关变化的策略分布。
它复用已发布 Profile，不重写字形、不产生新的笔顺或压感。自动指标预训练与人类偏好训练分别标记；
数学错误先处理，策略不自动激活。命令、已跑结果与数据要求见 [RL 实验指南](RL_RENDERING.md)。

## 0.4.0 候选与补字队列

`hwsol-studio collection-plan --profile profile@vN --source answer.md --output new-directory`
复用 Handnote 生成独立补字模板。日常 Runtime 不要求为少量数学回退立即重建 Profile。
CPU `train --append` 仍是全历史字体构建与统计校准；新的候选加入度量资产，不能称为神经训练。
支持只含词/行轨迹的 calibration；扫描估计必须显式标注 origin，不能标为 recorded-pen。
新候选没有自动继承旧版的作者验收，默认 Profile 保持原状。见 [本轮实现与未完成项](NEXT_STAGE_DEVELOPMENT.md)。
