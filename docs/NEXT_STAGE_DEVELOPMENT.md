# Next development stage — CPU realism candidate

2026-11-18。本轮在现有工程上继续开发。前置审计是 [CURRENT_STATE_AUDIT.md](CURRENT_STATE_AUDIT.md)，其数字与结论仍是历史记录。本轮并未取得作者对新版自然度的验收。

## 实现与边界

| 阶段 | 实现 | 保留的限制 |
|---|---|---|
| M0 | `67c2b88` 保存审计时源码；0.4.0 固定依赖；输出记录 commit、源码/上游摘要、lock、Profile 资产、模式、策略摘要和 seed；发行包嵌入 commit/lock | 本地发行，未推送远端；其他操作系统未实测 |
| M1 | FontTools 测量个人真实轮廓的精确墨迹边界、advance、面积重心、基线偏移、类别；类别和字符参考、异常权重；排版前派生字体；词内共享尺度及相关残差 | 英文仍由轮廓拼排，没有生成连笔；测得的是已构建扫描字体的尺度，不是原纸张毫米尺度 |
| M2 | 每个可见 recorder item 的来源；数学结构分类；复用个人横线和适配的个人括号；历史作业缺字队列与 Handnote 模板 | 复杂嵌套结构标签使用局部几何推断；不足的根号、箭头、方括号继续显式回退 |
| M3 | 解释/起始式/续行/结论/行内数学/多行推导角色；栏起点坐标、公式组起点和等号约束；按角色分布取值 | 角色为启发式，可人工标注校准；未知旧标注使用 inset 分布，不声称已学会全页布局；不自动生成双栏 |
| M4 | 可选 TextLineRenderer，真实 Handnote DiffBrush 子进程入口、实测失败记录；独立转写 CER/WER；真实词轨迹 CPU 组合接口 | 没有成功的神经推理或模型对照，见[独立实验报告](CONTINUOUS_WRITING_EXPERIMENT.md) |
| M5 | 全历史构建加入 glyph-metrics 资产；支持单独词/行轨迹导入并区分扫描轮廓、估计轨迹和真实采集；创建不激活的候选 | 当前本人无真实词/行轨迹，无生成模型适配参数；新字符需要以后独立采集 |

所有字体和统计构建均不是神经网络训练。原有 Profile 发布、回滚、原始样本、bandit 实验、PDF/PNG 校验继续保留。

## 显式开关与重放

旧 manifest 的默认行为保留。新因素在 `render` 内，与四种 `realism` 模式独立：

```json
{
  "realism": "combined",
  "profile": "personal@v3",
  "render": {
    "glyph_metrics": "contextual",
    "semantic_layout": true,
    "math_strokes": "personal",
    "seed": 42
  }
}
```

`glyph_metrics=raw` 关闭新度量层；`normalized` 开启类别规范化；`contextual` 额外开启共享词尺度。
`semantic_layout=false` 回到历史规划器；`math_strokes=off` 保留解析器原来的字形/几何线条。
纸张吸附仍优先，报告中的 `layout_applied` 表明实际是否启用布局。
字体的原始轮廓不写回；新字体只在进程临时缓存中派生。CFF 暂不支持规范化，明确报错后可使用 raw。
以共同估计基线缩放整个字形；不会单独截掉、抬起或压短 `p/y/g/q` 的尾部。

不要把当前 `baseline` 称为 v0.3.0 原版。四模式共有的字号、单次填充和历史横线修正仍存在。
受控对照必须同时记录上述因素。默认个人库仍是 `personal@v1`；不设 `HWSOL_DATA_DIR` 的 Runtime
按现有用户设置解析，无设置时使用 bundled-default。任何候选均没有自动激活。

从固定归档 manifest 重放；不要追加参数后仍声称为同一份实验。新度量对照关闭旧 bandit 策略，避免
把旧图评分迁移成新图评分。旧策略的运行报告标为 `legacy-profile-only-unvalidated-migration`。
新的 bandit 题包校验完整 renderer fingerprint，旧题包继续受到源码变化保护。

## Studio

```bash
export HWSOL_DATA_DIR="$PWD/profiles/realism-studio"
hwsol-studio collection-plan --profile personal@v3 \
  --source examples/phase2/mixed/03.md --output out/new-collection
hwsol-studio train --profile personal --append
hwsol-studio evaluate --profile personal --version vN
```

采集计划按出现次数、数学用途和现有变体数量排序，并调用锁定 Handnote 的模板生成器。
`source_failures` 不为空时表示覆盖分析不完整。普通解题不调用此采集流程。
当前缺字表覆盖已分析历史作业，不能外推到任意 LaTeX。

候选保留原始字体和 178 个真实变体，增加度量资产及角色统计；版本对比单独报告字符覆盖是否改变。
`Trace.origin` 可以是 `recorded-pen`、`estimated-trajectory` 或 `scan-contour`。估计轨迹不算真实笔顺。
只含 traces 的 calibration JSON 可导入；真实数字轨迹只有显式 `--include-traces` 才进入可迁移资产。
`model-adapter` 资产角色可登记可选参数，但本轮没有创建空模型或宣称已完成适配。

## 证据与六项质量维度

最终交付清单、固定版本五题、三页度量对照、数学前后覆盖、公式反例、候选比较、自动测试和新包重放
由 [本轮验收记录](evidence/next-stage-2026-11-18/README.md) 索引。个人 PDF 与 Profile 留在本地 `out/`、`profiles/`，不纳入源码发行。

1. 内容正确性：保留已有数值复算和原始解答；结构和像素校验不等于完整推导证明或盲读。
2. 单词尺度协调性：报告字形类别上沿的变异系数、真实三组 PDF；该代理量不等于作者主观评价。
3. 数学符号覆盖与可读性：以实际可见 item 为分母，字符和结构分开列；回退和未采集项不计为个人笔迹。
4. 作者风格相似性：待新版独立盲评，没有沿用历史 12 组分数。
5. 整页布局自然度：已知位置反例和组内约束通过，不等于跨题泛化。曾用于横线校准的页面不再算 holdout。
6. 工程交付一致性：测试、固定源码、发行摘要、干净安装、五题逐页校验分别报告。

## 未完成事项

- 新的作者盲评和独立 OCR/人工转写，尤其 `1/l`、`2/z`、根号钩部和小号上下标。
- 真实双字母、词、行的连续书写样本；目前没有证据表明拼贴感已消除。
- 箭头、方括号、求和及撇号等独立补字；不同尺寸根号和分数线的真实结构采集。
- 更可靠的语义结构标注、明确栏边界和角色分布；对复杂 TeX 的构件识别继续增加反例。
- 成功运行真实 DiffBrush/DiffInk/InkSight 后，才进行生成质量、OCR CER/WER、峰值内存和吞吐比较。
- 真实用户的完整整页数据与新 holdout；跨系统原生验收；策略发布生命周期 A8 尚待单独完善。
