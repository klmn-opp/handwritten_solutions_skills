# Session O：个人数学原迹接入与整页验收

2026-11-20。新上传的 `source_pdf/session-O_full_source.pdf` 已完成 19 页、123 行的
实际书写审核。本轮批准 **33 枚原始字形、13 种符号**，完成原 20 页修订和另外
4 道完整作业。原 20 页的 30 处数学字体回退降至 0，20 个偏重 π 换为新采集的真实 π。
24 页共 2,932 个可见数学项目、124 次新原迹调用，未记录字体回退。

主 Agent 与独立 Agent 对来源、局部及整页的工程审核通过；**作者风格验收待定，
生产发布 No-Go**。这是原迹与 Resolver 改进，没有神经网络训练或权重更新。
零回退只适用于这些受测页面，不表示任意公式已经覆盖。

## 可检查交付

| 内容 | 入口 |
| --- | --- |
| 新候选 B，完整 24 页 | [PDF](../workspace/runs/20261120T141332892747Z-session-o-reviewed-24-page-delivery-e4bc22e8/rendered/homework_solutions.pdf) |
| 相同题目与布局的 A 基线，24 页 | [PDF](../workspace/runs/20261120T141332892747Z-session-o-reviewed-24-page-delivery-e4bc22e8/comparisons/baseline-24-pages.pdf) |
| 逐页 300 DPI PNG | [ZIP](../workspace/runs/20261120T141332892747Z-session-o-reviewed-24-page-delivery-e4bc22e8/rendered/pages-300dpi.zip) |
| 原 20 页的 50 处替换 | [原生局部 A/B](../workspace/runs/20261120T140811736339Z-session-o-reviewed-math-complete-20-de331ba4/comparisons/all-targets/formula-regions.pdf) |
| 原 20 页的全部 173 个公式区域 | [A/B PDF](../workspace/runs/20261120T140811736339Z-session-o-reviewed-math-complete-20-de331ba4/comparisons/all-formulas/formula-regions.pdf) |
| 原回退位置与修订后标注 | [A 回退区域](../workspace/runs/20261120T140811736339Z-session-o-reviewed-math-complete-20-de331ba4/comparisons/fallback-overlay-A/math-fallback-overlay.pdf)、[B 来源标注](../workspace/runs/20261120T140811736339Z-session-o-reviewed-math-complete-20-de331ba4/comparisons/fallback-overlay-B/math-fallback-overlay.pdf) |
| 新增四题公式与整页 A/B | [公式 PDF](../workspace/runs/20261120T140350095212Z-session-o-fresh-four-homework-pages-ad9aab55/comparisons/formulas/formula-regions.pdf) |
| 逐项来源、旧字形与新变换 | [Glyph Provenance JSON](../workspace/runs/20261120T141332892747Z-session-o-reviewed-24-page-delivery-e4bc22e8/reports/glyph-provenance.json) |
| 来源与完整采集审核 | [语料专项报告](SESSION_O_CORPUS_IMPORT.md) |
| 汇总与逐页装配核验 | [summary](../workspace/runs/20261120T141332892747Z-session-o-reviewed-24-page-delivery-e4bc22e8/reports/summary.json)、[checks](../workspace/runs/20261120T141332892747Z-session-o-reviewed-24-page-delivery-e4bc22e8/reports/package-checks.json) |

前 20 页使用原第三轮 `4af6334e` 同题基线，保留其真实 f0/f1、英文、根式与横笔。
后四页为势函数与振动、Hermitian 内积及矩阵、连乘与指数分布似然、环流与磁通。
其 A/B 均由相同已校核答案、CPU 英文、相同页面与背景直接生成，包含上下分数、
高角括号、矩阵、根式、希腊字母及上下标。这是补充工程新题，不是封存独立 Holdout。
24 页英文均为当前 CPU 路径，不能宣称新神经英文泛化。

## 缺口、来源与实际替换

原迹文件 SHA256：`23a84ff4cf5dc4882b5fb76d5eb4cc3a32c59e9799ea7a351f545598401f4d43`。
最终 bank：`20261120T135929419362Z-session-o-reviewed-glyph-bank-b803206a`，
`solutions/glyphs/sources.json` SHA256：
`2ae93d8c9f2261abfae4811a82e87f275232e3231bc7c2fc6cf53c7fc9372f18`。

| 符号 | O-FULL 原页/行 | 可靠原迹数 | 原 20 页调用 | 新四页调用 | 原问题 |
| --- | --- | ---: | ---: | ---: | --- |
| π | P01 R04 | 3 | 20 | 14 | 旧真实 π 自身偏重；另一历史轻笔与 2 连笔，不能强拆 |
| κ | P01 R03 | 3 | 6 | 10 | 原基线缺少已准入的孤立个人源，仍走字体回退 |
| Φ | P02 R06 | 3 | 5 | 5 | 缺少准入源；新源直接按全高归一又会使圆腹过小 |
| ξ | P01 R04 | 3 | 4 | 10 | 原基线缺少已准入源 |
| η | P01 R02 | 3 | 3 | 8 | 同上 |
| ∏ | P11 R02 | 3 | 2 | 3 | 同上 |
| ≈ | P07 R06 | 3 | 1 | 2 | 同上 |
| ζ | P01 R02 | 3 | 1 | 1 | 同上 |
| ψ | P01 R06 | 3 | 1 | 5 | 同上 |
| ⟨ / ⟩ | P13 R01 | 各 1 | 各 3 | 各 6 | 缺少准入源；长槽位使用明确记录的有限拉伸 |
| ∮ | P11 R05 | 1 | 1 | 3 | 缺少准入源；居中放置的新原迹曾接触邻近 E |
| * | P07 R02 | 3 | 0 | 1 | 新四题中的真实五射星号；区分相邻四射乘号 |

除 π 外，原 20 页这 30 个替换位置原本均为明确的字体回退。新材料解决的是
可靠来源缺口，随后修复 Φ/∮ 的调用几何；不把未准入候选称作已有可用字体。
选择顺序依据原页面真实使用频率、π 的明确作者反馈和符号误认风险。

每枚批准源保留原 PDF、原页、上下文、300 DPI RGB 裁剪、原始/裁剪 SHA、
精确矩形及双向坐标变换、实际标签和审核依据。同一会话多枚变体仍属于一个来源组。
轮廓阈值为 L160，灰色印刷辅助线保留在原色图中；没有擦笔、腐蚀、膨胀、补画或
按标准答案改标签。原轮廓经记录的仿射变换直接放入 MathText 结构。

## 具体机制与验收依据

**π：更换真实来源。** 旧历史 π 的墨迹本身偏厚，不能靠侵蚀来制造新手写。
新 O-FULL 的三个独立 π 均保留完整顶横与两腿。种子固定 1909，由题目、LaTeX、
项目位置和字符的哈希决定真实变体；原 20 页三枚分别调用 4/10/6 次，没有挑种子。

**Φ：用圆腹定义字身高度。** 三枚 Φ 的原始全高约为圆腹的 1.82 倍，圆腹与同排 Ω
高度接近。把长竖在内的整个 Φ 缩到普通大写字高，会使圆腹变小、整字显淡。
现在将圆腹高度与大写槽位对齐，整枚原迹统一等比缩放，长竖自然伸出；不加墨。
允许区域由源测量决定为原槽顶扩 0.4 高、底扩 0.6 高，且必须通过实际墨迹邻接检查。
初版对称 ±0.5 高曾拒绝三个 V3：真实下伸约 0.508 高。改为符合原迹的非对称区域
后，三个既定候选的形状和位置不变，五处 Φ 均通过。第 13 页 dΦ_B 与分数线仍有
约 2.478 px 间隙；没有靠裁切长竖或改笔画解决。

**∮：保留比例，改变对齐依据。** 首轮原迹全高与原积分槽一致，但水平居中后碰到 E。
固定按原槽左边对齐，使整枚原迹左移约 3.280 px，不改变尺度、y 位置或源笔迹。
与 E 的间隙为约 3.038 px，与箭头为约 5.158 px。变换从原轮廓一次计算，避免
对已放置坐标再次平移造成浮点不一致。完整圈与积分上下端均保留。

普通符号等比缩放和平移；角括号可有限拉伸并标记来源类型。失败、未支持或无可靠源
的替换保留原内容和明确拒绝原因，绝不静默删符号。此阶段只用可选择的实验 Resolver，
未改正式 MathText 解析规则、页面布局或默认 Profile。

## 数学、边界、来源与英文检查

原第三轮基线先精确重放：20 页 PDF/PNG、合并 PDF 与 10,774 条来源记录一致。
最终 20 页 50 次替换全数通过；2,265 数学项目的身份和顺序保持，173 组 parser trace
保持；8,509 条英文记录及落位轮廓完全一致，所有非目标来源/几何不变，目标区域外
变化像素为零。旧 f 的真实变体、拒绝项与所有 ff 组合都没有重新采样。
新轮廓逐点等于原始裁剪导出的轮廓及记录变换；没有新接触、页面裁切或新增小于
0.60 px 的拥挤。已有非目标问题不会被此增量检查冒充修复。

新四题的明确解析结果另用 SymPy 核对，包括势与场、正交归一、行列式、连乘望远镜
消去、似然曲率、环流与磁通变化；这不是自动证明全部 LaTeX 语义等价。原题答案
沿用已校核基线，渲染通过也不被用作答案正确性的证明。

主 Agent 和独立 Agent 均查看原迹、原生局部及最终整页。独立最终证据：
`workspace/analysis/20261120T140416012048Z-session-o-final-pages-independent-a45c1888/FINAL_REVIEW.json`，
SHA256 `02c8ad2686540d0d7cecdf76614e129ff968d4d712664d41b4fecaee0a9786e3`。
主审：`workspace/analysis/session-o-root-priority/FINAL_ROOT_REVIEW.json`。
父运行中生成时的 review-pending 字段保留原状；上述后续实际审核和 24 页包记录审核完成，
没有倒改已封存父运行。

24 页来源按实际记录保留细分，避免将结构绘制冒充直接原迹：

| 来源类型 | 可见项目数 |
| --- | ---: |
| personal-outline | 2300 |
| personal-session-outline | 251 |
| personal-scan-outline | 71 |
| personal-outline-stretched | 84 |
| personal-stretched-outline | 18 |
| personal-outline-geometric-calibration | 14 |
| personal-scan-structural-rule | 181 |
| personal-component-composition | 6 |
| procedural-geometry | 7 |
| inherited-font / external-font | 0 / 0 |

因此仍有 7 条旧短根号横笔为程序几何，181 条个人风格结构横笔也不是孤立字形复制。
每项有原语义、位置、来源和必要的旧记录；完整清单见交付中的 provenance JSON。

## 失败与尚未解决的边界

- 首轮 20 页 `adf8f1c8` 仅 49/50 替换，∮ 留回退；Φ 圆腹偏小，保留为失败证据。
- ∮ 首个证明 `a27cad9f` 因重复变换的浮点末位差未通过严格源重建；改为一次从源计算，
  `e58e4155` 通过，没有放宽轮廓等值检查。
- Φ 对称区域探针 `62afb332` 2/5、新四页 `99e59a44` 留 3 回退；源测量后的
  `a765cef0` 5/5。已换成新材料不等于初次放置就合格。
- 新四题初次 `5ce9e516` 明确拒绝不支持的 `\ell`；预检也发现 `\widehat` 不支持。
  自行撰写的新答案明确声明 Latin l 为 log-likelihood，并使用单字母 `\hat`。
  没有改原始作者标签、伪称通用结构支持或调整原 Holdout 的 7/8 结果。
- 20 页封装 `8a64ba48` 图像正确但嵌套 `run.json` 被产物清单排除，封包核验失败；
  出错处理又追加日志，旧 receipt 不再有效，不能称其为封存通过。改用
  `source-run-receipt.json` 后创建新 run `de331ba4`；91 个图像/PDF 与前次字节相同，
  证明见 `20261120T141043341166Z-session-o-packaging-replay-verification-e1c2d8d6`。
- 新星号在 U 的上标尺寸下仍细小，五射原迹完整但不是每一射都显著可辨；保留此视觉限制。
- 14 行存在具体语义角色不确定性，333 个列 region 仍未标注；不能将其批量准入。
  例如原写 `π π Π` 不能自动标为 π/ϖ/Π，普通 N/Z/Q/R/C 不能冒充双线体，
  第三遍矩阵实际为 `[a b; b d]`，不能改成提示的 c。详见语料报告。
- 旧 30 处小 f、7 条程序短横笔及已有通用解析限制保持；本轮未把这些算作已修复。

目前不需要再抄整套数学符号。已收到完整 O-FULL，继续扩展覆盖应先审核现有原迹；
特定变体的标签确认或不同字体角色确有必要时，再按原 Session 形式给出明确任务。
当前需要作者的是新 π/Φ/∮ 与整页风格判断，不要求作者替 Agent 做来源或边界检查。

## 版本、命令与复现

源码分支 `feature/session-o-math-completion`；实现 commit **`ae9c9db`**，父 commit
`9881dc744dc609643aca6923c0a7bfc022f049b4`。10 个 `research/session_o_*.py`
实现导入、准入、源测量、Resolver、基线重放、四题和整页封包。
运行发生在提交前，通用 build 的 tracked-only dirty 字段没有包含新文件；
`reports/source-identity.json` 另保存实际未跟踪状态和完整源码 SHA 快照。已逐项确认
10 个最终源码文件与封包快照一致，不能仅用父 commit 声称源码完整。

环境：Linux x86_64、Python 3.12.16、Intel Core i9-14900HX，全部本轮任务使用 CPU。
`.venv` 无 Torch；`uv.lock` 未变。seed 1909，实际页面/阈值/来源与命令由各 run 保存。
没有 checkpoint 或优化器，因为没有神经训练。关键实际命令如下，复跑会创建新 run：

```bash
.venv/bin/python -m research.session_o_baseline
.venv/bin/python -m research.session_o_sources extract \
  --source-run workspace/runs/20261120T133948143789Z-session-o-full-source-rgb-import-7e119738 \
  --selection workspace/analysis/session-o-root-priority/selection-with-star.json
.venv/bin/python -m research.session_o_sources admit \
  --parent workspace/runs/20261120T135639190512Z-session-o-priority-original-crops-1153dd89 \
  --approval workspace/analysis/session-o-root-priority/source-approval-33-body.json
.venv/bin/python -m research.session_o_delivery \
  --bank workspace/runs/20261120T135929419362Z-session-o-reviewed-glyph-bank-b803206a/solutions/glyphs \
  --source-run workspace/runs/20261120T133948143789Z-session-o-full-source-rgb-import-7e119738
.venv/bin/python -m research.session_o_fresh_pages \
  --bank workspace/runs/20261120T135929419362Z-session-o-reviewed-glyph-bank-b803206a/solutions/glyphs
.venv/bin/python -m research.session_o_package \
  --approval workspace/analysis/session-o-root-priority/package-approval.json
.venv/bin/python -m research.project_state verify
.venv/bin/ruff check research/session_o_*.py
.venv/bin/ruff format --check research/session_o_*.py
.venv/bin/python -m compileall -q research/session_o_*.py
```

导入命令与坐标检查见语料报告。上述静态检查、运行完整性及保护检查已执行通过；
质量结论依据真实原迹、内容、整页视觉与增量不变性，不能用测试数量代替风格验收。

最终 24 页运行 `20261120T141332892747Z-session-o-reviewed-24-page-delivery-e4bc22e8`，
`run.json` SHA256 `e0c65bd37fdc3c5af478a29364a7a669b2f9de97314909dea83d82a9a45c6075`；
PDF SHA256 `4405763fe1e152371a891e8ef74baa2ed8e424393d7d2d51d5e6ee19d55415b2`。
24 张合并 PDF 栅格逐像素等于审核父页；状态为 held，不是发布状态。
私人 registry 增量位于 `workspace/registry/increments/<该run_id>.json`。
候选 Profile 清单位于包内 `solutions/candidate-profile/profile.json`，引用原 baseline
与回滚指针，未激活或覆盖原 Profile。

## Go / No-Go 与下一入口

**Go：** 本轮原迹导入、33 枚明确源准入、受测页的数学替换、内容与边界检查、
同条件整页对照及作者可检查交付。π 原迹与 Φ/∮ 放置问题已形成可复核修订。

**待作者：** 本人对 π 粗细、Φ 圆腹与长竖、∮ 及整体协调性的风格评价。
Agent 的工程结论不代替该判断。此前“B 更好”不视为对这份新候选的批准。

**No-Go：** 生产默认切换或 v1.0 发布；还缺本人验收、正式个人数学路径整合与
macOS 等既定 M2 条件。M4 上下文神经假设的独立 No-Go 不因新符号采集而解除，
M5/RL 也未解锁。本次不重启无关神经训练或旧实验。

正式 main、v0.6.0、latest、默认 personal@v1、旧 Profile/权重/标签/扫描及封存 C
均保持。源 PDF、latest、uv.lock、总计划和已冻结注册资产经过 SHA 复核；没有外传。
本轮数学完成增量已交付，长期 M0–M8 没有整体完成。
