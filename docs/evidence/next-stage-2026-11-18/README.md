# 2026-11-18 下一阶段验收证据

实现基线 `67c2b88`，CPU 实现 `6d04b70`，数学覆盖分母修正 `b34c0b5`。
本地交付根目录 `out/next-stage-2026-11-18/`；中途产物未覆盖，最终对照在 `delivery/`。
新版作者自然度和风格验收尚未完成。

## 可直接查看的本地产物

| 产物 | 路径 |
|---|---|
| 固定审计源码的五题基线 | [PDF](../../../out/next-stage-2026-11-18/frozen-baseline/homework_solutions.pdf) |
| raw / normalized / contextual，依次各一页 | [三页对照 PDF](../../../out/next-stage-2026-11-18/delivery/glyph-normalization-comparison.pdf) |
| 最终候选五题 | [personal@v5 PDF](../../../out/next-stage-2026-11-18/delivery/five-candidate/homework_solutions.pdf) |
| 数学原始结构 / 个人替换 | [之前](../../../out/next-stage-2026-11-18/delivery/math-off/homework_solutions.pdf) / [之后](../../../out/next-stage-2026-11-18/delivery/math-personal/homework_solutions.pdf) |
| 位置反例 | [旧规划器](../../../out/next-stage-2026-11-18/delivery/formula-legacy/homework_solutions.pdf) / [语义锚点](../../../out/next-stage-2026-11-18/delivery/formula-fixed/homework_solutions.pdf) |
| 缺字采集计划及模板 | [计划](../../../out/next-stage-2026-11-18/delivery/collection-queue/collection-plan.json) / [Handnote PDF](../../../out/next-stage-2026-11-18/delivery/collection-queue/symbols-01.pdf) |
| 干净 wheel 的五题重放 | [PDF](../../../out/next-stage-2026-11-18/wheel-replay/homework_solutions.pdf)，实测见 wheel-replay.json |

个人 PDF、扫描、字体、候选以及模型缓存均保留在本地忽略目录，不随源码包分发；上面的个人产物链接只在本工作区可用。
所有新增 PDF 遵守一题一页；三个样式比较页是三个独立的单页作业的合并展示。

## 实测结论

- [CPU 对照数据](cpu-comparisons.json)：11 个新作业、27 页，逐页 PDF/PNG 核验通过。固定审计五题另逐字节重放旧交付成功。
- [测试 Python 3.12](tests-final-py312.log) / [Python 3.13](tests-final-py313.log)：各 110 passed，包含两项隔离 MTH 测试，无跳过。JUnit 见同名 XML。3.13 运行当前源码，发行 wheel 另在干净 3.12 环境验证。
- [Ruff](ruff-check.log) / [格式检查](ruff-format.log)：通过。未配置独立静态类型检查或测试覆盖率工具；未做联网漏洞库扫描。
- [公式反例回归](formula-regression.json)：LaTeX 与公式内部几何摘要一致；四条公式左端 x 从 546/148/303/725 px 变为 159/154/176/148 px。整组等号约束可核对。此页已用于既往校准，不再称为独立 holdout。
- 规范化字体小写主体上沿 CV 2.14% → 0.57%，数字 2.45% → 0.42%。这是字体几何代理量；没有直接证明词内自然度提高。
- 数学示例按全部可见数学 item（含矩阵单字符及结构笔画）计算，个人来源 30/45 → 37/45，即 66.7% → 82.2%。矩阵方括号仍为 6 条程序几何；向量箭头和不能安全拉伸的根号仍为外部字形。
- 历史作业按内容去重后分析 6 份，均成功；当前实际缺少个人求和符号和撇号。补字进入独立 Studio 队列，不阻断现有日常作业。
- [候选与保护记录](preservation-and-candidate.json)：旧有 1,292 个文件未修改。新增 `personal@v4/v5` 候选；最终对照使用 v5，均未发布/激活，默认仍为 v1。
- v5 仍为 75 个采集码点、178 个轮廓，没有新增本人采集。v4 与 v5 的重建轮廓一致，但与 v3 不逐字节相同，最大墨迹边界及 advance 差为 0.093 em；没有隐瞒或把该重建差异算成风格提升。历史 v3 原字体保持不变。

## 六项质量分开报告

| 维度 | 当前证据与限制 |
|---|---|
| 内容正确性 | 原解答/数值式校验保留，公式几何和逐页像素通过；完整推导、OCR/人工盲读未独立验收 |
| 单词尺度协调性 | 类别度量与三组实际 PDF；英文仍逐字符，连笔与自然度需作者评价 |
| 数学覆盖及可读性 | 逐 item 来源、明确回退和几何类别；不能仅据覆盖率认定易读 |
| 作者风格相似性 | 新版未独立评分，旧 12 组分数未迁移 |
| 全页布局自然度 | 已知错误修复、单页约束保持；无跨题自然度泛化证据 |
| 工程交付一致性 | 双环境测试、来源快照、历史保护、发行对应及干净重放各自留证 |

生成式路线独立见 [实验报告](../../CONTINUOUS_WRITING_EXPERIMENT.md)，不使用 mock 质量结果。
