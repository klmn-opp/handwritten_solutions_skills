# 实施状态

更新于 2026-11-18。Phase 0 和 Phase 1 的基座已复验，Phase 2 的本地作业流程已在 Linux x86_64 跑通。当前包版本为 0.2.0。

## Phase 2 作业交付

- Agent 读取输入、核对题号和子题、完成推导与复核。Runtime 归档来源、记录 AI 作者和验证证据。扫描 PDF 可用 `read-page` 就地生成页图供 Agent 阅读。
- `render --manifest` 使用已发布 Profile。每题独立生成一页，检查绘制前的向量墨迹范围和纸上可见性；最后统一合并，从最终 PDF 栅格化逐页 PNG。
- 支持白纸、横线、方格、点阵、PNG/JPG 和指定 PDF 页。自定义背景作为独立底图，PDF 底图栅格化。超页和越界有有限重试，失败时保留题号和报告。
- `verify --job` 检查原件及产物摘要、题目顺序、页内身份、解答关联、尺寸和数值证据。已交付的 Manifest 可以在具有同一 Profile 的机器重新渲染。
- 已用公开的 Project Euler 第 1 题从题干完成求解、独立枚举、生成和验收。五题工程样例的 PDF 恰好五页，保留原编号和子题。[作业协议](../references/jobs.md) · [测试记录](evidence/phase2/tests.xml)

## 已实现并验证

| 能力 | 证据 |
|---|---|
| 上游真实仓库、提交、CLI/API、依赖与关键许可核对 | [Phase 0 报告](PHASE0.md)、[许可证清单](LICENSES.md) |
| Python src 工程、固定依赖、Pydantic 统一配置、Typer CLI、Adapter 协议 | `pyproject.toml`、`uv.lock`、`src/hwsol/` |
| `hwsol doctor` | CPU 环境及字体检查，不下载字体/模型 |
| `hwsol render-probe` 主后端 Handnote | Markdown → 向量 PDF + PNG + 源文件 + 结构验证报告 |
| 指定字体、普通数学、矩阵、多页、A4/300 DPI | [14 个实验结果](evidence/phase0/probe-summary.json)，其中预期拒绝也有记录 |
| 独立 MTH 实验 Adapter | 显式 STIX 回退、关闭未授权轨迹资产、隔离 Pillow 版本 |
| 本地隐私与失败处理 | 无云 API；空输入/缺字/公式降级拒绝；非空输出目录不覆盖 |
| 种子及字体/输入/产物摘要、环境来源记录 | `validation_report.json`，可重复 PNG/向量 PDF |
| 只读默认 Profile | bundled-default@v1：Patrick Hand / OFL-1.1，515 码点；不是作者真实笔迹 |
| Profile 合约、版本、切换/回滚/比较 | schema_version 1；资产摘要、候选与发布目录、原子激活指针；[格式](PROFILE_FORMAT.md) |
| 独立 CPU Studio | hwsol-studio 或 hwsol profile；模板 PDF/JSON、轨迹、扫描和字体导入；Runtime 不导入 Studio |
| 多轮累积与真实变体 | 全历史 accepted 样本重建；保留原件；低质量隔离；两轮及九轮变体测试 |
| 评估与发布 | 英文/数学测试页、覆盖和空字形检查、比较报告；完整评估后发布，默认不激活 |
| 读写隔离 | 写入锁、暂存目录、原子重命名；构建时旧版本仍可完成渲染；失败不改已发布版本 |
| Profile 迁移 | ZIP 仅含 Runtime 资产、字体、许可证和来源；新数据目录无历史样本也可渲染 |
| Skill 原型入口与边界说明 | `SKILL.md`、`agents/openai.yaml`；格式校验通过 |

## 验证结果

- 项目自动化：**49 passed**，含前两阶段 29 项和本阶段 20 项，测试执行时包含两个隔离兼容环境用例。
- Phase 1 D/E/I/J 工程门槛：切换改变像素；v1/v2 历史/轮廓保留；禁用 Studio/神经模块仍可渲染；构建时旧版持续渲染。
- 21 条实际 CLI 演练通过；回滚/导出导入后的 PDF 与 PNG 分别逐字节一致；[摘要](evidence/phase1/demo-summary.json)。
- 轨迹、扫描模板和 CFF OTF 测试采用明确标记的合成数据；不等同于真实作者笔迹质量验收。
- Handnote 上游针对性验证：**69 passed**，涵盖 render/vector/capture/symbols/sets/replace，使用其默认 Zhi Mang Xing。
- 使用 Patrick Hand 替代上游默认字体的首轮：40 passed / 3 failed，保留失败证据并分析，未伪装全通过。
- draw-your-font 上游合成图片端到端测试通过；模板生成通过。真实作者样本质量尚未验证。
- Ruff、Python 编译、Skill 格式、sdist/wheel 构建及仅 Runtime 的 wheel 离线安装/渲染检查见 [验证记录](VERIFICATION.md)。
- 实际 PDF/PNG 样例位于 `out/phase0-final/` 和 `out/phase1-demo/`；后者含两轮版本、回滚、迁移及标准评估预览。

早期 `render-probe` 的 `status=passed` 只代表当前结构和诊断检查通过；**不代表题目解答正确、无所有裁切风险或严格一题一页**。
数学测试 corpus 已人工看图，不等于完整 LaTeX 兼容认证。

## 未实现

| 阶段 | 尚未交付的功能 |
|---|---|
| Profile 后续增强 | 自动风格/缺笔画识别、按权重的变体采样、CFF 累积合并、任意 PDF/照片自动标注；真实作者样本与字体分发授权 |
| 作业流程的验证边界 | Agent 负责识别原稿与判断推导。Runtime 自动验证给出的精确数值等式和产物结构。复杂图示、一般符号证明、任意 LaTeX、真实扫描作业识别准确率及新 Codex 会话发现流程仍待进一步验证。 |
| Phase 3 | 跨平台 A–J 回归、真实样本质量验证、完整图示支持、性能/内存基准、macOS/Windows 验证、发布级许可证与安全审计 |
| Phase 4 | 神经微调与 DiffusionPen/DiffBrush/One-DM 等生成式 Adapter |

已存在 `doctor`、`profiles list/activate/rollback/compare/export/import`、
`profile create/template/import/samples/train/evaluate/publish` 和 `hwsol-studio`。
`read-page`、`render --manifest`、`verify --job` 用于正式作业。已有 Markdown 也可使用早期 `render-probe` 接口。

当前权重只影响字形首选优先级，运行时均匀采样真实变体；普通图片/PDF 只归档；CFF OTF 仅独立使用。
CPU Studio 复用主包已有依赖，普通渲染不需要 Studio 导入、训练样本、GPU 或模型；尚未拆成独立发行包。
详见 [训练指南](TRAINING.md)。

## 下一步

下一阶段按 [ROADMAP](ROADMAP.md) 扩展复杂公式/排版回归、真实扫描输入与 macOS/Windows 验收。
