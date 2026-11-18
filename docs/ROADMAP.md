# 当前研发里程碑：v0.9 Personal Homework Corpus

本轮优先级：真实语料 > 准确转写/质量筛选 > 最小神经生成训练 > 生产集成 > RL/RSI。
[当前状态](CURRENT_STATE.md) · [数据与审核说明](PERSONAL_HOMEWORK_CORPUS.md)。

已完成：先前5处复核项修订；phy1第15/25页和AS第70页增量导入；Sol xhigh直接清洗准入69个
单符号；冻结数据94条；数字1的真实GPU训练和16张生成对照。训练链路通过，输出偏淡模糊，
生成质量未通过。用户已明确授权模型标签直接用于训练，不再要求主Agent逐条复读或人工确认。

1. 继续让Sol从原始真实页面保守清洗。多公式混合、遮挡、涂改、边界不完整、变量/运算符歧义
   默认弃用；仅在可可靠隔离完整干净内容时另提新框。保留拒绝原因和模型来源。
2. 按固定分组补覆盖：0至少缺11 Train/5 Validation；2缺19 Train/2 Validation；
   x缺9 Train/5 Holdout。20/5/5仅是研究启动下限，不应以凑数量代替真实质量。
3. 冻结checkpoint诊断已确认后验/先验不匹配；先在Train/Validation修生成目标与采样的一致性，
   再结合Handnote原始个人字形、度量与真实作业扩充笔宽/写法分布，比较参考条件生成。
   详细依据见[V09_GLYPH_DIAGNOSIS](V09_GLYPH_DIAGNOSIS.md)。
   已展示的Holdout不得用于针对性调参；正式下一轮质量验收需新增未查看的独立文档或会话。
4. 版本化数据、保留原始像素，明确区分人工标签、授权Sol标签与普通OCR候选。
   每次训练固定源码、依赖、原件和裁剪摘要、权重、优化器、种子及预处理。
5. 生成模型需通过独立字符身份、自然变化、记忆与作者风格检验后，才尝试少量新类别、词或短片段。
   未达标内容继续font-cpu；数学结构仍由Handnote/MathText控制。
6. 内容与风格评估稳定后再收集偏好，不提前推进RL/RSI自动发布，不预设LoRA或从零大模型训练。

暂停DiffBrush、DiffusionPen、DiffInk比较与训练，保留全部旧证据。当前完整个人模型、连续英文、
生产集成及后续SFT/RL均No-Go；默认Profile与正式latest不变。

---

以下为原v0.8及更早路线，保留历史，不代表当前执行计划。

# 当前研发里程碑：v0.8 Evidence-Based Go/No-Go

v0.8已完成固定120行英文数据、两个CPU Profile实测、DiffBrush官方/Adapter/PDF诊断、多种子与holdout、真实DiffusionPen英文CUDA推理、588图双识别器评测和72份内容先行盲评材料。[验收报告](V08_GO_NO_GO.md)。生产仍为v0.6 font-cpu。

下一阶段按以下顺序推进，尚未授权自动发布或模型训练：

1. 回收独立内容校核和作者盲评；校准识别器标点空格、长行压缩及IAM域偏差。保留原始指标，不能只用OCR自证内容正确。
2. 主候选DiffusionPen先解决词间x-height/基线、数字标点与长词；DiffBrush作为整行对照保留。研究Reforge的方法，但其固定快照没有明确LICENSE，不复制实现；程序化笔画必须单独标明来源。
3. 完成DiffusionPen完整开发集及独立holdout，内容可靠且作者风格认可后，才设计本人扫描词/行的监督微调小试。按采集页/会话隔离数据；不把五词风格条件当成已完成训练或充分训练集。
4. DiffInk等待可靠英文权重/词表与可用在线输入方案；保留全部中文CUDA证据，不把中文续写算英文成功。
5. 不扩展旧bandit，不进行未经论证的LoRA、D3PO/RL或从零预训练。数学结构和生产CPU后端持续保留。

明确未完成：人工风格分数、真实人工错误率、DiffusionPen全120/holdout、整页端到端神经生成、个人监督微调、商业发布许可确认。当前个人训练与生产准入均No-Go。

---

以下为原阶段路线图及历史目标，保留供追溯。

# 后续阶段实施规划

v0.3.0 已交付 Linux 稳定性验收和阶段 4 本地实验工具。下文保留各阶段原始目标。已验证范围见[当前实施状态](IMPLEMENTATION_STATUS.md)。
完整 v1 的 Definition of Done 尚未达到。
排序依据：先独立 CPU Runtime + 可导出的 Profile，再接 Agent 解题闭环，最后完善跨平台与高级模型。

## Phase 1 — Profile 与独立 CPU 字迹 Studio

**状态：1A/1B/1C 已实现并通过合成样本验收。** 操作见 [TRAINING](TRAINING.md)，
格式见 [PROFILE_FORMAT](PROFILE_FORMAT.md)。真实作者样本、自动风格评分、采样概率权重及 CFF 累积合并不在当前已验证范围；
相应限制见 [实现状态](IMPLEMENTATION_STATUS.md)。以下保留原验收要求作为设计依据。

### 1A：Profile 合约与只读默认资源

定义 `profile.json`：profile ID、不可变 version ID、parent、来源和许可证、资产 SHA-256、
覆盖率、样本索引、风格参数、后端构建版本。运行包只加载发布版本；Studio 写 samples、training-runs
和 candidate 目录。默认数据目录使用现有 `HWSOL_DATA_DIR` 解析逻辑，不向仓库写入个人数据。

先将获授权测试字体封装为明确标记的 `bundled-default` 临时资源，保留字体原名和 OFL；
实际作者笔迹发布前必须取得真实样本和分发授权。默认资源禁止原地修改，派生 Profile 单独保存。

实现 `profiles list`、`activate <profile@version>`、历史切换/回滚及导入导出。
激活是更新指针，版本不可覆盖；导出包包含 Runtime 所需字体、变体、风格、覆盖和许可证，
模型 checkpoint 不是普通推理必需品。打包路径必须防越界，校验资产摘要和版本完整性。

验收：CPU/无训练环境可用默认字体；在另一数据目录导入后可独立渲染；切换回旧版后输出可复现。

### 1B：原始素材归档与累计构建

实现 `profile template/import/train --append`，优先接 Handnote 的已验证模板、capture JSON、
扫描构建接口和 TTF/OTF 导入。先制定样本级 schema：字符/标签、原始文件摘要、采集轮次、
分割结果、真实轨迹是否存在、质量及用户权重。不得由照片虚构速度、压感或笔顺。

PDF/普通照片先归档，只有经过标注/分割的字形进入构建；未支持的导入形式不能标为已训练。
历史样本永久保留；默认历史 + 新样本构建新 candidate。重复字符保留真实多变体，低质量
样本隔离待评估，不覆盖历史优质字形。不要直接用上游 `--replace` 实现整个版本管理。

先补 Phase 0 暴露的西文字体合并测试：比较 cmap 覆盖、字形轮廓与度量、GSUB alternates，
不把可变化的内部字形名当作唯一身份。draw-your-font 仅作为可选外部 CLI，许可与真实照片评估
通过后再接入；目前只有单字形输出，不能替代多轮变体库。

### 1C：评估、发布与并发隔离

`profile evaluate/publish`：保存训练快照 → candidate → 英数与数学标准页 → 覆盖和构建检查 →
旧版比较报告 → 发布不可变版本 → 按配置激活。失败不触碰已发布版本。使用暂存目录、原子重命名
及必要的构建锁，Runtime 始终持有已解析的发布版本。将来模型权重与神经训练依赖放可选 Studio 环境；
预留 `TrainingBackend` 接口，不实现基础模型训练。

**Phase 1 门槛：Test D/E/I/J**，包括两轮真实格式样本、第一轮保留、v1/v2 关系、回滚、
导出复用、构建失败不破坏旧版、构建时持续渲染。合成样本覆盖自动化，真实作者样本另行人工验收。

## Phase 2 — Agent 解题与严格分页闭环

Linux 工程验收记录在 [实施状态](IMPLEMENTATION_STATUS.md) 和 [验证记录](VERIFICATION.md)。以下保留设计与验收范围。

### 2A：输入、解答和 Manifest

正式完善 SKILL.md 与安装说明。Agent 读取文本/图片/PDF、多份来源，按顶层大题建立编号清单，
小题留在所属大题，尊重范围选择。空输入或关键题干不清晰时不编造。默认英文简明推导。
Python 不内嵌特定云 API 或另造解题 Agent。

定义任务 Manifest 与每题 Markdown：原编号、来源页/区域及输入摘要、来源完整性、解答路径、
验证状态与证据、Profile ID/version/hash、背景、字号、随机种子、依赖版本、AI 生成记录。
验证至少区分 `not_checked / passed / failed / inconclusive`，符号计算或数值复算必须有真实记录。

### 2B：数学兼容、背景和布局检查

由 Phase 0 corpus 扩展明确 LaTeX 白名单；测试语法解析与最终字形，不以 `unknown_commands`
列表或退出码替代渲染正确性。记录每个缺字与实际回退；关键字符无可靠字形即失败。
`aligned` 优先由 Agent 保持等价地改写成多个公式块；兼容后端只能显式选择，不自动隐式切换。

背景按题覆盖 → 本次统一 → 用户默认 → A4 白纸。支持横线/方格/点阵与 PNG/JPG/PDF，
独立背景层，统一尺寸/缩放/内边距/墨色。PDF 操作采用 PyMuPDF 还是 pypdf/Pillow 组合，
必须在发布前明确许可方案及功能覆盖；当前 Phase 0 的 pypdf 只验证结构，不替代完整背景系统。

每题独立渲染到暂存区，检查页数、布局边界、内容完整性。超页按空白压缩 → 公式折行 →
Agent 去除重复文字（保留必要推导）→ 边距 → 合理字体/行距下限重试。设最大重试次数和明确失败码，
禁止静默裁切、漏题或拆题。预留确定性 SVG 图示接口，只在题目需要时使用。

### 2C：事务交付与验证

实现目标接口 `hwsol render --manifest ...`、`verify --job ...`，成功条件是
`selected_problem_count == PDF page_count == PNG count == source solution count` 且题序对应。
所有题成功后才发布 `homework_solutions.pdf`、`pages/NN.png`、`solutions/NN.md`、
`manifest.json`、`validation_report.json`。任一失败不得留下可被误判为完整作业的成功文件。

**Phase 2 门槛：Test A/B/C/F/G/H**；与 Phase 1 的 D/E/I/J 一起覆盖完整核心闭环。
必须用真正题目从 Agent 输入端演练一次，不能只把渲染 fixture 当作完整解题验收。

## Phase 3 — 稳定性、打包与跨平台

扩展矩阵、嵌套分式、多行公式、单位、统计符号和长解答回归库；验证内容边界、fallback
和源文件对应关系。引入真实变体的受控采样、固定随机种子、不同纸张尺寸；记录峰值内存和冷/热启动
时间，避免把单次 Phase 0 计时当性能基准。

建立 Linux、macOS（含 Apple Silicon）、Windows CI，使用仅 Runtime 环境和独立 Studio 环境，
验证安装、升级、回滚、只读默认资源、非 ASCII 路径、离线缓存及失败报告。Profile 解析和渲染并发
需要单独测试，不将 Matplotlib 全局状态默认视为线程安全。

完成运行依赖/二进制内嵌库/字体/数据集/权重的发布清单。MTH 内置轨迹许可未解决前不分发；
高级模型权重不进入主包。修复或明确阻止已知会损坏内容的组合。

**v1 发布门槛：A–J 全通过 + 三平台实测 + 完整 Skill 安装演练 + 许可证清单 + 作者或明确测试
默认 Profile + 正确交付目录 + 无训练组件的 CPU 闭环**。跨平台或真实数据验证未完成就不能标记 v1 完成。

## Phase 4 — 可选生成式增强

单独研究 DiffusionPen、DiffBrush、One-DM 等 Adapter：代码/权重/训练集许可、CPU/Apple Silicon/CUDA、
真实样本风格相似度、误写/漏字率、资源需求，以及与确定性数学排版的组合。
先建立固定测试集和可量化准入阈值，再决定是否进入生产；默认禁用，不阻塞 v1，不让生成图像重画整张公式。

## 下一次实施切片

在原生 macOS 和 Windows 上记录已配置 CI 的首次运行，复核平台 wheel 内的第三方许可。接入有授权的真实作者样本，使用固定语料执行实际神经采样并扩大留出集；记录显存、误写、漏字与盲评风格分。
