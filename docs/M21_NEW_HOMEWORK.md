# M2.1 — 新作业 PDF Intake 与完整出稿验收

本轮实际完成四道新题的“题目 PDF → Agent 阅读 → 解题 → 独立核验 → 普通 CLI 出稿”。原题由 Agent 新编并先冻结，没有使用旧六页答案重放；这属于**受控合成原题的端到端工程验收**，不是外部真实未见作业、封存 Holdout 或扫描 OCR 能力证明。

## 题目与解答核验

四页原题先保存为 `original-questions.pdf`，SHA256 为 `d6e41b4e17b69e784e2473cb18bf5ab342e275d7ce75fe95db64622c78eb2603`。原题 run 的 `logs/source.py` 只有出题和阅读逻辑，没有答案代码。随后提取 PDF 文字，Agent 实际阅读提取结果及四页原题图像，再编写完整答案。根 Agent 独立解析推导，另以 SymPy 核验 16 个数学恒等式；符号核验和页面视觉核验分开记录。

|课程与新题|主要内容与独立结论|
|---|---|
|线性代数|三阶对称三对角矩阵，det=4，逆矩阵、Ax=b、全部特征值与规范正交特征向量、QᵀAQ。解为 (1,−1,1)；特征值 2、2±√2。|
|微积分|两个定积分，换元、长分式、分部积分、原函数求导与正性核验。结果 log 2−5/8 和 2 log 2−1。所有数学分数明确使用上下格式。|
|统计推断|已知总体标准差的双侧正态检验、p 值、95%区间与线性变换。z=2，p≈0.04550，区间 [100.08,107.92]；变换后均值估计203、标准误4。|
|电磁学|同轴电容器的场、电势、单位长度电容与两种能量推导。保留均匀介质、无限长近似和相反线电荷假设；使用高斯定律及场能密度独立验证。|

每道题完整保留在一页，没有为排版删除计算步骤，也没有预生成特定词图。CPU Profile 与本地安装 CLI 负责出稿，解题仍由宿主 Agent 执行；不能将这次记录描述为 Renderer 内置了解题器。

## 保留的失败及修复边界

第一次安装包全 Job 被不支持的 `\det` 明确拒绝。Agent 创建新的答案版本，将运算符标签改为 `\mathrm{det}` / `\mathrm{…}`，单个 μ 的 `\widehat` 改为 `\hat`，并显式将数学比值写成堆叠分数。完整差异及前后 SHA 留存，原题 PDF 不变；这是 Agent 编写自身答案时的等价修订，不是 Renderer 获得了新宏支持，也不是第一遍成功。

修订后的第二次全 Job 暴露了真实的几何问题：第四页标题 `Coaxial electrostatic field` 的 l/d 新增接触。未改标题、随机种子或数学输入规避问题。修复针对个人字形候选调整逐项检查实际轮廓接触：只拒绝导致新接触的 d 变体调整，保留原本可用的真实字形。原失败事务及日志完整保存。相关 Runtime 冻结 commit：`4a0e52a`。

修复后 editable 全 Job 通过，四道题各出一页；前三页 PNG 与之前实际安装包单题结果 SHA 完全一致。第四页已独立以原生 PNG 阅读，电场方向、电势积分界限、两种能量结果、标题和边界未发现错漏。此 editable run 为 dirty tree，**不能充当干净安装验收**。

## 数学来源与质量边界

受测四页共有 827 个数学可见项目：

|来源|数量|
|---|---:|
|个人字体真实轮廓|666|
|已审核 Session 轮廓|51|
|个人扫描原迹结构横笔|47|
|拉伸个人轮廓|33|
|历史扫描个人轮廓|24|
|明确标注的程序结构|6|
|外部字体回退|0|

六个程序结构分别是矩阵题的三个右方括号、统计题的三个根号横线，不冒充真实原迹；除这六项外，来源记录均标记为已采集的个人素材。每页均保留 Glyph Provenance、公式组、矩阵组及原始来源；四份导出 Markdown 与冻结输入字节相同，语义身份、解析轨迹、原始字形来源均保持。零回退仅适用于这四页，不能推导任意数学表达式均覆盖。

Renderer 的自动算术栏仍是 `not_checked`；本次数学正确性依据是原题阅读、独立推导与保存的符号恒等式，不用渲染通过替代解答正确。Agent 页面检查不代表作者本人风格接受。

## 可复现证据链

所有下列目录均位于 `workspace/runs/`；每个 run 保留源码快照、输入、参数、日志或输出，并通过其对应 manifest 验证。

|阶段|Run ID|
|---|---|
|只含原题的冻结 PDF|`20261120T154540021418Z-m21-original-question-pdf-6e403c01`|
|PDF 文字读取|`20261120T154548170269Z-m21-question-reading-2c04a6a5`|
|解答、图像阅读回执、独立校核|`20261120T154726089949Z-m21-solved-intake-7be046ad`|
|第一遍不支持宏失败|`20261120T154741128033Z-m21-installed-e2e-31616c37`|
|明确的等价答案源修订|`20261120T154856740364Z-m21-explicit-source-recovery-ab6496e5`|
|第二遍真实 l/d 接触失败|`20261120T154901487945Z-m21-installed-e2e-703389ec`|
|矩阵题安装包成功|`20261120T154958794787Z-m21-installed-e2e-6a844528`|
|积分题安装包成功|`20261120T155004257299Z-m21-installed-e2e-c0f2d07a`|
|统计题安装包成功|`20261120T155004294399Z-m21-installed-e2e-e1da2014`|
|修复后 editable 全四页|`20261120T155354507677Z-m21-installed-e2e-07f2da79`|
|独立来源、数学与视觉链审计|`20261120T155625916402Z-m21-controlled-intake-review-b43a6032`|
|冻结 wheel 双 Python 全四页复验|`20261120T155718248776Z-m21-clean-installed-new-job-43bb9000`|

唯一冻结解答入口为源修订 run 的 `solutions/job.json`。实际命令与 stdout/stderr 位于渲染 run 的 `logs/commands.json`。原题、阅读、求解和修订的源码为 `research/m21_e2e.py`，可显式使用 `--cli` 指向隔离安装的运行时；推理进程在 `/tmp` 运行，未调用 research 模块。

## 冻结安装包复验

Linux Python **3.12.16 与 3.13.18** 均在独立空环境中安装同一个冻结 wheel，完整四题通过。源码 commit 为 `4a0e52ac1f8c643b3a3714c0867cbea562fbee6b`；wheel SHA256 为 `df96bf879070431c85c8e8b6541235afc14162098b5375ab98c97a1db704accb`。实际安装使用本地锁定依赖闭包与精确 wheel，执行目录不在仓库中；环境没有 research 模块或 Torch。

该复验保持原题与全部答案，只明确选择新的 `personal-cpu-candidate@v3`。两个 Python 环境的全部四页均与上述 editable 参考逐页 PNG 字节、像素、语义、字形来源、几何和公式组一致；两环境相互也一致。导入、导出、重新导入及 v3→v2→v3 回滚通过，默认 Profile 保护通过。平台 run 已由本 E2E Agent 再次执行 `verify_run` 并核读 `reports/installations.json`。

平台检查器使用“审核外部已安装 runtime”模式，因此子回执没有冒称自己执行安装；实际干净安装证明在同一 run 的 `reports/installations.json`、安装命令及包审计中关联到子回执。此前独立审计 run 的 clean-install pending 是当时状态，保留不改写；本节由后续真实安装证据解除该 Linux 阻塞。

[最终四页 PDF](../workspace/runs/20261120T155718248776Z-m21-clean-installed-new-job-43bb9000/rendered/python-3.12/candidate/homework_solutions.pdf)；[逐页 PNG](../workspace/runs/20261120T155718248776Z-m21-clean-installed-new-job-43bb9000/rendered/python-3.12/candidate/pages/)。对应目录保留可编辑答案和完整来源报告。

**真实 macOS 尚未执行。** Linux 双版本通过不构成 macOS 证明。

当前结论：受控原题 intake、独立求解、Linux 双 Python 干净安装和完整四页工程出稿 **Go**；真实 macOS、外部真实新作业范围和作者完整页面风格接受尚未完成，生产发布 **No-Go**。
