# M2.1–M2.3 本轮验收结果

2026-11-21（运行 ID 使用 UTC）。本轮本地增量通过；**跨平台与正式 RC 仍为 No-Go**，真实 Mac 尚未执行，新的完整页面还需要作者视觉确认。

|项目|实际结论|
|---|---|
|Linux 安装与通用出稿|源码 `4a0e52ac1f8c643b3a3714c0867cbea562fbee6b`，全新 Python 3.12.16 / 3.13.18 环境，普通安装 CLI，无 Torch/research/特定页面缓存依赖。|
|新题端到端|先冻结四道新编原题 PDF，再由 Agent 阅读、解题和独立校核。完整四页通过，两环境与已审参考逐页像素、结构、来源和几何相同。属于受控新题工程证据，不冒充外部未见作业或扫描 OCR 验收。|
|实际缺陷修复|新题标题 l/d 暴露真实轮廓接触；逐项拒绝导致接触的字形调整，保留原字形及来源。不改标题、数学内容或页面门禁。|
|已接受页面保护|旧24页 PDF 像素、语义、来源及几何全部相同；没有把测试数量作为笔迹改善证据。|
|数学来源|新增可靠原迹 ∀，仅其区域440像素改变；ℤ保持显式字体回退，密集Φ保持间距不足的明确拒绝。四道新题827项数学内容无外部字体回退，6个程序结构明确记录。|
|迁移与回滚|v3→v2→v3，导出、独立资料库重新导入、逐项资产哈希、默认设置和 latest 保护通过。|
|安装包|wheel / sdist / Skill 实际构建和私有数据排除检查通过。当前为未发布4a构建，元数据仍0.6.0；`1.0.0rc1`仅保留为未来候选编号，不能改文件名冒充 RC。|
|真实 Mac|作者确认设备归属并授权复制。SSH已可达但公钥登录被拒，按作者偏好改为其终端输入密码并建立可复用连接。未传输、未运行，不能以Linux结果代替。|
|后续连续训练|已有真实连续墨迹与具体数据任务，但产品验收尚未完成。保留v21骨干及旧残差假设No-Go；未启动新训练、偏好优化或RSI。|

[最终四页 PDF](../workspace/runs/20261120T155718248776Z-m21-clean-installed-new-job-43bb9000/rendered/python-3.12/candidate/homework_solutions.pdf) · [逐页 PNG](../workspace/runs/20261120T155718248776Z-m21-clean-installed-new-job-43bb9000/rendered/python-3.12/candidate/pages/) · [原题与核验链](M21_NEW_HOMEWORK.md) · [来源与回退原因](M22_SOURCE_GAPS.md)。

[干净安装回执](../workspace/runs/20261120T155718248776Z-m21-clean-installed-new-job-43bb9000/run.json)保留完整命令和环境。wheel SHA256 为 `df96bf879070431c85c8e8b6541235afc14162098b5375ab98c97a1db704accb`。在本机 CPU 上，四页候选渲染约23–24秒；不是其他机器的性能承诺。

[私有 Mac 验收包](../workspace/runs/20261120T160130778545Z-m21-authorized-mac-payload-104256e2/solutions/mac-acceptance.tar.gz)包含相同安装包、两版 Profile、完整 Job 和独立验收脚本；与实际安装输入的对应关系经主/独立 Agent 核验，见[绑定回执](../workspace/runs/20261120T160257851729Z-m21-mac-payload-independent-review-b37f8cc7/reports/bindings.json)。包内没有 SSH 密钥，不应公开发布。

失败31616c37（不支持宏）、703389ec（新接触）、004eec3b（Skill打包顺序）均保留。最后一项只修研究打包协调器的顺序，没有修改冻结运行时。原始扫描、标签、旧 Profile、神经权重、main、正式 latest 和封存 C 均未改变。
