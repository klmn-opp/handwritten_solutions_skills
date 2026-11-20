# M2.3 — Qualified personal CPU RC2

2026-11-21。**本地 Release Candidate 工程验收 Go；生产激活与正式发布未执行。**
当前唯一待发布候选为 `1.0.0rc2`，运行时源码冻结于
`3f8bff9c1659ecb9636f45764d103a0bccbc4911`，独立分支
`feature/m23-qualified-personal-cpu`。历史RC1仍保留，不覆盖旧包。

## 技术假设与结果

已有真实个人墨迹及确定性结构配方已经有效。产品化缺口是可审核的本地准入，
不需要再训练一个新模型。新增资格记录绑定实际安装、完整作业与作者授权后，
个人Profile可以经普通 `font-cpu` Job出稿，效果应与已接受的实验路径一致。

这个假设通过真实安装和CLI验证：Linux及Mac的普通出稿、激活后默认出稿与各自
已审核v5候选逐页像素、语义、来源、几何和公式一致。无需GPU、Torch、research
模块或预先生成的题目词图。跨系统RAQM/BASIC差异保留并解释，不宣称跨系统像素相同。

## 实现与权限边界

- `profiles qualify-ink`：验证严格类型和哈希绑定的完整证据，保存不可覆盖的本地资格快照。
- `profiles publish-ink`：单独发布到本地资料库，不改变默认设置。
- `profiles activate` / `rollback`：单独选择已合格版本或回到历史版本。
- 普通 `font-cpu`：仅在Profile、配方、wheel、源码、依赖及当前平台文字能力仍匹配时调用个人配方。
- 导出后重新导入仍进入候选区；ZIP中的任意admission字段不能取得发布资格。

资格不存入不可变Profile资产，不能随ZIP自动传播。Linux/macOS、架构、Python小版本、
Pillow/RAQM/BASIC/FreeType等实际能力分别记录。构建或环境变化需要相应证据重新核验。
这是本地证据一致性机制，不是对能改写整份本地资料库的恶意拥有者的远程证明。

作者对原四页整体满意，要求v4再加8%词距，明确无需再验收。最终v5为1.1448，
192项字体/墨迹资产不变。资格记录明确区分作者已接受的v3、指定修改起点v4及v5
范围豁免；不虚构作者看过Mac页面。Linux的一个既有d选取、Mac的两个f适配变化
有独立审核，每个平台比较自己的原生基线。未扩大成任意笔迹修改授权。

## 实际证据

|证据|不可复用运行ID后缀|结果|
|---|---|---|
|RC2干净Linux安装|2880993d|Python3.12.16/3.13.18，全部安装与候选操作通过|
|RC2真实Mac干净安装|a8b00dcb|macOS15.6.1 arm64，Python3.12.16/3.13.7通过；仅复用公开下载缓存|
|完整资格证据包|bc7b1c05|四个实际环境、254个绑定文件；新包页面与既有审核记录精确对应|
|Linux实际资格生命周期|90cbf090|实际qualify/publish/普通出稿/默认出稿/rollback/export/reimport全部通过|
|Mac实际资格生命周期|fd78ef1d|两套解释器全部通过；普通与激活后四页均与该Mac候选一致|
|新内容验证|b1f90723|RC2冻结后新增概率题，独立推导与符号核验，普通已准入CPU入口生成完整一页；197数学项目无字体回退|

上述新题属于明确标注的Agent编写受控输入，不是外部未见大学试卷或新作者Holdout。
原四题先冻结题目PDF，再阅读、解题和独立核验；新概率题先冻结原题文本再保存解答。
新题经主/独立Agent检查无确认的漏项、重叠或裁切；高根号钩形较弱但仍可辨，保留为风格观察，
不将Agent检查冒充作者的新增风格批准。
渲染器负责结构/来源/边界，数学正确性来自单独核验，二者没有混淆。

[普通CPU四页PDF](../workspace/runs/20261120T170621036164Z-m23-qualified-cpu-lifecycle-90cbf090/rendered/qualified/homework_solutions.pdf) · [逐页PNG](../workspace/runs/20261120T170621036164Z-m23-qualified-cpu-lifecycle-90cbf090/rendered/qualified/pages/) · [新增概率题PDF](../workspace/runs/20261120T170751899204Z-m23-fresh-qualified-question-b1f90723/rendered/homework/homework_solutions.pdf) · [新题PNG](../workspace/runs/20261120T170751899204Z-m23-fresh-qualified-question-b1f90723/rendered/homework/pages/01.png)。

[wheel/sdist/Skill及校验文件](../workspace/runs/20261120T165554868752Z-m21-clean-installed-new-job-2880993d/reports/dist/)由冻结Git对象生成，公开文件白名单与私有数据排除检查通过。
用户文档在公开包的GUIDE、README与SKILL中说明一般出稿和独立资格流程。

wheel SHA256：`3e6a07279b78bd9f0bffc80c1c579c9e393c2ce248209583351a8b30d22c3c08`。
资格request SHA256：`87f6e847d7f9be6fd16a97ed25ddf1241727efcc07208bdc2b10b90ebcf56021`。
资格ID：`6c58cc6a45e714c578ff43cc8d006c7d98a2e151459906c9f6062d047f16a94a`。
Linux生命周期run SHA：`68b156e186e33ed51995c38e6c098eb8d9e74457fbed61b1789ebeff670901fa`。
Mac生命周期run SHA：`d419de16dd0e93a05b9191bef3e8cbaf5311c9fc7f5796cbfdcf473e48585652`。

运行源码、配置、Profile、种子、完整命令、环境、失败日志、PDF/PNG和来源报告均保留。
研究协调器为 `m21_clean_install`、`m21_native_mac`、`m23_qualification_bundle`、
`m23_qualified_lifecycle`、`m23_portable_qualified_check`、`m23_fresh_qualified_question`；
它们不属于已安装产品的运行依赖，每次执行的精确副本留在相应运行目录。

## 失败、限制与下一阶段

独立审查发现并在冻结前修复了单一Linux基线不能代表Mac的问题。合成DPI301用例
先被已有范围校验拒绝，改为合法但不同的299以验证资格配置门禁；没有放宽渲染限制。
子运行 `run.json` 嵌套会冲突的研究包装问题在实际执行前修复：可变库/工作区置于独立
临时目录，原回执保留，最终归档使用重命名副本。RC2真实安装和资格执行没有用失败
结果冒充成功。更早的宏拒绝、轮廓接触、打包顺序及词距严格比较失败记录继续保留。

此Go仅覆盖实际环境和支持范围。ℤ、密集Φ的限制和不支持结构仍有明确报告，见
[数学来源与回退](M22_SOURCE_GAPS.md)。不承诺任意公式零回退，也不将安装次数或
自动测试数当作笔迹自然度提升。神经模型未更新；现有v21骨干及连续监督研究的
旧残差假设No-Go保留，后续M3须先提出可区分形态、几何和墨色的监督任务。

生产main、latest、默认个人Profile、原始扫描、历史标签/权重与封存C保持不变。
所有本轮publish/activate均发生在隔离验收库。正式发布/替换默认仍需要用户明确批准；
本轮作者词距免复核不意味着批准生产推广。该保护不影响普通CPU产品候选的工程Go。
