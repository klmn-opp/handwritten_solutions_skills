# M2.3 — RC 准入缺口审计

2026-11-21。本次只读检查源码和公开发行入口；本文件是唯一新增文件。没有修改冻结运行时 `4a0e52ac1f8c643b3a3714c0867cbea562fbee6b`、Profile、训练数据或发布门禁。真实 Mac 验收正在其他任务执行，结果以实际回执为准。

结论：**仅把版本改成 `1.0.0rc1`，不能宣称用户要求的 M2.3 已完成。** 除 Mac 和作者页面验收外，还需要统一版本身份、支持真实的预发行构建，并明确 RC 对个人 CPU 候选的承诺范围。已经实现的 CPU 安装、出稿、Profile 迁移和明确拒绝机制无需重建。

## 实际工程缺口

|项目|源码证据|所需有界修改|
|---|---|---|
|版本身份不一致|`pyproject.toml:8` 为 `0.6.0`，`src/hwsol/__init__.py:3` 为 `0.3.0`；`src/hwsol/cli.py:35` 的 doctor 仅报告依赖版本，没有本包版本和 build 身份。|建立单一版本来源并在普通 CLI 暴露分发版本、源码 commit 和构建身份。检查 wheel、sdist、Skill 及锁文件的本项目版本一致性。不得重命名旧产物模拟 RC。|
|正式打包脚本不支持 RC 编号|`scripts/package_release.py:237` 正则仅接受 `vN.N.N`，会拒绝 `v1.0.0rc1`；`:240` 必须解析正式 tag；`:243` 核对 tag 与项目版本。|提供明确的本地预发行构建入口：从已冻结 commit 和实际预发行元数据生成不可复用目录，或扩展受控 prerelease 支持。保留版本匹配、构建来源、公开资产白名单及哈希审计。生成本地 RC 不隐含创建正式 tag、上传或生产发布。|
|个人 ink 的独立发布准入没有实现|`src/hwsol/training/studio.py:373` 的 publish 对任何 ink_bundle 无条件拒绝，提示需要 separate release qualification；`src/hwsol/personal_ink/bundle.py:52` 的 admission 只是通用 dict；当前源码未实现对应的发行资格校验流程。|若 RC 要宣称个人 CPU 配方已获正式资格，应先实现独立、显式、绑定源码/模型或配方/Profile/数据/评价哈希的准入记录和验证。缺少证据或证据过期继续拒绝。不能把任意 admission 字段或 passed 渲染回执当作用户授权。|
|当前个人候选始终是实验后端|`src/hwsol/jobs/runtime.py:162` 只以 personal-cpu-experimental 从 candidates 区解析；`:179` 阻止默认后端直接用 ink_bundle。`src/hwsol/workspace.py:237` 保持 held、禁止更新 latest；`src/hwsol/personal_ink/runtime.py:143` 固定 production_qualified=false。|明确本轮 RC 是“稳定 font-cpu + 可选实验个人墨迹”，还是还要完成“已准入个人 CPU Profile”。前者可保持门禁不动；后者需要真实准入设计及独立验收，不能仅换名称。|

`src/hwsol/profiles/store.py:177` 仍拒绝将实验 ink 设为默认；`:246` 的导入会校验资产并按 ink_bundle 放入 candidates，`:286` 明确 activated=false。`src/hwsol/profiles/schema.py:137` 保持个人 ink 为 local-only。这些是应保留的保护，不是为了通过 RC 而应移除的障碍。导入、导出和 v3→v2→v3 切换已经有实际安装证据；它们不等于通过了上述独立发布生命周期。

## 用户文档的实际状态

公开包的 README 和 Skill **不是仓库根目录旧文档的原样复制**。`public_docs/distribution.json:63` 用版本化模板覆盖入口：`public_docs/README.md.in`、`public_docs/SKILL.md.in` 已明确区分稳定 CPU、个人实验候选、私有资产和神经研究；Skill 模板 `:29` 起提供普通 Job 工作流，并说明候选不能推进 latest。`references/jobs.md` 还记录了不支持宏的等价源修订与失败保留方式。不能把根 README 的旧发行说明误报成已经打包出去的错误说明。

需要补的是：将真实 RC 版本、实际平台支持范围、安装产物和限制写入新发行说明；核对公开 Profile 指南的普通 font publish 与个人候选 import/export 两条路径不会混淆。源码 checkout 的 `SKILL.md` 第5步仍先推荐旧 combined/contextual，再在末尾声明个人 ink 应用 baseline/raw，宜在下一次文档修改时合并成单一、按后端分流的规则。公开模板当前没有这个冲突。

## 两种安全的 RC 范围

1. **稳定 font-cpu 产品 RC，个人 ink 保持显式实验选项。** 普通用户使用授权的内置字体、标准 Job 和 CPU CLI，个人 v3 继续 held、不激活、不公开分发。这个范围允许构建诚实的本地预发行包，但不能宣称个人数学 Profile 的正式发布流程已经完成，也不能据此将整个 M2 标成完成。
2. **包含已准入个人 CPU 能力的 RC。** 增加独立资格机制，冻结配方和 Profile，绑定真实平台、完整作业与作者验收。先生成可审查的资格记录和本地候选；生产激活、默认 Profile 替换和正式发布仍需用户明确批准。不要复用仅针对普通字体的 Studio publish 来绕开 ink 门禁。

先推进范围1中的共同工程是有界且可逆的：版本统一、预发行打包、文档和精确安装核验。范围2的资格记录可以同步设计，但在证据不齐时保持未通过；本轮不需要改渲染架构或启动神经训练。

## 下一次可执行验收边界

- 保留 `4a0e52a` 与所有已有产物，另建版本候选 commit 和运行目录。
- 用真实的 `1.0.0rc1` 元数据构建 wheel、sdist 和 Skill；保存源码、依赖、公开文件清单和每个产物 SHA。只生成本地产物，不隐含发布。
- 新包在真实 Linux/macOS 的隔离环境重新安装，完成正常新 Job、导入、公式、PDF/PNG、导出重新导入和版本回滚。此前 `0.6.0` 标记的 `4a` 安装成功不能直接作为新 RC 安装成功。
- 把新包输出与冻结页面和语义来源进行比较；如保持一致，明确记录继承了哪一份视觉证据。作者验收绑定具体候选和产物，不能静默沿用泛泛的历史满意评价。
- 未支持符号、复杂密集结构拒绝和受控合成新题的局限继续公开。已有四题证据不冒充外部未见真实作业覆盖。
- 决定并记录两种 RC 范围中的实际选择；没有资格机制时，不宣传个人 ink 已发布。发布授权与工程验收始终分开。

这份审计没有要求扩大自动测试数量，也没有以版本号、测试数或训练次数作为笔迹质量收益。真正剩余的准入问题是：普通用户能否从明确版本的安装包完成完整工作流，以及每项产品承诺是否有绑定到该版本的实际证据。
