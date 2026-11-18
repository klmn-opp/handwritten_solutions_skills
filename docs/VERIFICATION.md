# 验证记录

## Phase 2（2026-11-18）

Linux x86_64、Python 3.12。复查前 29 项基线通过。扩展后 49 项通过，包括两个 MTH 兼容环境用例。[JUnit](evidence/phase2/tests.xml)

| 检查 | 实际观察 |
|---|---|
| 前序实现复查 | 发布门槛原先只遍历报告列出的文件，清单为空时会跳过校验。现在按页数重建必需清单，检查每个摘要，并兼容旧 Windows 报告的路径分隔符。Probe 与正式排版共用 Handnote 配置和 Matplotlib 状态锁。 |
| A/B/H | A4 300 DPI 单题、五题混合、同题子题、重复原编号的不同来源、多个输入文件。空清单、缺题干、验证失败等情况仅写失败报告。 |
| C | 对四种预设纸张、自定义 JPG 和 PDF 指定页验证了背景优先级、底色像素和墨迹可见性。同一图像解码路径支持 PNG。 |
| F/G | 正文/矩阵缺字、白名单外命令、未闭合公式、显式分页、过长推导、单行超宽公式和不可见墨水会报错。自动去除空行的重试成功后保留原始 Markdown。 |
| 交付和复用 | 验证页面编号、顺序、哈希、数值等式和归档完整性。交换页序或修改 PNG 会使验证失败。固定环境从交付 Manifest 重渲染 PDF/PNG 字节一致。 |
| 公开题演练 | 阅读 Project Euler 第 1 题，按倍数求和后扣除重复项。独立枚举与精确有理数复算均得 233168。已检查真实 PDF 页图中的英文、矩阵、分式和统计子题。 |
| 待实测 | 新 Codex 进程中的 Skill 自动发现、真实扫描题识别率、真实字体风格、macOS/Windows。 |

[公开题来源](https://projecteuler.net/problem=1) · [作业输入与解答](../examples/phase2/euler) · [合成五题回归材料](../examples/phase2/mixed)

## Phase 1（2026-11-18）

沿用已锁定 Handnote 和 Phase 0 依赖，不新增 Runtime 依赖。环境仍为 Linux x86_64 / Python 3.12.16。

| 项目 | 结果与范围 |
|---|---|
| 项目自动化 | 29 passed，含两个隔离 MTH 测试；[JUnit](evidence/phase1/project-tests.xml) |
| D：Profile 切换 | bundled 与个人合成字形输出不同；固定种子渲染重复、回滚复现 |
| E：多轮构建 | v1/v2 原件和样本索引保留；旧轮廓/度量保留，第二轮不同真实变体加入；九轮无八套截断 |
| I：无训练环境 | 子进程阻止导入 Studio/神经模块仍可默认渲染；独立 wheel 环境临时移走 hwsol.training 目录，默认及导入个人 Profile 均成功；[证据](evidence/phase1/wheel-runtime.json) |
| J：构建时渲染 | 构建线程持锁暂停，旧发布版本完成渲染；第二写入者拒绝；未完成候选不可见 |
| 实际后端 | Handnote capture → TTF、合成扫描 → TTF、TTF 导入；合成 CFF OTF 单独导入/评估/发布/渲染及合并失败边界 |
| 异常与完整性 | 低质量隔离、无可用样本、构建失败、损坏字体/轨迹、预览/字体篡改、重复导入、路径穿越/符号链接拒绝 |
| CLI 演练 | 21 次调用：默认渲染、两轮构建/评估/发布、比较、回滚、导出/新目录导入；摘要与产物见 out/phase1-demo |
| 打包与代码检查 | 锁定离线安装、Ruff、compileall、Skill 校验、sdist/wheel 和独立 Runtime 安装（27 包，无 pytest/神经包）；[构建](evidence/phase1/build.txt) |
| 可视检查 | 已查看 v2 PNG，合成 x 与数学渲染正常；没有宣称作者字迹相似度验收 |

检查资产复制暂存和原子发布逻辑；修正 .gitignore 的 profiles 规则为根目录限定，避免遗漏源代码包。
未配置专用类型检查器、测试覆盖率或联网漏洞扫描；没有 macOS/Windows 实测结果。新命令机器错误退出 2，
argparse/Typer 自身的用法错误仍采用其标准文本提示。没有手写真实性/自动笔画完整性判定。

## Phase 0（2026-11-17，历史记录）

环境：Linux x86_64，Python 3.12.16，Node 22.19.0。上游提交见 [PHASE0](PHASE0.md)。
只在该环境验证执行，不把声明的 Python 范围或可移植路径解析当作跨平台实测。

| 项目 | 结果与范围 |
|---|---|
| 安装 | `uv sync --locked` 成功；随后 `--offline` 成功 |
| 项目测试 | `HWSOL_MTH_PYTHON=... pytest`：14 passed；[JUnit](evidence/phase0/project-tests.xml) |
| 实验 corpus | 两后端共 14 项，全部符合预期行为；不支持项的失败是正确结果 |
| 上游 Handnote | 默认字体六组共 69 项通过；[日志](evidence/phase0/handnote-upstream-default-font-tests.txt) |
| 首轮字体差异 | Patrick Hand 四组 43 项中 3 失败；[原始日志](evidence/phase0/handnote-upstream-tests.txt)，原因见 PHASE0 |
| draw-your-font | 原始 `npm test` 通过：6 字形、TTF/WOFF/WOFF2、字腔和度量；另生成模板 |
| 构建 | `uv build --offline` 生成 sdist 和 wheel，包含测试字体、OFL 和 provenance |
| wheel 运行 | 独立临时环境从锁定缓存安装 wheel 和仅 Runtime 依赖，运行 doctor 与公式 probe；检查无 torch/pytest/训练包 |
| 语法/格式 | `compileall`、`ruff check`、`ruff format --check`；未配置专用静态类型检查器，不宣称类型检查通过 |
| Skill | 官方 `quick_validate.py` 通过；尚未做完整生产 Skill 安装与真实题目闭环测试 |
| 安全边界 | 检查无 shell=True、动态 eval、云 API、硬编码凭据；渲染测试阻断 socket.connect；不是全面安全审计 |
| Git | 初始目录不是有效 Git 仓库，受管理 .git 为只读；未创建 commit，按文件清单检查改动 |

当前没有测试覆盖率百分比、联网漏洞数据库结果、跨平台结果或真实手写相似度分数，均不虚报。
上游源码没有改动；本地执行产生的 ignored 构建/测试目录不属于上游源码变更。
