# 许可证与依赖审查（Phase 0–2）

本项目新增代码采用 [MIT](../LICENSE)。这不改变任何依赖、字体、数据或模型的许可。
核对范围是固定源码提交的 LICENSE/依赖文件、实际安装包元数据及关键资产声明；不把元数据扫描
当作所有二进制内嵌第三方代码的完整发布审查。正式发布前继续完成 Phase 3 清单。

## Phase 2 新增运行依赖

| 组件 | 版本 | 用途与已核对许可 |
|---|---|---|
| pypdfium2 | 5.15.2 | PDFium 栅格化。上游 wheel 元数据列 BSD-3-Clause、Apache-2.0 和依赖许可。发行时保留实际平台 wheel 的 PDFium/第三方声明。 |
| SymPy | 1.14.0 | 有理数精确复算，BSD-3-Clause。 |
| mpmath | 1.3.0 | SymPy 的间接依赖，BSD-3-Clause。 |

Linux 安装包版本、依赖和所附许可摘要见 [本阶段清单](evidence/phase2/licenses.json)。运行安装仍采用锁定上游包。pypdfium2 发行信息可查 [PyPI](https://pypi.org/project/pypdfium2/5.15.2/)。

## 默认环境

| 组件 | 实际版本/来源 | 许可及处理 |
|---|---|---|
| Handnote | `114919f...` / 2.0.0.dev0 | [MIT](https://github.com/zh271/handnote/blob/114919f183817643c795e47e851ea25b96e3c393/LICENSE)，固定提交，不复制或修改源码 |
| Patrick Hand | Google Fonts `4aa2f964...`，515 个 cmap 字符 | [OFL-1.1](../src/hwsol/assets/presets/patrick-hand/OFL.txt)，TTF/许可/来源摘要随包附带；未修改字体；工程测试用 |
| FontTools | 4.68.0 | MIT |
| Pillow | 13.0.0 | MIT-CMU；二进制依赖许可见安装包附带文本 |
| Matplotlib | 3.11.2 | Matplotlib License（PSF 衍生）；包含字体及其他组件的独立条款 |
| NumPy | 2.5.5 | 元数据为 BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0；查看 wheel 附带声明 |
| OpenCV headless | 5.0.0.93 | 安装包元数据 Apache 2.0；上游说明另列打包脚本与二进制第三方条款 |
| pypdf | 7.1.0 | BSD-3-Clause；用于 PDF 组合、页身份、页数和尺寸核验 |
| Pydantic / Typer | 2.14.3 / 0.29.1 | MIT |
| STIX / DejaVu | 来自已锁定 Matplotlib | STIX 字体许可 / Bitstream Vera 及 DejaVu 条款；不另复制进项目 assets |

其他实际安装的间接依赖包括 contourpy、cycler、kiwisolver、pyparsing、python-dateutil、tzdata、
annotated-types、pydantic-core、typing-extensions、typing-inspection、annotated-doc、rich、
markdown-it-py、mdurl、Pygments、shellingham、packaging。逐包版本、许可字段、依赖关系和许可文件
摘要见 [runtime-licenses.json](evidence/phase0/runtime-licenses.json)；该开发环境清单同时包含
pytest/build/Ruff 等开发工具。仅 Runtime 的新环境安装了 27 个发行包（含本项目）。

上游 [CREDITS](https://github.com/zh271/handnote/blob/114919f183817643c795e47e851ea25b96e3c393/CREDITS.md)
还列出了 FreeType、模型与字体各自的条款。模型 extra 没有启用；xkcd 的非商业字体也没有使用。
不能把上游代码 MIT 推导为所有模型权重和生成产物都可商用。

## 隔离的兼容环境

MarkdownToHandwrite 源码 [MIT](https://github.com/lyc07/MarkdownToHandwrite/blob/634cf07987a9c84c59f808e03f879c9d93485374/LICENSE)，
但其 [ASSET_LICENSES.md](https://github.com/lyc07/MarkdownToHandwrite/blob/634cf07987a9c84c59f808e03f879c9d93485374/ASSET_LICENSES.md)
将 SDT 轨迹列为 **Pending verification / Do not redistribute until cleared**。该文件的旧目录名
与实际代码路径不同，实际 ZIP 位于 `src/markdown_to_handwrite/assets/sdt_trajectories.zip`。
它会被上游 package-data 打入 wheel，不能认为“pip 安装”就解决了分发授权。

本项目不把该包及轨迹资产列入默认依赖或 wheel；仅保留用户独立安装的评估 Adapter，测试时关闭并避免打开轨迹库。
在资产授权明确或上游提供可审计的无资产发行包前，不将其并入生产分发。

实际解析到了 Pillow 10.4.0、Handright 8.2.0、ReportLab 5.1.0、PyMuPDF 2.1 及其他依赖。
PyMuPDF 元数据明确为 **AGPL-3.0 或商业许可**；官方说明见
[License and Copyright](https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright)。
其 2.1 依赖链还安装了 ONNX Runtime、pymupdf-layout、pymupdf4llm 等包；后两者元数据未给出明确许可字段，
当前不标为已清理。详见 [compatibility-licenses.json](evidence/phase0/compatibility-licenses.json)。

因此 Phase 0 主包暂以 pypdf 完成结构检查。Phase 2 如采用 PyMuPDF 做背景、合并和光栅化，
须明确整个分发方案的许可及依赖边界；不能将带有它的组合发行物直接宣称全部为 MIT。

## 照片转字体路线

handwriting-to-font 与 draw-your-font 顶层代码声明 MIT，但后者运行依赖
[`potrace@2.1.8`](https://github.com/tooolbox/node-potrace/blob/master/LICENSE) 为 GPL-2.0，
`sharp` 的 native/libvips 等组件另有条款。已保存上游锁文件中的运行依赖版本/许可/integrity：
[draw-your-font-licenses.json](evidence/phase0/draw-your-font-licenses.json)。
当前仅在 `/tmp` 独立安装、执行合成测试，不将其代码或依赖再分发到主包。

Handnote 模板模拟使用 Patrick Hand 产生的测试字体属于该字体相关衍生产物，不能称为作者真实手写字迹。
这些中间实验资产不进入发行包；实际 Profile 发布时仍须逐项携带来源、字体与样本授权信息。

## 后续发布门槛

Phase 1 未增加第三方依赖。bundled-default 保持原始 Patrick Hand 字节和 OFL；个人累积构建更改字体 family 名，
并携带基底许可、来源文件及用户提供的样本许可。导入时未提供许可的样本标为
`LicenseRef-Unspecified-Local-Only`，生成 Profile 默认 `distribution=local-only`；这不是法律授权判断，
也不会替代底层字体本身的条款。合成 demo 不含真人笔迹，其基底字体仍受 OFL 约束。
Runtime ZIP 不含原始样本，完整 Studio 数据目录不进入发行包。

- 代码、字体、原始样本、数据集、模型权重、生成字体分别建账，记录授权来源与固定哈希。
- 不分发授权不明的 MTH 轨迹、SDT/IAM/CASIA 相关权重或用户个人原始样本。
- 当前未做联网漏洞数据库审计；许可证检查不能替代安全审计。
- macOS/Windows native wheel 的附带许可需在对应平台安装后复核。
