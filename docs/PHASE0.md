# Phase 0 技术验证报告

日期：2026-11-17。范围按用户最新指示收敛为 **Phase 0 + 后续规划**。
当前选择 **Handnote 作为主渲染后端**；MarkdownToHandwrite 保留独立环境中的实验兼容 Adapter。
此结论针对下述固定提交和 Linux CPU 环境，不代表全部 LaTeX 或所有平台已通过验证。

## 1. 核实的上游

| 项目 | 固定提交 | 实际用途与结论 |
|---|---|---|
| [Handnote](https://github.com/zh271/handnote) | `114919f183817643c795e47e851ea25b96e3c393` | Python ≥3.11；本项目固定 3.12/3.13。CPU Markdown、MathText、向量 PDF、PNG、采集和字体构建可用；模型为可选 extra |
| [MarkdownToHandwrite](https://github.com/lyc07/MarkdownToHandwrite) | `634cf07987a9c84c59f808e03f879c9d93485374` | Python ≥3.10；`Pillow<11` 与主环境分离；复杂公式补充，输出栅格 PDF |
| [handwriting-to-font](https://github.com/obsrvgmi/handwriting-to-font) | `27c048a3de2315bd896ecf7b6836dcd007bceeed` | Agent Skill，不是独立 Python 字体库；代理负责识别与标注，实际调用 draw-your-font |
| [draw-your-font](https://github.com/danilo-znamerovszkij/draw-your-font) | `ef367bb2193b523a03e313b9219e31055558ae70` | Node ≥18；实际 CLI 版本 0.1.0；合成图片 → 字体成功，暂不纳入默认安装 |

规划给出的 `danielgatis/draw-your-font` 未取得有效仓库内容。实际来源由 handwriting-to-font
README 的链接、解析脚本和目标仓库 `package.json.repository` 交叉确认，不能直接沿用原地址。
未修改任何上游源码；下载、模型、网页服务都不在日常 probe 路径上。

## 2. 验证过的接口

Handnote 的真实 CLI：

```bash
python -m handnote render input.md --font personal.ttf -o output.pdf
python -m handnote analyze input.md --font personal.ttf
python -m handnote symbols template --chars 'Abgiox012+-=' -o template.pdf
python -m handnote symbols simulate --spec template.json --font personal.ttf -o scans
python -m handnote symbols build --spec template.json -o symbols.ttf scans/scan-1.jpg
```

源码确认的公共 Python 接口：`handnote.markdown.parse`、`RenderConfig.from_mapping`、
`Renderer.render`、`Renderer.pdf`、`Renderer.coverage`、`capture.build_capture_font`、
`symbols.merge_fonts`。当前 Adapter 只使用渲染接口。采集与合并验证属于上游技术评估，
并不表示已实现 hwsol 的持续训练或 Profile 发布机制。

MarkdownToHandwrite 的真实 CLI：

```bash
python -m markdown_to_handwrite input.md --config config.json --font personal.ttf -o output.pdf
```

公共接口为 `config_from_dict`、`parse_markdown`、`ReportRenderer.render` 和 `save_pdf`。
实验 Adapter 通过指定 Python 启动隔离 worker，显式关闭 SDT、指定不存在的轨迹路径
（单独设置 enabled=false 仍会初始化内置轨迹包），关闭素材自动搜索，并使用明确的 STIX 回退字体。

draw-your-font 的真实命令为 `template`、`segment`、`build`、`make`、`preview`。
已运行上游 `npm test`：从合成图片找到 6 个字形，构建 TTF/WOFF/WOFF2/CSS，校验字形、
基线、下降部和字腔方向；另生成了 minimal 采集模板。没有真实作者样本，**没有验证真实照片的还原质量**。

## 3. 对比实验

统一输入见 [common.md](../examples/phase0/common.md)。两后端使用相同的原始文本、
Patrick Hand、A4、150 DPI、13 pt、18 mm 边距、seed=42；后端排版策略仍有差异。
人工检查过两后端的整张公式测试 PNG，覆盖分式、上下标、单位、积分、求和、极限、根式、向量、
希腊字母、均值/方差、2×2 矩阵，以及多个独立公式块。

| 检查 | Handnote | MarkdownToHandwrite |
|---|---|---|
| 统一英文/数学稿 | 1 页，已检查 | 1 页，显式 STIX 回退后已检查 |
| 同一稿重复 3 次 | 3 页 | 4 页 |
| 字体可指定 | Patrick Hand / DejaVu Sans 切换，像素哈希变化 | 明确传入正文、数学和回退字体 |
| `aligned` 环境 | 失败；Adapter 拒绝交付 | 1 页成功，作为兼容能力保留 |
| 无效 LaTeX | 上游 CLI 返回 0 并绘制降级内容；Adapter 拒绝 | 上游 CLI 返回 0 并留空公式；Adapter 拒绝 |
| 缺失正文字符 / 空输入 | 拒绝，保存失败报告，无 PDF | 拒绝，保存失败报告，无 PDF |
| 固定种子复现 | PNG 与向量 PDF 哈希一致，自动测试 | 未验证二进制 PDF 可复现；ReportLab 可能含时间元数据 |
| 300 DPI | 2480×3508 PNG，PDF 仍为 A4 | 接口支持，本阶段未做专项验收 |

原始计时、错误信息和环境见 [probe-summary.json](evidence/phase0/probe-summary.json) 与
[reports](evidence/phase0/reports/)。本机单页约 0.5 秒量级、三次重复稿约 0.5–0.9 秒；
这些是已热启动环境下的单次测量，**不是严格性能排名**，不含安装时间，不保证跨机器一致。
完整输出保存在工作区 `out/phase0-final/`；不将个人作业或大型生成文件纳入源码。

上游 CLI 退出码、原始输出及实际参数见 [upstream-cli.json](evidence/phase0/upstream-cli.json)。

## 4. 发现的问题与处理

1. **不能以退出码 0 证明公式成功。** 两后端的无效公式原始 CLI 都返回 0。当前 Adapter
   读取 Handnote warnings / MTH 的 `[latex-render-error]` 后拒绝发布 PDF。
2. **MTH 可静默漏符号。** 初次仅指定 Patrick Hand 时，α、β、σ、→ 未显示，虽然结构检查通过。
   修复为显式 STIX 字体链与预检查，并重新查看输出。仍未实现逐个实际选中字形的审计。
3. **Handnote 不支持本测试的 `aligned`。** Agent 可在 Phase 2 改写为多个 display 块，或显式选择
   兼容后端；当前不自动修改数学内容。
4. **分页语法不同。** Handnote 用独立 `---` 分页；MTH 用 `<newpage>`，不能盲目通用。
   自动分页都不是一题一页。超宽、超高对象和内容完整性检查仍属于 Phase 2/3。
5. **字体变体不能等同真实样本。** Handnote 支持真实 alternates，也支持合成变形。
   本阶段设置 `glyph_sets=1` 并关闭抖动，避免把合成字形误称真实个人写法。
6. **RGBA 处理。** 直接转换透明页为 RGB 会得到黑底；已使用上游 flatten 并显式白色纸张，
   添加白底像素回归检查。
7. **版本约束。** 未限制的 OpenCV 解析到了只有 Windows wheel 的 5.1.0.95；已固定实测
   5.0.0.93。MTH 要求 Pillow<11，单独安装，避免降低主环境的 Pillow 版本。
8. **上游字体测试有前提。** 以 Patrick Hand 代替预期中文字体跑 43 项采集/合并测试，40 通过、
   3 失败：新增字符数量阈值、中文字符覆盖和保存后字形名映射。恢复上游默认 Zhi Mang Xing 后，
   采集/合并加渲染/向量相关 69 项全部通过。保留两次原始记录；Phase 1 必须为西文字体补上
   轮廓/度量保持测试，不能仅比较内部 glyph name。源码会把 `post.formatType` 设为 3.0，名称可能规范化。
9. **范围限制。** 未运行神经模型、未用真实手写扫描件验收、未完成 macOS/Windows 运行验证。

## 5. 为什么选择 Handnote

已在 CPU 上通过英数与常用数学渲染，支持向量 PDF，公开 API 清晰，且与模板、轨迹采集、
多字形合并共享同一上游。无需另写字体生成器或公式排版器。主环境不含 PyTorch、ONNX Runtime、
PyMuPDF 或未知轨迹资产；额外 `pypdf` 仅负责独立页数/尺寸检查，不是自研 PDF 引擎。
上游核心依然强制安装 OpenCV；未来若需拆分训练依赖，应评估上游支持，当前不伪称零采集依赖。

MarkdownToHandwrite 的 `aligned` 能力有价值，但 Pillow 约束、未知许可的轨迹包、PyMuPDF
依赖及缺字静默失败，使其暂不适合作为默认生产依赖。详见 [许可证审查](LICENSES.md)。

## 6. 复现实验

主环境：

```bash
uv sync --locked
uv run --frozen hwsol doctor
uv run --frozen python scripts/phase0_probe.py --output out/new-probe
uv run --frozen pytest
uv build --offline
```

可选兼容环境（只用于评估，不打包其未授权资产）：

```bash
uv venv --python 3.12 /tmp/hwsol-mth-venv
uv pip install --python /tmp/hwsol-mth-venv/bin/python -r requirements-compatibility.lock
uv run --frozen python scripts/phase0_probe.py --output out/new-comparison \
  --mth-python /tmp/hwsol-mth-venv/bin/python
HWSOL_MTH_PYTHON=/tmp/hwsol-mth-venv/bin/python uv run --frozen pytest
```

Windows 使用自己的临时目录和 `Scripts/python.exe`，通过 PowerShell 设置环境变量。
`uv.lock` 固定主环境完整解析和哈希；兼容环境的等效锁文件固定所有已安装版本和上游提交。
draw-your-font 依赖锁来自上表固定提交的 `package-lock.json`。
