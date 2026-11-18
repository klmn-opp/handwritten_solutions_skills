# Handwritten Solutions

本地运行的手写作业 Agent Skill。Agent 读取文本、图片或 PDF，按原题号解答并保留验证记录。Runtime 用选定的字迹 Profile 排版，每道顶层题占一页，交付 PDF、逐页 PNG 和可编辑 Markdown。

Phase 2 已在 Linux 验证，内置 Profile 可直接运行，也支持导入已发布的个人字迹。示例中分别有公开题目和五题合成测试作业。真实作者字迹质量和三平台验收进度见 [实施状态](docs/IMPLEMENTATION_STATUS.md)。

## 安装与运行

需要 Python 3.12/3.13、Git 和 [uv](https://docs.astral.sh/uv/)。当前实测环境为 Linux x86_64 / Python 3.12。

```bash
uv sync --locked
uv run --frozen hwsol doctor
uv run --frozen hwsol render --manifest examples/phase2/euler/job.json --output out/euler
uv run --frozen hwsol verify --job out/euler
uv run --frozen hwsol profiles list
uv run --frozen hwsol render-probe --input examples/phase0/common.md --output out/common
uv run --frozen pytest
```

安装需要下载锁定依赖；安装完成后渲染不需要网络、GPU、PyTorch 或模型权重。运行 `uv` 时使用
`--offline` 可禁止依赖管理器访问网络；直接调用虚拟环境中的 `hwsol` 同样不下载任何资源。
Windows 对应可执行文件位于 `.venv/Scripts/`；macOS/Linux 位于 `.venv/bin/`。

`out/common/` 包含 `probe.pdf`、`pages/01.png` 等逐页图、`source.md` 和 `validation_report.json`。
输出目录必须为空或不存在，防止旧文件被误认作本次成功产物。失败退出码为 2；渲染期间失败会保存报告。

```bash
uv run --frozen hwsol render-probe --input examples/phase0/common.md --output out/custom \
  --font /path/to/your-font.ttf --config config/defaults.json
```

配置采用 JSON，尺寸使用 mm、字号使用 pt，未知字段报错。默认 A4 白纸、150 DPI；可设为 300 DPI。
`HWSOL_DATA_DIR` 可覆盖用户数据目录。默认 Linux 使用 `~/.local/share/handwritten-solutions`，
macOS 使用 `~/Library/Application Support/handwritten-solutions`，Windows 使用 `%LOCALAPPDATA%/handwritten-solutions`。
`render-probe` 的交付文件写入指定输出目录；样本、版本、构建记录保存在用户数据目录。

内附 **Patrick Hand / OFL-1.1 工程测试字体**，封装为只读 `bundled-default@v1`，不是项目作者笔迹。
完整许可及固定来源见 `src/hwsol/assets/presets/patrick-hand/`。正文和数字使用选定字体，缺失的数学字形可能回退到 STIX，报告保留覆盖率信息。

## 个人字迹 Profile

```bash
uv run --frozen hwsol profile template --output out/letters.pdf --chars "ABCabc123x" --sets 2
# 使用相应模板填写后导入；这里的路径替换成自己的扫描图片
uv run --frozen hwsol profile import --profile my-handwriting \
  --source /path/to/scan.jpg --template out/letters.json
uv run --frozen hwsol profile train --profile my-handwriting --append
uv run --frozen hwsol profile evaluate --profile my-handwriting --version v1
uv run --frozen hwsol profile publish --profile my-handwriting --version v1
uv run --frozen hwsol profiles activate my-handwriting@v1
uv run --frozen hwsol render-probe --input examples/phase0/common.md --output out/personal
```

再次导入新样本并构建会生成 v2 候选版本，历史原始素材和 v1 都保留。发布不自动激活；可以显式使用
`publish --activate`。运行时按 `--profile` → 用户默认 → `bundled-default@v1` 选择；普通渲染不调用 Studio。
Studio 也可独立通过 `uv run --frozen hwsol-studio ...` 调用，参数与 `hwsol profile ...` 相同。

```bash
uv run --frozen hwsol profiles compare my-handwriting@v1 my-handwriting@v2
uv run --frozen hwsol profiles rollback my-handwriting@v1
uv run --frozen hwsol profiles export my-handwriting@v2 --output out/my-handwriting-v2.zip
# 另一台电脑或另一数据目录，无需原始样本或训练环境
uv run --frozen hwsol profiles import --source /path/to/my-handwriting-v2.zip
uv run --frozen hwsol render-probe --input examples/phase0/common.md --output out/imported \
  --profile my-handwriting@v2
```

轨迹 JSON、扫描、TTF/OTF 的支持边界与多轮操作见 [训练指南](docs/TRAINING.md)；
可运行 `uv run --frozen python scripts/phase1_demo.py --output out/phase1-demo-new`
完成两轮**合成轨迹** CLI 演练，不需要个人样本。

详细证据：[Phase 0 报告](docs/PHASE0.md) · [Profile 格式](docs/PROFILE_FORMAT.md) · [实现状态](docs/IMPLEMENTATION_STATUS.md) ·
[后续阶段规划](docs/ROADMAP.md) · [许可证清单](docs/LICENSES.md)。

## 在 Codex 中使用

将源码发行包解压到目标工作区的 `.agents/skills/handwritten-solutions/`，或把此 checkout 链接到该位置。在该 Skill 目录运行 `uv sync --locked --no-dev`。在可以发现该 Skill 的 Codex 会话中使用 `$handwritten-solutions` 并提供题目。Agent 按 [SKILL.md](SKILL.md) 读取原始材料，完成解题、验证与交付。单独安装 wheel 可使用 CLI，Skill 指令和示例在源码发行包中。

[Manifest 格式和排版规则](references/jobs.md) 描述输入文件、验证状态、背景优先级、可复算等式和故障恢复。运行五题示例可使用 `hwsol render --manifest examples/phase2/mixed/job.json --output out/mixed`。
