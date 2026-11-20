# Handwritten Solutions v1.0.0

首个正式公开的 CPU 发行版本：将经核验的 Markdown / LaTeX 解答输出为手写风格 PDF、逐页 PNG 和可追溯报告。每道顶层题目一页，支持背景、版本化 Profile 和本地导入/审核/发布/回滚。

## 下载与使用

- `handwritten-solutions-1.0.0-skill.zip`：完整 Agent Skill、源码、使用指南与可复现 Demo；解压后按 README 安装。
- `handwritten_solutions-1.0.0-py3-none-any.whl`：Python 3.12/3.13 CLI 安装包。
- `handwritten_solutions-1.0.0.tar.gz`：可重建的完整源码包。
- `SHA256SUMS`、`release.json` 和 `PUBLICATION.json`：原始文件校验及公开/开发源码身份绑定。

正常出稿不需要 GPU 或 PyTorch。题目识别、解题与独立数学核验由宿主 Agent 完成，CLI 不包含通用解题模型。安装锁定依赖需要网络和 Git；安装完成后可使用本地运行环境出稿。

## 真实示例与边界

README 两页示例来自实际 CPU 渲染，稳定版重放后的 PNG 与公开图片逐字节一致。示例使用 OFL-1.1 的 Patrick Hand 默认字体，并记录 STIX 根号/撇号回退。公开包不包含作者私人原稿、Profile 或研究权重。

个人字形解析与数学结构保持受控；不支持的公式明确拒绝，缺失字形明确记录回退。有限测试页的零回退不是任意公式全面覆盖的保证。连续神经手写、上下文训练及偏好优化仍在研究阶段。

## 验收记录

同一份最终 wheel 已在 Linux x86_64 与真实 macOS 15.6.1 arm64 上，分别使用 Python 3.12 / 3.13 完成干净安装、Profile 导入/导出、公式 PDF/PNG 和回滚。四页个人作业与各平台已认可版本保持相同像素及结构；跨平台字体布局库不同，不承诺 Linux/Mac 像素一致。

最终运行时代码的 GitHub CI 在 Linux/macOS × Python 3.12/3.13 四组均通过，包含完整测试、源码包构建及 wheel 冒烟检查。发行前修复了 Benchmark 原题图片错误依赖系统字体的问题，并在真实 Mac 完整测试中复现和验证修复。Windows 自动测试仍失败，本版不声明支持 Windows。

公开仓库保留了原先 261 个研发提交的原始 ID、作者及时间。主页另有经作者授权公开的[真实个人笔迹四页演示](https://github.com/klmn-opp/handwritten_solutions_skills/tree/main/public_docs/personal-demo)，与安装包内可独立复现的默认字体示例分别说明。原始扫描、私有 Profile 和研究权重未随包分发。

所有下载文件均为已验收的原始预构建文件，GitHub 发布流程只核验和上传，不重新构建。
