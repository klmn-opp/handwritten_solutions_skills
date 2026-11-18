# 2026-11-18 当前实现审计证据

对应 [全面审计](../../CURRENT_STATE_AUDIT.md)。本目录只保存摘要、检查结果和测试日志，不复制原始笔迹照片或字体。

| 文件 | 内容 |
|---|---|
| snapshot.json | 文档入口修改前的 Git/源码摘要、安装依赖、个人版本与样本账本、训练报告摘要、最新五题来源、旧发行文件摘要检查 |
| runtime-checks.json | 最新五题像素核验、重新渲染 PDF/PNG 字节相等、四模式个人横线探查、实际默认库 |
| additional-checks.json | schema 同步、源码导入位置、旧题包拒绝、Lab 目录、当前 GPU 查询、代码规模 |
| tests-py312.log / tests-py312.xml | 本轮 Python 3.12 全部 93 项通过 |
| tests-py313.log / tests-py313.xml | 本轮 Python 3.13 全部 93 项通过 |
| ruff-check.log / ruff-format.log | src、tests、scripts 静态检查 |
| document-checks.json | 新文档本地链接、运行源码及个人输入保持原摘要的核对 |
| SHA256SUMS | 最终审计报告与本目录记录的摘要；不包括本清单本身 |

本轮测试在已有隔离开发环境执行，确认 hwsol 导入自本项目 src，而非旧 wheel。以下命令从项目根目录运行；临时环境路径是本机执行证据，不是安装要求。

```bash
MPLCONFIGDIR=/tmp/hwsol-mpl HANDNOTE_CACHE=/tmp/hwsol-handnote-cache \
  HWSOL_MTH_PYTHON=/tmp/hwsol-mth-venv/bin/python \
  /tmp/hwsol-personal-round1-testenv/bin/python -m pytest -ra
MPLCONFIGDIR=/tmp/hwsol-mpl313 HANDNOTE_CACHE=/tmp/hwsol-handnote-cache \
  HWSOL_MTH_PYTHON=/tmp/hwsol-mth-venv/bin/python \
  /tmp/hwsol-v030-py313/bin/python -m pytest -ra
/tmp/hwsol-personal-round1-testenv/bin/ruff check src tests scripts
/tmp/hwsol-personal-round1-testenv/bin/ruff format --check src tests scripts
```

真实数据核查通过 ProfileStore 显式指向 profiles/realism-studio。重放从 out/realism/rl-size-feedback/candidate/manifest.json 开始，输出 out/audit-2026-11-18/five-replay；四模式选取同一 manifest 的第 6 题，policy 设为 null，分别渲染并 verify --pixels，保留在该输出根目录。再次执行需换新输出目录。

snapshot 包含本机路径、样本标识和摘要，以本地开发交接为目的。历史 out/ 产物仍留在原处。摘要用于识别文件变化，不是作者真实性签名。

从项目根目录执行 `sha256sum -c docs/evidence/audit-2026-11-18/SHA256SUMS` 可核对本轮报告和记录。
