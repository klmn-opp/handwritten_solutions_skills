# M2 Linux package readiness preflight

2026-11-20。**Linux CPU 安装与公开包隔离预检 Go；v1.0 / 生产发布仍 No-Go。**
没有发布、建 tag、改版本或依赖、训练、替换个人 Profile / latest。
包仍标示 `0.6.0`，是未发布的工作树候选，不是重新发行已有 v0.6.0。
实验数学 Resolver、个人字形及神经权重没有进入 wheel。

## 最终产物与验证

假设：普通 CPU 用户能在新的 Linux Python 3.12/3.13 环境安装当前 wheel，
使用 bundled-default@v1 完成已有五题及新内容；公开分发不需要私人研发资料。

最终三包及关闭证据：
[20261120T111049994793Z-m2-public-package-closure-2859920d](../workspace/runs/20261120T111049994793Z-m2-public-package-closure-2859920d/run.json)。
实际双 Python 六页出稿：
[20261120T110552414281Z-m2-linux-package-preflight-b84ef643](../workspace/runs/20261120T110552414281Z-m2-linux-package-preflight-b84ef643/run.json)。
原始构建拒绝、editable 安装和 stage CLI 证据：
[20261120T110550529422Z-m2-pep517-mode-verification-a64dfd93](../workspace/runs/20261120T110550529422Z-m2-pep517-mode-verification-a64dfd93/run.json)。

| 检查 | 实际结果 |
| --- | --- |
| Linux Python 3.12.16 / 3.13.18 | 各自新临时环境、离线锁定依赖、安装 wheel；doctor、旧五题及新 ODE 题通过 |
| Runtime 隔离 | 从临时目录运行实际安装包；Torch、pytest、research 均不可导入 |
| 字迹与身份 | bundled-default@v1；唯一字体为 Patrick Hand/OFL，不是作者笔迹；旧题号为 2/4/6/7/9，新题为 New 1 |
| PDF / PNG | 每环境 5+1 页，verify 通过；六对 PNG 跨 Python 逐字节一致，并与已目视审阅的 c65b99a1 六页一致 |
| 最终包绑定 | 最终 wheel 全部 70 个成员字节与 b84ef643 已验 wheel 相同；末次仅重建包、新装两环境 doctor，未重复渲染 |
| 公开文件 | Skill / sdist 的实际文件、receipt、SHA、overlay 和所有真实 Markdown 相对链接通过；声明扫描范围内未发现私人资产或身份路径 |
| 原始构建 | 仓库根目录裸 uv build 实际拒绝，没有 wheel/sdist；普通 wheel metadata hook 同样先拒绝 |
| Editable | 独立新环境 uv sync --offline --locked --no-dev 通过；direct_url 为 editable，doctor 通过且无 Torch |
| 污染构建 | 两种未登记文件分别实际触发拒绝，无 wheel，路径见下 |
| sdist 再构建 | 解压最终 sdist 后构建 wheel，所有成员与最终 wheel 字节相同 |
| macOS / Windows | 本机未执行，不声明原生验证；归档链接使用 POSIX 路径，生成配置允许严格等价 LF/CRLF |

新题是 Agent 编写的 `x'=-2x+6, x(0)=1`，包含解与校验、`x(log 2)`、积分及稳定性。
ODE、初值、端点与积分分别符号重算；这是新文本 render-only 安装练习，不是作者数据、
问题识别准确性或风格盲评。Agent 检查六页未确认漏字、错题号或裁切。

[五题 PDF](../workspace/runs/20261120T110552414281Z-m2-linux-package-preflight-b84ef643/rendered/python-3.12/five-existing/homework_solutions.pdf)、
[新题 PDF](../workspace/runs/20261120T110552414281Z-m2-linux-package-preflight-b84ef643/rendered/python-3.12/new-text/homework_solutions.pdf)、
[新题 PNG](../workspace/runs/20261120T110552414281Z-m2-linux-package-preflight-b84ef643/rendered/python-3.12/new-text/pages/01.png)。

## 分发修复与保留失败

本地原 README/SKILL、用户 master plan、私人报告和历史数据保持原样。
[公共清单](../public_docs/distribution.json) 只定义分发快照：CPU 源码、许可证、唯一 bundled
字体及必要 Profile 资产、benchmark 摘要、Job schema、示例、CPU 测试、公开指南。
公共层不复制 Runtime 实现。入口 overlay 必须来自所选 ref 自身；旧 ref 缺公共层就拒绝。
已实际调用旧 v0.6.0 验证拒绝，没有向旧 tag 借用当前指南。默认不创建 Git bundle。

PEP 517 wrapper 在 wheel/sdist 的依赖查询、metadata 和构建 hook 先验证 receipt、
policy、逐文件 SHA 与实际清单；editable 单独委托旧后端。CI 先 stage 再 build。
root build 残留一律拒绝，要求新快照；源码中叫 build/dist 的目录不能跳审。仅容许导入
产生的 __pycache__/*.pyc，symlink 先拒绝。doctor 所述 docs/PHASE0.md 使用新公开兼容说明
覆盖，未引入原私人报告。入口模板明确为 .md.in，所有实际分发 .md 的链接均检查。

| 保留 run | 问题与修复前源码定位 |
| --- | --- |
| 20261120T104451429742Z-m2-linux-package-preflight-1f4c2376 | 安装成功但 sdist 20 份文档含本机 home 路径；logs/source 为原始构建快照，无公开资格 |
| 20261120T105314538009Z-m2-linux-package-preflight-5eae7de5 | allowlist 漏 METADATA.pb，真实 doctor 失败；同时补齐 benchmark SHA256SUMS；未填充占位 PDF |
| 20261120T105403598186Z-m2-linux-package-preflight-c65b99a1 | 六页通过，独立审查却发现模板原位置 12 处死链；logs/source 保留旧 .md，不沿用当时全链接结论 |
| 20261120T105950323821Z-m2-public-documentation-rebuild-9c67cda8 | 模板 .md.in 修复后链接通过，但裸 PEP 517 仍可绕过 stage；logs/private-input-snapshot 保留修复前源码 |
| 20261120T110552414281Z-m2-linux-package-preflight-b84ef643 | 初版 guard/出稿通过，独立审查实际证明按目录名忽略 build 的夹带漏洞；logs/private-input-snapshot 保留旧 guard；临时污染包已清理，没有伪称留存二进制 |
| 20261120T111025532874Z-m2-public-package-closure-ff9220ae | 修复后的两污染拒绝和正常构建成功，但辅助 driver 把 uv 自动 .gitignore 误计为第 4 包而断言失败；原样保留 |
| 20261120T111049994793Z-m2-public-package-closure-2859920d | 新 run 修正产物筛选；三包、两污染拒绝、sdist 重建、双 Python doctor 均通过，发布仍 held |

缓存复现：stage 到新目录，写入 `build/lib/hwsol/review_sentinel.py` 或
`src/hwsol/build/review_nested.py`，再执行 `uv build --offline --wheel --out-dir <独立目录>`。
旧版可夹带；最终 run 的 logs/source.py、两个污染快照、logs/commands.json 和
reports/build-pollution.json 保留修复后的真实拒绝证据。独立 Agent 已复核并关闭 P1/P2。

## 来源、命令与边界

最终基于 commit `ad9e41ef0f6204efd539b75ffea53452da49bc98` 的未提交包装改动。
454 个输入文件摘要、policy/receipt、overlay、源码快照与命令日志均在最终 run。
_build.json 中的 commit 是父源码标识，不能代替工作树完整文件 SHA。
锁定依赖、版本、原 README/SKILL、master plan 和正式 latest 均保持原 SHA。

公开构建入口（使用新目录，不发布）：

```sh
python scripts/package_release.py --stage /tmp/hwsol-public-unique
cd /tmp/hwsol-public-unique
uv build --offline --out-dir /tmp/hwsol-artifacts-unique
```

实际验证入口与完整参数保存在日志：

```sh
MPLCONFIGDIR=/tmp/hwsol-package-preflight-mpl .venv/bin/python -m research.package_readiness
.venv/bin/python -m pytest -q tests/test_public_distribution.py
.venv/bin/ruff check public_build_backend.py scripts/package_release.py research/package_readiness.py tests/test_public_distribution.py
```

18 项针对性包装测试与 Ruff 通过。公开指南另核对了 20 条实际 CLI help 入口，示例通过
Job schema。数量只表示边界回归范围，不是笔迹质量进步证据。
最终三包位于最终 run 的 reports/dist/：

| 产物 | SHA-256 |
| --- | --- |
| wheel，70 条目 | b82bc8592e964159ab45bfdada2bd750ec7c8a2a21b72005372d1fa28f391b23 |
| sdist，138 条目 | 6e3067a583732cd24b4f3dbe7646d9ad6fb9aa8db17712be1d1b3352b5c90807 |
| Skill ZIP，130 条目 | 1809667f68e20fce1f07bdc66ce047403c022cac12f0f7fa00075a102bcb5fd0 |

完整 M2 仍需 macOS 实际验证、作者页面验收、正式候选准入和明确发布批准。
个人数学、上下文神经生成、Profile 发布各有独立门禁，本轮 CPU 包不含这些研究成果。
隐私扫描是声明的文件、路径与资产核验，不是任意个人信息或许可的完备证明；新增公共文件
必须重审。本文是本地研发记录，不进入公共包。
