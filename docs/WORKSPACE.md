# Workspace contract v1

代码与非私密基准在 Git；运行数据在 `HWSOL_WORKSPACE`（默认当前目录的 `workspace/`）。所有新的 CLI 渲染、Studio 构建/评估/测量、lab 实验都有不可复用的运行 ID 和最终摘要。历史内部 Python API 保持兼容，新的实验编排必须通过 workspace API。

```text
workspace/
  workspace.json                 # 可指向已有 Profile 库
  latest.json                    # 最近通过 PDF/PNG 工程核验的作业
  runs/<run_id>/
    run.json                     # 代码/依赖/配置/种子/Profile/checkpoint/硬件/状态/摘要
    solutions/
    rendered/homework_solutions.pdf
    rendered/pages/
    reports/
    comparisons/
    logs/
  archive/                       # 历史文件原位索引；不删除原件
  datasets/
  checkpoints/
  environments/
  sources/
```

失败、准备输入或纯测量实验可能没有 PDF，必须报告原因，不生成空白 PDF 冒充完成。以上目录仍会建立。ID 不能重用；终态运行不能通过 API 改写；`workspace verify` 检查文件集合和摘要。崩溃留下 running 记录，重试使用新 ID。摘要提供损坏检测，不是防恶意重写签名。

```bash
hwsol workspace init --path workspace --profile-store "$PWD/profiles/realism-studio"
export HWSOL_WORKSPACE="$PWD/workspace"
hwsol render --manifest examples/phase2/mixed/job.json
hwsol workspace latest
hwsol verify --job workspace/runs/<run_id> --pixels
hwsol workspace archive --source profiles --source samples --source out
```

`HWSOL_DATA_DIR` 仍有最高优先级。显式初始化可连接旧库，不改默认 Profile；新库默认位于 workspace/data。旧 `render --output` 继续产生兼容导出，规范产物始终在 runs。失败兼容导出只有失败报告。旧 lab 的多步 authoring 目录仍可使用；每次调用的输出和结果分别快照到新运行，历史调用不改写。采集模板、导入源、发布版 Profile 属于独立资产库，不是实验目录。

latest.json 只在整个作业独立重验通过后原子更新；失败、holdout 基准、模型试验和仅工程接口测试不会更新它。指针含 run.json/PDF 摘要。它不代表作者已接受风格。

本次先将已验证 v0.4.0 合入本地 main，开发在 feature/workspace-benchmark-v06；experiment/diffink-cuda-v07 保留独立研究分支。无远端发布。旧 v0.4.0 标签未移动。历史数据先以 workspace/archive/pre-v06.json 固定摘要，既有目录保持原位。已有历史提交的二进制不改写 Git 历史。

## DVC 评估

DVC 可用 Git 内的小型 `.dvc` 元数据跟踪外部大文件，并使用本地/私有远端存储；参见[官方文件格式](https://doc.dvc.org/user-guide/project-structure/dvc-files)和[存储说明](https://doc.dvc.org/user-guide/data-management/remote-storage)。本轮选择先使用 SHA-256 资产清单和不可变目录，未安装或上传 DVC：还没有完成 CUDA 复现，也没有新增个人在线训练集，部署远端没有实际收益。

以后可以在独立私有数据仓库中 `dvc add datasets/<version> checkpoints/<version>`，元数据和锁文件进 Git，数据进受控存储。运行目录不整体进入 DVC，避免重复缓存 PDF；不得把个人扫描上传公共远端。DVC 不是必需运行依赖。
