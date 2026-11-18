# Profile 格式与存储（schema_version 1）

`src/hwsol/profiles/schema.py` 是 Pydantic 合约；未知字段拒绝，所有资产使用 SHA-256。
`bundled-default@v1` 位于包内，当前为明确标记的 OFL 测试字体；所有个人数据默认在包外。

## 不可变快照

每个 `profile.json` 包含：

| 字段 | 含义 |
|---|---|
| profile / version | 小写字母、数字、连字符的 Profile 名；版本为 v1、v2 等 |
| parent / base | 上一已发布版本及派生基底，固定 `profile@vN` 引用 |
| created_at / description | 构建时间、用途说明 |
| source_kind / distribution | `test-font`、`personal-assets` 或 `derived`；`OFL-1.1` 或 `local-only` |
| assets / font | 字体、许可证、来源文件的相对路径、角色与摘要；字体必须是已登记资产 |
| coverage / personal_codepoints | 字体总覆盖及参与个人构建的 Unicode 码点；未采集字符仍可来自基底 |
| variants | 十进制码点字符串到真实形状总数的映射，含首选字形；没有变体的字符不列出 |
| samples | 原始样本索引 ID、sample.json 摘要及 included；导出中不携带原始笔迹 |
| render | 经校验的 RenderConfig：纸张、尺寸、字号、种子等 |
| build | 后端与软件版本、构建 run_id、全量历史重建策略、首选和变体选择策略 |

首选和变体保存在字体 cmap 与 GSUB `rand` 中。Profile 是资产快照，非可变训练目录；
应用不覆盖已存在版本，每次加载验证资产。不能用手工编辑 JSON 代替新版本构建。
这是应用层不可变与摘要检测，不是操作系统防篡改或签名认证。

## 用户数据布局

```text
data/
  settings.json                        # default_profile 与各 Profile 的 active_versions
  profiles/name/
    definition.json                    # 创建时的 base 引用和快照摘要
    base/                              # 完整基底副本，避免依赖另一机器的原始样本
    candidates/vN/profile.json          # 未发布版本，Runtime 默认不可见
    versions/vN/profile.json            # 已发布版本及全部资产
    evaluations/vN.json                # 最新评估索引；历史评估仍保存在 training-runs
  samples/name/s-uuid/
    sample.json                        # 轮次、种类、质量/权重、原始资产摘要、处理状态
    original-000.json                   # 原始字节；扫描/字体保留相应后缀
    template.json                      # 如输入有模板则原样保存
    LICENSE.txt                        # 如用户提供则原样保存
  training-runs/name/r-uuid/run.json    # 本次输入快照、构建结果与错误；中间字体保留
  training-runs/name/e-uuid/            # 标准测试页、比较与评估报告
```

神经 checkpoint 尚未使用；不创建空占位目录。默认数据位置和 `HWSOL_DATA_DIR` 见 README。
新版本序号取已有候选/发布版本最大值加一；失败构建不消耗版本号，其独立 run_id 和失败报告保留。
parent 指构建时最后发布的版本；回滚仅改变激活指针，不删除版本，也不删历史样本。

## 写入和读取隔离

同一 Profile 只允许一个构建/导入/评估/发布写入者；其他写入立即报错，不等待。
样本和快照在同一文件系统暂存，完成后重命名；发布将 candidate 原子移动到 versions。
激活单独原子替换 settings.json。Runtime 只读选定快照，不取构建锁；训练失败或未发布不会改变默认版本。

进程异常终止可能留下 `.writer.lock` 或 `.settings.lock`。先读取锁中的 PID，确认对应写入进程已经结束，
再人工移除该锁文件并重试。不要在构建运行时删锁。未完成的 `.import-*`、`.snapshot-*` 和运行目录
不被当作已发布版本；原始样本、已发布版本不应作为清理目标。网络文件系统原子性未验证。

## 导出与导入

ZIP 仅含 profile.json 和登记资产，**不是训练数据备份**。不会包含样本原文件、checkpoint、作业或绝对路径。
完整 Studio 迁移应在写入者停止后备份整个数据目录。导出 ZIP 不含私人原始笔迹，但字体本身仍体现个人字迹，按个人数据管理。

导入验证相对路径、链接、重复条目、128 MiB / 256 文件上限、摘要和字体实际覆盖/变体元数据，
拒绝同名版本和覆盖 bundled-default。导入是对用户提供的已发布资产的显式信任，不提供作者身份认证。
导入后可直接渲染；继续本机采集时以导入快照为基底，旧样本只保留来源索引。

渲染报告记录最终 Profile 引用、快照和字体摘要、配置与依赖；相同快照/种子/已验证环境可复现产物。
重新训练会有新时间戳与版本标识，不承诺构建字体文件逐字节相同，也不承诺不同平台栅格化逐字节相同。
