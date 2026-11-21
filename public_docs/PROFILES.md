# Personal Profiles and CPU Studio

Clean installations use **`author-default@v1`**, the author's reviewed CPU font
and mathematical outlines. Existing explicitly selected defaults are preserved.
The licensed Patrick Hand engineering font remains available as the optional
`bundled-default@v1`; it is not the author's handwriting.
Use `hwsol profiles list` to inspect installed versions.
Use an explicit `profile@version` to reproduce a specific version, or omit the
Job's `profile` field to use the selected default. A Profile imported from your
own scans stays local unless you choose to export and share it.

Studio can build versioned fonts from user-provided templates, supported scans,
fonts, trajectories and calibration annotations. This deterministic font and
geometry construction is not neural-network training. Read each command's
`--help` for the supported input fields; retain the actual written label and
source, and do not treat an OCR suggestion or reference answer as ground truth.

## 从零建立自己的 Profile：三步

本代支持在 CPU 上把真实笔迹提取为个人字形、合并变体并生成预览。
命令中的 `train` 是历史名称；这里没有训练神经网络，也不会自动学习连笔或整页习惯。

**1. 生成并填写采集表。**

```sh
hwsol profile template --output out/my-handwriting.pdf --sets 2
```

会同时生成 `out/my-handwriting.json`，请保留。默认采集表包含 202 个英文大小写、
数字、标点、希腊字母和数学符号，每套 3 页；两套共 6 页。用黑笔书写，保留四角
定位标记，以约 300 DPI 扫描为一个 PDF，或按纸面顺序保存为 PNG/JPEG。
不要把未填写的打印样张作为个人笔迹导入。

这是内置 Handnote 字符网格的基础导入路径。项目已有的 Session 提示—作答采集册、
自然句子、真实公式和历史作业继续保留；它们需要相应的切分与标签审核，不能直接当作
这一网格模板导入。该命令也不是任意作业 PDF 的自动识字训练器。

**2. 核对实际笔迹，再用一条命令构建与预览。**

检查每个格子里真正写出的字是否与提示一致；有写错、越格、模糊的内容先重新采集。
`--labels-reviewed` 表示你或你的 Agent 已做了这项核对，打印提示本身不算标签验证。

```sh
hwsol profile build --profile my-handwriting \
  --template out/my-handwriting.json --source /path/to/filled-scans.pdf \
  --labels-reviewed
```

图片输入按页重复 `--source /path/to/page-01.jpg --source /path/to/page-02.jpg`。
命令会保留原始文件与哈希，导入扫描，累积构建新候选，并执行工程检查。
结果打印 `candidate`（例如 `my-handwriting@v1`）、`preview_pdf`、`preview_pages`
以及缺失个人字形的 `requested_without_personal_ink`。打开 PDF/PNG 检查字形与数学
可读性；尚未采集的字符可能仍来自基础字体或明确记录的数学字体回退，不能把它们
算作个人覆盖。候选保持 `held`，表示待你审阅，不会自动改变当前默认。
新建基础字体 Profile 使用 Patrick Hand 作为补缺基础；只有实际采集到的字形才会
被标为你的个人字形，这与发行版内置作者 Profile 的选择是两件独立的事。

**3. 看过预览后，将返回的版本设为自己的默认。**

```sh
hwsol profile publish --profile my-handwriting --version v1 --activate
```

将 `v1` 替换为上一步实际返回的版本。这里的 publish 只发布到本机 Profile 资料库，
不会上传网络。之后新 Job 不填写 `profile` 字段即可使用它；如果 Job 仍显式指定
`bundled-default@v1`，它就会继续使用该固定版本。

补充笔迹时，重新生成一个文件名不同的采集表，填写后再次运行第二步。
旧原件、版本和失败记录保留，累计构建产生 `v2`、`v3` 等新版本。
新版本的预览审核和发布步骤相同，失败不会替换已生效版本。

## 细分操作与版本管理

需要单独控制各阶段时，原有命令仍可使用：

```sh
hwsol profile import --profile my-handwriting --source /path/to/page.jpg --template out/letters.json
hwsol profile train --profile my-handwriting --append
hwsol profile evaluate --profile my-handwriting --version v2
hwsol profile publish --profile my-handwriting --version v2
hwsol profiles activate my-handwriting@v2
```

低层 `import` 不会替你核验实际字符身份，先完成与上述相同的标签审核。

Inspect the candidate before publishing or activating it. A publish does not
implicitly authorize replacing a user's production default. New imports and
append builds create new versions; keep original samples, existing labels and
old versions. The separate `hwsol-studio` entry point exposes the same Studio.

```sh
hwsol profiles compare my-handwriting@v1 my-handwriting@v2
hwsol profiles rollback my-handwriting@v1
hwsol profiles export my-handwriting@v2 --output out/profile-v2.zip
hwsol profiles import --source /path/to/profile-v2.zip
```

Runtime exports contain the version's declared assets and integrity metadata;
they are not a backup of all raw training history. Keep a separate private data
backup. Export does not grant permission to redistribute scans, fonts or weights.
Unspecified local-only sample licenses must not be assumed public.

For a safe engineering demonstration with synthetic, nonpersonal trajectories,
run `python scripts/phase1_demo.py --output out/new-demo`. Calibration format is
illustrated by [the synthetic example](../examples/realism/calibration.json).
Do not claim inferred contours contain observed pen timing or pressure. The
system's structural math layout remains separate from personalized ink.

An explicitly audited personal ink recipe can be kept as a private CPU candidate.
Use `hwsol profiles export name@vN --candidate --output out/candidate.zip` to transfer
it locally. Import detects the recipe and installs it in the candidate area, without
activation. The archive includes original crop evidence and source coordinates;
keep it private. A compatible, reviewed recipe must already exist: importing a new
scan does not automatically establish its labels, mathematical roles or geometry.
This special ink-recipe lifecycle is separate from the basic scan-to-font workflow
above; basic font Profiles do not require cross-machine ink qualification bundles.
The [Job contract](../references/jobs.md#reviewed-personal-cpu-candidate) describes
explicit selection and rejection behavior. Existing published font Profiles and
their explicitly selected defaults remain available.
