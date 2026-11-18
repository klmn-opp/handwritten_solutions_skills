# 扩充真实语料后的两轮续训

用户反馈上一版5页对照中Handnote最左列仍最好，授权Sol清理更多真实样本并至少再训练两轮。
在`experiment/expanded-corpus-two-rounds-v09`承接`bd103a0`，不改默认Profile/latest。

## 训练前协议

两名Sol xhigh分别负责微积分与物理/计算机Train文件；另一个Sol负责AS Validation和phy1 Holdout。
按原文档固定分组，避开已用留出页。每份证据独立run封存，300 DPI原色图不描画、不修补。
有遮挡、涂改、相邻笔画侵入、作者或标签不确定的候选弃用。主Agent机械核对原PDF重渲染、
裁剪像素、SHA、文档分组、标签策略和跨分辨率重复，不重复逐项识别Sol已经确定的标签。

新增300 DPI视图以独立数据集版本追加；不会覆盖Studio已有150 DPI页面和人工审核。
每个版本保留旧样本、标签、数据分组与完整来源，记录渲染到PDF坐标的变换。
字体178个辅助轮廓和75个类别参考保持不变；字体派生样本不计入真实样本数。
新Holdout不导入任何训练/验证版本，冻结所有候选后才能读取。

第一轮使用先到达的微积分Train批+新Validation，第二轮追加物理/计算机Train批。
两轮均计划8000次真实CUDA AdamW更新、学习率1e-4、batch32，其中30字体/2真实样本，
字体类均匀取样、真实样本按原数量取样；种子2001/2002。保留现有网络和损失，以辨别数据补充效果。
每250步记录验证，从1000步起按既有先验采样代理保存best。第二轮继承第一轮best及优化器，
每阶段重新设定固定随机种子。最终权重祖先步数与全部实验计算量分别报告。

扩展数据续训必须显式`--allow-data-extension`：校验新数据来自checkpoint的数据父版本，
旧图片/标签/参考像素均保留，错误父版本或未经证明的数据替换均拒绝。
采用原600图固定采样，Handnote、上一版、第一轮和第二轮全部保存，不只展示好看的随机样本。
两轮完成后先冻结权重和选择，再打开新Holdout；内容/风格验收不过就保持held。

## 命令

```bash
PYTHONPATH=.:src MPLCONFIGDIR=/tmp/hwsol-mpl .venv/bin/python -m research.corpus.dataset_extension \
  PARENT_DATA --train-package TRAIN_RUN --validation-package VALIDATION_RUN
PYTHONPATH=.:src .venv/bin/python -m research.corpus.reference_study train NEW_RUN/reports/data \
  --resume PARENT_TRAINING --steps 8000 --learning-rate 0.0001 --seed 2001 \
  --real-per-batch 2 --real-sampling proportional --all-variant-selection --allow-data-extension
```

第二轮parent改为第一轮数据/权重，追加另一Train批，不重复加入Validation包，seed2002。
所有依赖继续隔离；没有OCR/字体构建冒充神经训练，也没有上传私有PDF。
字体依旧是作者认可的效果基准，训练步数增加不自动意味着质量提升。

## 开发集触发的有界第三轮修正（打开新Holdout之前）

两轮固定600图断裂代理为16/20，上一版6；新Validation形状距离却更低。这是质量维度冲突，
不能凭距离降低宣称输出更好。额外4000步，seed2003，沿用第二轮best与同一数据版本/优化器，
新增权重0.5的prior-support项：直接采样条件先验，约束预测距离场接近同类别完整Train视图。
支持库只含字体辅助和真实Train，Validation/Holdout不能进入；原后验重建与KL仍保留。
不直接替换或复制像素，网络参数承担全部生成。该约束可能增加记忆或减少变化，必须同步报告。
所有参数在本轮开始前固定；第三轮后统一冻结旧版和三轮候选，才打开新Holdout。


## 实际执行结果（2026-11-19，北京时间）

已完成两轮各8000次CUDA优化，另完成4000次有界断笔修正；没有只启动后台任务就交付。
主交付：`workspace/runs/20261118T174726751468Z-expanded-corpus-two-rounds-58bbf5d2/`。
[新5页对照PDF](../workspace/runs/20261118T174726751468Z-expanded-corpus-two-rounds-58bbf5d2/rendered/homework_solutions.pdf)
包含全部75类，列顺序为Handnote、上一版、第一轮两种固定采样、第二轮两种固定采样、修正轮两种固定采样。
全部方法相同白色背景、参考尺度；固定sample0/4，不按好看程度挑图。
左列为现有Handnote个人轮廓导出的单字参考，不是另一个训练模型，也不是整页Handnote排版性能测试。

结论：**补数据与真实续训完成；有限生成研究Go，生产替换、连续英文与RL仍No-Go。**
作者认可Handnote左列最好，本轮没有足够证据推翻这一点。新候选均held，未宣布胜出者。
没有变更font-cpu、personal@v1、正式latest，未合并main或创建生产发布标签。

### 真实数据增长与边界

| 数据 | 上一版 | 第一轮 | 第二轮/修正轮 |
|---|---:|---:|---:|
| 真实Train裁剪 | 41 | 207 | 339 |
| 真实Train字符种类 | 4 | 42 | 52 |
| 训练选择用Validation | 13 | 71 | 71 |
| 个人字体派生辅助视图 | 178/75类 | 178/75类 | 178/75类 |

两名gpt-6.2-sol xhigh直接清洗并准入298张新真实Train：微积分166张/13页，物理与计算机132张/13页。
微积分批明确记录67个弃用候选；各批保留拒绝原因与裁剪调整证据，不能把它当作全语料统一拒绝率。
遮挡、交叉笔画无法归属、涂改、混合符号与不确定标签继续弃用，不描画补笔、不以正确答案改写实际字迹。
计算机笔记中的红色连续个人笔迹有作者来源判断记录；并非把所有红笔视为本人或老师。
标签标记model-accepted，human_verified=false；主Agent验证原PDF、坐标、像素、摘要和分组，没有再次逐项识别。

新增58张Validation覆盖40类；加旧13张为71张，参与checkpoint选择，不能冒充独立测试。
另57张/38类新Holdout来自固定留出文档phy1未用页；所有权重冻结后才打开，未进入任何训练版本。
同源文档分组不变；新页不等于新增独立作者或文档。47页跨split近重复dHash筛查无命中，
这只是页级近重复代理，不能证明不存在任何局部重复。原裁剪跨150/300DPI按PDF物理坐标检测重复。

仍只有字体辅助、缺少真实作业Train的23类：
`, - . / G H N O P Q R U V W X Y i j q z σ √ ⋅`。
52类有真实样本不代表每类样本充分；部分仅1–3张。既有保守数据就绪门槛仅字符1满足，
本轮75类是带字体锚点的研究实验，不绕过生产数据门槛。
最终数据集含450条各层级记录（包括旧人工整行/公式等），不能称为450张参与当前单字训练的图。
可训练视图为96×96；历史通用readiness报告中的64×64 task字段指旧原型，不是本轮架构。

新增300DPI视图独立版本化，不覆盖Studio已有150DPI页面、审核历史或人工修正。
因此它们不自动出现在旧Studio队列；已获授权的Sol准入无需用户重新逐条确认。
每个sample可反查原PDF摘要、原页、像素bbox、PDF坐标、原色裁剪、清理Agent和策略证据。
数据扩展证明要求旧标签/全部原记录/旧Train、Validation与字体像素完全保留；错误父版本续训会拒绝。

### 真实优化与复现

网络仍是579449参数`reference-sdf-cvae/1`，96×96离线墨迹距离场输出；并未恢复真实在线笔顺。
继续学习参数和优化器状态，全部75类参与；batch32含30字体辅助/2真实样本，真实样本按数量取样。
这是保守的字体锚定协议，真实样本仅占6.25%的每批输入，仍可能限制模型吸收新变化；没有证据只增加步数即可解决。
三阶段均lr=1e-4，训练seed2001/2002/2003，评估seed2909；每250步记录开发指标。

| 阶段 | 更新数 | 更新后的best步 | 训练秒 | 峰值PyTorch分配MiB | 参数绝对变化量 |
|---|---:|---:|---:|---:|---:|
| round1 | 8000 | 1000 | 174.76 | 441.8 | 2398.89 |
| round2 | 8000 | 2500 | 174.23 | 451.5 | 2165.64 |
| repair | 4000 | 1500 | 167.75 | 938.4 | 1460.01 |

本轮总计20000次更新、训练循环约516.74秒（含内部验证，不含数据清理/独立评估时间）。
最终修正候选的权重祖先为旧4500+第一轮1000+第二轮2500+修正1500=9500步，不能写成继承全部20000步。
best是在本阶段第1000步起的已更新checkpoint中选择，不代表一定比step0或上一模型更好。
每阶段重新固定随机种子，继承优化器但不声称逐指令无缝恢复同一训练进程。
初始、best、last、优化器、验证记录、曲线、配置、参数变化和checkpoint SHA全部保留。

设备RTX5060 Laptop；隔离环境PyTorch2.7.1+cu128/CUDA12.8，驱动580.126.09。
PyTorch报告设备总内存8080195584字节，nvidia-smi报告8151MiB，分配峰值不是整卡占用。
生产.venv没有Torch。各run的build、reports/environment.json、request.json保存固定版本和来源；
主交付reports/training-runs.json/candidates.json汇总全部SHA与配置。

### 冻结后的独立评估

| 方法 | 新Validation形状距离↓ | 新Holdout形状距离↓ | 600图断裂代理 | 二值轮廓同字体 | 变化偏小的类 |
|---|---:|---:|---:|---:|---|
| Handnote可用变体最优 | 0.145984 | 0.135318 | — | — | — |
| 非学习几何8选1 | 0.138132 | 0.128807 | — | — | — |
| previous | 0.145698 | 0.133715 | 6 | 11 | C,m |
| round1 | 0.137594 | 0.129522 | 16 | 9 | 无 |
| round2 | 0.137307 | 0.127473 | 20 | 11 | C |
| repair | 0.146862 | 0.128465 | 11 | 14 | C,H,a,e,g,k,m,u |

各神经模型600图均无空白或越界。距离在均匀尺度对齐后计算，神经/几何为8选1，字体只有可用的2–4变体；
这是对真实轮廓的覆盖代理，候选预算不同，不能转换成内容正确率或自然度。
第二轮Holdout距离比上一版约低4.7%、略低于几何基线，但断裂代理从6升到20。
修正轮将断裂代理降到11，仍高于上一版；8类变化过小，二值同字体14张，存在更强记忆/变化坍缩风险。
修正轮Validation距离也退步，不能只摘取有利指标宣称整体提升。
二值同形只是统计阈值下相同，不等于实现直接复制；像素不同同样不证明新的自然字形。

冻结run先于所有新Holdout调用。冻结后只采样重放、评估和报告，没有再训练或按Holdout调参换模型。
修正轮同模型/设备/种子600张真实GPU生成PNG逐字节重放相同；这是复现证据，不是质量证据。

### 独立盲转写

375张：五种方法各75类固定sample0，打乱编号，不向两名Sol评审提供目标字符或方法。
两评审分别审188/187张，分组不同且弃答习惯不同，没有双人一致性测量；结果只能作描述性观察。

| 方法 | 清晰 | 歧义 | 缺损 | 弃答 | 非空转写 | 与目标一致 | 非空不匹配 |
|---|---:|---:|---:|---:|---:|---:|---:|
| handnote | 57 | 17 | 1 | 17 | 58 | 56 | 2 |
| previous | 49 | 19 | 7 | 23 | 52 | 49 | 3 |
| round1 | 48 | 17 | 10 | 23 | 52 | 50 | 2 |
| round2 | 52 | 16 | 7 | 20 | 55 | 53 | 2 |
| repair | 55 | 18 | 2 | 18 | 57 | 55 | 2 |

缺损与转写非空可同时成立，故列值不应交叉相加。弃答不算真实错字。
Handnote也出现5→J、o→σ；上一版另有√→T，新三轮均有5→J、√→T。
这些是模型对孤立字形的判断，不是作者确认的错误；上下文、大小写和字点位置也会影响识别。
修正轮75图缺损标记2比上一版7少、接近Handnote1，但仍不足以声称自然度或个人相似性胜出。
原始转写、备选、置信度、不可见的目标映射、逐图摘要全部保留，不重新解释为人工真值。

`comparisons/content-failures.png`并列展示5、√、o、+、G的固定输出：修正轮加号完整性改善，
5和o的孤立歧义也存在于个人参考；G的小分离墨点沿用参考。未擅自纠正作者字形或改Profile。
`comparisons/font-variant-audit.png`保留源轮廓核对，不能将所有生成断笔归因于输入源有问题。

### 分维度验收

| 维度 | 本轮结论 |
|---|---|
| 内容正确性 | 独立Sol盲检仍有歧义/不匹配；没有人工字符错误率，不能放行生产 |
| 单词尺度与连笔 | 仅单字符实验，未训练或验证词/行与连笔 |
| 数学符号覆盖/可读性 | 75类只含有限数学符号；√仍盲检不匹配，未验收完整公式 |
| 作者风格相似性 | 本轮新候选尚无作者评分；保留作者对上一版Handnote最佳的判断 |
| 页面布局自然度 | 5页为研究字表，不是新神经数学作业；既有CPU作业系统未改 |
| 工程一致性 | 完整200测试、额外3项Torch损失/梯度检查、600PNG重放、23子run封存验证通过 |

### 验证、提交与交付路径

实际完整命令：

```bash
HWSOL_MTH_PYTHON=/tmp/hwsol-mth-venv/bin/python PYTHONPATH=.:src .venv/bin/pytest -o addopts='' -q
```

200 passed，最终日志在主交付logs/full-tests-final.log；不是200个视觉或模型质量测试。
额外3项可选Torch验证：最近同类支持距离、朝完整墨迹的梯度、精确支持零损失；不是3次训练。
Ruff/格式/diff检查通过。14原件、Corpus审核快照、5项受保护资产SHA保持一致，CPU源码/依赖diff为空。
23个本轮子run（训练、数据、评估等）完整性通过；主run以held封存，正式latest不动。

分支`experiment/expanded-corpus-two-rounds-v09`；源码阶段：
`4e83fd8`数据扩展与续训证明；`8fddb9f`绑定分组来源、允许同一Validation精确重放；
`2e6de84`可关闭prior-support约束（默认0，回退原损失）。最终文档提交记录在主run的final-source.json。

以下均相对于`workspace/runs/`，各run保留run.json、solutions/、rendered/pages/、reports/、comparisons/、logs/；
不产生PDF的数据/训练run在result中说明用途，不用空白PDF伪造交付。

| 用途 | run ID |
|---|---|
| 主交付/5页PDF/总报告 | `20261118T174726751468Z-expanded-corpus-two-rounds-58bbf5d2` |
| 微积分Train清洗 | `20261118T174609871113Z-cal-train-cleaning-300dpi-b4d5b78a` |
| 物理/计算机Train清洗 | `20261118T174508418216Z-train-phy2-ics-glyphs-300dpi-5d56440c` |
| 新Validation封存 | `20261118T175110517017Z-multichar-validation-v2-seal-041a159a` |
| 新Holdout封存 | `20261118T175430545693Z-multichar-holdout-v2-seal-620a64f9` |
| 第一轮数据 | `20261118T175344677211Z-expanded-real-glyph-data-92b34f5f` |
| 第二轮/最终数据 | `20261118T180020607941Z-expanded-real-glyph-data-fe7d5df4` |
| 第一轮训练 | `20261118T175506422522Z-all-personal-reference-training-b28d1bca` |
| 第二轮训练 | `20261118T180201116727Z-all-personal-reference-training-80cd482a` |
| 修正轮训练 | `20261118T180825097314Z-all-personal-reference-training-6922e82c` |
| 第一轮600图 | `20261118T175837985359Z-all-character-sampling-836d8353` |
| 第二轮600图 | `20261118T180518515242Z-all-character-sampling-5af66fcd` |
| 修正轮600图 | `20261118T181154952573Z-all-character-sampling-bbc1de5e` |
| 冻结全部候选 | `20261118T181321465091Z-all-character-candidate-freeze-e657b06f` |
| 375图盲转写 | `20261118T181345344901Z-expanded-data-blind-content-7831ac94` |
| 修正轮真实重放 | `20261118T181754134470Z-all-character-sampling-4b0e2dfc` |

数据版本入口`reports/dataset/manifest.json`及`reports/data/manifest.json`；权重入口`logs/best-checkpoint.pt`，
last与优化器也保留。主交付reports/comparison-summary.json给出全部Validation/Holdout run绝对路径；
character-acceptance.csv逐类列出真实覆盖/输出异常，candidates.json给出权重摘要，final-integrity.json记录保护检查。
rendered/pages/comparison-01.png至05.png为逐页预览；comparisons/另含各模型全字表、原全采样PDF和训练曲线。

### 下一里程碑

优先补23个无真实Train类及其余低覆盖类，继续从原PDF保守找完整笔迹；不是重新要求用户抄整本字母表。
针对断笔与同形歧义，引入可靠上下文/结构检查和多样性约束的小规模受控对照，再决定扩大真实样本权重。
目前模型输出仍强依赖个人字体参考；完整性约束能改善部分笔画，却可能把随机变化压回字体轮廓。
应同时量化身份、完整性、变化与记忆，不能靠增加同一损失的迭代次数解决所有问题。
下一次独立验收另封存未查看页面；本次57张已打开测试不得用于调参后再报新泛化成绩。
词/行生成需对应真实词行监督后另立阶段；完整数学结构继续由Handnote/MathText负责。
当前不做RL/从零预训练/自动发布；不继续DiffBrush、DiffusionPen、DiffInk比较。
