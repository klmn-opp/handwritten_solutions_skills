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
