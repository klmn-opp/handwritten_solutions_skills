# 全部个人字符联合训练

本实验承接 `9200793`，在独立分支 `experiment/all-personal-glyphs-v09` 实施。
用户授权全字符实验候选，不代表发布或更改默认 Profile。

## 冻结方法

原有个人字库覆盖75字符、178轮廓。它们是字体派生监督，不能计作新增真实扫描。
旧辅助导出误排句号和逗号，本轮在独立96像素视图中保留这4个自然小轮廓。
按原始字符尺度/基线渲染；必要时整体等比适配，不裁掉下伸结构。
真实训练仍仅使用通过既有来源与标签准入的单字符裁剪，原文档分组不变。
允许稀疏类别进入本轮实验；旧20/5/5门槛未达标仍须报告，不能宣称生产合格。

旧CVAE只依赖标签和潜变量，非目标字形失败。新 `reference-sdf-cvae/1` 接收
真实个人字体的基准轮廓、字符标签及随机变量，卷积解码器预测完整距离场。
没有直接像素复制、轮廓选择或坐标抖动的输出路径。参考特征经学习网络处理，
后验和参考条件先验通过KL约束联训，距离场到墨迹的边缘规则在训练/推理一致。
这是新小模型的参数训练，不能称为不兼容旧CVAE的直接续训；仅支持单作者离线单字。

初始阶段4000次AdamW更新，32批量，28个按类别均衡字体样本+4个按类别均衡真实样本。
精确字符表在每次数据run的manifest和coverage中。
每250次记录开发集诊断，1000次后按冻结代理评分保存最优模型。
模板匹配仅用于开发诊断，不是独立识字正确率。允许根据Train/Validation失败继续有界训练；
新Holdout在候选冻结后才打开，不能用于调参或选择权重。

## 复现入口

CPU协调器：`PYTHONPATH=.:src .venv/bin/python -m research.corpus.reference_study prepare DATASET`

CUDA训练：`PYTHONPATH=.:src .venv/bin/python -m research.corpus.reference_study train DATA_RUN/reports/data`

可选 `--steps 4000 --seed 1909 --resume PRIOR_RUN --learning-rate 0.0001`。
resume保留已学习权重和优化器，单独run记录来源摘要；重新固定采样种子，
不是原始进程逐位续跑。PyTorch只在独立环境导入，生产Runtime没有新增依赖。

所有数据、checkpoint、曲线、预览均在不可复用run中；不修改latest指针。
最后必须报告每字符来源、失败、多样性、记忆及独立样本覆盖，不能仅凭全类出图宣布成功。
