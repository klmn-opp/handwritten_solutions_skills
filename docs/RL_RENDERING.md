# CPU 渲染策略强化学习

当前开发版新增 `hwsol lab rl`。它训练已有渲染策略的选择概率，使用同一码点的真实字形变体、笔宽偏好、相邻笔宽连续性、重复抑制和相关的正文倾斜变化。训练不生成新字形、不学习新连笔或笔顺，也不推断照片里没有的真实压感。原始 LaTeX、数学字号、个人横线和一题一页规则继续保留。

## 本轮实际结果

在本地 `personal@v3` 上完成 6 种动作 × 3 个种子 × 3 道训练题的 54 页实测。用这些输出的自动奖励表进行了 800 次单步策略梯度更新，记录在 `out/realism/rl-proxy-v1/training.json`。这不是新渲染了 800 页，也没有人类评分混入自动奖励。

训练奖励的分布加权期望由 0.7413656 升至 0.7464170。两道未参与奖励计算的题目各用两个新种子抽查，训练前后恰好抽到相同动作，均值均为 0.7202005。该小样本验证没有显示提升，不能作为整体自然度更好的证据。

一个候选动作的平均笔宽对数误差为 0.1498，原 B 为 0.1816，约下降 17.5%；相应字形熵为 0.8851 和 0.9684。笔宽更稳定与更多变体之间存在取舍。这个动作的自动分数不能代替作者判断。

候选五题 PDF 位于 `out/realism/rl-proxy-v1/candidate/homework_solutions.pdf`；12 组匿名对照位于 `out/realism/rl-round1/review/compare.pdf`。候选没有替换默认 Profile。旧 B 偏好仅用于保留参照动作，不充当新对照的人类标签。

## 奖励与动作

每页先选一个动作，再在原有同字符字形池中采样。候选字形先按既有规则适配自己的字形盒，随后测量轮廓中线附近的近似笔宽。测量使用 192 px/em 的二值轮廓与距离变换，单位为 em。参考值是个人已采集字母和数字的主字形笔宽中位数。这是几何代理指标，不是书写压力测量。

字形采样偏好由相对参考笔宽的对数偏差、相邻字形的笔宽偏差、同字形重复惩罚组成。为每个真实样本保留小概率，正文倾斜采用相关漂移并限制在原有字形盒内；不独立旋转、移动公式里的数字或运算符。

自动奖励为 `0.55 exp(-2 × 笔宽对数RMSE) + 0.25 exp(-2 × 相邻笔宽变化) + 0.20 × 归一化字形熵`。这些权重是本轮工程选择，尚未经作者偏好校准。随机性并非越大越好。

优化器是有限动作、单步环境中的 REINFORCE / gradient bandit，包含移动奖励基线、相对初始分布的 KL 惩罚和熵项。输入是实际渲染得到的有限奖励表；反复抽样并不会增加独立证据量。数学结构与像素核验在产生奖励表之前运行，失败输出不参与优化。

收到偏好后，A/B 胜者记 1、败者记 0；tie 两边各 0.5，neither 两边各 0。每个动作从自己的已评价样本中回放奖励。这是简单的人类偏好 bandit，不是训练了一个深度奖励模型。每轮至少六组且覆盖全部动作，建议完成十二组；这是流程门槛，不是样本量充分性的统计保证。任何数学错误反馈均先中止训练并要求调查。

算法背景见 Sutton 与 Barto 的 [Gradient Bandits](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)。成对偏好与奖励学习的研究背景见 [Christiano 等人的论文](https://arxiv.org/abs/1706.03741)；本项目未复现该文的深度强化学习实验。

## 使用

```bash
export HWSOL_DATA_DIR="$PWD/profiles/realism-studio"
export MPLCONFIGDIR=/tmp/hwsol-mpl
export HANDNOTE_CACHE=/tmp/hwsol-handnote-cache
.venv/bin/hwsol lab rl prepare --manifest out/realism/math-size/input-five/job.json --output out/my-rl-trial
.venv/bin/hwsol lab rl train --run out/my-rl-trial --output out/my-rl-proxy
```

填写 `review/ratings.csv` 后运行新一轮，输出目录必须不存在：

```bash
.venv/bin/hwsol lab rl train --run out/my-rl-trial --ratings out/my-rl-trial/review/ratings.csv --parent out/my-rl-proxy/policy.json --output out/my-rl-human
```

每轮保留独立的 policy.json、training.json、候选作业、另题验证及可选评分文件副本。`--parent` 只读取历史分布，既有策略文件不覆盖；数据集、评分和父策略摘要进入来源记录。重复使用同一份偏好不是新一轮独立反馈。

策略通过 manifest 的 `render.policy` 内嵌，绑定完整 Profile 摘要，仅支持 `stroke-enhanced` 或 `combined`。运行时只读策略参数，不导入训练模块。交付 manifest 包含全部动作和概率，迁移个人 Profile 后可以重放，无需原训练照片或评分文件。未设置策略的作业继续采用当前 B 的流程。需要回到原 B 时移除 `render.policy`，或使用先前无策略的 manifest。

`seal.json` 绑定题包的图像、动作映射及实际输出；`dataset.json` 记录渲染代码摘要。图像、映射或渲染代码改变后，应重新生成对照。评分 CSV 单独可编辑。文件摘要用于一致性校验，不是作者身份或真实性证明。

## 首轮作者反馈与尺寸修正

已收到 12 组作者评价：5 组选择一侧、1 组相当、6 组两侧都不满意。先在匹配的原渲染器和封存图像上
执行 800 次偏好更新，保存到 `out/realism/rl-human-v1/`，再修正渲染器。
用户明确指出 l、p、y 偏小与 0 偏大；正文变体的等比适配会把宽字缩矮，宽槽又会把窄字放大。
现在横向单独适配，保留正常变体的高度和基线；高度异常者限制到该字符主字形高度的 95%–105%，
并围绕主字形的垂直中心校准。该范围是工程约束，不是从这十二个评分估计出的统计置信区间。
数学路径沿用已有字号适配。此正文修正在 stroke-enhanced/combined 的变体选择阶段生效。

修正版五题和字形对照在 `out/realism/rl-size-feedback/`。使用旧评分得到的策略分布作为试用起点，
新图尚无作者评分；`transfer.json` 区分训练时和修正后的渲染器来源。旧题包的训练会因代码摘要变化
而拒绝继续运行，不能给新图套用旧标签。已有的偏好训练结果和原题包保留。
当前两种 Python 各通过 93 项测试，包含上伸/下伸笔画、过宽/过大的变体和跨渲染器评分拒绝。

## 作者接下来提供什么

首轮十二组已记录到 `samples/realism-inbox/rl-round1/feedback/ratings.csv`，不需要重评。
现在先查看尺寸修正版即可。未来新一轮对照可继续用 A、B、tie、neither，并注明数字/运算符是否有误；
字段仍为 `pair_id,winner,math_error_a,math_error_b,notes`。之前四张字格无需重传。

若要继续学个人随机变化与粗细，建议再提供两页训练、一页独立验收：

| 内容 | 数量与形式 | 上传位置 |
|---|---|---|
| 自然英文 | 1 页，10–12 行；常用词自然重复，用平时速度、常用笔。JPG/PNG 或扫描 PDF，附同名 TXT 转写 | samples/realism-inbox/rl-round1/train/english.* |
| 数学推导 | 1 页，8–12 行；数字、正负号、等号、分式、上下标、均值横线、根号，允许自然重复。相同图像格式并附转写 | samples/realism-inbox/rl-round1/train/math.* |
| 全新混合解答 | 1 页，英语说明加数学推导，内容与训练页不同；仅留作验收 | samples/realism-inbox/rl-round1/holdout/ |
| 设备原始轨迹（可选） | 上述两页的原始导出 JSON/CSV；有则保留 x、y、时间、压力、抬笔及原文，注明坐标和时间单位 | samples/realism-inbox/rl-round1/train/tablet/ |

纸面样本建议 300 dpi 扫描或宽度至少 2000 px 的清晰照片，保留整页和纸张尺寸，不锐化或美化笔画。若设备没有时间/压力，不要补造。原始设备格式收到后再转换到当前校准格式，不要求用户手工编写轨迹。

新照片会用于增加真实字形/布局观察量并评估笔宽，不能直接变成真实压力轨迹。新 Profile 的摘要变化后，旧策略会拒绝绑定，需重新生成对照并训练该版本的策略。独立验收页不会加入奖励表。

## 验证与限制

Linux Python 3.12、3.13 各通过 92 项测试，覆盖梯度更新方向、测量、旧 B 逐字节保留、公式几何、固定种子、Profile 绑定、真实渲染训练回合、合成偏好回合、数学错误拒绝、证据篡改与跨目录迁移。测试中的偏好标签明确为合成数据，未用于本人的训练。正式候选五题五页通过逐页像素核验，迁移后 PDF/PNG 逐字节一致。

当前已有首轮十二组偏好，但尺寸修正版和独立新手写页尚未验收。只读照片无法验证笔顺或压感，字形池也不能生成尚未采集的连笔。此轮未做神经网络训练；本次 nvidia-smi 无法连接驱动，未把历史 GPU 检测当作当前可用环境。CPU 实验不依赖 CUDA。旧 v0.3.0 发行包尚未包含此功能。

矩阵框等结构线未参与本轮笔宽优化。公式几何与数值核验不能判断每个手写变体是否清楚易读，仍需作者检查数字和运算符。
