# M1 第三轮：保留 f 形状，恢复真实变化并处理 π

2026-11-20。作者对第二轮完整20页的回评：

> 有改善，f的形状不错了，但是出现了所有f趋同的现象，pi（圆周率符号）仍然有加粗问题

原话、所评 PDF SHA 已保存在
[反馈记录](../workspace/analysis/20261120-master-execution/author-feedback-m1-round3.json)。
`956bbad6` 保留为当前已获 f 形状肯定的对照，不因此认为变化质量或 π 已通过。

**第三轮已交付：真实 f0/f1 变体恢复通过主/独立审核，π 独立替换仍 No-Go。**
本轮不是神经训练，也不代表连续生成或作者风格准入。正式 Profile/latest 保持。

## 最终完整作业与对照

最终运行 `20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e`：

- [完整20页 PDF](../workspace/runs/20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e/rendered/homework_solutions.pdf)
- [逐页 PNG/PDF](../workspace/runs/20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e/rendered/pages)
- [124个完整 f 单词前后对照](../workspace/runs/20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e/comparisons/all-f-words/formula-regions.pdf)
- [全部145个 f 与20个 π 的局部对照](../workspace/runs/20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e/comparisons/all-f-pi-targets/formula-regions.pdf)
- [173组完整公式对照](../workspace/runs/20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e/comparisons/all-formulas/formula-regions.pdf)
- [前版完整PDF](../workspace/runs/20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e/comparisons/previous-reviewed20.pdf) · [逐页并排](../workspace/runs/20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e/comparisons/pages)
- [字体回退区域标注](../workspace/runs/20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e/comparisons/fallback-overlay/math-fallback-overlay.pdf) · [逐项来源与检查](../workspace/runs/20261120T124522448757Z-author-feedback-round3-authentic-f-complete-20-4af6334e/reports)

所有正文、公式、背景与排版条件与第二轮相同，20页均先精确重放原候选。
最终20张 PNG 还与通过审核的父候选 `fb1722a8` **逐字节相同**，集成交付未改其选择或形状。

## 本轮有界问题与准入

1. 变体塌缩来自 CPU 路径统一选择 f0。先查实际 GSUB 与真实来源，不能把循环访问的
   f2/f3 索引当作额外独立笔迹。目前编译字体实际只有 f0/f1；历史已核验局部还有
   final/for/unfiltered 中的 f，须核对原始 RGB、完整性、坐标及来源重叠。
2. 保留已认可的 f 比例，通过最多3个可靠真实来源候选检查可辨认性、横臂/下伸关系、
   自然差异和邻字净空。不得复制 f0 加噪声冒充学到变化，也不因增加来源数量忽略内容。
3. 当前 π 唯一已接入的来源本身偏粗；重新对未封存历史作业进行有界候选搜索，最多4个
   原迹。旧 resolver 仅有一个 accepted π 不等于历史材料中没有其他 π。
4. 真实原迹、原标签、原色300DPI裁剪、父区域、源 SHA、坐标与变换完整保留。OCR 是候选，
   不是标签真值。禁止擦细、补画或把相邻内容切损后强行接入。封存 C 不读取。
5. 新 `research/author_feedback_round3.py` 精确重放第二轮实际输出，旧源文件均按 SHA
   校验。先完成源与局部复核，再组合完整20页，保留非目标英文/公式/版式/背景。
   同词 ff 必须保持组内协调和原子准入，不能恢复旧“一大一小”候选。

本轮属于已曝光的作者反馈回归，不是新的独立 Holdout。不启动未经数据及机制准入的神经
训练。若真实变体或轻笔 π 无可靠来源，明确保留并使用已完成的广覆盖采集册补充，
不能用改笔迹或增训次数制造成功。

## 实际 f 结果

已保存作者反馈、建立第三轮精确回放入口；第1/6/12/17/20页像素、来源及 parser trace
均与已封存第二轮完全相同。f 原始来源审核已看两张表格真实 f 与三枚历史词 f，
进入 f1 和 final 的有界原生词级探针。两个红笔源不使用旧训练中加深的派生图；
final 原比例较宽，不能为了塞进小槽而压扁。

编译字体实际只有两枚独立表格 f，来自同一次采集的两次书写；不是4个新来源。
原 f0 作为当前已认可形状，另一真实 f1 保留不同的顶钩、横臂和下钩，通过共同坐标恢复
相对比例。f1 实际细干约1.52 px，f0约2.19 px；没有加粗/增墨。主和独立 Agent 在原生
字号及完整词中确认其可辨认、不断笔，不能只因比 f0细就判定退步，也不替代本人偏好。

历史 final 的真实比例约1225×700字体单位，当前D约1034×407；四页24处探针只有2处
空间允许，实际效果仍明显大/厚，整条路线 **No-Go**。不压缩改造来凑第三个来源。
两个红笔来源保留审核证据，未将约1.9倍加深的旧训练派生图当原迹。

固定一次 seed1909，通过 `SHA256(seed/title/实际词span或LaTeX上下文)` 名义50/50选择
f0/f1，不筛种子；同一词的 f 共享选择，同词 ff 还须全组通过。结果：

| 项目 | 实际结果 |
| --- | ---: |
| 全部 f | 145：140正文/标题，5数学 |
| 实际 f1 | 53：52正文/标题，1数学（第6页 f_avg） |
| 实际 f0 | 92：30原保留项＋2新增组内保留＋60固定选择 |
| 名义选择 | f1为67，f0为78；安全门禁后得到上述53/92 |
| 真实同词 ff | 14组全部保持实际来源一致 |
| 新增拒绝 | 第16页 cutoff 一枚 f1会碰邻字，246/247两枚共同保留 f0 |

原30处小 f 未被本轮绕过旧安全门禁。其余接受的 f1 与邻字最小距离约1.199 px，
超过0.60 px门槛。改变只有53个既有f item，完整10,629条非f来源记录保持精确相同；
92个未改f记录也相同。数学2,265项、字体回退30处（29外部、1继承）均不变，
20个π的轮廓及来源保持。非目标几何与目标区域外像素差为0，无新增不相关接触。

这证明两枚已采真实 f 被调用，修复了统一选择 f0 的机械重复；**没有证明学到了新字形、
连续书写或开放式自然变化**。只有两枚真实来源仍然有限，数学与英语神经长期目标不因此完成。

主 Agent 看完整五源原色对照、所有53处变化、原生探针及最终父候选第1/4/6/16/18/20页。
独立 Agent 看所有53处变化与全部14组完整ff词，复算145次选择，并核验真实字体文件和SHA。
[独立审核](../workspace/analysis/20261120T123058133770Z-round3-independent-1381bc61/README.md) ·
[主审核](../workspace/analysis/20261120-round3-root-review/FINAL_REVIEW.json)。
Go 仅用于作者对照，尚无本人对混合 f 结果的认可。

## π 搜索与明确 No-Go

新增来源终审：
[0029bd3f](../workspace/runs/20261120T123612508971Z-pi-connected-source-review-0029bd3f/reports/REVIEW.md)，
其五份原色证据与先前已审 `97cc64e2` 完全相同。
两份本地索引的11条 π 标签命中去重为4份文档8个区域；逐原图复核后7处为 x、p、中文
或英文等误识别，1处是目前已用的重 π。额外复核物理作业第55页的另一周期公式和整页
上下文，实际存在被 OCR 漏掉的较轻 π，但它的左腿与前面的2真实相接。

主/独立 Agent 均查看原色放大和源上下文，四个阈值的连通检查一致：2 与 π 顶部/左腿
同一连通域，π 右腿另成域。完整原色2π片段、PDF SHA、300DPI框、坐标变换、原 OCR
和单独的 Agent 实际转写保留；不改写旧标签。**孤立 π 提取 No-Go**，没有裁断笔迹或
制造“新轻 π”。这次只检查了有界的索引候选，不声称穷尽所有历史 PDF。

原索引过滤还发现旧 `derivation.selection.split` 中的 holdout 标记。**首轮旧入口确实
读入了 physics1 的16条 sample JSON，其中15条含旧 holdout 字符标签，排查时还输出了
部分元数据。**该来源无 recognition 行，没有读取其 OCR recognition 文字，但这不等于
没有接触留出标签。没有打开其原 PDF/页图/字形，也未据此选样或训练。封存 C 的已知 SHA
及名称排除一直在读取标签前生效。现已新增共享 `research/corpus_access.py`，旧
`missing_math_review.catalogue` 和新 π 审计都调用它，在标签获取前排除整来源、祖先、
后代与兄弟记录，覆盖旧嵌套 split。合成临时 SQLite 和受保护标签哨兵验证通过，真实索引
补充回执只读元数据与计数，未再读受保护文字或图像。详见
[完整访问披露](../workspace/runs/20261120T124151564297Z-corpus-access-guard-audit-7702c97c/reports/ACCESS_DISCLOSURE.md)。
该修复是实际访问边界问题的工程证据，不是笔迹质量进步；既有 run 不改写，也不将已曝光
数据冒称新独立 Holdout。

本轮不扩大到跨2/π的片段替换：它需要多字符来源/布局约定，会牵动已认可的2，且不能
覆盖独立 π 的所有场景。20处 π 继续明确保留，等待已交付 O-FULL 中的自然独立 π。
这不妨碍完成 f 变体工作。

## 复现、失败和下一入口

源码提交 `33eb737`，包含真实变体源审计/接入、第三轮精确回放、组合交付及共享索引防护。
最终 run 保存研究源码、实际配置、seed1909、Profile、全部原色来源/变换、父运行清单与
SHA批准回执。Linux x86_64、Python3.12.16、CPU；没有新神经更新/优化器/checkpoint。
现有 M0 注册表、正式 latest、锁文件、主计划原文及 CPU 无 Torch 均再次核验保持。

实际命令：

```sh
MPLCONFIGDIR=/tmp/hwsol-round3-final-mpl .venv/bin/python -m research.author_feedback_round3_delivery --approval workspace/analysis/20261120-round3-root-review/integration-approval.json
MPLCONFIGDIR=/tmp/hwsol-mpl .venv/bin/python -m pytest -q tests/test_corpus_access.py
.venv/bin/ruff check research/author_feedback_round3.py research/author_feedback_round3_delivery.py research/f_variation_feedback.py research/pi_source_feedback.py research/corpus_access.py research/missing_math_review.py tests/test_corpus_access.py
.venv/bin/python -m research.project_state verify
```

保留源审计 `0ad146b2`、早期探针 `15a1b8a9` 的执行失败，以及首次完整词输出
`92e783cc` 的 crop_pair 参数顺序错误；修复后使用新ID，未改种子、来源或门槛。
四页双源探针 `af65bccb` 保留 final 的视觉失败；最终父候选为 `fb1722a8`。
这些执行/视觉失败不能合并成成功训练次数。全部表格/历史原迹均未覆盖。

最终 run SHA：`9419cb9713df8d5118603bd166f5f6c4ec31e2148afcea5b2058a11e4789248f`。
PDF SHA：`c10f557a97ef0485b583d7111f0ecf418d9d6aec1b4cfe1f4f571cc410732933`。
run 保持 held，封存后的视觉审核另存 analysis，不反写原清单。

下一实际作者输入是[完整采集册](COMPREHENSIVE_MATH_COLLECTION.md)的原色填写和混合 f 风格反馈。
采集册已覆盖19页123行，无需再缩减；扫描保存 `source_pdf/session-O_full_source.pdf`。
π和其他缺项必须审核新原迹后才能替换。新输入到达前，当前有界修订完成，M2正式发布及
M4连续神经生成仍遵守各自未通过的门槛；不把真实字形选择叫作神经训练。
