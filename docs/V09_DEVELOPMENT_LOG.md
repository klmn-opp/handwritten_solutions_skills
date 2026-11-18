# v0.9 研发日志

日期：2026-11-18；分支：experiment/personal-corpus-v09。

1. 从v0.8 `24589ba`建立独立分支；读取历史状态、基准和CUDA报告，保留全部旧实验。
   `8d6343e`保存只读审计工具和source_pdf私有入口。初始仅7张原始个人图片，用户随后上传PDF。
2. `f404bf6`实现按摘要原件归档、页面坐标/派生图、SQLite审核历史、数据质量门禁。
   首次旧图导入因EXIF IFDRational不能JSON序列化失败；显式转换数值后新run恢复成功。
3. 收到7 PDF（831页），原件全归档，抽查21张缩略页，选择10页中部实际内容做试点。
   用户指出大量涂改/遮挡；加入质量分类、缺陷标记和“可读但污染也不进干净生成训练”的门禁。
   没有按颜色认定作者，也没有擦除分数线、生成式补画或将参考答案当真实笔迹。
4. `b44e802`提交独立识别Adapter和依赖锁。真实TrOCR/UniMERNet/Tesseract共58条输出。
   UniMERNet官方代码加载真实tiny权重，strict=True；无上游模型代码补丁。
   transformers固定4.42.4；pyarrow17/datasets2.21解决NumPy1.26兼容冲突，失败日志保留。
5. `1d89c3f`提交本地审核Studio、训练门禁、轻量CVAE原型和数学回退/偏好研究契约。
   浏览器测试发现KaTeX目录应包含dist，修正后公式预览成功；草稿恢复及未声明拒绝确认成功。
   本轮未向真实语料提交任何伪造人工确认。
6. 导出v1发现纯哈希桶将7份新文档全部划为Train。训练尚未开始、无成绩可调；
   改为初始组摘要排序保留独立Validation/Holdout，之后增量追加不洗牌，形成v1.1。
   v1和首次No-Go收据保留，分组升级有数据库事件。旧校准图不得成为新独立Holdout。
7. 生成器优化单元测试在RTX5060真实执行，约40万参数有非零更新；仅合成条纹夹具。
   个人训练因0个已核验干净样本被门禁拒绝，没有个人checkpoint、SFT、RL或新默认模型。
8. 四学科CPU PDF初次因run预建空pages目录触发Runtime输出保护失败；仅移除新run空占位目录，
   新run成功渲染四页，内容/边界/页数/来源检查通过，未更新latest。
9. Python3.12/3.13各159项全套测试通过。Ruff、JS语法、Python编译通过。
   默认生产venv无setuptools导致无隔离构建失败；使用归档源码副本和固定setuptools隔离构建成功，
   没有修改生产venv或历史发行包。新构建仍是CPU0.6.0兼容校验，不是发布v0.9神经Runtime。
10. 12,822历史文件、全部7新PDF、生产src/依赖、用户.vscode、正式latest、默认Profile摘要复核不变。
    失败与未准入运行不作为成功质量证据；测试通过不等于笔迹真实性提高。

详细交付：[V09_GO_NO_GO](V09_GO_NO_GO.md)；数据规范和本地命令：
[PERSONAL_HOMEWORK_CORPUS](PERSONAL_HOMEWORK_CORPUS.md)。后续先人工核验与扩充覆盖，再论证个人训练。
