# 当前开发状态

本文件跟踪 v0.6–v0.8 路线。历史审计保留于 CURRENT_STATE_AUDIT.md；v0.4.0 证据不改写。

- Phase 0 已实现：本地稳定 main、功能/实验分支；不可复用 run ID；旧 Profile 库兼容；新输出进入 workspace；失败不更新 latest；历史文件原位摘要索引。提交 d392235。
- Phase 1 验证中：60 例固定基准、render-only 与独立提交的 end-to-end 路径、资源/来源/结构记录、独立观察和真实 Tesseract 入口。首轮 48 例暴露两例关系符号别名问题，修复后 48/48 预期通过（42 页，6 次预期拒绝）。holdout 尚未用于调参。
- Phase 2 尚待上述交付验证完成后检查真实 CUDA，禁止把接口测试算作推理。
- Phase 3/4：须先成功推理、取得合格个人在线数据和独立评测；本轮尚未训练。
- Phase 5：font-cpu 保持默认，实验模型未进入生产路径。个人默认仍 personal@v1，未发布新模型或激活候选。

工作区契约：WORKSPACE.md；基准协议：BENCHMARK_V1.md。最终提交、标签、实测报告和路径将在验收后补充。
