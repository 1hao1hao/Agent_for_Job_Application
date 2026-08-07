---
source_type: project_logs
source_platform: project_repository
source_url: unknown
collected_at: 2026-07-29
public_status: project_owned
anonymized: true
content_origin: project_record
human_reviewed: true
---

# Router 与 Trace 迭代日志

规则 Router 定义岗位分析、简历匹配、面试准备、项目讲解、投递计划和 unknown 六类意图，并记录命中的关键词。

Trace 使用 JSONL 追加写入，最初记录路由、召回和耗时，端到端链路完成后扩展了 Context、Generation 和 Validation 阶段。
