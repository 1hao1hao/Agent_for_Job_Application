---
source_type: interview
source_platform: project_authored
source_url: unknown
collected_at: 2026-07-29
public_status: project_owned
anonymized: true
content_origin: project_authored_note
human_reviewed: true
---

# Context Budget 面试笔记

Context Builder 按检索 rank 选择完整 Chunk 前缀，预算包含证据 header、正文和分隔符。下一个完整 Chunk 放不下时停止，不从中间截断。

字符预算实现简单且不绑定模型，但字符和 Token 不是一一对应。接入具体模型后应使用 tokenizer 计算完整 Prompt 的 Token 预算。
