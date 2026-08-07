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

# Agent Trace 面试笔记

一次 Query 对应一条请求级 Trace，记录 routing、retrieval、context、generation、validation 和 total latency。错误请求也应尽可能保存已经完成的阶段。

Trace 的目标是定位失败发生在哪一层、使用了什么配置以及修改后是否改善，而不是无边界地保存所有隐私数据。
