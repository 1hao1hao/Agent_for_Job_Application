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

# Router 设计面试笔记

规则 Router 根据 Query 中的关键词预测 intent，并将 intent 映射到优先知识源。它的优点是行为可解释、测试简单，缺点是同义表达和多意图问题容易误判。

Router Accuracy 应独立计算，不能用全库检索碰巧找到答案来掩盖路由错误。
