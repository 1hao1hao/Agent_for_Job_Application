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

# 检索指标面试笔记

Recall@k 衡量前 k 个检索结果覆盖了多少人工标注的相关证据。MRR 只关注第一条相关证据的位置，相关证据越靠前，倒数排名越高。

评测时要固定语料、Query、切分参数和 top-k。检索指标只能说明证据召回情况，不能直接等同于最终答案准确率。
