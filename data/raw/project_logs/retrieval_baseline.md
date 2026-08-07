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

# Keyword Retrieval Baseline 日志

第一版 Retriever 不引入中文分词依赖。英文和数字按连续字符串提取，中文按单字和相邻双字生成 token，再使用 Query token 重叠率评分。

检索支持 source type 过滤和 top-k，返回 score、rank、chunk id 与命中 token。该分数不是语义相似度。
