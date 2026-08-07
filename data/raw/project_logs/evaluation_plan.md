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

# Evaluation 计划日志

评测集计划包含四类 Query：单来源、多来源、语义改写和不可回答，并划分 dev 与冻结 test。系统 prediction 必须由真实 Router 和 Retriever 运行产生。

Keyword baseline 记录 Router Accuracy、Recall@3、Recall@5、MRR 和 P50/P95 延迟，失败 case 不得删除。
