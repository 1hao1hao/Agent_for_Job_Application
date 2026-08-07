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

# Ingestion 模块迭代日志

项目将本地 Markdown 和文本文件统一转换为 Document，再按空行和最大字符数生成 Chunk。Chunk 完整继承文档 metadata，并使用来源、路径、索引和文本生成稳定 ID。

岗位数据额外支持 JSON/CSV importer，但采集爬虫不属于核心依赖。
