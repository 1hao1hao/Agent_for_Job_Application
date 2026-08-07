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

# Citation Validator 面试笔记

模型返回的 cited chunk ids 是不可信输入。Validator 要检查 ID 是否存在于本轮 Context、是否重复，以及 sufficient 与引用是否为空的组合是否一致。

Citation Validity 只证明引用 ID 合法，不证明回答结论得到证据的语义支持。后者需要人工核验或独立 Evidence Checker。
