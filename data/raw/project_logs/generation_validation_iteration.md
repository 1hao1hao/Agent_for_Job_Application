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

# Generation 与 Validation 迭代日志

Generator 通过 LlmClient Protocol 注入 Fake 或真实模型 adapter，要求模型输出 answer、cited chunk ids、sufficient 和 reason。

模型 JSON 先做结构解析，再由 Citation Validator 检查引用。非法 JSON 和非法 citation 会转成不同的受控错误，不允许直接返回 answered。
