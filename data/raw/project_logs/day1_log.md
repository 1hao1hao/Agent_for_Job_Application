---
source_type: project_logs
source_platform: project_repository
source_url: unknown
collected_at: 2026-06-13
public_status: project_owned
anonymized: true
content_origin: project_record
human_reviewed: true
---

# 项目日志 Day 1

## 今日目标

完成项目骨架和文档，不急着实现复杂业务代码。

## 重要决策

1. 第一版先用原生 Python 实现核心 RAG 流程，不提前引入 LangGraph 或 LlamaIndex。
2. 每个模块都要保持足够小，方便在实习面试中讲清楚输入、输出和失败排查方式。
3. 失败样例不能只停留在口头复盘，后续要沉淀为 regression test（回归测试）。

## 当前限制

1. 还没有实现检索和评测。
2. 样本数据只用于开发和测试，不代表真实岗位数据。
