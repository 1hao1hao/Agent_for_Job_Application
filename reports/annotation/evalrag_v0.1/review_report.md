# EvalRAG v0.1 数据审核报告

## 汇总

- 语料：30 份已审核，30 份批准，0 份未批准。
- 评测 case：60 条已审核，60 条批准，0 条未批准。
- Split：dev 40 条、test 20 条，全部批准。
- Category：single_source、multi_source、semantic_paraphrase、unanswerable 各 15 条。
- 缺失 Chunk ID：0。
- 结构性 source 不一致：0。
- frozen test 未运行。

## 修正记录

- single_001：删除与技术问题无关的“实习时间”要点，补齐岗位技术项。
- multi_001：删除 JD 未要求的 Docker 缺口，改为 REST API 与关系型数据库实践。
- multi_002：补充 skill_matrix 证据，使未完成项有 Chunk 支持。
- multi_012：删除证据外的 Dense/Hybrid 要点，保留现有检索与岗位要求。
- multi_014：补充 learning_plan 证据，支持 Dense/Hybrid 未完成的判断。
- multi_015：改为结合 Ingestion 项目记录与 Context Budget 面试笔记。
- semantic_003：补充项目评测与 Generation/Validation 日志证据。
- semantic_014：替换与 dev 重复的服务能力问题，消除跨 split 同事实泄漏。
- semantic_015：将主要意图修正为 interview_prepare。

## 最终结论

evalrag_v0.1 的 60 条标签均已通过审核和正式数据校验，可以用于 dev
baseline。20 条 test 保持冻结，未用于本轮调参或运行。
