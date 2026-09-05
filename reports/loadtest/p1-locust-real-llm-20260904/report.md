# EvalRAG 真实 LLM 小流量验证

本次使用固定 DeepSeek 模型配置，在单并发、15 秒窗口内对四类 Query 轮询请求。共完成 10 次
`/v1/query` 请求，失败数为 0，平均延迟 1,220.46 ms，P50 为 8 ms，P95/P99 约为
6,500 ms，吞吐为 0.77 RPS。

P50 很低是因为部分请求被 Evidence Gate 在生成前快速拒答；进入 LLM 的多来源与关系推理请求
约耗时 1.5-6.5 秒。样本量只有 10，不能据此推断真实模型容量或稳定 P95，也没有采集
provider token/cost。本目录 CSV 是可复查原始结果，完整环境与 deterministic 对照见
[`../p1-locust-local-20260904/report.md`](../p1-locust-local-20260904/report.md)。
