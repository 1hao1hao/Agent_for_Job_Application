# Model Gateway Fault Matrix

> 本报告使用确定性 Fake Provider，只证明控制流，不代表线上 SLO。

| Scenario | Status | Selected | Fallback | Attempts | Latency ms |
|---|---|---|---:|---:|---:|
| primary_success | succeeded | primary | 0 | 1 | 2.394 |
| timeout_fallback | succeeded | backup | 1 | 3 | 3.530 |
| rate_limit_retry | succeeded | primary | 0 | 2 | 2.518 |
| provider_5xx_fallback | succeeded | backup | 1 | 3 | 3.481 |
| auth_fallback | succeeded | backup | 1 | 2 | 2.339 |
| all_unavailable | controlled_failure | None | 0 | 2 | 2.321 |

- Success rate: 83.33%
- Fallback rate: 50.00%
