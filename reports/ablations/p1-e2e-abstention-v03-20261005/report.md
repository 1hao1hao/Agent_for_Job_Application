# E2E Unexpected Abstention Closure

## Protocol

- Dataset: `evalrag_v0.3`; dev=160 (answerable=120), release test=80 (answerable=60).
- Pipeline: Feedback Hybrid Router -> Adaptive v2 -> Evidence Gate -> Adaptive Context -> DeepSeek -> Citation Validator.
- Dev 用于定位和修复；配置锁定后 test 仅运行一次，未据此继续调参。
- 最终 dev 复用 candidate 的模型输出，只对 1 条 `sufficient=false` 非法引用执行确定性 Validator replay。

## Dev Before / After

| Metric | Baseline | Final | Delta |
|---|---:|---:|---:|
| Answerable E2E Success | 11.67% | 30.83% | +19.17% |
| Unexpected Abstention Rate | 75.00% | 66.67% | -8.33% |
| Unanswerable Abstention Accuracy | 92.50% | 100.00% | +7.50% |
| Citation Validity (answered) | 100.00% | 100.00% | +0.00% |
| Error count | 17 | 0 | -17 |

## Failure Funnel

| Terminal stage | Baseline all | Final all | Baseline answerable | Final answerable |
|---|---:|---:|---:|---:|
| router_abstain | 0 | 0 | 0 | 0 |
| retrieval_miss | 48 | 45 | 48 | 45 |
| gate_reject | 33 | 33 | 19 | 19 |
| context_evidence_dropped | 0 | 2 | 0 | 2 |
| generator_abstain | 46 | 40 | 23 | 14 |
| citation_invalid | 16 | 0 | 13 | 0 |
| model_error | 1 | 0 | 1 | 0 |
| answered | 16 | 40 | 16 | 40 |

## Root Cause And Fix

1. `EvidenceRequirement` 未保存 Query 明确点名的来源，导致跨来源问题只检索 Router 返回的部分来源；现由 Gate 按明确来源判断并最多扩源一次。
2. freshness metadata 与 Graph path 已被 Retriever 持有但未进入 Prompt；Context 现在只注入必要 provenance 字段和已验证关系路径。
3. Context Engine 曾在同优先级内按 chunk id 而非 rank 装箱，且 Citation 白名单包含被 token budget 丢弃的证据；现保持 rank/source 顺序并同步真实白名单。
4. 模型安全拒答但携带引用时不再升级成系统错误：Validator 仍记录问题，Pipeline 丢弃引用并返回 `insufficient_evidence`。

## Locked Release Test

- Answerable E2E Success: 38.33%
- Unexpected Abstention Rate: 60.00%
- Unanswerable Abstention Accuracy: 100.00%
- Citation Validity (answered): 100.00%
- Error count: 0

## Remaining Dev Failures

`context_evidence_dropped`=2, `gate_reject`=19, `generator_abstain`=14, `retrieval_miss`=45

最大剩余项是 retrieval miss，应留给后续 Retriever/Rerank；关系问题的 graph path 缺失应留给 Graph Retrieval。Context 仍有 2 条预算丢证据，适合后续做 query-aware evidence packing。本任务不据此修改 test、Retriever 或 benchmark 标签。

## Artifacts

- Baseline: `reports/runs/p1-e2e-abstention-v03-dev-baseline-20261005-rerun/`
- Final dev case/failure: 本目录 `case_results.jsonl` / `failures.jsonl`
- Release test: `reports/runs/p1-e2e-abstention-v03-release-test-20261005/`
