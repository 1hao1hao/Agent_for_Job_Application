# p1-e2e-abstention-v03-release-test-20261005

- Dataset: `evalrag_v0.3/test`
- Cases: 80 (answerable=60, unanswerable=20)
- Retriever: `adaptive-retrieval-v2.0`
- Evidence Gate: `evidence-gate-v2.1-e2e-locked`
- Prompt: `p1-e2e-funnel-v2`

## Failure Funnel

| Terminal stage | All | Answerable |
|---|---:|---:|
| router_abstain | 0 | 0 |
| retrieval_miss | 19 | 19 |
| gate_reject | 18 | 12 |
| context_evidence_dropped | 1 | 1 |
| generator_abstain | 18 | 4 |
| citation_invalid | 0 | 0 |
| model_error | 0 | 0 |
| answered | 24 | 24 |

## Metrics

- Answerable E2E Success: 38.33%
- Unexpected Abstention Rate: 60.00%
- Unanswerable Abstention Accuracy: 100.00%
- Citation Validity (answered): 100.00%
- Error count: 0

逐 Case 证据流见 `case_results.jsonl`，非 answered Case 见 `failures.jsonl`。
