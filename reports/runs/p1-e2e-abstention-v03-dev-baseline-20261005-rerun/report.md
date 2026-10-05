# p1-e2e-abstention-v03-dev-baseline-20261005-rerun

- Dataset: `evalrag_v0.3/dev`
- Cases: 160 (answerable=120, unanswerable=40)
- Retriever: `adaptive-retrieval-v2.0`
- Evidence Gate: `evidence-gate-v2.0-dev-locked`
- Prompt: `p1-e2e-funnel-v1`

## Failure Funnel

| Terminal stage | All | Answerable |
|---|---:|---:|
| router_abstain | 0 | 0 |
| retrieval_miss | 48 | 48 |
| gate_reject | 33 | 19 |
| context_evidence_dropped | 0 | 0 |
| generator_abstain | 46 | 23 |
| citation_invalid | 16 | 13 |
| model_error | 1 | 1 |
| answered | 16 | 16 |

## Metrics

- Answerable E2E Success: 11.67%
- Unexpected Abstention Rate: 75.00%
- Unanswerable Abstention Accuracy: 92.50%
- Citation Validity (answered): 100.00%
- Error count: 17

逐 Case 证据流见 `case_results.jsonl`，非 answered Case 见 `failures.jsonl`。
