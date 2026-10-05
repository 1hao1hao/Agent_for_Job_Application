# p1-e2e-abstention-v03-dev-release-candidate-20261005

- Dataset: `evalrag_v0.3/dev`
- Cases: 160 (answerable=120, unanswerable=40)
- Retriever: `adaptive-retrieval-v2.0`
- Evidence Gate: `evidence-gate-v2.1-e2e-dev`
- Prompt: `p1-e2e-funnel-v2`

## Failure Funnel

| Terminal stage | All | Answerable |
|---|---:|---:|
| router_abstain | 0 | 0 |
| retrieval_miss | 45 | 45 |
| gate_reject | 33 | 19 |
| context_evidence_dropped | 2 | 2 |
| generator_abstain | 39 | 14 |
| citation_invalid | 1 | 0 |
| model_error | 0 | 0 |
| answered | 40 | 40 |

## Metrics

- Answerable E2E Success: 30.83%
- Unexpected Abstention Rate: 66.67%
- Unanswerable Abstention Accuracy: 97.50%
- Citation Validity (answered): 100.00%
- Error count: 1

逐 Case 证据流见 `case_results.jsonl`，非 answered Case 见 `failures.jsonl`。
