# p1-task2-evidence-rerank-context-v03-dev-20261006

- Dataset: `evalrag_v0.3/dev`
- Cases: 160 (answerable=120, unanswerable=40)
- Retriever: `adaptive-retrieval-v2.1-evidence-rerank-dev-locked`
- Evidence Gate: `evidence-gate-v2.1-e2e-locked`
- Prompt: `p1-e2e-funnel-v2`

## Failure Funnel

| Terminal stage | All | Answerable |
|---|---:|---:|
| router_abstain | 0 | 0 |
| retrieval_miss | 45 | 45 |
| gate_reject | 34 | 21 |
| context_evidence_dropped | 4 | 4 |
| generator_abstain | 29 | 2 |
| citation_invalid | 1 | 1 |
| model_error | 0 | 0 |
| answered | 47 | 47 |

## Metrics

- Answerable E2E Success: 38.33%
- Unexpected Abstention Rate: 60.00%
- Unanswerable Abstention Accuracy: 100.00%
- Citation Validity (answered): 100.00%
- Error count: 1

逐 Case 证据流见 `case_results.jsonl`，非 answered Case 见 `failures.jsonl`。
