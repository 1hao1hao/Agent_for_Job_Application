# Task 2: Evidence-Need Rerank and Rewrite Decision

- Dataset: `evalrag_v0.3/dev` (160 cases; no test access).
- Candidate pool: top-20.

| Policy | Recall@5 | MRR | NDCG@5 | Invoke | P95 ms | Ranking misses recovered |
|---|---:|---:|---:|---:|---:|---:|
| never | 0.5833 | 0.5211 | 0.5121 | 0.00% | 1217.55 | 5 |
| always | 0.6042 | 0.5621 | 0.5507 | 62.50% | 4497.26 | 6 |
| low_confidence | 0.5833 | 0.5211 | 0.5121 | 1.88% | 1514.15 | 5 |
| evidence_need | 0.6042 | 0.5621 | 0.5507 | 50.00% | 3604.30 | 6 |

## Failure Attribution

- Historical retrieval misses: 45
- ranking_miss: 21
- recall_miss: 24

## Locked Decision

- Rerank need types: `['exact_fact', 'semantic_explanation', 'multi_source_synthesis']`
- Query rewrite: `NOT_JUSTIFIED`
- Config: `configs/retrieval/adaptive_evidence_rerank_v0.3.json`

This report measures retrieval placement only; it does not claim answer accuracy.

## Evidence-first Context

- Historical cross-source context drops under the same candidates/budget: `2 -> 0`.
- 60 groups / 300 turns: Follow-up Success `100%`, Citation Validity `100%`.
- Adaptive mean Prompt Token: historical `60.68` -> Evidence-first `60.92`.
- Real E2E still found four new two-hop path-level context drops; source-level priority does
  not yet guarantee that every relation-path evidence chunk fits the budget.

## Real DeepSeek Dev E2E

| Metric | Previous | Task 2 |
|---|---:|---:|
| Answerable E2E Success | 30.83% | 38.33% |
| Unexpected Abstention | 66.67% | 60.00% |
| Unanswerable Abstention Accuracy | 100.00% | 100.00% |
| Citation Validity (answered) | 100.00% | 100.00% |

Task 2 answerable failure funnel: retrieval miss 45, Gate reject 21, Context drop 4,
Generator abstain 2, Citation invalid 1. The run made 110 generation calls and reported
149,721 input + 20,640 output = 170,361 tokens. Provider pricing was not pinned, so cost
is intentionally unavailable rather than estimated.
