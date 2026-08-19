# P1-D5 Context Engine and Memory Dev Ablation

- Dataset: `evalrag_context_v0.1`; split: dev; 60 groups / 300 turns.
- Predictions are produced by ContextEngine; no LLM calls, estimated cost $0.

| Strategy | Follow-up | Citation | Key-point | Grounding | Prompt tokens | Compression | Repeat reads | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_memory | 36.67% | 100.00% | 28.30% | 36.67% | 37.2 | 31.50% | 0.0 | 0.089 |
| recent_window | 36.67% | 100.00% | 28.30% | 36.67% | 66.2 | 0.00% | 3.0 | 0.114 |
| summary_recent | 100.00% | 100.00% | 100.00% | 100.00% | 75.7 | 0.00% | 3.0 | 0.125 |
| semantic_memory | 100.00% | 100.00% | 100.00% | 100.00% | 47.9 | 15.08% | 0.0 | 0.101 |

## Difference cases

- `context-v01-001` (reference): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-002` (ellipsis): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-004` (cross_session): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-005` (memory_conflict): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-006` (topic_switch): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-009` (reference): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-010` (ellipsis): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-012` (cross_session): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-013` (memory_conflict): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-014` (topic_switch): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-017` (reference): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-018` (ellipsis): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-020` (cross_session): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-021` (memory_conflict): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}
- `context-v01-022` (topic_switch): {'no_memory': False, 'recent_window': False, 'summary_recent': True, 'semantic_memory': True}

## Boundary

Context-level scenario benchmark; key-point uses scenario-grounded expected facts and does not equal free-form answer accuracy
