# Adaptive Context Engine and Layered Memory Dev Ablation

- Dataset: `evalrag_context_v0.1`; split: dev; 60 groups / 300 turns.
- Predictions are produced by ContextEngine; no LLM calls, estimated cost $0.

| Strategy | Follow-up | Key-point | Prompt tokens | History redundancy | Memory recall | P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| recent_window | 36.67% | 28.30% | 67.2 | 42.86% | 0.00% | 1800.787 |
| summary_recent | 100.00% | 100.00% | 75.5 | 42.86% | 0.00% | 3903.684 |
| semantic_memory | 100.00% | 100.00% | 65.4 | 0.00% | 84.91% | 3795.673 |
| adaptive_policy | 100.00% | 100.00% | 60.7 | 0.00% | 56.60% | 4108.420 |

## Quality-equivalent comparison

Adaptive and Summary+Recent both keep 100% Follow-up Success. Adaptive reduces mean Prompt Token by 19.68% and History Redundancy by 100.00%.
Adaptive Memory Recall is lower than always-on Semantic Memory; this is an explicit on-demand recall trade-off, not a global Pareto claim.

## Difference cases

- `context-v01-001` (reference): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-002` (ellipsis): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-004` (cross_session): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-005` (memory_conflict): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-006` (topic_switch): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-009` (reference): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-010` (ellipsis): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-012` (cross_session): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-013` (memory_conflict): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-014` (topic_switch): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-017` (reference): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-018` (ellipsis): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-020` (cross_session): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-021` (memory_conflict): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}
- `context-v01-022` (topic_switch): {'recent_window': False, 'summary_recent': True, 'semantic_memory': True, 'adaptive_policy': True}

## Boundary

Context-level scenario benchmark; key-point uses scenario-grounded expected facts and does not equal free-form answer accuracy
