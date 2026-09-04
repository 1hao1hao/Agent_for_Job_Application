# Evidence Gate Low-Confidence Retry Dev Ablation

`evalrag_v0.3/dev`: 160 cases, 120 answerable. Test split was not read.

| Gate | FAR | FRR | Answerable acceptance | Retry rate | Retry success | Evidence recovery | E2E success |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural_v2 | 0.4327 | 0.0000 | 0.8417 | 0.1187 | 0.0000 | 0.0000 | 0.6000 |
| retry_v2_1 | 0.3274 | 0.0000 | 0.7000 | 0.6500 | 0.3077 | 0.0000 | 0.5437 |

Raw top-1 score is used only to trigger one retry. It never hard-rejects structurally sufficient evidence after retry.

## Decision

Rejected as the default Gate: retries increased but recovered no missing structural evidence, while deterministic E2E success regressed.
