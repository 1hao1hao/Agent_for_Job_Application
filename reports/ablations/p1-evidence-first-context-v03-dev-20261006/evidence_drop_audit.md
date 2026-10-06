# Evidence-first Context Audit

- Dataset: `evalrag_v0.3/dev`.
- Same candidates and token budget as the historical run.

| Case | Required sources | Before covered | After covered | Before tokens | After tokens |
|---|---|---|---|---:|---:|
| v03_cross_source_010 | resume, user_profile | False | True | 1790 | 1773 |
| v03_cross_source_026 | project_logs, resume | False | True | 1608 | 1664 |

Context evidence drops: 2 -> 0.
