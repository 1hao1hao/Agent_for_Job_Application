# EvalRAG Corpus v0.3 Data Quality Report

- Dataset: `evalrag_v0.3`
- Collection date: `2026-08-16`
- Input / retained / rejected: 669 / 658 / 11 documents
- Chunks: 4208; length P50/P95: 283 / 420 characters
- Scope: 558 public, 100 anonymized user-owned documents
- Methods: 319 dataset imports, 239 fixed-revision repository imports, 100 user-owned

| Source | Documents |
|---|---:|
| JD | 339 |
| Interview | 120 |
| Project logs | 159 |
| Resume | 20 |
| User profile | 20 |

Exact hash rejected 6 documents and SimHash rejected 5 near duplicates. There are no empty
documents after cleaning; template-line ratio is 3.06%. Full provenance and rejection reasons
are available in `corpus_manifest_v0.3.jsonl`, `collection_report.json`, and `rejected.jsonl`.

Public records preserve source URL/domain, fixed revision where available, collection date,
public status, license status, review method, and content hash. Evaluation labels are
corpus-grounded AI-assisted labels, not human-reviewed ground truth.
