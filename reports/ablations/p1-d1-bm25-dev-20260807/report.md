# P1-D1 BM25 Dev Ablation

- Dataset: `evalrag_v0.2`
- Split: `dev`
- Cases: 80
- 相同标签、Router source filter 与 top-k；本报告不代表答案准确率。

| Retriever | Recall@3 | Recall@5 | MRR | P50 ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| keyword | 0.7333 | 0.8056 | 0.8597 | 6.990 | 10.701 |
| bm25 | 0.5556 | 0.7889 | 0.6833 | 0.509 | 1.130 |
| dense | 0.7278 | 0.8083 | 0.6742 | 96.946 | 196.223 |
| bm25_hybrid | 0.7389 | 0.8000 | 0.7214 | 96.992 | 696.178 |

## Strategy Differences

- `v02_semantic_010` (semantic_paraphrase): 简历优势：怎样生成前判断材料是否足够，并避免证据不足仍强行回答？；MRR keyword/bm25/dense/bm25_hybrid=1.000/0.250/0.000/0.200。
- `v02_semantic_014` (semantic_paraphrase): 面试怎么回答：怎样让智能体可靠执行外部操作，并避免参数错误产生副作用？；MRR keyword/bm25/dense/bm25_hybrid=1.000/0.000/0.500/0.000。
- `v02_semantic_015` (semantic_paraphrase): 面试怎么回答：怎样用统一接口切换推理后端，并避免供应商响应格式不同？；MRR keyword/bm25/dense/bm25_hybrid=1.000/0.000/0.500/0.000。
- `v02_semantic_016` (semantic_paraphrase): 面试怎么回答：怎样生成前判断材料是否足够，并避免证据不足仍强行回答？；MRR keyword/bm25/dense/bm25_hybrid=1.000/0.000/0.200/0.000。
- `v02_semantic_017` (semantic_paraphrase): 面试怎么回答：怎样统一接收手工、CSV 和 JSON 岗位，并避免平台字段不一致？；MRR keyword/bm25/dense/bm25_hybrid=1.000/0.000/0.000/0.000。
- `v02_single_015` (single_source): 面试准备中，请求追踪应该怎样解释和排查？；MRR keyword/bm25/dense/bm25_hybrid=1.000/1.000/0.000/0.500。
- `v02_multi_012` (multi_source): 结合岗位、简历和面试资料准备请求追踪追问；MRR keyword/bm25/dense/bm25_hybrid=1.000/1.000/0.250/0.500。
- `v02_multi_002` (multi_source): 结合岗位、简历和面试资料准备混合检索追问；MRR keyword/bm25/dense/bm25_hybrid=0.500/1.000/0.500/0.500。
- `v02_multi_007` (multi_source): 结合岗位、简历和面试资料准备引用校验追问；MRR keyword/bm25/dense/bm25_hybrid=0.500/1.000/0.500/1.000。
- `v02_multi_017` (multi_source): 结合岗位、简历和面试资料准备意图路由追问；MRR keyword/bm25/dense/bm25_hybrid=0.500/1.000/1.000/1.000。
