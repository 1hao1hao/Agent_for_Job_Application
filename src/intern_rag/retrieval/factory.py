from __future__ import annotations

from pathlib import Path

from intern_rag.retrieval.base import Retriever
from intern_rag.retrieval.bm25 import BM25Retriever, load_bm25_index
from intern_rag.retrieval.dense import DenseRetriever, load_dense_index
from intern_rag.retrieval.hybrid import HybridRetriever
from intern_rag.retrieval.keyword import retrieve_top_k
from intern_rag.retrieval.rerank import (
    CrossEncoderRerankScorer,
    ChineseTokenOverlapScorer,
    RerankRetriever,
)


def build_retriever_from_config(config: dict[str, object]) -> Retriever:
    """根据运行配置构造统一 Retriever，避免 Pipeline/Runner 分支膨胀。"""

    retriever_name = str(config.get("retriever_name", ""))
    if retriever_name == "keyword":
        return retrieve_top_k
    bm25: BM25Retriever | None = None
    if retriever_name in {"bm25", "bm25_hybrid"}:
        bm25_index_path = Path(str(config.get("bm25_index_path", "")))
        if not str(bm25_index_path) or not bm25_index_path.exists():
            raise ValueError(f"BM25 index does not exist: {bm25_index_path}")
        bm25 = BM25Retriever(
            load_bm25_index(bm25_index_path),
            k1=float(config.get("k1", 1.5)),
            b=float(config.get("b", 0.75)),
        )
        if retriever_name == "bm25":
            return bm25
    if retriever_name not in {
        "dense", "hybrid", "hybrid_rerank", "bm25_hybrid"
    }:
        raise ValueError(f"unknown retriever: {retriever_name}")

    index_dir = Path(str(config.get("index_dir", "")))
    if not str(index_dir) or not index_dir.exists():
        raise ValueError(f"dense index does not exist: {index_dir}")
    index, model = load_dense_index(index_dir)
    dense = DenseRetriever(index, model)
    if retriever_name == "dense":
        return dense
    lexical_retriever = bm25 if bm25 is not None else retrieve_top_k
    hybrid = HybridRetriever(
        lexical_retriever,
        dense,
        rrf_k=int(config.get("rrf_k", 60)),
        candidate_multiplier=int(config.get("candidate_multiplier", 4)),
        lexical_name="bm25" if bm25 is not None else "keyword",
    )
    if retriever_name in {"hybrid", "bm25_hybrid"}:
        return hybrid
    scorer_kind = str(config.get("reranker_kind", "cross_encoder"))
    if scorer_kind == "token_overlap":
        scorer = ChineseTokenOverlapScorer()
    elif scorer_kind == "cross_encoder":
        scorer = CrossEncoderRerankScorer(
            model_name=str(config["reranker_model"]),
            revision=str(config["reranker_revision"]),
            device=str(config.get("reranker_device", "cpu")),
            local_files_only=bool(config.get("local_files_only", True)),
            batch_size=int(config.get("reranker_batch_size", 16)),
        )
    else:
        raise ValueError(f"unknown reranker_kind: {scorer_kind}")
    return RerankRetriever(
        hybrid,
        scorer,
        candidate_k=int(config.get("reranker_candidate_k", 20)),
    )
