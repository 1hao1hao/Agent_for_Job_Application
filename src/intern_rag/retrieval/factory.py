from __future__ import annotations

from pathlib import Path
import os

from intern_rag.graph import load_knowledge_graph
from intern_rag.graph.neo4j import create_neo4j_repository
from intern_rag.retrieval.adaptive import (
    AdaptiveRetriever,
    AdaptiveRetrieverConfig,
)
from intern_rag.retrieval.base import Retriever
from intern_rag.retrieval.bm25 import BM25Retriever, load_bm25_index
from intern_rag.retrieval.dense import DenseRetriever, load_dense_index
from intern_rag.retrieval.hybrid import HybridRetriever
from intern_rag.retrieval.graph import GraphRetriever, GraphVectorRetriever
from intern_rag.retrieval.keyword import retrieve_top_k
from intern_rag.retrieval.pgvector import (
    PgVectorIndexConfig,
    PgVectorIndexRepository,
    PgVectorRetriever,
    psycopg_connection_factory,
)
from intern_rag.retrieval.rerank import (
    CrossEncoderRerankScorer,
    ChineseTokenOverlapScorer,
    RerankRetriever,
    RerankScorer,
)


def build_retriever_from_config(config: dict[str, object]) -> Retriever:
    """根据运行配置构造统一 Retriever，避免 Pipeline/Runner 分支膨胀。"""

    retriever_name = str(config.get("retriever_name", ""))
    if retriever_name == "keyword":
        return retrieve_top_k
    graph_retriever: GraphRetriever | None = None
    if retriever_name in {
        "graph", "graph_vector", "graph_adaptive", "neo4j_graph_adaptive"
    }:
        if str(config.get("graph_backend", "file")) == "neo4j":
            repository = create_neo4j_repository(
                os.environ["NEO4J_URI"],
                os.environ["NEO4J_USER"],
                os.environ["NEO4J_PASSWORD"],
                database=str(config.get("neo4j_database", "neo4j")),
            )
            graph = repository.load(str(config["dataset_version"]))
            repository.close()
        else:
            graph_path = Path(str(config.get("graph_index_path", "")))
            if not str(graph_path) or not graph_path.exists():
                raise ValueError(f"graph index does not exist: {graph_path}")
            graph = load_knowledge_graph(graph_path)
        graph_retriever = GraphRetriever(
            graph,
            max_hops=int(config.get("graph_max_hops", 2)),
            max_nodes=int(config.get("graph_max_nodes", 80)),
            timeout_ms=float(config.get("graph_timeout_ms", 50.0)),
        )
        if retriever_name == "graph":
            return graph_retriever
    bm25: BM25Retriever | None = None
    if retriever_name in {
        "bm25", "bm25_hybrid", "adaptive", "graph_vector", "graph_adaptive",
        "neo4j_graph_adaptive", "pgvector_hybrid"
    }:
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
        "dense", "hybrid", "hybrid_rerank", "bm25_hybrid", "adaptive",
        "graph_vector",
        "graph_adaptive", "neo4j_graph_adaptive", "pgvector_dense",
        "pgvector_exact", "pgvector_hybrid"
    }:
        raise ValueError(f"unknown retriever: {retriever_name}")

    index_dir = Path(str(config.get("index_dir", "")))
    if not str(index_dir) or not index_dir.exists():
        raise ValueError(f"dense index does not exist: {index_dir}")
    index, model = load_dense_index(index_dir)
    if retriever_name in {"pgvector_dense", "pgvector_exact", "pgvector_hybrid"}:
        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise ValueError("DATABASE_URL is required for pgvector retriever")
        repository = PgVectorIndexRepository(
            psycopg_connection_factory(database_url),
            PgVectorIndexConfig(
                dataset_version=str(config["dataset_version"]),
                embedding_name=index.metadata.embedding_name,
                embedding_version=index.metadata.embedding_version,
                dimensions=index.metadata.dimensions,
                table_name=str(config.get("pgvector_table", "rag_chunk_embeddings")),
            ),
        )
        dense: Retriever = PgVectorRetriever(
            repository, model, exact=retriever_name == "pgvector_exact"
        )
    else:
        dense = DenseRetriever(index, model)
    if retriever_name in {"dense", "pgvector_dense", "pgvector_exact"}:
        return dense
    use_bm25_lexical = (
        retriever_name in {"bm25_hybrid", "pgvector_hybrid"}
        or (
            retriever_name in {
                "adaptive", "graph_vector", "graph_adaptive",
                "neo4j_graph_adaptive"
            }
            and str(config.get("adaptive_lexical", "keyword")) == "bm25"
        )
    )
    lexical_retriever = bm25 if use_bm25_lexical else retrieve_top_k
    hybrid = HybridRetriever(
        lexical_retriever,
        dense,
        rrf_k=int(config.get("rrf_k", 60)),
        candidate_multiplier=int(config.get("candidate_multiplier", 4)),
        lexical_name="bm25" if use_bm25_lexical else "keyword",
    )
    if retriever_name in {"hybrid", "bm25_hybrid", "pgvector_hybrid"}:
        return hybrid
    if retriever_name == "graph_vector":
        if graph_retriever is None:
            raise ValueError("graph_vector retriever requires graph index")
        return GraphVectorRetriever(
            graph_retriever,
            hybrid,
            rrf_k=int(config.get("graph_rrf_k", 60)),
            candidate_multiplier=int(config.get("graph_candidate_multiplier", 4)),
        )
    scorer = _build_rerank_scorer(config)
    if retriever_name in {"adaptive", "graph_adaptive", "neo4j_graph_adaptive"}:
        if bm25 is None:
            raise ValueError("adaptive retriever requires BM25 index")
        graph_vector = (
            GraphVectorRetriever(
                graph_retriever,
                hybrid,
                rrf_k=int(config.get("graph_rrf_k", 60)),
                candidate_multiplier=int(
                    config.get("graph_candidate_multiplier", 4)
                ),
            )
            if graph_retriever is not None
            else None
        )
        return AdaptiveRetriever(
            {"bm25": bm25, "dense": dense, "hybrid": hybrid},
            scorer,
            config=AdaptiveRetrieverConfig(
                confidence_threshold=float(
                    config.get("adaptive_confidence_threshold", 0.55)
                ),
                candidate_k=int(config.get("reranker_candidate_k", 20)),
                rerank_rrf_k=int(config.get("adaptive_rerank_rrf_k", 60)),
                original_rank_weight=float(
                    config.get("adaptive_original_rank_weight", 2.0)
                ),
                rerank_rank_weight=float(
                    config.get("adaptive_rerank_rank_weight", 1.0)
                ),
                rerank_policy=str(
                    config.get("adaptive_rerank_policy", "low_confidence")
                ),  # type: ignore[arg-type]
                force_strategy=(
                    str(config["adaptive_force_strategy"])  # type: ignore[arg-type]
                    if config.get("adaptive_force_strategy") is not None
                    else None
                ),
            ),
            graph_retriever=graph_vector,
        )
    return RerankRetriever(
        hybrid,
        scorer,
        candidate_k=int(config.get("reranker_candidate_k", 20)),
    )


def _build_rerank_scorer(config: dict[str, object]) -> RerankScorer:
    """按配置构造轻量或真实 CrossEncoder scorer。"""

    scorer_kind = str(config.get("reranker_kind", "cross_encoder"))
    if scorer_kind == "token_overlap":
        return ChineseTokenOverlapScorer()
    if scorer_kind == "cross_encoder":
        return CrossEncoderRerankScorer(
            model_name=str(config["reranker_model"]),
            revision=str(config["reranker_revision"]),
            device=str(config.get("reranker_device", "cpu")),
            local_files_only=bool(config.get("local_files_only", True)),
            batch_size=int(config.get("reranker_batch_size", 16)),
        )
    raise ValueError(f"unknown reranker_kind: {scorer_kind}")
