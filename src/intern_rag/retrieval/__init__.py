"""统一检索接口及 Keyword、Dense、Hybrid 实现。"""

from intern_rag.retrieval.base import RetrievalResult, Retriever
from intern_rag.retrieval.bm25 import (
    BM25Index,
    BM25Retriever,
    build_bm25_index,
    load_bm25_index,
    save_bm25_index,
    tokenize_bm25,
)
from intern_rag.retrieval.dense import (
    DenseIndex,
    DenseIndexMetadata,
    DenseRetriever,
    EmbeddingModel,
    SklearnLsaEmbedder,
    SentenceTransformerEmbedder,
    build_dense_index,
    build_pretrained_dense_index,
    load_dense_index,
    save_dense_index,
)
from intern_rag.retrieval.hybrid import HybridRetriever
from intern_rag.retrieval.rerank import (
    CrossEncoderRerankScorer,
    ChineseTokenOverlapScorer,
    FakeRerankScorer,
    RerankRetriever,
    RerankScorer,
)
from intern_rag.retrieval.factory import build_retriever_from_config

from intern_rag.retrieval.keyword import (
    retrieve_top_k,
    score_chunk,
    tokenize_text,
)

__all__ = [
    "RetrievalResult",
    "Retriever",
    "BM25Index",
    "BM25Retriever",
    "build_bm25_index",
    "load_bm25_index",
    "save_bm25_index",
    "tokenize_bm25",
    "DenseIndex",
    "DenseIndexMetadata",
    "DenseRetriever",
    "EmbeddingModel",
    "SklearnLsaEmbedder",
    "SentenceTransformerEmbedder",
    "HybridRetriever",
    "CrossEncoderRerankScorer",
    "ChineseTokenOverlapScorer",
    "FakeRerankScorer",
    "RerankRetriever",
    "RerankScorer",
    "build_dense_index",
    "build_pretrained_dense_index",
    "load_dense_index",
    "save_dense_index",
    "build_retriever_from_config",
    "retrieve_top_k",
    "score_chunk",
    "tokenize_text",
]
