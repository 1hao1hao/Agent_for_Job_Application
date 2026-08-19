from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Sequence

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.base import RetrievalResult
from intern_rag.retrieval.dense import EmbeddingModel


ConnectionFactory = Callable[[], Any]


@dataclass(frozen=True)
class PgVectorIndexConfig:
    """pgvector 索引的版本、表名、维度和 HNSW 参数。"""

    dataset_version: str
    embedding_name: str
    embedding_version: str
    dimensions: int
    table_name: str = "rag_chunk_embeddings"
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64


class PgVectorIndexRepository:
    """把离线 Chunk embedding 持久化到 PostgreSQL/pgvector。

    Repository 只管理索引元数据、向量和 provenance JSON。建库阶段批量 upsert，
    查询阶段使用 cosine distance 与可选 source filter；业务 Chunk 正文仍由版本化
    corpus 提供，避免数据库返回结构与现有 Retriever 契约冲突。
    """

    def __init__(
        self, connection_factory: ConnectionFactory, config: PgVectorIndexConfig
    ) -> None:
        if config.dimensions <= 0:
            raise ValueError("pgvector dimensions must be positive")
        if not config.table_name.replace("_", "").isalnum():
            raise ValueError("unsafe pgvector table name")
        self.connection_factory = connection_factory
        self.config = config

    def migrate(self) -> None:
        """创建 extension、索引表与 HNSW cosine index。"""

        table = self.config.table_name
        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        dataset_version TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        source_type TEXT NOT NULL,
                        embedding_name TEXT NOT NULL,
                        embedding_version TEXT NOT NULL,
                        provenance JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.config.dimensions}) NOT NULL,
                        PRIMARY KEY (dataset_version, chunk_id)
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {table}_hnsw_cosine
                    ON {table} USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {self.config.hnsw_m},
                          ef_construction = {self.config.hnsw_ef_construction})
                    """
                )
            connection.commit()

    def replace(
        self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]
    ) -> None:
        """原子替换当前 dataset version 的全部向量。"""

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length do not match")
        if any(len(vector) != self.config.dimensions for vector in vectors):
            raise ValueError("vector dimensions do not match index config")
        table = self.config.table_name
        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {table} WHERE dataset_version = %s",
                    (self.config.dataset_version,),
                )
                cursor.executemany(
                    f"""
                    INSERT INTO {table} (
                        dataset_version, chunk_id, source_type,
                        embedding_name, embedding_version, provenance, embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    """,
                    [
                        (
                            self.config.dataset_version,
                            chunk.id,
                            chunk.source_type,
                            self.config.embedding_name,
                            self.config.embedding_version,
                            json.dumps(
                                {
                                    "document_id": chunk.metadata.get("document_id", ""),
                                    "source_url": chunk.metadata.get("source_url", ""),
                                },
                                ensure_ascii=False,
                            ),
                            _vector_literal(vector),
                        )
                        for chunk, vector in zip(chunks, vectors)
                    ],
                )
            connection.commit()

    def query(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int,
        source_types: set[str] | None = None,
        exact: bool = False,
    ) -> list[tuple[str, float]]:
        """返回 chunk id 与 cosine similarity；`exact` 用于 ANN 对照实验。"""

        if top_k <= 0:
            return []
        if len(query_vector) != self.config.dimensions:
            raise ValueError("query vector dimensions do not match index")
        table = self.config.table_name
        filters = ["dataset_version = %s"]
        params: list[object] = [self.config.dataset_version]
        if source_types is not None:
            filters.append("source_type = ANY(%s)")
            params.append(sorted(source_types))
        vector = _vector_literal(query_vector)
        params.extend([vector, top_k])
        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                if exact:
                    cursor.execute("SET LOCAL enable_indexscan = off")
                cursor.execute(
                    f"""
                    SELECT chunk_id, 1 - (embedding <=> %s::vector) AS score
                    FROM {table}
                    WHERE {' AND '.join(filters)}
                    ORDER BY embedding <=> %s::vector, chunk_id
                    LIMIT %s
                    """.replace(
                        "SELECT chunk_id, 1 - (embedding <=> %s::vector)",
                        "SELECT chunk_id, 1 - (embedding <=> %s::vector)",
                    ),
                    [vector, *params[:-2], vector, top_k],
                )
                return [(str(row[0]), float(row[1])) for row in cursor.fetchall()]

    def count(self) -> int:
        """返回当前 dataset version 的索引行数。"""

        with self.connection_factory() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.config.table_name} "
                    "WHERE dataset_version = %s",
                    (self.config.dataset_version,),
                )
                return int(cursor.fetchone()[0])


class PgVectorRetriever:
    """使用现有 EmbeddingModel 编码 Query，并从 pgvector 返回统一结果。"""

    def __init__(
        self,
        repository: PgVectorIndexRepository,
        embedding_model: EmbeddingModel,
        *,
        exact: bool = False,
    ) -> None:
        config = repository.config
        if config.embedding_name != embedding_model.name:
            raise ValueError("pgvector index and embedding model name do not match")
        if config.embedding_version != embedding_model.version:
            raise ValueError("pgvector index and embedding model version do not match")
        self.repository = repository
        self.embedding_model = embedding_model
        self.exact = exact

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """编码一次 Query，查询持久化索引并恢复 Chunk 对象。"""

        if not query.strip() or top_k <= 0:
            return []
        vector = self.embedding_model.encode([query])[0]
        rows = self.repository.query(
            vector, top_k=top_k, source_types=source_types, exact=self.exact
        )
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        results: list[RetrievalResult] = []
        for chunk_id, score in rows:
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    rank=len(results) + 1,
                    chunk=chunk,
                    reason=f"pgvector_cosine={score:.6f}, exact={self.exact}",
                    details={"dense_score": score, "pgvector_exact": self.exact},
                )
            )
        return results


def psycopg_connection_factory(database_url: str) -> ConnectionFactory:
    """根据 DATABASE_URL 构造延迟连接工厂，密钥不会写入对象日志。"""

    def connect() -> Any:
        import psycopg

        return psycopg.connect(database_url)

    return connect


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.10g}" for value in vector) + "]"
