CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_chunk_embeddings (
    dataset_version TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    embedding_name TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(512) NOT NULL,
    PRIMARY KEY (dataset_version, chunk_id)
);

CREATE INDEX IF NOT EXISTS rag_chunk_embeddings_hnsw_cosine
ON rag_chunk_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
