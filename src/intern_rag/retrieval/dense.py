from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Protocol, Sequence

from intern_rag.ingestion import Chunk
from intern_rag.retrieval.base import RetrievalResult


class EmbeddingModel(Protocol):
    """Dense Retriever 依赖的最小文本向量模型接口。"""

    name: str
    version: str

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本编码为等长向量。"""


@dataclass(frozen=True)
class DenseIndexMetadata:
    """持久化索引的版本、模型和维度信息。"""

    dataset_version: str
    embedding_name: str
    embedding_version: str
    dimensions: int
    chunk_count: int
    created_at: str


@dataclass(frozen=True)
class DenseIndex:
    """离线生成的 Chunk 向量及其稳定 ID。"""

    metadata: DenseIndexMetadata
    chunk_ids: list[str]
    vectors: list[list[float]]


class SklearnLsaEmbedder:
    """中文字符 TF-IDF + LSA 的轻量本地稠密向量模型。

    该模型用于建立可离线复现的 Dense baseline，不宣称等价于 BGE 等
    预训练语义模型。字符 n-gram 适配中文，LSA 把稀疏词面特征投影为稠密向量。
    """

    name = "sklearn-char-tfidf-lsa"

    def __init__(self, vectorizer: Any, projector: Any, version: str) -> None:
        self.vectorizer = vectorizer
        self.projector = projector
        self.version = version

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        *,
        dimensions: int = 128,
        max_features: int = 4096,
    ) -> "SklearnLsaEmbedder":
        """只在离线建索引阶段拟合语料特征与 LSA 投影。"""

        if not texts:
            raise ValueError("cannot fit embedding model on empty texts")
        if dimensions <= 0 or max_features <= 1:
            raise ValueError("dimensions and max_features must be positive")

        import sklearn
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            max_features=max_features,
            sublinear_tf=True,
        )
        sparse_vectors = vectorizer.fit_transform(texts)
        max_dimensions = min(
            dimensions,
            max(1, sparse_vectors.shape[0] - 1),
            max(1, sparse_vectors.shape[1] - 1),
        )
        projector = TruncatedSVD(
            n_components=max_dimensions,
            random_state=42,
        )
        projector.fit(sparse_vectors)
        return cls(
            vectorizer,
            projector,
            version=f"v1-sklearn-{sklearn.__version__}",
        )

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """使用已经拟合的特征空间编码文档或 Query。"""

        if not texts:
            return []
        sparse_vectors = self.vectorizer.transform(texts)
        dense_vectors = self.projector.transform(sparse_vectors)
        return [_normalize(vector.tolist()) for vector in dense_vectors]

    def save(self, path: Path) -> None:
        """保存 Query 编码所需的 vectorizer 和 projector。"""

        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "projector": self.projector,
                "version": self.version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SklearnLsaEmbedder":
        """读取本项目离线生成的可信模型工件。"""

        import joblib

        payload = joblib.load(path)
        return cls(
            payload["vectorizer"],
            payload["projector"],
            str(payload["version"]),
        )


class SentenceTransformerEmbedder:
    """使用固定 revision 的预训练 SentenceTransformer 编码中文文本。"""

    def __init__(
        self,
        model_name: str,
        revision: str,
        *,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if not model_name.strip() or not revision.strip():
            raise ValueError("model_name and revision must not be empty")
        self.name = model_name
        self.version = revision
        self.device = device
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            local_files_only=local_files_only,
        )

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码并归一化，便于用点积计算余弦相似度。"""

        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

class DenseRetriever:
    """查询时只编码 Query，并与离线文档向量计算余弦相似度。"""

    def __init__(self, index: DenseIndex, embedding_model: EmbeddingModel) -> None:
        if index.metadata.embedding_name != embedding_model.name:
            raise ValueError("dense index and embedding model name do not match")
        if index.metadata.embedding_version != embedding_model.version:
            raise ValueError("dense index and embedding model version do not match")
        self.index = index
        self.embedding_model = embedding_model

    def __call__(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
        source_types: set[str] | None = None,
    ) -> list[RetrievalResult]:
        """对 Query 编码一次，返回过滤后的 Dense top-k。"""

        if not query.strip() or top_k <= 0:
            return []
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        query_vector = self.embedding_model.encode([query])[0]
        candidates: list[tuple[float, str, Chunk]] = []
        for chunk_id, vector in zip(self.index.chunk_ids, self.index.vectors):
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            if source_types is not None and chunk.source_type not in source_types:
                continue
            score = _dot(query_vector, vector)
            if score > 0:
                candidates.append((score, chunk_id, chunk))

        candidates.sort(key=lambda item: (-item[0], item[2].source_type, item[1]))
        return [
            RetrievalResult(
                chunk_id=chunk_id,
                score=score,
                rank=rank,
                chunk=chunk,
                reason=f"dense_cosine={score:.6f}",
                details={"dense_score": score},
            )
            for rank, (score, chunk_id, chunk) in enumerate(
                candidates[:top_k], start=1
            )
        ]


def build_dense_index(
    chunks: list[Chunk],
    *,
    dataset_version: str,
    dimensions: int = 128,
    max_features: int = 4096,
) -> tuple[DenseIndex, SklearnLsaEmbedder]:
    """离线拟合轻量 embedding，并为所有 Chunk 生成归一化向量。"""

    if not chunks:
        raise ValueError("cannot build dense index from empty chunks")
    model = SklearnLsaEmbedder.fit(
        [chunk.text for chunk in chunks],
        dimensions=dimensions,
        max_features=max_features,
    )
    vectors = model.encode([chunk.text for chunk in chunks])
    index = DenseIndex(
        metadata=DenseIndexMetadata(
            dataset_version=dataset_version,
            embedding_name=model.name,
            embedding_version=model.version,
            dimensions=len(vectors[0]),
            chunk_count=len(chunks),
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        chunk_ids=[chunk.id for chunk in chunks],
        vectors=vectors,
    )
    return index, model


def build_pretrained_dense_index(
    chunks: list[Chunk],
    *,
    dataset_version: str,
    model_name: str,
    revision: str,
    device: str = "cpu",
) -> tuple[DenseIndex, SentenceTransformerEmbedder]:
    """用 固定版本的 预训练模型 离线编码 全部 Chunk。"""

    if not chunks:
        raise ValueError("cannot build dense index from empty chunks")
    model = SentenceTransformerEmbedder(model_name, revision, device=device)
    vectors = model.encode([chunk.text for chunk in chunks])
    index = DenseIndex(
        metadata=DenseIndexMetadata(
            dataset_version=dataset_version,
            embedding_name=model.name,
            embedding_version=model.version,
            dimensions=len(vectors[0]),
            chunk_count=len(chunks),
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        chunk_ids=[chunk.id for chunk in chunks],
        vectors=vectors,
    )
    return index, model


def save_dense_index(
    index: DenseIndex,
    model: EmbeddingModel,
    index_dir: Path,
) -> None:
    """保存可审计的索引 JSON 与 Query encoder 工件。"""

    index_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": asdict(index.metadata),
        "chunk_ids": index.chunk_ids,
        "vectors": index.vectors,
    }
    (index_dir / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if isinstance(model, SklearnLsaEmbedder):
        model.save(index_dir / "model.joblib")
        model_kind = "sklearn_lsa"
    elif isinstance(model, SentenceTransformerEmbedder):
        model_kind = "sentence_transformer"
    else:
        raise TypeError("unsupported embedding model for persistence")
    (index_dir / "model_config.json").write_text(
        json.dumps({
            "kind": model_kind,
            "name": model.name,
            "version": model.version,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_dense_index(index_dir: Path) -> tuple[DenseIndex, EmbeddingModel]:
    """加载离线索引，并检查模型和索引的版本是否一致。"""

    payload = json.loads((index_dir / "index.json").read_text(encoding="utf-8"))
    index = DenseIndex(
        metadata=DenseIndexMetadata(**payload["metadata"]),
        chunk_ids=[str(value) for value in payload["chunk_ids"]],
        vectors=[
            [float(component) for component in vector]
            for vector in payload["vectors"]
        ],
    )
    model_config_path = index_dir / "model_config.json"
    if model_config_path.exists():
        model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
        if model_config["kind"] == "sentence_transformer":
            model: EmbeddingModel = SentenceTransformerEmbedder(
                str(model_config["name"]),
                str(model_config["version"]),
                local_files_only=True,
            )
        elif model_config["kind"] == "sklearn_lsa":
            model = SklearnLsaEmbedder.load(index_dir / "model.joblib")
        else:
            raise ValueError(f"unknown embedding model kind: {model_config['kind']}")
    else:
        model = SklearnLsaEmbedder.load(index_dir / "model.joblib")
    if index.metadata.chunk_count != len(index.chunk_ids):
        raise ValueError("dense index metadata chunk_count is inconsistent")
    return index, model


def _normalize(vector: list[float]) -> list[float]:
    """把向量归一化，使点积等价于余弦相似度。"""

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    """计算两个等长归一化向量的点积。"""

    if len(left) != len(right):
        raise ValueError("query and document vector dimensions do not match")
    return sum(a * b for a, b in zip(left, right))
