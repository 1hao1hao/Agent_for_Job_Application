from __future__ import annotations

from pathlib import Path

from intern_rag.retrieval.dense import load_dense_index
from intern_rag.routing.base import Router
from intern_rag.routing.hybrid import HybridRouter, HybridRouterConfig
from intern_rag.routing.intent_router import route_query
from intern_rag.routing.semantic import SemanticRouter, SemanticRouterConfig


def build_router_from_config(config: dict[str, object]) -> Router:
    """根据配置构造 Router，避免 Pipeline 和评测脚本写策略分支。"""

    name = str(config.get("router_name", "rule"))
    if name == "rule":
        return route_query
    if name not in {"semantic", "hybrid"}:
        raise ValueError(f"unknown router: {name}")

    index_dir = Path(str(config.get("embedding_index_dir", "")))
    if not index_dir.exists():
        raise ValueError(f"embedding index does not exist: {index_dir}")
    _, embedding_model = load_dense_index(index_dir)
    semantic = SemanticRouter(
        embedding_model,
        config=SemanticRouterConfig(
            min_score=float(config.get("semantic_min_score", 0.45)),
            min_margin=float(config.get("semantic_min_margin", 0.02)),
        ),
    )
    if name == "semantic":
        return semantic
    return HybridRouter(
        route_query,
        semantic,
        config=HybridRouterConfig(
            semantic_override_score=float(
                config.get("semantic_override_score", 0.55)
            ),
            semantic_override_margin=float(
                config.get("semantic_override_margin", 0.04)
            ),
            max_weak_rule_keywords=int(
                config.get("max_weak_rule_keywords", 2)
            ),
        ),
    )
