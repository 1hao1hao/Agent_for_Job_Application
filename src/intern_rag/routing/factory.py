from __future__ import annotations

from pathlib import Path

from intern_rag.retrieval.dense import load_dense_index
from intern_rag.routing.base import Router
from intern_rag.routing.hybrid import HybridRouter, HybridRouterConfig
from intern_rag.routing.intent_router import route_query
from intern_rag.routing.semantic import SemanticRouter, SemanticRouterConfig
from intern_rag.routing.feedback import (
    FeedbackRouter,
    JsonlRouterFeedbackStore,
    RouterVersionRegistry,
)


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


def build_active_router_from_registry(registry_path: Path) -> Router:
    """从版本 registry 构建 active Router，并应用审核后的离线反馈层。"""

    state = RouterVersionRegistry(registry_path).load()
    active = state.get("active_version")
    if not active:
        raise ValueError("router registry has no active version")
    versions = {
        str(item["version"]): item
        for item in state.get("versions", [])
    }
    if active not in versions:
        raise ValueError(f"active router version is missing: {active}")
    config = dict(versions[str(active)]["config"])
    router = build_router_from_config(config)
    if config.get("feedback_strategy") == "confirmed_anchor_override":
        feedback_path = Path(str(config.get("feedback_dataset", "")))
        if not feedback_path.exists():
            raise ValueError(f"router feedback dataset does not exist: {feedback_path}")
        router = FeedbackRouter(
            router, JsonlRouterFeedbackStore(feedback_path).read_all()
        )
    return router
