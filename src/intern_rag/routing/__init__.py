"""可配置的规则、语义与混合意图路由模块。"""

from intern_rag.routing.base import Router
from intern_rag.routing.factory import build_active_router_from_registry, build_router_from_config
from intern_rag.routing.hybrid import HybridRouter, HybridRouterConfig
from intern_rag.routing.intent_router import (
    INTENT_TO_SOURCES,
    Intent,
    RouteDecision,
    route_query,
)
from intern_rag.routing.semantic import (
    DEFAULT_INTENT_PROTOTYPES,
    SemanticRouter,
    SemanticRouterConfig,
)

__all__ = [
    "INTENT_TO_SOURCES",
    "DEFAULT_INTENT_PROTOTYPES",
    "HybridRouter",
    "HybridRouterConfig",
    "Intent",
    "RouteDecision",
    "Router",
    "SemanticRouter",
    "SemanticRouterConfig",
    "build_router_from_config",
    "build_active_router_from_registry",
    "route_query",
]
