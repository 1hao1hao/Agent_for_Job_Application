from __future__ import annotations

from typing import Protocol

from intern_rag.routing.intent_router import RouteDecision


class Router(Protocol):
    """Rule、Semantic 与 Hybrid Router 共同遵守的最小接口。"""

    def __call__(self, query: str) -> RouteDecision:
        """根据 query 返回意图、来源和可解释路由信息。"""
