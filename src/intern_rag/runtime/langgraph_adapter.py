from __future__ import annotations

from typing import Callable, Protocol


class CompiledGraph(Protocol):
    def invoke(self, state: dict[str, object]) -> dict[str, object]: ...


def build_langgraph_app(
    handler: Callable[[dict[str, object]], dict[str, object]],
) -> CompiledGraph:
    """编译一次单节点 LangGraph，供公平的重复调用延迟对照。"""

    from langgraph.graph import END, StateGraph

    graph = StateGraph(dict)
    graph.add_node("execute", handler)
    graph.set_entry_point("execute")
    graph.add_edge("execute", END)
    return graph.compile()


def run_with_langgraph(initial_state: dict[str, object], handler: Callable[[dict[str, object]], dict[str, object]]) -> dict[str, object]:
    """用单节点 LangGraph 对照同一 handler，不重写算法模块。"""

    return dict(build_langgraph_app(handler).invoke(dict(initial_state)))
