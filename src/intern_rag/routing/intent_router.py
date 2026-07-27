from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Intent = Literal[
    "analyze_jd",
    "match_resume",
    "interview_prepare",
    "project_explain",
    "application_plan",
    "unknown",
]

INTENT_TO_SOURCES: dict[Intent, list[str]] = {
    "analyze_jd": ["jd"],
    "match_resume": ["jd", "resume"],
    "interview_prepare": ["interview", "jd", "resume"],
    "project_explain": ["project_logs", "resume"],
    "application_plan": ["user_profile", "jd", "resume"],
    "unknown": [],
}

INTENT_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    "analyze_jd": (
        "jd",
        "岗位",
        "职位",
        "职责",
        "要求",
        "招聘",
        "公司",
        "实习天数",
        "分析岗位",
    ),
    "match_resume": (
        "简历",
        "匹配",
        "适配",
        "差距",
        "优势",
        "经历",
        "项目经历",
        "怎么改简历",
        "胜任",
    ),
    "interview_prepare": (
        "面试",
        "八股",
        "面经",
        "准备",
        "常问",
        "追问",
        "怎么回答",
        "复习",
    ),
    "project_explain": (
        "项目",
        "项目日志",
        "怎么讲",
        "亮点",
        "难点",
        "架构",
        "模块",
        "实现细节",
    ),
    "application_plan": (
        "投递",
        "求职",
        "规划",
        "计划",
        "城市",
        "目标",
        "用户画像",
        "时间安排",
        "申请",
    ),
}


@dataclass(frozen=True)
class RouteDecision:
    """一次 query 路由的结果。

    intent 表示系统理解的用户意图；
    routed_sources 表示后续 retrieval 应优先检索哪些知识源；
    matched_keywords 用于解释规则为什么命中。
    """

    intent: Intent
    routed_sources: list[str]
    matched_keywords: list[str]


def route_query(query: str) -> RouteDecision:
    """用简单关键词规则判断 query 意图并输出检索来源。

    第一版 router 不调用模型，只做可解释的规则匹配。
    若多个 intent 同时命中，选择命中关键词数量最多的 intent；
    若完全没有命中则返回 unknown。
    """

    normalized_query = query.lower().strip()
    if not normalized_query:
        return _build_decision("unknown", [])

    best_intent: Intent = "unknown"
    best_keywords: list[str] = []

    for intent, keywords in INTENT_KEYWORDS.items():
        matched_keywords = [
            keyword for keyword in keywords if keyword.lower() in normalized_query
        ]
        if _is_better_match(matched_keywords, best_keywords, intent, best_intent):
            best_intent = intent
            best_keywords = matched_keywords

    if not best_keywords:
        return _build_decision("unknown", [])
    return _build_decision(best_intent, best_keywords)


def _build_decision(intent: Intent, matched_keywords: list[str]) -> RouteDecision:
    """根据 intent 构造 RouteDecision。"""

    return RouteDecision(
        intent=intent,
        routed_sources=INTENT_TO_SOURCES[intent],
        matched_keywords=matched_keywords,
    )


def _is_better_match(
    matched_keywords: list[str],
    best_keywords: list[str],
    intent: Intent,
    best_intent: Intent,
) -> bool:
    """判断当前 intent 是否比已有最佳 intent 更适合。

    主要比较命中关键词数量；数量相同时保持已有结果稳定，避免同一个 query
    因 dict 顺序之外的因素产生不稳定路由。
    """

    if len(matched_keywords) > len(best_keywords):
        return True
    if len(matched_keywords) == len(best_keywords) and best_intent == "unknown":
        return intent != "unknown" and bool(matched_keywords)
    return False
