from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intern_rag.evaluation.context_dataset import (  # noqa: E402
    ContextEvaluationCase,
    validate_context_dataset,
)


CATEGORIES = (
    "reference", "ellipsis", "history_constraint", "cross_session",
    "memory_conflict", "topic_switch", "multi_source", "unanswerable",
)
FACTS = (
    "候选人每周可实习四天", "候选人优先考虑广州岗位", "候选人熟悉 Python 和 RAG",
    "候选人希望连续实习六个月", "目标岗位要求掌握向量检索", "项目使用 FastAPI 提供接口",
    "候选人参与过知识库问答项目", "目标岗位位于深圳",
)


def main() -> int:
    """生成 60 组五轮场景化 dev Case，并保存结构校验结果。"""

    cases = [_case(index) for index in range(60)]
    output = ROOT / "data/evaluation/evalrag_context_v0.1.jsonl"
    output.write_text(
        "".join(json.dumps(case.to_dict(), ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    validation = validate_context_dataset(cases)
    validation_path = ROOT / "data/evaluation/evalrag_context_v0.1_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["is_valid"] else 1


def _case(index: int) -> ContextEvaluationCase:
    category = CATEGORIES[index % len(CATEGORIES)]
    point = "" if category == "unanswerable" else FACTS[index % len(FACTS)]
    base = datetime(2026, 8, 16, tzinfo=timezone.utc) + timedelta(hours=index)
    first_fact = point or "用户没有提供内部薪资信息"
    messages = (
        _message("m1", "user", f"请记住：{first_fact}。", base),
        _message("m2", "assistant", "已记录这条已确认信息。", base + timedelta(minutes=1)),
        _message("m3", "user", "顺便帮我关注面试准备。", base + timedelta(minutes=2)),
        _message("m4", "assistant", "可以继续讨论岗位和面试问题。", base + timedelta(minutes=3)),
        _message("m5", "user", _query(category), base + timedelta(minutes=4)),
    )
    profile_facts = (
        ({"key": "已确认条件", "value": point, "source": "explicit", "confirmed": True},)
        if category == "history_constraint" and point
        else ()
    )
    memories: tuple[dict[str, object], ...] = ()
    if point:
        memories = (
            {
                "memory_id": f"mem-{index}-new", "memory_type": "fact", "content": point,
                "source": "confirmed_chat", "importance": 1.0, "version": 2,
            },
            *(
                ({
                    "memory_id": f"mem-{index}-old", "memory_type": "fact",
                    "content": "候选人只考虑北京岗位", "source": "older_chat",
                    "importance": 0.3, "version": 1,
                },)
                if category == "memory_conflict" else ()
            ),
            {
                "memory_id": f"mem-{index}-distractor-a", "memory_type": "experience",
                "content": FACTS[(index + 2) % len(FACTS)],
                "source": "other_confirmed_session", "importance": 0.2, "version": 1,
            },
            {
                "memory_id": f"mem-{index}-distractor-b", "memory_type": "preference",
                "content": FACTS[(index + 3) % len(FACTS)],
                "source": "other_confirmed_session", "importance": 0.1, "version": 1,
            },
        )
    evidence = (
        ({"chunk_id": f"ctx-{index}-jd", "source_type": "jd", "text": point},)
        if category == "multi_source" and point else ()
    )
    return ContextEvaluationCase(
        case_id=f"context-v01-{index + 1:03d}", category=category, split="dev",
        user_id=f"user-{index % 10}", session_id=f"session-{index}", messages=messages,
        expected_point=point, answerable=category != "unanswerable",
        profile_facts=profile_facts, summary=point, memories=memories, evidence=evidence,
    )


def _message(message_id: str, role: str, content: str, created_at: datetime) -> dict[str, str]:
    return {"message_id": message_id, "role": role, "content": content, "created_at": created_at.isoformat()}


def _query(category: str) -> str:
    return {
        "reference": "我刚才说的那个条件是什么？",
        "ellipsis": "那这个安排呢？",
        "history_constraint": "结合我的稳定条件回答。",
        "cross_session": "我之前确认过的长期信息是什么？",
        "memory_conflict": "以最新确认记录为准，我的条件是什么？",
        "topic_switch": "回到最开始的话题，我提供了什么信息？",
        "multi_source": "结合个人条件和岗位证据说明匹配点。",
        "unanswerable": "请给出我从未提供过的公司内部薪资。",
    }[category]


if __name__ == "__main__":
    raise SystemExit(main())
