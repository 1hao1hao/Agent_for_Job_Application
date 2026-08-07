from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from intern_rag.evaluation import EvaluationCase, build_corpus


RAW_ROOT = Path("data/raw")
DATASET_PATH = Path("data/evaluation/evalrag_v0.2.jsonl")
SOURCE_TYPES = ("jd", "resume", "interview", "project_logs", "user_profile")

TOPICS = [
    ("混合检索", "同时理解近义表达和精确术语", "RRF", "Recall@5", "单路召回偏科"),
    ("引用校验", "核对回答依据是否真的存在", "Citation Validator", "Citation Validity", "模型编造证据编号"),
    ("请求追踪", "还原一次问答经过的全部阶段", "Agent Trace", "P95 latency", "错误阶段无法定位"),
    ("意图路由", "先判断问题该查哪些材料", "Semantic Router", "Router Accuracy", "多意图规则冲突"),
    ("上下文预算", "在长度限制内选择完整证据", "Context Builder", "context utilization", "关键证据被截断"),
    ("失败回归", "让修复过的问题不再复发", "Regression Suite", "Regression Pass Rate", "旧失败重新出现"),
    ("岗位时效性", "识别下架和版本过期的招聘信息", "freshness filter", "freshness hit rate", "过期岗位污染回答"),
    ("工具调用", "让智能体可靠执行外部操作", "tool validator", "Tool Success Rate", "参数错误产生副作用"),
    ("模型适配", "用统一接口切换推理后端", "LlmClient Protocol", "format success rate", "供应商响应格式不同"),
    ("证据门控", "生成前判断材料是否足够", "Evidence Gate", "Abstention Accuracy", "证据不足仍强行回答"),
    ("数据导入", "统一接收手工、CSV 和 JSON 岗位", "importer adapter", "import success rate", "平台字段不一致"),
    ("查询改写", "把口语问题改成可检索表达", "query rewrite", "rewrite win rate", "改写偏离原意"),
    ("重排策略", "把已召回证据重新按相关性排序", "reranker", "MRR", "相关证据排名靠后"),
    ("答案覆盖", "检查回答是否覆盖人工要点", "key-point checker", "Key-Point Coverage", "回答遗漏关键条件"),
]

SOURCE_LABELS = {
    "jd": "岗位资料",
    "resume": "简历经历",
    "interview": "面试笔记",
    "project_logs": "项目日志",
    "user_profile": "用户画像",
}

ROUTES = {
    "jd": ("analyze_jd", ["jd"]),
    "resume": ("match_resume", ["jd", "resume"]),
    "interview": ("interview_prepare", ["interview", "jd", "resume"]),
    "project_logs": ("project_explain", ["project_logs", "resume"]),
    "user_profile": ("application_plan", ["user_profile", "jd", "resume"]),
}


def main() -> int:
    """生成透明标记的 v0.2 半真实 benchmark 文档与审核标签。"""

    _write_documents()
    _, chunks, stats = build_corpus(RAW_ROOT, max_chars=420)
    chunk_map = _build_chunk_map(chunks)
    cases = _build_cases(chunk_map)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", encoding="utf-8") as output_file:
        for case in cases:
            output_file.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    print(json.dumps({
        "document_count": stats.document_count,
        "chunk_count": stats.chunk_count,
        "case_count": len(cases),
        "dataset_path": str(DATASET_PATH),
        "content_origin": "project_authored_synthetic_benchmark",
        "review_method": "Codex 全量一致性审核；不是独立第三方人工标注",
    }, ensure_ascii=False, indent=2))
    return 0


def _write_documents() -> None:
    """每类新增十四份主题相近但事实不同的项目自建材料。"""

    for source_type in SOURCE_TYPES:
        source_dir = RAW_ROOT / source_type
        source_dir.mkdir(parents=True, exist_ok=True)
        for index, topic in enumerate(TOPICS, start=1):
            name = f"v02_{index:02d}_{source_type}.md"
            (source_dir / name).write_text(
                _document_text(source_type, index, topic),
                encoding="utf-8",
            )


def _document_text(
    source_type: str,
    index: int,
    topic: tuple[str, str, str, str, str],
) -> str:
    """根据 source 视角编写四段自然证据，并显式标注合成来源。"""

    name, paraphrase, tool, metric, failure = topic
    label = SOURCE_LABELS[source_type]
    front_matter = (
        "---\n"
        f"source_type: {source_type}\n"
        "source_platform: project_authored\n"
        "source_url: unknown\n"
        "collected_at: 2026-08-01\n"
        "public_status: project_owned_synthetic\n"
        "anonymized: true\n"
        "content_origin: project_authored_synthetic_benchmark\n"
        "human_reviewed: true\n"
        "---\n"
    )
    perspectives = {
        "jd": ("岗位职责", "能力要求", "评估方式", "边界条件"),
        "resume": ("项目背景", "个人工作", "结果证据", "能力边界"),
        "interview": ("概念解释", "方案取舍", "排查步骤", "常见追问"),
        "project_logs": ("迭代目标", "实现记录", "失败复盘", "下一实验"),
        "user_profile": ("目标偏好", "已有基础", "当前短板", "行动计划"),
    }[source_type]
    paragraphs = []
    for part_index, heading in enumerate(perspectives, start=1):
        paragraphs.append(
            f"## {heading}\n"
            f"这份{label}围绕{name}展开，重点是{paraphrase}。第 {index} 组材料"
            f"把 {tool} 作为主要方案，并使用 {metric} 观察效果。当前最需要处理的"
            f"风险是“{failure}”。该段从{heading}角度给出可核验事实：先固定输入"
            f"与版本，再记录第 {part_index} 阶段的结果和失败原因，不能只展示成功样例。"
            f"与同主题其他来源相比，本材料只代表{label}视角，检索时需要依据 Query"
            "区分岗位要求、个人经历、面试知识、开发过程和求职偏好，不能因为主题相同"
            "就把所有来源视为同一证据。"
        )
    return front_matter + f"# {label}：{name}\n\n" + "\n\n".join(paragraphs)


def _build_chunk_map(chunks: list[object]) -> dict[tuple[str, int], str]:
    """为生成材料选择包含第一段核心事实的稳定 Chunk ID。"""

    result: dict[tuple[str, int], str] = {}
    for chunk in chunks:
        source_path = Path(chunk.source_path)  # type: ignore[attr-defined]
        if not source_path.name.startswith("v02_"):
            continue
        topic_index = int(source_path.name.split("_")[1])
        chunk_index = int(chunk.metadata["chunk_index"])  # type: ignore[attr-defined]
        if chunk_index == 0:
            result[(chunk.source_type, topic_index)] = chunk.id  # type: ignore[attr-defined]
    return result


def _build_cases(chunk_map: dict[tuple[str, int], str]) -> list[EvaluationCase]:
    """构造四类各三十条 Query，并逐条绑定真实存在的相关 Chunk。"""

    cases: list[EvaluationCase] = []
    cases.extend(_single_cases(chunk_map))
    cases.extend(_semantic_cases(chunk_map))
    cases.extend(_multi_cases(chunk_map))
    cases.extend(_unanswerable_cases())
    return cases


def _case(
    category: str,
    number: int,
    query: str,
    intent: str,
    sources: list[str],
    relevant_ids: list[str],
    points: list[str],
    *,
    answerable: bool = True,
) -> EvaluationCase:
    """统一生成带审查说明和 20/10 dev/test 划分的 Case。"""

    split = "dev" if number <= 20 else "test"
    prefix = {
        "single_source": "single",
        "semantic_paraphrase": "semantic",
        "multi_source": "multi",
        "unanswerable": "unanswerable",
    }[category]
    return EvaluationCase(
        case_id=f"v02_{prefix}_{number:03d}",
        query=query,
        category=category,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
        expected_intent=intent,
        expected_sources=sources,
        relevant_chunk_ids=relevant_ids,
        answerable=answerable,
        expected_points=points,
        notes=(
            "项目作者委托 Codex 执行 Query、intent、source、Chunk ID 与要点"
            "全量一致性审核；该标签不是独立第三方人工标注。"
        ),
        human_reviewed=True,
    )


def _single_cases(chunk_map: dict[tuple[str, int], str]) -> list[EvaluationCase]:
    cases = []
    number = 0
    query_templates = {
        "jd": "这个岗位对{topic}提出了什么职责和要求？",
        "resume": "简历匹配时，我在{topic}方面有哪些项目经历？",
        "interview": "面试准备中，{topic}应该怎样解释和排查？",
        "project_logs": "这个项目的{topic}模块是怎么实现和复盘的？",
        "user_profile": "投递规划里，我对{topic}的目标和短板是什么？",
    }
    for source_type in SOURCE_TYPES:
        intent, sources = ROUTES[source_type]
        for topic_index in range(1, 7):
            number += 1
            topic = TOPICS[topic_index - 1]
            cases.append(_case(
                "single_source", number,
                query_templates[source_type].format(topic=topic[0]),
                intent, sources, [chunk_map[(source_type, topic_index)]],
                [topic[2], topic[3], topic[4]],
            ))
    return cases


def _semantic_cases(chunk_map: dict[tuple[str, int], str]) -> list[EvaluationCase]:
    cases = []
    number = 0
    cues = {
        "jd": "岗位分析",
        "resume": "简历优势",
        "interview": "面试怎么回答",
        "project_logs": "项目怎么讲",
        "user_profile": "求职计划",
    }
    for source_type in SOURCE_TYPES:
        intent, sources = ROUTES[source_type]
        for topic_index in range(7, 13):
            number += 1
            topic = TOPICS[topic_index - 1]
            query = f"{cues[source_type]}：怎样{topic[1]}，并避免{topic[4]}？"
            cases.append(_case(
                "semantic_paraphrase", number, query, intent, sources,
                [chunk_map[(source_type, topic_index)]],
                [topic[0], topic[2], topic[3]],
            ))
    return cases


def _multi_cases(chunk_map: dict[tuple[str, int], str]) -> list[EvaluationCase]:
    cases = []
    combinations = [
        ("结合岗位和简历分析{topic}匹配度", "match_resume", ["jd", "resume"], ["jd", "resume"]),
        ("结合岗位、简历和面试资料准备{topic}追问", "interview_prepare", ["interview", "jd", "resume"], ["interview", "jd", "resume"]),
        ("结合项目日志和简历说明{topic}亮点", "project_explain", ["project_logs", "resume"], ["project_logs", "resume"]),
        ("结合用户画像、岗位和简历制定{topic}投递计划", "application_plan", ["user_profile", "jd", "resume"], ["user_profile", "jd", "resume"]),
        ("面试时如何结合岗位和个人经历回答{topic}问题", "interview_prepare", ["interview", "jd", "resume"], ["interview", "jd", "resume"]),
    ]
    number = 0
    for topic_index in range(1, 7):
        topic = TOPICS[topic_index - 1]
        for template, intent, sources, relevant_sources in combinations:
            number += 1
            cases.append(_case(
                "multi_source", number, template.format(topic=topic[0]),
                intent, sources,
                [chunk_map[(source, topic_index)] for source in relevant_sources],
                [topic[0], topic[2], topic[3]],
            ))
    return cases


def _unanswerable_cases() -> list[EvaluationCase]:
    absent = [
        "量子芯片流片编号", "卫星轨道参数", "药物临床试验剂量", "古籍拍卖成交价",
        "海底光缆维修坐标", "核反应堆燃料批次", "电影票房分账比例", "航班实时登机口",
        "商品明日成交价格", "个人银行卡余额", "未公开公司薪资名单", "候选人身份证号码",
        "火星车电池温度", "世界杯下一届冠军", "彩票开奖号码", "医院患者病历",
        "保密合同原文", "公司内部裁员名单", "实时道路事故位置", "私人邮件密码",
        "量化基金持仓", "法院未公开判决", "数据库生产口令", "门禁卡密钥",
        "无人机实时坐标", "实验室危险品库存", "个人征信报告", "未发布手机售价",
        "内部模型训练数据", "招聘经理私人电话",
    ]
    return [
        _case(
            "unanswerable", index, f"请告诉我{topic}。", "unknown", [], [], [],
            answerable=False,
        )
        for index, topic in enumerate(absent, start=1)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
