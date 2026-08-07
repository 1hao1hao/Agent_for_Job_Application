from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from time import perf_counter
from typing import Literal, Mapping, Protocol, Sequence

from intern_rag.agent import LlmClient, LlmClientError, LlmTimeoutError


GradeStatus = Literal["completed", "unavailable", "skipped"]
PointVerdict = Literal["covered", "not_covered", "unknown"]
ClaimVerdict = Literal["supported", "unsupported", "unknown"]


@dataclass(frozen=True)
class EvidenceSpan:
    """一段可回查的证据文本；chunk_id 为空时表示来自 Answer。"""

    text: str
    chunk_id: str = ""


@dataclass(frozen=True)
class PointJudgment:
    """语义要点评分器对一个 expected point 的判断。"""

    point: str
    verdict: PointVerdict
    reason: str
    answer_evidence: str


@dataclass(frozen=True)
class ClaimJudgment:
    """Grounding Grader 对一条事实主张及其引用证据的判断。"""

    claim: str
    verdict: ClaimVerdict
    citation_ids: list[str]
    evidence: list[EvidenceSpan]
    reason: str


@dataclass(frozen=True)
class KeyPointGrade:
    """一条答案的逐要点评分结果和调用元数据。"""

    status: GradeStatus
    judgments: list[PointJudgment]
    grader_name: str
    grader_version: str
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""

    @property
    def coverage(self) -> float | None:
        """全部要点可判断时返回覆盖率；存在 unknown 时返回不可用。"""

        if self.status != "completed" or not self.judgments:
            return None
        if any(item.verdict == "unknown" for item in self.judgments):
            return None
        covered = sum(item.verdict == "covered" for item in self.judgments)
        return covered / len(self.judgments)


@dataclass(frozen=True)
class GroundingGrade:
    """一条答案的逐事实支持性判断和调用元数据。"""

    status: GradeStatus
    claims: list[ClaimJudgment]
    grader_name: str
    grader_version: str
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""

    @property
    def unsupported_answer(self) -> bool | None:
        """有 unsupported claim 时为真；unknown 或无 claim 时不作结论。"""

        if self.status != "completed" or not self.claims:
            return None
        if any(item.verdict == "unsupported" for item in self.claims):
            return True
        if any(item.verdict == "unknown" for item in self.claims):
            return None
        return False


class KeyPointGrader(Protocol):
    """将 Answer 与 expected points 对照的最小接口。"""

    name: str
    version: str

    def grade(self, answer: str, expected_points: Sequence[str]) -> KeyPointGrade:
        """返回逐 point 的 covered/not_covered/unknown 判断。"""


class GroundingGrader(Protocol):
    """将 Answer 的事实主张与 cited Context 对照的最小接口。"""

    name: str
    version: str

    def grade(
        self,
        answer: str,
        cited_context: Mapping[str, str],
    ) -> GroundingGrade:
        """返回逐 claim 的 supported/unsupported/unknown 判断。"""


class LexicalKeyPointGrader:
    """保留规范化字符串包含行为，作为可解释的 lexical baseline。"""

    name = "normalized-substring"
    version = "v1"

    def grade(self, answer: str, expected_points: Sequence[str]) -> KeyPointGrade:
        """逐点检查规范化字符串是否出现在答案中。"""

        normalized_answer = _normalize_text(answer)
        judgments = [
            PointJudgment(
                point=point,
                verdict=(
                    "covered"
                    if _normalize_text(point) in normalized_answer
                    else "not_covered"
                ),
                reason="normalized_substring_match",
                answer_evidence=(point if _normalize_text(point) in normalized_answer else ""),
            )
            for point in expected_points
        ]
        return KeyPointGrade(
            status="completed" if judgments else "skipped",
            judgments=judgments,
            grader_name=self.name,
            grader_version=self.version,
        )


@dataclass
class FakeKeyPointGrader:
    """自动化测试使用的可预测语义要点评分器。"""

    responses: list[KeyPointGrade]
    name: str = "fake-key-point-grader"
    version: str = "v1"
    calls: list[tuple[str, list[str]]] = field(default_factory=list, init=False)

    def grade(self, answer: str, expected_points: Sequence[str]) -> KeyPointGrade:
        """按顺序返回预设结果并记录输入。"""

        self.calls.append((answer, list(expected_points)))
        if not self.responses:
            raise RuntimeError("Fake KeyPointGrader has no response left")
        return self.responses.pop(0)


@dataclass
class FakeGroundingGrader:
    """自动化测试使用的可预测事实支持性评分器。"""

    responses: list[GroundingGrade]
    name: str = "fake-grounding-grader"
    version: str = "v1"
    calls: list[tuple[str, dict[str, str]]] = field(default_factory=list, init=False)

    def grade(
        self,
        answer: str,
        cited_context: Mapping[str, str],
    ) -> GroundingGrade:
        """按顺序返回预设结果并记录输入。"""

        self.calls.append((answer, dict(cited_context)))
        if not self.responses:
            raise RuntimeError("Fake GroundingGrader has no response left")
        return self.responses.pop(0)


class LlmKeyPointGrader:
    """通过结构化 LLM 判断答案是否以同义表达覆盖 expected points。"""

    def __init__(
        self,
        llm_client: LlmClient,
        *,
        model: str,
        temperature: float,
        prompt_version: str,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.name = "llm-semantic-key-point"
        self.version = prompt_version

    def grade(self, answer: str, expected_points: Sequence[str]) -> KeyPointGrade:
        """调用 LLM 并严格解析逐 point verdict，失败时返回 unavailable。

        输入是最终答案和人工 expected points。模型必须为每个 point 返回一次判断，
        covered 时还要复制答案中的原文片段。
        函数校验 point 集合、字段类型与 evidencespan，
        任何 API、JSON 或契约错误都会转成 unknown judgments，
        避免错误地计为 covered 或 not_covered。
        """

        points = list(expected_points)
        if not points:
            return KeyPointGrade("skipped", [], self.name, self.version)
        started = perf_counter()
        try:
            raw = self.llm_client.generate(
                build_key_point_prompt(answer, points, self.version),
                model=self.model,
                temperature=self.temperature,
            )
            judgments = parse_key_point_judgments(raw, answer, points)
            return KeyPointGrade(
                "completed",
                judgments,
                self.name,
                self.version,
                _elapsed_ms(started),
                _client_token_usage(self.llm_client),
            )
        except (LlmClientError, ValueError) as error:
            return _unavailable_key_point_grade(
                points,
                self.name,
                self.version,
                _elapsed_ms(started),
                _error_type(error),
                str(error),
                _client_token_usage(self.llm_client),
            )


class LlmGroundingGrader:
    """通过结构化 LLM 拆分 factual claims 并核对 cited Context。"""

    def __init__(
        self,
        llm_client: LlmClient,
        *,
        model: str,
        temperature: float,
        prompt_version: str,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.name = "llm-claim-grounding"
        self.version = prompt_version

    def grade(
        self,
        answer: str,
        cited_context: Mapping[str, str],
    ) -> GroundingGrade:
        """调用 LLM 生成逐 claim verdict，失败时返回 unavailable。

        输入只包含 Answer 和本轮实际引用的 Chunk，
        模型先抽取答案中的事实性陈述，
        再判断证据是支持、缺失/矛盾还是无法确定。
        解析阶段校验 claim 来自答案、citation id 存在、evidence span 来自对应 Chunk；
        契约错误不会被降级成 supported，而会返回 unknown/unavailable。
        """

        started = perf_counter()
        try:
            raw = self.llm_client.generate(
                build_grounding_prompt(answer, cited_context, self.version),
                model=self.model,
                temperature=self.temperature,
            )
            claims = parse_claim_judgments(raw, answer, cited_context)
            if not claims:
                raise ValueError("grounding response contains no factual claims")
            return GroundingGrade(
                "completed",
                claims,
                self.name,
                self.version,
                _elapsed_ms(started),
                _client_token_usage(self.llm_client),
            )
        except (LlmClientError, ValueError) as error:
            return GroundingGrade(
                "unavailable",
                [],
                self.name,
                self.version,
                _elapsed_ms(started),
                _client_token_usage(self.llm_client),
                _error_type(error),
                str(error),
            )


def build_key_point_prompt(
    answer: str,
    expected_points: Sequence[str],
    prompt_version: str,
) -> str:
    """构造语义要点评分 Prompt。"""

    return (
        f"prompt_version: {prompt_version}\n"
        "判断回答是否在语义上覆盖每个 expected point，不要求逐字相同。\n"
        "只依据 Answer；不要补充外部知识。covered 必须复制一段 Answer 原文作为 "
        "answer_evidence。无法可靠判断时用 unknown，不得猜测。\n"
        "只输出 JSON：{\"judgments\":[{\"point\":str,"
        "\"verdict\":\"covered|not_covered|unknown\","
        "\"answer_evidence\":str,\"reason\":str}]}。\n"
        "每个 expected point 必须且只能出现一次，point 保持原文。\n\n"
        f"Expected points:\n{json.dumps(list(expected_points), ensure_ascii=False)}\n\n"
        f"Answer:\n{answer}"
    )


def build_grounding_prompt(
    answer: str,
    cited_context: Mapping[str, str],
    prompt_version: str,
) -> str:
    """构造逐事实支持性审核 Prompt。"""

    evidence = [
        {"chunk_id": chunk_id, "text": text}
        for chunk_id, text in cited_context.items()
    ]
    return (
        f"prompt_version: {prompt_version}\n"
        "把 Answer 拆成最小、可验证的 factual claims，并逐条只依据 Cited Context 判断。\n"
        "supported：证据明确蕴含该事实；unsupported：证据缺失或与事实矛盾；"
        "unknown：上下文含糊，无法可靠确定。建议、问题和主观措辞不是 factual claim。\n"
        "claim 必须复制 Answer 原文；supported 的 evidence 必须复制对应 Chunk 原文。\n"
        "只输出 JSON：{\"claims\":[{\"claim\":str,"
        "\"verdict\":\"supported|unsupported|unknown\","
        "\"citation_ids\":[str],\"evidence\":[{\"chunk_id\":str,"
        "\"text\":str}],\"reason\":str}]}。\n\n"
        f"Answer:\n{answer}\n\n"
        f"Cited Context:\n{json.dumps(evidence, ensure_ascii=False)}"
    )


def parse_key_point_judgments(
    raw_response: str,
    answer: str,
    expected_points: Sequence[str],
) -> list[PointJudgment]:
    """解析并校验语义要点 JSON，防止漏点、重复点和伪造 evidence。"""

    payload = _parse_object(raw_response)
    raw_items = payload.get("judgments")
    if not isinstance(raw_items, list):
        raise ValueError("key-point judgments must be a list")
    expected = list(expected_points)
    if len(raw_items) != len(expected):
        raise ValueError("key-point judgment count does not match expected points")
    judgments: list[PointJudgment] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("key-point judgment must be an object")
        point = _required_string(raw_item, "point")
        verdict = _literal_value(
            raw_item, "verdict", {"covered", "not_covered", "unknown"}
        )
        evidence = _required_string(raw_item, "answer_evidence", allow_empty=True)
        reason = _required_string(raw_item, "reason")
        if verdict == "covered" and (
            not evidence or _normalize_text(evidence) not in _normalize_text(answer)
        ):
            raise ValueError("covered point evidence is not present in answer")
        judgments.append(PointJudgment(point, verdict, reason, evidence))  # type: ignore[arg-type]
    if sorted(item.point for item in judgments) != sorted(expected):
        raise ValueError("key-point response changed, omitted, or duplicated points")
    return judgments


def parse_claim_judgments(
    raw_response: str,
    answer: str,
    cited_context: Mapping[str, str],
) -> list[ClaimJudgment]:
    """解析逐 claim JSON，并校验 claim、引用 ID 和 evidence span 的来源。"""

    payload = _parse_object(raw_response)
    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        raise ValueError("grounding claims must be a list")
    claims: list[ClaimJudgment] = []
    for raw_claim in raw_claims:
        if not isinstance(raw_claim, dict):
            raise ValueError("grounding claim must be an object")
        claim = _required_string(raw_claim, "claim")
        if _normalize_text(claim) not in _normalize_text(answer):
            raise ValueError("grounding claim is not present in answer")
        verdict = _literal_value(
            raw_claim, "verdict", {"supported", "unsupported", "unknown"}
        )
        citation_ids = _string_list(raw_claim, "citation_ids")
        if any(chunk_id not in cited_context for chunk_id in citation_ids):
            raise ValueError("grounding claim uses unknown citation id")
        raw_evidence = raw_claim.get("evidence")
        if not isinstance(raw_evidence, list):
            raise ValueError("grounding evidence must be a list")
        evidence: list[EvidenceSpan] = []
        for raw_span in raw_evidence:
            if not isinstance(raw_span, dict):
                raise ValueError("grounding evidence span must be an object")
            chunk_id = _required_string(raw_span, "chunk_id")
            text = _required_string(raw_span, "text")
            if chunk_id not in cited_context:
                raise ValueError("evidence span uses unknown citation id")
            if _normalize_text(text) not in _normalize_text(cited_context[chunk_id]):
                raise ValueError("evidence span is not present in cited chunk")
            evidence.append(EvidenceSpan(text=text, chunk_id=chunk_id))
        if verdict == "supported" and (not citation_ids or not evidence):
            raise ValueError("supported claim must include citation and evidence")
        reason = _required_string(raw_claim, "reason")
        claims.append(
            ClaimJudgment(
                claim=claim,
                verdict=verdict,  # type: ignore[arg-type]
                citation_ids=citation_ids,
                evidence=evidence,
                reason=reason,
            )
        )
    return claims


def grade_to_dict(grade: KeyPointGrade | GroundingGrade) -> dict[str, object]:
    """把 grader dataclass 转成可写入 JSONL 的普通字典。"""

    return asdict(grade)


def _unavailable_key_point_grade(
    points: Sequence[str],
    name: str,
    version: str,
    latency_ms: float,
    error_type: str,
    message: str,
    token_usage: dict[str, int],
) -> KeyPointGrade:
    return KeyPointGrade(
        status="unavailable",
        judgments=[
            PointJudgment(point, "unknown", "grader_unavailable", "")
            for point in points
        ],
        grader_name=name,
        grader_version=version,
        latency_ms=latency_ms,
        token_usage=token_usage,
        error_type=error_type,
        error_message=message,
    )


def _parse_object(raw_response: str) -> dict[str, object]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ValueError("grader returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("grader response must be a JSON object")
    return payload


def _required_string(
    payload: Mapping[str, object],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _literal_value(
    payload: Mapping[str, object],
    field_name: str,
    allowed: set[str],
) -> str:
    value = _required_string(payload, field_name)
    if value not in allowed:
        raise ValueError(f"{field_name} has unsupported value: {value}")
    return value


def _string_list(payload: Mapping[str, object], field_name: str) -> list[str]:
    value = payload.get(field_name)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return value


def _client_token_usage(llm_client: LlmClient) -> dict[str, int]:
    usage = getattr(llm_client, "last_token_usage", None)
    if not isinstance(usage, dict):
        return {}
    return {
        str(key): int(value)
        for key, value in usage.items()
        if isinstance(value, int)
    }


def _error_type(error: Exception) -> str:
    if isinstance(error, LlmTimeoutError):
        return "llm_timeout"
    if isinstance(error, LlmClientError):
        return "llm_error"
    message = str(error)
    if "invalid JSON" in message:
        return "invalid_json"
    return "invalid_grader_output"


def _normalize_text(text: str) -> str:
    return "".join(text.casefold().split())


def _elapsed_ms(started: float) -> float:
    return (perf_counter() - started) * 1000
