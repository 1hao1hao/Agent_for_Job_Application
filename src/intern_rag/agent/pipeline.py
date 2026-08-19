from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping
from uuid import uuid4

from intern_rag.agent.context import ContextStrategy, build_context
from intern_rag.agent.context_engine import (
    ContextBudgetError,
    ContextEngine,
    ContextEngineConfig,
    ContextInputs,
    ContextMode,
)
from intern_rag.agent.evidence import EvidenceConfig, check_evidence
from intern_rag.agent.generation import (
    GenerationParseError,
    GenerationResult,
    LlmClient,
    LlmClientError,
    LlmTimeoutError,
    build_generation_prompt,
    generate_answer,
)
from intern_rag.agent.schemas import BuiltContext, RagRequest, RagResponse
from intern_rag.agent.validation import ValidationResult, validate_generation
from intern_rag.ingestion import Chunk
from intern_rag.retrieval import RetrievalResult, Retriever, retrieve_top_k
from intern_rag.routing import RouteDecision, Router, route_query
from intern_rag.tracing import (
    AgentTrace,
    ErrorType,
    build_agent_trace,
    write_trace_jsonl,
)


INSUFFICIENT_ANSWER = "当前证据不足，无法基于已提供的资料可靠回答该问题。"
FORMAT_ERROR_ANSWER = "模型输出格式不符合系统契约，本次请求未返回答案。"
CITATION_ERROR_ANSWER = "模型返回了无法验证的引用，本次请求未返回答案。"
SYSTEM_ERROR_ANSWER = "请求处理失败，请根据 trace 中的错误阶段排查。"

@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline 的路由、门控、重试、生成与上下文配置。"""

    model: str
    temperature: float = 0.0
    prompt_version: str = "p0-v1"
    context_max_chars: int = 4000
    context_strategy: ContextStrategy = "rank_prefix"
    router_name: str = "rule"
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    max_source_retries: int = 1
    max_format_retries: int = 1
    context_token_budget: int = 1800
    context_mode: ContextMode = "recent_window"
    system_prompt: str = "仅依据提供的证据回答；证据不足时明确拒答。"

    def __post_init__(self) -> None:
        """拒绝无法执行或无法复现的空配置。"""

        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.temperature < 0:
            raise ValueError("temperature must not be negative")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version must not be empty")
        if self.context_max_chars <= 0:
            raise ValueError("context_max_chars must be greater than 0")
        if self.context_strategy not in {"rank_prefix", "source_balanced"}:
            raise ValueError("unknown context_strategy")
        if self.context_token_budget <= 0:
            raise ValueError("context_token_budget must be greater than 0")
        if self.context_mode not in {
            "no_memory", "full_history", "recent_window", "summary_recent", "semantic_memory"
        }:
            raise ValueError("unknown context_mode")
        if not self.router_name.strip():
            raise ValueError("router_name must not be empty")
        if self.max_source_retries not in {0, 1}:
            raise ValueError("max_source_retries must be 0 or 1")
        if self.max_format_retries not in {0, 1}:
            raise ValueError("max_format_retries must be 0 or 1")


class RagPipeline:
    """串联 Router、可配置 Retriever、Context、Generator 与 Validator。

    run() 对一次请求只构造并追加一条 AgentTrace。模型输出或引用失败时，
    Pipeline 返回受控 RagResponse，并尽可能保留已经完成阶段的信息。
    """

    def __init__(
        self,
        chunks: list[Chunk],
        llm_client: LlmClient,
        config: PipelineConfig,
        *,
        trace_path: Path = Path("traces/agent_trace.jsonl"),
        router: Router = route_query,
        routers: Mapping[str, Router] | None = None,
        retriever: Retriever = retrieve_top_k,
        retrievers: Mapping[str, Retriever] | None = None,
        trace_sink: Callable[[AgentTrace], None] | None = None,
        context_engine: ContextEngine | None = None,
        context_provider: Callable[[RagRequest], ContextInputs] | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.llm_client = llm_client
        self.config = config
        self.trace_path = trace_path
        self.router = router
        self.routers: dict[str, Router] = {"rule": router}
        if routers is not None:
            self.routers.update(routers)
        self.retriever = retriever
        self.retrievers: dict[str, Retriever] = {"keyword": retriever}
        if retrievers is not None:
            self.retrievers.update(retrievers)
        self.trace_sink = trace_sink
        self.context_engine = context_engine
        self.context_provider = context_provider
        self.last_trace: AgentTrace | None = None
        self.last_trace_persistence_errors: list[str] = []

    def run(self, request: RagRequest) -> RagResponse:
        """执行门控与有限重试，并始终只追加一条请求级 Trace。"""

        trace_id = str(uuid4())
        started_at = perf_counter()
        stage_started_at = started_at
        current_stage = "routing"
        route_decision = RouteDecision(
            intent="unknown",
            routed_sources=[],
            matched_keywords=[],
        )
        retrieved_results: list[RetrievalResult] = []
        latency_ms = {
            "routing": 0.0,
            "retrieval": 0.0,
            "context": 0.0,
            "generation": 0.0,
            "validation": 0.0,
            "evidence": 0.0,
            "total": 0.0,
        }
        context_trace: dict[str, object] = {}
        generation_trace: dict[str, object] = {}
        validation_trace: dict[str, object] = {}
        evidence_trace: dict[str, object] = {}
        retrieval_decision_trace: dict[str, object] = {}
        attempts: list[dict[str, object]] = []
        citations: list[dict[str, object]] = []
        response = self._error_response(
            request=request,
            trace_id=trace_id,
            latency_ms=0.0,
            error_type="unknown_error",
            answer=SYSTEM_ERROR_ANSWER,
        )
        error_type: ErrorType = "none"
        error_message = ""
        source_retry_count = 0
        format_retry_count = 0

        try:
            selected_router = self.routers.get(self.config.router_name)
            if selected_router is None:
                raise ValueError(
                    f"router '{self.config.router_name}' is not configured"
                )
            selected_retriever = self.retrievers.get(request.retriever)
            if selected_retriever is None:
                current_stage = "retrieval"
                raise ValueError(
                    f"retriever '{request.retriever}' is not configured"
                )
            # Router 路由，根据query匹配并缩小检索路由
            stage_started_at = perf_counter()
            route_decision = selected_router(request.query)
            latency_ms["routing"] = _elapsed_ms(stage_started_at)

            while True:
                current_stage = "retrieval"
                stage_started_at = perf_counter()
                source_types = (
                    set(route_decision.routed_sources)
                    if source_retry_count == 0
                    else None
                )
                retrieved_results = selected_retriever(
                    request.query,
                    self.chunks,
                    top_k=request.top_k,
                    source_types=source_types,
                )
                retrieval_decision_trace = _retriever_trace(selected_retriever)
                retrieval_latency = _elapsed_ms(stage_started_at)
                latency_ms["retrieval"] += retrieval_latency

                current_stage = "evidence"
                stage_started_at = perf_counter()
                evidence_decision = check_evidence(
                    route_decision,
                    retrieved_results,
                    retriever_name=request.retriever,
                    retry_count=source_retry_count,
                    max_retries=self.config.max_source_retries,
                    config=self.config.evidence,
                )
                evidence_latency = _elapsed_ms(stage_started_at)
                latency_ms["evidence"] += evidence_latency
                evidence_trace = asdict(evidence_decision)
                attempts.append({
                    "attempt": len(attempts) + 1,
                    "type": (
                        "initial_retrieval"
                        if source_retry_count == 0
                        else "source_expansion"
                    ),
                    "source_filter": sorted(source_types) if source_types else None,
                    "retrieved_chunk_ids": [
                        result.chunk_id for result in retrieved_results
                    ],
                    "retrieval_decision": retrieval_decision_trace,
                    "evidence": evidence_trace,
                    "latency_ms": {
                        "retrieval": retrieval_latency,
                        "evidence": evidence_latency,
                    },
                })
                if evidence_decision.status != "retryable":
                    break
                source_retry_count += 1

            if evidence_decision.status != "sufficient":
                is_normal_unanswerable = (
                    evidence_decision.reason == "unanswerable_route"
                )
                error_type = "none" if is_normal_unanswerable else "retrieval_miss"
                error_message = evidence_decision.message
                response = RagResponse(
                    request_id=request.request_id,
                    trace_id=trace_id,
                    answer=INSUFFICIENT_ANSWER,
                    citations=[],
                    routed_sources=route_decision.routed_sources,
                    status="insufficient_evidence",
                    latency_ms=0.0,
                    error_type=None if is_normal_unanswerable else error_type,
                )
            else:
                # Context Builder 利用检索返回的结果构造模型上下文。
                current_stage = "context"
                stage_started_at = perf_counter()
                managed_context = None
                if self.context_engine is None:
                    built_context = build_context(
                        request.query,
                        retrieved_results,
                        max_chars=self.config.context_max_chars,
                        strategy=self.config.context_strategy,
                        required_source_types=route_decision.routed_sources,
                    )
                else:
                    inputs = (
                        self.context_provider(request)
                        if self.context_provider is not None
                        else ContextInputs()
                    )
                    # 先估算 Generator 外层指令、Query 与候选 citation id 的固定开销，
                    # Context Engine 的 token_budget 因而覆盖最终完整 Prompt，而非仅证据正文。
                    candidate_ids = [item.chunk_id for item in retrieved_results]
                    empty_context = BuiltContext(
                        query=request.query,
                        text="",
                        items=[],
                        used_chunk_ids=candidate_ids,
                        skipped_chunk_ids=[],
                        char_count=0,
                        max_chars=0,
                    )
                    reserved_tokens = self.context_engine.estimator.count(
                        build_generation_prompt(
                            request.query, empty_context, self.config.prompt_version
                        )
                    )
                    managed_context = self.context_engine.build(
                        query=request.query,
                        system_prompt=self.config.system_prompt,
                        retrieved_results=retrieved_results,
                        config=ContextEngineConfig(
                            token_budget=self.config.context_token_budget,
                            evidence_char_budget=self.config.context_max_chars,
                            mode=self.config.context_mode,
                            evidence_strategy=self.config.context_strategy,
                            reserved_token_count=reserved_tokens,
                        ),
                        required_source_types=route_decision.routed_sources,
                        profile=inputs.profile,
                        history=inputs.history,
                        memories=inputs.memories,
                        history_summary=inputs.history_summary,
                    )
                    built_context = managed_context.as_built_context()
                    actual_prompt_tokens = self.context_engine.estimator.count(
                        build_generation_prompt(
                            request.query, built_context, self.config.prompt_version
                        )
                    )
                    if actual_prompt_tokens > self.config.context_token_budget:
                        raise ContextBudgetError("formatted generation prompt exceeds token budget")
                latency_ms["context"] = _elapsed_ms(stage_started_at)
                context_trace = {
                    "used_chunk_ids": built_context.used_chunk_ids,
                    "skipped_chunk_ids": built_context.skipped_chunk_ids,
                    "char_count": built_context.char_count,
                    "max_chars": built_context.max_chars,
                    "is_truncated": built_context.is_truncated,
                    "selection_strategy": built_context.selection_strategy,
                    "covered_source_types": built_context.covered_source_types,
                    "missing_source_types": built_context.missing_source_types,
                }
                if managed_context is not None:
                    context_trace.update({
                        "mode": managed_context.mode,
                        "token_count": managed_context.token_count,
                        "token_budget": managed_context.token_budget,
                        "reserved_token_count": managed_context.reserved_token_count,
                        "actual_prompt_token_count": actual_prompt_tokens,
                        "token_estimator_version": managed_context.token_estimator_version,
                        "kept_segment_ids": list(managed_context.kept_ids),
                        "dropped": list(managed_context.dropped),
                        "recalled_memory_ids": list(managed_context.recalled_memory_ids),
                        "memory_write_ids": [],
                        "memory_write_reason": "disabled_without_explicit_confirmation",
                        "compression_fallbacks": list(managed_context.compression_fallbacks),
                    })

                while True:
                    current_stage = "generation"
                    stage_started_at = perf_counter()
                    try:
                        generation_result = generate_answer(
                            request.query,
                            built_context,
                            self.llm_client,
                            model=self.config.model,
                            temperature=self.config.temperature,
                            prompt_version=self.config.prompt_version,
                        )
                        generation_latency = _elapsed_ms(stage_started_at)
                        latency_ms["generation"] += generation_latency
                        generation_trace = _generation_to_trace(generation_result)
                        gateway_trace = _read_model_gateway_trace(self.llm_client)
                        if gateway_trace:
                            generation_trace["model_gateway"] = gateway_trace
                        call_token_usage = _read_token_usage(self.llm_client)
                        attempts.append({
                            "attempt": len(attempts) + 1,
                            "type": (
                                "initial_generation"
                                if format_retry_count == 0
                                else "format_repair"
                            ),
                            "status": "parsed",
                            "latency_ms": {"generation": generation_latency},
                            "token_usage": call_token_usage,
                            "model_gateway": gateway_trace,
                        })
                        break
                    except GenerationParseError as error:
                        generation_latency = _elapsed_ms(stage_started_at)
                        latency_ms["generation"] += generation_latency
                        attempts.append({
                            "attempt": len(attempts) + 1,
                            "type": (
                                "initial_generation"
                                if format_retry_count == 0
                                else "format_repair"
                            ),
                            "status": "invalid_format",
                            "reason": error.error_type,
                            "latency_ms": {"generation": generation_latency},
                            "token_usage": _read_token_usage(self.llm_client),
                            "model_gateway": _read_model_gateway_trace(self.llm_client),
                        })
                        if format_retry_count >= self.config.max_format_retries:
                            raise
                        format_retry_count += 1

                current_stage = "validation"
                stage_started_at = perf_counter()
                validation_result = validate_generation(
                    generation_result,
                    built_context,
                )
                latency_ms["validation"] = _elapsed_ms(stage_started_at)
                validation_trace = _validation_to_trace(validation_result)

                if not validation_result.is_valid:
                    error_type = "citation_invalid"
                    error_message = "; ".join(
                        issue.message for issue in validation_result.issues
                    )
                    response = self._error_response(
                        request=request,
                        trace_id=trace_id,
                        latency_ms=0.0,
                        error_type=error_type,
                        answer=CITATION_ERROR_ANSWER,
                        routed_sources=route_decision.routed_sources,
                    )
                elif not generation_result.sufficient:
                    response = RagResponse(
                        request_id=request.request_id,
                        trace_id=trace_id,
                        answer=(
                            generation_result.answer.strip()
                            or INSUFFICIENT_ANSWER
                        ),
                        citations=[],
                        routed_sources=route_decision.routed_sources,
                        status="insufficient_evidence",
                        latency_ms=0.0,
                    )
                else:
                    citations = [
                        citation.to_dict()
                        for citation in validation_result.citations
                    ]
                    response = RagResponse(
                        request_id=request.request_id,
                        trace_id=trace_id,
                        answer=generation_result.answer,
                        citations=validation_result.citations,
                        routed_sources=route_decision.routed_sources,
                        status="answered",
                        latency_ms=0.0,
                    )
        except GenerationParseError as error:
            error_type = "llm_format_error"
            error_message = f"{error.error_type}: {error}"
            generation_trace = {
                "status": "invalid",
                "error_type": error.error_type,
                "error_message": str(error),
            }
            response = self._error_response(
                request=request,
                trace_id=trace_id,
                latency_ms=0.0,
                error_type=error_type,
                answer=FORMAT_ERROR_ANSWER,
                routed_sources=route_decision.routed_sources,
            )
        except LlmClientError as error:
            generation_latency = _elapsed_ms(stage_started_at)
            latency_ms["generation"] += generation_latency
            error_type = (
                "llm_timeout" if isinstance(error, LlmTimeoutError) else "llm_error"
            )
            error_message = str(error)
            generation_trace = {
                "status": "error",
                "error_message": str(error),
                "model_gateway": _read_model_gateway_trace(self.llm_client),
            }
            attempts.append({
                "attempt": len(attempts) + 1,
                "type": "generation",
                "status": "timeout" if error_type == "llm_timeout" else "error",
                "reason": error_type,
                "latency_ms": {"generation": generation_latency},
                "token_usage": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "source": "not_reported_by_client_contract",
                },
                "model_gateway": _read_model_gateway_trace(self.llm_client),
            })
            response = self._error_response(
                request=request,
                trace_id=trace_id,
                latency_ms=0.0,
                error_type=error_type,
                answer=SYSTEM_ERROR_ANSWER,
                routed_sources=route_decision.routed_sources,
            )
        except Exception as error:
            error_type = _stage_error_type(current_stage)
            error_message = str(error)
            response = self._error_response(
                request=request,
                trace_id=trace_id,
                latency_ms=0.0,
                error_type=error_type,
                answer=SYSTEM_ERROR_ANSWER,
                routed_sources=route_decision.routed_sources,
            )

        latency_ms["total"] = _elapsed_ms(started_at)
        response = RagResponse(
            request_id=response.request_id,
            trace_id=response.trace_id,
            answer=response.answer,
            citations=response.citations,
            routed_sources=response.routed_sources,
            status=response.status,
            latency_ms=latency_ms["total"],
            error_type=response.error_type,
        )
        trace = build_agent_trace(
            query=request.query,
            route_decision=route_decision,
            retrieved_results=retrieved_results,
            latency_ms=latency_ms,
            error_type=error_type,
            request_id=request.request_id,
            trace_id=trace_id,
            citations=citations,
            answer=response.answer,
            routing={
                "intent": route_decision.intent,
                "routed_sources": route_decision.routed_sources,
                "matched_keywords": route_decision.matched_keywords,
                "strategy": route_decision.strategy,
                "confidence": route_decision.confidence,
                "reason": route_decision.reason,
                "details": route_decision.details,
            },
            retrieval={
                "retriever": request.retriever,
                "top_k": request.top_k,
                "result_count": len(retrieved_results),
                "chunk_ids": [result.chunk_id for result in retrieved_results],
                "decision": retrieval_decision_trace,
            },
            context=context_trace,
            evidence=evidence_trace,
            generation=generation_trace,
            validation=validation_trace,
            model_config={
                "model": self.config.model,
                "temperature": self.config.temperature,
                "context_max_chars": self.config.context_max_chars,
                "context_strategy": self.config.context_strategy,
                "router_name": self.config.router_name,
                "max_source_retries": self.config.max_source_retries,
                "max_format_retries": self.config.max_format_retries,
            },
            prompt_version=self.config.prompt_version,
            response_status=response.status,
            error_message=error_message,
            attempts=attempts,
            token_usage={
                "attempts": [
                    item.get("token_usage")
                    for item in attempts
                    if "token_usage" in item
                ],
                "source": "client_reported_or_explicitly_unavailable",
            },
        )
        self.last_trace = trace
        self.last_trace_persistence_errors = []
        try:
            write_trace_jsonl(trace, self.trace_path)
        except Exception as error:
            self.last_trace_persistence_errors.append(
                f"jsonl:{type(error).__name__}: {error}"
            )
        if self.trace_sink is not None:
            try:
                self.trace_sink(trace)
            except Exception as error:
                self.last_trace_persistence_errors.append(
                    f"sink:{type(error).__name__}: {error}"
                )
        return response

    @staticmethod
    def _error_response(
        request: RagRequest,
        trace_id: str,
        latency_ms: float,
        error_type: ErrorType,
        answer: str,
        routed_sources: list[str] | None = None,
    ) -> RagResponse:
        """构造不会携带未验证引用的受控错误响应。"""

        return RagResponse(
            request_id=request.request_id,
            trace_id=trace_id,
            answer=answer,
            citations=[],
            routed_sources=routed_sources or [],
            status="error",
            latency_ms=latency_ms,
            error_type=error_type,
        )


def _elapsed_ms(started_at: float) -> float:
    """计算阶段耗时，统一转换为毫秒。"""

    return (perf_counter() - started_at) * 1000


def _generation_to_trace(result: GenerationResult) -> dict[str, object]:
    """提取不包含模型原始输出的生成阶段 Trace。"""

    return {
        "status": "parsed",
        "sufficient": result.sufficient,
        "cited_chunk_ids": result.cited_chunk_ids,
        "reason": result.reason,
    }


def _validation_to_trace(result: ValidationResult) -> dict[str, object]:
    """把校验结果转换为可序列化 Trace。"""

    return {
        "is_valid": result.is_valid,
        "citation_ids": [citation.chunk_id for citation in result.citations],
        "issues": [issue.to_dict() for issue in result.issues],
    }


def _retriever_trace(retriever: Retriever) -> dict[str, object]:
    """读取可观测 Retriever 的本次决策；普通 Retriever 返回空字典。"""

    get_last_trace = getattr(retriever, "get_last_trace", None)
    if not callable(get_last_trace):
        return {}
    trace = get_last_trace()
    return dict(trace) if isinstance(trace, dict) else {}


def _stage_error_type(stage: str) -> ErrorType:
    """把未预期异常归类到当前执行阶段。"""

    if stage == "routing":
        return "router_error"
    if stage == "retrieval":
        return "retriever_error"
    if stage == "evidence":
        return "retrieval_miss"
    return "unknown_error"


def _read_token_usage(client: LlmClient) -> dict[str, object]:
    """读取 adapter 暴露的真实 token usage，不用字符数伪装 token。"""

    usage = getattr(client, "last_token_usage", None)
    if isinstance(usage, dict):
        return {**usage, "source": "llm_client"}
    return {
        "input_tokens": None,
        "output_tokens": None,
        "source": "not_reported_by_client",
    }


def _read_model_gateway_trace(client: LlmClient) -> dict[str, object]:
    """读取 Gateway 的脱敏 provider attempt，不要求普通 LLM client 实现。"""

    trace = getattr(client, "last_gateway_trace", None)
    return dict(trace) if isinstance(trace, dict) else {}
