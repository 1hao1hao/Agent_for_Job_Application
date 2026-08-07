"""Agent 契约、上下文、生成、校验与单轮 Pipeline 模块。"""

from intern_rag.agent.answer import (
    AnswerResult,
    Citation,
    compose_answer,
)
from intern_rag.agent.context import (
    build_context,
    context_item_from_result,
    format_context_item,
)
from intern_rag.agent.evidence import (
    EvidenceConfig,
    EvidenceDecision,
    check_evidence,
)
from intern_rag.agent.generation import (
    DeepSeekChatClient,
    FakeLlmClient,
    GenerationParseError,
    GenerationResult,
    LlmClient,
    LlmClientError,
    LlmTimeoutError,
    OpenAIResponsesClient,
    build_generation_prompt,
    generate_answer,
    parse_generation_result,
)
from intern_rag.agent.pipeline import PipelineConfig, RagPipeline
from intern_rag.agent.schemas import (
    BuiltContext,
    ContextItem,
    RagRequest,
    RagResponse,
    RagStatus,
    RetrieverName,
    RouterName,
)
from intern_rag.agent.validation import (
    ValidationIssue,
    ValidationResult,
    validate_generation,
)

__all__ = [
    "AnswerResult",
    "BuiltContext",
    "Citation",
    "ContextItem",
    "EvidenceConfig",
    "EvidenceDecision",
    "DeepSeekChatClient",
    "FakeLlmClient",
    "GenerationParseError",
    "GenerationResult",
    "LlmClient",
    "LlmClientError",
    "LlmTimeoutError",
    "OpenAIResponsesClient",
    "PipelineConfig",
    "RagRequest",
    "RagPipeline",
    "RagResponse",
    "RagStatus",
    "RetrieverName",
    "RouterName",
    "build_context",
    "build_generation_prompt",
    "check_evidence",
    "compose_answer",
    "context_item_from_result",
    "format_context_item",
    "generate_answer",
    "parse_generation_result",
    "ValidationIssue",
    "ValidationResult",
    "validate_generation",
]
