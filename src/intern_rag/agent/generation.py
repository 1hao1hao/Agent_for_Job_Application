from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from intern_rag.agent.schemas import BuiltContext


GenerationErrorType = Literal[
    "invalid_json",
    "missing_field",
    "invalid_field_type",
]


@dataclass(frozen=True)
class GenerationResult:
    """ 结构化模型输出。Generator 输出给 Validator

    cited_chunk_ids 只是模型声称使用的证据，必须继续交给 Citation Validator
    校验，不能直接当成合法引用。
    """

    answer: str
    cited_chunk_ids: list[str]
    sufficient: bool
    reason: str


class LlmClient(Protocol):
    """Generator 可注入的最小模型客户端契约。"""
    """Generator 调用“任意模型客户端”时遵守的 Python 接口契约。"""
    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        """根据 prompt 返回模型原始文本。"""


class GenerationParseError(ValueError):
    """模型输出不符合 GenerationResult 契约时的受控异常。"""

    def __init__(self, error_type: GenerationErrorType, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class LlmClientError(RuntimeError):
    """真实模型调用失败时的受控异常。"""


class LlmTimeoutError(LlmClientError):
    """模型调用超过客户端超时时间。"""


@dataclass
class FakeLlmClient:
    """供自动化测试使用的离线 Fake LLM。

    responses 按调用顺序返回；prompts 保存收到的输入，方便测试 Prompt 是否
    包含问题、证据和约束。该实现不会访问网络。
    """

    responses: list[str]
    prompts: list[str] = field(default_factory=list, init=False)
    last_token_usage: dict[str, int] | None = field(default=None, init=False)

    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        """返回下一条预设响应，并记录本次调用参数。"""

        self.prompts.append(prompt)
        if not self.responses:
            raise LlmClientError("Fake LLM has no response left")
        return self.responses.pop(0) #按调用顺序返回，一次调用只返回一次模型输出


class OpenAIResponsesClient:
    """基于标准库调用 OpenAI Responses API 的真实模型 adapter。

    API key 只在发起请求时从 OPENAI_API_KEY 环境变量读取，不接受源码参数，
    也不会出现在异常信息中。
    """

    api_url = "https://api.openai.com/v1/responses"

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        self.timeout_seconds = timeout_seconds
        self.last_token_usage: dict[str, int] | None = None

    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        """调用 Responses API 并提取第一段 output_text。"""

        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise LlmClientError("OPENAI_API_KEY is not configured")

        payload = {
            "model": model,
            "input": prompt,
            "temperature": temperature,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "evalrag_generation",
                    "strict": True,
                    "schema": _generation_json_schema(),
                }
            },
        }

        request = Request(
            self.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),#把python对象序列化成 JSON 格式字符串
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))#将 JSON 字符串反序列化成 python对象
        except HTTPError as error:
            raise LlmClientError(
                f"OpenAI Responses API returned HTTP {error.code}"
            ) from error
        except TimeoutError as error:
            raise LlmTimeoutError("OpenAI Responses API request timed out") from error
        except (URLError, json.JSONDecodeError) as error:
            raise LlmClientError("OpenAI Responses API request failed") from error

        output_text = _extract_output_text(response_data) #提取模型输出文本
        if output_text is None:
            raise LlmClientError("OpenAI response does not contain output_text")
        self.last_token_usage = _extract_token_usage(response_data)
        return output_text


class DeepSeekChatClient:
    """通过 OpenAI 兼容 Chat Completions 调用 DeepSeek JSON Mode。"""

    def __init__(
        self,
        timeout_seconds: float = 60.0,
        *,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str = "https://api.deepseek.com",
        max_tokens: int = 1200,
        system_instruction: str = "你是 EvalRAG 的结构化答案生成器，只输出合法 JSON。",
        max_retries: int = 2,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.timeout_seconds = timeout_seconds
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.system_instruction = system_instruction
        self.max_retries = max_retries
        self.last_token_usage: dict[str, int] | None = None
        self.last_response_metadata: dict[str, str] = {}

    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        """调用非思考模式并返回 JSON 文本，密钥只从环境变量读取。"""

        self.last_token_usage = None
        self.last_response_metadata = {}
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise LlmClientError(f"{self.api_key_env} is not configured")

        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                OpenAI,
            )
        except ImportError as error:
            raise LlmClientError("openai package is not installed") from error

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instruction,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
            )
        except APITimeoutError as error:
            raise LlmTimeoutError("DeepSeek API request timed out") from error
        except APIStatusError as error:
            raise LlmClientError(
                f"DeepSeek API returned HTTP {error.status_code}"
            ) from error
        except APIConnectionError as error:
            raise LlmClientError("DeepSeek API connection failed") from error

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise LlmClientError("DeepSeek response does not contain content")
        self.last_token_usage = _extract_chat_token_usage(response.usage)
        self.last_response_metadata = {
            "response_id": str(response.id),
            "model": str(response.model),
        }
        return content


def build_generation_prompt(
    query: str,
    context: BuiltContext,
    prompt_version: str,
) -> str:
    """构造只允许依据本轮 Context 回答的结构化生成 Prompt。

    “本轮 Context”就是针对这一次问题检索并经过预算筛选后，真正提供给模型的证据。
    """

    allowed_ids = ", ".join(context.used_chunk_ids) or "无"
    evidence_text = context.text or "（本轮没有检索到可用证据）"
    return (
        f"prompt_version: {prompt_version}\n"
        "你是 EvalRAG 的回答生成器。请严格遵守以下规则：\n"
        "1. 只能依据“检索证据”回答，不得补充上下文外事实。\n"
        "2. cited_chunk_ids 只能使用“允许引用的 chunk id”中的值。\n"
        "3. sufficient=true 时，answer 必须有依据且 cited_chunk_ids 不能为空。\n"
        "4. 证据不足时返回 sufficient=false、cited_chunk_ids=[]，"
        "并在 answer 和 reason 中说明不确定性。\n"
        "5. 只输出一个 JSON 对象，不要输出 Markdown 代码块或额外文字。\n"
        "6. JSON 必须包含 answer、cited_chunk_ids、sufficient、reason 四个字段。\n\n"
        f"用户问题：\n{query.strip()}\n\n"
        f"允许引用的 chunk id：{allowed_ids}\n\n"
        f"检索证据：\n{evidence_text}"
    )


def generate_answer(
    query: str,
    context: BuiltContext,
    llm_client: LlmClient,
    *,
    model: str,
    temperature: float,
    prompt_version: str,
) -> GenerationResult:
    """调用注入的 LLM client，并把原始输出解析为 GenerationResult。
                            ┌─ FakeLlmClient
        generate_answer() ──┤
                            └─ OpenAIResponsesClient

        FakeLlmClient 和 OpenAIResponsesClient 都提供了相同形式的 generate()，
        所以类型检查器可以把它们视为 LlmClient。
    """

    prompt = build_generation_prompt(query, context, prompt_version)
    raw_output = llm_client.generate(
        prompt,
        model=model,
        temperature=temperature,
    )
    return parse_generation_result(raw_output)


def parse_generation_result(raw_output: str) -> GenerationResult:
    """严格解析模型 JSON，并区分语法、缺字段和字段类型错误。"""

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise GenerationParseError(
            "invalid_json",
            "model output is not valid JSON",
        ) from error

    if not isinstance(payload, dict):
        raise GenerationParseError(
            "invalid_field_type",
            "model output root must be a JSON object",
        )

    required_fields = {"answer", "cited_chunk_ids", "sufficient", "reason"}
    missing_fields = sorted(required_fields - payload.keys())
    if missing_fields:
        raise GenerationParseError(
            "missing_field",
            f"model output misses fields: {', '.join(missing_fields)}",
        )

    answer = payload["answer"]
    cited_chunk_ids = payload["cited_chunk_ids"]
    sufficient = payload["sufficient"]
    reason = payload["reason"]
    if not isinstance(answer, str):
        raise _field_type_error("answer", "string")
    if not isinstance(cited_chunk_ids, list) or not all(
        isinstance(chunk_id, str) for chunk_id in cited_chunk_ids
    ):
        raise _field_type_error("cited_chunk_ids", "list[string]")
    if not isinstance(sufficient, bool):
        raise _field_type_error("sufficient", "boolean")
    if not isinstance(reason, str):
        raise _field_type_error("reason", "string")

    return GenerationResult(
        answer=answer,
        cited_chunk_ids=cited_chunk_ids,
        sufficient=sufficient,
        reason=reason,
    )


def _field_type_error(field_name: str, expected_type: str) -> GenerationParseError:
    """构造统一的字段类型错误。"""

    return GenerationParseError(
        "invalid_field_type",
        f"{field_name} must be {expected_type}",
    )


def _generation_json_schema() -> dict[str, object]:
    """返回真实模型 adapter 使用的严格 JSON Schema。
        正常情况下模型会按照 Schema 输出
    """

    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "cited_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "sufficient": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["answer", "cited_chunk_ids", "sufficient", "reason"],
        "additionalProperties": False,
    }


def _extract_output_text(response_data: object) -> str | None:
    """从 Responses API 原始 JSON 中提取模型文本。"""

    if not isinstance(response_data, dict):
        return None
    output = response_data.get("output")
    if not isinstance(output, list):
        return None

    for output_item in output:
        if not isinstance(output_item, dict):
            continue
        content = output_item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") != "output_text":
                continue
            text = content_item.get("text")
            if isinstance(text, str):
                return text
    return None


def _extract_token_usage(response_data: object) -> dict[str, int] | None:
    """提取真实 API 返回的 token usage；字段缺失时不做估算。"""

    if not isinstance(response_data, dict):
        return None
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        return None
    supported = ("input_tokens", "output_tokens", "total_tokens")
    parsed = {
        name: int(usage[name])
        for name in supported
        if isinstance(usage.get(name), int)
    }
    return parsed or None


def _extract_chat_token_usage(usage: object) -> dict[str, int] | None:
    """提取 Chat Completions usage，包括 DeepSeek 缓存命中字段。"""

    if usage is None:
        return None
    fields = {
        "input_tokens": "prompt_tokens",
        "output_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
        "prompt_cache_hit_tokens": "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens": "prompt_cache_miss_tokens",
    }
    parsed: dict[str, int] = {}
    for target, source in fields.items():
        value = getattr(usage, source, None)
        if isinstance(value, int):
            parsed[target] = value
    return parsed or None
