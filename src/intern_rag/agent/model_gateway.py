from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from threading import BoundedSemaphore, Lock
from time import monotonic, perf_counter, sleep
from typing import Callable, Literal, Sequence

from intern_rag.agent.generation import LlmClient, LlmClientError, LlmTimeoutError
from intern_rag.agent.generation import DeepSeekChatClient, OpenAIResponsesClient


CircuitState = Literal["closed", "open", "half_open"]


@dataclass(frozen=True)
class GatewayProvider:
    """一个可调用 Provider 及其模型、版本和价格快照。"""

    name: str
    client: LlmClient
    model: str
    input_usd_per_million: float = 0.0
    output_usd_per_million: float = 0.0


@dataclass(frozen=True)
class ModelGatewayConfig:
    """Model Gateway 的所有硬上限。"""

    max_attempts_per_provider: int = 2
    timeout_seconds: float = 90.0
    backoff_base_seconds: float = 0.25
    backoff_max_seconds: float = 2.0
    concurrency_limit: int = 4
    circuit_failure_threshold: int = 3
    circuit_recovery_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts_per_provider <= 0 or self.concurrency_limit <= 0:
            raise ValueError("attempt and concurrency limits must be positive")
        if self.timeout_seconds <= 0 or self.circuit_recovery_seconds <= 0:
            raise ValueError("timeout and recovery seconds must be positive")
        if self.backoff_base_seconds < 0 or self.backoff_max_seconds < 0:
            raise ValueError("backoff must not be negative")
        if self.circuit_failure_threshold <= 0:
            raise ValueError("circuit_failure_threshold must be positive")


@dataclass
class _Circuit:
    state: CircuitState = "closed"
    failures: int = 0
    opened_at: float | None = None
    probe_in_flight: bool = False


class ModelGatewayUnavailable(LlmClientError):
    """所有 Provider 都失败或被熔断后的受控错误。"""


class ModelGateway:
    """实现有界重试、fallback、并发限制和熔断的 LlmClient。

    Gateway 按 Provider 顺序调用。每个 Provider 只对 timeout、连接错误、429 和 5xx
    重试；鉴权和请求错误直接进入下一个 Provider。失败达到阈值后熔断，恢复窗口后
    仅允许一次 half-open 探测。`last_gateway_trace` 只保存结构化元数据，不保存 Prompt。
    """

    def __init__(
        self,
        providers: Sequence[GatewayProvider],
        config: ModelGatewayConfig = ModelGatewayConfig(),
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        names = [item.name for item in providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")
        self.providers = tuple(providers)
        self.config = config
        self.clock = clock
        self.sleeper = sleeper
        self._circuits = {item.name: _Circuit() for item in providers}
        self._lock = Lock()
        self._semaphore = BoundedSemaphore(config.concurrency_limit)
        self._pool = ThreadPoolExecutor(
            max_workers=config.concurrency_limit,
            thread_name_prefix="evalrag-model-gateway",
        )
        self._trace: ContextVar[dict[str, object]] = ContextVar(
            f"model_gateway_trace_{id(self)}", default={}
        )
        self._usage: ContextVar[dict[str, int] | None] = ContextVar(
            f"model_gateway_usage_{id(self)}", default=None
        )
        self.last_response_metadata: dict[str, str] = {}

    @property
    def last_gateway_trace(self) -> dict[str, object]:
        return dict(self._trace.get())

    @property
    def last_token_usage(self) -> dict[str, int] | None:
        usage = self._usage.get()
        return dict(usage) if usage is not None else None

    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        """按优先级调用 Provider，成功即返回，全部失败则快速受控报错。"""

        del model
        attempts: list[dict[str, object]] = []
        started = perf_counter()
        self._usage.set(None)
        selected: str | None = None
        for provider_index, provider in enumerate(self.providers):
            if not self._allow_call(provider.name):
                attempts.append({
                    "provider": provider.name, "attempt": 0,
                    "status": "circuit_open", "reason": "circuit_open",
                    "latency_ms": 0.0, "tokens": None, "estimated_cost_usd": 0.0,
                })
                continue
            for attempt_number in range(1, self.config.max_attempts_per_provider + 1):
                call_started = perf_counter()
                try:
                    if not self._semaphore.acquire(timeout=self.config.timeout_seconds):
                        raise LlmTimeoutError("gateway concurrency slot timed out")
                    future = self._pool.submit(
                        provider.client.generate,
                        prompt,
                        model=provider.model,
                        temperature=temperature,
                    )
                    future.add_done_callback(lambda _: self._semaphore.release())
                    output = future.result(timeout=self.config.timeout_seconds)
                    latency = (perf_counter() - call_started) * 1000
                    usage = _client_usage(provider.client)
                    cost = _estimate_cost(provider, usage)
                    attempts.append({
                        "provider": provider.name, "attempt": attempt_number,
                        "status": "succeeded", "reason": "provider_success",
                        "latency_ms": latency, "tokens": usage,
                        "estimated_cost_usd": cost,
                    })
                    self._record_success(provider.name)
                    self._usage.set(usage)
                    selected = provider.name
                    self.last_response_metadata = {
                        "provider": provider.name,
                        "model": provider.model,
                    }
                    self._set_trace(attempts, selected, provider_index > 0, started)
                    return output
                except FutureTimeout:
                    error = LlmTimeoutError("provider call exceeded gateway timeout")
                except Exception as caught:
                    error = caught
                latency = (perf_counter() - call_started) * 1000
                kind, transient = _classify_error(error)
                attempts.append({
                    "provider": provider.name, "attempt": attempt_number,
                    "status": "failed", "reason": kind,
                    "latency_ms": latency, "tokens": None,
                    "estimated_cost_usd": 0.0,
                })
                self._record_failure(provider.name)
                if not transient or attempt_number >= self.config.max_attempts_per_provider:
                    break
                delay = min(
                    self.config.backoff_base_seconds * (2 ** (attempt_number - 1)),
                    self.config.backoff_max_seconds,
                )
                if delay:
                    self.sleeper(delay)
        self._set_trace(attempts, selected, False, started)
        raise ModelGatewayUnavailable("all configured model providers are unavailable")

    def circuit_states(self) -> dict[str, str]:
        """返回可观测熔断状态，不暴露客户端或密钥。"""

        with self._lock:
            return {name: circuit.state for name, circuit in self._circuits.items()}

    def _allow_call(self, name: str) -> bool:
        with self._lock:
            circuit = self._circuits[name]
            if circuit.state != "open":
                return not (circuit.state == "half_open" and circuit.probe_in_flight)
            assert circuit.opened_at is not None
            if self.clock() - circuit.opened_at < self.config.circuit_recovery_seconds:
                return False
            circuit.state = "half_open"
            circuit.probe_in_flight = True
            return True

    def _record_success(self, name: str) -> None:
        with self._lock:
            circuit = self._circuits[name]
            circuit.state = "closed"
            circuit.failures = 0
            circuit.opened_at = None
            circuit.probe_in_flight = False

    def _record_failure(self, name: str) -> None:
        with self._lock:
            circuit = self._circuits[name]
            circuit.probe_in_flight = False
            circuit.failures += 1
            if circuit.failures >= self.config.circuit_failure_threshold:
                circuit.state = "open"
                circuit.opened_at = self.clock()

    def _set_trace(
        self,
        attempts: list[dict[str, object]],
        selected: str | None,
        fallback_used: bool,
        started: float,
    ) -> None:
        self._trace.set({
            "selected_provider": selected,
            "fallback_used": fallback_used,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "total_latency_ms": (perf_counter() - started) * 1000,
            "estimated_cost_usd": sum(float(item["estimated_cost_usd"]) for item in attempts),
            "circuit_states": self.circuit_states(),
        })


def build_model_gateway_from_config(config_or_path: dict[str, object] | Path) -> ModelGateway:
    """从版本化配置构造 Gateway，只启用已配置凭证的 Provider。

    输入可以是配置字典或 JSON 路径。函数只读取环境变量是否存在，不读取或记录密钥值；
    DeepSeek 与 OpenAI-compatible Responses adapter 都由 Gateway 统一控制 timeout/retry。
    """

    if isinstance(config_or_path, Path):
        config = json.loads(config_or_path.read_text(encoding="utf-8"))
    else:
        config = dict(config_or_path)
    gateway_raw = dict(config.get("gateway", {}))
    providers: list[GatewayProvider] = []
    for raw_item in config.get("providers", []):
        item = dict(raw_item)
        api_key_env = str(item["api_key_env"])
        if not os.environ.get(api_key_env, "").strip():
            continue
        adapter = str(item["adapter"])
        timeout = float(gateway_raw.get("timeout_seconds", 60.0))
        if adapter == "deepseek_chat":
            client: LlmClient = DeepSeekChatClient(
                timeout_seconds=timeout,
                api_key_env=api_key_env,
                base_url=str(item.get("base_url", "https://api.deepseek.com")),
                max_tokens=int(item.get("max_tokens", 1200)),
                max_retries=0,
            )
        elif adapter == "openai_responses":
            if api_key_env != "OPENAI_API_KEY":
                raise ValueError("OpenAIResponsesClient requires OPENAI_API_KEY")
            client = OpenAIResponsesClient(timeout_seconds=timeout)
        else:
            raise ValueError(f"unknown model provider adapter: {adapter}")
        providers.append(GatewayProvider(
            name=str(item["name"]),
            client=client,
            model=str(item["model"]),
            input_usd_per_million=float(item.get("input_usd_per_million", 0.0)),
            output_usd_per_million=float(item.get("output_usd_per_million", 0.0)),
        ))
    if not providers:
        raise ModelGatewayUnavailable("no configured model provider credential is available")
    return ModelGateway(providers, ModelGatewayConfig(**gateway_raw))


def _classify_error(error: Exception) -> tuple[str, bool]:
    if isinstance(error, LlmTimeoutError):
        return "timeout", True
    message = str(error).lower()
    status_match = re.search(r"http\s+(\d{3})", message)
    status = int(status_match.group(1)) if status_match else None
    if status in {401, 403} or "api_key" in message or "auth" in message:
        return "authentication_error", False
    if status == 429:
        return "rate_limited", True
    if status is not None and 500 <= status <= 599:
        return "provider_5xx", True
    if "connection" in message:
        return "connection_error", True
    return "provider_error", False


def _client_usage(client: LlmClient) -> dict[str, int] | None:
    usage = getattr(client, "last_token_usage", None)
    return dict(usage) if isinstance(usage, dict) else None


def _estimate_cost(provider: GatewayProvider, usage: dict[str, int] | None) -> float:
    if usage is None:
        return 0.0
    return (
        int(usage.get("input_tokens", 0)) * provider.input_usd_per_million
        + int(usage.get("output_tokens", 0)) * provider.output_usd_per_million
    ) / 1_000_000
