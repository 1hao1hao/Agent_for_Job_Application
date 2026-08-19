from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
from unittest.mock import patch
import unittest

from intern_rag.agent import (
    GatewayProvider,
    LlmClientError,
    LlmTimeoutError,
    ModelGateway,
    ModelGatewayConfig,
    ModelGatewayUnavailable,
    build_model_gateway_from_config,
)


class FakeProvider:
    def __init__(self, outcomes, *, delay: float = 0.0) -> None:
        self.outcomes = list(outcomes)
        self.delay = delay
        self.calls = 0
        self.last_token_usage = {"input_tokens": 100, "output_tokens": 20}
        self.active = 0
        self.max_active = 0
        self.lock = Lock()

    def generate(self, prompt: str, *, model: str, temperature: float) -> str:
        del prompt, model, temperature
        with self.lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                sleep(self.delay)
            outcome = self.outcomes.pop(0) if self.outcomes else "ok"
            if isinstance(outcome, Exception):
                raise outcome
            return str(outcome)
        finally:
            with self.lock:
                self.active -= 1


def _provider(name: str, client: FakeProvider) -> GatewayProvider:
    return GatewayProvider(name, client, f"{name}-model", 1.0, 2.0)


class ModelGatewayTests(unittest.TestCase):
    def test_primary_success_records_tokens_cost_and_provider(self) -> None:
        primary = FakeProvider(["primary-ok"])
        gateway = ModelGateway([_provider("primary", primary)])

        output = gateway.generate("prompt", model="ignored", temperature=0.0)

        self.assertEqual(output, "primary-ok")
        self.assertEqual(gateway.last_token_usage["input_tokens"], 100)
        self.assertEqual(gateway.last_gateway_trace["selected_provider"], "primary")
        self.assertAlmostEqual(gateway.last_gateway_trace["estimated_cost_usd"], 0.00014)

    def test_timeout_uses_fallback_with_hard_attempt_limit(self) -> None:
        primary = FakeProvider([LlmTimeoutError("timeout"), LlmTimeoutError("timeout")])
        backup = FakeProvider(["backup-ok"])
        gateway = ModelGateway(
            [_provider("primary", primary), _provider("backup", backup)],
            ModelGatewayConfig(max_attempts_per_provider=2, backoff_base_seconds=0),
        )

        self.assertEqual(gateway.generate("p", model="x", temperature=0), "backup-ok")
        self.assertEqual(primary.calls, 2)
        self.assertEqual(backup.calls, 1)
        self.assertTrue(gateway.last_gateway_trace["fallback_used"])

    def test_429_and_5xx_retry_but_auth_does_not(self) -> None:
        transient = FakeProvider([
            LlmClientError("provider HTTP 429"),
            LlmClientError("provider HTTP 503"),
            "ok",
        ])
        gateway = ModelGateway(
            [_provider("primary", transient)],
            ModelGatewayConfig(max_attempts_per_provider=3, backoff_base_seconds=0),
        )
        self.assertEqual(gateway.generate("p", model="x", temperature=0), "ok")
        self.assertEqual(transient.calls, 3)

        auth = FakeProvider([LlmClientError("provider HTTP 401")])
        backup = FakeProvider(["backup"])
        gateway = ModelGateway(
            [_provider("primary", auth), _provider("backup", backup)],
            ModelGatewayConfig(max_attempts_per_provider=3, backoff_base_seconds=0),
        )
        self.assertEqual(gateway.generate("p", model="x", temperature=0), "backup")
        self.assertEqual(auth.calls, 1)

    def test_circuit_open_half_open_and_close(self) -> None:
        now = [0.0]
        primary = FakeProvider([LlmTimeoutError("timeout"), "recovered"])
        gateway = ModelGateway(
            [_provider("primary", primary)],
            ModelGatewayConfig(
                max_attempts_per_provider=1,
                circuit_failure_threshold=1,
                circuit_recovery_seconds=10,
                backoff_base_seconds=0,
            ),
            clock=lambda: now[0],
        )
        with self.assertRaises(ModelGatewayUnavailable):
            gateway.generate("p", model="x", temperature=0)
        self.assertEqual(gateway.circuit_states()["primary"], "open")
        with self.assertRaises(ModelGatewayUnavailable):
            gateway.generate("p", model="x", temperature=0)
        self.assertEqual(primary.calls, 1)

        now[0] = 11.0
        self.assertEqual(gateway.generate("p", model="x", temperature=0), "recovered")
        self.assertEqual(gateway.circuit_states()["primary"], "closed")

    def test_provider_timeout_and_all_unavailable_are_controlled(self) -> None:
        slow = FakeProvider(["late"], delay=0.05)
        failed = FakeProvider([LlmClientError("provider HTTP 400")])
        gateway = ModelGateway(
            [_provider("slow", slow), _provider("failed", failed)],
            ModelGatewayConfig(
                max_attempts_per_provider=1,
                timeout_seconds=0.005,
                concurrency_limit=2,
            ),
        )
        with self.assertRaisesRegex(ModelGatewayUnavailable, "providers are unavailable"):
            gateway.generate("secret prompt", model="x", temperature=0)
        self.assertNotIn("secret prompt", str(gateway.last_gateway_trace))

    def test_shared_executor_enforces_concurrency_limit(self) -> None:
        provider = FakeProvider([], delay=0.02)
        gateway = ModelGateway(
            [_provider("primary", provider)],
            ModelGatewayConfig(concurrency_limit=2, max_attempts_per_provider=1),
        )
        with ThreadPoolExecutor(max_workers=6) as pool:
            outputs = list(pool.map(
                lambda _: gateway.generate("p", model="x", temperature=0),
                range(6),
            ))
        self.assertEqual(outputs, ["ok"] * 6)
        self.assertLessEqual(provider.max_active, 2)

    def test_config_builder_enables_only_providers_with_credentials(self) -> None:
        config = {
            "gateway": {"max_attempts_per_provider": 1},
            "providers": [
                {"name": "deepseek", "adapter": "deepseek_chat", "model": "deepseek-test", "api_key_env": "DEEPSEEK_API_KEY"},
                {"name": "openai", "adapter": "openai_responses", "model": "backup-test", "api_key_env": "OPENAI_API_KEY"},
            ],
        }
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            gateway = build_model_gateway_from_config(config)
        self.assertEqual([item.name for item in gateway.providers], ["deepseek"])

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ModelGatewayUnavailable):
                build_model_gateway_from_config(config)


if __name__ == "__main__":
    unittest.main()
