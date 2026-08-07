import json
import os
import unittest
from unittest.mock import patch

from intern_rag.agent import (
    BuiltContext,
    DeepSeekChatClient,
    FakeLlmClient,
    GenerationParseError,
    LlmClientError,
    OpenAIResponsesClient,
    build_generation_prompt,
    generate_answer,
    parse_generation_result,
)


def _context() -> BuiltContext:
    return BuiltContext(
        query="岗位要求是什么？",
        text=(
            "chunk_id: jd-1\n"
            "source_type: jd\n"
            "source_path: data/raw/jd/test.md\n"
            "title: 测试岗位\n"
            "rank: 1\n"
            "text:\n岗位要求熟悉 Python。"
        ),
        items=[],
        used_chunk_ids=["jd-1"],
        skipped_chunk_ids=[],
        char_count=120,
        max_chars=1000,
    )


class GenerationTests(unittest.TestCase):
    def test_prompt_contains_grounding_and_citation_constraints(self) -> None:
        prompt = build_generation_prompt("岗位要求是什么？", _context(), "p0-v1")

        self.assertIn("只能依据", prompt)
        self.assertIn("不得补充上下文外事实", prompt)
        self.assertIn("证据不足", prompt)
        self.assertIn("cited_chunk_ids", prompt)
        self.assertIn("jd-1", prompt)
        self.assertIn("prompt_version: p0-v1", prompt)

    def test_generate_answer_uses_fake_llm_and_parses_valid_json(self) -> None:
        raw_response = json.dumps(
            {
                "answer": "岗位要求熟悉 Python。",
                "cited_chunk_ids": ["jd-1"],
                "sufficient": True,
                "reason": "证据直接说明岗位要求。",
            },
            ensure_ascii=False,
        )
        client = FakeLlmClient([raw_response])

        result = generate_answer(
            "岗位要求是什么？",
            _context(),
            client,
            model="fake-model",
            temperature=0.0,
            prompt_version="p0-v1",
        )

        self.assertEqual(result.answer, "岗位要求熟悉 Python。")
        self.assertEqual(result.cited_chunk_ids, ["jd-1"])
        self.assertTrue(result.sufficient)
        self.assertEqual(len(client.prompts), 1)

    def test_invalid_json_has_controlled_error_type(self) -> None:
        with self.assertRaises(GenerationParseError) as raised:
            parse_generation_result("not-json")

        self.assertEqual(raised.exception.error_type, "invalid_json")

    def test_missing_field_has_controlled_error_type(self) -> None:
        raw_response = json.dumps(
            {
                "answer": "回答",
                "cited_chunk_ids": ["jd-1"],
                "sufficient": True,
            }
        )

        with self.assertRaises(GenerationParseError) as raised:
            parse_generation_result(raw_response)

        self.assertEqual(raised.exception.error_type, "missing_field")
        self.assertIn("reason", str(raised.exception))

    def test_invalid_field_type_has_controlled_error_type(self) -> None:
        raw_response = json.dumps(
            {
                "answer": "回答",
                "cited_chunk_ids": "jd-1",
                "sufficient": "true",
                "reason": "",
            }
        )

        with self.assertRaises(GenerationParseError) as raised:
            parse_generation_result(raw_response)

        self.assertEqual(raised.exception.error_type, "invalid_field_type")

    def test_real_adapter_requires_environment_api_key_without_network(self) -> None:
        client = OpenAIResponsesClient()

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LlmClientError):
                client.generate("test", model="gpt-4.1-mini", temperature=0.0)

    def test_deepseek_adapter_requires_environment_api_key_without_network(self) -> None:
        client = DeepSeekChatClient()

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(LlmClientError):
                client.generate(
                    "输出 JSON", model="deepseek-v4-flash", temperature=0.0
                )

    def test_deepseek_adapter_uses_json_mode_and_records_usage(self) -> None:
        class FakeUsage:
            prompt_tokens = 100
            completion_tokens = 20
            total_tokens = 120
            prompt_cache_hit_tokens = 60
            prompt_cache_miss_tokens = 40

        class FakeResponse:
            id = "response-test"
            model = "deepseek-v4-flash"
            usage = FakeUsage()
            choices = [
                type(
                    "Choice",
                    (),
                    {
                        "message": type(
                            "Message", (), {"content": '{"answer":"ok"}'}
                        )()
                    },
                )()
            ]

        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.chat = type(
                    "Chat", (), {"completions": FakeCompletions()}
                )()

        client = DeepSeekChatClient(max_tokens=500)
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True):
            with patch("openai.OpenAI", FakeOpenAI):
                output = client.generate(
                    "请输出 JSON", model="deepseek-v4-flash", temperature=0.0
                )

        self.assertEqual(output, '{"answer":"ok"}')
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(
            calls[0]["extra_body"], {"thinking": {"type": "disabled"}}
        )
        self.assertEqual(client.last_token_usage["input_tokens"], 100)
        self.assertEqual(client.last_token_usage["prompt_cache_hit_tokens"], 60)


if __name__ == "__main__":
    unittest.main()
