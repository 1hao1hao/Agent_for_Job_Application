from types import SimpleNamespace
import unittest

from intern_rag.agent import RagRequest, RagResponse
from intern_rag.serving.service import PipelineQueryService
from tests.support import InMemoryPersistenceRepository


class RuntimeEntrypointTests(unittest.TestCase):
    def test_http_query_service_passes_versioned_run_context(self) -> None:
        class RecordingRuntime:
            def __init__(self) -> None:
                self.context = None

            def execute(self, request, context):
                self.context = context
                response = RagResponse(
                    request.request_id, "trace-http", "ok", [], [],
                    "insufficient_evidence", 1.0, None,
                )
                return SimpleNamespace(response=response)

        runtime = RecordingRuntime()
        pipeline = SimpleNamespace(config=SimpleNamespace(
            router_name="hybrid", model="fake-v1", prompt_version="prompt-v1"
        ))
        repository = InMemoryPersistenceRepository()
        service = PipelineQueryService(pipeline, repository, runtime=runtime)
        request = RagRequest("HTTP 入口测试", retriever="hybrid")

        response = service.execute(request)

        self.assertEqual(response.trace_id, "trace-http")
        self.assertEqual(runtime.context.entrypoint, "http")
        self.assertEqual(runtime.context.model_version, "fake-v1")
        self.assertIn(request.request_id, repository.requests)


if __name__ == "__main__":
    unittest.main()
