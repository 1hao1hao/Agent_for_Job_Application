from __future__ import annotations

from itertools import count
import json
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

from locust import HttpUser, between, task

from loadtest.scenarios import QUERY_CASES


_case_counter = count()
_ledger_lock = Lock()


def _session_scopes() -> list[dict[str, str]]:
    value = os.environ.get("EVALRAG_LOADTEST_SESSIONS", "").strip()
    if not value:
        return []
    return [
        {"user_id": str(item["user_id"]), "session_id": str(item["session_id"])}
        for item in json.loads(value)
    ]


def _write_ledger(record: dict[str, object]) -> None:
    path_value = os.environ.get("EVALRAG_LOADTEST_LEDGER", "").strip()
    if not path_value:
        return
    path = Path(path_value)
    with _ledger_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


class EvalRagQueryUser(HttpUser):
    """轮询四类 Query，避免吞吐结果只代表单一检索路径。"""

    wait_time = between(0.05, 0.15)

    def on_start(self) -> None:
        self.session_scopes = _session_scopes()

    @task
    def query(self) -> None:
        case_index = next(_case_counter)
        case = QUERY_CASES[case_index % len(QUERY_CASES)]
        payload = {key: value for key, value in case.items() if key != "name"}
        request_id = str(uuid4())
        payload["request_id"] = request_id
        # 固定 30% session 请求，使 History cache 与 PostgreSQL 回源进入真实链路。
        if self.session_scopes and case_index % 10 < 3:
            payload.update(self.session_scopes[case_index % len(self.session_scopes)])
        started = perf_counter()
        with self.client.post(
            "/v1/query",
            json=payload,
            name=f"POST /v1/query [{case['name']}]",
            catch_response=True,
        ) as response:
            response_payload: dict[str, object] = {}
            try:
                response_payload = response.json()
            except Exception:
                pass
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            status = response_payload.get("status")
            if response.status_code == 200 and status not in {
                "answered", "insufficient_evidence"
            }:
                response.failure(f"unexpected response status: {status}")
            _write_ledger({
                "request_id": request_id,
                "trace_id": response_payload.get("trace_id"),
                "case_type": case["name"],
                "session_request": "session_id" in payload,
                "http_status": response.status_code,
                "response_status": status,
                "latency_ms": (perf_counter() - started) * 1000,
                "error_type": response_payload.get("error_type"),
            })
