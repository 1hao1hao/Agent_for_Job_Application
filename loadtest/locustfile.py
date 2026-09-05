from __future__ import annotations

from locust import HttpUser, between, task

from loadtest.scenarios import QUERY_CASES


class EvalRagQueryUser(HttpUser):
    """轮询四类 Query，避免吞吐结果只代表单一检索路径。"""

    wait_time = between(0.05, 0.15)

    def on_start(self) -> None:
        self.case_index = 0

    @task
    def query(self) -> None:
        case = QUERY_CASES[self.case_index % len(QUERY_CASES)]
        self.case_index += 1
        payload = {key: value for key, value in case.items() if key != "name"}
        with self.client.post(
            "/v1/query",
            json=payload,
            name=f"POST /v1/query [{case['name']}]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            status = response.json().get("status")
            if status not in {"answered", "insufficient_evidence"}:
                response.failure(f"unexpected response status: {status}")
