import unittest

from intern_rag.routing import RouteDecision, route_query


class RoutingTests(unittest.TestCase):
    def test_route_jd_analysis_query(self) -> None:
        decision = route_query("帮我分析这个大模型应用研发实习岗位的职责和要求")

        self.assertIsInstance(decision, RouteDecision)
        self.assertEqual(decision.intent, "analyze_jd")
        self.assertEqual(decision.routed_sources, ["jd"])
        self.assertIn("岗位", decision.matched_keywords)

    def test_route_resume_match_query(self) -> None:
        decision = route_query("我的简历和这个 JD 匹配吗，有哪些差距")

        self.assertEqual(decision.intent, "match_resume")
        self.assertEqual(decision.routed_sources, ["jd", "resume"])
        self.assertIn("简历", decision.matched_keywords)

    def test_route_interview_prepare_query(self) -> None:
        decision = route_query("面试 RAG 项目时常见追问应该怎么回答")

        self.assertEqual(decision.intent, "interview_prepare")
        self.assertEqual(decision.routed_sources, ["interview", "jd", "resume"])
        self.assertIn("面试", decision.matched_keywords)

    def test_route_project_explain_query(self) -> None:
        decision = route_query("这个多源 RAG 项目有哪些亮点和实现细节")

        self.assertEqual(decision.intent, "project_explain")
        self.assertEqual(decision.routed_sources, ["project_logs", "resume"])
        self.assertIn("项目", decision.matched_keywords)

    def test_route_application_plan_query(self) -> None:
        decision = route_query("根据我的目标城市和求职规划，下一周应该怎么投递")

        self.assertEqual(decision.intent, "application_plan")
        self.assertEqual(decision.routed_sources, ["user_profile", "jd", "resume"])
        self.assertIn("投递", decision.matched_keywords)

    def test_unknown_query_returns_unknown(self) -> None:
        decision = route_query("今天午饭吃什么")

        self.assertEqual(decision.intent, "unknown")
        self.assertEqual(decision.routed_sources, [])
        self.assertEqual(decision.matched_keywords, [])

    def test_empty_query_returns_unknown(self) -> None:
        decision = route_query("   ")

        self.assertEqual(decision.intent, "unknown")
        self.assertEqual(decision.routed_sources, [])


if __name__ == "__main__":
    unittest.main()
