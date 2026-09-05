import unittest

from intern_rag.agent import (
    AgentController,
    AgentControllerConfig,
    AgentState,
)


class AgentControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = AgentController(
            AgentControllerConfig(max_action_steps=4)
        )
        self.initial = AgentState(
            requested_retriever="adaptive",
            route_intent="match_resume",
            routed_sources=("jd", "resume"),
        )

    def test_sufficient_evidence_selects_retrieve_then_generate(self) -> None:
        retrieve = self.controller.choose(self.initial)
        generate = self.controller.choose(AgentState(
            **{
                **self.initial.__dict__,
                "step": retrieve.step,
                "evidence_status": "sufficient",
                "evidence_reason": "sufficient_evidence",
            }
        ))

        self.assertEqual(retrieve.action, "retrieve")
        self.assertEqual(generate.action, "generate")
        self.assertEqual(generate.step, 2)

    def test_retryable_evidence_expands_once_then_abstains(self) -> None:
        action = self.controller.choose(AgentState(
            **{
                **self.initial.__dict__,
                "step": 1,
                "evidence_status": "retryable",
                "source_retry_count": 0,
            }
        ))
        exhausted = self.controller.choose(AgentState(
            **{
                **self.initial.__dict__,
                "step": action.step,
                "evidence_status": "insufficient",
                "source_retry_count": 1,
            }
        ))

        self.assertEqual(action.action, "expand_sources")
        self.assertEqual(exhausted.action, "abstain")

    def test_format_repair_is_bounded(self) -> None:
        repair = self.controller.choose(AgentState(
            **{
                **self.initial.__dict__,
                "step": 2,
                "evidence_status": "sufficient",
                "generation_started": True,
                "generation_format_error": True,
                "format_retry_count": 0,
            }
        ))
        exhausted = self.controller.choose(AgentState(
            **{
                **self.initial.__dict__,
                "step": 3,
                "evidence_status": "sufficient",
                "generation_started": True,
                "generation_format_error": True,
                "format_retry_count": 1,
            }
        ))

        self.assertEqual(repair.action, "generate")
        self.assertEqual(exhausted.action, "abstain")

    def test_explicit_out_of_domain_can_abstain_before_retrieval(self) -> None:
        action = self.controller.choose(AgentState(
            requested_retriever="adaptive",
            route_intent="unknown",
            routed_sources=(),
            explicit_out_of_domain=True,
        ))

        self.assertEqual(action.action, "abstain")
        self.assertEqual(action.step, 1)

    def test_action_budget_has_non_overridable_upper_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 4"):
            AgentControllerConfig(max_action_steps=5)

        action = self.controller.choose(AgentState(
            **{**self.initial.__dict__, "step": 4, "evidence_status": "retryable"}
        ))
        self.assertEqual(action.action, "abstain")
        self.assertEqual(action.step, 4)


if __name__ == "__main__":
    unittest.main()
