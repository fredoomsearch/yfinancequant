from __future__ import annotations

import unittest

from assistant.contracts import AssistantRoute, AssistantState
from assistant.executor import AssistantExecutor
from assistant.policy import AssistantPolicyEngine
from schemas.planner import AssistantPlan, PlannerStep


class AssistantExecutorTest(unittest.TestCase):
    def test_policy_blocks_unknown_tools(self) -> None:
        policy = AssistantPolicyEngine()
        plan = AssistantPlan(
            intent="show_latest_summary",
            route={"intent": "show_latest_summary", "language": "en"},
            risk="low",
            steps=[
                PlannerStep(tool="load_run_bundle"),
                PlannerStep(tool="shell_outside_guardrails"),
            ],
        )

        decision = policy.evaluate(plan)

        self.assertFalse(decision.allowed)
        self.assertIn("shell_outside_guardrails", decision.unknown_tools)

    def test_executor_dispatches_allowed_plan(self) -> None:
        executor = AssistantExecutor()
        state = AssistantState(session_id="executor-test")
        plan = AssistantPlan(
            intent="show_latest_summary",
            route={"intent": "show_latest_summary", "language": "en"},
            risk="low",
            steps=[PlannerStep(tool="load_run_bundle"), PlannerStep(tool="render_response")],
        )

        def dispatch(current_state: AssistantState, route: AssistantRoute):  # type: ignore[no-untyped-def]
            self.assertEqual(route.intent, "show_latest_summary")
            current_state.last_intent = route.intent
            return "ok", current_state

        answer, updated = executor.execute(plan, state, dispatch)

        self.assertEqual(answer, "ok")
        self.assertEqual(updated.last_intent, "show_latest_summary")

    def test_executor_fails_closed_when_policy_blocks_the_plan(self) -> None:
        executor = AssistantExecutor()
        state = AssistantState(session_id="executor-test", preferred_language="es")
        plan = AssistantPlan(
            intent="show_latest_summary",
            route={"intent": "show_latest_summary", "language": "es"},
            risk="low",
            steps=[PlannerStep(tool="unknown_tool")],
        )

        answer, updated = executor.execute(plan, state, lambda current_state, route: ("should not happen", current_state))

        self.assertIn("Ejecución bloqueada por política", answer)
        self.assertEqual(updated.last_intent, "blocked_by_policy")

    def test_policy_blocks_pipeline_execution_when_comp_disallows_run(self) -> None:
        policy = AssistantPolicyEngine()
        plan = AssistantPlan(
            intent="run_pipeline",
            route={"intent": "run_pipeline", "language": "en", "allow_run": False},
            risk="medium",
            steps=[
                PlannerStep(tool="build_pipeline_request"),
                PlannerStep(tool="execute_pipeline"),
                PlannerStep(tool="load_run_bundle"),
            ],
        )

        decision = policy.evaluate(plan)

        self.assertFalse(decision.allowed)
        self.assertIn("non-executable", decision.reason)


if __name__ == "__main__":
    unittest.main()
