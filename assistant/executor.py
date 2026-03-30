from __future__ import annotations

from typing import Callable, Tuple

from assistant.contracts import AssistantRoute, AssistantState
from assistant.policy import AssistantPolicyEngine
from schemas.planner import AssistantPlan


class AssistantExecutor:
    def __init__(self, policy: AssistantPolicyEngine | None = None) -> None:
        self.policy = policy or AssistantPolicyEngine()

    def execute(
        self,
        plan: AssistantPlan,
        state: AssistantState,
        dispatch: Callable[[AssistantState, AssistantRoute], Tuple[str, AssistantState]],
    ) -> tuple[str, AssistantState]:
        decision = self.policy.evaluate(plan)
        if not decision.allowed:
            language = str(plan.route.get("language") or state.preferred_language or "en").lower()
            es = language.startswith("es")
            unknown_tools = ", ".join(decision.unknown_tools) if decision.unknown_tools else "n/a"
            message = (
                f"Ejecución bloqueada por política. Motivo: {decision.reason} "
                f"Riesgo={plan.risk}; límite_de_pasos={decision.max_steps}; tools_desconocidas={unknown_tools}."
                if es
                else f"Execution blocked by policy. Reason: {decision.reason} "
                f"risk={plan.risk}; max_steps={decision.max_steps}; unknown_tools={unknown_tools}."
            )
            state.last_intent = "blocked_by_policy"
            state.last_route = plan.route
            state.pending_route = {}
            state.pending_task = ""
            return message, state

        route = AssistantRoute.from_dict(plan.route)
        return dispatch(state, route)

