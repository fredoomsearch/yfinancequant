from __future__ import annotations

from dataclasses import dataclass, field

from schemas.planner import AssistantPlan


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""
    max_steps: int = 0
    unknown_tools: list[str] = field(default_factory=list)


class AssistantPolicyEngine:
    def __init__(self) -> None:
        self.allowed_tools = {
            "build_pipeline_request",
            "execute_pipeline",
            "load_context_bundle",
            "load_run_bundle",
            "load_run_comparison_bundle",
            "retrieve_web_facts",
            "merge_grounding_facts",
            "render_response",
            "fallback_router_resolution",
        }
        self.max_steps_by_risk = {
            "low": 3,
            "medium": 5,
            "high": 7,
        }

    def evaluate(self, plan: AssistantPlan) -> PolicyDecision:
        unknown_tools = [step.tool for step in plan.steps if step.tool not in self.allowed_tools]
        max_steps = self.max_steps_by_risk.get(plan.risk, 3)
        route = plan.route or {}
        allow_run = bool(route.get("allow_run", True))
        requests_execution = plan.intent in {"run_pipeline", "compare_sources"} or any(
            step.tool == "execute_pipeline" for step in plan.steps
        )

        if not route:
            return PolicyDecision(False, "Planner route is empty.", max_steps=max_steps, unknown_tools=unknown_tools)
        if not allow_run and requests_execution:
            return PolicyDecision(
                False,
                "The conversational interpreter marked this turn as non-executable, so pipeline execution is blocked.",
                max_steps=max_steps,
                unknown_tools=unknown_tools,
            )
        if unknown_tools:
            return PolicyDecision(
                False,
                "The plan requested tools that are not registered in the assistant policy layer.",
                max_steps=max_steps,
                unknown_tools=unknown_tools,
            )
        if len(plan.steps) > max_steps:
            return PolicyDecision(
                False,
                f"Plan exceeds the step budget for risk={plan.risk}.",
                max_steps=max_steps,
                unknown_tools=unknown_tools,
            )
        return PolicyDecision(True, "", max_steps=max_steps, unknown_tools=unknown_tools)
