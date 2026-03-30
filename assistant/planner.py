from __future__ import annotations

from assistant.contracts import AssistantRoute, AssistantState, ConversationInterpretation
from assistant.context import AssistantContextResolver
from assistant.source_selector import AssistantSourceSelector
from assistant.router import AssistantRouter
from assistant.tools import AssistantToolLayer
from schemas.planner import AssistantPlan


class AssistantPlanner:
    """Planning layer for the assistant control plane.

    This is the first phase of the architecture shift from a pure router to a
    planner/executor shape. The planner produces a canonical plan even when the
    runtime still delegates execution to the existing handlers.
    """

    def __init__(
        self,
        router: AssistantRouter,
        tool_layer: AssistantToolLayer | None = None,
        context_resolver: AssistantContextResolver | None = None,
        source_selector: AssistantSourceSelector | None = None,
    ) -> None:
        self.router = router
        self.tool_layer = tool_layer or AssistantToolLayer()
        self.context_resolver = context_resolver or AssistantContextResolver()
        self.source_selector = source_selector or AssistantSourceSelector()

    def _apply_interpretation(
        self,
        route: AssistantRoute,
        interpretation: ConversationInterpretation | None,
    ) -> AssistantRoute:
        if interpretation is None:
            return route
        refined = AssistantRoute.from_dict(route.to_dict())
        if interpretation.route_intent:
            refined.intent = interpretation.route_intent
        if interpretation.stage:
            refined.stage = interpretation.stage
        if interpretation.question_focus:
            refined.question_focus = interpretation.question_focus
        if interpretation.tickers:
            refined.tickers = list(interpretation.tickers)
        if interpretation.run_id:
            refined.run_id = interpretation.run_id
        elif interpretation.override_memory and refined.intent not in {"run_pipeline", "compare_sources"}:
            refined.run_id = None
            refined.secondary_run_id = None
        if interpretation.canonical_query:
            refined.interpreted_query = interpretation.canonical_query
        if interpretation.source_policy:
            refined.source_policy = interpretation.source_policy
        refined.web_required = interpretation.web_required
        refined.allow_run = interpretation.allow_run
        refined.override_memory = interpretation.override_memory
        refined.conversation_act = interpretation.act or refined.conversation_act
        if interpretation.route_intent and refined.answer_mode == "strict" and refined.intent != "run_pipeline":
            refined.answer_mode = "interpreted"
        return refined

    def plan(
        self,
        message: str,
        state: AssistantState,
        interpretation: ConversationInterpretation | None = None,
    ) -> AssistantPlan:
        route = self.router.route(message, state)
        route = self._apply_interpretation(route, interpretation)
        context = self.context_resolver.resolve(message, state, route)
        route = self.context_resolver.refine_route(route, context)
        route = self._apply_interpretation(route, interpretation)
        source_selection = self.source_selector.select(message, route, state)
        route.source_mode = source_selection.mode
        route.web_queries = list(source_selection.queries)
        context_dict = context.to_dict()
        context_dict["source_mode"] = source_selection.mode
        context_dict["web_queries"] = list(source_selection.queries)
        if interpretation is not None:
            context_dict["conversation_act"] = interpretation.act
            context_dict["conversation_subject"] = interpretation.subject
            context_dict["source_policy"] = interpretation.source_policy
            context_dict["web_required"] = interpretation.web_required
            context_dict["allow_run"] = interpretation.allow_run
            context_dict["override_memory"] = interpretation.override_memory
        steps = self.tool_layer.build_steps(route, context_dict)
        risk = "low"
        if route.intent in {"run_pipeline", "compare_sources"}:
            risk = "medium"
        elif route.source_mode in {"web", "mixed"}:
            risk = "medium"
        if route.intent in {"run_pipeline", "compare_sources"} and (route.experimental_groq_brain or route.compare_binance):
            risk = "high"
        explanation = (
            "The planner resolved the user request into a canonical route, attached compact run context, "
            f"selected answer_mode={route.answer_mode} with certainty={route.certainty}, and chose source_mode={route.source_mode}."
        )
        return AssistantPlan(
            intent=route.intent,
            route=route.to_dict(),
            steps=steps,
            grounded=True,
            risk=risk,
            explanation=explanation,
            answer_mode=route.answer_mode,  # type: ignore[arg-type]
            certainty=route.certainty,  # type: ignore[arg-type]
            context_snapshot=context_dict,
            source_mode=route.source_mode,  # type: ignore[arg-type]
        )
