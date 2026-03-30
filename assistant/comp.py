from __future__ import annotations

import re

from assistant.contracts import AssistantState, ConversationInterpretation
from assistant.router import (
    _detect_language,
    _extract_run_id,
    _looks_like_agent_guide_request,
    _looks_like_help_request,
    _looks_like_market_type_request,
    _looks_like_mode_guide_request,
    _looks_like_semantic_lookup_request,
    _looks_like_session_status_full_request,
    _looks_like_session_status_request,
    _looks_like_source_scope_request,
    _looks_like_symbol_guide_request,
    _looks_like_time_status_request,
    _looks_like_web_status_request,
    _normalize_whitespace,
    _normalize_yfinance_ticker,
    _ticker_candidates,
)


_GREETING_PATTERNS = (
    r"^(?:hola|hello|hi|hey|buenas|hola assistant|hello assistant|hey assistant)$",
    r"^(?:cómo estás|como estas|how are you)$",
)

_AGENT_GUIDE_HINTS = (
    "muestrame los agentes",
    "muéstrame los agentes",
    "agentes disponibles",
    "que agentes tiene",
    "qué agentes tiene",
    "que agentes tiene el asistente",
    "qué agentes tiene el asistente",
)

_MODE_GUIDE_HINTS = (
    "que modos tienes",
    "qué modos tienes",
    "que modos hay",
    "qué modos hay",
    "que modos puedes usar",
    "qué modos puedes usar",
)

_MODE_HUB_LABELS = {
    "mode",
    "mode guide",
    "modes",
    "modo",
    "modos",
    "guia de modos",
    "guía de modos",
}

_SYMBOL_GUIDE_HINTS = (
    "que activos puedo usar",
    "qué activos puedo usar",
    "que simbolos puedo usar",
    "qué simbolos puedo usar",
    "que símbolos puedo usar",
    "qué símbolos puedo usar",
)

_HELP_HINTS = (
    "que puedes hacer",
    "qué puedes hacer",
    "que más puedes hacer",
    "qué más puedes hacer",
    "que mas puedes hacer",
    "what can you do",
    "what else can you do",
)

_WEB_REQUIRED_HINTS = (
    "busca en internet",
    "buscar en internet",
    "con internet",
    "with internet",
    "use internet",
    "usa internet",
    "web",
)

_WEB_STATUS_HINTS = (
    "retriever",
    "retriever web",
    "estado del retriever",
)

_TRACE_HINTS = (
    "trace",
    "traza",
    "turn trace",
    "session trace",
    "show trace",
    "muestra la traza",
    "muestrame la traza",
    "muéstrame la traza",
    "como interpretaste",
    "cómo interpretaste",
    "why did you interpret that way",
)

_TRACE_FULL_HINTS = (
    "trace full",
    "full trace",
    "detailed trace",
    "trace detail",
    "traza completa",
    "traza detallada",
    "how did you interpret that",
    "explain your interpretation",
    "explica tu interpretación",
    "explica tu interpretacion",
)

_GENERIC_SEMANTIC_SUBJECT_PREFIXES = (
    "that",
    "this",
    "it",
    "that better",
    "this better",
    "it better",
    "eso",
    "esa",
    "ese",
    "eso mejor",
    "esa mejor",
    "ese mejor",
)

_DIRECT_SESSION_STATUS_LABELS = {
    "status",
    "estado",
    "session status",
    "estado de sesion",
    "estado de sesión",
    "mode status",
    "estado del modo",
}

_DIRECT_SESSION_STATUS_FULL_LABELS = {
    "status full",
    "estado completo",
    "full session status",
    "estado completo de la sesion",
    "estado completo de la sesión",
    "dame el status full de la sesion",
    "dame el status full de la sesión",
}


def _explicit_tickers(message: str) -> list[str]:
    return [_normalize_yfinance_ticker(item) for item in _ticker_candidates(message)]


def _semantic_subject_from_message(message: str) -> str:
    text = str(message or "").strip()
    patterns = (
        r"^(?:busca(?:\s+entonces)?\s+en\s+internet|search(?:\s+the\s+web)?|search\s+on\s+the\s+web|use internet|usa internet)\s+(?:what is|what's|qué es|que es)\s+(.+?)\??$",
        r"^(?:what does|qué significa|que significa)\s+(.+?)\s+mean\??$",
        r"^(?:what is|what's|qué es|que es)\s+(.+?)\??$",
        r"^(?:define|definition of|meaning of)\s+(.+?)\??$",
        r"^(?:tell me about|what can you tell me about|explain|describe)\s+(.+?)\??$",
        r"^(?:háblame de|hablame de|explica|describe)\s+(.+?)\??$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        subject = re.sub(r"[?.!,;:]+$", "", match.group(1)).strip()
        subject = re.sub(r"^(?:the|a|an|el|la|un|una)\s+", "", subject, flags=re.IGNORECASE)
        if subject:
            return subject
    return ""


def _semantic_comparison_subjects(message: str) -> list[str]:
    text = str(message or "").strip()
    pattern = (
        r"^(?:what is the difference between|what's the difference between|que diferencia hay entre|qué diferencia hay entre|difference between|diferencia entre|compare|compara)\s+(.+?)\s+(?:and|vs\.?|versus|y|con)\s+(.+?)(?:\?|$)"
    )
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return []
    first = re.sub(r"[?.!,;:]+$", "", match.group(1)).strip()
    second = re.sub(r"[?.!,;:]+$", "", match.group(2)).strip()
    first = re.sub(r"^(?:the|a|an|el|la|un|una)\s+", "", first, flags=re.IGNORECASE)
    second = re.sub(r"^(?:the|a|an|el|la|un|una)\s+", "", second, flags=re.IGNORECASE)
    if first and second:
        return [first, second]
    return []


def _is_generic_semantic_subject(subject: str) -> bool:
    lowered = _normalize_whitespace(subject).lower()
    if not lowered:
        return True
    return any(lowered.startswith(prefix) for prefix in _GENERIC_SEMANTIC_SUBJECT_PREFIXES)


def _comparison_stage_from_message(message: str) -> str:
    lowered = _normalize_whitespace(message).lower()
    if re.search(r"\b(cleaning|limpieza|clean)\b", lowered):
        return "cleaning"
    if re.search(r"\b(extraction|extracción|extraccion|extract)\b", lowered):
        return "extraction"
    if re.search(r"\b(modeling|modelado)\b", lowered):
        return "modeling"
    if re.search(r"\b(orchestrator|orquestador|orchestration|orquestación|orquestacion)\b", lowered):
        return "orchestrator"
    return "comparison"


class AssistantCompAgent:
    """Primary conversational interpreter.

    It does not execute anything. It classifies the conversational act,
    extracts the explicit subject for the current turn, and returns a compact
    contract the planner/runtime can enforce before routing and execution.
    """

    def interpret(self, message: str, state: AssistantState) -> ConversationInterpretation:
        normalized = _normalize_whitespace(message)
        lowered = normalized.lower()
        lowered_key = re.sub(r"[?.!,;:]+$", "", lowered).strip()
        language = state.preferred_language if state.preferred_language in {"en", "es"} else _detect_language(normalized)
        tickers = _explicit_tickers(normalized)
        run_id = _extract_run_id(normalized)
        web_required = any(hint in lowered for hint in _WEB_REQUIRED_HINTS)

        if any(re.search(pattern, lowered_key, flags=re.IGNORECASE) for pattern in _GREETING_PATTERNS):
            return ConversationInterpretation(
                act="greeting",
                route_intent="greet",
                source_policy="local_only",
                allow_run=False,
                override_memory=True,
                confidence=0.99,
                explanation="Greeting-style turn; answer socially instead of grounding on the latest run.",
            )

        if lowered in _WEB_STATUS_HINTS or _looks_like_web_status_request(normalized):
            return ConversationInterpretation(
                act="web_status",
                route_intent="show_web_status",
                stage="web_status",
                question_focus="web_status",
                source_policy="local_only",
                allow_run=False,
                override_memory=True,
                confidence=0.98,
                explanation="The turn asks about retriever/web status, so runtime config should answer directly.",
            )

        if any(hint in lowered for hint in _TRACE_HINTS):
            return ConversationInterpretation(
                act="trace",
                route_intent="show_turn_trace",
                stage="turn_trace_full" if any(hint in lowered for hint in _TRACE_FULL_HINTS) else "turn_trace",
                question_focus="turn_trace",
                source_policy="local_only",
                allow_run=False,
                override_memory=False,
                confidence=0.98,
                explanation="The user is asking to inspect how the last turn was interpreted.",
            )

        if any(hint in lowered for hint in _AGENT_GUIDE_HINTS) or _looks_like_agent_guide_request(normalized):
            return ConversationInterpretation(
                act="agent_guide",
                route_intent="show_agent_guide",
                stage="agent_guide",
                question_focus="agent_guide",
                source_policy="local_only",
                allow_run=False,
                override_memory=True,
                confidence=0.97,
                explanation="The user is asking about available agents and roles.",
            )

        if lowered_key in _MODE_HUB_LABELS or any(hint in lowered for hint in _MODE_GUIDE_HINTS):
            return ConversationInterpretation(
                act="mode_guide",
                route_intent="show_mode_guide",
                stage="mode_guide",
                question_focus="mode_guide",
                source_policy="local_only",
                allow_run=False,
                override_memory=True,
                confidence=0.97,
                explanation="The user is asking about runtime modes, not the latest run summary.",
            )

        if any(hint in lowered for hint in _SYMBOL_GUIDE_HINTS) or _looks_like_symbol_guide_request(normalized):
            return ConversationInterpretation(
                act="symbol_guide",
                route_intent="show_symbol_guide",
                stage="symbols",
                question_focus="symbols",
                source_policy="local_only",
                allow_run=False,
                override_memory=True,
                confidence=0.97,
                explanation="The user is asking which assets/symbols can be used.",
            )

        if any(hint in lowered for hint in _HELP_HINTS) or _looks_like_help_request(normalized):
            return ConversationInterpretation(
                act="help",
                route_intent="help",
                stage="help",
                question_focus="general",
                source_policy="local_only",
                allow_run=False,
                override_memory=False,
                confidence=0.97,
                explanation="Capability/help questions should stay in the help layer and not collapse into run summaries.",
            )

        if lowered_key in _DIRECT_SESSION_STATUS_FULL_LABELS or _looks_like_session_status_full_request(normalized):
            return ConversationInterpretation(
                act="session_status_full",
                route_intent="show_session_status_full",
                stage="session_status_full",
                question_focus="session_status_full",
                run_id=run_id,
                source_policy="local_only",
                allow_run=False,
                override_memory=False,
                confidence=0.98,
                explanation="The turn explicitly asks for the full session status.",
            )

        if lowered_key in _DIRECT_SESSION_STATUS_LABELS:
            return ConversationInterpretation(
                act="session_status",
                route_intent="show_session_status",
                stage="session_status",
                question_focus="session_status",
                run_id=run_id,
                source_policy="local_only",
                allow_run=False,
                override_memory=False,
                confidence=0.98,
                explanation="The turn explicitly asks for session status.",
            )

        if len(re.findall(r"\brun_\d+\b", lowered)) >= 2 and re.search(r"\b(compare|comparison|comparar|comparación|comparacion|vs|versus)\b", lowered):
            return ConversationInterpretation(
                act="run_comparison",
                route_intent="show_run_comparison",
                stage=_comparison_stage_from_message(normalized),
                question_focus="run_comparison",
                run_id=run_id,
                source_policy="",
                web_required=web_required,
                allow_run=False,
                override_memory=False,
                confidence=0.98,
                explanation="The turn explicitly compares two runs and should stay in run-comparison mode.",
            )

        comparison_subjects = _semantic_comparison_subjects(normalized)
        if comparison_subjects:
            left, right = comparison_subjects[0], comparison_subjects[1]
            return ConversationInterpretation(
                act="definition",
                subject=f"{left} vs {right}",
                canonical_query=f"what is the difference between {left} and {right}?",
                route_intent="show_semantic_lookup",
                stage="semantic_lookup",
                question_focus="semantic_lookup",
                run_id=run_id,
                source_policy="web_required" if web_required else "local_then_web",
                web_required=web_required,
                allow_run=False,
                override_memory=True,
                confidence=0.98,
                explanation="The turn asks for a conceptual comparison, so the canonical query should preserve both subjects.",
            )

        if _looks_like_time_status_request(normalized):
            return ConversationInterpretation(
                act="time_status",
                route_intent="show_time_status",
                stage="time_status",
                question_focus="time_status",
                source_policy="local_only",
                allow_run=False,
                override_memory=True,
                confidence=0.99,
                explanation="Time/date questions should use local tools and avoid run context.",
            )

        if _looks_like_source_scope_request(normalized):
            return ConversationInterpretation(
                act="source_scope",
                route_intent="show_source_scope",
                stage="source_scope",
                question_focus="source_scope",
                source_policy="local_only",
                allow_run=False,
                override_memory=True,
                confidence=0.98,
                explanation="Source-scope questions should route to the assistant source explanation.",
            )

        if _looks_like_market_type_request(normalized) and tickers:
            return ConversationInterpretation(
                act="classification",
                subject=tickers[0],
                route_intent="show_market_type",
                stage="market_type",
                question_focus="market_type",
                tickers=tickers[:1],
                run_id=run_id,
                source_policy="web_required" if web_required else "local_then_web",
                web_required=web_required,
                allow_run=False,
                override_memory=True,
                confidence=0.98,
                explanation="The turn asks for the market classification of the explicit symbol in this turn.",
            )

        if _looks_like_semantic_lookup_request(normalized):
            subject = _semantic_subject_from_message(normalized)
            if _is_generic_semantic_subject(subject):
                subject = ""
            if run_id and not tickers:
                return ConversationInterpretation(
                    act="general",
                    run_id=run_id,
                    tickers=tickers[:1],
                    source_policy="",
                    web_required=web_required,
                    allow_run=False,
                    override_memory=False,
                    confidence=0.0,
                    explanation="A run-scoped question should keep the explicit run context instead of collapsing into a generic semantic lookup.",
                )
            return ConversationInterpretation(
                act="definition",
                subject=subject or (tickers[0] if tickers else ""),
                canonical_query=f"what is {subject or tickers[0]}?" if (subject or tickers) else "",
                route_intent="show_semantic_lookup",
                stage="semantic_lookup",
                question_focus="semantic_lookup",
                tickers=tickers[:1],
                run_id=run_id,
                source_policy="web_required" if web_required else "local_then_web",
                web_required=web_required,
                allow_run=False,
                override_memory=bool(subject or tickers),
                confidence=0.97,
                explanation="The turn asks for a definition or conceptual explanation.",
            )

        subject = _semantic_subject_from_message(normalized)
        if _is_generic_semantic_subject(subject):
            subject = ""
        if subject and not tickers and not run_id:
            return ConversationInterpretation(
                act="definition",
                subject=subject,
                canonical_query=f"what is {subject}?",
                route_intent="show_semantic_lookup",
                stage="semantic_lookup",
                question_focus="semantic_lookup",
                source_policy="web_required" if web_required else "local_then_web",
                web_required=web_required,
                allow_run=False,
                override_memory=True,
                confidence=0.9,
                explanation="Open concept prompt; resolve semantically before any run-oriented routing.",
            )

        return ConversationInterpretation(
            act="general",
            run_id=run_id,
            tickers=tickers[:1],
            source_policy="",
            web_required=False,
            allow_run=True,
            override_memory=False,
            confidence=0.0,
            explanation=f"No strong conversational override was needed; language={language}.",
        )
