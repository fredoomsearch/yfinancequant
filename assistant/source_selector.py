from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List

from assistant.contracts import AssistantRoute, AssistantState

_WEB_HINTS = (
    "internet",
    "web",
    "search",
    "busca",
    "buscar",
    "google",
    "news",
    "noticias",
    "today",
    "hoy",
    "latest",
    "actual",
    "current",
)

_SEMANTIC_LOOKUP_EXCLUSIONS = (
    "volume",
    "volumen",
    "volatility",
    "volatilidad",
    "open",
    "high",
    "low",
    "close",
    "precio",
    "price",
    "metric",
    "metrics",
    "métrica",
    "métricas",
    "metrica",
    "metricas",
    "row",
    "rows",
    "fila",
    "filas",
    "prediction",
    "predicción",
    "prediccion",
    "decision",
    "decisión",
    "run_",
    "summary",
    "status",
    "result",
    "report",
    "latest",
    "current",
    "today",
    "now",
    "schema",
    "clean",
    "cleaned",
    "cleaning",
    "data",
)


def _looks_like_semantic_lookup(normalized: str) -> bool:
    text = str(normalized or "").strip().lower()
    if re.search(
        r"\b(what is|what's|qué es|que es)\s+(?:(?:the|a|an|el|la|un|una)\s+)?(volatility|volatilidad|risk|riesgo|return|retorno|rendimiento|alpha|beta|correlation|correlación|correlacion|leverage|apalancamiento|arbitrage|hedge|spread|liquidity|drawdown|momentum|mean reversion|slippage|hold|long|short|ticker|benchmark|portfolio|position|exposure|factor|signal|trend|variance|standard deviation|sharpe ratio|sortino ratio|covariance|pnl|profit and loss|excess return|benchmark return|tracking error|information ratio|dataset|artifact|manifest|target|feature engineering|grounding|memory|artifact store|evidence ledger|drift|conversational layer|validator|shadow run|promotion gate|challenger|champion|policy engine|retraining scheduler|feature registry|adaptive selector|shadow runner|promotion policy|row|fila|schema|esquema|raw column|clean column|model variable|column|columns|columna|columnas|variable|variables|feature|features|adj_close|adjusted close)\b(?!\s+(?:of|for|de|para|in|en)\b)",
        text,
    ):
        return True
    if any(token in text for token in _SEMANTIC_LOOKUP_EXCLUSIONS):
        return False
    if "binance" in text:
        return False
    if _semantic_comparison_subjects(text):
        return True
    if re.search(
        r"\b(what does .+ mean|qué significa .+|que significa .+|define|definition|definición|definicion|meaning|significado)\b",
        text,
    ):
        return True
    if re.fullmatch(
        r"(?:(?:the|a|an|el|la|un|una)\s+)?(?:volatility|volatilidad|risk|riesgo|return|retorno|rendimiento|alpha|beta|correlation|correlación|correlacion|leverage|apalancamiento|arbitrage|hedge|spread|liquidity|drawdown|momentum|mean reversion|slippage|hold|long|short|ticker|yfinance|benchmark|portfolio|position|exposure|factor|signal|trend|variance|standard deviation|sharpe ratio|sortino ratio|covariance|pnl|profit and loss|excess return|benchmark return|tracking error|information ratio|dataset|artifact|manifest|target|feature engineering|grounding|memory|artifact store|evidence ledger|drift|conversational layer|validator|shadow run|promotion gate|challenger|champion|policy engine|retraining scheduler|feature registry|adaptive selector|shadow runner|promotion policy|forex|crypto|equity|fund|index|stock|etf|currency pair|digital asset pair|market index|column|columns|columna|columnas|variable|variables|feature|features|adj_close|adjusted close)",
        text,
    ):
        return True
    if not re.search(r"\b(what is|what's|qué es|que es)\b", text):
        return False
    return True


def _semantic_subject(normalized: str) -> str:
    text = str(normalized or "").strip()
    patterns = (
        r"^(?:busca(?:\s+entonces)?\s+en\s+internet|search(?:\s+the\s+web)?|search\s+on\s+the\s+web|use internet|usa internet)\s+(?:what is|what's|qué es|que es)\s+(.+?)\??$",
        r"^(?:what does|qué significa|que significa)\s+(.+?)\s+mean\??$",
        r"^(?:what is|what's|qué es|que es)\s+(.+?)\??$",
        r"^(?:define|definition of|meaning of)\s+(.+?)\??$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            subject = re.sub(r"[?.!,;:]+$", "", match.group(1)).strip()
            subject = re.sub(r"^(?:the|a|an|el|la|un|una)\s+", "", subject, flags=re.IGNORECASE)
            if subject:
                return subject
    return ""


def _semantic_subject_variants(normalized: str) -> List[str]:
    subject = _semantic_subject(normalized)
    if not subject:
        return []
    variants: List[str] = []
    simplified = re.split(
        r"\b(?:in|en|for|para|on|sobre|with|con|about|de|del)\b",
        subject,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" -,:")
    if simplified and simplified.lower() != subject.lower():
        variants.append(simplified)
    variants.append(subject)
    seen: set[str] = set()
    ordered: List[str] = []
    for item in variants:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            ordered.append(item)
    return ordered


def _looks_like_semantic_follow_up(normalized: str) -> bool:
    text = str(normalized or "").strip().lower()
    if not text:
        return False
    return any(
        hint in text
        for hint in (
            "complement",
            "complementa",
            "expand",
            "expande",
            "expandelo",
            "expándelo",
            "use internet",
            "usa internet",
            "using internet",
            "con internet",
            "with internet",
            "external context",
            "contexto externo",
            "search that",
            "busca eso",
            "busca en internet",
        )
    )


def _semantic_comparison_subjects(normalized: str) -> List[str]:
    text = str(normalized or "").strip()
    patterns = (
        r"^(?:what is the difference between|what's the difference between|que diferencia hay entre|difference between|diferencia entre|compare|compara)\s+(.+?)\s+(?:and|vs\.?|versus|y|con)\s+(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            first = re.sub(r"[?.!,;:]+$", "", match.group(1)).strip()
            second = re.sub(r"[?.!,;:]+$", "", match.group(2)).strip()
            first = re.sub(r"^(?:the|a|an|el|la|un|una)\s+", "", first, flags=re.IGNORECASE)
            second = re.sub(r"^(?:the|a|an|el|la|un|una)\s+", "", second, flags=re.IGNORECASE)
            if first and second:
                return [first, second]
    return []


@dataclass
class SourceSelection:
    mode: str = "local"
    reason: str = ""
    queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AssistantSourceSelector:
    """Decides whether a request should use local, web, or mixed grounding."""

    def select(self, message: str, route: AssistantRoute, state: AssistantState) -> SourceSelection:
        normalized = str(message or "").strip().lower()
        focus = str(getattr(route, "question_focus", "general") or "general")
        run_id = route.run_id or state.last_run_id
        ticker = route.tickers[0] if route.tickers else (state.current_asset or "")
        semantic_query_basis = str(getattr(route, "interpreted_query", "") or "").strip().lower() or normalized
        if focus == "semantic_lookup" and not _semantic_subject(normalized) and not _semantic_comparison_subjects(normalized):
            last_raw = str((state.last_route or {}).get("raw_message") or "").strip().lower()
            if str(state.last_intent or "").strip() == "show_semantic_lookup" and last_raw and _looks_like_semantic_follow_up(normalized):
                semantic_query_basis = last_raw
        local_capable_intents = {
            "show_semantic_lookup",
            "show_source_scope",
            "show_time_status",
            "show_latest_summary",
            "show_session_status",
            "show_session_status_full",
            "show_asset_used",
            "show_market_metrics",
            "show_market_type",
            "show_model_variables",
            "show_decision_explanation",
            "show_clean_data",
            "show_run_comparison",
        }
        semantic_lookup = _looks_like_semantic_lookup(normalized)

        if bool(getattr(route, "web_required", False)):
            mode = "mixed" if (run_id or ticker) and focus not in {"time_status", "source_scope", "web_status"} else "web"
            return SourceSelection(
                mode=mode,
                reason="The conversational interpreter explicitly required web context for this turn.",
                queries=self._queries_for_focus(focus, ticker, semantic_query_basis if focus == "semantic_lookup" else normalized, semantic_lookup=focus == "semantic_lookup"),
            )

        if str(getattr(route, "source_policy", "") or "").strip() == "local_then_web" and focus in {"semantic_lookup", "market_type"}:
            return SourceSelection(
                mode="mixed" if run_id or ticker else "web",
                reason="The conversational interpreter resolved an explicit subject and requested local-first confirmation with optional web support.",
                queries=self._queries_for_focus(focus, ticker, semantic_query_basis if focus == "semantic_lookup" else normalized, semantic_lookup=focus == "semantic_lookup"),
            )

        if any(hint in normalized for hint in _WEB_HINTS):
            return SourceSelection(
                mode="mixed" if run_id or ticker or route.intent in local_capable_intents else "web",
                reason="The request explicitly asked for web context or current external information.",
                queries=self._queries_for_focus(focus, ticker, semantic_query_basis if focus == "semantic_lookup" else normalized),
            )

        if semantic_lookup:
            return SourceSelection(
                mode="mixed" if run_id or ticker else "web",
                reason="Semantic meaning/look-up questions benefit from external context before final synthesis.",
                queries=self._queries_for_focus(focus, ticker, semantic_query_basis, semantic_lookup=True),
            )

        if focus == "semantic_lookup":
            return SourceSelection(
                mode="mixed" if run_id or ticker else "web",
                reason="Semantic lookups are web-augmented and can optionally keep a local run or ticker anchor.",
                queries=self._queries_for_focus(focus, ticker, semantic_query_basis, semantic_lookup=True),
            )

        if focus == "market_type":
            return SourceSelection(
                mode="mixed" if run_id or ticker else "web",
                reason="Market-type questions benefit from local symbol facts plus optional external confirmation.",
                queries=self._queries_for_focus(focus, ticker, normalized),
            )

        if focus in {"time_status", "source_scope"}:
            return SourceSelection(mode="local", reason="Local tools and runtime configuration answer this request.", queries=[])

        if focus in {"run_comparison", "model_variables", "decision_explanation", "clean_data", "prediction", "cleaning", "extraction"}:
            if focus in {"model_variables", "decision_explanation", "clean_data", "run_comparison"} and getattr(route, "answer_mode", "strict") == "exploratory" and (run_id or ticker or route.secondary_run_id):
                return SourceSelection(
                    mode="mixed",
                    reason="Exploratory explanatory reads can use external context while keeping local run facts primary.",
                    queries=self._queries_for_focus(focus, ticker, normalized),
                )
            return SourceSelection(
                mode="local",
                reason="The question targets a run-specific artifact that should stay local-first.",
                queries=[],
            )

        if getattr(route, "answer_mode", "strict") == "exploratory" and ticker:
            return SourceSelection(
                mode="mixed",
                reason="Exploratory mode can benefit from external context while preserving local facts.",
                queries=self._queries_for_focus(focus, ticker, normalized),
            )

        return SourceSelection(mode="local", reason="Local artifacts provide the primary source of truth.", queries=[])

    def _queries_for_focus(self, focus: str, ticker: str, normalized: str, semantic_lookup: bool = False) -> List[str]:
        queries: List[str] = []
        symbol = str(ticker or "").strip().upper()
        if semantic_lookup and normalized:
            comparison_subjects = _semantic_comparison_subjects(normalized)
            if comparison_subjects:
                left, right = comparison_subjects
                queries.extend(
                    [
                        f"{left} meaning",
                        f"{right} meaning",
                        f"{left} vs {right}",
                        f"{left} versus {right}",
                        f"difference between {left} and {right}",
                    ]
                )
            else:
                subjects = _semantic_subject_variants(normalized)
                if subjects:
                    queries.append(f"{subjects[0]} meaning")
                    queries.append(f"{subjects[0]} definition")
                    if len(subjects) > 1:
                        queries.append(f"{subjects[1]} meaning")
            queries.append(normalized)
            if symbol:
                queries.append(f"{symbol} meaning market context")
            if comparison_subjects:
                return queries[:5]
        elif focus == "semantic_lookup" and normalized:
            comparison_subjects = _semantic_comparison_subjects(normalized)
            if comparison_subjects:
                left, right = comparison_subjects
                queries.extend(
                    [
                        f"{left} meaning",
                        f"{right} meaning",
                        f"{left} vs {right}",
                        f"{left} versus {right}",
                        f"difference between {left} and {right}",
                    ]
                )
            else:
                subjects = _semantic_subject_variants(normalized)
                if subjects:
                    queries.append(f"{subjects[0]} meaning")
                    queries.append(f"{subjects[0]} definition")
                    if len(subjects) > 1:
                        queries.append(f"{subjects[1]} meaning")
            queries.append(normalized)
            if symbol:
                queries.append(f"{symbol} market meaning context")
            if comparison_subjects:
                return queries[:5]
        elif focus == "market_type" and symbol:
            queries.append(f"{symbol} asset class market type")
            queries.append(f"{symbol} yfinance symbol type")
        elif focus == "decision_explanation" and symbol:
            queries.append(f"{symbol} latest market context trend volatility")
            queries.append(f"{symbol} current market sentiment price action")
        elif focus == "model_variables" and symbol:
            queries.append(f"{symbol} technical indicators moving averages volatility context")
            queries.append(f"{symbol} price action feature context")
        elif focus == "clean_data" and symbol:
            queries.append(f"{symbol} latest market context price action")
            queries.append(f"{symbol} current market backdrop volatility")
        elif focus == "run_comparison" and symbol:
            queries.append(f"{symbol} current market context versus prior run")
            queries.append(f"{symbol} latest market backdrop")
        elif focus == "market_metrics" and symbol:
            queries.append(f"{symbol} latest market data volume volatility")
            queries.append(f"{symbol} current price volume")
        elif focus == "general" and symbol:
            queries.append(f"{symbol} latest market context")
        elif symbol:
            queries.append(f"{symbol} market context")
        elif normalized:
            queries.append(normalized)
        return queries[:3]
