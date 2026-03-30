from __future__ import annotations

import ast
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from assistant.comp import AssistantCompAgent
from assistant.contracts import AssistantRoute, AssistantState, ConversationInterpretation
from assistant.context import AssistantContextResolver
from assistant.domain import (
    compare_semantic_definitions,
    resolve_assistant_identity,
    resolve_semantic_definition,
    resolve_semantic_definition_for_term,
)
from assistant.grounding import build_grounding_packet
from assistant.executor import AssistantExecutor
from assistant.planner import AssistantPlanner
from assistant.policy import AssistantPolicyEngine
from assistant.router import AssistantRouter
from assistant.state import load_state, save_state
from assistant.web import AssistantWebRetriever
from assistant.web import build_web_provider_env, build_web_provider_probe_command, describe_web_provider, list_web_provider_presets
from schemas.planner import AssistantPlan

_RUN_INTENTS = {"run_pipeline", "compare_sources"}
_SHOW_INTENTS = {
    "greet",
    "show_semantic_lookup",
    "show_source_scope",
    "show_time_status",
    "show_latest_summary",
    "show_extraction",
    "show_cleaning",
    "show_clean_data",
    "show_market_metrics",
    "show_prediction",
    "show_model_variables",
    "show_legacy_status",
    "show_symbol_guide",
    "show_asset_used",
    "show_market_type",
    "show_decision_explanation",
    "show_groq_status",
    "show_web_status",
    "show_assistant_scorecard",
    "show_session_status",
    "show_session_status_full",
    "show_turn_trace",
    "show_mode_guide",
    "show_agent_guide",
    "show_agent_card",
    "show_run_comparison",
}


def _is_spanish(language: Optional[str]) -> bool:
    return str(language or "en").lower().startswith("es")


def _resolve_language(state: AssistantState, route_language: Optional[str]) -> str:
    preferred = str(state.preferred_language or "").lower()
    if preferred in {"en", "es"}:
        return preferred
    return "es" if _is_spanish(route_language) else "en"


def _visible_answer_mode(mode: str) -> str:
    normalized = str(mode or "strict").lower()
    if normalized == "exploratory":
        return "exploratory"
    return "interpreted"


def _response_contract_line(route: AssistantRoute, language: str) -> str:
    es = _is_spanish(language)
    mode = _visible_answer_mode(str(getattr(route, "answer_mode", "strict") or "strict"))
    certainty = str(getattr(route, "certainty", "confirmed") or "confirmed").lower()
    source_mode = str(getattr(route, "source_mode", "local") or "local").lower()
    semantic_lookup = str(getattr(route, "question_focus", "") or "").strip() == "semantic_lookup"
    mode_labels = {
        "interpreted": ("interpretado", "interpreted"),
        "exploratory": ("exploratorio", "exploratory"),
    }
    certainty_labels = {
        "confirmed": ("confirmada desde artifacts del run", "confirmed from run artifacts"),
        "inferred": ("inferida desde artifacts del run", "inferred from run artifacts"),
        "hypothesis": ("hipótesis guiada por el contexto disponible", "hypothesis guided by the available context"),
        "confirmed_mixed": (
            "confirmada desde artifacts del run con apoyo de contexto externo",
            "confirmed from run artifacts with external context support",
        ),
        "inferred_mixed": (
            "inferida desde artifacts del run con contexto externo complementario",
            "inferred from run artifacts with complementary external context",
        ),
        "conflict": (
            "los artifacts locales tienen prioridad sobre contexto externo en conflicto",
            "local artifacts take priority over conflicting external context",
        ),
    }
    if semantic_lookup:
        certainty_labels = {
            "confirmed": ("confirmada desde contexto local del dominio", "confirmed from local domain context"),
            "inferred": ("inferida desde contexto local del dominio", "inferred from local domain context"),
            "hypothesis": (
                "hipótesis guiada por el contexto conceptual disponible",
                "hypothesis guided by the available conceptual context",
            ),
            "confirmed_mixed": (
                "confirmada con contexto local del dominio y apoyo externo",
                "confirmed with local domain context and external support",
            ),
            "inferred_mixed": (
                "inferida con contexto local del dominio y apoyo externo",
                "inferred with local domain context and external support",
            ),
            "conflict": (
                "el contexto local del dominio tiene prioridad sobre contexto externo en conflicto",
                "local domain context takes priority over conflicting external context",
            ),
        }
    grounding_labels = {
        "local": ("Base factual: local.", "Grounding: local."),
        "mixed": ("Base factual: mixta (local-first).", "Grounding: mixed (local-first)."),
        "web": ("Base factual: web.", "Grounding: web."),
    }
    mode_es, mode_en = mode_labels.get(mode, ("interpretado", "interpreted"))
    certainty_es, certainty_en = certainty_labels.get(certainty, certainty_labels["confirmed"])
    grounding_es, grounding_en = grounding_labels.get(source_mode, grounding_labels["local"])
    if es:
        return f"Modo de respuesta: {mode_es}. Certeza: {certainty_es}. {grounding_es}"
    return f"Response mode: {mode_en}. Certainty: {certainty_en}. {grounding_en}"


def _prepend_response_contract(answer: str, route: AssistantRoute, language: str) -> str:
    line = _response_contract_line(route, language)
    text = str(answer or "").strip()
    if not text:
        return line
    if text.startswith(line):
        return text
    return f"{line} {text}"


def _should_semantic_rewrite(route: AssistantRoute) -> bool:
    return str(getattr(route, "answer_mode", "strict") or "strict").lower() in {"interpreted", "exploratory"}


def _append_exploratory_tail(
    answer: str,
    route: AssistantRoute,
    summary: Dict[str, Any],
    result: Dict[str, Any],
    language: str,
) -> str:
    if str(getattr(route, "answer_mode", "strict") or "strict").lower() != "exploratory":
        return answer
    es = _is_spanish(language)
    focus = str(getattr(route, "question_focus", "general") or "general")
    models = summary.get("models") or {}
    data = summary.get("data") or {}
    rows = summary.get("rows") or {}
    confidence = float(models.get("confidence") or result.get("final_confidence") or 0.0)
    disagreement = models.get("disagreement")
    hypothesis = ""
    if focus == "decision_explanation":
        if disagreement:
            hypothesis = (
                "la señal parece frágil porque hubo desacuerdo entre modelos; conviene contrastarla con otra ventana o corrida."
                if es
                else "the signal looks fragile because the models disagreed; it is worth contrasting it with another window or run."
            )
        else:
            hypothesis = (
                f"la decisión parece relativamente estable porque la confianza local fue {confidence:.4f}, aunque esto no reemplaza validación out-of-sample."
                if es
                else f"the decision looks relatively stable because local confidence was {confidence:.4f}, although this does not replace out-of-sample validation."
            )
    elif focus == "model_variables":
        derived = data.get("derived_columns") or []
        hypothesis = (
            f"las señales derivadas como {', '.join(derived[:3]) or 'return_1d y medias móviles'} probablemente separan mejor el contexto que las columnas crudas solas, pero esto no es feature importance formal."
            if es
            else f"derived signals such as {', '.join(derived[:3]) or 'return_1d and moving averages'} probably separate context better than raw columns alone, but this is not formal feature importance."
        )
    elif focus == "clean_data":
        clean_rows = rows.get("cleaned")
        hypothesis = (
            f"esta lectura sirve mejor como contexto de mercado que como señal aislada; el dataset limpio aporta {clean_rows if clean_rows is not None else 'n/a'} filas para situar el patrón."
            if es
            else f"this read is more useful as market context than as a standalone signal; the cleaned dataset contributes {clean_rows if clean_rows is not None else 'n/a'} rows to situate the pattern."
        )
    elif focus == "run_comparison":
        hypothesis = (
            "la corrida con mayor confianza local y mejor preservación de filas probablemente ofrece la lectura más robusta, pero debe confirmarse contra validación consistente."
            if es
            else "the run with higher local confidence and better row preservation probably offers the more robust read, but it should be confirmed against consistent validation."
        )
    elif focus == "general":
        hypothesis = (
            "el estado resume la corrida actual, pero la parte más débil sigue siendo la decisión final si la confianza es baja o hubo desacuerdo entre modelos."
            if es
            else "the status summarizes the current run, but the weakest part is still the final decision when confidence is low or the models disagreed."
        )
    if not hypothesis:
        return answer
    label = "Hipótesis guiada" if es else "Guided hypothesis"
    return f"{answer} {label}: {hypothesis}"


def _format_run_comparison(
    left_summary: Dict[str, Any],
    left_result: Dict[str, Any],
    right_summary: Dict[str, Any],
    right_result: Dict[str, Any],
    stage: str | None = None,
    language: str = "en",
) -> str:
    es = _is_spanish(language)

    def _decision(summary: Dict[str, Any], result: Dict[str, Any]) -> str:
        return str((summary.get("models") or {}).get("final_decision") or result.get("final_decision") or "n/a")

    def _confidence(summary: Dict[str, Any], result: Dict[str, Any]) -> float:
        return float((summary.get("models") or {}).get("confidence") or result.get("final_confidence") or 0.0)

    def _ticker(summary: Dict[str, Any], result: Dict[str, Any]) -> str:
        tickers = summary.get("tickers") or ((result.get("manifest") or {}).get("request") or {}).get("tickers") or []
        return str(tickers[0]) if tickers else "n/a"

    def _rows(summary: Dict[str, Any]) -> tuple[Any, Any]:
        rows = summary.get("rows") or {}
        return rows.get("raw", "n/a"), rows.get("cleaned", "n/a")

    left_run = left_summary.get("run_id") or left_result.get("run_id") or "n/a"
    right_run = right_summary.get("run_id") or right_result.get("run_id") or "n/a"
    left_decision = _decision(left_summary, left_result)
    right_decision = _decision(right_summary, right_result)
    left_conf = _confidence(left_summary, left_result)
    right_conf = _confidence(right_summary, right_result)
    left_ticker = _ticker(left_summary, left_result)
    right_ticker = _ticker(right_summary, right_result)
    left_raw, left_clean = _rows(left_summary)
    right_raw, right_clean = _rows(right_summary)
    left_selected = str((left_result.get("modeling") or {}).get("selected_model") or (left_summary.get("motor") or {}).get("selected") or "n/a")
    right_selected = str((right_result.get("modeling") or {}).get("selected_model") or (right_summary.get("motor") or {}).get("selected") or "n/a")
    stronger = left_run if left_conf >= right_conf else right_run
    stage_key = str(stage or "").lower()
    if stage_key == "cleaning":
        if es:
            return (
                f"Comparación de limpieza entre {left_run} y {right_run}. "
                f"{left_run}: filas={left_raw}->{left_clean}. "
                f"{right_run}: filas={right_raw}->{right_clean}. "
                f"La corrida que preservó más filas limpias fue {left_run if (left_clean if isinstance(left_clean, (int, float)) else -1) >= (right_clean if isinstance(right_clean, (int, float)) else -1) else right_run}."
            )
        return (
            f"Cleaning comparison between {left_run} and {right_run}. "
            f"{left_run}: rows={left_raw}->{left_clean}. "
            f"{right_run}: rows={right_raw}->{right_clean}. "
            f"The run that preserved more cleaned rows was {left_run if (left_clean if isinstance(left_clean, (int, float)) else -1) >= (right_clean if isinstance(right_clean, (int, float)) else -1) else right_run}."
        )
    if stage_key == "modeling":
        if es:
            return (
                f"Comparación de modelado entre {left_run} y {right_run}. "
                f"{left_run}: decisión={left_decision}, confianza={left_conf:.4f}, motor={left_selected}. "
                f"{right_run}: decisión={right_decision}, confianza={right_conf:.4f}, motor={right_selected}. "
                f"La lectura más fuerte por confianza local es {stronger}."
            )
        return (
            f"Modeling comparison between {left_run} and {right_run}. "
            f"{left_run}: decision={left_decision}, confidence={left_conf:.4f}, motor={left_selected}. "
            f"{right_run}: decision={right_decision}, confidence={right_conf:.4f}, motor={right_selected}. "
            f"The stronger read by local confidence is {stronger}."
        )
    if stage_key == "extraction":
        if es:
            return (
                f"Comparación de extracción entre {left_run} y {right_run}. "
                f"{left_run}: ticker={left_ticker}, filas crudas={left_raw}. "
                f"{right_run}: ticker={right_ticker}, filas crudas={right_raw}. "
                f"La corrida con mayor cobertura cruda fue {left_run if (left_raw if isinstance(left_raw, (int, float)) else -1) >= (right_raw if isinstance(right_raw, (int, float)) else -1) else right_run}."
            )
        return (
            f"Extraction comparison between {left_run} and {right_run}. "
            f"{left_run}: ticker={left_ticker}, raw_rows={left_raw}. "
            f"{right_run}: ticker={right_ticker}, raw_rows={right_raw}. "
            f"The run with higher raw coverage was {left_run if (left_raw if isinstance(left_raw, (int, float)) else -1) >= (right_raw if isinstance(right_raw, (int, float)) else -1) else right_run}."
        )
    if es:
        return (
            f"Comparación entre {left_run} y {right_run}. "
            f"{left_run}: ticker={left_ticker}, decisión={left_decision}, confianza={left_conf:.4f}, filas={left_raw}->{left_clean}, motor={left_selected}. "
            f"{right_run}: ticker={right_ticker}, decisión={right_decision}, confianza={right_conf:.4f}, filas={right_raw}->{right_clean}, motor={right_selected}. "
            f"La lectura más fuerte por confianza local es {stronger}."
        )
    return (
        f"Comparison between {left_run} and {right_run}. "
        f"{left_run}: ticker={left_ticker}, decision={left_decision}, confidence={left_conf:.4f}, rows={left_raw}->{left_clean}, motor={left_selected}. "
        f"{right_run}: ticker={right_ticker}, decision={right_decision}, confidence={right_conf:.4f}, rows={right_raw}->{right_clean}, motor={right_selected}. "
        f"The stronger read by local confidence is {stronger}."
    )


def _route_to_request(route: AssistantRoute, artifact_root: str) -> Any:
    try:
        from schemas.pipeline import ModelFamily, PipelineRequest
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Pipeline request models are unavailable: {exc}. Activate the project venv or install dependencies."
        ) from exc

    tickers = [item.upper().strip() for item in route.tickers if str(item).strip()]
    if not tickers:
        tickers = ["AAPL"]
    if route.compare_binance and not route.comparison_yfinance_ticker and tickers:
        route.comparison_yfinance_ticker = tickers[0]
    if route.compare_binance and not route.comparison_asset and tickers:
        route.comparison_asset = tickers[0].split("-")[0].upper()
    if route.compare_binance and not route.comparison_binance_symbol and route.comparison_asset:
        route.comparison_binance_symbol = f"{route.comparison_asset.upper()}USDT"

    review_mode = route.review_mode if route.review_mode in {"auto", "off", "on"} else "auto"
    use_reviewer = bool(route.use_reviewer and review_mode != "off")
    if review_mode == "off":
        use_reviewer = False

    end = route.end or date.today()
    start = route.start or (end - timedelta(days=365))
    if start > end:
        start = end - timedelta(days=365)

    return PipelineRequest(
        tickers=tickers,
        start=start,
        end=end,
        interval=route.interval,
        model_choice=ModelFamily.auto,
        confidence_threshold=0.60,
        use_reviewer=use_reviewer,
        review_mode=review_mode,
        reviewer_provider="groq",
        artifact_root=artifact_root,
        compare_binance=bool(route.compare_binance),
        comparison_asset=route.comparison_asset,
        comparison_yfinance_ticker=route.comparison_yfinance_ticker,
        comparison_binance_symbol=route.comparison_binance_symbol,
        language=route.language if route.language in {"en", "es"} else "en",
        experimental_groq_brain=bool(getattr(route, "experimental_groq_brain", False)),
    )


def _format_concise_summary(summary: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    models = summary.get("models") or {}
    rows = summary.get("rows") or {}
    decision = models.get("final_decision", "n/a")
    confidence = models.get("confidence", 0.0)
    raw_rows = rows.get("raw")
    cleaned_rows = rows.get("cleaned")
    pieces = [
        f"Run {summary.get('run_id')} completed." if summary.get("run_id") else "Run completed.",
        (
            (
                f"En lenguaje simple, extraje {raw_rows} filas de yfinance y las limpié hasta {cleaned_rows} filas."
                if es
                else f"In plain language, I pulled {raw_rows} rows from yfinance and cleaned them into {cleaned_rows} rows."
            )
            if raw_rows is not None and cleaned_rows is not None
            else (
                "En lenguaje simple, extraje los datos de mercado solicitados y los preparé para el modelado."
                if es
                else "In plain language, I pulled the requested market data and prepared it for modeling."
            )
        ),
        f"{'Decisión' if es else 'Decision'}: {decision} ({'confianza' if es else 'confidence'} {confidence:.4f}).",
        f"{'Modo' if es else 'Mode'}: {summary.get('run_mode', 'local_only')}.",
        f"{'Filas' if es else 'Rows'}: raw={raw_rows} cleaned={cleaned_rows}.",
    ]
    return " ".join(pieces)


def _run_prefix(summary: Dict[str, Any]) -> str:
    run_id = summary.get("run_id")
    if not run_id:
        return ""
    return f"Run {run_id}. "


def _ensure_run_prefix(answer: str, summary: Dict[str, Any]) -> str:
    prefix = _run_prefix(summary).strip()
    text = str(answer or "").strip()
    if not prefix or not text or prefix in text:
        return text
    return f"{prefix} {text}"


def _clean_data_header_label(clean_mode: str, language: str) -> str:
    es = _is_spanish(language)
    if clean_mode == "hub":
        return "Centro de datos limpios" if es else "Clean data hub"
    if clean_mode == "analysis":
        return "Análisis de clean_market_data" if es else "Clean market data analysis"
    if clean_mode == "metrics":
        return "Métricas de clean_market_data" if es else "Clean market data metrics"
    return "Resumen de clean_market_data" if es else "Clean market data overview"


def _ensure_clean_data_header(answer: str, summary: Dict[str, Any], clean_mode: str, language: str) -> str:
    label = _clean_data_header_label(clean_mode, language)
    text = _ensure_run_prefix(answer, summary)
    if label in text:
        return text
    return f"{text} {label}.".strip()


def _ensure_model_variables_markers(
    answer: str,
    summary: Dict[str, Any],
    result: Dict[str, Any],
    language: str,
) -> str:
    es = _is_spanish(language)
    data = summary.get("data") or {}
    raw_columns = data.get("raw_columns") or []
    feature_columns = data.get("feature_columns") or []
    derived_columns = data.get("derived_columns") or [column for column in feature_columns if column not in raw_columns]
    base_model_columns = [column for column in feature_columns if column in raw_columns]
    base_label = "Variables base" if es else "Base variables"
    derived_label = "Variables derivadas" if es else "Derived variables"
    text = _ensure_run_prefix(answer, summary)
    if base_model_columns and base_label not in text:
        text = f"{text} {base_label}: {', '.join(base_model_columns)}."
    if derived_columns and derived_label not in text:
        text = f"{text} {derived_label}: {', '.join(derived_columns)}."
    return text


def _compact_reason_text(text: Any, *, limit: int = 120) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if not value:
        return "n/a"
    if "{" in value:
        value = value.split("{", 1)[0].strip()
    value = value.rstrip(" :-")
    if len(value) > limit:
        value = value[: limit - 1].rstrip() + "…"
    return value or "n/a"


def _format_sectioned_notice(
    title_en: str,
    title_es: str,
    sections: list[tuple[str, str, Any]],
    language: str = "en",
) -> str:
    es = _is_spanish(language)
    lines = [title_es if es else title_en]
    for en_label, es_label, value in sections:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        lines.append(f"{es_label if es else en_label}: {text}")
    return "\n".join(lines)


def _format_availability_notice(
    *,
    title_en: str,
    title_es: str,
    status_en: str,
    status_es: str,
    action_en: str,
    action_es: str,
    prompt_en: str,
    prompt_es: str,
    language: str = "en",
    detail_en: str | None = None,
    detail_es: str | None = None,
    extra_sections: Optional[list[tuple[str, str, Any]]] = None,
) -> str:
    es = _is_spanish(language)
    sections: list[tuple[str, str, Any]] = [
        ("Status", "Estado", status_es if es else status_en),
    ]
    if detail_en or detail_es:
        sections.append(("Detail", "Detalle", detail_es if es else detail_en))
    sections.extend(extra_sections or [])
    sections.extend(
        [
            ("Action", "Acción", action_es if es else action_en),
            ("Prompt", "Pregunta natural", prompt_es if es else prompt_en),
        ]
    )
    return _format_sectioned_notice(title_en, title_es, sections, language)


def _bundle_error(bundle: Dict[str, Any]) -> Dict[str, Any]:
    error = bundle.get("error") or (bundle.get("manifest") or {}).get("error") or {}
    return error if isinstance(error, dict) else {}


def _parse_failure_items(error: Dict[str, Any]) -> list[Any]:
    details = error.get("details") or {}
    failures = details.get("failures")
    if isinstance(failures, list) and failures:
        return failures
    message = str(error.get("message") or "")
    marker = "Failures:"
    if marker in message:
        tail = message.split(marker, 1)[1].strip()
        try:
            parsed = ast.literal_eval(tail)
        except Exception:
            return []
        if isinstance(parsed, list):
            return parsed
    return []


def _format_failure_list(failures: list[Any], language: str = "en") -> str:
    es = _is_spanish(language)
    compact: list[str] = []
    for item in failures[:4]:
        if isinstance(item, dict):
            ticker = str(item.get("ticker") or item.get("symbol") or "").strip().upper()
            reason = _compact_reason_text(item.get("reason") or item.get("error") or item.get("message"), limit=72)
            if ticker and reason != "n/a":
                compact.append(f"{ticker}: {reason}")
            elif ticker:
                compact.append(ticker)
            else:
                compact.append(reason)
        else:
            compact.append(_compact_reason_text(item, limit=72))
    if len(failures) > 4:
        compact.append(f"+{len(failures) - 4} {'más' if es else 'more'}")
    return "; ".join(item for item in compact if item) or ("sin detalles" if es else "no details")


def _format_run_failure(bundle: Dict[str, Any], language: str = "en", default_message: str = "") -> str:
    es = _is_spanish(language)
    manifest = bundle.get("manifest") or {}
    error = _bundle_error(bundle)
    details = error.get("details") or {}
    request = manifest.get("request") or {}
    message = str(error.get("message") or default_message or "").strip()
    stage = str(error.get("stage") or details.get("stage") or "").strip().lower()
    stage_aliases = {
        "extract": "extraction",
        "extraction_agent": "extraction",
        "clean": "cleaning",
        "cleaning_agent": "cleaning",
        "model": "modeling",
        "modeling_agent": "modeling",
        "comparison": "source_comparison",
        "source_comparison_failed": "source_comparison",
        "legacy": "legacy_bridge",
        "legacy_bridge_failed": "legacy_bridge",
    }
    stage = stage_aliases.get(stage, stage)
    if not stage:
        message_lower = message.lower()
        if "no market data could be extracted" in message_lower or "extraction failed" in message_lower or "no market data" in message_lower:
            stage = "extraction"

    stage_copy = {
        "extraction": {
            "title_en": "Extraction error",
            "title_es": "Error de extracción",
            "status_en": "No market data could be extracted.",
            "status_es": "No se pudo extraer datos de mercado.",
            "action_en": "Check the ticker, dates, and source connectivity, then retry.",
            "action_es": "Revisa el símbolo, las fechas y la conexión a la fuente; luego reintenta.",
            "prompt_en": "what symbol was used? | what columns were extracted?",
            "prompt_es": "qué símbolo se usó? | qué columnas se extrajeron?",
            "source_default": "yfinance",
        },
        "cleaning": {
            "title_en": "Cleaning error",
            "title_es": "Error de limpieza",
            "status_en": "The cleaning stage failed before the model could run.",
            "status_es": "La etapa de limpieza falló antes de que corriera el modelo.",
            "action_en": "Check the raw columns, ticker column, and date column, then rerun.",
            "action_es": "Revisa las columnas crudas, la columna ticker y la columna date; luego vuelve a correr.",
            "prompt_en": "what symbols are in the cleaned data? | what is the clean data schema?",
            "prompt_es": "qué símbolos hay en los datos limpios? | cuál es el esquema?",
        },
        "modeling": {
            "title_en": "Modeling error",
            "title_es": "Error de modelado",
            "status_en": "The modeling stage failed after cleaning.",
            "status_es": "La etapa de modelado falló después de limpiar los datos.",
            "action_en": "Check the cleaned rows, labels, and feature columns, then retry.",
            "action_es": "Revisa las filas limpias, las etiquetas y las columnas de modelo; luego reintenta.",
            "prompt_en": "what did the model predict? | why did it decide that way?",
            "prompt_es": "qué predijo el modelo? | por qué decidió eso?",
        },
        "source_comparison": {
            "title_en": "Source comparison error",
            "title_es": "Error de comparación de fuentes",
            "status_en": "The Binance comparison failed during the run.",
            "status_es": "La comparación con Binance falló durante la corrida.",
            "action_en": "Check the compare-binance inputs and source availability, then retry.",
            "action_es": "Revisa los parámetros de compare-binance y la disponibilidad de las fuentes; luego reintenta.",
            "prompt_en": "what changes when Binance is added? | which source wins?",
            "prompt_es": "qué cambia al añadir Binance? | qué fuente gana?",
        },
        "legacy_bridge": {
            "title_en": "Legacy bridge error",
            "title_es": "Error del puente legacy",
            "status_en": "The legacy bridge failed during the run.",
            "status_es": "El puente legacy falló durante la corrida.",
            "action_en": "Use a BTC, ETH, or LTC asset and verify the legacy mapping.",
            "action_es": "Usa un activo BTC, ETH o LTC y verifica el mapeo legacy.",
            "prompt_en": "is the legacy bridge active? | what asset uses the old mapping?",
            "prompt_es": "el puente legacy está activo? | qué activo usa el mapeo antiguo?",
        },
        "orchestrator": {
            "title_en": "Run error",
            "title_es": "Error de ejecución",
            "status_en": "The run failed before a complete summary could be built.",
            "status_es": "La corrida falló antes de construir un resumen completo.",
            "action_en": "Inspect the latest run, fix the issue, and try again.",
            "action_es": "Revisa la última corrida, corrige el problema y vuelve a intentarlo.",
            "prompt_en": "status | latest run | what happened?",
            "prompt_es": "estado | última corrida | qué pasó?",
        },
    }
    if stage not in stage_copy:
        stage = "orchestrator"
    copy = stage_copy[stage]
    tickers = request.get("tickers") or details.get("tickers") or []
    failures = _parse_failure_items(error)
    failure_text = _format_failure_list(failures, language) if failures else ""
    source = details.get("source") or copy.get("source_default") or "n/a"
    date_range = details.get("date_range") or {
        "start": request.get("start"),
        "end": request.get("end"),
    }
    sections: list[tuple[str, str, Any]] = [
        ("Status", "Estado", copy["status_es"] if es else copy["status_en"]),
        ("Stage", "Etapa", stage),
        ("Run", "Corrida", manifest.get("run_id") or details.get("run_id") or "n/a"),
        ("Tickers", "Símbolos", ", ".join(str(item) for item in tickers) if tickers else "n/a"),
    ]
    if source and (stage == "extraction" or details.get("source") or stage in {"source_comparison", "legacy_bridge"}):
        sections.append(("Source", "Fuente", source))
    if date_range.get("start") and date_range.get("end"):
        sections.append(
            (
                "Date range",
                "Rango de fechas",
                f"{date_range.get('start')} -> {date_range.get('end')}",
            )
        )
    rows_in = details.get("rows_in")
    rows_out = details.get("rows_out")
    raw_columns = details.get("raw_columns") or []
    feature_columns = details.get("feature_columns") or []
    target_column = details.get("target_column")
    if rows_in is not None:
        sections.append(("Rows in", "Filas de entrada", rows_in))
    if rows_out is not None:
        sections.append(("Rows out", "Filas de salida", rows_out))
    if raw_columns:
        sections.append(("Raw columns", "Columnas crudas", ", ".join(str(item) for item in raw_columns)))
    if feature_columns:
        sections.append(("Feature columns", "Columnas de modelo", ", ".join(str(item) for item in feature_columns)))
    if target_column:
        sections.append(("Target column", "Columna objetivo", target_column))
    sections.append(
        (
            "Cause",
            "Causa",
            _compact_reason_text(message or "n/a", limit=160),
        )
    )
    if failure_text:
        sections.append(("Failures", "Fallos", failure_text))
    sections.extend(
        [
            ("Action", "Acción", copy["action_es"] if es else copy["action_en"]),
            ("Prompt", "Pregunta natural", copy["prompt_es"] if es else copy["prompt_en"]),
        ]
    )
    return _format_sectioned_notice(copy["title_en"], copy["title_es"], sections, language)


def _format_latest_report(summary: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    if not summary:
        return _format_availability_notice(
            title_en="Session status",
            title_es="Estado de sesión",
            status_en="No completed runs are available yet.",
            status_es="No hay ejecuciones completadas todavía.",
            action_en="Start a run first.",
            action_es="Inicia una corrida primero.",
            prompt_en="what symbol was used? | what did the model predict?",
            prompt_es="qué símbolo se usó? | qué predijo el modelo?",
            language=language,
        )

    lines = [
        _format_concise_summary(summary, language),
    ]

    selection = summary.get("selection") or {}
    comparison = summary.get("comparison") or {}
    models = summary.get("models") or {}
    data = summary.get("data") or {}
    motor = summary.get("motor") or {}
    summary_parts: list[str] = []
    if selection:
        summary_parts.append(
            f"{'Selección' if es else 'Selection'}: {selection.get('strategy', 'n/a')} "
            f"{'porque' if es else 'because'} {_compact_reason_text(selection.get('reason', 'n/a'))}."
        )
    if comparison:
        summary_parts.append(
            f"{'Salud' if es else 'Health'}: extraction={comparison.get('extraction_health', 'n/a')} "
            f"modeling={comparison.get('modeling_health', 'n/a')}; "
            f"{'ruta' if es else 'path'}={comparison.get('decision_path', 'n/a')}."
        )
    if models.get("disagreement") is not None:
        reason = str(models.get("disagreement_reason") or "").strip()
        summary_parts.append(
            (
                f"{'Los modelos discreparon' if es else 'Models disagreed'}: {models.get('disagreement')}"
                + (f"; {reason}" if reason else "")
            )
        )
    if summary_parts:
        lines.append(" ".join(summary_parts))

    detail_parts: list[str] = []
    if data:
        detail_parts.append(
            f"{'Datos' if es else 'Data'}: raw columns={len(data.get('raw_columns') or [])}, "
            f"feature columns={len(data.get('feature_columns') or [])}, "
            f"derived columns={len(data.get('derived_columns') or [])}."
        )
    if motor:
        detail_parts.append(
            f"{'Motor' if es else 'Motor'}: requested={motor.get('requested', 'n/a')} selected={motor.get('selected', 'n/a')} "
            f"decision={motor.get('decision', 'n/a')}."
        )
    brain = summary.get("brain") or {}
    if brain:
        detail_parts.append(
            (
                f"{'Modo Groq brain' if es else 'Groq brain mode'}: "
                f"{'habilitado' if es else 'enabled'}={brain.get('enabled')} "
                f"{'usado' if es else 'used'}={brain.get('used')} "
                f"{'fuente_decisión' if es else 'decision_source'}={brain.get('decision_source')}."
            )
        )
        if brain.get("decision") is not None:
            detail_parts.append(
                (
                    f"{'Decisión de Groq brain' if es else 'Groq brain decision'}: "
                    f"{brain.get('decision')} ({float(brain.get('confidence', 0.0)):.4f}); "
                    f"{'determinista' if es else 'deterministic'}: {brain.get('deterministic_decision')} ({float(brain.get('deterministic_confidence', 0.0)):.4f})."
                )
            )
    if detail_parts:
        lines.append(" ".join(detail_parts))
    return "\n".join(lines)


def _format_run_recap(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    rows = summary.get("rows") or {}
    models = summary.get("models") or {}
    cleaning = result.get("cleaning") or {}
    data = summary.get("data") or {}
    motor = summary.get("motor") or result.get("motor") or {}
    extraction_rows = rows.get("raw")
    cleaned_rows = rows.get("cleaned")
    features = cleaning.get("feature_columns") or data.get("feature_columns") or []
    target = cleaning.get("target_column") or data.get("target_column") or "target_direction"
    decision = models.get("final_decision", "n/a")
    confidence = models.get("confidence", 0.0)
    selected = models.get("selected", "n/a")

    parts = []
    if extraction_rows is not None and cleaned_rows is not None:
        parts.append(
            (
                f"Versión corta: extraje {extraction_rows} filas de yfinance y las limpié hasta {cleaned_rows} filas."
                if es
                else f"Short version: I pulled {extraction_rows} rows from yfinance and cleaned them into {cleaned_rows} rows."
            )
        )
    else:
        parts.append(
            "Versión corta: extraje los datos de mercado solicitados y los limpié para el modelado."
            if es
            else "Short version: I pulled the requested market data and cleaned it for modeling."
        )
    if features:
        parts.append(
            (
                f"El conjunto limpio conservó {len(features)} variables y la columna objetivo {target}."
                if es
                else f"The cleaned set kept {len(features)} features and the target column {target}."
            )
        )
    parts.append(
        (
            f"Los modelos terminaron con una decisión {decision} y una confianza de {confidence:.4f}, usando {selected} como motor seleccionado."
            if es
            else f"The models finished with a {decision} decision at confidence {confidence:.4f}, using {selected} as the selected motor."
        )
    )
    if summary.get("review_mode") is not None:
        parts.append(f"{'Modo revisor' if es else 'Reviewer mode'}: {summary.get('review_mode')}.")
    if motor:
        parts.append(
            f"Motor: requested={motor.get('requested', 'n/a')} selected={motor.get('selected', 'n/a')} "
            f"decision={motor.get('decision', 'n/a')}."
        )
    return " ".join(parts)


def _summary_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    manifest = result.get("manifest") or {}
    request = manifest.get("request") or {}
    extraction = result.get("extraction") or {}
    cleaning = result.get("cleaning") or {}
    modeling = result.get("modeling") or {}
    source_comparison = result.get("source_comparison")
    legacy_analysis = result.get("legacy_analysis")
    motor = result.get("motor") or manifest.get("motor") or {}
    groq_brain = result.get("groq_brain") or {}

    raw_data = extraction.get("raw_data") or {}
    clean_data = cleaning.get("clean_data") or {}
    models = modeling.get("models") or []
    trained = [item.get("model_name") for item in models if item.get("model_name")]
    compare_binance = bool(request.get("compare_binance"))
    brain_enabled = bool(request.get("experimental_groq_brain"))
    reviewer_used = bool(manifest.get("reviewer_used", False))
    run_mode = "local_only"
    if brain_enabled and compare_binance and legacy_analysis and legacy_analysis.get("enabled", False):
        run_mode = "local_plus_binance_legacy_groq_brain"
    elif brain_enabled and compare_binance:
        run_mode = "local_plus_binance_groq_brain"
    elif brain_enabled:
        run_mode = "local_only_groq_brain"
    elif compare_binance and legacy_analysis and legacy_analysis.get("enabled", False):
        run_mode = "local_plus_binance_legacy"
    elif compare_binance:
        run_mode = "local_plus_binance"
    elif reviewer_used or result.get("reviewer"):
        run_mode = "local_plus_reviewer"

    return {
        "run_id": result.get("run_id") or manifest.get("run_id"),
        "status": result.get("status") or manifest.get("status"),
        "tickers": request.get("tickers") or extraction.get("tickers") or [],
        "date_range": {
            "start": request.get("start"),
            "end": request.get("end"),
            "interval": request.get("interval"),
        },
        "rows": {
            "raw": extraction.get("rows") or raw_data.get("rows"),
            "cleaned": cleaning.get("rows_out") or clean_data.get("rows_out"),
        },
        "models": {
            "trained": trained,
            "selected": modeling.get("selected_model"),
            "final_decision": result.get("final_decision") or manifest.get("decision"),
            "confidence": result.get("final_confidence") or manifest.get("confidence", 0.0),
            "deterministic_decision": result.get("deterministic_decision") or manifest.get("deterministic_decision") or modeling.get("ensemble_prediction"),
            "deterministic_confidence": result.get("deterministic_confidence") or manifest.get("deterministic_confidence") or modeling.get("ensemble_probability", 0.0),
            "disagreement": modeling.get("disagreement"),
            "disagreement_reason": modeling.get("rationale"),
            "reviewer_used": reviewer_used,
            "reviewer_provider": manifest.get("reviewer_provider"),
            "groq_brain_used": bool(manifest.get("groq_brain_used")) or bool(groq_brain),
            "groq_brain_provider": manifest.get("groq_brain_provider") or groq_brain.get("provider"),
            "decision_source": result.get("decision_source") or manifest.get("decision_source") or motor.get("decision_source"),
            "details": [
                {
                    "model": item.get("model_name"),
                    "prediction": item.get("latest_prediction"),
                    "probability": item.get("latest_probability"),
                    "confidence": item.get("confidence"),
                    "accuracy": (item.get("validation_metrics") or {}).get("accuracy", 0.0),
                    "roc_auc": (item.get("validation_metrics") or {}).get("roc_auc", 0.0),
                }
                for item in models
            ],
        },
        "selection": {
            "strategy": "legacy_orchestrated"
            if run_mode == "local_plus_binance_legacy"
            else ("compare_binance" if compare_binance else "local"),
            "reason": "Derived from result bundle.",
        },
        "comparison": {
            "extraction_health": "good" if raw_data else "needs_review",
            "modeling_health": "stable" if not modeling.get("disagreement") else "mixed",
            "decision_path": run_mode,
            "extraction_vs_models": "Derived from result bundle.",
        },
        "brain": {
            "enabled": brain_enabled,
            "used": bool(manifest.get("groq_brain_used")) or bool(groq_brain),
            "provider": manifest.get("groq_brain_provider") or groq_brain.get("provider"),
            "decision_source": result.get("decision_source") or manifest.get("decision_source") or motor.get("decision_source"),
            "deterministic_decision": result.get("deterministic_decision") or manifest.get("deterministic_decision") or modeling.get("ensemble_prediction"),
            "deterministic_confidence": result.get("deterministic_confidence") or manifest.get("deterministic_confidence") or modeling.get("ensemble_probability", 0.0),
            "decision": result.get("final_decision") or manifest.get("decision"),
            "confidence": result.get("final_confidence") or manifest.get("confidence", 0.0),
            "rationale": result.get("rationale") or manifest.get("rationale"),
            "risks": groq_brain.get("risks") or [],
            "explanation": groq_brain.get("explanation") or "",
        },
        "data": {
            "raw_columns": raw_data.get("columns") or [],
            "feature_columns": clean_data.get("feature_columns") or [],
            "derived_columns": [
                column
                for column in (clean_data.get("feature_columns") or [])
                if column not in (raw_data.get("columns") or [])
            ],
            "target_column": clean_data.get("target_column") or "target_direction",
        },
        "source_comparison": source_comparison,
        "legacy_analysis": legacy_analysis,
        "adaptive": result.get("adaptive"),
        "operations": result.get("operations") or manifest.get("operations"),
        "motor": motor,
        "review_mode": request.get("review_mode") or "auto",
        "run_mode": run_mode,
        "artifacts_count": len(manifest.get("artifacts") or []),
        "logs_count": len(manifest.get("logs") or []),
    }


def _format_extraction(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    extraction = result.get("extraction") or {}
    raw = extraction.get("raw_data") or {}
    cols = raw.get("columns") or summary.get("data", {}).get("raw_columns", [])
    missing = raw.get("missing_columns") or []
    tickers = ", ".join(result.get("extraction", {}).get("tickers", []) or summary.get("tickers", []) or ["n/a"])
    date_range = summary.get("date_range") or {}
    date_text = ""
    if date_range.get("start") and date_range.get("end"):
        date_text = (
            f" entre {date_range.get('start')} y {date_range.get('end')}"
            if es
            else f" between {date_range.get('start')} and {date_range.get('end')}"
        )
    return (
        f"{_run_prefix(summary)}"
        f"{'Extraje' if es else 'I pulled'} {extraction.get('rows', summary.get('rows', {}).get('raw', 0))} "
        f"{'filas de yfinance para' if es else 'rows from yfinance for'} {tickers}{date_text}. "
        f"{'Columnas faltantes' if es else 'Missing columns'}: {', '.join(missing) if missing else ('ninguna' if es else 'none')}. "
        f"{'Columnas crudas extraídas' if es else 'Raw columns extracted'}: {', '.join(cols)}."
    )


def _format_cleaning(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    cleaning = result.get("cleaning") or {}
    report = (cleaning.get("clean_data") or {}).get("quality_report") or {}
    features = cleaning.get("feature_columns") or summary.get("data", {}).get("feature_columns", [])
    return (
        f"{_run_prefix(summary)}"
        f"{'Limpié' if es else 'I cleaned'} {cleaning.get('rows_in', summary.get('rows', {}).get('raw', 0))} -> "
        f"{cleaning.get('rows_out', summary.get('rows', {}).get('cleaned', 0))} "
        f"{'filas' if es else 'rows'}. "
        f"{'Duplicados' if es else 'Duplicates'}: {report.get('dropped_duplicates', 0)}; "
        f"{'faltantes' if es else 'missing'}: {report.get('dropped_missing', 0)}. "
        f"{'Objetivo' if es else 'Target'}: "
        f"{cleaning.get('target_column', summary.get('data', {}).get('target_column', 'target_direction'))}. "
        f"{'Features' if es else 'Features'}: {', '.join(features)}."
    )


def _format_prediction(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    modeling = result.get("modeling") or {}
    models = modeling.get("models") or []
    motor = summary.get("motor") or result.get("motor") or {}
    details = []
    long_votes = sum(item.get("latest_prediction") == "long" for item in models)
    short_votes = sum(item.get("latest_prediction") == "short" for item in models)
    for item in models:
        details.append(
            f"{item.get('model_name')}: {item.get('latest_prediction')} "
            f"({item.get('latest_probability', 0.0):.4f}, confidence {item.get('confidence', 0.0):.4f})"
        )
    output = (
        f"{_run_prefix(summary)}"
        f"{'Los modelos predicen' if es else 'The models predict'} "
        f"{result.get('final_decision', summary.get('models', {}).get('final_decision'))} "
        f"({result.get('final_confidence', summary.get('models', {}).get('confidence', 0.0)):.4f}). "
        f"{'Votos' if es else 'Votes'}: {long_votes} long / {short_votes} short. "
        f"{'Modelos' if es else 'Models'}: {'; '.join(details)}."
    )
    if motor:
        output += (
            f" Motor: requested={motor.get('requested', 'n/a')} selected={motor.get('selected', 'n/a')} "
            f"decision={motor.get('decision', 'n/a')}."
        )
    output += (
        " Leyenda: long=subida, short=bajada, hold=señal débil."
        if es
        else " Legend: long=up, short=down, hold=weak signal."
    )
    return output


def _format_model_variables(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    data = summary.get("data") or {}
    rows = summary.get("rows") or {}
    modeling = result.get("modeling") or {}
    raw_columns = data.get("raw_columns") or []
    feature_columns = data.get("feature_columns") or []
    derived_columns = data.get("derived_columns") or [column for column in feature_columns if column not in raw_columns]
    base_model_columns = [column for column in feature_columns if column in raw_columns]
    target_column = data.get("target_column") or "target_direction"
    selected_model = modeling.get("selected_model") or (summary.get("models") or {}).get("selected") or "n/a"
    clean_rows = rows.get("cleaned")

    lines = [
        (
            f"{_run_prefix(summary)}En modelado trabajé sobre {clean_rows if clean_rows is not None else 'n/a'} filas limpias "
            f"y usé {len(feature_columns)} variables explicativas."
            if es
            else f"{_run_prefix(summary)}Modeling used {clean_rows if clean_rows is not None else 'n/a'} cleaned rows "
            f"and {len(feature_columns)} explanatory variables."
        )
    ]
    if base_model_columns:
        lines.append(
            f"{'Variables base' if es else 'Base variables'}: {', '.join(base_model_columns)}."
        )
    if derived_columns:
        lines.append(
            f"{'Variables derivadas' if es else 'Derived variables'}: {', '.join(derived_columns)}."
        )
    lines.append(
        (
            f"Objetivo: {target_column}. Se construye con future_close y future_return para marcar si el siguiente cierre sube (1) o baja (0)."
            if es
            else f"Target: {target_column}. It is built from future_close and future_return to mark whether the next close goes up (1) or down (0)."
        )
    )
    lines.append(
        (
            "Transformaciones aplicadas: tipado de fecha y numéricos, eliminación de filas sin ticker/date/close, "
            "deduplicación por ticker+date, creación de señales derivadas, y limpieza de inf/nan."
            if es
            else "Applied transformations: date and numeric coercion, dropping rows without ticker/date/close, "
            "deduplication by ticker+date, creation of derived signals, and inf/nan cleanup."
        )
    )
    lines.append(
        (
            "Antes de entrenar, ticker se codifica con one-hot; date, future_close, future_return y target_direction "
            "se excluyen de X; luego se aplica ffill, bfill y fillna(0)."
            if es
            else "Before training, ticker is one-hot encoded; date, future_close, future_return, and target_direction "
            "are excluded from X; then ffill, bfill, and fillna(0) are applied."
        )
    )
    lines.append(
        f"{'Motor seleccionado' if es else 'Selected motor'}: {selected_model}."
    )
    return " ".join(lines)


def _format_legacy(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    legacy = result.get("legacy_analysis") or summary.get("legacy_analysis") or {}
    if not legacy:
        return _format_availability_notice(
            title_en="Legacy bridge",
            title_es="Puente legacy",
            status_en="Legacy bridge is not enabled for this run.",
            status_es="El puente legacy no está habilitado en esta ejecución.",
            action_en="Use compare-binance for BTC, ETH, or LTC symbols that need the legacy mapping.",
            action_es="Usa compare-binance para BTC, ETH o LTC que necesiten el mapeo legacy.",
            prompt_en="is the legacy bridge active? | what asset uses the old mapping?",
            prompt_es="el puente legacy está activo? | qué activo usa el mapeo antiguo?",
            language=language,
        )
    if not legacy.get("enabled", False):
        return _format_availability_notice(
            title_en="Legacy bridge",
            title_es="Puente legacy",
            status_en="Legacy bridge is disabled for this run.",
            status_es="El puente legacy está deshabilitado en esta corrida.",
            detail_en=legacy.get("error") or legacy.get("bridge_note") or "no bridge context",
            detail_es=legacy.get("error") or legacy.get("bridge_note") or "sin contexto del puente",
            action_en="Use compare-binance with a BTC, ETH, or LTC symbol to enable the legacy path.",
            action_es="Usa compare-binance con un símbolo BTC, ETH o LTC para habilitar la ruta legacy.",
            prompt_en="why is legacy disabled? | how do I enable it?",
            prompt_es="por qué está deshabilitado legacy? | cómo lo activo?",
            language=language,
        )
    return (
        f"{_run_prefix(summary)}"
        f"{'Legacy habilitado para el activo' if es else 'Legacy enabled for asset'} {legacy.get('asset') or legacy.get('requested_asset')}. "
        f"{'Activo coincidente' if es else 'Matched asset'}: {legacy.get('matched_asset') or 'n/a'}. "
        f"{'Puntaje de confianza' if es else 'Trust score'}: {legacy.get('trust_score_pct', 0.0)}%. "
        f"{'Fuente del modelo' if es else 'Model source'}: {legacy.get('model_source', 'n/a')}. "
        f"{'Alineación de esquema' if es else 'Schema alignment'}: {legacy.get('schema_alignment_pct', 0.0)}%."
    )


def _format_comparison(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    comparison = summary.get("source_comparison") or result.get("source_comparison") or {}
    if not comparison:
        return _format_availability_notice(
            title_en="Source comparison",
            title_es="Comparación de fuentes",
            status_en="No Binance comparison is present in this run.",
            status_es="No hay comparación con Binance en esta ejecución.",
            action_en="Use compare-binance to compare yfinance with Binance.",
            action_es="Usa compare-binance para comparar yfinance con Binance.",
            prompt_en="what changes when Binance is added? | which source wins?",
            prompt_es="qué cambia al añadir Binance? | qué fuente gana?",
            language=language,
        )
    if not comparison.get("enabled", False):
        return _format_availability_notice(
            title_en="Source comparison",
            title_es="Comparación de fuentes",
            status_en="Binance comparison was not enabled.",
            status_es="La comparación con Binance no estaba habilitada.",
            detail_en=comparison.get("note") or comparison.get("error") or "no comparison context",
            detail_es=comparison.get("note") or comparison.get("error") or "sin contexto de comparación",
            action_en="Use compare-binance with the asset and date range you want to inspect.",
            action_es="Usa compare-binance con el activo y el rango de fechas que quieres revisar.",
            prompt_en="why is comparison disabled? | how do I turn it on?",
            prompt_es="por qué está deshabilitada la comparación? | cómo la activo?",
            language=language,
        )
    coverage = comparison.get("coverage") or {}
    close_alignment = comparison.get("close_price_alignment") or {}
    return (
        f"{_run_prefix(summary)}"
        f"{'Comparación de fuentes habilitada para' if es else 'Source comparison enabled for'} {comparison.get('asset') or 'the selected asset'} "
        f"({comparison.get('source_1')} vs {comparison.get('source_2')}). "
        f"{'Filas en común' if es else 'Overlap rows'}: {coverage.get('overlap_rows', 0)}. "
        f"{'Solo yfinance' if es else 'yfinance only'}: {coverage.get('yfinance_only_rows', 0)}. "
        f"{'Solo Binance' if es else 'binance only'}: {coverage.get('binance_only_rows', 0)}. "
        f"{'MAE de alineación del cierre' if es else 'Close alignment MAE'}: {close_alignment.get('mae')}. "
        f"{'MAPE' if es else 'MAPE'}: {close_alignment.get('mape_pct')}."
    )


def _format_stage_brief(summary: Dict[str, Any], result: Dict[str, Any], stage: str, language: str = "en") -> str:
    es = _is_spanish(language)
    stage_key = stage if stage in {"extraction", "cleaning", "modeling", "orchestrator"} else "orchestrator"
    stage_briefs = summary.get("stage_briefs") or result.get("stage_briefs") or {}
    brief = stage_briefs.get(stage_key) or {}
    if stage_key == "extraction":
        fallback = _format_extraction(summary, result, language)
    elif stage_key == "cleaning":
        fallback = _format_cleaning(summary, result, language)
    elif stage_key == "modeling":
        fallback = _format_prediction(summary, result, language)
    else:
        fallback = _format_decision_explanation(summary, result, language)
    if not brief:
        return f"{fallback} {_format_stage_handoff_hint(stage_key, language)}"
    pieces = [brief.get("summary") or fallback]
    if brief.get("motor"):
        pieces.append(f"{'Motor' if es else 'Motor'}: {brief.get('motor')}.")
    if brief.get("key_points"):
        pieces.append(f"{'Puntos clave' if es else 'Key points'}: {'; '.join(brief.get('key_points', []))}.")
    if brief.get("risks"):
        pieces.append(f"{'Riesgos' if es else 'Risks'}: {'; '.join(brief.get('risks', []))}.")
    if stage_key == "orchestrator":
        motor = summary.get("motor") or result.get("motor") or {}
        if motor:
            pieces.append(
                f"{'Motor de decisión' if es else 'Decision motor'}: requested={motor.get('requested', 'n/a')} "
                f"selected={motor.get('selected', 'n/a')} decision={motor.get('decision', 'n/a')}."
            )
    pieces.append(_format_stage_handoff_hint(stage_key, language))
    return f"{_run_prefix(summary)}" + " ".join(pieces)


def _format_motor(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    motor = summary.get("motor") or result.get("motor") or {}
    if not motor:
        return "No hay información de motor disponible para esta ejecución." if es else "No motor information is available for this run."
    pieces = [
        f"{'Motor solicitado' if es else 'Motor'}: requested={motor.get('requested', 'n/a')}.",
        f"{'Seleccionado' if es else 'Selected'}={motor.get('selected', 'n/a')}.",
        f"{'Decisión' if es else 'Decision'}={motor.get('decision', 'n/a')}.",
        f"{'Ruta de decisión' if es else 'Decision path'}={motor.get('decision_path', 'n/a')}.",
    ]
    if motor.get("compare_binance") is not None:
        pieces.append(f"compare_binance={motor.get('compare_binance')}.")
    if motor.get("legacy_enabled") is not None:
        pieces.append(f"legacy_enabled={motor.get('legacy_enabled')}.")
    if motor.get("reviewer_used") is not None:
        pieces.append(f"reviewer_used={motor.get('reviewer_used')}.")
    if motor.get("groq_brief_motor"):
        pieces.append(f"Groq motor label={motor.get('groq_brief_motor')}.")
    return " ".join(pieces)


def _format_symbol_guide(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    current = summary.get("tickers") or result.get("manifest", {}).get("request", {}).get("tickers") or []
    current_text = ", ".join(current) if current else ("ninguno" if es else "none")
    examples = [
        ("Stocks", "AAPL, MSFT, NVDA"),
        ("ETFs", "SPY, QQQ"),
        ("Indexes", "^GSPC, ^IXIC"),
        ("Forex", "EURUSD=X, USDJPY=X"),
        ("Crypto", "BTC-USD, ETH-USD"),
        ("Legacy crypto aliases", "BTC, ETH, LTC"),
    ]
    lines = [
        f"{'Símbolos sugeridos' if es else 'Suggested symbols'}:",
        f"- {'Activo actual' if es else 'Current asset'}: {current_text}.",
    ]
    for label, value in examples:
        lines.append(f"- {label if not es else {'Stocks':'Acciones','ETFs':'ETFs','Indexes':'Índices','Forex':'Forex','Crypto':'Cripto','Legacy crypto aliases':'Alias legacy cripto'}[label]}: {value}.")
    lines.append(
        (
            "Opción opcional: `compare-binance` compara yfinance con Binance y, para BTC/ETH/LTC, puede activar el puente legacy."
            if es
            else "Optional option: `compare-binance` compares yfinance with Binance and, for BTC/ETH/LTC, can surface the legacy bridge."
        )
    )
    lines.append(
        (
            "Cómo usarlo: escribe un ticker solo, como AAPL, para correr un análisis por defecto; agrega fechas si quieres acotar la ventana; o pide una fila limpia con `show the clean row for AAPL on 2026-03-21`."
            if es
            else "How to use them: type a ticker alone, like AAPL, to run a default analysis; add dates if you want to narrow the window; or ask for a clean row with `show the clean row for AAPL on 2026-03-21`."
        )
    )
    lines.append(
        (
            "Ejemplos rápidos: AAPL · MSFT · NVDA · SPY · QQQ · ^GSPC · BTC-USD · ETH-USD · EURUSD=X."
            if es
            else "Quick examples: AAPL · MSFT · NVDA · SPY · QQQ · ^GSPC · BTC-USD · ETH-USD · EURUSD=X."
        )
    )
    return f"{_run_prefix(summary)}" + " ".join(lines)


_CLEAN_COLUMN_ALIASES = [
    ("date", ("date", "fecha")),
    ("ticker", ("ticker", "symbol", "símbolo", "simbolo")),
    ("open", ("open", "apertura")),
    ("high", ("high", "high price", "máximo", "maximo", "alto")),
    ("low", ("low", "low price", "mínimo", "minimo", "bajo")),
    ("close", ("close", "close price", "cierre")),
    ("adj_close", ("adj close", "adj_close", "adjusted close", "adjusted")),
    ("volume", ("volume", "volumen")),
    ("return_1d", ("return 1d", "return_1d", "daily return", "retorno 1d", "retorno diario")),
    ("range_pct", ("range pct", "range_pct", "rango pct", "rango")),
    ("body_pct", ("body pct", "body_pct", "cuerpo pct", "cuerpo")),
    ("sma_5", ("sma 5", "sma_5", "moving average 5", "media 5")),
    ("sma_10", ("sma 10", "sma_10", "moving average 10", "media 10")),
    ("volatility_5", ("volatility 5", "volatility_5", "volatilidad 5")),
    ("volume_sma_5", ("volume sma 5", "volume_sma_5", "volumen sma 5")),
    ("future_close", ("future close", "future_close")),
    ("future_return", ("future return", "future_return")),
    ("target_direction", ("target direction", "target_direction", "objetivo", "direction target")),
]

_CLEAN_METRIC_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "return_1d",
    "range_pct",
    "body_pct",
    "sma_5",
    "sma_10",
    "volatility_5",
    "volume_sma_5",
    "future_close",
    "future_return",
    "target_direction",
]


def _extract_requested_columns(message: str) -> list[str]:
    lowered = (message or "").lower()
    selected: list[str] = []
    for canonical, hints in _CLEAN_COLUMN_ALIASES:
        if any(hint in lowered for hint in hints):
            selected.append(canonical)
    if not selected and any(hint in lowered for hint in ("open", "high", "low", "close", "volume", "ticker", "date")):
        for canonical in ("date", "ticker", "open", "high", "low", "close", "adj_close", "volume"):
            if canonical not in selected and canonical in lowered:
                selected.append(canonical)
    return list(dict.fromkeys(selected))


def _extract_clean_data_mode(message: str) -> str:
    lowered = (message or "").lower()
    analysis_hints = (
        "analysis",
        "analyze",
        "analyse",
        "analize",
        "analiza",
        "analizar",
        "interpret",
        "interpretation",
        "break down",
        "row analysis",
        "clean row analysis",
        "cleaned row analysis",
        "analyze the clean row",
        "analyze clean row",
        "analyze the cleaned row",
        "analyze cleaned row",
        "analiza la fila",
        "analiza la fila limpia",
        "analiza esta fila",
        "explica esta fila",
        "what does this row mean",
        "qué dice esta fila",
        "que dice esta fila",
        "qué dice esta fila limpia",
        "que dice esta fila limpia",
        "how do i read this row",
        "how to read this row",
        "read this row",
        "explain this row",
        "análisis de fila",
        "analisis de fila",
        "análisis de la fila",
        "analisis de la fila",
    )
    metric_hints = (
        "metrics",
        "metric",
        "métricas",
        "metricas",
        "clean data metrics",
        "cleaned csv metrics",
        "cleaned data metrics",
        "metrics in the cleaned data",
        "what metrics are in the cleaned data",
        "what metrics are available in the cleaned data",
        "cleaned metrics",
        "clean data fields",
        "clean data columns",
        "qué métricas hay",
        "que metricas hay",
        "qué métricas están",
        "que metricas estan",
        "métricas del clean",
        "metricas del clean",
    )
    row_hints = (
        "row",
        "rows",
        "fila",
        "filas",
        "registro",
        "registros",
        "specific row",
        "exact row",
        "last row",
        "latest row",
        "last cleaned row",
        "latest cleaned row",
        "clean row",
        "cleaned row",
        "última fila",
        "ultima fila",
        "último registro",
        "ultimo registro",
        "qué dice esta fila",
        "que dice esta fila",
        "qué dice esta fila limpia",
        "que dice esta fila limpia",
        "what does this clean row say",
        "what does this row say",
        "what does this row mean",
        "show me the clean row",
    )
    if any(hint in lowered for hint in analysis_hints) and any(
        hint in lowered
        for hint in (
            "clean",
            "cleaned",
            "clean market data",
            "cleaned data",
            "clean market row",
            "clean row",
            "cleaned row",
            "row",
            "fila",
            "registro",
            "datos limpios",
            "clean_market_data",
        )
    ):
        return "analysis"
    symbol_hints = (
        "symbols in the cleaned data",
        "symbols in clean data",
        "what symbols are in",
        "which symbols are in",
        "cleaned data symbols",
        "clean data symbols",
        "clean market data symbols",
        "symbols per ticker",
        "symbols per symbol",
        "per ticker",
        "per symbol",
        "tickers in the cleaned data",
        "activos del clean",
        "símbolos del clean",
        "simbolos del clean",
        "símbolos en los datos limpios",
        "simbolos en los datos limpios",
    )
    schema_hints = (
        "schema",
        "structure",
        "data structure",
        "format",
        "columns",
        "columnas",
        "what is scraped",
        "what is written",
        "what gets written",
        "how is it written",
        "qué estructura",
        "que estructura",
        "qué columnas",
        "que columnas",
    )
    if any(hint in lowered for hint in symbol_hints):
        return "symbols"
    if any(hint in lowered for hint in metric_hints) and any(
        hint in lowered
        for hint in (
            "clean",
            "cleaned",
            "clean market data",
            "cleaned data",
            "datos limpios",
            "clean_market_data",
        )
    ):
        return "metrics"
    if any(hint in lowered for hint in schema_hints):
        return "schema"
    if any(hint in lowered for hint in row_hints):
        return "analysis" if any(hint in lowered for hint in analysis_hints) else "row"
    return "hub"


def _extract_clean_data_row_selector(message: str) -> Dict[str, Any]:
    lowered = (message or "").lower()
    selector: Dict[str, Any] = {"row_index": None, "date": None}
    last_row_hints = (
        "last row",
        "latest row",
        "last cleaned row",
        "latest cleaned row",
        "última fila",
        "ultima fila",
        "último registro",
        "ultimo registro",
        "final row",
        "ultima fila limpia",
        "última fila limpia",
    )
    if any(hint in lowered for hint in last_row_hints):
        selector["row_index"] = -1

    row_match = re.search(r"(?:row|fila|registro)\s*(?:number\s*)?(?P<index>\d+)", lowered)
    if row_match:
        selector["row_index"] = int(row_match.group("index"))

    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", message or "")
    if not date_match:
        date_match = re.search(r"\b\d{4}/\d{2}/\d{2}\b", message or "")
    if not date_match:
        date_match = re.search(r"\b\d{4}\.\d{2}\.\d{2}\b", message or "")
    if date_match:
        selector["date"] = date_match.group(0)
    return selector


def _ticker_key(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", _normalize_metric_ticker(value).upper())


def _extract_clean_data_ticker(message: str, available_tickers: Optional[list[str]] = None) -> Optional[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in available_tickers or []:
        candidate = _normalize_metric_ticker(value)
        key = _ticker_key(candidate)
        if candidate and key and key not in seen:
            seen.add(key)
            candidates.append(candidate)

    message_key = re.sub(r"[^A-Z0-9]", "", (message or "").upper())
    for candidate in candidates:
        candidate_key = _ticker_key(candidate)
        if not candidate_key:
            continue
        candidate_variants = {candidate_key, candidate_key.rstrip("X"), candidate_key.replace("X", "")}
        if any(variant and variant in message_key for variant in candidate_variants):
            return candidate

    if len(candidates) == 1:
        return candidates[0]
    return None


def _format_clean_data_schema(
    summary: Dict[str, Any],
    result: Dict[str, Any],
    df,
    language: str = "en",
) -> str:
    es = _is_spanish(language)
    request = ((result.get("manifest") or {}).get("request") or {})
    raw_columns = summary.get("data", {}).get("raw_columns") or []
    feature_columns = summary.get("data", {}).get("feature_columns") or []
    available_columns = list(df.columns) if df is not None else []
    clean_columns = available_columns or feature_columns
    target_column = summary.get("data", {}).get("target_column") or "target_direction"
    tickers = summary.get("tickers") or request.get("tickers") or []
    unique_tickers = []
    if df is not None and "ticker" in df.columns:
        unique_tickers = sorted({str(item).upper().strip() for item in df["ticker"].dropna().astype(str) if str(item).strip()})
    lines = [
        f"{_run_prefix(summary)}"
        f"{'Esquema de clean_market_data' if es else 'Clean market data schema'}: "
        f"{'filas limpias' if es else 'cleaned rows'}={len(df) if df is not None else 0}; "
        f"{'columnas escritas' if es else 'written columns'}={len(clean_columns)}.",
    ]
    if tickers:
        lines.append(
            f"{'Símbolo actual' if es else 'Current symbol'}: {', '.join(str(item) for item in tickers)}."
        )
    if unique_tickers:
        lines.append(
            f"{'Símbolos presentes' if es else 'Symbols present'}: {', '.join(unique_tickers)}."
        )
    lines.append(
        f"{'Columnas crudas' if es else 'Raw columns'}: {', '.join(raw_columns) if raw_columns else 'n/a'}."
    )
    lines.append(
        f"{'Columnas limpias' if es else 'Clean columns'}: {', '.join(clean_columns) if clean_columns else 'n/a'}."
    )
    lines.append(
        (
            f"{'Cada fila limpia representa un ticker y una fecha únicos. target_direction marca si el próximo cierre sube (1), baja (0) o no existe (n/a).'
            if es
            else 'Each cleaned row is one unique ticker/date observation. target_direction marks whether the next close goes up (1), down (0), or is unavailable (n/a).'}"
        )
    )
    if feature_columns:
        lines.append(
            f"{'Columnas de modelo' if es else 'Model columns'}: {', '.join(feature_columns)}."
    )
    return " ".join(lines)


def _format_clean_data_metrics(
    summary: Dict[str, Any],
    result: Dict[str, Any],
    df,
    language: str = "en",
) -> str:
    es = _is_spanish(language)
    request = ((result.get("manifest") or {}).get("request") or {})
    available_columns = list(df.columns) if df is not None else []
    raw_columns = summary.get("data", {}).get("raw_columns") or []
    feature_columns = summary.get("data", {}).get("feature_columns") or []
    tickers = summary.get("tickers") or request.get("tickers") or []
    unique_tickers = []
    if df is not None and "ticker" in df.columns:
        unique_tickers = sorted({str(item).upper().strip() for item in df["ticker"].dropna().astype(str) if str(item).strip()})

    canonical_available = [column for column in _CLEAN_METRIC_COLUMNS if column in available_columns]
    raw_present = [column for column in raw_columns if column in canonical_available]
    derived_present = [column for column in canonical_available if column not in raw_present]
    model_present = [column for column in feature_columns if column in canonical_available]

    lines = [
        f"{_run_prefix(summary)}"
        f"{'Métricas de clean_market_data' if es else 'Clean market data metrics'}: "
        f"{'columnas disponibles' if es else 'available columns'}={len(canonical_available)}.",
    ]
    if tickers:
        lines.append(
            f"{'Símbolo actual' if es else 'Current symbol'}: {', '.join(str(item) for item in tickers)}."
        )
    if unique_tickers:
        lines.append(
            f"{'Símbolos presentes' if es else 'Symbols present'}: {', '.join(unique_tickers)}."
        )
    if summary.get("date_range"):
        date_range = summary["date_range"]
        if date_range.get("start") and date_range.get("end"):
            lines.append(
                f"{'Rango' if es else 'Date range'}: {date_range.get('start')} -> {date_range.get('end')}."
            )
    if raw_present:
        lines.append(
            f"{'Campos base' if es else 'Base fields'}: {', '.join(raw_present)}."
        )
    if derived_present:
        lines.append(
            f"{'Métricas derivadas' if es else 'Derived metrics'}: {', '.join(derived_present)}."
        )
    if model_present:
        lines.append(
            f"{'Columnas listas para modelo' if es else 'Model-ready columns'}: {', '.join(model_present)}."
        )
    lines.append(
        (
            "Prueba con `what is the volume of AAPL?`, `what is the volatility of AAPL?`, o `show the clean row for AAPL on 2026-03-21`."
            if not es
            else "Prueba con `cuál es el volumen de AAPL?`, `cuál es la volatilidad de AAPL?`, o `muestra la fila limpia de AAPL el 2026-03-21`."
        )
    )
    lines.append(
        "Para el impacto en la señal, mira el orquestador."
        if es
        else "For signal impact, ask the orchestrator."
    )
    return " ".join(lines)


def _format_clean_data_row_analysis(
    summary: Dict[str, Any],
    row,
    subset,
    language: str = "en",
) -> str:
    es = _is_spanish(language)
    if row is None or getattr(row, "empty", True):
        return (
            "No hay una fila seleccionada para analizar todavía."
            if es
            else "No cleaned row has been selected for analysis yet."
        )

    record = row.iloc[0]
    open_value = _coerce_float(record.get("open"))
    high_value = _coerce_float(record.get("high"))
    low_value = _coerce_float(record.get("low"))
    close_value = _coerce_float(record.get("close"))
    volume_value = _coerce_float(record.get("volume"))
    volatility_value = _coerce_float(record.get("volatility_5"))
    range_value = _coerce_float(record.get("range_pct"))
    body_value = _coerce_float(record.get("body_pct"))
    target_value = _coerce_float(record.get("target_direction"))
    future_close_value = _coerce_float(record.get("future_close"))
    future_return_value = _coerce_float(record.get("future_return"))

    move_pct = None
    if open_value not in (None, 0.0) and close_value is not None:
        move_pct = (close_value - open_value) / open_value

    intraday_range_pct = None
    if open_value not in (None, 0.0) and high_value is not None and low_value is not None:
        intraday_range_pct = (high_value - low_value) / open_value

    close_position_pct = None
    if high_value is not None and low_value is not None and high_value != low_value and close_value is not None:
        close_position_pct = (close_value - low_value) / (high_value - low_value)

    subset_volume_avg = None
    subset_volatility_avg = None
    subset_range_avg = None
    try:
        import pandas as pd  # type: ignore

        if subset is not None and not subset.empty:
            if "volume" in subset.columns:
                volume_series = pd.to_numeric(subset["volume"], errors="coerce").dropna()
                if not volume_series.empty:
                    subset_volume_avg = float(volume_series.mean())
            if "volatility_5" in subset.columns:
                volatility_series = pd.to_numeric(subset["volatility_5"], errors="coerce").dropna()
                if not volatility_series.empty:
                    subset_volatility_avg = float(volatility_series.mean())
            if "range_pct" in subset.columns:
                range_series = pd.to_numeric(subset["range_pct"], errors="coerce").dropna()
                if not range_series.empty:
                    subset_range_avg = float(range_series.mean())
    except Exception:
        pass

    volume_ratio = None
    if volume_value is not None and subset_volume_avg not in (None, 0.0):
        volume_ratio = volume_value / subset_volume_avg if subset_volume_avg else None

    target_reading = "unavailable" if not es else "no disponible"
    if target_value == 1:
        target_reading = "next close higher" if not es else "el siguiente cierre fue mayor"
    elif target_value == 0:
        target_reading = "next close lower" if not es else "el siguiente cierre fue menor"

    move_text = f"{move_pct:+.2%}" if move_pct is not None else "n/a"
    range_text = f"{intraday_range_pct:.2%}" if intraday_range_pct is not None else "n/a"
    close_position_text = f"{close_position_pct:.0%}" if close_position_pct is not None else "n/a"
    volume_ratio_text = f"{volume_ratio:.2f}x" if volume_ratio is not None else "n/a"

    bullish = bool(move_pct is not None and move_pct > 0) or target_value == 1
    bearish = bool(move_pct is not None and move_pct < 0) or target_value == 0
    if bullish and not bearish:
        local_read = "mildly bullish" if not es else "ligeramente alcista"
    elif bearish and not bullish:
        local_read = "mildly bearish" if not es else "ligeramente bajista"
    else:
        local_read = "mixed or neutral" if not es else "mixta o neutra"

    header = "Análisis de fila" if es else "Row analysis"
    lines = [
        f"{header}: {record.get('ticker') or 'n/a'} {'el' if es else 'on'} {_format_value(record.get('date'))}.",
    ]
    range_phrase = "del rango del día" if es else "of the day's range"
    lines.append(
        (
            f"{'Movimiento de cierre' if es else 'Close move'}: {move_text}; "
            f"{'rango intradía' if es else 'intraday range'}: {range_text}; "
            f"{'cierre en' if es else 'close at'} {close_position_text} {range_phrase}."
        ),
    )
    if volume_value is not None:
        volume_line = f"{'Volumen' if es else 'Volume'}: {_format_market_number('volume', volume_value)}"
        if volume_ratio_text != "n/a":
            volume_line += f" vs {('promedio' if es else 'avg')} {volume_ratio_text}"
        if subset_volume_avg is not None:
            volume_line += f"; {('promedio del subconjunto' if es else 'subset average')} {_format_market_number('volume', subset_volume_avg)}"
        lines.append(volume_line + ".")
    metric_bits: list[str] = []
    if volatility_value is not None:
        metric_bits.append(f"{'volatilidad' if es else 'volatility'}={_format_market_number('volatility_5', volatility_value)}")
    if range_value is not None:
        metric_bits.append(f"{'rango_pct' if es else 'range_pct'}={_format_market_number('range_pct', range_value)}")
    if body_value is not None:
        metric_bits.append(f"{'cuerpo_pct' if es else 'body_pct'}={_format_market_number('body_pct', body_value)}")
    if metric_bits:
        lines.append(f"{'Señales limpias' if es else 'Clean signals'}: {', '.join(metric_bits)}.")
    lines.append(
        f"{'Etiqueta objetivo' if es else 'Target label'}: target_direction={_format_value(record.get('target_direction'))} ({target_reading})."
    )
    if future_close_value is not None or future_return_value is not None:
        forward_bits = []
        if future_close_value is not None:
            forward_bits.append(f"future_close={_format_value(future_close_value)}")
        if future_return_value is not None:
            forward_bits.append(f"future_return={_format_market_number('future_return', future_return_value)}")
        lines.append(f"{'Proyección limpia' if es else 'Forward values'}: {', '.join(forward_bits)}.")
    lines.append(f"{'Lectura local' if es else 'Local read'}: {local_read}.")
    if subset_volatility_avg is not None or subset_range_avg is not None:
        context_bits = []
        if subset_volatility_avg is not None:
            context_bits.append(f"{'promedio volatilidad' if es else 'avg volatility'}={_format_market_number('volatility_5', subset_volatility_avg)}")
        if subset_range_avg is not None:
            context_bits.append(f"{'promedio rango' if es else 'avg range'}={_format_market_number('range_pct', subset_range_avg)}")
        lines.append(f"{'Contexto comparativo' if es else 'Comparative context'}: {', '.join(context_bits)}.")
    lines.append(
        "El contexto long/short/hold vive en el orquestador."
        if es
        else "Long/short/hold context lives in the orchestrator."
    )
    return "\n".join(lines)


def _should_enrich_clean_data_with_groq(message: str) -> bool:
    lowered = (message or "").lower()
    return any(
        hint in lowered
        for hint in (
            "analyze",
            "analysis",
            "explain",
            "explanation",
            "why",
            "deep",
            "deeper",
            "insight",
            "understand",
            "what does this mean",
            "how does this row",
            "how do these rows",
            "interpreta",
            "analiza",
            "análisis",
            "explica",
            "por qué",
            "profundo",
            "más profundo",
            "mas profundo",
            "entender",
            "qué significa",
            "que significa",
        )
    )


def _load_clean_dataframe(summary: Dict[str, Any], result: Dict[str, Any]):
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None

    cleaned_path = (
        (summary.get("files") or {}).get("cleaned")
        or (((result.get("cleaning") or {}).get("clean_data") or {}).get("artifact") or {}).get("path")
        or (((result.get("manifest") or {}).get("request") or {}).get("artifact_root"))
    )
    if not cleaned_path:
        return None
    path = Path(str(cleaned_path))
    if path.is_dir():
        path = path / "cleaned" / "clean_market_data.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass
    if "ticker" in df.columns:
        try:
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        except Exception:
            pass
    return df


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:  # NaN check
            return "n/a"
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    try:
        if numeric != numeric:  # NaN check
            return None
    except Exception:
        return None
    return numeric


def _format_clean_row_definition(language: str = "en") -> list[str]:
    es = _is_spanish(language)
    base_fields = "date, ticker, open, high, low, close, adj_close, volume"
    derived_fields = "return_1d, range_pct, body_pct, sma_5, sma_10, volatility_5, volume_sma_5, future_close, future_return, target_direction"
    if es:
        return [
            "Fila limpia: un registro único por fecha y ticker.",
            f"Campos base: {base_fields}.",
            f"Campos derivados: {derived_fields}.",
        ]
    return [
        "Clean row: one unique record per date and ticker.",
        f"Base fields: {base_fields}.",
        f"Derived fields: {derived_fields}.",
    ]


def _format_asset_used(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    request = ((result.get("manifest") or {}).get("request") or {})
    tickers = summary.get("tickers") or request.get("tickers") or []
    current = ", ".join(tickers) if tickers else ("ninguno" if es else "none")
    date_range = summary.get("date_range") or {}
    parts = [
        f"{_run_prefix(summary)}"
        f"{'Símbolo usado en la última corrida' if es else 'Last symbol used in the latest run'}: {current}.",
    ]
    if date_range.get("start") and date_range.get("end"):
        parts.append(
            f"{'Rango de fechas' if es else 'Date range'}: {date_range.get('start')} -> {date_range.get('end')}."
        )
    compare_binance = bool(request.get("compare_binance"))
    comparison_asset = request.get("comparison_asset")
    comparison_yf = request.get("comparison_yfinance_ticker")
    comparison_binance = request.get("comparison_binance_symbol")
    if compare_binance and (comparison_asset or comparison_yf or comparison_binance):
        parts.append(
            (
                f"{'Comparación Binance' if es else 'Binance comparison'}: yfinance={comparison_yf or current}, "
                f"{'Binance' if es else 'Binance'}={comparison_binance or 'n/a'}, "
                f"{'activo' if es else 'asset'}={comparison_asset or 'n/a'}."
            )
        )
    legacy = summary.get("legacy_analysis") or result.get("legacy_analysis") or {}
    if legacy:
        if legacy.get("enabled", False):
            parts.append(
                f"{'Puente legacy activo para' if es else 'Legacy bridge active for'} {legacy.get('asset') or legacy.get('requested_asset') or 'n/a'}."
            )
        elif legacy.get("error"):
            parts.append(f"{'Puente legacy no disponible' if es else 'Legacy bridge unavailable'}: {legacy.get('error')}.")
    return " ".join(parts)


def _translate_market_label(value: str, language: str = "en") -> str:
    es = _is_spanish(language)
    labels = {
        "crypto": ("cripto", "crypto"),
        "forex": ("forex", "forex"),
        "index": ("índice", "index"),
        "fund": ("fondo", "fund"),
        "equity": ("renta variable", "equity"),
        "unknown": ("desconocido", "unknown"),
        "digital asset pair": ("par de activo digital", "digital asset pair"),
        "currency pair": ("par de divisas", "currency pair"),
        "market index": ("índice de mercado", "market index"),
        "ETF": ("ETF", "ETF"),
        "stock": ("acción", "stock"),
    }
    label_es, label_en = labels.get(str(value or "").strip(), (str(value or "desconocido").strip(), str(value or "unknown").strip()))
    return label_es if es else label_en


def _translate_market_rationale(value: str, language: str = "en") -> str:
    es = _is_spanish(language)
    mapping = {
        "no symbol provided": (
            "no se proporcionó un símbolo",
            "no symbol provided",
        ),
        "symbols ending in =x are treated as forex pairs in yfinance": (
            "los símbolos que terminan en =X se tratan como pares forex en yfinance",
            "symbols ending in =X are treated as forex pairs in yfinance",
        ),
        "symbols starting with ^ are treated as market indexes": (
            "los símbolos que empiezan con ^ se tratan como índices de mercado",
            "symbols starting with ^ are treated as market indexes",
        ),
        "symbols ending in -usd/-usdt/-usdc are treated as crypto pairs": (
            "los símbolos que terminan en -USD/-USDT/-USDC se tratan como pares cripto",
            "symbols ending in -USD/-USDT/-USDC are treated as crypto pairs",
        ),
        "known etf ticker set": (
            "el símbolo coincide con un conjunto conocido de ETFs",
            "the symbol matches a known ETF ticker set",
        ),
        "default listed ticker classification": (
            "clasificación por defecto para un ticker listado",
            "default listed ticker classification",
        ),
    }
    key = str(value or "").strip().lower()
    translated = mapping.get(key)
    if not translated:
        return str(value or "").strip()
    return translated[0] if es else translated[1]


def _infer_market_type_from_web_facts(web_facts: List[Any]) -> str:
    text_parts: list[str] = []
    for item in web_facts[:3]:
        text_parts.append(str(_web_fact_value(item, "title") or ""))
        text_parts.append(str(_web_fact_value(item, "snippet") or ""))
    normalized = " ".join(text_parts).lower()
    keyword_map = {
        "crypto": ("crypto", "digital asset", "cryptocurrency", "token", "bitcoin", "ethereum"),
        "forex": ("forex", "currency pair", "fx"),
        "index": ("index", "benchmark"),
        "fund": ("etf", "fund", "exchange-traded fund"),
        "equity": ("stock", "equity", "shares"),
    }
    for market_type, keywords in keyword_map.items():
        if any(keyword in normalized for keyword in keywords):
            return market_type
    return ""


def _format_source_priority(language: str = "en") -> str:
    es = _is_spanish(language)
    return (
        "Prioridad de fuente: artifacts locales > contexto externo."
        if es
        else "Source priority: local artifacts > external context."
    )


def _has_market_type_conflict(local_market_type: str, web_facts: List[Any]) -> bool:
    inferred_web_type = _infer_market_type_from_web_facts(web_facts)
    return bool(inferred_web_type and str(local_market_type or "").strip() and inferred_web_type != str(local_market_type or "").strip())


def _infer_direction_from_web_facts(web_facts: List[Any]) -> str:
    text_parts: list[str] = []
    for item in web_facts[:3]:
        text_parts.append(str(_web_fact_value(item, "title") or ""))
        text_parts.append(str(_web_fact_value(item, "snippet") or ""))
    normalized = " ".join(text_parts).lower()
    long_keywords = ("bullish", "uptrend", "upside", "rally", "rebound", "gains", "strong demand", "recovery")
    short_keywords = ("bearish", "downtrend", "downside", "selloff", "drop", "decline", "weakness", "falling")
    long_hit = any(keyword in normalized for keyword in long_keywords)
    short_hit = any(keyword in normalized for keyword in short_keywords)
    if long_hit and not short_hit:
        return "long"
    if short_hit and not long_hit:
        return "short"
    return ""


def _has_decision_conflict(local_decision: str, web_facts: List[Any]) -> bool:
    external_bias = _infer_direction_from_web_facts(web_facts)
    return bool(external_bias and str(local_decision or "").strip() and external_bias != str(local_decision or "").strip())


def _infer_metric_bias_from_web_facts(metric: str, web_facts: List[Any]) -> str:
    text_parts: list[str] = []
    for item in web_facts[:3]:
        text_parts.append(str(_web_fact_value(item, "title") or ""))
        text_parts.append(str(_web_fact_value(item, "snippet") or ""))
    normalized = " ".join(text_parts).lower()
    if metric == "volume":
        high_keywords = ("surging volume", "heavy volume", "high volume", "strong volume", "elevated volume", "unusual volume")
        low_keywords = ("light volume", "low volume", "thin trading", "weak volume")
    elif metric == "volatility_5":
        high_keywords = ("high volatility", "volatile", "elevated volatility", "large swings", "sharp swings", "choppy")
        low_keywords = ("low volatility", "stable", "muted volatility", "calm trading", "quiet trading")
    else:
        return ""
    if any(keyword in normalized for keyword in high_keywords):
        return "high"
    if any(keyword in normalized for keyword in low_keywords):
        return "low"
    return ""


def _infer_local_metric_bias(metric: str, latest: float | None, avg: float | None) -> str:
    if latest is None:
        return ""
    if metric == "volume":
        if avg and avg > 0:
            if latest >= avg * 1.1:
                return "high"
            if latest <= avg * 0.9:
                return "low"
        return ""
    if metric == "volatility_5":
        if avg and avg > 0:
            if latest >= avg * 1.1:
                return "high"
            if latest <= avg * 0.9:
                return "low"
        if latest >= 0.03:
            return "high"
        if latest <= 0.01:
            return "low"
    return ""


def _build_market_metric_snapshot(
    summary: Dict[str, Any],
    result: Dict[str, Any],
    message: str,
    tickers: Optional[list[str]],
) -> Dict[str, Dict[str, float | None]]:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return {}
    df = _load_clean_dataframe(summary, result)
    if df is None or df.empty:
        return {}
    lowered = (message or "").lower()
    requested_metrics: list[str] = []
    if any(hint in lowered for hint in ("volume", "volumen")) and "volume" in df.columns:
        requested_metrics.append("volume")
    if any(hint in lowered for hint in ("volatility", "volatilidad")) and "volatility_5" in df.columns:
        requested_metrics.append("volatility_5")
    if not requested_metrics:
        requested_metrics = [column for column in ("volume", "volatility_5") if column in df.columns]
    selected_ticker = None
    for item in (tickers or summary.get("tickers") or ((result.get("manifest") or {}).get("request") or {}).get("tickers") or []):
        text = _normalize_metric_ticker(item)
        if text:
            selected_ticker = text
            break
    subset = df
    if selected_ticker and "ticker" in df.columns:
        normalized_df_ticker = df["ticker"].astype(str).map(_normalize_metric_ticker)
        filtered = df[normalized_df_ticker == selected_ticker]
        if not filtered.empty:
            subset = filtered
    if "date" in subset.columns:
        subset = subset.sort_values("date")
    if subset.empty:
        return {}
    latest = subset.tail(1).iloc[0]
    snapshot: Dict[str, Dict[str, float | None]] = {}
    for metric in requested_metrics:
        if metric not in subset.columns:
            continue
        series = pd.to_numeric(subset[metric], errors="coerce").dropna()
        latest_value = _coerce_float(latest.get(metric))
        avg_value = float(series.mean()) if not series.empty else None
        snapshot[metric] = {"latest": latest_value, "avg": avg_value}
    return snapshot


def _has_market_metrics_conflict(
    summary: Dict[str, Any],
    result: Dict[str, Any],
    message: str,
    tickers: Optional[list[str]],
    web_facts: List[Any],
) -> bool:
    snapshot = _build_market_metric_snapshot(summary, result, message, tickers)
    for metric, payload in snapshot.items():
        external_bias = _infer_metric_bias_from_web_facts(metric, web_facts)
        local_bias = _infer_local_metric_bias(metric, payload.get("latest"), payload.get("avg"))
        if external_bias and local_bias and external_bias != local_bias:
            return True
    return False


def _resolve_dynamic_certainty(route: AssistantRoute, web_facts: List[Any], *, conflict: bool = False) -> str:
    base = str(getattr(route, "certainty", "confirmed") or "confirmed").lower()
    if conflict:
        return "conflict"
    if route.source_mode not in {"web", "mixed"} or not web_facts:
        return base
    if base == "confirmed":
        return "confirmed_mixed"
    if base == "inferred":
        return "inferred_mixed"
    return base


def _web_fact_value(item: Any, key: str, default: Any = "") -> Any:
    if isinstance(item, dict):
        value = item.get(key, default)
    else:
        value = getattr(item, key, default)
    return default if value is None else value


def _summarize_web_facts(web_facts: List[Any]) -> Dict[str, Any]:
    domains: list[str] = []
    queries: list[str] = []
    trust_scores: list[float] = []
    for item in web_facts or []:
        domain = str(_web_fact_value(item, "domain") or "").strip().lower()
        query = str(_web_fact_value(item, "query") or "").strip()
        trust_raw = _web_fact_value(item, "trust_score", 0.0)
        try:
            trust_score = float(trust_raw or 0.0)
        except Exception:
            trust_score = 0.0
        if domain and domain not in domains:
            domains.append(domain)
        if query and query not in queries:
            queries.append(query)
        if trust_score > 0:
            trust_scores.append(trust_score)
    return {
        "fact_count": len(web_facts or []),
        "domain_count": len(domains),
        "domains": domains,
        "query_count": len(queries),
        "queries": queries,
        "top_trust_score": round(max(trust_scores), 2) if trust_scores else 0.0,
    }


def _append_direction_conflict_note(answer: str, local_decision: str, web_facts: List[Any], *, language: str = "en") -> str:
    external_bias = _infer_direction_from_web_facts(web_facts)
    if not external_bias or not str(local_decision or "").strip() or external_bias == str(local_decision or "").strip():
        return answer
    es = _is_spanish(language)
    note = (
        f"Conflicto de sesgo: el contexto web favorece una lectura {external_bias}, pero el run decidió {local_decision}."
        if es
        else f"Bias conflict: the web context leans {external_bias}, but the run decided {local_decision}."
    )
    text = str(answer or "").strip()
    if note in text:
        return text
    return f"{text} {note}".strip()


def _append_metric_conflict_note(
    answer: str,
    summary: Dict[str, Any],
    result: Dict[str, Any],
    message: str,
    tickers: Optional[list[str]],
    web_facts: List[Any],
    *,
    language: str = "en",
) -> str:
    snapshot = _build_market_metric_snapshot(summary, result, message, tickers)
    es = _is_spanish(language)
    notes: list[str] = []
    metric_names = {
        "volume": ("volumen", "volume"),
        "volatility_5": ("volatilidad", "volatility"),
    }
    for metric, payload in snapshot.items():
        external_bias = _infer_metric_bias_from_web_facts(metric, web_facts)
        local_bias = _infer_local_metric_bias(metric, payload.get("latest"), payload.get("avg"))
        if not external_bias or not local_bias or external_bias == local_bias:
            continue
        metric_es, metric_en = metric_names.get(metric, (metric, metric))
        notes.append(
            (
                f"Conflicto de métrica: la web sugiere {metric_es} {external_bias}, pero el dato local cae en {local_bias} frente a su promedio."
                if es
                else f"Metric conflict: the web suggests {metric_en} is {external_bias}, but the local reading is {local_bias} versus its average."
            )
        )
    if not notes:
        return answer
    text = str(answer or "").strip()
    for note in notes:
        if note not in text:
            text = f"{text} {note}".strip()
    return text


def _format_market_type(
    packet: Dict[str, Any],
    *,
    language: str = "en",
    web_enabled: bool = False,
) -> str:
    es = _is_spanish(language)
    local_facts = packet.get("local_facts") or {}
    identity = local_facts.get("market_identity") or {}
    symbol = str(identity.get("symbol") or local_facts.get("symbol") or "n/a").strip() or "n/a"
    market_type = _translate_market_label(str(identity.get("market_type") or "unknown"), language)
    market_type_key = str(identity.get("market_type") or "unknown").strip()
    asset_type = _translate_market_label(str(identity.get("asset_type") or "unknown"), language)
    rationale = _translate_market_rationale(str(identity.get("rationale") or ""), language)
    source_mode = str(packet.get("source_mode") or "local").strip() or "local"
    web_facts = packet.get("web_facts") or []

    lines = [
        (
            f"{_run_prefix(local_facts)}{symbol} pertenece al mercado {market_type} y se trata como {asset_type}."
            if es
            else f"{_run_prefix(local_facts)}{symbol} belongs to the {market_type} market and is treated as a {asset_type}."
        )
    ]
    if rationale:
        lines.append(
            f"{'Clasificación local' if es else 'Local classification'}: {rationale}."
        )
    if web_facts:
        top = web_facts[0] if isinstance(web_facts[0], dict) else {}
        snippet = str(top.get("snippet") or top.get("title") or "").strip()
        url = str(top.get("url") or "").strip()
        if snippet:
            lines.append(
                (
                    f"Contexto externo: {snippet}."
                    if es
                    else f"External context: {snippet}."
                )
            )
        if url:
            lines.append(f"{'Fuente' if es else 'Source'}: {url}.")
        inferred_web_type = _infer_market_type_from_web_facts(web_facts)
        if inferred_web_type:
            if inferred_web_type == market_type_key:
                lines.append(
                    (
                        f"Verificación externa: el contexto web es consistente con la clasificación local ({_translate_market_label(inferred_web_type, language)})."
                        if es
                        else f"External check: the web context is consistent with the local classification ({_translate_market_label(inferred_web_type, language)})."
                    )
                )
            else:
                lines.append(
                    (
                        f"Conflicto de fuentes: la web sugiere {_translate_market_label(inferred_web_type, language)}, pero mantengo la clasificación local como fuente primaria."
                        if es
                        else f"Source conflict: the web suggests {_translate_market_label(inferred_web_type, language)}, but I keep the local classification as the primary source."
                    )
                )
        lines.append(_format_source_priority(language))
    elif source_mode in {"web", "mixed"}:
        if web_enabled:
            lines.append(
                (
                    f"Selección de fuentes: {source_mode} (local-first), pero no encontré confirmación externa útil y mantuve la clasificación local."
                    if es
                    else f"Source selection: {source_mode} (local-first), but I did not find useful external confirmation so I kept the local classification."
                )
            )
        else:
            lines.append(
                (
                    f"Selección de fuentes: {source_mode} (local-first). El retriever web no está configurado, así que mantuve la clasificación local."
                    if es
                    else f"Source selection: {source_mode} (local-first). The web retriever is not configured, so I kept the local classification."
                )
            )
        lines.append(_format_source_priority(language))
    else:
        lines.append(
            f"{'Selección de fuentes' if es else 'Source selection'}: local."
        )
    return " ".join(item for item in lines if item)


def _format_source_attribution(
    route: AssistantRoute,
    web_facts: List[Any],
    *,
    language: str = "en",
    web_enabled: bool = False,
) -> str:
    if route.source_mode not in {"web", "mixed"}:
        return ""
    es = _is_spanish(language)
    mode = str(route.source_mode or "local")
    if web_facts:
        top = web_facts[0]
        snippet = str(_web_fact_value(top, "snippet") or _web_fact_value(top, "title") or "").strip()
        url = str(_web_fact_value(top, "url") or "").strip()
        parts = [
            (
                f"Selección de fuentes: {mode} (local-first)."
                if es
                else f"Source selection: {mode} (local-first)."
            )
        ]
        if snippet:
            parts.append(
                (
                    f"Contexto externo: {snippet}."
                    if es
                    else f"External context: {snippet}."
                )
            )
        if url:
            parts.append(f"{'Fuente' if es else 'Source'}: {url}.")
        extra_urls: list[str] = []
        for item in web_facts[1:3]:
            extra_url = str(_web_fact_value(item, "url") or "").strip()
            if extra_url:
                extra_urls.append(extra_url)
        if extra_urls:
            parts.append(
                (
                    f"Fuentes adicionales: {', '.join(extra_urls)}."
                    if es
                    else f"Additional sources: {', '.join(extra_urls)}."
                )
            )
        domains = (_summarize_web_facts(web_facts).get("domains") or [])[:3]
        if domains:
            parts.append(
                (
                    f"Dominios: {', '.join(domains)}."
                    if es
                    else f"Domains: {', '.join(domains)}."
                )
            )
        return " ".join(parts)
    if web_enabled:
        return (
            f"Selección de fuentes: {mode} (local-first). No encontré contexto externo útil y mantuve la respuesta sobre artifacts locales."
            if es
            else f"Source selection: {mode} (local-first). I did not find useful external context, so I kept the answer grounded in local artifacts."
        )
    return (
        f"Selección de fuentes: {mode} (local-first). El retriever web no está configurado, así que mantuve la respuesta local."
        if es
        else f"Source selection: {mode} (local-first). The web retriever is not configured, so I kept the answer local."
    )


def _append_source_attribution(
    answer: str,
    route: AssistantRoute,
    web_facts: List[Any],
    *,
    language: str = "en",
    web_enabled: bool = False,
) -> str:
    attribution = _format_source_attribution(route, web_facts, language=language, web_enabled=web_enabled)
    if not attribution:
        return answer
    text = str(answer or "").strip()
    if attribution in text:
        return text
    return f"{text} {attribution}".strip()


def _append_source_priority(answer: str, route: AssistantRoute, *, language: str = "en") -> str:
    if route.source_mode not in {"web", "mixed"}:
        return answer
    priority = _format_source_priority(language)
    text = str(answer or "").strip()
    if priority in text:
        return text
    return f"{text} {priority}".strip()


def _format_source_summary(route: AssistantRoute, web_facts: List[Any], *, language: str = "en") -> str:
    if route.source_mode not in {"web", "mixed"}:
        return ""
    es = _is_spanish(language)
    fact_count = len(web_facts or [])
    certainty = str(route.certainty or "").strip().lower()
    if route.source_mode == "mixed":
        if fact_count:
            summary = (
                f"Resumen de fuentes: facts del run + {fact_count} verificación externa"
                if es
                else f"Source summary: run facts + {fact_count} external check"
            )
            if fact_count != 1:
                summary += "es" if es else "s"
        else:
            summary = (
                "Resumen de fuentes: solo facts del run; no hubo contexto externo usable"
                if es
                else "Source summary: run facts only; no usable external context"
            )
    else:
        if fact_count:
            summary = (
                f"Resumen de fuentes: {fact_count} fuente externa"
                if es
                else f"Source summary: {fact_count} external source"
            )
            if fact_count != 1:
                summary += "s"
        else:
            summary = (
                "Resumen de fuentes: se pidió contexto web, pero no hubo facts externos usables"
                if es
                else "Source summary: web context was requested, but no usable external facts were retrieved"
            )
    if certainty == "conflict":
        summary += " en conflicto" if es else " in conflict"
    return f"{summary}."


def _format_source_blend_summary(route: AssistantRoute, web_facts: List[Any], *, language: str = "en") -> str:
    if route.source_mode not in {"web", "mixed"} or not web_facts:
        return ""
    es = _is_spanish(language)
    overview = _summarize_web_facts(web_facts)
    fact_count = int(overview.get("fact_count") or 0)
    domain_count = int(overview.get("domain_count") or 0)
    query_count = int(overview.get("query_count") or 0)
    top_trust = float(overview.get("top_trust_score") or 0.0)
    details: list[str] = []
    if fact_count:
        details.append(
            (
                f"Mezcla: {fact_count} fact{'s' if fact_count != 1 else ''} externo{'s' if fact_count != 1 else ''}"
                if es
                else f"Blend detail: {fact_count} external fact{'s' if fact_count != 1 else ''}"
            )
        )
    if domain_count:
        details.append(
            f"{domain_count} dominio{'s' if domain_count != 1 else ''}"
            if es
            else f"{domain_count} domain{'s' if domain_count != 1 else ''}"
        )
    if query_count > 1:
        details.append(
            f"{query_count} consultas"
            if es
            else f"{query_count} query variants"
        )
    if top_trust > 0:
        details.append(
            f"trust máximo={top_trust:.2f}"
            if es
            else f"top trust={top_trust:.2f}"
        )
    return ", ".join(details) + "." if details else ""


def _append_source_summary(answer: str, route: AssistantRoute, web_facts: List[Any], *, language: str = "en") -> str:
    summary = _format_source_summary(route, web_facts, language=language)
    text = str(answer or "").strip()
    if summary and summary not in text:
        text = f"{text} {summary}".strip()
    blend = _format_source_blend_summary(route, web_facts, language=language)
    if blend and blend not in text:
        text = f"{text} {blend}".strip()
    return text


def _append_external_backdrop(
    answer: str,
    route: AssistantRoute,
    web_facts: List[Any],
    *,
    language: str = "en",
) -> str:
    if route.source_mode not in {"web", "mixed"} or not web_facts:
        return answer
    es = _is_spanish(language)
    top = web_facts[0]
    snippet = str(_web_fact_value(top, "snippet") or _web_fact_value(top, "title") or "").strip()
    domain = str(_web_fact_value(top, "domain") or "").strip()
    if not snippet:
        return answer
    backdrop = (
        f"Contexto externo integrado: {snippet}. La prioridad sigue siendo el run y sus artifacts."
        if es
        else f"Integrated external backdrop: {snippet}. The run and its artifacts remain the primary source of truth."
    )
    if domain:
        backdrop += f" {'Dominio' if es else 'Domain'}: {domain}."
    text = str(answer or "").strip()
    if backdrop in text:
        return text
    return f"{text} {backdrop}".strip()


def _format_semantic_web_brief(route: AssistantRoute, web_facts: List[Any], language: str = "en") -> str:
    if not web_facts:
        return ""
    es = _is_spanish(language)
    semantic_message = route.interpreted_query or route.raw_message or ""
    comparison_subjects = _semantic_comparison_subjects_from_message(semantic_message)
    if comparison_subjects and len(web_facts) >= 2:
        left_subject, right_subject = comparison_subjects[0], comparison_subjects[1]
        left_fact = web_facts[0]
        right_fact = web_facts[1]
        left_title = str(_web_fact_value(left_fact, "title") or "").strip()
        left_snippet = str(_web_fact_value(left_fact, "snippet") or "").strip()
        right_title = str(_web_fact_value(right_fact, "title") or "").strip()
        right_snippet = str(_web_fact_value(right_fact, "snippet") or "").strip()
        label = "Brief semántico comparativo" if es else "Semantic comparison brief"
        pieces = []
        if left_title or left_snippet:
            left_text = left_snippet or left_title
            if left_title and left_snippet and left_title.lower() not in left_snippet.lower():
                left_text = f"{left_title}. {left_snippet}"
            pieces.append(f"{left_subject}: {left_text}")
        if right_title or right_snippet:
            right_text = right_snippet or right_title
            if right_title and right_snippet and right_title.lower() not in right_snippet.lower():
                right_text = f"{right_title}. {right_snippet}"
            pieces.append(f"{right_subject}: {right_text}")
        if pieces:
            joined = " | ".join(pieces)
            return f"{label}: {joined}"
    first = web_facts[0]
    title = str(_web_fact_value(first, "title") or "").strip()
    snippet = str(_web_fact_value(first, "snippet") or "").strip()
    text = snippet or title
    if not text:
        return ""
    title_prefix = "Brief semántico web" if es else "Semantic web brief"
    if title and snippet and title.lower() not in snippet.lower():
        return f"{title_prefix}: {title}. {snippet}"
    return f"{title_prefix}: {text}"


def _semantic_subject_from_message(message: str) -> str:
    text = str(message or "").strip()
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


def _looks_like_semantic_follow_up(message: str) -> bool:
    lowered = str(message or "").strip().lower()
    if not lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "complement",
            "complementa",
            "expand",
            "expande",
            "expándelo",
            "expandelo",
            "with internet",
            "con internet",
            "use internet",
            "usa internet",
            "using internet",
            "contexto externo",
            "external context",
            "search that",
            "busca eso",
            "busca en internet",
        )
    )


def _semantic_comparison_subjects_from_message(message: str) -> list[str]:
    text = str(message or "").strip()
    patterns = (
        r"^(?:what is the difference between|what's the difference between|que diferencia hay entre|qué diferencia hay entre|difference between|diferencia entre|compare|compara)\s+(.+?)\s+(?:and|vs\.?|versus|y|con)\s+(.+?)(?:\?|$)",
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


def _format_semantic_local_anchor(route: AssistantRoute, state: AssistantState, language: str = "en") -> str:
    es = _is_spanish(language)
    ticker = ""
    if route.tickers:
        ticker = str(route.tickers[0]).strip()
    elif state.current_asset:
        ticker = str(state.current_asset).strip()
    run_id = str(route.run_id or state.last_run_id or "").strip()
    parts: list[str] = []
    if run_id:
        parts.append(f"run={run_id}")
    if ticker:
        parts.append(f"símbolo={ticker}" if es else f"symbol={ticker}")
    if not parts:
        return ""
    prefix = "Ancla local" if es else "Local anchor"
    return f"{prefix}: {' | '.join(parts)}."


def _semantic_basis_message(message: str, state: AssistantState) -> str:
    current = str(message or "").strip()
    if _semantic_subject_from_message(current) or _semantic_comparison_subjects_from_message(current):
        return current
    if str(state.last_intent or "").strip() == "show_semantic_lookup" and _looks_like_semantic_follow_up(current):
        previous = str((state.last_route or {}).get("raw_message") or "").strip()
        if previous:
            return previous
    return current


def _is_semantic_subject_candidate(subject: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(subject or "").strip().lower())
    if not lowered:
        return False
    generic_status_tokens = {"latest", "today", "current", "now", "hoy", "ahora"}
    subject_tokens = {token for token in re.split(r"\s+", lowered) if token}
    if subject_tokens and subject_tokens.issubset(generic_status_tokens):
        return False
    if re.search(r"\brun_\d+\b", lowered):
        return False
    exclusions = (
        "extraction",
        "extraccion",
        "extracción",
        "cleaning",
        "limpieza",
        "modeling",
        "modelado",
        "orchestrator",
        "orquestador",
        "status",
        "estado",
        "help",
        "ayuda",
        "web status",
        "estado web",
        "source",
        "fuente",
        "time",
        "fecha",
        "hora",
        "ticker",
        "symbol",
        "símbolo",
        "simbolo",
        "latest run",
        "last run",
        "última corrida",
        "ultima corrida",
        "compare-binance",
        "--groq-brain",
    )
    if any(token in lowered for token in exclusions):
        return False
    if len(lowered.split()) > 6:
        return False
    return True


def _comparison_subjects_from_fragment(fragment: str) -> list[str]:
    text = re.sub(r"[?.!,;:]+$", "", str(fragment or "").strip())
    if not text:
        return []
    match = re.search(r"^(.+?)\s+(?:and|vs\.?|versus|y|con)\s+(.+?)$", text, flags=re.IGNORECASE)
    if not match:
        return []
    first = re.sub(r"^(?:the|a|an|el|la|un|una)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
    second = re.sub(r"^(?:the|a|an|el|la|un|una)\s+", "", match.group(2).strip(), flags=re.IGNORECASE)
    if not first or not second:
        return []
    if not _is_semantic_subject_candidate(first) or not _is_semantic_subject_candidate(second):
        return []
    return [first, second]


def _canonical_semantic_interpretation_query(message: str, state: AssistantState) -> str:
    basis = _semantic_basis_message(message, state)
    comparison_subjects = _semantic_comparison_subjects_from_message(basis)
    if comparison_subjects:
        left, right = comparison_subjects[0], comparison_subjects[1]
        return f"what is the difference between {left} and {right}?"
    subject = _semantic_subject_from_message(basis)
    if subject and _is_semantic_subject_candidate(subject):
        return f"what is {subject}?"

    normalized = re.sub(r"\s+", " ", str(message or "").strip())
    opener_patterns = (
        r"^(?:tell me about|what can you tell me about|what about|explain|describe|walk me through)\s+(.+?)\??$",
        r"^(?:háblame de|hablame de|cuéntame sobre|cuentame sobre|explica|describe)\s+(.+?)\??$",
    )
    for pattern in opener_patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        subject_fragment = re.sub(r"[?.!,;:]+$", "", match.group(1)).strip()
        comparison_subjects = _comparison_subjects_from_fragment(subject_fragment)
        if comparison_subjects:
            left, right = comparison_subjects[0], comparison_subjects[1]
            return f"what is the difference between {left} and {right}?"
        if _is_semantic_subject_candidate(subject_fragment):
            return f"what is {subject_fragment}?"

    bare = re.sub(r"[?.!,;:]+$", "", normalized).strip()
    if _is_semantic_subject_candidate(bare):
        return f"what is {bare}?"
    return ""


def _interpretation_queries_for_semantic_query(query: str) -> list[str]:
    comparison_subjects = _semantic_comparison_subjects_from_message(query)
    if comparison_subjects:
        left, right = comparison_subjects[0], comparison_subjects[1]
        queries = [f"{left} meaning", f"{right} meaning", query]
        return [str(item).strip().lower() for item in queries if str(item).strip()]
    subject = _semantic_subject_from_message(query)
    if not subject:
        return [str(query).strip().lower()] if str(query or "").strip() else []
    queries = [f"{subject} meaning", f"{subject} definition", query]
    return [str(item).strip().lower() for item in queries if str(item).strip()]


def _has_local_semantic_coverage(query: str) -> bool:
    comparison_subjects = _semantic_comparison_subjects_from_message(query)
    if comparison_subjects:
        return all(resolve_semantic_definition_for_term(item) is not None for item in comparison_subjects[:2])
    subject = _semantic_subject_from_message(query)
    if not subject:
        return False
    return resolve_semantic_definition_for_term(subject) is not None


def _append_interpretation_trace(answer: str, route: AssistantRoute, language: str = "en") -> str:
    if str(route.interpretation_source or "").strip() != "web_search":
        return answer
    es = _is_spanish(language)
    basis = route.interpreted_query or route.raw_message or ""
    comparison_subjects = _semantic_comparison_subjects_from_message(basis)
    if comparison_subjects:
        detail = (
            f"Interpretación aumentada por búsqueda: traté esto como una comparación entre {comparison_subjects[0]} y {comparison_subjects[1]} antes de sintetizar la respuesta."
            if es
            else f"Search-augmented interpretation: I treated this as a comparison between {comparison_subjects[0]} and {comparison_subjects[1]} before synthesizing the answer."
        )
    else:
        subject = _semantic_subject_from_message(basis) or basis.strip()
        detail = (
            f"Interpretación aumentada por búsqueda: traté esto como una pregunta conceptual sobre {subject} antes de sintetizar la respuesta."
            if es
            else f"Search-augmented interpretation: I treated this as a concept question about {subject} before synthesizing the answer."
        )
    text = str(answer or "").strip()
    if not detail or detail in text:
        return text
    return f"{text} {detail}".strip()


def _message_explicitly_mentions_ticker(message: str, ticker: str) -> bool:
    message_text = str(message or "").strip()
    ticker_text = str(ticker or "").strip()
    if not message_text or not ticker_text:
        return False
    normalized_message = re.sub(r"[^A-Za-z0-9^.=:-]+", "", message_text).upper()
    normalized_ticker = re.sub(r"[^A-Za-z0-9^.=:-]+", "", ticker_text).upper()
    if not normalized_message or not normalized_ticker:
        return False
    return normalized_ticker in normalized_message


def _explicit_tickers_in_message(message: str, tickers: list[str]) -> list[str]:
    explicit: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        text = str(ticker or "").strip()
        key = text.upper()
        if not text or key in seen:
            continue
        if _message_explicitly_mentions_ticker(message, text):
            explicit.append(text)
            seen.add(key)
    return explicit


def _format_semantic_local_brief(message: str, symbol: str, language: str = "en") -> str:
    definition = resolve_semantic_definition(message, symbol=symbol)
    if not definition:
        return ""
    es = _is_spanish(language)
    title_prefix = "Brief semántico local" if es else "Local semantic brief"
    return f"{title_prefix}: {definition.label}. {definition.render(language)}"


def _format_semantic_local_comparison_brief(message: str, language: str = "en") -> str:
    comparison_subjects = _semantic_comparison_subjects_from_message(message)
    if not comparison_subjects or len(comparison_subjects) < 2:
        return ""
    left, right = comparison_subjects[0], comparison_subjects[1]
    left_definition = resolve_semantic_definition_for_term(left)
    right_definition = resolve_semantic_definition_for_term(right)
    if not left_definition or not right_definition:
        return ""
    es = _is_spanish(language)
    label = "Brief semántico comparativo local" if es else "Local semantic comparison brief"
    comparison = compare_semantic_definitions(left_definition, right_definition, language=language)
    return f"{label}: {comparison}"


def _format_clean_data_view(
    summary: Dict[str, Any],
    result: Dict[str, Any],
    message: str,
    language: str = "en",
    tickers: Optional[list[str]] = None,
) -> str:
    es = _is_spanish(language)
    df = _load_clean_dataframe(summary, result)
    if df is None or df.empty:
        return _format_availability_notice(
            title_en="Clean market data overview",
            title_es="Resumen de clean_market_data",
            status_en="No clean market data is available yet.",
            status_es="No hay datos limpios disponibles todavía.",
            action_en="Start a run first, then open clean data again.",
            action_es="Ejecuta una corrida primero y vuelve a datos limpios.",
            prompt_en="what symbols are in the cleaned data? | what does this clean row say?",
            prompt_es="qué símbolos hay en los datos limpios? | qué dice esta fila?",
            language=language,
        )

    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore

    available_columns = [column for column in df.columns]
    requested_columns = _extract_requested_columns(message)
    explicit_requested_columns = list(requested_columns)
    if not requested_columns:
        requested_columns = [
            column
            for column in ("date", "ticker", "open", "high", "low", "close", "volume", "target_direction")
            if column in available_columns
        ]
    requested_columns = [column for column in requested_columns if column in available_columns]
    base_columns = [column for column in ("date", "ticker") if column in available_columns]
    preview_columns = list(dict.fromkeys(base_columns + requested_columns))
    if not preview_columns:
        preview_columns = available_columns[: min(len(available_columns), 8)]

    clean_mode = _extract_clean_data_mode(message)
    if clean_mode == "hub" and explicit_requested_columns:
        clean_mode = "preview"
    request_tickers = tickers or summary.get("tickers") or ((result.get("manifest") or {}).get("request") or {}).get("tickers") or []
    ticker_filter = _extract_clean_data_ticker(message, request_tickers if request_tickers else list(df["ticker"].dropna().astype(str).unique()) if "ticker" in df.columns else [])
    prompt_symbol = ticker_filter or (_normalize_metric_ticker(request_tickers[0]) if request_tickers else "AAPL")
    row_definition_lines = _format_clean_row_definition(language)
    ticker_notice_lines: list[str] = []
    analysis_exact_match = True
    subset = df
    if ticker_filter and "ticker" in df.columns:
        normalized_df_ticker = df["ticker"].astype(str).map(_normalize_metric_ticker)
        filtered = df[normalized_df_ticker == ticker_filter]
        if filtered.empty:
            analysis_exact_match = False
            available = sorted({str(item).upper() for item in df["ticker"].dropna().astype(str) if str(item).strip()})
            if available:
                available_text = ", ".join(available[:8])
                ticker_notice_lines = [
                    f"{'Estado' if es else 'Status'}: {'No encontré filas limpias para' if es else 'No clean rows were found for'} {ticker_filter}.",
                    f"{'Símbolos disponibles' if es else 'Available tickers'}: {available_text}.",
                    (
                        f"{'Acción' if es else 'Action'}: {'Ejecuta extracción para ese activo primero, o revisa limpieza para ver los símbolos disponibles.' if es else 'Run extraction for that asset first, or inspect the available symbols in cleaning.'}"
                    ),
                ]
            else:
                ticker_notice_lines = [
                    f"{'Estado' if es else 'Status'}: {'No encontré filas limpias para' if es else 'No clean rows were found for'} {ticker_filter}.",
                    (
                        f"{'Acción' if es else 'Action'}: {'Ejecuta extracción para ese símbolo primero y luego vuelve a limpiar.' if es else 'Run extraction for that symbol first, then come back to cleaning.'}"
                    ),
                ]
            compact_selected = re.sub(r"[^A-Z0-9]", "", ticker_filter)
            if len(compact_selected) <= 1:
                ticker_notice_lines.append(
                    f"{'Sugerencia' if es else 'Tip'}: {'usa un símbolo más completo, por ejemplo EURUSD=X, AAPL o NVDA.' if es else 'use a fuller symbol such as EURUSD=X, AAPL, or NVDA.'}"
                )
        else:
            subset = filtered
    elif "ticker" in df.columns and request_tickers:
        for candidate in request_tickers:
            normalized_candidate = _normalize_metric_ticker(candidate)
            if not normalized_candidate:
                continue
            normalized_df_ticker = df["ticker"].astype(str).map(_normalize_metric_ticker)
            filtered = df[normalized_df_ticker == normalized_candidate]
            if not filtered.empty:
                subset = filtered
                ticker_filter = normalized_candidate
                break

    if "date" in subset.columns:
        subset = subset.sort_values("date")

    row_selector = _extract_clean_data_row_selector(message)
    explicit_analysis_request = bool(row_selector.get("date") or row_selector.get("row_index") is not None or ticker_filter)
    selected_row = None
    if clean_mode in {"row", "analysis"}:
        row_subset = subset
        if row_selector.get("date") and pd is not None and "date" in row_subset.columns:
            try:
                row_date = pd.to_datetime(row_selector["date"], errors="coerce")
                if row_date is not None and row_date == row_date:
                    date_filtered = row_subset[pd.to_datetime(row_subset["date"], errors="coerce").dt.normalize() == row_date.normalize()]
                    if date_filtered.empty:
                        analysis_exact_match = False
                    row_subset = date_filtered
            except Exception:
                pass
        row_index = row_selector.get("row_index")
        if row_index is not None and not row_subset.empty:
            if row_index == -1:
                selected_row = row_subset.tail(1)
            else:
                zero_index = int(row_index) - 1
                if 0 <= zero_index < len(row_subset):
                    selected_row = row_subset.iloc[[zero_index]]
                else:
                    analysis_exact_match = False
        elif row_index is not None and row_subset.empty:
            analysis_exact_match = False
        if selected_row is None and not row_subset.empty and not (clean_mode == "analysis" and explicit_analysis_request and not analysis_exact_match):
            selected_row = row_subset.tail(1)
        if selected_row is None and not subset.empty and not (clean_mode == "analysis" and explicit_analysis_request and not analysis_exact_match):
            selected_row = subset.tail(1)

    unique_tickers = []
    if "ticker" in df.columns:
        unique_tickers = sorted({str(item).upper().strip() for item in df["ticker"].dropna().astype(str) if str(item).strip()})

    header_label = _clean_data_header_label(clean_mode, language)
    lines = [
        f"{_run_prefix(summary)}"
        f"{header_label}: "
        f"{'filas limpias' if es else 'cleaned rows'}={len(df)}; "
        f"{'columnas' if es else 'columns'}={len(available_columns)}.",
    ]
    if request_tickers:
        lines.append(
            f"{'Símbolo actual' if es else 'Current symbol'}: {', '.join(str(item) for item in request_tickers)}."
        )
    if ticker_notice_lines:
        lines.extend(ticker_notice_lines)
    if summary.get("date_range"):
        date_range = summary["date_range"]
        if date_range.get("start") and date_range.get("end"):
            lines.append(
                f"{'Rango' if es else 'Date range'}: {date_range.get('start')} -> {date_range.get('end')}."
            )

    if clean_mode == "hub":
        lines.append("Vistas abiertas:" if es else "Open views:")
        lines.append(
            "S Símbolos | M Métricas | R Fila | A Análisis | E Esquema"
            if es
            else "S Symbols | M Metrics | R Row | A Analysis | E Schema"
        )
        if unique_tickers:
            lines.append(
                f"{'Símbolos presentes' if es else 'Symbols present'}: {', '.join(unique_tickers[:6])}."
            )
        lines.append(
            (
                "Pide una subvista para bajar al detalle, o escribe un ticker si quieres contexto del activo."
                if es
                else "Ask for a subview to drill into details, or type a ticker if you want asset context."
            )
        )
        lines.append(
            "Preguntas naturales:"
            if es
            else "Natural prompts:"
        )
        lines.append(
            "qué símbolos hay en los datos limpios? | qué métricas hay en los datos limpios? | qué dice esta fila limpia?"
            if es
            else "what symbols are in the cleaned data? | what metrics are in the cleaned data? | what does this clean row say?"
        )
        lines.append(
            "Escribe 8 o clean data para volver aquí."
            if es
            else "Type 8 or clean data to come back here."
        )
        return "\n".join(lines)

    prompt_line = ""
    if clean_mode == "symbols":
        prompt_line = (
            "Preguntas naturales: qué símbolos hay en los datos limpios? | qué símbolos puedo usar?"
            if es
            else "Natural prompts: what symbols are in the cleaned data? | what symbols can I use?"
        )
    elif clean_mode == "metrics":
        prompt_line = (
            f"Preguntas naturales: qué métricas hay en los datos limpios de {prompt_symbol}? | métricas | fila"
            if es
            else f"Natural prompts: what metrics are in the cleaned data for {prompt_symbol}? | metrics | row"
        )
    elif clean_mode == "schema":
        prompt_line = (
            "Preguntas naturales: cuál es el esquema? | qué dice la estructura de clean_market_data?"
            if es
            else "Natural prompts: what is the clean data schema? | what does clean_market_data look like?"
        )
    elif clean_mode == "analysis":
        prompt_line = (
            f"Preguntas naturales: qué dice esta fila de {prompt_symbol}? | analiza la fila limpia de {prompt_symbol}"
            if es
            else f"Natural prompts: what does this clean row say for {prompt_symbol}? | analyze the clean row for {prompt_symbol}"
        )
    elif clean_mode == "row":
        prompt_line = (
            f"Preguntas naturales: muestra la fila limpia de {prompt_symbol} | qué dice esta fila?"
            if es
            else f"Natural prompts: show me the clean row for {prompt_symbol} | what does this row say?"
        )
    if prompt_line:
        lines.append(prompt_line)
    if clean_mode == "row":
        lines.append(row_definition_lines[0])

    if clean_mode == "symbols":
        if unique_tickers:
            counts = df["ticker"].dropna().astype(str).str.upper().str.strip().value_counts()
            rendered_counts = ", ".join(f"{ticker} ({int(counts[ticker])})" for ticker in counts.index[:10])
            lines.append(
                f"{'Símbolos presentes' if es else 'Symbols present'}: {', '.join(unique_tickers)}."
            )
            lines.append(
                f"{'Filas por símbolo' if es else 'Rows per symbol'}: {rendered_counts}."
            )
        lines.append(
            (
                "Pide un ticker y una fecha o número de fila si quieres una fila limpia exacta."
                if es
                else "Ask for a ticker plus a date or row number if you want one exact cleaned row."
            )
        )
    elif clean_mode == "metrics":
        lines.append(_format_clean_data_metrics(summary, result, df, language))
    elif clean_mode == "schema":
        lines.append(_format_clean_data_schema(summary, result, df, language))
    elif clean_mode == "analysis":
        if selected_row is not None and not selected_row.empty and analysis_exact_match:
            row = selected_row.iloc[0]
            row_columns = list(dict.fromkeys(preview_columns))
            for extra in ("future_close", "future_return", "target_direction"):
                if extra in available_columns and extra not in row_columns:
                    row_columns.append(extra)
            lines.append(f"{'Fila seleccionada' if es else 'Selected row'}:")
            lines.append(" - " + " | ".join(f"{column}={_format_value(row.get(column))}" for column in row_columns))
            if row_selector.get("row_index") is not None:
                lines.append(f"{'Número de fila' if es else 'Row number'}: {row_selector.get('row_index')}.")
            if row_selector.get("date"):
                lines.append(f"{'Fecha filtrada' if es else 'Filtered date'}: {row_selector.get('date')}.")
            lines.append(_format_clean_data_row_analysis(summary, selected_row, subset, language))
        elif explicit_analysis_request:
            lines.append(
                f"{'Estado' if es else 'Status'}: {'Ninguna fila limpia exacta coincidió con el símbolo, la fecha o el número pedidos.' if es else 'No exact clean row matched the requested symbol, date, or row number.'}"
            )
            lines.append(
                f"{'Símbolos disponibles' if es else 'Available tickers'}: {', '.join(unique_tickers) if unique_tickers else 'n/a'}."
            )
            lines.append(
                f"{'Acción' if es else 'Action'}: {'Pide extracción para correr ese símbolo primero, o pregunta limpieza para ver lo que sí existe.' if es else 'Run extraction for that symbol first, or inspect what is available in cleaning.'}"
            )
        else:
            lines.append(f"{'Estado' if es else 'Status'}: {'No había una fila exacta para analizar.' if es else 'No exact row was available for analysis.'}")
            lines.append(
                f"{'Acción' if es else 'Action'}: {'Se muestra la última fila disponible.' if es else 'Showing the latest available row.'}"
            )
            lines.append(_format_clean_data_row_analysis(summary, subset.tail(1), subset, language))
    else:
        if requested_columns:
            lines.append(
                f"{'Columnas solicitadas' if es else 'Requested columns'}: {', '.join(requested_columns)}."
            )
        if clean_mode == "row" and selected_row is not None and not selected_row.empty:
            row = selected_row.iloc[0]
            row_columns = list(dict.fromkeys(preview_columns))
            for extra in ("future_close", "future_return", "target_direction"):
                if extra in available_columns and extra not in row_columns:
                    row_columns.append(extra)
            lines.append(
                f"{'Fila seleccionada' if es else 'Selected row'}:"
            )
            lines.append(
                " - " + " | ".join(f"{column}={_format_value(row.get(column))}" for column in row_columns)
            )
            lines.append(row_definition_lines[0])
            if row_selector.get("row_index") is not None:
                lines.append(
                    f"{'Número de fila' if es else 'Row number'}: {row_selector.get('row_index')}."
                )
            if row_selector.get("date"):
                lines.append(
                    f"{'Fecha filtrada' if es else 'Filtered date'}: {row_selector.get('date')}."
                )
        else:
            lines.append(
                f"{'Vista rápida de las últimas filas' if es else 'Quick view of the latest rows'}:"
            )
            preview = subset.tail(3)
            for _, row in preview.iterrows():
                row_parts = []
                for column in preview_columns:
                    row_parts.append(f"{column}={_format_value(row.get(column))}")
                lines.append(" - " + " | ".join(row_parts))
            lines.append(
                f"{'Cada fila limpia es un registro único por fecha y ticker' if es else 'Each clean row is one unique record per date and ticker'}."
            )
        lines.append(
            (
                "Estas columnas limpias estabilizan la señal y reducen ruido."
                if es
                else "These clean columns stabilize the signal and reduce noise."
            )
        )
    if unique_tickers and clean_mode != "symbols":
        lines.append(
            f"{'Símbolos presentes' if es else 'Symbols present'}: {', '.join(unique_tickers)}."
        )
    return "\n".join(lines)


def _format_decision_explanation(summary: Dict[str, Any], result: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    rows = summary.get("rows") or {}
    data = summary.get("data") or {}
    models = summary.get("models") or {}
    selection = summary.get("selection") or {}
    comparison = summary.get("comparison") or {}
    motor = summary.get("motor") or result.get("motor") or {}
    cleaned_rows = rows.get("cleaned")
    raw_rows = rows.get("raw")
    features = data.get("feature_columns") or []
    target = data.get("target_column") or "target_direction"
    decision = models.get("final_decision", result.get("final_decision", "n/a"))
    confidence = models.get("confidence", result.get("final_confidence", 0.0))
    disagreement = models.get("disagreement")
    selection_reason = selection.get("reason", "")
    lines = [
        f"{_run_prefix(summary)}"
        f"{'La decisión final fue' if es else 'The final decision was'} {decision} "
        f"{'con confianza' if es else 'with confidence'} {float(confidence or 0.0):.4f}.",
        (
            f"{'Hold significa esperar' if es else 'Hold means wait'}; "
            f"{'long significa que el modelo espera subida' if es else 'long means the model expects an upward move'}; "
            f"{'short significa bajada' if es else 'short means a downward move'}. "
            f"{'Los datos limpios reducen ruido y ayudan a separar esas señales.' if es else 'cleaner data reduces noise and helps separate those signals.'}"
        ),
    ]
    if raw_rows is not None and cleaned_rows is not None:
        lines.append(
            (
                f"Se limpiaron {raw_rows} filas crudas hasta {cleaned_rows} filas listas para modelado."
                if es
                else f"{raw_rows} raw rows were cleaned into {cleaned_rows} model-ready rows."
            )
        )
    if features:
        feature_text = ", ".join(features[:10])
        lines.append(
            (
                f"El modelo usó columnas limpias como {feature_text} para aprender la dirección futura de {target}."
                if es
                else f"The model used clean features such as {feature_text} to learn the future direction in {target}."
            )
        )
    if disagreement is not None:
        lines.append(
            (
                f"Los modelos {'discreparon' if disagreement else 'estuvieron alineados'}."
                if es
                else f"The models {'disagreed' if disagreement else 'were aligned'}."
            )
        )
    if selection_reason:
        lines.append(
            (
                f"El orquestador eligió {selection.get('strategy', 'n/a')} porque {_compact_reason_text(selection_reason)}."
                if es
                else f"The orchestrator chose {selection.get('strategy', 'n/a')} because {_compact_reason_text(selection_reason)}."
            )
        )
    if comparison.get("decision_path"):
        lines.append(
            f"{'Ruta de decisión' if es else 'Decision path'}: {comparison.get('decision_path')}."
        )
    brain = summary.get("brain") or {}
    if brain and brain.get("enabled"):
        lines.append(
            (
                f"Groq brain experimental: {brain.get('decision')} reemplazó la decisión determinista {brain.get('deterministic_decision')}."
                if es
                else f"Experimental Groq brain: {brain.get('decision')} replaced the deterministic decision {brain.get('deterministic_decision')}."
            )
        )
    if motor:
        lines.append(
            f"Motor: requested={motor.get('requested', 'n/a')} selected={motor.get('selected', 'n/a')} decision={motor.get('decision', 'n/a')}."
        )
    return " ".join(lines)


def _format_market_number(column: str, value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:  # NaN check
            return "n/a"
    except Exception:
        pass
    try:
        numeric = float(value)
    except Exception:
        return _format_value(value)
    if column == "volume":
        return f"{numeric:,.0f}"
    if column in {"volatility_5", "range_pct", "body_pct", "return_1d", "future_return"}:
        return f"{numeric * 100:.2f}%"
    return _format_value(value)


_FOREX_SYMBOLS = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "CAD",
    "AUD",
    "NZD",
    "SEK",
    "NOK",
    "DKK",
    "MXN",
    "ZAR",
    "SGD",
    "HKD",
    "PLN",
    "TRY",
    "HUF",
    "CZK",
    "ILS",
}


def _normalize_metric_ticker(token: Any) -> str:
    text = str(token or "").strip().upper()
    if not text:
        return ""
    if text.endswith("=X") or text.endswith(".TO") or text.endswith(".DE"):
        return text
    compact = re.sub(r"[^A-Z0-9]", "", text)
    if len(compact) == 6:
        base = compact[:3]
        quote = compact[3:]
        if base in _FOREX_SYMBOLS and quote in _FOREX_SYMBOLS:
            return f"{compact}=X"
    return text.replace(" ", "")


def _format_market_metrics(
    summary: Dict[str, Any],
    result: Dict[str, Any],
    message: str,
    tickers: Optional[list[str]] = None,
    language: str = "en",
) -> str:
    es = _is_spanish(language)
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return _format_clean_data_view(summary, result, message, language)

    df = _load_clean_dataframe(summary, result)
    if df is None or df.empty:
        return _format_availability_notice(
            title_en="Market metrics",
            title_es="Métricas de mercado",
            status_en="No clean market data is available yet.",
            status_es="No hay datos limpios disponibles todavía.",
            action_en="Start a run first, then ask for volume or volatility.",
            action_es="Ejecuta una corrida primero y luego pide volumen o volatilidad.",
            prompt_en="what is the volume of AAPL? | what is the volatility of AAPL?",
            prompt_es="cuál es el volumen de AAPL? | cuál es la volatilidad de AAPL?",
            language=language,
        )

    lowered = (message or "").lower()
    requested_metrics: list[str] = []
    if any(hint in lowered for hint in ("volume", "volumen")) and "volume" in df.columns:
        requested_metrics.append("volume")
    if any(hint in lowered for hint in ("volatility", "volatilidad")) and "volatility_5" in df.columns:
        requested_metrics.append("volatility_5")
    if not requested_metrics:
        requested_metrics = [column for column in ("volume", "volatility_5") if column in df.columns]

    selected_ticker = None
    for item in (tickers or summary.get("tickers") or ((result.get("manifest") or {}).get("request") or {}).get("tickers") or []):
        text = _normalize_metric_ticker(item)
        if text:
            selected_ticker = text
            break

    subset = df
    note = ""
    if selected_ticker and "ticker" in df.columns:
        normalized_df_ticker = df["ticker"].astype(str).map(_normalize_metric_ticker)
        filtered = df[normalized_df_ticker == selected_ticker]
        if filtered.empty:
            available = sorted({str(item).upper() for item in df["ticker"].dropna().astype(str) if str(item).strip()})
            if available:
                available_text = ", ".join(available[:8])
                note = _format_sectioned_notice(
                    "Market metrics",
                    "Métricas de mercado",
                    [
                        (
                            "Status",
                            "Estado",
                            f"{'No clean rows were found for' if not es else 'No encontré filas limpias para'} {selected_ticker}.",
                        ),
                        ("Available tickers", "Símbolos disponibles", available_text),
                        (
                            "Action",
                            "Acción",
                            "Run extraction for that asset first, or inspect the available symbols in cleaning."
                            if not es
                            else "Ejecuta extracción para ese activo primero, o revisa limpieza para ver los símbolos disponibles.",
                        ),
                    ],
                    language,
                )
            else:
                note = _format_sectioned_notice(
                    "Market metrics",
                    "Métricas de mercado",
                    [
                        (
                            "Status",
                            "Estado",
                            f"{'No clean rows were found for' if not es else 'No encontré filas limpias para'} {selected_ticker}.",
                        ),
                        (
                            "Action",
                            "Acción",
                            "Run extraction for that asset first, then come back for metrics."
                            if not es
                            else "Ejecuta extracción para ese activo primero y luego vuelve por las métricas.",
                        ),
                    ],
                    language,
                )
            compact_selected = re.sub(r"[^A-Z0-9]", "", selected_ticker)
            if len(compact_selected) <= 1:
                return (
                    f"{note}\n"
                    + (
                        "Tip: use a fuller symbol such as EURUSD=X, AAPL, or NVDA."
                        if not es
                        else "Tip: usa un símbolo más completo, por ejemplo EURUSD=X, AAPL o NVDA."
                    )
                )
            return note
        subset = filtered

    if "date" in subset.columns:
        subset = subset.sort_values("date")
    latest = subset.tail(1).iloc[0]

    lines = [
        "Market metrics" if not es else "Métricas de mercado",
    ]
    if note:
        lines.append(note)
    if selected_ticker:
        lines.append(f"{'Symbol' if not es else 'Símbolo'}: {selected_ticker}.")
    if "date" in subset.columns and not subset.tail(1).empty:
        lines.append(
            f"{'Latest date' if not es else 'Fecha más reciente'}: {_format_value(latest.get('date'))}."
        )

    metric_labels = {
        "volume": "Volume" if not es else "Volumen",
        "volatility_5": "Volatility_5" if not es else "Volatilidad_5",
    }
    for metric in requested_metrics:
        if metric not in subset.columns:
            continue
        series = pd.to_numeric(subset[metric], errors="coerce")
        latest_value = latest.get(metric)
        avg_value = None
        if series is not None:
            try:
                avg_value = float(series.dropna().mean())
            except Exception:
                avg_value = None
        metric_text = f"{metric_labels.get(metric, metric)}: {_format_market_number(metric, latest_value)}"
        if avg_value is not None and avg_value == avg_value:
            metric_text += f"; {'avg' if not es else 'promedio'}: {_format_market_number(metric, avg_value)}"
        lines.append(metric_text + ".")

    if "volume" in requested_metrics or "volatility_5" in requested_metrics:
        lines.append(
            (
                "Volume shows how much traded; volatility shows how much price moved. Higher volume usually means more activity, and higher volatility means larger swings."
                if not es
                else "El volumen muestra cuánto se negoció; la volatilidad muestra cuánto se movió el precio. Más volumen suele significar más actividad, y más volatilidad significa oscilaciones más fuertes."
            )
        )

    if summary.get("run_id"):
        lines.append(f"Run: {summary.get('run_id')}.")
    return " ".join(lines)


def _format_groq_status(enabled: bool, model: str, base_url: str, language: str = "en") -> str:
    es = _is_spanish(language)
    lines = [
        "Estado de Groq" if es else "Groq status",
        f"{'Activo' if es else 'Enabled'}: {'sí' if enabled else 'no'}.",
        f"{'Modelo' if es else 'Model'}: {model or 'n/a'}.",
        f"{'Endpoint' if es else 'Endpoint'}: {base_url or 'n/a'}.",
        (
            "Se usa para enrutamiento, resúmenes de etapa y revisión final cuando la API key está disponible."
            if es
            else "It is used for routing, stage briefs, and final review when the API key is available."
        ),
        (
            "Si Groq falla, el asistente vuelve al modo local sin romper la sesión."
            if es
            else "If Groq fails, the assistant falls back to local mode without breaking the session."
        ),
        (
            "El modo experimental Groq brain puede tomar la decisión final del run cuando se activa con `--groq-brain` o con un prompt explícito."
            if es
            else "Experimental Groq brain mode can take the final run decision when enabled with `--groq-brain` or an explicit prompt."
        ),
    ]
    return " ".join(lines)


def _format_web_status(status: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    issues = [str(item).strip() for item in (status.get("issues") or []) if str(item).strip()]
    known_presets = ", ".join(list_web_provider_presets())
    lines = [
        "Estado web" if es else "Web status",
        f"{'Activo' if es else 'Enabled'}: {('sí' if es else 'yes') if status.get('enabled') else 'no'}.",
        f"{'Configuración válida' if es else 'Config valid'}: {('sí' if es else 'yes') if status.get('config_valid') else 'no'}.",
        f"{'Listo en runtime' if es else 'Runtime ready'}: {('sí' if es else 'yes') if status.get('runtime_ready') else 'no'}.",
        f"{'Proveedor' if es else 'Provider'}: {status.get('provider') or 'custom'}.",
        (
            f"Preset conocido: {('sí' if es else 'yes') if status.get('provider_known', True) else 'no'}; "
            f"host consistente: {('sí' if es else 'yes') if status.get('provider_host_match', True) else 'no'}."
            if es
            else f"Preset known: {('yes' if status.get('provider_known', True) else 'no')}; "
            f"host consistent: {('yes' if status.get('provider_host_match', True) else 'no')}."
        ),
        f"{'Endpoint' if es else 'Endpoint'}: {status.get('search_url') or 'n/a'}.",
        f"{'Método' if es else 'Method'}: {status.get('method') or 'n/a'}.",
        (
            f"Contrato: query_param={status.get('query_param') or 'n/a'}, limit_param={status.get('limit_param') or 'n/a'}, results_path={status.get('results_path') or 'n/a'}."
            if es
            else f"Contract: query_param={status.get('query_param') or 'n/a'}, limit_param={status.get('limit_param') or 'n/a'}, results_path={status.get('results_path') or 'n/a'}."
        ),
        (
            f"Autenticación: header={status.get('auth_header') or 'n/a'}, param={status.get('auth_param') or 'n/a'}."
            if es
            else f"Auth: header={status.get('auth_header') or 'n/a'}, param={status.get('auth_param') or 'n/a'}."
        ),
        (
            f"Cache: {('sí' if es else 'yes') if status.get('cache_enabled') else 'no'}; ttl={status.get('cache_ttl_seconds') or 0}s."
            if es
            else f"Cache: {('yes' if status.get('cache_enabled') else 'no')}; ttl={status.get('cache_ttl_seconds') or 0}s."
        ),
    ]
    if issues:
        lines.append(f"{'Problemas' if es else 'Issues'}: {' | '.join(issues)}.")
        lines.append(f"{'Presets soportados' if es else 'Supported presets'}: {known_presets}.")
    else:
        lines.append(
            "El retriever está listo para complementar respuestas exploratorias y mixtas."
            if es
            else "The retriever is ready to complement exploratory and mixed answers."
        )
    lines.append(
        "Siguiente paso: corre assistant_web_probe.py o imprime un preset con --print-env para activar internet real."
        if es
        else "Next step: run assistant_web_probe.py or print a preset with --print-env to activate real web retrieval."
    )
    return " ".join(lines)


def _format_time_status(language: str = "en") -> str:
    es = _is_spanish(language)
    try:
        now = datetime.now(ZoneInfo("America/Bogota"))
    except Exception:
        now = datetime.now().astimezone()
    iso_date = now.date().isoformat()
    weekday_en = now.strftime("%A")
    weekday_es = {
        "Monday": "lunes",
        "Tuesday": "martes",
        "Wednesday": "miércoles",
        "Thursday": "jueves",
        "Friday": "viernes",
        "Saturday": "sábado",
        "Sunday": "domingo",
    }.get(weekday_en, weekday_en.lower())
    zone_label = getattr(now.tzinfo, "key", str(now.tzinfo or "local"))
    if es:
        return f"Hoy es {weekday_es} {iso_date}. Hora local: {now.strftime('%H:%M:%S')} ({zone_label})."
    return f"Today is {weekday_en} {iso_date}. Local time: {now.strftime('%H:%M:%S')} ({zone_label})."


def _format_source_scope(state: AssistantState, web_retriever: Any, language: str = "en") -> str:
    es = _is_spanish(language)
    config_status_fn = getattr(web_retriever, "config_status", None)
    status = config_status_fn().to_dict() if callable(config_status_fn) else {}
    provider = str(status.get("provider") or "n/a").strip() if isinstance(status, dict) else "n/a"
    web_ready = bool(status.get("runtime_ready")) if isinstance(status, dict) else False
    lines = [
        "Fuentes del assistant" if es else "Assistant sources",
        (
            "1. Artifacts locales: corridas, columnas extraídas, filas limpias, variables del modelo, decisiones y métricas."
            if es
            else "1. Local artifacts: runs, extracted columns, clean rows, model variables, decisions, and metrics."
        ),
        (
            "2. Herramientas locales: fecha, hora, estado de la sesión y contexto del runtime."
            if es
            else "2. Local tools: date, time, session status, and runtime context."
        ),
        (
            "3. Internet: significado, contexto externo, definiciones y contraste cuando hace falta."
            if es
            else "3. Internet: meaning, external context, definitions, and contrast when needed."
        ),
        (
            "Jerarquía: artifacts locales > herramientas locales > internet. No invento si no hay una fuente confiable."
            if es
            else "Hierarchy: local artifacts > local tools > internet. I do not invent when there is no trustworthy source."
        ),
        (
            f"Web actual: {'lista' if web_ready else 'parcial o sin contexto útil todavía'}; proveedor={provider}."
            if es
            else f"Current web status: {'ready' if web_ready else 'partial or not useful yet'}; provider={provider}."
        ),
        (
            "Puedo buscar conceptos como forex, yfinance, adj_close o contexto externo actual. "
            "Para una corrida como run_0013, la verdad sale primero de los artifacts locales."
            if es
            else "I can search concepts such as forex, yfinance, adj_close, or current external context. "
            "For a run such as run_0013, truth comes from local artifacts first."
        ),
    ]
    if state.last_run_id:
        lines.append(
            f"Contexto actual: run={state.last_run_id} | símbolo={state.current_asset or 'n/a'}."
            if es
            else f"Current context: run={state.last_run_id} | symbol={state.current_asset or 'n/a'}."
        )
    return "\n".join(lines)


def _extract_web_provider_hint(message: str) -> str:
    lowered = str(message or "").lower()
    for provider in list_web_provider_presets():
        if provider in lowered:
            return provider
    return ""


def _resolve_web_provider_hint(message: str, state: AssistantState) -> str:
    explicit = _extract_web_provider_hint(message)
    if explicit:
        return explicit
    entity_memory = state.entity_memory if isinstance(state.entity_memory, dict) else {}
    by_kind = entity_memory.get("by_kind") if isinstance(entity_memory.get("by_kind"), dict) else {}
    for key in ("web", "assistant_scorecard"):
        entry = by_kind.get(key)
        if isinstance(entry, dict):
            provider_hint = str(entry.get("provider_hint") or "").strip().lower()
            if provider_hint in list_web_provider_presets():
                return provider_hint
    last_route = AssistantRoute.from_dict(state.last_route if isinstance(state.last_route, dict) else {})
    fallback = _extract_web_provider_hint(last_route.raw_message or "")
    return fallback if fallback in list_web_provider_presets() else ""


def _wants_web_provider_catalog(message: str) -> bool:
    lowered = str(message or "").lower()
    return bool(
        re.search(r"\b(provider|providers|preset|presets|proveedor|proveedores)\b", lowered)
        and re.search(r"\b(web|internet|search|retriever|supported|support|which|what|soporta|soporta)\b", lowered)
    )


def _append_web_provider_catalog(answer: str, provider: str, message: str, language: str = "en") -> str:
    wants_catalog = _wants_web_provider_catalog(message) or not provider
    if not wants_catalog and not provider:
        return answer
    es = _is_spanish(language)
    providers = [provider] if provider else list_web_provider_presets()
    details: list[str] = []
    for item in providers:
        try:
            description = describe_web_provider(item)
        except ValueError:
            continue
        details.append(
            (
                f"{description['provider']}: method={description['method']}, auth={description['auth_mode']}({description['auth_target']}), query={description['query_param']}, limit={description['limit_param']}, results={description['results_path']}, endpoint={description['example_search_url']}"
            )
        )
    if not details:
        return answer
    label = "Catálogo de presets" if es else "Preset catalog"
    return f"{answer} {label}: {' | '.join(details)}."


def _append_web_setup_snippet(answer: str, provider: str, language: str = "en") -> str:
    if not provider:
        return answer
    try:
        lines = build_web_provider_env(provider)
        probe_command = build_web_provider_probe_command(provider)
    except ValueError:
        return answer
    es = _is_spanish(language)
    label = "Snippet de setup" if es else "Setup snippet"
    probe_label = "Comando de probe" if es else "Probe command"
    rendered = " | ".join(lines)
    return f"{answer} {label}: {rendered}. {probe_label}: {probe_command}."


def _format_assistant_scorecard(report: Dict[str, Any], language: str = "en", provider_hint: str = "") -> str:
    es = _is_spanish(language)
    runtime = dict(report.get("runtime") or {})
    local_runtime = int(report.get("local_first_runtime_ready_score_pct") or 0)
    local_impl = int(report.get("local_first_implementation_score_pct") or 0)
    hybrid_runtime = int(report.get("hybrid_runtime_ready_score_pct") or 0)
    hybrid_impl = int(report.get("hybrid_implementation_score_pct") or 0)
    hybrid_gap = max(0, 100 - hybrid_runtime)
    issues = [str(item).strip() for item in runtime.get("web_config_issues") or [] if str(item).strip()]
    runtime_gaps: list[tuple[int, str, int, int]] = []
    for raw_key, layer in (report.get("layers") or {}).items():
        layer_dict = layer if isinstance(layer, dict) else {}
        runtime_ready = int(layer_dict.get("runtime_ready_score_pct") or 0)
        implementation = int(layer_dict.get("implementation_score_pct") or 0)
        label = str(layer_dict.get("label") or raw_key).strip() or str(raw_key)
        if runtime_ready < implementation:
            runtime_gaps.append((implementation - runtime_ready, label, runtime_ready, implementation))
    runtime_gaps.sort(key=lambda item: (-item[0], item[2], item[1]))
    top_gap_text = " | ".join(
        f"{label} {runtime_ready}/{implementation}"
        for _, label, runtime_ready, implementation in runtime_gaps[:3]
    )
    lines = [
        "Scorecard del assistant" if es else "Assistant scorecard",
        (
            f"Local-first: runtime_ready={local_runtime}%, implementación={local_impl}%."
            if es
            else f"Local-first: runtime_ready={local_runtime}%, implementation={local_impl}%."
        ),
        (
            f"Híbrido: runtime_ready={hybrid_runtime}%, implementación={hybrid_impl}%."
            if es
            else f"Hybrid: runtime_ready={hybrid_runtime}%, implementation={hybrid_impl}%."
        ),
        (
            f"Runtime web: configurado={'sí' if runtime.get('web_retriever_configured') else 'no'}, "
            f"válido={'sí' if runtime.get('web_retriever_config_valid') else 'no'}, "
            f"listo={'sí' if runtime.get('web_retriever_runtime_ready') else 'no'}."
            if es
            else f"Runtime web: configured={'yes' if runtime.get('web_retriever_configured') else 'no'}, "
            f"valid={'yes' if runtime.get('web_retriever_config_valid') else 'no'}, "
            f"ready={'yes' if runtime.get('web_retriever_runtime_ready') else 'no'}."
        ),
    ]
    if hybrid_gap:
        lines.append(
            f"Brecha a 100 para runtime híbrido: {hybrid_gap}%."
            if es
            else f"Gap to 100 for hybrid runtime: {hybrid_gap}%."
        )
    else:
        lines.append(
            "El runtime híbrido ya está completamente listo; las capas web están al 100% y no queda brecha por cerrar."
            if es
            else "The hybrid runtime is already fully ready; the web layers are at 100% and there is no remaining gap to close."
        )
    if top_gap_text:
        lines.append(
            f"Brechas principales de runtime: {top_gap_text}."
            if es
            else f"Main runtime gaps: {top_gap_text}."
        )
    if issues:
        lines.append(
            f"Bloqueos: {' | '.join(issues)}."
            if es
            else f"Blocking issues: {' | '.join(issues)}."
        )
    if issues and not runtime.get("web_retriever_runtime_ready"):
        presets = list_web_provider_presets()
        recommended_provider = provider_hint if provider_hint in presets else ("tavily" if "tavily" in presets else presets[0])
        try:
            provider_env = " | ".join(build_web_provider_env(recommended_provider))
            provider_probe = build_web_provider_probe_command(recommended_provider)
        except Exception:
            provider_env = ""
            provider_probe = ""
        lines.append(
            (
                f"Preset sugerido: {recommended_provider}. Pregunta 'what env do I need for {recommended_provider}?' o usa: {provider_env}."
                if es
                else f"Suggested preset: {recommended_provider}. Ask 'what env do I need for {recommended_provider}?' or use: {provider_env}."
            )
        )
        if provider_probe:
            lines.append(
                f"Comando de probe sugerido: {provider_probe}."
                if es
                else f"Suggested probe command: {provider_probe}."
            )
    lines.append(
        "Siguiente paso: pregunta estado web para ver el setup del proveedor o corre assistant_web_probe.py."
        if es
        else "Next step: ask web status for provider setup details or run assistant_web_probe.py."
    )
    return " ".join(lines)


def _wants_scorecard_layer_breakdown(message: str) -> bool:
    lowered = str(message or "").lower()
    return bool(re.search(r"\b(layer|layers|capa|capas)\b", lowered))


def _append_scorecard_layer_breakdown(answer: str, report: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    details: list[str] = []
    for key, layer in (report.get("layers") or {}).items():
        layer_dict = layer if isinstance(layer, dict) else {}
        label = str(layer_dict.get("label") or key).strip() or str(key)
        runtime_ready = int(layer_dict.get("runtime_ready_score_pct") or 0)
        implementation = int(layer_dict.get("implementation_score_pct") or 0)
        details.append(f"{label} {runtime_ready}/{implementation}")
    if not details:
        return answer
    label = "Capas" if es else "Layers"
    return f"{answer} {label}: {' | '.join(details)}."


def _wants_scorecard_web_impact(message: str) -> bool:
    lowered = str(message or "").lower()
    return bool(
        re.search(r"\b(web|internet|retriever)\b", lowered)
        and re.search(
            r"\b(change|changes|impact|improve|improves|improvement|if i enable|if i configure|si configuro|si activo|qué cambia|que cambia|impacto)\b",
            lowered,
        )
    )


def _append_scorecard_web_impact(answer: str, report: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    layers = report.get("layers") or {}
    affected: list[str] = []
    for key in ("web_retrieval_layer", "web_fact_extraction", "mixed_grounding_engine"):
        layer = layers.get(key) or {}
        runtime_ready = int(layer.get("runtime_ready_score_pct") or 0)
        implementation = int(layer.get("implementation_score_pct") or 0)
        label = str(layer.get("label") or key).strip() or str(key)
        if runtime_ready < implementation:
            affected.append(f"{label} {runtime_ready}->{implementation}")
    hybrid_runtime = int(report.get("hybrid_runtime_ready_score_pct") or 0)
    hybrid_impl = int(report.get("hybrid_implementation_score_pct") or 0)
    if not affected and hybrid_runtime >= hybrid_impl:
        return (
            f"{answer} "
            + (
                "El runtime híbrido ya está completamente listo; las capas web están al 100% y no queda brecha por cerrar."
                if es
                else "The hybrid runtime is already fully ready; the web layers are at 100% and there is no remaining gap to close."
            )
        )
    impact_line = (
        f"Si configuras web real, el runtime híbrido puede pasar de {hybrid_runtime}% a {hybrid_impl}%."
        if es
        else f"If you configure real web retrieval, hybrid runtime can move from {hybrid_runtime}% to {hybrid_impl}%."
    )
    if affected:
        impact_line += (
            f" Capas afectadas: {' | '.join(affected)}."
            if es
            else f" Affected layers: {' | '.join(affected)}."
        )
    return f"{answer} {impact_line}"


def _wants_scorecard_web_impact(message: str) -> bool:
    lowered = str(message or "").lower()
    return bool(
        re.search(r"\b(web|internet|retriever)\b", lowered)
        and re.search(
            r"\b(change|changes|impact|improve|improves|improvement|if i enable|if i configure|si configuro|si activo|qué cambia|que cambia|impacto)\b",
            lowered,
        )
    )


def _append_scorecard_web_impact(answer: str, report: Dict[str, Any], language: str = "en") -> str:
    es = _is_spanish(language)
    layers = report.get("layers") or {}
    affected = []
    for key in ("web_retrieval_layer", "web_fact_extraction", "mixed_grounding_engine"):
        layer = layers.get(key) or {}
        runtime_ready = int(layer.get("runtime_ready_score_pct") or 0)
        implementation = int(layer.get("implementation_score_pct") or 0)
        label = str(layer.get("label") or key).strip() or str(key)
        if runtime_ready < implementation:
            affected.append(f"{label} {runtime_ready}->{implementation}")
    hybrid_runtime = int(report.get("hybrid_runtime_ready_score_pct") or 0)
    hybrid_impl = int(report.get("hybrid_implementation_score_pct") or 0)
    if not affected and hybrid_runtime >= hybrid_impl:
        return (
            f"{answer} "
            + (
                "El runtime híbrido ya está completamente listo; las capas web están al 100% y no queda brecha por cerrar."
                if es
                else "The hybrid runtime is already fully ready; the web layers are at 100% and there is no remaining gap to close."
            )
        )
    impact_line = (
        f"Si configuras web real, el runtime híbrido puede pasar de {hybrid_runtime}% a {hybrid_impl}%."
        if es
        else f"If you configure real web retrieval, hybrid runtime can move from {hybrid_runtime}% to {hybrid_impl}%."
    )
    if affected:
        details = " | ".join(affected)
        impact_line += (
            f" Capas afectadas: {details}."
            if es
            else f" Affected layers: {details}."
        )
    return f"{answer} {impact_line}"


def _format_stage_handoff_hint(stage: str, language: str = "en") -> str:
    es = _is_spanish(language)
    stage_key = stage if stage in {"extraction", "cleaning", "modeling", "orchestrator"} else "orchestrator"
    hints = {
        "extraction": (
            "Siguiente paso: limpieza. Si faltan filas, símbolo o rango, vuelve a la extracción."
            if es
            else "Next agent: cleaning. If rows, symbol, or date range are wrong, go back to extraction."
        ),
        "cleaning": (
            "Siguiente paso: modelado. Si la fila exacta no existe, vuelve a la extracción."
            if es
            else "Next agent: modeling. If the exact row is missing, go back to extraction."
        ),
        "modeling": (
            "Siguiente paso: orquestador. Si quieres la verdad final o comparar modos, consulta el orquestador."
            if es
            else "Next agent: orchestrator. If you want the final truth or to compare modes, consult the orchestrator."
        ),
        "orchestrator": (
            "Siguiente paso: vuelve a extracción, limpieza o modelado según el error o la duda."
            if es
            else "Next step: return to extraction, cleaning, or modeling depending on the issue."
        ),
    }
    return hints[stage_key]


def _suggest_session_next_agent(summary: Dict[str, Any], state: AssistantState, language: str = "en") -> str:
    es = _is_spanish(language)
    brain = summary.get("brain") or {}
    comparison = summary.get("comparison") or {}
    last_intent = str(state.last_intent or "").strip()
    run_mode = str(summary.get("run_mode") or state.current_mode or "").strip()

    if not summary or not summary.get("run_id"):
        return "extracción" if es else "extraction"

    if brain.get("used") or "groq_brain" in run_mode:
        return "orquestador" if es else "orchestrator"

    if last_intent in {"show_extraction", "show_asset_used", "show_clean_data"}:
        return "limpieza" if es else "cleaning"
    if last_intent in {"show_cleaning", "show_market_metrics"}:
        return "modelado" if es else "modeling"
    if last_intent in {"show_prediction", "show_model_variables", "show_decision_explanation", "show_stage_brief"}:
        return "orquestador" if es else "orchestrator"

    if comparison.get("extraction_health") == "needs_review":
        return "extracción" if es else "extraction"
    if comparison.get("modeling_health") == "mixed":
        return "modeling" if not es else "modelado"

    return "limpieza" if es else "cleaning"


def _normalize_context_stage(intent: str, stage: str = "") -> str:
    intent = str(intent or "").strip().lower()
    stage = str(stage or "").strip().lower()
    if stage in {
        "extraction",
        "cleaning",
        "modeling",
        "orchestrator",
        "semantic_lookup",
        "clean_data",
        "metrics",
        "symbols",
        "asset",
        "decision",
        "summary",
        "comparison",
        "assistant_scorecard",
        "legacy",
        "groq_status",
        "web_status",
        "session_status",
        "session_status_full",
        "mode_guide",
        "agent_guide",
        "continue",
        "help",
        "language",
        "agent_card",
    }:
        return stage
    intent_map = {
        "run_pipeline": "orchestrator",
        "compare_sources": "comparison",
        "show_latest_summary": "summary",
        "show_semantic_lookup": "semantic_lookup",
        "show_assistant_scorecard": "assistant_scorecard",
        "show_extraction": "extraction",
        "show_cleaning": "cleaning",
        "show_clean_data": "clean_data",
        "show_market_metrics": "metrics",
        "show_prediction": "modeling",
        "show_model_variables": "model_variables",
        "show_legacy_status": "legacy",
        "show_symbol_guide": "symbols",
        "show_asset_used": "asset",
        "show_decision_explanation": "decision",
        "show_groq_status": "groq_status",
        "show_web_status": "web_status",
        "show_session_status": "session_status",
        "show_session_status_full": "session_status_full",
        "show_mode_guide": "mode_guide",
        "show_agent_guide": "agent_guide",
        "continue_task": "continue",
        "help": "help",
        "set_language": "language",
    }
    if intent == "show_agent_card":
        return stage if stage in {"extraction", "cleaning", "modeling"} else "agent_guide"
    if intent == "show_stage_brief":
        return stage if stage in {"extraction", "cleaning", "modeling", "orchestrator", "motor"} else "orchestrator"
    return intent_map.get(intent, "")


def _latest_meaningful_note(state: AssistantState) -> tuple[str, str]:
    known_stages = {
        "extraction",
        "cleaning",
        "modeling",
        "orchestrator",
        "semantic_lookup",
        "clean_data",
        "metrics",
        "symbols",
        "asset",
        "decision",
        "summary",
        "comparison",
        "assistant_scorecard",
        "legacy",
        "groq_status",
        "web_status",
        "session_status",
        "session_status_full",
        "mode_guide",
        "agent_guide",
        "continue",
        "help",
        "language",
        "agent_card",
    }
    for raw_note in reversed(state.notes or []):
        note = str(raw_note or "").strip()
        if not note or note.startswith("language:"):
            continue
        parts = [part.strip() for part in note.split(":") if part.strip()]
        if not parts:
            continue
        intent = parts[0].lower()
        if intent == "help":
            continue
        stage = ""
        if len(parts) >= 3 and parts[1].lower() in known_stages:
            stage = parts[1].lower()
        return intent, stage
    return "", ""


def _session_context_snapshot(
    summary: Dict[str, Any],
    state: AssistantState,
    language: str = "en",
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    summary = summary if isinstance(summary, dict) else {}
    result = result if isinstance(result, dict) else {}
    error = error if isinstance(error, dict) else {}
    details = error.get("details") if isinstance(error.get("details"), dict) else {}

    run_id = (
        summary.get("run_id")
        or result.get("run_id")
        or details.get("run_id")
        or state.last_run_id
        or "n/a"
    )

    tickers: list[str] = []
    ticker_sources: tuple[Any, ...] = (
        summary.get("tickers"),
        details.get("tickers"),
        ((result.get("manifest") or {}).get("request") or {}).get("tickers"),
        state.last_request.get("tickers") if isinstance(state.last_request, dict) else [],
    )
    for source in ticker_sources:
        if isinstance(source, list):
            tickers = [str(item).strip().upper() for item in source if str(item).strip()]
        elif isinstance(source, str) and source.strip():
            tickers = [source.strip().upper()]
        if tickers:
            break
    if not tickers and state.current_asset:
        tickers = [str(state.current_asset).strip().upper()]
    symbol = tickers[0] if tickers else "n/a"
    if len(tickers) > 1:
        symbol = f"{tickers[0]} +{len(tickers) - 1}"

    note_intent, note_stage = _latest_meaningful_note(state)
    route = state.last_route if isinstance(state.last_route, dict) else {}
    route_intent = str(route.get("intent") or "").strip().lower()
    route_stage = str(route.get("stage") or "").strip().lower()
    context_stage = note_stage or _normalize_context_stage(note_intent, "")
    if not context_stage and error and not summary and not result:
        context_stage = str(error.get("stage") or "").strip().lower()
    if not context_stage:
        context_stage = _normalize_context_stage(route_intent, route_stage)
    if not context_stage and (summary.get("run_id") or result or error):
        context_stage = "orchestrator"

    stage_labels = {
        "en": {
            "extraction": "extraction",
            "cleaning": "cleaning",
            "modeling": "modeling",
            "orchestrator": "orchestrator",
            "semantic_lookup": "concepts",
            "clean_data": "clean data",
            "metrics": "metrics",
            "symbols": "symbols",
            "asset": "asset",
            "decision": "decision",
            "summary": "summary",
            "comparison": "comparison",
            "assistant_scorecard": "assistant scorecard",
            "legacy": "legacy",
            "groq_status": "groq status",
            "web_status": "web status",
            "session_status": "status",
            "session_status_full": "full status",
            "turn_trace": "trace",
            "mode_guide": "modes",
            "agent_guide": "agents",
            "continue": "continue",
            "help": "help",
            "language": "language",
            "agent_card": "agent detail",
        },
        "es": {
            "extraction": "extracción",
            "cleaning": "limpieza",
            "modeling": "modelado",
            "orchestrator": "orquestador",
            "semantic_lookup": "conceptos",
            "clean_data": "datos limpios",
            "metrics": "métricas",
            "symbols": "símbolos",
            "asset": "símbolo",
            "decision": "decisión",
            "summary": "resumen",
            "comparison": "comparación",
            "assistant_scorecard": "scorecard del assistant",
            "legacy": "legacy",
            "groq_status": "estado de groq",
            "web_status": "estado web",
            "session_status": "estado",
            "session_status_full": "estado completo",
            "turn_trace": "traza",
            "mode_guide": "modos",
            "agent_guide": "agentes",
            "continue": "continuar",
            "help": "ayuda",
            "language": "idioma",
            "agent_card": "detalle del agente",
        },
    }
    stage_label = stage_labels["es" if _is_spanish(language) else "en"].get(context_stage, context_stage or "n/a")
    return {
        "run_id": str(run_id),
        "symbol": symbol,
        "stage_key": context_stage or "n/a",
        "stage": stage_label or "n/a",
    }


def _format_session_context_line(
    summary: Dict[str, Any],
    state: AssistantState,
    language: str = "en",
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> str:
    es = _is_spanish(language)
    snapshot = _session_context_snapshot(summary, state, language, result=result, error=error)
    return (
        f"{'Contexto actual' if es else 'Current context'}: "
        f"run={snapshot['run_id']} | "
        f"{'símbolo' if es else 'symbol'}={snapshot['symbol']} | "
        f"{'etapa' if es else 'stage'}={snapshot['stage']}."
    )


def _format_last_turn_trace(state: AssistantState, language: str = "en") -> str:
    trace = state.last_turn_trace if isinstance(state.last_turn_trace, dict) else {}
    if not trace:
        return ""
    es = _is_spanish(language)
    act = str(trace.get("act") or "general").strip()
    subject = str(trace.get("subject") or "n/a").strip() or "n/a"
    final_intent = str(trace.get("final_intent") or trace.get("route_intent") or "unknown").strip() or "unknown"
    source_mode = str(trace.get("source_mode") or "local").strip() or "local"
    source_policy = str(trace.get("source_policy") or "n/a").strip() or "n/a"
    risk = str(trace.get("planner_risk") or "low").strip() or "low"
    planner_steps = int(trace.get("planner_steps") or 0)
    allow_run = bool(trace.get("allow_run", True))
    override_memory = bool(trace.get("override_memory", False))
    web_required = bool(trace.get("web_required", False))
    return (
        (
            f"Traza conversacional: acto={act}; sujeto={subject}; intención={final_intent}; "
            f"fuente={source_mode}; política={source_policy}; web obligatoria={'sí' if web_required else 'no'}."
            if es
            else f"Conversational trace: act={act}; subject={subject}; intent={final_intent}; "
            f"source={source_mode}; policy={source_policy}; web_required={'yes' if web_required else 'no'}."
        )
        + " "
        + (
            f"Planner: riesgo={risk}; pasos={planner_steps}; allow_run={'sí' if allow_run else 'no'}; "
            f"override_memory={'sí' if override_memory else 'no'}."
            if es
            else f"Planner: risk={risk}; steps={planner_steps}; allow_run={'yes' if allow_run else 'no'}; "
            f"override_memory={'yes' if override_memory else 'no'}."
        )
    ).strip()


def _format_turn_trace_panel(state: AssistantState, language: str = "en", *, detailed: bool = False) -> str:
    trace_payload = state.last_turn_trace if isinstance(state.last_turn_trace, dict) else {}
    trace = _format_last_turn_trace(state, language)
    if trace:
        es = _is_spanish(language)
        title = (
            "Traza completa del turno"
            if detailed and es
            else ("Turn trace full" if detailed else ("Traza del turno" if es else "Turn trace"))
        )
        raw = str(trace_payload.get("raw_message") or "").strip()
        if not detailed:
            if raw:
                prompt_line = f"{'Último mensaje' if es else 'Last message'}: {raw}."
                return f"{title}\n{prompt_line}\n{trace}"
            return f"{title}\n{trace}"
        lines = [title]
        if raw:
            lines.append(f"{'Último mensaje' if es else 'Last message'}: {raw}.")
        canonical_query = str(trace_payload.get("canonical_query") or "").strip() or "n/a"
        route_intent = str(trace_payload.get("route_intent") or "").strip() or "n/a"
        final_intent = str(trace_payload.get("final_intent") or "unknown").strip() or "unknown"
        stage = str(trace_payload.get("stage") or "").strip() or "n/a"
        focus = str(trace_payload.get("question_focus") or "").strip() or "n/a"
        run_id = str(trace_payload.get("run_id") or "").strip() or "n/a"
        secondary_run_id = str(trace_payload.get("secondary_run_id") or "").strip() or "n/a"
        tickers = trace_payload.get("tickers") if isinstance(trace_payload.get("tickers"), list) else []
        planner_explanation = str(trace_payload.get("planner_explanation") or "").strip()
        lines.append(trace)
        lines.append(
            (
                f"Resolución: canonical_query={canonical_query}; intención_inicial={route_intent}; intención_final={final_intent}; etapa={stage}; foco={focus}."
                if es
                else f"Resolution: canonical_query={canonical_query}; initial_intent={route_intent}; final_intent={final_intent}; stage={stage}; focus={focus}."
            )
        )
        lines.append(
            (
                f"Scope: run_id={run_id}; secondary_run_id={secondary_run_id}; tickers={', '.join(str(item) for item in tickers) if tickers else 'n/a'}."
                if not es
                else f"Scope: run_id={run_id}; secondary_run_id={secondary_run_id}; tickers={', '.join(str(item) for item in tickers) if tickers else 'n/a'}."
            )
        )
        if planner_explanation:
            lines.append(
                (
                    f"Explicación del planner: {planner_explanation}"
                    if es
                    else f"Planner explanation: {planner_explanation}"
                )
            )
        return "\n".join(lines)
    return _format_availability_notice(
        title_en="Turn trace",
        title_es="Traza del turno",
        status_en="No turn trace is available yet.",
        status_es="Todavía no hay una traza de turno disponible.",
        action_en="Ask one question first, then request trace.",
        action_es="Haz una pregunta primero y luego pide trace o traza.",
        prompt_en="trace | trace full | how did you interpret that?",
        prompt_es="trace | traza | traza completa | cómo interpretaste eso?",
        language=language,
    )


def _contextual_help_prompts(
    summary: Dict[str, Any],
    state: AssistantState,
    language: str = "en",
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> list[str]:
    es = _is_spanish(language)
    snapshot = _session_context_snapshot(summary, state, language, result=result, error=error)
    stage_key = snapshot["stage_key"]
    symbol = snapshot["symbol"]
    has_symbol = symbol != "n/a"
    symbol_text_es = f" de {symbol}" if has_symbol else ""
    symbol_text_en = f" for {symbol}" if has_symbol else ""
    decision = str((summary.get("models") or {}).get("final_decision") or "").strip()
    if stage_key in {"cleaning", "clean_data", "metrics"}:
        if es:
            return [
                f"qué métricas hay en los datos limpios{symbol_text_es}?",
                f"qué dice esta fila limpia{symbol_text_es}?",
                f"muestra la fila limpia{symbol_text_es}",
            ]
        return [
            f"what metrics are in the cleaned data{symbol_text_en}?",
            "what does this clean row say?",
            f"show me the clean row{symbol_text_en}",
        ]
    if stage_key == "extraction":
        if es:
            return [
                f"qué columnas se extrajeron{symbol_text_es}?",
                "qué símbolo se usó?",
                "qué pasó en la extracción?",
            ]
        return [
            f"what columns were extracted{symbol_text_en}?",
            "what symbol was used?",
            "what happened in extraction?",
        ]
    if stage_key in {"modeling", "model_variables"}:
        if es:
            return [
                f"qué predijo el modelo{symbol_text_es}?",
                "qué variables usó el modelo?",
                "cómo afecta la limpieza a long short hold?",
            ]
        return [
            f"what did the model predict{symbol_text_en}?",
            "what variables did the model use?",
            "how does cleaning affect long short hold?",
        ]
    if stage_key in {"orchestrator", "decision", "summary", "comparison", "legacy"}:
        if es:
            first = f"por qué decidió {decision}?" if decision else "por qué decidió eso?"
            return [
                first,
                "qué fue lo que decidió el orquestador?",
                "qué significa long short hold?",
            ]
        first = f"why did it decide {decision}?" if decision else "why did it decide that way?"
        return [
            first,
            "what was the final decision?",
            "what does long short hold mean?",
        ]
    if stage_key == "semantic_lookup":
        return _semantic_lookup_help_prompts(state, language)
    if stage_key == "assistant_scorecard":
        if es:
            return [
                "assistant scorecard layers",
                "estado web",
                "qué cambia si activo web?",
            ]
        return [
            "assistant scorecard layers",
            "web status",
            "what changes if I enable web?",
        ]
    if stage_key == "symbols":
        if es:
            return [
                "qué símbolos puedo usar?",
                "cómo uso AAPL, BTC-USD o EURUSD=X?",
                "qué compara compare-binance?",
            ]
        return [
            "what symbols can I use?",
            "how do I use AAPL, BTC-USD, or EURUSD=X?",
            "what does compare-binance compare?",
        ]
    if stage_key == "mode_guide":
        if es:
            return [
                "compare-binance ETH-USD 2024-01-01 2025-01-01",
                "--groq-brain",
                "local_only",
            ]
        return [
            "compare-binance ETH-USD 2024-01-01 2025-01-01",
            "--groq-brain",
            "local_only",
        ]
    if stage_key in {"session_status", "session_status_full"}:
        if es:
            return [
                "estado",
                "estado completo",
                "cuál es el modo activo?",
            ]
        return [
            "status",
            "status full",
            "what mode am I in?",
        ]
    if stage_key == "agent_guide":
        if es:
            return [
                "extracción",
                "limpieza",
                "modelado",
            ]
        return [
            "extraction",
            "cleaning",
            "modeling",
        ]
    if stage_key == "continue":
        if es:
            return [
                "continuar",
                "estado",
                "última corrida",
            ]
        return [
            "continue",
            "status",
            "latest run",
        ]
    if stage_key == "groq_status":
        if es:
            return [
                "estado de groq",
                "url de groq",
                "modelo de groq",
            ]
        return [
            "groq status",
            "groq url",
            "groq model",
        ]
    if stage_key == "web_status":
        if es:
            return [
                "estado del retriever",
                "config de web",
                "internet activo",
            ]
        return [
            "web status",
            "web retriever status",
            "is internet retrieval active?",
        ]
    if es:
        return [
            "qué columnas se extrajeron?",
            "qué dice esta fila limpia?",
            "qué predijo el modelo?",
        ]
    return [
        "what columns were extracted?",
        "what does this clean row say?",
        "what did the model predict?",
    ]


def _humanize_task_reference(task: str, language: str = "en") -> str:
    es = _is_spanish(language)
    normalized = " ".join(str(task or "").split()).lower()
    mapping = {
        "pregunta extracción": "extracción" if es else "extraction",
        "ask extraction": "extraction",
        "pregunta limpieza": "limpieza" if es else "cleaning",
        "ask cleaning": "cleaning",
        "pregunta modelado": "modelado" if es else "modeling",
        "ask modeling": "modeling",
        "pregunta al orquestador": "orquestador" if es else "orchestrator",
        "ask orchestrator": "orchestrator",
    }
    return mapping.get(normalized, task)


def _describe_run_mode(mode: str, language: str = "en") -> str:
    es = _is_spanish(language)
    value = str(mode or "local_only").strip() or "local_only"
    labels = {
        "local_only": "determinista" if es else "deterministic",
        "local_plus_reviewer": "determinista + revisor" if es else "deterministic + reviewer",
        "local_plus_binance": "determinista + compare-binance" if es else "deterministic + compare-binance",
        "local_plus_binance_legacy": "determinista + compare-binance + legacy" if es else "deterministic + compare-binance + legacy",
        "local_only_groq_brain": "Groq brain experimental" if es else "Groq brain experimental",
        "local_plus_binance_groq_brain": "Groq brain experimental + compare-binance" if es else "Groq brain experimental + compare-binance",
        "local_plus_binance_legacy_groq_brain": "Groq brain experimental + compare-binance + legacy"
        if es
        else "Groq brain experimental + compare-binance + legacy",
    }
    return labels.get(value, value)


def _format_session_status(
    summary: Dict[str, Any],
    state: AssistantState,
    language: str = "en",
    groq_available: bool = False,
    detailed: bool = False,
) -> str:
    es = _is_spanish(language)
    models = summary.get("models") or {}
    brain = summary.get("brain") or {}
    comparison = summary.get("comparison") or {}
    source_comparison = summary.get("source_comparison") or {}
    legacy = summary.get("legacy_analysis") or {}
    selection = summary.get("selection") or {}
    current_mode = summary.get("run_mode") or state.current_mode or "local_only"
    readable_mode = _describe_run_mode(current_mode, language)
    run_id = summary.get("run_id") or state.last_run_id or "n/a"
    tickers = summary.get("tickers") or []
    final_decision = models.get("final_decision") or "n/a"
    final_confidence = float(models.get("confidence", 0.0) or 0.0)
    deterministic_decision = models.get("deterministic_decision") or "n/a"
    deterministic_confidence = float(models.get("deterministic_confidence", 0.0) or 0.0)
    decision_source = models.get("decision_source") or brain.get("decision_source") or "n/a"
    context_snapshot = _session_context_snapshot(summary, state, language)
    include_context_line = any(context_snapshot[key] != "n/a" for key in ("run_id", "symbol", "stage_key"))
    context_line = (
        _format_session_context_line(summary, state, language)
        if include_context_line
        else ""
    )
    if not detailed:
        groq_text = (
            f"{'Groq' if es else 'Groq'}: {'sí' if groq_available else 'no'}; "
            f"{'brain usado' if es else 'brain used'}: {'sí' if brain.get('used') else 'no'}."
            if es
            else f"Groq: {'yes' if groq_available else 'no'}; brain used: {'yes' if brain.get('used') else 'no'}."
        )
        confidence_text = (
            f"{'Confianza' if es else 'Confidence'}: {final_decision} {final_confidence:.4f}; "
            f"{'base' if es else 'baseline'} {deterministic_decision} {deterministic_confidence:.4f}; "
            f"{'fuente' if es else 'source'} {decision_source}."
        )
        compact_parts = [
            "Estado de sesión" if es else "Session status",
            f"{'Modo' if es else 'Mode'}: {readable_mode} ({current_mode}).",
        ]
        if context_line:
            compact_parts.append(context_line)
        compact_parts.extend(
            [
                groq_text,
                confidence_text,
                f"{'Siguiente agente' if es else 'Next agent'}: {_suggest_session_next_agent(summary, state, language)}.",
            ]
        )
        return " ".join(compact_parts)

    lines = [
        "Estado de sesión completo" if es else "Session status full",
    ]
    if context_line:
        lines.append(context_line)
    lines.extend(
        [
            (
                f"{'Resumen' if es else 'Overview'}: "
                f"{'modo' if es else 'mode'}={readable_mode} ({current_mode}); "
                f"run={run_id}; "
                f"{'idioma' if es else 'language'}={language}; "
                f"{'activos' if es else 'tickers'}={', '.join(tickers) if tickers else (state.current_asset or 'n/a')}."
            ),
            (
                f"{'Estado de Groq' if es else 'Groq state'}: "
                f"{'disponible' if es else 'available'}={'sí' if groq_available else 'no'}; "
                f"{'brain habilitado' if es else 'brain enabled'}={'sí' if brain.get('enabled') else 'no'}; "
                f"{'brain usado' if es else 'brain used'}={'sí' if brain.get('used') else 'no'}."
            ),
            (
                f"{'Decisión' if es else 'Decision'}: "
                f"{final_decision} ({final_confidence:.4f}); "
                f"{'base' if es else 'baseline'} {deterministic_decision} ({deterministic_confidence:.4f}); "
                f"{'fuente' if es else 'source'} {decision_source}."
            ),
        ]
    )
    if summary.get("review_mode") is not None or models.get("reviewer_used") is not None:
        lines.append(
            (
                f"{'Revisor' if es else 'Reviewer'}: {summary.get('review_mode', 'auto')} "
                f"({'usado' if es else 'used'}: {'sí' if models.get('reviewer_used') else 'no'})."
                if es
                else f"Reviewer: {summary.get('review_mode', 'auto')} (used: {'yes' if models.get('reviewer_used') else 'no'})."
            )
        )
    if comparison.get("decision_path"):
        lines.append(
            f"{'Ruta de decisión' if es else 'Decision path'}: {comparison.get('decision_path')}."
        )
    if selection.get("strategy"):
        lines.append(
            (
                f"{'Selección' if es else 'Selection'}: {selection.get('strategy')} "
                f"{'porque' if es else 'because'} {_compact_reason_text(selection.get('reason', 'n/a'))}."
            )
        )
    if comparison.get("extraction_health") or comparison.get("modeling_health"):
        lines.append(
            f"{'Salud' if es else 'Health'}: extraction={comparison.get('extraction_health', 'n/a')} "
            f"modeling={comparison.get('modeling_health', 'n/a')}."
        )
    if models.get("disagreement") is not None:
        reason = str(models.get("disagreement_reason") or "").strip()
        lines.append(
            (
                f"{'Los modelos discreparon' if es else 'Models disagreed'}: {models.get('disagreement')}."
                + (f" {reason}" if reason else "")
            ).strip()
        )
    if source_comparison:
        if source_comparison.get("enabled", False):
            lines.append(
                (
                    f"{'Comparación Binance' if es else 'Binance comparison'}: sí "
                    f"({source_comparison.get('source_1', 'n/a')} vs {source_comparison.get('source_2', 'n/a')})."
                )
            )
        else:
            note = source_comparison.get("note") or source_comparison.get("error")
            if note:
                lines.append(
                    f"{'Comparación Binance' if es else 'Binance comparison'}: {note}."
                )
    turn_trace = _format_last_turn_trace(state, language)
    if turn_trace:
        lines.append(turn_trace)
    if legacy:
        if legacy.get("enabled", False):
            lines.append(
                (
                    f"{'Puente legacy' if es else 'Legacy bridge'}: sí "
                    f"({legacy.get('asset') or legacy.get('requested_asset') or 'n/a'})."
                )
            )
        else:
            note = legacy.get("error") or legacy.get("bridge_note") or "n/a"
            lines.append(
                f"{'Puente legacy' if es else 'Legacy bridge'}: {note}."
            )
    lines.append(
        (
            f"{'Navegación' if es else 'Navigation'}: "
            f"{'siguiente agente' if es else 'next agent'}={_suggest_session_next_agent(summary, state, language)}; "
            f"{'tarea pendiente' if es else 'pending task'}={_humanize_task_reference(state.pending_task, language) if state.pending_task else 'n/a'}."
        )
    )
    return "\n".join(lines)


def _format_mode_detail(mode_key: str, language: str = "en") -> str:
    es = _is_spanish(language)
    key = re.sub(r"[\s\-\+]+", "_", (mode_key or "").strip().lower()).strip("_")
    display = {
        "local_only": "local_only",
        "local_plus_reviewer": "local_plus_reviewer",
        "local_plus_binance": "local_plus_binance",
        "local_plus_binance_legacy": "local_plus_binance_legacy",
        "local_only_groq_brain": "local_only_groq_brain",
        "local_plus_binance_groq_brain": "local_plus_binance_groq_brain",
        "local_plus_binance_legacy_groq_brain": "local_plus_binance_legacy_groq_brain",
        "compare_binance": "compare-binance",
        "groq_brain": "--groq-brain",
        "compare_binance_groq_brain": "compare-binance + --groq-brain",
    }.get(key, key or "local_only")

    def _label(en_text: str, es_text: str) -> str:
        return es_text if es else en_text

    def _section(en_text: str, es_text: str, value: str) -> str:
        return f"{_label(en_text, es_text)}: {value}"

    summary = ""
    activation = ""
    usage = ""
    flow = ""
    agent_flow = ""
    examples = ""
    prompts = ""

    if key == "local_only":
        summary = _label("Default deterministic local mode.", "Modo local determinista por defecto.")
        activation = _label("Write `local_only` or a ticker like `AAPL`.", "Escribe `local_only` o un ticker como `AAPL`.")
        usage = _label(
            "Use it when you only want the local baseline, without comparison or experimental brain.",
            "Úsalo si solo quieres la ruta local base, sin comparación ni cerebro experimental.",
        )
        flow = _label(
            "Uses only the local baseline without comparison or experimental brain.",
            "Solo usa la ruta local base sin comparación ni cerebro experimental.",
        )
        examples = "local_only | AAPL 2024-01-01 2025-01-01"
        prompts = _label(
            "what mode is active? | how do I activate it? | what changes if I compare with Binance?",
            "qué modo está activo? | cómo lo activo? | qué cambia si comparo con Binance?",
        )
    elif key == "local_plus_reviewer":
        summary = _label(
            "Deterministic mode with an optional reviewer when the result needs a second look.",
            "Modo determinista con revisor opcional cuando el resultado necesita segunda mirada.",
        )
        activation = _label("Write `local_plus_reviewer`.", "Escribe `local_plus_reviewer`.")
        usage = _label(
            "Use it when you want a more conservative local readout.",
            "Úsalo cuando quieres una lectura local más conservadora.",
        )
        flow = _label(
            "Can add a reviewer before the decision is finalized.",
            "Puede añadir un revisor antes de cerrar la decisión.",
        )
        examples = "local_plus_reviewer | AAPL 2024-01-01 2025-01-01"
        prompts = _label(
            "when does the reviewer step in? | why did the reviewer change the decision? | how do I activate it?",
            "cuándo entra el revisor? | por qué el revisor cambió la decisión? | cómo se activa?",
        )
    elif key == "local_plus_binance":
        summary = _label(
            "Deterministic mode with yfinance-to-Binance comparison.",
            "Modo determinista con comparación de yfinance contra Binance.",
        )
        activation = _label(
            "Write `local_plus_binance` or `compare-binance ETH-USD 2024-01-01 2025-01-01`.",
            "Escribe `local_plus_binance` o `compare-binance ETH-USD 2024-01-01 2025-01-01`.",
        )
        usage = _label(
            "Use it when you want to check whether the signal changes with the source.",
            "Úsalo cuando quieras revisar si la señal cambia según la fuente.",
        )
        flow = _label(
            "Adds a comparison branch between yfinance and Binance.",
            "Añade una rama de comparación entre yfinance y Binance.",
        )
        agent_flow = _label(
            "Extraction pulls yfinance, Cleaning prepares the cleaned rows, Modeling compares yfinance vs Binance, and Orchestrator closes the run.",
            "Extracción toma yfinance, Limpieza prepara las filas limpias, Modelado compara yfinance vs Binance y el Orquestador cierra la corrida.",
        )
        examples = "local_plus_binance | compare-binance ETH-USD 2024-01-01 2025-01-01"
        prompts = _label(
            "how do I compare with Binance? | what changes in the signal? | which asset am I comparing?",
            "cómo comparo con Binance? | qué cambia en la señal? | qué activo estás comparando?",
        )
    elif key == "local_plus_binance_legacy":
        summary = _label(
            "Deterministic mode with Binance comparison and the legacy bridge for BTC/ETH/LTC.",
            "Modo determinista con comparación Binance y el puente legacy para BTC/ETH/LTC.",
        )
        activation = _label(
            "Write `local_plus_binance_legacy` or `compare-binance BTC-USD 2024-01-01 2025-01-01`.",
            "Escribe `local_plus_binance_legacy` o `compare-binance BTC-USD 2024-01-01 2025-01-01`.",
        )
        usage = _label(
            "Use it when comparing crypto assets that still use the legacy mapping.",
            "Úsalo si comparas activos cripto que tienen mapeo antiguo.",
        )
        flow = _label(
            "Adds Binance comparison plus the older crypto mapping.",
            "Agrega la comparación Binance y el mapeo antiguo para cripto.",
        )
        agent_flow = _label(
            "Extraction pulls crypto history, Cleaning prepares the cleaned rows, Modeling compares the source paths, and Orchestrator closes the run.",
            "Extracción toma el historial cripto, Limpieza prepara las filas limpias, Modelado compara las rutas de fuente y el Orquestador cierra la corrida.",
        )
        examples = "local_plus_binance_legacy | compare-binance BTC-USD 2024-01-01 2025-01-01"
        prompts = _label(
            "which crypto uses the legacy bridge? | how do I compare BTC? | what changes with the old mapping?",
            "qué cripto usa el puente legacy? | cómo comparo BTC? | qué cambia con el mapeo antiguo?",
        )
    elif key == "local_only_groq_brain":
        summary = _label(
            "Local mode with the experimental Groq brain taking the final decision.",
            "Modo local con el cerebro experimental de Groq tomando la decisión final.",
        )
        activation = _label(
            "Write `local_only_groq_brain` or `--groq-brain AAPL 2024-01-01 2025-01-01`.",
            "Escribe `local_only_groq_brain` o `--groq-brain AAPL 2024-01-01 2025-01-01`.",
        )
        usage = _label(
            "Use it when you want the experimental brain to make the final call.",
            "Úsalo cuando quieres que el brain experimental tome la señal final.",
        )
        flow = _label(
            "Keeps the local pipeline but lets Groq replace the final decision.",
            "Mantiene la ruta local pero permite que Groq reemplace la decisión final.",
        )
        agent_flow = _label(
            "Extraction and Cleaning stay local, Modeling still produces the signal, and Groq can replace the final decision before Orchestrator records it.",
            "Extracción y Limpieza siguen locales, Modelado sigue produciendo la señal, y Groq puede reemplazar la decisión final antes de que el Orquestador la registre.",
        )
        examples = "local_only_groq_brain | --groq-brain AAPL 2024-01-01 2025-01-01"
        prompts = _label(
            "what did Groq decide? | why did the signal change? | how do I enable the experimental brain?",
            "qué decidió Groq? | por qué cambió la señal? | cómo activo el brain experimental?",
        )
    elif key == "local_plus_binance_groq_brain":
        summary = _label(
            "Combines source comparison with the experimental Groq brain.",
            "Combina comparación de fuentes con el cerebro experimental de Groq.",
        )
        activation = _label(
            "Write `local_plus_binance_groq_brain` or `compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01`.",
            "Escribe `local_plus_binance_groq_brain` o `compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01`.",
        )
        usage = _label(
            "Use it when you want source comparison and the experimental final decision together.",
            "Úsalo cuando quieres comparar la fuente y dejar que Groq tome la decisión final.",
        )
        flow = _label(
            "Adds source comparison and then lets Groq make the final call.",
            "Añade comparación de fuentes y luego deja la decisión final en manos de Groq.",
        )
        agent_flow = _label(
            "Extraction compares sources, Cleaning keeps the rows aligned, Modeling summarizes the gap, and Groq can make the final call.",
            "Extracción compara fuentes, Limpieza alinea filas, Modelado resume la diferencia y Groq puede tomar la decisión final.",
        )
        examples = "local_plus_binance_groq_brain | compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01"
        prompts = _label(
            "what changed after comparing sources? | what did Groq say? | how is the final decision made?",
            "qué cambió al comparar fuentes? | qué dijo Groq? | cómo queda la decisión final?",
        )
    elif key == "local_plus_binance_legacy_groq_brain":
        summary = _label(
            "Combines source comparison, the legacy bridge, and the experimental Groq brain.",
            "Combina comparación de fuentes, puente legacy y el cerebro experimental de Groq.",
        )
        activation = _label(
            "Write `local_plus_binance_legacy_groq_brain` or `compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01`.",
            "Escribe `local_plus_binance_legacy_groq_brain` o `compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01`.",
        )
        usage = _label(
            "Use it when you need the legacy crypto bridge and the experimental decision in one run.",
            "Úsalo cuando necesitas el puente cripto antiguo y la decisión experimental en una sola corrida.",
        )
        flow = _label(
            "Adds source comparison, the legacy bridge, and the experimental final decision.",
            "Suma comparación de fuentes, puente legacy y decisión final experimental.",
        )
        agent_flow = _label(
            "Extraction compares the source paths, Cleaning aligns the rows, Modeling summarizes the difference, and Groq can close the run.",
            "Extracción compara las rutas de fuente, Limpieza alinea las filas, Modelado resume la diferencia y Groq puede cerrar la corrida.",
        )
        examples = "local_plus_binance_legacy_groq_brain | compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01"
        prompts = _label(
            "what happens with BTC or ETH? | what did Groq decide? | how does the legacy bridge enter?",
            "qué pasa con BTC o ETH? | qué decidió Groq? | cómo entra el puente legacy?",
        )
    elif key == "compare_binance_groq_brain":
        summary = _label(
            "Compares yfinance with Binance and lets Groq make the final decision.",
            "Compara yfinance con Binance y deja que Groq tome la decisión final.",
        )
        activation = _label(
            "Write `compare-binance + --groq-brain` or `compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01`.",
            "Escribe `compare-binance + --groq-brain` o `compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01`.",
        )
        usage = _label(
            "Use it when you want source comparison and the final call from the experimental brain.",
            "Úsalo cuando quieres comparar fuentes y dejar la decisión final en el brain experimental.",
        )
        flow = _label(
            "Adds Binance comparison and sends the final decision to the experimental brain.",
            "Añade la comparación de Binance y pasa la decisión final al cerebro experimental.",
        )
        agent_flow = _label(
            "Extraction compares yfinance and Binance, Cleaning keeps both sources aligned, Modeling summarizes the gap, and Groq can make the final call.",
            "Extracción compara yfinance y Binance, Limpieza mantiene ambas fuentes alineadas, Modelado resume la diferencia y Groq puede tomar la decisión final.",
        )
        examples = "compare-binance + --groq-brain | compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01"
        prompts = _label(
            "what changed after comparing with Binance? | what did Groq say? | how does the final signal look?",
            "qué cambió al comparar con Binance? | qué dijo Groq? | cómo queda la señal final?",
        )
    elif key == "compare_binance":
        summary = _label(
            "Compares yfinance with Binance so you can see source differences for the same asset.",
            "Compara yfinance con Binance para ver diferencias de fuente en el mismo activo.",
        )
        activation = _label(
            "Write `compare-binance` or `compare-binance ETH-USD 2024-01-01 2025-01-01`.",
            "Escribe `compare-binance` o `compare-binance ETH-USD 2024-01-01 2025-01-01`.",
        )
        usage = _label(
            "Use it when you want to check whether the signal changes with the source.",
            "Úsalo cuando quieras revisar si la señal cambia según la fuente.",
        )
        flow = _label(
            "Adds a source comparison and can surface the legacy bridge for BTC/ETH/LTC.",
            "Añade una comparación de fuentes y puede mostrar el puente legacy para BTC/ETH/LTC.",
        )
        agent_flow = _label(
            "Extraction compares yfinance with Binance, Cleaning prepares the same timeline, Modeling checks the source gap, and Orchestrator closes the run.",
            "Extracción compara yfinance con Binance, Limpieza prepara la misma línea temporal, Modelado revisa la diferencia de fuente y el Orquestador cierra la corrida.",
        )
        examples = "compare-binance | compare-binance ETH-USD 2024-01-01 2025-01-01"
        prompts = _label(
            "which source wins? | does the signal change? | which asset am I comparing?",
            "qué fuente gana? | cambia la señal? | qué activo comparo?",
        )
    elif key == "groq_brain":
        summary = _label(
            "Lets Groq take the final decision in the experimental run.",
            "Permite que Groq tome la decisión final del run experimental.",
        )
        activation = _label(
            "Write `--groq-brain` or `--groq-brain AAPL 2024-01-01 2025-01-01`.",
            "Escribe `--groq-brain` o `--groq-brain AAPL 2024-01-01 2025-01-01`.",
        )
        usage = _label(
            "Use it when you want the experimental brain to decide the run.",
            "Úsalo cuando quieras que el brain experimental decida la corrida.",
        )
        flow = _label(
            "Replaces the local final decision with the experimental brain.",
            "Reemplaza la decisión final local por la del cerebro experimental.",
        )
        agent_flow = _label(
            "Extraction and Cleaning stay unchanged, Modeling scores the run, and Groq can replace the local final decision.",
            "Extracción y Limpieza siguen iguales, Modelado puntúa la corrida y Groq puede reemplazar la decisión final local.",
        )
        examples = "--groq-brain | --groq-brain AAPL 2024-01-01 2025-01-01"
        prompts = _label(
            "what did Groq decide? | how do I enable --groq-brain? | what changes in the final decision?",
            "qué decidió Groq? | cómo activo --groq-brain? | qué cambia en la decisión final?",
        )
    elif "reviewer" in key:
        summary = _label(
            "Deterministic mode with an optional reviewer when the result needs a second look.",
            "Modo determinista con revisor opcional cuando el resultado necesita segunda mirada.",
        )
        activation = _label("Write `local_plus_reviewer`.", "Escribe `local_plus_reviewer`.")
        usage = _label(
            "Use it when you want a more conservative local readout.",
            "Úsalo cuando quieres una lectura local más conservadora.",
        )
        flow = _label(
            "Can add a reviewer before the decision is finalized.",
            "Puede añadir un revisor antes de cerrar la decisión.",
        )
        examples = "local_plus_reviewer | AAPL 2024-01-01 2025-01-01"
        prompts = _label(
            "when does the reviewer step in? | why did the reviewer change the decision? | how do I activate it?",
            "cuándo entra el revisor? | por qué el revisor cambió la decisión? | cómo se activa?",
        )
    elif "legacy" in key:
        summary = _label(
            "Adds the legacy bridge for BTC/ETH/LTC together with Binance comparison.",
            "Añade el puente legacy para BTC/ETH/LTC junto con la comparación con Binance.",
        )
        activation = _label("Write `local_plus_binance_legacy`.", "Escribe `local_plus_binance_legacy`.")
        usage = _label(
            "Use it when comparing crypto assets that still use the legacy mapping.",
            "Úsalo si comparas activos cripto que tienen mapeo antiguo.",
        )
        flow = _label(
            "Adds Binance comparison plus the older crypto mapping.",
            "Agrega la comparación Binance y el mapeo antiguo para cripto.",
        )
        examples = "local_plus_binance_legacy | BTC-USD 2024-01-01 2025-01-01"
        prompts = _label(
            "which crypto uses the legacy bridge? | how do I compare BTC? | what changes with the old mapping?",
            "qué cripto usa el puente legacy? | cómo comparo BTC? | qué cambia con el mapeo antiguo?",
        )
    elif "binance" in key:
        summary = _label(
            "Deterministic mode with yfinance-to-Binance comparison.",
            "Modo determinista con comparación de yfinance contra Binance.",
        )
        activation = _label("Write `compare-binance`.", "Escribe `compare-binance`.")
        usage = _label(
            "Use it when you want to check whether the signal changes with the source.",
            "Úsalo cuando quieras revisar si la señal cambia según la fuente.",
        )
        flow = _label(
            "Adds yfinance versus Binance comparison.",
            "Añade comparación de yfinance frente a Binance.",
        )
        examples = "compare-binance | compare-binance ETH-USD 2024-01-01 2025-01-01"
        prompts = _label(
            "how do I compare with Binance? | what changes in the signal? | which asset am I comparing?",
            "cómo comparo con Binance? | qué cambia en la señal? | qué activo estás comparando?",
        )
    else:
        summary = _label("Default deterministic local mode.", "Modo local determinista por defecto.")
        activation = _label("Write `local_only`.", "Escribe `local_only`.")
        usage = _label(
            "Use it when you want only the local baseline, without comparison or experimental brain.",
            "Úsalo si solo quieres la ruta local base, sin comparación ni cerebro experimental.",
        )
        flow = _label(
            "Uses only the local baseline without comparison or experimental brain.",
            "Solo usa la ruta local base sin comparación ni cerebro experimental.",
        )
        examples = "local_only | AAPL 2024-01-01 2025-01-01"
        prompts = _label(
            "what mode is active? | how do I activate it? | what changes if I compare with Binance?",
            "qué modo está activo? | cómo lo activo? | qué cambia si comparo con Binance?",
        )

    lines = [
        "Detalle de modo" if es else "Mode detail",
        _section("Label", "Etiqueta", display),
        _section("Summary", "Resumen", summary),
        _section("Activation", "Activación", activation),
        _section("When to use", "Cuándo usarlo", usage),
        _section("Flow impact", "Impacto en el flujo", flow),
    ]
    if agent_flow:
        lines.append(_section("Agent flow", "Flujo de agentes", agent_flow))
    lines.extend(
        [
            _section("Examples", "Ejemplos", examples),
            _section("Natural prompts", "Preguntas naturales", prompts),
        ]
    )
    return "\n".join(lines)


def _format_mode_guide(language: str = "en", mode: str = "") -> str:
    es = _is_spanish(language)
    key = re.sub(r"[\s\-]+", "_", (mode or "").strip().lower())
    if key and key != "mode_guide":
        return _format_mode_detail(key, language)

    lines = [
        "Centro de modos" if es else "Mode hub",
        (
            "Principales: local_only | compare-binance | --groq-brain | compare-binance + --groq-brain"
            if es
            else "Main modes: local_only | compare-binance | --groq-brain | compare-binance + --groq-brain"
        ),
        (
            "Escribe un modo para ver su detalle; el hub queda compacto."
            if es
            else "Type a mode label to see its detail; the hub stays compact."
        ),
        (
            "Estilos de respuesta visibles: interpretado | exploratorio."
            if es
            else "Visible response styles: interpreted | exploratory."
        ),
        (
            "Interpretado explica artifacts locales; exploratorio puede sumar contexto externo e hipótesis guiadas."
            if es
            else "Interpreted explains local artifacts; exploratory can add external context and guided hypotheses."
        ),
        (
            "Ejemplo: compare-binance ETH-USD 2024-01-01 2025-01-01."
            if es
            else "Example: compare-binance ETH-USD 2024-01-01 2025-01-01."
        ),
        (
            "Casos para ejecutar: compare-binance ETH-USD 2024-01-01 2025-01-01 | --groq-brain AAPL 2024-01-01 2025-01-01 | compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01."
            if es
            else "Cases to run: compare-binance ETH-USD 2024-01-01 2025-01-01 | --groq-brain AAPL 2024-01-01 2025-01-01 | compare-binance + --groq-brain ETH-USD 2024-01-01 2025-01-01."
        ),
    ]
    lines.append(
        "Flujo: local_only por defecto; compare-binance opcional; --groq-brain experimental."
        if es
        else "Flow: local_only by default; compare-binance optional; --groq-brain experimental."
    )
    return "\n".join(lines)


def _format_agent_guide(language: str = "en") -> str:
    es = _is_spanish(language)
    lines = [
        "Menú de agentes" if es else "Agent menu",
        (
            "Principales: Extracción | Limpieza | Modelado | Orquestador"
            if es
            else "Main stages: Extraction | Cleaning | Modeling | Orchestrator"
        ),
        (
            "Abre un agente por nombre para ver su detalle."
            if es
            else "Open an agent by name to see its detail."
        ),
    ]
    lines.append(
        "Flujo: extracción -> limpieza -> modelado -> orquestador."
        if es
        else "Flow: extraction -> cleaning -> modeling -> orchestrator."
    )
    lines.append(
        "Extracción obtiene símbolo, rango y columnas crudas. Limpieza prepara filas y esquema estable. Modelado genera variables, votos y confianza. Orquestador cierra la decisión final."
        if es
        else "Extraction gets the symbol, range, and raw columns. Cleaning prepares stable rows and schema. Modeling builds variables, votes, and confidence. Orchestrator closes the final decision."
    )
    lines.append(
        "Preguntas naturales:"
        if es
        else "Natural prompts:"
    )
    lines.append(
        "qué símbolo se usó? | qué columnas se extrajeron? | fila | análisis de fila | qué dice esta fila limpia? | qué predijo el modelo? | por qué decidió eso?"
        if es
        else "what symbol was used? | what columns were extracted? | row | row analysis | what does this clean row say? | what did the model predict? | why did it decide that way?"
    )
    lines.append(
        "GroqAdvisor reescribe el detalle cuando Groq está activo."
        if es
        else "GroqAdvisor rewrites the detail when Groq is active."
    )
    return "\n".join(lines)


def _wants_data_help(message: str) -> bool:
    lowered = str(message or "").strip().lower()
    if not lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "what can i do with the data",
            "what can i do with data",
            "what can i do using the data",
            "what can i do with the dataset",
            "what do i do with that",
            "and what do i do with that",
            "what do i do with this",
            "que puedo hacer con los datos",
            "qué puedo hacer con los datos",
            "que hago con eso",
            "qué hago con eso",
            "y que hago con eso",
            "y qué hago con eso",
            "que hacer con los datos",
            "qué hacer con los datos",
            "with the data",
            "con los datos",
        )
    )


def _wants_next_step_help(message: str) -> bool:
    lowered = str(message or "").strip().lower()
    if not lowered:
        return False
    return any(
        hint in lowered
        for hint in (
            "what should i do next",
            "what do i do next",
            "what can i do next",
            "what now",
            "and now",
            "what do you recommend now",
            "que hago ahora",
            "qué hago ahora",
            "ahora que sigue",
            "ahora qué sigue",
            "que me recomiendas ahora",
            "qué me recomiendas ahora",
        )
    )


def _format_data_capabilities(language: str = "en") -> str:
    es = _is_spanish(language)
    lines = [
        "Con los datos puedes:" if es else "With the data you can:",
        (
            "1. ver columnas crudas extraídas desde yfinance."
            if es
            else "1. inspect raw columns extracted from yfinance."
        ),
        (
            "2. ver filas limpias, métricas, esquema y análisis de fila."
            if es
            else "2. inspect clean rows, metrics, schema, and row analysis."
        ),
        (
            "3. ver variables del modelo, transformaciones y target_direction."
            if es
            else "3. inspect model variables, transformations, and target_direction."
        ),
        (
            "4. comparar corridas, símbolos, decisiones y confianza."
            if es
            else "4. compare runs, symbols, decisions, and confidence."
        ),
        (
            "5. pedir definiciones de conceptos como row, schema, column, variable o adj_close."
            if es
            else "5. ask for concepts such as row, schema, column, variable, or adj_close."
        ),
    ]
    return "\n".join(lines)


def _last_semantic_key(state: AssistantState) -> str:
    last_route = state.last_route if isinstance(state.last_route, dict) else {}
    if str(last_route.get("intent") or state.last_intent or "").strip() != "show_semantic_lookup":
        return ""
    last_raw = str(last_route.get("raw_message") or "").strip()
    if not last_raw:
        return ""
    basis = _semantic_basis_message(last_raw, state)
    subject = _semantic_subject_from_message(basis)
    if subject:
        entry = resolve_semantic_definition_for_term(subject)
        if entry:
            return entry.key
    entry = resolve_semantic_definition(basis)
    return str(entry.key) if entry else ""


def _semantic_lookup_help_prompts(state: AssistantState, language: str = "en") -> list[str]:
    es = _is_spanish(language)
    semantic_key = _last_semantic_key(state)
    data_keys = {
        "dataset",
        "artifact",
        "manifest",
        "target",
        "feature engineering",
        "raw column",
        "clean column",
        "column",
        "row",
        "schema",
        "model variable",
        "variable",
        "adj_close",
    }
    governance_keys = {
        "validator",
        "shadow run",
        "promotion gate",
        "challenger",
        "champion",
        "policy engine",
        "retraining scheduler",
        "feature registry",
        "adaptive selector",
        "shadow runner",
        "promotion policy",
        "drift",
    }
    if semantic_key in data_keys:
        if es:
            return [
                "qué puedo hacer con los datos?",
                "qué diferencia hay entre una raw column y una model variable?",
                "qué dice esta fila limpia?",
            ]
        return [
            "what can I do with the data?",
            "what is the difference between a raw column and a model variable?",
            "what does this clean row say?",
        ]
    if semantic_key in governance_keys:
        if es:
            return [
                "qué diferencia hay entre challenger y champion?",
                "qué es un shadow run?",
                "qué es un promotion gate?",
            ]
        return [
            "what is the difference between challenger and champion?",
            "what is a shadow run?",
            "what is a promotion gate?",
        ]
    if es:
        return [
            "qué es grounding?",
            "qué es memory?",
            "qué es un artifact store?",
        ]
    return [
        "what is grounding?",
        "what is memory?",
        "what is an artifact store?",
    ]


def _format_next_step_guidance(
    summary: Dict[str, Any],
    state: AssistantState,
    language: str = "en",
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> str:
    es = _is_spanish(language)
    snapshot = _session_context_snapshot(summary, state, language, result=result, error=error)
    stage_key = snapshot["stage_key"]
    symbol = snapshot["symbol"]
    has_symbol = symbol != "n/a"
    symbol_text_es = f" de {symbol}" if has_symbol else ""
    symbol_text_en = f" for {symbol}" if has_symbol else ""
    decision = str((summary.get("models") or {}).get("final_decision") or "").strip()

    if stage_key == "extraction":
        prompts = [
            f"abre limpieza y revisa la fila limpia{symbol_text_es}" if es else f"open cleaning and inspect the clean row{symbol_text_en}",
            "verifica qué columnas se extrajeron" if es else "verify which columns were extracted",
            "continúa hacia modelado" if es else "continue into modeling",
        ]
    elif stage_key in {"cleaning", "clean_data", "metrics"}:
        prompts = [
            f"revisa qué dice esta fila limpia{symbol_text_es}" if es else f"review what this clean row says{symbol_text_en}",
            "abre modelado para ver variables y votos" if es else "open modeling to inspect variables and votes",
            "compara la señal limpia con la decisión final" if es else "compare the clean signal with the final decision",
        ]
    elif stage_key in {"modeling", "model_variables"}:
        prompts = [
            f"pide la decisión final del orquestador{symbol_text_es}" if es else f"ask for the orchestrator final decision{symbol_text_en}",
            "revisa por qué decidió eso" if es else "review why it decided that way",
            "contrasta variables del modelo con la fila limpia" if es else "contrast model variables with the clean row",
        ]
    elif stage_key in {"orchestrator", "decision", "summary", "comparison", "legacy"}:
        first = f"revisa por qué decidió {decision}" if es and decision else (
            f"review why it decided {decision}" if decision else ("revisa por qué decidió eso" if es else "review why it decided that way")
        )
        prompts = [
            first,
            "vuelve a modelado para ver votos y confianza" if es else "go back to modeling to inspect votes and confidence",
            "compara esta corrida con otra si quieres validar estabilidad" if es else "compare this run with another one if you want to validate stability",
        ]
    elif stage_key == "semantic_lookup":
        prompts = _semantic_lookup_help_prompts(state, language)
    elif stage_key == "assistant_scorecard":
        prompts = [
            "assistant scorecard layers",
            "estado web" if es else "web status",
            "qué cambia si activo web?" if es else "what changes if I enable web?",
        ]
    else:
        prompts = _contextual_help_prompts(summary, state, language, result=result, error=error)
    title = "Siguientes movimientos útiles:" if es else "Next useful moves now:"
    lines = [title]
    for index, prompt in enumerate(prompts[:3], start=1):
        lines.append(f"{index}. {prompt}")
    return "\n".join(lines)


def _format_agent_card(stage: str, language: str = "en") -> str:
    es = _is_spanish(language)
    stage_key = stage if stage in {"extraction", "cleaning", "modeling"} else "extraction"
    stage_titles = {
        "extraction": ("Extraction", "Extracción"),
        "cleaning": ("Cleaning", "Limpieza"),
        "modeling": ("Modeling", "Modelado"),
    }
    title_en, title_es = stage_titles[stage_key]
    title = title_es if es else title_en

    if stage_key == "extraction":
        if es:
            purpose = "Símbolo, rango, filas crudas y columnas faltantes."
            useful = [
                "2 / extracción / qué símbolo se usó?",
                "qué columnas se extrajeron?",
                "qué pasó en la extracción?",
            ]
            examples = [
                "AAPL",
                "BTC-USD",
                "EURUSD=X",
            ]
            handoff = "Handoff: extracción pasa el mismo run a limpieza. Si faltan filas, símbolo o rango, vuelve a extracción."
        else:
            purpose = "Symbol, date range, raw rows, and missing columns."
            useful = [
                "2 / extraction / what symbol was used?",
                "what columns were extracted?",
                "what happened in extraction?",
            ]
            examples = [
                "AAPL",
                "BTC-USD",
                "EURUSD=X",
            ]
            handoff = "Handoff: extraction passes the same run to cleaning. If rows, symbol, or date range are wrong, go back to extraction."
    elif stage_key == "cleaning":
        if es:
            purpose = "Esquema, métricas limpias, símbolos presentes, filas limpias, una fila exacta y análisis de fila."
            useful = [
                "3 / limpieza / qué símbolos hay en los datos limpios?",
                "metricas",
                "fila",
                "row",
                "análisis de fila",
                "esquema",
                "qué dice esta fila limpia?",
                "muestra la fila limpia de MSFT el 2026-03-21",
            ]
            examples = [
                "3",
                "limpieza",
            ]
            handoff = "Handoff: limpieza entrega el mismo run a modelado. Si la fila exacta, la métrica o el análisis faltan, vuelve a extracción."
        else:
            purpose = "Schema, clean metrics, present symbols, clean rows, one exact row, and row analysis."
            useful = [
                "3 / cleaning / what symbols are in the cleaned data?",
                "metrics",
                "row",
                "row analysis",
                "schema",
                "what does this clean row say?",
                "show me the clean row for MSFT on 2026-03-21",
                "analyze the clean row for MSFT on 2026-03-21",
            ]
            examples = [
                "3",
                "cleaning",
            ]
            handoff = "Handoff: cleaning passes the same run to modeling. If the exact row, metric, or analysis is missing, go back to extraction."
    else:
        if es:
            purpose = "Votos, confianza y señales long/short/hold."
            useful = [
                "4 / modelado / qué predijo el modelo?",
                "cómo afecta la limpieza a long short hold?",
                "qué dijo Groq en modelado?",
            ]
            examples = [
                "4",
                "modelado",
            ]
            handoff = "Handoff: modelado entrega el mismo run al orquestador. Si quieres la verdad final o comparar modos, consulta el orquestador."
        else:
            purpose = "Votes, confidence, and long/short/hold signals."
            useful = [
                "4 / modeling / what did the model predict?",
                "how does clean data affect long short hold?",
                "what did Groq say in modeling?",
            ]
            examples = [
                "4",
                "modeling",
            ]
            handoff = "Handoff: modeling passes the same run to the orchestrator. If you want the final truth or to compare modes, consult the orchestrator."

    body_lines = [
        f"{'Propósito' if es else 'Purpose'}: {purpose}",
        f"{'Preguntas naturales' if es else 'Natural prompts'}: {' · '.join(useful)}",
        f"{'Ejemplos' if es else 'Examples'}: {' · '.join(examples)}",
    ]
    if stage_key == "extraction":
        body_lines.append(
            (
                "Símbolos útiles: AAPL, MSFT, NVDA, SPY, QQQ, ^GSPC, BTC-USD, ETH-USD, EURUSD=X."
                if es
                else "Useful symbols: AAPL, MSFT, NVDA, SPY, QQQ, ^GSPC, BTC-USD, ETH-USD, EURUSD=X."
            )
        )
    body_lines.append(handoff)
    return "__agent_card__:{stage}:{lang}\n".format(stage=stage_key, lang="es" if es else "en") + "\n".join(body_lines)


class AssistantRuntime:
    def __init__(self, artifact_root: str = "artifacts", session_id: str = "default") -> None:
        self.artifact_root = artifact_root
        self.session_id = session_id
        self._orchestrator = None
        self._orchestrator_error: Optional[Exception] = None
        self.comp = AssistantCompAgent()
        self.router = AssistantRouter()
        self.planner = AssistantPlanner(self.router, context_resolver=AssistantContextResolver(self.artifact_root))
        self.policy = AssistantPolicyEngine()
        self.executor = AssistantExecutor(self.policy)
        self.web_retriever = AssistantWebRetriever()

    def _run_dir(self, run_id: str) -> Path:
        return Path(self.artifact_root) / "runs" / run_id

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def _load_bundle_from_fs(self, run_id: str) -> Dict[str, Any]:
        run_dir = self._run_dir(run_id)
        if not run_dir.exists():
            return {"manifest": None, "result": None, "summary": None, "logs": [], "error": None}
        summary = self._read_json(run_dir / "summary.json")
        result = self._read_json(run_dir / "result.json")
        manifest = self._read_json(run_dir / "manifest.json")
        logs = self._read_json(run_dir / "logs.json")
        error = self._read_json(run_dir / "error.json")
        return {
            "manifest": manifest or None,
            "result": result or None,
            "summary": summary or None,
            "logs": logs if isinstance(logs, list) else [],
            "error": error or None,
        }

    def _latest_run_id(self) -> Optional[str]:
        runs_root = Path(self.artifact_root) / "runs"
        if not runs_root.exists():
            return None
        run_ids = [path.name for path in runs_root.iterdir() if path.is_dir()]
        if not run_ids:
            return None
        return sorted(run_ids)[-1]

    def _ensure_orchestrator(self):
        if self._orchestrator is not None:
            return self._orchestrator
        try:
            from pipeline.orchestrator import PipelineOrchestrator

            self._orchestrator = PipelineOrchestrator(artifact_root=self.artifact_root)
            self._orchestrator_error = None
            return self._orchestrator
        except Exception as exc:  # noqa: BLE001
            self._orchestrator_error = exc
            raise RuntimeError(
                f"Pipeline runtime is unavailable: {exc}. Install the project dependencies or activate the venv."
            ) from exc

    def _load_bundle(self, run_id: str) -> Dict[str, Any]:
        bundle = self._load_bundle_from_fs(run_id)
        return bundle if isinstance(bundle, dict) else {"manifest": None, "result": None, "summary": None, "logs": [], "error": None}

    def _latest_bundle(self, state: AssistantState) -> Dict[str, Any]:
        run_id = state.last_run_id or self._latest_run_id()
        if not run_id:
            return {"manifest": None, "result": None, "summary": None, "logs": [], "error": None}
        return self._load_bundle(run_id)

    def _bundle_for_route(self, state: AssistantState, route: AssistantRoute) -> Dict[str, Any]:
        run_id = route.run_id
        if not run_id and not bool(getattr(route, "override_memory", False)):
            run_id = state.last_run_id or self._latest_run_id()
        if not run_id:
            return {"manifest": None, "result": None, "summary": None, "logs": [], "error": None}
        bundle = self._load_bundle(run_id)
        if bundle.get("manifest") or bundle.get("result") or bundle.get("summary") or bundle.get("error"):
            return bundle
        if route.run_id:
            return {
                "manifest": {"run_id": run_id},
                "result": None,
                "summary": None,
                "logs": [],
                "error": {
                    "stage": "orchestrator",
                    "message": f"Run {run_id} was not found.",
                    "details": {"run_id": run_id},
                },
            }
        return bundle

    def _comparison_bundle_for_route(self, state: AssistantState, route: AssistantRoute) -> Dict[str, Any]:
        left_run_id = route.run_id or state.last_run_id or self._latest_run_id()
        right_run_id = route.secondary_run_id
        left_bundle = self._load_bundle(left_run_id) if left_run_id else {"manifest": None, "result": None, "summary": None, "logs": [], "error": None}
        right_bundle = self._load_bundle(right_run_id) if right_run_id else {"manifest": None, "result": None, "summary": None, "logs": [], "error": None}
        return {"left": left_bundle, "right": right_bundle}

    def _update_state_after_run(self, state: AssistantState, route: AssistantRoute, result: Any, summary: Dict[str, Any]) -> AssistantState:
        state.last_run_id = getattr(result, "run_id", None) or summary.get("run_id")
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_request = _route_to_request(route, self.artifact_root).model_dump()
        state.current_asset = (route.comparison_asset or (route.tickers[0] if route.tickers else "") or state.current_asset)
        state.current_mode = summary.get("run_mode", state.current_mode)
        state.last_summary = summary
        state.pending_route = {}
        state.pending_task = ""
        context_stage = _normalize_context_stage(route.intent, route.stage)
        state.notes.append(f"{route.intent}:{context_stage or 'n/a'}:{state.last_run_id or 'n/a'}")
        return state

    def _handle_set_language(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = route.language if route.language in {"en", "es"} else "en"
        state.preferred_language = language
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.notes.append(f"language:{language}")
        if language == "es":
            return "Idioma cambiado a español. A partir de ahora responderé en español hasta que me lo cambies.", state
        return "Language set to English. I’ll keep replying in English until you change it.", state

    def _format_identity_response(self, state: AssistantState, language: str, *, detailed: bool) -> str:
        es = _is_spanish(language)
        if detailed:
            identity = resolve_assistant_identity()
            mode = str(state.current_mode or "").strip() or "local_only"
            current_asset = str(state.current_asset or "").strip()
            last_run_id = str(state.last_run_id or "").strip()
            config_status_fn = getattr(self.web_retriever, "config_status", None)
            web_status = config_status_fn() if callable(config_status_fn) else None
            web_ready = bool(getattr(web_status, "runtime_ready", False))
            web_enabled = bool(getattr(web_status, "enabled", False) if web_status is not None else getattr(self.web_retriever, "enabled", False))
            lines: list[str] = [
                f"{'Soy' if es else 'I am'} {identity.name}, {identity.render(language)}."
            ]
            lines.append(
                (
                    f"En esta sesión estoy en modo {mode}."
                    if es
                    else f"In this session I am operating in {mode} mode."
                )
            )
            if last_run_id:
                lines.append(
                    f"{'Último run' if es else 'Latest run'}: {last_run_id}."
                )
            if current_asset:
                lines.append(
                    f"{'Foco actual' if es else 'Current focus'}: {current_asset}."
                )
            lines.append(identity.capability_clause(language))
            lines.append(
                (
                    "Mi rol en esta sesión es interpretar preguntas, mantener contexto, y responder con facts locales o contexto externo cuando haga falta."
                    if es
                    else "My role in this session is to interpret questions, keep context, and answer with local facts or external context when needed."
                )
            )
            if web_ready:
                lines.append(
                    (
                        "La búsqueda web está lista, así que puedo complementar con contexto externo cuando haga falta."
                        if es
                        else "Web search is ready, so I can complement with external context when needed."
                    )
                )
            elif web_enabled:
                lines.append(
                    (
                        "La búsqueda web está configurada parcialmente, pero todavía no está lista en runtime."
                        if es
                        else "Web search is configured partially, but it is not runtime-ready yet."
                    )
                )
            else:
                lines.append(
                    (
                        "La búsqueda web todavía no está configurada."
                        if es
                        else "Web search is not configured yet."
                    )
                )
            return " ".join(line.strip() for line in lines if line).strip()

        return "Hola. ¿En qué puedo ayudarte?" if es else "Hello. How can I help?"

    def _handle_greeting(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        answer = self._format_identity_response(state, language, detailed=False)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_identity(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        answer = self._format_identity_response(state, language, detailed=True)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _build_context(self, route: AssistantRoute, bundle: Dict[str, Any], summary_text: str, fallback_text: str) -> Dict[str, Any]:
        return {
            "route": route.to_dict(),
            "summary": bundle.get("summary") or {},
            "result": bundle.get("result") or {},
            "summary_text": summary_text,
            "fallback_text": fallback_text,
        }

    def _collect_web_facts(self, route: AssistantRoute) -> List[Any]:
        if route.source_mode not in {"web", "mixed"}:
            return []
        if not self.web_retriever.enabled:
            return []
        facts: List[Any] = []
        seen: set[str] = set()
        for query in route.web_queries[:3]:
            for fact in self.web_retriever.search(query, limit=2):
                url = str(_web_fact_value(fact, "url") or "").strip()
                key = url or f"{_web_fact_value(fact, 'title')}:{_web_fact_value(fact, 'snippet')}"
                if not key or key in seen:
                    continue
                seen.add(key)
                facts.append(fact)
        if not facts:
            return []
        sorted_facts = sorted(
            facts,
            key=lambda item: (
                -float(_web_fact_value(item, "trust_score", 0.0) or 0.0),
                int(_web_fact_value(item, "rank", 999) or 999),
                str(_web_fact_value(item, "domain") or ""),
            ),
        )
        selected: List[Any] = []
        seen_domains: set[str] = set()
        overflow: List[Any] = []
        for fact in sorted_facts:
            domain = str(_web_fact_value(fact, "domain") or "").strip().lower()
            if domain and domain not in seen_domains:
                seen_domains.add(domain)
                selected.append(fact)
            else:
                overflow.append(fact)
        final = (selected + overflow)[:3]
        return final

    def _probe_search_interpretation(self, queries: list[str]) -> List[Any]:
        if not self.web_retriever.enabled:
            return []
        facts: List[Any] = []
        seen: set[str] = set()
        for query in queries[:3]:
            for fact in self.web_retriever.search(query, limit=1):
                url = str(_web_fact_value(fact, "url") or "").strip()
                key = url or f"{_web_fact_value(fact, 'title')}:{_web_fact_value(fact, 'snippet')}"
                if not key or key in seen:
                    continue
                seen.add(key)
                facts.append(fact)
        return facts[:3]

    def _should_attempt_search_augmented_interpretation(
        self,
        message: str,
        route: AssistantRoute,
        canonical_query: str,
    ) -> bool:
        if not canonical_query:
            return False
        explicit_tickers = _explicit_tickers_in_message(message, list(route.tickers or []))
        if explicit_tickers:
            return False
        blocked_intents = {
            "show_time_status",
            "show_source_scope",
            "show_web_status",
            "show_assistant_scorecard",
            "show_market_metrics",
            "show_clean_data",
            "show_model_variables",
            "show_decision_explanation",
            "show_extraction",
            "show_cleaning",
            "show_prediction",
            "show_run_comparison",
            "show_agent_guide",
            "show_agent_card",
            "show_mode_guide",
            "set_language",
            "greet",
            "continue_task",
            "compare_sources",
        }
        if route.intent in blocked_intents:
            return False
        if route.intent == "run_pipeline":
            lowered = str(message or "").lower()
            if re.search(r"\b(run|execute|ejecuta|extrae|extract)\b", lowered):
                return False
        return route.intent in {
            "unknown",
            "show_semantic_lookup",
            "show_stage_brief",
            "show_latest_summary",
            "help",
            "run_pipeline",
        }

    def _augment_plan_with_search_interpretation(
        self,
        message: str,
        state: AssistantState,
        plan: AssistantPlan,
    ) -> AssistantPlan:
        route = AssistantRoute.from_dict(plan.route)
        canonical_query = str(route.interpreted_query or "").strip() or _canonical_semantic_interpretation_query(message, state)
        if not self._should_attempt_search_augmented_interpretation(message, route, canonical_query):
            return plan

        has_local_coverage = _has_local_semantic_coverage(canonical_query)
        probe_facts: List[Any] = []
        interpretation_queries = _interpretation_queries_for_semantic_query(canonical_query)
        if not has_local_coverage and interpretation_queries:
            probe_facts = self._probe_search_interpretation(interpretation_queries)

        if not has_local_coverage and not probe_facts and route.intent != "show_semantic_lookup":
            return plan

        explicit_tickers = _explicit_tickers_in_message(message, list(route.tickers or []))
        route.intent = "show_semantic_lookup"
        route.stage = "semantic_lookup"
        route.question_focus = "semantic_lookup"
        route.answer_mode = "interpreted"
        route.certainty = "inferred"
        route.interpreted_query = canonical_query
        route.interpretation_source = "web_search" if probe_facts else ""
        route.interpretation_note = (
            "Search-augmented interpretation resolved the conceptual subject before routing."
            if probe_facts
            else ""
        )
        route.tickers = explicit_tickers

        source_selection = self.planner.source_selector.select(canonical_query, route, state)
        route.source_mode = source_selection.mode
        ordered_queries: list[str] = []
        seen_queries: set[str] = set()
        for item in interpretation_queries + list(source_selection.queries):
            text = str(item or "").strip()
            key = text.lower()
            if not text or key in seen_queries:
                continue
            seen_queries.add(key)
            ordered_queries.append(text)
        route.web_queries = ordered_queries[:5]

        context_snapshot = dict(plan.context_snapshot or {})
        context_snapshot.update(
            {
                "intent": route.intent,
                "stage": route.stage or "",
                "question_focus": route.question_focus,
                "answer_mode": route.answer_mode,
                "certainty": route.certainty,
                "source_mode": route.source_mode,
                "web_queries": list(route.web_queries),
                "interpreted_query": route.interpreted_query,
                "interpretation_source": route.interpretation_source,
            }
        )
        steps = self.planner.tool_layer.build_steps(route, context_snapshot)
        risk = "medium" if route.source_mode in {"web", "mixed"} else "low"
        explanation = (
            "The planner used search-augmented interpretation to resolve the conceptual subject before final routing, "
            f"then rebuilt the plan as intent={route.intent} with source_mode={route.source_mode}."
        )
        return AssistantPlan(
            intent=route.intent,
            route=route.to_dict(),
            steps=steps,
            grounded=True,
            risk=risk,  # type: ignore[arg-type]
            explanation=explanation,
            answer_mode=route.answer_mode,  # type: ignore[arg-type]
            certainty=route.certainty,  # type: ignore[arg-type]
            context_snapshot=context_snapshot,
            source_mode=route.source_mode,  # type: ignore[arg-type]
        )

    def _handle_run(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        route.language = language
        request = _route_to_request(route, self.artifact_root)
        orchestrator = self._ensure_orchestrator()
        try:
            result = orchestrator.run(request)
        except Exception as exc:
            run_id = self._latest_run_id() or state.last_run_id
            bundle = self._load_bundle(run_id) if run_id else {"manifest": None, "result": None, "summary": None, "logs": [], "error": None}
            answer = _format_run_failure(bundle, language, str(exc))
            state.last_run_id = (bundle.get("manifest") or {}).get("run_id") or run_id or state.last_run_id
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_request = request.model_dump()
            state.current_asset = (route.comparison_asset or (route.tickers[0] if route.tickers else "") or state.current_asset)
            state.last_summary = {}
            state.pending_route = {}
            state.pending_task = ""
            context_stage = _normalize_context_stage(route.intent, route.stage)
            state.notes.append(f"{route.intent}:{context_stage or 'n/a'}:{state.last_run_id or 'n/a'}:failed")
            return answer, state
        bundle = self._load_bundle(result.run_id)
        summary = bundle.get("summary") or {}
        local_answer = _format_concise_summary(summary, language)
        groq_answer = self.router.synthesize(
            route.raw_message or "run pipeline",
            self._build_context(route, bundle, local_answer, local_answer) | {"language": language},
            local_answer,
        )
        state = self._update_state_after_run(state, route, result, summary)
        return groq_answer, state

    def _handle_show_latest(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        if error and not summary and not result:
            answer = _format_run_failure(bundle, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        if not summary and not result:
            web_facts = self._collect_web_facts(route)
            if web_facts:
                route.certainty = _resolve_dynamic_certainty(route, web_facts)
                answer = _format_semantic_web_brief(route, web_facts, language)
                answer = _append_source_summary(answer, route, web_facts, language=language)
                answer = _append_source_attribution(answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
                answer = _prepend_response_contract(answer, route, language)
                state.last_intent = route.intent
                state.last_route = route.to_dict()
                state.last_summary = {}
                return answer, state
            return (
                _format_availability_notice(
                    title_en="Session status",
                    title_es="Estado de sesión",
                    status_en="No runs have been created yet.",
                    status_es="No hay ejecuciones completadas todavía.",
                    action_en="Start a run first, then ask for status again.",
                    action_es="Inicia una corrida primero y luego pide el estado.",
                    prompt_en="status | status full | latest run",
                    prompt_es="estado | estado completo | última corrida",
                    language=language,
                ),
                state,
            )
        summary = summary or _summary_from_result(result)
        web_facts = self._collect_web_facts(route)
        local_decision = str((summary.get("models") or {}).get("final_decision") or result.get("final_decision") or "")
        route.certainty = _resolve_dynamic_certainty(route, web_facts, conflict=_has_decision_conflict(local_decision, web_facts))
        answer = _format_latest_report(summary, language)
        answer = _append_direction_conflict_note(answer, local_decision, web_facts, language=language)
        answer = _append_external_backdrop(answer, route, web_facts, language=language)
        answer = _append_source_priority(answer, route, language=language)
        answer = _append_source_summary(answer, route, web_facts, language=language)
        answer = _append_source_attribution(answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
        answer = _append_exploratory_tail(answer, route, summary, result, language)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = summary
        return answer, state

    def _handle_semantic_lookup(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        route.question_focus = "semantic_lookup"
        bundle = self._bundle_for_route(state, route)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        semantic_basis = route.interpreted_query or _semantic_basis_message(route.raw_message or "", state)
        if not route.tickers:
            tickers = summary.get("tickers") or ((result.get("manifest") or {}).get("request") or {}).get("tickers") or []
            if tickers:
                route.tickers = [str(tickers[0]).strip()]
        web_facts = self._collect_web_facts(route)
        route.certainty = _resolve_dynamic_certainty(route, web_facts)
        if web_facts:
            answer = _format_semantic_web_brief(route, web_facts, language)
            local_anchor = _format_semantic_local_anchor(route, state, language)
            if local_anchor:
                answer = f"{answer} {local_anchor}".strip()
            answer = _append_interpretation_trace(answer, route, language)
            answer = _append_source_priority(answer, route, language=language)
            answer = _append_source_summary(answer, route, web_facts, language=language)
            answer = _append_source_attribution(answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
            answer = _prepend_response_contract(answer, route, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        comparison_subjects = _semantic_comparison_subjects_from_message(semantic_basis)
        if comparison_subjects:
            local_comparison_brief = _format_semantic_local_comparison_brief(semantic_basis, language)
            if local_comparison_brief:
                route.source_mode = "local"
                route.certainty = "inferred"
                answer = local_comparison_brief
                local_anchor = _format_semantic_local_anchor(route, state, language)
                if local_anchor:
                    answer = f"{answer} {local_anchor}".strip()
                answer = (
                    f"{answer} {'Usé comparaciones del glosario local porque el glosario local ya cubre ambos términos.' if _is_spanish(language) else 'I used local glossary comparisons because the local glossary already covers both terms.'}"
                ).strip()
                answer = _append_interpretation_trace(answer, route, language)
                answer = _prepend_response_contract(answer, route, language)
                state.last_intent = route.intent
                state.last_route = route.to_dict()
                state.last_summary = summary or {}
                return answer, state
        local_brief = _format_semantic_local_brief(semantic_basis, route.tickers[0] if route.tickers else "", language)
        if local_brief:
            route.source_mode = "local"
            route.certainty = "inferred"
            answer = local_brief
            local_anchor = _format_semantic_local_anchor(route, state, language)
            if local_anchor:
                answer = f"{answer} {local_anchor}".strip()
            answer = (
                f"{answer} {'Usé contexto local del dominio porque el glosario local ya tenía una definición directa.' if _is_spanish(language) else 'I used local domain context because the local glossary already had a direct definition.'}"
            ).strip()
            answer = _append_interpretation_trace(answer, route, language)
            answer = _prepend_response_contract(answer, route, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = summary or {}
            return answer, state
        if comparison_subjects:
            left, right = comparison_subjects
            route.source_mode = "web" if self.web_retriever.enabled else "local"
            route.certainty = "inferred" if self.web_retriever.enabled else "hypothesis"
            if _is_spanish(language):
                prefix = "Entendí esto como una comparación entre"
                reason = (
                    "y puedo complementar esa comparación con contexto externo."
                    if self.web_retriever.enabled
                    else "pero todavía no tengo búsqueda web activa para completar la comparación."
                )
            else:
                prefix = "I understood this as a comparison between"
                reason = (
                    "and I can complement that comparison with external context."
                    if self.web_retriever.enabled
                    else "but I do not have web retrieval active yet to complete the comparison."
                )
            answer = f"{prefix} {left} and {right}. {reason}"
            if self.web_retriever.enabled:
                answer += (
                    " I can search for both terms and contrast their context."
                    if not _is_spanish(language)
                    else " Puedo buscar ambos términos y contrastar su contexto."
                )
            answer = _append_interpretation_trace(answer, route, language)
            answer = _prepend_response_contract(answer, route, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = summary or {}
            return answer, state
        subject = _semantic_subject_from_message(semantic_basis)
        if subject:
            route.source_mode = "web" if self.web_retriever.enabled else "local"
            route.certainty = "inferred" if self.web_retriever.enabled else "hypothesis"
            prefix = "Entendí esto como una pregunta sobre" if _is_spanish(language) else "I understood this as a question about"
            if self.web_retriever.enabled:
                reason = (
                    "y puedo complementar esa idea con contexto externo, pero no encontré una definición local directa."
                    if _is_spanish(language)
                    else "and I can complement that idea with external context, but I did not find a direct local definition."
                )
            else:
                reason = (
                    "pero todavía no tengo búsqueda web activa para completar la definición."
                    if _is_spanish(language)
                    else "but I do not have web retrieval active yet to complete the definition."
                )
            answer = f"{prefix} {subject}. {reason}"
            if self.web_retriever.enabled:
                answer += (
                    " Puedo buscar contexto externo para completar esa idea."
                    if _is_spanish(language)
                    else " I can search for external context to complete that idea."
                )
            answer = _append_interpretation_trace(answer, route, language)
            answer = _prepend_response_contract(answer, route, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = summary or {}
            return answer, state
        answer = _format_availability_notice(
            title_en="Semantic lookup",
            title_es="Búsqueda semántica",
            status_en="I do not have enough external semantic context yet.",
            status_es="Todavía no tengo suficiente contexto semántico externo.",
            action_en="Enable web retrieval, or ask a run-specific question grounded in local artifacts.",
            action_es="Activa el retriever web o pregunta algo específico de una corrida grounded en artifacts locales.",
            prompt_en="web status | what env do I need for tavily? | what does forex mean?",
            prompt_es="estado del retriever | what env do I need for tavily? | qué es forex?",
            language=language,
        )
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = {}
        return answer, state

    def _handle_stage_view(self, state: AssistantState, route: AssistantRoute, stage: str) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        if stage == "symbols":
            answer = _format_symbol_guide(summary, result, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = summary or state.last_summary
            return answer, state
        if error and not summary and not result:
            answer = _format_run_failure(bundle, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        if not summary and not result:
            stage_titles = {
                "extraction": ("Extraction", "Extracción"),
                "cleaning": ("Cleaning", "Limpieza"),
                "modeling": ("Modeling", "Modelado"),
                "legacy": ("Legacy bridge", "Puente legacy"),
                "comparison": ("Source comparison", "Comparación de fuentes"),
                "stage_brief": ("Stage brief", "Brief de etapa"),
                "motor": ("Motor", "Motor"),
            }
            title_en, title_es = stage_titles.get(stage, ("Stage view", "Vista de etapa"))
            return (
                _format_availability_notice(
                    title_en=title_en,
                    title_es=title_es,
                    status_en="No runs available yet.",
                    status_es="No hay ejecuciones disponibles todavía.",
                    action_en="Start a run first, then come back here.",
                    action_es="Inicia una corrida primero y luego vuelve aquí.",
                    prompt_en="what symbol was used? | what columns were extracted?",
                    prompt_es="qué símbolo se usó? | qué columnas se extrajeron?",
                    language=language,
                ),
                state,
            )
        summary = summary or _summary_from_result(result)
        if stage == "extraction":
            answer = _format_extraction(summary, result, language)
        elif stage == "cleaning":
            answer = _format_cleaning(summary, result, language)
        elif stage == "modeling":
            answer = _format_prediction(summary, result, language)
        elif stage == "legacy":
            answer = _format_legacy(summary, result, language)
        elif stage == "comparison":
            answer = _format_comparison(summary, result, language)
        elif stage == "symbols":
            answer = _format_symbol_guide(summary, result, language)
        elif stage == "stage_brief":
            fallback = (
                _format_motor(summary, result, language)
                if route.stage == "motor"
                else _format_stage_brief(summary, result, route.stage or "orchestrator", language)
            )
            if self.router.groq_router.enabled and _should_semantic_rewrite(route):
                answer = self.router.synthesize(
                    route.raw_message or "explain the current stage",
                    {
                        "summary": summary,
                        "result": result,
                        "stage": route.stage,
                        "stage_brief": (summary.get("stage_briefs") or {}).get(route.stage or "orchestrator"),
                        "motor": summary.get("motor") or result.get("motor") or {},
                        "language": language,
                    },
                    fallback,
                )
            else:
                answer = fallback
        elif stage == "motor":
            answer = _format_motor(summary, result, language)
        else:
            answer = _format_concise_summary(summary, language)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = summary
        return answer, state

    def _handle_asset_used(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        if error and not summary and not result:
            answer = _format_run_failure(bundle, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        if not summary and not result:
            return (
                _format_availability_notice(
                    title_en="Asset",
                    title_es="Activo",
                    status_en="No runs available yet.",
                    status_es="No hay ejecuciones disponibles todavía.",
                    action_en="Start a run first, then ask which symbol was used.",
                    action_es="Inicia una corrida primero y luego pregunta qué símbolo se usó.",
                    prompt_en="what symbol was used? | what asset did the last run use?",
                    prompt_es="qué símbolo se usó? | qué activo usó la última corrida?",
                    language=language,
                ),
                state,
            )
        summary = summary or _summary_from_result(result)
        web_facts = self._collect_web_facts(route)
        route.certainty = _resolve_dynamic_certainty(route, web_facts)
        answer = _format_asset_used(summary, result, language)
        answer = _append_external_backdrop(answer, route, web_facts, language=language)
        answer = _append_source_priority(answer, route, language=language)
        answer = _append_source_summary(answer, route, web_facts, language=language)
        answer = _append_source_attribution(answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = summary
        return answer, state

    def _handle_market_type(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        if error and not summary and not result and not route.tickers and not state.current_asset:
            answer = _format_run_failure(bundle, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        if not summary and not result and not route.tickers and not state.current_asset:
            return (
                _format_availability_notice(
                    title_en="Market type",
                    title_es="Tipo de mercado",
                    status_en="No symbol context is available yet.",
                    status_es="Todavía no hay contexto de símbolo disponible.",
                    action_en="Provide a run id or symbol, for example `run_0001` or `BTC-USD`.",
                    action_es="Proporciona un run id o un símbolo, por ejemplo `run_0001` o `BTC-USD`.",
                    prompt_en="what market type does BTC-USD belong to?",
                    prompt_es="de qué tipo de mercado hace parte BTC-USD?",
                    language=language,
                ),
                state,
            )
        summary = summary or _summary_from_result(result)
        if route.tickers:
            symbol = route.tickers[0]
            summary = {**summary, "tickers": [symbol], "run_id": summary.get("run_id") or route.run_id}
        elif not summary.get("tickers"):
            symbol = state.current_asset or ""
            if symbol:
                summary = {**summary, "tickers": [symbol], "run_id": summary.get("run_id") or route.run_id}
        web_facts = self._collect_web_facts(route)
        packet = build_grounding_packet(summary, result, route.source_mode, web_facts).to_dict()
        local_identity = ((packet.get("local_facts") or {}).get("market_identity") or {})
        route.certainty = _resolve_dynamic_certainty(
            route,
            web_facts,
            conflict=_has_market_type_conflict(str(local_identity.get("market_type") or ""), web_facts),
        )
        answer = _format_market_type(packet, language=language, web_enabled=self.web_retriever.enabled)
        answer = _append_exploratory_tail(answer, route, summary, result, language)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = summary or state.last_summary
        return answer, state

    def _handle_clean_data(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        if error and not summary and not result:
            answer = _format_run_failure(bundle, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        if not summary and not result:
            return (
                _format_availability_notice(
                    title_en="Clean market data overview",
                    title_es="Resumen de clean_market_data",
                    status_en="No runs available yet.",
                    status_es="No hay ejecuciones disponibles todavía.",
                    action_en="Start a run first, then open clean data again.",
                    action_es="Ejecuta una corrida primero y vuelve a datos limpios.",
                    prompt_en="what symbols are in the cleaned data? | what does this clean row say?",
                    prompt_es="qué símbolos hay en los datos limpios? | qué dice esta fila?",
                    language=language,
                ),
                state,
            )
        summary = summary or _summary_from_result(result)
        web_facts = self._collect_web_facts(route)
        route.certainty = _resolve_dynamic_certainty(route, web_facts)
        answer = _format_clean_data_view(summary, result, route.raw_message or "", language, route.tickers)
        if self.router.groq_router.enabled and _should_semantic_rewrite(route) and _should_enrich_clean_data_with_groq(route.raw_message or ""):
            groq_answer = self.router.synthesize(
                route.raw_message or "analyze cleaned market data",
                {
                    "summary": summary,
                    "result": result,
                    "language": language,
                    "clean_data_view": answer,
                    "request_tickers": route.tickers,
                },
                answer,
            )
            if groq_answer and groq_answer.strip() and groq_answer.strip() != answer.strip():
                answer = groq_answer.strip()
        answer = _ensure_clean_data_header(answer, summary, _extract_clean_data_mode(route.raw_message or ""), language)
        answer = _append_external_backdrop(answer, route, web_facts, language=language)
        answer = _append_source_priority(answer, route, language=language)
        answer = _append_source_summary(answer, route, web_facts, language=language)
        answer = _append_source_attribution(answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
        answer = _append_exploratory_tail(answer, route, summary, result, language)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = summary
        return answer, state

    def _handle_market_metrics(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        if error and not summary and not result:
            answer = _format_run_failure(bundle, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        if not summary and not result:
            return (
                _format_availability_notice(
                    title_en="Market metrics",
                    title_es="Métricas de mercado",
                    status_en="No runs available yet.",
                    status_es="No hay ejecuciones disponibles todavía.",
                    action_en="Start a run first, then ask for volume or volatility.",
                    action_es="Ejecuta una corrida primero y luego pide volumen o volatilidad.",
                    prompt_en="what is the volume of AAPL? | what is the volatility of AAPL?",
                    prompt_es="cuál es el volumen de AAPL? | cuál es la volatilidad de AAPL?",
                    language=language,
                ),
                state,
            )
        summary = summary or _summary_from_result(result)
        web_facts = self._collect_web_facts(route)
        route.certainty = _resolve_dynamic_certainty(
            route,
            web_facts,
            conflict=_has_market_metrics_conflict(summary, result, route.raw_message or "", route.tickers, web_facts),
        )
        answer = _format_market_metrics(summary, result, route.raw_message or "", route.tickers, language)
        answer = _append_metric_conflict_note(answer, summary, result, route.raw_message or "", route.tickers, web_facts, language=language)
        answer = _append_external_backdrop(answer, route, web_facts, language=language)
        answer = _append_source_priority(answer, route, language=language)
        answer = _append_source_summary(answer, route, web_facts, language=language)
        answer = _append_source_attribution(answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
        answer = _append_exploratory_tail(answer, route, summary, result, language)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = summary
        return answer, state

    def _handle_groq_status(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        groq_router = self.router.groq_router
        answer = _format_groq_status(
            groq_router.enabled,
            groq_router.model,
            groq_router.base_url,
            language,
        )
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_time_status(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        answer = _format_time_status(language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_source_scope(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        answer = _format_source_scope(state, self.web_retriever, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_web_status(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        provider_hint = _resolve_web_provider_hint(route.raw_message or "", state)
        answer = _format_web_status(self.web_retriever.config_status().to_dict(), language)
        answer = _append_web_provider_catalog(answer, provider_hint, route.raw_message or "", language)
        answer = _append_web_setup_snippet(answer, provider_hint, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_assistant_scorecard(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        from assistant.scorecard import build_assistant_scorecard

        provider_hint = _resolve_web_provider_hint(route.raw_message or "", state)
        report = build_assistant_scorecard().to_dict()
        answer = _format_assistant_scorecard(report, language, provider_hint=provider_hint)
        if _wants_scorecard_layer_breakdown(route.raw_message or ""):
            answer = _append_scorecard_layer_breakdown(answer, report, language)
        if _wants_scorecard_web_impact(route.raw_message or ""):
            answer = _append_scorecard_web_impact(answer, report, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_session_status(
        self,
        state: AssistantState,
        route: AssistantRoute,
        *,
        detailed: bool = False,
    ) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or state.last_summary or {}
        result = bundle.get("result") or {}
        context_snapshot = _session_context_snapshot(summary, state, language, result=result, error=error)
        include_context_line = any(context_snapshot[key] != "n/a" for key in ("run_id", "symbol", "stage_key"))
        context_line = (
            _format_session_context_line(summary, state, language, result=result, error=error)
            if include_context_line
            else ""
        )
        if error and not bundle.get("summary") and not bundle.get("result"):
            answer = _format_run_failure(bundle, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        if not summary and result:
            summary = _summary_from_result(result)
        web_facts = self._collect_web_facts(route)
        local_decision = str((summary.get("models") or {}).get("final_decision") or result.get("final_decision") or "")
        route.certainty = _resolve_dynamic_certainty(route, web_facts, conflict=_has_decision_conflict(local_decision, web_facts))
        local_answer = _format_session_status(summary, state, language, self.router.groq_router.enabled, detailed=detailed)
        if self.router.groq_router.enabled and _should_semantic_rewrite(route):
            groq_answer = self.router.synthesize(
                route.raw_message
                or (
                    "show detailed session status with mode, Groq state, confidence, selection, and next agent"
                    if detailed
                    else "show current mode, Groq brain state, and confidence"
                ),
                {
                    "summary": summary,
                    "result": result,
                    "state": state.to_dict(),
                    "language": language,
                    "session_status": local_answer,
                    "session_status_variant": "full" if detailed else "compact",
                },
                local_answer,
            )
            if groq_answer and groq_answer.strip() and groq_answer.strip() != local_answer.strip():
                groq_answer = groq_answer.strip()
                title = "Estado de sesión completo" if detailed and _is_spanish(language) else (
                    "Session status full" if detailed else ("Estado de sesión" if _is_spanish(language) else "Session status")
                )
                if detailed:
                    if not groq_answer.startswith(title):
                        groq_answer = f"{title}\n{groq_answer}"
                    elif "\n" not in groq_answer:
                        remainder = groq_answer[len(title):].strip()
                        groq_answer = f"{title}\n{remainder}" if remainder else title
                    if context_line and context_line not in groq_answer:
                        groq_lines = groq_answer.splitlines()
                        if groq_lines:
                            if groq_lines[0].strip() != title:
                                groq_lines.insert(0, title)
                            if len(groq_lines) == 1:
                                groq_lines.append(context_line)
                            elif groq_lines[1] != context_line:
                                groq_lines.insert(1, context_line)
                            groq_answer = "\n".join(groq_lines)
                elif not groq_answer.startswith(title):
                    groq_answer = f"{title} {groq_answer}"
                local_answer = groq_answer
        local_answer = _append_direction_conflict_note(local_answer, local_decision, web_facts, language=language)
        local_answer = _append_external_backdrop(local_answer, route, web_facts, language=language)
        local_answer = _append_source_priority(local_answer, route, language=language)
        local_answer = _append_source_summary(local_answer, route, web_facts, language=language)
        local_answer = _append_source_attribution(local_answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
        local_answer = _append_exploratory_tail(local_answer, route, summary, result, language)
        local_answer = _prepend_response_contract(local_answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = summary or state.last_summary
        return local_answer, state

    def _handle_session_status_full(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        return self._handle_session_status(state, route, detailed=True)

    def _handle_turn_trace(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        answer = _format_turn_trace_panel(state, language, detailed=str(route.stage or "").strip() == "turn_trace_full")
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_mode_guide(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        answer = _format_mode_guide(language, route.stage or "")
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_agent_guide(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        answer = _format_agent_guide(language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_agent_card(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        answer = _format_agent_card(route.stage or "extraction", language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_decision_explanation(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        if error and not summary and not result:
            answer = _format_run_failure(bundle, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        if not summary and not result:
            return (
                _format_availability_notice(
                    title_en="Decision",
                    title_es="Decisión",
                    status_en="No runs available yet.",
                    status_es="No hay ejecuciones disponibles todavía.",
                    action_en="Start a run first, then ask why it decided that way.",
                    action_es="Inicia una corrida primero y luego pregunta por qué decidió eso.",
                    prompt_en="why did it decide that way? | what did the model predict?",
                    prompt_es="por qué decidió eso? | qué predijo el modelo?",
                    language=language,
                ),
                state,
            )
        summary = summary or _summary_from_result(result)
        web_facts = self._collect_web_facts(route)
        local_decision = str((summary.get("models") or {}).get("final_decision") or result.get("final_decision") or "")
        route.certainty = _resolve_dynamic_certainty(route, web_facts, conflict=_has_decision_conflict(local_decision, web_facts))
        answer = _format_decision_explanation(summary, result, language)
        answer = _append_direction_conflict_note(answer, local_decision, web_facts, language=language)
        answer = _append_external_backdrop(answer, route, web_facts, language=language)
        answer = _append_source_priority(answer, route, language=language)
        answer = _append_source_summary(answer, route, web_facts, language=language)
        answer = _append_source_attribution(answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
        answer = _append_exploratory_tail(answer, route, summary, result, language)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = summary
        return answer, state

    def _handle_model_variables(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        if error and not summary and not result:
            answer = _format_run_failure(bundle, language)
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            state.last_summary = {}
            return answer, state
        if not summary and not result:
            return (
                _format_availability_notice(
                    title_en="Model variables",
                    title_es="Variables de modelado",
                    status_en="No runs available yet.",
                    status_es="No hay ejecuciones disponibles todavía.",
                    action_en="Start a run first, then ask about features or transformations.",
                    action_es="Inicia una corrida primero y luego pregunta por variables o transformaciones.",
                    prompt_en="what variables did the model use? | what transformations were applied?",
                    prompt_es="qué variables usó el modelo? | qué transformaciones se aplicaron?",
                    language=language,
                ),
                state,
            )
        summary = summary or _summary_from_result(result)
        web_facts = self._collect_web_facts(route)
        route.certainty = _resolve_dynamic_certainty(route, web_facts)
        answer = _format_model_variables(summary, result, language)
        if self.router.groq_router.enabled and route.answer_mode != "strict":
            groq_answer = self.router.synthesize(
                route.raw_message or "explain the modeling variables and transformations",
                {
                    "summary": summary,
                    "result": result,
                    "language": language,
                    "model_variables_view": answer,
                },
                answer,
            )
            if groq_answer and groq_answer.strip() and groq_answer.strip() != answer.strip():
                answer = groq_answer.strip()
        answer = _ensure_model_variables_markers(answer, summary, result, language)
        answer = _append_external_backdrop(answer, route, web_facts, language=language)
        answer = _append_source_priority(answer, route, language=language)
        answer = _append_source_summary(answer, route, web_facts, language=language)
        answer = _append_source_attribution(answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
        answer = _append_exploratory_tail(answer, route, summary, result, language)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = summary
        return answer, state

    def _handle_run_comparison(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        if not route.run_id or not route.secondary_run_id:
            return (
                _format_availability_notice(
                    title_en="Run comparison",
                    title_es="Comparación de corridas",
                    status_en="Two run ids are required to compare runs.",
                    status_es="Se necesitan dos run ids para comparar corridas.",
                    action_en="Ask for example: compare run_0007 vs run_0008.",
                    action_es="Pregunta por ejemplo: compara run_0007 vs run_0008.",
                    prompt_en="compare run_0007 vs run_0008",
                    prompt_es="compara run_0007 vs run_0008",
                    language=language,
                ),
                state,
            )
        bundles = self._comparison_bundle_for_route(state, route)
        left = bundles.get("left") or {}
        right = bundles.get("right") or {}
        left_summary = left.get("summary") or {}
        left_result = left.get("result") or {}
        right_summary = right.get("summary") or {}
        right_result = right.get("result") or {}
        if not left_summary and not left_result:
            return _format_run_failure(left, language), state
        if not right_summary and not right_result:
            return _format_run_failure(right, language), state
        left_summary = left_summary or _summary_from_result(left_result)
        right_summary = right_summary or _summary_from_result(right_result)
        web_facts = self._collect_web_facts(route)
        route.certainty = _resolve_dynamic_certainty(route, web_facts)
        answer = _format_run_comparison(left_summary, left_result, right_summary, right_result, route.stage, language)
        answer = _append_external_backdrop(answer, route, web_facts, language=language)
        answer = _append_source_priority(answer, route, language=language)
        answer = _append_source_summary(answer, route, web_facts, language=language)
        answer = _append_source_attribution(answer, route, web_facts, language=language, web_enabled=self.web_retriever.enabled)
        answer = _append_exploratory_tail(answer, route, left_summary, left_result, language)
        answer = _prepend_response_contract(answer, route, language)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        state.last_summary = left_summary
        return answer, state

    def _handle_help(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        bundle = self._bundle_for_route(state, route)
        error = _bundle_error(bundle)
        summary = bundle.get("summary") or {}
        result = bundle.get("result") or {}
        raw_message = route.raw_message or ""
        wants_data_help = _wants_data_help(raw_message)
        wants_next_steps = _wants_next_step_help(raw_message) or str(getattr(route, "question_focus", "") or "") == "next_steps"
        if _last_semantic_key(state) in {
            "dataset",
            "artifact",
            "manifest",
            "target",
            "feature engineering",
            "raw column",
            "clean column",
            "column",
            "row",
            "schema",
            "model variable",
            "variable",
            "adj_close",
        } and wants_next_steps:
            wants_data_help = True
        if not summary and result:
            summary = _summary_from_result(result)
        if not _is_spanish(language):
            lines = [
                "Help hub",
                "Ask naturally with a ticker, an agent, a mode, or a question.",
                "Main routes: AAPL | MSFT | BTC-USD | ETH-USD | EURUSD=X.",
                "Agents: Extraction | Cleaning | Modeling | Orchestrator.",
                "Clean data: clean data, then symbols | metrics | row | analysis | schema.",
                "Row: one cleaned record per date and ticker. Analysis: explains that row with base fields and derived clean signals.",
                "Modes: local_only | compare-binance | --groq-brain.",
                "Response styles: interpreted | exploratory.",
                "Interpreted keeps the answer grounded in local artifacts. Exploratory can add external context and guided hypotheses.",
                "Concepts: ask what is X or what does X mean to search a definition and build context around it.",
                "Connectivity: ask web status to check whether external retrieval is configured and runtime-ready.",
                "Trace: ask trace or trace full to inspect how the last turn was interpreted.",
                "Architecture: ask assistant scorecard to see local-first vs hybrid readiness and the current runtime gap.",
                "Quick keys: 1-9 or A/B/H.",
                "Groq rewrites the reply in a more conversational voice when it is active.",
                "Ask for a plain-language explanation of extraction, cleaning, modeling, or the orchestrator when you want the deeper readout.",
            ]
            context_line = _format_session_context_line(summary, state, language, result=result, error=error)
            context_snapshot = _session_context_snapshot(summary, state, language, result=result, error=error)
            if any(context_snapshot[key] != "n/a" for key in ("run_id", "symbol", "stage_key")):
                lines.append(context_line)
            if wants_next_steps:
                lines.append(_format_next_step_guidance(summary, state, language, result=result, error=error))
            if wants_data_help:
                lines.append(_format_data_capabilities(language))
            lines.append(f"Examples: {' | '.join(_contextual_help_prompts(summary, state, language, result=result, error=error))}")
            if error and not summary and not result:
                lines.extend(
                    [
                        "Latest detected error:",
                        *_format_run_failure(bundle, language).split("\n"),
                    ]
                )
        else:
            lines = [
                "Centro de ayuda",
                "Pregunta con naturalidad usando un ticker, un agente, un modo o una duda.",
                "Rutas principales: AAPL | MSFT | BTC-USD | ETH-USD | EURUSD=X.",
                "Agentes: Extracción | Limpieza | Modelado | Orquestador.",
                "Datos limpios: clean data o datos limpios, luego symbols | metrics | row | analysis | schema.",
                "Fila: un registro limpio por fecha y ticker. Análisis: explica esa fila con señales limpias derivadas y campos base.",
                "Modos: local_only | compare-binance | --groq-brain.",
                "Estilos de respuesta: interpretado | exploratorio.",
                "Interpretado se mantiene grounded en artifacts locales. Exploratorio puede sumar contexto externo e hipótesis guiadas.",
                "Conceptos: pregunta qué es X o qué significa X para buscar una definición y armar contexto alrededor.",
                "Conectividad: pregunta estado web para ver si el retriever externo está configurado y listo en runtime.",
                "Traza: pregunta trace, traza o traza completa para ver cómo interpreté el último turno.",
                "Fuentes: pregunta de dónde proviene tu información o qué puedes buscar para ver cómo se mezclan artifacts locales, herramientas locales e internet.",
                "Arquitectura: pregunta assistant scorecard para ver local-first vs híbrido y la brecha actual de runtime.",
                "Atajos rápidos: 1-9 o A/B/H.",
                "Si Groq está activo, reescribe la respuesta con una voz más conversacional.",
                "Pide una explicación simple de extracción, limpieza, modelado o el orquestador cuando quieras la lectura profunda.",
            ]
            context_line = _format_session_context_line(summary, state, language, result=result, error=error)
            context_snapshot = _session_context_snapshot(summary, state, language, result=result, error=error)
            if any(context_snapshot[key] != "n/a" for key in ("run_id", "symbol", "stage_key")):
                lines.append(context_line)
            if wants_next_steps:
                lines.append(_format_next_step_guidance(summary, state, language, result=result, error=error))
            if wants_data_help:
                lines.append(_format_data_capabilities(language))
            lines.append(f"Ejemplos: {' | '.join(_contextual_help_prompts(summary, state, language, result=result, error=error))}")
            if error and not summary and not result:
                lines.extend(
                    [
                        "Último error detectado:",
                        *_format_run_failure(bundle, language).split("\n"),
                    ]
                )
        answer = "\n".join(lines)
        state.last_intent = route.intent
        state.last_route = route.to_dict()
        return answer, state

    def _handle_continue(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        language = _resolve_language(state, route.language)
        if state.pending_route:
            pending = AssistantRoute.from_dict(state.pending_route)
            pending.raw_message = route.raw_message
            answer, state = self._dispatch(state, pending)
            return answer, state
        if state.last_run_id:
            bundle = self._load_bundle(state.last_run_id)
            error = _bundle_error(bundle)
            summary = bundle.get("summary") or {}
            if error and not bundle.get("summary") and not bundle.get("result"):
                answer = _format_run_failure(bundle, language)
                state.last_intent = route.intent
                state.last_route = route.to_dict()
                state.last_summary = {}
                return answer, state
            if summary:
                answer = (
                    (
                        f"No hay una tarea pendiente. La última corrida completada es {summary.get('run_id')} "
                        f"para {', '.join(summary.get('tickers', []))} con decisión "
                        f"{summary.get('models', {}).get('final_decision')}."
                    )
                    if _is_spanish(language)
                    else (
                        f"No unfinished task is pending. The latest completed run is {summary.get('run_id')} "
                        f"for {', '.join(summary.get('tickers', []))} with decision "
                        f"{summary.get('models', {}).get('final_decision')}."
                    )
                )
                state.last_intent = route.intent
                state.last_route = route.to_dict()
                return answer, state
        return (
            _format_availability_notice(
                title_en="Continue",
                title_es="Continuar",
                status_en="No unfinished task is pending yet.",
                status_es="No hay una tarea pendiente todavía.",
                action_en="Start a run first, or ask for the latest status.",
                action_es="Inicia una corrida primero o pide el estado más reciente.",
                prompt_en="continue | status | what was the latest run?",
                prompt_es="continuar | estado | cuál fue la última corrida?",
                language=language,
            ),
            state,
        )

    def _dispatch(self, state: AssistantState, route: AssistantRoute) -> tuple[str, AssistantState]:
        if route.needs_follow_up:
            state.pending_route = route.to_dict()
            state.pending_task = route.follow_up_question
            state.last_intent = route.intent
            state.last_route = route.to_dict()
            return route.follow_up_question, state

        if route.intent != "continue_task":
            state.pending_route = {}
            state.pending_task = ""

        if route.intent in _RUN_INTENTS:
            return self._handle_run(state, route)
        if route.intent == "show_latest_summary":
            return self._handle_show_latest(state, route)
        if route.intent == "show_semantic_lookup":
            return self._handle_semantic_lookup(state, route)
        if route.intent == "show_source_scope":
            return self._handle_source_scope(state, route)
        if route.intent == "show_time_status":
            return self._handle_time_status(state, route)
        if route.intent == "show_extraction":
            return self._handle_stage_view(state, route, "extraction")
        if route.intent == "show_cleaning":
            return self._handle_stage_view(state, route, "cleaning")
        if route.intent == "show_prediction":
            return self._handle_stage_view(state, route, "modeling")
        if route.intent == "show_model_variables":
            return self._handle_model_variables(state, route)
        if route.intent == "show_legacy_status":
            return self._handle_stage_view(state, route, "legacy")
        if route.intent == "show_stage_brief":
            return self._handle_stage_view(state, route, "stage_brief" if route.stage != "motor" else "motor")
        if route.intent == "show_symbol_guide":
            return self._handle_stage_view(state, route, "symbols")
        if route.intent == "show_asset_used":
            return self._handle_asset_used(state, route)
        if route.intent == "show_market_type":
            return self._handle_market_type(state, route)
        if route.intent == "show_clean_data":
            return self._handle_clean_data(state, route)
        if route.intent == "show_market_metrics":
            return self._handle_market_metrics(state, route)
        if route.intent == "show_decision_explanation":
            return self._handle_decision_explanation(state, route)
        if route.intent == "show_run_comparison":
            return self._handle_run_comparison(state, route)
        if route.intent == "show_groq_status":
            return self._handle_groq_status(state, route)
        if route.intent == "show_web_status":
            return self._handle_web_status(state, route)
        if route.intent == "show_assistant_scorecard":
            return self._handle_assistant_scorecard(state, route)
        if route.intent == "show_session_status_full":
            return self._handle_session_status_full(state, route)
        if route.intent == "show_turn_trace":
            return self._handle_turn_trace(state, route)
        if route.intent == "show_session_status":
            return self._handle_session_status(state, route)
        if route.intent == "show_mode_guide":
            return self._handle_mode_guide(state, route)
        if route.intent == "show_agent_guide":
            return self._handle_agent_guide(state, route)
        if route.intent == "show_agent_card":
            return self._handle_agent_card(state, route)
        if route.intent == "set_language":
            return self._handle_set_language(state, route)
        if route.intent == "greet":
            return self._handle_greeting(state, route)
        if route.intent == "compare_sources":
            return self._handle_run(state, route)
        if route.intent == "continue_task":
            return self._handle_continue(state, route)
        if route.intent == "help":
            return self._handle_help(state, route)
        return self._handle_show_latest(state, route)

    def _refresh_entity_memory(self, state: AssistantState) -> AssistantState:
        route = AssistantRoute.from_dict(state.last_route if isinstance(state.last_route, dict) else {})
        summary = state.last_summary if isinstance(state.last_summary, dict) else {}
        focus = self._focus_from_route(route)
        ticker = ""
        if route.tickers:
            ticker = str(route.tickers[0]).strip()
        elif isinstance(summary.get("tickers"), list) and summary.get("tickers"):
            ticker = str(summary.get("tickers")[0]).strip()
        elif state.current_asset:
            ticker = str(state.current_asset).strip()
        run_id = route.run_id or state.last_run_id or str(summary.get("run_id") or "").strip() or None
        provider_hint = _extract_web_provider_hint(route.raw_message or "")
        entity_kind_map = {
            "asset_used": "symbol",
            "market_type": "market",
            "semantic_lookup": "semantic",
            "assistant_identity": "identity",
            "assistant_scorecard": "assistant_scorecard",
            "web_status": "web",
            "model_variables": "variable",
            "decision_explanation": "decision",
            "market_metrics": "metric",
            "clean_data": "row",
            "extraction": "extraction",
            "cleaning": "cleaning",
            "prediction": "prediction",
            "run_comparison": "comparison",
            "summary": "summary",
        }
        kind = entity_kind_map.get(focus, "")
        entry = {
            "focus": focus,
            "run_id": run_id,
            "secondary_run_id": route.secondary_run_id,
            "stage": route.stage or "",
            "ticker": ticker,
        }
        if provider_hint:
            entry["provider_hint"] = provider_hint
        previous_memory = state.entity_memory if isinstance(state.entity_memory, dict) else {}
        by_kind = dict(previous_memory.get("by_kind") or {}) if isinstance(previous_memory.get("by_kind"), dict) else {}
        if kind:
            by_kind[kind] = entry
        state.entity_memory = {
            "last_focus": focus,
            "last_entity_kind": kind,
            "run_id": run_id,
            "secondary_run_id": route.secondary_run_id,
            "stage": route.stage or "",
            "ticker": ticker,
            "provider_hint": provider_hint or str(previous_memory.get("provider_hint") or "").strip(),
            "by_kind": by_kind,
        }
        return state

    def _focus_from_route(self, route: AssistantRoute) -> str:
        focus_map = {
            "show_extraction": "extraction",
            "show_cleaning": "cleaning",
            "show_prediction": "prediction",
            "show_model_variables": "model_variables",
            "show_decision_explanation": "decision_explanation",
            "show_market_metrics": "market_metrics",
            "show_clean_data": "clean_data",
            "show_asset_used": "asset_used",
            "show_market_type": "market_type",
            "show_semantic_lookup": "semantic_lookup",
            "show_source_scope": "source_scope",
            "show_time_status": "time_status",
            "show_assistant_scorecard": "assistant_scorecard",
            "show_turn_trace": "turn_trace",
            "show_web_status": "web_status",
            "show_run_comparison": "run_comparison",
            "show_latest_summary": "summary",
        }
        if route.intent == "show_stage_brief":
            stage_map = {
                "extraction": "extraction",
                "cleaning": "cleaning",
                "modeling": "prediction",
                "orchestrator": "decision_explanation",
                "comparison": "run_comparison",
            }
            return stage_map.get(str(route.stage or "").strip(), "general")
        return focus_map.get(route.intent, str(route.question_focus or "").strip() or "general")

    def ask(self, message: str) -> str:
        state = load_state(self.artifact_root, self.session_id)
        interpretation = self.comp.interpret(message, state)
        plan = self.planner.plan(message, state, interpretation=interpretation)
        plan = self._augment_plan_with_search_interpretation(message, state, plan)
        answer, state = self.executor.execute(plan, state, self._dispatch)
        planned_language = str((plan.route or {}).get("language") or "").lower()
        if planned_language in {"en", "es"}:
            state.preferred_language = planned_language
        state = self._refresh_entity_memory(state)
        state.last_turn_trace = self._build_turn_trace(message, interpretation, plan, state)
        save_state(self.artifact_root, state)
        return answer

    def get_state(self) -> AssistantState:
        return load_state(self.artifact_root, self.session_id)

    def _build_turn_trace(
        self,
        message: str,
        interpretation: ConversationInterpretation,
        plan: AssistantPlan,
        state: AssistantState,
    ) -> Dict[str, Any]:
        final_route = AssistantRoute.from_dict(state.last_route if isinstance(state.last_route, dict) else plan.route)
        return {
            "raw_message": str(message or "").strip(),
            "act": str(interpretation.act or "general").strip() or "general",
            "subject": str(interpretation.subject or "").strip(),
            "canonical_query": str(interpretation.canonical_query or "").strip(),
            "route_intent": str(interpretation.route_intent or "").strip(),
            "final_intent": str(final_route.intent or plan.intent or "unknown").strip() or "unknown",
            "stage": str(final_route.stage or "").strip(),
            "question_focus": str(final_route.question_focus or "").strip(),
            "run_id": str(final_route.run_id or state.last_run_id or "").strip(),
            "secondary_run_id": str(final_route.secondary_run_id or "").strip(),
            "tickers": list(final_route.tickers or []),
            "source_policy": str(final_route.source_policy or interpretation.source_policy or "").strip(),
            "source_mode": str(final_route.source_mode or plan.source_mode or "local").strip() or "local",
            "web_required": bool(final_route.web_required or interpretation.web_required),
            "allow_run": bool(final_route.allow_run if final_route.allow_run is not None else interpretation.allow_run),
            "override_memory": bool(final_route.override_memory or interpretation.override_memory),
            "planner_risk": str(plan.risk or "low").strip() or "low",
            "planner_steps": len(plan.steps or []),
            "planner_explanation": str(plan.explanation or "").strip(),
        }
