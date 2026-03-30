from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from assistant.contracts import AssistantRoute, AssistantState

_EXPLORATORY_PATTERNS = (
    "what if",
    "could",
    "would",
    "should",
    "maybe",
    "hypothesis",
    "explore",
    "opinion",
    "recommend",
    "conviene",
    "seria mejor",
    "sería mejor",
    "hipotesis",
    "hipótesis",
    "explora",
)

_INTERPRETED_PATTERNS = (
    "why",
    "how",
    "explain",
    "interpret",
    "analyze",
    "analysis",
    "por que",
    "por qué",
    "como",
    "cómo",
    "explica",
    "interpreta",
    "analiza",
    "analisis",
    "análisis",
    "que dice",
    "qué dice",
    "meaning",
    "mean",
    "define",
    "significa",
    "significado",
)

_FOLLOW_UP_PATTERNS = (
    "complement that",
    "complement that with internet",
    "expand that",
    "expand that with internet",
    "explain that",
    "explain that better",
    "explain better",
    "tell me more",
    "go deeper",
    "dig deeper",
    "what about that",
    "what does that mean",
    "and why",
    "and that",
    "explica eso",
    "complementa eso",
    "complementa esa idea",
    "complementa con internet",
    "complementa eso con internet",
    "expande eso",
    "expande esa idea",
    "explica mejor",
    "explica mas",
    "explica más",
    "más detalle",
    "mas detalle",
    "profundiza",
    "y eso",
    "y por que",
    "y por qué",
    "qué significa eso",
    "que significa eso",
    "and then",
    "what happened there",
    "what happened next",
    "what happened there?",
    "what happened next?",
    "y entonces",
    "entonces",
    "y luego",
    "y eso",
    "eso",
    "y el mercado",
    "el mercado",
    "ese mercado",
    "este mercado",
    "that market",
    "y el activo",
    "el activo",
    "ese activo",
    "este activo",
    "ese símbolo",
    "ese simbolo",
    "este símbolo",
    "este simbolo",
    "ese ticker",
    "este ticker",
    "that asset",
    "that symbol",
    "that ticker",
    "y las columnas",
    "las columnas",
    "esa columna",
    "esta columna",
    "that column",
    "y las variables",
    "las variables",
    "esa variable",
    "esta variable",
    "that variable",
    "esa feature",
    "esta feature",
    "that feature",
    "ese término",
    "ese termino",
    "ese significado",
    "esa definición",
    "esa definicion",
    "that term",
    "that meaning",
    "that definition",
    "y las métricas",
    "y las metricas",
    "las métricas",
    "las metricas",
    "esa métrica",
    "esa metrica",
    "esta métrica",
    "esta metrica",
    "that metric",
    "y la fila",
    "la fila",
    "esa fila",
    "esta fila",
    "that row",
    "y esa comparación",
    "y esa comparacion",
    "esa comparación",
    "esa comparacion",
    "that comparison",
    "the comparison",
    "what is missing",
    "what should i configure next",
    "what should i set next",
    "and the setup",
    "the setup",
    "and the probe",
    "the probe",
    "and the layers",
    "the layers",
    "show layers",
    "show the layers",
    "what changes if i enable web",
    "if i enable web",
    "if i configure web",
    "y si configuro web",
    "si configuro web",
    "y si activo web",
    "si activo web",
    "y las capas",
    "las capas",
    "muestra las capas",
    "y el probe",
    "el probe",
    "y el comando",
    "el comando",
    "y el setup",
    "el setup",
    "qué falta",
    "que falta",
    "qué configuro",
    "que configuro",
    "y esa corrida",
    "esa corrida",
    "esta corrida",
    "esa etapa",
    "esta etapa",
    "ese proceso",
    "este proceso",
    "ese paso",
    "este paso",
    "esa parte",
    "esta parte",
    "esa parte del proceso",
    "esta parte del proceso",
    "ese paso del proceso",
    "este paso del proceso",
    "that stage",
    "this stage",
    "that process",
    "this process",
    "that step",
    "this step",
    "that part",
    "this part",
    "lo de extracción",
    "lo de extraccion",
    "lo de la extracción",
    "lo de la extraccion",
    "la parte de extracción",
    "la parte de extraccion",
    "esa extracción",
    "esa extraccion",
    "esta extracción",
    "esta extraccion",
    "that extraction",
    "this extraction",
    "lo de limpieza",
    "lo de la limpieza",
    "la parte de limpieza",
    "esa limpieza",
    "esta limpieza",
    "that cleaning",
    "this cleaning",
    "lo de modelado",
    "lo del modelado",
    "la parte de modelado",
    "el paso del modelado",
    "eso de modelado",
    "ese modelado",
    "este modelado",
    "that modeling",
    "this modeling",
    "lo del orquestador",
    "esa orquestación",
    "esa orquestacion",
    "este orquestador",
    "ese orquestador",
    "that orchestrator",
    "that orchestration",
    "esa ejecución",
    "esa ejecucion",
    "esta ejecución",
    "esta ejecucion",
    "that run",
    "the run",
    "ese resultado",
    "este resultado",
    "that result",
    "the result",
    "y la decisión",
    "y la decision",
    "la decisión",
    "la decision",
    "esa decisión",
    "esa decision",
    "esta decisión",
    "esta decision",
    "that decision",
    "qué pasó ahí",
    "que paso ahi",
    "ahi que paso",
    "ahí que pasó",
    "qué pasó",
    "que paso",
    "so",
    "so?",
    "then",
    "then?",
)

_CAUSAL_FOLLOW_UP_PATTERNS = (
    "and then",
    "then",
    "what happened there",
    "what happened next",
    "y entonces",
    "entonces",
    "y luego",
    "y eso",
    "qué pasó ahí",
    "que paso ahi",
    "ahi que paso",
    "ahí que pasó",
    "qué pasó",
    "que paso",
    "so",
)

_PIPELINE_PROGRESS_FOLLOW_UP_PATTERNS = (
    "and then",
    "and then?",
    "then",
    "then?",
    "y luego",
    "y luego?",
    "entonces",
    "entonces?",
)


def _looks_like_next_step_request(normalized: str) -> bool:
    return bool(
        re.search(
            r"\b(what should i do next|what do i do next|what can i do next|what now|and now|what do you recommend now|que hago ahora|qué hago ahora|ahora que sigue|ahora qué sigue|que me recomiendas ahora|qué me recomiendas ahora)\b",
            normalized,
        )
    )


@dataclass
class AssistantContextSnapshot:
    run_id: Optional[str] = None
    secondary_run_id: Optional[str] = None
    stage: Optional[str] = None
    ticker: str = ""
    current_mode: str = ""
    question_focus: str = "general"
    answer_mode: str = "strict"
    certainty: str = "confirmed"
    available_sections: list[str] = field(default_factory=list)
    raw_column_count: int = 0
    feature_count: int = 0
    target_column: str = ""
    selected_model: str = ""
    final_decision: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AssistantContextResolver:
    """Loads compact run/session context before final response planning."""

    def __init__(self, artifact_root: str = "artifacts") -> None:
        self.artifact_root = artifact_root

    def resolve(self, message: str, state: AssistantState, route: AssistantRoute) -> AssistantContextSnapshot:
        normalized = self._normalize(message)
        run_ids = self._extract_run_ids(message)
        last_route = AssistantRoute.from_dict(state.last_route if isinstance(state.last_route, dict) else {})
        question_focus = self._infer_question_focus(normalized, route)
        question_focus = self._infer_follow_up_focus(normalized, state, route, last_route, question_focus)
        memory_entry = self._memory_entry_for_focus(question_focus, state.entity_memory)
        run_id = (
            (run_ids[0] if run_ids else None)
            or route.run_id
            or memory_entry.get("run_id")
            or last_route.run_id
            or self._summary_run_id(state.last_summary)
            or state.last_run_id
        )
        secondary_run_id = (
            (run_ids[1] if len(run_ids) > 1 else None)
            or route.secondary_run_id
            or memory_entry.get("secondary_run_id")
            or last_route.secondary_run_id
        )
        bundle = self._load_bundle(run_id) if run_id else {"summary": {}, "result": {}, "manifest": {}}
        summary = bundle.get("summary") or state.last_summary or {}
        result = bundle.get("result") or {}
        manifest = bundle.get("manifest") or {}
        stage = route.stage or self._infer_requested_stage(normalized) or self._infer_stage(route.intent, question_focus)
        answer_mode = self._infer_answer_mode(normalized, route.intent, question_focus)
        certainty = self._infer_certainty(answer_mode, summary, result, manifest, question_focus=question_focus)

        data = summary.get("data") or {}
        models = summary.get("models") or {}
        modeling = result.get("modeling") or {}
        request = manifest.get("request") or {}
        tickers = (
            summary.get("tickers")
            or request.get("tickers")
            or route.tickers
            or ([memory_entry.get("ticker")] if memory_entry.get("ticker") else [])
            or last_route.tickers
        )
        raw_columns = data.get("raw_columns") or []
        feature_columns = data.get("feature_columns") or []

        return AssistantContextSnapshot(
            run_id=run_id or summary.get("run_id") or result.get("run_id"),
            secondary_run_id=secondary_run_id,
            stage=stage,
            ticker=str(tickers[0]).strip() if tickers else (state.current_asset or ""),
            current_mode=state.current_mode or str(summary.get("run_mode") or ""),
            question_focus=question_focus,
            answer_mode=answer_mode,
            certainty=certainty,
            available_sections=[name for name, payload in bundle.items() if payload],
            raw_column_count=len(raw_columns),
            feature_count=len(feature_columns),
            target_column=str(data.get("target_column") or "target_direction"),
            selected_model=str(modeling.get("selected_model") or models.get("selected") or ""),
            final_decision=str(models.get("final_decision") or result.get("final_decision") or ""),
        )

    def refine_route(self, route: AssistantRoute, context: AssistantContextSnapshot) -> AssistantRoute:
        refined = AssistantRoute.from_dict(route.to_dict())
        explicit_run_ids = self._extract_run_ids(refined.raw_message or "")
        if explicit_run_ids and refined.intent not in {"run_pipeline", "compare_sources"}:
            refined.run_id = explicit_run_ids[0]
            if len(explicit_run_ids) > 1:
                refined.secondary_run_id = explicit_run_ids[1]
        elif not refined.run_id and context.run_id and refined.intent not in {"run_pipeline", "compare_sources"}:
            refined.run_id = context.run_id
        if not refined.secondary_run_id and context.secondary_run_id:
            refined.secondary_run_id = context.secondary_run_id
        if not refined.tickers and context.ticker:
            refined.tickers = [context.ticker]
        if not refined.stage and context.stage:
            refined.stage = context.stage
        if context.question_focus == "run_comparison":
            refined.intent = "show_run_comparison"
        overridable_intents = {
            "unknown",
            "help",
            "show_help",
            "explanation_request",
            "show_latest_summary",
            "show_extraction",
            "show_cleaning",
            "show_prediction",
            "show_model_variables",
            "show_clean_data",
            "show_market_metrics",
            "show_decision_explanation",
            "show_stage_brief",
            "show_run_comparison",
            "show_asset_used",
            "show_session_status",
        }
        if refined.intent in overridable_intents:
            refined.intent = self._focus_to_intent(context.question_focus, refined.intent)
        if refined.intent == "show_prediction" and not refined.stage:
            refined.stage = "modeling"
        if refined.intent == "show_model_variables" and not refined.stage:
            refined.stage = "modeling"
        if refined.intent == "show_decision_explanation" and not refined.stage:
            refined.stage = "orchestrator"
        if refined.intent == "show_run_comparison" and not refined.stage:
            refined.stage = "comparison"
        if refined.intent == "show_clean_data" and not refined.stage:
            refined.stage = "cleaning"
        refined.answer_mode = context.answer_mode
        refined.certainty = context.certainty
        refined.question_focus = context.question_focus
        return refined

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _load_bundle(self, run_id: str) -> Dict[str, Dict[str, Any]]:
        run_dir = Path(self.artifact_root) / "runs" / run_id
        if not run_dir.exists():
            return {"summary": {}, "result": {}, "manifest": {}}
        return {
            "summary": self._read_json(run_dir / "summary.json"),
            "result": self._read_json(run_dir / "result.json"),
            "manifest": self._read_json(run_dir / "manifest.json"),
        }

    def _infer_question_focus(self, normalized: str, route: AssistantRoute) -> str:
        if len(self._extract_run_ids(normalized)) >= 2 and re.search(r"\b(compare|comparison|contrast|vs|versus|differences?|diferencia|compara|comparar|comparación)\b", normalized):
            return "run_comparison"
        if _looks_like_next_step_request(normalized):
            return "next_steps"
        if route.intent == "show_semantic_lookup":
            return "semantic_lookup"
        if route.intent == "show_source_scope":
            return "source_scope"
        if route.intent == "show_time_status":
            return "time_status"
        if route.intent == "show_identity":
            return "assistant_identity"
        if re.search(r"\b(what day is it|what date is it|what time is it|today's date|current date|current time|fecha correcta|fecha actual|hora actual|qué día es|que día es|qué dia es|que dia es|qué fecha es|que fecha es|qué hora es|que hora es)\b", normalized):
            return "time_status"
        if re.search(r"\b(where does your information come from|where does your info come from|where do your data come from|what sources do you use|what can you search|what can you look up|qué puedes buscar|que puedes buscar|de dónde proviene tu información|de donde proviene tu informacion|de dónde vienen tus datos|de donde vienen tus datos|qué fuentes usas|que fuentes usas)\b", normalized):
            return "source_scope"
        if re.search(r"\b(market type|tipo de mercado|asset class|asset type|clase de activo|tipo de activo)\b", normalized):
            return "market_type"
        if re.search(r"\b(scorecard|architecture score|architecture status|maturity|porcentaje de arquitectura|madurez de arquitectura)\b", normalized):
            return "assistant_scorecard"
        if re.search(r"\b(web status|internet status|retriever status|web config|internet config|estado web|estado de internet|estado del retriever|config del retriever)\b", normalized):
            return "web_status"
        if re.search(r"\b(extraction|extraccion|extracción|extract|extracted|columns were extracted|columnas se extrajeron)\b", normalized):
            return "extraction"
        if re.search(r"\b(variable|variables|feature|features|transform|transformation|transformacion|transformación|column|columns|columna|columnas)\b", normalized):
            return "model_variables"
        if re.search(r"\b(why|decision|decided|por que|por qué|decidio|decidió|explica.*decision|explica.*decisión)\b", normalized):
            return "decision_explanation"
        if re.search(r"\b(metric|metrics|volume|volumen|close|open|high|low|precio)\b", normalized):
            return "market_metrics"
        if re.search(r"\b(row|fila|schema|esquema|clean data|datos limpios|analiza la fila|analysis)\b", normalized):
            return "clean_data"
        if re.search(r"\b(cleaning|limpieza)\b", normalized):
            return "cleaning"
        if re.search(r"\b(modeling|modelado|prediction|predic|predijo|predicted)\b", normalized):
            return "prediction"
        if route.intent in {
            "show_model_variables",
            "show_decision_explanation",
            "show_clean_data",
            "show_market_metrics",
            "show_cleaning",
            "show_prediction",
            "show_asset_used",
            "show_market_type",
            "show_semantic_lookup",
            "show_identity",
            "show_assistant_scorecard",
            "show_web_status",
            "show_source_scope",
            "show_time_status",
        }:
            return route.intent.removeprefix("show_")
        return "general"

    def _infer_follow_up_focus(
        self,
        normalized: str,
        state: AssistantState,
        route: AssistantRoute,
        last_route: AssistantRoute,
        question_focus: str,
    ) -> str:
        entity_memory_focus = self._infer_entity_memory_follow_up_focus(normalized, state.entity_memory)
        if entity_memory_focus != "general":
            return entity_memory_focus
        if _looks_like_next_step_request(normalized):
            return "next_steps"
        if not any(pattern in normalized for pattern in _FOLLOW_UP_PATTERNS):
            return question_focus
        explanation_like = any(pattern in normalized for pattern in _INTERPRETED_PATTERNS)
        causal_like = any(pattern in normalized for pattern in _CAUSAL_FOLLOW_UP_PATTERNS)
        last_intent = str(state.last_intent or "").strip()
        if last_intent == "show_assistant_scorecard":
            scorecard_focus = self._infer_scorecard_follow_up_focus(normalized)
            if scorecard_focus != "general":
                return scorecard_focus
        progress_focus = self._infer_pipeline_progress_follow_up_focus(normalized, last_intent)
        if progress_focus != "general":
            return progress_focus
        elliptical_focus = self._infer_elliptical_follow_up_focus(normalized, last_intent, last_route.stage)
        if elliptical_focus != "general":
            return elliptical_focus
        if question_focus != "general":
            return question_focus
        mapping = {
            "show_model_variables": "model_variables",
            "show_decision_explanation": "decision_explanation",
            "show_clean_data": "clean_data",
            "show_market_metrics": "market_metrics",
            "show_cleaning": "cleaning",
            "show_extraction": "extraction",
            "show_asset_used": "asset_used",
            "show_market_type": "market_type",
            "show_semantic_lookup": "semantic_lookup",
            "show_identity": "assistant_identity",
            "show_assistant_scorecard": "assistant_scorecard",
            "show_web_status": "web_status",
            "show_source_scope": "source_scope",
            "show_time_status": "time_status",
            "show_prediction": "decision_explanation" if explanation_like or causal_like else "prediction",
            "show_stage_brief": self._focus_from_stage(last_route.stage),
            "show_session_status": "decision_explanation" if causal_like else "general",
            "show_latest_summary": "decision_explanation" if explanation_like or causal_like else "general",
        }
        return mapping.get(last_intent, question_focus)

    def _infer_pipeline_progress_follow_up_focus(self, normalized: str, last_intent: str) -> str:
        compact = re.sub(r"[?.!,;:]+$", "", normalized).strip()
        if compact not in _PIPELINE_PROGRESS_FOLLOW_UP_PATTERNS:
            return "general"
        stage_map = {
            "show_extraction": "cleaning",
            "show_cleaning": "prediction",
            "show_prediction": "decision_explanation",
            "show_model_variables": "decision_explanation",
        }
        return stage_map.get(last_intent, "general")

    def _infer_elliptical_follow_up_focus(self, normalized: str, last_intent: str, last_stage: Optional[str]) -> str:
        if re.search(r"\b(término|termino|term|meaning|significado|definition|definición|definicion)\b", normalized):
            return "semantic_lookup"
        if last_intent == "show_semantic_lookup" and re.search(
            r"\b(complement|complementa|expand|expande|internet|web|search|buscar|contexto externo|external context)\b",
            normalized,
        ):
            return "semantic_lookup"
        if re.search(r"\b(fecha|hora|day|date|time)\b", normalized):
            return "time_status"
        if re.search(r"\b(fuente|fuentes|source|sources|internet|web|buscar|search)\b", normalized):
            return "source_scope"
        if re.search(r"\b(comparación|comparacion|comparison)\b", normalized):
            return "run_comparison"
        if re.search(r"\b(quién eres|quien eres|who are you|what is yfinance|what's yfinance|introduce yourself|present yourself|tell me about yourself|about yourself)\b", normalized):
            return "assistant_identity"
        if re.search(r"\b(extracción|extraccion|extraction|extract)\b", normalized):
            return "extraction"
        if re.search(r"\b(limpieza|cleaning|clean)\b", normalized):
            return "cleaning"
        if re.search(r"\b(modelado|modeling)\b", normalized):
            return "prediction"
        if re.search(r"\b(orquestador|orchestrator|orquestación|orquestacion|orchestration)\b", normalized):
            return "decision_explanation"
        if re.search(r"\b(corrida|ejecución|ejecucion|run)\b", normalized) and last_intent == "show_run_comparison":
            return "run_comparison"
        if re.search(r"\b(corrida|ejecución|ejecucion|run|resultado|result)\b", normalized):
            result_like_map = {
                "show_latest_summary": "summary",
                "show_session_status": "summary",
                "show_extraction": "extraction",
                "show_cleaning": "cleaning",
                "show_prediction": "prediction",
                "show_stage_brief": self._focus_from_stage(last_stage),
            }
            mapped = result_like_map.get(last_intent)
            if mapped:
                return mapped
        if re.search(r"\b(etapa|stage|proceso|process|paso|step|parte|part)\b", normalized):
            stage_like_map = {
                "show_extraction": "extraction",
                "show_cleaning": "cleaning",
                "show_prediction": "prediction",
                "show_model_variables": "model_variables",
                "show_clean_data": "clean_data",
                "show_market_metrics": "market_metrics",
                "show_decision_explanation": "decision_explanation",
                "show_stage_brief": self._focus_from_stage(last_stage),
            }
            mapped = stage_like_map.get(last_intent)
            if mapped:
                return mapped
        if re.search(r"\b(mercado|market)\b", normalized):
            return "market_type"
        if re.search(r"\b(activo|asset|símbolo|simbolo|symbol|ticker)\b", normalized):
            return "asset_used"
        if re.search(r"\b(métrica|métricas|metrica|metricas|metric|metrics)\b", normalized):
            return "market_metrics"
        if re.search(r"\b(fila|row)\b", normalized):
            return "clean_data"
        if re.search(r"\b(variable|variables|feature|features)\b", normalized):
            return "model_variables"
        if re.search(r"\b(columna|columnas|column|columns)\b", normalized):
            return "extraction" if last_intent == "show_extraction" else "model_variables"
        if re.search(r"\b(decisión|decision)\b", normalized):
            return "decision_explanation"
        return "general"

    def _infer_scorecard_follow_up_focus(self, normalized: str) -> str:
        if re.search(
            r"\b(change|changes|impact|improve|improves|improvement|if i enable|if i configure|si configuro|si activo|qué cambia|que cambia|impacto)\b",
            normalized,
        ):
            return "assistant_scorecard"
        if re.search(r"\b(layer|layers|capa|capas)\b", normalized):
            return "assistant_scorecard"
        if re.search(
            r"\b(setup|probe|command|comando|env|provider|providers|preset|presets|web|internet|retriever|config|configure|configurar|tavily|serper|searxng|searchapi)\b",
            normalized,
        ):
            return "web_status"
        if re.search(r"\b(missing|left|gap|falta|cu[aá]nto|cuanto|score|scorecard|runtime|h[ií]brido|hibrido|hybrid)\b", normalized):
            return "assistant_scorecard"
        return "general"

    def _infer_answer_mode(self, normalized: str, intent: str, question_focus: str) -> str:
        semantic_summary_patterns = (
            "what does",
            "what is",
            "what's",
            "qué es",
            "que es",
            "meaning",
            "mean",
            "define",
            "significa",
            "significado",
        )
        if any(pattern in normalized for pattern in _EXPLORATORY_PATTERNS):
            return "exploratory"
        if intent in {"show_asset_used", "show_market_type"} or question_focus in {"market_type", "asset_used"}:
            return "strict"
        if intent == "show_semantic_lookup" or question_focus == "semantic_lookup":
            return "interpreted"
        if intent == "show_source_scope" or question_focus == "source_scope":
            return "strict"
        if intent == "show_time_status" or question_focus == "time_status":
            return "strict"
        if intent == "show_assistant_scorecard" or question_focus == "assistant_scorecard":
            return "strict"
        if intent == "show_web_status" or question_focus == "web_status":
            return "strict"
        if intent == "show_latest_summary" and not any(
            pattern in normalized
            for pattern in (
                "explain",
                "interpret",
                "analyze",
                "analysis",
                "explica",
                "interpreta",
                "analiza",
                "analisis",
                "análisis",
                "por que",
                "por qué",
                "what does that mean",
                "what does",
                "what is",
                "qué es",
                "que es",
                "meaning",
                "mean",
                "define",
                "significa",
                "significado",
            )
        ):
            return "strict"
        if intent == "show_latest_summary" and any(pattern in normalized for pattern in semantic_summary_patterns):
            return "interpreted"
        if any(pattern in normalized for pattern in _INTERPRETED_PATTERNS):
            return "interpreted"
        if question_focus in {"decision_explanation", "clean_data"} and normalized:
            return "interpreted"
        if intent in {"show_decision_explanation"}:
            return "interpreted"
        if intent == "show_identity" or question_focus == "assistant_identity":
            return "strict"
        return "strict"

    def _infer_certainty(
        self,
        answer_mode: str,
        summary: Dict[str, Any],
        result: Dict[str, Any],
        manifest: Dict[str, Any],
        *,
        question_focus: str = "general",
    ) -> str:
        if question_focus in {"assistant_identity", "time_status", "source_scope"}:
            return "confirmed"
        has_artifacts = bool(summary or result or manifest)
        if not has_artifacts:
            return "hypothesis" if answer_mode == "exploratory" else "inferred"
        if answer_mode == "strict":
            return "confirmed"
        if answer_mode == "interpreted":
            return "inferred"
        return "hypothesis"

    def _infer_stage(self, intent: str, question_focus: str) -> Optional[str]:
        stage_map = {
            "show_extraction": "extraction",
            "show_latest_summary": "summary",
            "help": "help",
            "show_cleaning": "cleaning",
            "show_clean_data": "cleaning",
            "show_market_metrics": "cleaning",
            "show_market_type": "market_type",
            "show_semantic_lookup": "semantic_lookup",
            "show_source_scope": "source_scope",
            "show_time_status": "time_status",
            "show_identity": "assistant_identity",
            "show_assistant_scorecard": "assistant_scorecard",
            "show_web_status": "web_status",
            "show_asset_used": "asset",
            "show_prediction": "modeling",
            "show_model_variables": "modeling",
            "show_decision_explanation": "orchestrator",
            "prediction": "modeling",
            "summary": "summary",
            "next_steps": "help",
            "cleaning": "cleaning",
            "clean_data": "cleaning",
            "market_metrics": "cleaning",
            "market_type": "market_type",
            "semantic_lookup": "semantic_lookup",
            "assistant_identity": "identity",
            "assistant_scorecard": "assistant_scorecard",
            "web_status": "web_status",
            "asset_used": "asset",
            "model_variables": "modeling",
            "decision_explanation": "orchestrator",
        }
        return stage_map.get(intent) or stage_map.get(question_focus)

    def _focus_to_intent(self, question_focus: str, fallback_intent: str) -> str:
        focus_map = {
            "extraction": "show_extraction",
            "run_comparison": "show_run_comparison",
            "summary": "show_latest_summary",
            "next_steps": "help",
            "market_type": "show_market_type",
            "semantic_lookup": "show_semantic_lookup",
            "source_scope": "show_source_scope",
            "time_status": "show_time_status",
            "assistant_identity": "show_identity",
            "assistant_scorecard": "show_assistant_scorecard",
            "web_status": "show_web_status",
            "asset_used": "show_asset_used",
            "model_variables": "show_model_variables",
            "decision_explanation": "show_decision_explanation",
            "clean_data": "show_clean_data",
            "market_metrics": "show_market_metrics",
            "cleaning": "show_cleaning",
            "prediction": "show_prediction",
        }
        return focus_map.get(question_focus, fallback_intent)

    def _focus_from_stage(self, stage: Optional[str]) -> str:
        stage_map = {
            "extraction": "extraction",
            "cleaning": "cleaning",
            "modeling": "prediction",
            "orchestrator": "decision_explanation",
            "comparison": "run_comparison",
            "semantic_lookup": "semantic_lookup",
            "identity": "assistant_identity",
        }
        return stage_map.get(str(stage or "").strip(), "general")

    def _infer_entity_memory_follow_up_focus(self, normalized: str, entity_memory: Any) -> str:
        if not isinstance(entity_memory, dict):
            return "general"
        compact = re.sub(r"[?.!,;:]+$", "", normalized).strip()
        if compact not in {"esa", "ese", "eso", "esta", "este", "that", "that one", "this", "this one"}:
            return "general"
        focus = str(entity_memory.get("last_focus") or "").strip()
        allowed = {
            "asset_used",
            "market_type",
            "assistant_scorecard",
            "semantic_lookup",
            "assistant_identity",
            "model_variables",
            "decision_explanation",
            "market_metrics",
            "clean_data",
            "extraction",
            "cleaning",
            "prediction",
            "run_comparison",
            "summary",
        }
        return focus if focus in allowed else "general"

    def _memory_entry_for_focus(self, focus: str, entity_memory: Any) -> Dict[str, Any]:
        if not isinstance(entity_memory, dict):
            return {}
        kind_map = {
            "asset_used": ["symbol", "market", "summary"],
            "market_type": ["market", "symbol", "summary"],
            "model_variables": ["variable", "prediction", "summary"],
            "decision_explanation": ["decision", "prediction", "summary"],
            "market_metrics": ["metric", "row", "summary"],
            "clean_data": ["row", "metric", "summary"],
            "extraction": ["extraction", "summary"],
            "cleaning": ["cleaning", "row", "summary"],
            "prediction": ["prediction", "variable", "decision", "summary"],
            "run_comparison": ["comparison"],
            "summary": ["summary"],
            "semantic_lookup": ["semantic", "market", "symbol", "summary"],
            "assistant_identity": ["identity", "summary"],
        }
        by_kind = entity_memory.get("by_kind")
        if not isinstance(by_kind, dict):
            return {}
        for key in kind_map.get(focus, []):
            entry = by_kind.get(key)
            if isinstance(entry, dict) and entry:
                return entry
        return {}

    def _normalize(self, message: str) -> str:
        return re.sub(r"\s+", " ", str(message or "").strip().lower())

    def _extract_run_ids(self, message: str) -> list[str]:
        matches = re.findall(r"\brun_\d+\b", str(message or "").lower())
        ordered: list[str] = []
        for item in matches:
            if item not in ordered:
                ordered.append(item)
        return ordered

    def _infer_requested_stage(self, normalized: str) -> Optional[str]:
        if re.search(r"\b(extraction|extraccion|extracción|extract)\b", normalized):
            return "extraction"
        if re.search(r"\b(cleaning|limpieza|clean data|datos limpios|fila|row|schema|esquema)\b", normalized):
            return "cleaning"
        if re.search(r"\b(modeling|modelado|prediction|predicted|predijo)\b", normalized):
            return "modeling"
        if re.search(r"\b(orchestrator|orquestador|decision|decisión|brief)\b", normalized):
            return "orchestrator"
        return None

    def _summary_run_id(self, summary: Dict[str, Any]) -> Optional[str]:
        if not isinstance(summary, dict):
            return None
        run_id = summary.get("run_id")
        text = str(run_id or "").strip()
        return text or None
