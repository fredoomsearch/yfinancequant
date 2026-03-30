from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from assistant import comp, context, contracts, grounding, planner, policy, router, runtime, source_selector, web


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LayerCheck:
    name: str
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerScore:
    key: str
    label: str
    score_pct: int
    implementation_score_pct: int
    runtime_ready_score_pct: int
    passed_checks: int
    total_checks: int
    checks: List[LayerCheck] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "score_pct": self.score_pct,
            "implementation_score_pct": self.implementation_score_pct,
            "runtime_ready_score_pct": self.runtime_ready_score_pct,
            "passed_checks": self.passed_checks,
            "total_checks": self.total_checks,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass
class AssistantScorecard:
    generated_at: str
    local_first_score_pct: int
    hybrid_score_pct: int
    local_first_implementation_score_pct: int
    local_first_runtime_ready_score_pct: int
    hybrid_implementation_score_pct: int
    hybrid_runtime_ready_score_pct: int
    runtime: Dict[str, Any]
    layers: Dict[str, LayerScore]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "local_first_score_pct": self.local_first_score_pct,
            "hybrid_score_pct": self.hybrid_score_pct,
            "local_first_implementation_score_pct": self.local_first_implementation_score_pct,
            "local_first_runtime_ready_score_pct": self.local_first_runtime_ready_score_pct,
            "hybrid_implementation_score_pct": self.hybrid_implementation_score_pct,
            "hybrid_runtime_ready_score_pct": self.hybrid_runtime_ready_score_pct,
            "runtime": self.runtime,
            "layers": {key: layer.to_dict() for key, layer in self.layers.items()},
        }


def _source(module: Any) -> str:
    return inspect.getsource(module)


def _check(name: str, condition: bool) -> LayerCheck:
    return LayerCheck(name=name, passed=bool(condition))


def _build_layer(
    key: str,
    label: str,
    checks: List[LayerCheck],
    *,
    runtime_ready_cap: int | None = None,
) -> LayerScore:
    passed = sum(1 for check in checks if check.passed)
    total = len(checks) or 1
    implementation_score = round((passed / total) * 100)
    runtime_ready_score = min(implementation_score, runtime_ready_cap) if runtime_ready_cap is not None else implementation_score
    return LayerScore(
        key=key,
        label=label,
        score_pct=runtime_ready_score,
        implementation_score_pct=implementation_score,
        runtime_ready_score_pct=runtime_ready_score,
        passed_checks=passed,
        total_checks=total,
        checks=checks,
    )


def build_assistant_scorecard() -> AssistantScorecard:
    context_src = _source(context)
    comp_src = _source(comp)
    runtime_src = _source(runtime)
    router_src = _source(router)
    selector_src = _source(source_selector)
    web_src = _source(web)
    contracts_src = _source(contracts)
    grounding_src = _source(grounding)
    planner_src = _source(planner)
    policy_src = _source(policy)

    visible_mode_check = (
        "_visible_answer_mode" in runtime_src
        and '"exploratory"' in runtime_src
        and '"interpreted"' in runtime_src
        and ('answer_mode: str = "strict"' in contracts_src or 'answer_mode: str = "strict"' in context_src)
    )

    semantic_checks = [
        _check("vague_summary_phrases", "qué pasa hoy" in router_src and "how's it going" in router_src),
        _check("causal_followups", "y entonces" in context_src and "what happened next" in context_src),
        _check("entity_followups", "ese símbolo" in context_src and "esa métrica" in context_src and "esa decisión" in context_src),
        _check("stage_process_followups", "esa parte del proceso" in context_src and "el paso del modelado" in context_src),
        _check("comparison_followups", "that comparison" in context_src and "run_comparison" in context_src),
        _check("bare_reference_resolution", "_infer_entity_memory_follow_up_focus" in context_src),
        _check("stage_brief_resolution", "show_stage_brief" in context_src and "_focus_from_stage" in context_src),
        _check("answer_modes", visible_mode_check),
    ]

    conversational_checks = [
        _check("time_status_intent", "show_time_status" in router_src and "_looks_like_time_status_request" in router_src),
        _check("source_scope_intent", "show_source_scope" in router_src and "_looks_like_source_scope_request" in router_src),
        _check("pending_route_reset", 'if route.intent != "continue_task"' in runtime_src and "state.pending_route = {}" in runtime_src),
        _check("semantic_followup_basis", "_semantic_basis_message" in runtime_src and "_looks_like_semantic_follow_up" in runtime_src),
        _check("data_help_capabilities", "_wants_data_help" in runtime_src and "_format_data_capabilities" in runtime_src),
        _check("agent_guide_stage_roles", "Extraction gets the symbol, range, and raw columns." in runtime_src),
        _check("follow_up_patterns", "complement that with internet" in context_src and "qué hago con eso" in router_src and "what should i do next" in router_src),
        _check(
            "semantic_architecture_terms",
            "artifact store" in router_src
            and "evidence ledger" in router_src
            and "conversational layer" in router_src
            and "promotion gate" in router_src
            and "challenger" in router_src,
        ),
    ]

    comp_checks = [
        _check("comp_agent_present", "class AssistantCompAgent" in comp_src),
        _check("conversation_contract", "ConversationInterpretation" in comp_src and "conversation_act: str = " in contracts_src),
        _check("planner_integration", "_apply_interpretation" in planner_src and "conversation_act" in planner_src),
        _check("runtime_hook", "self.comp = AssistantCompAgent()" in runtime_src and "interpretation = self.comp.interpret" in runtime_src),
        _check("state_turn_trace", "last_turn_trace" in contracts_src and "state.last_turn_trace" in runtime_src),
        _check("trace_panel", "_format_turn_trace_panel" in runtime_src and "show_turn_trace" in runtime_src),
        _check("policy_allow_run_guardrail", "allow_run" in policy_src and "non-executable" in policy_src),
        _check("explicit_web_policy", "web_required" in comp_src and "source_policy" in comp_src),
        _check("comparison_guardrails", "_comparison_stage_from_message" in comp_src and "_semantic_comparison_subjects" in comp_src),
        _check("help_and_mode_overrides", "_HELP_HINTS" in comp_src and "_MODE_HUB_LABELS" in comp_src),
    ]

    memory_checks = [
        _check("entity_memory_field", "entity_memory" in contracts_src),
        _check("entity_memory_refresh", "_refresh_entity_memory" in runtime_src),
        _check("entity_memory_by_kind", '"by_kind"' in runtime_src),
        _check("memory_entry_resolution", "_memory_entry_for_focus" in context_src),
        _check("cross_kind_fallback", '"asset_used": ["symbol", "market", "summary"]' in context_src),
        _check("stage_memory", "last_route.stage" in context_src),
        _check("secondary_run_memory", "secondary_run_id" in runtime_src and "secondary_run_id" in context_src),
        _check("bare_reference_tokens", '"that one"' in context_src and '"esa"' in context_src),
    ]

    local_fact_checks = [
        _check("bundle_loader", "_load_bundle" in context_src),
        _check("route_bundle_loader", "_bundle_for_route" in runtime_src),
        _check("comparison_bundle_loader", "_comparison_bundle_for_route" in runtime_src),
        _check("grounding_packet", "build_grounding_packet" in grounding_src),
        _check("stage_handlers", "show_extraction" in runtime_src and "show_cleaning" in runtime_src and "show_prediction" in runtime_src),
        _check("modeling_handlers", "show_model_variables" in runtime_src and "show_decision_explanation" in runtime_src),
        _check("clean_data_handlers", "show_clean_data" in runtime_src and "show_market_metrics" in runtime_src),
        _check("comparison_handler", "show_run_comparison" in runtime_src),
    ]

    grounded_checks = [
        _check("response_contract", "_prepend_response_contract" in runtime_src),
        _check("dynamic_certainty", "_resolve_dynamic_certainty" in runtime_src),
        _check("source_priority", "_append_source_priority" in runtime_src),
        _check("source_summary", "_append_source_summary" in runtime_src),
        _check("source_attribution", "_append_source_attribution" in runtime_src),
        _check("external_backdrop", "_append_external_backdrop" in runtime_src),
        _check("decision_conflict", "_append_direction_conflict_note" in runtime_src),
        _check("metric_conflict", "_append_metric_conflict_note" in runtime_src),
    ]

    selector_checks = [
        _check("selector_present", "AssistantSourceSelector" in selector_src),
        _check("web_hints", '"google"' in selector_src and '"today"' in selector_src and '"hoy"' in selector_src),
        _check("market_type_mixed", 'focus == "market_type"' in selector_src),
        _check("run_local_first", "run-specific artifact" in selector_src),
        _check("exploratory_mixed", 'answer_mode", "strict") == "exploratory"' in selector_src),
        _check("decision_queries", 'focus == "decision_explanation"' in selector_src),
        _check("variable_queries", 'focus == "model_variables"' in selector_src),
        _check("comparison_queries", 'focus == "run_comparison"' in selector_src),
    ]

    search_augmented_checks = [
        _check("interpreted_query_contract", 'interpreted_query: str = ""' in contracts_src),
        _check("interpretation_source_contract", 'interpretation_source: str = ""' in contracts_src),
        _check("interpretation_note_contract", 'interpretation_note: str = ""' in contracts_src),
        _check("canonical_interpretation_query", "_canonical_semantic_interpretation_query" in runtime_src),
        _check("interpretation_query_builder", "_interpretation_queries_for_semantic_query" in runtime_src),
        _check("local_semantic_coverage", "_has_local_semantic_coverage" in runtime_src),
        _check("plan_augmentation", "_augment_plan_with_search_interpretation" in runtime_src),
        _check("interpretation_trace", "_append_interpretation_trace" in runtime_src),
    ]

    web_retrieval_checks = [
        _check("search_url_config", "ASSISTANT_WEB_SEARCH_URL" in web_src),
        _check("method_config", "ASSISTANT_WEB_SEARCH_METHOD" in web_src),
        _check("provider_contract_config", "ASSISTANT_WEB_QUERY_PARAM" in web_src and "ASSISTANT_WEB_RESULTS_PATH" in web_src),
        _check("cache_config", "ASSISTANT_WEB_CACHE_TTL_SECONDS" in web_src and "_cache_key" in web_src),
        _check("provider_host_validation", "_provider_host_matches" in web_src and "expected_host_hint" in web_src),
        _check("get_support", "requests.get" in web_src),
        _check("post_support", "requests.post" in web_src),
        _check("dedupe", "seen: set[tuple[str, str]]" in web_src),
        _check("extract_items", "_extract_items" in web_src),
        _check("coerce_fact", "_coerce_fact" in web_src and "_dig_path" in web_src),
    ]

    web_fact_checks = [
        _check("title_fields", '"headline"' in web_src and '"name"' in web_src and '"title"' in web_src),
        _check("snippet_fields", '"description"' in web_src and '"summary"' in web_src and '"text"' in web_src),
        _check("url_fields", '"source_url"' in web_src and '"sourceUrl"' in web_src and '"link"' in web_src),
        _check("list_root", "if isinstance(data, list):" in web_src),
        _check("single_dict_root", "return [data]" in web_src),
        _check("nested_payloads", '"data"' in web_src and '"response"' in web_src and '"payload"' in web_src),
        _check("bucket_keys", '"results"' in web_src and '"items"' in web_src and '"hits"' in web_src),
        _check("domain_metadata", 'domain: str = ""' in web_src and "_infer_domain" in web_src),
        _check("query_metadata", 'query: str = ""' in web_src and "query=query" in web_src),
        _check("trust_scoring", 'trust_score: float = 0.0' in web_src and "_score_fact" in web_src),
        _check("probe_domain_summary", "domain_count" in web_src and "top_trust_score" in web_src),
        _check("invalid_item_skip", "return None" in web_src),
    ]

    mixed_checks = [
        _check("collect_web_facts", "_collect_web_facts" in runtime_src),
        _check("multi_query_loop", "route.web_queries[:3]" in runtime_src),
        _check("trust_sorting", '"trust_score"' in runtime_src and '"domain"' in runtime_src),
        _check("grounding_packet_use", "build_grounding_packet" in runtime_src),
        _check("grounding_packet_overview", "web_overview" in grounding_src),
        _check("mixed_certainty", "confirmed_mixed" in runtime_src or "inferred_mixed" in runtime_src),
        _check("market_type_conflict", "_has_market_type_conflict" in runtime_src),
        _check("decision_conflict", "_has_decision_conflict" in runtime_src),
        _check("metrics_conflict", "_has_market_metrics_conflict" in runtime_src),
        _check("source_selection_visible", "Source selection" in runtime_src or "Selección de fuentes" in runtime_src),
        _check("source_summary_visible", "Source summary" in runtime_src or "Resumen de fuentes" in runtime_src),
        _check("source_blend_visible", "_format_source_blend_summary" in runtime_src and "top trust" in runtime_src),
        _check("domain_visibility", "Domains:" in runtime_src and "Domain'}:" in runtime_src),
        _check("local_first_grounding", "mixed (local-first)" in runtime_src),
    ]

    retriever = web.AssistantWebRetriever()
    retriever_status = retriever.config_status()
    web_runtime_caps = {
        "web_retrieval_layer": None if retriever_status.runtime_ready else 70,
        "web_fact_extraction": None if retriever_status.runtime_ready else 80,
        "mixed_grounding_engine": None if retriever_status.runtime_ready else 95,
    }

    layers = {
        "semantic_layer": _build_layer("semantic_layer", "Semantic Layer", semantic_checks),
        "conversational_layer": _build_layer("conversational_layer", "Conversational Layer", conversational_checks),
        "primary_conversational_interpreter": _build_layer(
            "primary_conversational_interpreter",
            "Primary Conversational Interpreter",
            comp_checks,
        ),
        "memory_reference_layer": _build_layer("memory_reference_layer", "Memory/Reference Layer", memory_checks),
        "local_fact_layer": _build_layer("local_fact_layer", "Local Fact Layer", local_fact_checks),
        "grounded_answer_layer": _build_layer("grounded_answer_layer", "Grounded Answer Layer", grounded_checks),
        "source_selector": _build_layer("source_selector", "Source Selector", selector_checks),
        "search_augmented_interpretation_layer": _build_layer(
            "search_augmented_interpretation_layer",
            "Search-Augmented Interpretation Layer",
            search_augmented_checks,
            runtime_ready_cap=None if retriever_status.runtime_ready else 85,
        ),
        "web_retrieval_layer": _build_layer(
            "web_retrieval_layer",
            "Web Retrieval Layer",
            web_retrieval_checks,
            runtime_ready_cap=web_runtime_caps["web_retrieval_layer"],
        ),
        "web_fact_extraction": _build_layer(
            "web_fact_extraction",
            "Web Fact Extraction",
            web_fact_checks,
            runtime_ready_cap=web_runtime_caps["web_fact_extraction"],
        ),
        "mixed_grounding_engine": _build_layer(
            "mixed_grounding_engine",
            "Mixed Grounding Engine",
            mixed_checks,
            runtime_ready_cap=web_runtime_caps["mixed_grounding_engine"],
        ),
    }

    local_first_keys = (
        "semantic_layer",
        "conversational_layer",
        "primary_conversational_interpreter",
        "memory_reference_layer",
        "local_fact_layer",
        "grounded_answer_layer",
    )
    local_first_implementation_score = round(
        sum(layers[key].implementation_score_pct for key in local_first_keys) / len(local_first_keys)
    )
    local_first_runtime_ready_score = round(
        sum(layers[key].runtime_ready_score_pct for key in local_first_keys) / len(local_first_keys)
    )
    hybrid_implementation_score = round(
        sum(layer.implementation_score_pct for layer in layers.values()) / len(layers)
    )
    hybrid_runtime_ready_score = round(
        sum(layer.runtime_ready_score_pct for layer in layers.values()) / len(layers)
    )

    return AssistantScorecard(
        generated_at=_utc_now_iso(),
        local_first_score_pct=local_first_runtime_ready_score,
        hybrid_score_pct=hybrid_runtime_ready_score,
        local_first_implementation_score_pct=local_first_implementation_score,
        local_first_runtime_ready_score_pct=local_first_runtime_ready_score,
        hybrid_implementation_score_pct=hybrid_implementation_score,
        hybrid_runtime_ready_score_pct=hybrid_runtime_ready_score,
        runtime={
            "web_retriever_configured": retriever.enabled,
            "web_retriever_config_valid": retriever_status.config_valid,
            "web_retriever_runtime_ready": retriever_status.runtime_ready,
            "web_search_url": retriever.search_url,
            "web_method": retriever.method,
            "web_results_path": retriever.results_path,
            "web_cache_enabled": retriever_status.cache_enabled,
            "web_cache_ttl_seconds": retriever_status.cache_ttl_seconds,
            "web_config_issues": list(retriever_status.issues),
        },
        layers=layers,
    )
