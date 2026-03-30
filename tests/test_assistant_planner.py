from __future__ import annotations

import unittest

from assistant.contracts import AssistantState
from assistant.planner import AssistantPlanner
from assistant.router import AssistantRouter


class AssistantPlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = AssistantPlanner(AssistantRouter())

    def test_planner_builds_run_pipeline_plan(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("AAPL 2024-01-01 2025-01-01", state)

        self.assertEqual(plan.intent, "run_pipeline")
        self.assertEqual(plan.risk, "medium")
        self.assertEqual(plan.steps[0].tool, "build_pipeline_request")
        self.assertEqual(plan.steps[1].tool, "execute_pipeline")
        self.assertEqual(plan.steps[-1].tool, "render_response")
        self.assertEqual(plan.route.get("tickers"), ["AAPL"])

    def test_planner_routes_pure_greeting(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("hola", state)

        self.assertEqual(plan.intent, "greet")
        self.assertEqual(plan.risk, "low")
        self.assertEqual([step.tool for step in plan.steps], ["load_context_bundle", "render_response"])

    def test_planner_routes_capability_request_to_help(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what can you do?", state)

        self.assertEqual(plan.intent, "help")
        self.assertEqual(plan.risk, "low")
        self.assertEqual([step.tool for step in plan.steps], ["load_context_bundle", "load_run_bundle", "render_response"])

    def test_planner_ignores_greeting_prefix_when_real_intent_exists(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("hola, qué es forex?", state)

        self.assertEqual(plan.intent, "show_semantic_lookup")
        self.assertEqual(plan.answer_mode, "interpreted")
        self.assertEqual(plan.source_mode, "web")

    def test_planner_builds_grounded_read_plan_for_model_variables(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0007")

        plan = self.planner.plan("run_0007 qué variables usó el modelo?", state)

        self.assertEqual(plan.intent, "show_model_variables")
        self.assertEqual(plan.risk, "low")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual([step.tool for step in plan.steps], ["load_context_bundle", "load_run_bundle", "render_response"])
        self.assertEqual(plan.answer_mode, "strict")
        self.assertEqual(plan.certainty, "confirmed")
        self.assertEqual(plan.context_snapshot.get("question_focus"), "model_variables")
        self.assertTrue(plan.grounded)

    def test_planner_marks_interpreted_mode_for_explanatory_modeling_question(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0007")

        plan = self.planner.plan("explica las variables usadas en run_0007", state)

        self.assertEqual(plan.intent, "show_model_variables")
        self.assertEqual(plan.answer_mode, "interpreted")
        self.assertEqual(plan.certainty, "inferred")
        self.assertEqual(plan.route.get("question_focus"), "model_variables")

    def test_planner_uses_last_intent_for_vague_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_model_variables",
        )

        plan = self.planner.plan("explica mejor", state)

        self.assertEqual(plan.intent, "show_model_variables")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.answer_mode, "interpreted")
        self.assertEqual(plan.route.get("question_focus"), "model_variables")

    def test_planner_routes_causal_follow_up_from_latest_summary_to_decision_explanation(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("and then?", state)

        self.assertEqual(plan.intent, "show_decision_explanation")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.answer_mode, "interpreted")
        self.assertEqual(plan.route.get("question_focus"), "decision_explanation")

    def test_planner_routes_ultra_short_spanish_follow_up_to_decision_explanation(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("entonces?", state)

        self.assertEqual(plan.intent, "show_decision_explanation")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.answer_mode, "interpreted")

    def test_planner_routes_elliptical_market_follow_up_from_asset(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0001",
            last_intent="show_asset_used",
        )

        plan = self.planner.plan("y el mercado?", state)

        self.assertEqual(plan.intent, "show_market_type")
        self.assertEqual(plan.route.get("run_id"), "run_0001")
        self.assertEqual(plan.answer_mode, "strict")

    def test_planner_routes_elliptical_columns_follow_up_from_extraction(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0001",
            last_intent="show_extraction",
        )

        plan = self.planner.plan("y las columnas?", state)

        self.assertEqual(plan.intent, "show_extraction")
        self.assertEqual(plan.route.get("run_id"), "run_0001")
        self.assertEqual(plan.route.get("question_focus"), "extraction")

    def test_planner_routes_elliptical_variables_follow_up_to_model_variables(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("y las variables?", state)

        self.assertEqual(plan.intent, "show_model_variables")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "model_variables")

    def test_planner_routes_elliptical_decision_follow_up_to_decision_explanation(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("y la decisión?", state)

        self.assertEqual(plan.intent, "show_decision_explanation")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "decision_explanation")

    def test_planner_routes_elliptical_metrics_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("y las métricas?", state)

        self.assertEqual(plan.intent, "show_market_metrics")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "market_metrics")

    def test_planner_routes_entity_symbol_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("ese símbolo?", state)

        self.assertEqual(plan.intent, "show_asset_used")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "asset_used")

    def test_planner_routes_entity_decision_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("esa decisión?", state)

        self.assertEqual(plan.intent, "show_decision_explanation")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "decision_explanation")

    def test_planner_routes_entity_metric_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("esa métrica?", state)

        self.assertEqual(plan.intent, "show_market_metrics")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "market_metrics")

    def test_planner_routes_web_status_request(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("web status", state)

        self.assertEqual(plan.intent, "show_web_status")
        self.assertEqual(plan.route.get("stage"), "web_status")
        self.assertEqual(plan.answer_mode, "strict")
        self.assertEqual(plan.certainty, "inferred")

    def test_planner_routes_spanish_web_status_request(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("estado del retriever", state)

        self.assertEqual(plan.intent, "show_web_status")
        self.assertEqual(plan.route.get("stage"), "web_status")

    def test_planner_routes_provider_setup_request_to_web_status(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what env do I need for tavily?", state)

        self.assertEqual(plan.intent, "show_web_status")
        self.assertEqual(plan.route.get("stage"), "web_status")

    def test_planner_routes_semantic_lookup_to_dedicated_handler_with_web(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what does forex mean?", state)

        self.assertEqual(plan.intent, "show_semantic_lookup")
        self.assertEqual(plan.answer_mode, "interpreted")
        self.assertEqual(plan.source_mode, "web")
        self.assertEqual(plan.risk, "medium")

    def test_planner_routes_generic_definition_to_semantic_lookup(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what is arbitrage?", state)

        self.assertEqual(plan.intent, "show_semantic_lookup")
        self.assertEqual(plan.answer_mode, "interpreted")
        self.assertEqual(plan.source_mode, "web")
        self.assertEqual(plan.risk, "medium")

    def test_planner_routes_concept_comparison_to_semantic_lookup(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what is the difference between forex and crypto?", state)

        self.assertEqual(plan.intent, "show_semantic_lookup")
        self.assertEqual(plan.answer_mode, "interpreted")
        self.assertEqual(plan.source_mode, "web")
        self.assertEqual(plan.risk, "medium")
        self.assertGreaterEqual(len(plan.route.get("web_queries", [])), 5)
        self.assertEqual(plan.route.get("web_queries", [])[0], "forex meaning")
        self.assertEqual(plan.route.get("web_queries", [])[1], "crypto meaning")
        self.assertEqual(plan.route.get("web_queries", [])[2], "forex vs crypto")

    def test_planner_routes_local_glossary_comparison_to_semantic_lookup(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what is the difference between long and short?", state)

        self.assertEqual(plan.intent, "show_semantic_lookup")
        self.assertEqual(plan.answer_mode, "interpreted")
        self.assertEqual(plan.source_mode, "web")
        self.assertEqual(plan.route.get("web_queries", [])[0], "long meaning")
        self.assertEqual(plan.route.get("web_queries", [])[1], "short meaning")

    def test_planner_decomposes_semantic_subject_variants(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what is arbitrage in crypto?", state)

        self.assertEqual(plan.intent, "show_semantic_lookup")
        self.assertEqual(plan.source_mode, "web")
        self.assertGreaterEqual(len(plan.route.get("web_queries", [])), 3)
        self.assertEqual(plan.route.get("web_queries", [])[0], "arbitrage meaning")
        self.assertEqual(plan.route.get("web_queries", [])[1], "arbitrage definition")
        self.assertEqual(plan.route.get("web_queries", [])[2], "arbitrage in crypto meaning")

    def test_planner_routes_spanish_semantic_lookup_to_dedicated_handler_with_web(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("qué es forex?", state)

        self.assertEqual(plan.intent, "show_semantic_lookup")
        self.assertEqual(plan.answer_mode, "interpreted")
        self.assertEqual(plan.source_mode, "web")
        self.assertEqual(plan.risk, "medium")

    def test_planner_routes_semantic_term_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0001",
            last_intent="show_market_type",
            entity_memory={
                "last_focus": "market_type",
                "by_kind": {
                    "market": {"run_id": "run_0001", "ticker": "BTC-USD", "focus": "market_type"},
                },
            },
        )

        plan = self.planner.plan("y qué significa ese mercado?", state)

        self.assertEqual(plan.intent, "show_semantic_lookup")
        self.assertEqual(plan.route.get("run_id"), "run_0001")
        self.assertEqual(plan.route.get("tickers"), ["BTC-USD"])
        self.assertEqual(plan.source_mode, "mixed")

    def test_planner_routes_bare_semantic_term_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0001",
            last_intent="show_semantic_lookup",
            entity_memory={"last_focus": "semantic_lookup", "last_entity_kind": "semantic"},
        )

        plan = self.planner.plan("that term?", state)

        self.assertEqual(plan.intent, "show_semantic_lookup")
        self.assertEqual(plan.route.get("question_focus"), "semantic_lookup")

    def test_planner_keeps_volume_lookup_as_market_metrics(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what is the volume of NVDA?", state)

        self.assertEqual(plan.intent, "show_market_metrics")
        self.assertEqual(plan.route.get("question_focus"), "market_metrics")
        self.assertEqual(plan.source_mode, "local")

    def test_planner_routes_provider_activation_request_to_web_status(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("activate serper", state)

        self.assertEqual(plan.intent, "show_web_status")
        self.assertEqual(plan.route.get("stage"), "web_status")

    def test_planner_routes_provider_catalog_request_to_web_status(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what web providers are supported?", state)

        self.assertEqual(plan.intent, "show_web_status")
        self.assertEqual(plan.route.get("stage"), "web_status")

    def test_planner_routes_assistant_scorecard_request(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("assistant scorecard", state)

        self.assertEqual(plan.intent, "show_assistant_scorecard")
        self.assertEqual(plan.route.get("stage"), "assistant_scorecard")
        self.assertEqual(plan.route.get("question_focus"), "assistant_scorecard")

    def test_planner_routes_provider_specific_scorecard_request(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("assistant scorecard for serper", state)

        self.assertEqual(plan.intent, "show_assistant_scorecard")
        self.assertEqual(plan.route.get("stage"), "assistant_scorecard")

    def test_planner_routes_scorecard_layers_request(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("assistant scorecard layers", state)

        self.assertEqual(plan.intent, "show_assistant_scorecard")
        self.assertEqual(plan.route.get("stage"), "assistant_scorecard")

    def test_planner_routes_scorecard_web_impact_request(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("what changes if I enable web?", state)

        self.assertEqual(plan.intent, "show_assistant_scorecard")
        self.assertEqual(plan.route.get("stage"), "assistant_scorecard")

    def test_planner_routes_scorecard_setup_follow_up_to_web_status(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_intent="show_assistant_scorecard",
            last_route={"intent": "show_assistant_scorecard", "stage": "assistant_scorecard"},
            entity_memory={"last_focus": "assistant_scorecard", "last_entity_kind": "assistant_scorecard"},
        )

        plan = self.planner.plan("y el setup?", state)

        self.assertEqual(plan.intent, "show_web_status")
        self.assertEqual(plan.route.get("question_focus"), "web_status")

    def test_planner_routes_scorecard_probe_follow_up_to_web_status(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_intent="show_assistant_scorecard",
            last_route={"intent": "show_assistant_scorecard", "stage": "assistant_scorecard"},
            entity_memory={"last_focus": "assistant_scorecard", "last_entity_kind": "assistant_scorecard"},
        )

        plan = self.planner.plan("y el probe?", state)

        self.assertEqual(plan.intent, "show_web_status")
        self.assertEqual(plan.route.get("question_focus"), "web_status")

    def test_planner_routes_scorecard_layers_follow_up_to_scorecard(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_intent="show_assistant_scorecard",
            last_route={"intent": "show_assistant_scorecard", "stage": "assistant_scorecard"},
            entity_memory={"last_focus": "assistant_scorecard", "last_entity_kind": "assistant_scorecard"},
        )

        plan = self.planner.plan("y las capas?", state)

        self.assertEqual(plan.intent, "show_assistant_scorecard")
        self.assertEqual(plan.route.get("question_focus"), "assistant_scorecard")

    def test_planner_routes_scorecard_web_impact_follow_up_to_scorecard(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_intent="show_assistant_scorecard",
            last_route={"intent": "show_assistant_scorecard", "stage": "assistant_scorecard"},
            entity_memory={"last_focus": "assistant_scorecard", "last_entity_kind": "assistant_scorecard"},
        )

        plan = self.planner.plan("y si configuro web?", state)

        self.assertEqual(plan.intent, "show_assistant_scorecard")
        self.assertEqual(plan.route.get("question_focus"), "assistant_scorecard")

    def test_planner_routes_provider_setup_phrase_to_web_status(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("use serper setup", state)

        self.assertEqual(plan.intent, "show_web_status")
        self.assertEqual(plan.route.get("stage"), "web_status")

    def test_planner_routes_entity_row_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("esa fila?", state)

        self.assertEqual(plan.intent, "show_clean_data")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "clean_data")

    def test_planner_uses_entity_memory_for_bare_reference(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_help",
            entity_memory={"last_focus": "model_variables", "run_id": "run_0007", "last_entity_kind": "variable"},
        )

        plan = self.planner.plan("esa?", state)

        self.assertEqual(plan.intent, "show_model_variables")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "model_variables")

    def test_planner_prefers_entity_memory_by_kind_for_variable_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0008",
            last_intent="show_latest_summary",
            last_route={"intent": "show_latest_summary", "run_id": "run_0008"},
            entity_memory={
                "last_focus": "summary",
                "run_id": "run_0008",
                "last_entity_kind": "summary",
                "by_kind": {
                    "variable": {
                        "focus": "model_variables",
                        "run_id": "run_0007",
                        "ticker": "MSFT",
                    }
                },
            },
        )

        plan = self.planner.plan("esa variable?", state)

        self.assertEqual(plan.intent, "show_model_variables")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "model_variables")

    def test_planner_uses_symbol_memory_for_market_type_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0008",
            last_intent="show_latest_summary",
            last_route={"intent": "show_latest_summary", "run_id": "run_0008"},
            entity_memory={
                "last_focus": "summary",
                "run_id": "run_0008",
                "last_entity_kind": "summary",
                "by_kind": {
                    "symbol": {
                        "focus": "asset_used",
                        "run_id": "run_0007",
                        "ticker": "MSFT",
                    },
                    "summary": {
                        "focus": "summary",
                        "run_id": "run_0008",
                        "ticker": "AAPL",
                    },
                },
            },
        )

        plan = self.planner.plan("that market?", state)

        self.assertEqual(plan.intent, "show_market_type")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "market_type")

    def test_planner_uses_comparison_memory_after_unrelated_summary(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0009",
            last_intent="show_latest_summary",
            last_route={"intent": "show_latest_summary", "run_id": "run_0009"},
            entity_memory={
                "last_focus": "summary",
                "run_id": "run_0009",
                "last_entity_kind": "summary",
                "by_kind": {
                    "comparison": {
                        "focus": "run_comparison",
                        "run_id": "run_0007",
                        "secondary_run_id": "run_0008",
                    },
                    "summary": {
                        "focus": "summary",
                        "run_id": "run_0009",
                    },
                },
            },
        )

        plan = self.planner.plan("that comparison?", state)

        self.assertEqual(plan.intent, "show_run_comparison")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("secondary_run_id"), "run_0008")

    def test_planner_routes_elliptical_row_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("y la fila?", state)

        self.assertEqual(plan.intent, "show_clean_data")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "clean_data")

    def test_planner_detects_run_comparison_and_exploratory_mode(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("compare run_0007 vs run_0008 maybe", state)

        self.assertEqual(plan.intent, "show_run_comparison")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("secondary_run_id"), "run_0008")
        self.assertEqual(plan.answer_mode, "exploratory")
        self.assertEqual(plan.route.get("question_focus"), "run_comparison")

    def test_planner_detects_stage_specific_run_comparison(self) -> None:
        state = AssistantState(session_id="planner-test")

        plan = self.planner.plan("compare cleaning run_0007 vs run_0008", state)

        self.assertEqual(plan.intent, "show_run_comparison")
        self.assertEqual(plan.route.get("stage"), "cleaning")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("secondary_run_id"), "run_0008")

    def test_planner_routes_elliptical_comparison_follow_up_and_preserves_both_runs(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_run_comparison",
            last_route={"intent": "show_run_comparison", "run_id": "run_0007", "secondary_run_id": "run_0008"},
        )

        plan = self.planner.plan("y esa comparación?", state)

        self.assertEqual(plan.intent, "show_run_comparison")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("secondary_run_id"), "run_0008")
        self.assertEqual(plan.route.get("question_focus"), "run_comparison")

    def test_planner_routes_elliptical_run_follow_up_to_latest_summary(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("esa corrida?", state)

        self.assertEqual(plan.intent, "show_latest_summary")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "summary")

    def test_planner_routes_elliptical_result_follow_up_to_extraction(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0001",
            last_intent="show_extraction",
        )

        plan = self.planner.plan("ese resultado?", state)

        self.assertEqual(plan.intent, "show_extraction")
        self.assertEqual(plan.route.get("run_id"), "run_0001")
        self.assertEqual(plan.route.get("question_focus"), "extraction")

    def test_planner_routes_elliptical_cleaning_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_cleaning",
        )

        plan = self.planner.plan("esa limpieza?", state)

        self.assertEqual(plan.intent, "show_cleaning")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "cleaning")

    def test_planner_routes_modeling_follow_up_phrase_to_prediction(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("eso de modelado?", state)

        self.assertEqual(plan.intent, "show_prediction")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "prediction")

    def test_planner_routes_generic_process_follow_up_to_last_stage_view(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_cleaning",
        )

        plan = self.planner.plan("ese proceso?", state)

        self.assertEqual(plan.intent, "show_cleaning")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "cleaning")

    def test_planner_routes_extraction_nominal_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0001",
            last_intent="show_extraction",
        )

        plan = self.planner.plan("lo de extracción?", state)

        self.assertEqual(plan.intent, "show_extraction")
        self.assertEqual(plan.route.get("run_id"), "run_0001")
        self.assertEqual(plan.route.get("question_focus"), "extraction")

    def test_planner_routes_generic_part_follow_up_to_last_modeling_view(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_prediction",
        )

        plan = self.planner.plan("esa parte?", state)

        self.assertEqual(plan.intent, "show_prediction")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "prediction")

    def test_planner_routes_compound_cleaning_phrase(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("la parte de limpieza?", state)

        self.assertEqual(plan.intent, "show_cleaning")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "cleaning")

    def test_planner_routes_compound_modeling_step_phrase(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_latest_summary",
        )

        plan = self.planner.plan("el paso del modelado?", state)

        self.assertEqual(plan.intent, "show_prediction")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "prediction")

    def test_planner_routes_compound_process_phrase_to_last_stage(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_prediction",
        )

        plan = self.planner.plan("esa parte del proceso?", state)

        self.assertEqual(plan.intent, "show_prediction")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "prediction")

    def test_planner_uses_last_stage_brief_stage_for_generic_stage_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_stage_brief",
            last_route={"intent": "show_stage_brief", "run_id": "run_0007", "stage": "cleaning"},
        )

        plan = self.planner.plan("esa etapa?", state)

        self.assertEqual(plan.intent, "show_cleaning")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "cleaning")

    def test_planner_uses_last_stage_brief_stage_for_generic_process_follow_up(self) -> None:
        state = AssistantState(
            session_id="planner-test",
            last_run_id="run_0007",
            last_intent="show_stage_brief",
            last_route={"intent": "show_stage_brief", "run_id": "run_0007", "stage": "modeling"},
        )

        plan = self.planner.plan("ese proceso?", state)

        self.assertEqual(plan.intent, "show_prediction")
        self.assertEqual(plan.route.get("run_id"), "run_0007")
        self.assertEqual(plan.route.get("question_focus"), "prediction")

    def test_planner_marks_market_type_questions_as_mixed_grounding(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0001")

        plan = self.planner.plan("what market type does the symbol used in run_0001 belong to?", state)

        self.assertEqual(plan.intent, "show_market_type")
        self.assertEqual(plan.route.get("run_id"), "run_0001")
        self.assertEqual(plan.source_mode, "mixed")
        self.assertEqual(plan.risk, "medium")
        self.assertIn("retrieve_web_facts", [step.tool for step in plan.steps])
        self.assertIn("merge_grounding_facts", [step.tool for step in plan.steps])

    def test_planner_detects_spanish_asset_used_question(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0001")

        plan = self.planner.plan("dime el activo usado para analisis en run_0001", state)

        self.assertEqual(plan.intent, "show_asset_used")
        self.assertEqual(plan.route.get("run_id"), "run_0001")
        self.assertEqual(plan.source_mode, "local")

    def test_planner_detects_spanish_market_type_question(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0001")

        plan = self.planner.plan("de qué tipo de mercado hace parte el símbolo usado en run_0001?", state)

        self.assertEqual(plan.intent, "show_market_type")
        self.assertEqual(plan.route.get("run_id"), "run_0001")
        self.assertEqual(plan.source_mode, "mixed")

    def test_planner_marks_current_market_metrics_as_mixed_grounding(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0007", current_asset="MSFT")

        plan = self.planner.plan("what is the current volume of MSFT today?", state)

        self.assertEqual(plan.intent, "show_market_metrics")
        self.assertEqual(plan.source_mode, "mixed")
        self.assertEqual(plan.risk, "medium")
        self.assertIn("retrieve_web_facts", [step.tool for step in plan.steps])

    def test_planner_marks_current_decision_explanation_as_mixed_grounding(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0007", current_asset="MSFT")

        plan = self.planner.plan("why did run_0007 decide that way with current market context?", state)

        self.assertEqual(plan.intent, "show_decision_explanation")
        self.assertEqual(plan.source_mode, "mixed")
        self.assertEqual(plan.risk, "medium")
        self.assertIn("retrieve_web_facts", [step.tool for step in plan.steps])

    def test_planner_marks_clean_data_with_current_context_as_mixed_grounding(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0007", current_asset="MSFT")

        plan = self.planner.plan("analyze the clean row for MSFT with current market context", state)

        self.assertEqual(plan.intent, "show_clean_data")
        self.assertEqual(plan.source_mode, "mixed")
        self.assertEqual(plan.risk, "medium")
        self.assertIn("retrieve_web_facts", [step.tool for step in plan.steps])

    def test_planner_treats_latest_today_as_latest_summary(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0007")

        plan = self.planner.plan("latest today", state)

        self.assertEqual(plan.intent, "show_latest_summary")
        self.assertEqual(plan.source_mode, "mixed")
        self.assertEqual(plan.risk, "medium")
        self.assertIn("retrieve_web_facts", [step.tool for step in plan.steps])

    def test_planner_treats_spanish_whats_happening_today_as_latest_summary(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0007")

        plan = self.planner.plan("qué pasa hoy", state)

        self.assertEqual(plan.intent, "show_latest_summary")
        self.assertEqual(plan.source_mode, "mixed")
        self.assertEqual(plan.risk, "medium")

    def test_planner_treats_hows_it_going_as_latest_summary(self) -> None:
        state = AssistantState(session_id="planner-test", last_run_id="run_0007")

        plan = self.planner.plan("how's it going", state)

        self.assertEqual(plan.intent, "show_latest_summary")
        self.assertEqual(plan.answer_mode, "strict")


if __name__ == "__main__":
    unittest.main()
