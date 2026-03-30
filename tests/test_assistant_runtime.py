from __future__ import annotations

from datetime import datetime
import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from assistant.contracts import AssistantRoute, AssistantState
from assistant.runtime import (
    AssistantRuntime,
    _format_asset_used,
    _format_groq_status,
    _format_mode_guide,
    _format_clean_data_view,
    _format_decision_explanation,
    _format_agent_guide,
    _format_agent_card,
    _format_market_metrics,
    _format_latest_report,
    _format_session_context_line,
    _contextual_help_prompts,
    _format_session_status,
    _format_motor,
    _format_stage_brief,
    _format_symbol_guide,
    _extract_clean_data_mode,
    _summary_from_result,
)
from assistant.state import load_state, save_state
from assistant.web import WebFact


class _RuntimeSearchHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            payload = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            payload = {}
        self.server.requests_log.append(  # type: ignore[attr-defined]
            {
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers),
                "json": payload,
            }
        )
        body = json.dumps(self.server.response_payload).encode("utf-8")  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return None


@contextmanager
def _serve_runtime_search_payload(payload: dict) -> tuple[str, list[dict]]:
    server = HTTPServer(("127.0.0.1", 0), _RuntimeSearchHandler)
    server.response_payload = payload  # type: ignore[attr-defined]
    server.requests_log = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/search", server.requests_log  # type: ignore[attr-defined]
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


class _StubWebRetriever:
    def __init__(self, facts: list[WebFact]) -> None:
        self._facts = facts

    @property
    def enabled(self) -> bool:
        return True

    def search(self, query: str, limit: int = 3) -> list[WebFact]:
        return self._facts[:limit]


class _MappedWebRetriever:
    def __init__(self, mapping: dict[str, list[WebFact]]) -> None:
        self._mapping = mapping

    @property
    def enabled(self) -> bool:
        return True

    def search(self, query: str, limit: int = 3) -> list[WebFact]:
        return (self._mapping.get(query) or [])[:limit]


class AssistantRuntimeSummaryFallbackTest(unittest.TestCase):
    def test_runtime_routes_how_are_you_to_greeting_even_with_previous_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_state(
                tmpdir,
                AssistantState(
                    session_id="comp-greeting",
                    last_run_id="run_0013",
                    current_asset="SPY",
                    last_summary={"run_id": "run_0013", "tickers": ["SPY"]},
                ),
            )
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="comp-greeting")

            text = runtime.ask("como estas?")

        self.assertIn("Hola. ¿En qué puedo ayudarte?", text)

    def test_runtime_routes_agent_menu_request_even_with_previous_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_state(
                tmpdir,
                AssistantState(
                    session_id="comp-agent-guide",
                    last_run_id="run_0013",
                    current_asset="SPY",
                    last_summary={"run_id": "run_0013", "tickers": ["SPY"]},
                ),
            )
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="comp-agent-guide")

            text = runtime.ask("muestrame los agentes disponibles")

        self.assertIn("Menú de agentes", text)
        self.assertIn("Extracción | Limpieza | Modelado | Orquestador", text)

    def test_runtime_routes_mode_guide_request_even_with_previous_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_state(
                tmpdir,
                AssistantState(
                    session_id="comp-mode-guide",
                    last_run_id="run_0013",
                    current_asset="SPY",
                    last_summary={"run_id": "run_0013", "tickers": ["SPY"]},
                ),
            )
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="comp-mode-guide")

            text = runtime.ask("qué modos tienes?")

        self.assertIn("Centro de modos", text)
        self.assertIn("local_only | compare-binance | --groq-brain", text)

    def test_runtime_routes_asset_guide_request_even_with_previous_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_state(
                tmpdir,
                AssistantState(
                    session_id="comp-symbol-guide",
                    last_run_id="run_0013",
                    current_asset="SPY",
                    last_summary={"run_id": "run_0013", "tickers": ["SPY"]},
                ),
            )
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="comp-symbol-guide")

            text = runtime.ask("qué activos puedo usar?")

        self.assertIn("Símbolos sugeridos", text)
        self.assertIn("AAPL", text)
        self.assertIn("QQQ", text)

    def test_runtime_routes_spanish_capability_request_even_with_previous_run_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_state(
                tmpdir,
                AssistantState(
                    session_id="comp-help-es",
                    last_run_id="run_0013",
                    current_asset="SPY",
                    last_summary={"run_id": "run_0013", "tickers": ["SPY"]},
                ),
            )
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="comp-help-es")

            text = runtime.ask("que puedes hacer?")

        self.assertIn("Centro de ayuda", text)
        self.assertNotIn("Decisión:", text)

    def test_runtime_answers_time_status_from_local_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="time-status")

            text = runtime.ask("qué día es hoy?")

        expected_date = datetime.now(ZoneInfo("America/Bogota")).date().isoformat()
        self.assertIn(expected_date, text)
        self.assertIn("America/Bogota", text)
        self.assertNotIn("Run run_", text)

    def test_runtime_answers_source_scope_from_local_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="source-scope")

            text = runtime.ask("de dónde proviene tu información?")

        self.assertIn("Fuentes del assistant", text)
        self.assertIn("Artifacts locales", text)
        self.assertIn("Herramientas locales", text)
        self.assertIn("Internet", text)
        self.assertIn("Jerarquía: artifacts locales > herramientas locales > internet.", text)

    def test_runtime_clears_stale_pending_route_on_strong_new_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_state(
                tmpdir,
                AssistantState(
                    session_id="pending-clear",
                    pending_task="¿Qué columna deseas buscar?",
                    pending_route={
                        "intent": "show_semantic_lookup",
                        "needs_follow_up": True,
                        "follow_up_question": "¿Qué columna deseas buscar?",
                        "raw_message": "activa el retriever web y busca que es una columna ?",
                    },
                ),
            )
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="pending-clear")

            text = runtime.ask("qué día es hoy?")
            state = load_state(tmpdir, "pending-clear")

        self.assertNotIn("¿Qué columna deseas buscar?", text)
        self.assertEqual(state.pending_route, {})
        self.assertEqual(state.pending_task, "")
        self.assertEqual(state.last_intent, "show_time_status")

    def test_runtime_answers_pure_greeting_without_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="greeting")

            text = runtime.ask("hola")

        self.assertIn("Hola. ¿En qué puedo ayudarte?", text)
        self.assertNotIn("Idioma cambiado a español", text)

    def test_runtime_answers_capability_request_with_help_hub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="capability-request")

            text = runtime.ask("what can you do?")

        self.assertIn("Help hub", text)
        self.assertIn("Ask naturally with a ticker, an agent, a mode, or a question.", text)

    def test_runtime_collects_multi_query_web_facts_with_dedup_and_limit(self) -> None:
        runtime = AssistantRuntime(session_id="web-collect-test")
        runtime.web_retriever = _MappedWebRetriever(
            {
                "q1": [
                    WebFact(title="One", snippet="A", url="https://example.com/1"),
                    WebFact(title="Two", snippet="B", url="https://example.com/2"),
                ],
                "q2": [
                    WebFact(title="Two", snippet="B", url="https://example.com/2"),
                    WebFact(title="Three", snippet="C", url="https://example.com/3"),
                ],
                "q3": [
                    WebFact(title="Four", snippet="D", url="https://example.com/4"),
                ],
            }
        )
        route = AssistantRoute(intent="show_latest_summary", source_mode="mixed", web_queries=["q1", "q2", "q3"])

        facts = runtime._collect_web_facts(route)

        self.assertEqual(len(facts), 3)
        self.assertEqual([fact.title for fact in facts], ["One", "Two", "Three"])

    def test_runtime_can_answer_semantic_lookup_from_web_without_local_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-web-brief")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="Forex market",
                        snippet="Forex is the foreign exchange market where currency pairs trade globally.",
                        url="https://example.com/forex",
                    )
                ]
            )

            text = runtime.ask("what does forex mean?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: web.", text)
        self.assertIn("Semantic web brief:", text)
        self.assertIn("Forex is the foreign exchange market", text)
        self.assertIn("Source summary: 1 external source.", text)
        self.assertIn("Source: https://example.com/forex.", text)

    def test_runtime_can_answer_generic_semantic_lookup_from_web_without_local_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-web-arbitrage")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="Arbitrage",
                        snippet="Arbitrage is the practice of exploiting price differences of the same asset in different markets.",
                        url="https://example.com/arbitrage",
                    )
                ]
            )

            text = runtime.ask("what is arbitrage?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: web.", text)
        self.assertIn("Semantic web brief:", text)
        self.assertIn("Arbitrage is the practice", text)
        self.assertIn("Source summary: 1 external source.", text)
        self.assertIn("Source: https://example.com/arbitrage.", text)

    def test_runtime_can_force_web_semantic_lookup_from_explicit_web_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-explicit-web")
            runtime.web_retriever = _MappedWebRetriever(
                {
                    "groq meaning": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                    "groq definition": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                    "what is groq?": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                }
            )

            text = runtime.ask("busca en internet que es groq")

        self.assertIn("Brief semántico web", text)
        self.assertIn("Groq is an AI platform", text)
        self.assertNotIn("pregunta sobre busca en internet", text.lower())

    def test_runtime_persists_last_turn_trace_for_comp_interpretation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-trace-web")
            runtime.web_retriever = _MappedWebRetriever(
                {
                    "groq meaning": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                    "groq definition": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                    "what is groq?": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                }
            )

            runtime.ask("busca en internet que es groq")
            trace = runtime.get_state().last_turn_trace

        self.assertEqual(trace.get("act"), "definition")
        self.assertEqual(trace.get("subject"), "groq")
        self.assertEqual(trace.get("final_intent"), "show_semantic_lookup")
        self.assertEqual(trace.get("source_mode"), "web")
        self.assertEqual(trace.get("source_policy"), "web_required")
        self.assertTrue(trace.get("web_required"))
        self.assertFalse(trace.get("allow_run"))

    def test_runtime_can_show_turn_trace_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="trace-panel")
            runtime.web_retriever = _MappedWebRetriever(
                {
                    "groq meaning": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                    "groq definition": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                    "what is groq?": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                }
            )

            runtime.ask("busca en internet que es groq")
            text = runtime.ask("traza")

        self.assertIn("Traza del turno", text)
        self.assertIn("Último mensaje: busca en internet que es groq.", text)
        self.assertIn("acto=definition; sujeto=groq; intención=show_semantic_lookup;", text)
        self.assertIn("fuente=web; política=web_required; web obligatoria=sí.", text)

    def test_runtime_can_show_turn_trace_full_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="trace-panel-full")
            runtime.web_retriever = _MappedWebRetriever(
                {
                    "groq meaning": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                    "groq definition": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                    "what is groq?": [
                        WebFact(
                            title="Groq",
                            snippet="Groq is an AI platform and API provider focused on fast inference.",
                            url="https://example.com/groq",
                        )
                    ],
                }
            )

            runtime.ask("busca en internet que es groq")
            text = runtime.ask("traza completa")

        self.assertIn("Traza completa del turno", text)
        self.assertIn("Último mensaje: busca en internet que es groq.", text)
        self.assertIn("Resolución: canonical_query=what is groq?; intención_inicial=show_semantic_lookup; intención_final=show_semantic_lookup; etapa=semantic_lookup; foco=semantic_lookup.", text)
        self.assertIn("Scope: run_id=n/a; secondary_run_id=n/a; tickers=n/a.", text)
        self.assertIn("Explicación del planner:", text)

    def test_runtime_can_use_search_augmented_interpretation_for_open_concept_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-search-aug-open")
            runtime.web_retriever = _MappedWebRetriever(
                {
                    "black-litterman meaning": [
                        WebFact(
                            title="Black-Litterman model",
                            snippet="Black-Litterman is a portfolio construction model that blends market equilibrium with investor views.",
                            url="https://example.com/black-litterman",
                        )
                    ],
                    "black-litterman definition": [
                        WebFact(
                            title="Black-Litterman model",
                            snippet="Black-Litterman is a portfolio construction model that blends market equilibrium with investor views.",
                            url="https://example.com/black-litterman",
                        )
                    ],
                    "what is black-litterman?": [
                        WebFact(
                            title="Black-Litterman model",
                            snippet="Black-Litterman is a portfolio construction model that blends market equilibrium with investor views.",
                            url="https://example.com/black-litterman",
                        )
                    ],
                }
            )

            text = runtime.ask("tell me about Black-Litterman")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: web.", text)
        self.assertIn("Semantic web brief:", text)
        self.assertIn("portfolio construction model", text)
        self.assertIn("Search-augmented interpretation:", text)
        self.assertIn("Black-Litterman", text)
        self.assertNotIn("Run run_", text)

    def test_runtime_can_use_search_augmented_interpretation_for_bare_unknown_term(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-search-aug-bare")
            runtime.web_retriever = _MappedWebRetriever(
                {
                    "cointegration meaning": [
                        WebFact(
                            title="Cointegration",
                            snippet="Cointegration describes a long-run statistical relationship where non-stationary series move together.",
                            url="https://example.com/cointegration",
                        )
                    ],
                    "cointegration definition": [
                        WebFact(
                            title="Cointegration",
                            snippet="Cointegration describes a long-run statistical relationship where non-stationary series move together.",
                            url="https://example.com/cointegration",
                        )
                    ],
                    "what is cointegration?": [
                        WebFact(
                            title="Cointegration",
                            snippet="Cointegration describes a long-run statistical relationship where non-stationary series move together.",
                            url="https://example.com/cointegration",
                        )
                    ],
                }
            )

            text = runtime.ask("cointegration")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: web.", text)
        self.assertIn("Semantic web brief:", text)
        self.assertIn("long-run statistical relationship", text)
        self.assertIn("Search-augmented interpretation:", text)
        self.assertNotIn("BTC-USD", text)

    def test_runtime_can_use_canonical_semantic_interpretation_for_open_local_concept(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-search-aug-local")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("tell me about yfinance")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic brief: yfinance.", text)
        self.assertNotIn("Session status", text)

    def test_runtime_market_type_uses_explicit_symbol_instead_of_previous_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_state(
                tmpdir,
                AssistantState(
                    session_id="market-type-explicit-symbol",
                    last_run_id="run_0015",
                    current_asset="^GSPC",
                    last_summary={"run_id": "run_0015", "tickers": ["^GSPC"]},
                ),
            )
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="market-type-explicit-symbol")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("a qué mercado pertenece QQQ?")

        self.assertIn("QQQ pertenece", text)
        self.assertIn("ETF", text)
        self.assertNotIn("^GSPC pertenece", text)

    def test_runtime_can_answer_yfinance_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-yfinance")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is yfinance?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic brief: yfinance.", text)
        self.assertIn("python library", text.lower())
        self.assertIn("fetch market data from yahoo finance", text.lower())

    def test_runtime_can_answer_column_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-column")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is a column?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic brief: column.", text)
        self.assertIn("one field of a table or dataframe", text.lower())

    def test_runtime_can_answer_row_and_schema_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-row-schema")
            runtime.web_retriever = _StubWebRetriever([])

            row_text = runtime.ask("what is a row?")
            schema_text = runtime.ask("what is a schema?")

        self.assertIn("Local semantic brief: row.", row_text)
        self.assertIn("record or observation", row_text.lower())
        self.assertIn("Local semantic brief: schema.", schema_text)
        self.assertIn("structural contract of a dataset", schema_text.lower())

    def test_runtime_can_answer_dataset_artifact_manifest_target_and_feature_engineering_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-data-concepts")
            runtime.web_retriever = _StubWebRetriever([])

            dataset_text = runtime.ask("what is a dataset?")
            artifact_text = runtime.ask("what is an artifact?")
            manifest_text = runtime.ask("what is a manifest?")
            target_text = runtime.ask("what is a target?")
            feature_engineering_text = runtime.ask("what is feature engineering?")

        self.assertIn("Local semantic brief: dataset.", dataset_text)
        self.assertIn("collection of rows and columns", dataset_text.lower())
        self.assertIn("Local semantic brief: artifact.", artifact_text)
        self.assertIn("summary.json", artifact_text.lower())
        self.assertIn("Local semantic brief: manifest.", manifest_text)
        self.assertIn("structured record", manifest_text.lower())
        self.assertIn("Local semantic brief: target.", target_text)
        self.assertIn("label the model tries to predict", target_text.lower())
        self.assertIn("Local semantic brief: feature engineering.", feature_engineering_text)
        self.assertIn("turning raw data into derived signals", feature_engineering_text.lower())

    def test_runtime_can_answer_architecture_concepts_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-architecture-concepts")
            runtime.web_retriever = _StubWebRetriever([])

            grounding_text = runtime.ask("what is grounding?")
            memory_text = runtime.ask("what is memory?")
            artifact_store_text = runtime.ask("what is an artifact store?")
            evidence_ledger_text = runtime.ask("what is an evidence ledger?")
            drift_text = runtime.ask("what is drift?")
            conversational_layer_text = runtime.ask("what is a conversational layer?")

        self.assertIn("Local semantic brief: grounding.", grounding_text)
        self.assertIn("answering from evidence", grounding_text.lower())
        self.assertIn("Local semantic brief: memory.", memory_text)
        self.assertIn("keeps useful context between turns", memory_text.lower())
        self.assertIn("Local semantic brief: artifact store.", artifact_store_text)
        self.assertIn("where run outputs are kept", artifact_store_text.lower())
        self.assertIn("Local semantic brief: evidence ledger.", evidence_ledger_text)
        self.assertIn("trace of what evidence was used", evidence_ledger_text.lower())
        self.assertIn("Local semantic brief: drift.", drift_text)
        self.assertIn("meaningful change in data", drift_text.lower())
        self.assertIn("Local semantic brief: conversational layer.", conversational_layer_text)
        self.assertIn("keeps track of intent", conversational_layer_text.lower())

    def test_runtime_can_answer_governance_concepts_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-governance-concepts")
            runtime.web_retriever = _StubWebRetriever([])

            validator_text = runtime.ask("what is a validator?")
            shadow_run_text = runtime.ask("what is a shadow run?")
            promotion_gate_text = runtime.ask("what is a promotion gate?")
            challenger_vs_champion_text = runtime.ask("what is the difference between challenger and champion?")
            feature_registry_text = runtime.ask("what is a feature registry?")
            adaptive_selector_text = runtime.ask("what is an adaptive selector?")
            shadow_runner_text = runtime.ask("what is a shadow runner?")
            promotion_policy_text = runtime.ask("what is a promotion policy?")

        self.assertIn("Local semantic brief: validator.", validator_text)
        self.assertIn("required checks", validator_text.lower())
        self.assertIn("Local semantic brief: shadow run.", shadow_run_text)
        self.assertIn("without letting that candidate affect production", shadow_run_text.lower())
        self.assertIn("Local semantic brief: promotion gate.", promotion_gate_text)
        self.assertIn("into production after validation", promotion_gate_text.lower())
        self.assertIn("Local semantic comparison brief:", challenger_vs_champion_text)
        self.assertIn("current baseline", challenger_vs_champion_text.lower())
        self.assertIn("candidate trying to beat it", challenger_vs_champion_text.lower())
        self.assertIn("Local semantic brief: feature registry.", feature_registry_text)
        self.assertIn("versioned catalog", feature_registry_text.lower())
        self.assertIn("Local semantic brief: adaptive selector.", adaptive_selector_text)
        self.assertIn("chooses among approved models", adaptive_selector_text.lower())
        self.assertIn("Local semantic brief: shadow runner.", shadow_runner_text)
        self.assertIn("launches and records shadow runs", shadow_runner_text.lower())
        self.assertIn("Local semantic brief: promotion policy.", promotion_policy_text)
        self.assertIn("defines when a validated candidate", promotion_policy_text.lower())

    def test_runtime_can_answer_adj_close_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-adj-close")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is adj_close?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic brief: adj_close.", text)
        self.assertIn("adjusted close", text.lower())

    def test_runtime_can_answer_raw_column_vs_model_variable_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-raw-vs-model")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is the difference between a raw column and a model variable?")

        self.assertIn("Local semantic comparison brief:", text)
        self.assertIn("comes directly from the source", text.lower())
        self.assertIn("actually enters the model", text.lower())

    def test_runtime_reuses_previous_semantic_subject_for_web_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-follow-up-web")
            runtime.web_retriever = _StubWebRetriever([])
            runtime.ask("what is a column?")
            runtime.web_retriever = _MappedWebRetriever(
                {
                    "column meaning": [
                        WebFact(
                            title="Data column",
                            snippet="A column is a named field in a table that stores one attribute across all rows.",
                            url="https://example.com/column",
                        )
                    ],
                    "column definition": [
                        WebFact(
                            title="Data column",
                            snippet="A column is a named field in a table that stores one attribute across all rows.",
                            url="https://example.com/column",
                        )
                    ],
                    "what is a column?": [
                        WebFact(
                            title="Data column",
                            snippet="A column is a named field in a table that stores one attribute across all rows.",
                            url="https://example.com/column",
                        )
                    ],
                }
            )

            text = runtime.ask("complement that with internet")

        self.assertIn("Semantic web brief:", text)
        self.assertIn("named field in a table", text)
        self.assertNotIn("I understood this as a question about", text)
        self.assertNotIn("Assistant sources", text)

    def test_runtime_answers_data_help_with_targeted_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="data-help")

            text = runtime.ask("what can I do with the data?")

        self.assertIn("Help hub", text)
        self.assertIn("With the data you can:", text)
        self.assertIn("inspect raw columns extracted from yfinance", text)
        self.assertIn("inspect clean rows, metrics, schema, and row analysis", text)
        self.assertIn("inspect model variables", text)

    def test_runtime_routes_do_with_that_follow_up_to_data_help_after_data_concept(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="data-help-follow-up")
            runtime.web_retriever = _StubWebRetriever([])
            runtime.ask("what is a dataset?")

            text = runtime.ask("what do I do with that?")

        self.assertIn("Help hub", text)
        self.assertIn("With the data you can:", text)

    def test_runtime_routes_next_step_follow_up_to_contextual_help_after_data_concept(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="next-step-data-follow-up")
            runtime.web_retriever = _StubWebRetriever([])
            runtime.ask("what is a dataset?")

            text = runtime.ask("what should I do next?")

        self.assertIn("Help hub", text)
        self.assertIn("Next useful moves now:", text)
        self.assertIn("what can I do with the data?", text)
        self.assertIn("With the data you can:", text)

    def test_runtime_routes_spanish_next_step_follow_up_to_contextual_help(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="next-step-data-follow-up-es")
            runtime.web_retriever = _StubWebRetriever([])
            runtime.ask("qué es un dataset?")

            text = runtime.ask("qué hago ahora?")

        self.assertIn("Centro de ayuda", text)
        self.assertIn("Siguientes movimientos útiles:", text)
        self.assertIn("qué puedo hacer con los datos?", text)
        self.assertIn("Con los datos puedes:", text)

    def test_runtime_routes_next_step_after_modeling_to_actionable_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0042"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0042",
                "tickers": ["SPY"],
                "rows": {"raw": 250, "cleaned": 250},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                    "feature_columns": ["open", "close", "return_1d", "sma_5", "ticker"],
                    "derived_columns": ["return_1d", "sma_5"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.73, "disagreement": False},
                "motor": {"selected": "ensemble", "decision": "long"},
            }
            result = {
                "run_id": "run_0042",
                "manifest": {"request": {"tickers": ["SPY"]}},
                "modeling": {"selected_model": "ensemble"},
                "final_confidence": 0.73,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0042", "request": {"tickers": ["SPY"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="next-step-modeling")
            runtime.ask("what did the model predict?")
            text = runtime.ask("what should I do next?")

        self.assertIn("Help hub", text)
        self.assertIn("Next useful moves now:", text)
        self.assertIn("ask for the orchestrator final decision for SPY", text)
        self.assertIn("review why it decided that way", text)

    def test_runtime_routes_next_step_after_orchestrator_to_actionable_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0043"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0043",
                "tickers": ["BTC-USD"],
                "rows": {"raw": 365, "cleaned": 365},
                "models": {"final_decision": "short", "confidence": 0.61, "disagreement": True},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {
                "run_id": "run_0043",
                "manifest": {"request": {"tickers": ["BTC-USD"]}},
                "final_confidence": 0.61,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0043", "request": {"tickers": ["BTC-USD"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="next-step-orchestrator")
            runtime.ask("why did it decide that way?")
            text = runtime.ask("what should I do next?")

        self.assertIn("Help hub", text)
        self.assertIn("Next useful moves now:", text)
        self.assertIn("review why it decided short", text)
        self.assertIn("go back to modeling to inspect votes and confidence", text)

    def test_runtime_routes_y_luego_from_extraction_to_cleaning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0044"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0044",
                "tickers": ["AAPL"],
                "rows": {"raw": 120, "cleaned": 118},
                "date_range": {"start": "2025-01-01", "end": "2025-06-30"},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                    "feature_columns": ["open", "close", "return_1d", "sma_5", "ticker"],
                    "derived_columns": ["return_1d", "sma_5"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.66, "disagreement": False},
                "motor": {"selected": "ensemble", "decision": "long"},
            }
            result = {
                "run_id": "run_0044",
                "manifest": {"request": {"tickers": ["AAPL"]}},
                "modeling": {"selected_model": "ensemble"},
                "final_confidence": 0.66,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0044", "request": {"tickers": ["AAPL"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="followup-progress-cleaning")
            runtime.ask("what happened in extraction?")
            text = runtime.ask("y luego?")

        self.assertIn("Run run_0044. I cleaned 120 -> 118 rows.", text)
        self.assertIn("Features:", text)

    def test_runtime_routes_y_luego_from_cleaning_to_modeling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0045"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0045",
                "tickers": ["MSFT"],
                "rows": {"raw": 200, "cleaned": 200},
                "date_range": {"start": "2025-01-01", "end": "2025-09-01"},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                    "feature_columns": ["open", "close", "return_1d", "sma_5", "ticker"],
                    "derived_columns": ["return_1d", "sma_5"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "short", "confidence": 0.58, "disagreement": True},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {
                "run_id": "run_0045",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "modeling": {"selected_model": "majority"},
                "final_confidence": 0.58,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0045", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="followup-progress-modeling")
            runtime.ask("what happened in cleaning?")
            text = runtime.ask("y luego?")

        self.assertIn("Run run_0045. The models predict short", text)
        self.assertIn("Votes:", text)

    def test_runtime_can_answer_arbitrage_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-arbitrage")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is arbitrage?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic brief: arbitrage.", text)
        self.assertIn("price difference", text.lower())
        self.assertIn("local glossary already had a direct definition", text.lower())

    def test_runtime_can_answer_volatility_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-volatility")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is volatility?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic brief: volatility.", text)
        self.assertIn("how much and how quickly a price moves", text.lower())

    def test_runtime_can_answer_risk_vs_return_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-risk-return")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is the difference between risk and return?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic comparison brief:", text)
        self.assertIn("performance and risk concepts", text.lower())
        self.assertIn("risk:", text.lower())
        self.assertIn("return:", text.lower())

    def test_runtime_can_answer_benchmark_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-benchmark")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("benchmark")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic brief: benchmark.", text)
        self.assertIn("reference point", text.lower())

    def test_runtime_can_answer_benchmark_vs_portfolio_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-benchmark-portfolio")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is the difference between benchmark and portfolio?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic comparison brief:", text)
        self.assertIn("reference benchmark", text.lower())
        self.assertIn("portfolio being evaluated", text.lower())

    def test_runtime_can_answer_pnl_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-pnl")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is pnl?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic brief: PnL.", text)
        self.assertIn("profit and loss", text.lower())

    def test_runtime_can_answer_tracking_error_vs_information_ratio_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-te-ir")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is the difference between tracking error and information ratio?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic comparison brief:", text)
        self.assertIn("tracking error", text.lower())
        self.assertIn("information ratio", text.lower())

    def test_runtime_can_answer_concept_comparison_from_web_without_local_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-web-compare")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="Forex",
                        snippet="Forex is the global foreign exchange market where currency pairs trade.",
                        url="https://example.com/forex",
                    ),
                    WebFact(
                        title="Crypto",
                        snippet="Crypto markets trade digital assets against quote currencies or stablecoins.",
                        url="https://example.com/crypto",
                    ),
                ]
            )

            text = runtime.ask("what is the difference between forex and crypto?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: web.", text)
        self.assertIn("Semantic comparison brief:", text)
        self.assertIn("forex:", text.lower())
        self.assertIn("crypto:", text.lower())
        self.assertIn("Forex is the global foreign exchange market", text)
        self.assertIn("Crypto markets trade digital assets", text)
        self.assertIn("Source summary: 2 external sources.", text)
        self.assertIn("Source: https://example.com/forex.", text)
        self.assertIn("Additional sources: https://example.com/crypto.", text)

    def test_runtime_can_answer_local_concept_comparison_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-compare")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is the difference between long and short?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic comparison brief:", text)
        self.assertIn("both describe trading mechanics or risk management.", text.lower())
        self.assertIn("local glossary comparisons", text.lower())

    def test_runtime_can_answer_semantic_lookup_from_local_domain_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-local-brief")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what does hold mean?")

        self.assertIn("Response mode: interpreted.", text)
        self.assertIn("Certainty: inferred from local domain context.", text)
        self.assertIn("Grounding: local.", text)
        self.assertIn("Local semantic brief: hold.", text)
        self.assertIn("does not see enough edge", text)

    def test_runtime_explains_generic_semantic_lookup_without_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-no-web")
            runtime.web_retriever = _StubWebRetriever([])

            text = runtime.ask("what is wibble?")

        self.assertIn("I understood this as a question about wibble.", text)
        self.assertIn("I can search for external context to complete that idea.", text)

    def test_runtime_can_answer_spanish_semantic_lookup_from_web_without_local_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-web-brief-es")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="Mercado forex",
                        snippet="Forex es el mercado global donde se negocian pares de divisas.",
                        url="https://example.com/forex-es",
                    )
                ]
            )

            text = runtime.ask("qué es forex?")

        self.assertIn("Modo de respuesta: interpretado.", text)
        self.assertIn("Base factual: web.", text)
        self.assertIn("Brief semántico web:", text)
        self.assertIn("Forex es el mercado global", text)
        self.assertIn("Resumen de fuentes: 1 fuente externa.", text)
        self.assertIn("Fuente: https://example.com/forex-es.", text)

    def test_runtime_ignores_greeting_prefix_for_spanish_semantic_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="semantic-web-brief-greeting")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="Mercado forex",
                        snippet="Forex es el mercado global donde se negocian pares de divisas.",
                        url="https://example.com/forex-es",
                    )
                ]
            )

            text = runtime.ask("hola, qué es forex?")

        self.assertIn("Modo de respuesta: interpretado.", text)
        self.assertIn("Base factual: web.", text)
        self.assertIn("Brief semántico web:", text)

    def test_runtime_uses_market_memory_for_semantic_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0001"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0001",
                "tickers": ["BTC-USD"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.47},
            }
            result = {"run_id": "run_0001", "manifest": {"request": {"tickers": ["BTC-USD"]}}}
            manifest = {"run_id": "run_0001", "request": {"tickers": ["BTC-USD"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="semantic-follow-up-market")
            runtime.web_retriever = _StubWebRetriever([])
            runtime.ask("de qué tipo de mercado hace parte el símbolo usado en run_0001?")
            text = runtime.ask("y qué significa ese mercado?")

            self.assertIn("Modo de respuesta: interpretado.", text)
            self.assertIn("Certeza: inferida desde contexto local del dominio.", text)
            self.assertIn("Base factual: local.", text)
            self.assertIn("Brief semántico local:", text)
            self.assertIn("Ancla local: run=run_0001 | símbolo=BTC-USD.", text)

    def test_summary_from_result_marks_compare_binance_legacy(self) -> None:
        result = {
            "run_id": "run_9999",
            "status": "succeeded",
            "final_decision": "long",
            "final_confidence": 0.83,
            "motor": {
                "requested": "auto",
                "selected": "majority",
                "decision": "long",
            },
            "manifest": {
                "run_id": "run_9999",
                "reviewer_used": False,
                "reviewer_provider": None,
                "request": {
                    "tickers": ["BTC-USD"],
                    "start": "2024-01-01",
                    "end": "2025-01-01",
                    "interval": "1d",
                    "review_mode": "off",
                    "compare_binance": True,
                },
                "artifacts": [{}],
                "logs": [{}, {}],
            },
            "extraction": {
                "rows": 20,
                "tickers": ["BTC-USD"],
                "raw_data": {
                    "rows": 20,
                    "columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                },
            },
            "cleaning": {
                "rows_out": 18,
                "clean_data": {
                    "feature_columns": ["open", "high", "low", "close", "adj_close", "volume", "ticker"],
                    "target_column": "target_direction",
                    "rows_out": 18,
                },
            },
            "modeling": {
                "selected_model": "majority",
                "disagreement": True,
                "rationale": "Local models disagreed.",
                "models": [
                    {
                        "model_name": "logistic_regression",
                        "latest_prediction": "long",
                        "latest_probability": 0.71,
                        "confidence": 0.71,
                        "validation_metrics": {"accuracy": 0.6, "roc_auc": 0.65},
                    }
                ],
            },
            "legacy_analysis": {"enabled": True, "asset": "BTC", "trust_score_pct": 91.0},
        }

        summary = _summary_from_result(result)

        self.assertEqual(summary["run_mode"], "local_plus_binance_legacy")
        self.assertEqual(summary["models"]["final_decision"], "long")
        self.assertEqual(summary["rows"]["raw"], 20)
        self.assertEqual(summary["rows"]["cleaned"], 18)
        self.assertEqual(summary["motor"]["selected"], "majority")

        report = _format_latest_report(summary)
        report_es = _format_latest_report(summary, "es")
        self.assertIn("local_plus_binance_legacy", report)
        self.assertIn("Decision: long", report)
        self.assertIn("Motor: requested=auto selected=majority decision=long.", report)
        self.assertIn("En lenguaje simple, extraje 20 filas de yfinance", report_es)

    def test_summary_from_result_marks_groq_brain_mode(self) -> None:
        result = {
            "run_id": "run_1002",
            "status": "succeeded",
            "final_decision": "short",
            "final_confidence": 0.44,
            "deterministic_decision": "long",
            "deterministic_confidence": 0.6633333333,
            "decision_source": "groq_brain",
            "groq_brain": {
                "provider": "groq_brain",
                "decision": "short",
                "confidence": 0.44,
                "explanation": "Experimental Groq brain chose short.",
                "risks": ["experimental mode"],
            },
            "motor": {
                "requested": "auto",
                "selected": "ensemble",
                "decision": "short",
                "decision_path": "groq_brain_override",
            },
            "manifest": {
                "run_id": "run_1002",
                "reviewer_used": False,
                "reviewer_provider": None,
                "decision_source": "groq_brain",
                "groq_brain_used": True,
                "groq_brain_provider": "groq_brain",
                "deterministic_decision": "long",
                "deterministic_confidence": 0.6633333333,
                "request": {
                    "tickers": ["AAPL"],
                    "start": "2024-01-01",
                    "end": "2025-01-01",
                    "interval": "1d",
                    "review_mode": "off",
                    "experimental_groq_brain": True,
                },
                "artifacts": [{}],
                "logs": [{}, {}],
            },
            "extraction": {
                "rows": 20,
                "tickers": ["AAPL"],
                "raw_data": {
                    "rows": 20,
                    "columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                },
            },
            "cleaning": {
                "rows_out": 18,
                "clean_data": {
                    "feature_columns": ["open", "high", "low", "close", "adj_close", "volume", "ticker"],
                    "target_column": "target_direction",
                    "rows_out": 18,
                },
            },
            "modeling": {
                "selected_model": "majority",
                "disagreement": True,
                "rationale": "Local models disagreed.",
                "ensemble_prediction": "long",
                "ensemble_probability": 0.6633333333,
                "majority_prediction": "long",
                "models": [
                    {
                        "model_name": "logistic_regression",
                        "latest_prediction": "long",
                        "latest_probability": 0.71,
                        "confidence": 0.71,
                        "validation_metrics": {"accuracy": 0.6, "roc_auc": 0.65},
                    }
                ],
            },
            "legacy_analysis": None,
            "source_comparison": None,
        }

        summary = _summary_from_result(result)

        self.assertEqual(summary["run_mode"], "local_only_groq_brain")
        self.assertTrue(summary["brain"]["enabled"])
        self.assertTrue(summary["brain"]["used"])
        self.assertEqual(summary["brain"]["decision_source"], "groq_brain")
        self.assertEqual(summary["brain"]["deterministic_decision"], "long")
        self.assertEqual(summary["models"]["final_decision"], "short")
        self.assertEqual(summary["models"]["deterministic_decision"], "long")

        report = _format_latest_report(summary)
        self.assertIn("Groq brain mode", report)
        self.assertIn("short", report)
        self.assertIn("deterministic", report)

    def test_stage_brief_formatters_use_bundle_data(self) -> None:
        summary = {
            "run_id": "run_1000",
            "stage_briefs": {
                "extraction": {
                    "summary": "Pulled rows from yfinance.",
                    "motor": "yfinance",
                    "key_points": ["rows=10", "missing=none"],
                    "risks": ["low coverage"],
                }
            },
            "motor": {
                "requested": "auto",
                "selected": "majority",
                "decision": "short",
                "decision_path": "local_majority",
            },
        }
        result = {
            "legacy_analysis": None,
            "source_comparison": None,
        }

        self.assertIn("Pulled rows from yfinance.", _format_stage_brief(summary, result, "extraction"))
        self.assertIn("Motor: yfinance.", _format_stage_brief(summary, result, "extraction"))
        self.assertIn("Motor: requested=auto.", _format_motor(summary, result))
        self.assertIn("Run run_1000.", _format_stage_brief(summary, result, "extraction"))

    def test_agent_card_formatter_lists_commands_and_examples(self) -> None:
        card = _format_agent_card("cleaning", "es")

        self.assertIn("__agent_card__:cleaning:es", card)
        self.assertIn("Preguntas naturales", card)
        self.assertIn("qué símbolos hay en los datos limpios?", card)
        self.assertIn("Ejemplos", card)
        self.assertIn("qué dice esta fila limpia?", card)
        self.assertIn("Handoff", card)

    def test_session_status_formatter_reports_mode_and_confidence(self) -> None:
        summary = {
            "run_id": "run_1002",
            "run_mode": "local_only_groq_brain",
            "models": {
                "final_decision": "short",
                "confidence": 0.44,
                "deterministic_decision": "long",
                "deterministic_confidence": 0.6633333333,
                "decision_source": "groq_brain",
                "reviewer_used": False,
            },
            "brain": {
                "enabled": True,
                "used": True,
                "decision": "short",
                "confidence": 0.44,
                "decision_source": "groq_brain",
            },
            "comparison": {
                "decision_path": "groq_brain_override",
                "extraction_health": "good",
                "modeling_health": "mixed",
            },
            "selection": {
                "strategy": "majority_vote",
                "reason": "Los modelos locales discreparon.",
            },
            "source_comparison": {
                "enabled": True,
                "source_1": "yfinance",
                "source_2": "Binance",
            },
            "legacy_analysis": {
                "enabled": True,
                "asset": "AAPL",
            },
            "rows": {
                "raw": 251,
                "cleaned": 251,
            },
            "tickers": ["AAPL"],
            "review_mode": "off",
        }
        state = AssistantState(
            session_id="session-test",
            current_mode="local_only_groq_brain",
            preferred_language="es",
            last_intent="show_prediction",
            notes=["show_prediction:modeling:run_1002"],
        )

        text = _format_session_status(summary, state, "es", groq_available=True)

        self.assertIn("Estado de sesión", text)
        self.assertIn("Modo: Groq brain experimental (local_only_groq_brain).", text)
        self.assertIn("Contexto actual: run=run_1002 | símbolo=AAPL | etapa=modelado.", text)
        self.assertIn("Groq: sí; brain usado: sí.", text)
        self.assertIn("Confianza: short 0.4400; base long 0.6633; fuente groq_brain.", text)
        self.assertIn("Siguiente agente: orquestador.", text)

    def test_session_status_full_formatter_reports_detailed_fields(self) -> None:
        summary = {
            "run_id": "run_1002",
            "run_mode": "local_only_groq_brain",
            "tickers": ["AAPL"],
            "rows": {"raw": 251, "cleaned": 251},
            "models": {
                "final_decision": "short",
                "confidence": 0.44,
                "deterministic_decision": "long",
                "deterministic_confidence": 0.6633333333,
                "decision_source": "groq_brain",
                "reviewer_used": False,
                "disagreement": True,
                "disagreement_reason": "Los modelos locales discreparon.",
            },
            "brain": {
                "enabled": True,
                "used": True,
                "decision": "short",
                "confidence": 0.44,
                "decision_source": "groq_brain",
            },
            "comparison": {
                "decision_path": "groq_brain_override",
                "extraction_health": "good",
                "modeling_health": "mixed",
            },
            "selection": {
                "strategy": "majority_vote",
                "reason": "Los modelos locales discreparon.",
            },
            "source_comparison": {
                "enabled": True,
                "source_1": "yfinance",
                "source_2": "Binance",
            },
            "legacy_analysis": {
                "enabled": True,
                "asset": "AAPL",
            },
            "review_mode": "off",
        }
        state = AssistantState(
            session_id="session-test",
            current_mode="local_only_groq_brain",
            preferred_language="es",
            last_intent="show_prediction",
            current_asset="AAPL",
            pending_task="pregunta al orquestador",
            notes=["show_prediction:modeling:run_1002"],
        )

        text = _format_session_status(summary, state, "es", groq_available=True, detailed=True)
        lines = text.splitlines()

        self.assertIn("Estado de sesión completo", text)
        self.assertEqual("Contexto actual: run=run_1002 | símbolo=AAPL | etapa=modelado.", lines[1])
        self.assertIn("Resumen: modo=Groq brain experimental (local_only_groq_brain); run=run_1002; idioma=es; activos=AAPL.", text)
        self.assertIn("Estado de Groq: disponible=sí; brain habilitado=sí; brain usado=sí.", text)
        self.assertIn("Decisión: short (0.4400); base long (0.6633); fuente groq_brain.", text)
        self.assertIn("Revisor: off (usado: no).", text)
        self.assertIn("Ruta de decisión: groq_brain_override.", text)
        self.assertIn("Selección: majority_vote porque Los modelos locales discreparon.", text)
        self.assertIn("Salud: extraction=good modeling=mixed.", text)
        self.assertIn("Los modelos discreparon: True.", text)
        self.assertIn("Comparación Binance: sí (yfinance vs Binance).", text)
        self.assertIn("Puente legacy: sí (AAPL).", text)
        self.assertIn("Navegación: siguiente agente=orquestador; tarea pendiente=orquestador.", text)

    def test_format_session_status_full_includes_conversational_trace(self) -> None:
        summary = {
            "run_id": "run_1002",
            "tickers": ["AAPL"],
            "models": {"final_decision": "long", "confidence": 0.61},
        }
        state = AssistantState(
            session_id="session-trace-test",
            preferred_language="es",
            current_asset="AAPL",
            last_turn_trace={
                "act": "definition",
                "subject": "groq",
                "final_intent": "show_semantic_lookup",
                "source_mode": "web",
                "source_policy": "web_required",
                "web_required": True,
                "allow_run": False,
                "override_memory": True,
                "planner_risk": "medium",
                "planner_steps": 2,
            },
        )

        text = _format_session_status(summary, state, "es", groq_available=True, detailed=True)

        self.assertIn("Traza conversacional: acto=definition; sujeto=groq; intención=show_semantic_lookup;", text)
        self.assertIn("fuente=web; política=web_required; web obligatoria=sí.", text)
        self.assertIn("Planner: riesgo=medium; pasos=2; allow_run=no; override_memory=sí.", text)

    def test_orchestrator_stage_brief_uses_run_recap(self) -> None:
        summary = {
            "rows": {"raw": 251, "cleaned": 251},
            "models": {"final_decision": "long", "confidence": 0.6169, "selected": "majority"},
            "data": {
                "feature_columns": ["open", "high", "low", "close", "adj_close", "volume", "ticker"],
                "target_column": "target_direction",
            },
            "motor": {
                "requested": "auto",
                "selected": "majority",
                "decision": "long",
            },
            "review_mode": "auto",
            "stage_briefs": {},
        }
        result = {
            "cleaning": {
                "feature_columns": ["open", "high", "low", "close", "adj_close", "volume", "ticker"],
                "target_column": "target_direction",
            },
            "motor": {
                "requested": "auto",
                "selected": "majority",
                "decision": "long",
            },
        }

        recap = _format_stage_brief(summary, result, "orchestrator")

        self.assertIn("The final decision was long", recap)
        self.assertIn("251 raw rows were cleaned into 251 model-ready rows", recap)
        self.assertIn("cleaner data reduces noise", recap)

    def test_symbol_guide_lists_supported_tickers_and_compare_option(self) -> None:
        summary = {"run_id": "run_1001", "tickers": ["BTC-USD"]}
        result = {}

        guide = _format_symbol_guide(summary, result, "en")

        self.assertIn("Suggested symbols", guide)
        self.assertIn("BTC-USD", guide)
        self.assertIn("EURUSD=X", guide)
        self.assertIn("compare-binance", guide)
        self.assertIn("How to use them", guide)
        self.assertIn("Quick examples", guide)
        self.assertIn("AAPL", guide)

    def test_asset_used_and_clean_data_and_decision_explanations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cleaned_path = root / "clean_market_data.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-20",
                        "ticker": "MSFT",
                        "open": 100.0,
                        "high": 105.0,
                        "low": 99.0,
                        "close": 104.0,
                        "adj_close": 104.0,
                        "volume": 1000,
                        "target_direction": 1,
                    },
                    {
                        "date": "2026-03-21",
                        "ticker": "MSFT",
                        "open": 104.0,
                        "high": 106.0,
                        "low": 103.0,
                        "close": 105.0,
                        "adj_close": 105.0,
                        "volume": 1100,
                        "target_direction": 1,
                    },
                ]
            ).to_csv(cleaned_path, index=False)

            summary = {
                "run_id": "run_2000",
                "tickers": ["MSFT"],
                "date_range": {"start": "2026-03-01", "end": "2026-03-21"},
                "files": {"cleaned": str(cleaned_path)},
                "rows": {"raw": 2, "cleaned": 2},
                "data": {
                    "feature_columns": ["open", "high", "low", "close", "adj_close", "volume", "ticker"],
                    "target_column": "target_direction",
                },
                "models": {
                    "final_decision": "long",
                    "confidence": 0.8123,
                    "disagreement": True,
                },
                "selection": {"strategy": "majority_vote", "reason": "The local models disagreed."},
                "comparison": {"decision_path": "local_majority"},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {
                "manifest": {
                    "request": {
                        "compare_binance": True,
                        "comparison_asset": "MSFT",
                        "comparison_yfinance_ticker": "MSFT",
                        "comparison_binance_symbol": "MSFTUSDT",
                    }
                }
            }

            asset_text = _format_asset_used(summary, result, "en")
            hub_text = _format_clean_data_view(summary, result, "clean data", "en")
            hub_text_es = _format_clean_data_view(summary, result, "datos limpios", "es")
            clean_text = _format_clean_data_view(summary, result, "show clean market data open high", "en")
            analysis_text = _format_clean_data_view(summary, result, "analyze the clean row for MSFT on 2026-03-21", "en")
            analysis_text_es = _format_clean_data_view(summary, result, "analiza la fila limpia de MSFT el 2026-03-21", "es")
            decision_text = _format_decision_explanation(summary, result, "en")
            metrics_text = _format_market_metrics(summary, result, "what is the volume of MSFT?", ["MSFT"], "en")
            groq_text = _format_groq_status(True, "llama-3.1-8b-instant", "https://api.groq.com/openai/v1/chat/completions", "en")

            self.assertIn("Last symbol used", asset_text)
            self.assertIn("MSFT", asset_text)
            self.assertIn("Binance comparison", asset_text)
            self.assertIn("Clean data hub", hub_text)
            self.assertIn("Open views", hub_text)
            self.assertIn("Type 8 or clean data to come back here.", hub_text)
            self.assertIn("Centro de datos limpios", hub_text_es)
            self.assertIn("Vistas abiertas", hub_text_es)
            self.assertIn("Escribe 8 o clean data para volver aquí.", hub_text_es)
            self.assertIn("Requested columns: open, high", clean_text)
            self.assertIn("MSFT", clean_text)
            self.assertIn("Clean market data analysis", analysis_text)
            self.assertIn("Row analysis", analysis_text)
            self.assertIn("target_direction", analysis_text)
            self.assertIn("Análisis de clean_market_data", analysis_text_es)
            self.assertIn("Análisis de fila", analysis_text_es)
            self.assertIn("The final decision was long", decision_text)
            self.assertIn("long means the model expects an upward move", decision_text)
            self.assertIn("cleaner data reduces noise", decision_text)
            self.assertIn("Market metrics", metrics_text)
            self.assertIn("MSFT", metrics_text)
            self.assertIn("Volume", metrics_text)
            self.assertIn("Groq status", groq_text)
            self.assertIn("Endpoint", groq_text)
            mode_guide = _format_mode_guide("es")
            self.assertIn("Centro de modos", mode_guide)
            self.assertIn("local_only", mode_guide)
            self.assertIn("compare-binance", mode_guide)
            self.assertIn("--groq-brain", mode_guide)
            self.assertIn("Escribe un modo para ver su detalle", mode_guide)

            fx_path = root / "clean_market_data_fx.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-20",
                        "ticker": "EURUSD=X",
                        "open": 1.10,
                        "high": 1.11,
                        "low": 1.09,
                        "close": 1.105,
                        "adj_close": 1.105,
                        "volume": 0,
                        "volatility_5": 0.0123,
                        "target_direction": 1,
                    },
                    {
                        "date": "2026-03-21",
                        "ticker": "EURUSD=X",
                        "open": 1.11,
                        "high": 1.12,
                        "low": 1.10,
                        "close": 1.115,
                        "adj_close": 1.115,
                        "volume": 0,
                        "volatility_5": 0.0137,
                        "target_direction": 1,
                    },
                ]
            ).to_csv(fx_path, index=False)

            fx_summary = dict(summary)
            fx_summary["tickers"] = ["EURUSD=X"]
            fx_summary["files"] = {"cleaned": str(fx_path)}
            fx_result = dict(result)
            fx_result["manifest"] = {
                "request": {
                    "tickers": ["EURUSD=X"],
                }
            }

            fx_metrics = _format_market_metrics(fx_summary, fx_result, "cuál es el volumen de EURUSD?", ["EURUSD"], "es")
            self.assertIn("EURUSD=X", fx_metrics)
            self.assertIn("Volumen", fx_metrics)
            self.assertIn("Métricas de mercado", fx_metrics)

    def test_runtime_can_target_a_specific_run_by_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            for run_id, ticker in (("run_0007", "MSFT"), ("run_0008", "AAPL")):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                summary = {
                    "run_id": run_id,
                    "tickers": [ticker],
                    "date_range": {"start": "2026-03-01", "end": "2026-03-21"},
                    "rows": {"raw": 1, "cleaned": 1},
                    "data": {
                        "raw_columns": ["date", "ticker", "open", "high", "low", "close", "volume"],
                        "feature_columns": ["open", "high", "low", "close", "volume", "return_1d", "ticker"],
                        "derived_columns": ["return_1d"],
                        "target_column": "target_direction",
                    },
                    "models": {"final_decision": "long" if run_id == "run_0007" else "short", "confidence": 0.91},
                    "motor": {"requested": "auto", "selected": "majority", "decision": "long" if run_id == "run_0007" else "short"},
                }
                result = {
                    "run_id": run_id,
                    "manifest": {"request": {"tickers": [ticker]}},
                    "modeling": {
                        "selected_model": "majority",
                    },
                }
                (run_dir / "summary.json").write_text(json.dumps(summary))
                (run_dir / "result.json").write_text(json.dumps(result))
                (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "request": {"tickers": [ticker]}}))

            state = load_state(str(root), "run-id-test")
            state.last_run_id = "run_0008"
            save_state(str(root), state)

            runtime = AssistantRuntime(artifact_root=str(root), session_id="run-id-test")
            answer = runtime.ask("run_0007 what symbol was used?")
            cleaning_answer = runtime.ask("run_0007 limpieza")
            modeling_answer = runtime.ask("run_0007 modelado")
            model_variables_answer = runtime.ask("run_0007 what variables did the model use?")
            mid_sentence_answer = runtime.ask("explain the variables used in run_0007")
            decision_answer = runtime.ask("why did run_0007 decide that way?")

            self.assertIn("run_0007", answer)
            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Certainty: confirmed from run artifacts.", answer)
            self.assertIn("MSFT", answer)
            self.assertNotIn("AAPL", answer)
            self.assertIn("Response mode: interpreted.", cleaning_answer)
            self.assertIn("I cleaned", cleaning_answer)
            self.assertIn("run_0007", cleaning_answer)
            self.assertIn("Response mode: interpreted.", modeling_answer)
            self.assertIn("The models predict", modeling_answer)
            self.assertIn("run_0007", modeling_answer)
            self.assertIn("run_0007", model_variables_answer)
            self.assertIn("Response mode: interpreted.", model_variables_answer)
            self.assertIn("Certainty: confirmed from run artifacts.", model_variables_answer)
            self.assertIn("Base variables", model_variables_answer)
            self.assertIn("Derived variables", model_variables_answer)
            self.assertIn("Target: target_direction", model_variables_answer)
            self.assertIn("one-hot encoded", model_variables_answer)
            self.assertIn("Selected motor: majority", model_variables_answer)
            self.assertIn("run_0007", mid_sentence_answer)
            self.assertIn("Response mode: interpreted.", mid_sentence_answer)
            self.assertIn("Certainty: inferred from run artifacts.", mid_sentence_answer)
            self.assertIn("target_direction", mid_sentence_answer)
            self.assertIn("Response mode: interpreted.", decision_answer)
            self.assertIn("Certainty: inferred from run artifacts.", decision_answer)

    def test_runtime_uses_last_intent_for_vague_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "date_range": {"start": "2026-03-01", "end": "2026-03-21"},
                "rows": {"raw": 1, "cleaned": 1},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "high", "low", "close", "volume"],
                    "feature_columns": ["open", "high", "low", "close", "volume", "return_1d", "ticker"],
                    "derived_columns": ["return_1d"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "modeling": {"selected_model": "majority"},
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="follow-up-test")
            first_answer = runtime.ask("run_0007 what variables did the model use?")
            second_answer = runtime.ask("explain that better")

            self.assertIn("Response mode: interpreted.", first_answer)
            self.assertIn("Response mode: interpreted.", second_answer)
            self.assertIn("run_0007", second_answer)
            self.assertIn("Derived variables", second_answer)

    def test_runtime_routes_causal_follow_up_to_decision_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {"feature_columns": ["open", "close", "volume"], "target_column": "target_direction"},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"requested": "auto", "selected": "majority", "decision": "short"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "final_confidence": 0.48,
                "modeling": {"selected_model": "majority"},
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="follow-up-causal-test")
            first_answer = runtime.ask("qué pasa hoy")
            second_answer = runtime.ask("y entonces?")

            self.assertIn("Modo de respuesta: interpretado.", first_answer)
            self.assertIn("Modo de respuesta: interpretado.", second_answer)
            self.assertIn("run_0007", second_answer)
            self.assertIn("La decisión final fue", second_answer)

    def test_runtime_routes_ultra_short_spanish_follow_up_to_decision_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {"feature_columns": ["open", "close", "volume"], "target_column": "target_direction"},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"requested": "auto", "selected": "majority", "decision": "short"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "final_confidence": 0.48,
                "modeling": {"selected_model": "majority"},
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="follow-up-ultra-short-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("entonces?")

            self.assertIn("Modo de respuesta: interpretado.", answer)
            self.assertIn("run_0007", answer)
            self.assertIn("La decisión final fue", answer)

    def test_runtime_routes_elliptical_market_follow_up_from_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0001"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0001",
                "tickers": ["BTC-USD"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.47},
            }
            result = {"run_id": "run_0001", "manifest": {"request": {"tickers": ["BTC-USD"]}}}
            manifest = {"run_id": "run_0001", "request": {"tickers": ["BTC-USD"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-market-test")
            first_answer = runtime.ask("dime el activo usado para analisis en run_0001")
            second_answer = runtime.ask("y el mercado?")

            self.assertIn("BTC-USD", first_answer)
            self.assertIn("BTC-USD pertenece al mercado cripto", second_answer)
            self.assertIn("Modo de respuesta: interpretado.", second_answer)

    def test_runtime_routes_elliptical_columns_follow_up_from_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0001"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0001",
                "tickers": ["BTC-USD"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                    "feature_columns": ["open", "high", "low", "close", "adj_close", "volume", "ticker"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "short", "confidence": 0.47},
            }
            result = {"run_id": "run_0001", "manifest": {"request": {"tickers": ["BTC-USD"]}}, "extraction": {"rows": 10}}
            manifest = {"run_id": "run_0001", "request": {"tickers": ["BTC-USD"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-columns-test")
            runtime.ask("podrias explicarme el resultado de la extraccion run_0001?")
            answer = runtime.ask("y las columnas?")

            self.assertIn("date, ticker, open, high, low, close, adj_close, volume", answer)
            self.assertIn("run_0001", answer)

    def test_runtime_routes_elliptical_variables_follow_up_to_model_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "high", "low", "close", "volume", "return_1d", "ticker"],
                    "derived_columns": ["return_1d"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "modeling": {"selected_model": "majority"}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-variables-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("y las variables?")

            self.assertIn("Modo de respuesta: interpretado.", answer)
            self.assertIn("variables explicativas", answer)
            self.assertIn("run_0007", answer)

    def test_runtime_routes_elliptical_decision_follow_up_to_decision_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {"feature_columns": ["open", "close", "volume"], "target_column": "target_direction"},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"requested": "auto", "selected": "majority", "decision": "short"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "final_confidence": 0.48,
                "modeling": {"selected_model": "majority"},
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-decision-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("y la decisión?")

            self.assertIn("Modo de respuesta: interpretado.", answer)
            self.assertIn("La decisión final fue", answer)
            self.assertIn("run_0007", answer)

    def test_runtime_routes_elliptical_metrics_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            cleaned_path = root / "clean_market_data.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-21",
                        "ticker": "MSFT",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.5,
                        "close": 10.5,
                        "adj_close": 10.5,
                        "volume": 2000,
                        "return_1d": 0.02,
                        "range_pct": 0.15,
                        "body_pct": 0.05,
                        "sma_5": 10.1,
                        "sma_10": 9.9,
                        "volatility_5": 0.12,
                        "volume_sma_5": 1800,
                        "future_close": 10.8,
                        "future_return": 0.03,
                        "target_direction": 1,
                    }
                ]
            ).to_csv(cleaned_path, index=False)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "files": {"cleaned": str(cleaned_path)},
                "rows": {"raw": 10, "cleaned": 1},
                "models": {"final_decision": "long", "confidence": 0.91},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-metrics-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("y las métricas?")

            self.assertIn("Modo de respuesta: interpretado.", answer)
            self.assertIn("Métricas de mercado", answer)
            self.assertIn("run_0007", answer)

    def test_runtime_routes_entity_symbol_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "long", "confidence": 0.91},
                "motor": {"selected": "majority", "decision": "long"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="entity-symbol-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("ese símbolo?")

            self.assertIn("run_0007", answer)
            self.assertIn("MSFT", answer)

    def test_runtime_routes_entity_decision_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {"feature_columns": ["open", "close", "volume"], "target_column": "target_direction"},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"requested": "auto", "selected": "majority", "decision": "short"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "final_confidence": 0.48,
                "modeling": {"selected_model": "majority"},
            }
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="entity-decision-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("esa decisión?")

            self.assertIn("run_0007", answer)
            self.assertRegex(answer, r"(La decisión final fue|The final decision was)")

    def test_runtime_routes_entity_metric_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            cleaned_path = root / "clean_market_data.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-21",
                        "ticker": "MSFT",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.5,
                        "close": 10.5,
                        "adj_close": 10.5,
                        "volume": 2000,
                        "return_1d": 0.02,
                        "range_pct": 0.15,
                        "body_pct": 0.05,
                        "sma_5": 10.1,
                        "sma_10": 9.9,
                        "volatility_5": 0.12,
                        "volume_sma_5": 1800,
                        "future_close": 10.8,
                        "future_return": 0.03,
                        "target_direction": 1,
                    }
                ]
            ).to_csv(cleaned_path, index=False)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "files": {"cleaned": str(cleaned_path)},
                "rows": {"raw": 10, "cleaned": 1},
                "models": {"final_decision": "long", "confidence": 0.91},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="entity-metric-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("esa métrica?")

            self.assertIn("run_0007", answer)
            self.assertRegex(answer, r"(Métricas de mercado|Market metrics)")

    def test_runtime_routes_entity_row_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            cleaned_path = root / "clean_market_data.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-21",
                        "ticker": "MSFT",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.5,
                        "close": 10.5,
                        "adj_close": 10.5,
                        "volume": 2000,
                        "return_1d": 0.02,
                        "range_pct": 0.15,
                        "body_pct": 0.05,
                        "sma_5": 10.1,
                        "sma_10": 9.9,
                        "volatility_5": 0.12,
                        "volume_sma_5": 1800,
                        "future_close": 10.8,
                        "future_return": 0.03,
                        "target_direction": 1,
                    }
                ]
            ).to_csv(cleaned_path, index=False)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "files": {"cleaned": str(cleaned_path)},
                "rows": {"raw": 10, "cleaned": 1},
                "models": {"final_decision": "long", "confidence": 0.91},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="entity-row-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("esa fila?")

            self.assertIn("run_0007", answer)
            self.assertRegex(answer, r"(Fila|Row)")

    def test_runtime_persists_entity_memory_and_uses_bare_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "high", "low", "close", "volume", "return_1d", "ticker"],
                    "derived_columns": ["return_1d"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "modeling": {"selected_model": "majority"}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="entity-memory-test")
            first_answer = runtime.ask("run_0007 what variables did the model use?")
            second_answer = runtime.ask("that one?")
            state = load_state(str(root), "entity-memory-test")

            self.assertIn("run_0007", first_answer)
            self.assertIn("run_0007", second_answer)
            self.assertIn("Derived variables", second_answer)
            self.assertEqual((state.entity_memory or {}).get("last_focus"), "model_variables")
            self.assertEqual((state.entity_memory or {}).get("run_id"), "run_0007")
            self.assertEqual((state.entity_memory or {}).get("last_entity_kind"), "variable")
            self.assertEqual((((state.entity_memory or {}).get("by_kind") or {}).get("variable") or {}).get("run_id"), "run_0007")

    def test_runtime_uses_entity_memory_by_kind_for_variable_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for run_id, ticker in (("run_0007", "MSFT"), ("run_0008", "AAPL")):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                summary = {
                    "run_id": run_id,
                    "tickers": [ticker],
                    "rows": {"raw": 10, "cleaned": 9},
                    "data": {
                        "feature_columns": ["open", "high", "low", "close", "volume", "return_1d", "ticker"],
                        "derived_columns": ["return_1d"],
                        "target_column": "target_direction",
                    },
                    "models": {"final_decision": "long" if run_id == "run_0007" else "short", "confidence": 0.91},
                    "motor": {"requested": "auto", "selected": "majority", "decision": "long" if run_id == "run_0007" else "short"},
                }
                result = {"run_id": run_id, "manifest": {"request": {"tickers": [ticker]}}, "modeling": {"selected_model": "majority"}}
                manifest = {"run_id": run_id, "request": {"tickers": [ticker]}}
                (run_dir / "summary.json").write_text(json.dumps(summary))
                (run_dir / "result.json").write_text(json.dumps(result))
                (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="entity-memory-by-kind-test")
            runtime.ask("run_0007 what variables did the model use?")
            runtime.ask("run_0008 what happened today?")
            answer = runtime.ask("that variable?")
            state = load_state(str(root), "entity-memory-by-kind-test")

            self.assertIn("run_0007", answer)
            self.assertIn("Derived variables", answer)
            self.assertEqual((state.entity_memory or {}).get("run_id"), "run_0007")
            self.assertEqual((((state.entity_memory or {}).get("by_kind") or {}).get("variable") or {}).get("run_id"), "run_0007")
            self.assertEqual((((state.entity_memory or {}).get("by_kind") or {}).get("summary") or {}).get("run_id"), "run_0008")

    def test_runtime_uses_symbol_memory_for_market_type_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for run_id, ticker in (("run_0007", "MSFT"), ("run_0008", "AAPL")):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                summary = {
                    "run_id": run_id,
                    "tickers": [ticker],
                    "rows": {"raw": 10, "cleaned": 9},
                    "models": {"final_decision": "long" if run_id == "run_0007" else "short", "confidence": 0.91},
                    "motor": {"selected": "majority", "decision": "long" if run_id == "run_0007" else "short"},
                }
                result = {"run_id": run_id, "manifest": {"request": {"tickers": [ticker]}}}
                manifest = {"run_id": run_id, "request": {"tickers": [ticker]}}
                (run_dir / "summary.json").write_text(json.dumps(summary))
                (run_dir / "result.json").write_text(json.dumps(result))
                (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="entity-memory-market-test")
            runtime.ask("run_0007 what symbol was used?")
            runtime.ask("run_0008 what happened today?")
            answer = runtime.ask("that market?")

            self.assertIn("run_0007", answer)
            self.assertRegex(answer, r"(MSFT belongs to|MSFT pertenece)")

    def test_runtime_uses_comparison_memory_after_unrelated_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for run_id, ticker, decision, confidence in (
                ("run_0007", "MSFT", "long", 0.91),
                ("run_0008", "AAPL", "short", 0.66),
                ("run_0009", "NVDA", "long", 0.73),
            ):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                summary = {
                    "run_id": run_id,
                    "tickers": [ticker],
                    "rows": {"raw": 10, "cleaned": 9},
                    "models": {"final_decision": decision, "confidence": confidence},
                    "motor": {"selected": "majority", "decision": decision},
                }
                result = {
                    "run_id": run_id,
                    "manifest": {"request": {"tickers": [ticker]}},
                    "modeling": {"selected_model": "majority"},
                }
                manifest = {"run_id": run_id, "request": {"tickers": [ticker]}}
                (run_dir / "summary.json").write_text(json.dumps(summary))
                (run_dir / "result.json").write_text(json.dumps(result))
                (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="entity-memory-comparison-test")
            runtime.ask("compare run_0007 vs run_0008")
            runtime.ask("run_0009 what happened today?")
            answer = runtime.ask("that comparison?")

            self.assertIn("run_0007", answer)
            self.assertIn("run_0008", answer)
            self.assertIn("Comparison between run_0007 and run_0008.", answer)

    def test_runtime_routes_elliptical_row_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            cleaned_path = root / "clean_market_data.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-21",
                        "ticker": "MSFT",
                        "open": 10.0,
                        "high": 11.0,
                        "low": 9.5,
                        "close": 10.5,
                        "adj_close": 10.5,
                        "volume": 2000,
                        "return_1d": 0.02,
                        "range_pct": 0.15,
                        "body_pct": 0.05,
                        "sma_5": 10.1,
                        "sma_10": 9.9,
                        "volatility_5": 0.12,
                        "volume_sma_5": 1800,
                        "future_close": 10.8,
                        "future_return": 0.03,
                        "target_direction": 1,
                    }
                ]
            ).to_csv(cleaned_path, index=False)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "files": {"cleaned": str(cleaned_path)},
                "rows": {"raw": 10, "cleaned": 1},
                "models": {"final_decision": "long", "confidence": 0.91},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-row-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("y la fila?")

            self.assertIn("Modo de respuesta: interpretado.", answer)
            self.assertIn("MSFT", answer)
            self.assertIn("Fila", answer)

    def test_runtime_can_explain_market_type_from_local_first_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0001"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0001",
                "tickers": ["BTC-USD"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.47},
            }
            result = {
                "run_id": "run_0001",
                "manifest": {"request": {"tickers": ["BTC-USD"]}},
            }
            manifest = {"run_id": "run_0001", "request": {"tickers": ["BTC-USD"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="market-type-test")
            answer = runtime.ask("what market type does the symbol used in run_0001 belong to?")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Certainty: confirmed from run artifacts.", answer)
            self.assertIn("BTC-USD belongs to the crypto market", answer)
            self.assertIn("digital asset pair", answer)
            self.assertIn("Source selection: mixed (local-first), but I did not find useful external confirmation so I kept the local classification.", answer)

    def test_runtime_understands_spanish_asset_and_market_type_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0001"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0001",
                "tickers": ["BTC-USD"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.47},
            }
            result = {"run_id": "run_0001", "manifest": {"request": {"tickers": ["BTC-USD"]}}}
            manifest = {"run_id": "run_0001", "request": {"tickers": ["BTC-USD"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="market-type-es-test")
            asset_answer = runtime.ask("dime el activo usado para analisis en run_0001")
            market_answer = runtime.ask("de qué tipo de mercado hace parte el símbolo usado en run_0001?")

            self.assertIn("BTC-USD", asset_answer)
            self.assertIn("Modo de respuesta: interpretado.", asset_answer)
            self.assertIn("BTC-USD pertenece al mercado cripto", market_answer)
            self.assertIn("par de activo digital", market_answer)

    def test_runtime_appends_source_selection_to_current_market_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            cleaned_path = root / "clean_market_data.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-21",
                        "ticker": "MSFT",
                        "open": 100.0,
                        "high": 105.0,
                        "low": 99.0,
                        "close": 104.0,
                        "adj_close": 104.0,
                        "volume": 1000,
                        "volatility_5": 0.02,
                        "target_direction": 1,
                    }
                ]
            ).to_csv(cleaned_path, index=False)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "files": {"cleaned": str(cleaned_path)},
                "rows": {"raw": 1, "cleaned": 1},
                "models": {"final_decision": "long", "confidence": 0.51},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="mixed-metrics-test")
            answer = runtime.ask("what is the current volume of MSFT today?")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Market metrics", answer)
            self.assertIn("MSFT", answer)
            self.assertIn("Source selection: mixed (local-first).", answer)
            self.assertIn("kept the answer grounded in local artifacts", answer.lower())

    def test_runtime_appends_source_selection_to_current_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="mixed-status-test")
            answer = runtime.ask("status today")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Source selection: mixed (local-first).", answer)
            self.assertIn("kept the answer grounded in local artifacts", answer.lower())

    def test_runtime_appends_source_selection_to_current_decision_explanation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "return_1d", "ticker"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "selection": {"strategy": "majority_vote", "reason": "Local disagreement."},
                "comparison": {"decision_path": "local_majority"},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="mixed-decision-test")
            answer = runtime.ask("why did run_0007 decide that way with current market context?")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("The final decision was short", answer)
            self.assertIn("Source selection: mixed (local-first).", answer)
            self.assertIn("kept the answer grounded in local artifacts", answer.lower())

    def test_runtime_appends_source_selection_to_current_model_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 10},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "close", "volume"],
                    "feature_columns": ["open", "close", "volume", "return_1d", "sma_5", "ticker"],
                    "derived_columns": ["return_1d", "sma_5"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.72},
                "motor": {"selected": "majority", "decision": "long"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "modeling": {"selected_model": "majority"},
            }
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="mixed-variables-test")
            answer = runtime.ask("run_0007 what variables did the model use with current market context?")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Modeling used 10 cleaned rows", answer)
            self.assertIn("Source selection: mixed (local-first).", answer)
            self.assertIn("kept the answer grounded in local artifacts", answer.lower())

    def test_runtime_appends_source_selection_to_clean_data_with_current_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            cleaned_path = root / "clean_market_data.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-21",
                        "ticker": "MSFT",
                        "open": 100.0,
                        "high": 105.0,
                        "low": 99.0,
                        "close": 104.0,
                        "adj_close": 104.0,
                        "volume": 1000,
                        "return_1d": 0.01,
                        "range_pct": 0.06,
                        "body_pct": 0.04,
                        "sma_5": 103.0,
                        "sma_10": 102.0,
                        "volatility_5": 0.02,
                        "volume_sma_5": 995.0,
                        "future_close": 105.0,
                        "future_return": 0.0096,
                        "target_direction": 1,
                    }
                ]
            ).to_csv(cleaned_path, index=False)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "files": {"cleaned": str(cleaned_path)},
                "rows": {"raw": 1, "cleaned": 1},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                    "feature_columns": ["open", "high", "low", "close", "adj_close", "volume", "return_1d", "range_pct", "body_pct", "sma_5", "sma_10", "volatility_5", "volume_sma_5", "ticker"],
                    "target_column": "target_direction",
                },
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="mixed-clean-data-test")
            answer = runtime.ask("analyze the clean row for MSFT with current market context")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Clean market data analysis", answer)
            self.assertIn("Source selection: mixed (local-first).", answer)
            self.assertIn("kept the answer grounded in local artifacts", answer.lower())

    def test_runtime_appends_source_selection_to_run_comparison_with_current_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            for run_id, ticker, decision, confidence in (
                ("run_0007", "MSFT", "long", 0.91),
                ("run_0008", "AAPL", "short", 0.66),
            ):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                summary = {
                    "run_id": run_id,
                    "tickers": [ticker],
                    "rows": {"raw": 10, "cleaned": 9 if run_id == "run_0008" else 10},
                    "models": {"final_decision": decision, "confidence": confidence},
                    "motor": {"selected": "majority"},
                }
                result = {
                    "run_id": run_id,
                    "manifest": {"request": {"tickers": [ticker]}},
                    "modeling": {"selected_model": "majority"},
                }
                (run_dir / "summary.json").write_text(json.dumps(summary))
                (run_dir / "result.json").write_text(json.dumps(result))
                (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "request": {"tickers": [ticker]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="mixed-compare-test")
            answer = runtime.ask("compare run_0007 vs run_0008 with current market context")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Comparison between run_0007 and run_0008.", answer)
            self.assertIn("Source selection: mixed (local-first).", answer)
            self.assertIn("kept the answer grounded in local artifacts", answer.lower())

    def test_runtime_integrates_external_backdrop_when_web_facts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "return_1d", "ticker"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "selection": {"strategy": "majority_vote", "reason": "Local disagreement."},
                "comparison": {"decision_path": "local_majority"},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="web-mixed-test")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="MSFT market update",
                        snippet="Microsoft traded in a cautious range with attention on cloud demand and near-term volatility.",
                        url="https://example.com/msft-update",
                        domain="example.com",
                        query="MSFT latest market context",
                        rank=1,
                        trust_score=0.92,
                    ),
                    WebFact(
                        title="MSFT second source",
                        snippet="A second market note.",
                        url="https://news.example.org/msft-second",
                        domain="news.example.org",
                        query="MSFT latest market context",
                        rank=2,
                        trust_score=0.74,
                    ),
                ]
            )

            answer = runtime.ask("why did run_0007 decide that way with current market context?")

            self.assertIn("Certainty: inferred from run artifacts with complementary external context.", answer)
            self.assertIn("Grounding: mixed (local-first).", answer)
            self.assertIn("Integrated external backdrop: Microsoft traded in a cautious range", answer)
            self.assertIn("The run and its artifacts remain the primary source of truth.", answer)
            self.assertIn("Domain: example.com.", answer)
            self.assertIn("Source priority: local artifacts > external context.", answer)
            self.assertIn("Source selection: mixed (local-first).", answer)
            self.assertIn("Source: https://example.com/msft-update.", answer)
            self.assertIn("Additional sources: https://news.example.org/msft-second.", answer)
            self.assertIn("Domains: example.com, news.example.org.", answer)
            self.assertIn("Blend detail: 2 external facts, 2 domains, top trust=0.92.", answer)

    def test_runtime_can_use_real_http_web_retriever_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "return_1d", "ticker"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "selection": {"strategy": "majority_vote", "reason": "Local disagreement."},
                "comparison": {"decision_path": "local_majority"},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            payload = {
                "results": [
                    {
                        "title": "MSFT live context",
                        "snippet": "Cloud demand stayed firm while near-term volatility remained elevated.",
                        "url": "https://example.com/msft-live-context",
                    }
                ]
            }

            with _serve_runtime_search_payload(payload) as (search_url, requests_log):
                with patch.dict(
                    os.environ,
                    {
                        "ASSISTANT_WEB_SEARCH_URL": search_url,
                        "ASSISTANT_WEB_SEARCH_METHOD": "POST",
                    },
                    clear=False,
                ):
                    runtime = AssistantRuntime(artifact_root=str(root), session_id="web-http-e2e-test")
                    answer = runtime.ask("why did run_0007 decide that way with current market context?")

            self.assertGreaterEqual(len(requests_log), 1)
            self.assertEqual(requests_log[0]["method"], "POST")
            self.assertIn("MSFT", str(requests_log[0]["json"].get("q", "")))
            self.assertEqual(requests_log[0]["json"].get("limit"), 2)
            self.assertIn(
                "Certainty: inferred from run artifacts with complementary external context.",
                answer,
            )
            self.assertIn("Grounding: mixed (local-first).", answer)
            self.assertIn(
                "Integrated external backdrop: Cloud demand stayed firm while near-term volatility remained elevated.",
                answer,
            )
            self.assertIn("Domain: example.com.", answer)
            self.assertIn("Source summary: run facts + 1 external check.", answer)
            self.assertIn("Blend detail: 1 external fact, 1 domain, top trust=", answer)
            self.assertIn("Source: https://example.com/msft-live-context.", answer)

    def test_runtime_flags_market_type_conflict_between_local_and_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0001"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0001",
                "tickers": ["BTC-USD"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.47},
            }
            result = {"run_id": "run_0001", "manifest": {"request": {"tickers": ["BTC-USD"]}}}
            manifest = {"run_id": "run_0001", "request": {"tickers": ["BTC-USD"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="market-type-conflict-test")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="BTC stock classification",
                        snippet="Some external note incorrectly refers to BTC-USD as a stock equity listing.",
                        url="https://example.com/btc-stock-note",
                    )
                ]
            )

            answer = runtime.ask("what market type does the symbol used in run_0001 belong to with current context?")

            self.assertIn("BTC-USD belongs to the crypto market", answer)
            self.assertIn("Certainty: local artifacts take priority over conflicting external context.", answer)
            self.assertIn("Grounding: mixed (local-first).", answer)
            self.assertIn("Source conflict: the web suggests equity", answer)
            self.assertIn("Source priority: local artifacts > external context.", answer)

    def test_runtime_flags_decision_conflict_between_local_and_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "return_1d", "ticker"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "selection": {"strategy": "majority_vote", "reason": "Local disagreement."},
                "comparison": {"decision_path": "local_majority"},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="decision-conflict-test")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="MSFT bullish rebound",
                        snippet="Analysts described a bullish rebound with upside momentum and recovery in cloud demand.",
                        url="https://example.com/msft-bullish",
                    )
                ]
            )

            answer = runtime.ask("why did run_0007 decide that way with current market context?")

            self.assertIn("Certainty: local artifacts take priority over conflicting external context.", answer)
            self.assertIn("Grounding: mixed (local-first).", answer)
            self.assertIn("Bias conflict: the web context leans long, but the run decided short.", answer)

    def test_runtime_flags_market_metrics_conflict_between_local_and_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            cleaned_path = root / "clean_market_data.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-20",
                        "ticker": "MSFT",
                        "open": 100.0,
                        "high": 105.0,
                        "low": 99.0,
                        "close": 104.0,
                        "adj_close": 104.0,
                        "volume": 1800,
                        "volatility_5": 0.02,
                        "target_direction": 1,
                    },
                    {
                        "date": "2026-03-21",
                        "ticker": "MSFT",
                        "open": 104.0,
                        "high": 106.0,
                        "low": 103.0,
                        "close": 105.0,
                        "adj_close": 105.0,
                        "volume": 900,
                        "volatility_5": 0.02,
                        "target_direction": 1,
                    },
                ]
            ).to_csv(cleaned_path, index=False)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "files": {"cleaned": str(cleaned_path)},
                "rows": {"raw": 2, "cleaned": 2},
                "models": {"final_decision": "long", "confidence": 0.51},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="metrics-conflict-test")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="MSFT surging volume",
                        snippet="Market desks highlighted surging volume and heavy participation in the latest session.",
                        url="https://example.com/msft-volume",
                    )
                ]
            )

            answer = runtime.ask("what is the current volume of MSFT today?")

            self.assertIn("Certainty: local artifacts take priority over conflicting external context.", answer)
            self.assertIn("Grounding: mixed (local-first).", answer)
            self.assertIn("Metric conflict: the web suggests volume is high, but the local reading is low versus its average.", answer)

    def test_runtime_flags_session_status_conflict_between_local_and_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="status-conflict-test")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="MSFT bullish rebound",
                        snippet="Analysts described a bullish rebound with upside momentum and recovery in cloud demand.",
                        url="https://example.com/msft-bullish",
                    )
                ]
            )

            answer = runtime.ask("status today")

            self.assertIn("Certainty: local artifacts take priority over conflicting external context.", answer)
            self.assertIn("Grounding: mixed (local-first).", answer)
            self.assertIn("Bias conflict: the web context leans long, but the run decided short.", answer)

    def test_runtime_flags_latest_summary_conflict_between_local_and_web(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="latest-conflict-test")
            runtime.web_retriever = _StubWebRetriever(
                [
                    WebFact(
                        title="MSFT bullish rebound",
                        snippet="Analysts described a bullish rebound with upside momentum and recovery in cloud demand.",
                        url="https://example.com/msft-bullish",
                    )
                ]
            )

            answer = runtime.ask("latest today")

            self.assertIn("Certainty: local artifacts take priority over conflicting external context.", answer)
            self.assertIn("Grounding: mixed (local-first).", answer)
            self.assertIn("Source summary: run facts + 1 external check in conflict.", answer)
            self.assertIn("Bias conflict: the web context leans long, but the run decided short.", answer)

    def test_runtime_handles_spanish_whats_happening_today(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="latest-spanish-test")
            answer = runtime.ask("qué pasa hoy")

            self.assertIn("Modo de respuesta: interpretado.", answer)
            self.assertIn("Base factual: mixta (local-first).", answer)
            self.assertIn("Selección de fuentes: mixed (local-first).", answer)
            self.assertIn("run_0007", answer)

    def test_runtime_handles_loose_hows_it_going_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="latest-loose-test")
            answer = runtime.ask("how's it going")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Grounding: local.", answer)
            self.assertIn("run_0007", answer)

    def test_runtime_can_compare_two_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            for run_id, ticker, decision, confidence in (
                ("run_0007", "MSFT", "long", 0.91),
                ("run_0008", "AAPL", "short", 0.66),
            ):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                summary = {
                    "run_id": run_id,
                    "tickers": [ticker],
                    "rows": {"raw": 10, "cleaned": 9 if run_id == "run_0008" else 10},
                    "models": {"final_decision": decision, "confidence": confidence},
                    "motor": {"selected": "majority"},
                }
                result = {
                    "run_id": run_id,
                    "manifest": {"request": {"tickers": [ticker]}},
                    "modeling": {"selected_model": "majority"},
                }
                (run_dir / "summary.json").write_text(json.dumps(summary))
                (run_dir / "result.json").write_text(json.dumps(result))
                (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "request": {"tickers": [ticker]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="compare-test")
            answer = runtime.ask("compare run_0007 vs run_0008")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Comparison between run_0007 and run_0008.", answer)
            self.assertIn("run_0007: ticker=MSFT, decision=long", answer)
            self.assertIn("run_0008: ticker=AAPL, decision=short", answer)

    def test_runtime_routes_elliptical_comparison_follow_up_and_keeps_both_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            for run_id, ticker, decision, confidence in (
                ("run_0007", "MSFT", "long", 0.91),
                ("run_0008", "AAPL", "short", 0.66),
            ):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                summary = {
                    "run_id": run_id,
                    "tickers": [ticker],
                    "rows": {"raw": 10, "cleaned": 9 if run_id == "run_0008" else 10},
                    "models": {"final_decision": decision, "confidence": confidence},
                    "motor": {"selected": "majority"},
                }
                result = {
                    "run_id": run_id,
                    "manifest": {"request": {"tickers": [ticker]}},
                    "modeling": {"selected_model": "majority"},
                }
                (run_dir / "summary.json").write_text(json.dumps(summary))
                (run_dir / "result.json").write_text(json.dumps(result))
                (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "request": {"tickers": [ticker]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="compare-follow-up-test")
            runtime.ask("compare run_0007 vs run_0008")
            answer = runtime.ask("y esa comparación?")

            self.assertIn("run_0007", answer)
            self.assertIn("run_0008", answer)
            self.assertIn("Comparison between run_0007 and run_0008.", answer)

    def test_runtime_routes_elliptical_run_follow_up_to_latest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}, "final_confidence": 0.48}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-run-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("esa corrida?")

            self.assertIn("Modo de respuesta: interpretado.", answer)
            self.assertIn("run_0007", answer)
            self.assertIn("En lenguaje simple, extraje 10 filas", answer)

    def test_runtime_routes_elliptical_result_follow_up_to_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0001"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0001",
                "tickers": ["BTC-USD"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                },
                "models": {"final_decision": "short", "confidence": 0.47},
            }
            result = {"run_id": "run_0001", "manifest": {"request": {"tickers": ["BTC-USD"]}}, "extraction": {"rows": 10}}
            manifest = {"run_id": "run_0001", "request": {"tickers": ["BTC-USD"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-result-test")
            runtime.ask("podrias explicarme el resultado de la extraccion run_0001?")
            answer = runtime.ask("ese resultado?")

            self.assertIn("run_0001", answer)
            self.assertIn("date, ticker, open, high, low, close, adj_close, volume", answer)

    def test_runtime_routes_elliptical_cleaning_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "volume"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-cleaning-test")
            runtime.ask("run_0007 limpieza")
            answer = runtime.ask("esa limpieza?")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("run_0007", answer)
            self.assertIn("I cleaned", answer)

    def test_runtime_routes_modeling_follow_up_phrase_to_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "volume"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "modeling": {"selected_model": "majority"},
            }
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-modeling-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("eso de modelado?")

            self.assertIn("Modo de respuesta: interpretado.", answer)
            self.assertIn("run_0007", answer)
            self.assertIn("Los modelos predicen", answer)

    def test_runtime_routes_generic_process_follow_up_to_last_stage_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "volume"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-process-test")
            runtime.ask("run_0007 limpieza")
            answer = runtime.ask("ese proceso?")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("run_0007", answer)
            self.assertIn("I cleaned", answer)

    def test_runtime_routes_extraction_nominal_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0001"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0001",
                "tickers": ["BTC-USD"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                },
                "models": {"final_decision": "short", "confidence": 0.47},
            }
            result = {"run_id": "run_0001", "manifest": {"request": {"tickers": ["BTC-USD"]}}, "extraction": {"rows": 10}}
            manifest = {"run_id": "run_0001", "request": {"tickers": ["BTC-USD"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-extraction-nominal-test")
            runtime.ask("podrias explicarme el resultado de la extraccion run_0001?")
            answer = runtime.ask("lo de extracción?")

            self.assertIn("run_0001", answer)
            self.assertIn("date, ticker, open, high, low, close, adj_close, volume", answer)

    def test_runtime_routes_generic_part_follow_up_to_last_modeling_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "volume"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "modeling": {"selected_model": "majority"},
            }
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="elliptical-part-modeling-test")
            runtime.ask("run_0007 modelado")
            answer = runtime.ask("esa parte?")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("run_0007", answer)
            self.assertIn("The models predict", answer)

    def test_runtime_routes_compound_cleaning_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "volume"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="compound-cleaning-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("la parte de limpieza?")

            self.assertIn("run_0007", answer)
            self.assertRegex(answer, r"(Limpié|I cleaned|Se limpiaron)")

    def test_runtime_routes_compound_modeling_step_phrase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "volume"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "modeling": {"selected_model": "majority"},
            }
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="compound-modeling-test")
            runtime.ask("qué pasa hoy")
            answer = runtime.ask("el paso del modelado?")

            self.assertIn("run_0007", answer)
            self.assertRegex(answer, r"(Los modelos predicen|The models predict)")

    def test_runtime_routes_compound_process_phrase_to_last_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "volume"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "modeling": {"selected_model": "majority"},
            }
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="compound-process-test")
            runtime.ask("run_0007 modelado")
            answer = runtime.ask("esa parte del proceso?")

            self.assertIn("run_0007", answer)
            self.assertRegex(answer, r"(Los modelos predicen|The models predict)")

    def test_runtime_uses_last_stage_brief_stage_for_generic_stage_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "volume"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
            }
            result = {"run_id": "run_0007", "manifest": {"request": {"tickers": ["MSFT"]}}}
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            state = load_state(str(root), "stage-brief-follow-up-cleaning")
            state.last_run_id = "run_0007"
            state.last_intent = "show_stage_brief"
            state.last_route = {"intent": "show_stage_brief", "run_id": "run_0007", "stage": "cleaning"}
            save_state(str(root), state)

            runtime = AssistantRuntime(artifact_root=str(root), session_id="stage-brief-follow-up-cleaning")
            answer = runtime.ask("esa etapa?")

            self.assertIn("run_0007", answer)
            self.assertRegex(answer, r"(Limpié|I cleaned|Se limpiaron)")

    def test_runtime_uses_last_stage_brief_stage_for_generic_process_follow_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)
            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "data": {
                    "feature_columns": ["open", "close", "volume"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.91},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "modeling": {"selected_model": "majority"},
            }
            manifest = {"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps(manifest))

            state = load_state(str(root), "stage-brief-follow-up-modeling")
            state.last_run_id = "run_0007"
            state.last_intent = "show_stage_brief"
            state.last_route = {"intent": "show_stage_brief", "run_id": "run_0007", "stage": "modeling"}
            save_state(str(root), state)

            runtime = AssistantRuntime(artifact_root=str(root), session_id="stage-brief-follow-up-modeling")
            answer = runtime.ask("ese proceso?")

            self.assertIn("run_0007", answer)
            self.assertRegex(answer, r"(Los modelos predicen|The models predict)")

    def test_runtime_can_compare_two_runs_by_cleaning_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            for run_id, ticker, raw_rows, cleaned_rows in (
                ("run_0007", "MSFT", 10, 10),
                ("run_0008", "AAPL", 10, 8),
            ):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                summary = {
                    "run_id": run_id,
                    "tickers": [ticker],
                    "rows": {"raw": raw_rows, "cleaned": cleaned_rows},
                    "models": {"final_decision": "long", "confidence": 0.50},
                    "motor": {"selected": "majority"},
                }
                result = {
                    "run_id": run_id,
                    "manifest": {"request": {"tickers": [ticker]}},
                    "modeling": {"selected_model": "majority"},
                }
                (run_dir / "summary.json").write_text(json.dumps(summary))
                (run_dir / "result.json").write_text(json.dumps(result))
                (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "request": {"tickers": [ticker]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="compare-stage-test")
            answer = runtime.ask("compare cleaning run_0007 vs run_0008")

            self.assertIn("Response mode: interpreted.", answer)
            self.assertIn("Cleaning comparison between run_0007 and run_0008.", answer)
            self.assertIn("run_0007: rows=10->10.", answer)
            self.assertIn("run_0008: rows=10->8.", answer)

    def test_runtime_adds_guided_hypothesis_in_exploratory_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 10},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "close"],
                    "feature_columns": ["open", "close", "return_1d", "sma_5", "ticker"],
                    "derived_columns": ["return_1d", "sma_5"],
                    "target_column": "target_direction",
                },
                "models": {"final_decision": "long", "confidence": 0.72, "disagreement": True},
                "motor": {"selected": "majority", "decision": "long"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "modeling": {"selected_model": "majority"},
                "final_confidence": 0.72,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="explore-test")
            answer = runtime.ask("what variables did the model use maybe?")

            self.assertIn("Response mode: exploratory.", answer)
            self.assertIn("Certainty: hypothesis guided by the available context.", answer)
            self.assertIn("Guided hypothesis:", answer)

    def test_runtime_adds_guided_hypothesis_for_exploratory_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_0007"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_0007",
                "tickers": ["MSFT"],
                "rows": {"raw": 10, "cleaned": 9},
                "models": {"final_decision": "short", "confidence": 0.48, "disagreement": True},
                "motor": {"selected": "majority", "decision": "short"},
            }
            result = {
                "run_id": "run_0007",
                "manifest": {"request": {"tickers": ["MSFT"]}},
                "final_confidence": 0.48,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_0007", "request": {"tickers": ["MSFT"]}}))

            runtime = AssistantRuntime(artifact_root=str(root), session_id="explore-summary-test")
            answer = runtime.ask("status maybe")

            self.assertIn("Response mode: exploratory.", answer)
            self.assertIn("Guided hypothesis:", answer)

    def test_runtime_can_open_mode_and_agent_guides_without_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="guide-test")

            status = runtime.ask("status")
            web_status = runtime.ask("web status")
            mode_hub = runtime.ask("modes")
            mode_detail = runtime.ask("local_only")
            compare_mode = runtime.ask("compare-binance")
            combined_mode = runtime.ask("compare-binance + --groq-brain")
            agent_guide = runtime.ask("agents")
            help_text = runtime.ask("help")
            symbols_text = runtime.ask("symbols")
            extraction_card = runtime.ask("ask extraction")
            clean_data_empty = runtime.ask("clean data")
            metrics_empty = runtime.ask("volume")
            continue_empty = runtime.ask("continue")

            self.assertIn("Session status", status)
            self.assertIn("Mode:", status)
            self.assertIn("Groq:", status)
            self.assertIn("Web status", web_status)
            self.assertIn("Enabled: yes.", web_status)
            self.assertIn("Runtime ready: yes.", web_status)
            self.assertIn("Provider: duckduckgo.", web_status)
            self.assertIn("The retriever is ready to complement exploratory and mixed answers.", web_status)
            self.assertIn("Mode hub", mode_hub)
            self.assertIn("Type a mode label", mode_hub)
            self.assertIn("local_only", mode_hub)
            self.assertIn("Visible response styles: interpreted | exploratory.", mode_hub)
            self.assertIn("Mode detail", mode_detail)
            self.assertIn("Summary:", mode_detail)
            self.assertIn("Activation:", mode_detail)
            self.assertIn("When to use:", mode_detail)
            self.assertIn("Flow impact:", mode_detail)
            self.assertIn("Default deterministic local mode", mode_detail)
            self.assertIn("Mode detail", compare_mode)
            self.assertIn("compare-binance", compare_mode)
            self.assertIn("Summary:", compare_mode)
            self.assertIn("Activation:", compare_mode)
            self.assertIn("Flow impact:", compare_mode)
            self.assertIn("Agent flow:", compare_mode)
            self.assertIn("Extraction compares yfinance with Binance", compare_mode)
            self.assertIn("Compares yfinance with Binance", compare_mode)
            self.assertIn("Mode detail", combined_mode)
            self.assertIn("compare-binance + --groq-brain", combined_mode)
            self.assertIn("Summary:", combined_mode)
            self.assertIn("Flow impact:", combined_mode)
            self.assertIn("Agent flow:", combined_mode)
            self.assertIn("experimental brain", combined_mode)
            self.assertIn("Flow:", mode_hub)
            self.assertIn("Cases to run:", mode_hub)
            self.assertIn("--groq-brain AAPL 2024-01-01 2025-01-01", mode_hub)
            self.assertIn("Agent menu", agent_guide)
            self.assertIn("Open an agent by name to see its detail", agent_guide)
            self.assertIn("Flow:", agent_guide)
            self.assertIn("Help hub", help_text)
            self.assertIn("Ask naturally with a ticker", help_text)
            self.assertIn("Agents:", help_text)
            self.assertIn("Clean data:", help_text)
            self.assertIn("Row: one cleaned record per date and ticker.", help_text)
            self.assertIn("Analysis: explains that row with base fields and derived clean signals.", help_text)
            self.assertIn("Modes:", help_text)
            self.assertIn("Response styles: interpreted | exploratory.", help_text)
            self.assertIn("Interpreted keeps the answer grounded in local artifacts.", help_text)
            self.assertIn("Connectivity: ask web status", help_text)
            self.assertIn("Quick keys:", help_text)
            self.assertIn("Current context: run=n/a | symbol=n/a | stage=agents.", help_text)
            self.assertIn("Examples: extraction | cleaning | modeling", help_text)
            self.assertIn("Suggested symbols", symbols_text)
            self.assertIn("How to use them", symbols_text)
            self.assertIn("Clean market data overview", clean_data_empty)
            self.assertIn("Status:", clean_data_empty)
            self.assertIn("Action:", clean_data_empty)
            self.assertIn("Prompt:", clean_data_empty)
            self.assertIn("Market metrics", metrics_empty)
            self.assertIn("Status:", metrics_empty)
            self.assertIn("Action:", metrics_empty)
            self.assertIn("Continue", continue_empty)
            self.assertIn("No unfinished task is pending yet", continue_empty)
            self.assertIn("Action:", continue_empty)
            self.assertIn("__agent_card__:extraction:en", extraction_card)
            self.assertIn("Natural prompts", extraction_card)
            self.assertIn("Useful symbols:", extraction_card)

    def test_agent_guide_respects_sticky_spanish_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime = AssistantRuntime(artifact_root=tmpdir, session_id="guide-es")

            runtime.ask("switch to Spanish")
            status = runtime.ask("status")
            web_status = runtime.ask("estado del retriever")
            agent_guide = runtime.ask("agents")
            mode_hub = runtime.ask("modes")
            mode_detail = runtime.ask("local_only")
            compare_mode = runtime.ask("compare-binance")
            combined_mode = runtime.ask("compare-binance + --groq-brain")
            help_text = runtime.ask("help")
            symbols_text = runtime.ask("símbolos")
            extraction_card = runtime.ask("ask extraction")
            clean_data_empty = runtime.ask("datos limpios")
            metrics_empty = runtime.ask("volumen")
            continue_empty = runtime.ask("continuar")

            self.assertIn("Estado de sesión", status)
            self.assertIn("Modo:", status)
            self.assertIn("Groq:", status)
            self.assertIn("Estado web", web_status)
            self.assertIn("Activo: sí.", web_status)
            self.assertIn("Listo en runtime: sí.", web_status)
            self.assertIn("Proveedor: duckduckgo.", web_status)
            self.assertIn("El retriever está listo para complementar respuestas exploratorias y mixtas.", web_status)
            self.assertIn("Menú de agentes", agent_guide)
            self.assertIn("Abre un agente por nombre", agent_guide)
            self.assertIn("Flujo:", agent_guide)
            self.assertIn("Centro de modos", mode_hub)
            self.assertIn("Escribe un modo para ver su detalle", mode_hub)
            self.assertIn("Estilos de respuesta visibles: interpretado | exploratorio.", mode_hub)
            self.assertIn("Casos para ejecutar:", mode_hub)
            self.assertIn("--groq-brain AAPL 2024-01-01 2025-01-01", mode_hub)
            self.assertIn("Detalle de modo", mode_detail)
            self.assertIn("Resumen:", mode_detail)
            self.assertIn("Activación:", mode_detail)
            self.assertIn("Cuándo usarlo:", mode_detail)
            self.assertIn("Impacto en el flujo:", mode_detail)
            self.assertIn("Detalle de modo", compare_mode)
            self.assertIn("Compara yfinance con Binance", compare_mode)
            self.assertIn("Resumen:", compare_mode)
            self.assertIn("Activación:", compare_mode)
            self.assertIn("Impacto en el flujo:", compare_mode)
            self.assertIn("Flujo de agentes:", compare_mode)
            self.assertIn("Extracción compara yfinance con Binance", compare_mode)
            self.assertIn("Detalle de modo", combined_mode)
            self.assertIn("compare-binance + --groq-brain", combined_mode)
            self.assertIn("Resumen:", combined_mode)
            self.assertIn("Impacto en el flujo:", combined_mode)
            self.assertIn("Flujo de agentes:", combined_mode)
            self.assertIn("cerebro experimental", combined_mode)
            self.assertIn("__agent_card__:extraction:es", extraction_card)
            self.assertIn("Centro de ayuda", help_text)
            self.assertIn("Pregunta con naturalidad", help_text)
            self.assertIn("Agentes:", help_text)
            self.assertIn("Datos limpios:", help_text)
            self.assertIn("Fila: un registro limpio por fecha y ticker.", help_text)
            self.assertIn("Análisis: explica esa fila con señales limpias derivadas y campos base.", help_text)
            self.assertIn("Modos:", help_text)
            self.assertIn("Estilos de respuesta: interpretado | exploratorio.", help_text)
            self.assertIn("Interpretado se mantiene grounded en artifacts locales.", help_text)
            self.assertIn("Conectividad: pregunta estado web", help_text)
            self.assertIn("Contexto actual: run=n/a | símbolo=n/a | etapa=modos.", help_text)
            self.assertIn("Ejemplos: compare-binance ETH-USD 2024-01-01 2025-01-01 | --groq-brain | local_only", help_text)
            self.assertIn("Símbolos sugeridos", symbols_text)
            self.assertIn("Cómo usarlo", symbols_text)
            self.assertIn("Resumen de clean_market_data", clean_data_empty)
            self.assertIn("Estado:", clean_data_empty)
            self.assertIn("Acción:", clean_data_empty)
            self.assertIn("Pregunta natural:", clean_data_empty)
            self.assertIn("Métricas de mercado", metrics_empty)
            self.assertIn("Estado:", metrics_empty)
            self.assertIn("Acción:", metrics_empty)
            self.assertIn("Continuar", continue_empty)
            self.assertIn("No hay una tarea pendiente todavía", continue_empty)
            self.assertIn("Acción:", continue_empty)

    def test_runtime_reports_ready_web_status_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {
                    "ASSISTANT_WEB_PROVIDER": "tavily",
                    "ASSISTANT_WEB_SEARCH_URL": "https://search.test",
                    "ASSISTANT_WEB_SEARCH_API_KEY": "secret-key",
                },
                clear=True,
            ):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="web-status-ready")
                text = runtime.ask("web status")

        self.assertIn("Web status", text)
        self.assertIn("Enabled: yes.", text)
        self.assertIn("Config valid: yes.", text)
        self.assertIn("Runtime ready: yes.", text)
        self.assertIn("Provider: tavily.", text)
        self.assertIn("Preset known: yes; host consistent: yes.", text)

    def test_runtime_can_render_provider_setup_snippet_from_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="web-status-snippet")
                text = runtime.ask("what env do I need for tavily?")

        self.assertIn("Web status", text)
        self.assertIn("Setup snippet:", text)
        self.assertIn("Probe command:", text)
        self.assertIn('ASSISTANT_WEB_PROVIDER="tavily"', text)
        self.assertIn('ASSISTANT_WEB_SEARCH_URL="https://api.tavily.com/search"', text)
        self.assertIn('ASSISTANT_WEB_SEARCH_API_KEY="..."', text)
        self.assertIn(".venv/bin/python scripts/assistant_web_probe.py --provider tavily --search-url https://api.tavily.com/search", text)

    def test_runtime_can_render_provider_catalog_from_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="web-provider-catalog")
                text = runtime.ask("what web providers are supported?")

        self.assertIn("Web status", text)
        self.assertIn("Preset catalog:", text)
        self.assertIn("searxng: method=GET", text)
        self.assertIn("endpoint=https://YOUR-SEARXNG-ENDPOINT/search", text)
        self.assertIn("serper: method=POST", text)
        self.assertIn("endpoint=https://google.serper.dev/search", text)
        self.assertIn("tavily: method=POST", text)
        self.assertIn("endpoint=https://api.tavily.com/search", text)
        self.assertIn("searchapi: method=GET", text)

    def test_runtime_can_render_assistant_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="assistant-scorecard")
                text = runtime.ask("assistant scorecard")

        self.assertIn("Assistant scorecard", text)
        self.assertIn("Local-first: runtime_ready=100%, implementation=100%.", text)
        self.assertIn("Hybrid: runtime_ready=100%, implementation=100%.", text)
        self.assertIn("The hybrid runtime is already fully ready;", text)
        self.assertIn("Runtime web: configured=yes, valid=yes, ready=yes.", text)
        self.assertNotIn("Blocking issues:", text)
        self.assertNotIn("Suggested preset:", text)

    def test_runtime_can_render_assistant_scorecard_for_requested_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="assistant-scorecard-serper")
                text = runtime.ask("assistant scorecard for serper")

        self.assertIn("Assistant scorecard", text)
        self.assertIn("The hybrid runtime is already fully ready;", text)

    def test_runtime_can_render_assistant_scorecard_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="assistant-scorecard-layers")
                text = runtime.ask("assistant scorecard layers")

        self.assertIn("Assistant scorecard", text)
        self.assertIn("Layers:", text)
        self.assertIn("Semantic Layer 100/100", text)
        self.assertIn("Conversational Layer 100/100", text)
        self.assertIn("Search-Augmented Interpretation Layer 100/100", text)
        self.assertIn("Web Retrieval Layer 100/100", text)
        self.assertIn("Web Fact Extraction 100/100", text)
        self.assertIn("Mixed Grounding Engine 100/100", text)

    def test_runtime_can_render_assistant_scorecard_web_impact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="assistant-scorecard-web-impact")
                text = runtime.ask("what changes if I enable web?")

        self.assertIn("Assistant scorecard", text)
        self.assertIn("The hybrid runtime is already fully ready;", text)
        self.assertIn("layers are at 100%", text)

    def test_runtime_can_handoff_from_scorecard_to_web_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="assistant-scorecard-followup")
                runtime.ask("assistant scorecard for serper")
                text = runtime.ask("y el setup?")

        self.assertIn("Web status", text)
        self.assertIn("Provider: duckduckgo.", text)
        self.assertIn("Setup snippet:", text)
        self.assertIn('ASSISTANT_WEB_PROVIDER="serper"', text)
        self.assertIn('ASSISTANT_WEB_SEARCH_URL="https://google.serper.dev/search"', text)

    def test_runtime_can_handoff_from_scorecard_to_provider_setup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="assistant-scorecard-provider-followup")
                runtime.ask("assistant scorecard for serper")
                text = runtime.ask("use serper setup")

        self.assertIn("Web status", text)
        self.assertIn("Setup snippet:", text)
        self.assertIn('ASSISTANT_WEB_PROVIDER="serper"', text)
        self.assertIn('ASSISTANT_WEB_SEARCH_URL="https://google.serper.dev/search"', text)

    def test_runtime_can_route_direct_provider_activation_to_web_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="direct-provider-activation")
                text = runtime.ask("activate serper")

        self.assertIn("Web status", text)
        self.assertIn("Setup snippet:", text)
        self.assertIn('ASSISTANT_WEB_PROVIDER="serper"', text)
        self.assertIn('ASSISTANT_WEB_SEARCH_URL="https://google.serper.dev/search"', text)

    def test_runtime_can_handoff_from_scorecard_to_provider_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="assistant-scorecard-provider-probe")
                runtime.ask("assistant scorecard for serper")
                text = runtime.ask("y el probe?")

        self.assertIn("Web status", text)
        self.assertIn("Probe command:", text)
        self.assertIn("--provider serper", text)
        self.assertIn("--search-url https://google.serper.dev/search", text)

    def test_runtime_can_handoff_from_scorecard_to_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="assistant-scorecard-layers-followup")
                runtime.ask("assistant scorecard")
                text = runtime.ask("y las capas?")

        self.assertIn("Assistant scorecard", text)
        self.assertIn("Layers:", text)
        self.assertIn("Semantic Layer 100/100", text)
        self.assertIn("Conversational Layer 100/100", text)

    def test_runtime_can_handoff_from_scorecard_to_web_impact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                runtime = AssistantRuntime(artifact_root=tmpdir, session_id="assistant-scorecard-web-impact-followup")
                runtime.ask("assistant scorecard")
                text = runtime.ask("y si configuro web?")

        self.assertIn("Assistant scorecard", text)
        self.assertIn("The hybrid runtime is already fully ready;", text)

    def test_clean_data_view_can_list_symbols_show_schema_and_pick_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cleaned_path = root / "clean_market_data.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-20",
                        "ticker": "MSFT",
                        "open": 100.0,
                        "high": 105.0,
                        "low": 99.0,
                        "close": 104.0,
                        "adj_close": 104.0,
                        "volume": 1000,
                        "return_1d": 0.01,
                        "range_pct": 0.06,
                        "body_pct": 0.04,
                        "sma_5": 103.0,
                        "sma_10": 102.0,
                        "volatility_5": 0.02,
                        "volume_sma_5": 995.0,
                        "future_close": 105.0,
                        "future_return": 0.0096,
                        "target_direction": 1,
                    },
                    {
                        "date": "2026-03-21",
                        "ticker": "AAPL",
                        "open": 200.0,
                        "high": 203.0,
                        "low": 198.0,
                        "close": 202.0,
                        "adj_close": 202.0,
                        "volume": 1500,
                        "return_1d": 0.005,
                        "range_pct": 0.025,
                        "body_pct": 0.01,
                        "sma_5": 201.0,
                        "sma_10": 200.5,
                        "volatility_5": 0.01,
                        "volume_sma_5": 1490.0,
                        "future_close": 203.0,
                        "future_return": 0.00495,
                        "target_direction": 1,
                    },
                    {
                        "date": "2026-03-22",
                        "ticker": "MSFT",
                        "open": 106.0,
                        "high": 108.0,
                        "low": 105.0,
                        "close": 107.0,
                        "adj_close": 107.0,
                        "volume": 1100,
                        "return_1d": 0.02,
                        "range_pct": 0.028,
                        "body_pct": 0.0094,
                        "sma_5": 104.0,
                        "sma_10": 103.0,
                        "volatility_5": 0.03,
                        "volume_sma_5": 1050.0,
                        "future_close": 108.0,
                        "future_return": 0.0093,
                        "target_direction": 1,
                    },
                ]
            ).to_csv(cleaned_path, index=False)

            summary = {
                "run_id": "run_2001",
                "tickers": ["MSFT", "AAPL"],
                "date_range": {"start": "2026-03-20", "end": "2026-03-22"},
                "files": {"cleaned": str(cleaned_path)},
                "rows": {"raw": 3, "cleaned": 3},
                "data": {
                    "raw_columns": ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
                    "feature_columns": [
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
                        "ticker",
                    ],
                    "target_column": "target_direction",
                },
            }
            result = {
                "manifest": {
                    "request": {
                        "tickers": ["MSFT", "AAPL"],
                    }
                }
            }

            symbols_view = _format_clean_data_view(summary, result, "what symbols are in the cleaned data?", "en", ["MSFT", "AAPL"])
            metrics_view = _format_clean_data_view(summary, result, "what metrics are in the cleaned data?", "en", ["MSFT", "AAPL"])
            metrics_view_es = _format_clean_data_view(summary, result, "qué métricas hay en los datos limpios?", "es", ["MSFT", "AAPL"])
            schema_view = _format_clean_data_view(summary, result, "what is the clean data schema?", "en", ["MSFT", "AAPL"])
            bare_row_view = _format_clean_data_view(summary, result, "row", "en", ["MSFT"])
            bare_row_view_es = _format_clean_data_view(summary, result, "fila", "es", ["MSFT"])
            analysis_key_view = _format_clean_data_view(summary, result, "análisis de fila", "es", ["MSFT"])
            row_view = _format_clean_data_view(summary, result, "show the clean row for MSFT on 2026-03-22 open high", "en", ["MSFT"])

            self.assertIn("Symbols present", symbols_view)
            self.assertIn("MSFT", symbols_view)
            self.assertIn("AAPL", symbols_view)
            self.assertIn("Rows per symbol", symbols_view)

            self.assertIn("Clean market data metrics", metrics_view)
            self.assertIn("Base fields", metrics_view)
            self.assertIn("Derived metrics", metrics_view)
            self.assertIn("volume_sma_5", metrics_view)
            self.assertIn("target_direction", metrics_view)
            self.assertIn("orchestrator", metrics_view)
            self.assertIn("Métricas de clean_market_data", metrics_view_es)
            self.assertIn("Métricas derivadas", metrics_view_es)

            self.assertIn("Clean market data schema", schema_view)
            self.assertIn("Raw columns", schema_view)
            self.assertIn("Clean columns", schema_view)
            self.assertIn("target_direction", schema_view)
            self.assertIn("Each cleaned row is one unique ticker/date observation", schema_view)

            self.assertIn("Selected row", bare_row_view)
            self.assertIn("MSFT", bare_row_view)
            self.assertIn("Clean row: one unique record per date and ticker.", row_view)
            self.assertIn("Fila seleccionada", bare_row_view_es)
            self.assertIn("MSFT", bare_row_view_es)
            self.assertIn("Fila limpia: un registro único por fecha y ticker.", bare_row_view_es)
            self.assertIn("Análisis de clean_market_data", analysis_key_view)
            self.assertIn("Análisis de fila", analysis_key_view)
            self.assertIn("Fila seleccionada", analysis_key_view)
            self.assertIn("2026-03-22", analysis_key_view)

            self.assertIn("Selected row", row_view)
            self.assertIn("MSFT", row_view)
            self.assertIn("2026-03-22", row_view)
            self.assertIn("open=106.0000", row_view)
            self.assertIn("high=108.0000", row_view)

            missing_path = root / "clean_market_data_single.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2026-03-21",
                        "ticker": "AAPL",
                        "open": 200.0,
                        "high": 203.0,
                        "low": 198.0,
                        "close": 202.0,
                        "adj_close": 202.0,
                        "volume": 1500,
                        "return_1d": 0.005,
                        "range_pct": 0.025,
                        "body_pct": 0.01,
                        "sma_5": 201.0,
                        "sma_10": 200.5,
                        "volatility_5": 0.01,
                        "volume_sma_5": 1490.0,
                        "future_close": 203.0,
                        "future_return": 0.00495,
                        "target_direction": 1,
                    }
                ]
            ).to_csv(missing_path, index=False)
            missing_summary = dict(summary)
            missing_summary["files"] = {"cleaned": str(missing_path)}
            missing_summary["tickers"] = ["AAPL"]
            missing_result = {
                "manifest": {
                    "request": {
                        "tickers": ["AAPL"],
                    }
                }
            }
            missing_view = _format_clean_data_view(
                missing_summary,
                missing_result,
                "analyze the clean row for MSFT on 2026-03-21",
                "en",
                ["MSFT"],
            )
            self.assertIn("No exact clean row matched", missing_view)
            self.assertIn("Available tickers: AAPL", missing_view)
            self.assertNotIn("Selected row:", missing_view)

    def test_session_context_helpers_use_latest_symbol_and_stage(self) -> None:
        summary = {
            "run_id": "run_4000",
            "tickers": ["BTC-USD"],
            "models": {"final_decision": "short"},
        }
        state = AssistantState(
            session_id="context-test",
            current_asset="BTC-USD",
            last_run_id="run_4000",
            notes=["show_clean_data:clean_data:run_4000"],
        )

        context_line = _format_session_context_line(summary, state, "es")
        prompts = _contextual_help_prompts(summary, state, "es")

        self.assertIn("Contexto actual", context_line)
        self.assertIn("run=run_4000", context_line)
        self.assertIn("símbolo=BTC-USD", context_line)
        self.assertIn("etapa=datos limpios", context_line)
        self.assertTrue(any("qué métricas hay en los datos limpios de BTC-USD?" in prompt for prompt in prompts))
        self.assertTrue(any("muestra la fila limpia de BTC-USD" in prompt for prompt in prompts))
        self.assertEqual(_extract_clean_data_mode("row"), "row")
        self.assertEqual(_extract_clean_data_mode("fila"), "row")
        self.assertEqual(_extract_clean_data_mode("análisis de fila"), "analysis")

    def test_help_uses_contextual_examples_when_last_stage_is_clean_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_4000"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_4000",
                "tickers": ["BTC-USD"],
                "run_mode": "local_only",
                "models": {"final_decision": "short", "confidence": 0.45},
            }
            result = {"run_id": "run_4000", "manifest": {"request": {"tickers": ["BTC-USD"]}}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_4000", "request": {"tickers": ["BTC-USD"]}}))

            state = AssistantState(
                session_id="help-context",
                last_run_id="run_4000",
                current_asset="BTC-USD",
                preferred_language="es",
                notes=["show_clean_data:clean_data:run_4000"],
            )
            save_state(str(root), state)

            runtime = AssistantRuntime(artifact_root=str(root), session_id="help-context")
            text = runtime.ask("help")

            self.assertIn("Contexto actual", text)
            self.assertIn("run=run_4000", text)
            self.assertIn("símbolo=BTC-USD", text)
            self.assertIn("etapa=datos limpios", text)
            self.assertIn("Ejemplos:", text)
            self.assertIn("qué métricas hay en los datos limpios de BTC-USD?", text)
            self.assertIn("qué dice esta fila limpia de BTC-USD?", text)

    def test_sticky_spanish_session_overrides_english_prompt_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            run_dir = root / "runs" / "run_3000"
            run_dir.mkdir(parents=True)

            summary = {
                "run_id": "run_3000",
                "tickers": ["MSFT"],
                "date_range": {"start": "2026-03-01", "end": "2026-03-21"},
                "rows": {"raw": 1, "cleaned": 1},
                "data": {"raw_columns": ["date", "ticker", "open"], "feature_columns": ["open"], "target_column": "target_direction"},
                "models": {"final_decision": "long", "confidence": 0.9, "selected": "majority"},
                "motor": {"requested": "auto", "selected": "majority", "decision": "long"},
            }
            result = {"manifest": {"request": {"tickers": ["MSFT"]}}}
            (run_dir / "summary.json").write_text(json.dumps(summary))
            (run_dir / "result.json").write_text(json.dumps(result))
            (run_dir / "manifest.json").write_text(json.dumps({"run_id": "run_3000", "request": {"tickers": ["MSFT"]}}))

            state = load_state(str(root), "session-es")
            state.preferred_language = "es"
            state.last_run_id = "run_3000"
            save_state(str(root), state)

            runtime = AssistantRuntime(artifact_root=str(root), session_id="session-es")
            answer = runtime.ask("what symbol was used?")

            self.assertIn("Símbolo usado", answer)
            self.assertIn("MSFT", answer)

    def test_failed_extraction_returns_a_panel_and_keeps_the_latest_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            class FakeOrchestrator:
                def __init__(self, artifact_root: str) -> None:
                    self.artifact_root = Path(artifact_root)

                def run(self, request):  # type: ignore[no-untyped-def]
                    run_id = "run_0001"
                    run_dir = self.artifact_root / "runs" / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    manifest = {
                        "run_id": run_id,
                        "request": request.model_dump(),
                        "status": "failed",
                    }
                    error = {
                        "stage": "extraction",
                        "message": "No market data could be extracted. Failures: [{'ticker': 'AAPL', 'reason': 'no_rows_returned'}]",
                        "details": {
                            "run_id": run_id,
                            "source": "yfinance",
                            "tickers": ["AAPL"],
                            "failures": [{"ticker": "AAPL", "reason": "no_rows_returned"}],
                        },
                    }
                    (run_dir / "manifest.json").write_text(json.dumps(manifest, default=str))
                    (run_dir / "error.json").write_text(json.dumps(error, default=str))
                    raise ValueError(error["message"])

            runtime = AssistantRuntime(artifact_root=str(root), session_id="failure-test")
            runtime._orchestrator = FakeOrchestrator(str(root))

            answer = runtime.ask("AAPL")
            latest = runtime.ask("latest")
            status = runtime.ask("status")
            help_text = runtime.ask("help")
            continue_answer = runtime.ask("continue")

            self.assertIn("Extraction error", answer)
            self.assertIn("Status:", answer)
            self.assertIn("Stage:", answer)
            self.assertIn("Source:", answer)
            self.assertIn("Failures:", answer)
            self.assertIn("Action:", answer)
            self.assertIn("Prompt:", answer)
            self.assertNotIn("Traceback", answer)
            self.assertIn("Extraction error", latest)
            self.assertIn("Extraction error", status)
            self.assertIn("Latest detected error", help_text)
            self.assertIn("Extraction error", help_text)
            self.assertIn("Extraction error", continue_answer)
            self.assertEqual(runtime.get_state().last_run_id, "run_0001")

    def test_failed_cleaning_and_modeling_return_stage_specific_panels(self) -> None:
        scenarios = [
            (
                "cleaning",
                "Cleaning error",
                {
                    "stage": "cleaning",
                    "message": "Raw data must contain a ticker column",
                    "details": {
                        "run_id": "run_0002",
                        "tickers": ["AAPL"],
                        "rows_in": 251,
                        "raw_columns": ["date", "ticker", "open", "high", "low", "close"],
                        "target_column": "target_direction",
                    },
                },
            ),
            (
                "modeling",
                "Modeling error",
                {
                    "stage": "modeling",
                    "message": "Not enough labeled rows to train models.",
                    "details": {
                        "run_id": "run_0003",
                        "tickers": ["AAPL"],
                        "rows_in": 251,
                        "rows_out": 248,
                        "feature_columns": ["open", "high", "low", "close"],
                        "target_column": "target_direction",
                    },
                },
            ),
        ]

        for stage_name, expected_title, error_payload in scenarios:
            with self.subTest(stage=stage_name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)

                    class FakeOrchestrator:
                        def __init__(self, artifact_root: str, payload: dict) -> None:
                            self.artifact_root = Path(artifact_root)
                            self.payload = payload

                        def run(self, request):  # type: ignore[no-untyped-def]
                            run_id = self.payload["details"]["run_id"]
                            run_dir = self.artifact_root / "runs" / run_id
                            run_dir.mkdir(parents=True, exist_ok=True)
                            manifest = {
                                "run_id": run_id,
                                "request": request.model_dump(),
                                "status": "failed",
                            }
                            (run_dir / "manifest.json").write_text(json.dumps(manifest, default=str))
                            (run_dir / "error.json").write_text(json.dumps(self.payload, default=str))
                            raise ValueError(self.payload["message"])

                    runtime = AssistantRuntime(artifact_root=str(root), session_id=f"{stage_name}-test")
                    runtime._orchestrator = FakeOrchestrator(str(root), error_payload)

                    answer = runtime.ask("AAPL")
                    status = runtime.ask("status")

                    self.assertIn(expected_title, answer)
                    self.assertIn("Stage:", answer)
                    self.assertIn("Action:", answer)
                    self.assertIn("Prompt:", answer)
                    self.assertNotIn("Traceback", answer)
                    if stage_name == "cleaning":
                        self.assertIn("Raw columns:", answer)
                    if stage_name == "modeling":
                        self.assertIn("Feature columns:", answer)
                    self.assertIn(expected_title, status)
                    self.assertEqual(runtime.get_state().last_run_id, error_payload["details"]["run_id"])


if __name__ == "__main__":
    unittest.main()
