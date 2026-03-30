from __future__ import annotations

import json
import os
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

from assistant.web import AssistantWebRetriever


class _LocalSearchHandler(BaseHTTPRequestHandler):
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

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = {key: values[0] if len(values) == 1 else values for key, values in parse_qs(parsed.query).items()}
        self.server.requests_log.append(  # type: ignore[attr-defined]
            {
                "method": "GET",
                "path": parsed.path,
                "headers": dict(self.headers),
                "query": query,
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
def _serve_search_payload(payload: dict) -> tuple[str, list[dict]]:
    server = HTTPServer(("127.0.0.1", 0), _LocalSearchHandler)
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


class AssistantWebRetrieverTest(unittest.TestCase):
    def test_retriever_uses_default_provider_without_config(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("assistant.web.requests.get", side_effect=Exception("offline")):
                retriever = AssistantWebRetriever()
                status = retriever.config_status().to_dict()
                facts = retriever.search("MSFT latest market context")

        self.assertTrue(retriever.enabled)
        self.assertEqual(retriever.provider, "duckduckgo")
        self.assertTrue(status["config_valid"])
        self.assertTrue(status["runtime_ready"])
        self.assertEqual(retriever.search_url, "https://api.duckduckgo.com/")
        self.assertEqual(facts, [])

    def test_retriever_parses_duckduckgo_default_payload(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "Heading": "Arbitrage",
            "AbstractText": "Arbitrage is the practice of exploiting price differences of the same asset in different markets.",
            "AbstractURL": "https://example.com/arbitrage",
            "RelatedTopics": [
                {
                    "Text": "Arbitrage is exploiting price differences.",
                    "FirstURL": "https://example.com/arbitrage-2",
                }
            ],
        }

        with patch.dict(os.environ, {}, clear=True):
            with patch("assistant.web.requests.get", return_value=response) as get_mock:
                retriever = AssistantWebRetriever()
                status = retriever.config_status().to_dict()
                facts = retriever.search("arbitrage", limit=3)

        self.assertEqual(retriever.provider, "duckduckgo")
        self.assertTrue(status["runtime_ready"])
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0].title, "Arbitrage")
        self.assertIn("exploiting price differences", facts[0].snippet.lower())
        self.assertEqual(facts[0].url, "https://example.com/arbitrage")
        self.assertEqual(facts[1].url, "https://example.com/arbitrage-2")
        self.assertEqual(get_mock.call_args.kwargs["params"]["q"], "arbitrage")

    def test_retriever_uses_post_by_default_and_parses_results(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {"title": "MSFT market update", "snippet": "Volume is rising.", "url": "https://example.com/msft"},
                {"title": "MSFT market update", "snippet": "Volume is rising.", "url": "https://example.com/msft"},
            ]
        }

        with patch.dict(os.environ, {"ASSISTANT_WEB_SEARCH_URL": "https://search.test"}, clear=True):
            with patch("assistant.web.requests.post", return_value=response) as post_mock:
                retriever = AssistantWebRetriever()
                facts = retriever.search("MSFT latest market context", limit=3)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].title, "MSFT market update")
        self.assertEqual(facts[0].snippet, "Volume is rising.")
        self.assertEqual(facts[0].url, "https://example.com/msft")
        self.assertEqual(facts[0].domain, "example.com")
        self.assertEqual(facts[0].query, "MSFT latest market context")
        self.assertEqual(facts[0].rank, 1)
        self.assertGreater(facts[0].trust_score, 0.5)
        post_mock.assert_called_once()

    def test_retriever_supports_get_and_authorization(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "items": [
                {"name": "BTC-USD", "description": "Crypto pair context.", "link": "https://example.com/btc"}
            ]
        }

        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_SEARCH_URL": "https://search.test",
                "ASSISTANT_WEB_SEARCH_METHOD": "GET",
                "ASSISTANT_WEB_SEARCH_API_KEY": "secret-key",
            },
            clear=True,
        ):
            with patch("assistant.web.requests.get", return_value=response) as get_mock:
                retriever = AssistantWebRetriever()
                facts = retriever.search("BTC-USD asset class", limit=2)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].title, "BTC-USD")
        self.assertEqual(facts[0].snippet, "Crypto pair context.")
        self.assertEqual(facts[0].url, "https://example.com/btc")
        self.assertEqual(get_mock.call_args.kwargs["headers"]["Authorization"], "Bearer secret-key")

    def test_retriever_parses_nested_data_payloads(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": {
                "results": [
                    {
                        "headline": "AAPL overview",
                        "summary": "Equity context.",
                        "source_url": "https://example.com/aapl",
                    }
                ]
            }
        }

        with patch.dict(os.environ, {"ASSISTANT_WEB_SEARCH_URL": "https://search.test"}, clear=True):
            with patch("assistant.web.requests.post", return_value=response):
                retriever = AssistantWebRetriever()
                facts = retriever.search("AAPL current market context", limit=2)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].title, "AAPL overview")
        self.assertEqual(facts[0].snippet, "Equity context.")
        self.assertEqual(facts[0].url, "https://example.com/aapl")

    def test_retriever_supports_custom_contract(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "payload": {
                "hits": [
                    {
                        "headline": "QQQ market context",
                        "text": "ETF context.",
                        "sourceUrl": "https://example.com/qqq",
                    }
                ]
            }
        }

        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_SEARCH_URL": "https://search.test",
                "ASSISTANT_WEB_QUERY_PARAM": "query",
                "ASSISTANT_WEB_LIMIT_PARAM": "size",
                "ASSISTANT_WEB_AUTH_HEADER": "X-API-Key",
                "ASSISTANT_WEB_AUTH_SCHEME": "",
                "ASSISTANT_WEB_RESULTS_PATH": "payload.hits",
                "ASSISTANT_WEB_SEARCH_API_KEY": "plain-secret",
            },
            clear=True,
        ):
            with patch("assistant.web.requests.post", return_value=response) as post_mock:
                retriever = AssistantWebRetriever()
                facts = retriever.search("QQQ latest context", limit=4)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].title, "QQQ market context")
        self.assertEqual(facts[0].snippet, "ETF context.")
        self.assertEqual(facts[0].url, "https://example.com/qqq")
        self.assertEqual(post_mock.call_args.kwargs["json"], {"query": "QQQ latest context", "size": 4})
        self.assertEqual(post_mock.call_args.kwargs["headers"]["X-API-Key"], "plain-secret")

    def test_retriever_applies_serper_preset_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_PROVIDER": "serper",
                "ASSISTANT_WEB_SEARCH_URL": "https://search.test",
                "ASSISTANT_WEB_SEARCH_API_KEY": "secret-key",
            },
            clear=True,
        ):
            retriever = AssistantWebRetriever()
            status = retriever.config_status().to_dict()

        self.assertEqual(retriever.method, "POST")
        self.assertEqual(retriever.query_param, "q")
        self.assertEqual(retriever.limit_param, "num")
        self.assertEqual(retriever.auth_header, "X-API-KEY")
        self.assertEqual(retriever.auth_param, "")
        self.assertEqual(retriever.results_path, "organic")
        self.assertEqual(status["provider"], "serper")
        self.assertTrue(status["config_valid"])

    def test_retriever_infers_provider_from_search_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_SEARCH_URL": "https://google.serper.dev/search",
                "ASSISTANT_WEB_SEARCH_API_KEY": "secret-key",
            },
            clear=True,
        ):
            retriever = AssistantWebRetriever()
            status = retriever.config_status().to_dict()

        self.assertEqual(retriever.provider, "serper")
        self.assertTrue(status["provider_known"])
        self.assertTrue(status["provider_host_match"])
        self.assertTrue(status["runtime_ready"])

    def test_retriever_uses_provider_default_search_url_when_available(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_PROVIDER": "tavily",
                "ASSISTANT_WEB_SEARCH_API_KEY": "secret-key",
            },
            clear=True,
        ):
            retriever = AssistantWebRetriever()
            status = retriever.config_status().to_dict()

        self.assertEqual(retriever.search_url, "https://api.tavily.com/search")
        self.assertTrue(status["enabled"])
        self.assertTrue(status["config_valid"])
        self.assertTrue(status["runtime_ready"])

    def test_retriever_requires_api_key_for_authenticated_preset(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_PROVIDER": "tavily",
                "ASSISTANT_WEB_SEARCH_URL": "https://api.tavily.com/search",
            },
            clear=True,
        ):
            status = AssistantWebRetriever().config_status().to_dict()

        self.assertFalse(status["config_valid"])
        self.assertFalse(status["runtime_ready"])
        self.assertTrue(any("ASSISTANT_WEB_SEARCH_API_KEY is required" in issue for issue in status["issues"]))

    def test_retriever_applies_tavily_payload_auth_param(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {"title": "Macro backdrop", "snippet": "Risk appetite improved.", "url": "https://example.com/macro"}
            ]
        }

        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_PROVIDER": "tavily",
                "ASSISTANT_WEB_SEARCH_URL": "https://search.test",
                "ASSISTANT_WEB_SEARCH_API_KEY": "secret-key",
            },
            clear=True,
        ):
            with patch("assistant.web.requests.post", return_value=response) as post_mock:
                retriever = AssistantWebRetriever()
                facts = retriever.search("MSFT latest market context", limit=2)

        self.assertEqual(len(facts), 1)
        self.assertEqual(post_mock.call_args.kwargs["json"]["query"], "MSFT latest market context")
        self.assertEqual(post_mock.call_args.kwargs["json"]["max_results"], 2)
        self.assertEqual(post_mock.call_args.kwargs["json"]["api_key"], "secret-key")
        self.assertNotIn("Authorization", post_mock.call_args.kwargs["headers"])

    def test_config_status_accepts_auth_param_without_auth_header(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_PROVIDER": "tavily",
                "ASSISTANT_WEB_SEARCH_URL": "https://search.test",
                "ASSISTANT_WEB_SEARCH_API_KEY": "secret-key",
            },
            clear=True,
        ):
            status = AssistantWebRetriever().config_status().to_dict()

        self.assertTrue(status["enabled"])
        self.assertTrue(status["config_valid"])
        self.assertTrue(status["runtime_ready"])
        self.assertEqual(status["auth_header"], "")
        self.assertEqual(status["auth_param"], "api_key")
        self.assertTrue(status["cache_enabled"])
        self.assertEqual(status["cache_ttl_seconds"], 300)

    def test_config_status_reports_unknown_provider(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_PROVIDER": "unknown-provider",
                "ASSISTANT_WEB_SEARCH_URL": "https://search.test",
            },
            clear=True,
        ):
            status = AssistantWebRetriever().config_status().to_dict()

        self.assertFalse(status["config_valid"])
        self.assertFalse(status["provider_known"])
        self.assertTrue(any("Supported presets" in issue for issue in status["issues"]))

    def test_config_status_reports_provider_host_mismatch(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_PROVIDER": "tavily",
                "ASSISTANT_WEB_SEARCH_URL": "https://google.serper.dev/search",
            },
            clear=True,
        ):
            status = AssistantWebRetriever().config_status().to_dict()

        self.assertFalse(status["config_valid"])
        self.assertFalse(status["provider_host_match"])
        self.assertTrue(any("does not look consistent" in issue for issue in status["issues"]))

    def test_retriever_skips_invalid_items(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {"foo": "bar"},
                "not-a-dict",
                {"title": "Valid item", "description": "Useful snippet", "url": "https://example.com/valid"},
            ]
        }

        with patch.dict(os.environ, {"ASSISTANT_WEB_SEARCH_URL": "https://search.test"}, clear=True):
            with patch("assistant.web.requests.post", return_value=response):
                retriever = AssistantWebRetriever()
                facts = retriever.search("valid market context", limit=5)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].title, "Valid item")

    def test_retriever_hits_local_http_server_end_to_end(self) -> None:
        payload = {
            "results": [
                {
                    "title": "MSFT live context",
                    "snippet": "Cloud demand stayed firm with moderate volatility.",
                    "url": "https://example.com/msft-live",
                }
            ]
        }

        with _serve_search_payload(payload) as (search_url, requests_log):
            with patch.dict(
                os.environ,
                {
                    "ASSISTANT_WEB_SEARCH_URL": search_url,
                    "ASSISTANT_WEB_SEARCH_METHOD": "POST",
                },
                clear=True,
            ):
                retriever = AssistantWebRetriever()
                facts = retriever.search("MSFT latest market context", limit=2)

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].title, "MSFT live context")
        self.assertEqual(facts[0].url, "https://example.com/msft-live")
        self.assertEqual(facts[0].domain, "example.com")
        self.assertEqual(facts[0].query, "MSFT latest market context")
        self.assertGreater(facts[0].trust_score, 0.5)
        self.assertEqual(len(requests_log), 1)
        self.assertEqual(requests_log[0]["method"], "POST")
        self.assertEqual(requests_log[0]["json"], {"q": "MSFT latest market context", "limit": 2})

    def test_retriever_caches_repeated_queries_with_same_instance(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {"title": "Cached result", "snippet": "Useful context for caching.", "url": "https://example.com/cache"}
            ]
        }

        with patch.dict(os.environ, {"ASSISTANT_WEB_SEARCH_URL": "https://search.test"}, clear=True):
            with patch("assistant.web.requests.post", return_value=response) as post_mock:
                retriever = AssistantWebRetriever()
                facts_one = retriever.search("cached query", limit=2)
                facts_two = retriever.search("cached query", limit=2)

        self.assertEqual(len(facts_one), 1)
        self.assertEqual(len(facts_two), 1)
        self.assertEqual(facts_two[0].domain, "example.com")
        self.assertEqual(post_mock.call_count, 1)

    def test_retriever_reports_invalid_config(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_SEARCH_URL": "ftp://search.test",
                "ASSISTANT_WEB_SEARCH_METHOD": "PUT",
                "ASSISTANT_WEB_SEARCH_TIMEOUT": "0",
                "ASSISTANT_WEB_CACHE_TTL_SECONDS": "bad-cache",
            },
            clear=True,
        ):
            status = AssistantWebRetriever().config_status().to_dict()

        self.assertTrue(status["enabled"])
        self.assertFalse(status["config_valid"])
        self.assertFalse(status["runtime_ready"])
        self.assertTrue(any("http or https" in issue for issue in status["issues"]))
        self.assertTrue(any("GET or POST" in issue for issue in status["issues"]))
        self.assertTrue(any("greater than zero" in issue for issue in status["issues"]))
        self.assertTrue(any("CACHE_TTL_SECONDS must be numeric" in issue for issue in status["issues"]))

    def test_retriever_probe_returns_config_error_when_preset_requires_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ASSISTANT_WEB_PROVIDER": "tavily",
                "ASSISTANT_WEB_SEARCH_URL": "https://api.tavily.com/search",
            },
            clear=True,
        ):
            report = AssistantWebRetriever().probe().to_dict()

        self.assertTrue(report["configured"])
        self.assertFalse(report["config_valid"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["fact_count"], 0)
        self.assertIn("ASSISTANT_WEB_SEARCH_API_KEY", report["error"])

    def test_retriever_probe_uses_http_transport_when_configured(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "results": [
                {"title": "MSFT probe result", "snippet": "Probe context.", "url": "https://example.com/probe"}
            ]
        }

        with patch.dict(os.environ, {"ASSISTANT_WEB_SEARCH_URL": "https://search.test"}, clear=True):
            with patch("assistant.web.requests.post", return_value=response):
                report = AssistantWebRetriever().probe(query="MSFT probe", limit=1).to_dict()

        self.assertTrue(report["configured"])
        self.assertTrue(report["config_valid"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["fact_count"], 1)
        self.assertEqual(report["domain_count"], 1)
        self.assertEqual(report["domains"], ["example.com"])
        self.assertGreater(report["top_trust_score"], 0.5)
        self.assertEqual(report["facts"][0]["title"], "MSFT probe result")


if __name__ == "__main__":
    unittest.main()
