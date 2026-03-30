from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_loader import load_project_env

load_project_env()

from assistant.web import AssistantWebRetriever, build_web_provider_env, list_web_provider_presets


@contextmanager
def _env_overrides(values: dict[str, str]) -> None:
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe the configured assistant web retriever")
    parser.add_argument("--query", default="MSFT latest market context", help="Query to send to the retriever")
    parser.add_argument("--limit", type=int, default=2, help="Maximum number of facts to request")
    parser.add_argument("--provider", help="Optional provider preset override, for example searxng, serper, tavily, searchapi")
    parser.add_argument("--search-url", help="Optional search endpoint override")
    parser.add_argument("--api-key", help="Optional API key override")
    parser.add_argument("--method", help="Optional HTTP method override")
    parser.add_argument("--query-param", help="Optional query field override")
    parser.add_argument("--limit-param", help="Optional limit field override")
    parser.add_argument("--auth-header", help="Optional auth header override")
    parser.add_argument("--auth-scheme", help="Optional auth scheme override")
    parser.add_argument("--auth-param", help="Optional auth payload/query param override")
    parser.add_argument("--results-path", help="Optional dotted path to the results list")
    parser.add_argument("--timeout", type=float, help="Optional timeout override")
    parser.add_argument("--status-only", action="store_true", help="Print only config status without issuing a probe request")
    parser.add_argument("--print-env", action="store_true", help="Print a ready-to-copy env snippet for the selected provider")
    parser.add_argument("--save-report", help="Optional path to save the resulting status/probe report as JSON")
    parser.add_argument("--json", action="store_true", help="Print the probe result as JSON")
    return parser


def _render_text(report: dict) -> str:
    config = report.get("config") or {}
    lines = [
        f"Web retriever probe: ok={report.get('ok')} configured={report.get('configured')} config_valid={report.get('config_valid')}",
        f"Query: {report.get('query')} limit={report.get('limit')}",
        f"Endpoint: {config.get('search_url') or 'n/a'} method={config.get('method') or 'n/a'} provider={config.get('provider') or 'custom'}",
        f"Contract: query_param={config.get('query_param') or 'n/a'} limit_param={config.get('limit_param') or 'n/a'} auth_header={config.get('auth_header') or 'n/a'} auth_param={config.get('auth_param') or 'n/a'}",
        f"Runtime ready: {config.get('runtime_ready')}",
    ]
    if report.get("domain_count") is not None:
        lines.append(
            f"Domains: {report.get('domain_count')} top_trust={report.get('top_trust_score')} values={', '.join(report.get('domains') or []) or 'n/a'}"
        )
    if config.get("issues"):
        lines.append(f"Issues: {' | '.join(config.get('issues') or [])}")
    if report.get("error"):
        lines.append(f"Error: {report.get('error')}")
    lines.append(f"Fact count: {report.get('fact_count')}")
    for fact in report.get("facts") or []:
        lines.append(f"- {fact.get('title') or 'n/a'} | {fact.get('url') or 'n/a'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.print_env:
        provider = str(args.provider or "").strip().lower()
        if not provider:
            parser.error(f"--print-env requires --provider. Supported presets: {', '.join(list_web_provider_presets())}")
        lines = build_web_provider_env(provider, search_url=args.search_url or "", api_key_placeholder=args.api_key or "...")
        if args.json:
            print(json.dumps({"provider": provider, "env": lines}, indent=2))
        else:
            print("\n".join(lines))
        return 0
    overrides = {
        "ASSISTANT_WEB_PROVIDER": args.provider,
        "ASSISTANT_WEB_SEARCH_URL": args.search_url,
        "ASSISTANT_WEB_SEARCH_API_KEY": args.api_key,
        "ASSISTANT_WEB_SEARCH_METHOD": args.method,
        "ASSISTANT_WEB_QUERY_PARAM": args.query_param,
        "ASSISTANT_WEB_LIMIT_PARAM": args.limit_param,
        "ASSISTANT_WEB_AUTH_HEADER": args.auth_header,
        "ASSISTANT_WEB_AUTH_SCHEME": args.auth_scheme,
        "ASSISTANT_WEB_AUTH_PARAM": args.auth_param,
        "ASSISTANT_WEB_RESULTS_PATH": args.results_path,
    }
    if args.timeout is not None:
        overrides["ASSISTANT_WEB_SEARCH_TIMEOUT"] = str(args.timeout)
    env_overrides = {key: value for key, value in overrides.items() if value is not None}
    with _env_overrides(env_overrides):
        retriever = AssistantWebRetriever()
        if args.status_only:
            payload = retriever.config_status().to_dict()
        else:
            payload = retriever.probe(query=args.query, limit=args.limit).to_dict()
    if args.save_report:
        save_path = Path(args.save_report)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2))
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if args.status_only:
            status_lines = [
                f"Web retriever status: enabled={payload.get('enabled')} config_valid={payload.get('config_valid')} runtime_ready={payload.get('runtime_ready')}",
                f"Endpoint: {payload.get('search_url') or 'n/a'} method={payload.get('method') or 'n/a'} provider={payload.get('provider') or 'custom'}",
                f"Contract: query_param={payload.get('query_param') or 'n/a'} limit_param={payload.get('limit_param') or 'n/a'} auth_header={payload.get('auth_header') or 'n/a'} auth_param={payload.get('auth_param') or 'n/a'}",
            ]
            if payload.get("issues"):
                status_lines.append(f"Issues: {' | '.join(payload.get('issues') or [])}")
                status_lines.append(f"Supported presets: {', '.join(list_web_provider_presets())}")
            print("\n".join(status_lines))
        else:
            print(_render_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
