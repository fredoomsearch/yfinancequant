from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlparse
from typing import Any, Dict, List

import requests


_WEB_PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "duckduckgo": {
        "method": "GET",
        "query_param": "q",
        "limit_param": "num",
        "auth_header": "",
        "auth_scheme": "",
        "auth_param": "",
        "results_path": "",
        "example_search_url": "https://api.duckduckgo.com/",
        "expected_host_hint": "duckduckgo",
    },
    "searxng": {
        "method": "GET",
        "query_param": "q",
        "limit_param": "limit",
        "auth_header": "",
        "auth_scheme": "",
        "auth_param": "",
        "results_path": "results",
        "example_search_url": "https://YOUR-SEARXNG-ENDPOINT/search",
        "expected_host_hint": "searx",
    },
    "serper": {
        "method": "POST",
        "query_param": "q",
        "limit_param": "num",
        "auth_header": "X-API-KEY",
        "auth_scheme": "",
        "auth_param": "",
        "results_path": "organic",
        "example_search_url": "https://google.serper.dev/search",
        "expected_host_hint": "serper",
    },
    "tavily": {
        "method": "POST",
        "query_param": "query",
        "limit_param": "max_results",
        "auth_header": "",
        "auth_scheme": "",
        "auth_param": "api_key",
        "results_path": "results",
        "example_search_url": "https://api.tavily.com/search",
        "expected_host_hint": "tavily",
    },
    "searchapi": {
        "method": "GET",
        "query_param": "q",
        "limit_param": "num",
        "auth_header": "",
        "auth_scheme": "",
        "auth_param": "api_key",
        "results_path": "organic_results",
        "example_search_url": "https://YOUR-SEARCHAPI-ENDPOINT",
        "expected_host_hint": "searchapi",
    },
}


def list_web_provider_presets() -> List[str]:
    return sorted(_WEB_PROVIDER_PRESETS.keys())


def _looks_placeholder_url(url: str) -> bool:
    normalized = str(url or "").strip().lower()
    if not normalized:
        return True
    return any(token in normalized for token in ("your-", "your_", "your endpoint", "YOUR-".lower(), "YOUR_".lower()))


def _default_search_url_for_provider(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    preset = _WEB_PROVIDER_PRESETS.get(normalized)
    if not preset:
        return ""
    candidate = str(preset.get("example_search_url") or "").strip()
    if not candidate or _looks_placeholder_url(candidate):
        return ""
    return candidate


def _infer_provider_from_search_url(search_url: str) -> str:
    parsed = urlparse(str(search_url or "").strip())
    host = str(parsed.netloc or "").strip().lower()
    if not host:
        return ""
    for provider, preset in _WEB_PROVIDER_PRESETS.items():
        expected = str(preset.get("expected_host_hint") or "").strip().lower()
        if not expected:
            continue
        if expected in host:
            return provider
    return ""


def describe_web_provider(provider: str) -> Dict[str, str]:
    normalized = str(provider or "").strip().lower()
    preset = _WEB_PROVIDER_PRESETS.get(normalized)
    if not preset:
        raise ValueError(f"Unknown web provider preset: {provider}")
    auth_header = str(preset.get("auth_header") or "").strip()
    auth_param = str(preset.get("auth_param") or "").strip()
    auth_mode = "header" if auth_header else ("param" if auth_param else "none")
    auth_target = auth_header or auth_param or "n/a"
    return {
        "provider": normalized,
        "method": str(preset.get("method") or "").strip() or "POST",
        "query_param": str(preset.get("query_param") or "").strip() or "q",
        "limit_param": str(preset.get("limit_param") or "").strip() or "limit",
        "results_path": str(preset.get("results_path") or "").strip() or "n/a",
        "example_search_url": str(preset.get("example_search_url") or "").strip() or "https://YOUR-SEARCH-ENDPOINT",
        "auth_mode": auth_mode,
        "auth_target": auth_target,
    }


def build_web_provider_env(
    provider: str,
    *,
    search_url: str = "",
    api_key_placeholder: str = "...",
) -> List[str]:
    normalized = str(provider or "").strip().lower()
    preset = _WEB_PROVIDER_PRESETS.get(normalized)
    if not preset:
        raise ValueError(f"Unknown web provider preset: {provider}")
    lines = [f'ASSISTANT_WEB_PROVIDER="{normalized}"']
    default_search_url = str(preset.get("example_search_url") or "").strip() or "https://YOUR-SEARCH-ENDPOINT"
    if search_url:
        lines.append(f'ASSISTANT_WEB_SEARCH_URL="{search_url}"')
    else:
        lines.append(f'ASSISTANT_WEB_SEARCH_URL="{default_search_url}"')
    auth_header = str(preset.get("auth_header") or "").strip()
    auth_param = str(preset.get("auth_param") or "").strip()
    if auth_header or auth_param:
        lines.append(f'ASSISTANT_WEB_SEARCH_API_KEY="{api_key_placeholder}"')
    return lines


def build_web_provider_probe_command(
    provider: str,
    *,
    search_url: str = "",
    api_key_placeholder: str = "...",
    query: str = "MSFT latest market context",
) -> str:
    normalized = str(provider or "").strip().lower()
    preset = _WEB_PROVIDER_PRESETS.get(normalized)
    if not preset:
        raise ValueError(f"Unknown web provider preset: {provider}")
    endpoint = search_url or str(preset.get("example_search_url") or "").strip() or "https://YOUR-SEARCH-ENDPOINT"
    parts = [
        ".venv/bin/python",
        "scripts/assistant_web_probe.py",
        "--provider",
        normalized,
        "--search-url",
        endpoint,
        "--status-only",
        "--json",
    ]
    if str(preset.get("auth_header") or "").strip() or str(preset.get("auth_param") or "").strip():
        parts.extend(["--api-key", api_key_placeholder])
    if query:
        parts.extend(["--query", query])
    return " ".join(parts)


@dataclass
class WebFact:
    title: str
    snippet: str
    url: str
    domain: str = ""
    query: str = ""
    rank: int = 0
    trust_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WebRetrieverConfigStatus:
    enabled: bool
    config_valid: bool
    runtime_ready: bool
    provider: str
    provider_known: bool
    provider_host_match: bool
    search_url: str
    method: str
    query_param: str
    limit_param: str
    auth_header: str
    auth_param: str
    results_path: str
    timeout: float
    cache_enabled: bool
    cache_ttl_seconds: int
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WebRetrieverProbeResult:
    configured: bool
    config_valid: bool
    ok: bool
    query: str
    limit: int
    fact_count: int
    domain_count: int
    domains: List[str]
    top_trust_score: float
    error: str
    facts: List[Dict[str, Any]]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AssistantWebRetriever:
    """Optional web retriever. Disabled unless explicit config is present."""

    def __init__(self) -> None:
        raw_provider = str(os.getenv("ASSISTANT_WEB_PROVIDER") or "").strip().lower()
        raw_search_url = str(os.getenv("ASSISTANT_WEB_SEARCH_URL") or "").strip()
        inferred_provider = raw_provider or _infer_provider_from_search_url(raw_search_url)
        if not inferred_provider and not raw_search_url:
            inferred_provider = "duckduckgo"
        self.provider = inferred_provider
        preset = _WEB_PROVIDER_PRESETS.get(self.provider, {})
        self.search_url = raw_search_url or _default_search_url_for_provider(self.provider)
        self.api_key = str(os.getenv("ASSISTANT_WEB_SEARCH_API_KEY") or "").strip()
        self.method = str(os.getenv("ASSISTANT_WEB_SEARCH_METHOD") or preset.get("method") or "POST").strip().upper() or "POST"
        raw_timeout = str(os.getenv("ASSISTANT_WEB_SEARCH_TIMEOUT") or "10").strip()
        self.timeout_invalid = False
        try:
            self.timeout = float(raw_timeout or "10")
        except ValueError:
            self.timeout = 10.0
            self.timeout_invalid = True
        self.query_param = str(os.getenv("ASSISTANT_WEB_QUERY_PARAM") or preset.get("query_param") or "q").strip() or "q"
        self.limit_param = str(os.getenv("ASSISTANT_WEB_LIMIT_PARAM") or preset.get("limit_param") or "limit").strip() or "limit"
        auth_header_default = str(preset.get("auth_header") or "").strip()
        auth_param_default = str(preset.get("auth_param") or "").strip()
        if not auth_header_default and not auth_param_default:
            auth_header_default = "Authorization" if self.api_key and self.provider not in _WEB_PROVIDER_PRESETS else ""
        self.auth_header = str(os.getenv("ASSISTANT_WEB_AUTH_HEADER") or auth_header_default).strip()
        raw_auth_scheme = os.getenv("ASSISTANT_WEB_AUTH_SCHEME")
        preset_auth_scheme = preset.get("auth_scheme")
        if raw_auth_scheme is None:
            self.auth_scheme = "Bearer" if preset_auth_scheme is None else str(preset_auth_scheme).strip()
        else:
            self.auth_scheme = str(raw_auth_scheme).strip()
        self.auth_param = str(os.getenv("ASSISTANT_WEB_AUTH_PARAM") or auth_param_default or "").strip()
        self.results_path = str(os.getenv("ASSISTANT_WEB_RESULTS_PATH") or preset.get("results_path") or "").strip()
        raw_cache_ttl = str(os.getenv("ASSISTANT_WEB_CACHE_TTL_SECONDS") or "300").strip()
        self.cache_ttl_invalid = False
        try:
            self.cache_ttl_seconds = max(0, int(float(raw_cache_ttl or "300")))
        except ValueError:
            self.cache_ttl_seconds = 300
            self.cache_ttl_invalid = True
        self._cache: Dict[str, tuple[float, List[WebFact]]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.search_url)

    def config_status(self) -> WebRetrieverConfigStatus:
        issues: List[str] = []
        provider_known = not self.provider or self.provider in _WEB_PROVIDER_PRESETS
        provider_host_match = True
        if self.provider and self.provider not in _WEB_PROVIDER_PRESETS:
            issues.append(
                "ASSISTANT_WEB_PROVIDER is unknown. Supported presets: "
                + ", ".join(list_web_provider_presets())
                + "."
            )
        if not self.search_url:
            issues.append("ASSISTANT_WEB_SEARCH_URL is missing.")
        else:
            parsed = urlparse(self.search_url)
            if parsed.scheme not in {"http", "https"}:
                issues.append("ASSISTANT_WEB_SEARCH_URL must use http or https.")
            if not parsed.netloc:
                issues.append("ASSISTANT_WEB_SEARCH_URL must include a host.")
            provider_host_match = self._provider_host_matches(parsed.netloc)
            if self.provider and not provider_host_match:
                issues.append(
                    "ASSISTANT_WEB_SEARCH_URL does not look consistent with the selected provider preset."
                )
        requires_api_key = bool(self.provider and self.provider in _WEB_PROVIDER_PRESETS and (self.auth_header or self.auth_param))
        if requires_api_key and not self.api_key:
            issues.append(
                "ASSISTANT_WEB_SEARCH_API_KEY is required for the selected provider preset."
            )
        if self.method not in {"GET", "POST"}:
            issues.append("ASSISTANT_WEB_SEARCH_METHOD must be GET or POST.")
        if not self.query_param:
            issues.append("ASSISTANT_WEB_QUERY_PARAM cannot be empty.")
        if not self.limit_param:
            issues.append("ASSISTANT_WEB_LIMIT_PARAM cannot be empty.")
        if self.api_key and not (self.auth_header or self.auth_param):
            issues.append(
                "ASSISTANT_WEB_SEARCH_API_KEY requires ASSISTANT_WEB_AUTH_HEADER or ASSISTANT_WEB_AUTH_PARAM."
            )
        if self.timeout_invalid:
            issues.append("ASSISTANT_WEB_SEARCH_TIMEOUT must be numeric.")
        if self.timeout <= 0:
            issues.append("ASSISTANT_WEB_SEARCH_TIMEOUT must be greater than zero.")
        if self.cache_ttl_invalid:
            issues.append("ASSISTANT_WEB_CACHE_TTL_SECONDS must be numeric.")
        config_valid = not issues
        return WebRetrieverConfigStatus(
            enabled=self.enabled,
            config_valid=config_valid,
            runtime_ready=self.enabled and config_valid,
            provider=self.provider,
            provider_known=provider_known,
            provider_host_match=provider_host_match,
            search_url=self.search_url,
            method=self.method,
            query_param=self.query_param,
            limit_param=self.limit_param,
            auth_header=self.auth_header,
            auth_param=self.auth_param,
            results_path=self.results_path,
            timeout=self.timeout,
            cache_enabled=self.cache_ttl_seconds > 0,
            cache_ttl_seconds=self.cache_ttl_seconds,
            issues=issues,
        )

    def search(self, query: str, limit: int = 3) -> List[WebFact]:
        if not self.enabled or not str(query or "").strip():
            return []
        cache_key = self._cache_key(query=query, limit=limit)
        cached = self._cached_facts(cache_key)
        if cached is not None:
            return cached
        try:
            data = self._request_json(query=query, limit=limit)
        except Exception:
            return []
        items = self._extract_items(data)
        facts = self._facts_from_items(items, query=query, limit=limit)
        self._store_cached_facts(cache_key, facts)
        return facts

    def probe(self, query: str = "MSFT latest market context", limit: int = 2) -> WebRetrieverProbeResult:
        status = self.config_status()
        if not status.runtime_ready:
            return WebRetrieverProbeResult(
                configured=status.enabled,
                config_valid=status.config_valid,
                ok=False,
                query=query,
                limit=limit,
                fact_count=0,
                domain_count=0,
                domains=[],
                top_trust_score=0.0,
                error="; ".join(status.issues) if status.issues else "Web retriever is not configured.",
                facts=[],
                config=status.to_dict(),
            )
        try:
            data = self._request_json(query=query, limit=limit)
        except Exception as exc:
            return WebRetrieverProbeResult(
                configured=status.enabled,
                config_valid=status.config_valid,
                ok=False,
                query=query,
                limit=limit,
                fact_count=0,
                domain_count=0,
                domains=[],
                top_trust_score=0.0,
                error=str(exc),
                facts=[],
                config=status.to_dict(),
            )
        items = self._extract_items(data)
        facts = self._facts_from_items(items, query=query, limit=limit)
        domains = sorted({fact.domain for fact in facts if fact.domain})
        top_trust_score = round(max((fact.trust_score for fact in facts), default=0.0), 2)
        return WebRetrieverProbeResult(
            configured=status.enabled,
            config_valid=status.config_valid,
            ok=True,
            query=query,
            limit=limit,
            fact_count=len(facts),
            domain_count=len(domains),
            domains=domains,
            top_trust_score=top_trust_score,
            error="",
            facts=[fact.to_dict() for fact in facts],
            config=status.to_dict(),
        )

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.auth_header:
            headers[self.auth_header] = f"{self.auth_scheme} {self.api_key}".strip() if self.auth_scheme else self.api_key
        return headers

    def _build_payload(self, query: str, limit: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {self.query_param: query, self.limit_param: limit}
        if self.api_key and self.auth_param:
            payload[self.auth_param] = self.api_key
        return payload

    def _request_json(self, query: str, limit: int) -> Any:
        headers = self._build_headers()
        payload = self._build_payload(query, limit)
        if self.method == "GET":
            response = requests.get(self.search_url, headers=headers, params=payload, timeout=self.timeout)
        else:
            response = requests.post(self.search_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _cache_key(self, *, query: str, limit: int) -> str:
        return "|".join(
            [
                self.provider or "custom",
                self.search_url,
                self.method,
                self.query_param,
                self.limit_param,
                self.results_path,
                str(limit),
                str(query or "").strip().lower(),
            ]
        )

    def _cached_facts(self, cache_key: str) -> List[WebFact] | None:
        if self.cache_ttl_seconds <= 0:
            return None
        cached = self._cache.get(cache_key)
        if not cached:
            return None
        stored_at, facts = cached
        if (time.time() - stored_at) > self.cache_ttl_seconds:
            self._cache.pop(cache_key, None)
            return None
        return list(facts)

    def _store_cached_facts(self, cache_key: str, facts: List[WebFact]) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        self._cache[cache_key] = (time.time(), list(facts))

    def _facts_from_items(self, items: List[Any], *, query: str, limit: int) -> List[WebFact]:
        facts: List[WebFact] = []
        seen: set[tuple[str, str]] = set()
        for index, item in enumerate(items, start=1):
            fact = self._coerce_fact(item, query=query, rank=index)
            if fact is None:
                continue
            dedupe_key = (fact.url, fact.title)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            facts.append(fact)
            if len(facts) >= limit:
                break
        return facts

    def _extract_items(self, data: Any) -> List[Any]:
        duckduckgo_items = self._extract_duckduckgo_items(data)
        if duckduckgo_items:
            return duckduckgo_items
        if self.results_path:
            candidate = self._dig_path(data, self.results_path)
            if isinstance(candidate, list):
                return candidate
            if isinstance(candidate, dict):
                return [candidate]
        if isinstance(data, list):
            return data
        if not isinstance(data, dict):
            return []
        if any(key in data for key in ("title", "name", "headline", "snippet", "description", "content", "url", "link")):
            return [data]
        for key in ("results", "items", "documents", "entries", "hits"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                return candidate
        for key in ("data", "response", "payload"):
            candidate = data.get(key)
            extracted = self._extract_items(candidate)
            if extracted:
                return extracted
        return []

    def _extract_duckduckgo_items(self, data: Any) -> List[Any]:
        if self.provider != "duckduckgo" or not isinstance(data, dict):
            return []
        items: List[Any] = []
        abstract_text = self._normalize_text(str(data.get("AbstractText") or data.get("Abstract") or ""))
        heading = self._normalize_text(str(data.get("Heading") or data.get("AnswerType") or ""))
        abstract_url = str(data.get("AbstractURL") or data.get("Redirect") or "").strip()
        if abstract_text or heading or abstract_url:
            items.append(
                {
                    "title": heading or abstract_text[:80] or "DuckDuckGo result",
                    "snippet": abstract_text or heading,
                    "url": abstract_url,
                }
            )
        related = data.get("RelatedTopics")
        if isinstance(related, list):
            items.extend(self._flatten_duckduckgo_related_topics(related))
        results = data.get("Results")
        if isinstance(results, list):
            items.extend(self._flatten_duckduckgo_related_topics(results))
        return items

    def _flatten_duckduckgo_related_topics(self, topics: List[Any]) -> List[Any]:
        flattened: List[Any] = []
        for topic in topics:
            if not isinstance(topic, dict):
                continue
            nested = topic.get("Topics")
            if isinstance(nested, list) and nested:
                flattened.extend(self._flatten_duckduckgo_related_topics(nested))
                continue
            text = self._normalize_text(str(topic.get("Text") or topic.get("Result") or ""))
            url = str(topic.get("FirstURL") or topic.get("Url") or topic.get("URL") or "").strip()
            if not text and not url:
                continue
            title = self._normalize_text(str(topic.get("Name") or topic.get("Heading") or "")) or text[:80] or "DuckDuckGo topic"
            flattened.append(
                {
                    "title": title,
                    "snippet": text,
                    "url": url,
                }
            )
        return flattened

    def _coerce_fact(self, item: Any, *, query: str = "", rank: int = 0) -> WebFact | None:
        if not isinstance(item, dict):
            return None
        title = self._normalize_text(str(item.get("title") or item.get("name") or item.get("headline") or ""))
        snippet = self._normalize_text(
            str(
            item.get("snippet")
            or item.get("description")
            or item.get("summary")
            or item.get("content")
            or item.get("text")
            or ""
            )
        )
        url = str(item.get("url") or item.get("link") or item.get("source_url") or item.get("sourceUrl") or "").strip()
        if not any((title, snippet, url)):
            return None
        domain = self._infer_domain(url)
        return WebFact(
            title=title,
            snippet=snippet,
            url=url,
            domain=domain,
            query=str(query or "").strip(),
            rank=max(1, int(rank or 1)),
            trust_score=self._score_fact(title=title, snippet=snippet, url=url, domain=domain, query=query, rank=rank),
        )

    def _dig_path(self, data: Any, path: str) -> Any:
        current = data
        for part in [segment.strip() for segment in str(path or "").split(".") if segment.strip()]:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def _provider_host_matches(self, host: str) -> bool:
        normalized_host = str(host or "").strip().lower()
        if not self.provider or self.provider not in _WEB_PROVIDER_PRESETS or not normalized_host:
            return True
        expected = str(_WEB_PROVIDER_PRESETS.get(self.provider, {}).get("expected_host_hint") or "").strip().lower()
        if not expected:
            return True
        if "your-" in normalized_host or "your_" in normalized_host:
            return True
        if normalized_host.endswith(".test") or normalized_host.startswith("localhost") or normalized_host.startswith("127.0.0.1"):
            return True
        return expected in normalized_host

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip())

    def _infer_domain(self, url: str) -> str:
        parsed = urlparse(str(url or "").strip())
        return str(parsed.netloc or "").strip().lower().removeprefix("www.")

    def _score_fact(
        self,
        *,
        title: str,
        snippet: str,
        url: str,
        domain: str,
        query: str,
        rank: int,
    ) -> float:
        score = 0.35
        if title:
            score += 0.15
        if snippet:
            score += 0.15
        if domain:
            score += 0.1
        if str(url or "").startswith("https://"):
            score += 0.05
        if len(str(snippet or "").split()) >= 6:
            score += 0.05
        query_terms = {
            token
            for token in re.findall(r"[a-z0-9][a-z0-9_-]*", str(query or "").lower())
            if len(token) >= 3
        }
        body = f"{title} {snippet} {domain}".lower()
        overlap = sum(1 for token in query_terms if token in body)
        score += min(0.2, overlap * 0.04)
        score += max(0.0, 0.08 - (0.02 * max(int(rank or 1) - 1, 0)))
        return round(min(score, 0.99), 2)
