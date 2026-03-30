from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from assistant.domain import resolve_market_identity
from assistant.web import WebFact


@dataclass
class GroundingPacket:
    local_facts: Dict[str, Any] = field(default_factory=dict)
    web_facts: List[Dict[str, Any]] = field(default_factory=list)
    web_overview: Dict[str, Any] = field(default_factory=dict)
    source_mode: str = "local"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_grounding_packet(
    summary: Dict[str, Any],
    result: Dict[str, Any],
    source_mode: str,
    web_facts: List[WebFact] | None = None,
) -> GroundingPacket:
    tickers = summary.get("tickers") or ((result.get("manifest") or {}).get("request") or {}).get("tickers") or []
    symbol = str(tickers[0]).strip() if tickers else ""
    identity = resolve_market_identity(symbol)
    local_facts = {
        "run_id": summary.get("run_id") or result.get("run_id"),
        "symbol": symbol,
        "market_identity": identity.to_dict(),
        "decision": (summary.get("models") or {}).get("final_decision") or result.get("final_decision"),
        "confidence": (summary.get("models") or {}).get("confidence") or result.get("final_confidence"),
        "rows": summary.get("rows") or {},
    }
    web_fact_dicts = [item.to_dict() for item in (web_facts or [])]
    domains = [
        str(item.get("domain") or "").strip().lower()
        for item in web_fact_dicts
        if str(item.get("domain") or "").strip()
    ]
    trust_scores = [
        float(item.get("trust_score") or 0.0)
        for item in web_fact_dicts
        if float(item.get("trust_score") or 0.0) > 0
    ]
    return GroundingPacket(
        local_facts=local_facts,
        web_facts=web_fact_dicts,
        web_overview={
            "fact_count": len(web_fact_dicts),
            "domains": sorted(set(domains)),
            "top_trust_score": round(max(trust_scores), 2) if trust_scores else 0.0,
        },
        source_mode=source_mode,
    )
