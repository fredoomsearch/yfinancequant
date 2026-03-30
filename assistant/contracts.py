from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(text).date()
        except ValueError:
            return None
    return None


@dataclass
class AssistantRoute:
    intent: str = "unknown"
    tickers: List[str] = field(default_factory=list)
    start: Optional[date] = None
    end: Optional[date] = None
    interval: str = "1d"
    review_mode: str = "auto"
    use_reviewer: bool = True
    compare_binance: bool = False
    comparison_asset: Optional[str] = None
    comparison_yfinance_ticker: Optional[str] = None
    comparison_binance_symbol: Optional[str] = None
    experimental_groq_brain: bool = False
    artifact_root: str = "artifacts"
    language: str = "en"
    run_id: Optional[str] = None
    secondary_run_id: Optional[str] = None
    stage: Optional[str] = None
    needs_follow_up: bool = False
    follow_up_question: str = ""
    confidence: float = 0.0
    explanation: str = ""
    raw_message: str = ""
    answer_mode: str = "strict"
    certainty: str = "confirmed"
    question_focus: str = "general"
    source_mode: str = "local"
    web_queries: List[str] = field(default_factory=list)
    interpreted_query: str = ""
    interpretation_source: str = ""
    interpretation_note: str = ""
    conversation_act: str = "general"
    source_policy: str = ""
    web_required: bool = False
    allow_run: bool = True
    override_memory: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.start is not None:
            data["start"] = self.start.isoformat()
        if self.end is not None:
            data["end"] = self.end.isoformat()
        return data


## how method of interaction of the assistant for understand the user query and route to the correct handler
    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "AssistantRoute":
        if not isinstance(raw, dict):
            return cls()
        raw_tickers = raw.get("tickers") or []
        if isinstance(raw_tickers, str):
            tickers = [raw_tickers.strip()] if raw_tickers.strip() else []
        else:
            tickers = [str(item).strip() for item in raw_tickers if str(item).strip()]
        raw_web_queries = raw.get("web_queries") or []
        if isinstance(raw_web_queries, str):
            web_queries = [raw_web_queries.strip()] if raw_web_queries.strip() else []
        else:
            web_queries = [str(item).strip() for item in raw_web_queries if str(item).strip()]
        return cls(
            intent=str(raw.get("intent") or "unknown").strip() or "unknown",
            tickers=tickers,
            start=_coerce_date(raw.get("start")),
            end=_coerce_date(raw.get("end")),
            interval=str(raw.get("interval") or "1d").strip() or "1d",
            review_mode=str(raw.get("review_mode") or "auto").strip() or "auto",
            use_reviewer=bool(raw.get("use_reviewer", True)),
            compare_binance=bool(raw.get("compare_binance", False)),
            comparison_asset=(str(raw.get("comparison_asset") or "").strip() or None),
            comparison_yfinance_ticker=(str(raw.get("comparison_yfinance_ticker") or "").strip() or None),
            comparison_binance_symbol=(str(raw.get("comparison_binance_symbol") or "").strip() or None),
            artifact_root=str(raw.get("artifact_root") or "artifacts").strip() or "artifacts",
            language=str(raw.get("language") or "en").strip() or "en",
            run_id=(str(raw.get("run_id") or "").strip() or None),
            secondary_run_id=(str(raw.get("secondary_run_id") or "").strip() or None),
            stage=(str(raw.get("stage") or "").strip() or None),
            needs_follow_up=bool(raw.get("needs_follow_up", False)),
            follow_up_question=str(raw.get("follow_up_question") or "").strip(),
            confidence=float(raw.get("confidence") or 0.0),
            explanation=str(raw.get("explanation") or "").strip(),
            raw_message=str(raw.get("raw_message") or "").strip(),
            answer_mode=str(raw.get("answer_mode") or "strict").strip() or "strict",
            certainty=str(raw.get("certainty") or "confirmed").strip() or "confirmed",
            question_focus=str(raw.get("question_focus") or "general").strip() or "general",
            source_mode=str(raw.get("source_mode") or "local").strip() or "local",
            web_queries=web_queries,
            interpreted_query=str(raw.get("interpreted_query") or "").strip(),
            interpretation_source=str(raw.get("interpretation_source") or "").strip(),
            interpretation_note=str(raw.get("interpretation_note") or "").strip(),
            conversation_act=str(raw.get("conversation_act") or "general").strip() or "general",
            source_policy=str(raw.get("source_policy") or "").strip(),
            web_required=bool(raw.get("web_required", False)),
            allow_run=bool(raw.get("allow_run", True)),
            override_memory=bool(raw.get("override_memory", False)),
        )


@dataclass
class ConversationInterpretation:
    act: str = "general"
    subject: str = ""
    canonical_query: str = ""
    route_intent: str = ""
    stage: str = ""
    question_focus: str = ""
    tickers: List[str] = field(default_factory=list)
    run_id: Optional[str] = None
    source_policy: str = ""
    web_required: bool = False
    allow_run: bool = True
    override_memory: bool = False
    confidence: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssistantState:
    session_id: str = "default"
    last_run_id: Optional[str] = None
    last_intent: str = "idle"
    last_route: Dict[str, Any] = field(default_factory=dict)
    last_turn_trace: Dict[str, Any] = field(default_factory=dict)
    entity_memory: Dict[str, Any] = field(default_factory=dict)
    last_request: Dict[str, Any] = field(default_factory=dict)
    pending_route: Dict[str, Any] = field(default_factory=dict)
    pending_task: str = ""
    current_asset: str = ""
    current_mode: str = "local_only"
    preferred_language: str = ""
    last_summary: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "AssistantState":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            session_id=str(raw.get("session_id") or "default").strip() or "default",
            last_run_id=(str(raw.get("last_run_id") or "").strip() or None),
            last_intent=str(raw.get("last_intent") or "idle").strip() or "idle",
            last_route=dict(raw.get("last_route") or {}),
            last_turn_trace=dict(raw.get("last_turn_trace") or {}),
            entity_memory=dict(raw.get("entity_memory") or {}),
            last_request=dict(raw.get("last_request") or {}),
            pending_route=dict(raw.get("pending_route") or {}),
            pending_task=str(raw.get("pending_task") or "").strip(),
            current_asset=str(raw.get("current_asset") or "").strip(),
            current_mode=str(raw.get("current_mode") or "local_only").strip() or "local_only",
            preferred_language=str(raw.get("preferred_language") or "").strip(),
            last_summary=dict(raw.get("last_summary") or {}),
            notes=[str(item) for item in raw.get("notes") or [] if str(item).strip()],
            updated_at=str(raw.get("updated_at") or _utc_now_iso()).strip() or _utc_now_iso(),
        )
