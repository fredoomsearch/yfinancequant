from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class PlannerStep(BaseModel):
    tool: str
    purpose: str = ""
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AssistantPlan(BaseModel):
    version: Literal["v1"] = "v1"
    intent: str = "unknown"
    route: Dict[str, Any] = Field(default_factory=dict)
    steps: List[PlannerStep] = Field(default_factory=list)
    grounded: bool = True
    risk: Literal["low", "medium", "high"] = "low"
    explanation: str = ""
    answer_mode: Literal["strict", "interpreted", "exploratory"] = "strict"
    certainty: Literal["confirmed", "inferred", "hypothesis"] = "confirmed"
    context_snapshot: Dict[str, Any] = Field(default_factory=dict)
    source_mode: Literal["local", "web", "mixed"] = "local"
