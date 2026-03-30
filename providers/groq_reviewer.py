from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

import requests

from schemas.pipeline import ReviewerPacket, ReviewerResult
from providers.groq_payloads import compact_for_groq

logger = logging.getLogger(__name__)


def _extract_json(content: str) -> Optional[dict]:
    if not content:
        return None
    content = content.strip()
    if content.startswith("{") and content.endswith("}"):
        try:
            return json.loads(content)
        except Exception:
            pass
    match = re.search(r"\{.*\}", content, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


class GroqReviewer:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 45,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.base_url = base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def review(self, packet: ReviewerPacket) -> Optional[ReviewerResult]:
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You review a quant pipeline summary. Return only JSON with keys: "
                        "decision, confidence, explanation, risks. "
                        "Write the explanation in simple, human language with short sentences. "
                        "Do not sound like a checklist unless the facts require one. "
                        "Prefer a conversational explanation that sounds like one teammate briefing another. "
                        "If the packet contains a language field, answer in that language. "
                        "Avoid jargon unless it is necessary to keep the meaning precise."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(compact_for_groq(packet.model_dump(), max_depth=2, max_items=8, max_string=300), separators=(",", ":"), default=str),
                },
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            payload_json: Dict[str, Any] = response.json()
            content = payload_json["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            if not parsed:
                logger.debug("Groq returned a non-JSON payload; falling back to raw text")
                parsed = {
                    "decision": packet.candidate_decision,
                    "confidence": packet.confidence,
                    "explanation": content,
                    "risks": [],
                }
            return ReviewerResult(
                provider="groq",
                decision=str(parsed.get("decision", packet.candidate_decision)),
                confidence=float(parsed.get("confidence", packet.confidence)),
                explanation=str(parsed.get("explanation", "")),
                risks=[str(r) for r in parsed.get("risks", [])],
                raw_response=payload_json,
            )
        except Exception as exc:
            logger.debug("Groq review unavailable: %s", exc)
            return None
