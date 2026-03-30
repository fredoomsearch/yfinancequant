from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

import requests

from schemas.pipeline import StageBrief
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


class GroqAdvisor:
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

    def brief(self, stage: str, payload: Dict[str, Any]) -> Optional[StageBrief]:
        if not self.enabled:
            return None

        request_payload = {
            "stage": stage,
            "payload": payload,
        }
        stage_motor = payload.get("motor") if isinstance(payload, dict) else None
        payload_data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You produce a concise stage brief for one assistant with three skills: extract, clean, model. "
                        "Return only JSON with keys: stage, motor, summary, key_points, risks. "
                        "The motor should name the active engine, model family, or data source that dominates the stage. "
                        "Do not invent facts. Keep the reply grounded in the provided payload. "
                        "Write in simple, human language with short sentences. Avoid jargon when a plain word works. "
                        "Do not sound like a checklist unless the payload clearly requires a list. "
                        "Prefer a conversational summary that reads like one teammate talking to another. "
                        "Keep symbol names, ticker names, mode names, and command tokens unchanged so the text is easy to translate. "
                        "If the payload contains a language field, answer in that language."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(compact_for_groq(request_payload, max_depth=2, max_items=8, max_string=300), separators=(",", ":"), default=str),
                },
            ],
            "temperature": 0.1,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload_data, timeout=self.timeout)
            response.raise_for_status()
            payload_json: Dict[str, Any] = response.json()
            content = payload_json["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            if not parsed:
                parsed = {
                    "stage": stage,
                    "motor": stage_motor or stage,
                    "summary": content,
                    "key_points": [],
                    "risks": [],
                }
            return StageBrief(
                stage=str(parsed.get("stage") or stage),
                provider="groq",
                enabled=True,
                motor=str(parsed.get("motor") or stage_motor or stage),
                summary=str(parsed.get("summary") or "").strip(),
                key_points=[str(item) for item in parsed.get("key_points", []) if str(item).strip()],
                risks=[str(item) for item in parsed.get("risks", []) if str(item).strip()],
                raw_response=payload_json,
            )
        except Exception as exc:
            logger.debug("Groq stage brief unavailable for %s: %s", stage, exc)
            return None
