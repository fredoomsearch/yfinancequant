from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from assistant.contracts import AssistantState


def assistant_dir(artifact_root: str) -> Path:
    return Path(artifact_root) / "assistant"


def session_path(artifact_root: str, session_id: str) -> Path:
    return assistant_dir(artifact_root) / f"{session_id}.json"


def load_state(artifact_root: str, session_id: str = "default") -> AssistantState:
    path = session_path(artifact_root, session_id)
    if not path.exists():
        return AssistantState(session_id=session_id)
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return AssistantState(session_id=session_id)
    state = AssistantState.from_dict(raw)
    if not state.session_id:
        state.session_id = session_id
    return state


def save_state(artifact_root: str, state: AssistantState) -> Path:
    path = session_path(artifact_root, state.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, Any] = state.to_dict()
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
