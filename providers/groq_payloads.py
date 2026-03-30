from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def compact_for_groq(
    value: Any,
    *,
    max_depth: int = 2,
    max_items: int = 8,
    max_string: int = 400,
    _depth: int = 0,
) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            value = str(value)
    elif is_dataclass(value):
        try:
            value = asdict(value)
        except Exception:
            value = str(value)

    if isinstance(value, str):
        text = " ".join(value.split())
        if len(text) <= max_string:
            return text
        return f"{text[:max_string].rstrip()}…"

    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        items = list(value.items())
        for key, item in items[:max_items]:
            compacted[str(key)] = compact_for_groq(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                _depth=_depth + 1,
            )
        if len(items) > max_items:
            compacted["_truncated_keys"] = len(items) - max_items
        return compacted

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        compacted_list = [
            compact_for_groq(
                item,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
                _depth=_depth + 1,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            compacted_list.append(f"... {len(items) - max_items} more")
        return compacted_list

    return str(value)
