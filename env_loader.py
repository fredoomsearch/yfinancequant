from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Tuple


_PROJECT_ROOT = Path(__file__).resolve().parent
_INLINE_COMMENT_RE = re.compile(r"\s+#")


def _parse_env_line(line: str) -> Optional[Tuple[str, str]]:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.startswith("export "):
        raw = raw[len("export ") :].lstrip()
    if "=" not in raw:
        return None
    key, value = raw.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {"'", '"'}:
        value = value[1:-1]
    else:
        value = _INLINE_COMMENT_RE.split(value, maxsplit=1)[0].rstrip()
    return key, value


def load_project_env(path: str | Path | None = None, *, override: bool = False) -> bool:
    env_path = Path(path) if path is not None else _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False

    try:
        contents = env_path.read_text(encoding="utf-8")
    except OSError:
        return False

    loaded = False
    for line in contents.splitlines():
        parsed = _parse_env_line(line)
        if not parsed:
            continue
        key, value = parsed
        if not override and key in os.environ:
            continue
        os.environ[key] = value
        loaded = True
    return loaded
