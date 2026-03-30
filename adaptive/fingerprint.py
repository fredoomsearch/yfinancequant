from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from schemas.pipeline import ArtifactRef, RuntimeFingerprint


class RuntimeFingerprintBuilder:
    def build(self, *, request, modeling, cleaning, run_dir: Path) -> RuntimeFingerprint:
        request_snapshot = {
            "tickers": list(request.tickers),
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "interval": request.interval,
            "model_choice": request.model_choice.value,
            "review_mode": request.review_mode,
            "compare_binance": bool(request.compare_binance),
            "experimental_groq_brain": bool(request.experimental_groq_brain),
        }
        feature_count = len(cleaning.feature_columns or [])
        raw = json.dumps(
            {
                "request": request_snapshot,
                "selected_model": modeling.selected_model,
                "feature_count": feature_count,
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
            },
            sort_keys=True,
        ).encode("utf-8")
        fingerprint_id = hashlib.sha256(raw).hexdigest()[:16]

        adaptive_dir = run_dir / "adaptive"
        adaptive_dir.mkdir(parents=True, exist_ok=True)
        path = adaptive_dir / "runtime_fingerprint.json"
        payload = {
            "version": "v1",
            "fingerprint_id": fingerprint_id,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "request": request_snapshot,
            "selected_model": modeling.selected_model,
            "feature_count": feature_count,
        }
        path.write_text(json.dumps(payload, indent=2))
        artifact = ArtifactRef(
            name=path.name,
            path=str(path),
            kind="runtime_fingerprint",
            size_bytes=path.stat().st_size,
        )
        return RuntimeFingerprint(
            version="v1",
            fingerprint_id=fingerprint_id,
            python_version=sys.version.split()[0],
            platform=platform.platform(),
            request=request_snapshot,
            selected_model=modeling.selected_model,
            feature_count=feature_count,
            artifact=artifact,
        )

