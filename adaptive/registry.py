from __future__ import annotations

import json
from pathlib import Path

from schemas.pipeline import ArtifactRef, FeatureRegistrySnapshot


class FeatureRegistry:
    def __init__(self) -> None:
        self.approved_features = [
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "return_1d",
            "range_pct",
            "body_pct",
            "sma_5",
            "sma_10",
            "volatility_5",
            "volume_sma_5",
            "ticker",
        ]

    def snapshot(self, *, cleaning, run_dir: Path) -> FeatureRegistrySnapshot:
        observed = list(cleaning.feature_columns or [])
        approved = list(self.approved_features)
        missing = [feature for feature in approved if feature not in observed]
        extra = [feature for feature in observed if feature not in approved]
        approved_pct = round(((len(approved) - len(missing)) / len(approved) * 100.0), 2) if approved else 100.0

        adaptive_dir = run_dir / "adaptive"
        adaptive_dir.mkdir(parents=True, exist_ok=True)
        path = adaptive_dir / "feature_registry.json"
        payload = {
            "version": "v1",
            "approved_features": approved,
            "observed_features": observed,
            "missing_features": missing,
            "extra_features": extra,
            "approved_pct": approved_pct,
        }
        path.write_text(json.dumps(payload, indent=2))
        artifact = ArtifactRef(
            name=path.name,
            path=str(path),
            kind="feature_registry",
            size_bytes=path.stat().st_size,
        )
        return FeatureRegistrySnapshot(
            version="v1",
            approved_features=approved,
            observed_features=observed,
            missing_features=missing,
            extra_features=extra,
            approved_pct=approved_pct,
            artifact=artifact,
        )

