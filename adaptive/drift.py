from __future__ import annotations

from typing import Any, Dict, Optional

from schemas.pipeline import DriftAssessment


class DriftDetector:
    def assess(
        self,
        *,
        request,
        extraction,
        cleaning,
        modeling,
        source_comparison: Optional[Any] = None,
        legacy_analysis: Optional[Dict[str, Any]] = None,
    ) -> DriftAssessment:
        score = 0.0
        reasons: list[str] = []

        confidence = float(max(modeling.ensemble_probability, 1 - modeling.ensemble_probability))
        retention = (cleaning.rows_out / extraction.rows) if extraction.rows else 0.0

        if modeling.disagreement:
            score += 0.45
            reasons.append("Local models disagreed on the latest direction.")
        if confidence < float(request.confidence_threshold):
            score += 0.25
            reasons.append("Deterministic confidence is below the active threshold.")
        if retention < 0.90:
            score += 0.15
            reasons.append("Cleaning retained less than 90% of extracted rows.")

        if source_comparison and getattr(source_comparison, "enabled", False):
            coverage = getattr(source_comparison, "coverage", {}) or {}
            overlap_rows = int(coverage.get("overlap_rows") or 0)
            if overlap_rows == 0:
                score += 0.15
                reasons.append("Cross-source overlap is missing for the requested comparison window.")

        if legacy_analysis and legacy_analysis.get("enabled") and legacy_analysis.get("modeling_health") == "mixed":
            score += 0.10
            reasons.append("Legacy bridge reported mixed modeling health.")

        if score >= 0.55:
            level = "drifted"
        elif score >= 0.25:
            level = "watch"
        else:
            level = "stable"

        if not reasons:
            reasons.append("Signals are stable under the current confidence and retention checks.")

        return DriftAssessment(level=level, score=round(min(score, 1.0), 4), reasons=reasons)

