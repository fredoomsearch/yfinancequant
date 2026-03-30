from __future__ import annotations

import json
from pathlib import Path

from schemas.pipeline import ArtifactRef, RetrainingPlan


class RetrainingScheduler:
    def plan(self, *, request, drift, selection, validation, run_dir: Path) -> RetrainingPlan:
        status = "monitor"
        within_hours = None
        reasons: list[str] = []

        if validation.status == "candidate_ready":
            status = "scheduled"
            within_hours = 24
            reasons.append("Candidate-ready runs should schedule a challenger retraining window within 24 hours.")
        elif drift.level == "drifted":
            status = "immediate"
            within_hours = 6
            reasons.append("Severe drift requests immediate retraining preparation.")
        elif drift.level == "watch":
            status = "scheduled"
            within_hours = 72
            reasons.append("Watch-level drift should be revisited with a scheduled retraining cycle.")
        else:
            reasons.append("Current run remains under monitor status with no retraining pressure.")

        adaptive_dir = run_dir / "adaptive"
        adaptive_dir.mkdir(parents=True, exist_ok=True)
        path = adaptive_dir / "retraining_plan.json"
        payload = {
            "status": status,
            "recommended_within_hours": within_hours,
            "candidate_model": selection.recommended_model,
            "recommended_review_mode": selection.recommended_review_mode,
            "recommended_confidence_threshold": selection.recommended_confidence_threshold,
            "reasons": reasons,
        }
        path.write_text(json.dumps(payload, indent=2))
        artifact = ArtifactRef(
            name=path.name,
            path=str(path),
            kind="retraining_plan",
            size_bytes=path.stat().st_size,
        )
        return RetrainingPlan(
            status=status,
            recommended_within_hours=within_hours,
            candidate_model=selection.recommended_model,
            recommended_review_mode=selection.recommended_review_mode,
            recommended_confidence_threshold=selection.recommended_confidence_threshold,
            reasons=reasons,
            artifact=artifact,
        )

