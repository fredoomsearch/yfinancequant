from __future__ import annotations

import json
from pathlib import Path

from schemas.pipeline import ArtifactRef, ShadowRunReport


class ShadowRunner:
    def run(self, *, request, modeling, drift, selection, run_dir: Path) -> ShadowRunReport:
        if selection.recommended_action == "keep_current_strategy":
            return ShadowRunReport(
                executed=False,
                status="not_needed",
                candidate_model=selection.recommended_model,
                ready_for_promotion=False,
                reasons=["Current strategy remains the production baseline; no challenger shadow run is required."],
            )

        baseline_score = float(max(modeling.ensemble_probability, 1 - modeling.ensemble_probability))
        candidate_score = baseline_score
        reasons: list[str] = [
            "Shadow run benchmarks a candidate configuration without changing the live decision path.",
        ]

        if selection.recommended_model == "majority" and modeling.disagreement:
            candidate_score += 0.03
            reasons.append("Majority is favored during disagreement for the challenger benchmark.")
        if selection.recommended_review_mode == "on" and request.review_mode != "on":
            candidate_score += 0.02
            reasons.append("Reviewer-on candidate improves governance readiness for the benchmark.")
        if (selection.recommended_confidence_threshold or request.confidence_threshold) > request.confidence_threshold:
            candidate_score += 0.01
            reasons.append("A stricter threshold slightly increases challenger readiness.")
        if drift.level == "drifted":
            candidate_score -= 0.03
            reasons.append("Severe drift keeps the challenger below promotion-ready status.")

        candidate_score = round(max(0.0, min(candidate_score, 1.0)), 4)
        improvement = round(candidate_score - baseline_score, 4)
        ready_for_promotion = drift.level == "watch" and improvement >= 0.03
        if ready_for_promotion:
            reasons.append("The challenger benchmark cleared the promotion-ready threshold for manual review.")

        adaptive_dir = run_dir / "adaptive"
        adaptive_dir.mkdir(parents=True, exist_ok=True)
        report_path = adaptive_dir / "shadow_run.json"
        payload = {
            "executed": True,
            "status": "completed",
            "candidate_model": selection.recommended_model,
            "baseline_score": round(baseline_score, 4),
            "candidate_score": candidate_score,
            "improvement": improvement,
            "ready_for_promotion": ready_for_promotion,
            "reasons": reasons,
        }
        report_path.write_text(json.dumps(payload, indent=2))
        artifact = ArtifactRef(
            name=report_path.name,
            path=str(report_path),
            kind="shadow_run",
            size_bytes=report_path.stat().st_size,
        )
        return ShadowRunReport(
            executed=True,
            status="completed",
            candidate_model=selection.recommended_model,
            baseline_score=round(baseline_score, 4),
            candidate_score=candidate_score,
            improvement=improvement,
            ready_for_promotion=ready_for_promotion,
            reasons=reasons,
            artifact=artifact,
        )
