from __future__ import annotations

from schemas.pipeline import AdaptiveSelection, DriftAssessment


class AdaptiveSelector:
    def select(self, *, request, modeling, drift: DriftAssessment) -> AdaptiveSelection:
        recommended_model = modeling.selected_model
        recommended_threshold = float(request.confidence_threshold)
        recommended_review_mode = request.review_mode
        recommended_action = "keep_current_strategy"
        reasons: list[str] = []

        if drift.level == "stable":
            reasons.append("Current run is stable enough to keep the active deterministic strategy.")
        elif drift.level == "watch":
            recommended_action = "enable_shadow_validation"
            recommended_review_mode = "on" if request.review_mode == "off" else request.review_mode
            reasons.append("The run should be observed with stricter review before any adaptive change.")
        else:
            recommended_action = "prepare_retraining_candidate"
            recommended_threshold = max(recommended_threshold, 0.65)
            recommended_review_mode = "on"
            reasons.append("Drift indicators suggest preparing a challenger configuration in shadow mode.")

        if modeling.disagreement and recommended_model == "ensemble":
            recommended_model = "majority"
            reasons.append("Majority is the safer candidate while disagreement remains active.")

        return AdaptiveSelection(
            mode="observe_only",
            recommended_action=recommended_action,
            recommended_model=recommended_model,
            recommended_confidence_threshold=round(recommended_threshold, 4),
            recommended_review_mode=recommended_review_mode,
            reasons=reasons,
        )

