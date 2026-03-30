from __future__ import annotations

from schemas.pipeline import AdaptiveApproval, AdaptiveSelection, AdaptiveValidation, DriftAssessment, PromotionDecision, ShadowRunReport


class PromotionGate:
    def decide(
        self,
        *,
        drift: DriftAssessment,
        approval: AdaptiveApproval,
        selection: AdaptiveSelection,
        validation: AdaptiveValidation,
        shadow: ShadowRunReport,
    ) -> PromotionDecision:
        reasons = [
            "Adaptive promotions are manual-only in the current architecture phase.",
            "The gate can mark a candidate as ready, but never auto-applies it.",
        ]
        eligible = False
        mode = "manual_only"

        if approval.status == "blocked":
            reasons.append("The adaptive approval policy blocked this candidate before promotion review.")
        elif approval.status == "approved_for_promotion_review" and validation.status == "candidate_ready" and shadow.ready_for_promotion:
            eligible = True
            mode = "promotion_ready"
            reasons.append("The challenger cleared shadow validation and is ready for manual promotion.")
        elif approval.status == "manual_review" or validation.status == "shadow_only":
            mode = "shadow_only"
            reasons.append("A shadow run is required before promotion can be considered.")
        elif drift.level == "stable" and selection.recommended_action == "keep_current_strategy":
            reasons.append("The current strategy remains the preferred production baseline.")

        return PromotionDecision(eligible=eligible, mode=mode, reasons=reasons)
