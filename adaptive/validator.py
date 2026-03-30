from __future__ import annotations

from schemas.pipeline import AdaptiveSelection, AdaptiveValidation, DriftAssessment, ShadowRunReport


class AdaptiveValidator:
    def validate(
        self,
        *,
        drift: DriftAssessment,
        selection: AdaptiveSelection,
        shadow: ShadowRunReport,
    ) -> AdaptiveValidation:
        reasons: list[str] = [
            "Adaptive layer is still fail-safe: no live configuration is auto-applied.",
            "Validation only escalates candidates toward manual promotion review.",
        ]
        status = "observe_only"

        if shadow.executed and shadow.ready_for_promotion:
            status = "candidate_ready"
            reasons.append("Shadow benchmark marked the challenger as promotion-ready for manual review.")
        elif drift.level == "watch":
            status = "shadow_only"
            reasons.append("Shadow validation is recommended before changing the live decision path.")
        elif drift.level == "drifted":
            status = "shadow_only"
            reasons.append("Drifted runs require challenger validation before promotion.")

        if selection.recommended_action == "keep_current_strategy":
            reasons.append("No adaptive intervention is required for the current run.")

        if shadow.executed and not shadow.ready_for_promotion:
            reasons.append("The challenger remains below the promotion-ready threshold.")

        return AdaptiveValidation(passed=True, status=status, reasons=reasons)
