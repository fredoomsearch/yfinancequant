from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adaptive.policy import AdaptivePolicyEngine


class AdaptivePolicyEngineTest(unittest.TestCase):
    def test_policy_stays_observe_only_for_stable_run(self) -> None:
        engine = AdaptivePolicyEngine()
        request = SimpleNamespace(confidence_threshold=0.60, review_mode="off")
        drift = SimpleNamespace(level="stable")
        selection = SimpleNamespace(
            mode="observe_only",
            recommended_action="keep_current_strategy",
            recommended_model="ensemble",
            recommended_confidence_threshold=0.60,
            recommended_review_mode="off",
        )
        feature_registry = SimpleNamespace(approved_pct=100.0)
        shadow = SimpleNamespace(executed=False, ready_for_promotion=False)
        validation = SimpleNamespace(status="observe_only")
        retraining = SimpleNamespace(status="monitor", recommended_within_hours=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            approval = engine.evaluate(
                request=request,
                drift=drift,
                selection=selection,
                feature_registry=feature_registry,
                shadow=shadow,
                validation=validation,
                retraining=retraining,
                run_dir=Path(tmpdir),
            )

            self.assertTrue(Path(approval.artifact.path).exists())

        self.assertEqual(approval.status, "observe_only")
        self.assertFalse(approval.auto_apply_allowed)
        self.assertTrue(approval.requires_manual_signoff)
        self.assertTrue(approval.checks["approved_feature_catalog"])

    def test_policy_approves_candidate_for_manual_promotion_review(self) -> None:
        engine = AdaptivePolicyEngine()
        request = SimpleNamespace(confidence_threshold=0.60, review_mode="off")
        drift = SimpleNamespace(level="watch")
        selection = SimpleNamespace(
            mode="observe_only",
            recommended_action="prepare_retraining_candidate",
            recommended_model="majority",
            recommended_confidence_threshold=0.66,
            recommended_review_mode="on",
        )
        feature_registry = SimpleNamespace(approved_pct=100.0)
        shadow = SimpleNamespace(executed=True, ready_for_promotion=True)
        validation = SimpleNamespace(status="candidate_ready")
        retraining = SimpleNamespace(status="scheduled", recommended_within_hours=24)

        with tempfile.TemporaryDirectory() as tmpdir:
            approval = engine.evaluate(
                request=request,
                drift=drift,
                selection=selection,
                feature_registry=feature_registry,
                shadow=shadow,
                validation=validation,
                retraining=retraining,
                run_dir=Path(tmpdir),
            )

        self.assertEqual(approval.status, "approved_for_promotion_review")
        self.assertTrue(approval.checks["review_mode_hardened_if_escalated"])
        self.assertTrue(approval.checks["shadow_completed_for_candidate"])
        self.assertEqual(approval.proposed_changes["model"], "majority")

    def test_policy_blocks_candidate_when_feature_catalog_is_not_approved(self) -> None:
        engine = AdaptivePolicyEngine()
        request = SimpleNamespace(confidence_threshold=0.60, review_mode="off")
        drift = SimpleNamespace(level="watch")
        selection = SimpleNamespace(
            mode="observe_only",
            recommended_action="enable_shadow_validation",
            recommended_model="majority",
            recommended_confidence_threshold=0.60,
            recommended_review_mode="on",
        )
        feature_registry = SimpleNamespace(approved_pct=40.0)
        shadow = SimpleNamespace(executed=True, ready_for_promotion=False)
        validation = SimpleNamespace(status="shadow_only")
        retraining = SimpleNamespace(status="scheduled", recommended_within_hours=72)

        with tempfile.TemporaryDirectory() as tmpdir:
            approval = engine.evaluate(
                request=request,
                drift=drift,
                selection=selection,
                feature_registry=feature_registry,
                shadow=shadow,
                validation=validation,
                retraining=retraining,
                run_dir=Path(tmpdir),
            )

        self.assertEqual(approval.status, "blocked")
        self.assertFalse(approval.checks["approved_feature_catalog"])
        self.assertIn("approved catalog", " ".join(approval.reasons).lower())
