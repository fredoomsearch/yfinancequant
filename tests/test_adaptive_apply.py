from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from adaptive.apply import persist_promotion_application
from adaptive.review import persist_promotion_review
from ops.health import build_run_verification_report
from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import PipelineRequest
from tests.test_adaptive_review import (
    _SilentReviewer,
    _build_fake_cleaning,
    _build_fake_extraction,
    _build_fake_modeling,
)


class AdaptivePromotionApplicationTest(unittest.TestCase):
    def test_can_prepare_versioned_candidate_config_after_manual_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request = PipelineRequest(
                tickers=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 20),
                interval="1d",
                review_mode="off",
                use_reviewer=False,
            )
            orchestrator = PipelineOrchestrator(
                artifact_root=str(tmp_path / "artifacts"),
                reviewer=_SilentReviewer(),
            )
            extraction = _build_fake_extraction(tmp_path)
            cleaning = _build_fake_cleaning(tmp_path)
            modeling = _build_fake_modeling(tmp_path)
            modeling.disagreement = True
            modeling.selected_model = "ensemble"

            with patch("pipeline.orchestrator.ExtractionAgent.run", return_value=extraction), \
                 patch("pipeline.orchestrator.CleaningAgent.run", return_value=cleaning), \
                 patch("pipeline.orchestrator.ModelingAgent.run", return_value=modeling):
                result = orchestrator.run(request)

            persist_promotion_review(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                reviewer="ops-lead",
                decision="approve",
                notes="manual signoff",
            )
            application = persist_promotion_application(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                operator="release-operator",
                notes="prepare candidate package",
            )
            verify_report = build_run_verification_report(str(tmp_path / "artifacts"), result.run_id)

            self.assertEqual(application.status, "prepared")
            self.assertTrue(application.config_version.startswith(f"{result.run_id}-v"))
            self.assertTrue(Path(application.artifact.path).exists())
            self.assertTrue(Path(application.config_artifact.path).exists())
            self.assertEqual(verify_report["adaptive"]["promotion_application_status"], "prepared")
            self.assertEqual(verify_report["adaptive"]["promotion_application_operator"], "release-operator")
            self.assertEqual(verify_report["adaptive"]["promotion_application_version"], application.config_version)
            self.assertTrue(verify_report["artifacts"]["adaptive_candidate_config"]["exists"])
            self.assertTrue(verify_report["artifacts"]["adaptive_promotion_application"]["exists"])

    def test_apply_is_blocked_without_manual_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request = PipelineRequest(
                tickers=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 20),
                interval="1d",
                review_mode="off",
                use_reviewer=False,
            )
            orchestrator = PipelineOrchestrator(
                artifact_root=str(tmp_path / "artifacts"),
                reviewer=_SilentReviewer(),
            )
            extraction = _build_fake_extraction(tmp_path)
            cleaning = _build_fake_cleaning(tmp_path)
            modeling = _build_fake_modeling(tmp_path)
            modeling.disagreement = True
            modeling.selected_model = "ensemble"

            with patch("pipeline.orchestrator.ExtractionAgent.run", return_value=extraction), \
                 patch("pipeline.orchestrator.CleaningAgent.run", return_value=cleaning), \
                 patch("pipeline.orchestrator.ModelingAgent.run", return_value=modeling):
                result = orchestrator.run(request)

            with self.assertRaisesRegex(ValueError, "manually approved"):
                persist_promotion_application(
                    artifact_root=str(tmp_path / "artifacts"),
                    run_id=result.run_id,
                    operator="release-operator",
                )
