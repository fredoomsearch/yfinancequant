from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from adaptive.apply import persist_promotion_application
from adaptive.promote import persist_promotion_execution
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


class AdaptivePromotionExecutionTest(unittest.TestCase):
    def test_can_mark_prepared_package_as_applied(self) -> None:
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
            execution = persist_promotion_execution(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                operator="prod-operator",
                notes="applied to production baseline",
            )
            verify_report = build_run_verification_report(str(tmp_path / "artifacts"), result.run_id)

            self.assertEqual(execution.status, "applied")
            self.assertEqual(execution.applied_by, "prod-operator")
            self.assertEqual(execution.config_version, application.config_version)
            self.assertTrue(Path(execution.artifact.path).exists())
            self.assertEqual(verify_report["adaptive"]["promotion_execution_status"], "applied")
            self.assertEqual(verify_report["adaptive"]["promotion_execution_operator"], "prod-operator")
            self.assertEqual(verify_report["adaptive"]["promotion_lifecycle_status"], "applied")
            self.assertTrue(verify_report["artifacts"]["adaptive_promotion_execution"]["exists"])

    def test_promote_is_blocked_without_prepared_application(self) -> None:
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

            with self.assertRaisesRegex(ValueError, "prepared"):
                persist_promotion_execution(
                    artifact_root=str(tmp_path / "artifacts"),
                    run_id=result.run_id,
                    operator="prod-operator",
                )
