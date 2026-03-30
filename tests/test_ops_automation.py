from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from ops.automation import persist_operations_automation
from ops.health import build_run_verification_report
from ops.scheduler import persist_operations_schedule
from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import PipelineRequest
from tests.test_adaptive_review import (
    _SilentReviewer,
    _build_fake_cleaning,
    _build_fake_extraction,
    _build_fake_modeling,
)


class OperationsAutomationTest(unittest.TestCase):
    def test_can_materialize_automation_bundle_for_scheduled_run(self) -> None:
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

            with patch("pipeline.orchestrator.ExtractionAgent.run", return_value=extraction), \
                 patch("pipeline.orchestrator.CleaningAgent.run", return_value=cleaning), \
                 patch("pipeline.orchestrator.ModelingAgent.run", return_value=modeling):
                result = orchestrator.run(request)

            persist_operations_schedule(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                interval_seconds=300,
                start_immediately=True,
            )
            automation = persist_operations_automation(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
            )
            verify_report = build_run_verification_report(str(tmp_path / "artifacts"), result.run_id)

            self.assertTrue(Path(automation.artifact.path).exists())
            self.assertTrue(Path(automation.shell_artifact.path).exists())
            self.assertTrue(Path(automation.cron_artifact.path).exists())
            self.assertTrue(Path(automation.systemd_service_artifact.path).exists())
            self.assertTrue(Path(automation.systemd_timer_artifact.path).exists())
            self.assertTrue(verify_report["artifacts"]["operations_automation_bundle"]["exists"])
            self.assertTrue(verify_report["artifacts"]["operations_schedule_shell"]["exists"])
            self.assertTrue(verify_report["artifacts"]["operations_schedule_cron"]["exists"])
            self.assertTrue(verify_report["artifacts"]["operations_schedule_service"]["exists"])
            self.assertTrue(verify_report["artifacts"]["operations_schedule_timer"]["exists"])

    def test_automation_is_blocked_without_schedule(self) -> None:
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

            with patch("pipeline.orchestrator.ExtractionAgent.run", return_value=extraction), \
                 patch("pipeline.orchestrator.CleaningAgent.run", return_value=cleaning), \
                 patch("pipeline.orchestrator.ModelingAgent.run", return_value=modeling):
                result = orchestrator.run(request)

            with self.assertRaisesRegex(ValueError, "operations schedule"):
                persist_operations_automation(
                    artifact_root=str(tmp_path / "artifacts"),
                    run_id=result.run_id,
                )
