from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from ops.health import build_run_verification_report
from ops.scheduler import persist_operations_schedule, run_due_operations_schedule
from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import PipelineRequest
from tests.test_adaptive_review import (
    _SilentReviewer,
    _build_fake_cleaning,
    _build_fake_extraction,
    _build_fake_modeling,
)


class OperationsScheduleTest(unittest.TestCase):
    def test_schedule_can_be_persisted_and_run_when_due(self) -> None:
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

            schedule = persist_operations_schedule(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                interval_seconds=300,
                limit=10,
                required_hours=72,
                include_soak=True,
                enabled=True,
                start_immediately=True,
            )
            schedule_run = run_due_operations_schedule(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                force=False,
            )
            verify_report = build_run_verification_report(str(tmp_path / "artifacts"), result.run_id)

            self.assertTrue(Path(schedule.artifact.path).exists())
            self.assertTrue(schedule_run.executed)
            self.assertTrue(schedule_run.job is not None)
            self.assertEqual(schedule_run.job.run_id, result.run_id)
            self.assertTrue(Path(schedule_run.artifact.path).exists())
            self.assertTrue(verify_report["artifacts"]["operations_schedule"]["exists"])
            self.assertTrue(verify_report["artifacts"]["operations_schedule_run"]["exists"])
            self.assertTrue(verify_report["operations"]["schedule_enabled"])
            self.assertTrue(bool(verify_report["operations"]["schedule_next_run_at"]))
            self.assertTrue(bool(verify_report["operations"]["schedule_last_run_at"]))
            self.assertTrue(verify_report["operations"]["schedule_last_execution_status"])

    def test_schedule_run_skips_when_not_due(self) -> None:
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
                start_immediately=False,
            )
            schedule_run = run_due_operations_schedule(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                force=False,
            )

            self.assertFalse(schedule_run.executed)
            self.assertIsNone(schedule_run.job)
            self.assertIn("not due", schedule_run.reason)
