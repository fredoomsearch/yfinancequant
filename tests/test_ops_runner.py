from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from ops.health import build_run_verification_report
from ops.job_runner import persist_operations_job
from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import PipelineRequest
from tests.test_adaptive_review import (
    _SilentReviewer,
    _build_fake_cleaning,
    _build_fake_extraction,
    _build_fake_modeling,
)


class OperationsJobRunnerTest(unittest.TestCase):
    def test_job_runner_persists_periodic_refresh_report(self) -> None:
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

            report = persist_operations_job(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                cycles=2,
                interval_seconds=0,
                include_soak=True,
                limit=10,
                required_hours=72,
            )
            verify_report = build_run_verification_report(str(tmp_path / "artifacts"), result.run_id)

            self.assertEqual(report.run_id, result.run_id)
            self.assertEqual(report.cycles_requested, 2)
            self.assertEqual(report.cycles_completed, 2)
            self.assertEqual(len(report.cycles), 2)
            self.assertEqual(report.latest_release_status, "pending_soak")
            self.assertEqual(report.latest_soak_status, "pending")
            self.assertTrue(Path(report.artifact.path).exists())
            self.assertEqual(verify_report["operations"]["job_cycles_completed"], 2)
            self.assertEqual(verify_report["operations"]["job_latest_release_status"], "pending_soak")
            self.assertTrue(verify_report["artifacts"]["operations_job_runner"]["exists"])

    def test_job_runner_validates_cycle_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            persist_operations_job(cycles=0)
