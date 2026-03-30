from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from adaptive.apply import persist_promotion_application
from adaptive.promote import persist_promotion_execution
from adaptive.review import persist_promotion_review
from ops.health import build_release_gate_report, build_run_verification_report
from ops.soak import persist_soak_gate
from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import PipelineRequest
from tests.test_adaptive_review import (
    _SilentReviewer,
    _build_fake_cleaning,
    _build_fake_extraction,
    _build_fake_modeling,
)
from tests.test_soak_gate import _set_observability_window


class ReleaseLifecycleTest(unittest.TestCase):
    def test_release_gate_marks_pending_apply_for_prepared_candidate(self) -> None:
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
            persist_promotion_application(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                operator="release-operator",
                notes="prepare candidate package",
            )
            artifact_root = tmp_path / "artifacts"
            _set_observability_window(
                artifact_root,
                result.run_id,
                start="2026-03-20T00:00:00+00:00",
                end="2026-03-23T01:00:00+00:00",
            )
            persist_soak_gate(str(artifact_root), result.run_id, required_hours=72, limit=10)

            release_gate = build_release_gate_report(str(artifact_root), result.run_id)
            verify_report = build_run_verification_report(str(artifact_root), result.run_id)

            self.assertFalse(release_gate.ok)
            self.assertEqual(release_gate.status, "pending_apply")
            self.assertEqual(verify_report["adaptive"]["promotion_lifecycle_status"], "prepared")

    def test_release_gate_marks_release_applied_after_execution(self) -> None:
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
            persist_promotion_application(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                operator="release-operator",
                notes="prepare candidate package",
            )
            persist_promotion_execution(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                operator="prod-operator",
                notes="apply package",
            )
            artifact_root = tmp_path / "artifacts"
            _set_observability_window(
                artifact_root,
                result.run_id,
                start="2026-03-20T00:00:00+00:00",
                end="2026-03-23T01:00:00+00:00",
            )
            persist_soak_gate(str(artifact_root), result.run_id, required_hours=72, limit=10)

            release_gate = build_release_gate_report(str(artifact_root), result.run_id)
            verify_report = build_run_verification_report(str(artifact_root), result.run_id)

            self.assertTrue(release_gate.ok)
            self.assertEqual(release_gate.status, "release_applied")
            self.assertEqual(verify_report["adaptive"]["promotion_lifecycle_status"], "applied")
