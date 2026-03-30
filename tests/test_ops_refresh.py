from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from adaptive.apply import persist_promotion_application
from adaptive.review import persist_promotion_review
from ops.refresh import persist_operations_refresh
from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import PipelineRequest
from tests.test_adaptive_review import (
    _SilentReviewer,
    _build_fake_cleaning,
    _build_fake_extraction,
    _build_fake_modeling,
)


class OperationsRefreshTest(unittest.TestCase):
    def test_refresh_persists_operational_bundle_and_exposes_adaptive_statuses(self) -> None:
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

            refresh = persist_operations_refresh(
                artifact_root=str(tmp_path / "artifacts"),
                run_id=result.run_id,
                include_soak=True,
                limit=10,
                required_hours=72,
            )

            self.assertEqual(refresh.run_id, result.run_id)
            self.assertIn("readyz", refresh.artifacts)
            self.assertIn("release_gate", refresh.artifacts)
            self.assertIn("release_board", refresh.artifacts)
            self.assertIn("json", refresh.dashboard.artifacts)
            self.assertTrue(Path(refresh.artifacts["readyz"].path).exists())
            self.assertTrue(Path(refresh.artifacts["release_gate"].path).exists())
            self.assertTrue(Path(refresh.artifacts["release_board"].path).exists())
            self.assertTrue(Path(refresh.dashboard.artifacts["json"].path).exists())
            self.assertIn("soak_gate", refresh.soak)
            self.assertEqual(refresh.release_board.entries[0].manual_review_status, "approved")
            self.assertEqual(refresh.release_board.entries[0].promotion_application_status, "prepared")
            self.assertTrue(
                str(refresh.release_board.entries[0].promotion_application_version).startswith(f"{result.run_id}-v")
            )
