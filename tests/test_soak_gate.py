from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from ops.health import build_release_gate_report, build_run_verification_report
from ops.soak import persist_soak_gate
from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import (
    ArtifactRef,
    CleaningResult,
    CleanMarketData,
    ExtractionResult,
    ModelResult,
    ModelingResult,
    PipelineRequest,
    RawMarketData,
)


class _SilentReviewer:
    def review(self, packet):
        return None


def _build_fake_extraction(tmp_path: Path) -> ExtractionResult:
    raw_path = tmp_path / "raw_market_data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "ticker": ["AAPL"] * 12,
            "open": [100.0 + index for index in range(12)],
            "high": [101.0 + index for index in range(12)],
            "low": [99.0 + index for index in range(12)],
            "close": [100.5 + index for index in range(12)],
            "adj_close": [100.4 + index for index in range(12)],
            "volume": [1000 + index * 10 for index in range(12)],
        }
    ).to_csv(raw_path, index=False)
    return ExtractionResult(
        rows=12,
        tickers=["AAPL"],
        missing_columns=[],
        raw_data=RawMarketData(
            ticker="AAPL",
            start=date(2024, 1, 1),
            end=date(2024, 1, 20),
            interval="1d",
            rows=12,
            columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
            missing_columns=[],
            artifact=ArtifactRef(
                name=raw_path.name,
                path=str(raw_path),
                kind="raw_data",
                size_bytes=raw_path.stat().st_size,
            ),
        ),
    )


def _build_fake_cleaning(tmp_path: Path) -> CleaningResult:
    cleaned_path = tmp_path / "clean_market_data.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=11, freq="D"),
            "ticker": ["AAPL"] * 11,
            "open": [100.0 + index for index in range(11)],
            "high": [101.0 + index for index in range(11)],
            "low": [99.0 + index for index in range(11)],
            "close": [100.5 + index for index in range(11)],
            "adj_close": [100.4 + index for index in range(11)],
            "volume": [1000 + index * 10 for index in range(11)],
            "return_1d": [0.01] * 11,
            "range_pct": [0.02] * 11,
            "body_pct": [0.01] * 11,
            "sma_5": [100.0] * 11,
            "sma_10": [100.0] * 11,
            "volatility_5": [0.1] * 11,
            "volume_sma_5": [1000.0] * 11,
            "future_close": [101.0] * 11,
            "future_return": [0.01] * 11,
            "target_direction": [1.0] * 10 + [float("nan")],
        }
    ).to_csv(cleaned_path, index=False)
    feature_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "return_1d",
        "range_pct",
        "body_pct",
        "sma_5",
        "sma_10",
        "volatility_5",
        "volume_sma_5",
        "ticker",
    ]
    return CleaningResult(
        rows_in=12,
        rows_out=11,
        feature_columns=feature_columns,
        target_column="target_direction",
        clean_data=CleanMarketData(
            rows_in=12,
            rows_out=11,
            feature_columns=feature_columns,
            target_column="target_direction",
            artifact=ArtifactRef(
                name=cleaned_path.name,
                path=str(cleaned_path),
                kind="clean_data",
                size_bytes=cleaned_path.stat().st_size,
            ),
            quality_report={},
        ),
    )


def _build_fake_modeling(tmp_path: Path) -> ModelingResult:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for model_name, probability in (
        ("logistic_regression", 0.71),
        ("random_forest", 0.62),
        ("gradient_boosting", 0.66),
    ):
        artifact_path = models_dir / f"{model_name}.joblib"
        artifact_path.write_bytes(b"fake-model")
        model_results.append(
            ModelResult(
                model_name=model_name,
                validation_metrics={"accuracy": 0.75},
                latest_probability=probability,
                latest_prediction="long",
                confidence=probability,
                artifact=ArtifactRef(
                    name=artifact_path.name,
                    path=str(artifact_path),
                    kind="model",
                    size_bytes=artifact_path.stat().st_size,
                ),
            )
        )
    summary_path = models_dir / "model_summary.json"
    summary_path.write_text("{\"status\": \"ok\"}")
    return ModelingResult(
        models=model_results,
        ensemble_probability=0.6633333333,
        ensemble_prediction="long",
        majority_prediction="long",
        disagreement=False,
        selected_model="ensemble",
        rationale="Synthetic modeling result for soak coverage.",
        latest_sample_count=1,
        artifact=ArtifactRef(
            name=summary_path.name,
            path=str(summary_path),
            kind="model_summary",
            size_bytes=summary_path.stat().st_size,
        ),
    )


def _set_observability_window(artifact_root: Path, run_id: str, *, start: str, end: str) -> None:
    path = artifact_root / "runs" / run_id / "ops" / "runtime_observability.json"
    payload = json.loads(path.read_text())
    payload["run_started_at"] = start
    payload["run_finished_at"] = end
    payload["run_duration_ms"] = 1000
    path.write_text(json.dumps(payload, indent=2))


class SoakGateTest(unittest.TestCase):
    def test_persist_soak_gate_marks_pending_when_window_is_too_short(self) -> None:
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

            artifact_root = tmp_path / "artifacts"
            _set_observability_window(
                artifact_root,
                result.run_id,
                start="2026-03-20T00:00:00+00:00",
                end="2026-03-20T01:00:00+00:00",
            )

            payload = persist_soak_gate(str(artifact_root), result.run_id, required_hours=72, limit=10)
            release_gate = build_release_gate_report(str(artifact_root), result.run_id)
            verify_report = build_run_verification_report(str(artifact_root), result.run_id)

            self.assertEqual(payload["soak_gate"]["status"], "pending")
            self.assertFalse(payload["soak_gate"]["ok"])
            self.assertEqual(payload["release_summary"]["release_stage"], "ops_pending")
            self.assertEqual(release_gate.status, "pending_soak")
            self.assertEqual(verify_report["operations"]["soak_status"], "pending")

    def test_persist_soak_gate_marks_release_ready_after_72h_window(self) -> None:
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
                first = orchestrator.run(request)
                second = orchestrator.run(request)

            artifact_root = tmp_path / "artifacts"
            _set_observability_window(
                artifact_root,
                first.run_id,
                start="2026-03-20T00:00:00+00:00",
                end="2026-03-20T01:00:00+00:00",
            )
            _set_observability_window(
                artifact_root,
                second.run_id,
                start="2026-03-23T04:00:00+00:00",
                end="2026-03-23T05:00:00+00:00",
            )

            payload = persist_soak_gate(str(artifact_root), second.run_id, required_hours=72, limit=10)
            release_gate = build_release_gate_report(str(artifact_root), second.run_id)
            verify_report = build_run_verification_report(str(artifact_root), second.run_id)

            self.assertEqual(payload["soak_gate"]["status"], "passed")
            self.assertTrue(payload["soak_gate"]["ok"])
            self.assertGreaterEqual(payload["soak_gate"]["observed_hours"], 72.0)
            self.assertEqual(payload["soak_gate"]["sampled_runs"], 2)
            self.assertEqual(payload["soak_gate"]["ready_runs"], 2)
            self.assertTrue(payload["release_summary"]["ok"])
            self.assertEqual(payload["release_summary"]["release_stage"], "release_ready")
            self.assertTrue(release_gate.ok)
            self.assertEqual(release_gate.status, "release_ready")
            self.assertEqual(verify_report["operations"]["soak_status"], "passed")
            self.assertEqual(verify_report["operations"]["release_stage"], "release_ready")
