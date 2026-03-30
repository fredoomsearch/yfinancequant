from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pipeline.orchestrator import PipelineOrchestrator
from schemas.pipeline import (
    ArtifactRef,
    CleaningResult,
    ExtractionResult,
    ModelResult,
    ModelingResult,
    PipelineRequest,
    ReviewerResult,
    RawMarketData,
    CleanMarketData,
    SourceComparison,
    RunState,
)


class _SilentReviewer:
    def review(self, packet):
        return None


class _FixedReviewer:
    def __init__(self, result: ReviewerResult):
        self.result = result

    def review(self, packet):
        return self.result


class _FixedBrain:
    def __init__(self, result: ReviewerResult):
        self.result = result

    def decide(self, packet):
        return self.result


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

    quality_report = {
        "rows_in": 12,
        "rows_out": 11,
        "dropped_duplicates": 0,
        "dropped_missing": 0,
        "feature_columns": [
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
        ],
        "target_column": "target_direction",
    }

    return CleaningResult(
        rows_in=12,
        rows_out=11,
        feature_columns=quality_report["feature_columns"],
        target_column="target_direction",
        clean_data=CleanMarketData(
            rows_in=12,
            rows_out=11,
            feature_columns=quality_report["feature_columns"],
            target_column="target_direction",
            artifact=ArtifactRef(
                name=cleaned_path.name,
                path=str(cleaned_path),
                kind="clean_data",
                size_bytes=cleaned_path.stat().st_size,
            ),
            quality_report=quality_report,
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
                validation_metrics={
                    "accuracy": 0.75,
                    "precision": 0.7,
                    "recall": 0.8,
                    "f1": 0.74,
                    "roc_auc": 0.77,
                },
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
        rationale="Synthetic modeling result for orchestrator regression test.",
        latest_sample_count=1,
        artifact=ArtifactRef(
            name=summary_path.name,
            path=str(summary_path),
            kind="model_summary",
            size_bytes=summary_path.stat().st_size,
        ),
    )


def _build_fake_reviewer_result() -> ReviewerResult:
    return ReviewerResult(
        provider="groq",
        decision="hold",
        confidence=0.55,
        explanation="Synthetic reviewer result for run-mode coverage.",
        risks=[],
        raw_response={"synthetic": True},
    )


def _build_fake_brain_result() -> ReviewerResult:
    return ReviewerResult(
        provider="groq_brain",
        decision="short",
        confidence=0.44,
        explanation="Synthetic Groq brain result for experimental mode coverage.",
        risks=["experimental mode"],
        raw_response={"synthetic": True},
    )


def _build_source_comparison() -> SourceComparison:
    return SourceComparison(
        enabled=True,
        source_1="yfinance",
        source_2="binance",
        asset="BTC",
        timeframe="1d",
        date_range={"start": "2024-01-01", "end": "2025-01-01"},
        coverage={"overlap_rows": 366, "yfinance_only_rows": 0, "binance_only_rows": 1},
        row_counts={"yfinance": 366, "binance": 367, "overlap": 366, "union": 367},
        close_price_alignment={
            "overlap_rows": 366,
            "mae": 41.977292947404585,
            "mape_pct": 0.06092678643789259,
            "correlation": 0.9999912753798597,
        },
        note="Optional manual comparison between yfinance and Binance for the same asset and timeframe.",
    )


def _build_enabled_legacy_analysis(tmp_path: Path) -> dict:
    artifact_path = tmp_path / "legacy_bridge_summary.json"
    artifact_path.write_text("{\"status\": \"ok\"}")
    return {
        "enabled": True,
        "comparison_mode": "legacy_bridge",
        "asset": "BTC",
        "requested_asset": "BTC",
        "matched_asset": "BTC-USD",
        "asset_aliases": ["BTC", "BTC-USD", "BTC/USD", "BTCUSD"],
        "rows": 366,
        "clean_rows": 365,
        "selected_model": "ensemble",
        "disagreement": False,
        "modeling_health": "stable",
        "model_source": "pipeline_trained",
        "trust_score_pct": 100.0,
        "schema_alignment_pct": 100.0,
        "rationale": "Legacy bridge reused the pipeline-trained models on the same asset feed.",
        "bridge_note": "Legacy bridge reads the cleaned asset feed when available, normalizes BTC/ETH/LTC-style aliases, and reuses the pipeline-trained model outputs instead of training standalone models.",
        "source_comparison": {"enabled": True, "source_1": "yfinance", "source_2": "binance"},
        "artifact": {
            "name": artifact_path.name,
            "path": str(artifact_path),
            "kind": "legacy_summary",
            "size_bytes": artifact_path.stat().st_size,
        },
    }


def _build_disabled_legacy_analysis() -> dict:
    return {
        "enabled": False,
        "comparison_mode": "legacy_bridge",
        "asset": "BTC",
        "requested_asset": "BTC",
        "matched_asset": None,
        "selected_model": "n/a",
        "disagreement": False,
        "modeling_health": "mixed",
        "model_source": "n/a",
        "trust_score_pct": 0.0,
    }


class OrchestratorComparisonFailureTest(unittest.TestCase):
    def test_binance_failure_is_recorded_without_failing_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request = PipelineRequest(
                tickers=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 20),
                interval="1d",
                compare_binance=True,
                comparison_asset="ETH",
                comparison_yfinance_ticker="ETH-USD",
                comparison_binance_symbol="ETHUSDT",
                review_mode="off",
                use_reviewer=False,
            )

            orchestrator = PipelineOrchestrator(artifact_root=str(tmp_path / "artifacts"), reviewer=_SilentReviewer())
            extraction = _build_fake_extraction(tmp_path)
            cleaning = _build_fake_cleaning(tmp_path)
            modeling = _build_fake_modeling(tmp_path)

            with patch("pipeline.orchestrator.normalize_yfinance", return_value=pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]})), \
                 patch("pipeline.orchestrator.normalize_binance", side_effect=RuntimeError("binance down")), \
                 patch("pipeline.orchestrator.ExtractionAgent.run", return_value=extraction), \
                 patch("pipeline.orchestrator.CleaningAgent.run", return_value=cleaning), \
                 patch("pipeline.orchestrator.ModelingAgent.run", return_value=modeling):
                result = orchestrator.run(request)

            self.assertEqual(result.status, RunState.succeeded)
            self.assertIsNotNone(result.source_comparison)
            self.assertFalse(result.source_comparison.enabled)
            self.assertEqual(result.source_comparison.error, "binance down")
            self.assertIsNotNone(result.legacy_analysis)
            self.assertEqual(result.legacy_analysis["comparison_mode"], "legacy_bridge")
            self.assertEqual(result.legacy_analysis["asset"], "ETH")
            self.assertEqual(result.legacy_analysis["model_source"], "pipeline_trained")
            self.assertEqual(result.final_decision, "long")

            run_data = orchestrator.load_run(result.run_id)
            self.assertEqual(run_data["manifest"]["status"], "succeeded")
            self.assertEqual(run_data["summary"]["run_mode"], "local_plus_binance_legacy")
            self.assertTrue(
                any(log.get("message") == "source_comparison_failed" for log in run_data["logs"]),
                "Expected source_comparison_failed to be recorded in the run logs.",
            )


class OrchestratorExperimentalBrainTest(unittest.TestCase):
    def test_groq_brain_mode_overrides_the_deterministic_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            request = PipelineRequest(
                tickers=["AAPL"],
                start=date(2024, 1, 1),
                end=date(2024, 1, 20),
                interval="1d",
                review_mode="auto",
                use_reviewer=True,
                experimental_groq_brain=True,
            )

            orchestrator = PipelineOrchestrator(
                artifact_root=str(tmp_path / "artifacts"),
                reviewer=_SilentReviewer(),
                brain=_FixedBrain(_build_fake_brain_result()),
            )
            extraction = _build_fake_extraction(tmp_path)
            cleaning = _build_fake_cleaning(tmp_path)
            modeling = _build_fake_modeling(tmp_path)

            with patch("pipeline.orchestrator.ExtractionAgent.run", return_value=extraction), \
                 patch("pipeline.orchestrator.CleaningAgent.run", return_value=cleaning), \
                 patch("pipeline.orchestrator.ModelingAgent.run", return_value=modeling):
                result = orchestrator.run(request)

            self.assertEqual(result.status, RunState.succeeded)
            self.assertEqual(result.deterministic_decision, "long")
            self.assertAlmostEqual(result.deterministic_confidence, 0.6633333333, places=6)
            self.assertIsNotNone(result.groq_brain)
            self.assertEqual(result.groq_brain.provider, "groq_brain")
            self.assertEqual(result.final_decision, "short")
            self.assertAlmostEqual(result.final_confidence, 0.44, places=6)
            self.assertEqual(result.manifest.decision_source, "groq_brain")
            self.assertTrue(result.manifest.experimental_groq_brain)
            self.assertTrue(result.manifest.groq_brain_used)
            self.assertFalse(result.manifest.reviewer_used)
            self.assertEqual(result.motor["decision_path"], "groq_brain_override")

            run_data = orchestrator.load_run(result.run_id)
            self.assertEqual(run_data["summary"]["run_mode"], "local_only_groq_brain")
            self.assertTrue(run_data["summary"]["brain"]["enabled"])
            self.assertTrue(run_data["summary"]["brain"]["used"])
            self.assertEqual(run_data["summary"]["brain"]["decision_source"], "groq_brain")
            self.assertEqual(run_data["summary"]["models"]["final_decision"], "short")


class OrchestratorAdaptiveControlTest(unittest.TestCase):
    def test_adaptive_report_is_persisted_in_observe_only_mode(self) -> None:
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

            self.assertEqual(result.status, RunState.succeeded)
            self.assertIsNotNone(result.adaptive)
            self.assertEqual(result.adaptive.mode, "observe_only")
            self.assertEqual(result.adaptive.validation.status, "observe_only")
            self.assertEqual(result.adaptive.approval.status, "observe_only")
            self.assertFalse(result.adaptive.promotion.eligible)
            self.assertEqual(result.adaptive.shadow.status, "not_needed")
            self.assertEqual(result.adaptive.feature_registry.version, "v1")
            self.assertEqual(result.adaptive.retraining.status, "monitor")
            self.assertTrue(result.adaptive.runtime_fingerprint.fingerprint_id)
            self.assertIsNotNone(result.operations)
            self.assertTrue(result.operations.observability.artifact)
            self.assertTrue(result.operations.verify_gate.artifact)
            self.assertTrue(result.operations.soak_gate.artifact)
            self.assertTrue(result.operations.release_summary.artifact)
            self.assertTrue(Path(result.operations.observability.artifact.path).exists())
            self.assertTrue(Path(result.operations.verify_gate.artifact.path).exists())
            self.assertTrue(Path(result.operations.soak_gate.artifact.path).exists())
            self.assertTrue(Path(result.operations.release_summary.artifact.path).exists())
            self.assertTrue(result.operations.verify_gate.ok)
            self.assertFalse(result.operations.soak_gate.ok)
            self.assertEqual(result.operations.release_summary.release_stage, "ops_pending")
            self.assertGreaterEqual(result.operations.observability.run_duration_ms, 0)
            self.assertEqual(result.operations.observability.stage_statuses["orchestrator"], "completed")
            self.assertEqual(result.operations.observability.stage_statuses["extraction"], "completed")
            self.assertEqual(result.operations.observability.stage_statuses["cleaning"], "completed")
            self.assertEqual(result.operations.observability.stage_statuses["modeling"], "completed")
            self.assertIn("raw_data", result.operations.observability.artifact_kind_counts)
            self.assertIn("extraction_agent", result.operations.observability.agent_event_counts)
            self.assertEqual(result.operations.observability.data_profile["raw_rows"], 12)
            self.assertEqual(result.operations.observability.data_profile["clean_rows"], 11)
            self.assertEqual(result.operations.observability.data_profile["model_count"], 3)
            self.assertEqual(result.operations.observability.decision_profile["final_decision"], "long")
            self.assertIsNotNone(result.adaptive.artifact)
            self.assertTrue(Path(result.adaptive.artifact.path).exists())
            self.assertIsNotNone(result.adaptive.feature_registry.artifact)
            self.assertTrue(Path(result.adaptive.feature_registry.artifact.path).exists())
            self.assertIsNotNone(result.adaptive.retraining.artifact)
            self.assertTrue(Path(result.adaptive.retraining.artifact.path).exists())
            self.assertIsNotNone(result.adaptive.runtime_fingerprint.artifact)
            self.assertTrue(Path(result.adaptive.runtime_fingerprint.artifact.path).exists())
            self.assertIsNotNone(result.adaptive.approval.artifact)
            self.assertTrue(Path(result.adaptive.approval.artifact.path).exists())

            run_data = orchestrator.load_run(result.run_id)
            self.assertIn("adaptive", run_data["summary"])
            self.assertIn("operations", run_data["summary"])
            self.assertEqual(run_data["summary"]["adaptive"]["mode"], "observe_only")
            self.assertEqual(run_data["summary"]["adaptive"]["retraining"]["status"], "monitor")
            self.assertEqual(run_data["summary"]["adaptive"]["feature_registry"]["version"], "v1")
            self.assertEqual(run_data["summary"]["adaptive"]["approval"]["status"], "observe_only")
            self.assertEqual(run_data["summary"]["operations"]["release_summary"]["release_stage"], "ops_pending")
            self.assertEqual(run_data["summary"]["operations"]["observability"]["stage_statuses"]["orchestrator"], "completed")
            self.assertEqual(run_data["summary"]["operations"]["observability"]["data_profile"]["feature_count"], 14)
            self.assertTrue(
                any(log.get("message") == "adaptive_report_generated" for log in run_data["logs"]),
                "Expected adaptive_report_generated to be recorded in the run logs.",
            )
            self.assertTrue(
                any(log.get("message") == "operations_report_generated" for log in run_data["logs"]),
                "Expected operations_report_generated to be recorded in the run logs.",
            )

    def test_adaptive_report_can_mark_candidate_ready_after_shadow_benchmark(self) -> None:
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
            modeling.rationale = "Synthetic disagreement for adaptive promotion coverage."

            with patch("pipeline.orchestrator.ExtractionAgent.run", return_value=extraction), \
                 patch("pipeline.orchestrator.CleaningAgent.run", return_value=cleaning), \
                 patch("pipeline.orchestrator.ModelingAgent.run", return_value=modeling):
                result = orchestrator.run(request)

            self.assertEqual(result.status, RunState.succeeded)
            self.assertIsNotNone(result.adaptive)
            self.assertEqual(result.adaptive.shadow.status, "completed")
            self.assertTrue(result.adaptive.shadow.ready_for_promotion)
            self.assertEqual(result.adaptive.validation.status, "candidate_ready")
            self.assertEqual(result.adaptive.approval.status, "approved_for_promotion_review")
            self.assertTrue(result.adaptive.promotion.eligible)
            self.assertEqual(result.adaptive.promotion.mode, "promotion_ready")
            self.assertEqual(result.adaptive.retraining.status, "scheduled")
            self.assertEqual(result.adaptive.retraining.recommended_within_hours, 24)
            self.assertIsNotNone(result.operations)
            self.assertEqual(result.operations.release_summary.release_stage, "ops_pending")
            self.assertIsNotNone(result.adaptive.shadow.artifact)
            self.assertTrue(Path(result.adaptive.shadow.artifact.path).exists())

            run_data = orchestrator.load_run(result.run_id)
            self.assertEqual(run_data["summary"]["adaptive"]["validation"]["status"], "candidate_ready")
            self.assertEqual(run_data["summary"]["adaptive"]["approval"]["status"], "approved_for_promotion_review")
            self.assertEqual(run_data["summary"]["adaptive"]["promotion"]["mode"], "promotion_ready")
            self.assertEqual(run_data["summary"]["adaptive"]["retraining"]["status"], "scheduled")


class OrchestratorRunModeMatrixTest(unittest.TestCase):
    def test_run_mode_matrix_covers_core_variants(self) -> None:
        scenarios = [
            {
                "name": "local_only",
                "compare_binance": False,
                "use_reviewer": False,
                "review_mode": "off",
                "legacy_analysis": None,
                "source_comparison": None,
                "reviewer": _SilentReviewer(),
                "expected_run_mode": "local_only",
            },
            {
                "name": "local_plus_reviewer",
                "compare_binance": False,
                "use_reviewer": True,
                "review_mode": "on",
                "legacy_analysis": None,
                "source_comparison": None,
                "reviewer": _FixedReviewer(_build_fake_reviewer_result()),
                "expected_run_mode": "local_plus_reviewer",
            },
            {
                "name": "local_plus_binance",
                "compare_binance": True,
                "use_reviewer": False,
                "review_mode": "off",
                "legacy_analysis": _build_disabled_legacy_analysis(),
                "source_comparison": _build_source_comparison(),
                "reviewer": _SilentReviewer(),
                "expected_run_mode": "local_plus_binance",
            },
            {
                "name": "local_plus_binance_plus_reviewer",
                "compare_binance": True,
                "use_reviewer": True,
                "review_mode": "on",
                "legacy_analysis": _build_disabled_legacy_analysis(),
                "source_comparison": _build_source_comparison(),
                "reviewer": _FixedReviewer(_build_fake_reviewer_result()),
                "expected_run_mode": "local_plus_binance_plus_reviewer",
            },
            {
                "name": "local_plus_binance_legacy",
                "compare_binance": True,
                "use_reviewer": False,
                "review_mode": "off",
                "legacy_analysis": None,
                "source_comparison": _build_source_comparison(),
                "reviewer": _SilentReviewer(),
                "expected_run_mode": "local_plus_binance_legacy",
            },
            {
                "name": "local_plus_binance_legacy_plus_reviewer",
                "compare_binance": True,
                "use_reviewer": True,
                "review_mode": "on",
                "legacy_analysis": None,
                "source_comparison": _build_source_comparison(),
                "reviewer": _FixedReviewer(_build_fake_reviewer_result()),
                "expected_run_mode": "local_plus_binance_legacy_plus_reviewer",
            },
        ]

        for scenario in scenarios:
            with self.subTest(scenario=scenario["name"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    request = PipelineRequest(
                        tickers=["BTC-USD"],
                        start=date(2024, 1, 1),
                        end=date(2024, 1, 20),
                        interval="1d",
                        compare_binance=scenario["compare_binance"],
                        comparison_asset="BTC" if scenario["compare_binance"] else None,
                        comparison_yfinance_ticker="BTC-USD" if scenario["compare_binance"] else None,
                        comparison_binance_symbol="BTCUSDT" if scenario["compare_binance"] else None,
                        review_mode=scenario["review_mode"],
                        use_reviewer=scenario["use_reviewer"],
                    )

                    orchestrator = PipelineOrchestrator(artifact_root=str(tmp_path / "artifacts"), reviewer=scenario["reviewer"])
                    extraction = _build_fake_extraction(tmp_path)
                    cleaning = _build_fake_cleaning(tmp_path)
                    modeling = _build_fake_modeling(tmp_path)

                    legacy_patch = scenario["legacy_analysis"]
                    source_patch = scenario["source_comparison"]
                    if legacy_patch is None and scenario["compare_binance"]:
                        legacy_patch = _build_enabled_legacy_analysis(tmp_path)

                    with patch("pipeline.orchestrator.PipelineOrchestrator._build_source_comparison", return_value=source_patch), \
                         patch("pipeline.orchestrator.build_legacy_btc_analysis", return_value=legacy_patch), \
                         patch("pipeline.orchestrator.ExtractionAgent.run", return_value=extraction), \
                         patch("pipeline.orchestrator.CleaningAgent.run", return_value=cleaning), \
                         patch("pipeline.orchestrator.ModelingAgent.run", return_value=modeling):
                        result = orchestrator.run(request)

                    self.assertEqual(result.status, RunState.succeeded)
                    run_data = orchestrator.load_run(result.run_id)
                    self.assertEqual(run_data["summary"]["run_mode"], scenario["expected_run_mode"])
