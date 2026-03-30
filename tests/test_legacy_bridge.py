from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from schemas.pipeline import ArtifactRef, CleanMarketData, CleaningResult, ModelResult, ModelingResult
from utils.legacy_btc import _resolve_asset_frame, build_legacy_btc_analysis


class LegacyBridgeAssetResolutionTest(unittest.TestCase):
    def test_resolves_non_btc_aliases(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4, freq="D"),
                "ticker": ["BTC-USD", "ETH-USD", "ETHUSDT", "BTC-USD"],
                "close": [100.0, 200.0, 201.0, 101.0],
                "volume": [1000, 1200, 1300, 1100],
            }
        )

        selected, matched_asset, aliases = _resolve_asset_frame(frame, "ETH")

        self.assertEqual(matched_asset, "ETH-USD")
        self.assertIn("ETHUSDT", aliases)
        self.assertListEqual(sorted(selected["ticker"].unique().tolist()), ["ETH-USD"])


class LegacyBridgeReuseTest(unittest.TestCase):
    def _build_cleaning_result(self, tmp_path: Path) -> CleaningResult:
        cleaned_path = tmp_path / "clean_market_data.csv"
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        closes = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
        target_direction = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, float("nan")]
        cleaned_df = pd.DataFrame(
            {
                "date": dates,
                "ticker": ["ETH-USD"] * len(dates),
                "open": [value - 0.5 for value in closes],
                "high": [value + 1.0 for value in closes],
                "low": [value - 1.0 for value in closes],
                "close": closes,
                "adj_close": closes,
                "volume": [1000 + index * 25 for index in range(len(dates))],
                "return_1d": [0.01] * len(dates),
                "range_pct": [0.02] * len(dates),
                "body_pct": [0.01] * len(dates),
                "sma_5": [100.0] * len(dates),
                "sma_10": [100.0] * len(dates),
                "volatility_5": [0.1] * len(dates),
                "volume_sma_5": [1000.0] * len(dates),
                "target_direction": target_direction,
            }
        )
        cleaned_df.to_csv(cleaned_path, index=False)

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
        quality_report = {
            "rows_in": len(cleaned_df),
            "rows_out": len(cleaned_df),
            "dropped_duplicates": 0,
            "dropped_missing": 0,
            "feature_columns": feature_columns,
            "target_column": "target_direction",
        }
        return CleaningResult(
            rows_in=len(cleaned_df),
            rows_out=len(cleaned_df),
            feature_columns=feature_columns,
            target_column="target_direction",
            clean_data=CleanMarketData(
                rows_in=len(cleaned_df),
                rows_out=len(cleaned_df),
                feature_columns=feature_columns,
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

    def _build_modeling_result(self, tmp_path: Path) -> ModelingResult:
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        model_results = []
        for model_name, probability in (
            ("logistic_regression", 0.72),
            ("random_forest", 0.66),
            ("gradient_boosting", 0.68),
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
            ensemble_probability=0.6866666667,
            ensemble_prediction="long",
            majority_prediction="long",
            disagreement=False,
            selected_model="ensemble",
            rationale="Synthetic legacy bridge reuse test.",
            latest_sample_count=1,
            artifact=ArtifactRef(
                name=summary_path.name,
                path=str(summary_path),
                kind="model_summary",
                size_bytes=summary_path.stat().st_size,
            ),
        )

    def test_reuses_pipeline_trained_models_for_eth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cleaning = self._build_cleaning_result(tmp_path)
            modeling = self._build_modeling_result(tmp_path)
            raw_df = pd.read_csv(cleaning.clean_data.artifact.path)

            summary = build_legacy_btc_analysis(
                raw_df,
                tmp_path / "run_0001",
                asset="ETH",
                language="en",
                source_comparison={"enabled": True, "source_1": "yfinance", "source_2": "binance"},
                cleaning=cleaning,
                modeling=modeling,
            )

            self.assertEqual(summary["comparison_mode"], "legacy_bridge")
            self.assertEqual(summary["asset"], "ETH")
            self.assertEqual(summary["requested_asset"], "ETH")
            self.assertEqual(summary["model_source"], "pipeline_trained")
            self.assertEqual(summary["trust_score_pct"], 100.0)
            self.assertEqual(summary["schema_alignment_pct"], 100.0)
            self.assertIn("ETH-USD", summary["asset_aliases"])
            self.assertEqual(summary["models"][0]["model_name"], "logistic_regression")
            self.assertEqual(summary["models"][0]["source"], "pipeline_modeling")
            self.assertEqual(summary["models"][0]["bridge_mode"], "reused_trained_models")
            self.assertTrue(Path(summary["artifact"]["path"]).exists())
