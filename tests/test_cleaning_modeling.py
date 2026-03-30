from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from agents.cleaning_agent import CleaningAgent
from agents.modeling_agent import ModelingAgent
from schemas.pipeline import ArtifactRef, ExtractionResult, PipelineRequest, RawMarketData


class CleaningLeakageRegressionTest(unittest.TestCase):
    def test_latest_row_stays_unlabeled_and_is_excluded_from_training(self) -> None:
        closes = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=len(closes), freq="D"),
                "ticker": ["AAPL"] * len(closes),
                "open": [value - 0.5 for value in closes],
                "high": [value + 1.0 for value in closes],
                "low": [value - 1.0 for value in closes],
                "close": closes,
                "adj_close": closes,
                "volume": [1000 + index * 10 for index in range(len(closes))],
            }
        )

        request = PipelineRequest(
            tickers=["AAPL"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 20),
            interval="1d",
            review_mode="off",
            use_reviewer=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_path = tmp_path / "raw_market_data.csv"
            frame.to_csv(raw_path, index=False)

            extraction = ExtractionResult(
                rows=len(frame),
                tickers=["AAPL"],
                missing_columns=[],
                raw_data=RawMarketData(
                    ticker="AAPL",
                    start=request.start,
                    end=request.end,
                    interval=request.interval,
                    rows=len(frame),
                    columns=list(frame.columns),
                    missing_columns=[],
                    artifact=ArtifactRef(
                        name=raw_path.name,
                        path=str(raw_path),
                        kind="raw_data",
                        size_bytes=raw_path.stat().st_size,
                    ),
                ),
            )

            cleaning = CleaningAgent().run(request, extraction, tmp_path / "run_0001")
            cleaned_df = pd.read_csv(cleaning.clean_data.artifact.path, parse_dates=["date"])

            self.assertTrue(pd.isna(cleaned_df.iloc[-1]["target_direction"]))
            self.assertEqual(int(cleaned_df["target_direction"].isna().sum()), 1)

            X, y, latest_features = ModelingAgent()._prepare_matrices(cleaned_df)
            self.assertEqual(len(y), len(cleaned_df) - 1)
            self.assertEqual(len(latest_features), 1)
