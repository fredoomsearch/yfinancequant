from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from agents.extraction_agent import ExtractionAgent
from schemas.pipeline import PipelineRequest


class _FakeTicker:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def history(self, *args, **kwargs) -> pd.DataFrame:
        return self._frame.copy()


class _FakeYFinance:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def Ticker(self, ticker: str) -> _FakeTicker:  # noqa: N802 - mirror yfinance API
        return _FakeTicker(self._frame)


class ExtractionAgentTest(unittest.TestCase):
    def test_normalizes_yfinance_history_into_raw_artifact(self) -> None:
        frame = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Adj Close": [100.4, 101.4],
                "Volume": [1000, 1500],
            },
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date"),
        )

        request = PipelineRequest(
            tickers=["AAPL"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 5),
            interval="1d",
            review_mode="off",
            use_reviewer=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_0001"
            with patch("agents.extraction_agent._load_yfinance", return_value=_FakeYFinance(frame)):
                result = ExtractionAgent().run(request, run_dir)

            self.assertEqual(result.rows, 2)
            self.assertEqual(result.tickers, ["AAPL"])
            self.assertEqual(result.missing_columns, [])
            self.assertEqual(
                result.raw_data.columns[:8],
                ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
            )
            self.assertTrue(Path(result.raw_data.artifact.path).exists())
            self.assertTrue((run_dir / "raw" / "raw_market_data.csv").exists())
            self.assertTrue((run_dir / "raw" / "extraction_summary.json").exists())
