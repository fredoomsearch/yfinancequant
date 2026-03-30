from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from providers import GroqAdvisor
from schemas.pipeline import ArtifactRef, ExtractionResult, PipelineRequest, RawMarketData

logger = logging.getLogger(__name__)


def _load_yfinance():
    import yfinance as yf

    return yf


def _normalize_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "date" not in frame.columns:
        for candidate in ("Datetime", "Date", "timestamp"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "date"})
                break
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
    return frame


@dataclass
class ExtractionAgent:
    advisor: Optional[GroqAdvisor] = None

    def run(self, request: PipelineRequest, run_dir: Path) -> ExtractionResult:
        yf = _load_yfinance()
        output_dir = run_dir / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)

        expected_columns = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]

        frames: list[pd.DataFrame] = []
        failures: list[dict] = []
        missing_columns: set[str] = set()

        for ticker in request.tickers:
            ticker = ticker.upper().strip()
            if not ticker:
                continue
            try:
                history = yf.Ticker(ticker).history(
                    start=request.start,
                    end=request.end,
                    interval=request.interval,
                    auto_adjust=False,
                    actions=False,
                )
                if history.empty:
                    failures.append({"ticker": ticker, "reason": "no_rows_returned"})
                    continue

                history = history.reset_index()
                history = _normalize_date_column(history)
                rename_map = {
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
                history = history.rename(columns=rename_map)
                history["ticker"] = ticker

                for column in expected_columns:
                    if column not in history.columns:
                        missing_columns.add(column)

                keep_columns = [c for c in expected_columns if c in history.columns]
                keep_columns.extend([c for c in history.columns if c not in keep_columns])
                history = history[keep_columns]
                frames.append(history)
                logger.info("Extraction complete for %s with %s rows", ticker, len(history))
            except Exception as exc:
                failures.append({"ticker": ticker, "reason": str(exc)})
                logger.exception("Extraction failed for %s", ticker)

        if not frames:
            raise ValueError(f"No market data could be extracted. Failures: {failures}")

        raw_df = pd.concat(frames, ignore_index=True)
        raw_df = raw_df.sort_values(["ticker", "date"]).reset_index(drop=True)

        raw_csv = output_dir / "raw_market_data.csv"
        raw_df.to_csv(raw_csv, index=False)

        summary = {
            "tickers": request.tickers,
            "start": request.start.isoformat(),
            "end": request.end.isoformat(),
            "interval": request.interval,
            "rows": int(len(raw_df)),
            "columns": list(raw_df.columns),
            "missing_columns": sorted(missing_columns),
            "failures": failures,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "motor": "yfinance",
        }

        groq_brief = None
        if self.advisor and self.advisor.enabled:
            groq_brief = self.advisor.brief(
                "extraction",
                {
                    "motor": "yfinance",
                    "language": request.language,
                    "tickers": request.tickers,
                    "start": request.start.isoformat(),
                    "end": request.end.isoformat(),
                    "interval": request.interval,
                    "rows": int(len(raw_df)),
                    "columns": list(raw_df.columns),
                    "missing_columns": sorted(missing_columns),
                    "failures": failures,
                },
            )
            if groq_brief:
                summary["groq_brief"] = groq_brief.model_dump()
                summary["motor"] = groq_brief.motor or summary["motor"]

        (output_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2, default=str))

        artifact = ArtifactRef(
            name="raw_market_data.csv",
            path=str(raw_csv),
            kind="raw_data",
            size_bytes=raw_csv.stat().st_size,
        )
        raw_data = RawMarketData(
            ticker=",".join(request.tickers),
            start=request.start,
            end=request.end,
            interval=request.interval,
            rows=int(len(raw_df)),
            columns=list(raw_df.columns),
            missing_columns=sorted(missing_columns),
            artifact=artifact,
        )
        return ExtractionResult(
            rows=int(len(raw_df)),
            tickers=[t.upper().strip() for t in request.tickers if t.strip()],
            missing_columns=sorted(missing_columns),
            raw_data=raw_data,
            groq_brief=groq_brief,
        )
