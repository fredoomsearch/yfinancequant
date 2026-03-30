from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from providers import GroqAdvisor
from schemas.pipeline import ArtifactRef, CleaningResult, CleanMarketData, ExtractionResult, PipelineRequest

logger = logging.getLogger(__name__)


def _ensure_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert(None)


@dataclass
class CleaningAgent:
    advisor: Optional[GroqAdvisor] = None

    def run(self, request: PipelineRequest, extraction: ExtractionResult, run_dir: Path) -> CleaningResult:
        raw_path = Path(extraction.raw_data.artifact.path)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        output_dir = run_dir / "cleaned"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(raw_path)
        rows_in = int(len(df))
        report: dict = {
            "rows_in": rows_in,
            "dropped_duplicates": 0,
            "dropped_missing": 0,
            "feature_engineering": [],
        }

        if "ticker" not in df.columns:
            raise ValueError("Raw data must contain a ticker column")
        if "date" not in df.columns:
            raise ValueError("Raw data must contain a date column")

        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["date"] = _ensure_datetime(df["date"])

        numeric_columns = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        before = len(df)
        df = df.dropna(subset=["ticker", "date", "close"])
        report["dropped_missing"] = int(before - len(df))

        before = len(df)
        df = df.sort_values(["ticker", "date"]).drop_duplicates(subset=["ticker", "date"], keep="last")
        report["dropped_duplicates"] = int(before - len(df))

        grouped = df.groupby("ticker", group_keys=False)
        df["return_1d"] = grouped["close"].pct_change()
        df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        df["body_pct"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
        df["sma_5"] = grouped["close"].transform(lambda s: s.rolling(window=5, min_periods=3).mean())
        df["sma_10"] = grouped["close"].transform(lambda s: s.rolling(window=10, min_periods=5).mean())
        df["volatility_5"] = grouped["return_1d"].transform(lambda s: s.rolling(window=5, min_periods=3).std())
        df["volume_sma_5"] = grouped["volume"].transform(lambda s: s.rolling(window=5, min_periods=3).mean())
        df["future_close"] = grouped["close"].shift(-1)
        df["future_return"] = (df["future_close"] - df["close"]) / df["close"].replace(0, np.nan)
        df["target_direction"] = np.where(
            df["future_return"].notna(),
            (df["future_return"] > 0).astype(float),
            np.nan,
        )

        feature_columns = [
            column
            for column in df.select_dtypes(include=["number"]).columns
            if column not in {"future_close", "future_return", "target_direction"}
        ]
        feature_columns.extend(["ticker"])
        feature_columns = [column for column in feature_columns if column in df.columns]

        df = df.replace([np.inf, -np.inf], np.nan)

        cleaned_csv = output_dir / "clean_market_data.csv"
        df.to_csv(cleaned_csv, index=False)

        report.update(
            {
                "rows_out": int(len(df)),
                "feature_columns": feature_columns,
                "target_column": "target_direction",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "motor": "feature_engineering",
            }
        )

        groq_brief = None
        if self.advisor and self.advisor.enabled:
            groq_brief = self.advisor.brief(
                "cleaning",
                {
                    "motor": "feature_engineering",
                    "language": request.language,
                    "rows_in": rows_in,
                    "rows_out": int(len(df)),
                    "dropped_duplicates": report.get("dropped_duplicates", 0),
                    "dropped_missing": report.get("dropped_missing", 0),
                    "feature_columns": feature_columns,
                    "target_column": "target_direction",
                    "quality_report": report,
                },
            )
            if groq_brief:
                report["groq_brief"] = groq_brief.model_dump()
                report["motor"] = groq_brief.motor or report["motor"]

        (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2, default=str))

        artifact = ArtifactRef(
            name="clean_market_data.csv",
            path=str(cleaned_csv),
            kind="clean_data",
            size_bytes=cleaned_csv.stat().st_size,
        )
        clean_data = CleanMarketData(
            rows_in=rows_in,
            rows_out=int(len(df)),
            feature_columns=feature_columns,
            target_column="target_direction",
            artifact=artifact,
            quality_report=report,
        )
        return CleaningResult(
            rows_in=rows_in,
            rows_out=int(len(df)),
            feature_columns=feature_columns,
            target_column="target_direction",
            clean_data=clean_data,
            groq_brief=groq_brief,
        )
