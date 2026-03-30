
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


def parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def interval_to_binance(interval: str) -> str:
    mapping = {
        "1d": "1d",
        "1h": "1h",
        "4h": "4h",
        "1wk": "1w",
    }
    return mapping.get(interval, interval)


def normalize_yfinance(ticker: str, start: str, end: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    frame = yf.Ticker(ticker).history(start=start, end=end, interval=interval, auto_adjust=False, actions=False)
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame = frame.reset_index()
    if "Date" in frame.columns:
        frame = frame.rename(columns={"Date": "date"})
    if "Datetime" in frame.columns:
        frame = frame.rename(columns={"Datetime": "date"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_convert(None)
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in frame.columns]
    return frame[keep].copy()


def normalize_binance(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval_to_binance(interval),
        "startTime": to_ms(parse_date(start)),
        "endTime": to_ms(parse_date(end)),
        "limit": 1000,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    rows = response.json()
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["date", "open", "high", "low", "close", "volume"]].copy()


def compare_frames(yf_frame: pd.DataFrame, bn_frame: pd.DataFrame) -> Dict:
    yf_frame = yf_frame.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)
    bn_frame = bn_frame.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)

    merged = yf_frame.merge(bn_frame, on="date", how="outer", suffixes=("_yfinance", "_binance"), indicator=True)
    overlap = merged[merged["_merge"] == "both"].copy()
    missing_yfinance = merged[merged["_merge"] == "right_only"]
    missing_binance = merged[merged["_merge"] == "left_only"]

    close_alignment = {
        "overlap_rows": int(len(overlap)),
        "mae": None,
        "mape_pct": None,
        "correlation": None,
    }
    if not overlap.empty:
        close_diff = (overlap["close_yfinance"] - overlap["close_binance"]).abs()
        close_alignment["mae"] = float(close_diff.mean())
        denom = overlap["close_binance"].abs().replace(0, pd.NA)
        mape = (close_diff / denom).dropna()
        close_alignment["mape_pct"] = float(mape.mean() * 100.0) if not mape.empty else None
        try:
            close_alignment["correlation"] = float(overlap[["close_yfinance", "close_binance"]].corr().iloc[0, 1])
        except Exception:
            close_alignment["correlation"] = None

    return {
        "coverage": {
            "yfinance_rows": int(len(yf_frame)),
            "binance_rows": int(len(bn_frame)),
            "overlap_rows": int(len(overlap)),
            "yfinance_only_rows": int(len(missing_binance)),
            "binance_only_rows": int(len(missing_yfinance)),
        },
        "missing_fields": {
            "yfinance": [],
            "binance": [],
        },
        "row_counts": {
            "yfinance": int(len(yf_frame)),
            "binance": int(len(bn_frame)),
            "overlap": int(len(overlap)),
            "union": int(len(merged)),
        },
        "close_price_alignment": close_alignment,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare yfinance against Binance for one asset")
    parser.add_argument("--asset", default="BTC", help="Asset label used in the report")
    parser.add_argument("--yfinance-ticker", default="BTC-USD", help="yfinance ticker, e.g. BTC-USD")
    parser.add_argument("--binance-symbol", default="BTCUSDT", help="Binance symbol, e.g. BTCUSDT")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Interval to compare")
    parser.add_argument("--output", default="artifacts/source_comparison.json", help="Output JSON file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    yf_frame = normalize_yfinance(args.yfinance_ticker, args.start, args.end, args.interval)
    bn_frame = normalize_binance(args.binance_symbol, args.start, args.end, args.interval)
    comparison = compare_frames(yf_frame, bn_frame)

    payload = {
        "enabled": True,
        "source_1": "yfinance",
        "source_2": "binance",
        "asset": args.asset,
        "timeframe": args.interval,
        "date_range": {
            "start": args.start,
            "end": args.end,
        },
        "coverage": comparison["coverage"],
        "missing_fields": comparison["missing_fields"],
        "row_counts": comparison["row_counts"],
        "close_price_alignment": comparison["close_price_alignment"],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
