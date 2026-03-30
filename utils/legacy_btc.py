from __future__ import annotations

import importlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from schemas.pipeline import ArtifactRef, CleaningResult, ModelingResult

logger = logging.getLogger(__name__)


def _normalize_asset(value: Optional[str]) -> str:
    token = re.sub(r"[^A-Z0-9]", "", (value or "BTC").upper().strip())
    for suffix in ("USDT", "USDC", "BUSD", "USD"):
        if token.endswith(suffix) and len(token) > len(suffix):
            token = token[: -len(suffix)]
            break
    return token or "BTC"


def _asset_aliases(asset: str) -> List[str]:
    base = _normalize_asset(asset)
    aliases: List[str] = [base]
    for suffix in (
        "-USD",
        "/USD",
        "USD",
        "-USDT",
        "/USDT",
        "USDT",
        "-USDC",
        "/USDC",
        "USDC",
        "-BUSD",
        "/BUSD",
        "BUSD",
    ):
        candidate = f"{base}{suffix}"
        if candidate not in aliases:
            aliases.append(candidate)
    return aliases


def _normalized_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper().strip())


def _resolve_asset_frame(raw_df: pd.DataFrame, asset: str) -> Tuple[pd.DataFrame, Optional[str], List[str]]:
    frame = raw_df.copy()
    aliases = _asset_aliases(asset)
    if "ticker" not in frame.columns:
        return frame, None, aliases

    tickers = frame["ticker"].astype(str).str.upper().str.strip()
    normalized_tickers = tickers.map(_normalized_key)

    for candidate in aliases:
        exact_match = frame.loc[tickers == candidate].copy()
        if not exact_match.empty:
            return exact_match, candidate, aliases

    for candidate in aliases:
        normalized_candidate = _normalized_key(candidate)
        if not normalized_candidate:
            continue
        normalized_match = frame.loc[normalized_tickers == normalized_candidate].copy()
        if not normalized_match.empty:
            return normalized_match, candidate, aliases

    return frame, None, aliases


def _load_bridge_feed(raw_df: pd.DataFrame, cleaning: Optional[CleaningResult]) -> pd.DataFrame:
    if cleaning is not None:
        cleaned_path = Path(cleaning.clean_data.artifact.path)
        if cleaned_path.exists():
            try:
                return pd.read_csv(cleaned_path)
            except Exception as exc:
                logger.warning("Falling back to the raw feed because the cleaned artifact could not be read: %s", exc)
    return raw_df.copy()


def _coerce_legacy_frame(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    frame = df.copy()
    if "date" not in frame.columns:
        for candidate in ("Datetime", "Date", "timestamp"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "date"})
                break

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    for column in [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "current_price",
        "high_24h",
        "low_24h",
        "total_volume",
        "market_cap",
        "future_close",
        "future_return",
        "target_direction",
        "price_direction",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    else:
        frame["ticker"] = _normalize_asset(asset)

    frame = frame.sort_values([column for column in ["ticker", "date"] if column in frame.columns]).reset_index(drop=True)
    frame["symbol"] = _normalize_asset(asset)
    frame["legacy_asset"] = _normalize_asset(asset)
    frame["timestamp"] = frame["date"] if "date" in frame.columns else pd.NaT
    return frame


def _enrich_legacy_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if frame.empty:
        return frame

    if "current_price" not in frame.columns:
        if "close" in frame.columns:
            frame["current_price"] = frame["close"]
        elif "adj_close" in frame.columns:
            frame["current_price"] = frame["adj_close"]
        else:
            frame["current_price"] = np.nan

    if "high_24h" not in frame.columns:
        frame["high_24h"] = frame.get("high", frame["current_price"])
    if "low_24h" not in frame.columns:
        frame["low_24h"] = frame.get("low", frame["current_price"])
    if "total_volume" not in frame.columns:
        frame["total_volume"] = frame.get("volume", 0)

    frame["current_price"] = pd.to_numeric(frame["current_price"], errors="coerce")
    frame["high_24h"] = pd.to_numeric(frame["high_24h"], errors="coerce")
    frame["low_24h"] = pd.to_numeric(frame["low_24h"], errors="coerce")
    frame["total_volume"] = pd.to_numeric(frame["total_volume"], errors="coerce").fillna(0)
    frame["market_cap"] = frame.get("market_cap", frame["current_price"] * frame["total_volume"].replace(0, np.nan))
    frame["price_change_24h"] = frame["current_price"].diff()
    frame["price_change_percentage_24h"] = frame["current_price"].pct_change() * 100.0
    frame["market_cap_change_24h"] = frame["market_cap"].diff()
    frame["market_cap_change_percentage_24h"] = frame["market_cap"].pct_change() * 100.0
    frame["circulating_supply"] = frame.get("circulating_supply", 0.0)
    frame["total_supply"] = frame.get("total_supply", 0.0)
    frame["max_supply"] = frame.get("max_supply", 0.0)
    frame["ath"] = frame["current_price"].cummax()
    frame["ath_change_percentage"] = (frame["current_price"] / frame["ath"].replace(0, np.nan) - 1.0) * 100.0
    frame["ath_date"] = frame["timestamp"]
    frame["atl"] = frame["current_price"].cummin()
    frame["atl_change_percentage"] = (frame["current_price"] / frame["atl"].replace(0, np.nan) - 1.0) * 100.0
    frame["atl_date"] = frame["timestamp"]
    frame["last_updated"] = frame["timestamp"]
    frame["news_headlines"] = frame.get("news_headlines", "")

    price = frame["current_price"]
    volume = frame["total_volume"].fillna(0)
    frame["sma_20"] = price.rolling(20, min_periods=5).mean()
    frame["sma_50"] = price.rolling(50, min_periods=10).mean()
    frame["ema_20"] = price.ewm(span=20, adjust=False).mean()
    frame["ema_50"] = price.ewm(span=50, adjust=False).mean()
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(window=14, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14, min_periods=7).mean()
    rs = gain / loss.replace(0, np.nan)
    frame["rsi_14"] = 100 - (100 / (1 + rs))
    rolling_std = price.rolling(20, min_periods=5).std()
    frame["bb_upper"] = frame["sma_20"] + (2 * rolling_std)
    frame["bb_middle"] = frame["sma_20"]
    frame["bb_lower"] = frame["sma_20"] - (2 * rolling_std)
    frame["bb_percent"] = (price - frame["bb_lower"]) / (frame["bb_upper"] - frame["bb_lower"] + 1e-8)
    frame["atr_14"] = (frame["high_24h"] - frame["low_24h"]).rolling(14, min_periods=5).mean()
    frame["vwap"] = (price * volume).cumsum() / (volume.cumsum() + 1e-8)
    frame["obv"] = np.sign(price.diff().fillna(0.0)).mul(volume).cumsum()
    frame["return_1"] = price.pct_change(1)
    frame["return_3"] = price.pct_change(3)
    frame["rolling_vol_7"] = frame["return_1"].rolling(7, min_periods=3).std()
    frame["rolling_vol_14"] = frame["return_1"].rolling(14, min_periods=5).std()

    for lag in (1, 3, 5, 7):
        frame[f"current_price_lag_{lag}"] = price.shift(lag)
    for window in (3, 7, 14):
        frame[f"current_price_roll_mean_{window}"] = price.rolling(window=window, min_periods=max(3, window // 2)).mean()
        frame[f"current_price_roll_std_{window}"] = price.rolling(window=window, min_periods=max(3, window // 2)).std()

    frame["vol_over_marketcap"] = frame["total_volume"] / frame["market_cap"].replace(0, np.nan)
    frame["supply_ratio"] = 0.0
    frame["sentiment_score"] = 0.0
    frame["twitter_sentiment"] = 0.0
    frame["reddit_sentiment"] = 0.0
    frame["active_addresses"] = 0.0
    frame["transaction_count"] = 0.0
    frame["network_fee"] = 0.0
    frame["btc_correlation"] = 0.5
    frame["eth_correlation"] = 0.5
    frame["market_correlation"] = 0.5

    if "timestamp" in frame.columns:
        frame["hour"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.hour.fillna(0).astype(int)
        frame["day_of_week"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.dayofweek.fillna(0).astype(int)
        frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
        frame["month"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.month.fillna(0).astype(int)
        frame["quarter"] = pd.to_datetime(frame["timestamp"], errors="coerce").dt.quarter.fillna(0).astype(int)
    else:
        frame["hour"] = 0
        frame["day_of_week"] = 0
        frame["is_weekend"] = 0
        frame["month"] = 0
        frame["quarter"] = 0

    frame["orderbook_imbalance"] = 0.0

    if "target_direction" in frame.columns:
        frame["price_direction"] = pd.to_numeric(frame["target_direction"], errors="coerce")
    if "price_direction" not in frame.columns:
        future_close = frame["current_price"].shift(-1)
        frame["future_close"] = future_close
        frame["future_return"] = (future_close - frame["current_price"]) / frame["current_price"].replace(0, np.nan)
        frame["price_direction"] = np.where(future_close.notna(), (future_close > frame["current_price"]).astype(float), np.nan)
    else:
        future_close = frame["current_price"].shift(-1)
        if "future_close" not in frame.columns:
            frame["future_close"] = future_close
        if "future_return" not in frame.columns:
            frame["future_return"] = (future_close - frame["current_price"]) / frame["current_price"].replace(0, np.nan)
        frame.loc[future_close.isna(), "price_direction"] = np.nan

    return frame.replace([np.inf, -np.inf], np.nan)


def _load_shared_helpers() -> Dict[str, Any]:
    helpers: Dict[str, Any] = {}
    try:
        ml_models = importlib.import_module("utils.ml_models")
        helpers["ml_models_available"] = True
        helpers["ensure_price_direction"] = getattr(ml_models, "ensure_price_direction", None)
        helpers["prepare_training_data_from_enriched"] = getattr(ml_models, "prepare_training_data_from_enriched", None)
    except Exception as exc:
        helpers["ml_models_available"] = False
        helpers["ml_models_error"] = str(exc)

    try:
        data_processing = importlib.import_module("utils.data_processing")
        helpers["data_processing_available"] = True
        helpers["SetaFeatureEnricher"] = getattr(data_processing, "SetaFeatureEnricher", None)
    except Exception as exc:
        helpers["data_processing_available"] = False
        helpers["data_processing_error"] = str(exc)

    return helpers


def _label_bridge_frame(df: pd.DataFrame, helpers: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    frame = df.copy()
    if "price_direction" in frame.columns and frame["price_direction"].notna().any():
        frame["price_direction"] = pd.to_numeric(frame["price_direction"], errors="coerce")
        return frame

    if helpers and callable(helpers.get("ensure_price_direction")):
        try:
            return helpers["ensure_price_direction"](frame, price_col="current_price", label_col="price_direction")
        except Exception as exc:
            logger.debug("Shared label helper failed, falling back to local label generation: %s", exc)

    if "current_price" not in frame.columns:
        raise ValueError("Legacy bridge requires a current_price column to generate labels")

    future_close = frame["current_price"].shift(-1)
    frame["future_close"] = future_close
    frame["future_return"] = (future_close - frame["current_price"]) / frame["current_price"].replace(0, np.nan)
    frame["price_direction"] = np.where(future_close.notna(), (future_close > frame["current_price"]).astype(float), np.nan)
    return frame


def _local_prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    frame = df.copy()
    if "price_direction" not in frame.columns:
        raise ValueError("price_direction column required for shared feature preparation")
    frame = frame.dropna(subset=["price_direction"]).copy()
    if frame.empty:
        raise ValueError("No labeled rows available for legacy feature preparation")

    y = frame["price_direction"].astype(int).copy()
    X = frame.drop(columns=["price_direction"], errors="ignore")
    X = X.drop(columns=[c for c in ["date", "timestamp", "last_updated", "ath_date", "atl_date"] if c in X.columns], errors="ignore")
    if "ticker" in X.columns or "symbol" in X.columns:
        dummy_columns = [c for c in ["ticker", "symbol"] if c in X.columns]
        X = pd.get_dummies(X, columns=dummy_columns, prefix=dummy_columns, dummy_na=False)
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    X = X.select_dtypes(include=[np.number, "bool"]).astype(float)
    return X, y


def _prepare_shared_feature_view(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any], bool]:
    helpers = _load_shared_helpers()
    labeled = _label_bridge_frame(df, helpers)

    shared_helper_used = False
    shared_feature_frame: Optional[pd.DataFrame] = None
    shared_target: Optional[pd.Series] = None

    if callable(helpers.get("prepare_training_data_from_enriched")):
        try:
            shared_feature_frame, shared_target = helpers["prepare_training_data_from_enriched"](labeled)
            shared_helper_used = True
        except Exception as exc:
            logger.debug("Shared ml_models helper unavailable, falling back to local preparation: %s", exc)

    if shared_feature_frame is None or shared_target is None:
        shared_feature_frame, shared_target = _local_prepare_training_data(labeled)

    return shared_feature_frame, shared_target, helpers, shared_helper_used


def _reuse_pipeline_models(modeling: Optional[ModelingResult]) -> Tuple[List[Dict[str, Any]], bool, str, str, bool, float, str]:
    if modeling is None:
        return [], False, "compatibility_only", "n/a", False, 0.0, "n/a"

    model_items: List[Dict[str, Any]] = []
    for item in modeling.models:
        model_dump = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        model_dump["source"] = "pipeline_modeling"
        model_dump["bridge_mode"] = "reused_trained_models"
        model_items.append(model_dump)

    confidence = float(max(modeling.ensemble_probability, 1 - modeling.ensemble_probability))
    modeling_health = "stable" if not modeling.disagreement and confidence >= 0.60 else "mixed"
    return model_items, True, "pipeline_trained", modeling.selected_model, modeling.disagreement, confidence, modeling_health


def build_legacy_btc_analysis(
    raw_df: pd.DataFrame,
    run_dir: Path,
    asset: str,
    language: str = "en",
    source_comparison: Optional[Dict[str, Any]] = None,
    cleaning: Optional[CleaningResult] = None,
    modeling: Optional[ModelingResult] = None,
) -> Dict[str, Any]:
    requested_asset = (asset or "BTC").upper().strip()
    normalized_asset = _normalize_asset(requested_asset)
    legacy_dir = run_dir / "legacy_btc"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    feed_df = _load_bridge_feed(raw_df, cleaning)
    selected_frame, matched_asset, aliases = _resolve_asset_frame(feed_df, requested_asset)
    if matched_asset is None and cleaning is not None:
        fallback_frame, fallback_asset, _ = _resolve_asset_frame(raw_df, requested_asset)
        if fallback_asset is not None:
            selected_frame = fallback_frame
            matched_asset = fallback_asset

    if selected_frame.empty:
        raise ValueError(f"Legacy bridge could not resolve asset feed for {requested_asset}")

    legacy_frame = _coerce_legacy_frame(selected_frame, requested_asset)
    legacy_frame = _enrich_legacy_features(legacy_frame)
    shared_feature_frame, shared_target, helpers, shared_helper_used = _prepare_shared_feature_view(legacy_frame)

    model_items, model_reuse, model_source, selected_model, disagreement, confidence, modeling_health = _reuse_pipeline_models(modeling)
    ensemble_probability = float(modeling.ensemble_probability) if modeling else float(shared_target.mean()) if len(shared_target) else 0.0
    ensemble_prediction = modeling.ensemble_prediction if modeling else ("long" if ensemble_probability >= 0.5 else "short")
    majority_prediction = modeling.majority_prediction if modeling else ("long" if int(shared_target.sum()) >= max(len(shared_target) / 2, 1) else "short")

    label_column = "target_direction" if "target_direction" in legacy_frame.columns else "price_direction"
    clean_rows = int(legacy_frame[label_column].notna().sum()) if label_column in legacy_frame.columns else int(len(legacy_frame))
    if cleaning is not None:
        feature_reference = list(cleaning.feature_columns)
    else:
        feature_reference = list(shared_feature_frame.columns)
    shared_columns = [column for column in feature_reference if column in legacy_frame.columns or column in shared_feature_frame.columns]
    schema_alignment_pct = round((len(shared_columns) / len(feature_reference) * 100.0), 2) if feature_reference else 0.0

    trust_evidence = {
        "asset_alias_matched": matched_asset is not None,
        "cleaning_feed_used": cleaning is not None,
        "pipeline_models_reused": model_reuse,
        "holdout_label_preserved": bool(
            legacy_frame[label_column].isna().iloc[-1]
        )
        if label_column in legacy_frame.columns and not legacy_frame.empty
        else False,
        "schema_alignment_ok": schema_alignment_pct >= 70.0,
    }
    trust_score_pct = round((sum(1 for value in trust_evidence.values() if value) / len(trust_evidence)) * 100.0, 2)

    if model_items:
        feature_columns = list(cleaning.feature_columns) if cleaning is not None else list(shared_feature_frame.columns)
    else:
        feature_columns = list(shared_feature_frame.columns)

    rationale = (
        "Legacy bridge reused the pipeline-trained models on the same asset feed."
        if language == "en"
        else "El puente legacy reutilizó los modelos entrenados por el pipeline sobre el mismo feed del activo."
    )
    bridge_note = (
        "Legacy bridge reads the cleaned asset feed when available, normalizes BTC/ETH/LTC-style aliases, and reuses the pipeline-trained model outputs instead of training standalone models."
        if language == "en"
        else "El puente legacy lee el feed limpio cuando está disponible, normaliza alias tipo BTC/ETH/LTC y reutiliza los modelos entrenados por el pipeline en lugar de entrenar modelos aparte."
    )
    if shared_helper_used:
        bridge_note += (
            " Shared feature preparation from utils.ml_models was available and used."
            if language == "en"
            else " La preparación compartida de features desde utils.ml_models estuvo disponible y se usó."
        )

    summary = {
        "enabled": True,
        "comparison_mode": "legacy_bridge",
        "requested_asset": requested_asset,
        "asset": normalized_asset,
        "matched_asset": matched_asset,
        "asset_aliases": aliases,
        "rows": int(len(legacy_frame)),
        "clean_rows": clean_rows,
        "feature_columns": feature_columns,
        "shared_feature_columns": list(shared_feature_frame.columns),
        "latest_sample_count": int(modeling.latest_sample_count) if modeling else int(len(shared_feature_frame.tail(1))),
        "models": model_items,
        "ensemble_probability": ensemble_probability,
        "ensemble_prediction": ensemble_prediction,
        "majority_prediction": majority_prediction,
        "disagreement": disagreement,
        "selected_model": selected_model,
        "confidence": confidence,
        "modeling_health": modeling_health,
        "model_source": model_source,
        "source_comparison": source_comparison or {},
        "rationale": rationale,
        "bridge_note": bridge_note,
        "trust_evidence": trust_evidence,
        "trust_score_pct": trust_score_pct,
        "schema_alignment_pct": schema_alignment_pct,
        "shared_helpers": {
            "ml_models_available": bool(helpers.get("ml_models_available")),
            "data_processing_available": bool(helpers.get("data_processing_available")),
            "shared_ml_models_helper_used": shared_helper_used,
        },
        "target_column": label_column,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = legacy_dir / "legacy_bridge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    summary["artifact"] = ArtifactRef(
        name=summary_path.name,
        path=str(summary_path),
        kind="legacy_summary",
        size_bytes=summary_path.stat().st_size,
    ).model_dump()
    return summary
