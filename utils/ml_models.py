"""Legacy crypto model helpers and compatibility endpoints.

The active pipeline trains models in `agents/modeling_agent.py`. This module is
retained for legacy references and optional bridge helpers.
"""

from sched import scheduler
from typing import List
from flask import app
from matplotlib import _preprocess_data
from scripts.create_dummy_model import X
from scripts.trains_models import retrain_job
from schemas.prediction_trade_schemas import PredictionLogCreate
from schemas.schemas import validate_live_features
from utils.model_io import load_model_and_metadata, save_model_and_metadata
from services.paper_trading import PaperTrader
from fastapi import APIRouter, Depends, HTTPException, requests
from sqlalchemy.orm import Session
from utils.walkforward import walk_forward_cv
from models.calibrated_models import load_or_train_model
from api.crypto import PREDICTIONS_CSV_FILE_PATH
from api import events
from utils import database
from sklearn.linear_model import LinearRegression
import pandas as pd
from services import event_service, crypto_service
import logging
from models.prediction_trade_models import PredictionLog
from schemas.prediction_trade_schemas import PredictionLog as PredictionLogSchema

router = APIRouter()   # give it a prefix

CSV_FILE = "crypto_data_predictionURL.csv"
BINANCE_24HR_URL = "https://api.binance.com/api/v3/ticker/24hr"

logger = logging.getLogger(__name__)

def fetch_and_store_data(db: Session = Depends(database.get_db)):
    try:
        crypto_service_instance = crypto_service.CryptoService(db)
        response = crypto_service_instance.fetch_crypto_data_from_api(BINANCE_24HR_URL, db)
        data = response

        df = pd.DataFrame(data)
        df = df[['current_price', 'total_volume', 'symbol']]
        df.columns = ['feature1', 'target', 'symbol']
        df.to_csv(CSV_FILE, index=False)
        logger.info(f"Data successfully fetched and stored in {CSV_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error fetching and storing data: {e}")
        return False

def train_model(symbol, db: Session = Depends(database.get_db)):
    """
    Trains a linear regression model from a CSV file for a specific coin.
    Returns the trained model (in-memory).
    """
    try:
        if not fetch_and_store_data(db):
            raise HTTPException(status_code=500, detail="Error fetching and storing data")

        data = pd.read_csv(CSV_FILE)
        logger.info(f"Data successfully loaded from {CSV_FILE}")

        data = data[data['symbol'] == symbol]
        data = data.dropna()
        X = data[['feature1']]
        y = data['target']

        model = LinearRegression()
        model.fit(X, y)
        logger.info(f"Model successfully trained for {symbol}")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")

@router.get("/predict_price/")
def predict_price(db: Session = Depends(database.get_db)):
    """
    Endpoint to predict the price based on the trained model for each coin.
    """
    try:
        crypto_service_instance = crypto_service.CryptoService(db)
        response = crypto_service_instance.fetch_crypto_data_from_api(BINANCE_24HR_URL, db)
        coins_data = response

        predictions = {}
        predictions_list = []
        for coin_data in coins_data:
            symbol = coin_data['symbol']
            current_price = coin_data['current_price']

            # Train model in-memory for this symbol
            model = train_model(symbol, db)

            input_data = pd.DataFrame([[current_price]], columns=['feature1'])
            prediction = model.predict(input_data)[0]

            predictions[symbol] = prediction
            predictions_list.append({'symbol': symbol, 'predicted_price': prediction})

            event_service.create_event(events.EventCreate(
                event_name="PRICE_PREDICTION",
                event_value=prediction,
                event_message=f"Price prediction made for {symbol}",
                event_details={"prediction": prediction, "symbol": symbol}
            ), db)

        # Optionally, save predictions to CSV
        predictions_df = pd.DataFrame(predictions_list)
        predictions_df.to_csv("predictions.csv", index=False)

        return {"predictions": predictions, "message": "Predictions completed for all coins."}
    except HTTPException as e:
        raise e
    except Exception as e:
        event_service.create_event(events.EventCreate(
            event_name="PREDICTION_ERROR",
            event_value=1,
            event_message=f"Error during prediction: {e}",
            event_details={"error": str(e)}
        ), db)
        raise HTTPException(status_code=500, detail="Error during prediction")
    
@router.get("/csv/", summary="Download the prediction CSV")
def download_csv(db: Session = Depends(database.get_db)):
    try:
        from fastapi.responses import FileResponse
        # Serve the predictions CSV, not the raw data CSV
        return FileResponse("predictions.csv", media_type="text/csv", filename="predictions.csv")
    except Exception as e:
        logger.error("Could not serve CSV: %s", e)
        raise



import os

from fastapi import APIRouter, HTTPException
import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

MODEL_DIR = os.getenv("MODEL_DIR", "models")
CALIBRATED_MODEL_PATH = os.path.join(MODEL_DIR, "calibrated_stacking_model.joblib")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.joblib")

_cached_models: Dict[str, Any] = {"calibrated": None, "rf": None, "dl": None}

CLEAN_FEATURES = [
    "total_volume", "high_24h", "low_24h",
    "price_change_24h", "price_change_percentage_24h",
    "market_cap", "market_cap_change_24h", "market_cap_change_percentage_24h",
    "sma_20", "sma_50", "ema_20", "ema_50", "rsi_14", "macd",
    "bb_upper", "bb_middle", "bb_lower", "atr_14", "obv",
    "return_1", "return_3",
    "rolling_vol_7", "rolling_vol_14",
    "pct_change_1", "pct_change_3",
    "price_lag_1", "price_lag_3", "price_lag_5", "price_lag_7",
    "btc_correlation", "eth_correlation", "market_correlation",
    "hour", "day_of_week", "is_weekend"
]

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    logger.warning("XGBoost not available, falling back where necessary.")

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
    logger.warning("CatBoost not available, falling back where necessary.")

def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    sample = pd.read_csv(csv_path, nrows=0)
    possible_date_cols = [c for c in ["timestamp", "date", "time", "last_updated", "datetime"] if c in sample.columns]
    try:
        df = pd.read_csv(csv_path, parse_dates=possible_date_cols, low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False)
    return df

def ensure_price_direction(df: pd.DataFrame, price_col: str = "current_price", label_col: str = "price_direction") -> pd.DataFrame:
    if label_col in df.columns:
        return df
    if price_col not in df.columns:
        raise ValueError(f"Cannot generate {label_col}: '{price_col}' missing from dataframe")
    df = df.copy()
    ts_col = None
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        ts_col = "timestamp"
    else:
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower() or "datetime" in c.lower():
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    ts_col = c
                    break
                except Exception:
                    continue
    if ts_col is not None:
        df = df.sort_values(ts_col).reset_index(drop=True)
    future_price = df[price_col].shift(-1)
    df[label_col] = np.where(future_price.notna(), (future_price > df[price_col]).astype(float), np.nan)
    return df

def prepare_training_data_from_enriched(df_enriched: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    if drop_cols is None:
        drop_cols = ["name", "symbol", "image", "coin_id", "timestamp"]
    df = df_enriched.copy()
    if "price_direction" not in df.columns:
        raise ValueError("price_direction column required for training")
    df = df.dropna(subset=["price_direction"]).copy()
    if df.empty:
        raise ValueError("No labeled rows remain after dropping rows with missing price_direction")
    y = df["price_direction"].astype(int).copy()
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["price_direction"], errors="ignore")
    for col in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[f"{col}_year"] = X[col].dt.year.fillna(0).astype(int)
            X[f"{col}_month"] = X[col].dt.month.fillna(0).astype(int)
            X[f"{col}_day"] = X[col].dt.day.fillna(0).astype(int)
            X.drop(columns=[col], inplace=True)
    obj_cols = [c for c in X.columns if X[c].dtype == "object"]
    if obj_cols:
        logger.debug("Dropping object columns before numeric selection: %s", obj_cols)
        X = X.drop(columns=obj_cols)
    X_numeric = X.select_dtypes(include=[np.number, "bool"]).fillna(0).astype(np.float32)
    return X_numeric, y

def load_model_safe(path: str):
    if not os.path.exists(path):
        logger.debug("Model path missing: %s", path)
        return None
    try:
        return joblib.load(path)
    except Exception:
        logger.exception("Failed to load model %s", path)
        return None

def normalize_weights_for_available(weights: Optional[List[float]], available_flags: Dict[str, bool]) -> List[float]:
    names = ["rf", "dl", "seta"]
    available = [available_flags.get(n, False) for n in names]
    if weights is None:
        if any(available):
            count = sum(available)
            return [1.0/count if avail else 0.0 for avail in available]
        else:
            return [0.0, 0.0, 0.0]
    if len(weights) != 3 or any((w < 0) or (w > 1) for w in weights) or not np.isclose(sum(weights), 1.0):
        raise HTTPException(status_code=400, detail="Weights must be a list of three floats summing to 1.0")
    weights = list(map(float, weights))
    total = sum(w for w, avail in zip(weights, available) if avail)
    if total == 0:
        if any(available):
            count = sum(available)
            return [1.0/count if avail else 0.0 for avail in available]
        else:
            return [0.0, 0.0, 0.0]
    normalized = [ (w / total) if avail else 0.0 for w, avail in zip(weights, available) ]
    return normalized

def train_full_model_core(df_raw: pd.DataFrame, model_dir: str = MODEL_DIR, save_path: Optional[str] = None) -> Dict[str, Any]:
    if save_path is None:
        save_path = CALIBRATED_MODEL_PATH
    df_raw = ensure_price_direction(df_raw, price_col="current_price", label_col="price_direction")
    try:
        from api.data_processing import SetaFeatureEnricher
        enricher = SetaFeatureEnricher()
        df_enriched = enricher.enrich_features(df_raw)
        enricher_version = getattr(enricher, "__version__", None)
    except Exception as e:
        logger.warning("SetaFeatureEnricher not available or failed: %s — proceeding with raw df", e)
        df_enriched = df_raw.copy()
        enricher_version = None
    X, y = prepare_training_data_from_enriched(df_enriched)
    if X.shape[0] < 10:
        raise ValueError("Not enough rows to train")
    if y.nunique() < 2:
        raise ValueError("Target 'price_direction' must contain at least two classes for training")
    stratify_arg = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)
    estimators = []
    if XGBClassifier is not None:
        xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        xgb.fit(X_train, y_train)
        estimators.append(("xgb", xgb))
    else:
        logger.info("XGB not available; skipping XGB model")
    if CatBoostClassifier is not None:
        cat = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=8, verbose=0, random_state=42)
        cat.fit(X_train, y_train)
        estimators.append(("cat", cat))
    else:
        logger.info("CatBoost not available; skipping CatBoost")
    if len(estimators) == 0:
        rf_base = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_base.fit(X_train, y_train)
        estimators.append(("rf_base", rf_base))
    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5)
    stacking.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(stacking, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    y_pred = calibrated.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    logger.info("Trained calibrated stacking. test acc=%.4f", acc)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(calibrated, save_path)
    meta = {
        "feature_columns": list(X.columns),
        "train_rows": int(X.shape[0]),
        "enricher_version": enricher_version
    }
    with open(save_path + ".meta.json", "w") as f:
        json.dump(meta, f)
    _cached_models["calibrated"] = calibrated
    return {"calibrated_stacking": True, "accuracy": acc, "feature_columns": list(X.columns), "meta": meta}

def train_clean_model_core(df_raw: pd.DataFrame, model_dir: str = MODEL_DIR, save_path: Optional[str] = None) -> Dict[str, Any]:
    if save_path is None:
        save_path = RF_MODEL_PATH
    df_raw = ensure_price_direction(df_raw, price_col="current_price", label_col="price_direction")
    try:
        from api.data_processing import SetaFeatureEnricher
        enricher = SetaFeatureEnricher()
        df_enriched = enricher.enrich_features(df_raw)
    except Exception:
        df_enriched = df_raw.copy()
    cols_available = [c for c in CLEAN_FEATURES if c in df_enriched.columns]
    if not cols_available:
        raise ValueError("No clean features present in data")
    df_clean = df_enriched[cols_available + ["price_direction"]].dropna()
    X = df_clean[cols_available].astype(float).fillna(0)
    y = df_clean["price_direction"].astype(int)
    if X.shape[0] < 10:
        raise ValueError("Not enough rows to train clean model")
    result = train_and_evaluate(X, y)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(result["model"], save_path)
    _cached_models["rf"] = result["model"]
    return {"rf_trained": True, "accuracy": float(result["accuracy"]), "features_used": cols_available}

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    return {"accuracy": acc, "model": model}

@router.post("/crypto/train-model")
def train_model(csv_path: str = "data/crypto_data.csv", model_dir: str = MODEL_DIR, save_path: Optional[str] = None):
    try:
        df_raw = load_csv(csv_path)
        res = train_full_model_core(df_raw, model_dir=model_dir, save_path=save_path)
        return {"message": "Models trained", "accuracy": res["accuracy"], "features": res["feature_columns"], "meta": res["meta"]}
    except Exception as e:
        logger.exception("train_model error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-model-clean")
def train_model_clean(csv_path: str = "data/crypto_data.csv"):
    try:
        df_raw = load_csv(csv_path)
        res = train_clean_model_core(df_raw, model_dir=MODEL_DIR)
        return {"message": "Clean model trained", "accuracy": res["accuracy"], "features_used": res["features_used"]}
    except Exception as e:
        logger.exception("train_model_clean error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crypto/predict-ensemble")
async def predict_ensemble(data: List[dict], weights: Optional[List[float]] = None):
    try:
        df = pd.DataFrame(data)
        try:
            from api.data_processing import SetaFeatureEnricher
            enricher = SetaFeatureEnricher()
            enriched = enricher.enrich_features(df)
        except Exception:
            enriched = df.copy()
        meta_path = CALIBRATED_MODEL_PATH + ".meta.json"
        if os.path.exists(meta_path):
            feature_columns = json.load(open(meta_path)).get("feature_columns", [])
        else:
            feature_columns = [c for c in enriched.columns if c in CLEAN_FEATURES]
            if not feature_columns:
                feature_columns = list(enriched.select_dtypes(include=[np.number]).columns)
        features = enriched.reindex(columns=feature_columns).fillna(0)
        n = len(features)
        rf_probs = np.zeros(n)
        dl_probs = np.zeros(n)
        seta_probs = np.zeros(n)
        rf = _cached_models.get("rf") or (load_model_safe(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None)
        dl = _cached_models.get("dl") or (load_model_safe(DL_MODEL_PATH) if os.path.exists(DL_MODEL_PATH) else None)
        seta = _cached_models.get("calibrated") or (load_model_safe(CALIBRATED_MODEL_PATH) if os.path.exists(CALIBRATED_MODEL_PATH) else None)
        _cached_models["rf"] = rf
        _cached_models["dl"] = dl
        _cached_models["calibrated"] = seta
        if rf is not None:
            try:
                rf_probs = rf.predict_proba(features)[:, 1]
            except Exception:
                logger.warning("RF predict_proba failed; using zeros for rf_probs")
                rf_probs = np.zeros(n)
        if dl is not None:
            try:
                dl_probs = dl.predict_proba(features)[:, 1]
            except Exception:
                logger.warning("DL predict_proba failed; using zeros for dl_probs")
                dl_probs = np.zeros(n)
        if seta is not None:
            try:
                seta_probs = seta.predict_proba(features)[:, 1]
            except Exception:
                logger.warning("SETA predict_proba failed; using zeros for seta_probs")
                seta_probs = np.zeros(n)
        flags = {"rf": rf is not None, "dl": dl is not None, "seta": seta is not None}
        weights = normalize_weights_for_available(weights, flags)
        combined_probs = weights[0] * rf_probs + weights[1] * dl_probs + weights[2] * seta_probs
        results = []
        for i, (_, row) in enumerate(enriched.iterrows()):
            prob_up = float(combined_probs[i])
            decision = "BUY" if prob_up > 0.7 else ("SELL" if prob_up < 0.3 else "HOLD")
            results.append({
                "symbol": row.get("symbol", f"asset_{i}"),
                "current_price": float(row.get("current_price", 0.0) or 0.0),
                "probability_up": prob_up,
                "decision": decision
            })
        return {"predictions": results, "weights": weights}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("predict_ensemble error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



CSV_FILE = "crypto_data.csv"  # Change this line to use crypto_data.csv

def fetch_and_store_data(db: Session = Depends(database.get_db)):
    try:
        crypto_service_instance = crypto_service.CryptoService(db)
        response = crypto_service_instance.fetch_crypto_data_from_api(BINANCE_24HR_URL, db)
        data = response
        df = pd.DataFrame(data)
        df = df[['current_price', 'total_volume', 'symbol']]
        df.columns = ['feature1', 'target', 'symbol']
        df.to_csv(CSV_FILE, index=False)
        logger.info(f"Data successfully fetched and stored in {CSV_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error fetching and storing data: {e}")
        return False


@router.post("/crypto/tune-ensemble")
def tune_ensemble(step: float = 0.1, csv_path: str = "data/crypto_data.csv"):
    try:
        df_raw = load_csv(csv_path)
        df_raw = ensure_price_direction(df_raw, price_col="current_price", label_col="price_direction")
        try:
            from api.data_processing import SetaFeatureEnricher
            enricher = SetaFeatureEnricher()
            df_enriched = enricher.enrich_features(df_raw)
        except Exception:
            df_enriched = df_raw.copy()
        X = df_enriched.drop(columns=["price_direction"], errors="ignore").select_dtypes(include=[np.number]).fillna(0)
        y = df_enriched["price_direction"].astype(int).values
        n = X.shape[0]
        rf_probs = np.zeros(n)
        dl_probs = np.zeros(n)
        seta_probs = np.zeros(n)
        rf = _cached_models.get("rf") or (load_model_safe(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None)
        dl = _cached_models.get("dl") or (load_model_safe(DL_MODEL_PATH) if os.path.exists(DL_MODEL_PATH) else None)
        seta = _cached_models.get("calibrated") or (load_model_safe(CALIBRATED_MODEL_PATH) if os.path.exists(CALIBRATED_MODEL_PATH) else None)
        if rf is not None:
            try:
                rf_probs = rf.predict_proba(X)[:, 1]
            except Exception:
                logger.warning("RF predict_proba failed during tuning; using zeros")
        if dl is not None:
            try:
                dl_probs = dl.predict_proba(X)[:, 1]
            except Exception:
                logger.warning("DL predict_proba failed during tuning; using zeros")
        if seta is not None:
            try:
                seta_probs = seta.predict_proba(X)[:, 1]
            except Exception:
                logger.warning("SETA predict_proba failed during tuning; using zeros")
        vals = np.arange(0.0, 1.0 + 1e-9, step)
        best = {"weights": (0.0, 0.0, 1.0), "accuracy": -1.0}
        for w0 in vals:
            for w1 in vals:
                w2 = 1.0 - w0 - w1
                if w2 < 0 or w2 > 1:
                    continue
                combined = w0 * rf_probs + w1 * dl_probs + w2 * seta_probs
                preds = (combined >= 0.5).astype(int)
                acc = float((preds == y).mean())
                if acc > best["accuracy"]:
                    best["accuracy"] = acc
                    best["weights"] = (float(w0), float(w1), float(w2))
        return {"best_weights": best["weights"], "accuracy": best["accuracy"], "step": step}
    except Exception as e:
        logger.exception("tune_ensemble error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

from sklearn.inspection import permutation_importance

from services.feature_importance import compute_feature_importance
##HERE
@router.get("/crypto/feature-importance")
def feature_importance(top_n: int = 30, csv_path: str = "data/crypto_data.csv"):
    # Prepare df, X, y as in other endpoints
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Training CSV not found")
    df_raw = load_csv(csv_path)
    df_raw = ensure_price_direction(df_raw, price_col="current_price", label_col="price_direction")
    try:
        from api.data_processing import SetaFeatureEnricher
        enricher = SetaFeatureEnricher()
        df_enriched = enricher.enrich_features(df_raw)
    except Exception:
        df_enriched = df_raw.copy()
    meta_path = CALIBRATED_MODEL_PATH + ".meta.json"
    if os.path.exists(meta_path):
        feature_columns = json.load(open(meta_path)).get("feature_columns", [])
    else:
        feature_columns = list(df_enriched.select_dtypes(include=[np.number]).columns)
    X = df_enriched.reindex(columns=feature_columns).fillna(0)
    y = df_enriched["price_direction"].astype(int)
    model = load_model_safe(CALIBRATED_MODEL_PATH) or _cached_models.get("calibrated")
    if model is None:
        raise HTTPException(status_code=500, detail="Calibrated model not available")

    importance_df, method = compute_feature_importance(model, X, y, top_n)
    return {
        "feature_importance": importance_df.to_dict(orient="records"),
        "method": method
    }

@router.get("/crypto/threshold-stats")
def threshold_stats(threshold: float = 0.7, csv_path: str = "data/crypto_data.csv"):
    try:
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Training CSV not found")
        df_raw = load_csv(csv_path)
        df_raw = ensure_price_direction(df_raw, price_col="current_price", label_col="price_direction")
        try:
            from api.data_processing import SetaFeatureEnricher
            enricher = SetaFeatureEnricher()
            df_enriched = enricher.enrich_features(df_raw)
        except Exception:
            df_enriched = df_raw.copy()
        meta_path = CALIBRATED_MODEL_PATH + ".meta.json"
        if os.path.exists(meta_path):
            feature_columns = json.load(open(meta_path)).get("feature_columns", [])
        else:
            feature_columns = list(df_enriched.select_dtypes(include=[np.number]).columns)
        X = df_enriched.reindex(columns=feature_columns).fillna(0)
        y = df_enriched["price_direction"].astype(int)
        model = load_model_safe(CALIBRATED_MODEL_PATH) or _cached_models.get("calibrated")
        if model is None:
            raise HTTPException(status_code=500, detail="Calibrated model not available")
        probs = model.predict_proba(X)[:, 1]
        mask = probs >= threshold
        covered = int(mask.sum())
        coverage = float(covered / len(X)) if len(X) else 0.0
        hit_rate = float((y.iloc[mask] == 1).mean()) if covered > 0 else None
        return {"threshold": threshold, "coverage": coverage, "hit_rate": hit_rate, "sample_count": covered}
    except Exception as e:
        logger.exception("threshold_stats error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

class PaperTrader:
    def __init__(self, starting_capital=1000.0, confidence_threshold=0.7):
        self.capital = starting_capital
        self.starting_capital = starting_capital
        self.confidence_threshold = confidence_threshold
        self.position = 0.0
        self.entry_price = None
        self.trade_log = []

    def buy(self, price, confidence):
        if self.position != 0:
            return
        size = self.capital * min(0.5, max(0.05, (confidence - 0.5) * 2))
        qty = size / price if price > 0 else 0
        self.position = qty
        self.entry_price = price
        self.trade_log.append({"action": "BUY", "price": price, "qty": qty, "confidence": confidence})

    def sell(self, price):
        if self.position == 0:
            return
        profit = (price - self.entry_price) * self.position
        self.capital += profit
        self.trade_log.append({"action": "SELL", "price": price, "qty": self.position, "profit": profit})
        self.position = 0.0
        self.entry_price = None

    def get_results(self):
        final = self.capital
        return_pct = ((final - self.starting_capital) / self.starting_capital) * 100
        return {"initial_capital": self.starting_capital, "final_capital": final, "return_pct": return_pct, "trades": self.trade_log}

@router.post("/crypto/backtest")
def backtest_predictions(data: List[dict], starting_capital: float = 1000.0, confidence_threshold: float = 0.7):
    try:
        df = pd.DataFrame(data)
        if "current_price" not in df.columns:
            raise HTTPException(status_code=400, detail="Input must include 'current_price'")
        try:
            from api.data_processing import SetaFeatureEnricher
            enricher = SetaFeatureEnricher()
            enriched = enricher.enrich_features(df)
        except Exception:
            enriched = df.copy()
        meta_path = CALIBRATED_MODEL_PATH + ".meta.json"
        if os.path.exists(meta_path):
            feature_columns = json.load(open(meta_path)).get("feature_columns", [])
        else:
            feature_columns = [c for c in enriched.columns if np.issubdtype(enriched[c].dtype, np.number)]
        features = enriched.reindex(columns=feature_columns).fillna(0)
        model = load_model_safe(CALIBRATED_MODEL_PATH) or _cached_models.get("calibrated")
        if model is None:
            raise HTTPException(status_code=500, detail="Calibrated model not available")
        probs = model.predict_proba(features)[:, 1]
        trader = PaperTrader(starting_capital=starting_capital, confidence_threshold=confidence_threshold)
        for i, row in enriched.iterrows():
            price = float(row.get("current_price", 0.0) or 0.0)
            prob = float(probs[i])
            if prob > confidence_threshold and trader.position == 0:
                trader.buy(price, prob)
            elif prob < (1 - confidence_threshold) and trader.position != 0:
                trader.sell(price)
        results = trader.get_results()
        return {"backtest": results}
    except Exception as e:
        logger.exception("backtest error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

def threshold_curve(probs: np.ndarray, y_true: np.ndarray, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.95, 10)
    rows = []
    for t in thresholds:
        mask = probs >= t
        covered = int(mask.sum())
        coverage = covered / len(probs) if len(probs) else 0.0
        hit = float((y_true[mask] == 1).mean()) if covered > 0 else None
        rows.append({"threshold": float(t), "coverage": float(coverage), "hit_rate": hit, "sample_count": covered})
    return rows

def reliability_curve(probs: np.ndarray, y_true: np.ndarray, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            rows.append({"bin_low": float(bins[b]), "bin_high": float(bins[b + 1]), "avg_prob": None, "empirical_acc": None, "count": 0})
            continue
        avg_prob = float(probs[mask].mean())
        emp_acc = float((y_true[mask] == 1).mean())
        rows.append({"bin_low": float(bins[b]), "bin_high": float(bins[b + 1]), "avg_prob": avg_prob, "empirical_acc": emp_acc, "count": int(mask.sum())})
    return rows

@router.get("/crypto/diagnostics")
def diagnostics(csv_path: str = "data/crypto_data.csv", n_bins: int = 10):
    try:
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="CSV not found")
        df_raw = load_csv(csv_path)
        df_raw = ensure_price_direction(df_raw, price_col="current_price", label_col="price_direction")
        try:
            from api.data_processing import SetaFeatureEnricher
            enricher = SetaFeatureEnricher()
            df_enriched = enricher.enrich_features(df_raw)
        except Exception:
            df_enriched = df_raw.copy()
        meta_path = CALIBRATED_MODEL_PATH + ".meta.json"
        if os.path.exists(meta_path):
            feature_columns = json.load(open(meta_path)).get("feature_columns", [])
        else:
            feature_columns = list(df_enriched.select_dtypes(include=[np.number]).columns)
        X = df_enriched.reindex(columns=feature_columns).fillna(0)
        y = df_enriched["price_direction"].astype(int).values
        model = load_model_safe(CALIBRATED_MODEL_PATH) or _cached_models.get("calibrated")
        if model is None:
            raise HTTPException(status_code=500, detail="Calibrated model not available")
        probs = model.predict_proba(X)[:, 1]
        threshold_data = threshold_curve(probs, y)
        reliability_data = reliability_curve(probs, y, n_bins=n_bins)
        overall_acc = float(((probs >= 0.5).astype(int) == y).mean())
        high_mask = probs >= 0.7
        high_acc = float((y[high_mask] == 1).mean()) if high_mask.sum() else None
        high_cov = float(high_mask.mean())
        return {
            "overall_accuracy_at_0_5": overall_acc,
            "high_conf_accuracy_at_0_7": high_acc,
            "high_conf_coverage": high_cov,
            "threshold_curve": threshold_data,
            "reliability_curve": reliability_data
        }
    except Exception as e:
        logger.exception("diagnostics error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crypto/train-and-compare")
def train_and_compare(csv_path: str = "data/crypto_data.csv"):
    try:
        df_raw = load_csv(csv_path)
        full_res = train_full_model_core(df_raw, model_dir=MODEL_DIR)
        clean_res = train_clean_model_core(df_raw, model_dir=MODEL_DIR)
        return {"full": full_res, "clean": clean_res}
    except Exception as e:
        logger.exception("train_and_compare error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))





from fastapi import APIRouter, HTTPException, Query
import os
import json
import datetime
import logging
import requests
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from api.data_processing import SetaFeatureEnricher

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
router = APIRouter()

# Local model paths (avoid circular import)
MODEL_DIR = os.getenv("MODEL_DIR", "models")
CALIBRATED_MODEL_PATH = os.path.join(MODEL_DIR, "calibrated_stacking_model.joblib")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
CAT_MODEL_PATH = os.path.join(MODEL_DIR, "cat_model.joblib")

_cached_models = {
    "rf": None,
    "dl": None,
    "xgb": None,
    "cat": None,
    "calibrated": None,
    "meta": None,
}

# -----------------------------------------------------------------------------
# LIVE FETCH HELPERS
# -----------------------------------------------------------------------------
def _safe_get(url, timeout=10, params=None):
    r = requests.get(url, timeout=timeout, params=params or {})
    r.raise_for_status()
    return r


def fetch_legacy_snapshot(symbol_id="BTCUSDT"):
    r = _safe_get(BINANCE_24HR_URL, params={"symbol": symbol_id})
    data = r.json().get("market_data", {})
    return {
        "price": float(data.get("lastPrice", 0.0)),
        "high_24h": float(data.get("highPrice", 0.0)),
        "low_24h": float(data.get("lowPrice", 0.0)),
        "volume": float(data.get("volume", 0.0)),
        "change_24h": float(data.get("priceChangePercent", 0.0)),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


def fetch_binance(symbol_pair="BTCUSDT"):
    r = _safe_get("https://api.binance.com/api/v3/ticker/24hr", params={"symbol": symbol_pair})
    data = r.json()
    return {
        "price": float(data.get("lastPrice", 0.0)),
        "high_24h": float(data.get("highPrice", 0.0)),
        "low_24h": float(data.get("lowPrice", 0.0)),
        "volume": float(data.get("volume", 0.0)),
        "change_24h": float(data.get("priceChangePercent", 0.0)),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def _choose_features(df: pd.DataFrame, meta_path: str = None):
    if meta_path and os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            cols = meta.get("feature_columns", [])
            cols = [c for c in cols if c in df.columns]
            if cols:
                return cols
        except Exception as e:
            logger.warning("Failed to read meta feature_columns: %s", e)
    return list(df.select_dtypes(include=[np.number]).columns)


def load_model_safe(path: str):
    if not os.path.exists(path):
        logger.info("Model path does not exist: %s", path)
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        logger.exception("Failed to load model %s: %s", path, e)
        return None


def _ensemble_probs(w, rf_probs, dl_probs, xgb_probs, cat_probs, seta_probs):
    return (
        w[0] * rf_probs
        + w[1] * dl_probs
        + w[2] * xgb_probs
        + w[3] * cat_probs
        + w[4] * seta_probs
    )


def _precision_and_coverage(y_true, probs, thr):
    mask = probs >= thr
    if mask.sum() == 0:
        return None, 0.0, 0
    precision = float((y_true[mask] == 1).mean())
    coverage = float(mask.mean())
    return precision, coverage, int(mask.sum())


def _backtest_window(
    df_enriched: pd.DataFrame,
    probs: np.ndarray,
    threshold: float,
    window_hours: int = 24,
    ret_col: str = "future_return_1h",
):
    if len(df_enriched) != len(probs):
        return {
            "window_hours": window_hours,
            "samples": 0,
            "precision": None,
            "avg_return": None,
        }

    df = df_enriched.copy()
    ts_col = next(
        (c for c in ["timestamp", "last_updated", "time", "date"] if c in df.columns),
        None,
    )

    if ts_col:
        try:
            df["_ts"] = pd.to_datetime(df[ts_col])
            cutoff = df["_ts"].max() - pd.Timedelta(hours=window_hours)
            win = df["_ts"] >= cutoff
        except Exception:
            win = pd.Series([False] * len(df))
    else:
        k = max(10, int(0.2 * len(df)))
        win = pd.Series([False] * (len(df) - k) + [True] * k)

    df["_prob"] = probs
    df["_signal"] = (df["_prob"] >= threshold).astype(int)
    dfw = df[win].copy()

    if dfw.empty:
        return {
            "window_hours": window_hours,
            "samples": 0,
            "precision": None,
            "avg_return": None,
        }

    mask = dfw["_signal"] == 1
    samples = int(mask.sum())
    precision = (
        float((dfw.loc[mask, "price_direction"].astype(int) == 1).mean())
        if samples > 0
        else None
    )
    avg_ret = (
        float(dfw.loc[mask, ret_col].mean())
        if ret_col in dfw.columns and samples > 0
        else None
    )
    return {
        "window_hours": window_hours,
        "samples": samples,
        "precision": precision,
        "avg_return": avg_ret,
    }


def _calibration_summary(y_true, probs):
    try:
        bs = float(brier_score_loss(y_true, probs))
    except Exception as e:
        logger.warning("Brier score computation failed: %s", e)
        return {"brier_score": None, "reliability": []}

    cuts = np.linspace(0, 1, 11)
    out = []
    for i in range(10):
        lo, hi = cuts[i], cuts[i + 1]
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            out.append(
                {
                    "bin": f"[{lo:.1f},{hi:.1f})",
                    "count": 0,
                    "avg_prob": None,
                    "empirical": None,
                }
            )
        else:
            out.append(
                {
                    "bin": f"[{lo:.1f},{hi:.1f})",
                    "count": int(m.sum()),
                    "avg_prob": float(probs[m].mean()),
                    "empirical": float(y_true[m].mean()),
                }
            )
    return {"brier_score": bs, "reliability": out}

# -----------------------------------------------------------------------------
# MAIN ENDPOINT
# -----------------------------------------------------------------------------
@router.get("/crypto/target-precision-meta-live")
def target_precision_meta_live(
    csv_path: str = "data/crypto_data.csv",
    target_precision: float = 0.82,
    weight_step: float = 0.1,
    threshold_min: float = 0.55,
    threshold_max: float = 0.95,
    threshold_step: float = 0.02,
    use_cache: bool = False,
    cache_path: str = "data/enriched_cache.parquet",
    save_meta_path: str = os.path.join(MODEL_DIR, "meta_model.joblib"),
    symbols: str = Query(
        "bitcoin:BTCUSDT,ethereum:ETHUSDT,tether:USDT",
        description="Format: asset:binance_pair, multiple separated by comma",
    ),
    backtest_windows: str = "24,48",
):
    try:
        # ----------- LOAD + ENRICH DATA -----------
        if use_cache and os.path.exists(cache_path):
            df_enriched = pd.read_parquet(cache_path)
        else:
            if not os.path.exists(csv_path):
                raise HTTPException(status_code=404, detail="csv_path not found")

            df_raw = pd.read_csv(csv_path)

            # Enrichment wrapped safely: if it blows up internally (e.g. a .fillna on a str),
            # we fall back to using df_raw directly.
            try:
                enricher = SetaFeatureEnricher()
                df_enriched = enricher.enrich_features(df_raw)
            except Exception as e:
                logger.warning(
                    "SetaFeatureEnricher.enrich_features failed (%s); falling back to raw df",
                    e,
                )
                df_enriched = df_raw.copy()

            # Ensure we actually have a DataFrame
            if not isinstance(df_enriched, pd.DataFrame):
                logger.error(
                    "Enricher returned non-DataFrame (%s); falling back to raw df",
                    type(df_enriched),
                )
                df_enriched = df_raw.copy()

            if use_cache:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                df_enriched.to_parquet(cache_path, index=False)

        # ----------- CLEAN POSSIBLE LIST-CELLS + NAs -----------
        # If some columns hold Python lists, normal ML models will choke on them.
        for col in df_enriched.columns:
            try:
                if df_enriched[col].apply(lambda x: isinstance(x, list)).any():
                    # Simple strategy: take first element or 0
                    df_enriched[col] = df_enriched[col].apply(
                        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0
                    )
            except Exception:
                # If column can't be processed elementwise, skip it
                continue

        df_enriched = df_enriched.fillna(0)

        # ----------- FEATURES + MODELS -----------
        meta_path = CALIBRATED_MODEL_PATH + ".meta.json"
        feature_columns = _choose_features(df_enriched, meta_path)
        X = df_enriched.reindex(columns=feature_columns).fillna(0)

        if "price_direction" not in df_enriched.columns:
            raise HTTPException(
                status_code=400,
                detail="price_direction column missing in enriched data",
            )

        y = df_enriched["price_direction"].astype(int).values
        n = len(X)
        if n == 0:
            raise HTTPException(
                status_code=400, detail="No numeric rows after processing"
            )

        # Load models (safe)
        rf = _cached_models.get("rf") or load_model_safe(RF_MODEL_PATH)
        dl = _cached_models.get("dl") or load_model_safe(DL_MODEL_PATH)
        xgb = _cached_models.get("xgb") or load_model_safe(XGB_MODEL_PATH)
        cat = _cached_models.get("cat") or load_model_safe(CAT_MODEL_PATH)
        seta = _cached_models.get("calibrated") or load_model_safe(
            CALIBRATED_MODEL_PATH
        )
        _cached_models.update(
            rf=rf, dl=dl, xgb=xgb, cat=cat, calibrated=seta
        )

        def safe_predict_proba_arr(model, X_df):
            if model is None:
                return np.zeros(len(X_df))
            try:
                return model.predict_proba(X_df)[:, 1]
            except Exception as e:
                logger.warning(
                    "safe_predict_proba_arr failed for model %s: %s", type(model), e
                )
                return np.zeros(len(X_df))

        rf_probs = safe_predict_proba_arr(rf, X)
        dl_probs = safe_predict_proba_arr(dl, X)
        xgb_probs = safe_predict_proba_arr(xgb, X)
        cat_probs = safe_predict_proba_arr(cat, X)
        seta_probs = safe_predict_proba_arr(seta, X)

        # ----------- ENSEMBLE SEARCH -----------
        vals = np.arange(0.0, 1.0 + 1e-9, weight_step)
        thresholds = np.arange(
            threshold_min, threshold_max + 1e-9, threshold_step
        )
        best_hit_target, best_overall = None, None

        def eval_combo(w_rf, w_dl, w_xgb, w_cat, w_seta, thr):
            combined = _ensemble_probs(
                (w_rf, w_dl, w_xgb, w_cat, w_seta),
                rf_probs,
                dl_probs,
                xgb_probs,
                cat_probs,
                seta_probs,
            )
            prec, cov, cnt = _precision_and_coverage(y, combined, thr)
            if prec is None:
                return None
            return {
                "weights": (
                    float(w_rf),
                    float(w_dl),
                    float(w_xgb),
                    float(w_cat),
                    float(w_seta),
                ),
                "threshold": float(thr),
                "precision": float(prec),
                "coverage": float(cov),
                "samples": int(cnt),
            }

        for w_rf in vals:
            for w_dl in vals:
                for w_xgb in vals:
                    for w_cat in vals:
                        w_seta = 1 - w_rf - w_dl - w_xgb - w_cat
                        if w_seta < 0:
                            continue
                        for thr in thresholds:
                            res = eval_combo(
                                w_rf, w_dl, w_xgb, w_cat, w_seta, thr
                            )
                            if not res:
                                continue
                            if res["precision"] >= target_precision:
                                if (
                                    not best_hit_target
                                    or res["coverage"]
                                    > best_hit_target["coverage"]
                                ):
                                    best_hit_target = res
                            if (not best_overall) or (
                                abs(res["precision"] - target_precision)
                                < abs(
                                    best_overall["precision"] - target_precision
                                )
                            ):
                                best_overall = res

        chosen = best_hit_target or best_overall
        if not chosen:
            raise HTTPException(
                status_code=404, detail="No valid weight/threshold combos"
            )
        w = tuple(chosen["weights"])
        thr = chosen["threshold"]
        combined_probs = _ensemble_probs(
            w, rf_probs, dl_probs, xgb_probs, cat_probs, seta_probs
        )

        # ----------- META MODEL TRAINING -----------
        meta_df = pd.DataFrame(
            {
                "rf_prob": rf_probs,
                "dl_prob": dl_probs,
                "xgb_prob": xgb_probs,
                "cat_prob": cat_probs,
                "seta_prob": seta_probs,
            }
        )
        meta_X = meta_df.fillna(0)
        meta_y = y

        if len(meta_X) >= 10:
            Xtr, Xte, ytr, yte = train_test_split(
                meta_X,
                meta_y,
                test_size=0.2,
                random_state=42,
                stratify=meta_y,
            )
            meta_model = RandomForestClassifier(
                n_estimators=200, random_state=42
            )
            meta_model.fit(Xtr, ytr)
            meta_acc = float(
                accuracy_score(yte, meta_model.predict(Xte))
            )
            os.makedirs(os.path.dirname(save_meta_path) or ".", exist_ok=True)
            joblib.dump(meta_model, save_meta_path)
            _cached_models["meta"] = meta_model
            meta_info = {
                "status": "trained",
                "accuracy": meta_acc,
                "rows": int(len(meta_X)),
                "path": save_meta_path,
            }
        else:
            meta_model = None
            meta_info = {
                "status": "skipped",
                "reason": "not enough rows",
            }

        # ----------- CALIBRATION + BACKTESTS -----------
        calib = _calibration_summary(y, combined_probs)
        wins = [
            int(x.strip())
            for x in backtest_windows.split(",")
            if x.strip().isdigit()
        ]
        backtests = [
            _backtest_window(df_enriched, combined_probs, thr, window_hours=wh)
            for wh in wins
        ]

        # ----------- LIVE PREDICTIONS PER SYMBOL -----------
        live_results = {}
        enricher_live = SetaFeatureEnricher()

        # feature columns to use for per-record model input; prefer meta if available
        single_feature_cols = None
        if os.path.exists(meta_path):
            try:
                single_feature_cols = json.load(open(meta_path)).get(
                    "feature_columns", None
                )
            except Exception as e:
                logger.warning("Failed to read meta_path for live: %s", e)
                single_feature_cols = None

        def predict_single_for_row(model, row_dict, feature_cols):
            if model is None:
                return 0.0
            df_row = pd.DataFrame([row_dict])
            if feature_cols:
                df_row = df_row.reindex(columns=feature_cols)
            else:
                df_row = df_row.select_dtypes(include=[np.number])
            df_row = df_row.fillna(0)
            try:
                return float(model.predict_proba(df_row)[:, 1][0])
            except Exception as e:
                logger.warning(
                    "predict_single_for_row failed for %s: %s", type(model), e
                )
                return 0.0

        for s in [ss.strip() for ss in symbols.split(",") if ss.strip()]:
            try:
                parts = s.split(":")
                if len(parts) != 2:
                    continue
                asset_id, bin_symbol = parts[0], parts[1]
                asset_snapshot = fetch_legacy_snapshot(asset_id)
                bz = fetch_binance(bin_symbol)

                # Enrich single snapshots consistently, with safety
                try:
                    df_asset = enricher_live.enrich_features(pd.DataFrame([asset_snapshot]))
                    if not isinstance(df_asset, pd.DataFrame):
                        logger.warning(
                            "Live enrich_features returned %s; using raw snapshot",
                            type(df_asset),
                        )
                        df_asset = pd.DataFrame([asset_snapshot])
                except Exception as e:
                    logger.warning(
                        "Live enrich_features failed for %s: %s; using raw snapshot",
                        asset_id,
                        e,
                    )
                    df_asset = pd.DataFrame([asset_snapshot])

                # Clean NaNs and possible list-cells
                for col in df_asset.columns:
                    try:
                        if df_asset[col].apply(lambda x: isinstance(x, list)).any():
                            df_asset[col] = df_asset[col].apply(
                                lambda x: x[0]
                                if isinstance(x, list) and len(x) > 0
                                else 0
                            )
                    except Exception:
                        continue

                df_asset = df_asset.fillna(0)
                if df_asset.empty:
                    continue

                row_vec = df_asset.iloc[0].to_dict()

                p_rf = predict_single_for_row(
                    rf, row_vec, single_feature_cols
                )
                p_dl = predict_single_for_row(
                    dl, row_vec, single_feature_cols
                )
                p_xgb = predict_single_for_row(
                    xgb, row_vec, single_feature_cols
                )
                p_cat = predict_single_for_row(
                    cat, row_vec, single_feature_cols
                )
                p_seta = predict_single_for_row(
                    seta, row_vec, single_feature_cols
                )

                p_ens = float(
                    w[0] * p_rf
                    + w[1] * p_dl
                    + w[2] * p_xgb
                    + w[3] * p_cat
                    + w[4] * p_seta
                )
                dec_ens = (
                    "BUY"
                    if p_ens >= thr
                    else ("SELL" if p_ens <= (1 - thr) else "HOLD")
                )

                live_results[asset_id] = {
                    "asset": {
                        "price": asset_snapshot.get("price"),
                        "change_24h": asset_snapshot.get("change_24h"),
                        "ensemble_prob": p_ens,
                        "decision": dec_ens,
                    },
                    "binance": {
                        "price": bz.get("price"),
                        "change_24h": bz.get("change_24h"),
                        "ensemble_prob": p_ens,
                        "decision": dec_ens,
                    },
                }
            except Exception as e:
                logger.warning("Live prediction failed for symbol '%s': %s", s, e)
                continue

        return {
            "status": "success",
            "chosen_combo": chosen,
            "ensemble_metrics": {"weights": w, "threshold": thr},
            "calibration": calib,
            "backtests": backtests,
            "meta_model": meta_info,
            "live": live_results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("target_precision_meta_live error: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"target_precision_meta_live error: {str(e)}",
        )








# =========================
# Imports & Setup
# =========================
import os
import datetime, time, threading, requests, schedule
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, APIRouter, Depends, Body, Query, HTTPException
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.exc import ProgrammingError
# Custom modules
from services.trade_executor import execute_trade
from utils import database
from models.prediction_trade_models import PredictionLog
from schemas.prediction_trade_schemas import PredictionLog as PredictionLogSchema, PredictionLogCreate

# =========================
# Constants / Config
# =========================
SYMBOLS_DEFAULT = os.getenv("LIVE_SYMBOLS", "BTCUSDT,ETHUSDT").replace(" ", "").split(",")
PRED_INTERVAL_MIN = int(os.getenv("PRED_INTERVAL_MIN", "5"))
RETRAIN_CRON_HHMM   = os.getenv("RETRAIN_CRON_HHMM", "02:00")
EVAL_BAND = float(os.getenv("EVAL_BAND", "0.001"))
BINANCE_24HR_URL = "https://api.binance.com/api/v3/ticker/24hr"

# =========================
# Utilities
# =========================
def _safe_get(url: str, params: Dict[str, Any] = None, timeout: int = 10) -> dict:
    r = requests.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_binance_snapshot(symbol: str) -> dict:
    try:
        data = _safe_get(BINANCE_24HR_URL, params={"symbol": symbol})
        return {
            "price": float(data["lastPrice"]),
            "high_24h": float(data["highPrice"]),
            "low_24h": float(data["lowPrice"]),
            "volume": float(data["volume"]),
            "quoteVolume": float(data.get("quoteVolume", 0.0)),
            "count_trades": int(data.get("count", 0)),
            "change_24h": float(data["priceChangePercent"]),
            "bidPrice": float(data.get("bidPrice", data["lastPrice"])),
            "askPrice": float(data.get("askPrice", data["lastPrice"])),
            "openPrice": float(data.get("openPrice", data["lastPrice"])),
            "prevClosePrice": float(data.get("prevClosePrice", data["lastPrice"])),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Binance fetch failed: {e}")

def fetch_news_sentiment(symbol: str) -> str:
    return "Bullish sentiment detected based on recent headlines"

def calc_volatility(symbol: str, period: str = "1h") -> float:
    return 0.012

def fetch_orderbook_imbalance(symbol: str) -> float:
    return 0.07

def enrich_live_features(symbol: str) -> dict:
    snap = fetch_binance_snapshot(symbol)
    snap["volatility_1h"] = calc_volatility(symbol)
    snap["orderbook_imbalance"] = fetch_orderbook_imbalance(symbol)
    snap["news_sentiment"] = fetch_news_sentiment(symbol)
    snap["day_range_pct"] = (snap["high_24h"] - snap["low_24h"]) / max(1e-9, snap["low_24h"])
    snap["mid_price"] = (snap["bidPrice"] + snap["askPrice"]) / 2.0
    snap["spread_bps"] = (snap["askPrice"] - snap["bidPrice"]) / max(1e-9, snap["mid_price"]) * 1e4
    return snap

def current_price(symbol: str) -> float:
    return float(fetch_binance_snapshot(symbol)["price"])

# =========================
# Meta Model (Placeholder)
# =========================
def meta_live_predict(features: dict) -> Dict[str, Any]:
    prob_up = 0.8 if features["news_sentiment"].startswith("Bullish") else 0.45
    prediction = "BUY" if prob_up >= 0.55 else ("SELL" if prob_up <= 0.45 else "HOLD")
    trade_count = 1 if features["volatility_1h"] < 0.015 else 3
    explanation = (
        f"Human read: news={features['news_sentiment']}; "
        f"vol_1h={features['volatility_1h']:.3f}; "
        f"imbalance={features['orderbook_imbalance']:.3f}; "
        f"spread_bps={features['spread_bps']:.1f}."
    )
    return {
        "prediction": prediction,
        "confidence": round(prob_up if prediction == "BUY" else (1 - prob_up) if prediction == "SELL" else 0.5, 4),
        "trade_count": trade_count,
        "explanation": explanation
    }

# =========================
# Schemas
# =========================
class LiveIn(BaseModel):
    symbol: str
    prediction: str
    confidence: float
    trade_count: int
    reasoning: str
    entry_price: Optional[float] = None
    horizon_minutes: int = 30

    @validator("symbol")
    def _upper(cls, v): return v.upper()

    @validator("prediction")
    def _pred_ok(cls, v):
        v = v.upper()
        if v not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("prediction must be BUY/SELL/HOLD")
        return v

# =========================
# Core Prediction Logic
# =========================
def predict_from_features(features: dict):
    return {"prediction": "HOLD", "confidence": 0.5, "price": features.get("close", 0.0)}

def evaluate_outcome(entry_price: float, exit_price: float, side: str) -> str:
    change = (exit_price - entry_price) / max(1e-9, entry_price)
    if side == "BUY": return "correct" if change >= EVAL_BAND else "incorrect"
    if side == "SELL": return "correct" if change <= -EVAL_BAND else "incorrect"
    return "correct" if abs(change) <= EVAL_BAND else "incorrect"

def label_due_predictions(db):
    now = datetime.datetime.utcnow()
    
    try:
        pending = db.query(PredictionLog).all()
    except ProgrammingError as e:
        print("[labeler] Skipping: table does not exist yet.")
        db.rollback()  # Important to reset session after exception
        return
    
    for log in pending:
        # Only process logs that haven't been resolved and have a resolve_at time
        if getattr(log, "outcome", None) is None and getattr(log, "resolve_at", None) and log.resolve_at <= now:
            try:
                px = current_price(log.symbol)
                result = evaluate_outcome(log.entry_price, px, log.prediction)
                log.outcome = result
                log.resolved_at = now
                db.add(log)
                db.commit()
            except Exception as e:
                print(f"[labeler] fail {log.id} {log.symbol}: {e}")
                db.rollback()  # Keep session clean on errors

# =========================
# API Endpoints
# =========================
@router.post("/predict-live/")
def predict_live(sample: dict):
    features = enrich_live_features(sample.get("symbol", "BTCUSDT"))
    result = predict_from_features(features)
    return result

@router.post("/predictions/live")
def live_prediction(payload: LiveIn = Body(...), db: Session = Depends(database.get_db)):
    price = payload.entry_price if payload.entry_price else current_price(payload.symbol)
    now = datetime.datetime.utcnow()
    resolve_at = now + datetime.timedelta(minutes=payload.horizon_minutes)

    log = PredictionLog(
        symbol=payload.symbol,
        prediction=payload.prediction,
        confidence=payload.confidence,
        trade_count=payload.trade_count,
        reasoning=payload.reasoning,
        created_at=now,
        entry_price=float(price),
        horizon_minutes=int(payload.horizon_minutes),
        resolve_at=resolve_at,
        outcome=None,
        resolved_at=None
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return {"status": "ok", "id": log.id, "symbol": log.symbol, "prediction": log.prediction,
            "confidence": log.confidence, "entry_price": log.entry_price,
            "horizon_minutes": log.horizon_minutes, "resolve_at": resolve_at.isoformat()}

@router.post("/predict-and-trade/")
def predict_and_trade(payload: dict, mode: str = "paper"):
    features = enrich_live_features(payload.get("symbol", "BTCUSDT"))
    result = predict_from_features(features)
    prediction = result.get("prediction")
    confidence = float(result.get("confidence", 0.0))
    price = float(result.get("price", features.get("close", 0.0)))

    if confidence < 0.7:
        return {"ok": False, "reason": "low_confidence", "prediction": result}

    out = execute_trade(
        symbol=payload.get("symbol", "BTCUSDT"),
        side="BUY" if prediction == "BUY" else "SELL",
        entry_price=price,
        confidence=confidence,
        mode=mode,
        starting_capital=10000.0,
        risk_pct=0.01,
        stop_loss_pct=0.01,
        take_profit_pct=0.02,
    )
    return {"prediction": result, "trade_execution": out}

@router.get("/analysis/weekly")
def weekly_analysis(db: Session = Depends(database.get_db), days: int = Query(7, ge=1, le=30)):
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=days)
    logs = db.query(PredictionLog).filter(PredictionLog.created_at >= cutoff).all()
    agg, daily = {}, {}
    for log in logs:
        sym = log.symbol
        agg.setdefault(sym, {"total":0,"correct":0,"trade_count":0})
        agg[sym]["total"] += 1
        agg[sym]["trade_count"] += (log.trade_count or 0)
        if getattr(log, "outcome", None) == "correct":
            agg[sym]["correct"] += 1
        d = (log.created_at or datetime.datetime.utcnow()).strftime("%Y-%m-%d")
        daily.setdefault(d, {}).setdefault(sym, {"total":0,"correct":0})
        daily[d][sym]["total"] += 1
        if getattr(log, "outcome", None) == "correct":
            daily[d][sym]["correct"] += 1
    results = {s: {"accuracy_pct": round((data["correct"]/data["total"]*100.0),2) if data["total"]>0 else 0.0,
                   "trade_count": data["trade_count"],
                   "summary": f"{s} → {(data['correct']/data['total']*100.0) if data['total'] else 0:.2f}% accuracy, {data['trade_count']} trades in last {days} days"} for s,data in agg.items()}
    daily_delta = {day:{s: round((d["correct"]/d["total"]*100.0),2) if d["total"]>0 else 0.0 for s,d in per_sym.items()} for day,per_sym in sorted(daily.items())}
    return {"per_symbol": results, "daily_accuracy": daily_delta}

# =========================
# Background Jobs
# =========================
def run_live_prediction(symbol: str):
    try:
        feats = enrich_live_features(symbol)
        pred = meta_live_predict(feats)
        human = (f"{symbol} | {pred['explanation']} | trade_count={pred['trade_count']} | "
                 f"24h_change={feats['change_24h']:.2f}% | day_range={feats['day_range_pct']:.2%}")
        payload = {"symbol": symbol, "prediction": pred["prediction"], "confidence": pred["confidence"],
                   "trade_count": pred["trade_count"], "reasoning": human, "entry_price": feats["price"],
                   "horizon_minutes": 30}
        requests.post("http://localhost:8000/predictions/live", json=payload, timeout=8)
    except Exception as e:
        print(f"[runner] live prediction failed for {symbol}: {e}")

def retrain_job():
    print(f"[{datetime.datetime.utcnow().isoformat()}] Retraining meta-model(s)...")

def start_scheduler(symbols: List[str]):
    for s in symbols:
        schedule.every(PRED_INTERVAL_MIN).minutes.do(run_live_prediction, s)
    schedule.every(1).minutes.do(lambda: label_due_predictions(next(database.get_db())))
    schedule.every().day.at(RETRAIN_CRON_HHMM).do(retrain_job)
    def loop(): 
        while True: schedule.run_pending(); time.sleep(1)
    threading.Thread(target=loop, daemon=True).start()



