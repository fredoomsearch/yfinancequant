"""Legacy crypto feature helpers and compatibility endpoints.

The active pipeline lives in `agents/` and `pipeline/`. This module is kept as
reference material and for optional bridge logic, not as the primary training
path.
"""

import os
from pyexpat import features
from time import time
from catboost import CatBoostClassifier
from fastapi import APIRouter, Depends
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session
from ta import add_all_ta_features
from xgboost import XGBClassifier
from data_acquisition.crypto_api import fetch_on_chain_data
from models import calibrated_models
from api import events
from utils import database
from data_acquisition import web_scraper
from services import event_service
import logging
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data", tags=["data_processing"])

def assign_topic(article):
    """Assign a topic to an article based on its content."""
    if not isinstance(article, dict):
        logger.error(f"Invalid article format: {article}")
        return "Other"
    title = (article.get('title') or '').lower()
    description = (article.get('description') or '').lower()
    if any(word in title or word in description for word in ['stock', 'market', 'gdp', 'inflation']):
        return 'Economy'
    elif any(word in title or word in description for word in ['meta', 'microsoft', 'ai', 'tech', 'google', 'apple']):
        return 'Tech'
    elif any(word in title or word in description for word in ['trump', 'gop', 'republican', 'democrat', 'congress']):
        return 'Politics'
    else:
        return 'Other'

def process_news_data(news_data):
    """Process news data and add topics."""
    processed_data = []
    if isinstance(news_data, list):
        for article in news_data:
            if isinstance(article, dict):
                article_copy = article.copy()
                article_copy['topic'] = assign_topic(article)
                processed_data.append(article_copy)
    return processed_data


@router.get("/news/")
def get_news(db: Session = Depends(database.get_db)):
    news_url = "https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey=7711f5f714754e09a77c6d2640f1111e"
    news_url2 = "https://newsapi.org/v2/everything?domains=wsj.com&apiKey=7711f5f714754e09a77c6d2640f1111e"

    logger.info("Fetching news data...")
    news_data, scraped_data = web_scraper.scrape_website(news_url, news_url2, db)
   
    processed_news = process_news_data(news_data)
    processed_wsj = process_news_data(scraped_data)

    event_service.create_event(events.EventCreate(
        event_name="DATA_PROCESSED",
        event_value=0,
        event_message="Data processing complete",
        event_details={"news_data_len": len(processed_news), "scraped_data_len": len(processed_wsj)}
    ), db)

    combined_news = {"newsapi": processed_news, "wsj": processed_wsj}
    logger.info(f"Combined news: {combined_news}")
    return combined_news

import pandas as pd
import numpy as np
import pandas_ta as ta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
from typing import List, Optional
from functools import lru_cache
import requests
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_cast_series(series, dtype=float):
    return pd.to_numeric(series, errors="coerce").astype(dtype)

class SetaFeatureEnricher:
    def __init__(self, twitter_api=None, etherscan_api_key: Optional[str] = None):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.twitter_api = twitter_api
        self.etherscan_key = etherscan_api_key or os.getenv("ETHERSCAN_API_KEY")

    def enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        for col in ["current_price", "high_24h", "low_24h", "total_volume",
                   "market_cap", "circulating_supply", "max_supply"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
        df = self._add_technical_indicators(df)
        df = self._add_volatility_features(df)
        df = self._add_lagged_features(df)
        df = self._add_normalized_ratios(df)
        df = self._add_sentiment_features(df)
        df = self._add_onchain_metrics(df)
        df = self._add_market_correlation(df)
        df = self._add_temporal_features(df)
        df = self._add_meta_labels(df)
        df = self._add_orderbook_imbalance(df)
        df = df.replace([np.inf, -np.inf], np.nan)
        if "price_direction" in df.columns:
            target = df["price_direction"].copy()
            feature_columns = [column for column in df.columns if column != "price_direction"]
            df[feature_columns] = df[feature_columns].fillna(0)
            df["price_direction"] = target
            return df
        return df.fillna(0)

    def _add_technical_indicators(self, df):
        try:
            if "current_price" not in df.columns:
                return df
            price = df["current_price"]
            df["sma_20"] = ta.sma(price, length=20)
            df["sma_50"] = ta.sma(price, length=50)
            df["ema_20"] = ta.ema(price, length=20)
            df["ema_50"] = ta.ema(price, length=50)
            df["rsi_14"] = ta.rsi(price, length=14)
            bb = ta.bbands(price, length=20)
            if isinstance(bb, pd.DataFrame):
                df["bb_upper"], df["bb_middle"], df["bb_lower"] = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
                df["bb_percent"] = (price - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)
            if all(c in df.columns for c in ["high_24h", "low_24h"]):
                df["atr_14"] = ta.atr(df["high_24h"], df["low_24h"], price, length=14)
            if "total_volume" in df.columns:
                df["vwap"] = (price * df["total_volume"]).cumsum() / (df["total_volume"].cumsum() + 1e-8)
                df["obv"] = ta.obv(price, df["total_volume"])
            return df
        except Exception as e:
            logger.error(f"Error in _add_technical_indicators: {e}")
            return df

    def _add_volatility_features(self, df):
        if "current_price" in df.columns:
            df["return_1"] = df["current_price"].pct_change(1)
            df["return_3"] = df["current_price"].pct_change(3)
            df["rolling_vol_7"] = df["return_1"].rolling(7).std()
            df["rolling_vol_14"] = df["return_1"].rolling(14).std()
        return df

    def _add_lagged_features(self, df, lags=(1, 3, 5, 7), windows=(3, 7, 14)):
        for col in ["current_price", "total_volume"]:
            if col in df.columns:
                for lag in lags:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                for w in windows:
                    df[f"{col}_roll_mean_{w}"] = df[col].rolling(window=w).mean()
                    df[f"{col}_roll_std_{w}"] = df[col].rolling(window=w).std()
        return df

    def _add_normalized_ratios(self, df):
        if "total_volume" in df.columns and "market_cap" in df.columns:
            df["vol_over_marketcap"] = df["total_volume"] / df["market_cap"].replace({0: np.nan})
        if "circulating_supply" in df.columns and "max_supply" in df.columns:
            df["supply_ratio"] = df["circulating_supply"] / df["max_supply"].replace({0: np.nan})
        return df

    @lru_cache(maxsize=100)
    def _get_twitter_sentiment(self, symbol: str, count: int = 50):
        if not self.twitter_api or not symbol:
            return 0.0
        try:
            tweets = self.twitter_api.search_tweets(q=f"${symbol} -filter:retweets", count=count, tweet_mode="extended")
            scores = [self.sentiment_analyzer.polarity_scores(t.full_text)["compound"] for t in tweets]
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logger.error(f"Error fetching Twitter sentiment for {symbol}: {e}")
            return 0.0

    @lru_cache(maxsize=100)
    def _get_reddit_sentiment(self, symbol: str, limit: int = 50):
        try:
            url = f"https://api.pushshift.io/reddit/search/comment/?q={symbol}&size={limit}"
            r = requests.get(url, timeout=10).json()
            comments = [c["body"] for c in r.get("data", [])]
            scores = [self.sentiment_analyzer.polarity_scores(c)["compound"] for c in comments]
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment for {symbol}: {e}")
            return 0.0

    def _add_sentiment_features(self, df):
        df["sentiment_score"] = df.get("news_headlines", "").fillna("").astype(str).apply(
            lambda t: self.sentiment_analyzer.polarity_scores(t)["compound"]
        )
        df["twitter_sentiment"] = df.apply(lambda r: self._get_twitter_sentiment(r.get("symbol")), axis=1)
        df["reddit_sentiment"] = df.apply(lambda r: self._get_reddit_sentiment(r.get("symbol")), axis=1)
        return df

    def _add_onchain_metrics(self, df):
        df["active_addresses"] = 0
        df["transaction_count"] = 0
        df["network_fee"] = 0.0
        return df

    def _add_market_correlation(self, df):
        df["btc_correlation"] = 0.5
        df["eth_correlation"] = 0.5
        df["market_correlation"] = (df["btc_correlation"] + df["eth_correlation"]) / 2
        return df

    def _add_temporal_features(self, df):
        if "timestamp" in df.columns:
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
            df["month"] = df["timestamp"].dt.month
            df["quarter"] = df["timestamp"].dt.quarter
        return df

    def _add_meta_labels(self, df):
        if "current_price" in df.columns:
            future_price = df["current_price"].shift(-1)
            df["price_direction"] = np.where(future_price.notna(), (future_price > df["current_price"]).astype(float), np.nan)
        return df

    def _add_orderbook_imbalance(self, df):
        df["orderbook_imbalance"] = df.apply(lambda r: self.fetch_order_book_imbalance(r.get("symbol")), axis=1)
        return df

    @staticmethod
    def fetch_order_book_imbalance(symbol: str, depth: int = 20):
        if not symbol:
            return 0.0
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={symbol.upper()}USDT&limit={depth}"
            data = requests.get(url, timeout=5).json()
            bids = np.array([float(b[1]) for b in data["bids"]])
            asks = np.array([float(a[1]) for a in data["asks"]])
            return (bids.sum() - asks.sum()) / (bids.sum() + asks.sum() + 1e-8)
        except Exception as e:
            logger.error(f"Error fetching order book imbalance for {symbol}: {e}")
            return 0.0

class ModelManager:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        self.load_models()

    def load_models(self):
        model_paths = {
            "rf": os.path.join(self.model_dir, "rf_model.joblib"),
            "dl": os.path.join(self.model_dir, "lstm_model.joblib"),
            "calibrated": os.path.join(self.model_dir, "calibrated_stacking_model.joblib")
        }
        for name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    self.models[name] = joblib.load(path)
                    logger.info(f"Loaded model {name} from {path}")
                except Exception as e:
                    logger.error(f"Failed to load model {name}: {e}")
            else:
                logger.warning(f"Model file {path} not found")

    def predict(self, model_name, features):
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not loaded")
            return np.zeros(len(features))
        try:
            return model.predict_proba(features)[:, 1]
        except Exception as e:
            logger.error(f"Prediction error for model {model_name}: {e}")
            return np.zeros(len(features))

def ensemble_predictions(rf_pred, dl_pred, seta_pred, weights=(0.3, 0.3, 0.4)):
    return weights[0] * rf_pred + weights[1] * dl_pred + weights[2] * seta_pred

def predict_with_ensemble(data: List[dict], buy_thresh=0.7, sell_thresh=0.3):
    df = pd.DataFrame(data)
    enricher = SetaFeatureEnricher()
    enriched = enricher.enrich_features(df)
    non_numeric_cols = ["symbol", "current_price", "timestamp", "name"]
    features = enriched.drop(columns=[c for c in non_numeric_cols if c in enriched.columns]).select_dtypes(include=[np.number]).fillna(0)
    model_manager = ModelManager()
    rf_pred = model_manager.predict("rf", features)
    dl_pred = model_manager.predict("dl", features)
    seta_pred = model_manager.predict("calibrated", features)
    combined = ensemble_predictions(rf_pred, dl_pred, seta_pred)
    results = []
    for i, row in enriched.iterrows():
        prob_up = float(combined[i])
        decision = "HOLD"
        if prob_up >= buy_thresh:
            decision = "BUY"
        elif prob_up <= sell_thresh:
            decision = "SELL"
        results.append({
            "symbol": row.get("symbol", f"asset_{i}"),
            "current_price": float(row.get("current_price", 0.0)),
            "probability_up": prob_up,
            "decision": decision
        })
    return {"predictions": results}

# Legacy live loop removed; Binance helpers remain available above.
