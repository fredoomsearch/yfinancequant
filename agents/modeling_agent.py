from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from providers import GroqAdvisor
from schemas.pipeline import ArtifactRef, ModelFamily, ModelResult, ModelingResult, CleaningResult, PipelineRequest

logger = logging.getLogger(__name__)


def _label_from_probability(probability: float) -> str:
    return "long" if probability >= 0.5 else "short"


def _safe_metric(metric_fn, y_true, y_pred, **kwargs):
    try:
        return float(metric_fn(y_true, y_pred, **kwargs))
    except Exception:
        return 0.0


@dataclass
class ModelingAgent:
    model_names: Tuple[str, ...] = ("logistic_regression", "random_forest", "gradient_boosting")
    advisor: Optional[GroqAdvisor] = None

    def _prepare_matrices(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        trainable = df.dropna(subset=["target_direction"]).copy()
        if len(trainable) < 10:
            raise ValueError("Not enough labeled rows to train models.")

        trainable = trainable.sort_values(["date", "ticker"])
        y = trainable["target_direction"].astype(int)
        X = trainable.drop(columns=["target_direction"], errors="ignore")
        X = pd.get_dummies(X, columns=["ticker"], prefix="ticker")
        X = X.drop(columns=[c for c in ["date", "future_close", "future_return"] if c in X.columns], errors="ignore")
        X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        latest_rows = df.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
        latest_features = latest_rows.drop(columns=["target_direction"], errors="ignore")
        latest_features = pd.get_dummies(latest_features, columns=["ticker"], prefix="ticker")
        latest_features = latest_features.drop(columns=[c for c in ["date", "future_close", "future_return"] if c in latest_features.columns], errors="ignore")
        latest_features = latest_features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        latest_features = latest_features.reindex(columns=X.columns, fill_value=0)
        return X, y, latest_features

    def _time_split(self, X: pd.DataFrame, y: pd.Series):
        split_idx = max(int(len(X) * 0.8), 1)
        if split_idx >= len(X):
            split_idx = len(X) - 1
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        if len(X_test) == 0:
            X_train, X_test = X.iloc[:-1], X.iloc[-1:]
            y_train, y_test = y.iloc[:-1], y.iloc[-1:]
        if y_train.nunique() < 2 or y_test.nunique() < 1:
            raise ValueError("Need both classes in training data to fit the local models.")
        return X_train, X_test, y_train, y_test

    def _train_models(self, X_train, y_train):
        models = {
            "logistic_regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
                ]
            ),
            "random_forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample"),
            "gradient_boosting": GradientBoostingClassifier(random_state=42),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            logger.info("Trained %s on %s rows", name, len(X_train))
        return models

    def run(self, request: PipelineRequest, cleaning: CleaningResult, run_dir: Path) -> ModelingResult:
        cleaned_path = Path(cleaning.clean_data.artifact.path)
        if not cleaned_path.exists():
            raise FileNotFoundError(cleaned_path)

        output_dir = run_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(cleaned_path, parse_dates=["date"])
        if "ticker" not in df.columns:
            raise ValueError("Cleaned data must contain ticker")

        X, y, latest_features = self._prepare_matrices(df)
        X_train, X_test, y_train, y_test = self._time_split(X, y)
        models = self._train_models(X_train, y_train)

        model_results: List[ModelResult] = []
        latest_predictions: Dict[str, float] = {}

        for name, model in models.items():
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)[:, 1]
                latest_proba = model.predict_proba(latest_features)[:, 1]
            else:
                proba = y_pred.astype(float)
                latest_proba = np.asarray(model.predict(latest_features), dtype=float)

            latest_probability = float(np.mean(latest_proba))
            latest_prediction = _label_from_probability(latest_probability)
            confidence = float(max(latest_probability, 1 - latest_probability))

            metrics = {
                "accuracy": _safe_metric(accuracy_score, y_test, y_pred),
                "precision": _safe_metric(precision_score, y_test, y_pred, zero_division=0),
                "recall": _safe_metric(recall_score, y_test, y_pred, zero_division=0),
                "f1": _safe_metric(f1_score, y_test, y_pred, zero_division=0),
            }
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
            except Exception:
                metrics["roc_auc"] = 0.0

            artifact_path = output_dir / f"{name}.joblib"
            joblib.dump(model, artifact_path)

            model_results.append(
                ModelResult(
                    model_name=name,
                    validation_metrics=metrics,
                    latest_probability=latest_probability,
                    latest_prediction=latest_prediction,
                    confidence=confidence,
                    artifact=ArtifactRef(
                        name=f"{name}.joblib",
                        path=str(artifact_path),
                        kind="model",
                        size_bytes=artifact_path.stat().st_size,
                    ),
                )
            )
            latest_predictions[name] = latest_probability

        ensemble_probability = float(np.mean(list(latest_predictions.values())))
        ensemble_prediction = _label_from_probability(ensemble_probability)
        majority_vote = "long" if sum(result.latest_prediction == "long" for result in model_results) >= 2 else "short"
        disagreement = len({result.latest_prediction for result in model_results}) > 1
        confidence = float(max(ensemble_probability, 1 - ensemble_probability))

        if request.model_choice == ModelFamily.logistic_regression:
            selected_model = "logistic_regression"
        elif request.model_choice == ModelFamily.random_forest:
            selected_model = "random_forest"
        elif request.model_choice == ModelFamily.gradient_boosting:
            selected_model = "gradient_boosting"
        elif request.model_choice == ModelFamily.ensemble:
            selected_model = "ensemble"
        else:
            selected_model = "ensemble" if not disagreement else "majority"

        es = request.language == "es"
        rationale = (
            "All local models agreed on the same direction."
            if not es and not disagreement
            else (
                "Los modelos locales estuvieron alineados en la misma dirección."
                if es and not disagreement
                else (
                    "Local models disagreed, so the ensemble/majority result is used as the deterministic candidate."
                    if not es
                    else "Los modelos locales discreparon, así que el resultado de ensamble/mayoría se usa como candidato determinístico."
                )
            )
        )

        summary = {
            "models": [result.model_dump() for result in model_results],
            "ensemble_probability": ensemble_probability,
            "ensemble_prediction": ensemble_prediction,
            "majority_prediction": majority_vote,
            "disagreement": disagreement,
            "selected_model": selected_model,
            "rationale": rationale,
            "latest_sample_count": int(len(latest_features)),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "motor": selected_model,
        }

        groq_brief = None
        if self.advisor and self.advisor.enabled:
            groq_brief = self.advisor.brief(
                "modeling",
                {
                    "motor": selected_model,
                    "language": request.language,
                    "model_choice": request.model_choice.value,
                    "selected_model": selected_model,
                    "ensemble_probability": ensemble_probability,
                    "ensemble_prediction": ensemble_prediction,
                    "majority_prediction": majority_vote,
                    "disagreement": disagreement,
                    "latest_sample_count": int(len(latest_features)),
                    "training_rows": int(len(X_train)),
                    "test_rows": int(len(X_test)),
                    "models": [result.model_dump() for result in model_results],
                },
            )
            if groq_brief:
                summary["groq_brief"] = groq_brief.model_dump()
                summary["motor"] = groq_brief.motor or summary["motor"]

        summary_path = output_dir / "model_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

        artifact = ArtifactRef(
            name="model_summary.json",
            path=str(summary_path),
            kind="model_summary",
            size_bytes=summary_path.stat().st_size,
        )
        return ModelingResult(
            models=model_results,
            ensemble_probability=ensemble_probability,
            ensemble_prediction=ensemble_prediction,
            majority_prediction=majority_vote,
            disagreement=disagreement,
            selected_model=selected_model,
            rationale=rationale,
            latest_sample_count=int(len(latest_features)),
            artifact=artifact,
            groq_brief=groq_brief,
        )
