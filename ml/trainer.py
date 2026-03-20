import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from xgboost import XGBClassifier
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

MODELS_DIR = "models"
REPORTS_DIR = "outputs/reports"


def _get_models() -> Dict:
    """Returns all three models with sensible defaults."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # handles class imbalance
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1  # use all CPU cores
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0
        )
    }


def train_and_evaluate(
    X: pd.DataFrame,
    y_series: pd.Series,
    target_name: str
) -> Dict:
    """
    Trains all three models on a single target (risk_label OR sentiment_label).

    Steps:
        1. Encode labels (LabelEncoder)
        2. Train/test split (80/20)
        3. Train each model
        4. Evaluate: accuracy, F1, ROC-AUC, confusion matrix
        5. Save best model as .pkl

    Returns: results dict with metrics for all models
    """
    logger.info(f"\n{'='*40}")
    logger.info(f"Training target: {target_name}")
    logger.info(f"{'='*40}")

    # ── Label Encoding ────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_series)
    classes = le.classes_
    logger.info(f"Classes: {classes}")

    # ── Train/Test Split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded  # preserve class distribution
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Train & Evaluate Each Model ───────────────────────
    models = _get_models()
    results = {}
    best_f1 = -1
    best_model = None
    best_model_name = None

    for model_name, model in models.items():
        logger.info(f"\nTraining: {model_name}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        # ROC-AUC (multiclass)
        try:
            y_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            roc_auc = None

        # Cross-validation score
        cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring="f1_weighted")

        results[model_name] = {
            "accuracy":       round(accuracy, 4),
            "f1_weighted":    round(f1, 4),
            "roc_auc":        round(roc_auc, 4) if roc_auc else "N/A",
            "cv_mean_f1":     round(cv_scores.mean(), 4),
            "cv_std_f1":      round(cv_scores.std(), 4),
            "confusion_matrix": cm,
            "classification_report": report,
        }

        logger.info(f"  Accuracy : {accuracy:.4f}")
        logger.info(f"  F1       : {f1:.4f}")
        logger.info(f"  ROC-AUC  : {roc_auc:.4f}" if roc_auc else "  ROC-AUC : N/A")
        logger.info(f"  CV F1    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Track best model by F1
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = model_name

    logger.info(f"\nBest model for {target_name}: {best_model_name} (F1={best_f1:.4f})")

    # ── Save Best Model ───────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, f"{target_name}_best_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, f"{target_name}_label_encoder.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)

    logger.info(f"Saved: {model_path}")
    logger.info(f"Saved: {encoder_path}")

    return {
        "target":           target_name,
        "best_model":       best_model_name,
        "best_f1":          best_f1,
        "classes":          classes.tolist(),
        "model_results":    results,
        "model_path":       model_path,
        "encoder_path":     encoder_path,
    }
