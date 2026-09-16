from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.evaluate import (
    classification_metrics,
    print_classification_summary,
    save_classification_metrics,
)
from src.features.text_features import make_tfidf_vectorizer


def get_candidate_models():
    return {
        "linear_svc": LinearSVC(dual="auto", max_iter=1_000, class_weight="balanced", random_state=42),
        "sgd_logistic": SGDClassifier(loss="log_loss", max_iter=1_000, class_weight="balanced", random_state=42),
        "logistic_regression": LogisticRegression(max_iter=300, class_weight="balanced", random_state=42),
    }


def train_classifier(resumes: pd.DataFrame, model_path: Path, metrics_path: Path) -> dict:
    """Train a resume category classifier, save the model and full evaluation metrics."""
    counts = resumes["category"].value_counts()
    usable = resumes[resumes["category"].isin(counts[counts >= 2].index)].copy()
    n_classes = usable["category"].nunique()
    if n_classes < 2:
        raise ValueError("Classifier training requires at least two categories with two resumes each.")

    # Determine valid test size and stratification
    min_count = usable["category"].value_counts().min()
    can_stratify = min_count >= 2 and len(usable) >= n_classes * 2
    stratify = usable["category"] if can_stratify else None

    if can_stratify:
        test_size = max(0.2, min(0.5, n_classes / len(usable)))
    else:
        test_size = max(0.2, min(0.4, n_classes / len(usable)))

    train, test = train_test_split(
        usable, test_size=test_size, random_state=42, stratify=stratify
    )

    candidates = get_candidate_models()
    scores: dict[str, float] = {}
    fitted: dict[str, Pipeline] = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline([("tfidf", make_tfidf_vectorizer(max_features=8000)), ("model", clone(estimator))])
        pipeline.fit(train["text"], train["category"])
        preds = pipeline.predict(test["text"])
        scores[name] = float(f1_score(test["category"], preds, average="weighted", zero_division=0))
        fitted[name] = pipeline

    selected = max(scores, key=scores.get)

    # Evaluation on hold-out using the pipeline fitted on train
    best_preds = fitted[selected].predict(test["text"])
    y_prob = None
    if hasattr(fitted[selected].named_steps["model"], "predict_proba"):
        y_prob = fitted[selected].predict_proba(test["text"])

    label_names = sorted(usable["category"].unique().tolist())
    metrics = classification_metrics(
        test["category"].tolist(),
        best_preds.tolist(),
        y_prob=y_prob,
        label_names=label_names,
    )
    metrics["selected_model"] = selected
    metrics["candidate_weighted_f1"] = scores
    metrics["training_rows"] = int(len(usable))
    metrics["test_rows"] = int(len(test))
    metrics["n_categories"] = int(n_classes)

    # Retrain selected model on all available data for the saved artifact
    final = Pipeline([("tfidf", make_tfidf_vectorizer(max_features=8000)), ("model", clone(candidates[selected]))])
    final.fit(usable["text"], usable["category"])
    joblib.dump(final, model_path)

    save_classification_metrics(metrics, metrics_path)
    print_classification_summary(metrics)
    print(f"  Selected model    : {selected}")
    print(f"  Categories        : {metrics['n_categories']}")
    return metrics
