"""Tests for the evaluate.py metrics module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    classification_metrics,
    clustering_metrics,
    precision_at_k,
    recommender_metrics,
    regression_metrics,
)


# ── classification_metrics ───────────────────────────────────────────────────

def test_classification_metrics_perfect_predictions():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1_weighted"] == pytest.approx(1.0)


def test_classification_metrics_returns_required_keys():
    y_true = ["A", "B", "A"]
    y_pred = ["A", "A", "B"]
    m = classification_metrics(y_true, y_pred)
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted",
                "confusion_matrix", "classification_report", "labels", "n_samples"):
        assert key in m, f"Missing key: {key}"


def test_classification_metrics_n_samples():
    y_true = ["A", "B", "C", "A"]
    y_pred = ["A", "B", "C", "A"]
    m = classification_metrics(y_true, y_pred)
    assert m["n_samples"] == 4


def test_classification_metrics_with_probabilities():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    m = classification_metrics(y_true, y_pred, y_prob=y_prob, label_names=[0, 1])
    assert "roc_auc" in m
    assert m["roc_auc"] == pytest.approx(1.0, abs=0.01)


def test_classification_metrics_confusion_matrix_shape():
    y_true = ["A", "B", "C"]
    y_pred = ["A", "B", "A"]
    m = classification_metrics(y_true, y_pred, label_names=["A", "B", "C"])
    cm = m["confusion_matrix"]
    assert len(cm) == 3
    assert len(cm[0]) == 3


# ── precision_at_k ───────────────────────────────────────────────────────────

def test_precision_at_k_perfect():
    recs = ["data scientist", "ml engineer", "ai researcher"]
    relevant = ["data scientist", "ml engineer"]
    assert precision_at_k(recs, relevant, k=2) == pytest.approx(1.0)


def test_precision_at_k_zero():
    recs = ["java developer", "devops engineer"]
    relevant = ["data scientist"]
    assert precision_at_k(recs, relevant, k=2) == pytest.approx(0.0)


def test_precision_at_k_partial():
    recs = ["data scientist", "java developer", "ml engineer"]
    relevant = ["data scientist", "ml engineer"]
    # Top 2: 1 hit → 1/2
    assert precision_at_k(recs, relevant, k=2) == pytest.approx(0.5)


def test_precision_at_k_empty_relevant():
    assert precision_at_k(["data scientist"], [], k=3) == pytest.approx(0.0)


def test_precision_at_k_case_insensitive():
    recs = ["Data Scientist", "Java Developer"]
    relevant = ["data scientist"]
    assert precision_at_k(recs, relevant, k=2) == pytest.approx(0.5)


# ── recommender_metrics ──────────────────────────────────────────────────────

def test_recommender_metrics_returns_dict(tiny_jobs):
    recommended = tiny_jobs.head(3)
    m = recommender_metrics("data science", recommended, tiny_jobs, k=3)
    assert "precision_at_3" in m
    assert "coverage" in m
    assert "resume_category" in m


def test_recommender_metrics_coverage_fraction(tiny_jobs):
    recommended = tiny_jobs.head(2)
    m = recommender_metrics("hr", recommended, tiny_jobs, k=2)
    assert 0.0 <= m["coverage"] <= 1.0


# ── regression_metrics ───────────────────────────────────────────────────────

def test_regression_metrics_perfect():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_returns_keys():
    m = regression_metrics([1, 2, 3], [1.1, 2.2, 2.9])
    for key in ("mae", "rmse", "r2", "n_samples"):
        assert key in m


def test_regression_metrics_n_samples():
    m = regression_metrics([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert m["n_samples"] == 5


# ── clustering_metrics ───────────────────────────────────────────────────────

def test_clustering_metrics_structure():
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = ["python machine learning", "java spring boot", "aws kubernetes docker",
            "sql database postgresql", "react javascript typescript"]
    vec = TfidfVectorizer()
    matrix = vec.fit_transform(docs)
    model = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = model.fit_predict(matrix)
    candidates = [{"n_clusters": 2, "silhouette": 0.3, "inertia": 100.0}]
    m = clustering_metrics(matrix, labels, candidates)
    for key in ("n_samples", "n_clusters_actual", "silhouette", "cluster_sizes"):
        assert key in m
    assert m["n_samples"] == 5
