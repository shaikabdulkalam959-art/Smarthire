"""End-to-end pipeline integration tests.

Each test trains a model on the tiny fixtures, persists it to a tmp path,
loads it back, and verifies predictions are sensible.
"""
from __future__ import annotations

import pytest

from src.models.classifier import train_classifier
from src.models.clustering import cluster_skill_gaps, train_clustering
from src.models.fit_predictor import predict_fit, train_fit_predictor
from src.models.recommender import recommend, train_recommender


# ── Classifier ───────────────────────────────────────────────────────────────

def test_classifier_trains_and_predicts(tiny_resumes, tmp_path):
    metrics_path = tmp_path / "classifier.json"
    model_path = tmp_path / "classifier.joblib"
    metrics = train_classifier(tiny_resumes, model_path, metrics_path)

    assert model_path.exists()
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    import joblib
    model = joblib.load(model_path)
    predictions = model.predict(["python machine learning scikit-learn pandas data science"])
    assert len(predictions) == 1
    assert predictions[0] in tiny_resumes["category"].unique()


def test_classifier_requires_two_categories(tmp_path, tiny_resumes):
    single_cat = tiny_resumes[tiny_resumes["category"] == "Data Science"].copy()
    with pytest.raises(ValueError, match="two categories"):
        train_classifier(single_cat, tmp_path / "m.joblib", tmp_path / "m.json")


def test_classifier_metrics_file_written(tiny_resumes, tmp_path):
    import json
    metrics_path = tmp_path / "metrics.json"
    train_classifier(tiny_resumes, tmp_path / "m.joblib", metrics_path)
    data = json.loads(metrics_path.read_text())
    assert "selected_model" in data
    assert "classification_report" in data


# ── Recommender ──────────────────────────────────────────────────────────────

def test_recommender_trains_and_recommends(tiny_jobs, tmp_path):
    artifact_path = tmp_path / "recommender.joblib"
    meta = train_recommender(tiny_jobs, artifact_path)

    assert artifact_path.exists()
    assert meta["job_count"] == len(tiny_jobs)

    import joblib
    artifact = joblib.load(artifact_path)
    results = recommend("python machine learning scikit-learn sql statistics", artifact, top_n=3)
    assert len(results) <= 3
    assert "match_score" in results.columns
    assert (results["match_score"] >= 0).all()


def test_recommender_top_n_respected(tiny_jobs, tmp_path):
    artifact_path = tmp_path / "recommender.joblib"
    train_recommender(tiny_jobs, artifact_path)
    import joblib
    artifact = joblib.load(artifact_path)
    results = recommend("python django postgresql docker", artifact, top_n=2)
    assert len(results) <= 2


def test_recommender_scores_are_sorted_descending(tiny_jobs, tmp_path):
    artifact_path = tmp_path / "recommender.joblib"
    train_recommender(tiny_jobs, artifact_path)
    import joblib
    artifact = joblib.load(artifact_path)
    results = recommend("javascript react typescript html css", artifact, top_n=4)
    scores = results["match_score"].tolist()
    assert scores == sorted(scores, reverse=True)


# ── Clustering ───────────────────────────────────────────────────────────────

def test_clustering_trains_and_gaps(tiny_jobs, tmp_path):
    artifact_path = tmp_path / "clustering.joblib"
    meta = train_clustering(tiny_jobs, artifact_path)

    assert artifact_path.exists()
    assert "selected_clusters" in meta
    assert meta["selected_clusters"] >= 2

    import joblib
    artifact = joblib.load(artifact_path)
    gaps = cluster_skill_gaps("python django rest api", artifact)
    assert isinstance(gaps, list)


def test_clustering_gaps_are_strings(tiny_jobs, tmp_path):
    artifact_path = tmp_path / "clustering.joblib"
    train_clustering(tiny_jobs, artifact_path)
    import joblib
    artifact = joblib.load(artifact_path)
    gaps = cluster_skill_gaps("communication payroll recruitment", artifact)
    assert all(isinstance(g, str) for g in gaps)


def test_clustering_requires_three_jobs(tmp_path):
    import pandas as pd
    tiny = pd.DataFrame([
        {"title": "a", "company": "x", "location": "y", "document": "python sql"},
        {"title": "b", "company": "x", "location": "y", "document": "java spring"},
    ])
    with pytest.raises(ValueError, match="three"):
        train_clustering(tiny, tmp_path / "c.joblib")


# ── Fit Predictor ─────────────────────────────────────────────────────────────

def test_fit_predictor_trains_and_predicts(tiny_resumes, tiny_jobs, tmp_path):
    artifact_path = tmp_path / "fit.joblib"
    meta = train_fit_predictor(tiny_resumes, tiny_jobs, artifact_path)

    assert artifact_path.exists()
    assert "accuracy" in meta
    assert 0.0 <= meta["accuracy"] <= 1.0

    import joblib
    artifact = joblib.load(artifact_path)
    score = predict_fit("python machine learning scikit-learn", tiny_jobs.iloc[0]["document"], artifact)
    assert 0.0 <= score <= 1.0


def test_fit_predictor_score_in_unit_interval(tiny_resumes, tiny_jobs, tmp_path):
    artifact_path = tmp_path / "fit.joblib"
    train_fit_predictor(tiny_resumes, tiny_jobs, artifact_path)
    import joblib
    artifact = joblib.load(artifact_path)
    for _, job in tiny_jobs.iterrows():
        score = predict_fit("python developer", job["document"], artifact)
        assert 0.0 <= score <= 1.0, f"Score out of range for {job['title']}"
