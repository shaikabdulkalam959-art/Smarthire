from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import classification_metrics, print_classification_summary
from src.features.match_features import extract_skills, pair_features
from src.features.text_features import make_tfidf_vectorizer


def train_fit_predictor(resumes: pd.DataFrame, jobs: pd.DataFrame, artifact_path: Path) -> dict:
    """Train a binary fit/shortlisting model from weak supervision labels.

    Labels are derived from category-title agreement (positive) and random
    mismatches (negative).
    """
    vectorizer = make_tfidf_vectorizer(max_features=8000)
    vectorizer.fit(pd.concat([resumes["text"], jobs["document"]], ignore_index=True))

    resume_vecs = vectorizer.transform(resumes["text"])
    job_vecs = vectorizer.transform(jobs["document"])

    # Precompute skill sets for resumes and jobs
    resume_skills_list = [extract_skills(t) for t in resumes["text"]]
    job_skills_list = [extract_skills(d) for d in jobs["document"]]

    rng = np.random.default_rng(42)
    features, labels = [], []
    titles = jobs["title"].str.lower()

    for r_idx, (_, resume) in enumerate(resumes.iterrows()):
        category = str(resume["category"]).lower()
        matching_indices = jobs[titles.str.contains(category, regex=False)].index.tolist()
        positives_idx = matching_indices[:2] if matching_indices else [int(rng.integers(len(jobs)))]
        all_indices = set(range(len(jobs)))
        negative_candidates = list(all_indices - set(positives_idx))
        negatives_idx = rng.choice(negative_candidates, size=min(2, len(negative_candidates)), replace=False).tolist() if negative_candidates else []

        r_vec = resume_vecs[r_idx]
        r_skills = resume_skills_list[r_idx]

        for j_idx in positives_idx:
            cos = float(cosine_similarity(r_vec, job_vecs[j_idx])[0, 0])
            j_skills = job_skills_list[j_idx]
            overlap = len(r_skills & j_skills) / max(len(j_skills), 1)
            features.append([cos, overlap])
            labels.append(1)

        for j_idx in negatives_idx:
            cos = float(cosine_similarity(r_vec, job_vecs[j_idx])[0, 0])
            j_skills = job_skills_list[j_idx]
            overlap = len(r_skills & j_skills) / max(len(j_skills), 1)
            features.append([cos, overlap])
            labels.append(0)

    if len(set(labels)) < 2:
        raise ValueError("Fit predictor could not create both positive and negative training pairs.")

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=42, stratify=labels
    )
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = classification_metrics(
        y_test, y_pred, y_prob=y_prob, label_names=[0, 1]
    )
    metrics["synthetic_pair_count"] = len(labels)
    metrics["note"] = (
        "Labels are weak supervision from category/title agreement. "
        "Replace with real hiring outcomes for production quality."
    )

    joblib.dump({"vectorizer": vectorizer, "model": model}, artifact_path)

    print("\n  Fit Predictor (binary shortlisting):")
    print_classification_summary(metrics)
    return metrics


def predict_fit(resume_text: str, job_text: str, artifact: dict) -> float:
    """Return probability [0, 1] that the resume fits the given job."""
    features = [pair_features(resume_text, job_text, artifact["vectorizer"])]
    return float(artifact["model"].predict_proba(features)[0, 1])
