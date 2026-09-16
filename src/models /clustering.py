from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.evaluate import clustering_metrics, print_clustering_summary
from src.features.text_features import make_tfidf_vectorizer


def train_clustering(jobs: pd.DataFrame, artifact_path: Path) -> dict:
    """Train K-Means job clustering with silhouette-based k selection.

    Saves vectorizer, model, jobs dataframe, and candidate evaluation results.
    Returns a metrics dict suitable for train_all.py summary.
    """
    vectorizer = make_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(jobs["document"])
    n_rows = matrix.shape[0]
    if n_rows < 3:
        raise ValueError("Clustering requires at least three job listings.")

    # Elbow + silhouette across candidate k values
    k_range = range(2, min(11, n_rows))
    results: list[dict] = []
    models: dict[int, KMeans] = {}
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(matrix)
        sil = float(silhouette_score(matrix, labels)) if len(set(labels)) > 1 else -1.0
        results.append({"n_clusters": k, "silhouette": sil, "inertia": float(model.inertia_)})
        models[k] = model

    selected_k = max(results, key=lambda r: r["silhouette"])["n_clusters"]
    selected_model = models[selected_k]
    final_labels = selected_model.labels_

    # Full clustering metrics via evaluate.py
    cm = clustering_metrics(matrix, final_labels, k_candidates=results)
    cm["selected_clusters"] = selected_k
    cm["candidates"] = results  # Elbow data for plotting
    print_clustering_summary(cm)

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": selected_model,
            "jobs": jobs,
            "selection": results,
        },
        artifact_path,
    )
    return cm


def cluster_skill_gaps(resume_text: str, artifact: dict) -> list[str]:
    """Return skills present in the nearest cluster but absent from the resume."""
    from src.features.match_features import extract_skills

    vector = artifact["vectorizer"].transform([resume_text])
    cluster = int(artifact["model"].predict(vector)[0])
    jobs = artifact["jobs"].copy()
    labels = artifact["model"].labels_
    target_text = " ".join(jobs.loc[np.asarray(labels) == cluster, "document"])
    return sorted(extract_skills(target_text) - extract_skills(resume_text))
