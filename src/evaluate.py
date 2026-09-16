"""Evaluation utilities for all SmartHire models.

Provides ready-to-use metric computation for:
  - Classification (classifier, fit predictor): accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  - Clustering: silhouette score, inertia, elbow-method data
  - Recommender: Precision@K, coverage
  - Regression (optional salary model): MAE, RMSE, R²
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


# ── Classification ────────────────────────────────────────────────────────────

def classification_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Return a comprehensive classification metric dictionary.

    Parameters
    ----------
    y_true:
        Ground-truth class labels.
    y_pred:
        Predicted class labels.
    y_prob:
        Predicted probabilities (shape: n_samples × n_classes). Required for ROC-AUC.
    label_names:
        Class names used in the confusion matrix and report.

    Returns
    -------
    dict with keys: accuracy, precision_macro, recall_macro, f1_macro, f1_weighted,
                    roc_auc (if probabilities given), classification_report (dict),
                    confusion_matrix (list[list[int]]), labels.
    """
    classes = label_names or sorted(set(y_true) | set(y_pred))
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "labels": list(classes),
        "n_samples": int(len(y_true)),
    }
    if y_prob is not None:
        try:
            if len(classes) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                y_bin = label_binarize(y_true, classes=classes)
                metrics["roc_auc"] = float(
                    roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
                )
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def save_classification_metrics(metrics: dict, path: Path) -> None:
    """Serialise classification metrics to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


# ── Clustering ────────────────────────────────────────────────────────────────

def clustering_metrics(
    matrix,
    labels: np.ndarray,
    k_candidates: list[dict],
) -> dict[str, Any]:
    """Return clustering evaluation summary.

    Parameters
    ----------
    matrix:
        Feature matrix used for clustering (scipy sparse or dense).
    labels:
        Cluster label assignments from the selected model.
    k_candidates:
        List of dicts with keys {n_clusters, silhouette, inertia} from the
        candidate search. Returned as-is.

    Returns
    -------
    dict with keys: selected_k, n_samples, n_clusters_actual, silhouette,
                    inertia, cluster_sizes, candidates.
    """
    from sklearn.metrics import silhouette_score as _sil

    unique = sorted(set(int(l) for l in labels))
    cluster_sizes = {int(k): int((labels == k).sum()) for k in unique}
    try:
        sil = float(_sil(matrix, labels)) if len(unique) > 1 else -1.0
    except Exception:
        sil = -1.0

    return {
        "n_samples": int(len(labels)),
        "n_clusters_actual": int(len(unique)),
        "silhouette": sil,
        "cluster_sizes": cluster_sizes,
        "candidates": k_candidates,
    }


# ── Recommender ───────────────────────────────────────────────────────────────

def precision_at_k(
    recommended_titles: list[str],
    relevant_titles: list[str],
    k: int = 5,
) -> float:
    """Precision@K: fraction of top-K recommendations that are relevant.

    Parameters
    ----------
    recommended_titles:
        Ordered list of recommended job titles (most relevant first).
    relevant_titles:
        Ground-truth set of job titles considered relevant for this query.
    k:
        Cut-off position.

    Returns
    -------
    float in [0, 1].
    """
    if not relevant_titles:
        return 0.0
    top_k = recommended_titles[:k]
    relevant_set = {t.lower() for t in relevant_titles}
    hits = sum(1 for t in top_k if t.lower() in relevant_set)
    return hits / k


def recommender_metrics(
    resume_category: str,
    recommended: pd.DataFrame,
    all_jobs: pd.DataFrame,
    k: int = 5,
) -> dict[str, Any]:
    """Evaluate a single recommendation result.

    Uses resume category as a proxy for relevance:
    jobs whose title contains the category keywords are considered relevant.
    """
    category_keywords = resume_category.lower().split()
    relevant = all_jobs[
        all_jobs["title"].str.lower().apply(
            lambda t: any(kw in t for kw in category_keywords)
        )
    ]["title"].tolist()

    rec_titles = recommended["title"].tolist()
    p_at_k = precision_at_k(rec_titles, relevant, k=k)
    coverage = len(recommended) / max(len(all_jobs), 1)

    return {
        "resume_category": resume_category,
        "relevant_job_count": len(relevant),
        "recommended_count": len(recommended),
        f"precision_at_{k}": p_at_k,
        "coverage": float(coverage),
    }


def batch_recommender_metrics(
    sample_resumes: pd.DataFrame,
    artifact: dict,
    jobs: pd.DataFrame,
    k: int = 5,
    n_samples: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate recommender on a sample of resumes.

    Returns mean Precision@K and per-category breakdown.
    """
    from src.models.recommender import recommend

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sample_resumes), size=min(n_samples, len(sample_resumes)), replace=False)
    results = []
    for i in idx:
        row = sample_resumes.iloc[int(i)]
        recs = recommend(row["text"], artifact, top_n=k)
        results.append(recommender_metrics(row["category"], recs, jobs, k=k))

    mean_pk = float(np.mean([r[f"precision_at_{k}"] for r in results]))
    return {
        f"mean_precision_at_{k}": mean_pk,
        "n_evaluated": len(results),
        "per_sample": results,
    }


# ── Regression (optional salary model) ───────────────────────────────────────

def regression_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
) -> dict[str, float]:
    """Return MAE, RMSE, and R² for regression model evaluation."""
    from sklearn.metrics import r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2, "n_samples": int(len(y_true))}


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def print_classification_summary(metrics: dict) -> None:
    """Print a concise classification summary to stdout."""
    print(f"\n{'-'*55}")
    print(f"  Classification Metrics")
    print(f"{'-'*55}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision (macro) : {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro)    : {metrics['recall_macro']:.4f}")
    print(f"  F1 (macro)        : {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted)     : {metrics['f1_weighted']:.4f}")
    if "roc_auc" in metrics and metrics["roc_auc"] is not None:
        print(f"  ROC-AUC (macro)   : {metrics['roc_auc']:.4f}")
    print(f"  Samples           : {metrics['n_samples']}")
    print(f"{'-'*55}\n")


def print_clustering_summary(metrics: dict) -> None:
    """Print a concise clustering summary to stdout."""
    print(f"\n{'-'*55}")
    print(f"  Clustering Metrics")
    print(f"{'-'*55}")
    print(f"  Clusters          : {metrics['n_clusters_actual']}")
    print(f"  Silhouette score  : {metrics['silhouette']:.4f}")
    sizes = metrics.get("cluster_sizes", {})
    for k, v in sorted(sizes.items()):
        print(f"    Cluster {k:>2}      : {v} samples")
    print(f"{'-'*55}\n")
