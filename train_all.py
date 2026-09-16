#!/usr/bin/env python3
"""One command to build data and train every SmartHire ML artifact.

Usage:
    python train_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, RAW_DATA_DIR, ensure_project_directories
from src.data.load_data import load_resumes
from src.data.preprocess import build_job_corpus
from src.evaluate import batch_recommender_metrics
from src.models.classifier import train_classifier
from src.models.clustering import train_clustering
from src.models.fit_predictor import train_fit_predictor
from src.models.recommender import train_recommender


def main() -> None:
    ensure_project_directories()

    # ── Load data ──────────────────────────────────────────────────────────
    resume_path = RAW_DATA_DIR / "resumes.txt"
    if not resume_path.exists():
        # Try CSV fallback
        csv_candidates = list(RAW_DATA_DIR.glob("*resume*.csv")) + list(RAW_DATA_DIR.glob("*Resume*.csv"))
        if csv_candidates:
            resume_path = csv_candidates[0]
        else:
            raise SystemExit(
                "Missing resume data. Run first:\n"
                "  python scripts/seed_resumes.py\n"
                "Or place a Kaggle resume CSV in data/raw/ and re-run."
            )

    print(f"\nLoading resumes from: {resume_path}")
    resumes = load_resumes(resume_path)
    print(f"Loaded {len(resumes)} resumes across {resumes['category'].nunique()} categories.")

    print("\nBuilding job corpus...")
    jobs = build_job_corpus()
    print(f"Job corpus: {len(jobs)} listings.")

    metrics_dir = REPORTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Train models ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training Resume Category Classifier")
    print("=" * 60)
    classifier_metrics = train_classifier(
        resumes,
        MODELS_DIR / "resume_classifier.joblib",
        metrics_dir / "classifier.json",
    )

    print("\n" + "=" * 60)
    print("  Training Job Recommender")
    print("=" * 60)
    recommender_metrics_basic = train_recommender(jobs, MODELS_DIR / "recommender.joblib")
    import joblib
    rec_artifact = joblib.load(MODELS_DIR / "recommender.joblib")
    rec_eval = batch_recommender_metrics(resumes, rec_artifact, jobs, k=5, n_samples=40)
    recommender_metrics_basic.update(rec_eval)
    print(f"  Job count         : {recommender_metrics_basic['job_count']}")
    print(f"  Vocabulary size   : {recommender_metrics_basic['vocabulary_size']}")
    print(f"  Precision@5 (mean): {rec_eval['mean_precision_at_5']:.4f}")
    (metrics_dir / "recommender.json").write_text(
        json.dumps(recommender_metrics_basic, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("  Training Role Clustering")
    print("=" * 60)
    clustering_result = train_clustering(jobs, MODELS_DIR / "clustering.joblib")
    (metrics_dir / "clustering.json").write_text(
        json.dumps(clustering_result, indent=2, default=str), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("  Training Fit Predictor")
    print("=" * 60)
    fit_metrics = train_fit_predictor(
        resumes, jobs, MODELS_DIR / "fit_predictor.joblib"
    )
    (metrics_dir / "fit_predictor.json").write_text(
        json.dumps(fit_metrics, indent=2, default=str), encoding="utf-8"
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = {
        "classifier": classifier_metrics,
        "recommender": recommender_metrics_basic,
        "clustering": clustering_result,
        "fit_predictor": fit_metrics,
    }
    summary_path = metrics_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  [+] Training complete!")
    print("=" * 60)
    print(f"  Artifacts : {MODELS_DIR}")
    print(f"  Metrics   : {metrics_dir}")
    print()
    print(f"  Classifier accuracy   : {classifier_metrics.get('accuracy', 0):.3f}")
    print(f"  Classifier F1 (wt)    : {classifier_metrics.get('f1_weighted', 0):.3f}")
    print(f"  Recommender P@5       : {rec_eval['mean_precision_at_5']:.3f}")
    print(f"  Clustering k          : {clustering_result.get('selected_clusters', '?')}")
    print(f"  Clustering silhouette : {clustering_result.get('silhouette', 0):.3f}")
    print(f"  Fit predictor acc     : {fit_metrics.get('accuracy', 0):.3f}")
    print()
    print("  Run the app: streamlit run app/streamlit_app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
