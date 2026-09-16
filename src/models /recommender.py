from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.features.text_features import make_tfidf_vectorizer


def train_recommender(jobs: pd.DataFrame, artifact_path: Path) -> dict:
    vectorizer = make_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(jobs["document"])
    artifact = {"vectorizer": vectorizer, "matrix": matrix, "jobs": jobs.reset_index(drop=True)}
    joblib.dump(artifact, artifact_path)
    return {"job_count": int(len(jobs)), "vocabulary_size": int(len(vectorizer.vocabulary_))}


def recommend(resume_text: str, artifact: dict, top_n: int = 5) -> pd.DataFrame:
    scores = cosine_similarity(artifact["vectorizer"].transform([resume_text]), artifact["matrix"]).ravel()
    result = artifact["jobs"].copy()
    result["match_score"] = scores
    return result.nlargest(top_n, "match_score")
