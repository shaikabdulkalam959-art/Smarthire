from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def make_tfidf_vectorizer(max_features: int = 8_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        max_features=max_features,
    )
