"""Tests for feature extraction utilities."""
from __future__ import annotations

import pytest

from src.features.match_features import extract_skills, skill_overlap, pair_features
from src.features.text_features import make_tfidf_vectorizer


# ── extract_skills ──────────────────────────────────────────────────────────

def test_extract_skills_finds_known_terms() -> None:
    assert {"python", "sql", "pandas"} <= extract_skills("Python developer with SQL and pandas experience")


def test_extract_skills_is_case_insensitive() -> None:
    assert "docker" in extract_skills("DOCKER and Kubernetes on AWS")


def test_extract_skills_returns_set() -> None:
    result = extract_skills("python python python")
    assert isinstance(result, set)
    assert result == {"python"}


def test_extract_skills_no_false_positives() -> None:
    # "flasked" should not match "flask"; word boundaries enforced
    result = extract_skills("She flasked the solution quickly")
    assert "flask" not in result


def test_extract_skills_empty_string() -> None:
    assert extract_skills("") == set()


def test_extract_skills_multi_word_skill() -> None:
    # "machine learning" is a two-word skill in the vocabulary
    result = extract_skills("proficient in machine learning algorithms")
    assert "machine learning" in result


# ── skill_overlap ───────────────────────────────────────────────────────────

def test_skill_overlap_is_zero_when_no_job_skills_match() -> None:
    assert skill_overlap("Experience with communication", "Python SQL Tableau") == 0.0


def test_skill_overlap_perfect_when_all_job_skills_match() -> None:
    score = skill_overlap("python sql", "python sql")
    assert score == pytest.approx(1.0)


def test_skill_overlap_partial() -> None:
    score = skill_overlap("python pandas", "python sql pandas tableau")
    # resume has 2/4 job skills → 0.5
    assert score == pytest.approx(0.5)


def test_skill_overlap_no_job_skills() -> None:
    # denominator guard: job has no skills → 0.0
    assert skill_overlap("python sql", "cover letter template") == 0.0


# ── pair_features ───────────────────────────────────────────────────────────

def test_pair_features_returns_two_floats() -> None:
    docs = ["python machine learning", "python sql data science"]
    vectorizer = make_tfidf_vectorizer()
    vectorizer.fit(docs)
    result = pair_features(docs[0], docs[1], vectorizer)
    assert len(result) == 2
    assert all(isinstance(v, float) for v in result)


def test_pair_features_identical_docs_cosine_one() -> None:
    doc = "python sql pandas machine learning"
    vectorizer = make_tfidf_vectorizer()
    vectorizer.fit([doc, "python data analysis"])
    cosine, overlap = pair_features(doc, doc, vectorizer)
    assert cosine == pytest.approx(1.0, abs=1e-5)
    assert overlap == pytest.approx(1.0)


# ── TF-IDF vectorizer ───────────────────────────────────────────────────────

def test_make_tfidf_vectorizer_configurable() -> None:
    vec = make_tfidf_vectorizer(max_features=100)
    assert vec.max_features == 100


def test_make_tfidf_vectorizer_bigrams() -> None:
    vec = make_tfidf_vectorizer()
    assert vec.ngram_range == (1, 2)
