"""Tests for data loading and job corpus preprocessing."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_resumes
from src.data.preprocess import build_job_corpus, clean_text


# ── clean_text ───────────────────────────────────────────────────────────────

def test_clean_text_lowercases() -> None:
    assert clean_text("Python SQL") == "python sql"


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  python   sql  ") == "python sql"


def test_clean_text_handles_none() -> None:
    assert clean_text(None) == ""


def test_clean_text_handles_numeric() -> None:
    result = clean_text(42)
    assert result == "42"


# ── load_resumes from txt ────────────────────────────────────────────────────

def test_load_resumes_txt_parses_records(tmp_path: Path) -> None:
    content = textwrap.dedent("""\
        CATEGORY: Data Scientist
        RESUME:
        Experienced Python and ML engineer.
        ---

        CATEGORY: Backend Developer
        RESUME:
        Django and PostgreSQL specialist.
        ---
    """)
    path = tmp_path / "resumes.txt"
    path.write_text(content, encoding="utf-8")
    df = load_resumes(path)
    assert len(df) == 2
    assert set(df.columns) == {"category", "text"}
    assert "Data Scientist" in df["category"].values


def test_load_resumes_txt_drops_empty_entries(tmp_path: Path) -> None:
    content = "CATEGORY: Data Scientist\nRESUME:\n\n---\n\nCATEGORY: Backend Developer\nRESUME:\nDjango expert.\n---\n"
    path = tmp_path / "resumes.txt"
    path.write_text(content, encoding="utf-8")
    df = load_resumes(path)
    # Empty text row should be filtered out
    assert all(df["text"].str.len() > 0)


def test_load_resumes_txt_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_resumes(tmp_path / "nonexistent.txt")


def test_load_resumes_txt_bad_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "resumes.txt"
    path.write_text("This is completely wrong format with no markers.", encoding="utf-8")
    with pytest.raises(ValueError, match="CATEGORY"):
        load_resumes(path)


# ── load_resumes from CSV ────────────────────────────────────────────────────

def test_load_resumes_csv_standard_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "resume": ["Python developer", "Django backend dev"],
        "category": ["Data Scientist", "Backend Developer"],
    })
    path = tmp_path / "resumes.csv"
    df.to_csv(path, index=False)
    result = load_resumes(path)
    assert set(result.columns) == {"text", "category"}
    assert len(result) == 2


def test_load_resumes_csv_alias_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "resume_str": ["Python ML expert"],
        "Category": ["Data Scientist"],
    })
    path = tmp_path / "resumes.csv"
    df.to_csv(path, index=False)
    result = load_resumes(path)
    assert result.iloc[0]["text"] == "Python ML expert"


def test_load_resumes_csv_missing_columns_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"col_a": ["x"], "col_b": ["y"]})
    path = tmp_path / "resumes.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="text column"):
        load_resumes(path)


# ── build_job_corpus demo fallback ───────────────────────────────────────────

def test_build_job_corpus_returns_dataframe_with_document_column(tmp_path: Path) -> None:
    # No CSV files in tmp_path → falls back to DEMO_JOBS
    result = build_job_corpus(raw_dir=tmp_path, output_path=tmp_path / "out.csv")
    assert isinstance(result, pd.DataFrame)
    assert "document" in result.columns
    assert len(result) > 0


def test_build_job_corpus_demo_has_required_columns(tmp_path: Path) -> None:
    result = build_job_corpus(raw_dir=tmp_path, output_path=tmp_path / "out.csv")
    for col in ("title", "company", "location", "document"):
        assert col in result.columns


def test_build_job_corpus_from_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "job title": ["Data Scientist"],
        "company": ["TechCo"],
        "location": ["Remote"],
        "skills": ["python machine learning"],
        "description": ["Build and deploy ML models"],
        "experience": ["2-4 years"],
    })
    csv_path = tmp_path / "jobs.csv"
    df.to_csv(csv_path, index=False)
    result = build_job_corpus(raw_dir=tmp_path, output_path=tmp_path / "out.csv")
    assert len(result) >= 1
    assert "data scientist" in result["title"].values
