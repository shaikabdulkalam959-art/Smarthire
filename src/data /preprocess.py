from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

DEMO_JOBS = [
    ("Data Analyst", "Insight Labs", "Bengaluru", "python sql excel tableau statistics pandas", "Analyse business data, build SQL dashboards and communicate insights. Python, pandas, Tableau and statistics required.", "0-2 years"),
    ("Data Scientist", "Vertex AI", "Hyderabad", "python machine learning scikit-learn statistics sql", "Develop machine learning models, evaluate experiments and deploy data products with Python and SQL.", "1-3 years"),
    ("Machine Learning Engineer", "CloudScale", "Pune", "python pytorch tensorflow mlops docker aws", "Build production ML pipelines, train models, use Docker and cloud infrastructure.", "2-4 years"),
    ("Backend Developer", "API Works", "Mumbai", "python django fastapi postgresql redis docker", "Design REST APIs and reliable backend services using Python, Django, PostgreSQL and Docker.", "1-3 years"),
    ("Frontend Developer", "Pixel Studio", "Remote", "javascript react typescript html css", "Create accessible React interfaces with TypeScript, JavaScript, HTML and CSS.", "0-2 years"),
    ("HR Specialist", "People First", "Delhi", "recruitment onboarding hrms communication payroll", "Manage recruitment, employee onboarding, HRMS records and stakeholder communication.", "1-3 years"),
    ("Business Analyst", "Strategy Co", "Chennai", "sql excel tableau requirements stakeholder analysis", "Gather requirements, produce business analysis and create SQL and Tableau reporting.", "1-3 years"),
    ("DevOps Engineer", "InfraOps", "Remote", "aws kubernetes docker ci cd terraform linux", "Maintain cloud infrastructure with AWS, Kubernetes, Terraform, Docker and CI/CD.", "2-5 years"),
]

ALIASES = {
    "title": ("title", "job title", "job_title", "designation"),
    "company": ("company", "company name", "company_name", "employer"),
    "location": ("location", "job location", "job_location", "city"),
    "skills": ("skills", "skill", "key skills", "key_skills"),
    "description": ("description", "job description", "job_description", "job desc", "details"),
    "experience": ("experience", "experience required", "experience_required", "exp"),
}


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def _pick(columns: dict[str, str], choices: tuple[str, ...]) -> str | None:
    return next((columns[x] for x in choices if x in columns), None)


def _normalize_jobs(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {str(c).strip().lower(): c for c in frame.columns}
    output: dict[str, object] = {}
    for target, choices in ALIASES.items():
        source = _pick(columns, choices)
        output[target] = frame[source] if source else ""
    result = pd.DataFrame(output)
    result = result.apply(lambda series: series.map(clean_text))
    result = result[result["title"].ne("")]
    result["document"] = (result["title"] + " " + result["skills"] + " " + result["description"]).map(clean_text)
    return result[result["document"].str.len().gt(10)].drop_duplicates(subset=["title", "company", "document"])


def build_job_corpus(raw_dir: Path = RAW_DATA_DIR, output_path: Path | None = None) -> pd.DataFrame:
    """Merge any raw job-listing CSV files into the standard job corpus schema."""
    output_path = output_path or PROCESSED_DATA_DIR / "job_corpus.csv"
    frames: list[pd.DataFrame] = []
    for csv_path in raw_dir.glob("*.csv"):
        if "resume" in csv_path.name.lower():
            continue
        try:
            frames.append(_normalize_jobs(pd.read_csv(csv_path, low_memory=False)))
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    corpus = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(DEMO_JOBS, columns=["title", "company", "location", "skills", "description", "experience"])
    if "document" not in corpus:
        corpus["document"] = (corpus["title"] + " " + corpus["skills"] + " " + corpus["description"]).map(clean_text)
    corpus = corpus.drop_duplicates(subset=["title", "company", "document"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(output_path, index=False)
    return corpus
