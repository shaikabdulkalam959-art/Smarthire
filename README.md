# SmartHire — Resume Intelligence Platform

SmartHire is a classical machine-learning pipeline for **resume category prediction**,
**content-based job recommendations**, **role clustering**, **fit scoring**, and
**skill-gap reporting**.

Ollama is used *only* to generate synthetic labelled training resumes via
`scripts/generate_resumes.py` — it is not used during prediction or inference.
A self-contained offline generator (`scripts/seed_resumes.py`) is provided so the
full pipeline can be trained and run without any external services.

---

## Quick start (offline, no Ollama required)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Generate seed training data (no network needed — template-based)
python scripts/seed_resumes.py --per-category 10   # → data/raw/resumes.txt

# 3. Train all four ML models
python train_all.py                                 # → models/*.joblib

# 4. (Optional) Generate a training metrics report
python scripts/report_metrics.py                   # → reports/metrics_summary.md

# 5. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

---

## Ollama-powered data generation (optional, richer data)

```bash
# Pull a model and start Ollama
ollama pull qwen3:8b
ollama serve

# Generate synthetic labelled resumes (20 per category)
python scripts/generate_resumes.py --per-category 20

# Faster bootstrap (uses Qwen seeds + labelled variants)
python scripts/generate_resumes.py --fast-bootstrap --per-category 5

# Train and run as above
python train_all.py
streamlit run app/streamlit_app.py
```

Configure `OLLAMA_MODEL`, `OLLAMA_HOST`, and `OLLAMA_TIMEOUT_SECONDS` in `.env`.

---

## Project structure

```
resume_screening_ml/
├── app/
│   └── streamlit_app.py          # Premium dark-theme Streamlit UI
├── data/
│   ├── raw/                      # resumes.txt + optional job CSVs
│   ├── interim/
│   └── processed/                # job_corpus.csv (auto-generated)
├── models/                       # Trained .joblib artifacts
├── reports/
│   └── metrics/                  # JSON metrics + metrics_summary.md
├── scripts/
│   ├── seed_resumes.py           # Offline template-based data generator
│   ├── generate_resumes.py       # Ollama-powered data generator
│   └── report_metrics.py         # Post-training Markdown report
├── src/
│   ├── config.py                 # Paths and environment variables
│   ├── data/
│   │   ├── load_data.py          # Resume TXT/CSV loader
│   │   └── preprocess.py         # Job corpus normalisation
│   ├── features/
│   │   ├── match_features.py     # Skill extraction + pair features
│   │   └── text_features.py      # TF-IDF vectorizer factory
│   ├── models/
│   │   ├── classifier.py         # Resume category classifier
│   │   ├── clustering.py         # K-Means role clustering + skill gaps
│   │   ├── fit_predictor.py      # Pair-wise fit probability model
│   │   └── recommender.py        # Cosine-similarity job recommender
│   └── parsing/
│       └── resume_parser.py      # TXT / PDF / DOCX text extraction
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_features.py          # Feature extraction unit tests
│   ├── test_parser.py            # Resume parser tests
│   ├── test_pipeline.py          # End-to-end model pipeline tests
│   └── test_preprocess.py        # Data loading & preprocessing tests
├── train_all.py                  # One-command training entry point
├── pyproject.toml                # Pytest configuration
└── requirements.txt
```

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Adding real job data

Drop any job-listing CSV into `data/raw/` (excluding files with "resume" in the name).
The preprocessor accepts flexible column names:

| Field | Accepted column names |
|-------|-----------------------|
| Title | `title`, `job title`, `job_title`, `designation` |
| Company | `company`, `company name`, `company_name`, `employer` |
| Location | `location`, `job location`, `job_location`, `city` |
| Skills | `skills`, `skill`, `key skills`, `key_skills` |
| Description | `description`, `job description`, `job_description`, `details` |
| Experience | `experience`, `experience required`, `exp` |

Public datasets such as the Kaggle "Jobs on Naukri.com" CSV work out of the box.

---

## App features

The Streamlit app accepts pasted text as well as `.txt`, `.pdf`, and `.docx` uploads,
then presents results in three tabs:

| Tab | Content |
|-----|---------|
| 🏷️ Career Match | Predicted category, detected skills, confidence bars |
| 💼 Job Board | Top 5 jobs ranked by semantic match + fit score |
| 🔬 Skill Gap Report | Missing skills from nearest cluster, learning priority grid |

---

## Notes

- The 30-minute Ollama timeout accommodates slower local inference and does not enable model thinking.
- Generated resumes are synthetic; review for bias and data quality before use in production.
- Replace weak-supervision fit labels with real hiring outcomes for improved accuracy.
