.PHONY: help seed train test report app clean bootstrap

## Show this help message
help:
	@echo "SmartHire — ML Resume Matching"
	@echo ""
	@echo "Commands:"
	@echo "  make seed        Generate 500 seed resumes (25 categories × 20)"
	@echo "  make train       Train all 4 ML models"
	@echo "  make test        Run the full test suite"
	@echo "  make report      Generate metrics_summary.md"
	@echo "  make app         Launch the Streamlit UI"
	@echo "  make bootstrap   Full pipeline: seed → train → test → report"
	@echo "  make clean       Remove generated artifacts"

## Generate seed training data (no Ollama required — 25 categories × 20 = 500 records)
seed:
	.venv/bin/python scripts/seed_resumes.py --per-category 20

## Train all ML models (classifier, recommender, clustering, fit predictor)
train:
	.venv/bin/python train_all.py

## Generate a post-training markdown metrics report
report:
	.venv/bin/python scripts/report_metrics.py

## Run the full pytest test suite
test:
	.venv/bin/python -m pytest tests/ -v

## Launch the Streamlit web app
app:
	.venv/bin/streamlit run app/streamlit_app.py

## Full bootstrap: generate data → train → test → report
bootstrap: seed train test report
	@echo "✅ SmartHire bootstrap complete."

## Remove all generated artifacts (data, models, metrics)
clean:
	rm -f models/*.joblib
	rm -f data/raw/resumes.txt data/raw/resumes_manifest.json
	rm -f data/processed/job_corpus.csv
	rm -rf reports/metrics/
