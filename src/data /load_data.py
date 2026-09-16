from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def load_resumes(path: Path) -> pd.DataFrame:
    """Load either generated `resumes.txt` or a labelled resume CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Resume data not found: {path}")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        normalized = {str(c).strip().lower(): c for c in frame.columns}
        text_col = next((normalized[k] for k in ("resume", "resume_str", "text", "resume_text") if k in normalized), None)
        label_col = next((normalized[k] for k in ("category", "label", "category_name") if k in normalized), None)
        if not text_col or not label_col:
            raise ValueError("Resume CSV needs a text column (resume/text) and a label column (category/label).")
        return frame[[text_col, label_col]].rename(columns={text_col: "text", label_col: "category"}).dropna()

    raw = path.read_text(encoding="utf-8")
    pattern = re.compile(r"CATEGORY:\s*(?P<category>[^\n]+)\s*\nRESUME:\s*(?P<text>.*?)(?=\n---\s*|\Z)", re.S | re.I)
    rows = [{"category": m.group("category").strip(), "text": m.group("text").strip()} for m in pattern.finditer(raw)]
    if not rows:
        raise ValueError("resumes.txt must use CATEGORY: ... followed by RESUME: ... and --- separators.")
    return pd.DataFrame(rows).query("text != '' and category != ''")
