"""SmartHire — premium Streamlit UI with glassmorphism dark theme."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS_DIR, REPORTS_DIR
from src.features.match_features import extract_skills, SKILL_VOCABULARY
from src.models.clustering import cluster_skill_gaps
from src.models.fit_predictor import predict_fit
from src.models.recommender import recommend
from src.parsing.resume_parser import extract_resume_text


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartHire — Resume Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary:   #0d0f1a;
    --bg-card:      rgba(255, 255, 255, 0.04);
    --bg-card-hover:rgba(255, 255, 255, 0.08);
    --border:       rgba(255, 255, 255, 0.10);
    --accent-1:     #6c63ff;
    --accent-2:     #a78bfa;
    --accent-3:     #34d399;
    --accent-warn:  #fbbf24;
    --accent-red:   #f87171;
    --text-primary: #f1f5f9;
    --text-muted:   #94a3b8;
    --radius:       14px;
    --radius-sm:    8px;
    --glow:         0 0 40px rgba(108, 99, 255, 0.18);
}

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: var(--bg-primary) !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text-primary); }

/* ── Hero header ── */
.sh-hero {
    background: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(167,139,250,0.08) 60%, rgba(52,211,153,0.06) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem 2.5rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow);
}
.sh-hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(108,99,255,0.25) 0%, transparent 70%);
    pointer-events: none;
}
.sh-hero h1 {
    font-size: 2.4rem !important; font-weight: 800 !important;
    background: linear-gradient(135deg, #a78bfa, #6c63ff, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0 !important;
}
.sh-hero p { color: var(--text-muted); margin: 0; font-size: 1rem; }

/* ── Glass card ── */
.sh-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.sh-card:hover { border-color: rgba(108,99,255,0.35); box-shadow: var(--glow); }

/* ── Stat pill ── */
.sh-pill {
    display: inline-block;
    background: rgba(108,99,255,0.18);
    border: 1px solid rgba(108,99,255,0.35);
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem; font-weight: 600;
    color: var(--accent-2);
    margin: 0.15rem;
    transition: background 0.15s;
}
.sh-pill:hover { background: rgba(108,99,255,0.32); }
.sh-pill-green  { background:rgba(52,211,153,0.15); border-color:rgba(52,211,153,0.4); color:#34d399; }
.sh-pill-yellow { background:rgba(251,191,36,0.15); border-color:rgba(251,191,36,0.4); color:#fbbf24; }
.sh-pill-red    { background:rgba(248,113,113,0.15); border-color:rgba(248,113,113,0.4); color:#f87171; }

/* ── Confidence bar ── */
.sh-conf-wrap { margin: 0.5rem 0; }
.sh-conf-label { font-size: 0.82rem; color: var(--text-muted); margin-bottom: 4px; }
.sh-conf-bar-bg {
    height: 10px; border-radius: 999px;
    background: rgba(255,255,255,0.07);
    overflow: hidden;
}
.sh-conf-bar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, #6c63ff, #a78bfa, #34d399);
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}

/* ── Section label ── */
.sh-section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--accent-2);
    margin-bottom: 0.6rem;
}

/* ── Category badge ── */
.sh-cat-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, rgba(108,99,255,0.2), rgba(167,139,250,0.12));
    border: 1px solid rgba(108,99,255,0.4);
    border-radius: var(--radius);
    padding: 0.75rem 1.25rem;
    font-size: 1.15rem; font-weight: 700;
    color: var(--accent-2);
}

/* ── Score color helpers ── */
.score-high  { color: #34d399 !important; font-weight: 700; }
.score-mid   { color: #fbbf24 !important; font-weight: 700; }
.score-low   { color: #f87171 !important; font-weight: 700; }

/* ── Dataframe overrides ── */
.stDataFrame, iframe { border-radius: var(--radius-sm) !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    color: var(--text-muted) !important;
    font-weight: 500; padding: 0.5rem 1.1rem;
    border: 1px solid transparent !important;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: rgba(108,99,255,0.18) !important;
    border-color: var(--border) var(--border) transparent !important;
    color: var(--accent-2) !important;
}

/* ── Upload & textarea ── */
.stFileUploader, .stTextArea textarea {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
.stTextArea textarea { min-height: 220px !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6c63ff, #a78bfa) !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    font-weight: 700 !important; letter-spacing: 0.02em !important;
    padding: 0.6rem 2rem !important; font-size: 0.95rem !important;
    box-shadow: 0 4px 24px rgba(108,99,255,0.35) !important;
    transition: box-shadow 0.2s, transform 0.15s !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 32px rgba(108,99,255,0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Alert / warning ── */
.stAlert { border-radius: var(--radius-sm) !important; }

/* ── No match note ── */
.sh-no-gap {
    color: var(--accent-3); font-weight: 600; font-size: 0.95rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    required = [
        "resume_classifier.joblib",
        "recommender.joblib",
        "clustering.joblib",
        "fit_predictor.joblib",
    ]
    missing = [n for n in required if not (MODELS_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(
            "Run `python train_all.py` first. Missing: " + ", ".join(missing)
        )
    return {n: joblib.load(MODELS_DIR / n) for n in required}


# ── Helper renderers ──────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 0.6:
        return "score-high"
    if score >= 0.35:
        return "score-mid"
    return "score-low"


def _conf_bar(label: str, value: float) -> None:
    pct = int(value * 100)
    st.markdown(
        f"""
        <div class="sh-conf-wrap">
            <div class="sh-conf-label">{label}</div>
            <div class="sh-conf-bar-bg">
                <div class="sh-conf-bar-fill" style="width:{pct}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _pill(text: str, style: str = "") -> str:
    cls = f"sh-pill {style}".strip()
    return f'<span class="{cls}">{text}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 0.5rem">
            <div style="font-size:2.8rem">🎯</div>
            <div style="font-size:1.3rem;font-weight:800;
                background:linear-gradient(135deg,#a78bfa,#34d399);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent">
                SmartHire
            </div>
            <div style="font-size:0.75rem;color:#64748b;margin-top:2px">
                Classical ML · Resume Intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("#### 📋 How it works")
    st.markdown(
        """
1. **Upload** a `.txt`, `.pdf`, or `.docx` resume, **or** paste text below.
2. Click **Analyse Resume**.
3. Get your **predicted career category**, top **job matches**, and a personalised **skill-gap report**.
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("#### ⚙️ Pipeline")
    st.markdown(
        """
- **Classifier** — TF-IDF + LinearSVC / Logistic Regression
- **Recommender** — TF-IDF cosine similarity
- **Clustering** — K-Means (silhouette selection)
- **Fit Predictor** — Logistic Regression on pair features
        """
    )
    st.divider()
    st.caption(f"🔬 Skill vocabulary: **{len(SKILL_VOCABULARY)}** terms")

    # ── Live model metrics from training JSON ─────────────────────────────────
    summary_path = REPORTS_DIR / "metrics" / "training_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            clf = summary.get("classifier", {})
            rec = summary.get("recommender", {})
            clu = summary.get("clustering", {})
            fit = summary.get("fit_predictor", {})
            st.markdown("#### 📊 Model Metrics")
            st.markdown(
                f"""
| Model | Metric | Score |
|-------|--------|-------|
| Classifier | Accuracy | `{clf.get('accuracy', 0):.3f}` |
| Classifier | F1 (weighted) | `{clf.get('f1_weighted', 0):.3f}` |
| Classifier | Categories | `{clf.get('n_categories', '?')}` |
| Recommender | Precision@5 | `{rec.get('mean_precision_at_5', 0):.3f}` |
| Clustering | Silhouette | `{clu.get('silhouette', 0):.3f}` |
| Clustering | k | `{clu.get('selected_clusters', '?')}` |
| Fit Predictor | Accuracy | `{fit.get('accuracy', 0):.3f}` |
| Fit Predictor | ROC-AUC | `{fit.get('roc_auc') or 'N/A'}` |
                """
            )
        except Exception:
            pass

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="sh-hero">
    <h1>SmartHire</h1>
    <p>AI-powered resume intelligence — match your profile to the right opportunities.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Upload / paste ────────────────────────────────────────────────────────────
col_upload, col_paste = st.columns([1, 2], gap="large")

with col_upload:
    st.markdown('<div class="sh-section-label">Upload resume</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a file here",
        type=["txt", "pdf", "docx"],
        label_visibility="collapsed",
    )
    initial_text = ""
    if uploaded:
        try:
            initial_text = extract_resume_text(uploaded.name, uploaded.getvalue())
            st.success(f"✅ Loaded **{uploaded.name}**")
        except ValueError as e:
            st.error(str(e))

with col_paste:
    st.markdown('<div class="sh-section-label">Resume text</div>', unsafe_allow_html=True)
    resume_text = st.text_area(
        "Paste or review your resume",
        value=initial_text,
        height=240,
        placeholder="Paste your skills, education, experience, and projects here …",
        label_visibility="collapsed",
    )

# ── Analyse button ────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    analyse = st.button("🔍 Analyse Resume", type="primary", use_container_width=True)

st.divider()

# ── Results ───────────────────────────────────────────────────────────────────
if analyse:
    if len(resume_text.strip()) < 40:
        st.warning("⚠️ Please provide at least a short resume or profile (40+ characters).")
        st.stop()

    with st.spinner("Analysing your resume …"):
        try:
            artifacts = load_artifacts()
        except FileNotFoundError as err:
            st.error(str(err))
            st.stop()

        classifier = artifacts["resume_classifier.joblib"]
        predicted_category = classifier.predict([resume_text])[0]

        matches: pd.DataFrame = recommend(
            resume_text, artifacts["recommender.joblib"], top_n=5
        )
        fit_artifact = artifacts["fit_predictor.joblib"]
        matches = matches.copy()
        matches["fit_score"] = matches["document"].map(
            lambda text: predict_fit(resume_text, text, fit_artifact)
        )

        skill_gaps: list[str] = cluster_skill_gaps(
            resume_text, artifacts["clustering.joblib"]
        )
        found_skills: set[str] = extract_skills(resume_text)

    # ── Summary metrics row ─────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    top_match_score = float(matches["match_score"].iloc[0]) if not matches.empty else 0.0
    top_fit_score = float(matches["fit_score"].iloc[0]) if not matches.empty else 0.0
    with m1:
        st.markdown(
            '<div class="sh-card" style="text-align:center">'
            '<div style="font-size:1.8rem">🏷️</div>'
            '<div class="sh-section-label" style="margin-top:6px">Career Category</div>'
            f'<div style="font-size:1rem;font-weight:700;color:#a78bfa">{predicted_category}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="sh-card" style="text-align:center">'
            '<div style="font-size:1.8rem">🔗</div>'
            '<div class="sh-section-label" style="margin-top:6px">Top Match Score</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:#34d399">{top_match_score:.0%}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            '<div class="sh-card" style="text-align:center">'
            '<div style="font-size:1.8rem">⚡</div>'
            '<div class="sh-section-label" style="margin-top:6px">Top Fit Score</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:#fbbf24">{top_fit_score:.0%}</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            '<div class="sh-card" style="text-align:center">'
            '<div style="font-size:1.8rem">🔍</div>'
            '<div class="sh-section-label" style="margin-top:6px">Skills Detected</div>'
            f'<div style="font-size:1.4rem;font-weight:800;color:#6c63ff">{len(found_skills)}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_career, tab_jobs, tab_gaps = st.tabs(
        ["🏷️ Career Match", "💼 Job Board", "🔬 Skill Gap Report"]
    )

    # ── Tab 1: Career Match ─────────────────────────────────────────────────
    with tab_career:
        col_a, col_b = st.columns([1, 1], gap="large")
        with col_a:
            st.markdown('<div class="sh-section-label">Predicted category</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="sh-cat-badge">🎯 &nbsp; {predicted_category}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sh-section-label">Job board confidence</div>', unsafe_allow_html=True)
            _conf_bar("Semantic match", top_match_score)
            _conf_bar("Fit probability", top_fit_score)

        with col_b:
            st.markdown('<div class="sh-section-label">Detected skills</div>', unsafe_allow_html=True)
            if found_skills:
                pills_html = " ".join(_pill(s, "sh-pill-green") for s in sorted(found_skills))
                st.markdown(
                    f'<div class="sh-card" style="line-height:2.4">{pills_html}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.info("No skills from the current vocabulary detected in your resume.")

    # ── Tab 2: Job Board ────────────────────────────────────────────────────
    with tab_jobs:
        st.markdown('<div class="sh-section-label">Top 5 matching positions</div>', unsafe_allow_html=True)

        display = matches[["title", "company", "location", "match_score", "fit_score"]].copy()
        display["match_score"] = display["match_score"].map(lambda v: f"{v:.1%}")
        display["fit_score"] = display["fit_score"].map(lambda v: f"{v:.1%}")
        display.columns = ["Title", "Company", "Location", "Match Score", "Fit Score"]
        display = display.reset_index(drop=True)
        display.index = display.index + 1  # 1-based ranking

        st.dataframe(
            display,
            use_container_width=True,
            column_config={
                "Title": st.column_config.TextColumn("💼 Title"),
                "Company": st.column_config.TextColumn("🏢 Company"),
                "Location": st.column_config.TextColumn("📍 Location"),
                "Match Score": st.column_config.TextColumn("🔗 Match"),
                "Fit Score": st.column_config.TextColumn("⚡ Fit"),
            },
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sh-section-label">Match & fit breakdown</div>', unsafe_allow_html=True)
        for i, (_, row) in enumerate(matches.iterrows(), 1):
            title = str(row.get("title", "Unknown")).title()
            company = str(row.get("company", "")).title()
            ms = float(row["match_score"])
            fs = float(row["fit_score"])
            with st.container():
                c1, c2 = st.columns([2, 3])
                with c1:
                    st.markdown(
                        f"**{i}. {title}** &nbsp; <span style='color:#64748b;font-size:0.85rem'>{company}</span>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    _conf_bar(f"Match {ms:.0%} · Fit {fs:.0%}", (ms + fs) / 2)

    # ── Tab 3: Skill Gap Report ─────────────────────────────────────────────
    with tab_gaps:
        col_g1, col_g2 = st.columns([1, 1], gap="large")

        with col_g1:
            st.markdown('<div class="sh-section-label">Missing skills (nearest job cluster)</div>', unsafe_allow_html=True)
            if skill_gaps:
                gaps_html = " ".join(_pill(g, "sh-pill-yellow") for g in skill_gaps[:16])
                st.markdown(
                    f'<div class="sh-card" style="line-height:2.6">{gaps_html}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Adding these {len(skill_gaps)} skills to your profile would improve cluster alignment."
                )
            else:
                st.markdown(
                    '<div class="sh-card"><span class="sh-no-gap">✅ No skill gaps detected — '
                    "your profile aligns well with the nearest job cluster.</span></div>",
                    unsafe_allow_html=True,
                )

        with col_g2:
            st.markdown('<div class="sh-section-label">Your skill profile</div>', unsafe_allow_html=True)
            if found_skills:
                skill_pills = " ".join(_pill(s, "sh-pill-green") for s in sorted(found_skills))
                st.markdown(
                    f'<div class="sh-card" style="line-height:2.6">{skill_pills}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="sh-card" style="color:#64748b">'
                    "No skills detected. Try including specific tools and technologies in your resume."
                    "</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sh-section-label">Learning priority</div>', unsafe_allow_html=True)
        if skill_gaps:
            priority = skill_gaps[:6]
            cols = st.columns(len(priority))
            for col, skill in zip(cols, priority):
                with col:
                    st.markdown(
                        f'<div class="sh-card" style="text-align:center;padding:1rem 0.5rem">'
                        f'<div style="font-size:1.5rem">📚</div>'
                        f'<div style="font-size:0.85rem;font-weight:600;color:#a78bfa;margin-top:4px">{skill}</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#334155;font-size:0.75rem;padding-bottom:1rem">'
    "SmartHire uses classical ML — no data leaves your machine. "
    "Synthetic training data is generated separately via <code>scripts/generate_resumes.py</code>."
    "</div>",
    unsafe_allow_html=True,
)
