#!/usr/bin/env python3
"""Generate labelled, synthetic resumes through a locally running Ollama model."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (  # noqa: E402
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
    RAW_DATA_DIR,
    ensure_project_directories,
)

CATEGORIES = [
    "Data Analyst", "Data Scientist", "Machine Learning Engineer", "Backend Developer",
    "Frontend Developer", "HR Specialist", "Business Analyst", "DevOps Engineer",
]


def generate_one(category: str, number: int) -> str:
    prompt = f"""Create one entirely fictional entry-level to mid-level resume for the category: {category}.
It is synthetic training data, not a real person's biography. Vary skills, projects, education and experience.
Do not use real contact details, company names, postal addresses, or identify a real person.
Return exactly this format and no Markdown fence:
CATEGORY: {category}
RESUME:
<180-260 words, plain text resume>
---
This is item {number}."""
    response = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"temperature": OLLAMA_TEMPERATURE, "num_predict": 350},
        },
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    generated = response.json()["response"].strip()
    # Qwen may emit an internal reasoning section unless thinking is disabled.
    generated = re.sub(r"<think>.*?</think>", "", generated, flags=re.S | re.I).strip()
    # Small local models may ignore the requested delimiters. Normalize every
    # response so the training loader always receives a labelled record.
    marker = re.search(r"resume:\s*", generated, flags=re.I)
    if marker:
        generated = generated[marker.end():]
    generated = generated.replace("---", " ").strip()
    return f"CATEGORY: {category}\nRESUME:\n{generated}\n---"


def generate_bootstrap_set(per_category: int) -> list[str]:
    """Generate one compact Qwen seed per category, then make labelled variants."""
    records = {}
    for category in CATEGORIES:
        prompt = f"""Write one compact, entirely fictional resume profile for a {category}.
Use no real people, employers, addresses, email addresses or phone numbers. Write 40-60 words with relevant
skills, one project or accomplishment, and education or experience. Return only the resume text: no reasoning,
heading, label, Markdown, or delimiter."""
        print(f"Generating Qwen seed: {category}", flush=True)
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {"temperature": OLLAMA_TEMPERATURE, "num_predict": 110},
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        text = re.sub(r"<think>.*?</think>", "", response.json()["response"], flags=re.S | re.I)
        records[category.casefold()] = re.sub(r"\s+", " ", text.replace("---", " ")).strip()
    variants = [
        "Focuses on clear documentation, collaboration, and measurable project outcomes.",
        "Highlights practical problem-solving, version control, and stakeholder communication.",
        "Emphasizes testing, iterative delivery, and a portfolio-ready project contribution.",
    ]
    generated: list[str] = []
    for category in CATEGORIES:
        seed = records[category.casefold()]
        for number in range(per_category):
            text = seed if number == 0 else f"{seed} {variants[(number - 1) % len(variants)]}"
            generated.append(f"CATEGORY: {category}\nRESUME:\n{text}\n---")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=20, help="Synthetic resumes per category (default: 20).")
    parser.add_argument("--output", type=Path, default=RAW_DATA_DIR / "resumes.txt")
    parser.add_argument("--append", action="store_true", help="Append rather than overwrite the output file.")
    parser.add_argument("--fast-bootstrap", action="store_true", help="Generate compact Qwen seeds then labelled variants (best for slower local hardware).")
    args = parser.parse_args()
    if args.per_category < 2:
        parser.error("--per-category must be at least 2 for classifier validation.")
    ensure_project_directories()
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10).raise_for_status()
    except requests.RequestException as error:
        raise SystemExit(f"Cannot reach Ollama at {OLLAMA_HOST}. Start it with `ollama serve`. Details: {error}")

    try:
        if args.fast_bootstrap:
            print("Generating compact Qwen resume seeds with thinking disabled", flush=True)
            generated = generate_bootstrap_set(args.per_category)
        else:
            generated = []
            for category in CATEGORIES:
                for number in range(1, args.per_category + 1):
                    print(f"Generating {category} ({number}/{args.per_category})", flush=True)
                    generated.append(generate_one(category, number))
    except (requests.RequestException, ValueError) as error:
        raise SystemExit(f"Ollama generation failed: {error}")
    mode = "a" if args.append else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        if args.append and args.output.stat().st_size:
            handle.write("\n")
        handle.write("\n\n".join(generated).rstrip() + "\n")
    manifest = {"model": OLLAMA_MODEL, "per_category": args.per_category, "categories": CATEGORIES, "output": str(args.output), "fast_bootstrap": args.fast_bootstrap}
    (args.output.parent / "resumes_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(generated)} synthetic resumes to {args.output}")


if __name__ == "__main__":
    main()
