"""Keyword baseline for Phase 1: the labeling rubric applied mechanically.

Predicts material / color / technique codes for every described garment by
matching list words, rubric aliases and technique cues (hfa.rubric).
It's the floor the LLM and embedding methods have to beat.

Output: data/predictions/keyword.json  {accession: {material: [...], color: [...], technique: [...]}}
Run:    .venv/bin/python -m hfa.attr_keyword
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hfa.rubric import FIELDS, load_vocab, mentions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/predictions/keyword.json"


def predict(text: str, vocab: dict) -> dict[str, list[str]]:
    t = text.lower()
    return {f: sorted(item["code"] for item in vocab[f] if mentions(f, item, t)) for f in FIELDS}


def main() -> None:
    vocab = load_vocab()
    g = pd.read_parquet(ROOT / "data/processed/garments.parquet")
    g = g[g["has_description"]]
    preds = {a: predict(d, vocab) for a, d in zip(g["accession"], g["description"])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(preds, indent=1))
    filled = {f: sum(bool(p[f]) for p in preds.values()) for f in FIELDS}
    print(f"predicted {len(preds)} garments; with at least one value: {filled}")


if __name__ == "__main__":
    main()
