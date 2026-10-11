"""Build the small committed fixture GitHub Actions runs the search checks on.

Contents (tests/fixtures/search/, committed):
  garments.csv     judged garments + random distractors: accession, title, type, dates,
                   description, Phase 1 attributes. No donor or email columns, ever.
  emb.npy          their FashionCLIP embeddings (float16), rows aligned with garments.csv
  q_emb.npy / desc_emb.npy   CLIP text embeddings of each query and its parsed description
  queries.json     {qid: {text, kind, parse, qrels: {accession: grade}}}
  answers.json     cached grounded answers + the garments they were given
  baseline.json    nDCG@5 per system on this fixture, the floor CI checks against

This is UVA collection data, committed only because the repo is private. Confirm
with the archive that this subset may live on GitHub, and before making the repo
public remove the fixture or replace it with a synthetic one.
Run:  .venv/bin/python -m hfa.make_fixture [--distractors 100]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hfa.recsys import load_catalog
from hfa.search import clip_text

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests/fixtures/search"
COLUMNS = ["title", "ca_type", "date_raw", "date_earliest", "date_latest", "description"]
FORBIDDEN = {"donor_raw", "donor_name", "donation_date_text", "email"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractors", type=int, default=100)
    args = ap.parse_args()

    queries = yaml.safe_load((ROOT / "eval/search_queries.yaml").read_text())
    judg = json.loads((ROOT / "data/judgments/search_judgments.json").read_text())
    parses = json.loads((ROOT / "data/search/parses.json").read_text())
    answers = json.loads((ROOT / "data/search/answers.json").read_text())
    c = load_catalog()
    g = pd.read_parquet(ROOT / "data/processed/collection_clean.parquet").drop_duplicates("accession").set_index("accession")
    assert not FORBIDDEN & set(g.columns), "clean table must not carry donor columns"

    judged = sorted({x for q in queries for x in judg.get(q["id"], {})})
    rest = sorted(set(c.accessions) - set(judged))
    rng = np.random.default_rng(20261015)
    accs = judged + sorted(rng.choice(rest, size=min(args.distractors, len(rest)), replace=False).tolist())

    table = g.loc[accs, COLUMNS].copy()
    for f, attr in (("material", c.materials), ("color", c.colors), ("technique", c.techniques)):
        table[f] = [";".join(sorted(attr[a])) for a in accs]
    FIX.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(FIX / "garments.csv", index=False)
    np.save(FIX / "emb.npy", np.stack([c.emb[a] for a in accs]).astype(np.float16))

    q_emb = clip_text([q["text"] for q in queries])
    desc_emb = clip_text([parses[q["id"]]["parse"]["description"] for q in queries])
    np.save(FIX / "q_emb.npy", q_emb.astype(np.float16))
    np.save(FIX / "desc_emb.npy", desc_emb.astype(np.float16))
    qjson = {q["id"]: {"text": q["text"], "kind": q["kind"], "parse": parses[q["id"]]["parse"],
                       "qrels": {x: r["grade"] for x, r in judg.get(q["id"], {}).items()}} for q in queries}
    (FIX / "queries.json").write_text(json.dumps(qjson, indent=1))
    (FIX / "answers.json").write_text(json.dumps(
        {k: {kk: v[kk] for kk in ("query", "retrieved", "answer")} for k, v in answers.items()}, indent=1, ensure_ascii=False))

    # Baseline: what the current code scores on exactly this fixture, loaded the way CI loads it.
    from hfa.ci_check import fixture_scores
    base = fixture_scores(FIX)
    (FIX / "baseline.json").write_text(json.dumps(base, indent=1))
    print(f"fixture: {len(accs)} garments ({len(judged)} judged + {len(accs) - len(judged)} distractors), "
          f"{len(queries)} queries; baseline nDCG@5 {base}")


if __name__ == "__main__":
    main()
