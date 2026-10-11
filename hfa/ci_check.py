"""The check GitHub Actions runs on every push: search quality and answer grounding.

Runs the real ranking code (hfa.search.rank_queries) on the committed fixture,
with no model download and no LLM, and fails if:
  - any system's nDCG@5 falls more than TOLERANCE below baseline.json
  - any cached answer cites a garment it wasn't given
  - the fixture ever contains donor or email columns
Run:  python -m hfa.ci_check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests/fixtures/search"
TOLERANCE = 0.02
FORBIDDEN = {"donor_raw", "donor_name", "donation_date_text", "email", "donor"}


def load_fixture(fix: Path):
    table = pd.read_csv(fix / "garments.csv", dtype={"accession": str}).set_index("accession")
    table = table.astype(object).where(table.notna(), None)
    attrs = {a: {f: set(filter(None, (table.at[a, f] or "").split(";"))) for f in ("material", "color", "technique")}
             for a in table.index}
    queries = json.loads((fix / "queries.json").read_text())
    load = lambda n: np.load(fix / n).astype(np.float32)
    return table, attrs, queries, load("emb.npy"), load("q_emb.npy"), load("desc_emb.npy")


def fixture_scores(fix: Path = FIX, k: int = 5) -> dict[str, float]:
    from hfa.rec_eval import per_query
    from hfa.search import rank_queries

    table, attrs, queries, emb, q_emb, desc_emb = load_fixture(fix)
    qlist = [{"id": qid, "text": q["text"]} for qid, q in queries.items()]
    runs = rank_queries(qlist, {qid: q["parse"] for qid, q in queries.items()}, table, attrs, emb, q_emb, desc_emb)
    judged = [qid for qid, q in queries.items() if q["qrels"]]
    out = {}
    for system, run in runs.items():
        vals = [per_query({x: {"grade": gr} for x, gr in queries[qid]["qrels"].items()}, run[qid], k)["ndcg"]
                for qid in judged]
        out[system] = round(float(np.mean(vals)), 4)
    return out


def main() -> int:
    from hfa.rag import CITE

    failures = []
    cols = set(pd.read_csv(FIX / "garments.csv", nrows=0).columns)
    if FORBIDDEN & cols:
        failures.append(f"fixture contains personal-data columns: {sorted(FORBIDDEN & cols)}")

    base = json.loads((FIX / "baseline.json").read_text())
    now = fixture_scores()
    for system, b in base.items():
        flag = "FAIL" if now[system] < b - TOLERANCE else "ok"
        print(f"{flag:4} {system:7} nDCG@5 {now[system]:.4f} (baseline {b:.4f})")
        if flag == "FAIL":
            failures.append(f"{system} nDCG@5 dropped to {now[system]:.4f} from {b:.4f}")

    answers = json.loads((FIX / "answers.json").read_text())
    bad = {qid: sorted(set(CITE.findall(a["answer"])) - set(a["retrieved"])) for qid, a in answers.items()}
    bad = {k: v for k, v in bad.items() if v}
    print(f"{'FAIL' if bad else 'ok':4} answers citing garments they weren't given: {len(bad)}/{len(answers)}")
    if bad:
        failures.append(f"invalid citations: {bad}")

    for f in failures:
        print("::error::" + f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
