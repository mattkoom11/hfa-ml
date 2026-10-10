"""Draw the Phase 3 judged-query set and pool candidates from every system.

50 query garments from the photographed test split (25 clothing, 25 accessories,
stratified by CA type). For each, the top 5 from every system in hfa.recsys are
pooled and shuffled, so the judge can't tell which system found a garment.
Systems added later (the Phase 4 re-ranker) get their unjudged results appended
with --extend, without redrawing anything.

Output: data/judgments/pool.json  {query: {"candidates": [...], "found_by": {cand: [systems]}}}
Run:    .venv/bin/python -m hfa.judge_sample            (first draw)
        .venv/bin/python -m hfa.judge_sample --extend   (add new systems' results)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hfa.label_sample import allocate
from hfa.recsys import SYSTEMS, load_catalog, rank

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/judgments/pool.json"
SEED = 20261012
PER_GROUP = 25
DEPTH = 5  # the site shows 4 related garments; judge one past that


def draw_queries(c, g: pd.DataFrame) -> list[str]:
    rng = np.random.default_rng(SEED)
    pool = pd.DataFrame({"accession": [a for a in c.accessions if g.at[a, "split"] == "test"]})
    pool["group"] = pool["accession"].map(g["group"])
    pool["ca_type"] = pool["accession"].map(g["ca_type"])
    queries = []
    for grp in ("clothing", "accessory"):
        sub = pool[pool["group"] == grp]
        take = allocate(sub["ca_type"].value_counts(), PER_GROUP, floor=1)
        for t, k in take.items():
            rows = sub[sub["ca_type"] == t]["accession"].tolist()
            queries += [rows[i] for i in rng.permutation(len(rows))[:k]]
    return [queries[i] for i in rng.permutation(len(queries))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extend", action="store_true")
    ap.add_argument("--depth", type=int, default=DEPTH, help="pool each system's top N (5 by default)")
    args = ap.parse_args()
    c = load_catalog()
    g = pd.read_parquet(ROOT / "data/processed/garments.parquet").set_index("accession")
    rng = np.random.default_rng(SEED + 1)

    if OUT.exists() and not args.extend:
        raise SystemExit(f"{OUT} exists; use --extend to add systems, or delete it deliberately to redraw")
    pool = json.loads(OUT.read_text()) if args.extend else {q: {"candidates": [], "found_by": {}} for q in draw_queries(c, g)}

    cv_path = ROOT / "data/predictions/rerank_cv.json"
    rerank_cv = json.loads(cv_path.read_text()) if args.extend and cv_path.exists() else {}
    added = 0
    for q, entry in pool.items():
        ranked = {s: rank(c, q, s, args.depth) for s in SYSTEMS}
        if q in rerank_cv:
            ranked["rerank"] = rerank_cv[q][: args.depth]  # Phase 4 out-of-fold rankings
        for system, top in ranked.items():
            for x in top:
                entry["found_by"].setdefault(x, [])
                if system not in entry["found_by"][x]:
                    entry["found_by"][x].append(system)
        new = [x for x in entry["found_by"] if x not in entry["candidates"]]
        entry["candidates"] += [new[i] for i in rng.permutation(len(new))]
        added += len(new)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pool, indent=1))
    groups = pd.Series([g.at[q, "group"] for q in pool]).value_counts().to_dict()
    print(f"{len(pool)} queries {groups}; {added} candidates added; "
          f"{sum(len(e['candidates']) for e in pool.values())} to judge in total")


if __name__ == "__main__":
    main()
