"""Score the Phase 5 search systems against Matthew's blind 0/1/2 search judgments.

Same metrics as hfa.rec_eval (nDCG@K with 2^grade - 1 gains, rel@K, strong@K,
query-level bootstrap intervals, paired differences), reported overall and by
query kind (attribute / era / detail / visual / vague). Baseline: bm25.

Run:  .venv/bin/python -m hfa.search_eval [--k 5] [--log]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hfa.rec_eval import boot_ci, per_query

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ("bm25", "clip", "hybrid")
SEED = 20261014


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    queries = yaml.safe_load((ROOT / "eval/search_queries.yaml").read_text())
    judg = json.loads((ROOT / "data/judgments/search_judgments.json").read_text())
    runs = {s: json.loads((ROOT / f"data/predictions/search_{s}.json").read_text()) for s in SYSTEMS}
    qs = [q for q in queries if q["id"] in judg]
    rows = [{"query": q["id"], "kind": q["kind"], "system": s, **per_query(judg[q["id"]], runs[s][q["id"]], args.k)}
            for q in qs for s in SYSTEMS]
    df = pd.DataFrame(rows)
    grades = pd.Series([r["grade"] for q in qs for r in judg[q["id"]].values()])
    print(f"{len(qs)}/{len(queries)} queries judged, {len(grades)} judgments; grade mix "
          f"{grades.value_counts().sort_index().to_dict()}; unjudged in top {args.k}: "
          f"{df.groupby('system')['unjudged'].sum().to_dict()}\n")

    rng = np.random.default_rng(SEED)
    results = {}
    for scope in ["all"] + sorted(df["kind"].unique()):
        sub = df if scope == "all" else df[df["kind"] == scope]
        base = sub[sub["system"] == "bm25"].set_index("query")["ndcg"]
        print(f"{scope} ({sub['query'].nunique()} queries)")
        for s in SYSTEMS:
            m = sub[sub["system"] == s].set_index("query")
            lo, hi = boot_ci(m["ndcg"].to_numpy(), rng)
            line = (f"  {s:7} nDCG@{args.k} {m['ndcg'].mean():.3f} [{lo:.3f}, {hi:.3f}]  "
                    f"rel@{args.k} {m['rel'].mean():.2f}  strong@{args.k} {m['strong'].mean():.2f}")
            if s != "bm25":
                d = (m["ndcg"] - base.loc[m.index]).to_numpy()
                dlo, dhi = boot_ci(d, rng)
                line += f"   vs bm25 {d.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
            print(line)
            results[f"{scope}_{s}"] = float(m["ndcg"].mean())
        print()

    if args.log:
        from hfa import tracking
        with tracking.start("phase5-search", f"judged@{args.k}"):
            tracking.mlflow.log_params({"k": args.k, "queries": len(qs), "judgments": len(grades)})
            for key, v in results.items():
                tracking.mlflow.log_metric(f"{key}_ndcg", v)


if __name__ == "__main__":
    main()
