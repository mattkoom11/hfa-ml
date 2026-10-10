"""Score related-garment systems against Matthew's 0/1/2 relevance judgments.

Per query, each system's top K (default 5; the site shows 4):
  nDCG@K     graded gain (2^grade - 1), ideal ranking built from every judged
             candidate for that query
  rel@K      share of the top K graded >= 1
  strong@K   share of the top K graded 2
Intervals: bootstrap over queries. Differences vs the current recommender
("fixed") use a paired bootstrap: the same resampled queries for both systems.
Unjudged results count as 0 and are reported, so a system can't gain from gaps.

Run:  .venv/bin/python -m hfa.rec_eval [--k 5] [--log]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hfa.recsys import SYSTEMS, load_catalog, rank

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 2000
SEED = 20261012


def dcg(gains: list[float]) -> float:
    return float(sum(g / np.log2(i + 2) for i, g in enumerate(gains)))


def per_query(judg: dict, ranking: list[str], k: int) -> dict[str, float]:
    grades = [judg.get(x, {}).get("grade") for x in ranking[:k]]
    unjudged = sum(g is None for g in grades)
    grades = [0 if g is None else g for g in grades]
    ideal = sorted((r["grade"] for r in judg.values()), reverse=True)[:k]
    idcg = dcg([2**g - 1 for g in ideal])
    return {
        "ndcg": dcg([2**g - 1 for g in grades]) / idcg if idcg else 0.0,
        "rel": sum(g >= 1 for g in grades) / k,
        "strong": sum(g == 2 for g in grades) / k,
        "unjudged": unjudged,
    }


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = values[rng.integers(0, len(values), (N_BOOT, len(values)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--judgments", default="judgments.json",
                    help="file under data/judgments/ (judgments_final.json uses the blind re-grades)")
    args = ap.parse_args()

    judgments = json.loads((ROOT / "data/judgments" / args.judgments).read_text())
    pool = json.loads((ROOT / "data/judgments/pool.json").read_text())
    g = pd.read_parquet(ROOT / "data/processed/garments.parquet").set_index("accession")
    c = load_catalog()
    queries = [q for q in pool if q in judgments]
    # The learned re-ranker's out-of-fold rankings (hfa.rerank), if they exist.
    cv_path = ROOT / "data/predictions/rerank_cv.json"
    rerank_cv = json.loads(cv_path.read_text()) if cv_path.exists() else {}
    systems = list(SYSTEMS) + (["rerank"] if rerank_cv else [])

    rows = []
    for q in queries:
        for s in systems:
            ranking = rerank_cv[q] if s == "rerank" else rank(c, q, s, args.k)
            rows.append({"query": q, "group": g.at[q, "group"], "system": s,
                         **per_query(judgments[q], ranking, args.k)})
    df = pd.DataFrame(rows)
    grades = pd.Series([r["grade"] for q in queries for r in judgments[q].values()])
    print(f"{len(queries)} queries, {len(grades)} judgments; grade mix {grades.value_counts().sort_index().to_dict()}; "
          f"unjudged results in top {args.k}: {df.groupby('system')['unjudged'].sum().to_dict()}\n")

    rng = np.random.default_rng(SEED)
    results = {}
    for scope in ("all", "clothing", "accessory"):
        sub = df if scope == "all" else df[df["group"] == scope]
        n = sub["query"].nunique()
        print(f"{scope} ({n} queries)")
        base = sub[sub["system"] == "fixed"].set_index("query")["ndcg"]
        for s in systems:
            m = sub[sub["system"] == s].set_index("query")
            lo, hi = boot_ci(m["ndcg"].to_numpy(), rng)
            line = (f"  {s:12} nDCG@{args.k} {m['ndcg'].mean():.3f} [{lo:.3f}, {hi:.3f}]  "
                    f"rel@{args.k} {m['rel'].mean():.2f}  strong@{args.k} {m['strong'].mean():.2f}")
            if s != "fixed":
                diff = (m["ndcg"] - base.loc[m.index]).to_numpy()
                dlo, dhi = boot_ci(diff, rng)
                line += f"   vs fixed {diff.mean():+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
            print(line)
            results[f"{scope}_{s}"] = {"ndcg": m["ndcg"].mean(), "rel": m["rel"].mean(), "strong": m["strong"].mean()}
        print()

    if args.log:
        from hfa import tracking
        with tracking.start("phase3-recsys", f"judged@{args.k}:{args.judgments}"):
            tracking.mlflow.log_params({"k": args.k, "queries": len(queries), "judgments": len(grades)})
            for key, vals in results.items():
                for metric, v in vals.items():
                    tracking.mlflow.log_metric(f"{key}_{metric}", float(v))


if __name__ == "__main__":
    main()
