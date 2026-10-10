"""Draw the Phase 1 labeling set: described training-split garments, stratified by CA type.

Proportional allocation with a floor, so rare types (vests, sweaters, military)
still appear. Drawn once with a fixed seed; rerunning refuses to overwrite an
existing set so labels never end up attached to a different sample.

Output: data/labels/label_set.csv
Run:    .venv/bin/python -m hfa.label_sample [--n 200] [--floor 4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/labels/label_set.csv"
SEED = 20261011


def allocate(counts: pd.Series, n: int, floor: int) -> pd.Series:
    """How many to draw from each type: min(floor, available) first, the rest proportional."""
    base = counts.clip(upper=floor)
    remaining = n - int(base.sum())
    if remaining < 0:
        raise ValueError(f"floor {floor} across {len(counts)} types already exceeds n={n}")
    room = counts - base
    share = room / room.sum() * remaining
    extra = np.floor(share).astype(int)
    # hand leftover slots to the largest fractional remainders
    for t in (share - extra).sort_values(ascending=False).index[: remaining - int(extra.sum())]:
        extra[t] += 1
    return (base + extra).clip(upper=counts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--floor", type=int, default=4)
    args = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; delete it deliberately if you really want a new sample")

    g = pd.read_parquet(ROOT / "data/processed/garments.parquet")
    pool = g[(g["split"] == "train") & g["has_description"]]
    take = allocate(pool["ca_type"].value_counts(), args.n, args.floor)

    rng = np.random.default_rng(SEED)
    picks = []
    for t, k in take.items():
        rows = pool[pool["ca_type"] == t]
        picks.append(rows.iloc[rng.permutation(len(rows))[:k]])
    sample = pd.concat(picks)
    # Shuffle the order too, so labeling fatigue doesn't line up with one category.
    sample = sample.iloc[rng.permutation(len(sample))]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["accession", "ca_type", "group", "title", "date_raw", "description"]
    sample[cols].to_csv(OUT, index=False)
    print(f"labeling set: {len(sample)} garments from a pool of {len(pool)}")
    print(take[take > 0].to_dict())


if __name__ == "__main__":
    main()
