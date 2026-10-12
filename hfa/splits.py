"""Join the clean table to photo coverage and freeze train/val/test splits.

Splits are made once, by garment, and every later phase reuses them, so no
garment ever moves between training and testing. Stratified by group
(clothing/accessory) and CA type; strata too small to split go to train.

Output: data/processed/garments.parquet (clean table + photo coverage + split)
Run:    .venv/bin/python -m hfa.splits
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261010
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype=object)
    strata = df["group"].fillna("unknown") + "/" + df["ca_type"].fillna("unknown")
    for _, idx in df.groupby(strata, sort=True).groups.items():
        idx = np.array(sorted(idx))
        if len(idx) < 7:  # not enough to put at least one garment in val and test
            continue
        rng.shuffle(idx)
        n_val = max(1, round(len(idx) * FRACTIONS["val"]))
        n_test = max(1, round(len(idx) * FRACTIONS["test"]))
        split[idx[:n_test]] = "test"
        split[idx[n_test:n_test + n_val]] = "val"
    return split


def extend_splits(df: pd.DataFrame, previous: dict[str, str]) -> pd.Series:
    """Keep every garment's frozen split; assign only garments that are new since the last run.

    New garments are split among themselves with a seed derived from how many were
    already assigned, so re-running on the same data gives the same answer.
    """
    split = df["accession"].map(previous)
    new = split.isna()
    if new.any():
        fresh = df.loc[new].reset_index(drop=True)
        split.loc[new] = assign_splits(fresh, seed=SEED + len(previous)).values
    return split


def main() -> None:
    clean = pd.read_parquet(ROOT / "data/processed/collection_clean.parquet")
    cov = pd.read_parquet(ROOT / "data/processed/photo_coverage.parquet")
    # One row per usable accession; duplicate accession rows stay out until a curator resolves them.
    g = clean[clean["accession"].notna() & ~clean["accession"].duplicated(keep=False)].copy()
    g = g.merge(cov, on="accession", how="left")
    conv_path = ROOT / "data/interim/converted_jpgs.parquet"
    if conv_path.exists():
        conv = pd.read_parquet(conv_path)
        conv = conv[conv["status"].isin(["converted", "skipped"])].groupby("accession").size().rename("n_converted")
        g = g.merge(conv, left_on="accession", right_index=True, how="left")
    else:
        g["n_converted"] = 0
    for c in ("n_files", "n_shots", "n_web", "n_raw", "n_folders", "n_converted"):
        g[c] = g[c].fillna(0).astype(int)
    # JPGs the ML phases can read: ones on the share plus ones converted from RAW.
    g["n_ml_jpg"] = g["n_web"] + g["n_converted"]
    g["has_photos"] = g["n_shots"] > 0
    out = ROOT / "data/processed/garments.parquet"
    g = g.reset_index(drop=True)
    if out.exists():  # splits are frozen: never reshuffle garments that already have one
        prev = pd.read_parquet(out, columns=["accession", "split"])
        g["split"] = extend_splits(g, dict(zip(prev["accession"], prev["split"]))).values
        print(f"kept {g['accession'].isin(prev['accession']).sum()} frozen splits, "
              f"assigned {(~g['accession'].isin(prev['accession'])).sum()} new garments")
    else:
        g["split"] = assign_splits(g).values
    g.to_parquet(out, index=False)

    orphans = sorted(set(cov["accession"]) - set(clean["accession"].dropna()))
    print(f"garments: {len(g)}  photographed: {g['has_photos'].sum()}  "
          f"photographed+described: {(g['has_photos'] & g['has_description']).sum()}  "
          f"usable now (JPG + description): {((g['n_ml_jpg'] > 0) & g['has_description']).sum()}  "
          f"with any JPG: {(g['n_ml_jpg'] > 0).sum()}")
    print("split sizes:", g["split"].value_counts().to_dict())
    print(pd.crosstab(g["group"].fillna("unknown"), g["split"]).to_string())
    print(f"photo folders with no spreadsheet row: {len(orphans)}")
    (ROOT / "data/reports").mkdir(parents=True, exist_ok=True)
    pd.Series(orphans, name="accession").to_csv(ROOT / "data/reports/photo_orphans.csv", index=False)

    from hfa import tracking  # here, not at the top: CI imports this module without mlflow installed

    with tracking.start("phase0-data", "build-garments"):
        source = ROOT / "data/raw/Historic_Collection_Updated.xlsx"
        tracking.mlflow.log_params({
            "source_sha256": tracking.file_sha256(source)[:16],
            "split_seed": SEED,
            "split_fractions": "/".join(f"{k}={v}" for k, v in FRACTIONS.items()),
        })
        tracking.mlflow.log_metrics({
            "garments": len(g),
            "photographed": int(g["has_photos"].sum()),
            "described": int(g["has_description"].sum()),
            "photographed_and_described": int((g["has_photos"] & g["has_description"]).sum()),
            "jpg_and_described": int(((g["n_ml_jpg"] > 0) & g["has_description"]).sum()),
            "with_jpg": int((g["n_ml_jpg"] > 0).sum()),
            "raw_only_unconverted": int(((g["n_raw"] > 0) & (g["n_ml_jpg"] == 0)).sum()),
            "photo_orphans": len(orphans),
            **{f"split_{k}": int(v) for k, v in g["split"].value_counts().items()},
        })
        # Accession numbers and issue codes only; no donor columns reach MLflow.
        tracking.mlflow.log_artifact(str(ROOT / "data/reports/cleaning_issues.csv"))
        tracking.mlflow.log_artifact(str(ROOT / "data/reports/photo_orphans.csv"))


if __name__ == "__main__":
    main()
