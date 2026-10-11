"""Phase 6: export the learned re-ranker's related garments for the frontend.

Fits the Phase 4 re-ranker once on every judged query (blind-final judgments,
C = the median of what nested CV chose), then ranks candidates for every
photographed garment. The frontend reads the result instead of computing a
weighted score, and falls back to its old logic for garments not in the file.

Output: data/exports/related.json
        {"generated", "model", "k", "related": {accession: [up to k accessions, best first]}}
Run:    .venv/bin/python -m hfa.export_related [--k 8] [--frontend ../UVA_Collectiveaccess_Frontend/data/related.json]
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

from hfa.recsys import load_catalog
from hfa.rerank import fit, rerank, training_rows

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/exports/related.json"
_state: dict = {}


def _init(C: float) -> None:
    """Each worker loads the catalog and fits the same model once."""
    c = load_catalog()
    judgments = json.loads((ROOT / "data/judgments/judgments_final.json").read_text())
    X, y, _ = training_rows(c, judgments, list(judgments))
    _state.update(c=c, model=fit(X, y, C))


def _rank(args: tuple[str, int]) -> tuple[str, list[str]]:
    q, k = args
    scaler, model = _state["model"]
    return q, rerank(_state["c"], q, scaler, model, k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--frontend", type=Path, help="also copy the file here (e.g. the frontend's data/related.json)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    meta = json.loads((ROOT / "data/predictions/rerank_model.json").read_text())
    C = float(meta["C"])
    accs = load_catalog().accessions
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(C,)) as pool:
        related = dict(pool.map(_rank, [(a, args.k) for a in accs], chunksize=16))

    judged = json.loads((ROOT / "data/judgments/judgments_final.json").read_text())
    out = {
        "generated": date.today().isoformat(),
        "model": {"name": "hfa-ml rerank v1", "type": "logistic regression over pair features",
                  "C": C, "trained_on_queries": len(judged),
                  "trained_on_pairs": sum(len(v) for v in judged.values()),
                  "eval": "nDCG@5 0.929 under nested CV (vs 0.699 for the previous weighted score)"},
        "k": args.k,
        "related": related,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    sizes = np.array([len(v) for v in related.values()])
    print(f"exported related garments for {len(related)} garments (k={args.k}, min list {sizes.min()}) to {OUT}")
    if args.frontend:
        shutil.copyfile(OUT, args.frontend)
        print(f"copied to {args.frontend}")


if __name__ == "__main__":
    main()
