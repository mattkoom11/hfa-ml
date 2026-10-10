"""Phase 4: a learned re-ranker over the recommenders' combined candidates.

For each query, candidates = union of every system's top CANDIDATE_DEPTH.
Each (query, candidate) pair gets interpretable features; a logistic regression
predicts "graded 2" (strongly related), trained on Matthew's judgments.
Every feature also appears multiplied by "query is an accessory", so the model
can learn different weights for clothing and accessories (the Phase 3 finding).

Evaluation is nested cross-validation by query: 5 outer folds of 10 queries;
the regularization strength C is chosen inside each outer training set only.
So every query's ranking comes from a model that never saw that query.

Outputs:
  data/predictions/rerank_cv.json     {query: [top candidates, out-of-fold ranking]}
  data/predictions/rerank_model.json  coefficients of a model fit on all queries (for interpretation)
Run:  .venv/bin/python -m hfa.rerank
Then: .venv/bin/python -m hfa.rec_eval
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from hfa.recsys import SYSTEMS, Catalog, _overlap, load_catalog, rank

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_DEPTH = 10
OUT_DEPTH = 10
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
BASE = ["img_sim", "same_type", "same_era", "decade_close", "color_overlap", "material_overlap",
        "technique_overlap", "same_group"]
FEATURES = BASE + ["is_accessory"] + [f"{f}_x_acc" for f in BASE]


def features(c: Catalog, q: str, x: str) -> list[float]:
    dq, dx = c.decade[q], c.decade[x]
    decade = 0.0 if dq is None or dx is None else 1.0 if dq == dx else 0.5 if abs(dq - dx) <= 10 else 0.0
    base = [
        float(c.emb[q] @ c.emb[x]),
        float(bool(c.ca_type[q]) and c.ca_type[q] == c.ca_type[x]),
        float(bool(c.era[q]) and c.era[q] == c.era[x]),
        decade,
        _overlap(c.colors[q], c.colors[x]),
        _overlap(c.materials[q], c.materials[x]),
        _overlap(c.techniques[q], c.techniques[x]),
        float(c.group[q] == c.group[x]),
    ]
    acc = float(c.group[q] == "accessory")
    return base + [acc] + [v * acc for v in base]


def candidates(c: Catalog, q: str) -> list[str]:
    return sorted(set().union(*[rank(c, q, s, CANDIDATE_DEPTH) for s in SYSTEMS]))


def training_rows(c: Catalog, judgments: dict, queries: list[str]):
    X, y, groups = [], [], []
    for q in queries:
        for x, rec in judgments[q].items():
            X.append(features(c, q, x))
            y.append(int(rec["grade"] == 2))
            groups.append(q)
    return np.array(X), np.array(y), np.array(groups)


def fit(X: np.ndarray, y: np.ndarray, C: float):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, max_iter=5000).fit(scaler.transform(X), y)
    return scaler, model


def rerank(c: Catalog, q: str, scaler, model, k: int) -> list[str]:
    cands = candidates(c, q)
    p = model.predict_proba(scaler.transform(np.array([features(c, q, x) for x in cands])))[:, 1]
    order = sorted(range(len(cands)), key=lambda i: (-p[i], cands[i]))
    return [cands[i] for i in order[:k]]


def ndcg_at5(c: Catalog, judgments: dict, q: str, scaler, model) -> float:
    from hfa.rec_eval import per_query

    return per_query(judgments[q], rerank(c, q, scaler, model, 5), 5)["ndcg"]


def choose_C(c: Catalog, judgments: dict, queries: list[str]) -> float:
    """Inner CV over the outer training queries: pick C by mean nDCG@5."""
    qs = np.array(queries)
    best, best_score = C_GRID[0], -1.0
    for C in C_GRID:
        vals = []
        for tr, va in GroupKFold(n_splits=4).split(qs, groups=qs):
            X, y, _ = training_rows(c, judgments, list(qs[tr]))
            scaler, model = fit(X, y, C)
            vals += [ndcg_at5(c, judgments, q, scaler, model) for q in qs[va]]
        if np.mean(vals) > best_score:
            best, best_score = C, float(np.mean(vals))
    return best


def main() -> None:
    judgments = json.loads((ROOT / "data/judgments/judgments.json").read_text())
    pool = json.loads((ROOT / "data/judgments/pool.json").read_text())
    c = load_catalog()
    queries = np.array([q for q in pool if q in judgments])

    out, chosen = {}, []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(queries, groups=queries)):
        C = choose_C(c, judgments, list(queries[tr]))
        X, y, _ = training_rows(c, judgments, list(queries[tr]))
        scaler, model = fit(X, y, C)
        for q in queries[te]:
            out[q] = rerank(c, q, scaler, model, OUT_DEPTH)
        chosen.append(C)
        print(f"fold {fold}: C={C}, trained on {len(tr)} queries / {len(y)} pairs", flush=True)

    (ROOT / "data/predictions/rerank_cv.json").write_text(json.dumps(out, indent=1))

    # One model on every query, only to read what was learned (not for scoring).
    C_all = float(np.median(chosen))
    X, y, _ = training_rows(c, judgments, list(queries))
    scaler, model = fit(X, y, C_all)
    raw = model.coef_[0] / scaler.scale_  # log-odds change per unit of each raw feature
    coef = dict(zip(FEATURES, raw.round(3).tolist()))
    (ROOT / "data/predictions/rerank_model.json").write_text(json.dumps(
        {"C": C_all, "coef_raw_units": coef, "chosen_C_per_fold": chosen}, indent=1))
    print(f"\nall-query model (C={C_all}); log-odds per unit of each feature:")
    for f in BASE:
        print(f"  {f:18} clothing {coef[f]:+7.3f}   accessory {coef[f] + coef[f + '_x_acc']:+7.3f}")


if __name__ == "__main__":
    main()
