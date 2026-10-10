"""Score an attribute-extraction method against Matthew's labels.

Per field (material / color / technique), over garments labeled "done":
  micro P / R / F1  pooled over every (garment, code) pair
  macro F1          mean over codes that occur in the gold labels of this set
  exact match       share of garments whose predicted set equals the gold set
95% intervals come from a garment-level bootstrap. Only list codes are scored;
the free-text "other" boxes are not.

Dev = the first 106 labels, frozen 2026-10-11 before any rule tuning used them.
Test = everything labeled after that. Test is hidden unless --test is passed:
look at it once, when a method is final.

Run:  .venv/bin/python -m hfa.attr_eval keyword [--test] [--by-code] [--subset METHOD] [--log]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hfa.rubric import FIELDS, load_vocab

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 1000
SEED = 20261011


def load_gold(path: Path | None = None) -> tuple[dict, set[str], set[str]]:
    labels = json.loads((path or ROOT / "data/labels/labels.json").read_text())
    gold = {a: r for a, r in labels.items() if r.get("status") == "done"}
    dev = set(pd.read_csv(ROOT / "data/labels/eval_roles.csv")["accession"])
    return gold, set(gold) & dev, set(gold) - dev


def counts(gold: dict, pred: dict, accs: list[str], field: str, codes: list[str]) -> np.ndarray:
    """Array [garment, code, (tp, fp, fn)] so bootstrap resamples are cheap."""
    out = np.zeros((len(accs), len(codes), 3), dtype=np.int32)
    idx = {c: j for j, c in enumerate(codes)}
    for i, a in enumerate(accs):
        g, p = set(gold[a][field]), set(pred.get(a, {}).get(field, []))
        for c in g & p:
            out[i, idx[c], 0] = 1
        for c in p - g:
            out[i, idx[c], 1] = 1
        for c in g - p:
            out[i, idx[c], 2] = 1
    return out


def scores(arr: np.ndarray) -> dict[str, float]:
    tp, fp, fn = arr.sum(axis=(0, 1))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    per_code = arr.sum(axis=0)
    support = per_code[:, 0] + per_code[:, 2]  # codes present in gold
    denom = 2 * per_code[:, 0] + per_code[:, 1] + per_code[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        code_f1 = np.where(denom > 0, 2 * per_code[:, 0] / denom, 0.0)
    macro = float(code_f1[support > 0].mean()) if (support > 0).any() else 0.0
    exact = float((arr[:, :, 1:].sum(axis=(1, 2)) == 0).mean())
    return {"precision": p, "recall": r, "micro_f1": f1, "macro_f1": macro, "exact": exact}


def bootstrap(arr: np.ndarray, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    draws = [scores(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(N_BOOT)]
    return {k: (float(np.percentile([d[k] for d in draws], 2.5)), float(np.percentile([d[k] for d in draws], 97.5)))
            for k in ("micro_f1", "macro_f1")}


def evaluate(method: str, accs: set[str], gold: dict, pred: dict, vocab: dict, by_code: bool) -> dict:
    rng = np.random.default_rng(SEED)
    accs = sorted(accs)
    result = {}
    for f in FIELDS:
        codes = [x["code"] for x in vocab[f]]
        arr = counts(gold, pred, accs, f, codes)
        s, ci = scores(arr), bootstrap(arr, rng)
        result[f] = {**s, "micro_ci": ci["micro_f1"], "macro_ci": ci["macro_f1"]}
        print(f"  {f:9}  micro F1 {s['micro_f1']:.3f} [{ci['micro_f1'][0]:.3f}, {ci['micro_f1'][1]:.3f}]  "
              f"macro F1 {s['macro_f1']:.3f} [{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}]  "
              f"P {s['precision']:.3f}  R {s['recall']:.3f}  exact {s['exact']:.3f}")
        if by_code:
            pc = arr.sum(axis=0)
            rows = [(codes[j], *pc[j]) for j in range(len(codes)) if pc[j].sum()]
            for code, tp, fp, fn in sorted(rows, key=lambda r: -(r[1] + r[3])):
                print(f"        {code:16} tp {tp:3}  fp {fp:3}  fn {fn:3}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", help="name of data/predictions/<method>.json")
    ap.add_argument("--test", action="store_true", help="also score the held-out test labels")
    ap.add_argument("--by-code", action="store_true")
    ap.add_argument("--subset", help="only score garments present in data/predictions/<SUBSET>.json")
    ap.add_argument("--labels", type=Path, help="score against another label file (e.g. labels_pre_adjudication.json)")
    ap.add_argument("--log", action="store_true", help="log the scores to MLflow (phase1-attributes)")
    args = ap.parse_args()

    pred = json.loads((ROOT / f"data/predictions/{args.method}.json").read_text())
    gold, dev, test = load_gold(args.labels)
    if args.subset:
        # e.g. score keyword only on garments that have photos, to compare like with like
        covered = set(json.loads((ROOT / f"data/predictions/{args.subset}.json").read_text()))
        dev, test = dev & covered, test & covered
        print(f"restricted to garments covered by {args.subset}")
    vocab = load_vocab()
    sets = {"dev": dev} | ({"test": test} if args.test else {})
    results = {}
    for name, accs in sets.items():
        print(f"{args.method} on {name} ({len(accs)} garments)")
        results[name] = evaluate(args.method, accs, gold, pred, vocab, args.by_code)
    if not args.test:
        print(f"(test set: {len(test)} labeled garments, hidden; pass --test when the method is final)")

    if args.log:
        from hfa import tracking
        with tracking.start("phase1-attributes", args.method + (f"@{args.subset}" if args.subset else "")):
            tracking.mlflow.log_params({"method": args.method, "n_dev": len(dev), "n_test": len(test) if args.test else 0})
            for name, res in results.items():
                for f, s in res.items():
                    for k in ("micro_f1", "macro_f1", "precision", "recall", "exact"):
                        tracking.mlflow.log_metric(f"{name}_{f}_{k}", s[k])


if __name__ == "__main__":
    main()
