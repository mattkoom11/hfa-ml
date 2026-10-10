"""Predict material / color / technique from garment photos (Phase 1, track B).

Two modes, both on FashionCLIP garment embeddings (hfa.embed):
  zeroshot  cosine similarity to text prompts ("a fashion item made of velvet"),
            softmax within each field, keep codes above a per-field threshold
  probe     one logistic regression per code, trained on keyword labels
            (hfa.attr_keyword) of described, photographed training-split garments.
            Every labeled garment (dev and test) is left out of training.

Thresholds are tuned per field on the dev labels only; test stays untouched.
Outputs data/predictions/image_<mode>.json, in the same shape as keyword.json,
for every garment that has photos.

Run:  .venv/bin/python -m hfa.attr_image zeroshot|probe
Then: .venv/bin/python -m hfa.attr_eval image_probe --subset image_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hfa.attr_eval import counts, load_gold, scores
from hfa.rubric import FIELDS, load_vocab

ROOT = Path(__file__).resolve().parent.parent
MIN_POSITIVES = 8  # codes rarer than this in the training pool are never predicted by the probe
TECHNIQUE_PROMPTS = {
    "printing": "a fashion item with a printed pattern",
    "embroidery": "a fashion item with embroidery",
    "beading": "a beaded fashion item",
    "pleating": "a pleated fashion item",
    "knitting": "a knitted fashion item",
    "quilting": "a quilted fashion item",
    "applique": "a fashion item with appliqué",
    "painting": "a hand-painted fashion item",
    "braiding": "a fashion item with braid trim",
    "crocheting": "a crocheted fashion item",
    "weaving": "a fashion item with a woven pattern",
    "hand_sewing": "a hand-sewn fashion item",
    "machine_sewing": "a machine-sewn fashion item",
    "needlepoint": "a needlepoint fashion item",
}


def prompt(field: str, item: dict) -> str:
    if field == "color":
        return f"a {item['label']} fashion item"
    if field == "material":
        return f"a fashion item made of {item['label']}"
    return TECHNIQUE_PROMPTS.get(item["code"], f"a fashion item with {item['label']}")


def zeroshot_probs(emb: np.ndarray, vocab: dict) -> dict[str, np.ndarray]:
    import torch

    from hfa.embed import load_model

    model, _, tokenizer, device = load_model()
    out = {}
    with torch.no_grad():
        for f in FIELDS:
            tok = tokenizer([prompt(f, it) for it in vocab[f]]).to(device)
            te = torch.nn.functional.normalize(model.encode_text(tok).float(), dim=-1).cpu().numpy()
            logits = 100.0 * emb @ te.T
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            out[f] = p / p.sum(axis=1, keepdims=True)
    return out


def probe_probs(emb: np.ndarray, accs: np.ndarray, vocab: dict, exclude: set[str],
                score_on: np.ndarray | None = None) -> tuple[dict, dict]:
    """Train on garment embeddings; score `score_on` (per-photo embeddings) if given."""
    from sklearn.linear_model import LogisticRegression

    g = pd.read_parquet(ROOT / "data/processed/garments.parquet").set_index("accession")
    kw = json.loads((ROOT / "data/predictions/keyword.json").read_text())
    pool = [i for i, a in enumerate(accs)
            if a in kw and a not in exclude and a in g.index and g.at[a, "split"] == "train"]
    out, trained = {}, {}
    for f in FIELDS:
        target = emb if score_on is None else score_on
        probs = np.zeros((len(target), len(vocab[f])))
        for j, item in enumerate(vocab[f]):
            y = np.array([item["code"] in kw[accs[i]][f] for i in pool])
            if y.sum() < MIN_POSITIVES or y.sum() == len(y):
                continue
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)
            clf.fit(emb[pool], y)
            probs[:, j] = clf.predict_proba(target)[:, 1]
            trained[f"{f}:{item['code']}"] = int(y.sum())
        out[f] = probs
    return out, {"train_pool": len(pool), "codes_trained": trained}


def tune_threshold(probs: np.ndarray, codes: list[str], accs: np.ndarray, gold: dict, dev: list[str], field: str,
                   top1: bool = False) -> float:
    """Pick the cutoff that maximizes dev micro-F1 for this field."""
    idx = {a: i for i, a in enumerate(accs)}
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.quantile(probs, np.linspace(0.5, 0.999, 120))):
        pred = {a: {field: [codes[j] for j in decide(probs[idx[a]], t, top1)]} for a in dev}
        f1 = scores(counts(gold, pred, dev, field, codes))["micro_f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def decide(row: np.ndarray, t: float, top1: bool) -> np.ndarray:
    keep = row >= t
    if top1 and row.max() > 0:
        keep[row.argmax()] = True  # nearly every garment has some color / material
    return np.where(keep)[0]


def max_pool(probs_img: dict[str, np.ndarray], img_acc: np.ndarray, accs: np.ndarray) -> dict[str, np.ndarray]:
    """Garment score = highest score over its photos, so a detail seen in one view counts."""
    rows = {a: np.where(img_acc == a)[0] for a in accs}
    return {f: np.stack([p[rows[a]].max(axis=0) for a in accs]) for f, p in probs_img.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["zeroshot", "probe"])
    ap.add_argument("--top1", default="color,material", help="fields that always keep their top code ('' for none)")
    ap.add_argument("--pool", choices=["mean", "max"], default="mean",
                    help="mean: score the averaged garment embedding; max: score each photo, keep the max")
    ap.add_argument("--tag", default="", help="suffix for the output name, for comparing variants")
    args = ap.parse_args()
    top1 = {f for f in args.top1.split(",") if f}

    z = np.load(ROOT / "data/processed/garment_embeddings.npz")
    accs, emb = z["accession"], z["emb"]
    vocab = load_vocab()
    gold, dev, test = load_gold()
    dev_photo = sorted(set(accs) & dev)

    meta = {"mode": args.mode, "n_garments": len(accs), "n_dev_with_photos": len(dev_photo)}
    meta.update({"top1": sorted(top1), "pool": args.pool})
    img = np.load(ROOT / "data/processed/image_embeddings.npz") if args.pool == "max" else None
    if args.mode == "zeroshot":
        probs = zeroshot_probs(img["emb"].astype(np.float32) if img else emb, vocab)
    else:
        probs, info = probe_probs(emb, accs, vocab, exclude=dev | test,
                                  score_on=img["emb"].astype(np.float32) if img else None)
        meta.update(info)
    if img is not None:
        probs = max_pool(probs, img["accession"], accs)

    preds = {a: {} for a in accs}
    meta["thresholds"] = {}
    for f in FIELDS:
        codes = [it["code"] for it in vocab[f]]
        t = tune_threshold(probs[f], codes, accs, gold, dev_photo, f, f in top1)
        meta["thresholds"][f] = t
        for i, a in enumerate(accs):
            preds[a][f] = [codes[j] for j in decide(probs[f][i], t, f in top1)]

    name = f"image_{args.mode}{args.tag}"
    out = ROOT / f"data/predictions/{name}.json"
    out.write_text(json.dumps(preds, indent=1))
    (ROOT / f"data/predictions/{name}_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out.name} for {len(preds)} garments; thresholds {meta['thresholds']}")
    if args.mode == "probe":
        print(f"probe training pool: {meta['train_pool']} garments; codes trained: {len(meta['codes_trained'])}")


if __name__ == "__main__":
    main()
