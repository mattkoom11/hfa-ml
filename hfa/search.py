"""Phase 5: plain-English search over the photographed catalog, three ways.

  bm25    keyword search over title, description, type, date text and Phase 1 attributes
  clip    FashionCLIP: the query's text embedding against every garment's photo embedding
  fusion  bm25 and clip merged by reciprocal rank fusion (no LLM; what the CPU-only VPS runs)
  hybrid  a local LLM (Ollama, qwen3:8b) parses the query into a decade range, a CA
          type and attribute lists; those become hard filters (decade, type) and soft
          boosts (material / color / technique), and CLIP ranks within the filters.
          Filters are relaxed (type first, then decade) if fewer than MIN_RESULTS remain.

LLM parses are cached in data/search/parses.json, so reruns are reproducible and
the benchmark doesn't depend on the model being up. Nothing leaves the machine.

Outputs: data/predictions/search_<system>.json  {query_id: [accessions, best first]}
Run:     .venv/bin/python -m hfa.search [--reparse]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hfa.recsys import load_catalog
from hfa.rubric import load_vocab, mentions

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "eval/search_queries.yaml"
PARSES = ROOT / "data/search/parses.json"
OLLAMA = "http://127.0.0.1:11434/api/generate"
LLM = "qwen3:8b"
TOP = 10
MIN_RESULTS = 5
BOOST = 0.03  # per matching material / color / technique, on the cosine-similarity scale
# A decade range is only trusted if the query actually names a period.
PERIOD_CUE = re.compile(r"\b(1[5-9]\d{2}s?|\d{2}s|\d{1,2}(st|nd|rd|th)[- ]century|century|victorian|edwardian|regency|"
                        r"georgian|belle [eé]poque|art deco|roaring twenties|post-?war|mid-?century|wwi+|world war)\b", re.I)
# A CA type is only trusted if the query names some kind of object.
OBJECT_CUE = re.compile(r"\b(dress(es)?|gowns?|frocks?|blouses?|shirts?|skirts?|bodices?|jackets?|coats?|overcoats?|capes?|cloaks?|"
                        r"suits?|ensembles?|outfits?|vests?|waistcoats?|trousers|pants|sweaters?|robes?|kimonos?|pajamas?|"
                        r"slips?|corsets?|petticoats?|undergarments?|uniforms?|hats?|bonnets?|caps?|shoes?|boots?|slippers?|"
                        r"bags?|purses?|clutch(es)?|reticules?|fans?|gloves?|ties?|shawls?|scarf|scarves|collars?|aprons?|"
                        r"jewelry|necklaces?|brooch(es)?|bracelets?|earrings?|pins?|clothing|garments?|accessor(y|ies))\b", re.I)
EXAMPLES = """Examples (not from the collection):
Query: blue taffeta ball gown from the 1950s -> {"decade_from": 1950, "decade_to": 1959, "ca_type": "dr", "materials": ["taffeta"], "colors": ["blue"], "techniques": [], "description": "blue taffeta ball gown"}
Query: men's top hat -> {"decade_from": null, "decade_to": null, "ca_type": "acs", "materials": [], "colors": [], "techniques": [], "description": "men's top hat"}
Query: something to wear to a wedding -> {"decade_from": null, "decade_to": null, "ca_type": null, "materials": [], "colors": [], "techniques": [], "description": "formal wedding attire"}
Query: wool winter cape -> {"decade_from": null, "decade_to": null, "ca_type": "otw", "materials": ["wool"], "colors": [], "techniques": [], "description": "warm wool cape"}
"""

TYPE_LABELS = {
    "dr": "dress", "sh": "shirt or blouse", "sk": "skirt", "bd": "bodice", "jkt": "jacket", "otw": "outerwear / coat / cape",
    "st": "suit", "end": "ensemble", "vt": "vest", "tr": "trousers", "sw": "sweater", "rb": "robe", "pj": "pajamas",
    "und": "undergarment", "nw": "non-western garment", "mt": "military", "oc": "occupational / uniform",
    "acs": "accessory (hat, shoes, purse, fan, gloves, tie, shawl, ...)", "jwl": "jewelry", "tex": "textile piece",
}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avg = sum(map(len, docs)) / len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in docs]

    def score(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            for t in query:
                if t in tf:
                    f = tf[t]
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
        return out


def parse_schema(vocab: dict) -> dict:
    codes = lambda f: [x["code"] for x in vocab[f]]
    return {
        "type": "object",
        "properties": {
            "decade_from": {"type": ["integer", "null"]},
            "decade_to": {"type": ["integer", "null"]},
            "ca_type": {"type": ["string", "null"], "enum": list(TYPE_LABELS) + [None]},
            "materials": {"type": "array", "items": {"type": "string", "enum": codes("material")}},
            "colors": {"type": "array", "items": {"type": "string", "enum": codes("color")}},
            "techniques": {"type": "array", "items": {"type": "string", "enum": codes("technique")}},
            "description": {"type": "string"},
        },
        "required": ["decade_from", "decade_to", "ca_type", "materials", "colors", "techniques", "description"],
    }


def llm_parse(text: str, vocab: dict) -> dict:
    import urllib.request

    types = "; ".join(f"{k} = {v}" for k, v in TYPE_LABELS.items())
    prompt = (
        "You turn a museum visitor's search into filters for a historic clothing collection.\n"
        f"Garment type codes: {types}.\n"
        "Rules: set decade_from/decade_to only if the query names a period (\"Victorian\" = 1837-1901, "
        "\"early 1900s\" = 1900-1915, \"1920s\" = 1920-1929). Set ca_type only if the query clearly names a kind "
        "of object; hats, shoes, bags, fans, gloves and ties are 'acs'. List only materials, colors and techniques "
        "the query actually names; leave lists empty otherwise. Use null when the query doesn't say. "
        "'description' restates what the visitor wants to see in at most eight plain words.\n"
        f"{EXAMPLES}Query: {text}"
    )
    body = json.dumps({"model": LLM, "prompt": prompt, "stream": False, "think": False,
                       "format": parse_schema(vocab), "options": {"temperature": 0, "num_predict": 200}}).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read())["response"])


def guard(raw: dict, text: str, vocab: dict) -> tuple[dict, list[str]]:
    """Drop parse fields the query doesn't support. Returns the cleaned parse and what was dropped."""
    p, dropped, t = dict(raw), [], text.lower()
    for field, key in (("material", "materials"), ("color", "colors"), ("technique", "techniques")):
        items = {x["code"]: x for x in vocab[field]}
        keep = [code for code in raw[key] if code in items and mentions(field, items[code], t)]
        dropped += [f"{key}:{code}" for code in raw[key] if code not in keep]
        p[key] = keep
    if raw["ca_type"] and not OBJECT_CUE.search(text):
        dropped.append(f"type:{raw['ca_type']}")
        p["ca_type"] = None
    if (raw["decade_from"] or raw["decade_to"]) and not PERIOD_CUE.search(text):
        dropped.append(f"decade:{raw['decade_from']}-{raw['decade_to']}")
        p["decade_from"] = p["decade_to"] = None
    words = (raw.get("description") or "").split()
    p["description"] = " ".join(words[:12]) if words else text
    return p, dropped


def clip_text(texts: list[str]) -> np.ndarray:
    import torch

    from hfa.embed import load_model

    model, _, tokenizer, device = load_model()
    with torch.no_grad():
        e = model.encode_text(tokenizer(texts).to(device)).float()
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


RRF_K = 60  # the usual constant; damps the advantage of rank 1 over rank 2


def rrf(orders: list) -> list[int]:
    """Reciprocal rank fusion: each list adds 1 / (RRF_K + rank); highest total first."""
    score: dict[int, float] = {}
    for order in orders:
        for rank, j in enumerate(order):
            score[int(j)] = score.get(int(j), 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(score, key=lambda j: (-score[j], j))


def rank_queries(queries: list[dict], parses: dict, table: pd.DataFrame, attrs: dict, emb: np.ndarray,
                 q_emb: np.ndarray, desc_emb: np.ndarray) -> dict[str, dict[str, list[str]]]:
    """Rank `table`'s garments (index = accession, rows aligned with `emb`) for every query, three ways.

    Pure function of its inputs (no model, no LLM), so CI can run it on the committed fixture.
    attrs: {accession: {"material": set, "color": set, "technique": set}}; parses: {query_id: guarded parse}.
    """
    accs = np.array(table.index)
    docs = []
    for a in accs:
        row = table.loc[a]
        words = " ".join(sorted(attrs[a]["material"] | attrs[a]["color"] | attrs[a]["technique"]))
        docs.append(tokens(" ".join(str(v or "") for v in (row["title"], row["description"],
                                                         TYPE_LABELS.get(row["ca_type"], ""), row["date_raw"], words))))
    bm25 = BM25(docs)
    lo = table["date_earliest"].astype(float).to_numpy()
    hi = table["date_latest"].astype(float).to_numpy()
    types = table["ca_type"].to_numpy()

    out = {"bm25": {}, "clip": {}, "fusion": {}, "hybrid": {}}
    for i, q in enumerate(queries):
        s = bm25.score(tokens(q["text"]))
        bm25_order = [j for j in np.argsort(-s, kind="stable") if s[j] > 0]
        out["bm25"][q["id"]] = [str(accs[j]) for j in bm25_order[:TOP]]
        sim = emb @ q_emb[i]
        clip_order = np.argsort(-sim, kind="stable")
        out["clip"][q["id"]] = [str(accs[j]) for j in clip_order[:TOP]]
        out["fusion"][q["id"]] = [str(accs[j]) for j in rrf([bm25_order, clip_order])[:TOP]]

        p = parses[q["id"]]
        score = emb @ desc_emb[i]
        for j, a in enumerate(accs):
            score[j] += BOOST * sum(len(set(p[k + "s"]) & attrs[a][k]) for k in ("material", "color", "technique"))
        decade_ok = np.ones(len(accs), bool)
        if p["decade_from"] or p["decade_to"]:
            a0, a1 = p["decade_from"] or 0, (p["decade_to"] or 3000) + 9
            decade_ok = ~np.isnan(lo) & (hi >= a0) & (lo <= a1)
        type_ok = types == p["ca_type"] if p["ca_type"] else np.ones(len(accs), bool)
        for mask in (decade_ok & type_ok, decade_ok, np.ones(len(accs), bool)):
            if mask.sum() >= MIN_RESULTS:
                break
        order = [j for j in np.argsort(-score, kind="stable") if mask[j]]
        out["hybrid"][q["id"]] = [str(accs[j]) for j in order[:TOP]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reparse", action="store_true", help="ask the LLM again instead of using cached parses")
    args = ap.parse_args()

    queries = yaml.safe_load(QUERIES.read_text())
    vocab = load_vocab()
    c = load_catalog()
    g = pd.read_parquet(ROOT / "data/processed/garments.parquet").set_index("accession").loc[c.accessions]
    accs = np.array(c.accessions)

    emb = np.stack([c.emb[a] for a in accs])
    q_emb = clip_text([q["text"] for q in queries])

    # hybrid parses (cached)
    PARSES.parent.mkdir(parents=True, exist_ok=True)
    parses = {} if args.reparse or not PARSES.exists() else json.loads(PARSES.read_text())
    for q in queries:
        if q["id"] not in parses or parses[q["id"]].get("_text") != q["text"]:
            raw = llm_parse(q["text"], vocab)
            clean, dropped = guard(raw, q["text"], vocab)
            parses[q["id"]] = {"_text": q["text"], "raw": raw, "parse": clean, "dropped": dropped}
            PARSES.write_text(json.dumps(parses, indent=1))  # save as we go
            print(f"  {q['id']} {q['text']!r}: {clean}  dropped={dropped}", flush=True)
    n_dropped = sum(bool(v["dropped"]) for k, v in parses.items() if k in {q["id"] for q in queries})
    print(f"guards changed {n_dropped}/{len(queries)} parses")
    desc_emb = clip_text([parses[q["id"]]["parse"]["description"] for q in queries])

    attrs = {a: {"material": c.materials[a], "color": c.colors[a], "technique": c.techniques[a]} for a in accs}
    out = rank_queries(queries, {k: v["parse"] for k, v in parses.items()}, g, attrs, emb, q_emb, desc_emb)

    for name, ranking in out.items():
        (ROOT / f"data/predictions/search_{name}.json").write_text(json.dumps(ranking, indent=1))
    print(f"ranked {len(queries)} queries with {', '.join(out)}")


if __name__ == "__main__":
    main()
