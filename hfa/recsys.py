"""Related-garment recommenders compared in Phases 3-4.

  fixed        port of the frontend's lib/relatedGarments.ts: era .30, type .25,
               color .20, material .15, decade .10. After a CA import the catalog
               has no color or material values, so those weights score nothing;
               this is the system as it would actually run.
  fixed_attrs  the same weights, with Phase 1 attributes filled in
               (keywords for described garments, the photo probe otherwise)
  image        cosine similarity of FashionCLIP garment embeddings

Candidates are photographed garments, the ones the site can show with an image.
Ties are broken by accession so rankings are reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ("fixed", "fixed_attrs", "image")


def era(year: float | None) -> str | None:
    """Same buckets as getEraFromDecade in the frontend's types/garment.ts."""
    if year is None or np.isnan(year):
        return None
    if year < 1920:
        return "pre-1920"
    if year < 1950:
        return "1920-1950"
    if year < 1980:
        return "1950-1980"
    return "1980+"


@dataclass
class Catalog:
    accessions: list[str]
    ca_type: dict[str, str | None]
    era: dict[str, str | None]
    decade: dict[str, int | None]
    colors: dict[str, set[str]]
    materials: dict[str, set[str]]
    emb: dict[str, np.ndarray]
    attr_source: dict[str, str]


def load_catalog() -> Catalog:
    g = pd.read_parquet(ROOT / "data/processed/garments.parquet").set_index("accession")
    z = np.load(ROOT / "data/processed/garment_embeddings.npz")
    emb = dict(zip(z["accession"], z["emb"]))
    accs = sorted(a for a in emb if a in g.index)  # photographed garments with a spreadsheet row
    kw = json.loads((ROOT / "data/predictions/keyword.json").read_text())
    probe = json.loads((ROOT / "data/predictions/image_probe_top1.json").read_text())

    mid = ((g["date_earliest"].astype(float) + g["date_latest"].astype(float)) / 2).to_dict()
    colors, materials, source = {}, {}, {}
    for a in accs:
        src = kw if a in kw else probe
        source[a] = "keyword" if a in kw else "probe"
        colors[a], materials[a] = set(src[a]["color"]), set(src[a]["material"])
    return Catalog(
        accessions=accs,
        ca_type={a: g.at[a, "ca_type"] for a in accs},
        era={a: era(mid.get(a)) for a in accs},
        decade={a: None if np.isnan(mid.get(a, np.nan)) else int(mid[a] // 10 * 10) for a in accs},
        colors=colors, materials=materials, emb=emb, attr_source=source,
    )


def _overlap(x: set[str], y: set[str]) -> float:
    return len(x & y) / max(len(x), len(y)) if x and y else 0.0


def fixed_score(c: Catalog, q: str, x: str, use_attrs: bool) -> float:
    s = 0.0
    if c.era[q] and c.era[q] == c.era[x]:
        s += 0.30
    if c.ca_type[q] and c.ca_type[q] == c.ca_type[x]:
        s += 0.25
    if use_attrs:
        s += 0.20 * _overlap(c.colors[q], c.colors[x])
        s += 0.15 * _overlap(c.materials[q], c.materials[x])
    if c.decade[q] is not None and c.decade[x] is not None:
        diff = abs(c.decade[q] - c.decade[x])
        s += 0.10 if diff == 0 else 0.05 if diff <= 10 else 0.0
    return s  # maxScore is 1.0, so this is already the normalized score


def scores(c: Catalog, q: str, system: str) -> dict[str, float]:
    cands = [x for x in c.accessions if x != q]
    if system == "image":
        m = np.stack([c.emb[x] for x in cands])
        return dict(zip(cands, (m @ c.emb[q]).tolist()))
    return {x: fixed_score(c, q, x, use_attrs=system == "fixed_attrs") for x in cands}


def rank(c: Catalog, q: str, system: str, k: int = 10) -> list[str]:
    s = scores(c, q, system)
    return sorted(s, key=lambda x: (-s[x], x))[:k]


def tie_share(c: Catalog, q: str, system: str, k: int = 10) -> float:
    """Share of the top-k whose score equals the k-th score: how arbitrary the cut is."""
    s = scores(c, q, system)
    top = rank(c, q, system, k)
    kth = s[top[-1]]
    return sum(1 for x in top if s[x] == kth) / k
