"""Export a self-contained search index for the live search service (hfa.serve).

Everything the service needs, nothing it doesn't: no donor or email columns,
no photos, no access to the share. Copy the folder to the server as-is.

Output: data/exports/search_index/
  garments.csv  accession, title, ca_type, date_raw, date_earliest, date_latest, description,
                material, color, technique (Phase 1 attributes, ';'-separated)
  emb.npy       FashionCLIP photo embeddings, float16, rows aligned with garments.csv
  meta.json     what built it and when
Run:  .venv/bin/python -m hfa.search_index
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hfa.recsys import load_catalog

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/exports/search_index"
COLUMNS = ["title", "ca_type", "date_raw", "date_earliest", "date_latest", "description"]
FORBIDDEN = {"donor_raw", "donor_name", "donation_date_text", "email", "donor"}


def main() -> None:
    c = load_catalog()
    g = pd.read_parquet(ROOT / "data/processed/collection_clean.parquet").drop_duplicates("accession").set_index("accession")
    table = g.loc[c.accessions, COLUMNS].copy()
    for f, attr in (("material", c.materials), ("color", c.colors), ("technique", c.techniques)):
        table[f] = [";".join(sorted(attr[a])) for a in c.accessions]
    assert not FORBIDDEN & set(table.columns)

    OUT.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(OUT / "garments.csv", index=False)
    np.save(OUT / "emb.npy", np.stack([c.emb[a] for a in c.accessions]).astype(np.float16))
    (OUT / "meta.json").write_text(json.dumps({
        "built": date.today().isoformat(), "garments": len(table), "embedding": "Marqo/marqo-fashionCLIP (mean of photos)",
        "attributes": "keyword rubric for described garments, photo probe otherwise", "system": "fusion (bm25 + clip, RRF)",
    }, indent=1))
    size = sum(p.stat().st_size for p in OUT.iterdir()) / 1e6
    print(f"search index: {len(table)} garments, {size:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
