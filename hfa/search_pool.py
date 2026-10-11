"""Pool the search systems' top results per benchmark query for blind judging.

Output: data/judgments/search_pool.json  {query_id: {"text", "candidates", "found_by"}}
Judge:  .venv/bin/python -m hfa.judge_server --task search
Run:    .venv/bin/python -m hfa.search_pool [--depth 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/judgments/search_pool.json"
SYSTEMS = ("bm25", "clip", "fusion", "hybrid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=5)
    args = ap.parse_args()
    queries = yaml.safe_load((ROOT / "eval/search_queries.yaml").read_text())
    runs = {s: json.loads((ROOT / f"data/predictions/search_{s}.json").read_text()) for s in SYSTEMS}
    pool = json.loads(OUT.read_text()) if OUT.exists() else {}
    rng = np.random.default_rng(20261014)
    added = 0
    for q in queries:
        e = pool.setdefault(q["id"], {"text": q["text"], "candidates": [], "found_by": {}})
        if e["text"] != q["text"]:
            raise SystemExit(f"{q['id']} text changed since pooling; judgments would no longer match")
        for s, run in runs.items():
            for x in run.get(q["id"], [])[: args.depth]:
                e["found_by"].setdefault(x, [])
                if s not in e["found_by"][x]:
                    e["found_by"][x].append(s)
        new = [x for x in e["found_by"] if x not in e["candidates"]]
        e["candidates"] += [new[i] for i in rng.permutation(len(new))]
        added += len(new)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(pool, indent=1))
    print(f"{len(pool)} queries; {added} new pairs; {sum(len(e['candidates']) for e in pool.values())} in total")


if __name__ == "__main__":
    main()
