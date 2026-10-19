"""Export per-decade colour, material and technique totals for the frontend's woven timeline.

Only aggregates leave hfa-ml: no per-garment records, titles or descriptions. Each
dated garment adds a total weight of 1, split across the decades its date range
covers by year overlap (1890-1920 gives the 1890s, 1900s and 1910s 10/31 each and
the 1920s 1/31). Within a decade, a garment's weight is split evenly over its own
colour, material and technique codes.

Input:  data/exports/search_index/garments.csv   (or the CI fixture with --source fixture)
        data/processed/garments.parquet           (dates of scanned accessions only)
Output: data/exports/collection-weave.json
Run:    .venv/bin/python -m hfa.export_weave [--source fixture] [--frontend ../UVA_Collectiveaccess_Frontend]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from hfa.ci_check import FORBIDDEN

ROOT = Path(__file__).resolve().parent.parent
INDEX_CSV = ROOT / "data/exports/search_index/garments.csv"
FIXTURE_CSV = ROOT / "tests/fixtures/search/garments.csv"
CATALOG = ROOT / "data/processed/garments.parquet"
OUT = ROOT / "data/exports/collection-weave.json"

# Same threshold hfa.dates uses for "century_whole": too vague for decade labels.
CENTURY_SPAN = 99
LISTS = ("color", "material", "technique")
PLURAL = {"color": "colors", "material": "materials", "technique": "techniques"}


def _assert_no_personal_data(columns) -> None:
    # A raise, not an assert: `python -O` strips asserts, and this guard keeps
    # donor data out of a public repo.
    leaked = FORBIDDEN & set(columns)
    if leaked:
        raise ValueError(f"refusing to export: personal-data columns {sorted(leaked)}")


def load_vocab() -> dict[str, set[str]]:
    lists = json.loads((ROOT / "hfa/vocab.json").read_text())["lists"]
    return {name: {entry["code"] for entry in entries} for name, entries in lists.items()}


def _year(value) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else int(number)


def classify(earliest, latest) -> str:
    """'dated', 'century_only' (span of 99+ years) or 'undated' (missing, unparseable or reversed)."""
    e, l = _year(earliest), _year(latest)
    if e is None or l is None or l < e:
        return "undated"
    return "century_only" if l - e >= CENTURY_SPAN else "dated"


def decade_weights(earliest: int, latest: int) -> dict[int, float]:
    """Share of one garment per decade, by inclusive year overlap."""
    span = latest - earliest + 1
    weights = {}
    for decade in range(earliest // 10 * 10, latest // 10 * 10 + 1, 10):
        overlap = min(latest, decade + 9) - max(earliest, decade) + 1
        if overlap > 0:
            weights[decade] = overlap / span
    return weights


def split_codes(cell, vocab: set[str], unknown: dict[str, int]) -> list[str]:
    """Semicolon-separated vocabulary codes; unknown codes are dropped and counted."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    kept = []
    for code in (c.strip() for c in str(cell).split(";")):
        if not code:
            continue
        if code in vocab:
            kept.append(code)
        else:
            unknown[code] = unknown.get(code, 0) + 1
    return kept


def build_weave(
    rows: pd.DataFrame,
    vocab: dict[str, set[str]],
    scanned: dict,
    source: str,
    today: str,
) -> tuple[dict, dict[str, int]]:
    _assert_no_personal_data(rows.columns)
    counts = {"dated": 0, "century_only": 0, "undated": 0}
    garments: dict[int, float] = defaultdict(float)
    attrs = {name: defaultdict(lambda: defaultdict(float)) for name in LISTS}
    unknown: dict[str, int] = {}

    for row in rows.to_dict("records"):
        kind = classify(row.get("date_earliest"), row.get("date_latest"))
        counts[kind] += 1
        if kind != "dated":
            continue
        codes = {name: split_codes(row.get(name), vocab[name], unknown) for name in LISTS}
        for decade, weight in decade_weights(_year(row["date_earliest"]), _year(row["date_latest"])).items():
            garments[decade] += weight
            for name, cs in codes.items():
                for code in cs:
                    attrs[name][decade][code] += weight / len(cs)

    decades = []
    if garments:
        for decade in range(min(garments), max(garments) + 10, 10):
            entry = {"decade": decade, "garments": round(garments.get(decade, 0.0), 3)}
            for name in LISTS:
                entry[PLURAL[name]] = {
                    code: round(w, 3) for code, w in sorted(attrs[name][decade].items()) if round(w, 3) > 0
                }
            decades.append(entry)

    weave = {
        "generated": today,
        "source": source,
        "rows": len(rows),
        **counts,
        "decades": decades,
        "scanned": scanned,
    }
    return weave, unknown


def scanned_dates(catalog: pd.DataFrame, accessions: list[str]) -> dict:
    """Date ranges for scanned accessions that are catalogued with a dated (not century-only) range."""
    _assert_no_personal_data(catalog.columns)
    found = {}
    for accession in accessions:
        if accession not in catalog.index:
            continue
        row = catalog.loc[accession]
        if classify(row["date_earliest"], row["date_latest"]) == "dated":
            found[accession] = {"earliest": _year(row["date_earliest"]), "latest": _year(row["date_latest"])}
    return found


def frontend_scanned_accessions(frontend: Path) -> list[str]:
    garments = json.loads((frontend / "data/garments.json").read_text())
    return sorted({g["accessionNumber"] for g in garments if g.get("model3d_url") and g.get("accessionNumber")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=["index", "fixture"], default="index")
    ap.add_argument(
        "--frontend",
        type=Path,
        help="frontend repo: read its scanned accessions and copy the result into its data/",
    )
    args = ap.parse_args()

    fixture = args.source == "fixture"
    table = pd.read_csv(FIXTURE_CSV if fixture else INDEX_CSV, dtype=str, keep_default_na=False)

    scanned: dict = {}
    if args.frontend:
        accessions = frontend_scanned_accessions(args.frontend)
        if CATALOG.exists():
            scanned = scanned_dates(pd.read_parquet(CATALOG).set_index("accession"), accessions)
        else:
            print(f"note: {CATALOG.relative_to(ROOT)} not found; scanned garments get no dates", file=sys.stderr)

    weave, unknown = build_weave(
        table, load_vocab(), scanned, "ci-fixture" if fixture else "search-index", date.today().isoformat()
    )
    if unknown:
        print(f"warning: dropped unknown codes {dict(sorted(unknown.items()))}", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(weave, indent=1) + "\n")
    print(
        f"{weave['dated']} dated, {weave['century_only']} century-only, {weave['undated']} undated garments "
        f"-> {len(weave['decades'])} decades, {len(scanned)} scanned ({OUT.relative_to(ROOT)})"
    )
    if args.frontend:
        dest = args.frontend / "data/collection-weave.json"
        shutil.copyfile(OUT, dest)
        print(f"copied to {dest}")


if __name__ == "__main__":
    main()
