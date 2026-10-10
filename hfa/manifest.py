"""Build a manifest of every photo inside an accession folder.

Photo layout: <root>/Historic Clothing Collection/<type>/[year]/<accession #>/...
Converted JPGs usually sit in a `converted/` subfolder next to the Sony RAW files,
so one shot appears twice; `stem` lets later steps keep one file per shot.
Reads only; nothing on the photo share is changed.

Outputs:
  data/interim/photo_manifest.parquet   one row per image file
  data/processed/photo_coverage.parquet one row per accession: shot counts, JPG availability
Run:  .venv/bin/python -m hfa.manifest [--root /mnt/nas-garment-capture]
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd

from hfa.accession import normalize_accession

ROOT = Path(__file__).resolve().parent.parent
TOP_FOLDERS = ("Historic Clothing Collection", "HCC Accessories Data")
SKIP_DIRS = {"#recycle", "Matt Experiment", "photos to be sorted", "Label Research", "Master Excel"}
RAW_EXT = {".arw", ".cr2", ".cr3", ".nef", ".dng"}
WEB_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff"}
_ACC_DIR = re.compile(r"^[A-Za-z]{2,5}[.,\s]\d{4}[.,\s]\d+")
# Multi-piece objects get one folder per part ("ACS.2023.967a", "DR.2024.53.ab");
# stray duplicates end in " copy". Both belong to the parent accession.
_PART = re.compile(r"^(.*?\d{4}[.,\s]\d+)(?:\.?([a-z]{1,3}))?(?:\s+copy)?$", re.IGNORECASE)


def parse_folder(name: str) -> tuple[str | None, str | None]:
    """Folder name -> (normalized parent accession, part label or None)."""
    m = _PART.match(name.strip())
    base, part = (m.group(1), m.group(2)) if m else (name, None)
    return normalize_accession(base).value, (part.lower() if part else None)


def scan(root: Path) -> pd.DataFrame:
    rows = []
    for top in TOP_FOLDERS:
        for dirpath, dirnames, _ in os.walk(root / top):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            name = os.path.basename(dirpath)
            if not _ACC_DIR.match(name):
                continue
            dirnames[:] = []  # everything below an accession folder belongs to it
            accession, part = parse_folder(name)
            for sub, _, files in os.walk(dirpath):
                for f in files:
                    stem, ext = os.path.splitext(f)
                    ext = ext.lower()
                    if ext not in RAW_EXT | WEB_EXT:
                        continue
                    full = Path(sub) / f
                    rows.append({
                        "accession": accession,
                        "part": part,
                        "folder_name": name,
                        "folder_rel": str(Path(dirpath).relative_to(root)),
                        "path_rel": str(full.relative_to(root)),
                        "stem": stem.lower(),
                        "ext": ext,
                        "kind": "raw" if ext in RAW_EXT else "web",
                        "in_converted": "converted" in Path(sub).name.lower(),
                        "bytes": full.stat().st_size,
                    })
    return pd.DataFrame(rows)


def coverage(manifest: pd.DataFrame) -> pd.DataFrame:
    g = manifest.groupby("accession", dropna=True)
    return pd.DataFrame({
        "n_files": g.size(),
        "n_shots": g["stem"].nunique(),
        "n_web": g.apply(lambda x: (x["kind"] == "web").sum(), include_groups=False),
        "n_raw": g.apply(lambda x: (x["kind"] == "raw").sum(), include_groups=False),
        "n_folders": g["folder_rel"].nunique(),
    }).reset_index().assign(raw_only=lambda d: d["n_web"].eq(0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nas-garment-capture")
    root = Path(ap.parse_args().root)
    if not (root / TOP_FOLDERS[0]).is_dir():
        raise SystemExit(f"{root} doesn't look mounted (no '{TOP_FOLDERS[0]}' folder)")
    m = scan(root)
    cov = coverage(m)
    (ROOT / "data/interim").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    m.to_parquet(ROOT / "data/interim/photo_manifest.parquet", index=False)
    cov.to_parquet(ROOT / "data/processed/photo_coverage.parquet", index=False)
    print(f"image files: {len(m)}  accessions: {len(cov)}  shots: {cov['n_shots'].sum()}  "
          f"raw-only: {cov['raw_only'].sum()}  split across >1 folder: {(cov['n_folders'] > 1).sum()}")
    print(f"folders with unusable accession names: {m.loc[m['accession'].isna(), 'folder_name'].nunique()}")


if __name__ == "__main__":
    main()
