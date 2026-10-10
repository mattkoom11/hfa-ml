"""Make a local, reduced-size JPG for every garment photo the ML steps use.

Default: convert RAW-only garments' Sony .ARW files. --web: copy the JPGs that
already exist on the share (index: data/interim/cached_jpgs.parquet).

Reads from the photo share (never writes to it) and writes
data/interim/jpg/<accession>/<stem>.jpg, long edge 1600 px, sRGB.
Resumable: files that already exist are skipped.
Run:  .venv/bin/python -m hfa.convert [--web] [--root /mnt/nas-garment-capture] [--workers 6]
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/interim/jpg"
LONG_EDGE = 1600


def convert_one(src: Path, dst: Path) -> str:
    if dst.exists() and dst.stat().st_size > 0:
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part.jpg")
    cmd = ["magick", str(src), "-auto-orient", "-colorspace", "sRGB",
           "-resize", f"{LONG_EDGE}x{LONG_EDGE}>", "-quality", "90", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        return "failed: " + r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "failed"
    tmp.rename(dst)  # only complete files get the final name, so reruns can trust them
    return "converted"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nas-garment-capture")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N files (timing test)")
    ap.add_argument("--web", action="store_true", help="copy existing JPGs locally instead of converting RAW")
    ap.add_argument("--max-shots", type=int, default=8, help="with --web: at most this many shots per garment")
    args = ap.parse_args()
    root = Path(args.root)

    manifest = pd.read_parquet(ROOT / "data/interim/photo_manifest.parquet")
    if args.web:
        # Garments that already have JPGs on the share: copy one reduced JPG per shot,
        # preferring the converted/ copy, so the model steps never need the share mounted.
        todo = manifest[manifest["accession"].notna() & manifest["kind"].eq("web")]
        todo = todo.sort_values("in_converted", ascending=False)
    else:
        cov = pd.read_parquet(ROOT / "data/processed/photo_coverage.parquet")
        raw_only = set(cov.loc[cov["raw_only"], "accession"])
        todo = manifest[manifest["accession"].isin(raw_only) & manifest["kind"].eq("raw")]
    todo = todo.drop_duplicates(["accession", "stem"])
    if args.web:
        todo = todo.sort_values(["accession", "stem"]).groupby("accession").head(args.max_shots)
    todo = todo.reset_index(drop=True)
    if args.limit:
        todo = todo.head(args.limit)

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_one, root / row.path_rel, OUT / row.accession / f"{row.stem}.jpg"): i
            for i, row in todo.iterrows()
        }
        for n, fut in enumerate(as_completed(futures), 1):
            results[futures[fut]] = fut.result()
            if n % 100 == 0 or n == len(futures):
                print(f"{n}/{len(futures)}", flush=True)

    todo["status"] = pd.Series(results)
    todo["jpg_rel"] = [str((OUT / a / f"{s}.jpg").relative_to(ROOT)) for a, s in zip(todo["accession"], todo["stem"])]
    if not args.limit:
        name = "cached_jpgs.parquet" if args.web else "converted_jpgs.parquet"
        todo.to_parquet(ROOT / "data/interim" / name, index=False)
    print(todo["status"].str.split(":").str[0].value_counts().to_string())
    failed = todo[todo["status"].str.startswith("failed")]
    if len(failed):
        print(failed[["path_rel", "status"]].head(10).to_string())


if __name__ == "__main__":
    main()
