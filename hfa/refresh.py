"""Rebuild everything downstream of the spreadsheet and photos, check it, and optionally deploy.

Runs the pipeline steps in order and skips a step when its outputs are newer than
all of its inputs. Steps that read the photo share run whenever it's mounted (they
skip finished work themselves). Before deploying, the new search index must pass
the gate: tests, the CI quality check, and integrity checks against what's live.

  --dry-run      show which steps would run, run nothing
  --force        run every step
  --no-photos    skip the steps that need the photo share
  --deploy       after the gate passes, copy the index to the VPS, restart the
                 service, and roll back if /health doesn't report the new index

Run:  .venv/bin/python -m hfa.refresh [--dry-run] [--deploy]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SHARE = Path("/mnt/nas-garment-capture")
FRONTEND_RELATED = ROOT.parent / "UVA_Collectiveaccess_Frontend/data/related.json"
INDEX = ROOT / "data/exports/search_index"
VPS = "matthewkim@179.236.246.247"
REMOTE_INDEX = "~/hfa-search/data/exports/search_index"
HEALTH_URL = "https://search.uvahcc.org/health"
FORBIDDEN = {"donor", "donor_raw", "donor_name", "donation_date_text", "email"}
MIN_KEEP = 0.95  # the new index may not lose more than 5% of the garments that are live


@dataclass
class Step:
    name: str
    args: list[str]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    share: bool = False  # reads the photo share; runs whenever it's mounted

    def paths(self, which: list[str]) -> list[Path]:
        return [ROOT / p for p in which]


STEPS = [
    Step("clean", ["hfa.clean"], ["data/raw/Historic_Collection_Updated.xlsx"],
         ["data/processed/collection_clean.parquet", "data/interim/collection_full.parquet"]),
    Step("manifest", ["hfa.manifest", "--root", str(SHARE)], [],
         ["data/interim/photo_manifest.parquet", "data/processed/photo_coverage.parquet"], share=True),
    Step("convert-raw", ["hfa.convert", "--root", str(SHARE)], [], [], share=True),
    Step("convert-web", ["hfa.convert", "--web", "--root", str(SHARE)], [], [], share=True),
    Step("splits", ["hfa.splits"],
         ["data/processed/collection_clean.parquet", "data/processed/photo_coverage.parquet"],
         ["data/processed/garments.parquet"]),
    Step("embed", ["hfa.embed"], ["data/interim/jpg"], ["data/processed/garment_embeddings.npz"]),
    Step("attr-keyword", ["hfa.attr_keyword"], ["data/processed/garments.parquet"], ["data/predictions/keyword.json"]),
    Step("attr-probe", ["hfa.attr_image", "probe"],
         ["data/processed/garments.parquet", "data/processed/garment_embeddings.npz", "data/predictions/keyword.json"],
         ["data/predictions/image_probe.json"]),
    Step("related", ["hfa.export_related", "--frontend", str(FRONTEND_RELATED)],
         ["data/processed/garment_embeddings.npz", "data/predictions/keyword.json", "data/predictions/image_probe.json",
          "data/predictions/rerank_model.json", "data/judgments/judgments_final.json"],
         ["data/exports/related.json"]),
    Step("search-index", ["hfa.search_index"],
         ["data/processed/collection_clean.parquet", "data/processed/garment_embeddings.npz",
          "data/predictions/keyword.json", "data/predictions/image_probe.json"],
         ["data/exports/search_index/garments.csv", "data/exports/search_index/emb.npy"]),
]


def newest_mtime(p: Path) -> float:
    """A file's mtime, or the newest file inside a directory (0 if nothing exists)."""
    if p.is_file():
        return p.stat().st_mtime
    if p.is_dir():
        return max((f.stat().st_mtime for f in p.rglob("*") if f.is_file()), default=0.0)
    return 0.0


def is_stale(inputs: list[Path], outputs: list[Path]) -> bool:
    """Run a step if any output is missing or any input changed after the oldest output."""
    if not outputs or any(not o.exists() for o in outputs):
        return True
    return max((newest_mtime(i) for i in inputs), default=0.0) > min(o.stat().st_mtime for o in outputs)


def plan(force: bool, share_ok: bool) -> list[tuple[Step, str]]:
    """Each step with what will happen to it: run, skip (fresh), or skip (no share).

    Staleness is re-checked as steps run, so this preview can only under-count:
    a step that looks fresh now still runs if an earlier step rewrites its inputs.
    """
    out = []
    for s in STEPS:
        if s.share:
            out.append((s, "run" if share_ok else "skip: photo share not mounted"))
        elif force or is_stale(s.paths(s.inputs), s.paths(s.outputs)):
            out.append((s, "run"))
        else:
            out.append((s, "skip: up to date"))
    return out


def run_step(s: Step) -> None:
    print(f"\n=== {s.name}: python -m {' '.join(s.args)}", flush=True)
    t = time.time()
    subprocess.run([sys.executable, "-m", *s.args], cwd=ROOT, check=True)
    print(f"=== {s.name} done in {time.time() - t:.0f}s", flush=True)


def index_problems(folder: Path, live_garments: int | None) -> list[str]:
    """Integrity checks on an exported search index; an empty list means it may ship."""
    problems = []
    table = pd.read_csv(folder / "garments.csv", dtype={"accession": str})
    emb = np.load(folder / "emb.npy")
    if leaked := FORBIDDEN & set(table.columns):
        problems.append(f"personal-data columns in the index: {sorted(leaked)}")
    if table["accession"].duplicated().any():
        problems.append(f"{table['accession'].duplicated().sum()} duplicate accessions")
    if emb.shape != (len(table), 512):
        problems.append(f"embeddings are {emb.shape}, expected ({len(table)}, 512)")
    elif not np.isfinite(emb.astype(np.float32)).all():
        problems.append("embeddings contain NaN or inf")
    if live_garments and len(table) < MIN_KEEP * live_garments:
        problems.append(f"index has {len(table)} garments, live has {live_garments} "
                        f"(more than {1 - MIN_KEEP:.0%} lost)")
    return problems


def live_health(timeout: float = 10) -> dict | None:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def gate() -> bool:
    print("\n=== gate: tests, CI quality check, index integrity", flush=True)
    for cmd in (["-m", "pytest", "-q"], ["-m", "hfa.ci_check"]):
        if subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode != 0:
            print(f"GATE FAILED: python {' '.join(cmd)}")
            return False
    live = live_health()
    problems = index_problems(INDEX, live["garments"] if live else None)
    if live is None:
        print("note: live service didn't answer /health, so the size check was skipped")
    for p in problems:
        print(f"GATE FAILED: {p}")
    return not problems


def ssh(cmd: str) -> None:
    subprocess.run(["ssh", "-o", "BatchMode=yes", VPS, cmd], check=True)


def wait_for(garments: int, timeout: float = 180) -> bool:
    """Poll /health until the restarted service reports the new index size."""
    end = time.time() + timeout
    while time.time() < end:
        h = live_health(timeout=5)
        if h and h.get("garments") == garments:
            return True
        time.sleep(5)
    return False


def deploy() -> bool:
    n = len(pd.read_csv(INDEX / "garments.csv", usecols=["accession"]))
    print(f"\n=== deploy: {n} garments to {VPS}", flush=True)
    ssh(f"rm -rf {REMOTE_INDEX}.prev && cp -a {REMOTE_INDEX} {REMOTE_INDEX}.prev")
    subprocess.run(["rsync", "-a", "--delete", f"{INDEX}/", f"{VPS}:{REMOTE_INDEX}/"], check=True)
    ssh("systemctl --user restart hfa-search")
    if wait_for(n):
        print(f"deployed: {HEALTH_URL} reports {n} garments")
        return True
    print("new index didn't come up; rolling back to the previous one")
    ssh(f"rm -rf {REMOTE_INDEX} && mv {REMOTE_INDEX}.prev {REMOTE_INDEX} && systemctl --user restart hfa-search")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-photos", action="store_true")
    ap.add_argument("--deploy", action="store_true")
    args = ap.parse_args()

    share_ok = not args.no_photos and os.path.ismount(SHARE)
    steps = plan(args.force, share_ok)
    for s, what in steps:
        print(f"  {s.name:<13} {what}")
    if args.dry_run:
        return 0

    for s, _ in steps:  # re-check as we go: an earlier step may have made this one stale
        if s.share and not share_ok:
            continue
        if s.share or args.force or is_stale(s.paths(s.inputs), s.paths(s.outputs)):
            run_step(s)

    if not gate():
        return 1
    print("gate passed")
    if args.deploy and not deploy():
        return 1
    print(f"\nnext: commit and push {FRONTEND_RELATED.relative_to(ROOT.parent)} so Vercel picks up related.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
