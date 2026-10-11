"""Local judging page for Phase 3: grade pooled candidates 0 / 1 / 2 per query.

Serves hfa/static/judge.html, garment photos from data/interim/jpg, and a JSON
API on 127.0.0.1 only. Each grade is saved immediately (atomic replace) to
data/judgments/judgments.json. The page never shows which system found a
candidate.

Run:  .venv/bin/python -m hfa.judge_server [--port 8766] [--judge matthew] [--depth 5]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "static/judge.html"
JPG = (ROOT / "data/interim/jpg").resolve()
POOL = ROOT / "data/judgments/pool.json"
OUT = ROOT / "data/judgments/judgments.json"
REJUDGE_SET = ROOT / "data/judgments/rejudge_set.json"
MAX_PHOTOS = 4
_lock = threading.Lock()


def load_judgments() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_grade(query: str, cand: str, rec: dict) -> None:
    with _lock:
        data = load_judgments()
        data.setdefault(query, {})[cand] = rec
        tmp = OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
        os.replace(tmp, OUT)


def garment_card(a: str, g: pd.DataFrame) -> dict:
    photos = sorted(p.name for p in (JPG / a).glob("*.jpg"))[:MAX_PHOTOS] if (JPG / a).is_dir() else []
    row = g.loc[a]
    return {
        "accession": a,
        "title": row.get("title") or "",
        "type": row.get("ca_type") or "",
        "date": row.get("date_raw") or "",
        "description": row.get("description") or "",
        "photos": [f"/img/{a}/{p}" for p in photos],
    }


class Handler(BaseHTTPRequestHandler):
    payload: dict = {}
    valid: dict[str, set[str]] = {}
    judge = "matthew"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith(("text/html", "application/json")) else "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            return self._json(200, {**self.payload, "judgments": load_judgments()})
        if self.path.startswith("/img/"):
            target = (JPG / unquote(self.path[len("/img/"):])).resolve()
            # Only files inside the local photo cache, never anything else on disk.
            if target.is_relative_to(JPG) and target.suffix.lower() == ".jpg" and target.is_file():
                return self._send(200, target.read_bytes(), "image/jpeg")
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/grade":
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            q, cand, grade = body["query"], body["candidate"], int(body["grade"])
        except (ValueError, KeyError, TypeError):
            return self._json(400, {"error": "bad request"})
        if cand not in self.valid.get(q, set()) or grade not in (0, 1, 2):
            return self._json(400, {"error": "unknown pair or grade"})
        rec = {"grade": grade, "note": str(body.get("note", "")).strip()[:300], "judge": self.judge,
               "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        save_grade(q, cand, rec)
        self._json(200, {"ok": True, "record": rec})

    def log_message(self, fmt, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--judge", default="matthew")
    ap.add_argument("--task", choices=["recsys", "search"], default="recsys",
                    help="search: judge text queries from search_pool.json into search_judgments.json")
    ap.add_argument("--rejudge", action="store_true",
                    help="blind re-grade of [search_]rejudge_set.json into [search_]rejudgments.json; earlier grades hidden")
    ap.add_argument("--depth", type=int, default=5,
                    help="only show candidates in some system's top N (the site shows 4); 0 shows the whole pool")
    args = ap.parse_args()
    if not POOL.exists():
        raise SystemExit("no pool yet; run: .venv/bin/python -m hfa.judge_sample")

    global OUT
    rubric = None
    if args.task == "search":
        OUT = ROOT / "data/judgments/search_judgments.json"
        pool = json.loads((ROOT / "data/judgments/search_pool.json").read_text())
        args.depth = 0
        rubric = ["2 · Strongly matches: clearly what the search asked for.",
                  "1 · Partly matches: gets some of the search right (e.g. the right object but wrong color or period).",
                  "0 · Doesn't match the search.",
                  "Judge against the words of the search, not whether the garment is interesting."]
    else:
        pool = json.loads(POOL.read_text())
    if args.rejudge:
        prefix = "search_" if args.task == "search" else ""
        OUT = ROOT / f"data/judgments/{prefix}rejudgments.json"
        pairs = json.loads((ROOT / f"data/judgments/{prefix}rejudge_set.json").read_text())["pairs"]  # shuffled
        full = pool
        pool = {}
        for q, x in pairs:
            pool.setdefault(q, {"candidates": [], "text": full.get(q, {}).get("text", "")})["candidates"].append(x)
        args.depth = 0
    if args.depth:
        # Grades already given beyond this depth stay in judgments.json; they just aren't asked for.
        from hfa.recsys import SYSTEMS, load_catalog, rank
        c = load_catalog()
        cv_path = ROOT / "data/predictions/rerank_cv.json"
        rerank_cv = json.loads(cv_path.read_text()) if cv_path.exists() else {}
        for q, e in pool.items():
            keep = set().union(*[rank(c, q, s, args.depth) for s in SYSTEMS], rerank_cv.get(q, [])[: args.depth])
            e["candidates"] = [x for x in e["candidates"] if x in keep]
    g = pd.read_parquet(ROOT / "data/processed/garments.parquet").set_index("accession")
    g = g.astype(object).where(g.notna(), None)
    needed = {x for e in pool.values() for x in e["candidates"]} | (set() if args.task == "search" else set(pool))
    cards = {a: garment_card(a, g) for a in needed}
    if args.task == "search":
        for q, e in pool.items():  # the "query card" is the search text itself
            cards[q] = {"accession": q, "title": "", "type": "search", "date": "", "description": e["text"], "photos": []}
    Handler.payload = {
        "queries": [{"query": q, "candidates": e["candidates"]} for q, e in pool.items()],
        "cards": cards, "rubric": rubric, "task": args.task,
    }
    Handler.valid = {q: set(e["candidates"]) for q, e in pool.items()}
    judged = load_judgments()
    done = sum(1 for q, e in pool.items() for x in e["candidates"] if x in judged.get(q, {}))
    Handler.judge = args.judge

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    total = sum(len(e["candidates"]) for e in pool.values())
    print(f"judging {total} pairs across {len(pool)} queries ({done} done) at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
