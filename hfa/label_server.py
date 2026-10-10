"""Local labeling page for Phase 1 (material / color / technique from descriptions).

Serves hfa/static/label.html and a tiny JSON API on 127.0.0.1 only.
Labels are written after every garment to data/labels/labels.json (atomic
replace), so closing the tab or the server never loses more than the garment
on screen.

Run:  .venv/bin/python -m hfa.label_server [--port 8765] [--labeler matthew]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "static/label.html"
VOCAB = Path(__file__).resolve().parent / "vocab.json"
SET = ROOT / "data/labels/label_set.csv"
LABELS = ROOT / "data/labels/labels.json"
FIELDS = ("material", "color", "technique")
_lock = threading.Lock()


def load_labels() -> dict:
    return json.loads(LABELS.read_text()) if LABELS.exists() else {}


def save_label(accession: str, record: dict) -> None:
    with _lock:
        labels = load_labels()
        labels[accession] = record
        tmp = LABELS.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(labels, indent=1, ensure_ascii=False))
        os.replace(tmp, LABELS)


def validate(body: dict, allowed: dict[str, set[str]]) -> dict:
    """Keep only known codes, so a stale page can't write junk into the label file."""
    rec = {"status": body.get("status") if body.get("status") in ("done", "skipped") else "done"}
    for f in FIELDS:
        rec[f] = sorted(set(body.get(f, [])) & allowed[f])
        rec[f"{f}_none"] = bool(body.get(f"{f}_none"))
        rec[f"other_{f}"] = str(body.get(f"other_{f}", "")).strip()[:200]
    rec["note"] = str(body.get("note", "")).strip()[:500]
    return rec


class Handler(BaseHTTPRequestHandler):
    items: list[dict] = []
    vocab: dict = {}
    allowed: dict[str, set[str]] = {}
    labeler = "matthew"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/items":
            self._json(200, {"vocab": self.vocab, "items": self.items, "labels": load_labels()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/api/label":
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            accession = body["accession"]
        except (ValueError, KeyError):
            return self._json(400, {"error": "bad request"})
        if accession not in {i["accession"] for i in self.items}:
            return self._json(400, {"error": "unknown accession"})
        rec = validate(body, self.allowed)
        rec["labeler"] = self.labeler
        rec["labeled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_label(accession, rec)
        self._json(200, {"ok": True, "record": rec})

    def log_message(self, fmt, *args) -> None:  # keep the terminal quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--labeler", default="matthew")
    args = ap.parse_args()
    if not SET.exists():
        raise SystemExit("no labeling set yet; run: .venv/bin/python -m hfa.label_sample")

    df = pd.read_csv(SET, dtype=str).fillna("")
    Handler.items = df.to_dict("records")
    Handler.vocab = json.loads(VOCAB.read_text())["lists"]
    Handler.allowed = {f: {x["code"] for x in Handler.vocab[f]} for f in FIELDS}
    Handler.labeler = args.labeler

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    done = sum(1 for v in load_labels().values() if v.get("status"))
    print(f"labeling {len(Handler.items)} garments ({done} already labeled) at http://127.0.0.1:{args.port}", flush=True)
    print("Ctrl+C to stop; every answer is already saved.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
