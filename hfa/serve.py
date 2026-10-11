"""Live collection search: fusion of keyword (BM25) and FashionCLIP image search, CPU only.

The deployed counterpart of the Phase 5 "fusion" system (nDCG@5 0.80 on the
40-query benchmark). No LLM, so it fits a 2-core VPS.

  POST /search   {"q": "navy wool suit", "k": 12}  ->  {"results": [accessions], "system", "took_ms"}
                 requires "Authorization: Bearer $HFA_SEARCH_TOKEN"
  GET  /health   {"ok": true, "garments": N}

Guards: bearer token (compared in constant time), query length cap, k cap, and a
per-client sliding-window rate limit. Meant to sit behind HTTPS and be called only by
the website's server, never by browsers directly.

Run:  HFA_SEARCH_TOKEN=... HFA_SEARCH_INDEX=data/exports/search_index \\
      uvicorn hfa.serve:app --host 127.0.0.1 --port 8800
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from hfa.search import build_bm25, rrf, tokens

MAX_K = 24
RATE_LIMIT = 30  # requests per client per window
RATE_WINDOW = 60.0  # seconds


class Query(BaseModel):
    q: str = Field(min_length=1, max_length=200)
    k: int = Field(default=12, ge=1, le=MAX_K)


class SearchIndex:
    """The exported index (hfa.search_index) plus a keyword index built from it."""

    def __init__(self, folder: Path):
        table = pd.read_csv(folder / "garments.csv", dtype={"accession": str}).set_index("accession")
        self.table = table.astype(object).where(table.notna(), None)
        self.accs = np.array(self.table.index)
        self.emb = np.load(folder / "emb.npy").astype(np.float32)
        attrs = {a: {f: set(filter(None, (self.table.at[a, f] or "").split(";"))) for f in ("material", "color", "technique")}
                 for a in self.accs}
        self.bm25 = build_bm25(self.table, attrs)

    def search(self, q: str, k: int, q_emb: np.ndarray) -> list[str]:
        s = self.bm25.score(tokens(q))
        bm25_order = [j for j in np.argsort(-s, kind="stable") if s[j] > 0]
        clip_order = np.argsort(-(self.emb @ q_emb), kind="stable")
        return [str(self.accs[j]) for j in rrf([bm25_order, clip_order])[:k]]


class RateLimiter:
    def __init__(self, limit: int = RATE_LIMIT, window: float = RATE_WINDOW):
        self.limit, self.window = limit, window
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


def clip_text_encoder(threads: int = 2) -> Callable[[str], np.ndarray]:
    """FashionCLIP's text tower on CPU; ~600 MB download on first run."""
    import open_clip
    import torch

    torch.set_num_threads(threads)
    model, _, _ = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionCLIP", device="cpu")
    model.eval()
    tok = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")

    def encode(text: str) -> np.ndarray:
        with torch.no_grad():
            e = model.encode_text(tok([text])).float()
        return torch.nn.functional.normalize(e, dim=-1)[0].numpy()

    return encode


def create_app(index: SearchIndex, encode: Callable[[str], np.ndarray], token: str,
               limiter: RateLimiter | None = None) -> FastAPI:
    if not token:
        raise RuntimeError("refusing to start without a search token")
    limiter = limiter or RateLimiter()
    app = FastAPI(title="hfa-ml search", docs_url=None, redoc_url=None, openapi_url=None)

    def authorized(request: Request, authorization: str = Header(default="")) -> str:
        given = authorization.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="unauthorized")
        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            raise HTTPException(status_code=429, detail="too many requests")
        return client

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "garments": len(index.accs)}

    @app.post("/search")
    def search(query: Query, _: str = Depends(authorized)) -> dict:
        t = time.perf_counter()
        results = index.search(query.q, query.k, encode(query.q))
        return {"results": results, "system": "fusion", "took_ms": round((time.perf_counter() - t) * 1000)}

    return app


def _from_env() -> FastAPI:
    folder = Path(os.environ.get("HFA_SEARCH_INDEX", "data/exports/search_index"))
    threads = int(os.environ.get("HFA_SEARCH_THREADS", "2"))
    return create_app(SearchIndex(folder), clip_text_encoder(threads), os.environ.get("HFA_SEARCH_TOKEN", ""))


def __getattr__(name: str):
    # `uvicorn hfa.serve:app` builds the real app lazily, so importing this module
    # (tests, CI) never loads the model or needs a token.
    if name == "app":
        globals()["app"] = _from_env()
        return globals()["app"]
    raise AttributeError(name)
