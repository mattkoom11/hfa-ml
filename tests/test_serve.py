"""The live search service: auth, validation, rate limiting and ranking, with a fake encoder.

No model download: the encoder is injected, so this runs in CI.
"""

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hfa.serve import RateLimiter, SearchIndex, create_app  # noqa: E402

TOKEN = "test-token"


@pytest.fixture
def client(tmp_path):
    rows = [
        {"accession": "ST.2021.1", "title": "Suit", "ca_type": "st", "date_raw": "1940s", "date_earliest": 1940,
         "date_latest": 1949, "description": "Navy wool two-piece suit", "material": "wool", "color": "navy", "technique": ""},
        {"accession": "ACS.2024.1", "title": "Fan", "ca_type": "acs", "date_raw": "Late 19thc", "date_earliest": 1875,
         "date_latest": 1899, "description": None, "material": "", "color": "black", "technique": "painting"},
        {"accession": "DR.2023.1", "title": "Dress", "ca_type": "dr", "date_raw": "1920s", "date_earliest": 1920,
         "date_latest": 1929, "description": "Pink silk chiffon dress", "material": "silk;chiffon", "color": "pink", "technique": ""},
    ]
    pd.DataFrame(rows).to_csv(tmp_path / "garments.csv", index=False)
    np.save(tmp_path / "emb.npy", np.eye(3, 512, dtype=np.float16))  # garment i's vector points along axis i
    index = SearchIndex(tmp_path)
    fan_like = np.eye(1, 512, 1, dtype=np.float32)[0]  # the fake encoder thinks every query looks like the fan
    app = create_app(index, lambda q: fan_like, TOKEN, RateLimiter(limit=3, window=60))
    return TestClient(app)


def post(client, body, token=TOKEN):
    return client.post("/search", json=body, headers={"Authorization": f"Bearer {token}"})


def test_health_needs_no_token(client):
    assert client.get("/health").json() == {"ok": True, "garments": 3}


def test_rejects_missing_or_wrong_token(client):
    assert client.post("/search", json={"q": "suit"}).status_code == 401
    assert post(client, {"q": "suit"}, token="nope").status_code == 401


def test_validates_query(client):
    assert post(client, {"q": ""}).status_code == 422
    assert post(client, {"q": "x" * 201}).status_code == 422
    assert post(client, {"q": "suit", "k": 99}).status_code == 422


def test_fuses_keyword_and_image(client):
    r = post(client, {"q": "navy wool suit", "k": 3}).json()
    # keyword search ranks the suit first, image search the fan; RRF puts both on top
    assert set(r["results"][:2]) == {"ST.2021.1", "ACS.2024.1"}
    assert r["system"] == "fusion" and len(r["results"]) == 3


def test_missing_description_is_not_the_word_nan():
    # Regression: pandas NaN is truthy, so `v or ""` once indexed the literal word "nan"
    # for every garment without a description (391 of them), skewing keyword scores.
    from hfa.search import build_bm25

    table = pd.DataFrame([{"title": "Fan", "ca_type": "acs", "date_raw": float("nan"), "description": float("nan")}],
                         index=["ACS.2024.1"])
    bm25 = build_bm25(table, {"ACS.2024.1": {"material": set(), "color": set(), "technique": set()}})
    assert "nan" not in bm25.docs[0]


def test_rate_limit(client):
    codes = [post(client, {"q": "suit"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
