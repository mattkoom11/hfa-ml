import json
from pathlib import Path

import pandas as pd
import pytest

FIX = Path(__file__).parent / "fixtures/search"
pytestmark = pytest.mark.skipif(not (FIX / "garments.csv").exists(), reason="fixture not built yet")


def test_fixture_has_no_personal_data():
    cols = set(pd.read_csv(FIX / "garments.csv", nrows=0).columns)
    assert not cols & {"donor", "donor_raw", "donor_name", "donation_date_text", "email"}


def test_fixture_rows_line_up_with_embeddings():
    import numpy as np

    n = len(pd.read_csv(FIX / "garments.csv"))
    assert np.load(FIX / "emb.npy").shape == (n, 512)


def test_every_query_has_a_parse():
    queries = json.loads((FIX / "queries.json").read_text())
    assert all({"decade_from", "ca_type", "materials"} <= set(q["parse"]) for q in queries.values())
