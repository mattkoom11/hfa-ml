import json

import pandas as pd

from hfa import label_server
from hfa.label_sample import allocate


def test_allocate_respects_floor_and_total():
    counts = pd.Series({"dr": 346, "acs": 232, "vt": 6, "mt": 2})
    take = allocate(counts, 60, 4)
    assert take.sum() == 60
    assert take["mt"] == 2 and take["vt"] >= 4
    assert (take <= counts).all()


def test_validate_drops_unknown_codes():
    allowed = {"material": {"wool", "silk"}, "color": {"red"}, "technique": set()}
    rec = label_server.validate(
        {"material": ["wool", "made_up"], "color": ["red"], "technique_none": True, "status": "weird"}, allowed)
    assert rec["material"] == ["wool"]
    assert rec["status"] == "done"
    assert rec["technique"] == [] and rec["technique_none"] is True


def test_save_label_is_atomic_and_merges(tmp_path, monkeypatch):
    monkeypatch.setattr(label_server, "LABELS", tmp_path / "labels.json")
    label_server.save_label("DR.2023.1", {"status": "done"})
    label_server.save_label("DR.2023.2", {"status": "skipped"})
    data = json.loads((tmp_path / "labels.json").read_text())
    assert set(data) == {"DR.2023.1", "DR.2023.2"}
    assert not (tmp_path / "labels.json.tmp").exists()
