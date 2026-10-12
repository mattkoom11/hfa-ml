import os

import numpy as np
import pandas as pd

from hfa.refresh import index_problems, is_stale
from hfa.splits import assign_splits, extend_splits


def touch(p, mtime):
    p.write_text("x")
    os.utime(p, (mtime, mtime))
    return p


def test_stale_when_an_output_is_missing_or_older_than_an_input(tmp_path):
    src, out = touch(tmp_path / "in", 100), touch(tmp_path / "out", 200)
    assert not is_stale([src], [out])
    os.utime(src, (300, 300))
    assert is_stale([src], [out])
    assert is_stale([src], [tmp_path / "missing"])


def test_a_new_file_inside_an_input_folder_makes_the_step_stale(tmp_path):
    folder = tmp_path / "jpg"
    (folder / "DR.1").mkdir(parents=True)
    touch(folder / "DR.1" / "a.jpg", 100)
    out = touch(tmp_path / "emb.npz", 200)
    assert not is_stale([folder], [out])
    touch(folder / "DR.1" / "b.jpg", 300)
    assert is_stale([folder], [out])


def garments(n, start=0):
    return pd.DataFrame({"accession": [f"DR.{i}" for i in range(start, start + n)],
                         "group": "clothing", "ca_type": "dress"})


def test_new_garments_never_move_existing_ones_between_splits():
    old = garments(40)
    frozen = dict(zip(old["accession"], assign_splits(old)))
    grown = pd.concat([old, garments(25, start=40)], ignore_index=True)
    split = extend_splits(grown, frozen)
    assert (split[:40].values == old["accession"].map(frozen).values).all()
    assert split[40:].notna().all() and set(split[40:]) <= {"train", "val", "test"}
    assert (extend_splits(grown, frozen) == split).all()  # same data, same answer


def write_index(folder, n, columns=("accession", "title"), emb=None):
    folder.mkdir()
    pd.DataFrame({c: [f"{c}{i}" for i in range(n)] for c in columns}).to_csv(folder / "garments.csv", index=False)
    np.save(folder / "emb.npy", emb if emb is not None else np.ones((n, 512), dtype=np.float16))
    return folder


def test_a_good_index_passes(tmp_path):
    assert index_problems(write_index(tmp_path / "i", 100), live_garments=100) == []


def test_gate_blocks_personal_data_bad_embeddings_and_lost_garments(tmp_path):
    pii = write_index(tmp_path / "pii", 10, columns=("accession", "title", "donor_name"))
    assert any("personal-data" in p for p in index_problems(pii, None))

    short = write_index(tmp_path / "short", 10, emb=np.ones((9, 512), dtype=np.float16))
    assert any("embeddings are" in p for p in index_problems(short, None))

    nan = np.ones((10, 512), dtype=np.float16)
    nan[3, 7] = np.nan
    assert any("NaN" in p for p in index_problems(write_index(tmp_path / "nan", 10, emb=nan), None))

    assert any("lost" in p for p in index_problems(write_index(tmp_path / "small", 90), live_garments=100))
