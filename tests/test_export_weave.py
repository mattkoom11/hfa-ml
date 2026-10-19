import json

import pandas as pd
import pytest

from hfa.export_weave import (
    FIXTURE_CSV,
    build_weave,
    classify,
    decade_weights,
    load_vocab,
    scanned_dates,
    split_codes,
)

VOCAB = {
    "color": {"black", "white", "blue"},
    "material": {"silk", "cotton", "lace"},
    "technique": {"embroidery"},
}
COLUMNS = ["accession", "date_earliest", "date_latest", "material", "color", "technique"]


def rows(*records):
    return pd.DataFrame([dict(zip(COLUMNS, r)) for r in records], columns=COLUMNS, dtype=str)


def test_decade_weights_spread_by_year_overlap():
    w = decade_weights(1890, 1920)
    assert w == pytest.approx({1890: 10 / 31, 1900: 10 / 31, 1910: 10 / 31, 1920: 1 / 31})


def test_decade_weights_inside_one_decade():
    assert decade_weights(1910, 1919) == {1910: 1.0}
    assert decade_weights(1925, 1925) == {1920: 1.0}


@pytest.mark.parametrize("earliest,latest", [(1775, 1824), (1903, 1947), (1999, 2001), (1890, 1988)])
def test_decade_weights_always_sum_to_one(earliest, latest):
    assert sum(decade_weights(earliest, latest).values()) == pytest.approx(1)


@pytest.mark.parametrize(
    "earliest,latest,kind",
    [
        ("1890", "1920", "dated"),
        ("1890", "1988", "dated"),
        ("1800", "1899", "century_only"),
        ("1800", "1900", "century_only"),
        ("", "1920", "undated"),
        ("1890", "", "undated"),
        ("abc", "1900", "undated"),
        ("1950", "1900", "undated"),
        (1910.0, 1920.0, "dated"),
        (float("nan"), 1920.0, "undated"),
    ],
)
def test_classify(earliest, latest, kind):
    assert classify(earliest, latest) == kind


def test_split_codes_keeps_known_codes_and_counts_unknown_ones():
    unknown: dict[str, int] = {}
    assert split_codes("lace; silk;polka", VOCAB["material"], unknown) == ["lace", "silk"]
    assert unknown == {"polka": 1}
    assert split_codes("", VOCAB["material"], unknown) == []
    assert split_codes(None, VOCAB["material"], unknown) == []


def test_build_weave_splits_attributes_and_counts_every_row():
    weave, unknown = build_weave(
        rows(
            ("A.1", "1910", "1919", "silk;cotton", "black", "embroidery"),
            ("A.2", "1910", "1919", "silk", "", ""),
            ("A.3", "1800", "1899", "silk", "black", ""),
            ("A.4", "", "", "silk", "black", ""),
        ),
        VOCAB,
        {},
        "ci-fixture",
        "2026-10-19",
    )
    assert unknown == {}
    assert (weave["rows"], weave["dated"], weave["century_only"], weave["undated"]) == (4, 2, 1, 1)
    assert weave["decades"] == [
        {
            "decade": 1910,
            "garments": 2.0,
            "colors": {"black": 1.0},
            "materials": {"cotton": 0.5, "silk": 1.5},
            "techniques": {"embroidery": 1.0},
        }
    ]
    assert weave["source"] == "ci-fixture"
    assert weave["generated"] == "2026-10-19"
    assert weave["scanned"] == {}


def test_decades_are_contiguous_with_empty_gaps():
    weave, _ = build_weave(
        rows(("A.1", "1900", "1905", "silk", "black", ""), ("A.2", "1930", "1931", "cotton", "white", "")),
        VOCAB,
        {},
        "ci-fixture",
        "2026-10-19",
    )
    assert [d["decade"] for d in weave["decades"]] == [1900, 1910, 1920, 1930]
    assert weave["decades"][1] == {"decade": 1910, "garments": 0.0, "colors": {}, "materials": {}, "techniques": {}}


def test_no_dated_rows_gives_no_decades():
    weave, _ = build_weave(rows(("A.1", "", "", "silk", "black", "")), VOCAB, {}, "ci-fixture", "2026-10-19")
    assert weave["decades"] == []
    assert weave["undated"] == 1


def test_refuses_tables_with_personal_data_columns():
    table = rows(("A.1", "1910", "1919", "silk", "black", "")).assign(email="someone@example.com")
    with pytest.raises(ValueError, match="personal-data"):
        build_weave(table, VOCAB, {}, "ci-fixture", "2026-10-19")


def test_scanned_dates_keeps_only_dated_catalogued_accessions():
    catalog = pd.DataFrame(
        {
            "accession": ["DR.1", "DR.2", "DR.3"],
            "date_earliest": [1910.0, 1800.0, float("nan")],
            "date_latest": [1920.0, 1899.0, float("nan")],
        }
    ).set_index("accession")
    assert scanned_dates(catalog, ["DR.1", "DR.2", "DR.3", "DR.9"]) == {"DR.1": {"earliest": 1910, "latest": 1920}}


def test_scanned_dates_refuses_personal_data_columns():
    catalog = pd.DataFrame({"accession": ["DR.1"], "date_earliest": [1910.0], "date_latest": [1920.0], "email": ["x"]})
    with pytest.raises(ValueError, match="personal-data"):
        scanned_dates(catalog.set_index("accession"), ["DR.1"])


def test_end_to_end_on_the_ci_fixture():
    table = pd.read_csv(FIXTURE_CSV, dtype=str, keep_default_na=False)
    weave, _ = build_weave(table, load_vocab(), {}, "ci-fixture", "2026-10-19")
    assert weave["dated"] + weave["century_only"] + weave["undated"] == weave["rows"] == len(table)
    total = sum(d["garments"] for d in weave["decades"])
    assert total == pytest.approx(weave["dated"], abs=0.001 * len(weave["decades"]) + 1e-9)
    decades = [d["decade"] for d in weave["decades"]]
    assert decades == list(range(decades[0], decades[-1] + 10, 10))
    json.dumps(weave)
