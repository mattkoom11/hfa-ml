import pytest

from hfa.accession import decode_prefix, normalize_accession


@pytest.mark.parametrize("raw,expected", [
    ("DR.2023.17", "DR.2023.17"),
    ("JKT.2022 124", "JKT.2022.124"),
    ("SK.2023,34", "SK.2023.34"),
    ("DR,2023.458", "DR.2023.458"),
    ("ACS.2026/58", "ACS.2026.58"),
    ("acs.2024.007", "ACS.2024.7"),
    ("ACS.2023.967.1", "ACS.2023.967.1"),
])
def test_normalizes(raw, expected):
    assert normalize_accession(raw).value == expected


@pytest.mark.parametrize("raw,issue", [
    ("2023.214", "no_prefix"),
    ("2023.760-779", "range"),
    ("ACS.2025.207 ??", "uncertain"),
    ("ACS.20205.12", "bad_year"),
    ("XYZ.2023.1", "unknown_prefix"),
    ("", "blank"),
])
def test_flags(raw, issue):
    assert issue in normalize_accession(raw).issues


def test_bad_year_is_not_trusted():
    assert normalize_accession("ACS.20205.12").value is None


def test_decode_prefix():
    assert decode_prefix("CDR") == ("dr", None, "child")
    assert decode_prefix("WENS") == ("end", "female", None)
    assert decode_prefix("ETH") == ("nw", None, None)
    assert decode_prefix("NOPE") == (None, None, None)
