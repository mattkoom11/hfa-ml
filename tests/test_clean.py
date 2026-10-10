import pandas as pd
import pytest

from hfa.clean import normalize_condition, normalize_location, split_donor, split_object_name
from hfa.dates import century_phrase_to_range, clean_phrase, numeric_range, parse_date
from hfa.manifest import parse_folder
from hfa.splits import assign_splits


def test_object_name_prefix():
    assert split_object_name("M Overcoat") == ("Overcoat", "male", None)
    assert split_object_name("C Boots") == ("Boots", None, "child")
    assert split_object_name("Dress") == ("Dress", None, None)
    assert split_object_name("Wedding Dress") == ("Wedding Dress", None, None)


def test_condition():
    assert normalize_condition("Good") == "good"
    assert normalize_condition(" Excellent ") == "excellent"
    assert normalize_condition("EACS") is None


def test_location():
    assert normalize_location("Flat Cabinet, Draw O") == "Flat Cabinet / Drawer O"
    assert normalize_location("Flat Cabinet Draw a") == "Flat Cabinet / Drawer A"
    assert normalize_location("Box #7") == "Box 7"
    assert normalize_location("Hanging") == "Hanging"


def test_donor_split():
    assert split_donor("Marcy Linton 11/2021") == ("Marcy Linton", "11/2021")
    assert split_donor("John & Rebecca Disosway, Jan 22, 2024") == ("John & Rebecca Disosway", "Jan 22, 2024")
    assert split_donor("Peggy Williams November, 2025") == ("Peggy Williams", "November, 2025")
    assert split_donor("Anonymous") == ("Anonymous", None)


def test_numeric_dates():
    assert numeric_range("Mid 20thc, 1960-1980") == (1960, 1980)
    assert numeric_range("Early 20thc, 1912") == (1912, 1912)
    assert numeric_range("mid 20thc, 1940c") == (1940, 1940)
    assert numeric_range("20thc") is None
    assert parse_date("Late 19thc Early 20thc, 1890-1910").method == "numeric"


def test_typo_cleanup():
    assert clean_phrase("Eartly 20thc") == "early 20thc"
    assert clean_phrase("Late 19th Early 20stc") == "late 19thc early 20thc"


@pytest.mark.parametrize("phrase,century", [
    ("18thc", 18), ("late 19thc", 19), ("mid 20thc", 20), ("first half 20thc", 20),
    ("second half 19thc", 19), ("early-mid 20thc", 20),
])
def test_century_phrases_stay_in_their_century(phrase, century):
    r = century_phrase_to_range(phrase)
    assert r is not None
    lo, hi = r
    assert lo <= hi
    assert (century - 1) * 100 <= lo and hi <= century * 100


def test_spanning_phrase():
    assert century_phrase_to_range("late 19thc early 20thc") == (1875, 1924)
    assert century_phrase_to_range("late 18thc, early 19thc") == (1775, 1824)


def test_quarter_boundaries():
    assert century_phrase_to_range("late 19thc") == (1875, 1899)
    assert century_phrase_to_range("early 20thc") == (1900, 1924)
    assert century_phrase_to_range("mid 20thc") == (1925, 1974)
    assert century_phrase_to_range("second half of 19thc") == (1850, 1899)
    assert century_phrase_to_range("wwi") is None


def test_whole_century_is_tagged():
    d = parse_date("20thc")
    assert (d.earliest, d.latest, d.method) == (1900, 1999, "century_whole")
    assert parse_date("Late 19th Early 20thc").method == "century"


def test_part_folders():
    assert parse_folder("ACS.2023.967a") == ("ACS.2023.967", "a")
    assert parse_folder("DR.2024.53.ab") == ("DR.2024.53", "ab")
    assert parse_folder("DR.2023.11 copy") == ("DR.2023.11", None)
    assert parse_folder("DR.2023.11") == ("DR.2023.11", None)


def test_splits_are_frozen():
    df = pd.DataFrame({"group": ["clothing"] * 40 + ["accessory"] * 20, "ca_type": ["dr"] * 40 + ["acs"] * 20})
    a, b = assign_splits(df), assign_splits(df)
    assert a.equals(b)
    assert set(a) == {"train", "val", "test"}
