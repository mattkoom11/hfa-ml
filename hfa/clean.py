"""Turn Historic_Collection_Updated.xlsx into clean tables.

Outputs (all under data/, which git ignores):
  data/interim/collection_full.parquet    every column, including donor PII (CA import only)
  data/processed/collection_clean.parquet no donor or email columns (ML work uses this one)
  data/reports/cleaning_issues.csv        one row per flagged problem, accession + issue only

The source workbook is never modified.
Run:  .venv/bin/python -m hfa.clean
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from hfa.accession import ACCESSORY_TYPES, decode_prefix, normalize_accession
from hfa.dates import parse_date

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data/raw/Historic_Collection_Updated.xlsx"

COLUMNS = {
    "Accession #": "accession_raw",
    "Object Name": "object_name_raw",
    "Location": "location_raw",
    "Date": "date_raw",
    "Description": "description",
    "Maker_Label": "maker_label",
    "Old Number": "old_number",
    "Condition": "condition_raw",
    "Donor": "donor_raw",
    "Email": "email",
}
PII_COLUMNS = ["donor_raw", "donor_name", "donation_date_text", "email"]
CONDITIONS = {"excellent", "very good", "good", "fair", "poor"}
NAME_PREFIX = {"M": ("male", None), "W": ("female", None), "C": (None, "child")}
_DATE_TAIL = re.compile(
    r"[,\s]+((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{0,2},?\s*\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}/\d{4}|\d{4})\s*$",
    re.IGNORECASE,
)


def load_sheets(path: Path = SOURCE) -> pd.DataFrame:
    """Concatenate every year sheet by header name, so a missing column can't shift others."""
    frames = []
    for name, df in pd.read_excel(path, sheet_name=None, dtype=str).items():
        df = df.rename(columns=lambda c: str(c).strip()).rename(columns=COLUMNS)
        df = df.dropna(how="all")
        df["source_sheet"] = name
        df["source_row"] = df.index + 2  # 1-based, after the header row
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for col in COLUMNS.values():
        if col not in out:
            out[col] = None
    return out


def split_object_name(name: object) -> tuple[str | None, str | None, str | None]:
    """'M Overcoat' -> ('Overcoat', 'male', None); 'C Boots' -> ('Boots', None, 'child')."""
    if name is None or pd.isna(name):
        return None, None, None
    s = str(name).strip()
    m = re.match(r"^([MWC])\s+(.+)$", s)
    if m:
        gender, age = NAME_PREFIX[m.group(1)]
        return m.group(2).strip(), gender, age
    return s, None, None


def normalize_condition(raw: object) -> str | None:
    if raw is None or pd.isna(raw):
        return None
    s = str(raw).strip().lower()
    return s.replace(" ", "_") if s in CONDITIONS else None


def normalize_location(raw: object) -> str | None:
    if raw is None or pd.isna(raw):
        return None
    s = re.sub(r"\s+", " ", str(raw).strip())
    m = re.match(r"^flat cabinet,?\s*(?:draw|drawer)\s*(\w+)$", s, re.IGNORECASE)
    if m:
        return f"Flat Cabinet / Drawer {m.group(1).upper()}"
    m = re.match(r"^box\s*#?\s*(\w+)$", s, re.IGNORECASE)
    if m:
        return f"Box {m.group(1).upper()}"
    return s


def split_donor(raw: object) -> tuple[str | None, str | None]:
    """'Marcy Linton 11/2021' -> ('Marcy Linton', '11/2021')."""
    if raw is None or pd.isna(raw):
        return None, None
    s = re.sub(r"\s+", " ", str(raw).strip())
    m = _DATE_TAIL.search(s)
    if m:
        return s[: m.start()].strip(" ,") or None, m.group(1)
    return s, None


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    issues: list[dict] = []

    def flag(i, field, issue):
        issues.append({"accession_raw": df.at[i, "accession_raw"], "source_sheet": df.at[i, "source_sheet"],
                       "source_row": df.at[i, "source_row"], "field": field, "issue": issue})

    acc = df["accession_raw"].map(normalize_accession)
    df["accession"] = acc.map(lambda a: a.value)
    df["prefix"] = acc.map(lambda a: a.prefix)
    df["accession_year"] = acc.map(lambda a: a.year)
    for i, a in acc.items():
        for issue in a.issues:
            if issue != "punctuation_fixed":
                flag(i, "accession", issue)

    decoded = df["prefix"].map(decode_prefix)
    df["ca_type"] = decoded.map(lambda t: t[0])
    prefix_gender = decoded.map(lambda t: t[1])
    prefix_age = decoded.map(lambda t: t[2])

    names = df["object_name_raw"].map(split_object_name)
    df["title"] = names.map(lambda t: t[0])
    df["gender"] = names.map(lambda t: t[1]).fillna(prefix_gender)
    df["age_group"] = names.map(lambda t: t[2]).fillna(prefix_age)

    dates = df["date_raw"].map(parse_date)
    df["date_earliest"] = dates.map(lambda d: d.earliest).astype("Int64")
    df["date_latest"] = dates.map(lambda d: d.latest).astype("Int64")
    df["date_method"] = dates.map(lambda d: d.method)
    for i, d in dates.items():
        if d.method in ("unparsed", "blank"):
            flag(i, "date", d.method)

    df["condition"] = df["condition_raw"].map(normalize_condition)
    for i in df.index[df["condition"].isna()]:
        flag(i, "condition", "blank" if pd.isna(df.at[i, "condition_raw"]) else f"invalid:{df.at[i, 'condition_raw']}")

    df["storage_location"] = df["location_raw"].map(normalize_location)

    donors = df["donor_raw"].map(split_donor)
    df["donor_name"] = donors.map(lambda t: t[0])
    df["donation_date_text"] = donors.map(lambda t: t[1])

    df["group"] = df["ca_type"].map(lambda t: None if pd.isna(t) else ("accessory" if t in ACCESSORY_TYPES else "clothing"))
    df["has_description"] = df["description"].notna() & df["description"].astype(str).str.strip().ne("")

    dup = df["accession"].notna() & df["accession"].duplicated(keep=False)
    for i in df.index[dup]:
        flag(i, "accession", "duplicate")

    return df, pd.DataFrame(issues)


def main() -> None:
    df, issues = clean(load_sheets())
    for sub in ("interim", "processed", "reports"):
        (ROOT / "data" / sub).mkdir(parents=True, exist_ok=True)
    df.to_parquet(ROOT / "data/interim/collection_full.parquet", index=False)
    df.drop(columns=PII_COLUMNS).to_parquet(ROOT / "data/processed/collection_clean.parquet", index=False)
    issues.to_csv(ROOT / "data/reports/cleaning_issues.csv", index=False)

    print(f"rows: {len(df)}  with usable accession: {df['accession'].notna().sum()}")
    print("date parse methods:", df["date_method"].value_counts().to_dict())
    print("issues by field/type:")
    if len(issues):
        print(issues.groupby(["field", issues["issue"].str.split(":").str[0]]).size().to_string())


if __name__ == "__main__":
    main()
