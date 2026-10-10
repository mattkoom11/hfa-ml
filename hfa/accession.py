"""Accession-number normalization and prefix decoding.

Accession numbers look like ``DR.2023.17``: a category prefix, the accession
year, and a running number. The spreadsheet has typos (commas, spaces, missing
prefixes, ranges), so every raw value goes through ``normalize_accession``
before anything is joined on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# prefix -> (CA record type, gender, age group). C- means child, W- means women's,
# per the photo folders ("Children's garments") and the CA profile's type tree.
PREFIX_MAP: dict[str, tuple[str, str | None, str | None]] = {
    "DR": ("dr", None, None),
    "WDR": ("dr", "female", None),
    "CDR": ("dr", None, "child"),
    "JKT": ("jkt", None, None),
    "CJKT": ("jkt", None, "child"),
    "OTW": ("otw", None, None),
    "COTW": ("otw", None, "child"),
    "ST": ("st", None, None),
    "CST": ("st", None, "child"),
    "ENS": ("end", None, None),
    "CENS": ("end", None, "child"),
    "WENS": ("end", "female", None),
    "MT": ("mt", None, None),
    "OC": ("oc", None, None),
    "VT": ("vt", None, None),
    "TR": ("tr", None, None),
    "CTR": ("tr", None, "child"),
    "SH": ("sh", None, None),
    "CSH": ("sh", None, "child"),
    "SW": ("sw", None, None),
    "RB": ("rb", None, None),
    "PJ": ("pj", None, None),
    "SK": ("sk", None, None),
    "CSK": ("sk", None, "child"),
    "BD": ("bd", None, None),
    "WBD": ("bd", "female", None),
    "UND": ("und", None, None),
    "NW": ("nw", None, None),
    "CNW": ("nw", None, "child"),
    "ETH": ("nw", None, None),  # "Ethnic" photo folder; profile calls it Non-Western
    "ACS": ("acs", None, None),
    "CACS": ("acs", None, "child"),
    "JWL": ("jwl", None, None),
    "TEX": ("tex", None, None),
    "HW": ("hw", None, None),
    "SHO": ("sho", None, None),
}

# CA types that count as accessories when results are split clothing vs accessory.
ACCESSORY_TYPES = {"acs", "jwl", "tex", "hw", "sho"}

_CANON = re.compile(r"^([A-Z]{2,5})\.(\d{4,5})\.(\d+)((?:\.\d+)*)$")


@dataclass(frozen=True)
class Accession:
    raw: str
    value: str | None  # normalized "PREFIX.YYYY.N", or None if it can't be trusted
    prefix: str | None
    year: int | None
    issues: tuple[str, ...]


def normalize_accession(raw: object) -> Accession:
    text = "" if raw is None else str(raw).strip()
    issues: list[str] = []
    if not text or text.lower() == "nan":
        return Accession(text, None, None, None, ("blank",))

    s = text.upper()
    if "?" in s:
        issues.append("uncertain")
        s = s.replace("?", "").strip()
    if re.search(r"\d-\d", s):
        issues.append("range")
    # Commas and stray spaces between the three parts are typing slips for dots.
    t = re.sub(r"\s*[,/\s]\s*", ".", s)
    t = re.sub(r"\.+", ".", t).strip(".")

    m = _CANON.match(t)
    if not m:
        issues.append("no_prefix" if re.match(r"^\d{4}\.\d+", t) else "unparseable")
        return Accession(text, None, None, None, tuple(issues))

    prefix, year, num, sub = m.groups()
    if t != s:
        issues.append("punctuation_fixed")
    if sub:
        issues.append("sub_number")
    if prefix not in PREFIX_MAP:
        issues.append("unknown_prefix")
    y = int(year)
    if not 1900 <= y <= 2100:
        # e.g. "20205": obvious to a person, but a curator should confirm the fix.
        issues.append("bad_year")
        return Accession(text, None, prefix, None, tuple(issues))
    return Accession(text, f"{prefix}.{year}.{int(num)}{sub}", prefix, y, tuple(issues))


def decode_prefix(prefix: str | None) -> tuple[str | None, str | None, str | None]:
    """Return (ca_type, gender, age_group) for a prefix, or Nones if unknown."""
    return PREFIX_MAP.get(prefix or "", (None, None, None))
