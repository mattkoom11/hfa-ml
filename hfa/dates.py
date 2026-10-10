"""Parse the collection's free-text Date column into a year range.

Values look like "Mid 20thc, 1960-1980", "18thc", "Late 19thc Early 20thc,
1890-1910" or "First Half 19thc", with typos such as "Eartly". When a numeric
range is present it wins; otherwise the century phrase is translated.
The original text is always kept for CA's display field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TYPO_FIXES = {
    r"\beartly\b": "early",
    r"\bealry\b": "early",
    r"(\d{2})stc\b": r"\1thc",
    r"(\d{2})th\b(?!c)": r"\1thc",  # "Late 19th Early 20thc" -> "Late 19thc Early 20thc"
    r"(\d{2})th c\b": r"\1thc",
}


@dataclass(frozen=True)
class DateRange:
    raw: str
    earliest: int | None
    latest: int | None
    method: str  # "numeric", "century", "century_whole", "blank" or "unparsed"


def clean_phrase(text: str) -> str:
    s = text.lower().strip()
    for pat, rep in TYPO_FIXES.items():
        s = re.sub(pat, rep, s)
    return re.sub(r"\s+", " ", s)


def numeric_range(text: str) -> tuple[int, int] | None:
    """Pull an explicit year or year range out of the text, if there is one."""
    m = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)\s*[-–]\s*(1[5-9]\d{2}|20[0-2]\d)\b", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    m = re.search(r"\b(1[5-9]\d{2}|20[0-2]\d)(?:s|c)?\b", text)  # "1940c" = circa 1940
    if m:
        y = int(m.group(1))
        return (y, y)
    return None


# Offsets into a century (1800-1899 for "19thc"). Late = last quarter (Matthew,
# 2026-10-10); early and mid follow from that so the parts tile the century.
_SPANS = {
    "early": (0, 24),
    "mid": (25, 74),
    "late": (75, 99),
    "early-mid": (0, 74),
    "mid-late": (25, 99),
    "first half": (0, 49),
    "second half": (50, 99),
    "": (0, 99),
}
_SEGMENT = re.compile(
    r"(early-mid|mid-late|early|mid|late|first half|second half)?(?: of)?\s*(\d{2})thc"
)


def century_phrase_to_range(phrase: str) -> tuple[int, int] | None:
    """Translate a cleaned century phrase into (earliest, latest) years.

    "late 19thc" -> (1875, 1899); "late 19thc early 20thc" -> (1875, 1924);
    "20thc" -> (1900, 1999). Anything without a century ("wwi", "?") -> None.
    """
    segments = _SEGMENT.findall(phrase)
    if not segments:
        return None
    lows, highs = [], []
    for qualifier, century in segments:
        base = (int(century) - 1) * 100
        lo, hi = _SPANS[qualifier]
        lows.append(base + lo)
        highs.append(base + hi)
    return min(lows), max(highs)


def parse_date(raw: object) -> DateRange:
    text = "" if raw is None else str(raw).strip()
    if not text or text.lower() == "nan":
        return DateRange(text, None, None, "blank")
    num = numeric_range(text)
    if num:
        return DateRange(text, num[0], num[1], "numeric")
    cen = century_phrase_to_range(clean_phrase(text))
    if cen:
        # A bare "20thc" is stored for CA but too vague to train decade labels on.
        method = "century_whole" if cen[1] - cen[0] >= 99 else "century"
        return DateRange(text, cen[0], cen[1], method)
    return DateRange(text, None, None, "unparsed")
