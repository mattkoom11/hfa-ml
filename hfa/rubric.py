"""The Phase 1 labeling rubric as text patterns (decisions dated 2026-10-11).

Shared by the label checker (hfa.label_audit) and the keyword baseline
(hfa.attr_keyword), so both read text the same way the labeler does.
Every alias here was decided from dev-set garments only; never tune on test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"
FIELDS = ("material", "color", "technique")

# Technique codes whose text form differs from the list label.
TECHNIQUE_CUES = {
    "printing": r"\bprint(?:ed|s)?\b",
    "pleating": r"\bpleat(?:ed|s|ing)?\b",
    "knitting": r"\bknit(?:ted|ting)?\b",
    "embroidery": r"\bembroider(?:ed|y)\b",
    "beading": r"\bbead(?:ed|ing|work|s)\b",
    "quilting": r"\bquilt(?:ed|ing)\b",
    "crocheting": r"\bcrochet(?:ed|ing)?\b",
    "applique": r"\bappliqu[eé]d?\b",
    "painting": r"\bpaint(?:ed|ing)?\b",
    # Weaving only when the weave is a visible feature, not "plain woven lining".
    "weaving": r"\bwoven\s+(?:stripe|pattern|design|check|plaid|motif|band|border)s?\b|\bwoven[- ]in\b(?<!beads woven in)",
    "braiding": r"\bbraid(?:ed|ing|s)?\b",
}
# Spellings and phrases the rubric treats as a list word.
ALIASES = {
    ("color", "gray"): r"\bgr[ae]y\b",
    ("material", "velvet"): r"\bvelvet(?:een)?\b",  # velveteen counts, like crepe de chine -> crepe
    ("material", "satin"): r"\bsat(?:in|een)s?\b",  # sateen -> satin, same reasoning
    ("color", "turquoise"): r"\bturquoise?\b",  # catalog typo "turquois"
    ("material", "lace"): r"\blace(?!\s*(?:-|\s)?(?:up|over|through)\b)s?\b",  # "lace up boots" is a verb
}
# Two-word colors take the more specific word ("navy blue" -> navy, "rust brown" -> rust).
SUPPRESS = {
    ("color", "blue"): r"\bnavy blue\b",
    ("color", "brown"): r"\brust brown\b",
}
HARDWARE = re.compile(
    r"\b(metal|brass|silver|glass|plastic|celluloid|bakelite|bone|shell|pearl|mother.of.pearl|wood(?:en)?|"
    r"leather|rhinestone|jet|steel|copper)\s+(?:\w+\s+)?(button|zipper|zip|buckle|snap|hook|clasp|frame|stud|"
    r"closure|chain|handle|clip)s?\b",
    re.IGNORECASE,
)


def load_vocab() -> dict:
    return json.loads(VOCAB_PATH.read_text())["lists"]


def cue(field: str, item: dict) -> str:
    if (field, item["code"]) in ALIASES:
        return ALIASES[(field, item["code"])]
    if field == "technique" and item["code"] in TECHNIQUE_CUES:
        return TECHNIQUE_CUES[item["code"]]
    return r"\b" + re.escape(item["label"].lower()) + r"s?\b"


def mentions(field: str, item: dict, text: str) -> bool:
    """Does lowercased `text` mention this list value, by the rubric's reading?"""
    pattern = cue(field, item)
    if (field, item["code"]) in SUPPRESS:
        text = re.sub(SUPPRESS[(field, item["code"])], "", text)
    return re.search(pattern, text) is not None
