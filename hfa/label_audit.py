"""Consistency check for Phase 1 labels against the rubric (hfa.rubric).

Flags, per labeled garment:
  - a list word in the text that wasn't labeled
  - a label whose word doesn't appear in the text
  - hardware named in the text ("pearl buttons") with no material recorded for it
These are prompts to re-check, not verdicts: "gold trim" can be a color.

Dev labels only. The checker is keyword-based, so fixing test labels with it
would bend the test set toward the keyword baseline. Test labels are cleaned
at the end by adjudicating disagreements from every method, blind to which
method disagreed.

Run:  .venv/bin/python -m hfa.label_audit
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from hfa.rubric import FIELDS, HARDWARE, load_vocab, mentions

ROOT = Path(__file__).resolve().parent.parent


def audit(labels: dict, texts: pd.Series, vocab: dict) -> pd.DataFrame:
    rows = []
    for acc, rec in labels.items():
        if rec.get("status") != "done":
            continue
        text = texts[acc].lower()
        for f in FIELDS:
            for item in vocab[f]:
                present = mentions(f, item, text)
                labeled = item["code"] in rec[f]
                if present and not labeled:
                    rows.append((acc, f, item["code"], "in text, not labeled"))
                elif labeled and not present:
                    rows.append((acc, f, item["code"], "labeled, not in text"))
        other = rec.get("other_material", "").lower()
        for m in HARDWARE.finditer(text):
            mat = m.group(1).lower().replace("wooden", "wood")
            if mat not in rec["material"] and mat.split()[0] not in other:
                rows.append((acc, "material", mat, f"hardware '{m.group(0)}' not recorded"))
    return pd.DataFrame(rows, columns=["accession", "field", "code", "flag"])


def main() -> None:
    labels = json.loads((ROOT / "data/labels/labels.json").read_text())
    roles = ROOT / "data/labels/eval_roles.csv"
    if roles.exists():
        dev = set(pd.read_csv(roles)["accession"])
        hidden = len(set(labels) - dev)
        labels = {a: r for a, r in labels.items() if a in dev}
        print(f"auditing dev labels only; {hidden} test labels are not checked on purpose")
    s = pd.read_csv(ROOT / "data/labels/label_set.csv", dtype=str).set_index("accession")
    status = Counter(r["status"] for r in labels.values())
    print(f"labeled {len(labels)}/{len(s)}  {dict(status)}")

    flags = audit(labels, s["description"], load_vocab())
    done = status.get("done", 0)
    print(f"flags: {len(flags)} across {flags['accession'].nunique() if len(flags) else 0} garments "
          f"({len(flags) / max(done, 1):.2f} per garment)")
    if len(flags):
        print(flags.groupby(["field", "flag"]).size().to_string())
        print()
        for _, r in flags.iterrows():
            print(f"  {r.accession:14} {r.field:9} {r.code:12} {r.flag}")
            print(f"      {s.loc[r.accession, 'description'][:150]}")


if __name__ == "__main__":
    main()
