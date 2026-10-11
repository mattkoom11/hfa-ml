"""Phase 5: grounded answers over search results, with deterministic faithfulness checks.

For each query, the local LLM (Ollama, qwen3:8b) gets only the hybrid search's top
garments as numbered records and writes a short answer that cites accession
numbers in [brackets]. The answer is then checked without another LLM:
  invalid_citations  cited accession numbers that weren't retrieved (should be none)
  sentence_coverage  share of sentences that cite at least one garment
  unsupported        colors / materials / techniques named in a sentence that none of
                     that sentence's cited garments has in its record (hfa.rubric reading)

Output: data/search/answers.json
Run:    .venv/bin/python -m hfa.rag [--k 5] [--regenerate]
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

from hfa.rubric import FIELDS, load_vocab, mentions

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/search/answers.json"
OLLAMA = "http://127.0.0.1:11434/api/generate"
LLM = "qwen3:8b"
CITE = re.compile(r"\[([A-Z]{2,5}\.\d{4}\.\d+(?:\.\d+)*)\]")


def record_text(a: str, row: pd.Series) -> str:
    desc = " ".join(str(row["description"] or "").split()[:60])
    return f"[{a}] {row['title']}; type {row['ca_type']}; date {row['date_raw'] or 'unknown'}; {desc or 'no description'}"


def generate(query: str, records: list[str]) -> str:
    prompt = (
        "You answer a visitor's question about a historic clothing collection using ONLY the records below.\n"
        "Write 2-4 short sentences. After every claim about a garment, cite it with its accession number in "
        "square brackets, exactly as given, e.g. [DR.2023.17]. Mention only colors, materials and features that "
        "the cited record states. If the records don't match the question well, say so.\n\n"
        "Records:\n" + "\n".join(records) + f"\n\nQuestion: {query}\nAnswer:"
    )
    body = json.dumps({"model": LLM, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0, "num_predict": 300}}).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"].strip()


def check(answer: str, retrieved: list[str], records: dict[str, str], vocab: dict) -> dict:
    """Deterministic faithfulness checks; `records` maps accession -> its record text."""
    cited = CITE.findall(answer)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    unsupported = []
    for s in sentences:
        sent_cites = [a for a in CITE.findall(s) if a in records]
        if not sent_cites:
            continue
        text = CITE.sub("", s).lower()
        support = " ".join(records[a].lower() for a in sent_cites)
        for f in FIELDS:
            for item in vocab[f]:
                if mentions(f, item, text) and not mentions(f, item, support):
                    unsupported.append(f"{f}:{item['code']}")
    return {
        "cited": sorted(set(cited)),
        "invalid_citations": sorted({a for a in cited if a not in retrieved}),
        "sentence_coverage": sum(bool(CITE.search(s)) for s in sentences) / max(len(sentences), 1),
        "unsupported": unsupported,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--regenerate", action="store_true")
    args = ap.parse_args()

    queries = yaml.safe_load((ROOT / "eval/search_queries.yaml").read_text())
    hybrid = json.loads((ROOT / "data/predictions/search_hybrid.json").read_text())
    g = pd.read_parquet(ROOT / "data/processed/garments.parquet").set_index("accession")
    g = g.astype(object).where(g.notna(), None)
    vocab = load_vocab()
    out = {} if args.regenerate or not OUT.exists() else json.loads(OUT.read_text())

    for q in queries:
        retrieved = hybrid[q["id"]][: args.k]
        records = {a: record_text(a, g.loc[a]) for a in retrieved}
        if q["id"] not in out or out[q["id"]].get("retrieved") != retrieved:
            out[q["id"]] = {"query": q["text"], "retrieved": retrieved, "answer": generate(q["text"], list(records.values()))}
        out[q["id"]]["checks"] = check(out[q["id"]]["answer"], retrieved, records, vocab)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
        ch = out[q["id"]]["checks"]
        print(f"  {q['id']} cites {len(ch['cited'])}, invalid {ch['invalid_citations']}, "
              f"coverage {ch['sentence_coverage']:.2f}, unsupported {ch['unsupported']}", flush=True)

    vals = [out[q["id"]]["checks"] for q in queries]
    n = len(vals)
    print(f"\n{n} answers: invalid citations in {sum(bool(v['invalid_citations']) for v in vals)}; "
          f"mean sentence coverage {sum(v['sentence_coverage'] for v in vals) / n:.2f}; "
          f"answers with an unsupported attribute: {sum(bool(v['unsupported']) for v in vals)}")


if __name__ == "__main__":
    main()
