"""Subset label (CS, IL, SF, OTHER) of every IFEval instruction, in the order of follow_flags.

Usage: ifeval_subsets.py  (writes data/ifeval_subsets.json)
"""
from __future__ import annotations

import argparse
import json

from ipidefense.evaluation.ifeval import load_prompts
from ipidefense.paths import DATA
from ipidefense.utils import write_json

DEFINITION = (
    "  CS  completion/length-sensitive (228): number_words 'at least', number_sentences "
    "'at least',\n      number_paragraphs, nth_paragraph_first_word, multiple_sections, "
    "postscript, end_checker,\n      quotation, two_responses\n"
    "  IL  resembles training injections (188): change_case:*, constrained_response, "
    "json_format,\n      repeat_prompt, response_language\n"
    "  SF  short-favouring (46): number_words / number_sentences 'less than'\n"
    "  OTHER everything else (372)")
EXPECTED = {"CS": 228, "IL": 188, "SF": 46, "OTHER": 372}
CS_ANY = {"length_constraints:number_paragraphs", "length_constraints:nth_paragraph_first_word",
          "detectable_format:multiple_sections", "detectable_content:postscript",
          "startend:end_checker", "startend:quotation", "combination:two_responses"}
RELATIONAL = {"length_constraints:number_words", "length_constraints:number_sentences"}
IL_ANY = {"detectable_format:constrained_response", "detectable_format:json_format",
          "combination:repeat_prompt", "language:response_language"}
OUT = DATA / "ifeval_subsets.json"


def label(iid: str, kwargs: dict | None) -> str:
    rel = (kwargs or {}).get("relation")
    if iid in RELATIONAL:
        if rel == "at least":
            return "CS"
        if rel == "less than":
            return "SF"
        raise ValueError(f"{iid}: unexpected relation {rel!r}")
    if iid in CS_ANY:
        return "CS"
    if iid.startswith("change_case:") or iid in IL_ANY:
        return "IL"
    return "OTHER"


def build() -> dict:
    """One label per prompt and instruction; refuses if a subset count differs."""
    rows = load_prompts(0)
    labels = []
    for i, row in enumerate(rows):
        kws = row.get("kwargs") or [{}] * len(row["instruction_id_list"])
        labels.append({"idx": i, "key": row.get("key"),
                       "subsets": [label(iid, kw) for iid, kw in
                                   zip(row["instruction_id_list"], kws, strict=True)],
                       "instruction_id_list": row["instruction_id_list"]})
    counts = {k: sum(s == k for r in labels for s in r["subsets"]) for k in EXPECTED}
    if counts != EXPECTED:
        raise SystemExit(f"subset counts {counts} != expected {EXPECTED}")
    return {"source": "data/ifeval_input.jsonl via ipidefense.evaluation.ifeval.load_prompts",
            "definition": DEFINITION, "counts": counts,
            "prompts_with": {k: sum(k in r["subsets"] for r in labels) for k in EXPECTED},
            "labels": labels}


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    res = build()
    write_json(OUT, res)
    print(json.dumps({"counts": res["counts"], "prompts_with": res["prompts_with"]}))


if __name__ == "__main__":
    main()
