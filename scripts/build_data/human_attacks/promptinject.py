import re
from pathlib import Path

from common import PLACEHOLDER, esc, lang_of

PI_SOURCE = "agencyenterprise/PromptInject@2928a71:promptinject/prompt_data.py"
PI_TOKEN = "{REPLACE_rogue_string}"
_ENTRY = re.compile(r'"([\w-]+)":\s*\{\s*"label":\s*"([^"]*)",\s*"instruction":\s*'
                    r"\'([^\']*)\'", re.S)


def promptinject_records(path: Path) -> list[dict]:
    src = path.read_text(encoding="utf-8")
    start = src.index("goal_hikacking_attacks = {")
    block = src[start: src.index("\n}\n", start) + 3]
    recs = []
    for key, label, instr in _ENTRY.findall(block):
        if instr.count(PI_TOKEN) != 1:
            raise ValueError(f"unexpected placeholder count in {key}")
        head, tail = instr.split(PI_TOKEN)
        adapted = esc(head) + PLACEHOLDER + esc(tail)
        lid = lang_of(head + " " + tail)
        recs.append({
            "id": f"pinj-{key}", "source": PI_SOURCE,
            "source_row_id": f"goal_hikacking_attacks/{key}",
            "lang": "en", "langdetect_prob": round(float(lid["prob"]), 3), "original_text": instr,
            "adapted_template": adapted, "target_phrase_replaced": PI_TOKEN,
            "notes": f"PromptInject goal-hijacking attack '{label}' (Perez & Ribeiro 2022); "
                     "written by the paper authors (not crowd-sourced); no per-item success "
                     "record; the source's own rogue-string placeholder was renamed to {witness}",
            "meta": {"label": label, "langdetect_top": lid["lang"], "author_written": True},
        })
    if len(recs) != 5:
        raise ValueError(f"expected 5 PromptInject attacks, found {len(recs)}")
    return recs
