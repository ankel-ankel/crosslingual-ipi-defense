import json

from ipidefense.paths import DATA
from ipidefense.third_party.ifeval import instructions_registry as reg

PROMPTS = DATA / "ifeval_input.jsonl"


def load_prompts(n: int) -> list[dict]:
    rows = [
        json.loads(line)
        for line in PROMPTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:n] if n else rows


def follow_flags(prompt_row: dict, response: str) -> list[bool]:
    flags = []
    kwargs_list = prompt_row.get("kwargs") or [{}] * len(prompt_row["instruction_id_list"])
    for iid, kw in zip(prompt_row["instruction_id_list"], kwargs_list, strict=False):
        inst = reg.INSTRUCTION_DICT[iid](iid)
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        inst.build_description(**kw)
        iargs = inst.get_instruction_args()
        if iargs and "prompt" in iargs:
            inst.build_description(prompt=prompt_row["prompt"])
        # an empty response follows nothing
        flags.append(bool(response.strip()) and bool(inst.check_following(response)))
    return flags
