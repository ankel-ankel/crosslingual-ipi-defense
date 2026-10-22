import argparse
import json

from ipidefense.paths import RUNS

MT = RUNS / "suite" / "human_mt" / "summary.json"
EN = RUNS / "suite" / "human" / "summary.json"
LANGS = ("es", "vi", "fr", "de", "zh", "ko", "bn")
NAMES = {"es": "Spanish", "vi": "Vietnamese", "fr": "French", "de": "German", "zh": "Chinese",
         "ko": "Korean", "bn": "Bengali", "en": "English (not translated)"}
DIALOGUE, PREFIX = "hen-61e9844dcdf5", "hen-9cc5b1354799"


def _num(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def _rate(x: float) -> str:
    return f"${x:.3f}$"


def _ub(x: float) -> str:
    return f"${x:.4f}$"


def cell(summary: dict, fam: str, lang: str) -> dict:
    pooled = summary["pooled"][f"{fam}|{lang}"]
    block = pooled.get("qa_ok_only", pooled["all"])["wf_obey"]
    per_t = pooled["per_template"]
    out = {"k": block["k"], "n": block["n"], "rate": block["rate"], "ub": block["cp95_upper"],
           "max_template": max(v["wf"]["rate"] for v in per_t.values() if v.get("qa_ok", True)),
           "dialogue": per_t[DIALOGUE]["wf"], "prefix": per_t[PREFIX]["wf"]}
    if fam != "undef":
        claim = summary["decision"]["claims"][f"{fam}|{lang}|human"]
        out["pass"], out["failing"] = claim["pass"], claim["failing_templates"]
    return out


# one row per language: undef obey, then def and hard stats
def row(summary: dict, lang: str) -> tuple[str, dict]:
    u, d, h = (cell(summary, f, lang) for f in ("undef", "def", "hard"))
    tex = " & ".join([
        NAMES[lang], _rate(u["rate"]),
        f"{_num(d['k'])}/{_num(d['n'])}", _ub(d["ub"]), _rate(d["max_template"]),
        f"{d['dialogue']['k']}/{d['dialogue']['n']}", f"{d['prefix']['k']}/{d['prefix']['n']}",
        f"{_num(h['k'])}/{_num(h['n'])}", _ub(h["ub"]), _rate(h["max_template"]),
        f"{h['dialogue']['k']}/{h['dialogue']['n']}", f"{h['prefix']['k']}/{h['prefix']['n']}",
    ]) + r"\\"
    return tex, {"undef": u, "def": d, "hard": h}


def main() -> None:
    argparse.ArgumentParser().parse_args()
    mt = json.loads(MT.read_text(encoding="utf-8"))
    langs = [lang for lang in LANGS if f"def|{lang}|human" in mt["decision"]["claims"]]
    digest = {"langs": langs, "lands": mt["decision"]["lands"],
              "exact_agrees": all(v["agrees"] for v in mt["human_mt"]["exact_check"].values())}
    for lang in langs:
        tex, nums = row(mt, lang)
        print(tex)
        digest[lang] = nums
    en = json.loads(EN.read_text(encoding="utf-8"))
    tex, nums = row(en, "en")
    print("% English reference row (not translated):")
    print(tex)
    digest["en"] = nums
    print(json.dumps(digest, indent=1))


if __name__ == "__main__":
    main()
