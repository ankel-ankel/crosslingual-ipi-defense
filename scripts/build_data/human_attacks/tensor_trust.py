import bz2
import json
import re
import sys
from collections import Counter
from multiprocessing import Pool

import pandas as pd
from common import TRAIN_WITNESS, adapt, lang_of, quality_reason, residual

TARGET = re.compile(r"access\s+granted", re.I)
RESID_TARGET = re.compile(r"access\W{0,5}gr?ant|\bgranted\b|acceso concedido|accès autorisé"
                          r"|zugriff gewährt|доступ предоставлен|аццесс|授予访问|访问权限"
                          r"|quyền truy cập|toegang|zugang|\bacceso\b|\baccès\b", re.I)
# references to the game's defense prompt
GAME = {
    "password": r"pass\s?word|passcode|pass code|passphrase|access code"
                r"|secret (?:word|code|phrase|key)"
                r"|correct code|\bsecret\b|\bpwd\b|contraseña|mot de passe|passwort|senha"
                r"|пароль|код доступа|密码|密碼|访问代码|口令",
    "denied": r"denied|\bdeny\b|拒绝|отказ",
    "game": r"bancaire|bancario|банк|защит|银行|tensor ?trust|\bbank\b|account balance"
            r"|\bdefen[cs]e\b|\bdefender\b|\battacker\b"
            r"|\baccess\b|\bvault\b|\bdoor\b|let me in|\b(?:un)?authori[sz]ed\b|kermode"
            r"|spirit bear|search engine",
    "target_length": r"\b(?:14|fourteen) characters\b|\brevoked\b",
}
GAME_RE = {k: re.compile(v, re.I) for k, v in GAME.items()}


def as_bool(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    st = Counter()
    rows = []
    with bz2.open(sys.argv[1], "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            st["attacks_total"] += 1
            if not as_bool(r.get("output_is_access_granted")):
                continue
            st["granted"] += 1
            if as_bool(r.get("is_self_attack")):
                continue
            a = r.get("attacker_input") or ""
            if not a.strip():
                continue
            st["granted_nonself"] += 1
            code = (r.get("access_code") or "").strip()
            rows.append({"source_row_id": str(r["attack_id"]), "text": a,
                         "contains_code": len(code) >= 3 and code.casefold() in a.casefold(),
                         "llm": r.get("llm_choice") or "GPT_35_TURBO_0613(v1-era)",
                         "attacker": r.get("attacker_id_anonymized"),
                         "timestamp": r.get("timestamp")})
    df = pd.DataFrame(rows).sort_values("timestamp", kind="stable")
    g = df.groupby("text", sort=False).agg(
        source_row_id=("source_row_id", "first"), llm=("llm", "first"),
        attacker=("attacker", "first"),
        timestamp=("timestamp", "first"), contains_code=("contains_code", "max"),
        n_rows=("source_row_id", "size"), n_attackers=("attacker", "nunique"),
        llms=("llm", lambda x: ",".join(sorted(set(x))))).reset_index()
    st["unique_texts"] = len(g)
    recs = []
    for r in g.itertuples(index=False):
        t = r.text
        reason = "contains_access_code" if r.contains_code else quality_reason(t)
        a = adapt(t, TARGET) if not reason else {"ok": False, "n_match": -1}
        if not reason and not a["ok"]:
            reason = ("no_exact_target" if a["n_match"] == 0 else
                      "multiple_targets" if a["n_match"] > 1 else a["reason"])
        resid = residual(t, TARGET)
        if not reason and RESID_TARGET.search(resid):
            reason = "residual_target_fragment"
        if not reason and TRAIN_WITNESS.search(resid):
            reason = "training_witness_word"
        gh = ",".join(k for k, rx in GAME_RE.items() if rx.search(resid))
        if not reason and gh:
            reason = "game_specific"
        if not reason and len(re.sub(r"[\W\d_]+", "", resid)) < 4:
            reason = "no_instruction_text"
        recs.append({**r._asdict(), "drop_reason": reason, "game_hits": gh,
                     "adapted": a.get("adapted"),
                     "matched": a.get("matched"),
                     "braces_escaped": a.get("braces_escaped", False),
                     "resid": resid})
    out = pd.DataFrame(recs)
    keep = out["drop_reason"] == ""
    lidmask = out["drop_reason"].isin(["", "game_specific"])
    with Pool(14) as pool:
        lid = pool.map(lang_of, out.loc[lidmask, "resid"].tolist(), chunksize=200)
    out = out.join(pd.DataFrame(lid, index=out.index[lidmask]))
    for k, v in out["drop_reason"].value_counts().items():
        st[f"drop:{k or 'KEPT'}"] = int(v)
    print(dict(st))
    out.drop(columns=["resid"]).to_parquet(sys.argv[2])
    k = out[keep]
    print("kept LID:", k.lang.value_counts().head(20).to_dict())
    print("kept script:", k[k.method == "script"].lang.value_counts().to_dict())
    print("kept llm:", k.llm.value_counts().to_dict(), "| attackers:", k.attacker.nunique())


if __name__ == "__main__":
    main()
