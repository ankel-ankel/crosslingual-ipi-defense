import re
import sys
from collections import Counter
from multiprocessing import Pool

import pandas as pd
from common import TRAIN_WITNESS, adapt, lang_of, quality_reason, residual

TARGET = re.compile(r"I\s+have\s+been\s+PWNED", re.I)
RESID_TARGET = re.compile(r"p\W{0,3}w\W{0,3}n|he sido hackead|fui hackead|me han hackead"
                          r"|hve bin|i\s+have\s+been", re.I)
# game-specific references per HackAPrompt level
GAME = {
    "L0_space": r"space workers?|from space|positive things|\bin space\b|working in space"
                r"|space working|spaceship|outer space",
    "L1_translation": r"if you return|영어|번역|英語|翻訳|入力が日本語|tiếng anh|dịch"
                      r"|ingl[eé]s|anglais|englisch|inglese|английск|英语|英文"
                      r"|if the input is|翻译|翻譯|translat|spanish|espa[ñn]ol|gracia|glacia"
                      r"|traduc|traduz|übersetz|tradui",
    "L2_key": r"secret key|\bkey\b|clave secreta|密钥",
    "L3_fewshot": r"biblioteca|este libro|ipad|negro\b|donde esta",
    "L4_search": r"search engine|kermode|spirit bear|ursus|i don'?t know|gribbell|\bbears?\b"
                 r"|source \d|\bengine\b",
    "L5_grammar": r"gramm[ae]r|grammatic|feedback|spelling|stylistic",
    "L5_empty_input": r"if (?:the )?(?:text|input|sentence|sentec\w*) (?:is )?(?:empty|none|blank)"
                      r"|if it'?s empty"
                      r"|if (?:there is )?no (?:text|input|sentence|sentec\w*|errors?)|if none\b"
                      r"|if empty|if correct"
                      r"|no input|empty respond|\bbrackets?\b|square bracket|list of characters"
                      r"|if (?:there are )?no mistakes?|if not good"
                      r"|if (?:the )?text is (?:good|correct|fine)"
                      r"|if input'?s empty|if no\s*[,:]?\s*(?:say|write)|if right\b|if this is bad",
    "L6_story": r"stor(?:y|ies)|second person|second prompt|once upon a time",
    "target_length": r"first (?:\d+|two|three|four) words|\b(?:four|4) words\b"
                     r"|\b(?:17|seventeen) characters\b",
    "L7_movie": r"movie|\btitle",
    "L8_sigma": r"sigma|ucar|storyteller|townspeople|villager|yool|input\?>|<\?/?input",
    "L9_slashes": r"slash|bad idea",
    "L10_emoji": r"emoji",
    "meta": r"hackaprompt|\blevel\b|competition|prompt hacking",
}
GAME_RE = {k: re.compile(v, re.I) for k, v in GAME.items()}


def game_hit(resid: str) -> str:
    return ",".join(k for k, rx in GAME_RE.items() if rx.search(resid))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cols = ["level", "user_input", "model", "expected_completion", "correct", "dataset",
            "session_id", "timestamp"]
    df = pd.read_parquet(sys.argv[1], columns=cols)
    df["row"] = range(len(df))
    stages = Counter({"rows_total": len(df)})
    ok = df[df.correct & (df.expected_completion == "I have been PWNED")].copy()
    stages["successful_pwned_rows"] = len(ok)
    ok["pref"] = (ok.dataset != "submission_data").astype(int)
    ok = ok.sort_values(["pref", "row"])
    g = ok.groupby("user_input", sort=False).agg(
        source_row_id=("row", "first"), level=("level", "first"), model=("model", "first"),
        dataset=("dataset", "first"), session_id=("session_id", "first"),
        n_rows=("row", "size"), n_sessions=("session_id", "nunique"),
        levels=("level", lambda x: ",".join(map(str, sorted(set(x))))),
        models=("model", lambda x: ",".join(sorted(set(x))))).reset_index()
    stages["unique_successful_texts"] = len(g)

    recs = []
    for r in g.itertuples(index=False):
        t = r.user_input
        reason = quality_reason(t)
        a = adapt(t, TARGET) if not reason else {"ok": False, "reason": ""}
        if not reason and not a["ok"]:
            if a["n_match"] == 0:
                reason = "no_exact_target"
            elif a["n_match"] > 1:
                reason = "multiple_targets"
            else:
                reason = a["reason"]
        resid = residual(t, TARGET)
        if not reason and RESID_TARGET.search(resid):
            reason = "residual_target_fragment"
        if not reason and TRAIN_WITNESS.search(resid):
            reason = "training_witness_word"
        gh = game_hit(resid)
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
    L = pd.DataFrame(lid, index=out.index[lidmask])
    out = out.join(L)
    for k, v in out["drop_reason"].value_counts().items():
        stages[f"drop:{k or 'KEPT'}"] = int(v)
    print(dict(stages))
    out.drop(columns=["resid"]).to_parquet(sys.argv[2])
    k = out[keep]
    print("kept by LID top lang:", k.lang.value_counts().head(25).to_dict())
    print("kept method=script:", k[k.method == "script"].lang.value_counts().to_dict())


if __name__ == "__main__":
    main()
