"""Assemble pool_<lang>.jsonl from the HackAPrompt / Tensor Trust candidate tables.

Usage: python scripts/build_data/human_attacks/assemble.py CAND_DIR OUT_DIR
       PROMPTINJECT_PROMPT_DATA_PY
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from common import CHAT_TOKENS, MAX_LEN, MIN_LEN, PLACEHOLDER, WORD_COUNT, norm_key
from decisions import EXCLUDE_EN, NON_EN
from promptinject import promptinject_records
from sklearn.feature_extraction.text import TfidfVectorizer

SRC = {"HP": ("hap", "hackaprompt/hackaprompt-dataset@25b87fbedfb86840abaf8cd09af7a029208a971a"),
       "TT": ("tt", "qxcv/tensor-trust@4de2b2fe01ba0cb6fbf7cbb9f1a3fabaf8157372:raw-data/v2")}
NEAR_DUP, DIVERSITY = 0.80, 0.60
QUOTA = {"HP": 110, "TT": 90}
SEED = 0


def load(cand_dir: Path) -> pd.DataFrame:
    hp = pd.read_parquet(cand_dir / "hp_cand.parquet").rename(columns={"user_input": "text"})
    hp["src"], hp["players"] = "HP", hp.n_rows
    # session_id is not a player id (17 values)
    hp["player"] = "hp-row-" + hp.source_row_id.astype(str)
    hp["stratum"] = "L" + hp.level.astype(str) + "|" + hp.model
    tt = pd.read_parquet(cand_dir / "tt_cand.parquet")
    tt["src"], tt["players"], tt["player"], tt["stratum"] = ("TT", tt.n_attackers, tt.attacker,
                                                            tt.llm)
    df = pd.concat([hp, tt], ignore_index=True)
    df["source_row_id"] = df.source_row_id.astype(str)
    return df


def notes_meta(r) -> tuple[str, dict]:
    if r.src == "HP":
        meta = {"level": int(r.level), "model": r.model, "dataset": r.dataset,
                "session_id": r.session_id,
                "n_rows": int(r.n_rows), "levels": r.levels, "models": r.models}
        note = (f"HackAPrompt level {r.level}; succeeded on {r.models}; "
                f"submitted {r.n_rows} time(s)")
    else:
        meta = {"llm": r.llm, "attacker": r.attacker, "timestamp": r.timestamp,
                "n_rows": int(r.n_rows),
                "n_attackers": int(r.n_attackers), "llms": r.llms}
        note = f"Tensor Trust hijack; succeeded on {r.llms}; {r.n_attackers} attacker(s)"
    meta.update({"langdetect_top": r.lang, "langdetect_top_prob": float(r.prob),
                 "en_function_word_ratio": float(r.en_ratio),
                 "braces_escaped": bool(r.braces_escaped)})
    if r.braces_escaped:
        note += "; literal braces doubled for str.format"
    if WORD_COUNT.search(r.adapted):
        note += "; mentions a fixed word count (the original target had several words)"
    return note, meta


def record(r, lang: str, extra: str = "") -> dict:
    note, meta = notes_meta(r)
    if extra:
        note += "; " + extra
    pre, src = SRC[r.src]
    return {"id": f"{pre}-{r.source_row_id}", "source": src, "source_row_id": r.source_row_id,
            "lang": lang,
            "langdetect_prob": round(float(r.prob), 3), "original_text": r.text,
            "adapted_template": r.adapted,
            "target_phrase_replaced": r.matched, "notes": note, "meta": meta}


def validate(rec: dict) -> None:
    t, o = rec["adapted_template"], rec["original_text"]
    assert t.count(PLACEHOLDER) == 1, rec["id"]
    assert t.format(witness=rec["target_phrase_replaced"]) == o, rec["id"]
    assert "ZQXJW" in t.format(witness="ZQXJW"), rec["id"]
    assert MIN_LEN <= len(o) <= MAX_LEN and not CHAT_TOKENS.search(o), rec["id"]


def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Exact (normalised) then greedy near-duplicate removal; the kept rows and their TF-IDF
    matrix."""
    df = df.assign(key=df.adapted.map(norm_key)).sort_values(
        ["src", "players", "source_row_id"], ascending=[True, False, True], kind="stable")
    df = df.drop_duplicates("key").reset_index(drop=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform(df.adapted.str.replace(PLACEHOLDER, " ", regex=False))
    kept, blocked = [], set()
    for i0 in range(0, X.shape[0], 1000):
        S = (X[i0:i0 + 1000] @ X.T).toarray()
        for j in range(S.shape[0]):
            i = i0 + j
            if i in blocked:
                continue
            kept.append(i)
            blocked |= {int(c) for c in np.nonzero(S[j] >= NEAR_DUP)[0] if c > i}
    return df.iloc[kept].reset_index(drop=True), X[kept]


def select_en(df: pd.DataFrame, X) -> list[int]:
    """Stratified pick: per source quota, round-robin over strata x length tercile, one text per
    player."""
    chosen: list[int] = []
    for src, quota in QUOTA.items():
        idx = [i for i in range(len(df)) if df.src[i] == src]
        lens = df.loc[idx, "text"].str.len()
        terc = pd.cut(lens, [0, 80, 200, 10**6], labels=["short", "mid", "long"]).astype(str)
        buckets = defaultdict(list)
        for i in idx:
            buckets[(df.stratum[i], terc[i])].append(i)
        prio = lambda i: hashlib.sha1(  # noqa: E731
            f"{SEED}:{df.src[i]}:{df.source_row_id[i]}".encode()).hexdigest()
        for b in buckets.values():
            b.sort(key=prio, reverse=True)  # pop() takes the smallest hash first
        keys = sorted(buckets)
        used_players, got = set(), 0
        while got < quota and any(buckets.values()):
            for k in keys:
                if got >= quota:
                    break
                while buckets[k]:
                    i = buckets[k].pop()
                    if df.player[i] in used_players:
                        continue
                    if chosen and (X[i] @ X[chosen].T).toarray().max() >= DIVERSITY:
                        continue
                    chosen.append(i)
                    used_players.add(df.player[i])
                    got += 1
                    break
    return chosen


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cand, out = Path(sys.argv[1]), Path(sys.argv[2])
    df = load(cand)
    ok = df[df.drop_reason == ""]
    # English
    en = ok[(ok.lang == "en") & (ok.prob >= 0.90)]
    en = en[~(en.src + ":" + en.source_row_id).isin(EXCLUDE_EN)]
    en_d, Xen = dedup(en)
    print("EN candidates:", en.groupby("src").size().to_dict(), "-> after exact+near dedup:",
          en_d.groupby("src").size().to_dict())
    full = [record(r, "en") for r in en_d.itertuples()]
    sel = select_en(en_d, Xen)
    pools = defaultdict(list)
    pools["en"] = [full[i] for i in sel]
    pinj = promptinject_records(Path(sys.argv[3]))
    pools["en"] += pinj
    full += pinj
    print("EN selected:", Counter(en_d.src[i] for i in sel),
          "| HP strata:", Counter(en_d.stratum[i].split("|")[0] for i in sel
                                  if en_d.src[i] == "HP"),
          "| TT llm:", Counter(en_d.stratum[i] for i in sel if en_d.src[i] == "TT"))
    # non-English (decided item by item)
    key = ok.src + ":" + ok.source_row_id
    ne = ok[key.isin(NON_EN)].copy()
    missing = set(NON_EN) - set(key[key.isin(NON_EN)])
    assert not missing, f"adjudicated ids not among automatic survivors: {missing}"
    ne["lang_final"] = [NON_EN[k][0] for k in ne.src + ":" + ne.source_row_id]
    for lg, sub in ne.groupby("lang_final"):
        d, _ = dedup(sub)
        pools[lg] = [record(r, lg, NON_EN[f"{r.src}:{r.source_row_id}"][1])
                     for r in d.itertuples()]
        print(f"{lg}: adjudicated {len(sub)} -> after dedup {len(d)} "
              f"({d.groupby('src').size().to_dict()})")
    # write
    for lg, recs in sorted(pools.items()):
        for rec in recs:
            validate(rec)
        with open(out / f"pool_{lg}.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    for rec in full:
        validate(rec)
    with open(out / "pool_en_full.jsonl", "w", encoding="utf-8") as f:
        for rec in full:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("written:", {lg: len(v) for lg, v in sorted(pools.items())}, "| pool_en_full:", len(full))


if __name__ == "__main__":
    main()
