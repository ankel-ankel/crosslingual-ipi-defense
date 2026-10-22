"""Inputs of every experiment: AlpacaDataCleaned, the IFEval prompts and SQuAD v1.1 dev,
downloaded into data/ from pinned sources with their sha256 checked, and the Tensor Trust texts
(not shipped) of data/human_attacks/pool_<lang>.jsonl, restored from the pinned dump.

--force also replaces a downloaded file whose sha256 differs; --dump PATH reads a local copy of
raw_dump_attacks.jsonl.bz2 instead of downloading the Tensor Trust dump.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # runs without installing

from ipidefense.paths import DATA  # noqa: E402
from ipidefense.runner import Run, Step, main, shown, user_path  # noqa: E402

# file name under DATA -> (pinned URL, sha256)
FILES: dict[str, tuple[str, str]] = {
    "alpaca_cleaned.json": (
        "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/"
        "a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8/alpaca_data_cleaned.json",
        "bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b",
    ),
    "ifeval_input.jsonl": (
        "https://raw.githubusercontent.com/google-research/google-research/"
        "26d8ccdab6fec61b5c83ad6327ea8bda9e580288/instruction_following_eval/data/input_data.jsonl",
        "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
    ),
    "squad_dev_v1.1.json": (
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
        "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
    ),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, url: str, digest: str, force: bool) -> None:
    dest = DATA / name
    if dest.exists():
        if sha256_file(dest) == digest:
            print(f"{name}: present, sha256 ok", flush=True)
            return
        if not force:
            raise SystemExit(f"{dest} exists with another sha256; pass --force to replace it")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with part.open("wb") as fh:
            for block in resp.iter_content(1 << 20):
                fh.write(block)
    got = sha256_file(part)
    if got != digest:
        part.unlink()
        raise SystemExit(f"{name}: sha256 mismatch ({got}, expected {digest}) from {url}")
    part.replace(dest)
    print(f"{name}: downloaded, sha256 ok", flush=True)


REPO_ID = "qxcv/tensor-trust"
REVISION = "4de2b2fe01ba0cb6fbf7cbb9f1a3fabaf8157372"
DUMP = "raw-data/v2/raw_dump_attacks.jsonl.bz2"
DUMP_SHA256 = "87853cc8065a22156d15c4bdec777a8d35758749beb79769147d63a9644a73ce"
POOLS = DATA / "human_attacks"
SOURCES = POOLS / "sources" / "tensor_trust"
# sha1 of each complete pool, one json.dumps(record, ensure_ascii=False) per line
POOL_SHA1 = {"en": "0360b21a6852de3df6fa6f9f97318baf9166734c",
             "ru": "8f7f8f3cb98982d8d3e832ee9f4e5c40037e091c",
             "zh": "2fe3425f1931c53c4a52ef3742d7c48798a43ac9",
             "es": "8043ca5b2f897f09e781f6b095752ee67f6c1411"}
TARGET = re.compile(r"access\s+granted", re.I)  # the Tensor Trust target phrase
PLACEHOLDER = "{witness}"


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def adapt(text: str) -> str:
    """The single target match replaced by {witness}, literal braces doubled (common.adapt)."""
    (m,) = TARGET.finditer(text)
    head, tail = text[: m.start()], text[m.end():]
    return head.replace("{", "{{").replace("}", "}}") + PLACEHOLDER + tail.replace(
        "{", "{{").replace("}", "}}")


def fetch_dump() -> Path:
    """The raw attack dump at REVISION (public dataset: no token is sent)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, DUMP, repo_type="dataset", revision=REVISION,
                                local_dir=SOURCES, token=False))


def read_pool(lang: str) -> list[dict]:
    path = POOLS / f"pool_{lang}.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def dump_texts(path: Path, ids: set[str]) -> dict[str, str]:
    """attack_id -> attacker input for the wanted ids (untrusted file: bz2 + json only)."""
    out = {}
    with bz2.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if str(r.get("attack_id")) in ids:
                out[str(r["attack_id"])] = r.get("attacker_input") or ""
    return out


def restore(lang: str, recs: list[dict], texts: dict[str, str]) -> str:
    """The pool text with every withheld record filled in; refuses any mismatch."""
    for rec in recs:
        if "text_sha1" not in rec:
            continue
        text = texts.get(rec["source_row_id"])
        if text is None or sha1(text) != rec["text_sha1"]:
            raise SystemExit(f"{lang}/{rec['id']}: the dump text does not match text_sha1")
        adapted = adapt(text)
        if adapted.format(witness=rec["target_phrase_replaced"]) != text:
            raise SystemExit(f"{lang}/{rec['id']}: the adapted template does not round-trip")
        rec["original_text"], rec["adapted_template"] = text, adapted
        del rec["text_sha1"]
    body = "".join(json.dumps(rec, ensure_ascii=False) + "\n" for rec in recs)
    if sha1(body) != POOL_SHA1[lang]:
        raise SystemExit(f"pool_{lang}.jsonl: the restored pool differs from the evaluated pool")
    return body


def download(run: Run) -> None:
    for name, (url, digest) in FILES.items():
        if run.dry_run:
            print(f"fetch {name} from {url}", flush=True)
            continue
        fetch(name, url, digest, run.args.force)


def tensor_trust(run: Run) -> None:
    if run.dry_run:
        print(f"restore the Tensor Trust texts of {shown(POOLS)}/pool_<lang>.jsonl from "
              f"{REPO_ID}@{REVISION}:{DUMP}", flush=True)
        return
    pools = {lang: read_pool(lang) for lang in POOL_SHA1}
    ids = {r["source_row_id"] for recs in pools.values() for r in recs if "text_sha1" in r}
    texts: dict[str, str] = {}
    if ids:
        dump = run.args.dump or fetch_dump()
        if file_sha256(dump) != DUMP_SHA256:
            raise SystemExit(f"{dump}: sha256 is not the pinned {DUMP_SHA256}")
        texts = dump_texts(dump, ids)
    for lang, recs in pools.items():
        n = sum("text_sha1" in r for r in recs)
        body = restore(lang, recs, texts)
        if n:
            (POOLS / f"pool_{lang}.jsonl").write_text(body, encoding="utf-8", newline="\n")
        print(f"pool_{lang}.jsonl: {n} texts restored, {len(recs)} records")


def options(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dump", type=user_path, help="local copy of raw_dump_attacks.jsonl.bz2")


STEPS = [
    Step("download", "AlpacaDataCleaned, IFEval prompts and SQuAD v1.1 dev, sha256 checked",
         "inputs of every experiment", download),
    Step("tensor-trust", "Tensor Trust texts of the human-written attack pools, from the pinned "
         "dump", "inputs of the human-written attack results", tensor_trust),
]


if __name__ == "__main__":
    main(__file__, __doc__, STEPS, options)
