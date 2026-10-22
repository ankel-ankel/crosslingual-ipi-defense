"""Generation requests, append-only JSONL transcripts with resume, small I/O helpers."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Item:
    """One generation request; `key` must be unique within a transcript file."""
    key: str
    instruction: str
    data: str | None
    meta: dict


class Transcript:
    """Append-only JSONL transcript with resume by item key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        if path.exists():
            # split on "\n" only: ensure_ascii=False leaves U+2028/U+2029/U+0085 unescaped
            lines = path.read_text(encoding="utf-8").split("\n")
            for n, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    if n >= len(lines) - 2:  # last line cut off by a crash
                        print(f"WARNING: skipping truncated last line in {path}", flush=True)
                        continue
                    raise
                self.done[rec["key"]] = rec

    def pending(self, items: Iterable[Item]) -> list[Item]:
        return [it for it in items if it.key not in self.done]

    def add(self, recs: list[dict]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self.done[rec["key"]] = rec


def chunks(seq: list, size: int) -> Iterator[list]:
    for k in range(0, len(seq), size):
        yield seq[k : k + size]


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
