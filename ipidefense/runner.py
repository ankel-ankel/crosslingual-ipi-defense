"""Steps of the scripts under scripts/: command line, done markers, commands, files.

Each experiment command runs in its own Python process, started from the repository root, so the
GPU memory of one command is free before the next one starts.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ipidefense.paths import ROOT, RUNS

DONE = RUNS / ".done"  # one marker per finished step: .done/<script>/<step>


@dataclass(frozen=True)
class Step:
    name: str
    about: str  # one line for --list
    paper: str  # tables and figures of the paper it produces
    func: Callable[[Run], bool | None]  # returning False leaves the step unmarked
    marker: bool = True  # False: the step skips finished work itself, from its outputs


def root_arg(var: str, default: str) -> str:
    """IPI_DATA, IPI_MODELS or IPI_RUNS as written in command arguments; relative to the
    repository root, like the default."""
    return os.environ.get(var) or default


def user_path(text: str) -> Path:
    """A path given on the command line, taken from the folder the script was started in."""
    return Path(os.path.abspath(text))


def shown(path: str | Path) -> str:
    """A path as printed: relative to the repository root when it lies inside it."""
    if isinstance(path, Path) and path.is_absolute():
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)
    return str(path)


def command(module: str, args: Sequence[str]) -> str:
    """A command as printed: the module file and its arguments."""
    words = [module.replace(".", "/") + ".py", *args]
    return " ".join(w if w and not any(c.isspace() for c in w) else f'"{w}"' for w in words)


def _word(arg: object) -> str:
    if isinstance(arg, str):
        return arg
    if isinstance(arg, int) and not isinstance(arg, bool):
        return str(arg)
    raise TypeError(f"command arguments are str or int, got {arg!r}")


class Run:
    """The commands and file operations of a step; a dry run prints them and runs nothing."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dry_run: bool = args.dry_run

    def _start(self, module: str, args: Sequence[object],
               **kwargs) -> tuple[str, subprocess.CompletedProcess | None]:
        words = [_word(a) for a in args]
        shown_cmd = command(module, words)
        print("+", shown_cmd, flush=True)
        if self.dry_run:
            return shown_cmd, None
        return shown_cmd, subprocess.run([sys.executable, "-m", module, *words], cwd=ROOT,
                                         **kwargs)

    def py(self, module: str, *args: str | int) -> None:
        """Run an ipidefense module with these arguments; stop the script if it fails."""
        shown_cmd, done = self._start(module, args)
        if done is not None and done.returncode != 0:
            raise SystemExit(f"failed (exit {done.returncode}): {shown_cmd}")

    def ok(self, module: str, *args: str | int) -> bool:
        """Run a check and say whether it passed (a dry run assumes it does)."""
        _, done = self._start(module, args)
        if done is None:
            print("  (the rest of the step runs only if this check passes)", flush=True)
            return True
        return done.returncode == 0

    def output(self, module: str, *args: str | int, dry: str = "") -> str:
        """Run a module and return what it prints (a dry run returns `dry`)."""
        shown_cmd, done = self._start(module, args, stdout=subprocess.PIPE, text=True)
        if done is None:
            return dry
        if done.returncode != 0:
            raise SystemExit(f"failed (exit {done.returncode}): {shown_cmd}")
        return done.stdout

    def copy(self, src: str | Path, dst: str | Path, *, if_exists: bool = False) -> None:
        print(f"copy {shown(src)} -> {shown(dst)}" + (" (if it exists)" if if_exists else ""),
              flush=True)
        if self.dry_run or (if_exists and not Path(src).exists()):
            return
        shutil.copy(src, dst)

    def move(self, src: str | Path, dst: str | Path) -> None:
        print(f"move {shown(src)} -> {shown(dst)}", flush=True)
        if not self.dry_run:
            shutil.move(str(src), str(dst))

    @contextmanager
    def keep(self, out: str | Path, dst: str | Path) -> Iterator[None]:
        """Around a command that writes `out` under a fixed name: store its output as `dst` and
        put back the file that was at `out` (kept meanwhile as `out`.keep)."""
        kept = Path(f"{out}.keep") if isinstance(out, Path) else f"{out}.keep"
        had = os.path.exists(out)
        if self.dry_run:
            print(f"copy {shown(out)} -> {shown(kept)} (if it exists)", flush=True)
        elif had:
            self.copy(out, kept)
        yield
        self.move(out, dst)
        if self.dry_run:
            print(f"move {shown(kept)} -> {shown(out)} (if it was kept)", flush=True)
        elif had:
            self.move(kept, out)

    def require(self, path: str | Path) -> None:
        """Stop unless the file exists."""
        if self.dry_run:
            print(f"check that {shown(path)} exists", flush=True)
        elif not Path(path).is_file():
            raise SystemExit(f"{path} is missing")


def marker(script: str, step: str) -> Path:
    return DONE / script / step


def print_steps(script: str, steps: Sequence[Step]) -> None:
    width = max(len(s.name) for s in steps)
    print(f"{script}.py steps, in run order:\n")
    for s in steps:
        print(f"  {s.name:<{width}}  {s.about}")
        if s.paper:
            print(f"  {'':<{width}}  paper: {s.paper}")


def main(script_file: str, doc: str, steps: Sequence[Step],
         options: Callable[[argparse.ArgumentParser], None] | None = None) -> None:
    """Command line of a script under scripts/: every step in order unless --only names some."""
    script = Path(script_file).stem
    ap = argparse.ArgumentParser(prog=Path(script_file).name, description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="list the steps and the tables and figures they produce")
    ap.add_argument("--only", nargs="+", metavar="STEP", help="run only these steps")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and file operations, run nothing")
    ap.add_argument("--force", action="store_true",
                    help="run steps again even if they are marked done")
    if options is not None:
        options(ap)
    args = ap.parse_args()
    if args.list:
        print_steps(script, steps)
        return
    by_name = {s.name: s for s in steps}
    unknown = [n for n in args.only or () if n not in by_name]
    if unknown:
        ap.error(f"unknown step {' '.join(unknown)} (steps: {' '.join(by_name)})")
    os.chdir(ROOT)  # relative roots and outputs are read from the repository root
    chosen = [by_name[n] for n in dict.fromkeys(args.only)] if args.only else list(steps)
    done = [s.name for s in chosen
            if s.marker and not args.force and marker(script, s.name).exists()]
    print(f"{script}.py: {' '.join(s.name for s in chosen)}", flush=True)
    if done:
        print(f"skipped, marked done: {' '.join(done)} (--force runs them again)", flush=True)
    if args.dry_run:
        print("dry run: commands and file operations are printed, nothing runs", flush=True)
    run = Run(args)
    for step in chosen:
        if step.name in done:
            print(f"\n== {step.name}: done, skipped ({shown(marker(script, step.name))})",
                  flush=True)
            continue
        print(f"\n== {step.name}: {step.about}", flush=True)
        if step.func(run) is False:
            print(f"{step.name}: nothing was run, the step stays unmarked", flush=True)
        elif step.marker and not args.dry_run:
            path = marker(script, step.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(datetime.now().isoformat(timespec="seconds") + "\n",
                            encoding="utf-8")
    print(f"\n{script}.py: finished" + (" (dry run)" if args.dry_run else ""), flush=True)
