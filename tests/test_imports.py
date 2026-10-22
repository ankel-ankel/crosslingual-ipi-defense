import importlib
import os
import pkgutil
import re
import subprocess
import sys
from pathlib import Path

import pytest

import ipidefense

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = [
    "ipidefense.paths", "ipidefense.utils", "ipidefense.arms", "ipidefense.model",
    "ipidefense.gpumem", "ipidefense.runner",
    "ipidefense.data.hosts", "ipidefense.data.splits", "ipidefense.data.documents",
    "ipidefense.attacks.templates", "ipidefense.attacks.witnesses", "ipidefense.attacks.recipes",
    "ipidefense.attacks.gcg",
    "ipidefense.rendering.nllb", "ipidefense.rendering.literal",
    "ipidefense.training.lora", "ipidefense.training.defense", "ipidefense.training.sets",
    "ipidefense.training.sft",
    "ipidefense.evaluation.obey", "ipidefense.evaluation.ifeval",
    "ipidefense.evaluation.ifeval_logged", "ipidefense.evaluation.runtime",
    "ipidefense.metrics.stats", "ipidefense.metrics.scoring",
    "ipidefense.third_party.ifeval.instructions",
    "ipidefense.third_party.ifeval.instructions_registry",
    "ipidefense.third_party.ifeval.instructions_util",
    *(f"ipidefense.experiments.{name}" for name in (
        "transfer", "utility", "styles", "adaptive", "controls", "position", "benign",
        "preservation", "case_study", "probe",
        "nonemission", "suite", "human_mt", "human_mt_table", "translated_hosts",
        "language_plan", "render_audit", "translations", "review", "canary", "taskacc",
        "squad_rescore", "judge_dump", "judge", "judge_validate", "ifeval_breakdown",
        "ifeval_subsets", "legit", "channel_swap", "lora_mask", "nonwitness",
        "nonwitness_render", "gcg_run", "gcg_mcnemar", "gcg_table", "gcg_cache_check",
        "backbones", "backbone_rules",
    )),
]

# in run order
SCRIPT_NAMES = ("prepare_data", "main_results", "languages", "semantic", "mechanism", "whitebox",
                "backbones", "figures")
SCRIPTS = [ROOT / "scripts" / f"{name}.py" for name in SCRIPT_NAMES]
COMMAND = re.compile(r"^\+ (ipidefense/[\w/]+)\.py\b", re.M)


def _found() -> list[str]:
    return sorted(m.name for m in pkgutil.walk_packages(ipidefense.__path__, "ipidefense.")
                  if not m.ispkg)


def _script(path: Path, flag: str, runs: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "IPI_RUNS": str(runs)}
    return subprocess.run([sys.executable, str(path), flag], env=env, capture_output=True,
                          text=True, timeout=300)


@pytest.mark.parametrize("name", EXPECTED)
def test_expected_module_imports(name: str) -> None:
    importlib.import_module(name)


@pytest.mark.parametrize("name", _found())
def test_every_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_scripts_exist() -> None:
    assert sorted(p.name for p in (ROOT / "scripts").glob("*.py")) == sorted(
        p.name for p in SCRIPTS)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_lists_and_runs_dry(script: Path, tmp_path: Path) -> None:
    listed = _script(script, "--list", tmp_path / "runs")
    assert listed.returncode == 0, listed.stderr
    dry = _script(script, "--dry-run", tmp_path / "runs")
    assert dry.returncode == 0, dry.stderr
    assert not (tmp_path / "runs").exists()  # a dry run writes nothing
    for name in sorted(set(COMMAND.findall(dry.stdout))):
        module = importlib.import_module(name.replace("/", "."))
        assert callable(getattr(module, "main", None)), (script.name, name)
