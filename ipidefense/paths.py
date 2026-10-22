import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("IPI_DATA") or ROOT / "data")
MODELS = Path(os.environ.get("IPI_MODELS") or ROOT / "models")
RUNS = Path(os.environ.get("IPI_RUNS") or ROOT / "runs")
