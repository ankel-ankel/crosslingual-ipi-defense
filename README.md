# Cross-Lingual Transfer of a Defense Against Indirect Prompt Injection

A LoRA defense against indirect prompt injection for Qwen2.5-7B-Instruct. When the data passed to
the model contain an injected instruction, the defended model still answers the trusted
instruction. The adapter is trained on attacks in English, Spanish and Vietnamese and evaluated
on other attack languages as well.

## Installation

Python 3.11+ and one GPU with 32 GB of memory.

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
```

## Quick start

```bash
python scripts/prepare_data.py
python scripts/main_results.py --only train ifeval transfer summary
```

This trains the adapter with three seeds (`models/defense-qwen-seed{0,1,2}`) and writes the
results to `runs/experiment/summary.json`.

## Usage

```python
from ipidefense.model import generate, load_defense_lora_qwen

model, tok = load_defense_lora_qwen("models/defense-qwen-seed0")
reply = generate(model, tok, instruction="Summarize the text below.", data=document)
```

The untrusted data are passed separately from the instruction, in their own `input` turn.

## Experiments

Run the scripts in `scripts/` in this order, with the Run button or `python scripts/<name>.py`.
Finished steps are skipped; `--list` shows the steps and `--only` runs selected ones.

| Script | Contents |
|---|---|
| `prepare_data.py` | public data sets and Tensor Trust attack texts |
| `main_results.py` | training, security and utility, ablations, adaptive attack |
| `languages.py` | attack suite in ten languages, human-written attacks, non-English hosts |
| `semantic.py` | task accuracy, LLM judge, IFEval breakdown, legitimate instructions, non-witness attacks |
| `mechanism.py` | channel swap, adapter masking, representation probe |
| `whitebox.py` | white-box GCG attack |
| `backbones.py` | SEA-LION and Mistral backbones |
| `figures.py` | figures |

The SEA-LION and Mistral experiments also need these models in `models/`:

| Model | Note |
|---|---|
| `aisingapore/Llama-SEA-LION-v3-8B-IT` | |
| `mistralai/Mistral-7B-Instruct-v0.3` | |
| `facebook/Meta-SecAlign-8B` | chat template for SEA-LION; gated, needs `HF_TOKEN` |

## Repository layout

```text
ipidefense/   library code
scripts/      experiment scripts; build_data/ builds the shipped data
data/         shipped data; sources and licences in data/README.md
tests/        unit tests (CPU): pytest
```

## Licence

MIT. Third-party code, data and models keep their own licences (`THIRD_PARTY.md`).
