# Cross-Lingual Transfer of a Defense Against Indirect Prompt Injection

In indirect prompt injection, an attacker hides an instruction inside data that an LLM application
passes to the model, such as a document or a web page. The defense studied here fine-tunes a small
LoRA adapter (rank 16) on Qwen2.5-7B-Instruct so that, when the data contain an injected
instruction, the model still gives the answer it gives on the clean data. The repository trains
the adapter and tests whether it also resists injections written in languages it was not trained
on (training uses English, Spanish and Vietnamese attacks; the trusted instruction and the data
stay in English).

## Quick start

Requirements: Python 3.11 or newer and one GPU with 32 GB of memory.

```bash
# 1. install
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .

# 2. download the base model into models/
hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

# 3. download the public data sets
python scripts/prepare_data.py

# 4. train the defense (three seeds) and evaluate it on the held-out injections
python scripts/main_results.py --only train ifeval transfer summary
```

The adapters are written to `models/defense-qwen-seed{0,1,2}` and the results to
`runs/experiment/summary.json`.

## Using a trained adapter

```python
from ipidefense.model import generate, load_defense_lora_qwen

model, tok = load_defense_lora_qwen("models/defense-qwen-seed0")
reply = generate(model, tok, instruction="Summarize the text below.", data=document)
```

The untrusted data go in a separate `input` turn of the chat template, after the user's
instruction (see `build_messages` in `ipidefense/model.py`).

## Reproducing the paper

Run the scripts in `scripts/` in this order. Each one can be opened and run as is (Run button or
`python scripts/<name>.py`); it uses the settings of the paper, prints every command and skips the
steps that already finished, so it can be restarted after an interruption.

| Script | What it reproduces |
|---|---|
| `prepare_data.py` | downloads the public data and restores the Tensor Trust attack texts |
| `main_results.py` | defense adapters; main security and utility results; ablations; adaptive attack |
| `languages.py` | 26 attack templates in ten languages; human-written attacks; non-English hosts |
| `semantic.py` | task accuracy under attack; LLM judge; IFEval breakdown; legitimate instructions; non-witness attacks |
| `mechanism.py` | channel-swap control; adapter masking; representation probe |
| `whitebox.py` | white-box GCG attack |
| `backbones.py` | SEA-LION and Mistral backbones |
| `figures.py` | figures of the paper |

Useful options: `--list` shows the steps of a script and the tables they produce, `--only STEP`
runs selected steps, `--dry-run` prints the commands without running them, and `--force` reruns
finished steps. All outputs go to `runs/`. The comparisons with other defenses reported in the
paper are not part of this repository.

## Models

Download into `models/` with `hf download <id> --local-dir models/<name>`:

| Model | Needed for |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | everything (main backbone) |
| `aisingapore/Llama-SEA-LION-v3-8B-IT` | backbone experiments |
| `mistralai/Mistral-7B-Instruct-v0.3` | backbone experiments |
| `facebook/Meta-SecAlign-8B` | tokenizer and chat template used for SEA-LION (gated: set `HF_TOKEN`) |

`facebook/nllb-200-distilled-600M`, used to translate attacks, is downloaded automatically.

## Repository layout

```text
ipidefense/   library code: attacks, training, evaluation, experiments
scripts/      the scripts to run (see above); build_data/ shows how the shipped data were built
data/         shipped inputs (see below)
tests/        unit tests, run on the CPU with `pytest`
```

`data/` holds the inputs that the code cannot regenerate exactly: host splits for each seed,
attack templates and human-written attacks, the translated attacks with their quality checks, the
LLM-judge labels, training sets of the backbone experiments, task-accuracy items and synthetic
documents. `data/README.md` lists their sources, licences and checksums. The Tensor Trust attack
texts are not redistributed; `prepare_data.py` downloads them from the original release.

## Notes

- Decoding is greedy. Batched generation pads its inputs, so a different batch size, GPU or
  library version can change individual outputs; the scripts use the batch sizes of the paper.
- The paper's results were produced on one NVIDIA GeForce RTX 5090 (32 GB) with bf16 weights.
- `data/`, `models/` and `runs/` are read from the repository root; the environment variables
  `IPI_DATA`, `IPI_MODELS` and `IPI_RUNS` point them elsewhere.

## Licence

MIT (`LICENSE`). Third-party code, data and models keep their own licences: see
`THIRD_PARTY.md` and `data/README.md`.
