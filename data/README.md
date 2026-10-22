# Data

Inputs of the experiments. Files marked *downloaded* are fetched by a script and are not part of
the repository; everything else is shipped. The sha256 of every shipped file is listed under
[Checksums](#checksums).

| Path | Content | Origin |
|---|---|---|
| `alpaca_cleaned.json` | AlpacaDataCleaned, the host tasks | downloaded: `scripts/prepare_data.py`, step download |
| `ifeval_input.jsonl` | IFEval prompts | downloaded: `scripts/prepare_data.py`, step download |
| `squad_dev_v1.1.json` | SQuAD v1.1 development set | downloaded: `scripts/prepare_data.py`, step download |
| `splits/` | host slices per seed (Alpaca row indices) | `scripts/build_data/export_splits.py` |
| `human_attacks/` | human-written attack pools (en, ru, zh, es) | `scripts/build_data/human_attacks/`; Tensor Trust texts restored by `scripts/prepare_data.py`, step tensor-trust |
| `renderings/` | NLLB renderings, whole-sentence translations and their review | `ipidefense/experiments/suite.py render`, `translations.py export` / `import`, `review.py list` / `apply`, `canary.py build`; checked by `scripts/languages.py`, step reviews |
| `renderings/human_mt/` | translations of 24 English human-written attacks | `ipidefense/experiments/human_mt.py export` / `import` / `review-list` / `apply`; checked by `scripts/languages.py`, step human-mt |
| `renderings/nonwitness/` | translations of the five non-witness injections | `ipidefense/experiments/nonwitness.py export` / `import` / `review-list` / `apply`; checked by `scripts/semantic.py`, step nonwitness |
| `bactrian_x/` | 150 non-English hosts per language | `ipidefense/experiments/translated_hosts.py prepare` |
| `taskacc/` | SQuAD, BoolQ and HotpotQA items with decoys | `scripts/semantic.py`, step taskacc (`taskacc.py prepare`) |
| `legit_docs/` | 240 synthetic extraction documents | `scripts/semantic.py`, step legit (`ipidefense/data/documents.py`) |
| `ifeval_subsets.json` | subset label of every IFEval prompt and instruction | `scripts/semantic.py`, step ifeval (`ifeval_subsets.py`) |
| `judge/` | judge rubric, response pairs, blinding key, labels of the judge passes | `scripts/semantic.py`, step judge (response pairs and key; the labels come from the judge passes, merged by `ipidefense/experiments/judge.py merge`) |
| `baselines/` | NLLB render cache of the training wrappers, frozen Qwen targets of the seed-0 training hosts | `ipidefense/training/sets.py render`, `targets --seed 0` |
| `backbones/` | targets and training examples of the backbone configurations | `scripts/backbones.py`, steps targets and train |
| `human_attacks/sources/`, `bactrian_x/raw/` | raw downloads | downloaded: `scripts/prepare_data.py`, step tensor-trust (Tensor Trust dump); `ipidefense/experiments/translated_hosts.py prepare`; the pool build reads its raw sources from paths given on the command line |

## Downloaded inputs

### alpaca_cleaned.json

- Source: [gururise/AlpacaDataCleaned](https://github.com/gururise/AlpacaDataCleaned), file
  `alpaca_data_cleaned.json` at commit `a7d629079a95c2e4b7ec7dfe55087fbd18d9eba8`.
- sha256: `bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b`.
- Licence: the cleaned Alpaca data is CC BY-NC 4.0 (the repository code is Apache-2.0).
- 51,760 entries, of which 19,157 have a non-empty instruction and a non-empty input; these are
  the usable hosts (instruction = trusted task, input = untrusted data). The experiments do not
  use the reference responses.

### ifeval_input.jsonl

- Source: [google-research/google-research](https://github.com/google-research/google-research),
  `instruction_following_eval/data/input_data.jsonl` at commit
  `26d8ccdab6fec61b5c83ad6327ea8bda9e580288`.
- sha256: `67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49`.
- Licence: Apache-2.0. 541 prompts with 834 instructions.

### squad_dev_v1.1.json

- Source: SQuAD v1.1 development set,
  `https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json`.
- sha256: `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`.
- Licence: CC BY-SA 4.0. 48 articles, 10,570 questions; the answer-substitution experiments use
  the first 200 questions.

## Host splits (`splits/seed{0,1,2}.json`)

For each seed, the usable Alpaca rows are shuffled with `random.Random(seed)` and cut into
disjoint slices (`ipidefense/data/splits.py`): `train` 0-1500, `security` 1500-1650, `benign`
1650-1750, `adaptive` 1750-1850, `format` 1850-2000, `pilot` 2000-2030, `fresh` 2030-2530 and
`lrsel` 2530-2630 (used by no experiment here). Each file stores the sha256 of
`alpaca_cleaned.json`, the slice bounds and, per slice, the row indices of the hosts in order.
The code derives the hosts itself; the files document them and are checked against the code when
written (`scripts/build_data/export_splits.py`). Licence: indices into AlpacaDataCleaned.

## Human-written attacks (`human_attacks/pool_<lang>.jsonl`)

Successful attacks from public prompt-injection data, minimally adapted: the source's fixed target
phrase is replaced by the placeholder `{witness}` and nothing else changes (literal braces are
doubled for `str.format`). Candidates passed automatic filters (success in the source game, length
20-600 characters, no chat-template tokens, target phrase exactly once, no residual or translated
target fragment, no training witness, no game-specific references, seeded language identification,
near-duplicate removal) and were then read one by one by an AI assistant (Claude) against written
exclusion criteria. The build scripts are in `scripts/build_data/human_attacks/`; the raw sources
go to `human_attacks/sources/` (not shipped) and are passed on the command line.

| Pool | Records | Sources |
|---|---|---|
| `pool_en.jsonl` | 205 | HackAPrompt 110, Tensor Trust 90, PromptInject 5 |
| `pool_ru.jsonl` | 11 | Tensor Trust 11 |
| `pool_zh.jsonl` | 10 | Tensor Trust 9, HackAPrompt 1 |
| `pool_es.jsonl` | 8 | HackAPrompt 8 |

| Source | Revision | Licence |
|---|---|---|
| HackAPrompt (`hackaprompt/hackaprompt-dataset`, Hugging Face) | `25b87fbedfb86840abaf8cd09af7a029208a971a` | MIT |
| PromptInject (`agencyenterprise/PromptInject`, GitHub), `promptinject/prompt_data.py` | `2928a71` | MIT |
| Tensor Trust (`qxcv/tensor-trust`, Hugging Face mirror of the game's data dumps), `raw-data/v2` | `4de2b2fe01ba0cb6fbf7cbb9f1a3fabaf8157372` | no licence stated for the attack data (the game code is BSD-2-Clause) |

Fields: `id`, `source` (repository@revision), `source_row_id` (HackAPrompt row index, Tensor Trust
`attack_id`, PromptInject attack key), `lang`, `langdetect_prob`, `original_text`,
`adapted_template`, `target_phrase_replaced`, `notes`, `meta`.

Tensor Trust texts are not redistributed: in the shipped pools their `original_text` and
`adapted_template` are `null` and a `text_sha1` field identifies the text. Restore them with

```bash
python scripts/prepare_data.py --only tensor-trust
```

which downloads the pinned dump (public, no token), checks its sha256, fills the texts by attack id
with the same adaptation, checks every `text_sha1` and the sha1 of each complete pool, and drops the
`text_sha1` fields. The suite stops with a pointer to this script while texts are missing.

## Renderings (`renderings/`)

Every non-English injection of the paper is stored here, so the experiments do not translate at
run time.

- `<lang>.json` (en es vi fr de zh ko ar th bn my km ru): NLLB-200-distilled-600M renderings with
  the corrected witness-literal renderer (`ipidefense/rendering/literal.py`, `render_v2`), with
  back-translations, quality flags and review verdicts. `<lang>_old.json` (es vi fr de zh): the
  original renderer, for the renderer comparison. Licence: ours; produced with NLLB-200
  (CC BY-NC 4.0).
- `review_*.json`: review worklists, the verdicts of two independent LLM reviews (A and B, Claude)
  and their agreement reports for the NLLB renderings; the rubric is stored in each `<lang>.json`.
- `<lang>_llm.json` (en es vi fr de zh ko ar th bn my km): the 28 suite templates per language
  translated as whole sentences with Claude Opus 5.5 (Anthropic), keeping witnesses and markup
  verbatim. `llm/review_*.json`: worklists, the verdicts of two independent LLM reviews and their
  report; a rendering is used only if both reviews pass it. `llm/canary/`: the planted-defect set
  used to validate the reviews (key, worklists, both reviews' verdicts, report). No rendering was
  checked by a native speaker. Licence: ours.
- The `method` field of each record names the renderer (`llm:` marks whole-sentence translations);
  `sha1` identifies the rendered text.

### Human-written attacks, translated (`renderings/human_mt/`)

The 24 evaluated English human-written attacks (15 HackAPrompt-derived, 9 Tensor Trust-derived)
translated into es vi fr de zh ko bn as above, with the review files in `review_r1/` (key, canary
and review reports, rubric, worklist). The 9 Tensor Trust-derived records are withheld: they keep
their keys, witness, structure, review verdicts and `sha1`, but have no `english`, `rendered` or
`history` text and carry `release_ok: false`. `ipidefense/experiments/human_mt.py check` (step
human-mt of `scripts/languages.py`) lists them and stops; to run this experiment, restore the
English texts and translate these templates again (`human_mt.py export`, `import`, `review-list`,
`apply`).

### Non-witness injections, translated (`renderings/nonwitness/`)

The four format-hijacking injections and the answer-substitution injection of the non-emission
experiment, translated into es vi fr de zh ko bn as above (`JSON`, `P.S.` and the decoy
placeholder kept verbatim), with the review files in `review_r1/`.
`ipidefense/experiments/nonwitness.py check` (step nonwitness of `scripts/semantic.py`) verifies
them; its `export`, `import`, `review-list` and `apply` rebuild them.

## Non-English hosts (`bactrian_x/`)

- Source: [MBZUAI/Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X) at revision
  `3698480a001cb3f62c61c314d4acb181eb31e983` (Alpaca and Dolly instructions machine-translated
  with the Google Translate API). Licence: CC BY-NC 4.0; cite Li, Koto, Wu, Aji and Baldwin (2023),
  arXiv:2305.15011.
- `hosts_{es,vi,de,zh}.jsonl`: 150 hosts per language with `instruction` and `data` in the host
  language, `id` and the English source instruction. Items with an `alpaca-*` id and a non-empty
  instruction and input are joined to the English config; items whose normalized English
  instruction or input equals that of a training host of any seed are excluded; 150 are drawn with
  `random.Random('ml5-<lang>')`.
- `prepare_report.json`: the selection rule, the sha256 of the raw files and the counts.
- The raw files (`bactrian_x/raw/`, about 97 MB) are not shipped;
  `ipidefense/experiments/translated_hosts.py prepare` downloads them and draws the hosts again.

## Task accuracy items (`taskacc/`)

`squad500.json`, `boolq500.json` and `hotpot300.json`: items drawn with seed 0 from the SQuAD v1.1
development set (`data/squad_dev_v1.1.json`), BoolQ validation (`google/boolq` at
`35b264d03638db9f4ce671b711558bf7ff0f80d5`) and HotpotQA distractor validation
(`hotpotqa/hotpot_qa` at `1908d6afbbead072334abe2965f91bd2709910ab`), each with our decoys; the
source and revision are stored in each file. Licences: SQuAD CC BY-SA 4.0, BoolQ CC BY-SA 3.0,
HotpotQA CC BY-SA 4.0 (the files are shared under the same terms).
Step taskacc of `scripts/semantic.py` (`taskacc.py prepare`) rebuilds missing files.
`squad500_overlap_excluded.json`: the 34 SQuAD items whose in-context decoy shares a token with a
gold answer and which the answer-substitution experiment removes
(`ipidefense/experiments/nonwitness.py` computes the same list).

## Synthetic documents (`legit_docs/docs.json`)

240 synthetic extraction documents generated with seed 0 by `ipidefense/data/documents.py`, with
rule-based scorers; step legit of `scripts/semantic.py` regenerates and checks them. Licence:
ours.

## IFEval subsets (`ifeval_subsets.json`)

A subset label for every IFEval prompt and instruction (definitions and counts in the file),
derived from `ifeval_input.jsonl` by step ifeval of `scripts/semantic.py`
(`ipidefense/experiments/ifeval_subsets.py`). Licence: ours (labels over Apache-2.0 prompts).

## Judge data (`judge/`)

- `rubric.txt`: the judge prompt.
- `judge_items.jsonl`, `judge_key.json`: blinded response pairs (Qwen2.5-7B-Instruct outputs on
  Alpaca hosts) and the key that maps them back to arms; `judge_A.jsonl`, `judge_B.jsonl`: the
  labels of the two judge passes (Claude Opus 5.5).
- `validation/`: the items, key and calibration set of the agreement check and the labels of
  passes C, D and E.
- Licence: ours; host texts under the Alpaca licence. The labels cannot be regenerated exactly.

## Render cache and frozen Qwen targets (`baselines/`)

- `render_cache.json`: the NLLB-200-distilled-600M renderings (es, vi) of the wrapper segments of
  the training templates and of the held-out injections, as rendered in the defense runs
  (the `render` command of `ipidefense/training/sets.py`). Step train of `scripts/backbones.py` builds the
  training examples of configurations A and B with them, after checking them against the
  original-renderer strings of the render audit (step render-audit of `scripts/languages.py`).
- `seed0/targets.jsonl`: the answers of the frozen Qwen2.5-7B-Instruct to the 1,500 seed-0
  training hosts (160 new tokens, unbatched; the `targets` command of the same module, seed 0),
  the Qwen row of the target census (step targets of `scripts/backbones.py`).
- Licence: ours; renderings produced with NLLB-200 (CC BY-NC 4.0); Qwen outputs (Apache-2.0
  model) on Alpaca hosts (CC BY-NC 4.0).

## Backbone configurations (`backbones/`)

`{sealion,mistral}/targets_cap512.jsonl` and `targets_cap160.jsonl`: targets of the frozen
backbone regenerated with 512 and 160 new tokens (step targets of `scripts/backbones.py`);
`cfgA.jsonl` and `cfgB.jsonl`: the exact training examples of configurations A and B (step train).
Licence: ours; SEA-LION outputs under the Llama 3.1 Community License, Mistral outputs Apache-2.0;
Alpaca-derived inputs.

## Downloaded by the experiments (not shipped)

- `human_attacks/sources/`: raw HackAPrompt, Tensor Trust and PromptInject files for the pool
  build; step tensor-trust of `scripts/prepare_data.py` fetches the Tensor Trust dump.
- `bactrian_x/raw/`: `ipidefense/experiments/translated_hosts.py prepare`.

## Checksums

sha256 of the shipped files, relative to `data/` (check with `sha256sum -c` from `data/`
after saving the list to a file). The human-written attack pools are listed as shipped,
before `scripts/prepare_data.py` fills in the Tensor Trust texts. Some files have Windows (CRLF)
line endings; `.gitattributes` keeps every file under `data/` byte for byte. Code that writes
such a file again (`export_splits.py`, `documents.py`, `ifeval_subsets.py`) uses the line endings
of the platform, so the rewritten file can have another sha256 with the same content.

```text
b4f6ab5704754022d75e4755aed5c5d834070ccc88d5240387b269d204223c70  backbones/mistral/cfgA.jsonl
322c87516e540f2d301d96556ed2e2b2a9b43869792f372f2464525842816727  backbones/mistral/cfgB.jsonl
4dfb1df51f9c4996d5ea871c4c247c09c3a1e7272a36c67244d241ed94b24745  backbones/mistral/targets_cap160.jsonl
e2e87e4e63248fcc31f63054e738fcd6ef1efb7baeec4662578c9f120baaf48a  backbones/mistral/targets_cap512.jsonl
4adcf8df7e48c485dd36ab8c7b7779288fb4b2d8d57521b8d9e199b20dcda2ec  backbones/sealion/cfgA.jsonl
e1e70640e69765f8b4b32fc0d141b6f34e7654697023b2094c06f7eaebb4de4c  backbones/sealion/cfgB.jsonl
8a2e6e2983314b574643a1cbd63a9e569ef31ec3bfe841375aa0e7d6ceefc1bf  backbones/sealion/targets_cap160.jsonl
bc0d62887fc290ec0dbbd43bb68cc07063f31d133d023f9798c617a8e6ce2ac3  backbones/sealion/targets_cap512.jsonl
0510809b5b67055370985fd46b41d3b4f1a70f9e78fa133e03af1967e17de227  bactrian_x/hosts_de.jsonl
f35588890a23ab1866209d6a4a2fa3651a2c2abf5f9d3c2fdd52b46b7f3c7ede  bactrian_x/hosts_es.jsonl
8b87a3464acd2db49d3622c91d75c2dfc65096346d9972f44fa970fa4077c9c5  bactrian_x/hosts_vi.jsonl
be8e479945aeda6035f2493f532a6f74bbba1685de2c3e9e153f841037793040  bactrian_x/hosts_zh.jsonl
17ead87dad086891a91153a60857b0e4d74448f6b4c0524007b5e83883eea73e  bactrian_x/prepare_report.json
51314f065c51d32fadf61c3b0f87e001987582b3c7a66ae2dc5062f68147d1f3  baselines/render_cache.json
ded846d28421d39466ef7442dc32596d45fcdedb5d711e1de59beb3f87a01ca0  baselines/seed0/targets.jsonl
e5cc47b73c1900f85c85894445e87da05e1ef38ec6161e0807e7417bec768c3f  human_attacks/pool_en.jsonl
47db4141a43dadcaf09838518f213af8b5a609f68b2c81b902a11135379248f9  human_attacks/pool_es.jsonl
c8fceee6b88c6a16b61502d2e541c56b6d6273feccb2fe3e9815a2211658fa2b  human_attacks/pool_ru.jsonl
f12ea93904239b45e9fdd8dadf81ea1718d9dc3e653e6c7f4c9246f4ab936b59  human_attacks/pool_zh.jsonl
f90a9bb8289ec1c6f920167283f44e1b280473faf2ba1d2b4a53a9255a769fb2  ifeval_subsets.json
78f91990299819e0e9c2f0f335dbaced3057129e82a2b93902f5c8f3ab042caa  judge/judge_A.jsonl
cec6dea271622cadad74a340912684065a2516dae974158c486e347f56d4df00  judge/judge_B.jsonl
f67df6a3d158e691dc0986ccedeb1f0206f406ef87bfa6b24bd0b95158d293a8  judge/judge_items.jsonl
f6263df956cdcb712595c9f466f06cf6642cfc4d7890b96ec378aaba87eaca7b  judge/judge_key.json
72cef8cfff9acea234600549de739d57adc04056635818b5dffd8fea59106534  judge/rubric.txt
54eca4adb8978329ae7fc41d5f234bd9c62ace09fd52387430a235722f635070  judge/validation/audit_items.jsonl
ffa59306221744f5f6b281b717026912492d6610157f16fb19b2d6f70f996faf  judge/validation/audit_key.json
4191c2cf6f6eba4443666dcbca0d8d09e8515a8d10a8923588e35407a7e35dd7  judge/validation/calib.json
15f6c2cdee98a01f08fd35a5966b852e06ce0bbad0f57e5546164120123d9326  judge/validation/pass_C.json
994509ee4a0321f284a0b8b0eb2aa96da911132f4710334c39eeacd6ed80a409  judge/validation/pass_D.json
89be0d0d4451856375b8a6076f0263f03387683f7ebc7d85f012b93eebe39f36  judge/validation/pass_E.json
bd9836784516252b8d88c8b865908902f28a1b29881c48ecf909fc0eeba0bbcd  legit_docs/docs.json
c6174095361635f54116e635a4755d55a87e58e91f512282a98082f079db459d  renderings/ar.json
07c4f20b6cac875bdcd8b00c0195686934941f5e15c38647dad8042eeebebae5  renderings/ar_llm.json
44014506f45c022bb5944cbb4e3b81976a154d83296e2df1001bbb2cb03358d1  renderings/bn.json
73837f34fdabb53dd19a0afce18961d9be63e209e719e5d1c78034144ab1c273  renderings/bn_llm.json
37cb32970261824818fbeb41d96422a9a11373ae60d653648fda9f8b94bd377a  renderings/de.json
bb045f86b18c465be1f381143f0faa82c6a67f25ab845c714fbd37bd458ada23  renderings/de_llm.json
f3867e5448e814ceef821f7edc1799fc13b67c08f56e5ead951e6c80b06f84e5  renderings/de_old.json
58b0c68b7a6dbdb8017629fdfe56276d51a0a968ba4d2928ca04240812fec79b  renderings/en.json
419c5be06dbd6b69be2f235c70d149839ceb67ad0d67a9d747250017bf592ade  renderings/en_llm.json
672bcb263a1a4a8f41c4ad7ca2530e6fbe52bd169eb738520247f8dd4aa3c768  renderings/es.json
aac0a0bbb8c8c1d8a9900339661d644e4e89b3201eca0aa8f147fbb8549f6030  renderings/es_llm.json
7fef39a8bb55c6e3d8b1f2d3d3724af7296e04cc795b9a6301ed0d46229de149  renderings/es_old.json
fa7e5618540c34907b19d11f2af8eb47bf443e166e96f3c71daba2ce318916b4  renderings/fr.json
69914f46b1ec54958e23d1d07fb9f731cb53e7b1568f219a69e2a224d0847a6f  renderings/fr_llm.json
4fe0b2d8d0d7045503b9be7228f693a8886475ba74bae8db4b2daa8cbaea954d  renderings/fr_old.json
97349743521ed072d0d2f684181d0e11e625a89042fd9b66212f26e1623cb896  renderings/human_mt/bn.json
75b076c284263b2340acf9dcfa53c192f778aa96fd4756106f0453e0eb933564  renderings/human_mt/de.json
b881a2fc8d2ec889f84e0312a4d8945d79e52a848cb040004b9a3058fd49aae9  renderings/human_mt/es.json
e8638a38eda43f33e84d62fc050a2cdd5c2b141f8ffc2c504775ad48a41e15db  renderings/human_mt/fr.json
a078c3a265cc87e2b3c955c04f210d207ff16c468880de4904b92d470e19816b  renderings/human_mt/ko.json
91e56262de55e6a2b9cd9f0f640ee867395c196df0cf4238ff03e7efdd439a2f  renderings/human_mt/review_r1/canary_report_d8a0539ae1.json
79646ddb5848d58e98968545976a81dd050afca8a2a5e629a0157e9a8ca7c48b  renderings/human_mt/review_r1/key.json
b9f6c7d9d17be9061aae6abe1cb0fc67aabc2294430d5862625eda2993d9f86e  renderings/human_mt/review_r1/review_report_d8a0539ae1.json
a75bd211a327d4c7f9430c615e2b6db5484ac57e5bca7eb60a0445a24cda7215  renderings/human_mt/review_r1/rubric.txt
0f52621549fa2295e1b903c6183493e08d77690c831e6c22661e5be289320fb5  renderings/human_mt/review_r1/worklist.json
c25a7d4c9302b3638485caabba708f4ec33804ea0951012d58c0687eaf2cfae7  renderings/human_mt/vi.json
9c9ca3346e95378dfb4e364c62fe04ad5eba106d9e85eae49e48512963587382  renderings/human_mt/zh.json
88e4672526f578b59383c7eac567bded38e13eb49d08cf60a9930cfbd97ac7a9  renderings/km.json
23213eca68f0be799891ea62167d5e4d4d9b28292fd6cbb8f80c1e24ed62c0e4  renderings/km_llm.json
fa8bef4b08c38f6ce852b13ff18e8f9d881625891a2e5727ff06c683678d00ff  renderings/ko.json
6ac407f87b64db660f617df3d787df7dfc94c1d6ba9f85fa14fe7a7ef7f0fb42  renderings/ko_llm.json
4ba2ef2b38b0d3b5816f4cda64d290945487c39389e7415295272fc3fef4ed53  renderings/llm/canary/canary_report.json
3d27ea8a00fed3b2a8da1d6c584a2e29cc016c9f8746a877884ce75b204d885a  renderings/llm/canary/key.json
0cbe8783268366d7010a30be302a4064b0a641fcfe1dd2aa3997342062c06903  renderings/llm/canary/review_canary_A.json
95c5009dcbdbf2164d8b27a775c39dc1d732b2f1acc37d9a936cba64016d0a26  renderings/llm/canary/review_canary_B.json
936707fcc18a796af33f6a01359a55052bcaf65004171f071e267de157fe08e2  renderings/llm/canary/review_worklist_ar.json
e0063a5c778dd6eff622ae556f974eb82ba3a18360a3b333b95301249e32dcf7  renderings/llm/canary/review_worklist_bn.json
9a4e0c8cb23a0a21336299482bc86a4616a20279ed2a78a306f7cebb3e30079f  renderings/llm/canary/review_worklist_de.json
f05b924390985cfe3e1477a24c81e055d8382cee395a281579340cd8dd69b16a  renderings/llm/canary/review_worklist_es.json
769177df6a316dd0bee9649459bfc1440af7d9cb70587ba047c10d5bb885c873  renderings/llm/canary/review_worklist_fr.json
b33d16feb45a377f6fb7af255dab77542d5de21c89adedc1828d1dfd442777ae  renderings/llm/canary/review_worklist_km.json
6cc84f63832ca005fc45255a0825393f5f8769883468badd425a618ba2291c0c  renderings/llm/canary/review_worklist_ko.json
378860ad9b5e2aeb10efe6a483ebc2043358cdd6ff6e8b1a46168729e8252d87  renderings/llm/canary/review_worklist_my.json
ad143d7eac2ba1f4652aae6229eafa7b49a7474fb102b7cb7e57162bd3c3e8c2  renderings/llm/canary/review_worklist_th.json
11b685a4702d11d789ab5769c92fc30d67d09e3a61bf4143aac5bda492a2560b  renderings/llm/canary/review_worklist_vi.json
3f16bceb91d5e5d0bb46417a89b7938d0603fc697558711844d37637607e78cf  renderings/llm/canary/review_worklist_zh.json
d5076eb2bfd921a6f862bb1ae0edb541d2e1ed50c3e43673796a1c7c9cdfc7e8  renderings/llm/review_llm_A.json
ceb43071b2335446a85fe8db8d5916e289892e8725a6268af3ad06a158451c8d  renderings/llm/review_llm_B.json
10de6a48ff2ebe97344a501151a489dff0864c9341c026c398956f7ee53d18a4  renderings/llm/review_report_llm.json
1fb8f0d60fcd411e58378dfa853a52975ee1dadd74c9c14b1729c03b7a0ddf67  renderings/llm/review_worklist_all.json
f108cc62daf8ae4a6f99efd6d92435e865938cacdd1174a2cf4c9e97a690e9f5  renderings/llm/review_worklist_ar.json
74640955f9d002f0d0bc71bde18be284ace013e0f0470870d9462dc0ba12bbe5  renderings/llm/review_worklist_bn.json
98e2f616512c41c8a6dd4a830857e8ddf32d3b33ea388c1881768f24c20775e3  renderings/llm/review_worklist_de.json
387e566deda1d896a140938511af951e8a1616994a1a068c7f136c2cef94c377  renderings/llm/review_worklist_es.json
65e12f2d2f791c03cc28f616d9d575d73218afa26fd7d0c7dc38aee101aecf7e  renderings/llm/review_worklist_fr.json
3fa0c714afadb733e96901909de5e1daa14cce46049916976b2f286f44019bfd  renderings/llm/review_worklist_km.json
a4194d4131d3c4e62d2da662f7e03da8c66f2067af3e4d9f6663be6e018f5a7a  renderings/llm/review_worklist_ko.json
e1d5cba6e8b24cecafbccfe583bd52be5bc9956f7ba85f09dd59036f23432941  renderings/llm/review_worklist_my.json
4c1268efca97d6b37dd29cead274a8796d41ba223c4a056c73ec98e4979b76e9  renderings/llm/review_worklist_th.json
49f0753a65d8c2fcf1050a0ba9af8e9a959ae6f6b590fb6f4dccfa5da9c4662e  renderings/llm/review_worklist_vi.json
1487e25ec62b81a630dbffa23f0ce00c44aec550ee610d01add868a4cd7af6e5  renderings/llm/review_worklist_zh.json
4b3e51750da73206052c28c5e355e58b7d596b2af2d151ceb07e0b09c77ef44d  renderings/my.json
14b84213f7b8d002113a1c7462b61c6fbdfcbdaf50e41f733c8a164285f659f7  renderings/my_llm.json
f7a2a7ddb00f2116ae5e182533fdfd4faff7e44ba9c42a56ebab870563de0622  renderings/nonwitness/bn.json
db88e1dafb0dee3735e073780bc371b89c8901ca2ffe9762ec8e95a7fed3e318  renderings/nonwitness/de.json
5ecb738b4512023e444652187441fe4970a4c6dea18b27dad0ef116bc6ccc347  renderings/nonwitness/es.json
2d566bd024dcefd8e8930d46d79be51ffefdeced0e9a8fab99905f69eeba27ee  renderings/nonwitness/fr.json
425a674032546feaed1a5dd80293fb22fc46abca43773974acd9fb35d846fcb9  renderings/nonwitness/ko.json
d0355e6ed3d1194aae64f9fb334ba01e2c7769627a2d63d583714bf15969d379  renderings/nonwitness/review_r1/canary_report_bcb739717b.json
fdb8bb629f6338ae423c1e7e0c5c358e6186feca85c30551e87a3b9f28334fc4  renderings/nonwitness/review_r1/key.json
809801fd954c191d7818a68ccf9e0586cc5c206340c092c150984de1c1194d00  renderings/nonwitness/review_r1/review_report_bcb739717b.json
68d572039e235c7852c0b203711f685a7088ba804ff9eb73895aaf6ba4f23065  renderings/nonwitness/review_r1/rubric.txt
b01f426312dda9e509c733f1afd96430a6c0f9cb983e8a329dc33d6da0a89adb  renderings/nonwitness/review_r1/worklist.json
8a12eaad5a6fcfd00f90a3a47e3d52d20c7591f5173bec74b690e890b6055212  renderings/nonwitness/vi.json
a7e422b3293493d9d241c5975a5c8737031a79c612410ec434f90984ec9a11e9  renderings/nonwitness/zh.json
faf547da1a364b9292d4c81fd05c181e2fd20406351c29e3083bce933236ac7d  renderings/review_A.json
17e77d4d777a00276b727efdbfffcc93116d69ea23f8bd02efcc53b3c38af218  renderings/review_B.json
31dff7d87887842fdeaa71df23e47666f7a8eb928558158d011726be01bcc6b7  renderings/review_report_v2_round1.json
b934856cd30e7e6340c00bb3c8472796e824d6a49d552a178f22c7ca9c7320f1  renderings/review_report_v2_supp.json
d611041236f9281e71a8d75fe0ef2a944c127dd4526ee52ef363967c4f7d0a66  renderings/review_summary.json
2cf3fac815d9c0895af20b6e5751eaf787447423c8298226ff14074c2b5135f7  renderings/review_supp_A.json
565d34dcc1bdea1cb1f4d1582c2bda1894869605edab2db5e056c52dae676e1a  renderings/review_supp_B.json
011d23690e8893b7fab09c562f797af624ed4a42efb8224a097a5b50c1a8fc2d  renderings/review_worklist.json
28cffff4ab3b7f9d04015c52194e829c3dc1c74780e56ba74aa9d585fe4de0bf  renderings/review_worklist_base.json
13f73548420a68ea3a4690c5e3b7540f014051a066165fe21c0f71b6329ba577  renderings/review_worklist_supp.json
bf1c1fb51311b42c79ddd2e1859bfb29a79b764ef0b3592beba82c09c35c5aa2  renderings/ru.json
0d045ad0540c7ee9b1df2d21dac27302408673b96a7dfbc203d6fccb81716968  renderings/th.json
e617fcfa1a8e6ddb177d4246e3bfe90c972fa702d7c9b9c06b08512c7c4fe156  renderings/th_llm.json
782c3f44e27d65d3e8c433a4ea994c8c5bc1156838bfed0f0c21986dd52a78d2  renderings/vi.json
fb8eb4612778feb5d1730ce0f24a3996433199a0588a221f88dded853faf83e3  renderings/vi_llm.json
5ba816c60eb3a6a20f643dad22e5f3fe2406af8f4239e7c076647196ba22c479  renderings/vi_old.json
44ece28a67e6de63d883a21ffc33fd2015ad474ce3940360386409009eaa8286  renderings/zh.json
68829ecad7986b5769a8924d7fb8d02f6a88b5fb640fa01f33ac0069f19b0ea2  renderings/zh_llm.json
253f68e03969c4caa76926df180ed83790e36b4ef6a478ee9edd76931eab5285  renderings/zh_old.json
dd55cdd6aaff5975870f00c156a2a2dd9e291ac779610b9c3efed49ce82182ec  splits/seed0.json
d8be995f7bb0c4ab84847402f17c7fd4af5a5bc3de532976430c545b47e023ad  splits/seed1.json
284654abaf52fee6e8b45aaf8ad73f94edb33bb9a966549677a57996e755f8d0  splits/seed2.json
28f3ef08514cadc36c937eb07a24c89edbb1922642d1332d9d389d478e2f41c8  taskacc/boolq500.json
011c36f65407d8ccd8b09abb511a69ee38b3b006a3c229d469f250220a7ac47b  taskacc/hotpot300.json
78d27556fba6b0964ae7b4b00f5e7e95ac8bcbb552c9159fb7f88f7d155ca151  taskacc/squad500.json
b145a6a9c62b3f3e10467a761343b780e752a161bd87b4a4299d69ea5f708de2  taskacc/squad500_overlap_excluded.json
```
