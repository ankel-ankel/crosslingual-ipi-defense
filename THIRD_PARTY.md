# Third-party code, data and models

The repository code is released under the MIT licence (`LICENSE`). The code listed below is
vendored or ported from other projects and keeps the licence of its source. Data sets and models
are not redistributed unless stated in `data/README.md`.

## Vendored code

### IFEval instruction checker (`ipidefense/third_party/ifeval/`)

- Source: [google-research/google-research](https://github.com/google-research/google-research),
  directory `instruction_following_eval/` at commit `5b09c22d73a9d35eb6c5d2a99b95677a45053466`:
  `instructions.py`, `instructions_registry.py`, `instructions_util.py`.
- Copyright The Google Research Authors. Licensed under the Apache License, Version 2.0; the
  licence header is kept in each file, followed by a note on what was changed, and the full
  licence text is reproduced at the end of this file.
- Changes made for this repository:
  - `from absl import logging` is replaced by the standard `logging` module, and the imports
    between the three files are package-relative;
  - nltk is not used: `count_words` counts `\w+` matches with `re` (the pattern of the nltk
    tokenizer it replaces), `count_sentences` counts with the module's own
    `split_into_sentences` instead of nltk's punkt model, and the capital-word frequency check
    tokenizes with a new `word_tokenize` (`\w+` matches) instead of `nltk.word_tokenize`.
    Sentence and capital-word counts can therefore differ from the upstream checker.

## Ported code

### nanoGCG candidate sampler (`ipidefense/attacks/gcg.py`)

`sample_ids_from_grad` is ported from [nanoGCG](https://github.com/GraySwanAI/nanoGCG) 0.3.0
(tag `v0.3.0`, commit `33d84187f5ad9ec2c9f7f6ec3af6106be3dbc6f5`, `nanogcg/gcg.py`); the
retokenization check of candidate suffixes follows the same idea. nanoGCG is distributed under the
MIT License:

```text
MIT License

Copyright (c) 2024 Gray Swan AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### StruQ, SecAlign and Meta-SecAlign recipes (`ipidefense/attacks/recipes.py`)

- The injection recipes in `recipes.py` follow the reference code of
  [StruQ](https://github.com/Sizhe-Chen/StruQ), [SecAlign](https://github.com/facebookresearch/SecAlign)
  (`struq.py`, `align.py`, `config.py`, `test.py`) and
  [Meta_SecAlign](https://github.com/facebookresearch/Meta_SecAlign) (`config.py`, `utils.py`):
  naive and completion injections, the sets of textual delimiters, the injected-prompt format and
  StruQ's test-time filter. The delimiter lists are copied from `config.py`.
- These three repositories are released under the Creative Commons Attribution-NonCommercial 4.0
  International licence (CC BY-NC 4.0), https://creativecommons.org/licenses/by-nc/4.0/.

## Data sets

Sources, revisions, checksums and licences of every data item are listed in `data/README.md`.
In short:

| Data | Use | Licence | Shipped |
|---|---|---|---|
| AlpacaDataCleaned | host tasks | CC BY-NC 4.0 | no (downloaded) |
| IFEval prompts | utility | Apache-2.0 | no (downloaded) |
| SQuAD v1.1 dev | answer substitution, task accuracy | CC BY-SA 4.0 | no (downloaded); items with decoys in `data/taskacc/` |
| BoolQ validation | task accuracy | CC BY-SA 3.0 | items with decoys in `data/taskacc/` |
| HotpotQA distractor validation | task accuracy | CC BY-SA 4.0 | items with decoys in `data/taskacc/` |
| HackAPrompt | human-written attacks | MIT | adapted texts in `data/human_attacks/` |
| PromptInject | human-written attacks | MIT | adapted texts in `data/human_attacks/` |
| Tensor Trust | human-written attacks | no licence stated for the attack data | ids and sha1 only; texts restored by `scripts/prepare_data.py` |
| Bactrian-X | non-English hosts | CC BY-NC 4.0 | 150 hosts per language in `data/bactrian_x/` |

## Models

| Model | Use | Licence |
|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | primary backbone | Apache-2.0 |
| meta-llama/Llama-3.1-8B-Instruct | `--base llama` of the training and evaluation modules (no script uses it) | Llama 3.1 Community License |
| facebook/Meta-SecAlign-8B | tokenizer files, for the chat template of the SEA-LION backbone | Llama 3.1 Community License |
| aisingapore/Llama-SEA-LION-v3-8B-IT | additional backbone | Llama 3.1 Community License |
| mistralai/Mistral-7B-Instruct-v0.3 | additional backbone | Apache-2.0 |
| facebook/nllb-200-distilled-600M | fragment translation of injections | CC BY-NC 4.0 |

Model outputs shipped in `data/` (training targets, judged responses) are covered by the licence
of the model that produced them. The suite translations and the judge labels in `data/` were
produced with Claude Opus 5.5 (Anthropic).

## Apache License 2.0

The full text of the licence of the vendored IFEval checker:

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
