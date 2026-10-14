# H3 Technical Report: Question-Level Reasoning-Trace Trajectories (R1-Distill-Llama-8B)

**Project:** TTCA (Orange R&D Internship)
**Model analysed:** `r1-llama8b`
**Source notebooks:** `0_sentence_segmentation.ipynb`, `1_reasoning_trace_labeler.ipynb`, `2_reasoning_trace_trajectory_analysis.ipynb`
**Result folder:** `~/Documents/Orange R&D Internship/TTCA/src/results/r1-llama8b/`

> **Note on scope and assumption.** None of the three notebooks uses the label "H3" explicitly. This report treats **H3 as the question-level trajectory analysis of reasoning traces** (notebook 2 and the pipeline feeding it: notebooks 0 and 1). If H3 in your write-up is defined differently, only Section 2 (the hypothesis statement) needs to change. Every number below comes from the notebook outputs. Values marked *(read from figure)* were read off a plot and are approximate. Statements marked *(interpretation)* are my reading, not something the notebook computes.

---

## Table of contents

1. Summary
2. Hypothesis and research questions
3. Pipeline overview
4. Stage 0: Sentence segmentation
5. Stage 1: Segment labeling
6. Stage 2: Trajectory analysis (methods)
7. Data and eligibility
8. Results
   - 8.1 Question groups
   - 8.2 ALL-CORRECT
   - 8.3 MIXED
   - 8.4 ALL-WRONG
   - 8.5 Group comparison
   - 8.6 Annotation consistency
9. Cross-cutting findings
10. Limitations and caveats
11. Output files (tree)
12. Index of figures and tables
13. Reproducibility notes

---

## 1. Summary

- Reasoning traces of R1-Distill-Llama-8B were split into sentences (stage 0), labeled per segment by an LLM (stage 1), and analysed **per question** (stage 2).
- After quality filters, **6,575 eligible samples** over **268 questions** (≥ 5 eligible samples each) were analysed. Of the 270 questions with any eligible sample, 2 were too small to use.
- Questions fall into three groups: **ALL-CORRECT (43)**, **MIXED (167)**, **ALL-WRONG (58)**.
- Main observations (all observational, no causal claims):
  - **ALL-CORRECT:** in the 6 questions with a large length spread, long correct samples contain doubt and verification phases (hedge cues 0.907 per 100 words vs 0.000 in short samples). Short correct samples are pure setup followed by reasoning.
  - **MIXED (167 questions):** incorrect samples are *shorter* than correct ones on average (median ratio 0.84; incorrect longer in only 31% of questions). They show less verification, fewer answer replacements and more traces with no answer value.
  - **ALL-WRONG:** in the 12 questions with a large length spread, long wrong samples are about 16.5× longer (median ratio) and highly repetitive (4-gram repetition share 0.222 vs 0.037), with much less new vocabulary.
  - Across groups, ALL-WRONG has the highest hedge rate (1.647 per 100 words) and the highest share of text after the first answer (0.409).
- The group comparison is **between different questions** and must not be read as a within-question effect.

---

## 2. Hypothesis and research questions

**Working formulation of H3 (edit as needed):** *Within the same question, the temporal structure of a reasoning trace (how it distributes setup, reasoning, doubt, verification and correction, and how its answer develops) differs systematically between long and short samples and between correct and incorrect samples.*

The notebook asks three concrete questions, one per group:

| Group | Question asked in the notebook |
|---|---|
| ALL-CORRECT | When the model is right every time, how do its longer samples differ from its shorter ones? |
| MIXED | Inside the same question, how do incorrect samples differ from correct ones? |
| ALL-WRONG | Do the longer wrong samples differ from the shorter wrong samples? |

The study is **observational**: no LLM inference at analysis time, no new labels, no causal claims.

---

## 3. Pipeline overview

| Stage | Notebook | Input | Output |
|---|---|---|---|
| 0 | `0_sentence_segmentation.ipynb` | `*_raw_results.csv` with `thinking_text` | CSV with `⟦S1⟧ ⟦S2⟧ …` markers inserted (`*_segmented.csv`) |
| 1 | `1_reasoning_trace_labeler.ipynb` | `r1-llama8b_raw_results_segmented.csv` | `labeled_results.csv`, `labeled_segments.csv`, checkpoint JSONL |
| 2 | `2_reasoning_trace_trajectory_analysis.ipynb` | `r1-llama8b_raw_results_segmented_labeled.csv` | Figures, tables and the `src/results/r1-llama8b/textual_analysis` folder (also zipped as `src/results/r1-llama8b/textual_analysis.zip`) |

---

## 4. Stage 0: Sentence segmentation

**Goal.** Insert markers `⟦S1⟧ ⟦S2⟧ …` at the start of each sentence in `thinking_text` **without changing any original character**. Removing the markers always returns the exact input. The approach uses pure-Python rules and contextual checks (no LLM, no downloads, only `pandas`).

**Rules.**

- A sentence can end at `.` `!` `?` `…` (plus closing quotes, brackets, `*`) **only if whitespace follows** and the next token looks like a sentence start (capital letter, digit, opening quote or bracket, `$`, `\`, …). So `3.14`, `1.2e-5`, `file.py` never split, and lowercase continuations (`e.g. this`) do not either.
- For a `.`, the preceding word is checked. Titles and reference abbreviations (`Mr.`, `Dr.`, `vs.`, `Fig.`, `e.g.`) never end a sentence. List numbers and option labels (`1.`, `A.`) are not ends. Dotted acronyms and initials only split when a typical sentence opener follows.
- Newlines are clues, not automatic boundaries: a blank line splits unless the sentence clearly continues; a single newline splits only before list, option or heading lines, or after a tag-only line such as `<think>`.
- Markers go before the first character of each sentence; surrounding whitespace is never touched.
- **A line ending in `:` never ends a segment.** Following lines (equations, code, list items) stay in the same segment until ordinary prose resumes.
- **Very short fillers** (`Hmm.`, `Wait, no.`, `Okay, so.`; at most 3 words, all from a small interjection list `FILLER_WORDS`) are merged into the following sentence, even across a blank line.

**Quality controls.** The notebook includes acceptance tests (exact text preservation plus expected markers), error reporting with the offending row index (nothing skipped silently), a side-by-side preview of original and segmented text (random rows plus edge cases), and a `show_rows([...])` helper for manual inspection before saving.

**Supported formats.** `.csv`, `.tsv`, `.parquet` for input and output; the default output is next to the input with a `_segmented` suffix.

---

## 5. Stage 1: Segment labeling

**Labeler.** Qwen3.5-9B (4-bit AWQ, `lcu0312/Qwen3.5-9B-AWQ-4bit`) served by **vLLM**, one server per GPU on 2 × Tesla T4 (data parallel), run on Kaggle. Fallback options in the config: Qwen3.5-9B fp16 (tensor-parallel over both T4s) and Qwen3-8B AWQ.

**Serving details.**

- Continuous batching, prefix caching and a shared KV-cache pool; traces come from one shared queue served by both GPUs.
- Launch profiles tried in order: `fast` (prefix caching + CUDA graphs), `no-prefix-cache`, `eager`.
- `MAX_MODEL_LEN = 16384`, `DTYPE = half`, `GPU_MEMORY_UTILIZATION = 0.90`.
- Output is constrained by a **JSON schema** (grammar-guided decoding), so labels are always valid JSON with allowed values. Thinking mode is disabled (`enable_thinking: False`).
- Each trace is sent whole; only segments that are *not* being labeled are shortened when the context is too long. Every response is validated (right count, right ids, in order, valid enum values) with retries.
- Checkpoint line after every trace; output CSVs rewritten every 300 seconds.
- `SECOND_HALF_ONLY=True` was the default in the notebook (labels only the second half of the input CSV, matching the earlier llama.cpp version).

**Label scheme.** Each segment receives four labels, judged by *function in context*, never by keywords.

| Label | Values | Meaning |
|---|---|---|
| `discourse_role` | `setup` | Reading the input: restating the question, givens, options or tools, re-reading the problem. No fact or method applied yet. |
| | `reasoning` | Any step toward the answer: recalling a fact to apply it, planning, calculating, deducing, evaluating or rejecting an option, writing code, stating a conclusion. |
| | `doubt` | The model voices uncertainty about something it holds. Planning questions and approach switches are *not* doubt. |
| | `verification` | A check actually performed (or opened) on something already derived: recomputing, substituting back, a second method. |
| | `correction` | The model recognises its *own* earlier statement is wrong, rejects it and installs a replacement. |
| `answer_status` | `none` | No answer proposed. |
| | `candidate` | Proposes or endorses a specific answer, even tentatively, or a conclusion later followed by doubt, check, correction or reopening. |
| | `final` | The model commits and nothing after it doubts, re-checks or changes the answer. |
| `answer_value` | string or null | The answer as written in the segment (not normalised); null when status is `none` or the answer is long (code, text). |
| `temporal_role` | `new` | Starts a different move: a new sub-goal, a new approach, or any change of role. The first segment is always `new`. |
| | `continuation` | Continues or completes the move in progress; always has the same role as the previous segment. |

The prompt is a short system prompt plus **one worked example** (a doubt → correction → verification → final trace) sent as a real user/assistant turn. An optional one-line `note` field is available (`CFG.USE_NOTE`).

**Rule of thumb for `candidate` vs `final`:** scan *forward* from a conclusion-sounding segment. If a doubt, verification or correction follows, it is `candidate`. If only restatements follow, it is `final`.

---

## 6. Stage 2: Trajectory analysis (methods)

### 6.1 Unit of analysis

The unit is the **question** (`benchmark::question_id`). All eligible samples of a question are pooled across compute levels (the compute level is kept only as a label and as the panel layout in figures). Every pooled number is built as *samples → question (equal weight) → pooled*. Every confidence interval comes from a bootstrap that resamples **questions** (`N_BOOT = 2000`).

### 6.2 Length

Length is measured in **words** of the thinking text (whitespace tokens after replacing `<br>` with a space and removing zero-width characters). A trace's length is the sum of its segment word counts.

### 6.3 Segment reconstruction

- `thinking_text` is split on `⟦S<n>⟧` markers; annotations are matched **by `segment_id`**, never by position.
- A trace counts as *labeled* only if marker ids and annotation ids form the same set, there are no duplicate ids, the JSON is valid and every segment has a valid `discourse_role`.
- Invalid `answer_status` or `temporal_role` values are set to missing (never guessed).
- Assertions check that segment widths sum to the trace length and that text order is preserved.

### 6.4 Key parameters

| Parameter | Value | Meaning |
|---|---|---|
| `SEED` | 0 | Random seed |
| `MIN_SAMPLES` | 5 | A question needs ≥ 5 eligible samples |
| `LEN_STD_QUANTILE` | 0.85 (default in code; output shows an 85% quantile of 0.979) | Long-vs-short is computed only for questions whose `std(words)/mean(words)` exceeds this quantile |
| `SIDE_FRAC` | 0.25 | "Long"/"short" side = longest/shortest 25% of a question's samples (at least 1, never overlapping) |
| `N_BOOT` | 2000 | Question-level bootstrap resamples |
| `REL_BINS` | 10 | Relative-position deciles |
| `CMP_LEN_SIG_ONLY` | True | Group comparison uses only length-significant ALL-CORRECT/ALL-WRONG questions |
| `SHOW_FRAC` | 0.75 | Share of each group's questions whose trajectory figure is displayed in the notebook |
| `N_SAVE_PER_GROUP` | 5 | Example questions saved to disk per group |

### 6.5 Question groups

| Group | Rule (eligible samples only) | Comparison inside the question |
|---|---|---|
| ALL-CORRECT | every sample correct | long vs short samples |
| MIXED | at least one correct and one incorrect | incorrect vs correct samples |
| ALL-WRONG | every sample incorrect | long vs short samples |
| TOO-FEW | fewer than 5 eligible samples | not analysed |

### 6.6 Features (computed on eligible samples only)

*Length-free* features do not change just because a trace is longer. *Length-dependent* features are listed separately. Text features are **cues and similarity scores, not labels**.

**Role composition**
- `share_setup`, `share_reasoning`, `share_doubt`, `share_verification`, `share_correction`: share of words per role.

**Phase structure** (a *phase* is a run of consecutive segments with the same role)
- `changes_per100w`: role changes per 100 words.
- `runs100_doubt`, `runs100_verification`, `runs100_correction`: phase runs per 100 words.
- `doubt_to_verif100`, `doubt_to_corr100`, `verif_to_doubt100`: specific transitions per 100 words.
- `first_doubt_rel`, `first_verification_rel`, `first_correction_rel`: relative position of the first entry into a role (missing if never entered).

**Answer development** (positions measured at the *end* of the segment that first carries the event)
- `first_ans_rel`, `first_final_rel`, `stab_rel` (position where the last answer value becomes stable).
- `share_after_first_ans`, `share_after_stab`: share of the trace after those points.
- `n_replace_100w`, `any_replace`: observed answer replacements (a changed normalised value that is not a refinement such as `x=3` → `3`).
- Answer pattern (`pat_*`): `stable`, `answer replaced`, `gold candidate later replaced (outcome-linked)`, `no answer value`.

**Text cues**
- `hedge_100w`: hedge phrases per 100 words (regex over *wait, hmm, maybe, actually, let me check/double-check/verify, alternatively, or perhaps, on the other hand*).
- `novelty`: share of content words not seen earlier in the trace.
- `rep4_share`: share of word 4-grams already seen earlier in the trace.
- `sent_repeat_share`: share of sentences (≥ 4 tokens) that repeat an earlier sentence exactly.

**Length-dependent**
- `words`, `log_words`, `n_seg`, `mean_seg_words`.

### 6.7 Statistics

For each comparison, every question contributes one row per side (mean over the samples of that side). The per-question difference *A − B* is summarised by:

- the mean difference with a **bootstrap 95% CI** (resampling questions),
- the paired effect size **dz** (mean / SD of the question-level differences) with its bootstrap CI,
- a **sign-flip permutation p-value** and a **Wilcoxon p-value**.

All p-values are **exploratory with no multiplicity correction**. In the result tables, `sig` means the bootstrap CI of the mean difference excludes 0.

### 6.8 Figures

**Per-question trajectory figure.** One panel per compute level; each panel shows only the samples of that level. Each segment is a block whose **width is its word count**, drawn in original text order and never rescaled per trace. Colour is the discourse role (same in every figure). Rows are ordered by total words (shortest at the top). Markers: first answer candidate (triangle), first `final` (diamond), observed replacement or refinement. A black bar marks the start of every `new` segment.

**Overall figure (six panels a to f)** for each comparison:

| Panel | Content |
|---|---|
| (a), (b) | Role composition along relative trace position (deciles) for each side |
| (c) | Difference in role share between the two sides by position, with 95% question-bootstrap band |
| (d) | 12 largest length-free differences as paired effect sizes dz (orange = CI excludes 0) |
| (e) | Phase-transition probability difference between the sides (`*` = CI excludes 0) |
| (f) | Answer-development pattern shares per side |

**Colour code:** setup = grey, reasoning = blue, doubt = orange, verification = green, correction = red.

---

## 7. Data and eligibility

### 7.1 Dataset

- **9,388 rows**, 20 columns; 9 benchmarks; 7 compute levels (0, 256, 512, 1024, 2048, 4096, 8192).
- Samples per question before exclusions: mean 34.8 (min 31, max 35).
- `is_correct`: 4,265 correct and 5,123 incorrect. `was_truncated`: 1,008 truncated.

| Benchmark | Rows |
|---|---|
| MATH-full | 1,050 |
| BFCL-v3-simple | 1,050 |
| IFEval | 1,050 |
| MMLU-Redux | 1,050 |
| GSM8K | 1,050 |
| MMLU-Pro | 1,048 |
| LiveCodeBench-v6 | 1,033 |
| GPQA-Diamond | 1,031 |
| AIME2025 | 1,026 |

| Compute level | Rows |
|---|---|
| 0, 256, 512, 1024, 2048, 4096 | 1,350 each |
| 8192 | 1,288 |

### 7.2 Data quality checks

| Check | Rows | Share |
|---|---|---|
| Rows total | 9,388 | 100% |
| Duplicate (benchmark, question_id, compute_level, sample_index) key | 0 | 0% |
| Empty `thinking_text` | 1,350 | 14.4% |
| Empty `rtl_labels_json` | 1,350 | 14.4% |
| Correctness / truncation / gold unknown | 0 | 0% |
| `<br>` tags or zero-width characters | 0 | 0% |
| Malformed or missing markers | 0 | 0% |
| Same `thinking_text` twice inside one question | 1,150 | 12.2% |

*(Interpretation)* The 1,350 empty traces equal exactly one compute level's row count, which is consistent with the compute-level-0 condition having no thinking text. The notebook does not state this.

### 7.3 Label reconstruction checks

| Check | Traces | Share |
|---|---|---|
| JSON missing / empty | 1,350 | 14.4% |
| JSON malformed | 34 | 0.36% |
| Segment count in text ≠ annotation count | 34 | 0.36% |
| Segment-id set mismatch | 34 | 0.36% |
| Duplicate segment ids (text or JSON) | 0 | 0% |
| Unexpected role / status / temporal value | 0 | 0% |
| **Labeled traces (all checks passed)** | **8,004** | **85.3%** |

Total segments over all 9,388 traces: 410,887.

### 7.4 Eligibility funnel

| Step | Removed | Remaining |
|---|---|---|
| All rows | 0 | 9,388 |
| Not a duplicate key | 0 | 9,388 |
| Labeled (valid JSON, ids match, all roles valid) | 1,384 | 8,004 |
| Has words | 0 | 8,004 |
| Correctness known | 0 | 8,004 |
| Truncation status known | 0 | 8,004 |
| Complete (not truncated) | 906 | 7,098 |
| First copy of its thinking text inside the question | 523 | **6,575** |

Final eligible set: **6,575 samples**, **325,773 segments**, **34 features** per sample, 270 questions with at least one eligible sample.

### 7.5 Segment-level label counts (eligible samples)

| Label | Value | Segments |
|---|---|---|
| discourse_role | reasoning | 259,981 |
| | setup | 29,436 |
| | doubt | 24,825 |
| | verification | 10,741 |
| | correction | 790 |
| answer_status | none | 288,154 |
| | candidate | 33,476 |
| | final | 4,143 |
| temporal_role | continuation | 228,340 |
| | new | 97,433 |

Correction is very rare (790 of 325,773 segments, about 0.24%). This is why correction shares in the group comparison are near zero.

---

## 8. Results

### 8.1 Question groups

Questions: 270 in total; 268 with ≥ 5 eligible samples; median 26 samples per question (min 3, max 30). The standardised length spread (`std/mean`) has median 0.648; its 85% quantile is **0.979**, which is the threshold for "significant length spread".

**Table 1. Group sizes** (`tables/question_groups.csv` holds the per-question assignment)

| Group | Questions | Samples | Median samples | Length-significant |
|---|---|---|---|---|
| ALL-CORRECT | 43 | 1,154 | 27 | 6 |
| MIXED | 167 | 4,120 | 26 | 23 |
| ALL-WRONG | 58 | 1,294 | 25 | 12 |
| TOO-FEW | 2 | 7 | 3 | 0 |

**Table 2. Questions by difficulty tier**

| Tier | ALL-CORRECT | MIXED | ALL-WRONG |
|---|---|---|---|
| easy | 26 | 59 | 5 |
| medium | 14 | 54 | 22 |
| hard | 3 | 54 | 31 |

Group membership tracks difficulty as expected: ALL-CORRECT is mostly easy, ALL-WRONG mostly hard. This is the main reason cross-group comparisons are confounded (Section 8.5).

**Example questions saved to disk**

| Group | Questions |
|---|---|
| ALL-CORRECT | `MMLU-Pro::MMLU-Pro_26`, `MMLU-Redux::MMLU-Redux_18`, `MMLU-Redux::MMLU-Redux_20`, `IFEval::IFEval_18`, `IFEval::IFEval_10` |
| MIXED | `MMLU-Pro::MMLU-Pro_1`, `MMLU-Pro::MMLU-Pro_5`, `MATH-full::MATH-full_27`, `GSM8K::GSM8K_8`, `MMLU-Pro::MMLU-Pro_23` |
| ALL-WRONG | `AIME2025::AIME2025_14`, `LiveCodeBench-v6::LiveCodeBench-v6_9`, `LiveCodeBench-v6::LiveCodeBench-v6_12`, `LiveCodeBench-v6::LiveCodeBench-v6_14`, `LiveCodeBench-v6::LiveCodeBench-v6_21` |

---

### 8.2 ALL-CORRECT

**Setting.** 43 questions, of which **6** have a significant length spread (`std/mean` > 0.979) and enter long-vs-short. The median long/short length ratio is **5.66**. Figures displayed in the notebook: 33 of 43 (77%); 5 saved to disk.

#### 8.2.1 Overall figure

![Figure 1. ALL-CORRECT long vs short](src/results/r1-llama8b/textual_analysis/figures/overall_ALL-CORRECT_long_vs_short.png)

*Figure 1: `src/results/r1-llama8b/textual_analysis/figures/overall_ALL-CORRECT_long_vs_short.png` (n = 6 questions).*

Reading the panels:

- **(a) long:** after an initial setup phase, reasoning dominates, with verification (green) and doubt (orange) appearing from roughly 20% of the trace onward. Verification forms a visible band in the second half. *(read from figure)*
- **(b) short:** setup then pure reasoning; no doubt, verification or correction at all.
- **(c) role-share difference:** verification is consistently higher in long traces. Reasoning is higher early (about +0.2 around 10 to 20%) and lower late (about −0.2 around 70 to 90%). *(read from figure)*
- **(d) effect sizes:** hedging, verification runs, share of text after the first answer and share after stabilisation are higher in long samples; novelty and the relative position of the first answer are lower.
- **(e) transitions (long − short):** `START → setup` +0.17\*, `START → reasoning` −0.17\*, `reasoning → doubt` +0.27\*, `reasoning → verification` +0.29\*, `reasoning → END` −0.68\*. Long traces rarely end directly after reasoning; they pass through doubt or verification first.
- **(f) answer development:** short samples are 100% *stable*. Long samples are about 0.89 stable, with a small share of replaced answers and about 0.08 with no answer value *(read from figure)*.

#### 8.2.2 Largest within-question differences

**Table 3. ALL-CORRECT long vs short** (top 15 rows printed in the notebook; the full table is `src/results/r1-llama8b/textual_analysis/tables/ALL-CORRECT_long_vs_short.csv`). A = long, B = short, 6 questions per row.

| Feature | mean A | mean B | mean diff | CI lo | CI hi | dz | p sign-flip | sig |
|---|---|---|---|---|---|---|---|---|
| log_words | 5.770 | 4.190 | 1.580 | 1.245 | 1.872 | 3.777 | 0.027 | True |
| hedge_100w | 0.907 | 0.000 | 0.907 | 0.686 | 1.078 | 3.381 | 0.029 | True |
| words | 415.754 | 66.750 | 349.004 | 258.258 | 437.972 | 2.795 | 0.029 | True |
| novelty | 0.513 | 0.691 | −0.178 | −0.226 | −0.129 | −2.683 | 0.027 | True |
| runs100_verification | 0.237 | 0.000 | 0.237 | 0.174 | 0.299 | 2.604 | 0.029 | True |
| n_seg | 37.452 | 4.877 | 32.575 | 23.317 | 43.611 | 2.445 | 0.033 | True |
| stab_rel | 0.669 | 0.960 | −0.291 | −0.375 | −0.202 | −2.387 | 0.027 | True |
| share_after_stab | 0.331 | 0.040 | 0.291 | 0.202 | 0.380 | 2.387 | 0.030 | True |
| first_ans_rel | 0.665 | 0.960 | −0.295 | −0.384 | −0.200 | −2.386 | 0.033 | True |
| share_after_first_ans | 0.335 | 0.040 | 0.295 | 0.199 | 0.384 | 2.386 | 0.035 | True |
| share_verification | 0.087 | 0.000 | 0.087 | 0.055 | 0.124 | 1.808 | 0.034 | True |
| doubt_to_verif100 | 0.060 | 0.000 | 0.060 | 0.038 | 0.085 | 1.740 | 0.030 | True |
| runs100_doubt | 0.162 | 0.000 | 0.162 | 0.092 | 0.238 | 1.632 | 0.030 | True |
| share_doubt | 0.018 | 0.000 | 0.018 | 0.009 | 0.028 | 1.453 | 0.023 | True |
| pat_stable | 0.889 | 1.000 | −0.111 | −0.167 | −0.056 | −1.291 | 0.134 | True |

**Reading.** In ALL-CORRECT questions with a wide length spread, short correct samples state the answer almost at the end (first answer at 96% of the trace) and stop. Long correct samples reach a first answer at about 67% and spend the remaining third on doubt and verification. The model is right in both cases, so the extra text is confirmation rather than rescue.

**Caveat.** Only **6 questions** support this comparison. The CIs exclude 0 for most features, but the smallest achievable sign-flip p-value with n = 6 is limited (the printed values are about 0.03), so this is a small-sample pattern.

#### 8.2.3 Example figures (saved)

Per-question trajectory figures for the five saved ALL-CORRECT questions. Each has one panel per compute level, segments as blocks of width = word count, colour = role, black bars at `new` segments, markers at first answer candidate and first final.

![Figure 2a. IFEval_10](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_10.png)

*Figure 2a: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_10.png`*

![Figure 2b. IFEval_18](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_18.png)

*Figure 2b: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_18.png`*

![Figure 2c. MMLU-Pro_26](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Pro-MMLU-Pro_26.png)

*Figure 2c: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Pro-MMLU-Pro_26.png`*

![Figure 2d. MMLU-Redux_18](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_18.png)

*Figure 2d: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_18.png`*

![Figure 2e. MMLU-Redux_20](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_20.png)

*Figure 2e: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_20.png`*

---

### 8.3 MIXED

**Setting.** 167 questions, no length filter (comparison is by outcome). Incorrect samples are longer on average in only **31%** of questions; the median length ratio incorrect/correct is **0.84**. Figures displayed in the notebook: 126 of 167 (75%); 5 saved to disk.

#### 8.3.1 Overall figure

![Figure 3. MIXED incorrect vs correct](src/results/r1-llama8b/textual_analysis/figures/overall_MIXED_incorrect_vs_correct.png)

*Figure 3: `src/results/r1-llama8b/textual_analysis/figures/overall_MIXED_incorrect_vs_correct.png` (n = 167 questions).*

Reading the panels:

- **(a), (b):** both outcomes look alike: setup, then a long reasoning phase, with doubt and some verification growing toward the end. Correct traces show somewhat more verification late.
- **(c):** differences are small (within about ±0.06). Incorrect traces have a lower reasoning share in the first 20% (about −0.06) because they spend more of their early length on setup; verification is slightly lower in incorrect traces late in the trace.
- **(d):** novelty and share of setup are higher for incorrect samples; hedge cues, answer replacement, verification and repetition measures are lower. `first_correction_rel`, `verif_to_doubt100` and `first_final_rel` have CIs that include 0 (grey).
- **(e):** all transition differences are small (|Δ| ≤ 0.06). Significant ones: `reasoning → setup` −0.02\*, `reasoning → verification` −0.02\*, `reasoning → END` +0.05\*, `doubt → setup` −0.02\*.
- **(f):** the answer patterns of correct and incorrect samples (see Table 5) differ mainly in the "no answer value" and "replaced" shares.

#### 8.3.2 Largest within-question differences

**Table 4. MIXED incorrect vs correct** (A = incorrect, B = correct; full table in `src/results/r1-llama8b/textual_analysis/tables/MIXED_incorrect_vs_correct.csv`)

| Feature | n | mean A | mean B | mean diff | CI lo | CI hi | dz | p sign-flip | sig |
|---|---|---|---|---|---|---|---|---|---|
| log_words | 167 | 6.045 | 6.291 | −0.246 | −0.338 | −0.159 | −0.414 | 0.000 | True |
| novelty | 167 | 0.534 | 0.499 | 0.035 | 0.019 | 0.051 | 0.328 | 0.000 | True |
| pat_noanswer | 167 | 0.461 | 0.359 | 0.102 | 0.056 | 0.152 | 0.321 | 0.000 | True |
| n_seg | 167 | 49.649 | 63.100 | −13.452 | −20.732 | −6.848 | −0.304 | 0.000 | True |
| rep4_share | 167 | 0.078 | 0.094 | −0.016 | −0.024 | −0.008 | −0.303 | 0.000 | True |
| words | 167 | 735.581 | 908.035 | −172.454 | −273.139 | −85.834 | −0.285 | 0.001 | True |
| sent_repeat_share | 167 | 0.008 | 0.014 | −0.006 | −0.009 | −0.003 | −0.272 | 0.000 | True |
| share_verification | 167 | 0.016 | 0.025 | −0.009 | −0.014 | −0.003 | −0.237 | 0.002 | True |
| pat_replaced | 167 | 0.157 | 0.221 | −0.064 | −0.108 | −0.021 | −0.225 | 0.004 | True |
| runs100_verification | 167 | 0.051 | 0.071 | −0.019 | −0.033 | −0.007 | −0.215 | 0.005 | True |
| any_replace | 167 | 0.177 | 0.238 | −0.061 | −0.108 | −0.015 | −0.204 | 0.009 | True |
| share_setup | 167 | 0.160 | 0.142 | 0.018 | 0.003 | 0.031 | 0.190 | 0.014 | True |
| hedge_100w | 167 | 1.246 | 1.337 | −0.091 | −0.166 | −0.016 | −0.185 | 0.022 | True |
| first_doubt_rel | 142 | 0.403 | 0.369 | 0.033 | 0.001 | 0.063 | 0.178 | 0.037 | True |
| first_final_rel | 92 | 0.987 | 0.991 | −0.004 | −0.010 | 0.000 | −0.176 | 0.101 | False |

**Reading.** Effect sizes are small to moderate (|dz| ≤ 0.41), but they are estimated on 167 questions, so the CIs are tight.

- Incorrect samples are **shorter** (about 736 vs 908 words on average), with fewer segments.
- Incorrect samples are **less repetitive** (4-gram and sentence repetition) and carry **more new vocabulary** (novelty 0.534 vs 0.499).
- Incorrect samples have **less verification** and **fewer answer replacements**, and are more often **without any annotated answer value** (0.461 vs 0.359).
- Incorrect samples spend a larger share on setup and first voice doubt slightly later.

*(Interpretation)* One reading is that failures in MIXED questions are often early, thin traces that do not check their work, rather than long runaway traces. Section 10 lists a selection effect (truncated samples are excluded) that may partly drive this.

#### 8.3.3 Example figures (saved)

![Figure 4a. GSM8K_8](src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/GSM8K-GSM8K_8.png)

*Figure 4a: `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/GSM8K-GSM8K_8.png`*

![Figure 4b. MATH-full_27](src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MATH-full-MATH-full_27.png)

*Figure 4b: `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MATH-full-MATH-full_27.png`*

![Figure 4c. MMLU-Pro_1](src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_1.png)

*Figure 4c: `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_1.png`*

![Figure 4d. MMLU-Pro_23](src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_23.png)

*Figure 4d: `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_23.png`*

![Figure 4e. MMLU-Pro_5](src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_5.png)

*Figure 4e: `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_5.png`*

---

### 8.4 ALL-WRONG

**Setting.** 58 questions, of which **12** have a significant length spread. The median long/short length ratio is **16.51**. Figures displayed in the notebook: 44 of 58 (76%); 5 saved to disk.

#### 8.4.1 Overall figure

![Figure 5. ALL-WRONG long vs short](src/results/r1-llama8b/textual_analysis/figures/overall_ALL-WRONG_long_vs_short.png)

*Figure 5: `src/results/r1-llama8b/textual_analysis/figures/overall_ALL-WRONG_long_vs_short.png` (n = 12 questions).*

Reading the panels:

- **(a) long:** almost no setup after the first few percent; the trace is reasoning with a persistent orange (doubt) band throughout and little verification.
- **(b) short:** a long setup phase (about 40% of the trace) followed by reasoning, with a thin doubt band.
- **(c):** long − short reasoning share is about +0.65 in the first deciles and falls to about 0 by 50 to 60% of the trace. *(read from figure)* *(Interpretation)* Setup length is roughly constant in absolute words, so in a very long trace it occupies a negligible relative share.
- **(d):** long samples have more text after the first answer, more 4-gram and sentence repetition, a higher reasoning share, and lower novelty, setup share and first-doubt position. Several CIs are very wide (for example `share_after_first_ans`, `first_ans_rel`), reflecting few questions with that feature defined.
- **(e):** `reasoning → doubt` +0.49\*, `reasoning → verification` +0.04\*, `reasoning → END` −0.50\*. The `correction → reasoning` +0.5 cell and the empty `verification` row come from very few observations.
- **(f):** short samples are mostly "no answer value" (about 0.92); long samples have a stable answer in about 0.24 and a replaced answer in about 0.29 *(replaced read from figure)*.

#### 8.4.2 Largest within-question differences

**Table 5. ALL-WRONG long vs short** (A = long, B = short; full table in `src/results/r1-llama8b/textual_analysis/tables/ALL-WRONG_long_vs_short.csv`)

| Feature | n | mean A | mean B | mean diff | CI lo | CI hi | dz | p sign-flip | sig |
|---|---|---|---|---|---|---|---|---|---|
| log_words | 12 | 8.032 | 5.416 | 2.616 | 2.263 | 2.895 | 4.438 | 0.001 | True |
| novelty | 12 | 0.253 | 0.665 | −0.411 | −0.473 | −0.339 | −3.308 | 0.001 | True |
| first_ans_rel | 3 | 0.511 | 0.785 | −0.213 | −0.261 | −0.138 | −3.215 | 0.253 | True |
| share_after_first_ans | 3 | 0.489 | 0.215 | 0.213 | 0.138 | 0.261 | 3.215 | 0.257 | True |
| share_setup | 12 | 0.049 | 0.297 | −0.248 | −0.297 | −0.196 | −2.709 | 0.001 | True |
| first_doubt_rel | 8 | 0.136 | 0.471 | −0.384 | −0.477 | −0.285 | −2.599 | 0.010 | True |
| words | 12 | 3826.662 | 240.896 | 3585.767 | 2766.922 | 4333.812 | 2.413 | 0.001 | True |
| rep4_share | 12 | 0.222 | 0.037 | 0.186 | 0.136 | 0.227 | 2.140 | 0.000 | True |
| n_seg | 12 | 215.817 | 14.642 | 201.175 | 148.939 | 252.808 | 2.094 | 0.000 | True |
| share_reasoning | 12 | 0.879 | 0.658 | 0.221 | 0.160 | 0.275 | 2.073 | 0.001 | True |
| sent_repeat_share | 12 | 0.028 | 0.000 | 0.028 | 0.020 | 0.037 | 1.841 | 0.002 | True |
| pat_noanswer | 12 | 0.475 | 0.917 | −0.442 | −0.639 | −0.247 | −1.246 | 0.008 | True |
| verif_to_doubt100 | 12 | 0.008 | 0.000 | 0.008 | 0.003 | 0.013 | 0.850 | 0.017 | True |
| pat_stable | 12 | 0.236 | 0.056 | 0.181 | 0.069 | 0.306 | 0.826 | 0.039 | True |
| runs100_verification | 12 | 0.023 | 0.000 | 0.023 | 0.010 | 0.040 | 0.808 | 0.003 | True |

**Reading.** Long wrong samples are on average about 3,800 words vs about 240 for short wrong samples. They are dominated by reasoning, introduce little new vocabulary and repeat many 4-grams and sentences, which fits looping behaviour. Short wrong samples are mostly traces with no annotated answer value (0.917).

**Caveat.** `first_ans_rel` and `share_after_first_ans` are defined for only **3** questions (the p-values of about 0.25 reflect this); `first_doubt_rel` for 8.

#### 8.4.3 Example figures (saved)

![Figure 6a. AIME2025_14](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/AIME2025-AIME2025_14.png)

*Figure 6a: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/AIME2025-AIME2025_14.png`*

![Figure 6b. LiveCodeBench-v6_12](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_12.png)

*Figure 6b: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_12.png`*

![Figure 6c. LiveCodeBench-v6_14](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_14.png)

*Figure 6c: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_14.png`*

![Figure 6d. LiveCodeBench-v6_21](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_21.png)

*Figure 6d: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_21.png`*

![Figure 6e. LiveCodeBench-v6_9](src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_9.png)

*Figure 6e: `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_9.png`*

---

### 8.5 Group comparison

Five conditions side by side (question-level means over the samples of each condition): **ALL-CORRECT short**, **ALL-CORRECT long**, **MIXED correct**, **MIXED incorrect**, **ALL-WRONG**. ALL-CORRECT and ALL-WRONG are restricted to length-significant questions (n = 6 and n = 12); MIXED uses all 167 questions.

![Figure 7. Group comparison](src/results/r1-llama8b/textual_analysis/figures/overall_group_comparison.png)

*Figure 7: `src/results/r1-llama8b/textual_analysis/figures/overall_group_comparison.png`. Top row: role composition along the trace for each condition. Bottom left: role shares by condition with 95% CIs. Bottom right: answer-development patterns by condition.*

**Table 6. Group comparison** (`src/results/r1-llama8b/textual_analysis/tables/group_comparison.csv`)

| Feature | ALL-CORRECT short | ALL-CORRECT long | MIXED correct | MIXED incorrect | ALL-WRONG |
|---|---|---|---|---|---|
| share_doubt | 0.000 | 0.018 | 0.049 | 0.046 | 0.054 |
| share_verification | 0.000 | 0.087 | 0.025 | 0.016 | 0.002 |
| share_correction | 0.000 | 0.000 | 0.002 | 0.002 | 0.002 |
| share_reasoning | 0.793 | 0.740 | 0.781 | 0.775 | 0.796 |
| changes_per100w | 1.416 | 1.406 | 1.430 | 1.434 | 1.196 |
| first_ans_rel | 0.960 | 0.665 | 0.671 | 0.697 | 0.591 |
| share_after_first_ans | 0.040 | 0.335 | 0.329 | 0.303 | 0.409 |
| n_replace_100w | 0.000 | 0.005 | 0.058 | 0.057 | 0.012 |
| rep4_share | 0.025 | 0.066 | 0.094 | 0.078 | 0.120 |
| hedge_100w | 0.000 | 0.907 | 1.337 | 1.246 | 1.647 |
| log_words | 4.190 | 5.770 | 6.291 | 6.045 | 6.697 |

**Reading.**

- Doubt share rises from 0 (ALL-CORRECT short) to about 0.05 in MIXED and ALL-WRONG.
- **Verification share is highest in ALL-CORRECT long (0.087)** and drops across MIXED correct (0.025), MIXED incorrect (0.016) and ALL-WRONG (0.002): verification is where the groups differ most in the *role* mix.
- Reasoning share is nearly constant (0.74 to 0.80).
- Repetition (`rep4_share`) and hedge rate increase from ALL-CORRECT short to ALL-WRONG (0.025 → 0.120 and 0.000 → 1.647).
- ALL-WRONG spends the largest share after the first answer (0.409) but with the lowest rate of role changes (1.196 per 100 words), which fits long looping text.

**Caveat (stated in the notebook).** ALL-CORRECT, MIXED and ALL-WRONG contain *different questions* of different difficulty (Table 2). Differences *between* groups show where the groups sit relative to each other, not what causes an outcome. **Only ALL-CORRECT short vs long (and MIXED incorrect vs correct, and ALL-WRONG long vs short, each within-question) are within-question contrasts.**

---

### 8.6 Annotation consistency

Sanity check of the answer annotations: `is_correct` vs "last annotated answer equals gold".

**Table 7. Agreement of correctness with the last annotated answer**

| | last answer ≠ gold | last answer = gold |
|---|---|---|
| is_correct = 0 | 1,651 | 92 |
| is_correct = 1 | 912 | 1,449 |

- Traces with an annotated answer value: **4,104 of 6,575**.
- Agreement between `is_correct` and (last annotated answer = gold): **0.755**.
- Traces with a `final` segment: **44.3%**; `final` before any `candidate`: 3; first segment labeled `continuation`: 0.

**Reading.** Annotation agreement is moderate (75.5%). The 912 correct traces whose last annotated value differs from gold are likely a mix of formatting differences between the annotated string and the gold answer, label errors, and answers carried in text not captured as a value. The notebook does not separate these causes. This is why answer-based features are described as cues, and why the `gold_*` patterns are treated with care.

---

## 9. Cross-cutting findings

1. **Length and structure.** Long samples are *not* just short samples stretched out. In ALL-CORRECT they add doubt and verification phases; in ALL-WRONG they add repetition and low-novelty reasoning.
2. **Correct vs incorrect inside the same question** (MIXED): incorrect samples are shorter, check less, revise their answer less and more often end without a clear answer value. The effects are small to moderate (|dz| ≤ 0.41).
3. **Verification is the most outcome-linked role.** It is highest in ALL-CORRECT long, lower in MIXED correct than in ALL-CORRECT long, lower still in MIXED incorrect, and nearly absent in ALL-WRONG (0.087 → 0.025 → 0.016 → 0.002). Because the groups contain different questions, this is a descriptive gradient, not evidence that verification causes success.
4. **Novelty vs repetition.** Novelty falls and repetition rises with length in both ALL-CORRECT and ALL-WRONG, but only ALL-WRONG shows heavy 4-gram repetition (0.222).
5. **Correction is rare** (790 segments in total, share 0.002), so correction-based features carry little signal in this dataset.

---

## 10. Limitations and caveats

- **Observational only.** No causal claim; no multiplicity correction on p-values (the notebook marks them exploratory).
- **Small subsets for the length contrasts.** Only 6 (ALL-CORRECT) and 12 (ALL-WRONG) questions pass the length-spread filter; several features exist for as few as 3 questions.
- **Cross-group confounding by difficulty** (Table 2). Cross-group differences are not within-question effects.
- **Selection by eligibility.** The funnel drops 906 truncated samples and 1,384 unlabeled samples (14.4% of rows had no thinking text or labels). *(Interpretation)* Truncated samples are likely long and more often incorrect, so excluding them may shrink the length of incorrect samples in MIXED and ALL-WRONG, and may partly explain "incorrect samples are shorter". The notebook does not test this.
- **Pooling across compute levels.** Compute level is only a layout label in figures; effects that depend on compute level are not separated in the overall tables.
- **Duplicate traces.** 523 samples were removed because the same thinking text appeared under several compute levels in the same question (only the first copy is kept).
- **LLM labels.** Labels come from a 9B quantised model with a single worked example; no inter-annotator or human validation is reported in these notebooks. Agreement between `is_correct` and the last annotated answer is 75.5%.
- **Temporal role.** The labeler schema allows only `new` and `continuation`; the analysis notebook also supports a `repetition` value for drawing repetition arrows, but no such value occurs, so **0 repetition arrows** are drawn. Repetition is therefore measured only through text cues (`rep4_share`, `sent_repeat_share`).
- **Threshold wording.** The notebook's introductory text mentions a 75% quantile and the config default is 0.85; the printed output uses the 85% quantile (0.979). The results in this report follow the printed output.
- **Segmentation limits.** Rule-based sentence splitting can mis-split edge cases such as code, equations and unusual abbreviations; this affects segment counts and phase boundaries, not word counts.
- **Word-based length.** Length is counted in words, not model tokens.

---

## 11. Output files (tree)

```
~/Documents/Orange R&D Internship/TTCA/src/results$ tree .

└── r1-llama8b
    ├── src/results/r1-llama8b/textual_analysis
    │   ├── figures
    │   │   ├── examples
    │   │   │   ├── ALL-CORRECT
    │   │   │   │   ├── IFEval-IFEval_10.png
    │   │   │   │   ├── IFEval-IFEval_18.png
    │   │   │   │   ├── MMLU-Pro-MMLU-Pro_26.png
    │   │   │   │   ├── MMLU-Redux-MMLU-Redux_18.png
    │   │   │   │   └── MMLU-Redux-MMLU-Redux_20.png
    │   │   │   ├── ALL-WRONG
    │   │   │   │   ├── AIME2025-AIME2025_14.png
    │   │   │   │   ├── LiveCodeBench-v6-LiveCodeBench-v6_12.png
    │   │   │   │   ├── LiveCodeBench-v6-LiveCodeBench-v6_14.png
    │   │   │   │   ├── LiveCodeBench-v6-LiveCodeBench-v6_21.png
    │   │   │   │   └── LiveCodeBench-v6-LiveCodeBench-v6_9.png
    │   │   │   └── MIXED
    │   │   │       ├── GSM8K-GSM8K_8.png
    │   │   │       ├── MATH-full-MATH-full_27.png
    │   │   │       ├── MMLU-Pro-MMLU-Pro_1.png
    │   │   │       ├── MMLU-Pro-MMLU-Pro_23.png
    │   │   │       └── MMLU-Pro-MMLU-Pro_5.png
    │   │   ├── overall_ALL-CORRECT_long_vs_short.png
    │   │   ├── overall_ALL-WRONG_long_vs_short.png
    │   │   ├── overall_group_comparison.png
    │   │   └── overall_MIXED_incorrect_vs_correct.png
    │   └── tables
    │       ├── ALL-CORRECT_long_vs_short.csv
    │       ├── ALL-WRONG_long_vs_short.csv
    │       ├── group_comparison.csv
    │       ├── MIXED_incorrect_vs_correct.csv
    │       └── question_groups.csv
    └── src/results/r1-llama8b/textual_analysis.zip
```

Sizes reported by the notebook: overall figures 223 to 275 KB; example figures 172 to 293 KB; CSV tables 4 to 34 KB.

---

## 12. Index of figures and tables

### Figures (paths relative to `r1-llama8b/`)

| Ref | File | Section |
|---|---|---|
| Figure 1 | `src/results/r1-llama8b/textual_analysis/figures/overall_ALL-CORRECT_long_vs_short.png` | 8.2.1 |
| Figure 2a | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_10.png` | 8.2.3 |
| Figure 2b | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_18.png` | 8.2.3 |
| Figure 2c | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Pro-MMLU-Pro_26.png` | 8.2.3 |
| Figure 2d | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_18.png` | 8.2.3 |
| Figure 2e | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_20.png` | 8.2.3 |
| Figure 3 | `src/results/r1-llama8b/textual_analysis/figures/overall_MIXED_incorrect_vs_correct.png` | 8.3.1 |
| Figure 4a | `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/GSM8K-GSM8K_8.png` | 8.3.3 |
| Figure 4b | `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MATH-full-MATH-full_27.png` | 8.3.3 |
| Figure 4c | `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_1.png` | 8.3.3 |
| Figure 4d | `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_23.png` | 8.3.3 |
| Figure 4e | `src/results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_5.png` | 8.3.3 |
| Figure 5 | `src/results/r1-llama8b/textual_analysis/figures/overall_ALL-WRONG_long_vs_short.png` | 8.4.1 |
| Figure 6a | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/AIME2025-AIME2025_14.png` | 8.4.3 |
| Figure 6b | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_12.png` | 8.4.3 |
| Figure 6c | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_14.png` | 8.4.3 |
| Figure 6d | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_21.png` | 8.4.3 |
| Figure 6e | `src/results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_9.png` | 8.4.3 |
| Figure 7 | `src/results/r1-llama8b/textual_analysis/figures/overall_group_comparison.png` | 8.5 |

### Tables

| Ref | CSV file | Content | Section |
|---|---|---|---|
| Table 1 | `src/results/r1-llama8b/textual_analysis/tables/question_groups.csv` | Per-question group assignment and group sizes | 8.1 |
| Table 3 | `src/results/r1-llama8b/textual_analysis/tables/ALL-CORRECT_long_vs_short.csv` | ALL-CORRECT long vs short effects | 8.2.2 |
| Table 4 | `src/results/r1-llama8b/textual_analysis/tables/MIXED_incorrect_vs_correct.csv` | MIXED incorrect vs correct effects | 8.3.2 |
| Table 5 | `src/results/r1-llama8b/textual_analysis/tables/ALL-WRONG_long_vs_short.csv` | ALL-WRONG long vs short effects | 8.4.2 |
| Table 6 | `src/results/r1-llama8b/textual_analysis/tables/group_comparison.csv` | Five-condition comparison | 8.5 |

Tables 2 and 7 are printed in the notebook only (not saved as CSV). In-report tables 3, 4 and 5 show the 15 rows printed in the notebook; the CSV files contain the complete feature lists.

---

## 13. Reproducibility notes

- **Inputs:** `r1-llama8b_raw_results.csv` → (stage 0) `*_segmented.csv` → (stage 1) `*_segmented_labeled.csv` → (stage 2). Stage 2 reads the path from the `TRACE_CSV` environment variable (default `r1-llama8b_raw_results_segmented_labeled.csv`) and writes to `TRACE_OUT` (default `src/results/r1-llama8b/textual_analysis`).
- **Required columns:** `question_id`, `benchmark`, `compute_level`, `sample_index`, `thinking_text`, `gold`, `is_correct`, `was_truncated`, `rtl_labels_json`. Optional: `difficulty_tier`, `task_type`, `rtl_n_segments`.
- **Determinism:** `SEED = 0`; bootstraps use `numpy.random.default_rng`. LLM labeling (stage 1) is not guaranteed to be bit-identical across runs.
- **Environment (stage 1):** Kaggle, 2 × Tesla T4, vLLM, `MAX_MODEL_LEN = 16384`.
- **Environment (stage 2):** Python with `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn` (for English stop words).
- **Archive:** the notebook zips the `src/results/r1-llama8b/textual_analysis` folder into `src/results/r1-llama8b/textual_analysis.zip`.
