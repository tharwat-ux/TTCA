---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 24px; }
  section.small { font-size: 19px; }
  table { font-size: 0.8em; }
  img { max-height: 520px; }
---

<!-- Slides are separated by "---". Open with Marp (VS Code extension or marp-cli); it also reads as plain Markdown.
Image paths are relative to r1-llama8b/ : place this file in that folder. -->

# H3: Question-Level Reasoning-Trace Trajectories

**Model:** R1-Distill-Llama-8B (`r1-llama8b`)
**Project:** TTCA, Orange R&D Internship
**Notebooks:** `0_sentence_segmentation` · `1_reasoning_trace_labeler` · `2_reasoning_trace_trajectory_analysis`

> Assumption: the notebooks never say "H3". H3 is treated here as the question-level trajectory analysis. Values marked *(read from figure)* are approximate; *(interpretation)* is my reading, not computed.

---

# Agenda

1. Summary
2. Hypothesis and research questions
3. Pipeline (stages 0, 1, 2)
4. Methods
5. Data and eligibility
6. Results: ALL-CORRECT, MIXED, ALL-WRONG, group comparison
7. Annotation consistency
8. Cross-cutting findings
9. Limitations
10. Output files, index, reproducibility

---

# 1. Summary

- Traces were segmented (stage 0), labeled per segment by an LLM (stage 1), analysed **per question** (stage 2).
- **6,575 eligible samples**, **268 questions** (≥ 5 eligible samples each).
- Groups: **ALL-CORRECT (43)**, **MIXED (167)**, **ALL-WRONG (58)**.
- Observational only. No causal claims.

---

# 1. Summary: main observations

- **ALL-CORRECT:** long correct samples add doubt and verification (hedge cues 0.907 vs 0.000 per 100 words in short ones).
- **MIXED:** incorrect samples are *shorter* (median ratio 0.84; longer in only 31% of questions), check less, replace answers less, more often have no answer value.
- **ALL-WRONG:** long wrong samples are ~16.5× longer, highly repetitive (4-gram repetition 0.222 vs 0.037), low novelty.
- Group comparison is **between different questions**: not a within-question effect.

---

# 2. Hypothesis and research questions

**Working formulation of H3 (edit as needed):**
*Within the same question, the temporal structure of a reasoning trace (distribution of setup, reasoning, doubt, verification, correction, and answer development) differs systematically between long and short samples and between correct and incorrect samples.*

| Group | Question asked |
|---|---|
| ALL-CORRECT | When the model is always right, how do long samples differ from short ones? |
| MIXED | Inside one question, how do incorrect samples differ from correct ones? |
| ALL-WRONG | Do long wrong samples differ from short wrong samples? |

---

# 3. Pipeline overview

| Stage | Notebook | Input | Output |
|---|---|---|---|
| 0 | `0_sentence_segmentation` | `*_raw_results.csv` | CSV with `⟦S1⟧ ⟦S2⟧ …` markers |
| 1 | `1_reasoning_trace_labeler` | `r1-llama8b_raw_results_segmented.csv` | `labeled_results.csv`, `labeled_segments.csv`, checkpoint JSONL |
| 2 | `2_reasoning_trace_trajectory_analysis` | `*_segmented_labeled.csv` | Figures, tables, `textual_analysis.zip` |

---

<!-- _class: small -->
# Stage 0: Sentence segmentation

**Goal:** insert `⟦S1⟧ ⟦S2⟧ …` at sentence starts **without changing any original character** (removing markers returns the exact input). Pure Python rules, only `pandas`.

- Split at `. ! ? …` only if whitespace follows and next token looks like a sentence start. `3.14`, `file.py`, `e.g. this` never split.
- Abbreviations (`Mr.`, `Dr.`, `vs.`, `Fig.`), list numbers and option labels are not sentence ends.
- Newlines are clues: blank line splits unless the sentence clearly continues.
- **A line ending in `:` never ends a segment.**
- **Short fillers** (`Hmm.`, `Wait, no.`; ≤ 3 words from `FILLER_WORDS`) merge into the next sentence.
- Controls: acceptance tests, errors report row index, side-by-side preview, `show_rows([...])`.

---

<!-- _class: small -->
# Stage 1: Segment labeling: setup

- **Labeler:** Qwen3.5-9B 4-bit AWQ (`lcu0312/Qwen3.5-9B-AWQ-4bit`) on **vLLM**, one server per GPU, 2 × Tesla T4, Kaggle.
- Fallbacks: Qwen3.5-9B fp16 (tensor-parallel), Qwen3-8B AWQ.
- Continuous batching, prefix caching, shared KV-cache pool, one shared queue.
- Launch profiles: `fast` → `no-prefix-cache` → `eager`.
- `MAX_MODEL_LEN=16384`, `DTYPE=half`, `GPU_MEMORY_UTILIZATION=0.90`.
- **JSON-schema constrained output** (valid JSON, allowed values only); thinking mode off.
- Whole trace sent; only non-labeled segments are shortened if context is too long. Responses validated, with retries.
- Checkpoint after every trace; CSVs rewritten every 300 s.
- Prompt: short system prompt + **one worked example** (doubt → correction → verification → final).

---

<!-- _class: small -->
# Stage 1: Label scheme

| Label | Values and meaning |
|---|---|
| `discourse_role` | **setup** reading the input · **reasoning** any step toward the answer · **doubt** voiced uncertainty · **verification** a check actually performed · **correction** own earlier statement rejected and replaced |
| `answer_status` | **none** · **candidate** tentative or later reopened answer · **final** committed, nothing after doubts it |
| `answer_value` | string or null, copied as written, not normalised |
| `temporal_role` | **new** (new move or any role change; first segment always new) · **continuation** (same role as previous) |

Judged by **function in context**, never by keywords.
Rule of thumb: scan *forward* from a conclusion. Doubt/check/correction follows → `candidate`. Only restatements → `final`.

---

<!-- _class: small -->
# Stage 2: Methods: unit and length

- **Unit = the question** (`benchmark::question_id`). Samples pooled across compute levels.
- Pooling: *samples → question (equal weight) → pooled*.
- Every CI: bootstrap resampling **questions** (`N_BOOT=2000`).
- **Length = words** of thinking text (`<br>` → space, zero-width characters removed).
- Segments matched to annotations **by `segment_id`**, never by position.
- *Labeled* trace = same id sets, no duplicates, valid JSON, every role valid.
- Assertions: segment widths sum to trace length; text order preserved.

---

<!-- _class: small -->
# Stage 2: Parameters and groups

| Parameter | Value |
|---|---|
| `MIN_SAMPLES` | 5 eligible samples per question |
| `LEN_STD_QUANTILE` | 0.85 (printed threshold 0.979) |
| `SIDE_FRAC` | 0.25 (longest / shortest 25%, ≥ 1 sample) |
| `N_BOOT` / `REL_BINS` | 2000 / 10 deciles |
| `SHOW_FRAC` / `N_SAVE_PER_GROUP` | 0.75 / 5 |

| Group | Rule | Compared |
|---|---|---|
| ALL-CORRECT | every sample correct | long vs short |
| MIXED | ≥ 1 correct and ≥ 1 incorrect | incorrect vs correct |
| ALL-WRONG | every sample incorrect | long vs short |

Long/short only for questions with `std(words)/mean(words)` above the quantile.

---

<!-- _class: small -->
# Stage 2: Features

- **Role shares:** `share_setup / reasoning / doubt / verification / correction`
- **Phase structure:** `changes_per100w`, `runs100_*`, `doubt_to_verif100`, `doubt_to_corr100`, `verif_to_doubt100`, `first_*_rel`
- **Answer development:** `first_ans_rel`, `first_final_rel`, `stab_rel`, `share_after_first_ans`, `share_after_stab`, `n_replace_100w`, `any_replace`, patterns `stable / replaced / gold-lost / no answer value`
- **Text cues:** `hedge_100w`, `novelty`, `rep4_share`, `sent_repeat_share`
- **Length-dependent:** `words`, `log_words`, `n_seg`, `mean_seg_words`

Text features are **cues, not labels**. A phase = run of consecutive segments with the same role.

---

# Stage 2: Statistics and figures

**Per comparison:** one row per side per question → difference A − B → mean with bootstrap 95% CI, paired effect size **dz**, sign-flip p, Wilcoxon p.
All p-values **exploratory, no multiplicity correction**. `sig` = CI excludes 0.

**Overall figure panels (a to f):**
(a, b) role composition per side · (c) role-share difference with CI band · (d) 12 largest dz · (e) transition-probability difference · (f) answer-development patterns

**Colours:** setup grey · reasoning blue · doubt orange · verification green · correction red

---

<!-- _class: small -->
# 5. Data

- **9,388 rows**, 20 columns, 9 benchmarks, 7 compute levels (0, 256, 512, 1024, 2048, 4096, 8192).
- Samples per question: mean 34.8 (31 to 35).
- Correct 4,265 · incorrect 5,123 · truncated 1,008.

| Benchmark | Rows | Benchmark | Rows |
|---|---|---|---|
| MATH-full | 1,050 | MMLU-Pro | 1,048 |
| BFCL-v3-simple | 1,050 | LiveCodeBench-v6 | 1,033 |
| IFEval | 1,050 | GPQA-Diamond | 1,031 |
| MMLU-Redux | 1,050 | AIME2025 | 1,026 |
| GSM8K | 1,050 | | |

Compute levels: 1,350 rows each, except 8192 with 1,288.

---

<!-- _class: small -->
# 5. Data quality

| Check | Rows | Share |
|---|---|---|
| Duplicate key | 0 | 0% |
| Empty `thinking_text` / `rtl_labels_json` | 1,350 | 14.4% |
| Missing correctness / truncation / gold | 0 | 0% |
| Same trace twice inside one question | 1,150 | 12.2% |
| JSON malformed / count mismatch / id mismatch | 34 each | 0.36% |
| **Labeled traces** | **8,004** | **85.3%** |

*(Interpretation)* 1,350 empty traces = one compute level's row count, consistent with compute level 0 having no thinking. Not stated in the notebook.

---

# 5. Eligibility funnel

| Step | Removed | Remaining |
|---|---|---|
| All rows | 0 | 9,388 |
| Labeled (valid JSON, ids match) | 1,384 | 8,004 |
| Complete (not truncated) | 906 | 7,098 |
| First copy of its thinking text in question | 523 | **6,575** |

**6,575 samples · 325,773 segments · 34 features · 270 questions**

---

# 5. Segment label counts

| Label | Value | Segments |
|---|---|---|
| role | reasoning | 259,981 |
| | setup | 29,436 |
| | doubt | 24,825 |
| | verification | 10,741 |
| | correction | 790 |
| status | none / candidate / final | 288,154 / 33,476 / 4,143 |
| temporal | continuation / new | 228,340 / 97,433 |

Correction ≈ 0.24% of segments, so correction features carry little signal.

---

# 6.1 Question groups

| Group | Questions | Samples | Median samples | Length-significant |
|---|---|---|---|---|
| ALL-CORRECT | 43 | 1,154 | 27 | 6 |
| MIXED | 167 | 4,120 | 26 | 23 |
| ALL-WRONG | 58 | 1,294 | 25 | 12 |
| TOO-FEW | 2 | 7 | 3 | 0 |

| Tier | ALL-CORRECT | MIXED | ALL-WRONG |
|---|---|---|---|
| easy | 26 | 59 | 5 |
| medium | 14 | 54 | 22 |
| hard | 3 | 54 | 31 |

Group tracks difficulty → cross-group comparisons are confounded. Table: `tables/question_groups.csv`

---

# 6.2 ALL-CORRECT: setting

- 43 questions, **6** with significant length spread (`std/mean` > 0.979).
- Median long/short length ratio: **5.66**.
- 33 of 43 figures displayed in notebook; 5 saved.

![w:900](textual_analysis/figures/overall_ALL-CORRECT_long_vs_short.png)

*Figure 1: `textual_analysis/figures/overall_ALL-CORRECT_long_vs_short.png`*

---

<!-- _class: small -->
# 6.2 ALL-CORRECT: reading Figure 1

- **(a) long:** reasoning dominates; doubt and verification appear from ~20% on.
- **(b) short:** setup then pure reasoning; no doubt, verification or correction.
- **(c):** verification higher in long traces; reasoning ≈ +0.2 early, ≈ −0.2 late *(read from figure)*.
- **(e) transitions long − short:** `START→setup` +0.17\*, `START→reasoning` −0.17\*, `reasoning→doubt` +0.27\*, `reasoning→verification` +0.29\*, `reasoning→END` −0.68\*.
- **(f):** short 100% stable; long ≈ 0.89 stable, ≈ 0.08 no answer value *(read from figure)*.

---

<!-- _class: small -->
# 6.2 ALL-CORRECT: Table 3 (long A vs short B, 6 questions)

| Feature | A | B | diff | CI | dz | sig |
|---|---|---|---|---|---|---|
| log_words | 5.770 | 4.190 | 1.580 | 1.245 to 1.872 | 3.777 | ✓ |
| hedge_100w | 0.907 | 0.000 | 0.907 | 0.686 to 1.078 | 3.381 | ✓ |
| words | 415.8 | 66.8 | 349.0 | 258.3 to 438.0 | 2.795 | ✓ |
| novelty | 0.513 | 0.691 | −0.178 | −0.226 to −0.129 | −2.683 | ✓ |
| runs100_verification | 0.237 | 0.000 | 0.237 | 0.174 to 0.299 | 2.604 | ✓ |
| first_ans_rel | 0.665 | 0.960 | −0.295 | −0.384 to −0.200 | −2.386 | ✓ |
| share_after_first_ans | 0.335 | 0.040 | 0.295 | 0.199 to 0.384 | 2.386 | ✓ |
| share_verification | 0.087 | 0.000 | 0.087 | 0.055 to 0.124 | 1.808 | ✓ |
| pat_stable | 0.889 | 1.000 | −0.111 | −0.167 to −0.056 | −1.291 | ✓ |

Top 15 printed rows (others in `tables/ALL-CORRECT_long_vs_short.csv`: n_seg, stab_rel, doubt_to_verif100, runs100_doubt, share_doubt, …). p sign-flip ≈ 0.03.

---

# 6.2 ALL-CORRECT: takeaway

- Short correct samples state the answer at **96%** of the trace and stop.
- Long correct samples reach a first answer at **~67%**, then spend the last third on doubt and verification.
- The model is right in both: extra text is **confirmation, not rescue**.

**Caveat:** only **6 questions** support this contrast.

---

# 6.2 ALL-CORRECT: example figures

Per-question trajectory: one panel per compute level, block width = word count, colour = role, black bars = `new` segments, markers at first answer candidate and first final.

- Figure 2a: `textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_10.png`
- Figure 2b: `textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_18.png`
- Figure 2c: `textual_analysis/figures/examples/ALL-CORRECT/MMLU-Pro-MMLU-Pro_26.png`
- Figure 2d: `textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_18.png`
- Figure 2e: `textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_20.png`

---

# Figure 2a: ALL-CORRECT, IFEval_10

![h:560](textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_10.png)

---

# Figure 2b: ALL-CORRECT, IFEval_18

![h:560](textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_18.png)

---

# Figure 2c: ALL-CORRECT, MMLU-Pro_26

![h:560](textual_analysis/figures/examples/ALL-CORRECT/MMLU-Pro-MMLU-Pro_26.png)

---

# Figure 2d: ALL-CORRECT, MMLU-Redux_18

![h:560](textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_18.png)

---

# Figure 2e: ALL-CORRECT, MMLU-Redux_20

![h:560](textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_20.png)

---

# 6.3 MIXED: setting

- 167 questions, no length filter (compared by outcome).
- Incorrect longer in only **31%** of questions; median ratio incorrect/correct **0.84**.
- 126 of 167 figures displayed; 5 saved.

![w:900](textual_analysis/figures/overall_MIXED_incorrect_vs_correct.png)

*Figure 3: `textual_analysis/figures/overall_MIXED_incorrect_vs_correct.png`*

---

<!-- _class: small -->
# 6.3 MIXED: reading Figure 3

- **(a, b):** similar shapes; correct traces show a bit more late verification.
- **(c):** differences small (within ±0.06). Incorrect: reasoning ≈ −0.06 in first 20% (more setup).
- **(d):** incorrect higher in novelty and setup share; lower in hedging, replacement, verification, repetition. `first_correction_rel`, `verif_to_doubt100`, `first_final_rel` CIs include 0.
- **(e):** all |Δ| ≤ 0.06. Significant: `reasoning→setup` −0.02\*, `reasoning→verification` −0.02\*, `reasoning→END` +0.05\*, `doubt→setup` −0.02\*.
- **(f):** differences mainly in "no answer value" and "replaced" shares.

---

<!-- _class: small -->
# 6.3 MIXED: Table 4 (incorrect A vs correct B, 167 questions)

| Feature | A | B | diff | dz | sig |
|---|---|---|---|---|---|
| log_words | 6.045 | 6.291 | −0.246 | −0.414 | ✓ |
| novelty | 0.534 | 0.499 | 0.035 | 0.328 | ✓ |
| pat_noanswer | 0.461 | 0.359 | 0.102 | 0.321 | ✓ |
| n_seg | 49.6 | 63.1 | −13.5 | −0.304 | ✓ |
| rep4_share | 0.078 | 0.094 | −0.016 | −0.303 | ✓ |
| words | 735.6 | 908.0 | −172.5 | −0.285 | ✓ |
| sent_repeat_share | 0.008 | 0.014 | −0.006 | −0.272 | ✓ |
| share_verification | 0.016 | 0.025 | −0.009 | −0.237 | ✓ |
| pat_replaced | 0.157 | 0.221 | −0.064 | −0.225 | ✓ |
| any_replace | 0.177 | 0.238 | −0.061 | −0.204 | ✓ |
| share_setup | 0.160 | 0.142 | 0.018 | 0.190 | ✓ |
| hedge_100w | 1.246 | 1.337 | −0.091 | −0.185 | ✓ |
| first_doubt_rel (n=142) | 0.403 | 0.369 | 0.033 | 0.178 | ✓ |
| first_final_rel (n=92) | 0.987 | 0.991 | −0.004 | −0.176 | ✗ |

Top 15 printed rows (also `runs100_verification`; full: `tables/MIXED_incorrect_vs_correct.csv`).

---

# 6.3 MIXED: takeaway

- Effects small to moderate (|dz| ≤ 0.41), tight CIs (167 questions).
- Incorrect samples: **shorter** (~736 vs ~908 words), **less repetitive**, **more new vocabulary**, **less verification**, **fewer answer replacements**, more often **no answer value**.
- *(Interpretation)* Failures here look like early, thin traces that do not check their work, not runaway traces.
- Selection effect: truncated samples are excluded (see limitations).

---

# 6.3 MIXED: example figures

- Figure 4a: `textual_analysis/figures/examples/MIXED/GSM8K-GSM8K_8.png`
- Figure 4b: `textual_analysis/figures/examples/MIXED/MATH-full-MATH-full_27.png`
- Figure 4c: `textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_1.png`
- Figure 4d: `textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_23.png`
- Figure 4e: `textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_5.png`

---

# Figure 4a: MIXED, GSM8K_8

![h:560](textual_analysis/figures/examples/MIXED/GSM8K-GSM8K_8.png)

---

# Figure 4b: MIXED, MATH-full_27

![h:560](textual_analysis/figures/examples/MIXED/MATH-full-MATH-full_27.png)

---

# Figure 4c: MIXED, MMLU-Pro_1

![h:560](textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_1.png)

---

# Figure 4d: MIXED, MMLU-Pro_23

![h:560](textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_23.png)

---

# Figure 4e: MIXED, MMLU-Pro_5

![h:560](textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_5.png)

---

# 6.4 ALL-WRONG: setting

- 58 questions, **12** with significant length spread.
- Median long/short length ratio: **16.51**.
- 44 of 58 figures displayed; 5 saved.

![w:900](textual_analysis/figures/overall_ALL-WRONG_long_vs_short.png)

*Figure 5: `textual_analysis/figures/overall_ALL-WRONG_long_vs_short.png`*

---

<!-- _class: small -->
# 6.4 ALL-WRONG: reading Figure 5

- **(a) long:** almost no setup; reasoning with a persistent doubt band, little verification.
- **(b) short:** setup ≈ 40% of trace, then reasoning.
- **(c):** reasoning share long − short ≈ +0.65 early, ≈ 0 by 50 to 60% *(read from figure)*. *(Interpretation)* setup is roughly constant in absolute words, so negligible in relative terms for very long traces.
- **(d):** long: more text after first answer, more repetition, lower novelty and setup share. Several CIs very wide.
- **(e):** `reasoning→doubt` +0.49\*, `reasoning→verification` +0.04\*, `reasoning→END` −0.50\*. `correction→reasoning` +0.5 rests on very few observations.
- **(f):** short ≈ 0.92 no answer value; long ≈ 0.24 stable *(read from figure)*.

---

<!-- _class: small -->
# 6.4 ALL-WRONG: Table 5 (long A vs short B, 12 questions)

| Feature | A | B | diff | dz | sig |
|---|---|---|---|---|---|
| log_words | 8.032 | 5.416 | 2.616 | 4.438 | ✓ |
| novelty | 0.253 | 0.665 | −0.411 | −3.308 | ✓ |
| first_ans_rel (n=3) | 0.511 | 0.785 | −0.213 | −3.215 | ✓ |
| share_setup | 0.049 | 0.297 | −0.248 | −2.709 | ✓ |
| first_doubt_rel (n=8) | 0.136 | 0.471 | −0.384 | −2.599 | ✓ |
| words | 3826.7 | 240.9 | 3585.8 | 2.413 | ✓ |
| rep4_share | 0.222 | 0.037 | 0.186 | 2.140 | ✓ |
| n_seg | 215.8 | 14.6 | 201.2 | 2.094 | ✓ |
| share_reasoning | 0.879 | 0.658 | 0.221 | 2.073 | ✓ |
| sent_repeat_share | 0.028 | 0.000 | 0.028 | 1.841 | ✓ |
| pat_noanswer | 0.475 | 0.917 | −0.442 | −1.246 | ✓ |
| pat_stable | 0.236 | 0.056 | 0.181 | 0.826 | ✓ |

Top 15 printed rows (also `share_after_first_ans`, `verif_to_doubt100`, `runs100_verification`; full: `tables/ALL-WRONG_long_vs_short.csv`).

---

# 6.4 ALL-WRONG: takeaway

- Long wrong samples ≈ **3,800 words** vs ≈ **240** for short.
- Dominated by reasoning, little new vocabulary, many repeated 4-grams and sentences: consistent with **looping**.
- Short wrong samples are mostly traces with **no annotated answer value** (0.917).

**Caveat:** `first_ans_rel` and `share_after_first_ans` defined for only **3** questions (p ≈ 0.25); `first_doubt_rel` for 8.

---

# 6.4 ALL-WRONG: example figures

- Figure 6a: `textual_analysis/figures/examples/ALL-WRONG/AIME2025-AIME2025_14.png`
- Figure 6b: `textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_12.png`
- Figure 6c: `textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_14.png`
- Figure 6d: `textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_21.png`
- Figure 6e: `textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_9.png`

---

# Figure 6a: ALL-WRONG, AIME2025_14

![h:560](textual_analysis/figures/examples/ALL-WRONG/AIME2025-AIME2025_14.png)

---

# Figure 6b: ALL-WRONG, LiveCodeBench-v6_12

![h:560](textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_12.png)

---

# Figure 6c: ALL-WRONG, LiveCodeBench-v6_14

![h:560](textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_14.png)

---

# Figure 6d: ALL-WRONG, LiveCodeBench-v6_21

![h:560](textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_21.png)

---

# Figure 6e: ALL-WRONG, LiveCodeBench-v6_9

![h:560](textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_9.png)

---

# 6.5 Group comparison

Five conditions, question-level means. ALL-CORRECT (n=6) and ALL-WRONG (n=12) restricted to length-significant questions; MIXED (n=167) not filtered.

![w:950](textual_analysis/figures/overall_group_comparison.png)

*Figure 7: `textual_analysis/figures/overall_group_comparison.png`*

---

<!-- _class: small -->
# 6.5 Group comparison: Table 6

| Feature | AC short | AC long | MIXED correct | MIXED incorrect | ALL-WRONG |
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

`textual_analysis/tables/group_comparison.csv` · AC = ALL-CORRECT

---

# 6.5 Group comparison: reading

- Doubt share: 0 → ~0.05 (MIXED, ALL-WRONG).
- **Verification share** is where role mix differs most: **0.087 → 0.025 → 0.016 → 0.002**.
- Reasoning share almost constant (0.74 to 0.80).
- Repetition and hedging rise toward ALL-WRONG (`rep4_share` 0.025 → 0.120; hedge 0.000 → 1.647).
- ALL-WRONG: most text after first answer (0.409), fewest role changes (1.196/100 w): long looping text.

**Caveat:** groups contain *different questions* of different difficulty. Only the three within-question contrasts support effect claims.

---

# 7. Annotation consistency

| | last answer ≠ gold | last answer = gold |
|---|---|---|
| is_correct = 0 | 1,651 | 92 |
| is_correct = 1 | 912 | 1,449 |

- Traces with an annotated answer value: **4,104 of 6,575**.
- Agreement `is_correct` vs (last annotated answer = gold): **0.755**.
- With a `final` segment: **44.3%**; `final` before any `candidate`: 3; first segment `continuation`: 0.

The 912 mismatches mix formatting differences, label errors and answers not captured as a value (not separated by the notebook). Answer-based features are cues.

---

# 8. Cross-cutting findings

1. **Long ≠ stretched short.** ALL-CORRECT long adds doubt and verification; ALL-WRONG long adds repetition and low-novelty reasoning.
2. **MIXED:** incorrect samples are shorter, check less, revise less, more often lack an answer value (|dz| ≤ 0.41).
3. **Verification gradient:** 0.087 → 0.025 → 0.016 → 0.002. Descriptive, not causal.
4. **Novelty falls, repetition rises with length**; heavy 4-gram repetition (0.222) only in ALL-WRONG.
5. **Correction is rare** (790 segments, share 0.002).

---

<!-- _class: small -->
# 9. Limitations and caveats

- Observational; p-values exploratory, no multiplicity correction.
- Length contrasts rest on **6** and **12** questions; some features on as few as 3.
- Cross-group confounding by difficulty.
- **Selection:** 906 truncated and 1,384 unlabeled samples dropped. *(Interpretation)* truncated samples are likely long and more often wrong, which may partly explain "incorrect is shorter". Untested.
- Compute levels pooled; compute-level effects not separated.
- 523 duplicate traces removed.
- Labels from a 9B quantised model with one worked example; no human validation reported. Answer agreement 75.5%.
- Schema allows only `new`/`continuation`; **0 repetition arrows** drawn; repetition measured via text cues.
- Notebook text says 75% quantile, config 0.85, output uses 0.979: report follows output.
- Rule-based segmentation can mis-split code/equations; length counted in words, not tokens.

---

<!-- _class: small -->
# 10. Output files

```
~/Documents/Orange R&D Internship/TTCA/src/results$ tree .

└── r1-llama8b
    ├── textual_analysis
    │   ├── figures
    │   │   ├── examples
    │   │   │   ├── ALL-CORRECT   (IFEval_10, IFEval_18, MMLU-Pro_26, MMLU-Redux_18, MMLU-Redux_20)
    │   │   │   ├── ALL-WRONG     (AIME2025_14, LiveCodeBench-v6_9/_12/_14/_21)
    │   │   │   └── MIXED         (GSM8K_8, MATH-full_27, MMLU-Pro_1/_5/_23)
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
    └── textual_analysis.zip
```

---

<!-- _class: small -->
# Index: figures and tables

| Ref | File |
|---|---|
| Fig 1 | `textual_analysis/figures/overall_ALL-CORRECT_long_vs_short.png` |
| Fig 2a to 2e | `textual_analysis/figures/examples/ALL-CORRECT/*.png` |
| Fig 3 | `textual_analysis/figures/overall_MIXED_incorrect_vs_correct.png` |
| Fig 4a to 4e | `textual_analysis/figures/examples/MIXED/*.png` |
| Fig 5 | `textual_analysis/figures/overall_ALL-WRONG_long_vs_short.png` |
| Fig 6a to 6e | `textual_analysis/figures/examples/ALL-WRONG/*.png` |
| Fig 7 | `textual_analysis/figures/overall_group_comparison.png` |
| Table 1 | `textual_analysis/tables/question_groups.csv` |
| Table 3 | `textual_analysis/tables/ALL-CORRECT_long_vs_short.csv` |
| Table 4 | `textual_analysis/tables/MIXED_incorrect_vs_correct.csv` |
| Table 5 | `textual_analysis/tables/ALL-WRONG_long_vs_short.csv` |
| Table 6 | `textual_analysis/tables/group_comparison.csv` |

Tables 2 and 7 are notebook printouts only. Tables 3 to 5 show the 15 rows printed in the notebook.

---

<!-- _class: small -->
# Reproducibility

- **Chain:** `r1-llama8b_raw_results.csv` → `*_segmented.csv` → `*_segmented_labeled.csv` → stage 2.
- Stage 2 env vars: `TRACE_CSV` (default `r1-llama8b_raw_results_segmented_labeled.csv`), `TRACE_OUT` (default `textual_analysis`).
- **Required columns:** `question_id`, `benchmark`, `compute_level`, `sample_index`, `thinking_text`, `gold`, `is_correct`, `was_truncated`, `rtl_labels_json`. Optional: `difficulty_tier`, `task_type`, `rtl_n_segments`.
- `SEED=0`; stage-1 LLM labeling is not guaranteed bit-identical across runs.
- Stage 1: Kaggle, 2 × Tesla T4, vLLM, `MAX_MODEL_LEN=16384`.
- Stage 2: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`.

---

# Thank you

**H3: Question-Level Reasoning-Trace Trajectories**
Questions?
