# H3 — Textual Analysis of Thinking Traces

| Field | Value |
|---|---|
| **Document** | TTCA technical report H3, version 2.0 (appendix edition) |
| **Title** | Textual Analysis of Thinking Traces — what the model does inside its reasoning |
| **Model analysed** | `r1-llama8b` (Llama-3.1-8B, R1-style reasoning distillation) |
| **Analysis input** | `src/data/processed/r1-llama8b/r1-llama8b_raw_results_segmented_labeled.csv` (106 MB, 9,388 rows / 270 questions) |
| **Notebooks** | [`0.sentence_segmentation.ipynb`](0.sentence_segmentation.ipynb) → [`1.reasoning_trace_labeler.ipynb`](1.reasoning_trace_labeler.ipynb) → [`2_reasoning_trace_trajectory_analysis.ipynb`](2_reasoning_trace_trajectory_analysis.ipynb) |
| **Results** | `src/results/r1-llama8b/textual_analysis/` — 4 overview figures, 15 example figures, 5 tables (bundled as `textual_analysis.zip`) |
| **Analysis type** | Observational / descriptive. No LLM inference in the analysis stage, no new labels, no causal claims. |
| **Status** | Final — frozen at the committed artefacts; numbers taken from the saved result tables (see §9) |
| **Supersedes** | `H3.md` (v1.0, inline-table edition); identical content, different packaging |

---

## Abstract

This report answers hypothesis H3 of the TTCA project: *what does the model actually do inside its thinking trace, and how does that inner behaviour differ between longer and shorter samples of the same question, between correct and incorrect samples of the same question, and between the three outcome groups ALL-CORRECT, MIXED and ALL-WRONG?* Three notebooks produce the analysis: a lossless sentence segmenter inserts `⟦S#⟧` markers into each trace; a Qwen3.5-9B labeller assigns every segment a discourse role (`setup`, `reasoning`, `doubt`, `verification`, `correction`), an answer status and a temporal role; and a trajectory-analysis notebook pools eligible samples per question and compares conditions with question-level bootstrap statistics. From 9,388 rows covering 270 questions, 6,575 traces pass an explicit eligibility funnel. The three within-question contrasts tell one coherent story. In ALL-CORRECT questions (6 length-significant questions), extra length buys **self-checking**: doubt rises from 0 % to 1.8 % of words, verification from 0 % to 8.7 %, and the first answer appears at 66 % of a long trace versus 96 % of a short one. In ALL-WRONG questions (12 length-significant questions), extra length buys **looping**: 4-gram repetition rises from 0.037 to 0.222 while lexical novelty collapses from 0.665 to 0.253, and `setup` shrinks from 29.7 % to 4.9 % of words. Inside MIXED questions (167 questions, the strongest contrast), correct samples are ≈ 1.3× longer than incorrect ones and are marked by *verification* (0.025 vs 0.016 word share, p = 0.0025), not by doubt, which does not differ. A five-condition comparison places repetition and hedging on a monotone difficulty gradient (0.025 → 0.120 and 0.0 → 1.647), with the explicit caveat that the groups contain different questions. Full effect tables, configurations, distributions and example galleries are deferred to appendices A–G.

*Keywords:* chain-of-thought analysis, discourse labelling, reasoning trajectories, compute scaling, observational study.

---

## 1 Introduction

### 1.1 Motivation

H1 and H2 of TTCA treat a sample as a row of numbers — accuracy, tokens, latency — aggregated over benchmarks or models. H3 deliberately changes the lens. Instead of asking *how much* the model thinks, it asks *what the thinking consists of*: where the model restates the problem, where it reasons, where it doubts itself, where it actually performs a check, where it corrects an earlier claim, where it commits to an answer, how often it repeats content it has already produced, and how much of the trace is still ahead once an answer exists.

These are the quantities that a token budget is *spent on*, and they differ between a run that ends correct and a run that ends wrong even when both run to the same length. Measuring them requires segmenting the free-text trace, labelling each segment with its discourse function, and reconstructing the trajectory — which is what the three-notebook pipeline of §4 implements.

### 1.2 Research questions

| ID | Research question | Contrast | Unit |
|---|---|---|---|
| **RQ1** | When the model is right on every sample of a question, how do its *longer* samples differ from its *shorter* ones? | ALL-CORRECT long vs short | question |
| **RQ2** | When the model is wrong on every sample of a question, how do its longer samples differ from its shorter ones? | ALL-WRONG long vs short | question |
| **RQ3** | Inside a single mixed question, how do *incorrect* samples differ from *correct* ones? | MIXED incorrect vs correct | question |
| **RQ4** | How do the five outcome/length conditions sit relative to each other? | group comparison | question |

RQ1–RQ3 are *within-question* contrasts and support outcome statements. RQ4 compares different questions and is explicitly descriptive (see §7).

### 1.3 Contributions

1. A reproducible three-stage textual pipeline: deterministic lossless sentence segmentation, LLM segment labelling with a strict JSON grammar and checkpointed, crash-safe serving, and a question-level trajectory analysis with bootstrap inference.
2. The first within-question measurement of what an extra reasoning budget is spent on in *this* model: self-checking when the run is going well, self-repetition when it is going badly, and a verification gap — not a doubt gap — between correct and incorrect samples.
3. A five-condition descriptive map of the ALL-CORRECT / MIXED / ALL-WRONG space, with repetition and hedging identified as monotone difficulty meters.

### 1.4 Report structure

§2 places H3 in the TTCA project. §3 describes the data and its provenance, eligibility funnel and grouping. §4 describes the methodology. §5 presents the results (RQ1–RQ4). §6 discusses them, §7 states the threats to validity, §8 concludes, §9 documents reproducibility, and §10 lists the internal artefacts. Appendices A–G hold configuration references, the full effect tables, the group comparison with confidence intervals, group distributions, the label taxonomy and segmentation tests, the example figure galleries, and the artefact index.

---

## 2 Background and scope

The TTCA project is split into three analysis hypotheses:

| Folder | Hypothesis | Unit of analysis |
|---|---|---|
| `h1-overbenchmark_analysis` | How do compute levels relate to accuracy, latency and token efficiency — over benchmarks? | compute level × benchmark |
| `h2-overmodel_analysis` | Are the H1 patterns universal across models or model-specific? | model |
| **`h3-texual_analysis-of-thinking-traces`** | **What happens *inside* the thinking trace?** | **question** (pooled samples) |

H3 sits downstream of a fixed sampling sweep: every question of the committed r1-llama8b dataset was answered 5 times at each of 7 compute levels (0, 256, 512, 1024, 2048, 4096, 8192). The unit of analysis is the *question*; the compute level is pooled away except as a figure-panel label, because the question is the only unit at which "correct vs incorrect" and "long vs short" are meaningful within one prompt.

**Scope decisions (inherited from the pipeline):**

* **Rows in scope.** The analysed file holds **9,388 rows covering all 270 questions** of the committed raw sweep (`9 benchmarks × 30 questions × 7 compute levels × 5 samples = 9,450 rows`, 1,050 per benchmark), minus 62 rows — every one of them at compute level `8192`. Keys `(benchmark, question_id, compute_level, sample_index)` are unique and the question set is identical to the raw file's.
* **Provenance flag.** The labeler notebook *as committed* runs with `CFG.SECOND_HALF_ONLY=True` (keep only `len(df)//2:` of its input). The produced 9,388 rows are exactly half of 18,776 and at the same time cover the whole committed sweep minus the 62 rows above — so the Kaggle-side `r1-llama8b_raw_results_segmented.csv` evidently held more rows than a straight segmentation of the committed raw file (most plausibly two copies of each row). Nothing is double-counted (all keys are unique), and the 62 absent `8192` rows are simply never analysed.
* **Compute levels present:** 0 (1,350 rows), 256, 512, 1024, 2048, 4096 (1,350 each) and 8192 (1,288 rows); the 1,350 rows at level `0` have no thinking text at all (skipped by the labeller); 1,008 rows are truncated (`was_truncated = True`); 4,265 correct / 5,123 incorrect.
* **Length is measured in words** of `thinking_text` after `<br>` → space and removal of zero-width characters — never in tokens, so it is comparable across benchmarks.

---

## 3 Data

### 3.1 Source and pipeline

```
src/data/raw/r1-llama8b/r1-llama8b_raw_results.csv          (raw sweep)
        │
        │  0.sentence_segmentation.ipynb        lossless ⟦S1⟧ ⟦S2⟧ … markers
        ▼
r1-llama8b_raw_results_segmented.csv          (intermediate, lives on Kaggle)
        │
        │  1.reasoning_trace_labeler.ipynb      Qwen3.5-9B labels every segment
        ▼
r1-llama8b_raw_results_segmented_labeled.csv  (106 MB — the analysis input)
        │                                     + rtl_status / rtl_n_segments / rtl_labels_json
        │  2_reasoning_trace_trajectory_analysis.ipynb
        ▼
src/results/r1-llama8b/textual_analysis/
        ├── figures/  4 overview PNGs + examples/{ALL-CORRECT,MIXED,ALL-WRONG}/ 5 each
        └── tables/   question_groups.csv, 3 effect tables, group_comparison.csv
```

The labeler appended three columns to the original 17: `rtl_status` (`labeled`, 8,038 rows, or `skipped: no thinking text`, 1,350 rows), `rtl_n_segments`, and `rtl_labels_json` — a JSON array with one object per segment (`segment_id`, `note`, `discourse_role`, `answer_status`, `answer_value`, `temporal_role`). File-level inventory in Appendix G.

### 3.2 Labelled dataset — quality statistics

| Quantity | Value |
|---|---|
| rows in the labelled CSV | 9,388 |
| `rtl_status = labeled` | 8,038 |
| `rtl_status = skipped: no thinking text` (compute level 0) | 1,350 |
| segments carrying `⟦S#⟧` markers | **410,887** |
| rows truncated (`was_truncated = True`) | 1,008 |
| labelled rows whose marker/annotation ids or roles fail stage-2 reconstruction | 34 |
| correct / incorrect | 4,265 / 5,123 |

### 3.3 Eligibility criteria

The analysis notebook does not silently drop rows: every step of the funnel is counted. A trace is *labelled* only if the JSON parses, the marker-id set in the text equals the annotation-id set, no duplicate ids exist, and every segment has a valid `discourse_role` (invalid `answer_status`/`temporal_role` become missing — never guessed); segment word widths must sum to the whole trace and text order must be preserved. A *complete* trace is not truncated; *first-copy* keeps only the first sample of each distinct thinking text inside a question (exact duplicates are common at low compute levels).

**Table 1 — Eligibility funnel (n rows, recomputed with the notebook's own rules; reproduces `question_groups.csv` exactly).**

| Step | Removed | Remaining |
|---|---|---|
| all rows | — | **9,388** |
| not a duplicate key | 0 | 9,388 |
| labelled (valid JSON, ids match, all roles valid) | 1,384 | 8,004 |
| has words | 0 | 8,004 |
| correctness known | 0 | 8,004 |
| truncation status known | 0 | 8,004 |
| complete (not truncated) | 906 | 7,098 |
| first copy of its thinking text inside the question | 523 | **6,575** |
| **questions with ≥ 1 eligible sample** | — | **270** |

(The 1,384 rejected at the "labeled" step = 1,350 rows with no thinking text + 34 rows whose `rtl_status` says `labeled` but whose marker/annotation ids or roles disagree.)

### 3.4 Question grouping

A question is grouped from its eligible samples only: `TOO-FEW` (< 5 samples), `ALL-CORRECT` (all correct), `ALL-WRONG` (all incorrect), `MIXED` (≥ 1 of each). *Length-significant* (`len_sig`) = `std(words)/mean(words)` above the 85th percentile (**0.979**) across the 268 non-TOO-FEW questions (median 0.648). Length-significance is the admission ticket for RQ1/RQ2 only; RQ3 never filters on length.

**Table 2 — Question groups.**

| group | rule | questions | eligible samples | median samples / question | length-significant (`len_sig`) |
|---|---|---:|---:|---:|---:|
| **ALL-CORRECT** | every sample correct | 43 | 1,154 | 27 | 6 |
| **MIXED** | ≥ 1 correct and ≥ 1 incorrect | 167 | 4,120 | 26 | 23 |
| **ALL-WRONG** | every sample incorrect | 58 | 1,294 | 25 | 12 |
| **TOO-FEW** | < 5 eligible samples (excluded) | 2 | 7 | 4 | 0 |
| **total** | | **270** | **6,575** | | **41** |


Samples per group: 1,154 (ALL-CORRECT), 4,120 (MIXED), 1,294 (ALL-WRONG), 7 (TOO-FEW) — **6,575 in total**. The distribution behaves as expected by difficulty: easy-tier questions are mostly MIXED/ALL-CORRECT, hard-tier questions mostly ALL-WRONG/MIXED, and only 3 hard questions are ALL-CORRECT (full distributions in Appendix D).

---

## 4 Methodology

### 4.1 Design principles

1. **Unit of analysis = the question** (`benchmark::question_id`). All eligible samples of a question are pooled across compute levels. Every pooled number is built `samples → question (equal weight) → pooled`, and **every confidence interval resamples questions**, not samples — so 167 MIXED questions count as 167 units, not 4,120.
2. **Length in words**, not tokens (§2).
3. **Two classes of features.** *Length-free* features (shares, rates per 100 words, relative positions) do not change just because a trace is longer — these carry the scientific content. *Length-dependent* features (`words`, `log_words`, `n_seg`, `mean_seg_words`) are reported separately so they can never masquerade as behaviour.
4. **`final` is an annotation status, not correctness.** Text features are cues and similarity scores, not labels.
5. **No causal claims.** The notebook itself flags that cross-group differences compare different questions (§7).

### 4.2 Sentence segmentation (stage 0, notebook [1])

Each `thinking_text` is split into sentences by **inserting `⟦S1⟧ ⟦S2⟧ … ` markers at sentence starts without changing a single original character** — deleting every `⟦S<number>⟧` token always returns the byte-exact input. Pure Python + pandas: no LLM, no tokenizer, fully deterministic.

Boundary detection is four passes of contextual checks: (a) terminal punctuation with a strict sub-engine for the ambiguous period (lowercase continuations, titles such as `Mr.`, list numbering, option-label chains such as `A. … B. …`, dotted acronyms such as `U.S.` / `Ph.D.`, abbreviations such as `e.g.`, scientific notation, decimals); (b) newlines as clues, not boundaries — blank lines split only when neither side continues a sentence, single newlines split only before list/heading lines or after tag-only lines such as `<think>`; (c) colon blocks — a line ending in `:` never ends a segment, and the span until ordinary prose resumes (equations, code, indented blocks) stays together; (d) filler merging — interjection-only sentences (`Hmm.`, `Wait, no.`) of ≤ 3 filler words merge into the next sentence, while a short sentence with a content word (`Done.`) does not. Markers are written before the first character of each sentence and whitespace is never touched, so the round-trip guarantee holds by construction; input already containing markers is refused rather than mangled.

A table of 21 acceptance cases (abbreviations, decimals, code, lists, options, quotes, acronyms, fillers, colon blocks, empty input, pre-marked input) is asserted in both directions — strip(out) == input **and** out == expected — and all pass (**Table E2**, Appendix E).

### 4.3 Segment labelling (stage 1, notebook [2])

Every segment of every trace is labelled with four fields by **Qwen3.5-9B served with vLLM on 2 × Tesla T4 (Kaggle, data parallel)**, upgraded from an earlier llama.cpp version with the same prompt and CSV outputs. Continuous batching (16–32 traces per T4), prefix caching across the chunks of one trace, xgrammar-constrained JSON output, longest-traces-first scheduling and an adaptive in-flight controller driven by the server's own `/metrics` keep the two T4s saturated; checkpointing is per trace, keyed by an md5 fingerprint of everything that could change the labels (prompt, example, model, chunking, rule enforcement), so any change invalidates the cache instead of silently mixing label vintages. Failure handling is explicit: dead-server workers drain, finished work is never lost, rerun resumes.

The request carries the whole trace before the labelling command so the prefix cache hits between chunks; the output is a strict JSON array (`minItems = maxItems = n`, enums for every field, no extra properties) compiled by xgrammar. Up to 2 retries at temperature 0.3 with a rising seed precede a chunk failure; `enforce_rules()` then applies four mechanical consistency fixes: no `answer_value` without a status; first segment is always `new`; a `continuation` must repeat the previous role (else `new`); any `final` before the last doubt/verification/correction is demoted to `candidate`. Configuration details are in Appendix A; the full label taxonomy in Appendix E.

### 4.4 Trajectory features (stage 2, notebook [3])

**Role composition & phase structure** — word share of each role (`share_*`); consecutive same-role segments merged into *phases* (`runs`), giving `n_runs`, `changes_per100w`, `runs100_<role>` (phases per 100 words), `first_<role>_rel` (relative position of the first phase of a role), and directed phase transitions `doubt_to_verif100`, `doubt_to_corr100`, `verif_to_doubt100`, aggregated into row-normalised 7 × 7 matrices (`START` + 5 roles + `END`) for the difference heat-maps.

**Surface cues** — `hedge_100w`: occurrences of `wait / hmm / maybe / actually / let me check / double-check / alternatively / …` per 100 words.

**Novelty & repetition** (per trace, in text order) — `novelty` = share of content words (≥ 3 letters, non-stop) never seen before in that trace; `rep4_share` = share of 4-grams that already occurred earlier; `sent_repeat_share` = share of ≥ 4-token sentences already seen.

**Answer development** — positions measured at the *end* of the segment that first carries an event: `first_ans_rel`, `first_final_rel`, `stab_rel`, `share_after_first_ans` / `share_after_stab`, `n_replace` vs `n_refine` (a refinement is one normalised answer containing the other, e.g. `x=3` vs `3`), `n_replace_100w`, `any_replace`, plus `gold_ever`, `gold_lost`, `last_matches_gold` (light answer normalisation: `\boxed{}`, `$`, `\text`, `\frac`, thousands commas, numeric canonicalisation). **Answer patterns** (categorical): `stable`, `answer replaced`, `gold candidate later replaced`, `no answer value`.

**Length-dependent** — `words`, `log_words`, `n_seg`, `mean_seg_words`. **Relative-position profiles** — role words per decile, a (trace × role × decile) array used by every stack-plot, asserted to sum back to the trace length. In total 34 features per contrast.

**Marker legend used in all trajectory figures:** `▼` first answer candidate (`#08306b`), `◆` first final (`#6a3d9a`), `✕` observed answer replacement (`#d62728`), `✱` answer refinement/format change (`#17becf`), thick black bar = start of a `new` thought, dashed arrow = repetition of an earlier phase. Row labels read `s<sample> · correct/INCORRECT · <words>w`, shortest sample at the top; each compute level gets its own panel with its own x-scale.

### 4.5 Statistical procedures

For one contrast the notebook builds, for every question, the **mean feature value of side A and of side B**, then works on the differences `A − B` across questions:

* mean + **95 % question-bootstrap CI** (2,000 resamples),
* **paired effect size `dz`** = mean ÷ SD of the question-level differences (with its own bootstrap CI),
* **sign-flip p** (2,000 random ±1 resamplings) and **Wilcoxon p** — both exploratory, *no multiplicity correction*,
* `sig = True` when the bootstrap CI of the mean difference excludes 0.

Sides: *long / short* = longest / shortest `SIDE_FRAC = 0.25` of the question's samples (≥ 1 sample, never overlapping); *incorrect / correct* = the incorrect / correct samples of a MIXED question. A contrast needs ≥ 3 questions with both sides present. Result tables report the bootstrap mean-difference p; the CI-excludes-zero flag is the primary significance criterion used in the text.

---

## 5 Results

### 5.1 Overview

The three within-question contrasts (§5.2–§5.4) answer RQ1, RQ3 and RQ2 respectively, in order of statistical strength; §5.5 answers RQ4. Every result below is a question-level quantity with a 2,000-resample bootstrap CI; full 34-feature tables are in Appendix B.

**Executive findings:**

1. **Extra compute is not spent on re-reading the question.** In ALL-WRONG the longest samples devote only **4.9 %** of their words to `setup` versus **29.7 %** in the shortest (Δ = −0.248, dz = −2.71, CI [−0.297, −0.196]); the freed budget goes almost entirely into `reasoning` (65.8 % → 87.9 %).
2. **Long wrong traces loop.** 4-gram repetition rises from **0.037 → 0.222** (dz = 2.14) while lexical novelty collapses from **0.665 → 0.253** (dz = −3.31). Long correct traces show the same direction but far weaker (novelty 0.691 → 0.513, repetition 0.025 → 0.066).
3. **Doubt starts earlier when the run is going badly.** In ALL-WRONG the first `doubt` segment appears at **13.6 %** of a long trace versus **47.1 %** of a short one (dz = −2.60); in MIXED, incorrect samples doubt *later* than correct ones (+3.3 % of the trace, dz = 0.18).
4. **Verification is the clearest marker of correct samples.** MIXED correct samples give `verification` a **0.025** word share vs **0.016** for incorrect ones (dz = −0.24, p = 0.0025) and enter verification phases 0.019 times per 100 words more often. In ALL-CORRECT, short samples contain **zero** doubt and verification words; long ones contain 1.8 % doubt and **8.7 % verification**.
5. **When the model is right, the answer arrives early and the tail is check-work.** First answer candidate at **66.5 %** of a long ALL-CORRECT trace but at **96.0 %** of a short one (dz = −2.39): **33.5 %** of the long trace is written *after* the answer exists, versus 4.0 % of the short one.
6. **Answer churn is an outcome-linked signal, not a length artifact.** Within MIXED an answer is replaced at least once in **23.8 %** of correct vs **17.7 %** of incorrect samples (dz = −0.20, p = 0.009) — while the raw replacement *rate per 100 words* is indistinguishable (0.058 vs 0.057).
7. **Correct samples are longer inside the same MIXED question** (mean 908 vs 736 words; Δ log-words −0.246, dz = −0.41): the "more thinking helps" correlation at within-question granularity.

### 5.2 RQ1 — ALL-CORRECT questions: long vs short

**6 length-significant questions** (of 43 ALL-CORRECT); median long/short length ratio within a question ≈ **4.9×** (geometric mean, from Δ log-words 1.580).

![Figure 1: ALL-CORRECT long vs short](../../results/r1-llama8b/textual_analysis/figures/overall_ALL-CORRECT_long_vs_short.png)

*Figure 1 — ALL-CORRECT long vs short. Panels: (a)/(b) stacked role composition along the trace for long and short samples; (c) role-share difference long − short with a 95 % question-bootstrap band; (d) the 12 largest length-free effects with `dz` and CIs (orange = CI excludes 0); (e) phase-transition probability difference (long − short), `*` = CI excludes 0; (f) answer-development pattern mix.*

**Table 3 — Key effects, ALL-CORRECT long − short (n = 6; full table in Table B1).**

| feature | n | long | short | Δ (long − short) [95 % CI] | dz | p(sign-flip) | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `share_doubt` | 6 | 0.0182 | 0 | 0.0182 [0.00907, 0.0277] | 1.453 | 0.0235 | ✓ |
| `share_verification` | 6 | 0.0871 | 0 | 0.0871 [0.0555, 0.124] | 1.808 | 0.034 | ✓ |
| `changes_per100w` | 6 | 1.406 | 1.416 | -0.0105 [-0.163, 0.141] | -0.0516 | 0.915 |  |
| `runs100_verification` | 6 | 0.237 | 0 | 0.237 [0.174, 0.299] | 2.604 | 0.0295 | ✓ |
| `first_ans_rel` | 6 | 0.665 | 0.96 | -0.295 [-0.384, -0.2] | -2.386 | 0.0335 | ✓ |
| `first_final_rel` | 6 | 0.982 | 1.000 | -0.0178 [-0.0296, -0.00644] | -1.033 | 0.054 | ✓ |
| `share_after_first_ans` | 6 | 0.335 | 0.04 | 0.295 [0.199, 0.384] | 2.386 | 0.0355 | ✓ |
| `any_replace` | 6 | 0.0278 | 0 | 0.0278 [0, 0.0833] | 0.408 | 1.000 |  |
| `hedge_100w` | 6 | 0.907 | 0 | 0.907 [0.686, 1.078] | 3.381 | 0.0295 | ✓ |
| `novelty` | 6 | 0.513 | 0.691 | -0.178 [-0.226, -0.129] | -2.683 | 0.0275 | ✓ |
| `rep4_share` | 6 | 0.0657 | 0.0246 | 0.0411 [0.015, 0.0706] | 1.117 | 0.0685 | ✓ |
| `words` | 6 | 415.8 | 66.750 | 349.0 [258.3, 438.0] | 2.795 | 0.0295 | ✓ |
| `log_words` | 6 | 5.770 | 4.190 | 1.580 [1.245, 1.872] | 3.777 | 0.0275 | ✓ |
| `pat_stable` | 6 | 0.889 | 1.000 | -0.111 [-0.167, -0.0556] | -1.291 | 0.134 | ✓ |
| `pat_noanswer` | 6 | 0.0833 | 0 | 0.0833 [0.0278, 0.139] | 0.913 | 0.271 | ✓ |


**Reading:**

* **Short correct traces contain no doubt and no verification at all** (both shares exactly 0.000). Long correct traces devote **1.8 %** of words to doubt (dz = 1.45) and **8.7 %** to verification (dz = 1.81); they open `verification` phases 0.24× per 100 words (dz = 2.60) and follow `doubt → verification` 0.06× per 100 words (dz = 1.74). Correction never occurs on either side.
* **The answer comes much earlier in long traces:** first candidate at 66.5 % of the trace vs 96.0 % (dz = −2.39), first `final` at 98.2 % vs 100.0 % (dz = −1.03), so **33.5 % of the long trace is written after the answer exists vs 4.0 % of the short one** (dz = 2.39). A long correct sample does not "finally arrive" — it *arrives early and then keeps checking*.
* **Long correct traces are less novel and more repetitive:** novelty 0.513 vs 0.691 (dz = −2.68), 4-gram repetition 0.066 vs 0.025 (dz = 1.12), hedges 0.91 vs 0.0 per 100 words (dz = 3.38).
* **Nothing in the "how often the model changes gear" family moves:** `changes_per100w` 1.406 vs 1.416 (dz = −0.05) — the extra length is *more of the same phases*, not more phase switches.
* Pattern shifts are small but consistent: `stable` 88.9 % vs 100 %, `no answer value` 8.3 % vs 0 % (long traces more often carry a long/code answer that cannot be normalised). Not significant: `share_setup`, `share_reasoning`, `mean_seg_words`, replacement rates (≈ 0 on both sides).

### 5.3 RQ3 — MIXED questions: incorrect vs correct

**167 questions** (the largest group, 4,120 eligible samples). No length filter — this is the statistically strongest contrast of H3.

![Figure 2: MIXED incorrect vs correct](../../results/r1-llama8b/textual_analysis/figures/overall_MIXED_incorrect_vs_correct.png)

*Figure 2 — MIXED incorrect vs correct. Panels: (a) role composition of incorrect samples, (b) of correct samples, (c) incorrect − correct role-share difference with bootstrap band, (d) 12 largest length-free effects, (e) phase-transition difference matrix, (f) answer-development patterns.*

**Table 4 — Key effects, MIXED incorrect − correct (n = 167; full table in Table B2).**

| feature | n | incorrect | correct | Δ (incorrect − correct) [95 % CI] | dz | p(sign-flip) | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `share_setup` | 167 | 0.16 | 0.142 | 0.018 [0.00263, 0.0311] | 0.19 | 0.0145 | ✓ |
| `share_verification` | 167 | 0.0164 | 0.0252 | -0.00879 [-0.0145, -0.00321] | -0.237 | 0.0025 | ✓ |
| `runs100_verification` | 167 | 0.0515 | 0.0707 | -0.0192 [-0.0328, -0.00659] | -0.215 | 0.005 | ✓ |
| `first_doubt_rel` | 142 | 0.403 | 0.369 | 0.0331 [0.00145, 0.0627] | 0.178 | 0.037 | ✓ |
| `share_after_first_ans` | 139 | 0.303 | 0.329 | -0.0218 [-0.0555, 0.0127] | -0.106 | 0.215 |  |
| `any_replace` | 167 | 0.177 | 0.238 | -0.0613 [-0.108, -0.0152] | -0.204 | 0.009 | ✓ |
| `n_replace_100w` | 167 | 0.0567 | 0.0585 | -0.00178 [-0.0203, 0.0205] | -0.0133 | 0.882 |  |
| `hedge_100w` | 167 | 1.246 | 1.337 | -0.0911 [-0.166, -0.0165] | -0.185 | 0.022 | ✓ |
| `novelty` | 167 | 0.534 | 0.499 | 0.0346 [0.019, 0.0507] | 0.328 | 0.0005 | ✓ |
| `rep4_share` | 167 | 0.0784 | 0.0941 | -0.0157 [-0.0244, -0.00829] | -0.303 | 0.0005 | ✓ |
| `sent_repeat_share` | 167 | 0.00804 | 0.0138 | -0.0058 [-0.00924, -0.00282] | -0.272 | 0.0005 | ✓ |
| `words` | 167 | 735.6 | 908.0 | -172.5 [-273.1, -85.834] | -0.285 | 0.001 | ✓ |
| `log_words` | 167 | 6.045 | 6.291 | -0.246 [-0.338, -0.159] | -0.414 | 0.0005 | ✓ |
| `pat_replaced` | 167 | 0.157 | 0.221 | -0.0638 [-0.108, -0.0209] | -0.225 | 0.0045 | ✓ |
| `pat_noanswer` | 167 | 0.461 | 0.359 | 0.102 [0.0557, 0.152] | 0.321 | 0.0005 | ✓ |


**Reading:**

* **Correct samples are the longer ones** — mean 908 vs 736 words (Δ = −172, CI [−273, −86]), Δ log-words −0.246 (dz = −0.41) ⇒ a typical correct sample is ≈ 1.3× the length of an incorrect one of the same question; they also carry more segments (63.1 vs 49.6). The within-question form of "more thinking ↔ better outcome".
* **Verification separates the outcomes most cleanly:** `share_verification` 0.016 (incorrect) vs **0.025** (correct), Δ = −0.0088, CI [−0.014, −0.003], dz = −0.237, p = 0.0025; `runs100_verification` 0.051 vs 0.071 (dz = −0.215, p = 0.005). Correction and doubt shares do **not** differ.
* **Incorrect traces are more novel, less repetitive:** novelty 0.534 vs 0.499 (dz = 0.33, p = 0.0005), `rep4_share` 0.078 vs 0.094 (dz = −0.30, p = 0.0005), `sent_repeat_share` 0.008 vs 0.014 (dz = −0.27, p = 0.0005) — correct samples loop over their own content slightly more (they are re-checking it); incorrect ones keep generating new content.
* **Doubt arrives slightly later in incorrect samples:** `first_doubt_rel` 0.403 vs 0.369 (+3.3 % of the trace, dz = 0.18, p = 0.037) — significant but small, and opposite in direction to the ALL-WRONG long/short effect of §5.2, where doubt arrives *earlier* when the run is failing.
* **Answer churn:** `any_replace` 17.7 % vs 23.8 % (dz = −0.20, p = 0.009) and `pat_replaced` 15.7 % vs 22.1 % (dz = −0.23), while `n_replace_100w` is identical (0.057 vs 0.059) — the effect is "long enough to reach an answer, reject it and reach a better one", not a higher churn rate. `pat_noanswer` 46.1 % vs 35.9 % (dz = 0.32).
* Not significant: `share_reasoning`, `share_doubt`, `share_correction`, `changes_per100w`, `mean_seg_words`, `stab_rel`, `share_after_stab`.

### 5.4 RQ2 — ALL-WRONG questions: long vs short

**12 length-significant questions** (of 58 ALL-WRONG). Every eligible sample of these questions is wrong.

![Figure 3: ALL-WRONG long vs short](../../results/r1-llama8b/textual_analysis/figures/overall_ALL-WRONG_long_vs_short.png)

*Figure 3 — ALL-WRONG long vs short. Same panel layout as Figure 1.*

**Table 5 — Key effects, ALL-WRONG long − short (n = 12; full table in Table B3).**

| feature | n | long | short | Δ (long − short) [95 % CI] | dz | p(sign-flip) | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `share_setup` | 12 | 0.0491 | 0.297 | -0.248 [-0.297, -0.196] | -2.709 | 0.001 | ✓ |
| `share_reasoning` | 12 | 0.879 | 0.658 | 0.221 [0.16, 0.275] | 2.073 | 0.001 | ✓ |
| `share_doubt` | 12 | 0.0641 | 0.0429 | 0.0212 [0.00287, 0.0399] | 0.631 | 0.0535 | ✓ |
| `share_verification` | 12 | 0.00681 | 0 | 0.00681 [0.00261, 0.012] | 0.789 | 0.005 | ✓ |
| `runs100_verification` | 12 | 0.0228 | 0 | 0.0228 [0.00984, 0.0401] | 0.808 | 0.0035 | ✓ |
| `first_doubt_rel` | 8 | 0.136 | 0.471 | -0.384 [-0.477, -0.285] | -2.599 | 0.01 | ✓ |
| `share_after_first_ans` | 3 | 0.489 | 0.215 | 0.213 [0.138, 0.261] | 3.215 | 0.257 | ✓ |
| `any_replace` | 12 | 0.289 | 0.0278 | 0.261 [0.0833, 0.445] | 0.787 | 0.0385 | ✓ |
| `hedge_100w` | 12 | 1.931 | 1.247 | 0.683 [0.227, 1.156] | 0.807 | 0.0135 | ✓ |
| `novelty` | 12 | 0.253 | 0.665 | -0.411 [-0.473, -0.339] | -3.308 | 0.0015 | ✓ |
| `rep4_share` | 12 | 0.222 | 0.0366 | 0.186 [0.136, 0.227] | 2.140 | 0.0005 | ✓ |
| `sent_repeat_share` | 12 | 0.0284 | 0 | 0.0284 [0.0199, 0.0374] | 1.841 | 0.0025 | ✓ |
| `words` | 12 | 3826.7 | 240.9 | 3585.8 [2766.9, 4333.8] | 2.413 | 0.001 | ✓ |
| `n_seg` | 12 | 215.8 | 14.642 | 201.2 [148.9, 252.8] | 2.094 | 0.0005 | ✓ |
| `pat_replaced` | 12 | 0.289 | 0.0278 | 0.261 [0.0833, 0.45] | 0.787 | 0.0315 | ✓ |
| `pat_noanswer` | 12 | 0.475 | 0.917 | -0.442 [-0.639, -0.247] | -1.246 | 0.0085 | ✓ |


**Reading:**

* **Setup collapses.** Short wrong samples spend 29.7 % of their words restating the problem; long wrong samples only 4.9 % (Δ = −0.248, dz = −2.71, p = 0.001). Reasoning absorbs it: 65.8 % → 87.9 % (dz = 2.07).
* **Long wrong traces loop instead of exploring.** `rep4_share` 0.037 → 0.222 (dz = 2.14, p = 0.0005), `sent_repeat_share` 0.000 → 0.028 (dz = 1.84), while **novelty collapses 0.665 → 0.253** (dz = −3.31). Six-fold repetition with a third of the new content: the signature of a failing long run.
* **Doubt arrives early and stays.** `first_doubt_rel` 0.471 → 0.136 (dz = −2.60, p = 0.010); `share_doubt` +0.021 (dz = 0.63); `verif_to_doubt100` 0 → 0.008 (dz = 0.85) — checks decay back into doubt rather than resolving.
* **Verification appears only in long wrong traces** (`share_verification` 0 → 0.0068, dz = 0.79; `runs100_verification` 0 → 0.023, dz = 0.81) — but at a fraction of the 8.7 % seen in long *correct* traces (§5.2): checking alone does not rescue the answer.
* **Answer churn explodes:** `any_replace` 2.8 % → 28.9 % (dz = 0.79), `pat_noanswer` 91.7 % → 47.5 % (dz = −1.25) — short wrong samples mostly never write a normalisable answer at all, long ones keep rewriting one. (`first_ans_rel`, `first_final_rel` rest on only 3 and 1 questions; reported for completeness in Table B3 only.)
* **Hedges double:** 1.247 → 1.931 per 100 words (dz = 0.81). Not significant: `changes_per100w` (1.355 → 1.061, dz = −0.37 — long wrong traces switch gear *less* often), `share_correction`, `doubt_to_corr100`, `mean_seg_words`.

### 5.5 RQ4 — Five-condition group comparison

Five conditions side by side, each a question-level mean: **ALL-CORRECT short**, **ALL-CORRECT long** (shortest/longest 25 % of each question's samples), **MIXED correct**, **MIXED incorrect**, **ALL-WRONG** (all samples). With `CMP_LEN_SIG_ONLY = True` the ALL-CORRECT and ALL-WRONG sides are restricted to length-significant questions — exactly as in §5.2/§5.4 (n = 6 and n = 12); MIXED is compared by outcome and never length-filtered (n = 167).

![Figure 4: Group comparison](../../results/r1-llama8b/textual_analysis/figures/overall_group_comparison.png)

*Figure 4 — Five-condition comparison. Top row: role composition along the trace per condition (n in each title; the run used the threshold `std/mean > 0.98`). Bottom left: role shares by condition with 95 % CIs. Bottom right: answer-development pattern mix per condition.*

**Table 6 — Group comparison, question-level means (per-condition 95 % CIs in Table C1).**

| feature | ALL-CORRECT short | ALL-CORRECT long | MIXED correct | MIXED incorrect | ALL-WRONG |
|---|---:|---:|---:|---:|---:|
| `share_doubt` | 0 | 0.0182 | 0.0495 | 0.0464 | 0.054 |
| `share_verification` | 0 | 0.0871 | 0.0252 | 0.0164 | 0.00233 |
| `share_correction` | 0 | 0 | 0.0021 | 0.00185 | 0.00202 |
| `share_reasoning` | 0.793 | 0.74 | 0.781 | 0.775 | 0.796 |
| `changes_per100w` | 1.416 | 1.406 | 1.430 | 1.434 | 1.196 |
| `first_ans_rel` | 0.96 | 0.665 | 0.671 | 0.697 | 0.591 |
| `share_after_first_ans` | 0.04 | 0.335 | 0.329 | 0.303 | 0.409 |
| `n_replace_100w` | 0 | 0.00492 | 0.0585 | 0.0567 | 0.0119 |
| `rep4_share` | 0.0246 | 0.0657 | 0.0941 | 0.0784 | 0.12 |
| `hedge_100w` | 0 | 0.907 | 1.337 | 1.246 | 1.647 |
| `log_words` | 4.190 | 5.770 | 6.291 | 6.045 | 6.697 |


`n_questions` per condition: 6 / 6 / 167 / 167 / 12. For the answer-position features only questions whose samples carry an annotated answer contribute: 6 / 6 / 149 / 149 / 10.

**Reading:**

* **`setup` share falls monotonically toward the worst outcome:** 0.207 (AC short) → 0.155 (AC long) → 0.142 (MIXED correct) → 0.160 (MIXED incorrect) → **0.049** (ALL-WRONG, from Tables B1–B3).
* **`verification` is the opposite:** 0.000 (AC short) → **0.087** (AC long) → 0.025 (MIXED correct) → 0.016 (MIXED incorrect) → 0.002 (ALL-WRONG). Highest where the model is both correct and forced to think long.
* **`rep4_share` rises 0.025 → 0.066 → 0.078/0.094 → 0.120** across AC-short, AC-long, MIXED-inc/corr, ALL-WRONG, and `hedge_100w` follows (0.000 → 0.907 → 1.246/1.337 → 1.647): repetition and hedging track the *difficulty of the outcome*; novelty does the opposite.
* **Answer development:** `stable` 100 % (AC short) → 88.9 % (AC long) → ~39 %/36 % (MIXED) → 23.6 % (ALL-WRONG); `no answer value` 0 % → 8.3 % → 35.9 %/46.1 % → 47.5 %.
* **Length:** `log_words` 4.19 → 5.77 → 6.29 (MIXED correct) → 6.05 (MIXED incorrect) → 6.70 — MIXED correct samples are longer than MIXED incorrect ones, and the ALL-WRONG condition sits highest of all.

> **Caveat (stated by the notebook itself):** ALL-CORRECT, MIXED and ALL-WRONG contain *different questions*, typically of different difficulty. These differences show **where the groups sit relative to each other, not what causes an outcome**. Only the within-question contrasts of §5.2–§5.4 support that reading; the only within-question part of Figure 4 is AC short vs long.

### 5.6 Representative example trajectories

Five questions per group were saved as multi-panel trajectory figures (one panel per compute level, shortest sample at top, markers as in §4.4); the galleries are in **Appendix F**. What they add to the pooled tables: single questions *show* the spatial layout of `setup`/`doubt`/`verification` blocks and the answer markers, but the systematic differences only appear when questions are pooled — correctness does not visually "look like" anything on a single row (see the MIXED gallery note).

---

## 6 Discussion — what extra compute is actually spent on

1. **Length alone is not a behaviour.** The three contrasts disagree in instructively different ways. In *correct* runs, extra length buys **self-checking** (doubt 0 → 1.8 %, verification 0 → 8.7 %, answer at 66 % of the trace with a third of the trace left to verify it). In *failing* runs, extra length buys **looping** (repetition ×6, novelty ÷2.6, setup halved twice over, doubt from 13.6 % of the way in). In *mixed* runs, the difference between outcomes is not how much the model doubts (equal) but how much it **verifies**.
2. **Verification, not doubt, tracks success.** `share_doubt` is statistically indistinguishable between correct and incorrect samples within a question (Δ = −0.003, n = 167), while `share_verification` and `runs100_verification` both separate them at p < 0.01. Doubt is the cheap signal; performing the check is the expensive one.
3. **Repetition and hedging are difficulty meters.** Across the five conditions both rise monotonically as the outcome worsens (0.025 → 0.120 and 0.000 → 1.647), in *both* length contrasts and in the outcome contrast — the most robust cross-section pattern in H3.
4. **Answer churn matters through outcomes, not through rate.** Within MIXED the per-word replacement rate is identical, yet correct samples replace an answer at least once far more often (23.8 % vs 17.7 %) — they are long enough to reach an answer, reject it and reach a better one. In ALL-WRONG the churn is one-sided (2.8 % → 28.9 % short → long) and paired with a collapse in `no answer value` (91.7 % → 47.5 %).
5. **Nothing here is causal.** The design is observational and the compute level is pooled away; Table 6 in particular compares different questions.

---

## 7 Threats to validity and limitations

* **Cross-group confound** — ALL-CORRECT/MIXED/ALL-WRONG contain different questions of different difficulty; only the within-question contrasts (§5.2–§5.4) support outcome statements.
* **Labeler dependence** — every `discourse_role`/`answer_status` comes from one judge model (Qwen3.5-9B-AWQ, greedy decoding, one worked example). The consistency fixes in `enforce_rules()` are mechanical, not validation; no human agreement study is stored in the repo.
* **Input provenance** — the labeler's `SECOND_HALF_ONLY=True` flag and the 62 missing `8192` rows (§2): the analysed file is the committed 9,450-row raw sweep minus those 62 rows, while the Kaggle-side segmented input evidently held ~18,776 rows. All analysed keys are unique, so no sample is counted twice.
* **Small n in the length contrasts** — 6 (ALL-CORRECT) and 12 (ALL-WRONG) questions; several answer-position features fall back to n = 0–3 and are listed only for completeness in Appendix B. Wilcoxon/sign-flip p-values are exploratory with **no multiplicity correction** across 34 features × 3 contrasts.
* **Truncation excluded** — 906 truncated rows are dropped (they are exactly the largest compute levels), so the analysis never sees budget-exhaustion failures.
* **Cleared notebook outputs** — notebooks 1 and 2 are committed without their outputs; every number in this document was taken from the saved tables in `results/`, except the eligibility funnel and data-quality counts, which were recomputed from the labelled CSV with the notebook's own rules and reproduce `question_groups.csv` exactly (6,575 samples / 270 questions).
* **Doc/code drift** — the notebook's header markdown says the length-significance quantile is 75 %, while the code default is 0.85 (the saved figure titles confirm the run used `std/mean > 0.98`, the 85th-percentile value; see Appendix A).

---

## 8 Conclusion

H3 turns the thinking trace from an opaque text into a labelled trajectory and asks a budget question of every sample: *what was the extra compute spent on?* The answer is conditional on outcome. When the model is right, more budget is spent on arriving at the answer early and then verifying it; when it is wrong, more budget is spent restating the same content with rising doubt and almost no verification. Between correct and incorrect samples of the same question, the separating signal is verification and length — not doubt, and not the rate of answer changes. Repetition and hedging rise monotonically with outcome difficulty across every contrast, making them the most portable difficulty meters in this feature set. All claims are observational and question-level; the appendices carry every number behind them.

---

## 9 Reproducibility

**Stage 0 (local).** Open `0.sentence_segmentation.ipynb`, point `INPUT_PATH`/`OUTPUT_PATH` at the r1-llama8b raw/processed CSVs, run top to bottom; inspect the preview (section 4) before the save cell. Round-trip and acceptance tests must print `All acceptance tests passed.`

**Stage 1 (Kaggle, GPU T4 × 2 + Internet).** Upload/point `CFG.INPUT_FILENAME` at `r1-llama8b_raw_results_segmented.csv` (cell 1 pulls it from dataset `tharwatzzzz111/dataseta`), run cells 1–9 (installs vLLM, downloads the model, starts and smoke-tests both servers), then cell 10 for a visual quality check, then cell 11 for the run. Interrupt-safe: rerun cell 11 to resume; changing prompt/example/model/`CHUNK_SIZE` invalidates the checkpoint automatically. Outputs land in `/kaggle/working/` and must be downloaded into `src/data/processed/r1-llama8b/` as `r1-llama8b_raw_results_segmented_labeled.csv`.

**Stage 2 (local).**

```bash
# defaults baked into the notebook:
#   TRACE_CSV = ../../data/processed/r1-llama8b/r1-llama8b_raw_results_segmented_labeled.csv
#   TRACE_OUT = ../../results/r1-llama8b/textual_analysis
python -c "import jupyter"  # notebook run: 2_reasoning_trace_trajectory_analysis.ipynb, top to bottom
```

Environment overrides: `TRACE_CSV`, `TRACE_OUT`, `TRACE_LEN_Q` (length-significance quantile, default 0.85). Needs `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn` (for `ENGLISH_STOP_WORDS`). Expected output: the four `overall_*.png`, 15 `examples/**/*.png` and five CSVs listed in Appendix G.

---

## 10 References (internal artefacts)

| ID | Artefact |
|---|---|
| [1] | `src/notebooks/h3-texual_analysis-of-thinking-traces/0.sentence_segmentation.ipynb` |
| [2] | `src/notebooks/h3-texual_analysis-of-thinking-traces/1.reasoning_trace_labeler.ipynb` |
| [3] | `src/notebooks/h3-texual_analysis-of-thinking-traces/2_reasoning_trace_trajectory_analysis.ipynb` |
| [4] | `src/data/raw/r1-llama8b/r1-llama8b_raw_results.csv` (raw sweep) |
| [5] | `src/data/processed/r1-llama8b/r1-llama8b_raw_results_segmented_labeled.csv` (analysis input) |
| [6] | `src/results/r1-llama8b/textual_analysis/` (figures + tables, also as `textual_analysis.zip`) |

---

# Appendix A — Configuration reference

### A.1 Sentence segmentation (notebook [1], section 1)

| Setting | Default in the notebook | Meaning |
|---|---|---|
| `INPUT_PATH` | `../../data/raw/LiquidAILFM2.5-2.6B/LiquidAILFM2.5-2.6B_raw_results.csv` | input table (`.csv`/`.tsv`/`.parquet`) |
| `OUTPUT_PATH` | `…/data/processed/LiquidAILFM2.5-2.6B/…_segmented.csv` | output, `"_segmented"` suffix |
| `TEXT_COL` | `thinking_text` | column to segment |
| `SEED` / `N_PREVIEW` | 42 / 10 | preview sampling |
| `MARKER_START` | 1 | first marker is `⟦S1⟧` |

> **Caveat:** the notebook as committed still points at the *LiquidAI* sweep (its saved output shows a run on 29,050 rows / 24,899 texts on a Windows machine). For r1-llama8b the two paths must be switched to `../../data/raw/r1-llama8b/r1-llama8b_raw_results.csv` → `../../data/processed/r1-llama8b/r1-llama8b_raw_results_segmented.csv`. That intermediate CSV is not committed (it was staged on Kaggle as dataset `tharwatzzzz111/dataseta`, which cell 1 of the labeler copies into `/kaggle/working`).

Application details: `segment_column()` raises with the offending row index — nothing is skipped silently; the original frame is untouched; a pre-check refuses a frame that already contains markers. The preview (section 4) builds a side-by-side HTML view with markers highlighted, sampling half deliberately edge-case rows (abbreviation, decimal, equation, code, list, quote, many-newlines) and half random with `SEED=42`. The save cell re-loads the file and re-asserts shape, columns and the marker-stripping round trip for every non-empty text.

### A.2 Segment labelling (notebook [2])

`MODEL_CONFIGS` (pick one with `MODEL_KEY`, default `qwen3.5-9b-awq`):

| key | repo | tp | slots (start → max) | notes |
|---|---|---|---|---|
| `qwen3.5-9b-awq` | `lcu0312/Qwen3.5-9B-AWQ-4bit` | 1 | 16 → 32 | ~7–9 GB, one full copy per T4; hybrid Gated-DeltaNet → needs `--mamba-cache-mode align`; `--language-model-only` drops the vision tower |
| `qwen3.5-9b-fp16` | `Qwen/Qwen3.5-9B` | 2 | 24 → 48 | ~19 GB bf16, one server sharded over both T4s |
| `qwen3-8b-awq` | `Qwen/Qwen3-8B-AWQ` | 1 | 16 → 32 | plain-transformer fallback (labels would differ) |

Other `CFG` values that shape the run:

| Group | Setting | Value | Effect |
|---|---|---|---|
| I/O | `INPUT_FILENAME` / `SECOND_HALF_ONLY` | `r1-llama8b_raw_results_segmented.csv` / `True` | second half of the Kaggle-side input (provenance note in §2) |
| I/O | `OUT_RESULTS`, `OUT_SEGMENTS`, `OUT_CHECKPOINT` | `labeled_results.csv`, `labeled_segments.csv`, `labeling_checkpoint.jsonl` | all in `/kaggle/working/` |
| I/O | `CSV_ENCODING` / `AUTOSAVE_SECONDS` | `utf-8-sig` / 300 | Excel shows `⟦S1⟧` correctly; both CSVs rewritten every 5 min |
| server | `DTYPE`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `MAX_NUM_BATCHED_TOKENS` | `half`, 16384, 0.90, 8192 | fp16 (T4 has no bf16); long traces are the constraint |
| server | `LAUNCH_PROFILES`, `SMOKE_TEST`, `WARMUP_REQUESTS` | `["fast","no-prefix-cache","eager"]`, `True`, 8 | profiles tried in order; accepted only when every server is healthy *and* labels a 5-segment smoke trace with valid JSON |
| server | `GPU_INDICES`, `PORT_BASE` | `[0,1]`, 8089 | one server per T4, ports 8089/8090 |
| concurrency | `ADAPTIVE_SLOTS`, `ADAPT_INTERVAL_S`, `ADAPT_KV_TARGET`, `HEARTBEAT_SECONDS` | `True`, 15 s, 0.75, 60 s | every 15 s each server's `/metrics` is read; in-flight cap rises by 4 (up to `MAX_SERVER_SLOTS`) while KV cache < 0.75, drops (min 4, 90 s cool-down) when preemptions appear or KV ≥ 0.97 |
| labelling | `CHUNK_SIZE` | 20 | max segments per request |
| labelling | `USE_NOTE`, `NOTE_MAX_CHARS` | `True`, 100 | one-line note first, then labels (~+20 tokens/segment; `False` ≈ 30 % fewer output tokens but changes the prompt → full relabel) |
| labelling | `TEMPERATURE`, `RETRY_TEMPERATURE`, `MAX_RETRIES`, `SEED` | 0.0, 0.3, 2, 1234 | greedy first attempt; retries sample differently (seed only on retries) |
| labelling | `ENFORCE_RULES` | `True` | post-hoc consistency fixes (§4.3) |
| labelling | `SAMPLE_TRACES` | 3 | quality check in cell 10 (nothing saved) |

Context fitting: `max_tokens = n × (80 + note_tokens) + 100`; if the prompt would exceed `MAX_MODEL_LEN`, only the segments *not* being labelled are shortened (ladder 600 → 300 → 150 → 60 chars + `[...]`); a server-side `ContextOverflow` triggers the next rung. Checkpoint lines carry an md5 fingerprint of system prompt, example, model repo, `USE_NOTE`, `NOTE_MAX_CHARS`, `CHUNK_SIZE`, `ENFORCE_RULES` — changing any of them marks every old line stale; slot settings and concurrency are not part of the fingerprint. Only the main thread writes files; CSVs are written via `.tmp` + `os.replace` so a crash cannot leave a half-written file.

### A.3 Trajectory analysis (notebook [3], section 1)

| Setting | Value | Meaning |
|---|---|---|
| `SEED` | 0 | bootstrap / sampling seed |
| `MIN_SAMPLES` | 5 | a question needs ≥ 5 eligible samples to enter the analysis |
| `LEN_STD_QUANTILE` | **0.85** (`TRACE_LEN_Q`) | long-vs-short only for questions with `std/mean` above this quantile |
| `SIDE_FRAC` | 0.25 | "long"/"short" side = longest/shortest 25 % of the question's samples (never overlapping) |
| `N_BOOT` | 2000 | question-level bootstrap resamples |
| `REL_BINS` | 10 | relative-position deciles |
| `CMP_LEN_SIG_ONLY` | `True` | group comparison (§5.5) uses the same length filter as §5.2 / §5.4 |
| `N_SAVE_PER_GROUP` / `SHOW_FRAC` | 5 / 0.75 | 5 example figures saved per group; 75 % of questions shown interactively |
| `ROLES` / `KEY_ROLES` | `setup, reasoning, doubt, verification, correction` / `doubt, verification, correction` | label universe |

> **Documentation inconsistency:** the notebook's header markdown says the length-significance quantile is **75 %**, while the code default is **0.85** (and the saved figures' titles confirm the run used **std/mean > 0.98**, the 85th-percentile value, = 0.979 as recomputed). All tables in this report are the ones actually produced by the 0.85 setting.

---

# Appendix B — Full effect tables

Every contrast reports all 34 features (11 role/structure, 2 answer-position families, 7 answer-development, 4 surface/novelty, 4 length-dependent, 4 answer-pattern). Columns: per-side question-level means, mean difference with 95 % question-bootstrap CI, paired effect size `dz`, sign-flip p, and `sig` = bootstrap CI excludes 0.

### Table B1 — ALL-CORRECT long − short (n = 6 questions)

| feature | n | long | short | Δ (long − short) [95 % CI] | dz | p(sign-flip) | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `share_setup` | 6 | 0.155 | 0.207 | -0.0521 [-0.113, 0.00367] | -0.624 | 0.193 |  |
| `share_reasoning` | 6 | 0.74 | 0.793 | -0.0533 [-0.138, 0.0278] | -0.464 | 0.315 |  |
| `share_doubt` | 6 | 0.0182 | 0 | 0.0182 [0.00907, 0.0277] | 1.453 | 0.0235 | ✓ |
| `share_verification` | 6 | 0.0871 | 0 | 0.0871 [0.0555, 0.124] | 1.808 | 0.034 | ✓ |
| `share_correction` | 6 | 0 | 0 | 0 [0, 0] | — | 1.000 |  |
| `changes_per100w` | 6 | 1.406 | 1.416 | -0.0105 [-0.163, 0.141] | -0.0516 | 0.915 |  |
| `runs100_doubt` | 6 | 0.162 | 0 | 0.162 [0.092, 0.238] | 1.632 | 0.03 | ✓ |
| `runs100_verification` | 6 | 0.237 | 0 | 0.237 [0.174, 0.299] | 2.604 | 0.0295 | ✓ |
| `runs100_correction` | 6 | 0 | 0 | 0 [0, 0] | — | 1.000 |  |
| `doubt_to_verif100` | 6 | 0.0604 | 0 | 0.0604 [0.038, 0.0853] | 1.740 | 0.0305 | ✓ |
| `doubt_to_corr100` | 6 | 0 | 0 | 0 [0, 0] | — | 1.000 |  |
| `verif_to_doubt100` | 6 | 0.00308 | 0 | 0.00308 [0, 0.00925] | 0.408 | 1.000 |  |
| `first_doubt_rel` | 0 | — | — | — | — | — | — |
| `first_verification_rel` | 0 | — | — | — | — | — | — |
| `first_correction_rel` | 0 | — | — | — | — | — | — |
| `first_ans_rel` | 6 | 0.665 | 0.96 | -0.295 [-0.384, -0.2] | -2.386 | 0.0335 | ✓ |
| `first_final_rel` | 6 | 0.982 | 1.000 | -0.0178 [-0.0296, -0.00644] | -1.033 | 0.054 | ✓ |
| `stab_rel` | 6 | 0.669 | 0.96 | -0.291 [-0.375, -0.202] | -2.387 | 0.027 | ✓ |
| `share_after_first_ans` | 6 | 0.335 | 0.04 | 0.295 [0.199, 0.384] | 2.386 | 0.0355 | ✓ |
| `share_after_stab` | 6 | 0.331 | 0.04 | 0.291 [0.202, 0.38] | 2.387 | 0.03 | ✓ |
| `n_replace_100w` | 6 | 0.00492 | 0 | 0.00492 [0, 0.0147] | 0.408 | 1.000 |  |
| `any_replace` | 6 | 0.0278 | 0 | 0.0278 [0, 0.0833] | 0.408 | 1.000 |  |
| `hedge_100w` | 6 | 0.907 | 0 | 0.907 [0.686, 1.078] | 3.381 | 0.0295 | ✓ |
| `novelty` | 6 | 0.513 | 0.691 | -0.178 [-0.226, -0.129] | -2.683 | 0.0275 | ✓ |
| `rep4_share` | 6 | 0.0657 | 0.0246 | 0.0411 [0.015, 0.0706] | 1.117 | 0.0685 | ✓ |
| `sent_repeat_share` | 6 | 0.0022 | 0 | 0.0022 [0, 0.00542] | 0.56 | 0.503 |  |
| `words` | 6 | 415.8 | 66.750 | 349.0 [258.3, 438.0] | 2.795 | 0.0295 | ✓ |
| `log_words` | 6 | 5.770 | 4.190 | 1.580 [1.245, 1.872] | 3.777 | 0.0275 | ✓ |
| `n_seg` | 6 | 37.452 | 4.877 | 32.575 [23.317, 43.611] | 2.445 | 0.0335 | ✓ |
| `mean_seg_words` | 6 | 12.660 | 14.050 | -1.391 [-3.598, 1.241] | -0.388 | 0.359 |  |
| `pat_stable` | 6 | 0.889 | 1.000 | -0.111 [-0.167, -0.0556] | -1.291 | 0.134 | ✓ |
| `pat_replaced` | 6 | 0.0278 | 0 | 0.0278 [0, 0.0833] | 0.408 | 1.000 |  |
| `pat_goldlost` | 6 | 0 | 0 | 0 [0, 0] | — | 1.000 |  |
| `pat_noanswer` | 6 | 0.0833 | 0 | 0.0833 [0.0278, 0.139] | 0.913 | 0.271 | ✓ |


### Table B2 — MIXED incorrect − correct (n = 167 questions)

| feature | n | incorrect | correct | Δ (incorrect − correct) [95 % CI] | dz | p(sign-flip) | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `share_setup` | 167 | 0.16 | 0.142 | 0.018 [0.00263, 0.0311] | 0.19 | 0.0145 | ✓ |
| `share_reasoning` | 167 | 0.775 | 0.781 | -0.00591 [-0.0198, 0.00769] | -0.0632 | 0.418 |  |
| `share_doubt` | 167 | 0.0464 | 0.0495 | -0.00309 [-0.00809, 0.00149] | -0.0975 | 0.194 |  |
| `share_verification` | 167 | 0.0164 | 0.0252 | -0.00879 [-0.0145, -0.00321] | -0.237 | 0.0025 | ✓ |
| `share_correction` | 167 | 0.00185 | 0.0021 | -0.000246 [-0.00103, 0.000523] | -0.0467 | 0.557 |  |
| `changes_per100w` | 167 | 1.434 | 1.430 | 0.00323 [-0.0694, 0.0831] | 0.00611 | 0.935 |  |
| `runs100_doubt` | 167 | 0.335 | 0.344 | -0.00933 [-0.0388, 0.0197] | -0.0491 | 0.529 |  |
| `runs100_verification` | 167 | 0.0515 | 0.0707 | -0.0192 [-0.0328, -0.00659] | -0.215 | 0.005 | ✓ |
| `runs100_correction` | 167 | 0.0165 | 0.0167 | -0.000207 [-0.00663, 0.00624] | -0.00494 | 0.951 |  |
| `doubt_to_verif100` | 167 | 0.012 | 0.0167 | -0.0046 [-0.0108, 0.00317] | -0.0962 | 0.231 |  |
| `doubt_to_corr100` | 167 | 0.00749 | 0.00507 | 0.00242 [-0.00151, 0.00634] | 0.0956 | 0.229 |  |
| `verif_to_doubt100` | 167 | 0.00253 | 0.00395 | -0.00142 [-0.00298, 0.000157] | -0.138 | 0.0725 |  |
| `first_doubt_rel` | 142 | 0.403 | 0.369 | 0.0331 [0.00145, 0.0627] | 0.178 | 0.037 | ✓ |
| `first_verification_rel` | 79 | 0.67 | 0.678 | -0.0157 [-0.074, 0.0441] | -0.0572 | 0.606 |  |
| `first_correction_rel` | 35 | 0.565 | 0.546 | -0.0334 [-0.132, 0.0675] | -0.11 | 0.523 |  |
| `first_ans_rel` | 139 | 0.697 | 0.671 | 0.0218 [-0.0125, 0.0554] | 0.106 | 0.209 |  |
| `first_final_rel` | 92 | 0.987 | 0.991 | -0.00441 [-0.00991, 0.000463] | -0.176 | 0.101 |  |
| `stab_rel` | 133 | 0.834 | 0.82 | 0.0129 [-0.0168, 0.0428] | 0.0743 | 0.396 |  |
| `share_after_first_ans` | 139 | 0.303 | 0.329 | -0.0218 [-0.0555, 0.0127] | -0.106 | 0.215 |  |
| `share_after_stab` | 133 | 0.166 | 0.18 | -0.0129 [-0.0415, 0.0172] | -0.0743 | 0.384 |  |
| `n_replace_100w` | 167 | 0.0567 | 0.0585 | -0.00178 [-0.0203, 0.0205] | -0.0133 | 0.882 |  |
| `any_replace` | 167 | 0.177 | 0.238 | -0.0613 [-0.108, -0.0152] | -0.204 | 0.009 | ✓ |
| `hedge_100w` | 167 | 1.246 | 1.337 | -0.0911 [-0.166, -0.0165] | -0.185 | 0.022 | ✓ |
| `novelty` | 167 | 0.534 | 0.499 | 0.0346 [0.019, 0.0507] | 0.328 | 0.0005 | ✓ |
| `rep4_share` | 167 | 0.0784 | 0.0941 | -0.0157 [-0.0244, -0.00829] | -0.303 | 0.0005 | ✓ |
| `sent_repeat_share` | 167 | 0.00804 | 0.0138 | -0.0058 [-0.00924, -0.00282] | -0.272 | 0.0005 | ✓ |
| `words` | 167 | 735.6 | 908.0 | -172.5 [-273.1, -85.834] | -0.285 | 0.001 | ✓ |
| `log_words` | 167 | 6.045 | 6.291 | -0.246 [-0.338, -0.159] | -0.414 | 0.0005 | ✓ |
| `n_seg` | 167 | 49.649 | 63.100 | -13.452 [-20.732, -6.848] | -0.304 | 0.0005 | ✓ |
| `mean_seg_words` | 167 | 15.617 | 15.297 | 0.32 [-0.125, 0.812] | 0.106 | 0.177 |  |
| `pat_stable` | 167 | 0.36 | 0.392 | -0.0318 [-0.0765, 0.0125] | -0.106 | 0.191 |  |
| `pat_replaced` | 167 | 0.157 | 0.221 | -0.0638 [-0.108, -0.0209] | -0.225 | 0.0045 | ✓ |
| `pat_goldlost` | 167 | 0.0212 | 0.0278 | -0.00658 [-0.021, 0.00861] | -0.067 | 0.381 |  |
| `pat_noanswer` | 167 | 0.461 | 0.359 | 0.102 [0.0557, 0.152] | 0.321 | 0.0005 | ✓ |


### Table B3 — ALL-WRONG long − short (n = 12 questions)

| feature | n | long | short | Δ (long − short) [95 % CI] | dz | p(sign-flip) | sig |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `share_setup` | 12 | 0.0491 | 0.297 | -0.248 [-0.297, -0.196] | -2.709 | 0.001 | ✓ |
| `share_reasoning` | 12 | 0.879 | 0.658 | 0.221 [0.16, 0.275] | 2.073 | 0.001 | ✓ |
| `share_doubt` | 12 | 0.0641 | 0.0429 | 0.0212 [0.00287, 0.0399] | 0.631 | 0.0535 | ✓ |
| `share_verification` | 12 | 0.00681 | 0 | 0.00681 [0.00261, 0.012] | 0.789 | 0.005 | ✓ |
| `share_correction` | 12 | 0.00105 | 0.0027 | -0.00165 [-0.00754, 0.00173] | -0.173 | 1.000 |  |
| `changes_per100w` | 12 | 1.061 | 1.355 | -0.294 [-0.703, 0.117] | -0.372 | 0.241 |  |
| `runs100_doubt` | 12 | 0.38 | 0.284 | 0.0961 [-0.0686, 0.238] | 0.331 | 0.271 |  |
| `runs100_verification` | 12 | 0.0228 | 0 | 0.0228 [0.00984, 0.0401] | 0.808 | 0.0035 | ✓ |
| `runs100_correction` | 12 | 0.00795 | 0.0199 | -0.0119 [-0.055, 0.0131] | -0.169 | 1.000 |  |
| `doubt_to_verif100` | 12 | 0.00205 | 0 | 0.00205 [0, 0.0052] | 0.368 | 0.501 |  |
| `doubt_to_corr100` | 12 | 0.00393 | 0.0129 | -0.00896 [-0.0366, 0.00667] | -0.194 | 1.000 |  |
| `verif_to_doubt100` | 12 | 0.00789 | 0 | 0.00789 [0.00303, 0.0132] | 0.85 | 0.017 | ✓ |
| `first_doubt_rel` | 8 | 0.136 | 0.471 | -0.384 [-0.477, -0.285] | -2.599 | 0.01 | ✓ |
| `first_verification_rel` | 0 | — | — | — | — | — | — |
| `first_correction_rel` | 1 | 0.444 | 0.656 | — | — | — | — |
| `first_ans_rel` | 3 | 0.511 | 0.785 | -0.213 [-0.261, -0.138] | -3.215 | 0.253 | ✓ |
| `first_final_rel` | 1 | 0.999 | 1.000 | — | — | — | — |
| `stab_rel` | 2 | 0.72 | 0.885 | — | — | — | — |
| `share_after_first_ans` | 3 | 0.489 | 0.215 | 0.213 [0.138, 0.261] | 3.215 | 0.257 | ✓ |
| `share_after_stab` | 2 | 0.28 | 0.115 | — | — | — | — |
| `n_replace_100w` | 12 | 0.0141 | 0.0253 | -0.0112 [-0.0668, 0.0225] | -0.124 | 1.000 |  |
| `any_replace` | 12 | 0.289 | 0.0278 | 0.261 [0.0833, 0.445] | 0.787 | 0.0385 | ✓ |
| `hedge_100w` | 12 | 1.931 | 1.247 | 0.683 [0.227, 1.156] | 0.807 | 0.0135 | ✓ |
| `novelty` | 12 | 0.253 | 0.665 | -0.411 [-0.473, -0.339] | -3.308 | 0.0015 | ✓ |
| `rep4_share` | 12 | 0.222 | 0.0366 | 0.186 [0.136, 0.227] | 2.140 | 0.0005 | ✓ |
| `sent_repeat_share` | 12 | 0.0284 | 0 | 0.0284 [0.0199, 0.0374] | 1.841 | 0.0025 | ✓ |
| `words` | 12 | 3826.7 | 240.9 | 3585.8 [2766.9, 4333.8] | 2.413 | 0.001 | ✓ |
| `log_words` | 12 | 8.032 | 5.416 | 2.616 [2.263, 2.895] | 4.438 | 0.0015 | ✓ |
| `n_seg` | 12 | 215.8 | 14.642 | 201.2 [148.9, 252.8] | 2.094 | 0.0005 | ✓ |
| `mean_seg_words` | 12 | 18.012 | 16.853 | 1.159 [-0.214, 2.843] | 0.416 | 0.191 |  |
| `pat_stable` | 12 | 0.236 | 0.0556 | 0.181 [0.0694, 0.306] | 0.826 | 0.0395 | ✓ |
| `pat_replaced` | 12 | 0.289 | 0.0278 | 0.261 [0.0833, 0.45] | 0.787 | 0.0315 | ✓ |
| `pat_goldlost` | 12 | 0 | 0 | 0 [0, 0] | — | 1.000 |  |
| `pat_noanswer` | 12 | 0.475 | 0.917 | -0.442 [-0.639, -0.247] | -1.246 | 0.0085 | ✓ |


---

# Appendix C — Group comparison with confidence intervals

### Table C1 — All features × all five conditions (question-level mean [95 % question-bootstrap CI])

| feature | condition | mean [95 % CI] | n questions |
|---|---|---:|---:|
| `share_doubt` | ALL-CORRECT short | 0 [0, 0] | 6 |
| `share_doubt` | ALL-CORRECT long | 0.0182 [0.00941, 0.027] | 6 |
| `share_doubt` | MIXED correct | 0.0495 [0.0436, 0.0558] | 167 |
| `share_doubt` | MIXED incorrect | 0.0464 [0.0408, 0.0524] | 167 |
| `share_doubt` | ALL-WRONG | 0.054 [0.0398, 0.0684] | 12 |
| `share_verification` | ALL-CORRECT short | 0 [0, 0] | 6 |
| `share_verification` | ALL-CORRECT long | 0.0871 [0.0538, 0.124] | 6 |
| `share_verification` | MIXED correct | 0.0252 [0.0202, 0.0307] | 167 |
| `share_verification` | MIXED incorrect | 0.0164 [0.0118, 0.0218] | 167 |
| `share_verification` | ALL-WRONG | 0.00233 [0.00109, 0.00393] | 12 |
| `share_correction` | ALL-CORRECT short | 0 [0, 0] | 6 |
| `share_correction` | ALL-CORRECT long | 0 [0, 0] | 6 |
| `share_correction` | MIXED correct | 0.0021 [0.00134, 0.00304] | 167 |
| `share_correction` | MIXED incorrect | 0.00185 [0.00124, 0.00257] | 167 |
| `share_correction` | ALL-WRONG | 0.00202 [0.000761, 0.00384] | 12 |
| `share_reasoning` | ALL-CORRECT short | 0.793 [0.714, 0.859] | 6 |
| `share_reasoning` | ALL-CORRECT long | 0.74 [0.694, 0.785] | 6 |
| `share_reasoning` | MIXED correct | 0.781 [0.768, 0.794] | 167 |
| `share_reasoning` | MIXED incorrect | 0.775 [0.764, 0.786] | 167 |
| `share_reasoning` | ALL-WRONG | 0.796 [0.77, 0.825] | 12 |
| `changes_per100w` | ALL-CORRECT short | 1.416 [1.240, 1.599] | 6 |
| `changes_per100w` | ALL-CORRECT long | 1.406 [1.208, 1.615] | 6 |
| `changes_per100w` | MIXED correct | 1.430 [1.346, 1.518] | 167 |
| `changes_per100w` | MIXED incorrect | 1.434 [1.337, 1.536] | 167 |
| `changes_per100w` | ALL-WRONG | 1.196 [0.98, 1.415] | 12 |
| `first_ans_rel` | ALL-CORRECT short | 0.96 [0.927, 0.993] | 6 |
| `first_ans_rel` | ALL-CORRECT long | 0.665 [0.595, 0.741] | 6 |
| `first_ans_rel` | MIXED correct | 0.671 [0.644, 0.698] | 149 |
| `first_ans_rel` | MIXED incorrect | 0.697 [0.666, 0.73] | 149 |
| `first_ans_rel` | ALL-WRONG | 0.591 [0.462, 0.708] | 10 |
| `share_after_first_ans` | ALL-CORRECT short | 0.04 [0.00735, 0.0729] | 6 |
| `share_after_first_ans` | ALL-CORRECT long | 0.335 [0.259, 0.405] | 6 |
| `share_after_first_ans` | MIXED correct | 0.329 [0.302, 0.356] | 149 |
| `share_after_first_ans` | MIXED incorrect | 0.303 [0.27, 0.334] | 149 |
| `share_after_first_ans` | ALL-WRONG | 0.409 [0.292, 0.538] | 10 |
| `n_replace_100w` | ALL-CORRECT short | 0 [0, 0] | 6 |
| `n_replace_100w` | ALL-CORRECT long | 0.00492 [0, 0.0147] | 6 |
| `n_replace_100w` | MIXED correct | 0.0585 [0.0458, 0.0727] | 167 |
| `n_replace_100w` | MIXED incorrect | 0.0567 [0.0405, 0.0773] | 167 |
| `n_replace_100w` | ALL-WRONG | 0.0119 [0.00344, 0.0239] | 12 |
| `rep4_share` | ALL-CORRECT short | 0.0246 [0.0125, 0.0367] | 6 |
| `rep4_share` | ALL-CORRECT long | 0.0657 [0.0432, 0.0924] | 6 |
| `rep4_share` | MIXED correct | 0.0941 [0.0823, 0.106] | 167 |
| `rep4_share` | MIXED incorrect | 0.0784 [0.0688, 0.0876] | 167 |
| `rep4_share` | ALL-WRONG | 0.12 [0.0918, 0.146] | 12 |
| `hedge_100w` | ALL-CORRECT short | 0 [0, 0] | 6 |
| `hedge_100w` | ALL-CORRECT long | 0.907 [0.703, 1.075] | 6 |
| `hedge_100w` | MIXED correct | 1.337 [1.223, 1.446] | 167 |
| `hedge_100w` | MIXED incorrect | 1.246 [1.124, 1.366] | 167 |
| `hedge_100w` | ALL-WRONG | 1.647 [1.324, 1.932] | 12 |
| `log_words` | ALL-CORRECT short | 4.190 [4.108, 4.285] | 6 |
| `log_words` | ALL-CORRECT long | 5.770 [5.468, 6.062] | 6 |
| `log_words` | MIXED correct | 6.291 [6.163, 6.407] | 167 |
| `log_words` | MIXED incorrect | 6.045 [5.931, 6.146] | 167 |
| `log_words` | ALL-WRONG | 6.697 [6.374, 6.938] | 12 |


---

# Appendix D — Group distributions

### Table D1 — Question groups by benchmark

| benchmark | ALL-CORRECT | MIXED | ALL-WRONG | TOO-FEW | questions |
|---|---:|---:|---:|---:|---:|
| AIME2025 | 1 | 16 | 12 | 1 | 30 |
| BFCL-v3-simple | 3 | 13 | 14 | 0 | 30 |
| GPQA-Diamond | 1 | 28 | 1 | 0 | 30 |
| GSM8K | 9 | 20 | 1 | 0 | 30 |
| IFEval | 5 | 23 | 2 | 0 | 30 |
| LiveCodeBench-v6 | 1 | 10 | 18 | 1 | 30 |
| MATH-full | 8 | 18 | 4 | 0 | 30 |
| MMLU-Pro | 3 | 23 | 4 | 0 | 30 |
| MMLU-Redux | 12 | 16 | 2 | 0 | 30 |
| **all** | **43** | **167** | **58** | **2** | **270** |


### Table D2 — Question groups by difficulty tier

| difficulty tier | ALL-CORRECT | MIXED | ALL-WRONG | TOO-FEW | questions |
|---|---:|---:|---:|---:|---:|
| easy | 26 | 59 | 5 | 0 | 90 |
| medium | 14 | 54 | 22 | 0 | 90 |
| hard | 3 | 54 | 31 | 2 | 90 |
| **all** | **43** | **167** | **58** | **2** | **270** |


---

# Appendix E — Label taxonomy and segmentation tests

### Table E1 — `discourse_role` definitions (verbatim from the labeller's system prompt)

The role is the *function* of the segment in context — never judged by keywords ("wait", "let me check", "so the answer is" mean different things in different places).

| Role | Definition |
|---|---|
| `setup` | reading the **input**: restating the question, listing givens/options/tools, defining a term, re-reading the problem, restating the goal. No fact or method applied yet. |
| `reasoning` | any step toward the answer: recalling a fact/formula *to apply it*, announcing a plan, calculating, deducing, evaluating/rejecting an option, resolving a doubt, self-posed planning questions, switching approach without a suspected error, guess-and-test, writing code, stating a conclusion. |
| `doubt` | the model voices **uncertainty about something it holds** ("I'm not sure", "maybe I misread", "is that right?"). Planning questions and approach switches are *not* doubt. |
| `verification` | a check **actually performed** on something already derived: recomputing, substituting back, testing samples, a second method. First-time computations are not verification even if they say "make sure". |
| `correction` | the model recognises **its own earlier** statement/computation/interpretation as wrong and explicitly replaces it in that segment (even if the replacement is wrong). Merely recomputing is not correction. |

**Other fields.** `answer_status` (about the answer to the original question): `none` → `candidate` (proposed/endorsed, even tentatively, or a conclusion later reopened) → `final` (committed and never reopened); the rule given to the model: *scan forward from a conclusion-sounding segment — if a doubt/check/correction follows, it was only a candidate; if only restatements follow, it is final.* `answer_value`: the answer in the segment's own form (never normalised; `"two"` stays `"two"`; MCQ letter if named), `null` when there is no answer or the answer is long text/code; a correction that rejects 25 and installs 22 carries `"22"`. `temporal_role`: `new` (a new sub-goal/approach **or any change of `discourse_role`**; the first segment is always `new`) vs `continuation` (continues the move in progress — always the same role as the previous segment); a `repetition` value would mean re-stating earlier content and is never assigned automatically.

**Prompt layout.** One system prompt (the taxonomy + JSON-only answer format), then the worked example sent as a real user/assistant turn pair (an 8-segment trace walking through doubt → correction → verification → final: `16 + 9 = 25` → double-counting doubt → `16 + 9 − 3 = 22` → independent check → commit), then the actual request: `QUESTION:` (optional), `TRACE (n segments):`, and finally `Label these k segments, in this order: ⟦S…⟧ …` — the whole trace always precedes the request so vLLM's prefix cache hits between the chunks of one trace. `chat_template_kwargs.enable_thinking = False` because the grammar cannot contain a thinking block.

### Table E2 — Sentence-segmentation acceptance tests (all pass; each asserts strip(out) == input **and** out == expected)

| Case | Input | Expected output |
|---|---|---|
| title abbreviation | `Mr. X is here. He is ready.` | `⟦S1⟧Mr. X is here. ⟦S2⟧He is ready.` |
| decimal | `The result is 3.14. Therefore, it is valid.` | `⟦S1⟧The result is 3.14. ⟦S2⟧Therefore, it is valid.` |
| decimal mid-sentence | `For example, x = 2.5 and y = 3.0.` | one segment |
| `e.g.` | `We use e.g. this approach. It works.` | split only after `approach.` |
| scientific notation | `The value is approximately 1.2e-5.` | one segment |
| quote ending | `He wrote "This is correct." Then he continued.` | split after the closing quote |
| math + list | `Let me think (carefully).\n\nFirst, x = 3.5 * (2+1) = 10.5.\nSo y = 10.5 - 0.5 = 10.0! Is that right?\n\n- Check: 10.0 > 9.\n- Done.` | 6 segments |
| numbered list | `1. Check it. If yes, go.\n2. Otherwise stop.` | `1.` not treated as an ending |
| option labels | `Options:\nA. Safe, B. Unsafe.\nWe pick D. Done.` | 3 segments |
| tag line + filler | `<think>\nOkay, so. Let me see.` | split after the tag only |
| acronym + name | `In the U.S. The next step. John F. Kennedy spoke.` | 3 segments |
| ellipsis | `Is it done? yes, see Fig. 3.) Then stop...  Next.` | 3 segments |
| leading/trailing space | `  Leading space. Trailing space.  \n` | whitespace untouched |
| fillers merge | `Hmm.\n\nWait, no. Wait, the operation is…` | one segment |
| content word ≠ filler | `Hold on. Done.\n\nNext step.` | 2 segments (`Done.` keeps its segment) |
| trailing filler | `Hmm.` | stays `⟦S1⟧Hmm.` |
| colon + code block | `Else:\n\nfor each pair …:\n\n   swapped = …\n\nif swapped == "abc": …` | code stays in the `⟦S1⟧` block until prose resumes |
| colon + equations | `So, setting each factor equal to zero gives:\n\n\( x + 3 = 0 \) => \( x = -3 \) …` | all equations in the first segment |
| empty/NaN/number | `""`, `"   "`, `None`, `NaN`, `5` | returned untouched |
| pre-marked text | `Already ⟦S1⟧marked.` | `ValueError`, never silent |

---

# Appendix F — Example trajectory galleries

Each figure shows one question: one panel per compute level, samples of that level only, shortest at the top, word positions on each panel's own scale (panels are compared by *shape*, not by absolute length). The notebook displays 75 % of the questions of each group interactively and writes only these 5 "representative" ones (closest to the group median on log mean length, log length spread and accuracy). Marker legend as in §4.4.

### F.1 ALL-CORRECT

![Figure A1 — ALL-CORRECT example IFEval_10](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_10.png)

![Figure A2 — ALL-CORRECT example IFEval_18](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/IFEval-IFEval_18.png)

![Figure A3 — ALL-CORRECT example MMLU-Pro_26](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Pro-MMLU-Pro_26.png)

![Figure A4 — ALL-CORRECT example MMLU-Redux_18](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_18.png)

![Figure A5 — ALL-CORRECT example MMLU-Redux_20](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-CORRECT/MMLU-Redux-MMLU-Redux_20.png)

*What to look for (e.g. `MMLU-Redux_20`: 29 samples, 29 correct, length spread std/mean 0.32): the grey `setup` block ends within the first ~20 % of every row, `doubt` (orange) appears as isolated islands only from compute level 256 upward, `verification` (green) shows up sporadically from 512, and `▼`/`◆` markers cluster in the last third — the §5.2 signature of "short = straight to the point, long = answer early, then check".*

### F.2 MIXED

![Figure A6 — MIXED example GSM8K_8](../../results/r1-llama8b/textual_analysis/figures/examples/MIXED/GSM8K-GSM8K_8.png)

![Figure A7 — MIXED example MATH-full_27](../../results/r1-llama8b/textual_analysis/figures/examples/MIXED/MATH-full-MATH-full_27.png)

![Figure A8 — MIXED example MMLU-Pro_1](../../results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_1.png)

![Figure A9 — MIXED example MMLU-Pro_23](../../results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_23.png)

![Figure A10 — MIXED example MMLU-Pro_5](../../results/r1-llama8b/textual_analysis/figures/examples/MIXED/MMLU-Pro-MMLU-Pro_5.png)

*What to look for (e.g. `MMLU-Pro_23`: 30 samples, 19 correct, std/mean 0.69, panels 256 → 8192): green correct and red INCORRECT rows are interleaved at every level; correctness does not visibly "look like" anything on a single row — the systematic differences only appear when the 167 questions are pooled (§5.3). Watch instead for the density of green `verification` blocks and the red `✕` replacement markers, which the pooled tables show are outcome-linked; the 8192 panel shows the length explosion (one 2,490-word row beside 788–895-word rows).*

### F.3 ALL-WRONG

![Figure A11 — ALL-WRONG example AIME2025_14](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/AIME2025-AIME2025_14.png)

![Figure A12 — ALL-WRONG example LiveCodeBench-v6_9](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_9.png)

![Figure A13 — ALL-WRONG example LiveCodeBench-v6_12](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_12.png)

![Figure A14 — ALL-WRONG example LiveCodeBench-v6_14](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_14.png)

![Figure A15 — ALL-WRONG example LiveCodeBench-v6_21](../../results/r1-llama8b/textual_analysis/figures/examples/ALL-WRONG/LiveCodeBench-v6-LiveCodeBench-v6_21.png)

*What to look for (e.g. `AIME2025_14`: 14 samples, 0 correct, std/mean 0.93): rows grow from 165 to 2,527 words while the grey `setup` prefix *shrinks* in relative terms — by level 4096 it is a sliver at the left edge, and the rest is a nearly unbroken band of blue reasoning with black `new`-thought separators every few dozen words (the §5.4 repetition signature). Only one `▼` candidate marker appears in the whole figure: long failing runs often never commit to a normalisable answer.*

---

# Appendix G — Artefact index

| Report section | Produced by | File |
|---|---|---|
| §3.2 | `1.reasoning_trace_labeler.ipynb` | [`r1-llama8b_raw_results_segmented_labeled.csv`](../../data/processed/r1-llama8b/r1-llama8b_raw_results_segmented_labeled.csv) (analysis input) |
| §4.2 / App. A.1 | `0.sentence_segmentation.ipynb` | segmented CSV (not committed; staged on Kaggle dataset `tharwatzzzz111/dataseta`) |
| §4.3 / App. A.2 | `1.reasoning_trace_labeler.ipynb` | labelled CSV (above) |
| §3.4 / App. D | `2_…ipynb` §5 | [`tables/question_groups.csv`](../../results/r1-llama8b/textual_analysis/tables/question_groups.csv) |
| §5.2 / App. B | `2_…ipynb` §6 | [`tables/ALL-CORRECT_long_vs_short.csv`](../../results/r1-llama8b/textual_analysis/tables/ALL-CORRECT_long_vs_short.csv), [`figures/overall_ALL-CORRECT_long_vs_short.png`](../../results/r1-llama8b/textual_analysis/figures/overall_ALL-CORRECT_long_vs_short.png) (Figure 1) |
| §5.3 / App. B | `2_…ipynb` §7 | [`tables/MIXED_incorrect_vs_correct.csv`](../../results/r1-llama8b/textual_analysis/tables/MIXED_incorrect_vs_correct.csv), [`figures/overall_MIXED_incorrect_vs_correct.png`](../../results/r1-llama8b/textual_analysis/figures/overall_MIXED_incorrect_vs_correct.png) (Figure 2) |
| §5.4 / App. B | `2_…ipynb` §8 | [`tables/ALL-WRONG_long_vs_short.csv`](../../results/r1-llama8b/textual_analysis/tables/ALL-WRONG_long_vs_short.csv), [`figures/overall_ALL-WRONG_long_vs_short.png`](../../results/r1-llama8b/textual_analysis/figures/overall_ALL-WRONG_long_vs_short.png) (Figure 3) |
| §5.5 / App. C | `2_…ipynb` §9 | [`tables/group_comparison.csv`](../../results/r1-llama8b/textual_analysis/tables/group_comparison.csv), [`figures/overall_group_comparison.png`](../../results/r1-llama8b/textual_analysis/figures/overall_group_comparison.png) (Figure 4) |
| App. F | `2_…ipynb` §6.1/§7.1/§8.1 | `figures/examples/{ALL-CORRECT,MIXED,ALL-WRONG}/*.png` (Figures A1–A15) |
| — | bundle | [`textual_analysis.zip`](../../results/r1-llama8b/textual_analysis.zip) (4.4 MB, identical content) |
