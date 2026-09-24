# H1 Analysis: Reasoning-Trace Efficiency and Trajectory Study

**The complete H1 analysis** reconstructed from the single raw results file `h1_raw_results.csv` and implemented in `h1-analysis-with-outputs.ipynb`. All findings below are reproduced from the notebook outputs and the saved CSVs in `h1_testing/`.

> **Important framing (repeated throughout):** everything in this analysis is **descriptive / associational, not causal**. E.g., "longer correct paths contain more repetition" is an association *within* groups, not evidence that length *causes* repetition.

---

## Table of Contents

1. [Overview and Objectives](#1-overview-and-objectives)
2. [Data and Setup](#2-data-and-setup)
3. [Shared Text-Analysis Pipeline (Methodology)](#3-shared-text-analysis-pipeline)
4. [Analysis 1 — Over Samples (Studies 1, 2, 3)](#4-analysis-1--over-samples)
   - 4.1 Study 1 — All-Correct, High-Dispersion Groups
   - 4.2 Study 1 — Efficiency Score
   - 4.3 Study 1 — Score Validation Suite
   - 4.4 Study 1 — Within-Group Statistics
   - 4.5 Study 1 — Extra-Token Attribution
   - 4.6 Study 1 — Figures
   - 4.7 Study 2 — Mixed-Outcome Groups
   - 4.8 Study 3 — All-Wrong Groups
   - 4.9 Trajectory Classifier Validation
   - 4.10 Study 2/3 Findings and Figures
5. [Analysis 2 — Over Benchmarks](#5-analysis-2--over-benchmarks)
6. [Analysis 3 — Over Compute Level](#6-analysis-3--over-compute-level)
7. [Analysis 4 — Over Difficulty Level](#7-analysis-4--over-difficulty-level)
8. [Key Takeaways](#8-key-takeaways)
9. [Limitations](#9-limitations)

---

## 1. Overview and Objectives

The H1 study investigates **how models reason** inside their reasoning traces, and **how reasoning effort (compute) translates into accuracy**. It is organized into four top-level analyses built on one raw results file:

1. **Over Samples** — a per-path study of *how* a model reasons within a fixed `(question_id, compute_level)` group, split into three sub-studies by group outcome:
   - **Study 1 (all-correct):** what distinguishes an *efficient* correct path from a *verbose* correct path?
   - **Study 2 (mixed):** how do incorrect paths differ in timing/behavior from correct paths solving the *same* question at the *same* compute budget?
   - **Study 3 (all-wrong):** what trajectory shapes do failing paths take when there is no correct comparator?
2. **Over Benchmarks** — accuracy vs. compute level per benchmark, with a Kneedle-detected "optimal" compute level per benchmark.
3. **Over Compute Level** — a benchmark-size-robust, normalized aggregate accuracy-vs-compute curve across all benchmarks.
4. **Over Difficulty Level** — the same methodology as analyses 2 and 3, but with an **empirically derived** difficulty level replacing the benchmark dimension.

All text-analysis machinery (segmentation, answer extraction, SBERT similarity/novelty, segment labeling, answer tracking) is implemented **once** and **reused identically** across Studies 1–3.

---

## 2. Data and Setup

### Raw input
- **File:** `h1_raw_results.csv` (≈ 18.5 MB), the *only* raw input; everything else is derived.
- **Size:** **3,150 rows ・ 90 questions ・ 7 compute levels.**
- 11 rows flagged `hit_marker_empty_output` were dropped before analysis (no thinking text), leaving **3,139 usable rows**.

### Required / optional columns
- Required: `question_id`, `compute_level`, `is_correct`, `thinking_text`, `thinking_tokens`, `benchmark`.
- Optional: `raw_model_output`, `output_tokens`, `was_truncated`/`truncated`, `gold` (present here).

### Compute levels
7 levels: **0, 256, 512, 1024, 2048, 4096, 8192** (thinking-token budgets).

### Key configuration (`CFG`)
| Setting | Value |
|---|---|
| Seed | 0 |
| Similarity thresholds | repeat `0.85`, dup-verif `0.90`, revisit `0.50` |
| Semantic novelty weight | `0.5` (semantic + 0.5 content-word) |
| Novelty thresholds | `tau_new = 0.35`, high `0.60` |
| Segmentation | min 15 chars/segment; max 2000 segments |
| Efficiency-score caps | info `0.60`, red `0.60`, div `[0.50, 0.95]`, flips/1k `3.0`, doubt/1k `10.0` |
| Study 1 min group size | 3 paths |
| Studies 2/3 min group size | 2 paths |
| Min length ratio (extra-token study) | `max/min >= 1.5` |
| Bootstrap resamples | 1000 (over **questions**) |
| Difficulty bands (fixed pass-rate) | 0–25% hard, 25–50%, 50–75%, 75–100% easy |

### Finding early data notes
- `was_truncated` was **False for all rows**; `hit_response_marker` True for all 3,150 rows.
- This is a Kaggle-style run: SBERT `all-MiniLM-L6-v2` downloaded on first use; `kneed` installed for elbow detection.

---

## 3. Shared Text-Analysis Pipeline

One reusable pipeline transforms a raw reasoning trace into labeled segments and path-level metrics. There are 9 stages:

### 3.1 Answer normalization (`normalize_answer`)
- Converts answers to a canonical representation: numeric values → **6 significant figures** with no trailing zeros, fractions → evaluated (e.g. `3/4` → `0.75`), percent → stripped of `%`, thousands separators removed, MC letters → uppercase, lowercased text otherwise.
- Reference answers can be extracted from `\boxed{...}` (last boxed block is used) or from serialized grading metadata (JSON dict → reads `answer`/`value`/`gold`/`final_answer`).

### 3.2 Answer candidate extraction (`extract_candidates`)
- Regex-based detection of numeric values (integers, decimals, thousands-separated, signed) and single MC letters.
- Candidate mentions int the **immediately preceding ~25 characters** of a negation (e.g. "the answer is **not** 5") are **excluded**.

### 3.3 Explicit answer assertions (`extract_explicit_answer`)
- A committed answer requires an answer cue (*the answer is, final answer, therefore the answer is, we get, equals, is equal to, i choose…*) or `\boxed{...}` followed by a detectable value.
- Segments containing **hedge cues** (*maybe, perhaps, I think, possibly, could be, probably…*) are **never** treated as committed assertions — preventing tentative mentions from counting as genuine answer changes.

### 3.4 Segmentation
- Traces split on blank lines/sentence punctuation (`.!?` followed by uppercase/digit/`$`); fragments `< 15 chars` merged to neighbours; traces with `> 2000` segments coarsened by merging adjacent pairs. Result: the basic analysis unit.

### 3.5 Segment raw features
- Cues per segment: has candidate, explicit answer, hedge, **doubt/self-correction** (*wait, hmm, let me reconsider, that's not right, let me recheck…*), **rejection** (*that's wrong, scratch that, let me start over, i was wrong…*), **verification** (*let me verify, checking, double-check, confirm, sanity check, plugging back in…*), and char length.

### 3.6 Reference answer resolution
Reference priority:
1. **Gold** (`gold` column present → normalized most-frequent gold) → source `gold`.
2. **Group consensus** — most frequent own final answer if it covers **≥ 50%** of the group's paths → source `consensus`.
3. Otherwise **unresolved** (path's own final answer used upstream) → source `own_final`.

### 3.7 Semantic similarity & novelty
- Segments embedded with `sentence-transformers/all-MiniLM-L6-v2`.
- For each segment, `max_sim_prev` = max cosine similarity to any *earlier* segment in the same path; `sem_novelty = 1 − max_sim_prev`.
- Final `novelty = 0.5·sem_novelty + 0.5·content_word_novelty` (share of content words never seen before in the path). This distinguishes *semantic* repetition from genuinely new reasoning.

### 3.8 Answer-state tracking
Across the path: first explicit answer position, **answer flips** (changes between asserted values), **restates** (re-asserting the same value), **doubt/rejection** events, and **verification revisits** (verification cue + similarity to earlier content ≥ 0.50).

### 3.9 Segment labels (mutually exclusive, strict priority)
`first_answer` › `wrong_candidate` › `verification` › `doubt` › `restate` › `repetition` › `new_reasoning` / `before_answer`.

- **first_answer** — first committed answer assertion.
- **wrong_candidate** — explicit answer ≠ resolved reference.
- **verification** — verification cue present.
- **doubt** — hesitation/self-correction/rejection cue.
- **restate** — later assertion of an already-held answer.
- **repetition** — `max_sim_prev ≥ 0.85` (highly similar to earlier content).
- **new_reasoning / before_answer** — remaining segments split by whether the first answer has occurred yet and whether novelty clears `tau_new`.

---

## 4. Analysis 1 — Over Samples

Studies individual reasoning **paths** within a `(question_id, compute_level)` group. Groups are partitioned by outcome:

| Study | Group outcome | Question | Eligible groups | Paths |
|---|---|---|---|---|
| 1 | All correct + **high length dispersion** | What makes one correct path more efficient than another? | **26** | **130** |
| 2 | Mixed outcomes (0 < n_correct < n) | How do failing paths differ from correct ones on the same question/budget? | **214** | **1,061** |
| 3 | All wrong (n_correct = 0) | What trajectory shapes do failing paths take? | **133** | **663** |

### 4.1 Study 1 — All-Correct, High-Dispersion Groups

**Eligibility.** A `(question_id, compute_level)` group qualifies iff every sample is correct *and* it falls in the **top 10% of `std(tokens)/mean(tokens)`** across all groups (i.e. substantial length dispersion despite uniform correctness). Rows must have valid `is_correct` and non-empty text; ≥ 3 paths per group.

- Dispersion threshold (p90 of std/mean) = **0.512**.
- Result: **26 eligible groups, 130 paths** (Study 1 population), 7,987 segments.
- **All 26 groups** have `max/min thinking_tokens ≥ 1.5`, so all are usable for the extra-token study.

**Path-level metrics computed for every path** (shared pipeline in group mode):

| Metric | Definition |
|---|---|
| **Redundancy** | share of tokens in segments with `max_sim_prev ≥ 0.85`, excluding verification (which is *deliberately* similar). |
| **Information gain** | novelty-weighted share of tokens in post-first-answer segments that clear `tau_new`; whole-trace fallback if none qualify. |
| **First-answer position / post-answer fraction** | cumulative token fraction through the first explicit answer, and its complement. |
| **Post-answer verification** | fraction of post-answer segments labeled verification (only if post-answer budget clears `max(30, 2%·total)` tokens). |
| **Critical prefix** | token fraction through the segment where the correct answer is *last adopted and then held* (no later flips). |
| **Useful prefix** | critical prefix extended through immediately-following verification/new-reasoning segments that still clear the novelty threshold. |
| Plus | answer_flips, n_restate, n_doubt, n_revisit, unique-4-gram ratio. |

**Pooled distribution of Study 1 metrics (mean / median, n=130):**

| Metric | mean | median | min | max |
|---|---|---|---|---|
| thinking_tokens | **1,111** | 676 | 58 | 7,719 |
| redundancy | 0.094 | 0.060 | 0.000 | 0.497 |
| information_gain | 0.326 | 0.327 | 0.006 | 0.794 |
| unique_4gram_ratio | 0.965 | 0.979 | 0.727 | 1.000 |
| first_answer_pos | 0.878 | 1.000 | 0.077 | 1.000 |
| critical_prefix_ratio | 0.911 | 1.000 | 0.086 | 1.000 |
| useful_prefix_ratio | 0.921 | 1.000 | 0.086 | 1.000 |
| answer_flips | 0.07 | 0 | 0 | 2 |
| n_restate | 0.31 | 0 | 0 | 4 |
| n_doubt | 2.48 | 1 | 0 | 24 |
| n_revisit | 0.50 | 0 | 0 | 4 |

> Reading: correct paths are **sparse** in explicit answer events (most commit their first and final answer immediately — median first-answer position = 1.0), yet show frequent doubt cues (mean 2.5/ path).

### 4.2 Study 1 — Efficiency Score

A **descriptive composite** (not a validated ground-truth metric) combining **seven capped, [0,1]-clipped components** under a weighted **arithmetic mean** (default), with a geometric-mean alternative for comparison:

| Component | Weight | Meaning |
|---|---|---|
| c_info | 0.25 | information gain (capped at 0.60) |
| c_nonred | 0.20 | 1 − redundancy/0.60 |
| c_div | 0.10 | lexical diversity (unique 4-gram ratio mapped over [0.50, 0.95]) |
| c_stab | 0.10 | 1 − answer_flips per 1k tokens / 3.0 |
| c_conf | 0.10 | 1 − doubt events per 1k tokens / 10.0 |
| c_verif | 0.10 | post-answer verification (missing → neutral 0.5) |
| c_necess | 0.15 | 1 − useful_prefix_ratio |

- Rate components use `thinking_tokens` **with a floor of 1000 tokens** so short traces don't explode into per-1k rates.
- Weights are **renormalized** if a component is missing.

**Score distribution (n=130):**

| Stat | efficiency_score | c_info | c_nonred | c_div | c_stab | c_conf | c_verif | c_necess |
|---|---|---|---|---|---|---|---|---|
| mean | **0.633** | 0.528 | 0.844 | 0.978 | 0.991 | 0.856 | 0.380 | 0.079 |
| std | 0.125 | 0.324 | 0.188 | 0.067 | 0.042 | 0.188 | 0.208 | 0.187 |
| min | 0.287 | 0.010 | 0.171 | 0.505 | 0.667 | 0.266 | 0.000 | 0.000 |
| median | 0.649 | 0.546 | 0.900 | 1.000 | 1.000 | 0.914 | 0.500 | 0.000 |
| max | 0.800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.914 |

> The two weakest components are **c_verif** (post-answer verification is rare — median 0.5 is the neutral default) and **c_necess** (most paths reach a *useful* prefix ratio near 1.0, i.e. almost the whole trace is "needed"). **c_stab** and **c_div** are nearly saturated (medians = 1.0).

### 4.3 Study 1 — Score Validation Suite

Because the score is descriptive, it is stress-tested before use: **range/monotonicity, controlled scenarios, length invariance, weight-perturbation robustness (300 Dirichlet draws), leave-one-component-out, single-component dominance, and length-proxy tests**.

| # | Check | Status | Observed | Criterion |
|---|---|---|---|---|
| 1 | score_in_[0,1] | ✅ PASS | True | all in [0,1] |
| 2 | monotone_in_information_gain | ✅ PASS | 0.721 vs 0.513 | higher info → higher score |
| 3 | monotone_in_redundancy (inverse) | ✅ PASS | 0.700 vs 0.533 | lower redundancy → higher score |
| 4 | controlled_efficient_beats_rambling | ✅ PASS | 0.821 vs 0.274 | efficient > rambling |
| 5 | length_invariance_at_fixed_rates | ✅ PASS | delta = 0.0000 | < 0.05 when only length scales |
| 6 | weight_perturbation_rank_stability | ✅ PASS | p05 = 0.992 | ≥ 0.85 |
| 7 | leave_one_component_out_rank_stability | ✅ PASS | min = 0.788 (c_info) | ≥ 0.75 |
| 8 | **no_single_component_dominance** | ❌ **FAIL** | **0.956 (c_info)** | ≤ 0.90 |
| 9 | **score_not_pure_length_proxy_rho** | ❌ **FAIL** | **−0.538** | |ρ| ≤ 0.50 |
| 10 | score_not_pure_length_proxy_r2 | ✅ PASS | 0.217 | ≤ 0.50 |
| 11 | arith_vs_geo_aggregation_agreement | ✅ PASS (info) | ρ = 0.942 | ≥ 0.8 |

**Validation verdict (9 PASS / 2 FAIL):**
- The score is **robust to weight perturbations, robust to dropping any single component, length-invariant at fixed rates, and monotone** in its intended directions.
- **Two honest caveats:** (1) `c_info` (information gain) is heavily dominant in ranking (|ρ|=0.956), and (2) the score is **substantially correlated with length** (Spearman = −0.54; linear R² = 0.22) — i.e. it is partly a "shorter is better" proxy, exactly why all Study 1 findings are treated as associations and length is explicitly co-modeled.

### 4.4 Study 1 — Within-Group Statistics

Within every usable group (≥ 3 paths) we correlate `thinking_tokens` with each metric (Spearman) and compute longest-minus-shortest differences. Averaged to **question level** → **bootstrapped 1000× over questions** → 95% CI, two-sided bootstrap p, standardized effect size. Complemented by a **demeaned OLS** of each metric on centered log-tokens, clustered by question, Holm-corrected across metrics.

| Metric | Spearman mean | CI | long−short diff | CI | bootstrap p | std diff | p_holm | Reject 0.05 |
|---|---|---|---|---|---|---|---|---|
| **redundancy** | **+0.395** | (0.214, 0.569) | **+0.075** | (0.019, 0.136) | 0.008 | +0.669 | 0.005 | ✅ |
| **unique_4gram_ratio** | **−0.396** | (−0.613, −0.139) | −0.043 | (−0.089, −0.016) | 0.000 | −0.609 | 0.022 | ✅ |
| **information_gain** | **−0.376** | (−0.603, −0.158) | −0.114 | (−0.195, −0.045) | 0.004 | −0.713 | 0.022 | ✅ |
| post_answer_verification | +0.075 | (−1.0, 0.725) | −0.012 | (−0.076, 0.029) | 0.706 | −0.177 | 0.972 | ❌ |
| critical_prefix_ratio | −0.016 | (−0.214, 0.175) | −0.005 | (−0.046, 0.034) | 0.830 | −0.058 | 0.402 | ❌ |
| useful_prefix_ratio | −0.090 | (−0.272, 0.097) | −0.034 | (−0.077, 0.004) | 0.100 | −0.403 | 0.374 | ❌ |
| **efficiency_score** | **−0.404** | (−0.613, −0.153) | **−0.089** | (−0.136, −0.045) | **0.000** | **−0.989** | **0.000** | ✅ |

**Study 1 headline findings:**
- **Longer correct paths are less "efficient" in composition:** more token-weighted redundancy (Spearman +0.40), lower lexical diversity (−0.40), lower information gain (−0.38), and lower efficiency score (−0.40). All pass Holm-corrected 5% significance; the efficiency-score effect is large (standardized effect ≈ **−0.99**).
- **Crucially, the "structural" metrics do *not* shift with length:** critical prefix ratio, useful prefix ratio, and post-answer verification show **no association** with token count. In other words, the extra length is **not going into verifying or structurally re-deriving** the answer — it is spent, on average, on **repetition/new exploratory language that does not add novel content**.

### 4.5 Study 1 — Extra-Token Attribution

For groups with `max/min thinking_tokens ≥ 1.5` (all 26 in Study 1), the *extra* tokens of the longest path (beyond the shortest path's count) are attributed to the longest path's segment labels, then rolled into three buckets. Bootstrap over questions.

| Bucket | Mean share of extra tokens | 95% CI | bootstrap p |
|---|---|---|---|
| needed_to_reach_answer (before + first answer) | **25.9%** | (19.6%, 32.2%) | 0.000 |
| **post-answer verification / new reasoning** | **54.9%** | (45.5%, 63.9%) | 0.000 |
| repetition / restatement / doubt / wrong-candidate churn | **19.2%** | (11.9%, 26.8%) | 0.000 |

**Finding:** of the tokens that make a correct path *longer*, **~55% arrive after the answer is already committed** (verification-style or new-reasoning text), ~26% are needed to reach the answer in the first place, and ~19% are overt churn (repetition/restate/doubt/wrong candidate). Combined with §4.4 (post-answer verification *rate* does not rise with length), the picture is: **longer correct paths mostly add post-answer exploration that does not measurably increase verification activity or change the outcome — it is extra, largely non-informational, effort.**

**Per-label detail (example, GPQA-Diamond_0 @ 8192, 5,985 extra tokens):** new_reasoning 4,152; repetition 827; before_answer 746; doubt 195; verification 32; first_answer 16; wrong_candidate 16; restate 0.

### 4.6 Study 1 — Figures

| Figure | Content |
|---|---|
| **fig1** `fig1_length_vs_metrics_forest.png` | Forest plot of within-group Spearman(`thinking_tokens`, metric) with question-level bootstrap 95% CI; zero line shown. |
| **fig2** `fig2_token_composition.png` | Stacked bar of **extra-token buckets** for the top length-spread groups (longest−shortest tokens). |
| **fig3** `fig3_cumulative_information.png` | Cumulative novelty-weighted information vs. cumulative tokens for a representative group's paths (visual "slope" of info vs. length). |
| **fig4** `fig4_within_group_scatter.png` | Scatter of `thinking_tokens` vs. `efficiency_score`, each group in its own colour — shows the general downward trend with heavy within-group spread. |

### 4.7 Study 2 — Mixed-Outcome Groups

**Eligibility:** `0 < n_correct < n_samples` (at least one correct and one incorrect path), ≥ 2 paths. Controls for question difficulty and compute budget by comparing paths within the same group.

- **214 eligible mixed groups, 1,061 paths** (100,365 segments). Runtime ≈ 16 min (~0.9 s/path).
- Same shared pipeline in **group mode**, plus `trajectory_shape` and trajectory fields.

**Correctness × trajectory cross-tab (Study 2):**

| trajectory_shape | Incorrect | Correct | Total |
|---|---|---|---|
| never_found | 467 | 446 | 913 |
| stable_from_first | 21 | 116 | 137 |
| wavered_but_recovered | 2 | 4 | 6 |
| found_then_lost | 1 | 4 | 5 |
| **Total** | 491 | 570 | 1,061 |

- **Agreement `ends_correct` vs ground-truth `is_correct`: 55.4%** — a modest classifier consistency (matches ~ how often the last explicit answer equals the reference), not a new outcome measure.
- Trajectory shape is only weakly predictive of outcome: even "never_found" (per classifier) is nearly balanced between correct/incorrect (446 correct), because correctness is decided by the graded *final answer*, which the trajectory classifier labels only via explicit in-text assertions. **Use trajectory shapes as behavioral descriptions, not as correctness predictors.**

**Timing contrast — mean(metric | incorrect) − mean(metric | correct), within-group, bootstrapped over questions:**

| Metric | mean diff (inc−corr) | 95% CI | bootstrap p |
|---|---|---|---|
| first_correct_frac | −0.039 | (−0.160, 0.081) | 0.536 |
| n_doubt | +0.350 | (−0.411, 1.277) | 0.446 |
| **answer_flips** | **−0.040** | (−0.074, −0.009) | **0.010** |
| n_wrong_assert | +0.015 | (−0.062, 0.084) | 0.626 |

**Finding:** the **only statistically significant timing difference between incorrect and correct paths on the same question/budget is *fewer* explicit answer flips for incorrect paths** (p = 0.010). Incorrect paths do **not** show more doubt, later/earlier first-correct position, or more wrong assertions. Combined with the token means *(*correct = 1,579, incorrect = 1,767 thinking tokens — incorrect paths run slightly longer on average)*, the failure mode is not "thrashing," but a calmer, steadier drift — incorrect paths tend to **settle early into a wrong answer and stick to it**.

Additional trajectory stats (Study 2): among paths that ever touched the correct answer (n=148), `first_correct_frac` median = **0.96** — when the right answer does appear it usually appears very late in the trace.

### 4.8 Study 3 — All-Wrong Groups

**Eligibility:** `n_correct = 0`, ≥ 2 paths. No correct comparator → restricted to *within-path* trajectory composition. **133 eligible groups, 663 paths** (39,037 segments, ~6.6 min).

**Trajectory composition (share of all-wrong paths):**

| trajectory_shape | Study 3 share |
|---|---|
| **never_found** | **91.1%** |
| stable_from_first | 8.6% |
| found_then_lost | 0.3% |

- **never_found rate 91.1%** — in fully-failing groups, the model **almost never touches the correct answer at all** (within explicit assertions).
- **found_then_lost rate 0.3%** — losing a previously-found correct answer is *rare*; group failure is dominated by **never having found it**, not by drifting away.

### 4.9 Trajectory Classifier Validation

The four-class classifier (`never_found`, `stable_from_first`, `wavered_but_recovered`, `found_then_lost`) was unit-tested on **four synthetic traces**, one per class:
**All 4 PASS** (`divergence_classifier_checks.csv`).

Other pipeline validation on real data:
- Answer-normalization edge cases (`$1,234.50`→`1234.5`, `42%`→`42`, `3/4`→`0.75`, `  B  `→`B`, empty→None, `Answer: C.`→`answer: c`, …): **all PASS**.
- `\boxed{}` coverage in Study 1 population: **30.8%** of traces.
- MC-regex coverage (MMLU-Pro/Redux/GPQA-Diamond rows, first 5 segments): **63.2%**.
- Unresolved `is_correct` (dead branch): **0 rows**.
- `ref_source` distribution (Studies 2+3): **gold 1,212, consensus 300, own_final 212**.
- `answer_detect`: **1,723 / 1,724** paths had a detectable final answer.
- Label spot-check (10 random mixed segments): labels manually plausible (new_reasoning, verification, repetition, …).

### 4.10 Study 2/3 Findings and Figures

**Divergence position** (found_then_lost paths, Studies 2+3 combined): n = 7, **median = 0.95** of the trace length, IQR = (0.93, 0.98). When a path does lose the correct answer, it happens **very near the end** of the trace.

**Figure inventory (Study 1–3):**

| Figure | Content |
|---|---|
| **fig5** `fig5_divergence_position.png` | Histogram of divergence position (fraction of trace) for found_then_lost paths. |
| **fig6** (saved only when a representative found_then_lost path exists — not present in this run's `figures/` folder) | Cross-path grounding: max similarity of an incorrect path's segments to correct-path segments over normalized position, with divergence point marked. Since only 5 found_then_lost paths exist in Study 2, it did not get written. |
| **fig7** `fig7_trajectory_composition.png` | Side-by-side trajectory composition, Study 2 (mixed) vs Study 3 (all-wrong). |

---

## 5. Analysis 2 — Over Benchmarks

Operates on the **full raw dataframe** (not the Study 1–3 subsets). For each benchmark: accuracy at every compute level → plot → **Kneedle** elbow ("knee") = the compute level past which **more compute yields diminishing accuracy returns**.

### 5.1 Accuracy by compute level (n = 50 per cell, except a few 48–49)

| Benchmark \ level | 0 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| AIME2025 | 0.12 | 0.06 | 0.16 | 0.18 | 0.28 | 0.46 | **0.58** |
| BFCL-v3-simple | 0.38 | 0.40 | 0.40 | 0.36 | 0.38 | 0.40 | 0.40 |
| GPQA-Diamond | 0.32 | 0.46 | 0.45 | 0.47 | 0.43 | 0.56 | **0.70** |
| GSM8K | 0.68 | 0.80 | **0.88** | 0.82 | 0.90 | 0.90 | 0.86 |
| IFEval | 0.64 | 0.80 | **0.86** | 0.82 | 0.85 | 0.82 | 0.88 |
| LiveCodeBench-v6 | 0.26 | 0.36 | 0.34 | 0.42 | 0.44 | 0.62 | **0.68** |
| MATH-full | 0.66 | 0.64 | 0.65 | 0.68 | **0.76** | 0.72 | 0.72 |
| MMLU-Pro | 0.44 | 0.40 | 0.42 | 0.51 | 0.52 | **0.58** | 0.58 |
| MMLU-Redux | 0.62 | 0.46 | 0.63 | 0.58 | 0.62 | 0.54 | **0.64** |

### 5.2 Kneedle-based optimal compute level per benchmark

Kneedle runs on the **ordinal** x-axis (`curve="concave"` then `"convex"`, `direction="increasing"`); boundary "knees" (first/last position) are treated as **None**; curves with < 3 levels → `None`.

| Benchmark | Kneedle optimal compute level |
|---|---|
| AIME2025 | **256** |
| BFCL-v3-simple | **256** |
| GPQA-Diamond | **256** |
| GSM8K | **512** |
| IFEval | **512** |
| LiveCodeBench-v6 | **256** |
| MATH-full | **None** (no interior knee — nearly flat/noisy curve) |
| MMLU-Pro | **512** |
| MMLU-Redux | **4096** |

### 5.3 Observations
- **Most benchmarks saturate early** (knee at 256–512), i.e. the biggest accuracy gains come already at low budgets.
- **MATH-full** has *no* detectable interior knee → returns keep growing (or wobbling) through the whole range.
- **MMLU-Redux** is the lone benchmark with a *late* knee (4096) — its curve is non-monotone (dips at 256, 4096), showing more compute does not reliably help there.
- AIME2025's curve is the steepest gain profile in relative terms (+0.46 absolute from level 0→8192).

**Figure:** **fig8** `fig8_benchmark_accuracy_curves.png` — 9-panel grid, one accuracy-vs-compute curve per benchmark.

---

## 6. Analysis 3 — Over Compute Level

Goal: a **single accuracy-vs-compute curve** across benchmarks that is **not dominated by benchmark sample size**.

**Method:** naive pooling would let large benchmarks dominate, so instead:
1. accuracy per (benchmark, compute level) — from §5.1,
2. **min-max normalize within each benchmark** across its compute levels,
3. **average the normalized curves equally across benchmarks** (each benchmark gets one equal vote).

Constant-accuracy benchmarks (range 0) would make normalization undefined (0/0) and are **excluded**; here **0 of 9 benchmarks** were excluded.

### 6.1 Normalized aggregate by compute level (equal weight over 9 benchmarks)

| compute_level | mean normalized accuracy | sd | n_benchmarks |
|---|---|---|---|
| 0 | 0.212 | 0.303 | 9 |
| 256 | 0.313 | 0.363 | 9 |
| **512** | **0.527** | 0.403 | 9 |
| 1024 | 0.444 | 0.241 | 9 |
| 2048 | 0.677 | 0.276 | 9 |
| 4096 | 0.791 | 0.193 | 9 |
| 8192 | **0.943** | 0.120 | 9 |

### 6.2 Kneedle result

**Kneedle-selected compute level (aggregate normalized curve): 512.**

This is a striking headline: **on the size-robust aggregate, the "knee" of the accuracy-vs-compute curve is at 512 tokens** — past 512, the aggregate normalized accuracy keeps rising (to 0.94 at 8192) but at a much lower marginal rate, so 512 is where compute gains begin to plateau *across* benchmarks. The inter-benchmark spread (sd) also steadily shrinks from 0.40 → 0.12, i.e. benchmarks converge at large budgets.

**Figures:**
- **fig9** `fig9_compute_level_normalized_aggregate.png` — aggregate normalized curve with the red Kneedle marker at 512.
- **fig10** `fig10_benchmark_normalized_curves.png` — all 9 benchmark curves (thin) behind the aggregate (thick black) to show representativeness, not a few outliers.

---

## 7. Analysis 4 — Over Difficulty Level

Mirrors Analyses 2 and 3 with **difficulty level** as the grouping dimension. **Difficulty is derived, not given.**

### 7.1 Deriving difficulty (per-question pass rate)

For every question (benchmark + question_id, 90 questions) the **pass rate pooled over all samples and all compute levels** is computed, then binned into **fixed absolute pass-rate bands** (not quantiles, so band sizes are intentionally unequal):

| Level | Pass-rate band | Questions | Mean pass rate | Samples |
|---|---|---|---|---|
| **D1_hard** | [0%, 25%) | 26 | 0.077 | 905 |
| **D2_medium_hard** | [25%, 50%) | 13 | 0.381 | 451 |
| **D3_medium_easy** | [50%, 75%) | 15 | 0.646 | 523 |
| **D4_easy** | [75%, 100%] | 36 | 0.922 | 1,260 |

The **unequal split** (26 hard + 36 easy vs 13/15 in the middle) is expected given **fixed** cutoffs — it reflects a *bimodal* difficulty distribution in the data (many ~0% or ~100% questions, fewer in the middle), not an error.

### 7.2 Accuracy by compute level (equal-weight over benchmarks within each difficulty band)

| Difficulty \ level | 0 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| **D1_hard** | 0.075 | 0.065 | 0.147 | 0.036 | 0.128 | 0.103 | **0.161** |
| **D2_medium_hard** | 0.210 | 0.324 | 0.331 | 0.295 | 0.400 | 0.429 | **0.731** |
| **D3_medium_easy** | 0.481 | 0.452 | 0.591 | 0.748 | 0.705 | 0.762 | 0.757 |
| **D4_easy** | 0.841 | 0.859 | 0.917 | 0.885 | 0.918 | **0.970** | 0.963 |

(n_benchmarks per cell: 8/8/8/8/8/8/8 for D1 and D4; 7×7 for D2 and D3; n_samples ≈ 128–180 per cell.)

### 7.3 Kneedle results

**Raw (benchmark-equal-weight) accuracy curves:**

| Difficulty | Kneedle optimal compute level |
|---|---|
| D1_hard | **4096** |
| D2_medium_hard | **256** |
| D3_medium_easy | **None** |
| D4_easy | **512** |

**Normalized aggregate curves** (each (benchmark, difficulty) curve min-max normalized, then averaged equally):

| Difficulty | Kneedle optimal compute level |
|---|---|
| D1_hard | **4096** |
| D2_medium_hard | **256** |
| D3_medium_easy | **None** |
| D4_easy | **None** |

0 of 30 (benchmark, difficulty) curves were constant and excluded from normalization.

### 7.4 Difficulty findings
- **D1_hard (hardest)** — the *only* band where the knee is **late (4096)**: hard questions keep needing compute all the way up, with accuracy improving from 0.075 → 0.161 (still very low in absolute terms). More budget is required *and* still insufficient for hard questions.
- **D2_medium_hard** — knee at **256** on raw curves (fast early gains).
- **D3_medium_easy** — **no knee**: the curve is erratic (0.48 → 0.45 → 0.59 → 0.75 → 0.71 → 0.76 → 0.76), so no clean saturation point exists.
- **D4_easy** — knee at 512 on raw accuracy, and *no knee* on the normalized curve — beyond already-high baseline accuracy there is little relative headroom, so normalized gains are ambiguous (a known selection/ceiling artifact).

> **Caveat (from the notebook):** difficulty is computed from the same outcomes it is then used to stratify → the difficulty curves carry a **selection effect** (hard band is low-accuracy by construction; easy questions may be flat/ceiling). Read as descriptive shape, not causal evidence.

**Figures:**
- **fig11** `fig11_difficulty_accuracy_curves.png` — four panels, raw accuracy vs compute per difficulty band.
- **fig12** `fig12_difficulty_normalized_curves.png` — four overlaid normalized curves, with Kneedle marker per band where detected (D1 @ 4096, D2 @ 256).

---

## 8. Key Takeaways

**A. On reasoning efficiency (Study 1):**
1. Among all-correct, high-dispersion groups, **longer correct paths are compositionally less efficient**: more redundancy (+0.40 Spearman), less lexical diversity (−0.40), less information gain (−0.38), lower efficiency score (−0.40). All survive Holm-corrected 5% testing.
2. The length effect is **not** about verification or structure: critical/useful prefix ratios and post-answer verification are **independent of length**. The extra tokens go to **post-answer exploration** — of the tokens that separate the longest from the shortest correct path, **~55% are post-answer verification/new-reasoning text**, ~26% to reach the answer, ~19% to churn (repetition/restate/doubt/wrong-candidate).

**B. On how failure looks (Studies 2 & 3):**
3. In mixed groups, incorrect and correct paths differ mainly in **fewer explicit answer flips** for incorrect paths (p = 0.010) — incorrect paths **settle early on a wrong answer and stay with it** rather than thrash. No significant differences in doubt, wrong-assertions, or first-correct timing.
4. In all-wrong groups, failure is dominated by **never_found (91.1%)**; found_then_lost is rare (0.3%), and when it happens it occurs at **~95% of the trace** (median divergence position).
5. Trajectory shape alone is **weakly predictive of correctness** (ends-correct ↔ is_correct agreement only 55.4%) → treat shapes as behavioral, not diagnostic.

**C. On compute scaling (Analyses 2–4):**
6. **Most benchmarks' accuracy elbows at 256–512 tokens** (AIME2025, BFCL, GPQA, LiveCodeBench at 256; GSM8K, IFEval, MMLU-Pro at 512). **MATH-full has no knee**; **MMLU-Redux** is loaded *opposite* (knee at 4096 with a non-monotone curve). AIME2025 gains the most in absolute terms.
7. **Size-robust aggregate: the global knee is 512 tokens.** Normalized accuracy rises 0.21→0.53 by 512 and only to 0.94 at 8192; inter-benchmark spread halves (s.d. 0.40→0.12).
8. **Difficulty-band knee:** hard questions need the *most* compute (knee 4096, still ~16% absolute accuracy), medium-hard elbows at 256, medium-easy has no clean knee (erratic curve), easy questions saturate at 512 (with a ceiling artifact on normalized curves).

**D. Tooling caveat (Study 1 score):**
9. The efficiency score is **robust** (weights, LOO, monotonicity, length-invariance all pass) but **not a standalone verdict**: `c_info` dominates rankings (0.956) and the score is negatively tied to length (ρ = −0.54). Findings using it are associational only.

---

## 9. Limitations

- **Descriptive, not causal.** Every finding is an association, never a causal claim (e.g., "repetition correlates with length" ≠ "length causes repetition"). Within-group demeaning removes group-level confounds for Study 1 correlations, but question-level bootstrap CIs remain wide.
- **Heuristic text analysis.** Answer/behavior cues are **English-language regexes**, not semantic classifiers; best for short numeric and MC answers (MC coverage 63.2% on MC benchmarks). Segmentation is character-based, not true linguistic units. Novelty depends on the SBERT embedding + chosen thresholds.
- **Reference dependence.** Reference answers come from gold where available (1,212 paths), else consensus (300) or own-final (212); all reference-dependent metrics inherit that reliability.
- **Trajectory classifier limits.** Its labels depend on detecting *explicit* in-text assertions, so it under-counts implicit correctness (hence the low ends-correct agreement).
- **Difficulty selection effect.** Because difficulty is derived from the same outcomes, difficulty-band curves embed a selection/ceiling artifact (explicitly flagged in the notebook).
- **Kneedle fragility.** Kneedle can return boundary/None values on short/noisy or non-monotone curves (MATH-full, D3), and the x-axis treats unevenly spaced compute levels as equally spaced (ordinal positions).
- **Small tail phenomena.** found_then_lost is extremely rare (7 paths across Studies 2+3) — the divergence-position median (0.95) rests on n=7 and should be read as an anecdote.

---

### Output files (all written to `h1_testing/`)
- **Study 1:** `path_metrics.csv` (130×21), `segments.csv` (7,987 rows), `group_summary.csv` (26), `sanity_checks.csv` (11), `within_group_stats.csv` (7), `extra_token_attribution_{labels,buckets}.csv`.
- **Study 2/3:** `mixed_path_metrics.csv` (1,061), `mixed_segments.csv` (100,365), `allwrong_path_metrics.csv` (663), `allwrong_segments.csv` (39,037), `divergence_classifier_checks.csv`, `divergence_timing_contrast.csv`.
- **Benchmarks:** `benchmark_accuracy_by_compute_level.csv`, `benchmark_kneedle_optimal_compute_level.csv`, `benchmark_normalized_accuracy.csv`.
- **Compute level:** `compute_level_normalized_aggregate.csv`, `compute_level_kneedle_optimal.csv`.
- **Difficulty:** `difficulty_accuracy_by_compute_level.csv`, `difficulty_kneedle_{optimal_compute_level,normalized_optimal}.csv`, `difficulty_level_counts.csv`, `difficulty_normalized_{accuracy,aggregate}.csv`, `question_difficulty.csv`.
- **Figures:** `fig1`–`fig5`, `fig7`–`fig12` in `h1_testing/figures/` (11 files; fig6 is conditional and was not saved this run).