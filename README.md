# Dynamic Test-Time Compute Allocation for LLM Reasoning

## Abstract

Large reasoning models spend inference compute **adaptively**: harder queries call for
deeper chains of thought, easier ones for little or none. In practice, systems apply a
single fixed budget to every input — wasteful on easy queries, insufficient on hard ones.
The literature documents two failure modes — **underthinking** (stopping too early, e.g.
via frequent thought-switching without deeper reasoning) and **overthinking** (unnecessary
or domain-mismatched compute) — but treats them as separate phenomena and never reconciles
them. This project consolidates that literature into a single picture and proposes a
pipeline that unifies both directions of compute control.

---

## Background & the gap

### The two documented failure modes

1. **Underthinking** — the model either receives too little compute for the problem, or has
   enough but allocates it inefficiently. Evidence: incorrect responses show far more thought
   **switches** despite often being longer, i.e. compute is spent restarting thoughts rather
   than reasoning deeper.
2. **Overthinking** — extended chains of thought give diminishing or negative returns on easy
   problems. Evidence: a U-shaped entropy–difficulty relationship (entropy high on easy
   problems despite high accuracy, dropping at medium difficulty), and **domain-dependent**
   returns on chain-of-thought length — the optimal budget differs by task type, not only by
   difficulty.

### Open question

Whether overthinking is a **true inverted-U** (net accuracy loss) or **pure saturation**
(flat accuracy, wasted compute only) — and whether that depends on task type — is unresolved
in the literature and is treated here as an empirical question rather than an assumption.

### What is missing

Current methods fall into two camps, each with a blind spot:

| Camp | Examples | Blind spot |
|---|---|---|
| **Reactive** (decide during generation) | confidence/entropy early exit, conformal / risk-controlled stopping | cannot identify *in advance* how much compute a query will need; only knows when to stop |
| **Predictive** (decide before generation) | budget forcing, difficulty classifiers, learned routers, thinking-optimal scaling | cannot recover when the pre-set budget is wrong for an unusual query; no statistical guarantee |

None of these simultaneously provides **instance-level** allocation, cross-model
generalization, and **statistically calibrated** stopping — and none is evaluated under a
shared protocol against the two failure modes at once.

---

## Proposed approach

A four-component pipeline in which each stage covers the previous stage's blind spot:

| Stage | Role | Component |
|-------|------|-----------|
| 0 | Predictive | **Cheap difficulty signal** — a short forced-CoT prefix whose entropy trajectory is read out, or a hidden-state probe over the question alone; both avoid the chicken-and-egg problem of knowing difficulty before thinking |
| 1 | Predictive | **Calibrated theoretical baseline** — a scaling-law fit `C₀ = A·d̂^β + C_min` with only a few free parameters per model family, grounded in compute-optimal scaling and the marginal-value principle |
| 2 | Predictive | **Predictive correction** — a light regression `ΔC = g(C₀, x)` on the residual between the baseline and empirically optimal compute; directly compared against existing probes and marginal-value predictors |
| 3 | Reactive | **Exception layer with a statistical guarantee** — Learn-Then-Test conformal thresholds (stop when confident; pre-emptively stop likely-unsolvable) that override the pre-set budget only for flagged long-tail cases |

The pipeline is therefore **traceable at every step** — a query can be followed from a
difficulty estimate, to a predicted budget, to the point where a reactive gate either
confirms or overrides it.

---

## Hypotheses

Four falsifiable hypotheses, each widening the scope of the previous:

- **H1 — Allocation harms performance.** Poor allocation causes underthinking on hard
  problems and overthinking on easy ones, detectable in statistically controlled
  compute–accuracy curves (saturation vs. inverted-U).
- **H2 — It generalizes across model families.** Curve shape (saturating, inverted-U, or
  monotonic) is consistent rather than model-specific.
- **H3 — It varies by task type.** Generation-type tasks, selection-type tasks, and
  computation-heavy tasks shape the optimal budget curve differently — difficulty is not the
  whole story.
- **H4 — Difficulty explains only part of the variance.** A difficulty-only baseline leaves
  substantial residual variance in the compute a query actually needs — setting the bar any
  better predictor must clear.

---

## Investigation approach

- **Benchmark corpus.** A deliberate two-dimensional mix: multiple difficulty tiers relative
  to the target capability, crossed with task types (math, coding, multi-choice selection,
  tool use, science, instruction following) so no single tier or format dominates.
- **Controlled compute.** The budget is an enforced cap on pre-response reasoning tokens, so
  compute is varied without confounding reasoning with instruction following.
- **Simpson's-paradox-safe analysis.** Rather than pooling raw accuracy (which lets
  differently-shaped per-benchmark curves cancel out), each per-benchmark/per-tier curve is
  computed separately, **normalized on its own performance**, and only then aggregated;
  disaggregated curves are always reported alongside.
- **Structure of the analysis.** The investigation proceeds across three dimensions —
  *per sample* (within-group path structure), *per benchmark* (knee detection where added
  compute stops paying), and *per compute level & difficulty* (aggregate knee, tier-specific
  knees, gains per tier).

---

## Key findings

The executed investigation supports the central claim and sharpens it:

- **Gains come cheaply, then saturate.** The aggregate curve knees well below the maximal
  budget tested — most accuracy gain arrives at a fraction of the largest budget — while the
  hardest band keeps improving roughly an order of magnitude further.
- **No clean inverted-U.** Overthinking manifests as **saturation** — flat accuracy with
  wasted compute — rather than net accuracy loss on the tested set.
- **Longer correct paths are less efficient, not deeper.** A majority of the excess tokens on
  longer correct paths arrive *after* the answer is already committed (verification-style or
  new-reasoning tokens), and redundancy grows while information gain falls.
- **Failures settle early.** Incorrect paths differ by *fewer* explicit answer reconstructions
  — they commit to a wrong answer and stick rather than thrash, and rarely discover the correct
  one. This is consistent with **insufficient compute / a weak convergence signal**, i.e. the
  reactive layer the pipeline proposes.
- **Easy tests hit a ceiling.** Easy tiers saturate quickly (or at once), confirming
  over-allocating to easy queries is pure waste.

---

## Conclusion

Compute allocation is a first-class decision in test-time scaling. Controlling it requires
predicting where thinking should end *and* reacting while it ends. The evidence — saturation
rather than a global inverted-U, cheap aggregate gains, hard query long-tails, and early
commitment to wrong answers — directly motivates the difficulty-gated, statistically
calibrated, predict-then-react pipeline proposed here, and de-risks each of its four stages
against the actual failure structure of the reasoning model.