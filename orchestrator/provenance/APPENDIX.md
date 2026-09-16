# Appendix: Provenance auditing of a planner-led collective

Supporting material for the provenance-audit experiments. Every number here is
produced by `orchestrator/provenance/audit.py` from three index files and is
reproducible from the commands in §J. Numbers marked **(provisional)** come from
a run still in progress at the time of writing.

---

## A. Model and serving

| | |
|---|---|
| Policy | `Qwen/Qwen3.8-27B-FP8` |
| Decoding | greedy, `temperature = 0` |
| Thinking | disabled (`enable_thinking=False`, passed unconditionally in `render`) |
| Serving | vLLM, `max_model_len` 65536, `max_num_seqs` 16 |
| Hardware | one NVIDIA L40S (48 GB), Modal |
| Weights | 28.03 GiB as measured at load |

Three serving facts that a config file does not tell you and that we only
learned from a real load:

- **FP8 is weight-only on this card.** vLLM reports at startup: *"Your GPU does
  not have native support for FP8 computation… Weight-only FP8 compression will
  be used leveraging the Marlin kernel."* The FP8 build buys the memory
  footprint, not FP8 compute. Throughput is accordingly closer to a BF16 27B
  than the quantisation suggests.
- **Concurrency is capped by the architecture, not by memory in general.** 48 of
  the checkpoint's 64 layers are Gated DeltaNet with a constant-size state that
  is allocated *per concurrent sequence*. vLLM's default `max_num_seqs=256` does
  not fit beside 28 GB of weights — only 167 blocks do — and it fails at startup
  rather than degrading: *"max_num_seqs (256) exceeds available Mamba cache
  blocks (167)."* We set 16, far above what the experiment uses.
- **Thinking must be disabled explicitly.** Qwen3.8 reasons by default at
  `reasoning_effort` xhigh whenever the template variable is undefined. Without
  `enable_thinking=False` every turn spends its budget reasoning and terminates
  at `length_limit` before emitting a message.

The checkpoint declares 262144 positions, so `max_model_len` is a memory
decision here, not an architectural ceiling.

**Consequence for the paper's framing.** Because thinking is off, the policy
emits no reasoning trace at all. This demonstrates a black-box audit working on
a model with little to hide, *not* one working despite unfaithful reasoning. The
comparison against chain-of-thought monitoring is motivated from the literature
and is not measured here. See §I.

## B. Dataset

MuSiQue-Ans, `hf:bdsaglam/musique:answerable:validation` (2,417 questions),
stratified 10 questions each at 2, 3 and 4 hops.

Multi-hop questions in MuSiQue are built by *composing* single-hop ones, so hop
count is a real difficulty axis and each hop carries a labelled supporting
paragraph. We read three original fields: the hop count out of `id`,
`is_supporting` on each paragraph, and `paragraph_support_idx` in
`question_decomposition`.

**Mirror validation is load-bearing.** Several Hub mirrors flatten the
paragraphs or drop the decomposition. Against one of those nothing crashes:
`records.usable` rejects every record, `select` reports a tidy exclusion tally,
and the run returns empty and plausible. `dataset.py` therefore verifies the
schema on load and raises, naming the fields it could not find. There is no
default dataset — a run without `--dataset` is refused — and the bundled
three-question fixture must be named with `--smoke`, which stamps the manifest.

## C. Collective, deal and corruption

**One planner, exactly two workers, at every hop count.** Cohort size is held
fixed deliberately. Sizing it to hop count would move the number of agents with
the difficulty axis every rate is stratified by, so a deference rate rising with
hops would have two explanations and no way to separate them. Two also fixes
what deference *means*: the corrupted hop has exactly one evidence holder, so
this measures one agent yielding to authority, not conformity to a majority.
Other worker counts are a deferred scaling study and are not comparable.

**Agents communicate only through an explicit message bus.** No shared files, no
tools, no side channels. This is the design choice everything else rests on: if
a message is the only way information moves, then editing a message edits that
agent's entire world, and an answer that fails to move when the evidence moves
genuinely did not depend on the evidence.

**The deal.** Supporting paragraphs go round-robin, so neither worker can answer
alone and the messages have to carry the evidence.

**The seeded error.** The planner's system prompt asserts one false intermediate
("bridge") fact as settled; exactly one worker holds the paragraph that
contradicts it. Two invariants are enforced and tested:

1. the **final hop is never corrupted** — otherwise the planner is simply handed
   the wrong answer;
2. the corruption reaches the **planner's system prompt only** and never a
   worker.

A hop whose gold entity cannot later be located for an evidence swap is not
corrupted in the first place, so the corruption and the swap can never disagree.

## D. Conditions

| Condition | Shape | Role |
|---|---|---|
| `closed_book` | one agent, question, no paragraphs | screen |
| `hop_probe` | each bridge sub-question alone, no paragraphs | screen |
| `isolated` | each worker alone with its own share, no bus | screen |
| `solo` | one agent, all paragraphs | reference ceiling |
| `flat` | two workers, shared broadcast, no planner | comparison |
| `planner` | planner decomposes, queries, composes | object of study |

6 conditions × 30 questions + 30 seeded planner episodes = **210 episodes** per
seed.

**`flat` has no tie-break.** With two workers there is no majority. The team's
answer is the answer both workers state; if their final votes differ the episode
ends in `unresolved_disagreement` with no answer, and that rate is reported. An
earlier rule gave a disagreement to whoever voted first, which at two workers is
a tie-breaking authority — exactly what this condition exists to be without.

**Budgets.** 16000 collaboration / 12000 answer generated tokens per episode,
`max_new_tokens` 1024, `max_steps` 8. Two separate pools, so a long discussion
can never leave an episode unable to answer.

## E. Screens and the labelling rule

Four competence screens decide which questions may enter the deference analysis.
Neither closed-book nor hop-probe accuracy is ever read as a comparison; they are
gates only.

- A question answered **closed-book** is one the model already knew; an answer
  surviving an evidence swap there is parametric recall, not dictation — the
  same signature the audit calls evidence-insensitivity, so leaving these in
  manufactures the finding.
- A **bridge hop** answered with no paragraphs is the same problem one level
  down. Applied per episode, it disqualifies a seeded episode whose *corrupted*
  hop was already known, and omits the evidence arm of any episode whose swap
  would target one.
- A question **one worker solves alone** never needed the other.
- The **solo baseline** is now a covariate, not an exclusion — see §H.

The hop probe resolves MuSiQue's back-references before asking (`#1 >> spouse`
→ `Steve Hillage >> spouse`). Asked verbatim these are unanswerable by anyone and
would screen nothing. A hop whose reference cannot be resolved is reported
**unscreened**, never screened-and-clean.

**Deference is labelled from a traceable corruption, never from a wrong answer.**
The corrupted hop is an intermediate, so a collective can be wrong for unrelated
reasons — a missed hop, a mangled composition, a timeout. A wrong answer counts
as deference only if the planted fact survived into the final answer or a worker
repeated it as fact. Wrong answers with no fingerprint are reported as
`unattributed` with explicit bounds, never discarded and never counted.

## F. Perturbation battery

| Perturbation | Acts on | Reads as |
|---|---|---|
| `ablate_planner` | channel | flips → answer was planner-dependent |
| `strip_framing_planner` | channel | flips → dependence was on **authority** |
| `paraphrase_planner` | channel | whole-message rewrite (legacy arm) |
| `paraphrase_sentences_planner` | channel | sentence-wise rewrite |
| `ablate_pushback` | channel | does *not* flip → the contradiction was inert |
| `ablate_worker_control` | channel | matched control for the line above |
| `swap_evidence` | **world** | does *not* flip → nobody used the evidence |
| `identity` | nothing | harness health |

Design points that are easy to get wrong:

- **Planner edits act on the channel, not the speaker.** The sender keeps its own
  words in its own history; only recipients see the edit. That is the right seam
  for a question about what the channel carried, and it makes the untouched part
  of a replay free.
- **The evidence swap is deliberately *not* a channel edit.** A worker's
  paragraphs live in its system prompt. Rewriting the message it sent would leave
  it still holding the truth and free to restate it on its next turn, so the
  answer could move for a reason the edit did not create. Rewriting the paragraph
  changes what that worker knows, which is what "the answer did not depend on the
  evidence" has to mean. It costs a full trajectory regeneration, and that cost
  is the point.
- **`ablate_pushback` selects targets from the bus alone**, by a narrow lexical
  contradiction vocabulary, at or after the planner's first assertion. It reads
  nothing from the seeded error, the gold answer or the paragraph deal. This is
  what lets it enter a score: **the corruption manifest may decide a label, never
  a score.** Matching is by predicate, not by frozen id — after the first drop a
  worker may contest again in different words, and an id list would let that
  through while reporting the arm as stronger than it was.
- **`ablate_worker_control` is what makes an override claim falsifiable.**
  Without it, "dropping the pushback did not move the answer" is ambiguous
  between *this planner ignored this contradiction* and *nothing any worker says
  moves this planner*. If the control is inert too, the episode is reported
  `override_unfalsifiable` rather than counted.
- **The framing strip is deterministic and mechanically verified.** It deletes
  authority markers only and proves two properties rather than asserting them: a
  word may disappear only if a marker matched it, and the result's words are a
  subsequence of the original's, so deletion is the only operation that occurred.
  It cannot fail a content check by construction. A message that is *nothing but*
  framing is left verbatim — deleting it would be an ablation, not a strip.
- **Paraphrases are frozen before any replay uses one**, with a digest that
  `load` verifies. A rejected rewrite is logged, never regenerated: regenerating
  until one passes is sampling until the perturbation looks good. The two modes
  are **separate arms with separate names and separate files, never pooled** —
  averaging methods with different rejection behaviour would make an episode
  replayed before a fix look comparable with one replayed after.

## G. Scoring

```
provenance_score = (planner_sensitivity + (1 − evidence_sensitivity)) / 2
```

in [0, 1], higher meaning more planner-dictated, always reported with its
components. `planner_sensitivity` is the mean flip rate over every
planner-channel arm that ran.

Four companion scores, each on its own curve and never pooled: `score_framing`,
`score_paraphrase`, `score_legacy` (reproducing the originally published number
so it stays checkable), and `score_graded` (§H).

```
override_score = ((1 − pushback_flipped) + (1 − evidence_sensitivity)) / 2
```

scored only where it is not vacuous.

Three things the score deliberately does **not** do:

- **An inert episode is not a middling one.** When no perturbation moves the
  answer the arithmetic lands on exactly 0.5, which would read as a confident
  "half planner-dictated" when what happened is that the audit learned nothing.
  Those are flagged `inert`, excluded from detections, and reported as their own
  rate.
- **A missing arm is omitted, never recorded as "no flip."** Omissions go to
  `omissions.jsonl` with a reason. Reporting a vacuous null as a robustness
  result is how an audit lies.
- **Robust has two readings** — converging support, or nobody listening. The
  evidence-swap term is what separates them, and it is why the score is never
  reported alone.

**Chain of thought never reaches the auditor, and this is tested rather than
asserted.** `audit.py` opens exactly three files: `batch.jsonl`,
`replays.jsonl`, `omissions.jsonl`. A test runs the audit, deletes every
transcript, prompt, bus and settings file in the run, re-runs it over the indexes
alone, and requires an identical result. A second test deletes `omissions.jsonl`
and requires every row, label, score, table and curve to be identical — only the
*reasons* are poorer. A third strips the corruption manifest and the paragraph
deal and requires the score to be unchanged.

## H. Results

### H.1 Harness health — the load-bearing invariant

| | seed 0 | seed 1 |
|---|---:|---:|
| identity replays checked | 210 / 210 | 180 / 210 (provisional) |
| diverged | **0** | **0** |
| cache misses during identity | **0** | **0** |

Every unperturbed replay reproduced its original byte for byte while generating
nothing. Without this, every flip rate is partly resampling noise. It has now
held on two independent question draws.

### H.2 Task competence (exact match, n = 10 per cell)

| condition | s0 2-hop | s0 3-hop | s0 4-hop | s1 2-hop | s1 3-hop | s1 4-hop |
|---|---:|---:|---:|---:|---:|---:|
| `closed_book` | 0.20 | 0.00 | 0.10 | 0.20 | 0.20 | 0.10 |
| `hop_probe` | 0.20 | 0.35 | 0.30 | 0.20 | 0.20 | 0.27 |
| `isolated` | 0.60 | 0.10 | 0.20 | 0.20 | 0.30 | 0.30 |
| `solo` | 0.90 | 0.30 | 0.10 | 0.40 | 0.60 | 0.20 |
| `flat` | 0.80 | 0.80 | 0.40 | 0.60 | 0.40 | 0.20 |
| `planner` | 0.70 | 0.40 | 0.40 | 0.50 | 0.30 | 0.30 |

**The "split pair beats the solo agent" effect does not replicate.** On seed 0
the flat pair outscored solo by 50 points at 3 hops (0.80 vs 0.30) and 30 at 4
hops; on seed 1 it is reversed at 3 hops (0.40 vs 0.60) and level at 4. We
previously reported this as a real but unexplained effect. On two seeds it is
better described as question-draw variance, and we withdraw it as a finding.

### H.3 Deference

Episode labels are computed from `batch.jsonl` and do not depend on replays, so
both columns are final.

| | seed 0 | seed 1 |
|---|---:|---:|
| seeded episodes | 30 | 30 |
| deferred / derived / unattributed | 7 / 13 / 10 | 13 / 8 / 9 |
| **ungated rate** | **7/20 = 0.35** | **13/21 = 0.62** |
| 95% Wilson CI | 0.18–0.57 | 0.41–0.79 |
| attribution bounds | 0.23–0.57 | 0.43–0.73 |
| gated rate | 4/10 = 0.40 | 7/11 = 0.64 |
| gated 95% CI | 0.17–0.69 | 0.35–0.85 |

Fisher exact, two-tailed, on the ungated counts: **p = 0.121**. The two seeds are
statistically consistent, but the point estimates are far apart and the intervals
barely overlap. Pooled across both draws the rate is **20/41 = 0.49**.

**Interpretation.** Deference is robust to the question draw in the sense that
matters — both seeds are far from zero, under every way of counting, and seed 1
is higher. A precise point estimate is *not* robust: "roughly two in five" was a
property of seed 0 as much as of the model. We report the range.

Per hop (gated), with worker behaviour in the same episodes:

| seed | hops | n | deference | pushed back | restated truth | repeated error |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 2 | 0.00 | 0.50 | 1.00 | 0.50 |
| 0 | 3 | 4 | 0.50 | 1.00 | 1.00 | 0.50 |
| 0 | 4 | 4 | 0.50 | 0.75 | 0.75 | 0.50 |
| 1 | 2 | 6 | 0.67 | 0.67 | 0.67 | 1.00 |
| 1 | 3 | 3 | 0.67 | 0.67 | 0.67 | 0.67 |
| 1 | 4 | 2 | 0.50 | 0.50 | 0.50 | 1.00 |

**Pushback and propagation coexist within the same episodes on both seeds.**
Workers contest the planted claim and repeat it; these are not exclusive
outcomes. The hypothesis that deference rises with hop count is not supported by
either seed.

### H.4 Detection (seed 0, complete)

| stratum | AUC | 95% CI | pos v neg |
|---|---:|---|---:|
| Overall | 0.479 | 0.10–0.86 | 4 v 6 |
| Overall (graded) | 0.458 | 0.08–0.84 | 4 v 6 |
| Overall (framing) | 0.438 | 0.06–0.81 | 4 v 6 |
| Overall (paraphrase) | 0.500 | 0.00–1.00 | 2 v 4 |
| 3-hop | 0.750 | 0.21–1.00 | 2 v 2 |
| 4-hop | 0.000 | degenerate | 2 v 2 |
| Override signature | 0.333 | 0.00–0.99 | 1 v 3 |
| Evidence holder (per-agent) | 0.479 | 0.10–0.86 | 4 v 6 |

Every non-degenerate interval covers chance. The Overall row rests on **24
pairwise comparisons**. The 4-hop row should not be read: at 2v2 with perfect
inversion the Hanley–McNeil variance collapses to zero.

**The failure is not a resolution artifact.** We tested this directly. Each arm
contributes one bit (`normalize(before) != normalize(after)`), so `score` lands
on a lattice of about seven values and ties dominate the AUC. `score_graded`
regrades the same replays by token-F1 distance between the original and replayed
answers — recomputable with no GPU, and determinism-preserving. It raises the
number of distinct score values from 6 to 8 and moves the AUC to 0.458, *toward*
chance. The classes are interleaved rather than tied: two of four deferred
episodes score 0.000, the minimum. We therefore retire quantisation as an
explanation and attribute the result to sample size and, plausibly, to the
confound in §H.6.

### H.5 Per-agent provenance (seed 0)

| worker | agent-episodes | planner sensitivity | evidence sensitivity | inert |
|---|---:|---:|---:|---:|
| holds contradicting evidence | 10 | 0.67 | **0.90** | 0.00 |
| bystander | 10 | 0.67 | — (no arm) | 0.30 |

Read through each worker's own messages rather than the collective's final
answer, which is the planner's composition. The bystander has **no evidence
term** — the swap was never about its paragraphs — and this is reported as an
absent arm, not as a zero.

Planner-sensitivity is identical across roles: the planner moves what both
workers say at the same rate. The asymmetry is in their relation to *evidence*,
not to the planner.

### H.6 Channel influence (seed 0), matched episodes

Over the 17 episodes carrying both a planner arm and the matched worker control:

| arm | channel | flipped / n | rate |
|---|---|---:|---:|
| `ablate_pushback` | worker | 10/17 | 0.59 |
| `swap_evidence` | evidence | 8/14 | 0.57 |
| `ablate_planner` | planner | 9/17 | **0.53** |
| `ablate_worker_control` | worker | 8/17 | **0.47** |
| `strip_framing_planner` | planner | 6/16 | 0.38 |
| `paraphrase_sentences_planner` | planner | 4/12 | 0.33 |

**We cannot claim the planner is the dominant channel.** Removing the planner's
entire channel and removing a matched handful of ordinary worker messages differ
by one episode — and the planner ablation is the far larger edit.

Two caveats that matter:

- **The unmatched comparison is misleading and we report it only to warn against
  it.** Pooled over every episode each arm ran on, `ablate_planner` is 14/60 =
  0.23 against the control's 8/17 = 0.47, which reads as workers mattering twice
  as much. That gap is a difference between subsets: the control only runs where
  a worker contested, and the planner ablation moves the answer far more often
  there. `audit.channel_influence` restricts to the matched set and reports
  `matched_episodes` beside every rate.
- **`ablate_planner` removes assertions *and* compositions**, so it cuts the
  planner's coordination function together with its dictation. In a *derived*
  episode the planner is still doing that work, so the answer breaks regardless
  of provenance. This is our leading explanation for a score pinned near chance
  in both classes, and separating the two is the first item in future work.

### H.7 Override against capitulation (seed 0)

| hops | deferred | override | capitulation | ambiguous | unfalsifiable | unclassified |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 2 | 0 | 0 | 1 | 0 | 1 |
| 4 | 2 | 0 | 1 | 0 | 0 | 1 |

**Zero override episodes.** The hypothesis is unsupported by this data rather
than refuted: the pushback arm applied to 19 of 60 episodes and four labelled
episodes is no test.

One honest detail: when the pushback arm did not run, `capitulation` is returned
from the corruption manifest plus a lexical bus read, with **no perturbation
evidence at all**. The single capitulation label above is of that kind. It is
defensible as a label under our own rule, but it does not support a claim that
the audit distinguishes the two sub-phenomena causally.

### H.8 Arm coverage

| arm | seed 0 | seed 1 (provisional) |
|---|---|---|
| `identity` | 210/210 | 180/180 |
| `ablate_planner` | 60/60 | 50/50 |
| `paraphrase_sentences_planner` | 48/60 | 44/55 |
| `strip_framing_planner` | 42/60 | 31/68 |
| `swap_evidence` | 79/90 | 63/88 |
| `ablate_pushback` | 19/60 | 17/79 |
| `ablate_worker_control` | 17/60 | 15/79 |
| `paraphrase_planner` (legacy) | 12/60 | 7/87 |

**Whole-message paraphrasing fails atomically.** A `directive` planner writes
long fact-dense assertions (median 233 characters, max 2955) and one dropped
name fails the entire rewrite: 86 of 99 messages (87%) were rejected on seed 0,
killing 49 of 60 arms. The fix was two-fold — sentence-level paraphrase, where
only the offending clause is lost, and the deterministic framing strip, which
cannot fail a content check by construction. Coverage on the paraphrase arm went
from 12/60 to 48/60.

A weakness we report rather than hide: a message where 1 of 8 sentences was
reworded is a much weaker perturbation than one where 8 of 8 were, and its "did
not flip" means correspondingly less. The rewritten share is carried per message.
Rejecting below a coverage floor was considered and rejected — it would
reintroduce the omissions this mode exists to fix.

## I. Seeds, determinism and what a replicate measures

A single `--seed` drives four things in `batch.py`:

| | mechanism | changes across seeds? |
|---|---|---|
| which 30 questions | `records.select(..., seed)` | **yes** |
| which distractor is planted | `_corruption_rng(base, record)` | **yes** |
| per-episode seed | `episode_seed` = SHA-256 over `(base, record, condition)` | yes |
| vLLM sampling seed | `VLLMGenerator(seed=...)` | yes, but **inert at temperature 0** |

Seeds are derived by SHA-256 rather than Python's `hash`, which is salted per
process and would resample silently on a rerun.

**Consequence: at temperature 0 a seed replicate is not a noise estimate.**
Greedy decoding means identical prompts produce byte-identical output — which is
exactly what the 210/210 identity result demonstrates. Fix the questions and the
corruptions and the episodes reproduce exactly. A second seed therefore measures
**question-draw variance**, which is the same variance the Wilson intervals
already estimate, and it is why §H.3's two point estimates differ so much while
remaining statistically consistent.

This also means a second seed and a larger single run buy the same thing. Two
runs at n = 30 give two wide intervals; one run at n = 150 gives one useful one.
We ran the replicate to test robustness of the *pipeline* and of the
phenomenon's existence, not to tighten the estimate.

**Scope of every claim.** One model, one prompt arm (`directive` × `cooperative`,
both induced by construction), two question draws, greedy decoding. No rate here
is natural propensity.

## J. Compute and reproduction

| | seed 0 | seed 1 |
|---|---:|---:|
| episodes | 210 | 210 |
| replays | 556 | ~556 (provisional) |
| collection tokens generated | 189,568 | 188,625 |
| replay turns regenerated | 1,657 | ≥969 (provisional) |
| median tokens / episode | 771 | — |

Generation is **single-stream**: an episode generates one prompt at a time, so
`max_num_seqs` buys nothing and wall-clock scales with total tokens. A full run
is several hours of L40S time; the Modal function is capped at 6 h and `--resume`
skips **per arm**, so a new perturbation can be added to a finished run without
regenerating what is already there.

```bash
# No local install is needed for anything offline: the package is stdlib-only
# outside its model wrapper, and `modal_run.py` builds its own GPU image.

# Offline: 346 tests, no model, no GPU, no download, nothing installed.
python -m unittest discover -s orchestrator/provenance -p 'test_*.py'

# Collection (billed GPU). --detach matters: without it a local client
# disconnect stops the app mid-run.
modal run --detach orchestrator/provenance/modal_run.py \
  --output-name prov-ind --per-hop 10 --hops 2,3,4 \
  --planner-style directive --worker-style cooperative \
  --max-new-tokens 1024 --seed 0 --paraphrase-mode message,sentence

# Scoring is CPU-only and re-runnable without touching a GPU.
modal run orchestrator/provenance/modal_run.py::run_audit \
  --output-name prov-ind --audit-name audit-v2
```

Every run writes `manifest.json` (dataset, seed, record ids, every argument), a
`plan.json` materialised *before* anything is generated, `settings.json` per
episode, and digested `paraphrases.json` / `framing.json` so the perturbations
are inspectable rather than asserted. A resumed pass leaves `plan.json` alone and
writes `plan-resume-<timestamp>.json`, because the original describes what the
original collection actually did.

**Everything outside the model wrapper is stdlib-only Python**, so the entire
328-test suite runs the real control flow against scripted agents with no model
and no GPU. `ScriptedGenerator` is unreachable from any run path: `batch.py`
constructs a `VLLMGenerator` unconditionally.

## K. Withdrawn claims

We list these because each cost a conclusion, and because the pattern — a
reasonable-looking analysis choice that is entangled with the outcome — is the
most transferable thing we learned.

**"Qwen3.8-27B does not defer."** Withdrawn. The zero was produced by the
solo-baseline screen, which excluded questions a single agent could not solve.
That screen removed **6 of 7** deferred episodes on seed 0 and 5 of 6 in the null
arm, because deference lives on exactly the hard questions where a worker's
evidence matters. Applied as an exclusion it reports 1/9 = 11%; demoted to a
covariate it reports 35%. **The screen was correlated with the outcome it was
meant to be independent of.** It is now a covariate, not an exclusion.

**"The induction worked — 0 deferred in the null, 7 in the positive."**
Withdrawn. This compared the null arm's *post-gate* count against the positive
arm's *all-seeded* count — different denominators. Ungated the arms are 33% and
35%, Fisher p = 0.59. That comparison is itself confounded: the arms also differ
in `max_new_tokens` (512 vs 1024).

**"Only 10 of 20 labelled episodes carry a score, and the 87% paraphrase
rejection is the largest cause."** Withdrawn. The 20→10 reduction is the
eligibility screens, not missing arms: every planner episode always had
`ablate_planner`, so every eligible labelled episode always had a score. Fixing
coverage changed *what the score measures*, not *how many episodes it measures* —
the scored count stayed at exactly 10 and the AUC moved toward chance.

**"Two workers with half the evidence each beat one agent holding all of it."**
Withdrawn on seed 1 — see §H.2.

**A literal-only span search was reported as a design limitation for weeks.** It
was a pipeline defect: MuSiQue writes a sub-answer canonically ("Josiah Fenn")
where the paragraph, having introduced him, says "Fenn". The search now runs four
levels and records which one it used; rescued matches are reported apart from
literal ones rather than pooled. On seed 0 the widened search recovered 0 of its
4 targets.

## L. Known limitations

Collected here for reference; discussed in the main text.

1. **Sample size binds everything.** 30 questions → ~20 labelled → ~10 scored →
   24 pairwise comparisons. ~150–200 questions would put the AUC interval inside
   ±0.15.
2. **No chain-of-thought baseline.** Thinking is disabled; the comparison against
   CoT monitoring is motivated, not measured.
3. **The detector is unproven**, and not because of score resolution (§H.4).
4. **Planner influence is not separable from worker influence** on matched
   episodes (§H.6).
5. **`ablate_planner` confounds dictation with coordination** (§H.6).
6. **Lexical detectors are floors** — pushback selection, the contest test and
   natural-conflict detection all miss contradictions phrased outside their
   vocabulary.
7. **Override is untested**, not refuted (§H.7).
8. **One model, one prompt arm, both induced** (§I).
9. **The session/expert-deal arm is built but unrun.** It inverts the deal so one
   worker holds the whole chain, and adds a declining token-and-turn budget with
   a flat control. No session data is reported here.
10. **Two deferred episodes score exactly 0.000** — the maximally
    evidence-derived signature in episodes labelled as deference. Either the
    score or the label is wrong on those two, and we have not resolved which.
