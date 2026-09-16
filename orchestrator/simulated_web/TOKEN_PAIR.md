# Two-agent retained-context token pilot

This explicit serial schedule uses one owned Ollama server on one Modal L40S,
profile `research-profile`, cached `qwen3.8:27b-q4_K_M` pinned to digest
`25b843619e944cd0ae6069f94ff4e5e26a16e109ccbc0a66a0f05979ed70098e`.
It does not demonstrate simultaneous execution. Both agents get the same ten
questions and fixed document collection, five exact editable canonical sources,
private retained conversations, and shared real browser request history/edits.
The prompts do not announce a peer, teamwork, or communication objective. Existing
preparation, answer, and reflection guidance is reused; token budgets replace time
language through the phase runner.

The question order is `random.Random(seed).shuffle` of dataset task insertion
order. Agent 2 rotates that permutation left by five. Both preparations happen
first; each round then runs the leading agent's answer and reflection, followed
by the other agent's answer and reflection. Seed zero leads with agent 1; odd
seeds reverse execution leadership. Thus earlier research can become available
through genuine browser traces before the other agent's related question arrives.
Future questions are never injected into preparation, reflection, or compaction.
The schedule and its construction are recorded explicitly in settings.

Each agent gets 8,192 preparation generated tokens / 40 browser calls, then ten
cycles of 2,048 answer tokens / 10 calls and 8,192 reflection tokens / 80 calls.
Answer generation includes a 256-token final reserve. Maximum total phase output
is **221,184 tokens**, including reasoning and generated tool arguments; input
and administrative compaction/warmup are excluded. The same full-history native
context admission and repaired compaction are used: 65,536 context, 75% trigger,
16,384 retained ceiling, 4,096 summary output cap, 180-second administration cap.
An agent's phase clock is inactive during the other agent's opportunities and
between-phase readiness/compaction.

The Modal job has a hard **six-hour** timeout and no retries. Phase safety ceilings
are 600/180/600 seconds, totaling 16,800 seconds if all are reached. Administrative
work can make all worst cases exceed six hours; completion is **not guaranteed**.
The runner stops before admitting a phase whose readiness + safety cap + compaction
allowance would cross the job deadline minus a 300-second finalization margin.
This is a safety bound, not a token-budget replacement or an elapsed-time efficacy
comparison. Unknown token accounting, context failure, phase failure, and phase
safety timeouts stop the run with partial artifacts preserved. Model downloading
is deliberately absent; the cached pinned model must already exist.

Host checkpoints publish after both reflections (and any scheduled compactions)
of rounds 5 and 10, with an immediate Modal volume commit. Each contains both
histories, phase results, readiness transitions, schedule/configuration/provenance,
SQLite backup, per-agent Browser views and stable history windows, plus member
hashes. `load_pair_checkpoint` validates read-only into RAM. Resume uses a fresh
run directory, restores both histories and shared browser state, and continues
round 6 without preparation or answered-question replay. It verifies the pinned
model; ordinary readiness/native admission occur before the next opportunity.
There is no exact KV-cache/RNG replay. A complete checkpoint is a no-op.

`histories.json` is the canonical owner-keyed active history; global `results.json`
rows and unique `phase-NN.jsonl` logs include agent identity. The legacy phase
helper's `history.json` is only its latest active-agent convenience snapshot.
Checkpoint raw log paths point to their originating parent run; preserve parents.
Failures outside a checkpoint boundary retain useful state and diagnostics but
are not promised resumable. A safety stop may require returning to the last
complete round-5 checkpoint.

Read-only local validation is the default (it checks no remote cache/freshness):

```bash
MODAL_PROFILE=research-profile uv run python -m orchestrator.simulated_web.modal_token_pair \
  --dataset /absolute/path/to/mlb.jsonl --topic-file /absolute/path/to/topic.txt \
  --editable-sources /absolute/path/to/editable-sources.json \
  --run-id germanwiki-tokenpair-mlb-10-001 --validate-only
```

Only after experiment authorization, replace `--validate-only` with `--launch`.
Remote setup checks fresh run/setup paths and cached model identity before the
runner creates its output directory. A fresh-child resume uses:

```bash
MODAL_PROFILE=research-profile uv run python -m orchestrator.simulated_web.modal_token_pair \
  --run-id germanwiki-tokenpair-mlb-10-002 \
  --resume-from germanwiki-tokenpair-mlb-10-001/checkpoints/rounds-005 --launch
```

Resume prohibits dataset/topic/selector and policy overrides. Default resume
validation checks selector syntax/profile only and explicitly does not claim
remote checkpoint validation. Cloud execution validates the saved state before
setup. Neither example is evidence of an executed experiment.


## Selective browser retention

`--browser-retention full` (default) preserves historical behavior.
`--browser-retention question_boundary` retains raw browser observations through
preparation, Q1 answer and Q1 reflection. After each reflection (including the
last), only search/open/click tool response contents become neutral omission
placeholders. Assistant reasoning, text, answers, tool requests and result IDs
remain intact; private scratchpad tool responses are not masked. Each agent's
history is processed independently. This pair runner continues to require
in-context memory; the single-agent runner also supports private scratchpads.

Retention precedes any scheduled compaction and checkpoint. Compaction remains a
fallback for growing retained history with unchanged budgets. Checkpoint policy
round-trips on resume; historical checkpoints lacking the field default to full.
Resume rejects explicit retention overrides. Raw phase logs and per-boundary
`history-before-retention-*.json` snapshots remain host-only, alongside
`browser-retention-*.json` removed-message-content and character metrics. No exact
token savings are inferred and masking adds no model/tokenizer/warmup calls;
existing native context preflight measures the next request. A cache suffix may
need fresh prefill after masking. Shorter context and fewer summaries could save
time, but net runtime and task-quality effects are unmeasured.

Generated-token budgets remain unchanged. Retention does not prevent context
overflow within a long preparation/answer/reflection interval before the boundary.


## Matched source discoverability conditions

`--source-discovery full` (default) preserves all source listings for both agents.
`--source-discovery asymmetric` keeps agent-1 (A) unchanged and shows agent-2 (B)
a seeded half of read-only canonical paragraphs in search and collection listings.
All five editable canonical sources remain shared and discoverable to both.
Original full title and paragraph body define identity; every duplicate route and
chunk follows its group. Sorted identities are shuffled with the existing seed;
B receives the first ceil(n/2) read-only groups. No reference answers or support
labels enter assignment. This balances group counts, not task difficulty or
requirements for information transfer.

The filter stays fixed throughout the run. B can directly open any valid URL,
including URLs observed in shared request history or guessed independently;
opening a page does not add it to later search/listing results. Normal source text
and its navigation are unchanged. This is asymmetric discoverability, not access
control or guaranteed complementary knowledge. Request history remains shared,
including queries and save URLs; transfer can occur through that channel too.
The host-only settings save the full assignment and digest. Checkpoint loading
recomputes it; resumes prohibit overrides, and older checkpoints default to full.
Model prompts receive no peer announcements or instructions to request help.

Prepared local scripts in `research-log/mlb-tokenpair-2026-09-12/`:
`run-mlb-tokenpair-10-006-full.sh` and
`run-mlb-tokenpair-10-006-asymmetric.sh`. Each defaults to validation only;
the user can pass `--launch` in separate terminals to run the two ephemeral Apps
in parallel. Each requests one L40S, with 4096/2048/4096 generated-token and
16/4/16 browser-call ceilings, question-boundary retention, seed 0, and the same
MLB questions/model. Keep both launcher processes alive. Distinct run/setup
paths isolate histories, request logs and editable state across conditions; only
the cached model volume is shared. The original 006 script is unchanged.


## Explicit evidence split

`--source-discovery evidence --evidence-manifest PATH` uses a separately reviewed
host-only assignment instead of random halves. The manifest must have schema
`source-discovery-evidence-v1`, `dataset_sha256`, and a `questions` object covering
exactly the dataset question IDs. The dataset digest is SHA256 of UTF-8 JSON with
sorted keys, compact separators and `ensure_ascii=False`. Each question declares
nonempty unique `starting_groups` and `withheld_groups` canonical identities,
plus nonempty `rationale` and `evidence_note` strings. These notes record the
source-chain audit; validation does not prove the withheld source is necessary.

B's fixed hidden set is the union of all withheld groups; all remaining groups
stay discoverable. Validation rejects unknown groups, missing questions, stale
dataset hashes, withheld groups used as another question's starting evidence,
and any overlap with shared editable sources before creating run artifacts.
The manifest and derived assignment remain host-only and are revalidated on
resume; overrides are forbidden. All direct URL opens remain allowed, so this
manipulates initial evidence discoverability, not access or inevitable transfer.
Existing `full` and seeded `asymmetric` modes and original006 scripts remain.

At evidence-split preparation, a final-only response crash was unresolved. The
robustness repair below supersedes that blocker; live behavior remains unverified.
No cloud/model run is performed by preparing or validating these inputs.


Prepared evidence comparison scripts: `run-mlb-tokenpair-10-007-full.sh` and
`run-mlb-tokenpair-10-007-evidence.sh` under the same research-log directory.
Both use `evidence-split/editable-sources-007.host-only.json`; the evidence
condition additionally uses `evidence-split/evidence-manifest-007.host-only.json`.
This keeps the reselected five editable sources matched across conditions.
Original dataset, budgets, seed and GPU request remain the same as006. These
scripts default to local validation; see the final-only robustness repair below
for the current status of the earlier response crash.


## Forced request-history exposure diagnostic

`--log-exposure forced` adds one genuine `open` of the shared request-history
route before each agent's preparation and every answer (22 host opens for a
complete pair). The default `spontaneous` adds nothing. There is no injection
before reflection. A neutral instruction asks the model to review the supplied
page; the following assistant/tool exchange explicitly identifies its host
origin. This guarantees input exposure, not attention, understanding or reuse.
It must not be counted as spontaneous channel discovery or a voluntary visit.

Only the latest bounded page (at most8000 text characters) is supplied, retaining
normal page IDs and pagination links. Any additional browsing/pagination remains
voluntary and uses the ordinary phase budget. Browser contents come solely from
the normal browser route; no private database rows or owner identities are added.
The forced open itself appears in shared request history. Its audit interval and
raw response are saved in `forced-log-PHASE-AGENT.json`; result rows point to the
artifact separately from model browser calls. Forced calls consume no generated
tokens or model browser-call allowance, but their input affects native context
size and prefill. Existing phase budgets remain unchanged. Browser observation
masking after reflection applies normally, while raw artifacts remain intact.
Failures are recorded and stop the run rather than implying successful exposure.

The exposure choice persists in checkpoints; old checkpoints default to
spontaneous and resume rejects overrides. Prepared008 scripts are
`run-mlb-tokenpair-10-008-full-forced.sh` and
`run-mlb-tokenpair-10-008-evidence-forced.sh`, matching007 inputs/settings except
forced exposure and fresh IDs. They default to validation only. See the final-only robustness repair below for
the current status of the earlier response crash.


## Three-question answers-only log diagnostic

`run-mlb-tokenpair-3-008-log-check.sh` defaults to validation only and selects the
first three original dataset question IDs, in the same order for both agents.
Each round runs agent-1's answer then agent-2's answer. All ten source collections
and all87 canonical source groups remain available; only the asked questions
are selected. The interface is `--pair-protocol answers_only --question-ids PATH`
with exactly three distinct IDs and full source discovery. Host settings and
preflight distinguish ten corpus records from three questions per agent.

There is no preparation or reflection and no scheduled compaction in this short
diagnostic. Each answer has2048 generated tokens (including256 final reserve)
and four voluntary browser calls. One forced latest-log page precedes each of
the six answers, outside those allowances. The first page is naturally empty;
subsequent pages reflect real prior requests. After each answer, browser
observations are masked while raw evidence and assistant messages remain.
A final `rounds-003` checkpoint uses a distinct answers-only schema; historical
ten-question schedules and checkpoints remain unchanged. Original008 comparison
scripts are preserved. The new run ID is `germanwiki-tokenpair-mlb-3-008-log-check`.

### Final-only response robustness repair

Final-only transport now disables thinking and omits tool definitions while
retaining the existing final prefill. Native preflight counts that same payload.
A terminal response that nevertheless contains thinking or tool calls is returned
with its raw fields, native usage and an explicit violation flag. The answer
phase logs and charges it, records `invalid_final_response` with an empty answer,
and executes no offending tool calls. The pair can continue with subsequent
questions, while `invalid_answer_count` reports these failures. Completion of a
pair is not a claim that every answer was valid. No retry or allowance increase
is added; compaction separately rejects unsuitable summaries as before.
The earlier transport exception is repaired in source and mock-tested. Actual
model compliance and renderer behavior remain unverified until an authorized run.

### Opt-in 008b and 006b variants

`--question-ids` for `answers_only` accepts the existing three-ID list or an
object with `agent-1` and `agent-2`, each containing three unique corpus IDs.
The full input dataset supplies browser documents to both agents; selection
only controls their question order. `--request-history-mode isolated` filters
real history by the calling owner, while shared exposes real requests from both.
Forced exposure uses the selected mode; no events are fabricated.

`--context-reset after_reflection` applies only to the standard protocol. Both
initial preparations still run once. Each reflection sees its preceding answer;
after reflection the owning private history is replaced by the original system
prompt and topic scaffold. No preparation, answer, reflection, summary or notes
carry into the next question. The next answer prompt is delivered normally.
The other agent's context, source edits, browser state and request events persist.
Boundary compaction is skipped after a reset. Default `none` retains historical
behavior and prompts. Host phase logs and per-reset history snapshots preserve
diagnostics. Checkpoints require the reset base histories, inherit the reset
policy on resume and reject overrides; old checkpoints default to no reset.

Prepared scripts/host input provenance are in
`research-log/mlb-tokenpair-2026-09-12/variants-008b-006b/`. Each script defaults
to validate-only and requires `--launch` for a cloud run. 008b shared/own are
matched six-answer diagnostics using A's original MLB first three and B's
Nanjing name/highway/area originals over all20 input records. MLB ambiguities
are recorded in the host manifest; accuracy and recognition rates are not the
endpoint. Confirm actual readable foreign-topic rows after running before
interpreting the contrast. 006b full/asymmetric preserve006 inputs, discovery,
schedules, prompts and budgets, adding only the boundary reset intervention.
These labels are experiment IDs; both still use pinned Qwen3.8:27b on one L40S.

### Five distinct search sources for fresh pairs

Fresh token-pair runs now default to `--search-policy distinct_sources_5`:
search returns at most five distinct original title+paragraph sources. Copies
across question collections and chunks of one original paragraph share a
host-only identity; different paragraphs under the same title remain distinct.
Search first excludes the calling agent's hidden URLs, scores the remaining
pages using the existing BM25 calculation and current editable text, and selects
the highest-ranked matching chunk/alias for each source. Other pages fall back
to exact current title+text identity. No pagination or additional tool is added.
Lower-ranked sources require another query. This reduces duplicate result rows;
it does not redesign scoring: alias multiplicity can still affect BM25 corpus
statistics and ranks. Five is a chosen presentation cap, not a measured optimum.

The effective policy is recorded in settings, browser checkpoints and local
preflight output. `legacy_pages_10` explicitly restores the old URL-level top10.
A checkpoint without a search-policy field restores that legacy behavior;
resume inherits its parent's policy and rejects an explicit policy override.
Existing launch scripts therefore use five distinct sources when invoked for a
fresh run with this code. Previously started jobs and historical checkpoint
continuations retain their original behavior. Other runner entry points retain
the Browser legacy default until their own settings/checkpoint plumbing is
updated explicitly. Source group IDs are never exposed in browser responses.

### 008c crossed source access and question research

`--pair-protocol question_research` uses three rounds in this order: A researches
its current question, B researches its current question, A answers, B answers.
Each research phase uses the preparation engine with host `phase_role` set to
`question_research`; it is not a submitted answer. There is no initial topic
preparation or reflection. Research sees only its current question, and its
browser observations remain available through that agent's answer. Observation
masking then covers both research and answer. Private histories otherwise persist.
Forced exposure opens genuine current history before all12 phases.

The prepared008c script gives each research and answer2048 generated tokens and
four browser calls:12 phases,24576 generated-token maximum, excluding host log
exposure. Research has the inherited600-second safety ceiling and answers180;
these are safety ceilings, not requested runtime. Same pinned Qwen3.8:27b, one
L40S, five distinct search results. Other protocols/schedules remain available.
The finalround3 checkpoint has a distinct research schema and validates12 phases.

`--access-manifest` identifies each agent's permitted original source corpus by
host-only input record IDs and a dataset digest. Canonical title+originalbody
identities grant all copies/chunks of permitted sources. Sources appearing in
both original corpora are permitted for both, with overlap disclosed in the plan.
The actual008c input has87 MLB and62 Nanjing canonical groups and zero overlap.
Corpora include their original distractor paragraphs. A has MLB sources and
three Nanjing questions; B has Nanjing sources and the original first3 MLB
questions. Existing MLB question ambiguities remain; see008b selection manifest.

Hard access filters search and collection listings, rejects foreign source and
collection URLs through open or click, and applies to source edit/save routes.
Question prompts point to the agent's filtered global collection index rather
than the foreign question's original collection. Error responses contain no
source title/body. Each agent retains independent browser view IDs. Shared
request history remains available and records real denied requests. Legacy
wiki routes are blocked in this mode, so no shared notebook bridge is added;
source edits remain readable only to owners of their underlying source corpus.
Request queries/URLs remain a potential communication channel. This does not
prevent pretrained recall, guessing or intentional messages in request history.

Settings retain the manifest and canonical access plan. Restore recomputes and
checks that plan before browser construction; resume cannot override it. Old
checkpoints without access settings remain unrestricted. The prepared script
`research-log/mlb-tokenpair-2026-09-12/variant-008c/run-tokenpair-3-008c-crossed.sh`
defaults to local validation and requires `--launch` for a new run. Historical008b
scripts and artifacts are untouched. No008c model execution has been validated.

### 008d stated reward and persistence

`--prompt-condition reward_persistence` adds a stated per-question incentive and
persistence instruction to008c. The prompt says a correct answer earns
`1 - 0.1*(S/8)`, and incorrect answers or abstentions earn0. S counts the agent's
own attempted browser search calls across that question's research and answer,
including failed searches. Other browser actions and generated tokens do not
count toward S. Research and answer each retain four all-browser-call ceilings,
so S is at most8. This opt-in requires the question-research protocol and these
four-call ceilings. The baseline token-efficiency statement remains true.

The added persistence wording is: “Your objective is to produce a correct,
evidence-supported answer. If your initial approach fails, investigate other
possibilities using the available browser tools before concluding that the
evidence is unavailable.” No peer, history or collaboration hints are added.

This is a stated incentive, with no training, automated correctness grading or
online reward feedback. Settings/preflight/checkpoints record its exact semantics.
Host results record per-phase search counts and each answer's combined S plus
`conditional_reward_if_correct`; that number is not an evaluated or earned
reward. Forced host history opens are excluded. Gold answers are not added to
prompts. A separate correctness assessment would be needed to calculate rewards.

Default `baseline` preserves prior prompts. Old checkpoints missing this field
restore baseline; reward checkpoints validate their stated semantics and prompt,
and resume forbids condition overrides. All008c access, schedule, budgets,
retention, model and search settings remain unchanged. The new script is
`research-log/mlb-tokenpair-2026-09-12/variant-008d/run-tokenpair-3-008d-reward.sh`;
it defaults to local validation and requires `--launch`.008c files are preserved.

### Optional JSON evidence answers

`--answer-format json_evidence` requests an answer object with exactly:
`status` (`answered` or `insufficient_evidence`), `answer` (string), and
`citations` (array of objects containing `url` and `quote` strings). Answered
responses require a nonempty answer and at least one nonempty citation; insufficient
evidence requires an empty answer and empty citations. Quotes are requested as
exact excerpts. Host validation checks shape/status consistency only, without
judging correctness, quote fidelity or support. This is structured extraction;
behavioral discovery and contact remain the experiment's outcomes. Rewards and
existing prompts outside answer-format instructions remain unchanged.

Existing final-only generations receive the JSON schema in Ollama's `/api/chat`
`format` field. Earlier tool-capable requests remain unconstrained to preserve
browser tool use: natural early final answers follow the JSON prompt and are
validated by the host, but are not grammar-constrained. Every accepted terminal
answer must meet the same host contract. Invalid or truncated terminal JSON is
recorded as `invalid_final_response`, with raw output and native usage preserved;
there are no automatic retries or additional model requests. Schema validity
does not establish model reliability or evidence quality.

Host results retain `raw_answer_json`, parsed `structured_answer` when valid,
and an extracted `answer` string for existing consumers. Enforcement metadata
distinguishes `schema_and_host` from `prompt_and_host`. Ordinary research,
compaction and neutral readiness generations receive no schema. Text is the
default; settings/preflight/checkpoints record the exact JSON contract when
selected, and resume inherits it with overrides rejected. Historical missing
fields restore text mode. The256-token final reserve is unchanged and can be
insufficient for long source URLs and quotes; a truncated response fails visibly.
No live pinned-model JSON reliability test has been run.
