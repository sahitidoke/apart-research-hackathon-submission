# FarmShare MuSiQue array

Four fixed question shards, one allocated GPU and one local Ollama server per
array task. Default model is `qwen3.5:2b`; default is one agent per question for
calibration. Set `AGENTS=2` for the existing shared-wiki cohort condition.
The private/no-communication pair control is not implemented here.

## Before submitting

- Copy this project to FarmShare. Submit from its root, with `uv`, compatible
  Python (>=3.12), Linux Ollama supporting Qwen3.5, `curl`, `flock`, and NVIDIA
  GPU support available in the job environment. Prepare the uv environment
  ahead of time (`uv sync`); jobs use offline mode.
- Download **MuSiQue-Ans** using the [official repository](https://github.com/StonyBrookNLP/musique)
  and its `download_data.sh` instructions. Keep the original JSONL immutable and
  preserve its source/version and CC BY 4.0 attribution. No dataset is bundled
  or downloaded automatically by this array.
- Start with `musique_ans_v1.0_dev.jsonl` for accessible reference answers.
  Official test questions may omit gold annotations; this runner can produce
  predictions, but does not provide the withheld answers or a grading service.
- Install/download the model in advance using your existing Ollama setup:
  `ollama pull qwen3.5:2b`. If the store is on shared scratch, export
  `OLLAMA_MODELS` to that directory for both preparation and submission. All
  tasks must see the same model files. Do not run inference on a login node.
- FarmShare's documented batch partition is `normal`; the documented `gpu` QoS
  allows four jobs and six GPUs/user. The example `normal` QoS limits users to
  three GPUs. Verify your live association with `sacctmgr show assoc user="$USER"`
  and resources with `sinfo -o '%P %G %f'`; site configuration can change.
  See [FarmShare jobs](https://docs.farmshare.stanford.edu/slurm/).
- The script requests one **generic GPU**, because the live L40S GRES/feature
  label has not been verified. To specifically reserve L40S, add the exact
  GRES type or constraint reported by the cluster to the `sbatch` command.
  It logs GPU names; do not combine results from different hardware unnoticed.

## Submit

From the project root on FarmShare (create `logs/` before submitting; Slurm opens its log files before the script starts):

```sh
mkdir -p logs
sbatch slurm/musique-array.sbatch /absolute/path/musique_ans_v1.0_dev.jsonl
```

Defaults: four array tasks (`0-3%4`), 1 GPU, 4 CPU cores, 32 GB host RAM and
24 hours per task; 1 agent/question, 16K context, 24 tool-loop steps, 600 seconds
per agent, seed 0. The script does not assume four tasks start together.
The entire input file is used. For a pilot, provide a separately frozen JSONL
subset chosen before examining model outcomes; never silently filter failures.

For two-agent cohorts, on the same fixed input:

```sh
AGENTS=2 sbatch slurm/musique-array.sbatch /absolute/path/musique_ans_v1.0_dev.jsonl
```

Overrides are environment variables: `MODEL`, `AGENTS`, `CONTEXT_LENGTH`,
`STEPS`, `QUESTION_TIMEOUT`, `MAX_OUTPUT_TOKENS` (default 2048), `SEED`,
`RUN_ROOT`, `OLLAMA_MODELS`. Set `MAX_OUTPUT_TOKENS=8192` for the larger
per-response allowance, including thinking.
Ollama concurrency equals `AGENTS` and is fixed per job. Do not manually set
`CUDA_VISIBLE_DEVICES`; Slurm owns that mapping. Model tags are resolved to a
digest in each shard manifest. Compare the four manifests before pooling them.
Changing from 2B Q8 Ollama weights to a partner's trained BF16 checkpoint is a
precision/backend change as well as a training change; align checkpoints and
inference settings for a before/after MARL comparison.

## Outputs and restart

Slurm output: `logs/musique-<array-job-id>_<shard>.out` and `.err` under the project root.
Question progress is printed at start/end. Artifacts:

```text
runs/musique-<array-job-id>/shard-0/
  manifest.json                 # input/model/source hashes, settings, question IDs
  session.*/                    # server startup log, version, available model metadata
  <hashed-question-id>/
    source.json                 # original record, PRIVATE labels/decomposition
    pages.json                  # only paragraph text, titles and neutral navigation
    task.json                   # original question only
    attempt-001/                # standard runner logs, results, wiki, settings
    attempt-001.stdout
    outcome.json                # terminal attempt; exit status is not answer correctness
```

Question IDs are ordered by hop/shape prefix and hashed ID, then dealt
round-robin to four shards. Every input record is assigned exactly once. All
supplied paragraphs/distractors remain available. Long paragraphs become linked
parts without removing text; display titles can be shortened. A neutral index
links to the same Field notebook affordance used by the existing simulator.
The wiki, conversations and page handles start fresh for every question/cohort.
Question decompositions and support labels never enter browser pages.

If the 24-hour allocation ends early, resubmit the same shard IDs and reuse the
original output root (absolute path recommended):

```sh
RUN_ROOT=/absolute/path/project/runs/musique-ORIGINAL_JOB_ID \
  sbatch slurm/musique-array.sbatch /absolute/path/musique_ans_v1.0_dev.jsonl
```

Terminal outcomes, including errors, are retained and skipped. Interrupted
attempts without an outcome are preserved and get a new attempt directory.
Resume refuses changed data, model digest, runner source or experiment settings.
Use a new root for a new seed, configuration or deliberate repeat. A lock prevents
simultaneous writers to one shard. Inspect failures; a shard exits nonzero if any
question had a nonzero runner exit, but continues collecting the remaining tasks.

This is an inference harness. Official scoring is a separate command below;
behavioral outcome coding remains separate.
Official-format exports are now generated automatically as described below.
The simulated-browser adaptation is not directly comparable to official
reading-comprehension leaderboard scores.

## Validation status

Prepared by source inspection only. No syntax checks, tests, Slurm submission,
MuSiQue inference, dataset download, model download or FarmShare execution was
performed while writing these files. Before first submission, run:

```sh
bash -n slurm/musique-array.sbatch
uv run --offline python -m pytest -q
```

A small authorized GPU pilot is still needed to verify Linux Ollama compatibility,
actual GPU use, task latency, shard completion and exact dataset schema.

## Prediction exports

After each shard finishes, the Slurm script exports
`shard-N/predictions/agent-1.predictions.jsonl` (one file per agent),
`gold.jsonl`, `statuses.jsonl`, and `export.json`. The last finished shard also
writes combined files under `runs/musique-JOBID/predictions/`, in the original
input dataset order. Evaluate shard files against their accompanying gold file;
do not concatenate shard predictions and compare against the original input order.

Exports preserve complete final answers verbatim. Non-complete outcomes have
empty predicted answers and retain their actual status in `statuses.jsonl`.
There is no short-answer extraction or official scoring in this step. Long
explanations will depress official EM/F1. Supporting paragraph predictions are
empty placeholders; do not report their support score as evidence performance.
Answerability is true because this runner accepts MuSiQue-Ans only.

The exporter checks dataset hashes, question coverage, model/source consistency,
and terminal outcomes. If a shard is interrupted before producing all outcomes,
there is no complete combined export until it is successfully resumed. Individual
agent limits still export and do not change the batch's nonzero exit status.
A fresh run root is needed after source edits because the runner refuses source
changes on resume.

To export an existing finished run without model calls:

```sh
uv run --offline python -m orchestrator.simulated_web.musique_export \
  --dataset /absolute/path/to/input.jsonl \
  --run-root runs/musique-JOBID
```

## Official MuSiQue evaluation

Evaluate the saved raw predictions directly with the sibling MuSiQue evaluator.
This requires no model or GPU. For the recorded session:

```sh
SESSION_RUN=/home/your-user/agent-swarming/runs/musique-session-1711943
for agent in agent-1 agent-2; do
  uv run --offline --no-project python \
    /home/your-user/musique/evaluate_v1.0.py \
    "$SESSION_RUN/predictions/$agent.predictions.jsonl" \
    "$SESSION_RUN/predictions/gold.jsonl" \
    --output_filepath "$SESSION_RUN/$agent.raw-accuracy.json"
done
```

Read answer EM/F1. Verbose responses can receive low scores despite containing
an appropriate short answer. Support scores are not meaningful because support
predictions are empty placeholders. Recorded non-complete agent outcomes export
blank answers and remain identified in `statuses.jsonl`.

## Staggered questions with a persistent wiki

Use `musique-session.sbatch` for the explicitly encouraged collaboration condition.
It runs one coordinator process: all questions in the supplied file
share one live wiki for the entire session. The batch resource default remains one GPU;
request two GPUs below for one Ollama server per agent. Only single-agent neutral or pressure sessions support
assignment sharding with an array, as described below. Collaborative session arrays
are rejected before starting Ollama. The existing musique-array script above retains
its independent, fresh-wiki-per-question behavior and different retrieval corpus.

From the project root, using the existing frozen 20-question pilot and installed
model store:

```sh
mkdir -p logs
export OLLAMA_MODELS=/scratch/users/your-user/ollama-models
env -u QUESTION_ID -u RUN_ROOT -u TARGET_TOKENS -u TOTAL_TOKEN_BUDGET -u FINAL_RESERVE \
  PROMPT_CONDITION=maximal AGENTS=2 SEED=0 MODEL=qwen3.5:9b \
  SERVER_MODE=per-agent HISTORY_MODE=persistent CONTEXT_LENGTH=262144 \
  sbatch --export=ALL --gres=gpu:2 --cpus-per-task=8 --mem=64G \
  slurm/musique-session.sbatch \
  /scratch/users/your-user/datasets/musique/musique_ans_v1.0_dev_pilot20.jsonl
```

This submits **one job with two GPUs and two agents**. Each Ollama server sees one
entry from Slurm's `CUDA_VISIBLE_DEVICES`, listens on its own `127.0.0.1` port, and
uses `OLLAMA_NUM_PARALLEL=1`. Agent-1 uses the first endpoint and agent-2 the second.
One coordinator handles all browser calls against the same live SQLite wiki;
a save is available to the peer immediately during the slot, with no after-run
synchronization. Private histories and the barrier between question slots are unchanged.
Each GPU must independently fit the model and its context; the two GPUs do not pool
memory for a single agent. GPU execution and maximum-context fit remain unverified.

`SERVER_MODE=auto` is the default: two agents with two visible GPUs select
`per-agent`; one visible GPU selects `shared`, retaining the older one-GPU
workflow. Explicit `per-agent` requires exactly as many visible devices as agents;
other ambiguous multi-GPU counts fail in auto mode. Single-agent and array
workflows remain shared by default. For the older **one-GPU shared-server**
configuration, use the command above with `SERVER_MODE=shared`, `--gres=gpu:1`,
`--cpus-per-task=4`, and `--mem=32G`. Its one server uses
`OLLAMA_NUM_PARALLEL=2`; concurrent requests still depend on backend scheduling.

For the optional sourced starter-note condition, export this before the same
submission (omit it for an empty initial wiki):

```sh
export WIKI_SEED_FILE="$PWD/orchestrator/simulated_web/fixtures/hook-starter-note.json"
```

The launcher preserves Slurm's numeric device IDs or full GPU/MIG UUIDs verbatim;
it does not substitute physical 0/1 or `SLURM_JOB_GPUS` indices. Mixed index/UUID
lists, duplicates, malformed IDs and per-agent count mismatches fail before server
artifacts. Device identity is allocation provenance, not a runtime GPU-utilization
measurement; MIG identifiers refer to assigned instances. Slurm can renumber GPU
indices inside the job's cgroup; see [Slurm GPU environment](https://slurm.schedmd.com/gres.html).
Ollama documents [device selection](https://docs.ollama.com/gpu) and
[parallel-request settings](https://docs.ollama.com/faq). The launcher disables
Ollama's optional Vulkan backend so CUDA device visibility governs these NVIDIA jobs.

All endpoints must report the installed model with the same nonempty digest
before the session run directory is created. Startup failures preserve separate
server diagnostics and terminate all launched servers. `routing.tsv` records the
resolved server mode, ports, assigned CUDA tokens and parallelism; session
`settings.json` records `settings.server_mode`, `settings.agent_endpoints`, the
original device list and each endpoint's model metadata. Ports are selected while
all allocation sockets are held, then released for server startup; a later bind
failure aborts startup. No model downloads occur.

The default condition is `maximal`: agents know about their peer and the notebook,
are explicitly encouraged to save sourced findings before answering, and are
asked to conserve tokens. `PROMPT_CONDITION=neutral` uses the original question-
answering prompt plus the final-answer format instruction shared by both conditions.
Keep the dataset, agent count, seed and model settings identical
when comparing conditions; each submission must use fresh wiki state. This
comparison changes both collaboration encouragement and efficiency wording.
It does not isolate either component or establish spontaneous collaboration.

Every agent receives every question once, in a recorded seeded schedule. Agents
have distinct cyclic offsets of one shuffled question order, so they receive
different questions in each slot. All assignments in one slot finish before the
next slot begins. Each agent's own conversation and private page handles persist
across assignments by default (`HISTORY_MODE=persistent`). `HISTORY_MODE=reset`
restores fresh chats and handles per assignment. Agent identity and the shared
notebook persist in either mode. Per-question budgets, steps and timeouts reset. There must be at least as many
questions as agents; maximal encouragement requires at least two agents.
All supplied source paragraphs are searchable throughout
the session, with stable question-specific URLs. This expanded retrieval corpus
is a separate change from the original per-question array condition.

Defaults: 2 agents, `qwen3.5:9b`, seed 0, 36 steps and 600 seconds per assignment,
262144-token context and 8192 generated tokens per model response, with the same one-GPU,
4-CPU, 32 GB, 24-hour resource request as one array task. Overrides are `MODEL`,
`AGENTS`, `SEED`, `STEPS`, `QUESTION_TIMEOUT`, `CONTEXT_LENGTH`,
`MAX_OUTPUT_TOKENS`, `PROMPT_CONDITION`, `RUN_ROOT`, `SERVER_MODE`, and `OLLAMA_MODELS`.
Token conservation is a prompt instruction; existing generation limits remain
in force. No oracle, correctness feedback or extra post-answer turn is added.
The wiki's existing 100-write limit applies per agent across the entire session.
The session corpus must fit within the browser's 10,000-page limit.

The default root is `runs/musique-session-JOBID/`, with `settings.json`,
`manifest.json`, aggregate `results.json`, `web/wiki.sqlite3`, and per-slot
`assignments/slot-0001/logs/agent-N.jsonl` and `results.json`. Host-only task files
live under `tasks/<hashed-question-id>/task.json`. The runner writes raw exports
itself to `predictions/` in dataset order: `agent-N.predictions.jsonl`,
`gold.jsonl`, `statuses.jsonl`, and `export.json`. Do not invoke the old shard
exporter on a session root. Run the official evaluator directly on its raw
predictions using the command above; this job does not launch the evaluator.

Slurm logs are `logs/musique-session-JOBID.out` and `.err`. Ollama startup logs,
version, per-server `ollama-N.log` / `models-N.json` and `routing.tsv` are preserved separately in
`runs/musique-session-server-JOBID.XXXXXX/`; its exact path is printed in the
Slurm output. Keeping server files outside the run root lets the runner create
that root exclusively and refuse concurrent or accidental reuse.

Interrupted sessions cannot resume. Partial transcripts and wiki state are
preserved; retry with a fresh root. A completed collection can still contain
failed or truncated assignments: these retain their statuses and export empty
predicted answers. Completion is not correctness. This launcher has not been
submitted as part of implementing it; GPU/model behavior remains unverified.

Both session prompt conditions now request only a short, complete final answer,
without explanations, citations or Markdown, for direct MuSiQue scoring. This is
a prompt instruction rather than a separate hard final-answer token cap. The
8,192-token allowance includes thinking and applies to research/tool responses
as well. Existing 600-second assignment deadlines remain unchanged. An explicitly
exported `MAX_OUTPUT_TOKENS` overrides the default; set it to `8192` for the new
pilot if your shell still has an older value.

For a single-agent neutral session at Qwen3.5-9B's configured maximum context,
set `AGENTS=1,PROMPT_CONDITION=neutral,CONTEXT_LENGTH=262144` in the submission
export. Session context validation accepts 1024–262144 tokens; the default is
262144. This does not change the installed model precision, output budget, step
limit, or timeout. GPU memory and runtime support must be checked from the job logs.

### Four-GPU single-agent neutral pilot

Submit the same complete 20-question file to each of four one-GPU tasks. Each task
answers five disjoint questions from the seed-0 session schedule; all 20 questions'
documents remain searchable in every task. Do not split the dataset file or use
the older musique-array launcher for this comparison.

From the repository root (commands prepared only; not submitted here):

```sh
mkdir -p logs
export OLLAMA_MODELS=/scratch/users/your-user/ollama-models
sbatch --array=0-3%4 \
  --output=logs/musique-session-%A_%a.out \
  --error=logs/musique-session-%A_%a.err \
  --export=ALL,AGENTS=1,PROMPT_CONDITION=neutral,SEED=0,MODEL=qwen3.5:9b,CONTEXT_LENGTH=262144,MAX_OUTPUT_TOKENS=8192,STEPS=36,QUESTION_TIMEOUT=600 \
  slurm/musique-session.sbatch \
  /scratch/users/your-user/datasets/musique/musique_ans_v1.0_dev_pilot20.jsonl
```

The array requests up to four concurrent GPUs; actual start times depend on Slurm.
Each task retains the script's 32 GB host-memory and 24-hour requests. Maximum
context GPU fit and runtime remain unverified.

The launcher accepts only contiguous zero-based arrays and requires
`AGENTS=1` with `PROMPT_CONDITION=neutral` or `pressure`. CLI equivalents are `--shard 0 --shards 4`
through `--shard 3 --shards 4`. Shards take every fourth position of the original
full single-agent schedule, retaining `agent-1` and its seed 0 for generation.
Default unsharded settings remain unchanged.

Roots are `runs/musique-session-ARRAY_JOB_ID-shard-0/` through `-shard-3/`.
If `RUN_ROOT` is already exported, it is used as a prefix and `-shard-N` is
appended. Each root preserves the complete original `dataset.jsonl`, its SHA256,
all corpus pages, and schedule shard metadata (full order, assigned IDs and
original one-based slots). Each root's `predictions/gold.jsonl` and
`agent-1.predictions.jsonl` cover only that shard's five assignments, in original
dataset order. Evaluate each prediction file against its own subset gold file;
no merged export or automatic evaluation is performed.

Each shard starts a fresh wiki that persists across its five assignments. This
changes cross-question notebook history and the effective total write allowance:
the unchanged 100-write per-agent limit applies separately in each shard. It is
therefore not behaviorally identical to a single uninterrupted 20-question session,
even though initial document search, question prompts and generation seeds match.
Full dataset and shard validation happen before run/server directories are created;
failures after execution starts preserve partial artifacts and server logs.

### Rerun the three step-limit questions with 36 steps

The optional `QUESTION_ID` selects exactly one assignment while retaining the
complete 20-question document corpus and dataset provenance. It requires one
neutral or pressure agent and cannot be combined with multiple shards. Unknown/empty IDs,
duplicate dataset IDs and invalid source records fail before run/server artifacts.
Without `QUESTION_ID`, the existing session behavior is unchanged.

The following prepared command submits three independent one-GPU jobs, one per
previous step-limit question. It has not been run. Use the complete pilot file:

```sh
mkdir -p logs
export OLLAMA_MODELS=/scratch/users/your-user/ollama-models
for question_id in \
  4hop2__161602_426860_88460_20999 \
  3hop2__304722_667199_63959 \
  2hop__159215_779396
do
  env -u RUN_ROOT QUESTION_ID="$question_id" \
    AGENTS=1 PROMPT_CONDITION=neutral SEED=0 MODEL=qwen3.5:9b \
    CONTEXT_LENGTH=262144 MAX_OUTPUT_TOKENS=8192 STEPS=36 QUESTION_TIMEOUT=600 \
    sbatch --export=ALL slurm/musique-session.sbatch \
    /scratch/users/your-user/datasets/musique/musique_ans_v1.0_dev_pilot20.jsonl
done
```

Each job writes a fresh `runs/musique-session-JOBID/` with a fresh wiki, full
`dataset.jsonl` and `pages.json`, and one-question prediction/gold exports.
`settings.json` and `manifest.json` record `schedule.selection`: selected ID,
full original seeded order, source question count and original one-based slot.
The CLI equivalent is `--question-id ID`. Generation remains `agent-1`, seed 0;
only the selected assignment runs, with a 36-step cap. The 600-second deadline
and 8192-token per-response generation cap still apply. These are fresh attempts,
not continuations of previous transcripts or notebook state. Jobs and tests have
not been executed as part of preparing this selector.

### Token pressure with a cumulative generated-output cap

`PROMPT_CONDITION=pressure` adds an efficiency instruction to the neutral prompt:
target 3000 tokens across reasoning and tool arguments, with a hard requested cap
of 4000 **native generated tokens per question, including the final answer**.
This count includes generated reasoning, tool names/arguments and other assistant
text. Retrieved pages and other input tokens are excluded. It is a different
metric from the visible reasoning-plus-argument counts in `token_breakdown.py`;
that diagnostic remains separate and is not automatically run.

Defaults for this condition are `TARGET_TOKENS=3000`,
`TOTAL_TOKEN_BUDGET=4000`, and `FINAL_RESERVE=256`. CLI equivalents are
`--target-tokens`, `--total-token-budget`, and `--final-reserve`.
All are integers: total <=32768, 1 <= reserve < total, and
1 <= target <= total-reserve. Pressure requires at least two total model turns.
Budget options on neutral/maximal runs are rejected; without them, both older
conditions retain uncapped cumulative generation and their existing behavior.
Session steps remain 36 by default.

Each request's `num_predict` is bounded by the smaller of the per-response limit
and the remaining research allowance (3744 initially). Native `eval_count` is
deducted after every response. Once research reaches its allowance, a research
response is truncated, or only the last model turn remains, the runner requests
one final answer with tools omitted and `think:false`. This final request can
use the reserve plus unused balance, still bounded by the per-response limit.
It consumes one of the 36 turns; no 37th turn is added. A truncated research
response is logged in full; partial tool calls are never executed or forwarded
into finalization. Its textual reasoning/content can remain as context.

Missing/invalid native counts stop the assignment with `budget_error`. A backend
that exceeds its requested allowance also stops with `budget_error`; the observed
excess is retained, never clamped. Enforcement depends on the backend honoring
`num_predict` and accurately reporting counts; the client cannot undo tokens
already generated. A truncated final answer is `budget_limit`, distinct from
legacy step/generation limits. All non-complete outcomes export blank predictions.
Logs record request allowances, observed usage, remaining balance and phase;
results record accounting completeness, backend violations and `cap_verified`.
An uncounted response or budget error does not claim a verified cap.

Prepared four-GPU, full-pilot command (not submitted here):

```sh
mkdir -p logs
export OLLAMA_MODELS=/scratch/users/your-user/ollama-models
env -u QUESTION_ID -u RUN_ROOT \
  AGENTS=1 PROMPT_CONDITION=pressure SEED=0 MODEL=qwen3.5:9b \
  CONTEXT_LENGTH=262144 MAX_OUTPUT_TOKENS=8192 STEPS=36 QUESTION_TIMEOUT=600 \
  TARGET_TOKENS=3000 TOTAL_TOKEN_BUDGET=4000 FINAL_RESERVE=256 \
  sbatch --export=ALL --array=0-3%4 \
  --output=logs/musique-session-%A_%a.out \
  --error=logs/musique-session-%A_%a.err \
  slurm/musique-session.sbatch \
  /scratch/users/your-user/datasets/musique/musique_ans_v1.0_dev_pilot20.jsonl
```

This preserves the full document corpus, seed-0 assignments and per-shard output
layout. Each question has its own budget; shards retain separate fresh wikis.
The pressure prompt, cumulative cap and finalization behavior all change together,
so this comparison cannot isolate the soft prompt's effect. The 600-second
deadline still applies, including finalization; a final answer is not guaranteed.
Model/backend behavior and the prepared regression tests remain unrun.

### Maximal collaboration with and without pressure

`maximal-pressure` composes the unchanged maximal collaboration instructions
with exactly the same explicit target and budget as `pressure`: 3000 reasoning/
argument target, 4000 native generated tokens including the final answer, and
256 reserved for finalization. Both maximal conditions require at least two
agents and reject arrays and single-question selection, preserving one shared
wiki throughout the full session. Each agent gets a separate budget for each
question; no budget is pooled across agents.

The prepared loop below submits two matching one-GPU jobs, each with two agents
answering the same complete 20-question file. It clears inherited selectors, run
roots and budget overrides so the maximal control has no cumulative cap, while
maximal-pressure receives its standard 3000/4000/256 defaults. Both use the same
installed `qwen3.5:9b` model/precision; this does not requantize the model.

```sh
mkdir -p logs
export OLLAMA_MODELS=/scratch/users/your-user/ollama-models
for condition in maximal maximal-pressure
do
  env -u QUESTION_ID -u RUN_ROOT -u TARGET_TOKENS -u TOTAL_TOKEN_BUDGET -u FINAL_RESERVE \
    AGENTS=2 PROMPT_CONDITION="$condition" HISTORY_MODE=persistent SEED=0 MODEL=qwen3.5:9b \
    CONTEXT_LENGTH=262144 MAX_OUTPUT_TOKENS=8192 STEPS=36 QUESTION_TIMEOUT=600 \
    sbatch --export=ALL slurm/musique-session.sbatch \
    /scratch/users/your-user/datasets/musique/musique_ans_v1.0_dev_pilot20.jsonl
done
```

The maximal prompt already asks agents to conserve tokens. This comparison adds
the explicit numerical target, cumulative cap and reserved finalization together;
it cannot isolate the soft target's effect. Each job starts a separate fresh wiki
and writes `runs/musique-session-JOBID/`. These commands and the added regression
tests remain unrun; two-agent maximum-context GPU fit remains unverified.

Session history is retained separately for each agent on the host, without new
summarization or compaction. Backend templates may omit prior thinking, and growing
history can exceed even a 262144-token window, causing context omission or request
failure. The 2000 private page-view cap now spans the whole session in persistent
mode (per assignment in reset mode); the 100-write cap spans either session mode.
Keep history mode matched across comparisons and record it with context size.

### Preparation and urgent-answer protocol

Set `PROTOCOL=preparation-urgency` and `PREPARATION_TOPIC='PUBLIC TOPIC'`, and pass
a curated five-question JSONL dataset to the existing session launcher.
Keep `HISTORY_MODE=persistent` and select `PROMPT_CONDITION=maximal`, `neutral`,
`neutral-efficiency`, or `wiki-aware` (the latter requires `AGENTS` at least 2);
unset `TARGET_TOKENS`, `TOTAL_TOKEN_BUDGET`, `FINAL_RESERVE` and `QUESTION_ID`.
Do not use an array. Validation rejects incompatible options before starting
servers or creating server/run artifacts. Existing shared/per-agent GPU routing,
optional `WIKI_SEED_FILE`, fresh run-root and model-digest checks still apply;
the default model remains `qwen3.5:9b`.

`PROMPT_CONDITION=neutral-efficiency` adds the individual prompt-only correctness
minus generated-token cost objective and omits all prompt-level wiki/peer cues.
It requires this protocol, allows one or more agents, and leaves browser discovery
and editing available. There is no score calculation or feedback; see the
simulated-browser README for the exact objective and comparison caveat.

Budgets are fixed: initial preparation 8000, later preparation 4000, each answer
2000 including a 256-token final reserve. `STEPS` and `QUESTION_TIMEOUT` apply
per phase. The question-slot barrier remains, with independent prep-to-answer
transitions inside a slot. The current 20-question pilot must not be passed as
this protocol's dataset. The curated input is
`/scratch/users/your-user/datasets/musique/musique_ans_v1.0_dev_mlb_related5.jsonl`,
with public topic `Major League Baseball: championship records, player records,
awards, drafts and season dates.` (one line). Its adjacent manifest records
source-line fingerprints and the shared factual chain. These inputs have been
prepared; no preparation-protocol job has been launched. See the simulated-browser
README for phase visibility, logs and failure semantics. This documentation
does not authorize a submission.

For the paired standalone-initial-research design, use
`PROTOCOL=initial-preparation-cycles` with `PROMPT_CONDITION=neutral-answer-cost`
(user version) or `PROMPT_CONDITION=neutral-all-cost` (assistant version), and the
same `PREPARATION_TOPIC`, curated five-question dataset, agents, seed and resources.
Each run needs fresh wiki/output state. Both run 8000 topic-only research tokens
then five 4000+2000 cycles (256 final reserve), at most 38000 tokens per agent.
The former hides each question until answering and charges answer tokens only;
the latter reveals it before preparation and charges all generated tokens once.
Initial research failure aborts before assignments and preserves diagnostic logs.
These variables select the opt-in design; historical defaults remain unchanged.


### Terminal editability recall diagnostic

For `answer-reflection` sessions, opt in with `--editability-probe` (Slurm:
`EDITABILITY_PROBE=1`, default `0`). After all questions and the final successful
reflection, each agent receives one memory-only follow-up allowing yes/no/unsure,
asking for a remembered editable page/file, evidence, saving method and expected
persistence without naming the selected source or asserting one exists. The request
has no tools and disables thinking, with at most 1024 generated tokens (or the lower
client response limit). It uses the final retained, observation-masked history; no
new compaction, history restoration or truncation is performed. Estimated context
limits skip the probe explicitly; native boundary/accounting failures are recorded.
Context preflight is an estimate, not an exact tokenizer or backend-retention guarantee.

Host-only `editability-probe.json` records the exact request snapshot, response,
metadata, separate usage and per-agent status; `manifest.json` tracks diagnostic
completion separately. Task answers, exports, usage and scoring exclude the probe.
Probe failure preserves completed task results. Terminal recall does not prove
recognition during the task; no/unsure cannot distinguish never noticing from
forgetting. No automatic judge or correctness feedback is added.


For `musique-session.sbatch`, set `SEARCH_SNIPPETS=0` to return only titles and
URLs for all search hits; `[Editable]` display labels and ranking remain intact.
The default `SEARCH_SNIPPETS=1` retains snippets. Other values are rejected before
server startup or run artifacts. This maps to the session CLI's
`--search-snippets off|on` and is recorded in `settings.json`.
