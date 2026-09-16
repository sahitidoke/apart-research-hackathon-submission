# FarmShare parallel FP8 pilot (prepared; not launched)

The new `orchestrator.simulated_web.farmshare_parallel_notebooks.py` runs the
central notebook coordinator with two independent owned vLLM processes on one
Slurm host. It imports no Modal entrypoint. It fixes discovery-only source access,
mandatory post-answer freeform notes and shared searchable request metadata.
Three questions × two agents × (4096 QA + two 2048-token note attempts) gives a
49152 generated-token ceiling. Unanswered/token-exhausted questions do not acquire
a fabricated final answer or mandatory note. The own-agent publication gate does
not block the peer. No response bodies or note text enter the shared request log.

Exact model remains `Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, all 76 locked artifacts;
vLLM `0.24.0`, Transformers `5.8.0`, fine-grained FP8 weights and BF16 KV cache.
No model, precision, parser, context or runtime fallback is performed.

## Verified remote inspection, 2026-09-13

Read-only SSH to `login.farmshare.stanford.edu` reached `rice-01`.
`sinfo`/`scontrol` reported GPU partition `oat-[01-06]`, four L40S 48GB GPUs
per host, generic `gpu:4` GRES, and feature `GPU_SKU:L40S`. The GPU partition
uses QoS `gpu`; this user's `operator` association allows that QoS. Its reported
per-user GPU maximum is four. No user job was listed at inspection time.
Mixed node state does **not** establish available GPUs: oat-02 was MIXED with
all four GPUs allocated. Queue delay and immediate two-GPU availability remain unknown.

`MaxMemPerCPU=4000` MiB means the launcher requests 32 CPUs for 128000 MiB RAM,
a conservative near-128GB reservation, with one node, two GPUs and six hours.
Maximum requested GPU allocation is 12 GPU-hours. Lower RAM requires an explicit
resource choice; it was not validated. `sbatch --cpus-per-task=N --mem=...` can
change those values after review (keep memory <= 4000 × CPUs).

Login Python is 3.12.3 and `~/.local/bin/uv` exists in a login shell. No vLLM or
Transformers was found in system Python; no Apptainer/Singularity was found in
login PATH. The two standard HF cache paths checked did not contain this model.
This is not a full filesystem cache inventory. `/farmshare/user_data/your-user`
and `/scratch/users/your-user` exist. Existing `~/agent-swarming` is a dirty older
checkout on `your-user/multi-round-questioning`; do not overwrite it. Its `.venv`
exists but its package inventory was not checked. No remote file was changed.

## Preparation and execution boundaries

Remote execution is blocked until a reviewed new staging directory contains the
current source, host-only inputs, a Python 3.12 runtime with the exact packages,
and a hash-verified model cache. No install, transfer, weights download or Slurm
submission was performed. Native Linux installation is the prepared path; GPU
compatibility of this exact vLLM/CUDA stack on FarmShare remains unverified.

After separate authorization, stage into a **fresh** directory rather than the
existing dirty checkout. Review the actual source plus these six host-only files:

- `research-log/mlb-tokenpair-2026-09-12/variants-008b-006b/combined-008b.host-only.jsonl`
- `research-log/mlb-tokenpair-2026-09-12/variants-008b-006b/topic.txt`
- `research-log/mlb-tokenpair-2026-09-12/evidence-split/editable-sources-007.host-only.json`
- `research-log/async-fp8-notebooks-2026-09-13/question-ids-3mlb.host-only.json`
- `research-log/mlb-tokenpair-2026-09-12/variant-008c/access-008c.host-only.json`
- `research-log/async-fp8-notebooks-2026-09-13/visible-labels.host-only.json`

An environment preparation command, **unrun and requiring authorization**, is
`uv venv --python python3 /absolute/fresh/runtime`, followed by
`uv pip install --python /absolute/fresh/runtime/bin/python 'vllm==0.24.0' 'transformers==5.8.0' 'huggingface-hub==1.21.0'`.
This can download substantial dependencies; do not run it incidentally during preflight.
The launcher never invokes package installation or `uv sync`.

From the reviewed staged project root, set explicit absolute paths:

```bash
export PATH="$HOME/.local/bin:$PATH"
export FP8_PYTHON=/absolute/fresh/runtime/bin/python
export PATH="$(dirname "$FP8_PYTHON"):$PATH"
args=(
  --run-id germanwiki-9b-async-fp8-parallel-farmshare-3mlb-20260913
  --run-root /absolute/fresh/results
  --cache-dir /absolute/shared/hf-cache
  --dataset research-log/mlb-tokenpair-2026-09-12/variants-008b-006b/combined-008b.host-only.jsonl
  --topic-file research-log/mlb-tokenpair-2026-09-12/variants-008b-006b/topic.txt
  --editable-sources research-log/mlb-tokenpair-2026-09-12/evidence-split/editable-sources-007.host-only.json
  --question-ids research-log/async-fp8-notebooks-2026-09-13/question-ids-3mlb.host-only.json
  --access-manifest research-log/mlb-tokenpair-2026-09-12/variant-008c/access-008c.host-only.json
  --visible-labels research-log/async-fp8-notebooks-2026-09-13/visible-labels.host-only.json
)
# Default: input validation only, no output directory/cache/runtime/GPU access.
uv run --offline --no-project --python "$FP8_PYTHON" python \
  -m orchestrator.simulated_web.farmshare_parallel_notebooks "${args[@]}"
# Separately authorized weights download only; no model/GPU launch.
uv run --offline --no-project --python "$FP8_PYTHON" python \
  -m orchestrator.simulated_web.farmshare_parallel_notebooks "${args[@]}" --download-model
# Separately authorized submission; never invoke the .sbatch script on login directly.
sbatch slurm/fp8-parallel-notebooks.sbatch "${args[@]}"
```

The download preserves partial cache files on failure and fetches only exact lock
filenames at the pinned revision. Execution rechecks input freshness, Slurm device
allocation, package versions and all artifact hashes before creating setup/run
outputs. `CUDA_VISIBLE_DEVICES=2,3` is preserved as `2` and `3`, never replaced by
physical assumptions `0,1`; this transport intentionally rejects GPU UUID tokens.
The Slurm device namespace and reported L40S names are checked without loading CUDA.
Port selection is dynamic and rechecked by each server; a race fails with preserved logs.

Watch `fp8-parallel-JOBID.out` for model-validation/startup messages and changed
question/token/request summaries, QA/note phase and latest mandatory-note
status, attempts, generated tokens and persistence result. There is no periodic
warmup stdout heartbeat; first model startup can remain quiet until its request completes. Detailed server logs and transport/failure metadata live at
`RUN_ROOT/RUN_ID-setup`; central `events.jsonl`, settings and results live under
`RUN_ROOT/RUN_ID`. SIGTERM attempts worker cancellation and preserves diagnostics;
Slurm hard-kill or node loss cannot guarantee final application-level writes.
No live GPU fit, inference overlap, process-group cleanup or node runtime checks
have been performed by this launcher preparation.

## Concrete helper for this checkout

From the local Mac, these two commands use the verified FarmShare account and
fresh fixed staging path. They are prepared commands, **not already executed**:

```bash
bash /path/to/agent-swarming/research-log/async-fp8-notebooks-2026-09-13/run-farmshare-fp8-notebooks.sh --prepare
bash /path/to/agent-swarming/research-log/async-fp8-notebooks-2026-09-13/run-farmshare-fp8-notebooks.sh --submit
```

`--prepare` validates locally, builds a preserved source/input bundle, creates
`/farmshare/user_data/your-user/agent-swarming-fp8-20260913-notes-v1`, transfers
only 41 required source/input files plus a manifest and remote helper, validates
the staged inputs, installs the pinned native runtime and downloads/hash-checks
weights into `/farmshare/user_data/your-user/agent-swarming-fp8-hf-cache`.
It does not submit a GPU job. This installation/download is substantial and
its live compatibility remains unverified. Existing `~/agent-swarming` is untouched.

`--submit` revalidates and submits exactly one two-L40S, 32-CPU, 128000MiB,
six-hour job. It prints the job ID and absolute stdout/stderr paths. A submission
marker rejects duplicate attempts, including uncertain submission failures.
The run ID is `germanwiki-9b-async-fp8-parallel-fs-notes-v1-20260913`.

No flag means validation only. `--bundle-only` builds a local review bundle
without network actions. SSH/SCP use BatchMode and a 15-second connection timeout;
if authentication is needed, first open `ssh your-user@login.farmshare.stanford.edu`
interactively to establish the configured shared connection, then retry the helper.

Preparation refuses an existing stage; useful partial files/logs remain. If its
remote helper was transferred and extracted successfully, resume the same source
and cached download after inspecting `prepare.log` using:

```bash
ssh your-user@login.farmshare.stanford.edu 'bash /farmshare/user_data/your-user/agent-swarming-fp8-20260913-notes-v1/remote-pilot.sh --prepare'
```

Local verification: helper and generated remote script pass `bash -n`; its
isolated extracted 41-file bundle imports and validates the actual six inputs,
reports 49152 tokens and creates no run root. Evidence:
`/private/tmp/farmshare-helper-staged-preflight.json`. No network mutation,
installation, weights download or scheduler submission was executed during preparation.
