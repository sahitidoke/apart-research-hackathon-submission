#!/usr/bin/env bash
#SBATCH --job-name=musique-session
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/musique-session-%j.out
#SBATCH --error=logs/musique-session-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SLURM_SUBMIT_DIR:?Missing submission directory}"
[[ "$#" == 1 ]] || { echo 'Usage: sbatch slurm/musique-session.sbatch /absolute/path/pilot.jsonl' >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
for executable in uv ollama nvidia-smi curl; do
    command -v "$executable" >/dev/null || { echo "Missing executable: $executable" >&2; exit 1; }
done
[[ -f "$1" && -r "$1" ]] || { echo "Cannot read dataset: $1" >&2; exit 1; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'No Slurm-visible GPU allocated' >&2; exit 1; }

MODEL=${MODEL:-qwen3.5:9b}
AGENTS=${AGENTS:-2}
SERVER_MODE=${SERVER_MODE:-auto}
SEED=${SEED:-0}
STEPS=${STEPS:-36}
QUESTION_TIMEOUT=${QUESTION_TIMEOUT:-600}
PROTOCOL=${PROTOCOL:-standard}
if [[ "$PROTOCOL" == answer-reflection ]]; then
    CONTEXT_LENGTH=${CONTEXT_LENGTH:-65536}
else
    CONTEXT_LENGTH=${CONTEXT_LENGTH:-262144}
fi
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-8192}
PROMPT_CONDITION=${PROMPT_CONDITION:-maximal}
HISTORY_MODE=${HISTORY_MODE:-persistent}
SEARCH_SNIPPETS=${SEARCH_SNIPPETS-1}
case "$SEARCH_SNIPPETS" in
    1) SEARCH_SNIPPETS_MODE=on ;;
    0) SEARCH_SNIPPETS_MODE=off ;;
    *) echo 'SEARCH_SNIPPETS must be 0 or 1' >&2; exit 1 ;;
esac
PROTOCOL=${PROTOCOL:-standard}
PROTOCOL_ARGS=(--protocol "$PROTOCOL")
if [[ -n "${PREPARATION_TOPIC+x}" ]]; then
    PROTOCOL_ARGS+=(--preparation-topic "$PREPARATION_TOPIC")
fi
PROBE_ARGS=()
if [[ "${EDITABILITY_PROBE:-0}" == 1 ]]; then PROBE_ARGS=(--editability-probe); fi
REFLECTION_ARGS=()
if [[ "$PROTOCOL" == answer-reflection ]]; then
    : "${EDITABLE_SOURCE_TITLE:?Reflection requires original source title}"
    : "${EDITABLE_SOURCE_TEXT_SHA256:?Reflection requires original source text hash}"
    REFLECTION_ARGS=(--editable-source-title "$EDITABLE_SOURCE_TITLE"
        --editable-source-text-sha256 "$EDITABLE_SOURCE_TEXT_SHA256"
        --compaction-trigger "${COMPACTION_TRIGGER:-0.8}"
        --summary-tokens "${SUMMARY_TOKENS:-4096}" --recent-tokens "${RECENT_TOKENS:-4096}"
        --summary-generation-tokens "${SUMMARY_GENERATION_TOKENS:-8192}"
        --observation-window "${OBSERVATION_WINDOW:-0}")
    if [[ "${COMBINED_PHASE_BUDGETS:-0}" == 1 ]]; then
        REFLECTION_ARGS+=(--combined-phase-budgets)
    fi
fi
BUDGET_ARGS=()
if [[ -n "${TARGET_TOKENS+x}" ]]; then BUDGET_ARGS+=(--target-tokens "$TARGET_TOKENS"); fi
if [[ -n "${TOTAL_TOKEN_BUDGET+x}" ]]; then BUDGET_ARGS+=(--total-token-budget "$TOTAL_TOKEN_BUDGET"); fi
if [[ -n "${FINAL_RESERVE+x}" ]]; then BUDGET_ARGS+=(--final-reserve "$FINAL_RESERVE"); fi
WIKI_SEED_ARGS=()
if [[ -n "${WIKI_SEED_FILE+x}" ]]; then
    [[ -f "$WIKI_SEED_FILE" && -r "$WIKI_SEED_FILE" ]] || { echo "Cannot read WIKI_SEED_FILE: $WIKI_SEED_FILE" >&2; exit 1; }
    WIKI_SEED_ARGS=(--wiki-seed-file "$WIKI_SEED_FILE")
fi
QUESTION_ARGS=()
if [[ -n "${QUESTION_ID+x}" ]]; then
    QUESTION_ARGS=(--question-id "$QUESTION_ID")
fi
SHARD=0
SHARDS=1
RUN_ID=$SLURM_JOB_ID
if [[ -n "${SLURM_ARRAY_TASK_ID+x}" ]]; then
    : "${SLURM_ARRAY_JOB_ID:?Missing array job ID}"
    : "${SLURM_ARRAY_TASK_COUNT:?Missing array task count}"
    : "${SLURM_ARRAY_TASK_MIN:?Missing array minimum}"
    : "${SLURM_ARRAY_TASK_MAX:?Missing array maximum}"
    : "${SLURM_ARRAY_TASK_STEP:?Missing array step}"
    SHARD=$SLURM_ARRAY_TASK_ID
    SHARDS=$SLURM_ARRAY_TASK_COUNT
    RUN_ID="${SLURM_ARRAY_JOB_ID}-shard-${SHARD}"
    [[ "$AGENTS" == 1 && ( "$PROMPT_CONDITION" == neutral || "$PROMPT_CONDITION" == pressure ) ]] || {
        echo 'Session arrays require AGENTS=1 and PROMPT_CONDITION=neutral or pressure' >&2; exit 1;
    }
fi
# For arrays an explicit RUN_ROOT is a common prefix, never a shared run directory.
RUN_ROOT=${RUN_ROOT:-"$SLURM_SUBMIT_DIR/runs/musique-session-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"}
if [[ -n "${SLURM_ARRAY_TASK_ID+x}" ]]; then
    RUN_ROOT="${RUN_ROOT}-shard-${SHARD}"
fi
[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || { echo "Run root already exists: $RUN_ROOT; choose a fresh RUN_ROOT" >&2; exit 1; }
[[ "$PROMPT_CONDITION" == maximal || "$PROMPT_CONDITION" == neutral || "$PROMPT_CONDITION" == pressure || "$PROMPT_CONDITION" == maximal-pressure || "$PROMPT_CONDITION" == wiki-aware || "$PROMPT_CONDITION" == neutral-efficiency || "$PROMPT_CONDITION" == neutral-answer-cost || "$PROMPT_CONDITION" == neutral-all-cost || "$PROMPT_CONDITION" == neutral-reflection || "$PROMPT_CONDITION" == neutral-reflection-no-efficiency ]] || { echo 'PROMPT_CONDITION must be maximal, neutral, pressure, maximal-pressure, wiki-aware, neutral-efficiency, neutral-answer-cost, neutral-all-cost, neutral-reflection or neutral-reflection-no-efficiency' >&2; exit 1; }
[[ "$HISTORY_MODE" == persistent || "$HISTORY_MODE" == reset ]] || { echo 'HISTORY_MODE must be persistent or reset' >&2; exit 1; }
# Validate configuration before starting a server. Python avoids shell integer overflow.
GPU_CONFIG=$(uv run --offline --no-project python - "$AGENTS" "$SEED" "$STEPS" "$QUESTION_TIMEOUT" "$CONTEXT_LENGTH" "$MAX_OUTPUT_TOKENS" "$PROMPT_CONDITION" "$1" "$SHARD" "$SHARDS" "$HISTORY_MODE" "$SERVER_MODE" <<'PY'
import json
import math
import os
from pathlib import Path
import sys

from orchestrator.simulated_web.session import gpu_configuration, load_wiki_seed, prepare, validate_reflection_options, validate_editability_probe
from orchestrator.simulated_web.context import ContextPolicy

try:
    agents, seed, steps = map(int, sys.argv[1:4])
    timeout = float(sys.argv[4])
    context, output = map(int, sys.argv[5:7])
    if sys.argv[7] in ('maximal', 'maximal-pressure') and agents < 2:
        raise SystemExit('Maximal conditions require AGENTS >= 2')
    if not (1 <= agents <= 32 and 1 <= steps <= 500
            and math.isfinite(timeout) and 0 < timeout <= 3600
            and 1024 <= context <= 262144 and 1 <= output <= 32768):
        raise ValueError('out of range')
except ValueError as exc:
    raise SystemExit('Invalid configuration: require AGENTS 1..32, SEED an integer, '
                     'STEPS 1..500, QUESTION_TIMEOUT (0,3600], CONTEXT_LENGTH 1024..262144, '
                     'MAX_OUTPUT_TOKENS 1..32768') from exc
mode, devices = gpu_configuration(os.environ['CUDA_VISIBLE_DEVICES'], agents, sys.argv[12])
shard, shards = map(int, sys.argv[9:11])
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    minimum, maximum, step = (int(os.environ['SLURM_ARRAY_TASK_' + key])
                              for key in ('MIN', 'MAX', 'STEP'))
    if minimum != 0 or maximum != shards - 1 or step != 1:
        raise SystemExit('Use a contiguous zero-based array such as --array=0-3%4')
records = [json.loads(line) for line in Path(sys.argv[8]).read_bytes().splitlines() if line.strip()]
budget_values = [int(os.environ[key]) if key in os.environ else None
                 for key in ('TARGET_TOKENS', 'TOTAL_TOKEN_BUDGET', 'FINAL_RESERVE')]
if (sys.argv[7] in ('pressure', 'maximal-pressure') or os.environ.get('PROTOCOL') in ('preparation-urgency', 'initial-preparation-cycles', 'answer-reflection')) and steps < 2:
    raise SystemExit('Pressure condition requires at least two total turns')
pages, _, _, _ = prepare(records, agents, seed, sys.argv[7], shard, shards, os.environ.get('QUESTION_ID'), *budget_values, history_mode=sys.argv[11],
                         protocol=os.environ.get('PROTOCOL', 'standard'), preparation_topic=os.environ.get('PREPARATION_TOPIC'))
combined_phase_budgets = os.environ.get('COMBINED_PHASE_BUDGETS', '0')
if combined_phase_budgets not in ('0', '1') or (combined_phase_budgets == '1' and os.environ.get('PROTOCOL') != 'answer-reflection'):
    raise SystemExit('COMBINED_PHASE_BUDGETS must be 0 or 1; enabled requires answer-reflection')
observation_window = int(os.environ.get('OBSERVATION_WINDOW', '0'))
if observation_window < 0 or (observation_window and os.environ.get('PROTOCOL') != 'answer-reflection'):
    raise SystemExit('OBSERVATION_WINDOW requires answer-reflection and a nonnegative integer')
probe = os.environ.get('EDITABILITY_PROBE', '0')
if probe not in ('0', '1'):
    raise SystemExit('EDITABILITY_PROBE must be 0 or 1')
validate_editability_probe(probe == '1', os.environ.get('PROTOCOL', 'standard'))
policy = None
if os.environ.get('PROTOCOL') == 'answer-reflection':
    policy = ContextPolicy(context, float(os.environ.get('COMPACTION_TRIGGER', '.8')),
                           int(os.environ.get('SUMMARY_TOKENS', '4096')),
                           int(os.environ.get('RECENT_TOKENS', '4096')),
                           int(os.environ.get('SUMMARY_GENERATION_TOKENS', '8192')), observation_window, combined_phase_budgets == '1')
    if os.environ.get('WIKI_SEED_FILE') is not None:
        raise SystemExit('Reflection pilot does not accept separate wiki seeding')
validate_reflection_options(records, os.environ.get('PROTOCOL', 'standard'),
                            os.environ.get('EDITABLE_SOURCE_TITLE'), os.environ.get('EDITABLE_SOURCE_TEXT_SHA256'), policy)
load_wiki_seed(os.environ.get('WIKI_SEED_FILE'), pages)
print(mode, agents)
PY
)
read -r SERVER_MODE AGENTS <<< "$GPU_CONFIG"
IFS=',' read -r -a GPU_DEVICES <<< "$CUDA_VISIBLE_DEVICES"

# Each child receives its token from Slurm's CUDA namespace, not SLURM_JOB_GPUS
# or physical nvidia-smi indices (which can differ under cgroup isolation).
SERVER_COUNT=1
if [[ "$SERVER_MODE" == per-agent ]]; then SERVER_COUNT=$AGENTS; fi
PORT_LIST=$(uv run --offline --no-project python - "$SERVER_COUNT" <<'PY'
import socket
import sys

sockets = []
try:
    for _ in range(int(sys.argv[1])):
        handle = socket.socket()
        sockets.append(handle)
        handle.bind(('127.0.0.1', 0))
    print(' '.join(str(handle.getsockname()[1]) for handle in sockets))
finally:
    for handle in sockets:
        handle.close()
PY
)
read -r -a PORTS <<< "$PORT_LIST"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p "$SLURM_SUBMIT_DIR/runs"
SERVER_LOG_DIR=$(mktemp -d "$SLURM_SUBMIT_DIR/runs/musique-session-server-${RUN_ID}.XXXXXX")
printf 'Run root: %s\nServer logs: %s\nServer mode: %s\n' "$RUN_ROOT" "$SERVER_LOG_DIR" "$SERVER_MODE"
# Server artifacts stay outside RUN_ROOT, which is created only after all model
# endpoints pass the session CLI's metadata and matching-digest preflight.
export OLLAMA_CONTEXT_LENGTH="$CONTEXT_LENGTH"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NO_CLOUD=1
export OLLAMA_VULKAN=0
# Export OLLAMA_MODELS before submission to select an existing shared model store.
SERVER_PIDS=()
cleanup() {
    local pid
    for pid in "${SERVER_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${SERVER_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'server_mode\t%s\nallocated_cuda_visible_devices\t%s\n' "$SERVER_MODE" "$CUDA_VISIBLE_DEVICES" >"$SERVER_LOG_DIR/routing.tsv"
for ((index=0; index<SERVER_COUNT; index++)); do
    devices=$CUDA_VISIBLE_DEVICES
    parallel=$AGENTS
    if [[ "$SERVER_MODE" == per-agent ]]; then
        devices=${GPU_DEVICES[index]}
        parallel=1
    fi
    printf 'server_%s\t127.0.0.1:%s\tCUDA_VISIBLE_DEVICES=%s\tOLLAMA_NUM_PARALLEL=%s\n' \
        "$((index + 1))" "${PORTS[index]}" "$devices" "$parallel" >>"$SERVER_LOG_DIR/routing.tsv"
    CUDA_VISIBLE_DEVICES="$devices" OLLAMA_HOST="127.0.0.1:${PORTS[index]}" \
        OLLAMA_NUM_PARALLEL="$parallel" ollama serve >"$SERVER_LOG_DIR/ollama-$((index + 1)).log" 2>&1 &
    SERVER_PIDS+=("$!")
done
for ((index=0; index<SERVER_COUNT; index++)); do
    ready=0
    for ((attempt=0; attempt<60; attempt++)); do
        for pid in "${SERVER_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null || { cat "$SERVER_LOG_DIR"/ollama-*.log >&2; exit 1; }
        done
        if curl --noproxy '*' --silent --fail --max-time 2 \
            "http://127.0.0.1:${PORTS[index]}/api/tags" >"$SERVER_LOG_DIR/models-$((index + 1)).json"; then
            ready=1
            break
        fi
        sleep 1
    done
    [[ "$ready" == 1 ]] || { echo "Ollama server $((index + 1)) did not start" >&2; exit 1; }
done
for pid in "${SERVER_PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null || { cat "$SERVER_LOG_DIR"/ollama-*.log >&2; exit 1; }
done
OLLAMA_HOST="127.0.0.1:${PORTS[0]}" ollama --version >"$SERVER_LOG_DIR/ollama-version.txt" 2>&1
ENDPOINT_ARGS=(--port "${PORTS[0]}")
if [[ "$SERVER_MODE" == per-agent ]]; then ENDPOINT_ARGS=(--agent-ports "${PORTS[@]}"); fi

uv run --offline --no-project python -m orchestrator.simulated_web.session \
    --dataset "$1" --run-dir "$RUN_ROOT" --model "$MODEL" "${ENDPOINT_ARGS[@]}" \
    --gpu-devices "$CUDA_VISIBLE_DEVICES" --search-snippets "$SEARCH_SNIPPETS_MODE" \
    --agents "$AGENTS" --seed "$SEED" --steps "$STEPS" --timeout "$QUESTION_TIMEOUT" \
    --context-length "$CONTEXT_LENGTH" --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --prompt-condition "$PROMPT_CONDITION" --history-mode "$HISTORY_MODE" --shard "$SHARD" --shards "$SHARDS" "${QUESTION_ARGS[@]}" "${BUDGET_ARGS[@]}" "${WIKI_SEED_ARGS[@]}" "${PROTOCOL_ARGS[@]}" "${REFLECTION_ARGS[@]}" "${PROBE_ARGS[@]}"
