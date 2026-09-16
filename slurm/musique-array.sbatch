#!/usr/bin/env bash
#SBATCH --job-name=musique-2b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/musique-%A_%a.out
#SBATCH --error=logs/musique-%A_%a.err

set -euo pipefail

# Submit from the project root; pass the official MuSiQue-Ans JSONL as argument 1.
: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch}"
: "${SLURM_SUBMIT_DIR:?Missing submission directory}"
: "${1:?Usage: sbatch slurm/musique-array.sbatch /absolute/path/musique_ans_v1.0_dev.jsonl}"
cd "$SLURM_SUBMIT_DIR"
for executable in uv ollama nvidia-smi curl flock; do
    command -v "$executable" >/dev/null || { echo "Missing executable: $executable" >&2; exit 1; }
done
[[ -r "$1" ]] || { echo "Cannot read dataset: $1" >&2; exit 1; }
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "No Slurm-visible GPU allocated" >&2; exit 1; }
# Leave CUDA_VISIBLE_DEVICES exactly as set by Slurm.
nvidia-smi --query-gpu=name,memory.total --format=csv

MODEL=${MODEL:-qwen3.5:2b}
AGENTS=${AGENTS:-1}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-16384}
MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-2048}
STEPS=${STEPS:-24}
QUESTION_TIMEOUT=${QUESTION_TIMEOUT:-600}
RUN_ROOT=${RUN_ROOT:-"$SLURM_SUBMIT_DIR/runs/musique-${SLURM_ARRAY_JOB_ID}"}
SHARD_DIR="$RUN_ROOT/shard-$SLURM_ARRAY_TASK_ID"
mkdir -p "$SHARD_DIR"
# A resubmission may reuse RUN_ROOT, but never run the same shard simultaneously.
exec 9>"$SHARD_DIR/.lock"
flock -n 9 || { echo "Shard already active: $SHARD_DIR" >&2; exit 1; }
SESSION_DIR=$(mktemp -d "$SHARD_DIR/session.XXXXXX")

# One private server per array task, including when tasks share a four-GPU node.
PORT=$(uv run --offline --no-project python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
export OLLAMA_HOST="127.0.0.1:$PORT"
export OLLAMA_CONTEXT_LENGTH="$CONTEXT_LENGTH"
export OLLAMA_NUM_PARALLEL="$AGENTS"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NO_CLOUD=1
# Set OLLAMA_MODELS to an existing shared model store before submission if needed.
# No model downloads, package installation or automatic service restart in this job.
ollama serve >"$SESSION_DIR/ollama.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
ready=0
for ((attempt=0; attempt<60; attempt++)); do
    kill -0 "$SERVER_PID" 2>/dev/null || { cat "$SESSION_DIR/ollama.log" >&2; exit 1; }
    if curl --noproxy '*' --silent --fail --max-time 2 "http://127.0.0.1:$PORT/api/tags" >"$SESSION_DIR/models.json"; then
        ready=1
        break
    fi
    sleep 1
done
[[ "$ready" == 1 ]] || { echo "Ollama did not start" >&2; exit 1; }
# Catch a bind failure before accepting an unrelated server at the candidate port.
kill -0 "$SERVER_PID" 2>/dev/null || { cat "$SESSION_DIR/ollama.log" >&2; exit 1; }
ollama --version >"$SESSION_DIR/ollama-version.txt" 2>&1

batch_status=0
uv run --offline python -m orchestrator.simulated_web.musique_batch \
    --dataset "$1" --output "$SHARD_DIR" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards 4 \
    --model "$MODEL" --port "$PORT" --agents "$AGENTS" \
    --context-length "$CONTEXT_LENGTH" --max-output-tokens "$MAX_OUTPUT_TOKENS" --steps "$STEPS" \
    --timeout "$QUESTION_TIMEOUT" --seed "${SEED:-0}" || batch_status=$?

# Export even when individual agents hit limits; preserve the batch failure status.
# Each shard exports its subset; the last finished shard also exports the full run.
uv run --offline python -m orchestrator.simulated_web.musique_export \
    --dataset "$1" --run-root "$RUN_ROOT" --shard "$SLURM_ARRAY_TASK_ID"
exit "$batch_status"
