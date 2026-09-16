#!/usr/bin/env bash
#SBATCH --job-name=containment-diagnostic
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/containment-%j.out
#SBATCH --error=logs/containment-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SLURM_SUBMIT_DIR:?Missing submission directory}"
: "${CUDA_VISIBLE_DEVICES:?No allocated GPU}"
[[ "$#" == 0 ]] || { echo 'Usage: sbatch slurm/containment-diagnostic.sbatch' >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
for executable in uv ollama curl nvidia-smi; do
    command -v "$executable" >/dev/null || { echo "Missing executable: $executable" >&2; exit 1; }
done
export OLLAMA_MODELS=${OLLAMA_MODELS:-/scratch/users/your-user/ollama-models}
PILOT_MODEL=${PILOT_MODEL:-qwen3.5:9b}
[[ "$PILOT_MODEL" =~ ^[a-zA-Z0-9._-]+:[a-zA-Z0-9._-]+$ ]] || { echo 'Expected a locally cached library model:tag' >&2; exit 1; }
[[ -r "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/${PILOT_MODEL%:*}/${PILOT_MODEL##*:}" ]] || { echo 'Model must be cached before submission' >&2; exit 1; }
uv run --offline --no-project python -m orchestrator.containment_diagnostic.runner --help >/dev/null
PILOT_PORT=$(uv run --offline --no-project python - <<'PY'
import socket
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    print(sock.getsockname()[1])
PY
)
PILOT_ROOT="$SLURM_SUBMIT_DIR/runs/containment-${SLURM_JOB_ID}"
[[ ! -e "$PILOT_ROOT" ]] || { echo 'Run path already exists' >&2; exit 1; }
mkdir "$PILOT_ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv >"$PILOT_ROOT/gpu.csv"
export OLLAMA_HOST="127.0.0.1:$PILOT_PORT"
export OLLAMA_CONTEXT_LENGTH=8192
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NO_CLOUD=1
export OLLAMA_VULKAN=0
ollama serve >"$PILOT_ROOT/ollama.log" 2>&1 &
PILOT_SERVER_PID=$!
cleanup() {
    kill "$PILOT_SERVER_PID" 2>/dev/null || true
    wait "$PILOT_SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
PILOT_READY=0
for ((attempt=0; attempt<60; attempt++)); do
    kill -0 "$PILOT_SERVER_PID" 2>/dev/null || { echo 'Pilot server exited' >&2; exit 1; }
    if curl --noproxy '*' --silent --fail --max-time 2 "http://$OLLAMA_HOST/api/tags" >"$PILOT_ROOT/models.json"; then
        PILOT_READY=1
        break
    fi
    sleep 1
done
[[ "$PILOT_READY" == 1 ]] || { echo 'Pilot server did not become ready' >&2; exit 1; }
ollama --version >"$PILOT_ROOT/ollama-version.txt" 2>&1
uv run --offline --no-project python -m orchestrator.containment_diagnostic.runner \
    --mode ollama --model "$PILOT_MODEL" --run-dir "$PILOT_ROOT/diagnostic" \
    --base-url "http://$OLLAMA_HOST" --timeout 120 --max-tokens 512
