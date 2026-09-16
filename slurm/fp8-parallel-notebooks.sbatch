#!/usr/bin/env bash
#SBATCH --job-name=fp8-parallel-notebooks
#SBATCH --partition=gpu
#SBATCH --constraint=GPU_SKU:L40S
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000M
#SBATCH --time=06:00:00
#SBATCH --output=fp8-parallel-%j.out
#SBATCH --error=fp8-parallel-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch after validating inputs}"
: "${SLURM_SUBMIT_DIR:?Missing submission directory}"
: "${FP8_PYTHON:?Set absolute path of reviewed Python 3.12 vLLM environment}"
[[ -x "$FP8_PYTHON" ]] || { echo 'FP8_PYTHON is not executable' >&2; exit 1; }
command -v uv >/dev/null || { echo 'uv must be installed and on PATH' >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
# Action flags are owned here, not accepted from forwarded input arguments.
for arg in "$@"; do
    case "$arg" in --execute|--download-model|--validate-only|--wall-seconds*) echo "Forbidden forwarded option: $arg" >&2; exit 1;; esac
done
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$FP8_PYTHON"):$PATH"
exec uv run --offline --no-project --python "$FP8_PYTHON" python -m orchestrator.simulated_web.farmshare_parallel_notebooks \
    --execute --wall-seconds 21600 "$@"
