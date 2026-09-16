#!/usr/bin/env bash
set -euo pipefail
# 12b seed1: six-question reference; fresh initial research plus all six questions.
# One H100, no history, generic rendered titles;123904full-path maximumtokens. No automatic launch.
mode=(--validate-only)
run_id='germanwiki-6-012b-fp8-20260913-seed1-answer600-v3-concurrent-12b'
run_id_set=false
resume_options=(--resume-round 2)
if (( $# )); then
  case "$1" in
    --validate-only|--launch) mode=("$1"); shift ;;
    --run-id|--resume-checkpoint|--resume-run-id|--resume-round) ;;
    *) echo 'Expected --validate-only, --launch or --run-id' >&2; exit 2 ;;
  esac
fi
while (( $# )); do
  case "$1" in
    --run-id)
      if (( $# < 2 )) || [[ -z $2 || $2 == --* || $run_id_set == true ]]; then
        echo '--run-id requires one nonempty value and may appear only once' >&2; exit 2
      fi
      run_id=$2;run_id_set=true;shift 2 ;;
    --resume-checkpoint|--resume-run-id|--resume-round)
      if (( $# < 2 )) || [[ -z $2 || $2 == --* ]]; then echo 'Resume option requires a value' >&2; exit 2; fi
      resume_options+=("$1" "$2");shift 2 ;;
    --detach|--download-model)
      if [[ ${mode[0]} != --launch ]]; then
        echo 'Only launch accepts --detach or --download-model' >&2; exit 2
      fi
      mode+=("$1");shift ;;
    *) echo 'Expected --run-id, --detach or --download-model' >&2; exit 2 ;;
  esac
done
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${INPUT_DIR:?Set INPUT_DIR to the directory containing the seven required input files}"
export MODAL_PROFILE="${MODAL_PROFILE:-research-profile}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/agent-swarming-uv-cache}"
export PYTHONDONTWRITEBYTECODE=1
exec uv run --frozen python -m orchestrator.simulated_web.modal_hf_reference_12b \
  --dataset "${INPUT_DIR}/dataset.host-only.jsonl" \
  --topic-file "${INPUT_DIR}/topic-research.txt" \
  --editable-sources "${INPUT_DIR}/editable-sources.host-only.json" \
  --question-ids "${INPUT_DIR}/question-ids-same-6.host-only.json" \
  --access-manifest "${INPUT_DIR}/access-manifest.host-only.json" \
  --run-id "$run_id" --question-count 6 --preparation-safety-seconds 1200 --model-seed 1 \
  --visible-labels "${INPUT_DIR}/visible-labels.host-only.json" --round-leaders "${INPUT_DIR}/round-leaders-6.host-only.json" --source-access-policy discovery-only-v1 --short-note-retry --notebook-context-policy 9e-v1 --notebook-quota-policy notebook-exempt-v1 --question-pairing-policy same-question-v1 "${resume_options[@]}" "${mode[@]}"
