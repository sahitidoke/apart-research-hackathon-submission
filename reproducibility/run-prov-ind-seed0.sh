#!/usr/bin/env bash
set -euo pipefail
# prov-ind seed 0: provenance collection, 30 questions (10 each at 2, 3, 4 hops)
# x 6 conditions + 30 seeded planner episodes = 210 episodes. Greedy decoding.
# Billed GPU job (L40S, several hours). No automatic launch.
#
# Unlike the 12a-12e wrappers, the runner does not implement --validate-only
# itself: orchestrator/provenance/modal_run.py has a plain Modal entrypoint. So
# validation happens here -- the resolved command is printed and nothing is
# executed unless --launch is given explicitly.
mode=--validate-only
extra=()
output_name='prov-ind'
output_name_set=false
while (( $# )); do
  case "$1" in
    --validate-only|--launch)
      mode=$1; shift ;;
    --output-name)
      if (( $# < 2 )) || [[ -z $2 || $2 == --* || $output_name_set == true ]]; then
        echo '--output-name requires one nonempty value and may appear only once' >&2; exit 2
      fi
      output_name=$2; output_name_set=true; shift 2 ;;
    --detach|--resume)
      if [[ $mode != --launch ]]; then
        echo 'Only --launch accepts --detach or --resume' >&2; exit 2
      fi
      extra+=("$1"); shift ;;
    *) echo 'Expected --validate-only, --launch, --output-name, --detach or --resume' >&2; exit 2 ;;
  esac
done
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export MODAL_PROFILE="${MODAL_PROFILE:-research-profile}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/agent-swarming-uv-cache}"
export PYTHONDONTWRITEBYTECODE=1
# No --dataset: modal_run.py defaults to MuSiQue-Ans from the Hub
# (hf:bdsaglam/musique:answerable:validation), which preserves the original
# AllenAI schema this package validates. Pass a local export only for a pinned
# or air-gapped copy; see input-manifest.json.
command=(modal run ${extra[@]+"${extra[@]}"} orchestrator/provenance/modal_run.py
         --output-name "$output_name" --per-hop 10 --hops 2,3,4
         --planner-style directive --worker-style cooperative
         --max-new-tokens 1024 --seed 0 --paraphrase-mode message,sentence)
if [[ $mode == --validate-only ]]; then
  printf 'MODAL_PROFILE=%s\n' "$MODAL_PROFILE"
  printf 'Would run (nothing executed):\n  %s\n' "${command[*]}"
  exit 0
fi
exec "${command[@]}"
