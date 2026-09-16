#!/usr/bin/env bash
set -euo pipefail
# Score a finished provenance collection. CPU-only and re-runnable: it reads the
# episodes and replays already on the Modal volume and writes audit.json beside
# them, so re-scoring a run costs no GPU.
#
# This is the step that produces the input every figure is drawn from. It does
# not collect anything -- run-prov-ind-seed{0,1}.sh does that first.
#
# As with the collection wrappers, --validate-only prints the resolved command
# and executes nothing; modal_run.py implements no such flag itself.
mode=--validate-only
output_name='prov-ind'
audit_name='audit-v2'
output_name_set=false
audit_name_set=false
while (( $# )); do
  case "$1" in
    --validate-only|--launch)
      mode=$1; shift ;;
    --output-name)
      if (( $# < 2 )) || [[ -z $2 || $2 == --* || $output_name_set == true ]]; then
        echo '--output-name requires one nonempty value and may appear only once' >&2; exit 2
      fi
      output_name=$2; output_name_set=true; shift 2 ;;
    --audit-name)
      if (( $# < 2 )) || [[ -z $2 || $2 == --* || $audit_name_set == true ]]; then
        echo '--audit-name requires one nonempty value and may appear only once' >&2; exit 2
      fi
      audit_name=$2; audit_name_set=true; shift 2 ;;
    *) echo 'Expected --validate-only, --launch, --output-name or --audit-name' >&2; exit 2 ;;
  esac
done
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export MODAL_PROFILE="${MODAL_PROFILE:-research-profile}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/agent-swarming-uv-cache}"
export PYTHONDONTWRITEBYTECODE=1
# A fresh --audit-name writes a new audit beside the old one rather than over it,
# which is what lets a re-score be compared against the score it replaced.
command=(modal run 'orchestrator/provenance/modal_run.py::run_audit'
         --output-name "$output_name" --audit-name "$audit_name")
if [[ $mode == --validate-only ]]; then
  printf 'MODAL_PROFILE=%s\n' "$MODAL_PROFILE"
  printf 'Would run (nothing executed):\n  %s\n' "${command[*]}"
  exit 0
fi
exec "${command[@]}"
