#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SETUP_ROOT=$(dirname "$SCRIPT_PATH")
EXPECTED_COMMIT=c3e36b65d69681c23e053042b0bc21e2331bad17
MATRIX="$SETUP_ROOT/minimal_cross_shape_cases.csv"
RUNNER="$SETUP_ROOT/run_generalization_campaign.py"

main() {
  local output_root=${1:?usage: run_minimal_cross_shape_campaign.sh OUTPUT_ROOT [--resume]}
  local repo_override=${HELION_REPO_ROOT-}
  local python_executable=${STRICT_PYTHON_EXECUTABLE:-python}
  local repo
  shift

  if (( $# > 1 )); then
    printf 'usage: %s OUTPUT_ROOT [--resume]\n' "$0" >&2
    return 2
  fi
  if (( $# == 1 )) && [[ $1 != --resume ]]; then
    printf 'usage: %s OUTPUT_ROOT [--resume]\n' "$0" >&2
    return 2
  fi

  repo=$(realpath "${repo_override:-$(git rev-parse --show-toplevel)}")
  test "$(git -C "$repo" rev-parse HEAD)" = "$EXPECTED_COMMIT"
  exec "$python_executable" "$RUNNER" "$output_root" \
    --repo "$repo" \
    --python "$python_executable" \
    --matrix "$MATRIX" \
    "$@"
}

main "$@"
