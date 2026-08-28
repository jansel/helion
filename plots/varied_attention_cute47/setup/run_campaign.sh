#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
CAMPAIGN_PY="$SCRIPT_DIR/campaign.py"
STRICT_VALIDATOR="$SCRIPT_DIR/../../generalized_full_autotune/setup/build_strict_manifest.py"
lane_pids=()
TERM_GRACE_SECONDS=15
campaign_lock_fd=

usage() {
  cat >&2 <<'EOF'
Usage:
  run_campaign.sh OUTPUT_ROOT
  run_campaign.sh --resume OUTPUT_ROOT

First-run environment:
  VARIED_ATTENTION_SOURCE_REPO       Helion repository (default: git root)
  VARIED_ATTENTION_FA4_SOURCE_REPO   Existing FA4 repository (required)
  VARIED_ATTENTION_QUACK_SOURCE_REPO Existing Quack repository (required)
  VARIED_ATTENTION_PYTHON            Python executable (default: python)
EOF
}

remove_lane_pid() {
  local finished_pid=$1
  local pid
  local -a remaining=()
  for pid in "${lane_pids[@]}"; do
    if [[ $pid != "$finished_pid" ]]; then
      remaining+=("$pid")
    fi
  done
  lane_pids=("${remaining[@]}")
}

process_group_live() {
  local pid=$1
  local state
  while IFS= read -r state; do
    if [[ -n $state && $state != Z* ]]; then
      return 0
    fi
  done < <(ps -o stat= -g "$pid" 2>/dev/null || true)
  return 1
}

cleanup_lanes() {
  local pid deadline any_live
  for pid in "${lane_pids[@]}"; do
    if process_group_live "$pid"; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  deadline=$((SECONDS + TERM_GRACE_SECONDS))
  while ((SECONDS < deadline)); do
    any_live=0
    for pid in "${lane_pids[@]}"; do
      if process_group_live "$pid"; then
        any_live=1
        break
      fi
    done
    if ((any_live == 0)); then
      break
    fi
    sleep 1
  done
  for pid in "${lane_pids[@]}"; do
    if process_group_live "$pid"; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "${lane_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  lane_pids=()
}

resolve_python() {
  local requested=${VARIED_ATTENTION_PYTHON-python}
  local candidate
  candidate=$(command -v "$requested") || {
    printf 'Python executable not found: %s\n' "$requested" >&2
    return 1
  }
  realpath -e "$candidate"
}

require_external_output() {
  local output_root=$1
  local source_repo=$2
  case "$output_root" in
    "$source_repo" | "$source_repo"/*)
      printf 'OUTPUT_ROOT must be outside source repository: %s\n' \
        "$source_repo" >&2
      return 1
      ;;
  esac
}

acquire_campaign_lock() {
  local output_root=$1
  local lock_path="${output_root}.campaign.lock"
  mkdir -p "$(dirname "$lock_path")"
  exec {campaign_lock_fd}>"$lock_path"
  if ! flock -n "$campaign_lock_fd"; then
    printf 'another campaign process holds %s\n' "$lock_path" >&2
    return 1
  fi
}

adopt_campaign_lock() {
  local output_root=$1
  local inherited=${VARIED_ATTENTION_CAMPAIGN_LOCK_FD-}
  local expected actual
  if [[ ! $inherited =~ ^[0-9]+$ ]]; then
    echo "invalid inherited campaign lock descriptor" >&2
    return 1
  fi
  expected=$(realpath -m "${output_root}.campaign.lock")
  actual=$(readlink -f "/proc/$$/fd/$inherited" 2>/dev/null || true)
  if [[ $actual != "$expected" ]]; then
    printf 'inherited campaign lock points to %s, expected %s\n' \
      "${actual:-<closed>}" "$expected" >&2
    return 1
  fi
  if ! flock -n "$inherited"; then
    echo "inherited campaign lock is not held" >&2
    return 1
  fi
  campaign_lock_fd=$inherited
  unset VARIED_ATTENTION_CAMPAIGN_LOCK_FD
}

snapshot_launcher() {
  local output_root=$1
  local launcher="$output_root/launcher"
  mkdir -p "$launcher"
  if [[ -e $launcher/run_campaign.sh || -e $launcher/campaign.py || \
    -e $launcher/build_strict_manifest.py ]]; then
    echo "launcher snapshot already exists; use --resume" >&2
    return 1
  fi
  cp -- "$SCRIPT_PATH" "$launcher/run_campaign.sh"
  cp -- "$CAMPAIGN_PY" "$launcher/campaign.py"
  cp -- "$STRICT_VALIDATOR" "$launcher/build_strict_manifest.py"
  chmod 0555 \
    "$launcher/run_campaign.sh" \
    "$launcher/campaign.py" \
    "$launcher/build_strict_manifest.py"
}

run_parallel_lanes() {
  local python_executable=$1
  local output_root=$2
  local first_status finished_pid remaining_pid

  setsid "$python_executable" "$output_root/launcher/campaign.py" \
    run-lane "$output_root" --lane dense \
    >>"$output_root/dense-lane.log" 2>&1 &
  lane_pids+=("$!")
  setsid "$python_executable" "$output_root/launcher/campaign.py" \
    run-lane "$output_root" --lane causal \
    >>"$output_root/causal-lane.log" 2>&1 &
  lane_pids+=("$!")

  finished_pid=""
  if wait -n -p finished_pid "${lane_pids[@]}"; then
    first_status=0
  else
    first_status=$?
  fi
  if ((first_status != 0)); then
    cleanup_lanes
    return "$first_status"
  fi
  if [[ -n $finished_pid ]]; then
    remove_lane_pid "$finished_pid"
  fi

  if ((${#lane_pids[@]})); then
    remaining_pid=${lane_pids[0]}
    if wait "$remaining_pid"; then
      first_status=0
    else
      first_status=$?
    fi
    if ((first_status != 0)); then
      cleanup_lanes
      return "$first_status"
    fi
    remove_lane_pid "$remaining_pid"
  fi
}

main() {
  local mode=new
  if [[ ${1-} = --resume ]]; then
    mode=resume
    shift
  fi
  if [[ $# != 1 ]]; then
    usage
    return 2
  fi

  local output_root python_executable source_repo fa4_source_repo
  local quack_source_repo snapshot
  output_root=$(realpath -m "$1")
  python_executable=$(resolve_python)
  snapshot="$output_root/launcher/run_campaign.sh"

  if [[ $mode = new ]]; then
    acquire_campaign_lock "$output_root"
    if [[ -e $output_root ]]; then
      if [[ -n $(find "$output_root" -mindepth 1 -print -quit) ]]; then
        echo "OUTPUT_ROOT must be absent or empty for a new campaign" >&2
        return 1
      fi
    fi
    source_repo=${VARIED_ATTENTION_SOURCE_REPO-}
    if [[ -z $source_repo ]]; then
      source_repo=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
    fi
    source_repo=$(realpath -e "$source_repo")
    fa4_source_repo=${VARIED_ATTENTION_FA4_SOURCE_REPO-}
    if [[ -z $fa4_source_repo ]]; then
      echo "set VARIED_ATTENTION_FA4_SOURCE_REPO for the first run" >&2
      return 1
    fi
    fa4_source_repo=$(realpath -e "$fa4_source_repo")
    quack_source_repo=${VARIED_ATTENTION_QUACK_SOURCE_REPO-}
    if [[ -z $quack_source_repo ]]; then
      echo "set VARIED_ATTENTION_QUACK_SOURCE_REPO for the first run" >&2
      return 1
    fi
    quack_source_repo=$(realpath -e "$quack_source_repo")
    require_external_output "$output_root" "$source_repo"
    require_external_output "$output_root" "$fa4_source_repo"
    require_external_output "$output_root" "$quack_source_repo"
    export VARIED_ATTENTION_SOURCE_REPO="$source_repo"
    export VARIED_ATTENTION_FA4_SOURCE_REPO="$fa4_source_repo"
    export VARIED_ATTENTION_QUACK_SOURCE_REPO="$quack_source_repo"
    export VARIED_ATTENTION_PYTHON="$python_executable"
    snapshot_launcher "$output_root"
    export VARIED_ATTENTION_CAMPAIGN_LOCK_FD=$campaign_lock_fd
    exec "$snapshot" --resume "$output_root"
  fi

  if [[ $SCRIPT_PATH != "$snapshot" ]]; then
    if [[ ! -x $snapshot ]]; then
      echo "missing immutable launcher snapshot: $snapshot" >&2
      return 1
    fi
    unset VARIED_ATTENTION_CAMPAIGN_LOCK_FD
    exec "$snapshot" --resume "$output_root"
  fi

  if [[ -n ${VARIED_ATTENTION_CAMPAIGN_LOCK_FD-} ]]; then
    adopt_campaign_lock "$output_root"
  else
    acquire_campaign_lock "$output_root"
  fi
  trap cleanup_lanes INT TERM EXIT
  "$python_executable" "$CAMPAIGN_PY" initialize "$output_root" \
    --source-repo "${VARIED_ATTENTION_SOURCE_REPO-}" \
    --fa4-source-repo "${VARIED_ATTENTION_FA4_SOURCE_REPO-}" \
    --quack-source-repo "${VARIED_ATTENTION_QUACK_SOURCE_REPO-}"
  run_parallel_lanes "$python_executable" "$output_root"
  "$python_executable" "$CAMPAIGN_PY" validate "$output_root"
  "$python_executable" "$CAMPAIGN_PY" build "$output_root"
  trap - INT TERM EXIT
  echo "campaign complete: $output_root/published"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
