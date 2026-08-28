#!/usr/bin/env bash
set -euo pipefail

HELDOUT_SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
RUNNER=$(dirname "$HELDOUT_SCRIPT_PATH")/run_strict_all8.sh

# Reuse the all-eight launcher's environment, checkout, GPU, and lane checks.
source "$RUNNER"

EXPECTED_COMMIT=c3e36b65d69681c23e053042b0bc21e2331bad17

run_heldout_lane() {
  local repo=$1
  local output_root=$2
  local input_seed=$3
  local gpu=$4
  local causal=$5
  local python_executable=$6
  shift 6
  local seq_len tuner_seed variant

  variant=$([[ $causal = 1 ]] && echo causal || echo dense)
  for spec in "$@"; do
    read -r seq_len tuner_seed <<<"$spec"
    "$BASH" "$RUNNER" __run_lane \
      "$repo" "$output_root/$variant/seed_$tuner_seed" \
      "$input_seed" "$gpu" "$causal" "$python_executable" "$spec"
  done
}

start_heldout_lane() {
  local result_variable=$1
  local launcher=$2
  shift 2
  local pid

  setsid "$BASH" "$launcher" __run_heldout_lane "$@" &
  pid=$!
  active_lane_pids+=("$pid")
  printf -v "$result_variable" '%s' "$pid"
}

write_heldout_campaign_manifest() {
  local output_root=$1
  local python_executable=$2
  local all8_launcher_sha256=$3
  local heldout_launcher_sha256=$4

  printf '%s\n' \
    'schema_version,expected_commit,python_executable,all8_launcher_sha256,heldout_launcher_sha256,variant,seq_len,physical_gpu,input_seed,tuner_seed,result_path' \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,dense,81920,7,$INPUT_SEED,2026082301,dense/seed_2026082301/dense_s81920/result.json" \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,dense,81920,7,$INPUT_SEED,2026082302,dense/seed_2026082302/dense_s81920/result.json" \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,dense,81920,7,$INPUT_SEED,2026082303,dense/seed_2026082303/dense_s81920/result.json" \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,dense,81920,7,$INPUT_SEED,2026082304,dense/seed_2026082304/dense_s81920/result.json" \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,dense,81920,7,$INPUT_SEED,2026082305,dense/seed_2026082305/dense_s81920/result.json" \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,causal,196608,6,$INPUT_SEED,2026082311,causal/seed_2026082311/causal_s196608/result.json" \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,causal,196608,6,$INPUT_SEED,2026082312,causal/seed_2026082312/causal_s196608/result.json" \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,causal,196608,6,$INPUT_SEED,2026082313,causal/seed_2026082313/causal_s196608/result.json" \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,causal,196608,6,$INPUT_SEED,2026082314,causal/seed_2026082314/causal_s196608/result.json" \
    "3,$EXPECTED_COMMIT,$python_executable,$all8_launcher_sha256,$heldout_launcher_sha256,causal,196608,6,$INPUT_SEED,2026082315,causal/seed_2026082315/causal_s196608/result.json" \
    >"$output_root/campaign.csv"
}

heldout_main() {
  local repo_override=${HELION_REPO_ROOT-}
  local output_root=${1:?usage: run_strict_heldout.sh OUTPUT_ROOT}
  local repo dense_pid causal_pid power_limit gpu python_executable
  local launcher_dir heldout_launcher all8_launcher
  local heldout_launcher_sha256 all8_launcher_sha256

  sanitize_search_environment
  python_executable=$(resolve_python_executable)
  validate_python_executable "$python_executable"
  repo=$(realpath "${repo_override:-$(git rev-parse --show-toplevel)}")
  output_root=$(realpath -m "$output_root")

  test "$(git -C "$repo" rev-parse HEAD)" = "$EXPECTED_COMMIT"
  validate_checkout_clean "$repo"
  validate_output_root_outside_checkout "$repo" "$output_root"
  test ! -e "$output_root"
  mkdir -p "$output_root"
  launcher_dir="$output_root/launcher"
  heldout_launcher="$launcher_dir/run_strict_heldout.sh"
  all8_launcher="$launcher_dir/run_strict_all8.sh"
  snapshot_launcher "$HELDOUT_SCRIPT_PATH" "$heldout_launcher"
  snapshot_launcher "$RUNNER" "$all8_launcher"
  heldout_launcher_sha256=$(sha256sum "$heldout_launcher" | cut -d ' ' -f 1)
  all8_launcher_sha256=$(sha256sum "$all8_launcher" | cut -d ' ' -f 1)
  write_heldout_campaign_manifest \
    "$output_root" "$python_executable" \
    "$all8_launcher_sha256" "$heldout_launcher_sha256"

  for gpu in 6 7; do
    test "$(nvidia-smi -i "$gpu" --query-gpu=name --format=csv,noheader)" = \
      "NVIDIA B200"
    power_limit=$(nvidia-smi -i "$gpu" --query-gpu=power.limit \
      --format=csv,noheader,nounits)
    awk -v value="$power_limit" \
      'BEGIN { exit !(value >= 749.5 && value <= 750.5) }'
    validate_gpu_idle "$gpu"
  done

  cd "$repo"
  trap cleanup_lanes EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  start_heldout_lane dense_pid "$heldout_launcher" \
    "$repo" "$output_root" "$INPUT_SEED" 7 0 "$python_executable" \
    "81920 2026082301" \
    "81920 2026082302" \
    "81920 2026082303" \
    "81920 2026082304" \
    "81920 2026082305"
  start_heldout_lane causal_pid "$heldout_launcher" \
    "$repo" "$output_root" "$INPUT_SEED" 6 1 "$python_executable" \
    "196608 2026082311" \
    "196608 2026082312" \
    "196608 2026082313" \
    "196608 2026082314" \
    "196608 2026082315"
  wait_for_lanes "$dense_pid" "$causal_pid"
  trap - INT TERM EXIT
}

if [[ ${BASH_SOURCE[0]} = "$0" ]]; then
  if [[ ${1-} = __run_heldout_lane ]]; then
    shift
    sanitize_search_environment
    run_heldout_lane "$@"
  else
    heldout_main "$@"
  fi
fi
