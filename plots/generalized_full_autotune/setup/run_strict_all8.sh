#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT=c3e36b65d69681c23e053042b0bc21e2331bad17
EXPECTED_GIT_DESCRIBE_PREFIX=v1.4.0-157-g
EXPECTED_CUTE_VERSION=4.7.0
INPUT_SEED=2026081500
SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
active_lane_pids=()
campaign_lock_fd=

sanitize_search_environment() {
  local variable
  while IFS= read -r variable; do
    case "$variable" in
      HELION_* | CUTE_DSL_* | CUDA_MPS_* | CUDNN_* | TORCH_CUDNN_* | TRITON_* | \
        TORCHINDUCTOR_* | PYTORCH_TUNABLEOP_* | CUDA_CACHE_* | \
      CUDA_DEVICE_ORDER | PYTHONPYCACHEPREFIX | XDG_CACHE_HOME)
        unset "$variable"
        ;;
      CUBLAS_FORCE_TF32 | CUBLAS_WORKSPACE_CONFIG | CUDA_AUTO_BOOST | \
        CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT | \
        CUDA_DEVICE_MAX_CONNECTIONS | CUDA_DISABLE_PTX_JIT | \
        CUDA_FORCE_PTX_JIT | CUDA_LAUNCH_BLOCKING | \
        CUDA_MANAGED_FORCE_DEVICE_ALLOC | CUDA_MODULE_LOADING | \
        CUDA_VISIBLE_DEVICES | NVIDIA_TF32_OVERRIDE | \
        PYTHONPATH | PYTORCH_ALLOC_CONF | PYTORCH_CUDA_ALLOC_CONF | \
        TORCH_ALLOW_TF32_CUBLAS_OVERRIDE)
        unset "$variable"
        ;;
    esac
  done < <(compgen -e)
}

resolve_python_executable() {
  local requested=${STRICT_PYTHON_EXECUTABLE-}
  local candidate canonical

  if [[ -n $requested ]]; then
    candidate=$requested
  else
    candidate=$(command -v python) || {
      echo "python is not available on PATH; set STRICT_PYTHON_EXECUTABLE" >&2
      return 1
    }
  fi
  if [[ $candidate != */* ]]; then
    candidate=$(command -v "$candidate") || {
      printf 'Python command is not available on PATH: %s\n' "$candidate" >&2
      return 1
    }
  fi
  canonical=$(realpath -e "$candidate") || return 1
  if [[ ! -f $canonical || ! -x $canonical || -L $canonical ]]; then
    printf 'Python executable is not a canonical executable file: %s\n' \
      "$canonical" >&2
    return 1
  fi
  printf '%s\n' "$canonical"
}

validate_python_executable() {
  local python_executable=$1
  local canonical cute_version

  canonical=$(realpath -e "$python_executable") || return 1
  if [[ $python_executable != "$canonical" || ! -f $canonical || \
    ! -x $canonical || -L $canonical ]]; then
    printf 'Python executable changed or is not canonical: %s\n' \
      "$python_executable" >&2
    return 1
  fi
  cute_version=$("$python_executable" -c \
    'from importlib.metadata import version; print(version("nvidia-cutlass-dsl"))') || \
    return 1
  if [[ $cute_version != "$EXPECTED_CUTE_VERSION" ]]; then
    printf 'CuTe version changed or is unsupported: expected %s, got %s\n' \
      "$EXPECTED_CUTE_VERSION" "$cute_version" >&2
    return 1
  fi
}

validate_gpu_idle() {
  local gpu=$1
  local active_pids

  active_pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
    --format=csv,noheader,nounits)
  if grep -Eq '[0-9]' <<<"$active_pids"; then
    printf 'physical GPU %s is not idle; active compute PIDs: %s\n' \
      "$gpu" "$(tr '\n' ' ' <<<"$active_pids")" >&2
    return 1
  fi
}

validate_checkout_clean() {
  local repo=$1
  local listing path invalid
  local -a unexpected=()

  git -C "$repo" diff --quiet
  git -C "$repo" diff --cached --quiet
  listing=$(mktemp)
  if ! git -C "$repo" ls-files --others --exclude-standard -z >"$listing"; then
    rm -f "$listing"
    return 1
  fi
  invalid=0
  while IFS= read -r -d '' path; do
    case "$path" in
      .validation/generalized_paired/combine_results.py | \
        .validation/generalized_paired/paired_worker.py | \
        .validation/generalized_paired/run_all8.py | \
        .validation/generalized_paired/test_static.py)
        if [[ ! -f $repo/$path || -L $repo/$path ]]; then
          printf 'allowed staged harness path is not a regular file: %s\n' \
            "$path" >&2
          invalid=1
        fi
        ;;
      *) unexpected+=("$path") ;;
    esac
  done <"$listing"
  rm -f "$listing"
  if ((invalid)); then
    return 1
  fi
  if ((${#unexpected[@]})); then
    printf 'checkout has unexpected untracked paths:\n' >&2
    printf '  %q\n' "${unexpected[@]}" >&2
    return 1
  fi

  if ! git -C "$repo" ls-files --others --ignored --exclude-standard -z \
    >"$listing"; then
    rm -f "$listing"
    return 1
  fi
  unexpected=()
  while IFS= read -r -d '' path; do
    unexpected+=("$path")
  done <"$listing"
  rm -f "$listing"
  if ((${#unexpected[@]})); then
    printf 'checkout has ignored paths; caches must be external:\n' >&2
    printf '  %q\n' "${unexpected[@]}" >&2
    return 1
  fi
}

validate_checkout_identity() {
  local repo=$1
  local abbreviation describe head

  head=$(git -C "$repo" rev-parse HEAD) || return 1
  if [[ $head != "$EXPECTED_COMMIT" ]]; then
    printf 'measured repository HEAD changed: expected %s, got %s\n' \
      "$EXPECTED_COMMIT" "$head" >&2
    return 1
  fi
  describe=$(git -C "$repo" describe --tags --always --dirty) || return 1
  if [[ $describe != "$EXPECTED_GIT_DESCRIBE_PREFIX"* ]]; then
    printf 'measured repository version changed: expected %s<commit>, got %s\n' \
      "$EXPECTED_GIT_DESCRIBE_PREFIX" "$describe" >&2
    return 1
  fi
  abbreviation=${describe#"$EXPECTED_GIT_DESCRIBE_PREFIX"}
  if [[ ! $abbreviation =~ ^[0-9a-f]{7,40}$ || \
    $EXPECTED_COMMIT != "$abbreviation"* ]]; then
    printf 'measured repository version changed: expected %s<commit>, got %s\n' \
      "$EXPECTED_GIT_DESCRIBE_PREFIX" "$describe" >&2
    return 1
  fi
  validate_checkout_clean "$repo"
}

validate_output_root_outside_checkout() {
  local repo=$1
  local output_root=$2
  case "$output_root" in
    "$repo" | "$repo"/*)
      printf 'OUTPUT_ROOT must be outside the measured checkout: %s\n' \
        "$output_root" >&2
      return 1
      ;;
  esac
}

snapshot_launcher() {
  local source=$1
  local destination=$2

  test -f "$source"
  test ! -L "$source"
  test ! -e "$destination"
  mkdir -p "$(dirname "$destination")"
  cp -- "$source" "$destination"
  chmod 0555 "$destination"
  sha256sum "$destination" | cut -d ' ' -f 1 >"$destination.sha256"
}

validate_launcher_snapshot() {
  local launcher=$1
  local expected actual mode

  test -f "$launcher"
  test ! -L "$launcher"
  test -f "$launcher.sha256"
  test ! -L "$launcher.sha256"
  expected=$(<"$launcher.sha256")
  actual=$(sha256sum "$launcher" | cut -d ' ' -f 1)
  mode=$(stat -c %a "$launcher")
  if [[ $actual != "$expected" || $mode != 555 ]]; then
    printf 'launcher snapshot identity changed: %s\n' "$launcher" >&2
    return 1
  fi
}

acquire_campaign_lock() {
  local output_root=$1
  local lock_path="${output_root}.campaign.lock"

  mkdir -p "$(dirname "$lock_path")"
  exec {campaign_lock_fd}>"$lock_path"
  if ! flock -n "$campaign_lock_fd"; then
    printf 'another strict campaign process holds %s\n' "$lock_path" >&2
    return 1
  fi
}

shape_completion_contents() {
  local output=$1
  local input_seed=$2
  local gpu=$3
  local causal=$4
  local seq_len=$5
  local tuner_seed=$6
  local python_executable=$7
  local result_sha256

  result_sha256=$(sha256sum "$output/result.json" | cut -d ' ' -f 1)
  printf '%s\n' \
    'schema_version=1' \
    "commit=$EXPECTED_COMMIT" \
    "cute_version=$EXPECTED_CUTE_VERSION" \
    "python_executable=$python_executable" \
    "input_seed=$input_seed" \
    "tuner_seed=$tuner_seed" \
    "gpu=$gpu" \
    "causal=$causal" \
    "seq_len=$seq_len" \
    "result_sha256=$result_sha256"
}

write_shape_completion_marker() {
  local output=$1
  shift
  local marker="$output/.launcher-complete"
  local temporary="$output/.launcher-complete.tmp.$$"

  test -f "$output/result.json"
  test ! -L "$output/result.json"
  shape_completion_contents "$output" "$@" >"$temporary"
  chmod 0444 "$temporary"
  mv -- "$temporary" "$marker"
}

validate_shape_completion_marker() {
  local output=$1
  shift
  local marker="$output/.launcher-complete"

  if [[ ! -f $marker || -L $marker || ! -f $output/result.json || \
    -L $output/result.json ]]; then
    printf 'completed shape evidence is missing or unsafe: %s\n' "$output" >&2
    return 1
  fi
  if ! cmp -s "$marker" <(shape_completion_contents "$output" "$@"); then
    printf 'completed shape evidence changed: %s\n' "$output" >&2
    return 1
  fi
}

quarantine_incomplete_shape() {
  local output=$1
  local output_root=$2
  local quarantine_root="${output_root}.quarantine"
  local destination

  mkdir -p "$quarantine_root"
  destination="$quarantine_root/$(basename "$output").$(date +%Y%m%dT%H%M%S).$$"
  test ! -e "$destination"
  mv -- "$output" "$destination"
  printf 'quarantined incomplete shape at %s\n' "$destination"
}

run_lane() {
  local repo=$1
  local output_root=$2
  local input_seed=$3
  local gpu=$4
  local causal=$5
  local python_executable=$6
  shift 6
  local python_version seq_len tuner_seed output
  local -a common=(
    "$repo/benchmarks/cute/compare_attention_backends.py"
    --impl helion-cute
    --z 2
    --h 32
    --head-dim 64
    --dtype float16
    --num-runs 9
    --warmup-ms 1000
    --rep-ms 500
    --seed "$input_seed"
    --power-cap-w 750
    --helion-force-flash-config 0
    --helion-force-autotune 1
    --helion-require-full-autotune 1
    --helion-return-lse 0
    --helion-cute-benchmark-timer wall
    --helion-autotune-effort full
    --helion-autotune-best-of-k 1
    --helion-autotune-benchmark-timeout 60
    --helion-autotune-accuracy-check 1
    --helion-autotuner-initial-population from_random
    --json
  )

  for spec in "$@"; do
    read -r seq_len tuner_seed <<<"$spec"
    output="$output_root/$([[ $causal = 1 ]] && echo causal || echo dense)_s$seq_len"
    # The preceding foreground pipeline is fully reaped before this loop advances.
    validate_python_executable "$python_executable"
    validate_gpu_idle "$gpu"
    validate_checkout_identity "$repo"
    if [[ -e $output ]]; then
      if [[ -e $output/.launcher-complete ]]; then
        validate_shape_completion_marker \
          "$output" "$input_seed" "$gpu" "$causal" "$seq_len" \
          "$tuner_seed" "$python_executable"
        printf 'SKIP completed gpu=%s causal=%s S=%s seed=%s\n' \
          "$gpu" "$causal" "$seq_len" "$tuner_seed"
        continue
      fi
      quarantine_incomplete_shape "$output" "$output_root"
    fi
    mkdir -p \
      "$output/autotune" \
      "$output/cache/cuda" \
      "$output/cache/cute_dsl" \
      "$output/cache/helion" \
      "$output/cache/torchinductor" \
      "$output/cache/triton" \
      "$output/cache/xdg"
    python_version=$("$python_executable" -c \
      'import sys; print(sys.version.replace("\n", " "))')
    echo "START gpu=$gpu causal=$causal S=$seq_len seed=$tuner_seed $(date --iso-8601=seconds)"
    printf 'PYTHON executable=%s version=%s\n' \
      "$python_executable" "$python_version" | tee "$output/run.log"
    PYTHONUNBUFFERED=1 \
      PYTHONHASHSEED=0 \
      PYTHONPYCACHEPREFIX="$output/cache/pycache" \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="$gpu" \
      CUDA_CACHE_DISABLE=0 \
      CUDA_CACHE_PATH="$output/cache/cuda" \
      CUTE_DSL_CACHE_DIR="$output/cache/cute_dsl" \
      TORCHINDUCTOR_CACHE_DIR="$output/cache/torchinductor" \
      TRITON_CACHE_DIR="$output/cache/triton" \
      XDG_CACHE_HOME="$output/cache/xdg" \
      HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS=6,7 \
      HELION_CACHE_DIR="$output/cache/helion" \
      HELION_AUTOTUNE_LOG="$output/autotune" \
      HELION_AUTOTUNE_LOG_DETAILS=1 \
      "$python_executable" "${common[@]}" \
        --seq-len "$seq_len" \
        --causal "$causal" \
        --helion-env "HELION_AUTOTUNE_RANDOM_SEED=$tuner_seed" \
        --json-output "$output/result.json" \
        2>&1 | tee -a "$output/run.log"
    validate_checkout_identity "$repo"
    validate_gpu_idle "$gpu"
    write_shape_completion_marker \
      "$output" "$input_seed" "$gpu" "$causal" "$seq_len" \
      "$tuner_seed" "$python_executable"
    echo "DONE gpu=$gpu causal=$causal S=$seq_len $(date --iso-8601=seconds)"
  done
}

terminate_active_lanes() {
  local attempt pid running
  for pid in "${active_lane_pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  for ((attempt = 0; attempt < 50; attempt++)); do
    running=0
    for pid in "${active_lane_pids[@]}"; do
      if kill -0 -- "-$pid" 2>/dev/null; then
        running=1
      fi
    done
    if ((running == 0)); then
      break
    fi
    sleep 0.1
  done
  for pid in "${active_lane_pids[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
  for pid in "${active_lane_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  active_lane_pids=()
}

remove_active_lane() {
  local completed=$1
  local pid
  local -a remaining=()
  for pid in "${active_lane_pids[@]}"; do
    if [[ $pid != "$completed" ]]; then
      remaining+=("$pid")
    fi
  done
  active_lane_pids=("${remaining[@]}")
}

wait_for_lanes() {
  local dense_pid=$1
  local causal_pid=$2
  local completed_pid=""
  local remaining_pid status

  if wait -n -p completed_pid "$dense_pid" "$causal_pid"; then
    status=0
  else
    status=$?
  fi
  if ((status != 0)); then
    echo "benchmark lane $completed_pid failed with status $status" >&2
    terminate_active_lanes
    return "$status"
  fi
  remove_active_lane "$completed_pid"

  if [[ $completed_pid = "$dense_pid" ]]; then
    remaining_pid=$causal_pid
  else
    remaining_pid=$dense_pid
  fi
  if wait "$remaining_pid"; then
    status=0
  else
    status=$?
  fi
  if ((status != 0)); then
    echo "benchmark lane $remaining_pid failed with status $status" >&2
    return "$status"
  fi
  remove_active_lane "$remaining_pid"
  return "$status"
}

cleanup_lanes() {
  local status=$?
  trap - EXIT
  terminate_active_lanes
  exit "$status"
}

start_lane() {
  local result_variable=$1
  local launcher=$2
  shift 2
  local pid
  setsid "$BASH" "$launcher" __run_lane "$@" &
  pid=$!
  active_lane_pids+=("$pid")
  printf -v "$result_variable" '%s' "$pid"
}

main() {
  local mode=new
  if [[ ${1-} = --resume ]]; then
    mode=resume
    shift
  fi
  local repo_override=${HELION_REPO_ROOT-}
  local output_root=${1:?usage: run_strict_all8.sh [--resume] OUTPUT_ROOT}
  local repo dense_pid causal_pid power_limit gpu launcher python_executable

  sanitize_search_environment
  python_executable=$(resolve_python_executable)
  validate_python_executable "$python_executable"
  repo=$(realpath "${repo_override:-$(git rev-parse --show-toplevel)}")
  output_root=$(realpath -m "$output_root")

  validate_checkout_identity "$repo"
  validate_output_root_outside_checkout "$repo" "$output_root"
  launcher="$output_root/launcher/run_strict_all8.sh"
  if [[ $mode = resume && $SCRIPT_PATH != "$launcher" ]]; then
    validate_launcher_snapshot "$launcher"
    export HELION_REPO_ROOT="$repo"
    export STRICT_PYTHON_EXECUTABLE="$python_executable"
    exec "$launcher" --resume "$output_root"
  fi
  acquire_campaign_lock "$output_root"
  if [[ $mode = new ]]; then
    test ! -e "$output_root"
    mkdir -p "$output_root"
    snapshot_launcher "$SCRIPT_PATH" "$launcher"
  else
    test -d "$output_root"
    validate_launcher_snapshot "$launcher"
  fi

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
  start_lane dense_pid "$launcher" \
    "$repo" "$output_root" "$INPUT_SEED" 7 0 "$python_executable" \
    "32768 2026081501" \
    "65536 2026081502" \
    "131072 2026081503" \
    "262144 2026081504"
  start_lane causal_pid "$launcher" \
    "$repo" "$output_root" "$INPUT_SEED" 6 1 "$python_executable" \
    "65536 2026081511" \
    "131072 2026081512" \
    "262144 2026081513" \
    "524288 2026081514"
  wait_for_lanes "$dense_pid" "$causal_pid"
  trap - INT TERM EXIT
}

if [[ ${BASH_SOURCE[0]} = "$0" ]]; then
  if [[ ${1-} = __run_lane ]]; then
    shift
    sanitize_search_environment
    run_lane "$@"
  else
    main "$@"
  fi
fi
