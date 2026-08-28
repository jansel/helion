#!/bin/bash
set -euo pipefail

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
REPO_ROOT=$(cd "$(dirname "$SCRIPT_PATH")/../../.." && pwd)
STUDY_ROOT=/tmp/kernelagent-study
SOURCE_ROOT=$STUDY_ROOT/KernelAgent
SHIMS_ROOT=$STUDY_ROOT/shims
SANDBOX_LAUNCHER=$STUDY_ROOT/sandbox_launcher.sh
DRIVER_TEMPLATE=$REPO_ROOT/plots/kernelagent/setup/kernelagent_attention_driver.py.txt
RUN_ROOT=${KERNELAGENT_10X_ROOT:-/tmp/kernelagent-runs/attention_opus5_e0647170_10x}
RUNTIME_DRIVER=$RUN_ROOT/kernelagent_attention_driver_10x.py
EXPECTED_COMMIT=e0647170da36ef9b059ac0bd3d60103aa4ed378b
EXPECTED_TEMPLATE_SHA256=82b9d1fd0a71cd0dbb7a747bd81ac6c9e11deedc389bf707b729eee7b753e91c

export PYTHONPATH=$SHIMS_ROOT:$SOURCE_ROOT

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

run_one() {
    local variant=$1
    local seq_len=$2
    local causal=$3
    local gpu=$4
    local seconds=$5
    local run_dir=$RUN_ROOT/${variant}_${seq_len}_10x
    local supervisor_log=$RUN_ROOT/${variant}_${seq_len}_10x.supervisor.log

    if [[ -f $run_dir/selected_kernel.py && -f $run_dir/manifest.json ]]; then
        if python - "$run_dir/manifest.json" "$seq_len" "$causal" "$gpu" "$seconds" <<'PY'
import json
import math
import sys

manifest = json.load(open(sys.argv[1]))
expected = {
    "seq_len": int(sys.argv[2]),
    "causal": bool(int(sys.argv[3])),
    "physical_gpu": int(sys.argv[4]),
    "budget_label": "10x",
}
assert all(manifest.get(key) == value for key, value in expected.items())
assert math.isclose(manifest["budget_seconds"], float(sys.argv[5]))
PY
        then
            log "SKIP completed $variant $seq_len 10x gpu=$gpu"
            return 0
        fi
        log "ERROR completed artifacts do not match requested run: $run_dir"
        return 1
    fi
    if [[ -e $run_dir ]]; then
        log "ERROR partial run directory exists; move it aside before retrying: $run_dir"
        return 1
    fi

    log "START $variant $seq_len 10x time=${seconds}s gpu=$gpu"
    if python "$RUNTIME_DRIVER" \
        --run-dir "$run_dir" \
        --seq-len "$seq_len" \
        --causal "$causal" \
        --physical-gpu "$gpu" \
        --budget-seconds "$seconds" \
        --budget-label 10x >"$supervisor_log" 2>&1
    then
        log "END $variant $seq_len 10x exit=0"
        return 0
    else
        local status=$?
        log "END $variant $seq_len 10x exit=$status"
        return "$status"
    fi
}

run_stream() {
    local stream=$1
    local failures=0
    if [[ $stream == dense ]]; then
        run_one dense 32768 0 7 7086.0 || failures=$((failures + 1))
        run_one dense 65536 0 7 8409.0 || failures=$((failures + 1))
        run_one dense 131072 0 7 13867.0 || failures=$((failures + 1))
        run_one dense 262144 0 7 36827.0 || failures=$((failures + 1))
    elif [[ $stream == causal ]]; then
        run_one causal 65536 1 6 37322.0 || failures=$((failures + 1))
        run_one causal 131072 1 6 33746.0 || failures=$((failures + 1))
        run_one causal 262144 1 6 56194.0 || failures=$((failures + 1))
        run_one causal 524288 1 6 24139.0 || failures=$((failures + 1))
    else
        log "ERROR unknown stream: $stream"
        return 2
    fi
    return "$failures"
}

if [[ ${1:-} == --stream ]]; then
    run_stream "${2:?dense or causal}"
    exit $?
fi

for path in "$SOURCE_ROOT" "$SHIMS_ROOT" "$DRIVER_TEMPLATE"; do
    if [[ ! -e $path ]]; then
        log "ERROR required path is missing: $path"
        exit 1
    fi
done
if [[ ! -x $SANDBOX_LAUNCHER ]]; then
    log "ERROR sandbox launcher is missing or not executable: $SANDBOX_LAUNCHER"
    exit 1
fi
if ! command -v claude >/dev/null; then
    log "ERROR claude CLI is not available"
    exit 1
fi
actual_commit=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
if [[ $actual_commit != "$EXPECTED_COMMIT" ]]; then
    log "ERROR KernelAgent commit is $actual_commit, expected $EXPECTED_COMMIT"
    exit 1
fi
actual_template_sha256=$(sha256sum "$DRIVER_TEMPLATE" | cut -d' ' -f1)
if [[ $actual_template_sha256 != "$EXPECTED_TEMPLATE_SHA256" ]]; then
    log "ERROR driver template hash is $actual_template_sha256, expected $EXPECTED_TEMPLATE_SHA256"
    exit 1
fi

mkdir -p "$RUN_ROOT"
python - "$DRIVER_TEMPLATE" "$RUNTIME_DRIVER" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
old = 'parser.add_argument("--budget-label", choices=("1x", "2x"))'
new = 'parser.add_argument("--budget-label", choices=("1x", "2x", "10x"))'
if source.count(old) != 1:
    raise SystemExit("driver template did not contain the expected budget-label line")
Path(sys.argv[2]).write_text(source.replace(old, new))
PY

for gpu in 6 7; do
    active_pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader | xargs)
    if [[ -n $active_pids ]]; then
        log "ERROR GPU $gpu is busy with compute PID(s): $active_pids"
        exit 1
    fi
done

if [[ ${1:-} == --check ]]; then
    log "KernelAgent 10x setup validated; GPUs 6 and 7 are idle"
    exit 0
fi

original_power_6=$(nvidia-smi -i 6 --query-gpu=power.limit --format=csv,noheader,nounits | xargs)
original_power_7=$(nvidia-smi -i 7 --query-gpu=power.limit --format=csv,noheader,nounits | xargs)
dense_pid=
causal_pid=

cleanup() {
    local status=$?
    local alive attempt pid
    trap - EXIT INT TERM
    for pid in "$dense_pid" "$causal_pid"; do
        [[ -n $pid ]] && kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for attempt in {1..20}; do
        alive=0
        for pid in "$dense_pid" "$causal_pid"; do
            if [[ -n $pid ]] && kill -0 -- "-$pid" 2>/dev/null; then
                alive=1
            fi
        done
        [[ $alive = 0 ]] && break
        sleep 0.25
    done
    for pid in "$dense_pid" "$causal_pid"; do
        if [[ -n $pid ]] && kill -0 -- "-$pid" 2>/dev/null; then
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
    done
    for pid in "$dense_pid" "$causal_pid"; do
        [[ -n $pid ]] && wait "$pid" 2>/dev/null || true
    done
    sudo -n nvidia-smi -i 6 -pl "$original_power_6" >/dev/null 2>&1 || true
    sudo -n nvidia-smi -i 7 -pl "$original_power_7" >/dev/null 2>&1 || true
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sudo -n nvidia-smi -i 6,7 -pl 750
log "RUN_ROOT=$RUN_ROOT"
log "Expected critical path: about 42.1 hours plus setup overhead"

setsid bash "$SCRIPT_PATH" --stream dense >"$RUN_ROOT/queue_dense.log" 2>&1 &
dense_pid=$!
setsid bash "$SCRIPT_PATH" --stream causal >"$RUN_ROOT/queue_causal.log" 2>&1 &
causal_pid=$!

set +e
wait "$dense_pid"
dense_status=$?
dense_pid=
wait "$causal_pid"
causal_status=$?
causal_pid=
set -e

log "Dense stream exit=$dense_status; causal stream exit=$causal_status"
if (( dense_status != 0 || causal_status != 0 )); then
    exit 1
fi
log "All KernelAgent 10x tuning runs completed"
