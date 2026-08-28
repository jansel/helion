#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${HELION_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
SETUP="$REPO/plots/kernelagent_closed/setup"
if [[ -n ${KERNELAGENT_PYTHON:-} ]]; then
    PYTHON=$KERNELAGENT_PYTHON
elif [[ -n ${VIRTUAL_ENV:-} ]]; then
    PYTHON=$VIRTUAL_ENV/bin/python
elif ! PYTHON=$(command -v python3); then
    echo "KERNELAGENT_PYTHON or an active Python environment is required" >&2
    exit 2
fi
export KERNELAGENT_ENV_ROOT=${KERNELAGENT_ENV_ROOT:-$(cd -- "$(dirname -- "$PYTHON")/.." && pwd)}
ARCHIVE=${KERNELAGENT_ARCHIVE:?KERNELAGENT_ARCHIVE is required}
BUNDLE=${KERNELAGENT_BUNDLE_ROOT:?KERNELAGENT_BUNDLE_ROOT is required}
BINARY=${KERNELAGENT_BINARY:-$BUNDLE/kernelagent-bin}
REFERENCES=${KERNELAGENT_REFERENCE_ROOT:?KERNELAGENT_REFERENCE_ROOT is required}
RUN_ROOT=${KERNELAGENT_RUN_ROOT:?KERNELAGENT_RUN_ROOT is required}
mkdir -p "$RUN_ROOT"

GPU6_POWER_LIMIT=$(nvidia-smi -i 6 --query-gpu=power.limit --format=csv,noheader,nounits)
GPU7_POWER_LIMIT=$(nvidia-smi -i 7 --query-gpu=power.limit --format=csv,noheader,nounits)
DENSE_PID=
CAUSAL_PID=

restore_power_limits() {
    sudo nvidia-smi -i 6 -pl "$GPU6_POWER_LIMIT" || true
    sudo nvidia-smi -i 7 -pl "$GPU7_POWER_LIMIT" || true
}

cleanup() {
    local status=$?
    local alive attempt pid
    trap - EXIT INT TERM
    for pid in "$DENSE_PID" "$CAUSAL_PID"; do
        [[ -n $pid ]] && kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for attempt in {1..20}; do
        alive=0
        for pid in "$DENSE_PID" "$CAUSAL_PID"; do
            if [[ -n $pid ]] && kill -0 -- "-$pid" 2>/dev/null; then
                alive=1
            fi
        done
        [[ $alive = 0 ]] && break
        sleep 0.25
    done
    for pid in "$DENSE_PID" "$CAUSAL_PID"; do
        if [[ -n $pid ]] && kill -0 -- "-$pid" 2>/dev/null; then
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
    done
    for pid in "$DENSE_PID" "$CAUSAL_PID"; do
        [[ -n $pid ]] && wait "$pid" 2>/dev/null || true
    done
    restore_power_limits
    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sudo nvidia-smi -i 6 -pl 750
sudo nvidia-smi -i 7 -pl 750

run_one() {
    local variant=$1
    local causal=$2
    local seq_len=$3
    local gpu=$4
    local base_budget=$5
    local multiplier=$6
    local budget_label="${multiplier}x"
    local budget
    budget=$(awk -v base="$base_budget" -v scale="$multiplier" \
        'BEGIN { printf "%.1f", base * scale }')
    local run_dir="$RUN_ROOT/${variant}_${seq_len}_${budget_label}"
    local log="$RUN_ROOT/${variant}_${seq_len}_${budget_label}.log"
    if ! "$PYTHON" "$SETUP/run_campaign.py.txt" \
        --run-dir "$run_dir" \
        --archive "$ARCHIVE" \
        --binary "$BINARY" \
        --reference-root "$REFERENCES" \
        --seq-len "$seq_len" \
        --causal "$causal" \
        --physical-gpu "$gpu" \
        --budget-seconds "$budget" \
        --budget-label "$budget_label" \
        >"$log" 2>&1; then
        if [[ -f "$run_dir/selected_kernel.py.txt" ]] ||
            jq -e '.best_candidate_id != null' \
                "$run_dir/campaign_state.json" >/dev/null 2>&1; then
            "$PYTHON" "$SETUP/run_campaign.py.txt" \
                --run-dir "$run_dir" \
                --archive "$ARCHIVE" \
                --binary "$BINARY" \
                --reference-root "$REFERENCES" \
                --seq-len "$seq_len" \
                --causal "$causal" \
                --physical-gpu "$gpu" \
                --budget-seconds "$budget" \
                --budget-label "$budget_label" \
                --resume-selected >>"$log" 2>&1
        else
            "$PYTHON" "$SETUP/finalize_failed_run.py.txt" \
                --run-dir "$run_dir" >>"$log" 2>&1
        fi
    fi
}

retry_failed_dense_1x() {
    run_one dense 0 32768 7 708.6 1 &&
        run_one dense 0 65536 7 840.9 1 &&
        run_one dense 0 131072 7 1386.7 1 &&
        run_one dense 0 262144 7 3682.7 1
}

retry_failed_causal_1x() {
    run_one causal 1 524288 6 2413.9 1
}

run_dense_2x() {
    run_one dense 0 32768 7 708.6 2 &&
        run_one dense 0 65536 7 840.9 2 &&
        run_one dense 0 131072 7 1386.7 2 &&
        run_one dense 0 262144 7 3682.7 2
}

run_causal_2x() {
    run_one causal 1 65536 6 3732.2 2 &&
        run_one causal 1 131072 6 3374.6 2 &&
        run_one causal 1 262144 6 5619.4 2 &&
        run_one causal 1 524288 6 2413.9 2
}

export PYTHON SETUP ARCHIVE BINARY REFERENCES RUN_ROOT
export -f run_one retry_failed_dense_1x retry_failed_causal_1x
export -f run_dense_2x run_causal_2x

STATUS=0
if [[ ${KERNELAGENT_SKIP_1X:-0} != 1 ]]; then
    setsid bash -euo pipefail -c 'retry_failed_dense_1x' &
    DENSE_PID=$!
    setsid bash -euo pipefail -c 'retry_failed_causal_1x' &
    CAUSAL_PID=$!
    wait "$DENSE_PID" || STATUS=1
    wait "$CAUSAL_PID" || STATUS=1
    DENSE_PID=
    CAUSAL_PID=
    if [[ $STATUS -ne 0 ]]; then
        exit "$STATUS"
    fi

    if ! "$PYTHON" "$SETUP/validate_selected.py.txt" \
        --run-root "$RUN_ROOT" --budget-label 1x; then
        exit 1
    fi
fi

setsid bash -euo pipefail -c 'run_dense_2x' &
DENSE_PID=$!
setsid bash -euo pipefail -c 'run_causal_2x' &
CAUSAL_PID=$!
wait "$DENSE_PID" || STATUS=1
wait "$CAUSAL_PID" || STATUS=1
DENSE_PID=
CAUSAL_PID=
if [[ $STATUS -ne 0 ]]; then
    exit "$STATUS"
fi

"$PYTHON" "$SETUP/validate_selected.py.txt" \
    --run-root "$RUN_ROOT" --budget-label 2x
