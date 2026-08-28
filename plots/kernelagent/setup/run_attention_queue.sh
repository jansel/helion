#!/bin/bash
set -u

STREAM=${1:?dense or causal}
ROOT=/tmp/kernelagent-runs/attention_opus5_e0647170
DRIVER=/tmp/kernelagent-study/kernelagent_attention_driver.py
export PYTHONPATH=/tmp/kernelagent-study/shims:/tmp/kernelagent-study/KernelAgent
mkdir -p "$ROOT"

run_one() {
    local variant=$1
    local seq_len=$2
    local causal=$3
    local gpu=$4
    local seconds=$5
    local label=$6
    local run_dir="$ROOT/${variant}_${seq_len}_${label}"
    local log="$ROOT/${variant}_${seq_len}_${label}.supervisor.log"
    printf '%s START %s %s %s budget=%ss gpu=%s\n' \
        "$(date --iso-8601=seconds)" "$variant" "$seq_len" "$label" "$seconds" "$gpu" | tee -a "$ROOT/queue_${STREAM}.log"
    python "$DRIVER" \
        --run-dir "$run_dir" \
        --seq-len "$seq_len" \
        --causal "$causal" \
        --physical-gpu "$gpu" \
        --budget-seconds "$seconds" \
        --budget-label "$label" >"$log" 2>&1
    local status=$?
    printf '%s END %s %s %s exit=%s\n' \
        "$(date --iso-8601=seconds)" "$variant" "$seq_len" "$label" "$status" | tee -a "$ROOT/queue_${STREAM}.log"
}

if [[ $STREAM == dense ]]; then
    for spec in \
        'dense 32768 0 7 708.6' \
        'dense 65536 0 7 840.9' \
        'dense 131072 0 7 1386.7' \
        'dense 262144 0 7 3682.7'; do
        run_one $spec 1x
    done
    for spec in \
        'dense 32768 0 7 1417.2' \
        'dense 65536 0 7 1681.8' \
        'dense 131072 0 7 2773.4' \
        'dense 262144 0 7 7365.4'; do
        run_one $spec 2x
    done
elif [[ $STREAM == causal ]]; then
    for spec in \
        'causal 65536 1 6 3732.2' \
        'causal 131072 1 6 3374.6' \
        'causal 262144 1 6 5619.4' \
        'causal 524288 1 6 2413.9'; do
        run_one $spec 1x
    done
    for spec in \
        'causal 65536 1 6 7464.4' \
        'causal 131072 1 6 6749.2' \
        'causal 262144 1 6 11238.8' \
        'causal 524288 1 6 4827.8'; do
        run_one $spec 2x
    done
else
    echo "unknown stream: $STREAM" >&2
    exit 2
fi
