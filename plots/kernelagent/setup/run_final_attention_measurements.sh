#!/bin/bash
set -euo pipefail

STREAM=${1:?dense or causal}
IMPLS=${2:-"kernelagent-1x kernelagent-2x kernelagent-10x"}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${HELION_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}
RUN_ROOT=${KERNELAGENT_RESULTS_ROOT:-$REPO/plots/kernelagent/runs}
OUT_ROOT=${KERNELAGENT_FINAL_ROOT:-/tmp/kernelagent-final/attention_opus5_e0647170}
HARNESS=$REPO/benchmarks/cute/compare_attention_backends.py
mkdir -p "$OUT_ROOT"

run_one() {
    local variant=$1
    local seq_len=$2
    local causal=$3
    local gpu=$4
    local impl=$5
    local output="$OUT_ROOT/${variant}_${seq_len}_${impl}.json"
    local log="$OUT_ROOT/${variant}_${seq_len}_${impl}.log"
    printf '%s MEASURE %s %s %s gpu=%s\n' \
        "$(date --iso-8601=seconds)" "$variant" "$seq_len" "$impl" "$gpu"
    env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS=6,7 \
        python "$HARNESS" \
            --impl "$impl" \
            --kernelagent-results-root "$RUN_ROOT" \
            --z 2 --h 32 --seq-len "$seq_len" --head-dim 64 \
            --dtype float16 --causal "$causal" --biased 0 \
            --num-runs 9 --warmup-ms 1000 --rep-ms 500 \
            --seed 2026080106 --power-cap-w 750 \
            --skip-correctness 0 \
            --json --json-output "$output" >"$log" 2>&1
}

if [[ $STREAM == dense ]]; then
    for seq_len in 32768 65536 131072 262144; do
        for impl in $IMPLS; do
            run_one dense "$seq_len" 0 7 "$impl"
        done
    done
elif [[ $STREAM == causal ]]; then
    for seq_len in 65536 131072 262144 524288; do
        for impl in $IMPLS; do
            run_one causal "$seq_len" 1 6 "$impl"
        done
    done
else
    echo "unknown stream: $STREAM" >&2
    exit 2
fi
