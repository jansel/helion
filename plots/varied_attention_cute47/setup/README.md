# Varied BF16 attention campaign with CuTe 4.7

This directory defines the reproducible campaign used to refresh the eight-shape
BF16 attention plot with a single CuTe 4.7 toolchain. It measures Helion-CuTe,
PyTorch SDPA forced to cuDNN, FlashAttention-4, and FlexAttention forced to its
FLASH/CuTe backend. Dense work is pinned to physical GPU 7 and causal work to
physical GPU 6. Both GPUs must be NVIDIA B200s held at a 750 W power limit.

`EXPECTED_MEASURED_COMMIT` in `campaign.py` pins the measured Helion source to
`c3e36b65d69681c23e053042b0bc21e2331bad17`. FA4 is pinned to commit
`2409214a03797b168f648ea30df1adbc09ce658a`, described by
`fa4-v4.0.0.beta23`. Quack is source-pinned to commit
`b5b49dae477d39cb8ea8cca2820ef09ba548c72c`.

## Workloads

| Order | Group | Variant | B | H | S | D | Epilogue | GPU |
|---:|---|---|---:|---:|---:|---:|---|---:|
| 1 | head dim 128 | dense | 2 | 32 | 262144 | 128 | identity | 7 |
| 2 | head dim 128 | causal | 2 | 32 | 524288 | 128 | identity | 6 |
| 3 | batch 1 | dense | 1 | 32 | 524288 | 64 | identity | 7 |
| 4 | batch 1 | causal | 1 | 32 | 1048576 | 64 | identity | 6 |
| 5 | batch 8 | dense | 8 | 32 | 524288 | 64 | identity | 7 |
| 6 | batch 8 | causal | 8 | 32 | 786432 | 64 | identity | 6 |
| 7 | ReLU epilogue | dense | 2 | 32 | 524288 | 64 | ReLU | 7 |
| 8 | ReLU epilogue | causal | 2 | 32 | 1048576 | 64 | ReLU | 6 |

Every implementation gets nine steady-state samples after the benchmark's cache
and thermal warmups. Correctness is enabled. ReLU is part of the timed callable,
but its FLOPs are excluded from the attention throughput numerator. Helion uses
the backend wall timer required for strict CuTe measurements; the three baselines
use the harness CUDA-event timer.

## Prerequisites

Use the pinned Python environment containing:

- PyTorch `2.13.0.dev20260506+cu130`
- Triton `3.7.0+git88b227e2`
- `nvidia-cutlass-dsl` `4.7.0`
- `nvidia-cudnn-cu13` `9.20.0.48` with cuDNN runtime `9.20.0`
- `apache-tvm-ffi` `0.1.11`

Existing local checkouts containing the pinned FA4 and Quack commits are
required. The campaign does not install packages, fetch from the network, or
mutate any source repository. It creates clean detached worktrees below the
external output root. Every benchmark process starts with the measured Helion
worktree on `PYTHONPATH`; FA4 and FlexAttention additionally receive the pinned
Quack worktree. The Helion process removes `PYTHONPATH` immediately after Python
has initialized its import path because strict mode rejects an ambient override.
User-site packages are disabled. Preflights using both exact path classes require
resolved module files to live under the detached worktrees and record their
relative names and hashes. The campaign also records resolved import origins and
file hashes for Torch, Triton, CuTe, TVM FFI, and Torch's cuDNN binding, plus the
installed cuDNN shared-library hash. Inherited `LD_LIBRARY_PATH`, `LIBRARY_PATH`,
`PYTHONPATH`, and `PYTHONHOME` are not passed to benchmark children.

## Run and resume

Choose an output directory outside the Helion repository and point the launcher
at the existing FA4 repository:

```bash
OUTPUT_ROOT=$(realpath -m ../varied-attention-cute47-run)
export VARIED_ATTENTION_FA4_SOURCE_REPO=$HOME/src/flash-attention
export VARIED_ATTENTION_QUACK_SOURCE_REPO=$HOME/src/quack
export VARIED_ATTENTION_PYTHON=$(command -v python)
plots/varied_attention_cute47/setup/run_campaign.sh "$OUTPUT_ROOT"
```

The first command snapshots the launcher, creates the three pinned worktrees, and
then executes only the snapshot. Dense and causal lanes run concurrently in
separate process groups. There is no campaign wall-clock timeout. The 180-second
Helion setting is a per-candidate autotuner benchmark timeout, not a campaign
budget. A nonblocking lock at `"$OUTPUT_ROOT.campaign.lock"` protects the whole
initialize/run/validate/build sequence; a second launcher for the same output
root fails immediately.

For the multi-day run, launch it as a detached user service so it is independent
of the current terminal or Codex session:

```bash
systemd-run --user --collect \
  --unit=helion-varied-cute47 \
  --working-directory="$PWD" \
  --setenv=VARIED_ATTENTION_FA4_SOURCE_REPO="$VARIED_ATTENTION_FA4_SOURCE_REPO" \
  --setenv=VARIED_ATTENTION_QUACK_SOURCE_REPO="$VARIED_ATTENTION_QUACK_SOURCE_REPO" \
  --setenv=VARIED_ATTENTION_PYTHON="$VARIED_ATTENTION_PYTHON" \
  "$(realpath plots/varied_attention_cute47/setup/run_campaign.sh)" \
  "$OUTPUT_ROOT"
```

Inspect the detached job with:

```bash
systemctl --user status helion-varied-cute47.service
journalctl --user -u helion-varied-cute47.service -f
tail -f "$OUTPUT_ROOT/dense-lane.log" "$OUTPUT_ROOT/causal-lane.log"
```

After interruption or host restart, resume through either the source launcher or
the immutable snapshot:

```bash
plots/varied_attention_cute47/setup/run_campaign.sh --resume "$OUTPUT_ROOT"
```

The resume command can also be placed in a new `systemd-run` unit by changing the
unit name and adding `--resume` before `"$OUTPUT_ROOT"`.

A result is skipped only if its result, invocation, strict sidecar when
applicable, receipt hashes, input specification, versions, source identity, GPU,
power limit, correctness, and all nine finite timings validate. Any partial or
invalid implementation directory is moved intact to `quarantine/` before a fresh
attempt. Completed implementations are not rerun.

## Isolation and provenance

Each case and implementation runs in a new process with a private cache tree for
Python bytecode, CUDA, CuTe, Helion, TorchInductor, Triton, XDG, and temporary
files. Helion runs a forced, cache-bypassed standard `full` autotune from a random
initial population. No fixed config, user seed config, wall-clock autotune budget,
or shape-specific winner is supplied. Its accepted result must include one
measured LFBO trial, the selected generated-source hash, post-measurement source
verification, 64 successful correctness launches, and exact repeatability.

Helion acceptance also reuses a snapshotted and hashed copy of the current
generalized-campaign validator for the complete v22 structural provenance and
source ledger. A local adapter supplies
the actual BF16 shape, ReLU kernel/baseline names, isolated cache fields, and
180-second candidate timeout that the older all8 entry point hardcodes. It then
validates the canonical compiler-seed policy, every compiler seed's generation-0
measurement, structural qualification timeline, selected source/config linkage,
and the joined `autotune.csv`, `autotune.meta.jsonl`, and
`autotune.sources.csv` run ID. All three sidecar hashes and that run ID are in the
result receipt and compact evidence.

The manifest records commit and tree IDs, the source subset hash used by the
strict harness, benchmark and attention source hashes, setup hashes, Python and
package RECORD hashes (including `apache-tvm-ffi`), compiler-tool hashes, exact
implementation version strings, GPU UUIDs, driver, and power limit. FA4 and Quack
source commit/tree identities are independent of any installed Quack package.
`input_spec_sha256` authenticates the documented deterministic input recipe and
shape. It is intentionally not a hash of tens of GiB of materialized tensor
bytes. Published CSV and compact evidence contain only relative artifact names;
raw strict results remain in the external campaign directory.

## Outputs

After all 32 implementation results validate, the launcher writes:

```text
published/attention_varied_shapes_b200_750w.csv
published/attention_varied_shapes_b200_750w_evidence.json
```

The CSV has the same schema consumed by the existing varied-attention renderer.
To render the plot after reviewing and copying the two published data files into
their intended plot locations, run:

```bash
python plots/render_varied_attention.py \
  "$OUTPUT_ROOT/published/attention_varied_shapes_b200_750w.csv" \
  "$OUTPUT_ROOT/published/attention_varied_shapes_b200_750w.png" \
  --evidence \
  "$OUTPUT_ROOT/published/attention_varied_shapes_b200_750w_evidence.json"
```

CPU-only validation of this setup is available with:

```bash
pytest plots/varied_attention_cute47/setup/test_campaign.py
```
