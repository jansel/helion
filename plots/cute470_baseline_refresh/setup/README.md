# CuTe 4.7 all-eight baseline refresh

This campaign refreshes the CuTe-dependent baseline rows in the existing eight
FP16 attention payloads. It reruns FA4 and FlexAttention-CuTe for every shape and
reevaluates only the KernelAgent Closed kernels whose retained rows currently
pass. It does not rerun Helion, SDPA, or any Triton/TileIR/TLX baseline. The
published result is a full eight-payload overlay suitable for the existing plot
pipeline, with every untouched row copied byte-for-byte as a JSON value from the
historical input.

Dense cases run on physical GPU 7 and causal cases on physical GPU 6. Both must
be NVIDIA B200s at a 750 W power limit. Every refreshed row runs correctness and
uses nine CUDA-event timing samples after the harness warmup.

## Workloads

| Variant | Shape | GPU | FA4 | FlexAttention-CuTe | KA Closed 1x | KA Closed 2x |
|---|---|---:|:---:|:---:|:---:|:---:|
| dense | `2x32x32768x64` | 7 | yes | yes | no | no |
| dense | `2x32x65536x64` | 7 | yes | yes | no | no |
| dense | `2x32x131072x64` | 7 | yes | yes | yes | yes |
| dense | `2x32x262144x64` | 7 | yes | yes | yes | no |
| causal | `2x32x65536x64` | 6 | yes | yes | yes | yes |
| causal | `2x32x131072x64` | 6 | yes | yes | yes | no |
| causal | `2x32x262144x64` | 6 | yes | yes | yes | yes |
| causal | `2x32x524288x64` | 6 | yes | yes | yes | no |

That is 25 refreshed rows: 16 library baselines and nine retained KernelAgent
Closed kernels. The omitted KernelAgent rows remain historical failures in the
overlay. The retained kernels were selected under CuTe `4.5.1`; this campaign
requires them to pass the standard, repeated, and stress correctness checks
under CuTe `4.7.0`, and labels their measured version as CuTe `4.7.0` while
retaining the selection version in full provenance.

## Pins and isolation

`EXPECTED_MEASURED_COMMIT` in `campaign.py` pins the measured Helion benchmark
to `c3e36b65d69681c23e053042b0bc21e2331bad17`. FA4 is pinned to commit
`2409214a03797b168f648ea30df1adbc09ce658a`
(`fa4-v4.0.0.beta23`), and Quack is pinned to
`b5b49dae477d39cb8ea8cca2820ef09ba548c72c`.

The Python environment must contain:

- PyTorch `2.13.0.dev20260506+cu130`
- Triton `3.7.0+git88b227e2`
- `nvidia-cutlass-dsl` `4.7.0`
- `nvidia-cudnn-cu13` `9.20.0.48`, runtime `9.20.0`
- `apache-tvm-ffi` `0.1.11`
- `cuda-bindings` `13.2.0`

The launcher never installs or fetches dependencies. It creates clean detached
Helion, FA4, and Quack worktrees under an external output root. It snapshots and
hashes the launcher scripts before execution. Historical payloads, each retained
KernelAgent manifest and source, and each prior KernelAgent result are copied
from clean tracked files into the campaign and hashed. The manifest also records
source commits/trees, package RECORD hashes, resolved module and cuDNN-library
file hashes, compiler-tool hashes, GPU UUIDs, driver, and power limit.

Every implementation process uses the detached measured Helion and pinned Quack
roots on `PYTHONPATH`; FA4 resolution is pinned with `HELION_FA4_ROOT`.
`PYTHONNOUSERSITE=1` is set, and inherited `PYTHONPATH`, `PYTHONHOME`,
`LD_LIBRARY_PATH`, and `LIBRARY_PATH` are not passed through. Each row has a
private CUDA, CuTe, TorchInductor, Triton, XDG, temporary, and bytecode cache.

## Run and resume

Choose an output directory outside all source repositories and provide existing
FA4 and Quack repositories that contain the pinned commits:

```bash
OUTPUT_ROOT=$(realpath -m ../cute470-baseline-refresh)
export CUTE470_BASELINE_FA4_SOURCE_REPO=$HOME/src/flash-attention
export CUTE470_BASELINE_QUACK_SOURCE_REPO=$HOME/src/quack
export CUTE470_BASELINE_PYTHON=$(command -v python)
plots/cute470_baseline_refresh/setup/run_campaign.sh "$OUTPUT_ROOT"
```

The historical payload and KernelAgent paths default to their tracked locations
in the current Helion repository. The corresponding environment variables exist
for explicit launch configuration, but the validator requires them to resolve
to `plots/kernelagent/results/payloads`, `plots/kernelagent_closed/runs`, and
`plots/kernelagent_closed/results` within that source repository.

For a durable run independent of the current shell or Codex session:

```bash
systemd-run --user --collect \
  --unit=helion-cute470-baseline-refresh \
  --working-directory="$PWD" \
  --setenv=CUTE470_BASELINE_FA4_SOURCE_REPO="$CUTE470_BASELINE_FA4_SOURCE_REPO" \
  --setenv=CUTE470_BASELINE_QUACK_SOURCE_REPO="$CUTE470_BASELINE_QUACK_SOURCE_REPO" \
  --setenv=CUTE470_BASELINE_PYTHON="$CUTE470_BASELINE_PYTHON" \
  "$(realpath plots/cute470_baseline_refresh/setup/run_campaign.sh)" \
  "$OUTPUT_ROOT"
```

Inspect progress with:

```bash
systemctl --user status helion-cute470-baseline-refresh.service
journalctl --user -u helion-cute470-baseline-refresh.service -f
tail -f "$OUTPUT_ROOT/dense-lane.log" "$OUTPUT_ROOT/causal-lane.log"
```

Resume after interruption with:

```bash
plots/cute470_baseline_refresh/setup/run_campaign.sh --resume "$OUTPUT_ROOT"
```

A nonblocking lock at `"$OUTPUT_ROOT.campaign.lock"` prevents two launchers from
using the same campaign. Each completed row is skipped only after its exact
invocation, result statistics, versions, source hashes, correctness flags, and
receipt hashes revalidate. Invalid or partial row directories are moved to
`quarantine/`. Lane cleanup gives each process group a bounded TERM grace period,
then sends KILL and reaps the tracked children.

## Outputs

After all 25 refreshed rows validate, `published/payloads/` contains the full
eight JSON payloads. `published/baseline_refresh_evidence.json` records the raw
nine-sample timings and relative result/receipt names, plus hashes for historical
payloads, retained KernelAgent artifacts, source snapshots, toolchain, and every
overlay payload. Published artifacts reject absolute local paths and refreshed
rows omit `input_seed` to preserve the historical payload schema.

The direct merge below is a preview only: it retains the historical Helion and
SDPA rows from the input payloads. It is useful for checking the refreshed
baseline rows before the generalized campaign is complete, but it is not the
final publication path.

```bash
PAYLOADS=(
  "$OUTPUT_ROOT/published/payloads/dense_32768.json"
  "$OUTPUT_ROOT/published/payloads/dense_65536.json"
  "$OUTPUT_ROOT/published/payloads/dense_131072.json"
  "$OUTPUT_ROOT/published/payloads/dense_262144.json"
  "$OUTPUT_ROOT/published/payloads/causal_65536.json"
  "$OUTPUT_ROOT/published/payloads/causal_131072.json"
  "$OUTPUT_ROOT/published/payloads/causal_262144.json"
  "$OUTPUT_ROOT/published/payloads/causal_524288.json"
)
python "$OUTPUT_ROOT/checkouts/helion/benchmarks/cute/compare_attention_backends.py" \
  --merge-json "${PAYLOADS[@]}" \
  --output "$OUTPUT_ROOT/published/attention_backends_b200_750w.md" \
  --csv-output "$OUTPUT_ROOT/published/attention_backends_b200_750w.csv" \
  --plot-output "$OUTPUT_ROOT/published/attention_backends_b200_750w.png" \
  --summary-plot-output \
  "$OUTPUT_ROOT/published/attention_backends_b200_750w_geomean.png" \
  --plot-impl-label 'kernelagent-1x=KernelAgent Public (1x Helion tuning time; $107 tokens)' \
  --plot-impl-label 'kernelagent-2x=KernelAgent Public (2x Helion tuning time; $245 tokens)' \
  --plot-impl-label 'kernelagent-10x=KernelAgent Public (10x Helion tuning time; $1,368 tokens)' \
  --plot-impl-label 'kernelagent-closed-1x=KernelAgent Closed (1x Helion tuning time; $167 tokens)' \
  --plot-impl-label 'kernelagent-closed-2x=KernelAgent Closed (2x Helion tuning time; $564 tokens)'
```

For final publication, pass this campaign's overlay directory to the generalized
publisher. That publisher replaces the historical `helion-cute` and `sdpa` rows
with its fresh paired measurements while preserving the refreshed FA4,
FlexAttention-CuTe, and KernelAgent Closed rows:

```bash
python "$GENERALIZED_TOOLING_ROOT/publish_results.py" \
  --paired-raw "$GENERALIZED_WORK/generalized-paired/all8_paired_raw.json" \
  --run-manifest "$GENERALIZED_WORK/generalized-paired/run_manifest.json" \
  --static-validation "$GENERALIZED_WORK/generalized-paired/static_validation.json" \
  --strict-artifact-root "$STRICT_ROOT" \
  --strict-manifest "$GENERALIZED_WORK/strict_manifest.csv" \
  --heldout-artifact-root "$HELDOUT_ROOT" \
  --heldout-manifest "$GENERALIZED_WORK/heldout_manifest.csv" \
  --generalization-artifact-root "$GENERALIZATION_ROOT" \
  --generalization-manifest "$GENERALIZED_WORK/generalization_manifest.csv" \
  --baseline-payload-dir "$OUTPUT_ROOT/published/payloads" \
  --output-payload-dir "$GENERALIZED_WORK/generalized-payloads" \
  --raw-artifact-label plots/generalized_full_autotune/all8_paired_raw.json \
  --renderer "$OUTPUT_ROOT/checkouts/helion/benchmarks/cute/compare_attention_backends.py" \
  --render-csv "$GENERALIZED_WORK/attention_backends_b200_750w.csv" \
  --render-plot "$GENERALIZED_WORK/attention_backends_b200_750w.png" \
  --render-summary-plot \
  "$GENERALIZED_WORK/attention_backends_b200_750w_geomean.png" \
  --plot-impl-label 'kernelagent-1x=KernelAgent Public (1x Helion tuning time; $107 tokens)' \
  --plot-impl-label 'kernelagent-2x=KernelAgent Public (2x Helion tuning time; $245 tokens)' \
  --plot-impl-label 'kernelagent-10x=KernelAgent Public (10x Helion tuning time; $1,368 tokens)' \
  --plot-impl-label 'kernelagent-closed-1x=KernelAgent Closed (1x Helion tuning time; $167 tokens)' \
  --plot-impl-label 'kernelagent-closed-2x=KernelAgent Closed (2x Helion tuning time; $564 tokens)'
```

CPU-only checks for this setup are:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/tmp/cute470-baseline-refresh-pycache \
pytest plots/cute470_baseline_refresh/setup/test_baseline_refresh.py
```
