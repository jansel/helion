# B200 Attention Backend Benchmarks

This directory contains two forward-attention comparisons:

- `attention_backends_b200_750w.csv`: one row per shape, including versions,
  timing, throughput, correctness, and Helion autotuner metadata.
- `attention_backends_b200_750w.png`: grouped TFLOP/s bars generated from the
  CSV's source JSON payloads.
- `attention_backends_b200_750w_geomean.png`: one geometric-mean TFLOP/s bar
  per complete backend across all eight shapes, plus an explicit marker for an
  incomplete backend.
- `attention_varied_shapes_b200_750w.csv`: long-form provenance and median
  TFLOP/s for the eight BF16 varied-shape audits.
- `attention_varied_shapes_b200_750w_evidence.json`: compact raw timing
  samples, FLOP counts, and identifiers for 21 original source audit artifacts.
  Those original audit files are not included in this tracked archive.
- `attention_varied_shapes_b200_750w.png`: grouped comparison of Helion CuTe,
  cuDNN SDPA, FA4, and FlexAttention CuTe over those varied shapes.
- `render_varied_attention.py`: deterministic renderer for the varied-shape
  CSV and chart.
- `kernelagent_tuning_usage.csv`: per-run and aggregate tuning time, Claude
  token accounting, completed call counts, and CLI-reported cost for
  KernelAgent Public.
- `kernelagent_closed_tuning_usage.csv`: per-run and aggregate tuning time and
  GPT-5.6 token accounting for KernelAgent Closed, including an OpenAI
  public-list cost estimate. The provider did not report an actual dollar cost.
- `kernelagent/`: selected generated kernels, manifests, setup sources, and raw
  measurements for the three KernelAgent Public baselines.
- `kernelagent_closed/`: selected CuTeDSL kernels, manifests, controlled MCP
  setup sources, and raw measurements for KernelAgent Closed 1x and 2x.
- `tlx/results/`: raw JSON for the eight Meta TLX attention measurements.
- `generalized_full_autotune/`: strict fresh-config verification tooling and thin
  source bundles. A completed generalized eight-shape result set is not tracked.
- `cute461_rerun/`: the CuTe 4.6.1 refresh and its retained source, cache,
  search, and paired-measurement artifacts behind the superseded `0e1113f8`
  Helion CuTe and SDPA bars.
- `mainrefresh_20260827/`: the current Helion CuTe and SDPA rows -- gzipped raw
  result JSON (including the complete per-candidate autotune ledgers and
  strict-prevalidation snapshots) plus a SHA256 manifest for the eight-shape
  2026-08-27 refresh.
- `mainrefresh_20260826/`: the superseded 2026-08-26 refresh (same protocol
  family but wall-clock timed for Helion CuTe and without the pre-measurement
  thermal cooldown; see the measurement-bias note below).

For the original fixed-shape FP16 report, the non-CuTe baselines originated from
`benchmarks/cute/compare_attention_backends.py` at `7a754822`; TLX and the two
KernelAgent families were added later with the same shapes and timing protocol.
The current Helion CuTe and SDPA rows come from the 2026-08-27 refresh under
`mainrefresh_20260827/`: for every shape, a fresh strict cold full-effort
autotune (`--helion-require-full-autotune 1`) and standalone nine-sample
measurement at commit `7789ca326` (`1.4.0.dev168+g7789ca326`) with CuTe 4.7.0,
plus a same-protocol SDPA rerun. Standalone FA4, FlexAttention's FA4 path, and
passing KernelAgent Closed sources retain their existing CuTe 4.6.1
measurements. Failed KernelAgent attempts that produced no executable source
retain their truthful 4.5.1 selection metadata but do not contribute a bar or
legend version. No non-Helion/non-SDPA baseline was rerun for this refresh.
The superseded `0e1113f8` paired-campaign rows remain archived under
`cute461_rerun/`, and the superseded 2026-08-26 rows under
`mainrefresh_20260826/`.

## Varied BF16 shape report

The varied-shape chart covers four pairs of forward-attention workloads:

| Group | Dense | Causal |
| --- | --- | --- |
| Head dimension 128 | `2x32x262144x128` | `2x32x524288x128` |
| Batch 1 | `1x32x524288x64` | `1x32x1048576x64` |
| Batch 8 | `8x32x524288x64` | `8x32x786432x64` |
| ReLU epilogue | `2x32x524288x64` | `2x32x1048576x64` |

All inputs are BF16. The bars report the median throughput from the audited
artifacts named in the CSV's `evidence_file` column. The tracked evidence JSON
contains every timing sample used by the chart, the FLOP count used to derive
TFLOP/s, and the names of the original audit artifacts. Those original files
were not retained here, so the compact evidence reproduces the CSV and PNG but
does not independently audit source/config/correctness provenance. The Helion
and strongest baseline values use randomized paired campaigns where those campaigns exist;
the remaining baseline values come from same-shape standalone screens. This is
therefore a composite report, not a single four-way paired experiment. The CSV
records the physical GPU, sample count, timing mode, and correctness status for
every bar. Only physical GPUs 6 and 7 were used, on NVIDIA B200 GPUs configured
for a 750 W power limit.

Every row uses the same attention-forward FLOP model: QK and PV matrix
products, with triangular work for causal attention. ReLU execution is timed
but its pointwise operations are not added to the FLOP numerator. Helion fuses
ReLU into the attention epilogue. The SDPA and FA4 ReLU rows include a timed
eager `torch.relu`; FlexAttention uses `torch.compile(fullgraph=True)` around
FlexAttention and ReLU. Helion, SDPA, and FlexAttention use native BHSD inputs;
FA4 uses native BSHD inputs. No layout conversion is in the timed region.

The legend identifies the consolidated landable Helion stack at
`e094f171` (`1.4.0.dev95+ge094f171`) with CuTe 4.6.1. The underlying audits
span the equivalent pre-consolidation source commits recorded by their raw
artifacts; for example, the strongest D128 audits were clean at `89aae72e`.
The other versions are PyTorch `2.13.0.dev20260506`, cuDNN `9.20.0.48`,
FlashAttention `fa4-v4.0.0.beta23`, and CuTe 4.6.1.

One baseline has a narrower correctness claim: the batch-8 dense
FlexAttention CuTe timing completed, but exact-shape full-output correctness
was skipped because the validation allocation exceeded available memory. The
bar is retained and marked `SKIPPED_FULL_SHAPE_MEMORY` in the CSV; it is below
the fully checked SDPA result and cannot change the best-baseline conclusion.
Some older paired artifacts record the controlled GPU assignment but omit an
inline power-cap field; the CSV marks those entries as `campaign` rather than
`yes` in `power_cap_recorded`.

Regenerate the chart without rerunning a kernel:

```bash
python plots/render_varied_attention.py \
  plots/attention_varied_shapes_b200_750w.csv \
  plots/attention_varied_shapes_b200_750w.png \
  --evidence plots/attention_varied_shapes_b200_750w_evidence.json
```

The remaining sections document the original fixed-shape FP16 report.

## Hardware and software

| Component | Version used |
| --- | --- |
| GPU | NVIDIA B200, driver 580.82.07 |
| Power limit | 750 W, verified by the harness before every run |
| Python | 3.12.13 |
| PyTorch | 2.13.0.dev20260506+cu130 |
| CUDA reported by PyTorch | 13.0 |
| cuDNN | 9.20.0, package `nvidia-cudnn-cu13==9.20.0.48` |
| Helion CuTe kernel source | 1.4.0.dev168+g7789ca326 (commit `7789ca3269887c687c850a5a1f3e6c6823aac267`) |
| Standard Triton | 3.7.0+git88b227e2 |
| CuTe evaluation runtime | 4.7.0 for the Helion CuTe rows; 4.6.1 for the retained CuTe-backed baselines |
| Quack source | 0.4.1, commit `b5b49dae477d39cb8ea8cca2820ef09ba548c72c` |
| Apache TVM FFI | 0.1.11 |
| FlashAttention-4 | `fa4-v4.0.0.beta23` |
| Gluon source | Triton commit `88b227e23f0445f3f695bad05bbf1a363b4f50e0` |
| TLX | Meta Triton `3.7.4+fb`, commit `7632ef23c8f4667d659e19820d39d80f613fbed1` |
| TLX attention source | SHA256 `03b55b10cfb36daee36d504abeaaeca8003852158bfcac7a373b303ae7a394b3` |
| TileIR toolchain | `nvtriton==3.6.0`, TileIR 13.3 (`cuda_tile_v14.3.0`) |
| KernelAgent Public | `v2+e0647170` (commit `e0647170da36ef9b059ac0bd3d60103aa4ed378b`) |
| KernelAgent Public model | Claude Opus 5.0 via Claude Code 2.1.220 |
| KernelAgent Closed | `v3-20260730`; binary SHA256 `2859181c5786a782d8946032967a5ee70a9bf191e272137c97703a12550330c0` |
| KernelAgent Closed model | GPT-5.6 (`gpt-5.6-sol-ptu`, xhigh reasoning) |
| KernelAgent Closed selection runtime | CuTe 4.5.1 |

The full Gluon source SHA256 and every displayed implementation version are in
the CSV. Dense shapes ran on physical GPU 7 and causal shapes on physical GPU
6. The two groups can run concurrently, but implementations for a single shape
must run serially to avoid contention.

## Workload and metric

All inputs are FP16 with `B=2`, `H=32`, and `D=64`:

| Variant | Sequence lengths |
| --- | --- |
| Dense | 32K, 64K, 128K, 256K |
| Causal | 64K, 128K, 256K, 512K |

The timing harness creates independent normal Q, K, and V tensors in
`[B, H, S, D]` layout with input seed 2026080106. Reported
throughput uses the benchmark harness convention:

```text
dense FLOPs  = 4 * B * H * S^2 * D
causal FLOPs = 4 * B * H * (0.5 * S^2) * D
TFLOP/s      = FLOPs / (milliseconds * 1e9)
```

This counts the QK and PV matrix multiplications. It does not count softmax or
masking operations. Helion uses output-only attention, so LSE is not included.
The upstream Gluon entry point returns both output and LSE; its Gluon timing
therefore includes the extra LSE write. The TLX wrapper returns only output,
but its forward kernel writes FP32 softmax state for backward, so TLX timing
also includes an auxiliary state write.

## Measurement protocol

The retained non-Helion/non-SDPA baselines use the original fresh-process
nine-sample protocol:

1. Run five cache-warmup calls and synchronize.
2. Run sustained BF16 matrix multiplication for 10 seconds to stabilize B200
   clocks.
3. Take nine independent `triton.testing.do_bench` measurements, each with a
   1000 ms warmup window and a 500 ms measurement window.
4. Use CUDA-event timing for every series.
5. Report the median of the nine results as `mom_med_ms` and
   `mom_med_tflops`. Best-of-nine values are diagnostic only.

All current Helion CuTe and SDPA bars come from the 2026-08-27 standalone
refresh, which uses the exact fresh-process nine-sample protocol of the
retained baselines: every implementation, including Helion CuTe, is timed by
CUDA-event `triton.testing.do_bench` with the per-run **median** statistic
(median of nine per-run medians). Two measurement biases present in the
superseded 2026-08-26 refresh are fixed:

1. **Timer parity.** The 2026-08-26 Helion rows used a synchronized
   wall-clock timer (CUDA events once mis-read CuTe launches, an artifact of
   the pre-compiled-launcher ~200 ms host dispatch that no longer exists;
   event and wall timing now agree to <0.1 ms on these kernels). The wall
   interval additionally charges ~0.2 ms of per-launch host overhead to the
   kernel, penalizing only the Helion rows by ~0.4-0.8% on the shorter
   shapes.
2. **Thermal parity.** Under the 750 W cap a heat-soaked B200 sustains ~1%
   lower attention throughput than one starting near idle temperature. The
   strict rows were previously measured immediately after multi-hour autotune
   searches while every baseline ran on an idle GPU. The harness now idles
   before the warmup/measurement phase until the GPU returns to its
   process-startup temperature plus a small margin
   (`--measure-cooldown-margin-c`, default 3 C, with a plateau exit and a
   hard 300 s cap) and records the wait as `thermal_cooldown` in each row;
   this is a no-op for the already-cold baseline processes. The tracked
   2026-08-27 rows were produced by this cooldown's original absolute form
   (cool to 55 C; each row's `thermal_cooldown` records the exact request
   and outcome), which is equivalent on these machines, whose idle
   temperature is 37-44 C.

Each Helion CuTe row ran under `--helion-require-full-autotune 1`: a cold
`FROM_RANDOM` full-effort LFBO search with forced cache bypass, best-of-1, no
wall-clock or generation caps, no fixed, user, or seeded configs, canonical
compiler seeds measured in generation zero, and a 64-launch full-output
correctness plus bitwise-repeatability check against forced
`SDPBackend.CUDNN_ATTENTION` before timing. The measured source checkout is
hash-verified before and after tuning (commit `7789ca326`, clean tree).
Input tensors use seed 2026080106, matching the retained baselines; autotune
RNG seeds were `2026092100 + S/1024` for dense and `2026093100 + S/1024` for
causal. Dense shapes ran serially on physical GPU 7 and causal shapes
serially on physical GPU 6, with the two lanes concurrent and both GPUs
verified at 750 W. The complete raw artifacts, including every measured
candidate and the strict-prevalidation snapshots, are gzipped under
`mainrefresh_20260827/results/` with SHA256s in
`mainrefresh_20260827/manifest.json`; the payload rows embed the same strict
provenance with the bulkiest ledger fields replaced by
`{sha256, bytes, entries}` fingerprints that match the archived raw JSON.

| Shape | Helion CuTe TFLOP/s | SDPA TFLOP/s | Median ratio |
| --- | ---: | ---: | ---: |
| Dense 32K | 740.384 | 745.977 | -0.750% |
| Dense 64K | 745.897 | 742.271 | +0.488% |
| Dense 128K | 747.513 | 742.925 | +0.618% |
| Dense 256K | 742.639 | 743.621 | -0.132% |
| Causal 64K | 723.111 | 727.667 | -0.626% |
| Causal 128K | 723.088 | 738.264 | -2.056% |
| Causal 256K | 734.405 | 743.419 | -1.212% |
| Causal 512K | 736.194 | 719.329 | +2.344% |
| Overall geometric mean | 736.601 | 737.881 | -0.173% |

These absolute values are not comparable to the superseded `0e1113f8` paired
rows: the paired campaign timed single warm-cache calls with raw CUDA events,
while this protocol's `do_bench`-style timing flushes L2 between calls, which
lowers every number by roughly 3-4% on these shapes. Under this refresh
Helion CuTe beats SDPA on three of eight shapes and trails by 0.173% on the
eight-shape geometric mean; no paired-inference confidence interval is
claimed for the current rows. Repeated same-config measurements on these
GPUs still spread by up to ~0.7% with thermal history and day-to-day drift
(the retained baseline rows carry their historical median-of-nine-means
values), so single-shape ratios inside roughly +/-1% should be read as ties.

Bitwise repeatability is a useful race screen, not a formal race proof. The
historical largest-shape study below records the stronger numerical and
Compute Sanitizer checks performed on the earlier fixed configurations. A
failed KernelAgent Closed shape has no throughput bar and is marked `FAIL`.
FlexAttention was forced to `TRITON` or `FLASH` as named in the plot; its Flash
path uses the same FA4 checkout as the standalone FA4 baseline.

Grouped-chart bars are ordered by increasing arithmetic mean TFLOP/s across
their passing shapes. Summary-chart bars are ordered by increasing geometric
mean and require a passing result for every shape. An incomplete backend is
shown as `INCOMPLETE (passing/total)` at zero, without assigning it a partial
geometric mean.

## Source setup

Restore the cleaned generalized Helion CuTe source and strict benchmark from
the retained bundle. The bundle has SHA256
`a39cfcd01206c36609a178ee483e40194189a00f878e8d48f68a082aec3cdfad` and
contains the complete history through measured commit
`c3e36b65d69681c23e053042b0bc21e2331bad17`. Use a sibling worktree: checking
out the measured commit in the artifact repository would remove the tracked
plots and setup files needed by later commands.

From the artifact repository root:

```bash
export ARTIFACT_REPO=$(git rev-parse --show-toplevel)
export REPRO_ROOT=${REPRO_ROOT:-$HOME/helion-attention-repro}
export GENERALIZED_BUNDLE="$ARTIFACT_REPO/plots/generalized_full_autotune/helion-generalized-c3e36b65.bundle"
test "$(sha256sum "$GENERALIZED_BUNDLE" | cut -d' ' -f1)" = \
  a39cfcd01206c36609a178ee483e40194189a00f878e8d48f68a082aec3cdfad
git -C "$ARTIFACT_REPO" bundle verify "$GENERALIZED_BUNDLE"
git -C "$ARTIFACT_REPO" fetch --no-tags "$GENERALIZED_BUNDLE" \
  refs/archive/helion-generalized-c3e36b65:refs/archive/helion-generalized-c3e36b65
git -C "$ARTIFACT_REPO" worktree add --detach \
  "$REPRO_ROOT/helion-generalized-c3e36b65d" \
  refs/archive/helion-generalized-c3e36b65
export REPO="$REPRO_ROOT/helion-generalized-c3e36b65d"
test "$(git -C "$REPO" rev-parse HEAD)" = \
  c3e36b65d69681c23e053042b0bc21e2331bad17
export WORK="$REPRO_ROOT/work"
mkdir -p "$WORK/results" "$WORK/payloads"
```

Keep the fetched bundle ref under `refs/archive/`, as shown above. Installing
it as a local tag would change `git describe` and therefore the Helion version
recorded by the benchmark; the strict launcher rejects that state before GPU
work begins.

Check out FlashAttention at `fa4-v4.0.0.beta23` and Triton at
`88b227e23f0445f3f695bad05bbf1a363b4f50e0`, then point the harness at them:

```bash
export HELION_FA4_ROOT=/path/to/flash-attention
export TRITON_ROOT=/path/to/triton
export HELION_GLUON_ATTENTION_PATH="$TRITON_ROOT/python/examples/gluon/01-attention-forward.py"
export HELION_GLUON_VERSION=88b227e23f0445f3f695bad05bbf1a363b4f50e0
export HELION_BENCHMARK_HELION_VERSION=1.4.0.dev157+gc3e36b65d
export HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS=6,7
```

Clone the exact Quack source under the reproduction root:

```bash
git clone https://github.com/Dao-AILab/quack.git "$REPRO_ROOT/quack"
git -C "$REPRO_ROOT/quack" checkout b5b49dae477d39cb8ea8cca2820ef09ba548c72c
export PYTHONPATH="$REPRO_ROOT/quack${PYTHONPATH:+:$PYTHONPATH}"
```

The `PYTHONPATH` entry exposes `quack/quack/` as the `quack` package without
installing a different release into the environment.

Commit `c3e36b65d` contains the cleaned strict reporting harness and generalized
Helion CuTe source used for new verification runs. Install
`nvidia-cutlass-dsl==4.7.0`; do not reuse a CuTe
compile cache created by another version. The original non-Helion rows came
from the snapshots recorded in their CSV version fields and in the
backend-specific sections below. Helion Triton and Helion TileIR use the
separate historical worktrees restored next.

The pre-rebase `helion-generalized-d9cae807.bundle` remains an immutable CuTe
4.6.1 archive. Its results must not be relabeled as validation of the current
source. The later `helion-generalized-a6b5049a.bundle` is likewise retained
only as an immutable pre-`692bf81d` archive. The
`helion-generalized-edd79764.bundle` archive has SHA256
`a097f409f085cf5292d9b6686d6d64b08bd0b50e4d35b4008bce520c55a0803a` and
contains historical diagnostic evidence for
`edd79764349bdbd43dfb6afbcf4e620128aecc11`; its failed held-out campaign is
not publication evidence for `c3e36b65d`.

Restore the Helion source and reporting harness used for the historical Helion
Triton and TileIR rows into two more sibling worktrees. The source bundle has
SHA256 `be6419eb9e3ee1be7d3ef9165aec85d9fd0bed7321c54621308e8517c862df1e`
and requires `ba7ec31e02557cd6ddd3119eb2b2318ef52acd6f`. The harness bundle has
SHA256 `4a3982ebd2187a4515a3b184632191adcbdf538e5b3cff7f80d4a4fe741f2e78`
and requires the source commit restored by the first bundle:

```bash
export LEGACY_BUNDLE="$ARTIFACT_REPO/plots/cute461_rerun/helion-016ad645.bundle"
test "$(sha256sum "$LEGACY_BUNDLE" | cut -d' ' -f1)" = \
  be6419eb9e3ee1be7d3ef9165aec85d9fd0bed7321c54621308e8517c862df1e
git -C "$ARTIFACT_REPO" rev-parse --verify \
  ba7ec31e02557cd6ddd3119eb2b2318ef52acd6f^{commit}
git -C "$ARTIFACT_REPO" bundle verify "$LEGACY_BUNDLE"
git -C "$ARTIFACT_REPO" fetch "$LEGACY_BUNDLE" \
  refs/archive/helion-triton-tileir-016ad645:refs/archive/helion-triton-tileir-016ad645
git -C "$ARTIFACT_REPO" worktree add --detach \
  "$REPRO_ROOT/helion-016ad645" refs/archive/helion-triton-tileir-016ad645
export LEGACY_REPO="$REPRO_ROOT/helion-016ad645"
test "$(git -C "$LEGACY_REPO" rev-parse HEAD)" = \
  016ad64544743e5ce29f55972c24a960ee6cbc45

export HARNESS_BUNDLE="$ARTIFACT_REPO/plots/cute461_rerun/attention-harness-7a754822.bundle"
test "$(sha256sum "$HARNESS_BUNDLE" | cut -d' ' -f1)" = \
  4a3982ebd2187a4515a3b184632191adcbdf538e5b3cff7f80d4a4fe741f2e78
git -C "$ARTIFACT_REPO" bundle verify "$HARNESS_BUNDLE"
git -C "$ARTIFACT_REPO" fetch "$HARNESS_BUNDLE" \
  refs/noland/attention-harness-7a754822:refs/archive/attention-harness-7a754822
git -C "$ARTIFACT_REPO" worktree add --detach \
  "$REPRO_ROOT/harness-7a754822" refs/archive/attention-harness-7a754822
export HARNESS_REPO="$REPRO_ROOT/harness-7a754822"
test "$(git -C "$HARNESS_REPO" rev-parse HEAD)" = \
  7a754822961ef6f4a0d275d07516545c2d9221fd
```

The harness contains the compatibility shims needed to load this FA4 checkout
with CuTe 4.6.1. The local Quack 0.4.1 source still annotates
`cute.core.ThrMma` and `cute.core.ThrCopy`; the loader aliases those names to
their 4.6.1 top-level locations before importing FA4. It also retains the NVVM
enum and primitive-ABI adapters required by this checkout. These are import and
binding compatibility changes, not kernel changes. Do not substitute the
harness's older default FA4 ref.

FA4 beta23's package metadata pins CuTe 4.6.0.dev0, Quack 0.5.3 or newer, and
Apache TVM FFI 0.1.12 or newer. This environment instead uses CuTe 4.6.1,
Quack 0.4.1, and Apache TVM FFI 0.1.11; the loader bridge above is therefore a
deliberate compatibility setup rather than FA4's declared dependency set. Both
dense and causal FA4 paths passed the full-output check before the sweep.

Set the power limit immediately before benchmarking and restore the machine's
normal limit afterward:

```bash
sudo nvidia-smi -i 6 -pl 750
sudo nvidia-smi -i 7 -pl 750
# Run benchmarks.
sudo nvidia-smi -i 6 -pl 850
sudo nvidia-smi -i 7 -pl 850
```

## Canonical invocation

The standalone harness can rerun each implementation separately. For a CSV
row, set `GPU` from `physical_gpu`, `S` from `seq_len`, and `CAUSAL` from
`causal`. The tracked Helion CuTe and SDPA bars use exactly this invocation
from the repository checkout at `7789ca326`, with one addition for Helion
CuTe: `--helion-require-full-autotune 1 --helion-force-autotune 1
--helion-autotune-effort full`, per-shape fresh `HELION_CACHE_DIR` /
`TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` / `CUTE_DSL_CACHE_DIR`
`--helion-env` overrides, and the `HELION_AUTOTUNE_RANDOM_SEED` values recorded
in `mainrefresh_20260827/manifest.json`. The event timer and the
pre-measurement cooldown are harness defaults and need no extra flags. The
superseded `0e1113f8` paired rows remain archived under `cute461_rerun/`
(source bundle, winner configs, and raw timings in `helion-0e1113f8.bundle`,
`helion_cache/`, and `paired_all8.json`).

```bash
cd "$REPO"
GPU=7
S=32768
CAUSAL=0
IMPL=sdpa
OUT="$WORK/results/${CAUSAL}_${S}_${IMPL}.json"

CUDA_VISIBLE_DEVICES="$GPU" python \
  benchmarks/cute/compare_attention_backends.py \
  --impl "$IMPL" \
  --z 2 --h 32 --seq-len "$S" --head-dim 64 \
  --dtype float16 --causal "$CAUSAL" --biased 0 \
  --num-runs 9 --warmup-ms 1000 --rep-ms 500 \
  --seed 2026080106 --power-cap-w 750 \
  --skip-correctness 0 --helion-return-lse 0 \
  --helion-cute-benchmark-timer event \
  --json --json-output "$OUT"
```

Repeat this for the fourteen displayed implementations:

```text
sdpa
fa4
gluon
tlx
flexattention
flexattention-cute
helion-triton
helion-cute
helion-tileir
kernelagent-1x
kernelagent-2x
kernelagent-10x
kernelagent-closed-1x
kernelagent-closed-2x
```

The two historical Helion implementations must import Helion from commit
`016ad645` while using the reporting harness from `7a754822`. For
`IMPL=helion-triton` or `IMPL=helion-tileir`, replace the Python command above
with the following equivalent invocation; keep the same arguments after the
script path:

```bash
export HELION_BENCHMARK_HELION_VERSION=1.4.0.dev38+g016ad645
PYTHONPATH="$LEGACY_REPO${PYTHONPATH:+:$PYTHONPATH}" \
  CUDA_VISIBLE_DEVICES="$GPU" python \
  "$HARNESS_REPO/benchmarks/cute/compare_attention_backends.py" \
  --impl "$IMPL" \
  --z 2 --h 32 --seq-len "$S" --head-dim 64 \
  --dtype float16 --causal "$CAUSAL" --biased 0 \
  --num-runs 9 --warmup-ms 1000 --rep-ms 500 \
  --seed 2026080106 --power-cap-w 750 \
  --skip-correctness 0 --helion-return-lse 0 \
  --helion-cute-benchmark-timer event \
  --json --json-output "$OUT"
```

For the KernelAgent Public entries, also pass
`--kernelagent-results-root "$ARTIFACT_REPO/plots/kernelagent/runs"`. For KernelAgent Closed,
pass
`--kernelagent-closed-results-root "$ARTIFACT_REPO/plots/kernelagent_closed/runs"`.
KernelAgent produced one fixed-shape output kernel for each attempted workload;
those saved artifacts cover only the eight FP16, output-only shapes in this
report. This restriction does not describe Helion's autotuner, which runs a
live search for each bound shape as documented below.

Helion runs also require the exact settings recorded in the corresponding
`*_helion_overrides` CSV cell. Translate the JSON fields as follows:

| CSV field | Command-line translation |
| --- | --- |
| `force_autotune` | `--helion-force-autotune 0` or `1` |
| `env_overrides` | One `--helion-env KEY=VALUE` per entry |
| `seed_config_overrides` | One `--helion-seed-config KEY=JSON` per entry |
| `config_overrides` | One `--helion-config KEY=JSON` per entry |

### Generalized full-autotune verification

The `generalized_full_autotune/setup` harness exercises the cleaned search at
`c3e36b65d`; the tracked CSV and PNGs instead use the in-harness strict mode at
`7789ca326` documented above, not this external multi-campaign flow.
It forces `full` effort from a cold population whose normalized coverage design
reaches every active pipeline family and exp2 packet. V22 qualification then
establishes measured witnesses and conditional children for active pipeline
lanes, exercises legal CLC divisors and compound transfers, then probes every
live parent family before promoting the best four. Every compound leaf and one
unrestricted path remain active. The deterministic-row budget is derived from
the live normalized design and requested population size. Any remaining
initial slots are filled without a measured winner. The run rejects fixed
configs, user/CLI config seeds, cache reads, value priors, time or generation
budgets, custom tuners, noncanonical compiler heuristics, and CuTe codegen
overrides. The normal canonical CuTe-flash compiler seeds remain enabled. Their
ordered normalized IDs are authenticated against the live ConfigSpec and every
effective seed must succeed or source-deduplicate in generation zero. The
recorded per-shape RNG seed controls the remaining candidate generation; it
does not supply any config field. Each accepted run records every effective
candidate and requires the selected source digest to identify a candidate that
was actually measured. The launcher, manifest validator, paired worker, and
publication checks are documented in
`generalized_full_autotune/setup/README.md`.

“Shape-tuned” here does not mean “shape-hardcoded.” The B200/SM100 path measured
here contains no lookup for the eight reported sequence lengths or for
`B=2,H=32`. It derives a search surface from the recognized attention graph,
dtype, head dimension, causality, modifiers, batch-head count, and structural
tile divisibility. The ordered choices and defaults are identical for every
sequence length in the same legality class. Standard full tuning can therefore
search the same families for a new supported B200 shape, but the finite LFBO
search is stochastic and does not guarantee recovery of the absolute global
optimum. Separate upstream SM103 target policies retain measured target/shape
seed and lowering choices; they do not participate in this B200 campaign or
support a broader cross-target no-lookup claim.

Structural choices are searched as legal compound families. In particular,
`cute_flash_q_tile_count` is derived from the selected pipeline family (two for
FA4 families, one for the overlap family) rather than sampled independently;
the barrier graph and shared-memory layout make arbitrary cross-products
invalid. The legacy field remains accepted and canonicalized for compatibility.
Likewise, historical `deg1_*` packet strings name cadence/batching schedules;
their runtime path uses the degree-two exp2 evaluator. Renaming those strings
would change generated-source hashes, so neutral aliases are deferred to a
separate compatibility change.

No completed `c3e36b65d` eight-shape manifest, paired result, or publication
payload is tracked yet. The older `3af9343a` and `5809652a` thin bundles are
retained as source history only; they must not be treated as evidence that the
current chart was produced by the generalized search. The strict launcher uses
fresh per-shape
CUDA, CuTe, Triton, TorchInductor, XDG, and Helion cache roots and pins both
Helion and the attention example imports to the detached source checkout.

Helion Triton retains its original protocol: `from_random` except causal 512K,
which transferred the previous winner with `from_best_available`. The CSV is
the authoritative per-shape report and preserves the structured Helion
overrides.

### Paired verification protocol

Run two complete campaigns with orchestrator seeds `2026081101` and
`2026081102`. For each shape, derive the input/order seed as
`orchestrator_seed + S + 1_000_000` for causal and
`orchestrator_seed + S` for dense. Use physical GPU 7 for dense and physical
GPU 6 for causal, identified by the UUIDs recorded in the raw artifact. Reject
any device that is not an NVIDIA B200 at a 750 W power limit.

For each fresh per-shape process:

1. Create a CUDA `Generator`, seed it with the derived value, then allocate Q,
   K, and V in that order with three FP16 `torch.randn((2, 32, S, 64))` calls.
2. Bind the measured config, regenerate its source, require its SHA256 to match
   the selected measured source, compile that exact config, and force the
   reference through `SDPBackend.CUDNN_ATTENTION`.
3. Compare the complete output in 2,048-row chunks at `atol=0.05`,
   `rtol=0.02`; require finite output and two additional bitwise-identical
   Helion results.
4. Run ten seconds of sustained BF16 matrix multiplication, then three warmup
   calls per implementation with alternating order.
5. Create six `helion,sdpa` and six `sdpa,helion` orders, shuffle with
   `random.Random(seed ^ 0x5A17)`, and time one call per observation using
   start/end CUDA events plus synchronized host wall time.
6. Combine the two campaigns only after both pass. Plot each implementation's
   median over its 24 event samples. For paired inference, use
   `log(sdpa_ms / helion_ms)`, resample 12 pairs within each campaign/shape
   stratum, and take bootstrap elements 500 and 19,500 of 20,000 sorted draws.

The bootstrap interval is conditional on the two observed fixed campaign
strata. Publication also compares event and wall timing from the same calls: a
shared-pair bootstrap must bound relative timer bias within 0.5%, separate
checks catch common-mode event failures, and at least 22 of 24 calls must satisfy
the per-call bounds. A sub-0.5% point estimate is called directional only when
both timer intervals, both campaigns, and the plotted marginal medians agree;
otherwise it is recorded as inconclusive while the measured bars remain
available.

The raw artifact contains every measured value needed to recompute the medians,
geometric means, paired estimates, and stratified bootstrap intervals without
rerunning a kernel. It records base bootstrap seed `2026081103`, every derived
shape/timer seed, and the 20,000-draw intervals, so the endpoints are fully
reproducible.

## Historical largest-shape Helion+CuTe validation study

The largest dense and causal Helion+CuTe rows were independently rechecked
because both originally appeared to beat cuDNN SDPA. The saved generated
kernels were used first. After those kernels failed the stronger validation,
accurate degree-two replacements were compiled, phase-tuned, and measured.
The historical corrected bars were:

| Shape | Helion+CuTe TFLOP/s | Paired SDPA TFLOP/s | Difference |
| --- | ---: | ---: | ---: |
| Dense `2x32x262144x64` | 779.605 | 774.735 | +0.629% |
| Causal `2x32x524288x64` | 758.345 | 754.546 | +0.504% |

The two historical rows beat their paired references. Across that superseded
eight-shape dataset, Helion+CuTe's geometric mean was 752.461 TFLOP/s versus
766.674 for SDPA, a -1.854% aggregate difference. These are not the current
plot values.

These are 12-pair fixed-config measurements using raw CUDA events, not
full-autotune reruns. The table's differences are ratios of the two median
throughputs; confidence intervals use the paired samples directly. Dense
reproduced on both devices: GPU 7 measured a +0.660% mean paired gain with a
bootstrap 95% confidence interval of `[+0.573%, +0.757%]`; GPU 6 measured
789.839 versus 785.849 TFLOP/s, or +0.456% with interval
`[+0.291%, +0.578%]`. Causal also reproduced: GPU 6 measured a +0.509% gain
with interval `[+0.476%, +0.543%]`; GPU 7 measured 746.075 versus 743.187
TFLOP/s, or +0.388% with interval `[+0.308%, +0.464%]`.

The raw timing artifacts record base commit `a657aa4b` because the two phase
choices were supplied as explicit fixed configs. Commit `d227cd7e` promotes
those same choices into compiler seeds; it does not otherwise change the
generated paths measured here. Those commit identifiers describe only this
historical study; the current plot uses `7789ca326`.

The replacement configurations use these structural schedules; every explicit
override is retained in the payload `config` and `helion_overrides` objects:

| Shape | Validated schedule |
| --- | --- |
| Dense 256K | `fa4_2cta`, `deg2_16x6`, offsets `12/2`, KV depth 2, nonpersistent, `single_final`, wait 0, registers `192/72/40`, rescale threshold/chunk `8/8` |
| Causal 512K | `fa4`, `deg2_16x6`, masked and unmasked cadence `16/6`, offsets `14/12`, wait 0, discontinuity depth 3, rescale chunk 16, LPT swizzle 0 |

The full-output comparisons below use strict failures at `atol=2e-3` and
`rtol=1e-2`. NRMSE is normalized by the RMS magnitude of the SDPA output.
Peaky inputs multiply Q and K by two; uniform inputs set Q and K to zero and
also permit an analytic reference.

| Shape and input | Max absolute error vs SDPA | NRMSE | Strict failures |
| --- | ---: | ---: | ---: |
| Dense 256K, normal | 0.0000305 | 0.08190% | 0 / 1.074B |
| Dense 256K, peaky | 0.005859 | 0.06994% | 147 / 1.074B |
| Dense 256K, uniform | 0.00000763 | 0.09392% | 0 / 1.074B |
| Causal 512K, normal | 0.004883 | 0.08134% | 9 / 2.147B |
| Causal 512K, peaky | 0.005859 | 0.06969% | 374 / 2.147B |
| Causal 512K, uniform | 0.001953 | 0.08817% | 0 / 2.147B |

Five additional normal-input executions were bitwise identical for both
kernels. Sampled FP32 references also confirmed that the errors were small and
not merely agreement with SDPA rounding. Compute Sanitizer on the final dense
`12/2` schedule at the same-path `1x1x32768x64` shape reported zero racecheck
hazards and zero synccheck errors. The final causal `14/12` schedule also
reported zero racecheck hazards at that shape. Causal synccheck at 32K, 4K,
and 1K did not finish in practical time because instrumented mbarriers ran
prohibitively slowly, so no causal synccheck result is claimed.

The withdrawn kernels explain why the original loose check was insufficient.
The dense degree-one kernel had 16,386 shared-memory write-after-read hazards;
on peaky inputs its maximum error was 0.071289 and about 9.743% of elements
failed the strict comparison. The causal degree-one kernel was racecheck-clean,
but its normal maximum error was 0.01611 with 0.966% NRMSE, and its peaky
maximum error was 0.07593 with about 10.22% strict failures. The degree-one
exp2 approximation itself has 2.983% maximum relative error. The old dense
776.7 and causal 776.6 TFLOP/s plot values were therefore removed.

Profiling and controlled ablations identify the dense speedup mechanism. A
terminal-only statistics handoff is +1.608% over a fully acknowledged one-slot
handoff. Moving the degree-two phase from `0/2` to `12/2` adds a further
+0.525% mean paired gain with bootstrap interval `[+0.379%, +0.662%]`; it does
not reduce the instruction count, but NCU reports 0.623 versus 0.601 eligible
warps per scheduler. Against SDPA, the final kernel executes 8.32% fewer
instructions, reads 0.93% less DRAM, has 2.98% fewer L2 read misses, and uses
101,376 rather than 232,448 bytes of dynamic shared memory. Its long-scoreboard
stall ratio is 1.938 versus 3.371 for SDPA. NCU measures it 0.347% faster even
though issue utilization is lower, 52.08% versus 56.60%. Both kernels remain
limited to one block per SM, so the shared-memory reduction does not increase
the 25% theoretical occupancy by itself. The dense advantage therefore comes
from less instruction work and fewer long-latency waits, while the phase change
improves latency placement. NCU replay does not reproduce the phase-only
timing result--it records more traffic and a 0.236% slowdown for `12/2` versus
`0/2`--so the randomized unprofiled A/B is the evidence for that smaller
incremental gain.

The causal result has a different mechanism. Against SDPA, the final kernel
executes 2.32% fewer instructions and uses 68,608 rather than 232,448 bytes of
dynamic shared memory. In two NCU collections it also has about 93% fewer L2
read-miss sectors and off-chip read bytes; the full collection reports 54.9
GB versus 796.0 GB of DRAM reads and 3.35B versus 48.88B L2 read misses. Its
260.9B L2 read-hit sectors versus SDPA's 122.2B show that more of its repeated
input traffic is served on chip. This locality and lower instruction count
explain the unprofiled +0.509% paired gain despite lower issue utilization
(54.05% versus 56.00%), eligible warps (0.641 versus 0.774), and much higher
barrier stalls (1.549 versus 0.106). NCU itself reverses the timing by 0.667%
under multi-pass replay, so its counters are used for attribution; the
randomized raw-event study supplies the timing claim.

Commit `a657aa4b` repairs the inaccurate degree-one route and the persistent
statistics handoff. Commit `d227cd7e` promotes the validated `12/2` dense and
`14/12` causal phases into the exact-shape full-effort seeds and tests that
they survive normalization in the initial population. This guarantees that a
full search evaluates them; it does not claim that a full search produced the
historical bars. Recreate the explicit fixed configs to repeat that study; the
current paired protocol and raw artifact apply to the new full-search winners.

Every retained KernelAgent manifest binds the full shape and dtype, physical
GPU, 750 W power cap, seed, display metadata, and SHA256 of any selected
source. The benchmark harness rejects missing or mismatched provenance before
loading generated code and applies the same full-output, exact-repeat, and
stress checks to public and closed results.

## KernelAgent Public environment and tuning

KernelAgent Public was checked out at
`e0647170da36ef9b059ac0bd3d60103aa4ed378b` (`pyproject.toml` version 0.1.0).
The plot label uses `v2+e0647170` to identify this development snapshot. The
generated kernel language is Triton; KernelAgent Public has no
selectable CuTe backend. The version line identifies Triton without repeating
the backend in the KernelAgent name.
The complete local compatibility patch and launch sources are retained under
`plots/kernelagent/setup/`. Machine-specific launcher paths were normalized in
the checked-out reproduction copy; the byte-identical measured launcher and
its manifest-recorded hash remain available in the measured source bundle.

The retained launchers discover the Helion checkout relative to their own
location. Set `HELION_REPO_ROOT` only to override that checkout. The public
sandbox maps one Python environment read-only at `/env`; either activate its
Python 3.12 virtual environment or set `KERNELAGENT_ENV_ROOT` to the environment
root explicitly. The final-measurement launcher also accepts
`KERNELAGENT_RESULTS_ROOT` and `KERNELAGENT_FINAL_ROOT` for its input and output
trees:

```bash
export KERNELAGENT_ENV_ROOT=/path/to/python-environment
# Optional when running a copied launcher outside this checkout:
export HELION_REPO_ROOT=/path/to/helion
```

Each shape and time allocation was a fresh run. The only reference implementation in
the task was an output-only PyTorch FP16 SDPA module forced to
`SDPBackend.CUDNN_ATTENTION`. KernelAgent generated one initial Triton seed,
with up to three correctness-guided refinements, then ran its hardware-guided
optimizer with four workers, beam width two, and two bottleneck directions.
RAG was disabled. No kernel, cache, prompt history, or model session was
transferred between shapes or time multipliers; the 2x and 10x runs started
from scratch rather than resuming a shorter run.

Claude Code requested the canonical `claude-opus-5` model at high reasoning
effort with tool use, MCP servers, session persistence, and web search all
disabled. The model necessarily used its provider transport, but had no
browsing or shell interface. During tuning, generated-code correctness tests,
candidate benchmarks, and NCU ran in a separate user/mount/network/PID
namespace with no network. That namespace exposed only the selected physical
GPU, the isolated run directory, the Python environment, and the KernelAgent
checkout. The Helion and FA4 source trees were not mounted.

The total wall-clock deadline included seed generation, compilation,
correctness, NCU, model calls, and benchmarking. The budget table was frozen
from the original Helion+CuTe full-autotune campaign associated with Helion
commit `016ad645`, before the later CuTe 4.6.1 and generalized-search refreshes.
The 1x values are those historical elapsed times; 2x and 10x are exact
multiples. They do not describe the duration of the newer `d9cae807`
pre-rebase or historical `edd79764` rebased strict verification searches:

| Variant | Sequence | 1x seconds | 2x seconds | 10x seconds |
| --- | ---: | ---: | ---: | ---: |
| Dense | 32K | 708.6 | 1417.2 | 7086.0 |
| Dense | 64K | 840.9 | 1681.8 | 8409.0 |
| Dense | 128K | 1386.7 | 2773.4 | 13867.0 |
| Dense | 256K | 3682.7 | 7365.4 | 36827.0 |
| Causal | 64K | 3732.2 | 7464.4 | 37322.0 |
| Causal | 128K | 3374.6 | 6749.2 | 33746.0 |
| Causal | 256K | 5619.4 | 11238.8 | 56194.0 |
| Causal | 512K | 2413.9 | 4827.8 | 24139.0 |

KernelAgent candidate timing used Triton's `do_bench` with a 1000 ms warmup
window and 500 ms measurement window. NCU used launch skip three and launch
count one. The finalized runner allowed 120 seconds per seed correctness
attempt, longer than KernelAgent's upstream 30-second default to cover
compilation of these unusually large attention kernels. Four completed 1x
runs (dense 64K/128K/256K and causal 64K) used an earlier 900-second cap, but
each accepted seed actually passed in less than 120 seconds. Both exact runner
versions are retained in `plots/kernelagent/setup/`. Every selected kernel
passed a full-output comparison against cuDNN SDPA at `atol=5e-2` and
`rtol=2e-2` during tuning and again in the final nine-sample harness.

The published KernelAgent results used Claude Opus 4.5, four workers, eight
rounds, roughly one hour per KernelBench task, and an H100. This comparison
instead uses Claude Opus 5.0, B200 at 750 W, long fixed-shape attention, and
Helion-matched wall-clock budgets. The stricter no-tool/no-RAG isolation and
full-output long-shape correctness checks are also deliberate deviations.
The completed-call audit for the 24 published searches contains 3,443 Claude
calls, 103,197,222 total recorded tokens, zero web-search requests, and
$1,719.81 in CLI-reported model cost. The token total comprises 7,048 ordinary
input, 44,672,900 cache-creation input, 912,654 cache-read input, and
57,604,620 output tokens. Calls interrupted by a hard run deadline emit no
usage record, so tokens and cost are reproducible lower bounds rather than
exact billing totals. The full per-shape data and 1x/2x/10x/all summary rows
are in `plots/kernelagent_tuning_usage.csv`. Plot captions show the rounded
published completed-call totals: `$107 tokens` for 1x, `$245 tokens` for 2x,
and `$1,368 tokens` for 10x.

These published totals exclude discarded infrastructure-failed attempts. The
first causal-256K 10x attempt recorded 409 completed calls, 12,320,083 tokens,
and $210.48 before an expired local X.509 credential caused the optimizer to
terminate after 36,034 of 56,194 allocated seconds. The first causal-512K 10x
attempt failed before its first completed model call for the same reason. Both
shape directories were moved aside and restarted from empty directories; the
published replacements used their full allocations and passed the independent
correctness harness.

Across the eight final measurements, KernelAgent Public 1x averages 524.7 TFLOP/s
(521.6 geometric mean, 70.7% of the refreshed SDPA by geometric mean).
KernelAgent Public 2x
averages 572.5 TFLOP/s (568.5 geometric mean, 77.1% of SDPA). Because every 2x
search was fresh, not resumed, it beat the corresponding 1x result on five of
eight shapes rather than monotonically improving every shape. KernelAgent Public 10x
averages 640.5 TFLOP/s (635.7 geometric mean, 86.2% of SDPA) and beats the 2x
result on six of eight shapes. These percentages mix protocols: the KernelAgent
rows keep their historical mean-statistic values while the refreshed SDPA rows
use the median statistic.

## KernelAgent Closed environment and tuning

The KernelAgent Closed baseline uses the complete Oink bundle labeled
`v3-20260730`. The original archive and executable are identified by these
hashes; the binary's own `info` command reports package version `0.0.0`, so the
date label is required to distinguish this rewrite:

```text
kernelagent-oink-linux-x86_64-20260730.tgz
sha256: 258853a7f19046f0cd3a3cc74564514f2756d24b54b1fbd7efa60f38d9a26e0b
kernelagent-bin
sha256: 2859181c5786a782d8946032967a5ee70a9bf191e272137c97703a12550330c0
```

The model was `gpt-5.6-sol-ptu` at `xhigh` reasoning through the bundle's
`openai_responses` provider. The generated language was CuTeDSL, and every
search selected its source with CUTLASS DSL 4.5.1. Each published shape started
with an empty workspace, model session, cache, and candidate history; no kernel
or conversation was transferred between shapes.

For the CuTe 4.6.1 refresh, each selected source was loaded from scratch,
recompiled, and checked against cuDNN on both full-output input distributions.
Passing sources were then remeasured with the canonical nine-sample protocol.
This evaluation used no new model calls or tuning allocation. Passing plot
entries identify CuTe 4.6.1;
their JSON config records both `selection_cute_version=4.5.1` and
`evaluation_cute_version=4.6.1`. Five failed campaigns selected no source and
therefore could not be recompiled; those rows retain their selection-only 4.5.1
provenance and have no throughput bar. Separately, the dense 256K 2x source
passed the standard check but failed the independent stress check after
recompilation with 4.6.1, while the causal 128K 2x source failed the standard
check. Those rows also have no throughput bars. The plot legend derives
versions from passing measurements, so the failures do not create a second
displayed CuTe version.

The model had no built-in tools. A controlled MCP exposed campaign status,
read-only references, a correctness-only `test_candidate` action, authoritative
candidate submission, the current best source, and an NCU action. It had no
shell, arbitrary file access, web access, or subagents. The task required the
model to iterate with `test_candidate` until both distributions passed, then
submit that identical source. Submission independently repeated both checks,
so the model could not bypass the correctness gate.
The only attention specification was the output-only PyTorch FP16 SDPA call
forced to cuDNN by the evaluator. As approved for this run, the MCP also
exposed the binary's bundled B200 FlashKDA and PTX files. Two generic Blackwell
FP16 GEMM tutorials from CUTLASS commit
`da5e086dab31d63815acafdac9a9c5893b1c69e2` documented CuTeDSL tensor-core,
TMA, pipeline, and launch syntax. No attention implementation, Helion source,
FA4 source, prior selected kernel, or benchmark payload was exposed.

Candidate processes ran in a separate user, mount, PID, and network namespace.
The jail mounted only the selected GPU, Python environment, run directory, and
fixed evaluator sources. Static validation prohibited Triton, Helion,
FlashAttention, PyTorch attention/compilation, external extensions, networking,
and arbitrary file or process APIs. It also required the launched attention
kernel itself to contain a CuTe tensor-core MMA operation. A timed-out process
group was terminated in full so an orphaned CUDA process could not retain the
GPU or lock.

Every accepted candidate passed two full-output comparisons to cuDNN SDPA at
`atol=5e-2`, `rtol=2e-2` before search timing. The standard case used normal
Q/K/V with seed `2026080106`. The independent stress case used seed
`2026080107`, multiplied Q and K by 2, and added a random per-batch/head/feature
offset to V. This prevents long dense zero-mean attention from accepting a
degenerate zero output. The comparison visits every output element in bounded
sequence-row chunks; this preserves the full-output tolerance test without a
95 GB FP32 temporary allocation at causal 512K. Search timing still used the
standard case, with one CUDA-event median, 200 ms warmup, and a 200 ms
measurement window after cache and clock warmup.

A shape-dependent minimum throughput rejected candidates too slow to complete
the final nine-run measurement. After the fixed model deadline, the selected
source was loaded in a new network-isolated process, checked on both cases,
thermally warmed for 10 seconds, and measured nine times with the report's
canonical 1000/500 ms windows. Each passing source was then checked on both
cases once more in a fresh sandbox. Finalization uses shape-aware process limits
of 900, 1800, and 3600 seconds and records selection metadata before launching;
an interrupted or timed-out final measurement can therefore resume without
rerunning the model or extending its tuning allocation. Final measurement and
post-hoc validation time were outside the model's tuning allocation, matching
the KernelAgent Public accounting.

The original single-case harness exposed a real validity hole at dense 256K:
a zero output passed because the true output was close to zero at the absolute
tolerance. The stress case rejects that source on 96.1% of elements, with a
7.34 maximum absolute error. All pre-fix completed selections were quarantined;
dense 32K/64K/128K and causal 64K were rerun from empty histories for a full
1x allocation. The clean dense 256K replacement was also fresh. Discarded
harness-debug campaigns are not included in the published time or token totals.
The causal 128K campaign was already running when the gate changed; all of its
earlier submissions had failed the standard case, its first accepted candidate
passed both cases, and its original deadline was not extended.

Both final KernelAgent Closed baselines are incomplete:

| Variant | Sequence | 1x result | 1x TFLOP/s | 1x of SDPA | 2x result | 2x TFLOP/s | 2x of SDPA |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| Dense | 32K | FAIL | - | - | FAIL | - | - |
| Dense | 64K | FAIL | - | - | FAIL | - | - |
| Dense | 128K | PASS | 31.5 | 4.2% | PASS | 64.1 | 8.6% |
| Dense | 256K | PASS | 405.9 | 54.6% | FAIL | - | - |
| Causal | 64K | PASS | 126.7 | 17.4% | PASS | 79.7 | 10.9% |
| Causal | 128K | PASS | 50.8 | 6.9% | FAIL | - | - |
| Causal | 256K | PASS | 97.5 | 13.1% | PASS | 98.1 | 13.2% |
| Causal | 512K | PASS | 209.4 | 29.1% | FAIL | - | - |

`FAIL` means either no source was selected before the campaign deadline or the
selected source failed the CuTe 4.6.1 final-harness checks; no timing is
reported in either case. The summary chart labels 1x
`INCOMPLETE (6/8)` and 2x `INCOMPLETE (3/8)` rather than computing a misleading
geometric mean over only their successful shapes. Each 2x campaign was a fresh
run, not a continuation of 1x, so the additional time does not imply monotonic
per-shape performance.

The 1x campaigns consumed 21,759.0 allocated tuning seconds (6.04 aggregate
GPU-hours), 593 recorded model calls, and 137,792,292 tokens: 3,491,028 ordinary
input, 133,289,472 cache-read input, and 1,011,792 output. The 2x campaigns
consumed 43,518.0 seconds (12.09 aggregate GPU-hours), 1,294 calls, and
476,155,350 tokens: 7,859,825 ordinary input, 466,189,824 cache-read input, and
2,105,701 output. Combined, that is 65,277.0 allocated seconds, 1,887 calls,
and 613,947,642 tokens. Cache-creation input was zero. A deadline-killed model
call may have no usage record, so token and call counts are reproducible lower
bounds. Neither the GPT endpoint nor the closed binary reported a dollar cost.

For comparison, the report estimates a direct-OpenAI Standard list-price
equivalent using the GPT-5.6 Sol prices published on 2026-08-08: per million
short-context tokens, `$5.00` ordinary input, `$0.50` cached input, `$6.25`
cache writes, and `$30.00` output; the corresponding long-context prices are
`$10.00`, `$1.00`, `$12.50`, and `$60.00`. Each completed call is classified
from its recorded `context_tokens`, using 272K tokens as the short/long
boundary. This gives **$166.90 for 1x**, **$564.33 for 2x**, and **$731.23
combined**. The 1x total contains 390 short- and 203 long-context calls; 2x
contains 638 short- and 656 long-context calls. These are public-list token
estimates, not PTU invoices, and exclude any deadline-killed call lacking a
usage record. Current prices are at <https://platform.openai.com/docs/pricing>.

During causal 256K 2x, the temporary extracted executable was removed after the
model had selected a valid candidate. The executable was restored from the
original archive and its SHA256 rechecked; finalization resumed from the
recorded source without another model call or added tuning time. The manifest
records this recovery. Launchers now recover whenever campaign state contains
a selected candidate, even if failure occurred before `selected_kernel.py.txt`
was copied.

The reproduction harness is under `plots/kernelagent_closed/setup/`. Its
supervisor and MCP launch paths include post-run interruption-safety hardening;
these changes do not affect normal completed-run prompts, budgets, validation,
timing, or selected-kernel execution, but the files are not claimed to be
byte-identical to the supervisor used for the retained measurements. The run
manifests remain the authoritative evidence for measured outputs and recorded
binary, model, and reference hashes. To reconstruct the read-only reference
directory, extract the archive, start `kernelagent-bin` once with `TMPDIR`
pointing at an empty directory, and copy
`kernelagent/skills/cutedsl-kernel-expert/SKILL.md`, its `references/kda-b200`
directory, and `references/ptx_isa_9.2.agent.md` from the live `_MEI*` runtime
directory. Add `fp16_gemm_0.py` and `fp16_gemm_1.py` from the CUTLASS commit
above under `cutlass-4.5.1/blackwell/tutorial_gemm/`. Every exposed file hash
is recorded independently in each run manifest.

With the bundle and reference paths prepared, launch the fixed 1x streams:

```bash
export KERNELAGENT_PYTHON=/path/to/python-environment/bin/python
export KERNELAGENT_ENV_ROOT=/path/to/python-environment
export KERNELAGENT_ARCHIVE=/path/to/kernelagent-oink-linux-x86_64-20260730.tgz
export KERNELAGENT_BUNDLE_ROOT=/path/to/extracted/kernelagent_runner
export KERNELAGENT_REFERENCE_ROOT=/path/to/controlled-references
export KERNELAGENT_RUN_ROOT=/tmp/kernelagent-closed-v3-1x
export KERNELAGENT_OPENAI_BASE_URL=https://provider.example/openai/v1
export KERNELAGENT_OPENAI_CA_CERTIFICATE=/path/to/provider-ca.pem
export KERNELAGENT_OPENAI_NO_PROXY=127.0.0.1,localhost
export CODEX_CLIENT_CERT=/path/to/client-certificate.pem
export CODEX_CLIENT_KEY=/path/to/client-private-key.pem
"$ARTIFACT_REPO/plots/kernelagent_closed/setup/run_1x.sh"
```

`KERNELAGENT_PYTHON` defaults to the active virtual environment's Python, then
to `python3`; `KERNELAGENT_ENV_ROOT` defaults to the selected Python's
environment root. `HELION_REPO_ROOT` can override the script-relative checkout.
The base URL and CA variables are required because the original internal
endpoint and machine-specific CA path are intentionally not retained. The
no-proxy setting is optional and defaults to `127.0.0.1,localhost`. The client
certificate and key may instead be supplied through
`THRIFT_TLS_CL_CERT_PATH` and `THRIFT_TLS_CL_KEY_PATH`.

Then launch fresh 2x streams in the same root:

```bash
KERNELAGENT_SKIP_1X=1 \
  "$ARTIFACT_REPO/plots/kernelagent_closed/setup/run_failed_1x_then_2x.sh"
```

The launchers record a deadline-exhausted shape as a failed result. Validate
all selected sources and publish both allocations with:

```bash
python "$ARTIFACT_REPO/plots/kernelagent_closed/setup/validate_selected.py.txt" \
  --run-root "$KERNELAGENT_RUN_ROOT" --budget-label 1x
python "$ARTIFACT_REPO/plots/kernelagent_closed/setup/validate_selected.py.txt" \
  --run-root "$KERNELAGENT_RUN_ROOT" --budget-label 2x
python "$ARTIFACT_REPO/plots/kernelagent_closed/setup/publish_results.py.txt" \
  --run-root "$KERNELAGENT_RUN_ROOT" \
  --output-root "$ARTIFACT_REPO/plots/kernelagent_closed" \
  --payload-root "$ARTIFACT_REPO/plots/kernelagent/results/payloads"
```

The launchers run dense shapes serially on physical GPU 7 and causal shapes
serially on physical GPU 6, while the two streams overlap. They set both GPUs to
750 W and restore 850 W after both streams finish. The 1x seconds in the
budget table above are exact hard deadlines; 2x uses exact doubles. Both include
model calls, reference reads, compilation, correctness, profiling, and
candidate timing. Selected sources, manifests, and final result JSON are under
`plots/kernelagent_closed/`; raw per-run model tokens and times are in
`plots/kernelagent_closed_tuning_usage.csv`. The GPT provider and closed binary
did not report actual dollar cost. The chart marks these dollar values as
token costs and rounds them to `$167 tokens` for 1x and `$564 tokens` for 2x;
the detailed values above remain the reproducible public-list estimates.

## KernelAgent Public 10x run

`plots/kernelagent/setup/run_kernelagent_10x.sh` launched fresh 10x searches
for all eight shapes. Dense runs execute sequentially on physical GPU 7 while
causal runs execute sequentially on physical GPU 6; the two streams run in
parallel. It uses the same Claude Opus 5.0 adapter, isolation, correctness,
profiling, and fixed-shape setup described above. The script verifies idle
GPUs and the pinned KernelAgent checkout, applies the 750 W power cap, writes
under `/tmp/kernelagent-runs/attention_opus5_e0647170_10x`, and restores the
previous power limits on exit.

The dense stream allocates 66,189 seconds and the causal stream 151,401
seconds. Since they overlap, the expected critical path is about 42.1 hours
plus setup overhead. To launch it independently of an interactive session:

```bash
nohup bash "$ARTIFACT_REPO/plots/kernelagent/setup/run_kernelagent_10x.sh" \
  >/tmp/kernelagent-10x-launch.log 2>&1 </dev/null &
echo $!
```

Progress is recorded in `queue_dense.log` and `queue_causal.log` under the run
root. A repeated invocation skips validated completed shapes and fails closed
if it finds a partial shape directory; KernelAgent cannot safely resume a
partially completed search. The final published searches consumed 217,592.5
aggregate seconds across shapes, versus 217,590.0 allocated seconds including
supervisor overhead.

## TileIR environment

TileIR replaces the normal Triton package, so run it in a separate Python
environment. The exact wheel used was the CPython 3.12 Linux x86-64 wheel from
the Triton-to-Tile-IR `cuda_tile_v14.3.0` release:

```text
nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl
sha256: 79f11643cb83107df76a9177d9b873f540a603443ea7560b73e0b7df93069dc8
```

The wheel requires glibc 2.38 or newer. The canonical run used an isolated
Ubuntu 24.04 container because the host had glibc 2.34. A native environment
with a sufficiently new glibc is equivalent. Install nvtriton without replacing
the standard benchmark environment:

```bash
python -m pip install --no-deps \
  /path/to/nvtriton-3.6.0-cp312-cp312-linux_x86_64.whl
```

The wheel is available from the
[`cuda_tile_v14.3.0` release](https://github.com/triton-lang/Triton-to-tile-IR/releases/tag/cuda_tile_v14.3.0).
When using a container, expose the NVIDIA control/UVM and GPU device nodes and
preserve physical ordinals; the harness invokes `nvidia-smi -i $GPU` to verify
the power limit. Verify that importing `triton` resolves to this environment,
then run with:

```bash
export ENABLE_TILE=1
export HELION_BACKEND=tileir
export TILEIR_ENABLE_APPROX=1
export TILEIR_ENABLE_FTZ=1
export HELION_BENCHMARK_HELION_VERSION=1.4.0.dev38+g016ad645
```

Use the canonical invocation above with `IMPL=helion-tileir`. Decode
`helion_tileir_helion_overrides` from each CSV row using the same translation
table. Those cells record full search effort, random seed, compile/benchmark
timeouts, precompile mode, job count, and transferred seed config.

TileIR-specific exceptions are part of the recorded protocol:

- Dense 32K completed a 501-candidate full search. External process cleanup
  interrupted its original final launch, so the selected config was rerun with
  `HELION_AUTOTUNE_EFFORT=none` for correctness and nine-sample timing. The CSV
  records both the fixed config and the completed-search metadata.
- Causal 64K used parent-process autotune timing.
- Causal 512K used four precompile jobs, a 120-second compile timeout, and
  parent-process autotune timing. Internal per-candidate accuracy checking was
  disabled with `HELION_AUTOTUNE_ACCURACY_CHECK=0` because the spawned-worker
  path could not retain the isolated TileIR runtime at this size. The final
  selected kernel still passed the harness correctness comparison against SDPA
  before timing.

These are autotuning-process exceptions, not waived final correctness checks.

## TLX environment

TLX replaces the normal Triton package, so it was built and installed into an
isolated target directory. The baseline uses Meta's integrated TLX fork rather
than the standalone `triton-utlx` plugin: the published plugin wheel is
ABI-coupled to a development Triton build and its Python layer expects frontend
dispatch hooks that are absent from stock Triton 3.7. The integrated fork
contains those hooks and the tested Blackwell attention tutorial in one pinned
source tree. It does not expose a separate TLX release number; `3.7.4+fb` is
the Meta Triton runtime version and is therefore the version shown in the plot.

Build the exact CPython 3.12 wheel and install it without changing Helion's
normal environment:

```bash
git clone https://github.com/facebookexperimental/triton.git tlx-triton-repo
git -C tlx-triton-repo worktree add --detach "$PWD/tlx-triton" \
  7632ef23c8f4667d659e19820d39d80f613fbed1

mkdir -p tlx-wheels tlx-runtime
CC=clang CXX=clang++ MAX_JOBS=32 \
  python -m pip wheel --no-deps --wheel-dir tlx-wheels ./tlx-triton
python -m pip install --no-deps --target "$PWD/tlx-runtime" \
  tlx-wheels/triton-3.7.4-cp312-cp312-linux_x86_64.whl
```

The detached worktree is intentional: this fork's build script otherwise adds
a Git suffix to the package metadata when the source root contains a `.git`
directory. The worktree reproduces the measured package version `3.7.4` while
remaining pinned to the recorded commit.

The build used Clang 22.1.3. Its downloaded LLVM tools require
`GLIBCXX_3.4.30`; on hosts with an older system C++ runtime, prepend a newer
GCC runtime directory to `LD_LIBRARY_PATH` for the wheel build. The resulting
wheel used here has SHA256
`83d2db6a41ec78d9b4faf3bac5e7d68248e497c1db43d884bda6bba8a3503806`.
Verify the isolated install before benchmarking:

```bash
PYTHONPATH="$PWD/tlx-runtime" python - <<'PY'
import triton
import triton.language.extra.tlx as tlx

print(triton.__version__)  # 3.7.4+fb
print(tlx.__file__)
PY
```

Select this runtime only for the TLX process:

```bash
export TLX_RUNTIME=/path/to/tlx-runtime
export HELION_TLX_RUNTIME_ROOT="$TLX_RUNTIME"
export HELION_TLX_REVISION=7632ef23c8f4667d659e19820d39d80f613fbed1
```

For a direct `--impl tlx` invocation, prefix the canonical command with
`PYTHONPATH="$TLX_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"`. Do not export that
`PYTHONPATH` while running another backend. When TLX is selected through the
harness's `--impl all` mode, the harness prepends the isolated runtime only to
the TLX child process.

The harness loads
`triton.language.extra.tlx.tutorials.blackwell_fa_ws_pipelined_persistent`,
checks its full output against cuDNN SDPA, and records the exact selected
configuration. TLX's upstream autotuner prunes to six configurations for
`D=64`: KV depth 6 and the causal-dependent N grouping are fixed, while three
rescaling variants and the warp-barrier toggle are measured. Each shape ran in
a fresh process, dense on physical GPU 7 and causal on physical GPU 6.

All eight TLX outputs passed the full correctness comparison. TLX reached a
650.8 TFLOP/s overall geometric mean, 88.2% of the refreshed SDPA. Its dense
geometric mean was 680.8 TFLOP/s (91.5% of SDPA), and its causal geometric mean
was 622.1 TFLOP/s (85.0% of SDPA). Per-shape values are in the CSV; the TLX
rows keep their historical mean-statistic values.

## Rebuild the reports

Single-implementation JSON files must first be combined into one payload per
shape. This script consumes the filenames produced by the canonical invocation
and checks that all fourteen implementations describe the same shape:

```bash
python - <<'PY'
import json
import os
from pathlib import Path

work = Path(os.environ["WORK"])
implementations = (
    "sdpa",
    "fa4",
    "gluon",
    "tlx",
    "flexattention",
    "flexattention-cute",
    "helion-triton",
    "helion-cute",
    "helion-tileir",
    "kernelagent-1x",
    "kernelagent-2x",
    "kernelagent-10x",
    "kernelagent-closed-1x",
    "kernelagent-closed-2x",
)
shapes = (
    ("dense", 0, 32768),
    ("dense", 0, 65536),
    ("dense", 0, 131072),
    ("dense", 0, 262144),
    ("causal", 1, 65536),
    ("causal", 1, 131072),
    ("causal", 1, 262144),
    ("causal", 1, 524288),
)
for variant, causal, seq_len in shapes:
    results = [
        json.loads(
            (work / "results" / f"{causal}_{seq_len}_{impl}.json").read_text()
        )
        for impl in implementations
    ]
    shape = results[0]["shape"]
    assert all(result["shape"] == shape for result in results)
    payload = {"shape": shape, "results": results}
    output = work / "payloads" / f"{variant}_{seq_len}.json"
    output.write_text(json.dumps(payload) + "\n")
PY
```

Keep payloads in this order, matching the CSV and chart. The historical
all-backend payloads remain under `plots/kernelagent/results/payloads/` and are
inputs to the CuTe 4.7 baseline refresh documented under
`plots/cute470_baseline_refresh/setup/`. After that refresh and a completed
generalized campaign, set `CUTE470_BASELINE_ROOT` to the refresh output root.
The strict publisher deep-copies
`$CUTE470_BASELINE_ROOT/published/payloads`, replaces only the `helion-cute`
and `sdpa` entries, and writes:

```text
dense_32768.json
dense_65536.json
dense_131072.json
dense_262144.json
causal_65536.json
causal_131072.json
causal_262144.json
causal_524288.json
```

Those generalized payloads are intentionally absent until all eight strict
searches and both paired campaigns pass. The tracked chart currently comes from
the `7789ca326` strict-mode rows in the CSV; their raw timing and search
ledgers are archived under `mainrefresh_20260827/`. Do not run the generalized
publisher against a partial campaign or relabel the existing chart with
`c3e36b65d`.

To rebuild the CSV and both charts from a complete payload directory without
rerunning kernels:

```bash
PAYLOADS=(
  "$WORK/payloads/dense_32768.json"
  "$WORK/payloads/dense_65536.json"
  "$WORK/payloads/dense_131072.json"
  "$WORK/payloads/dense_262144.json"
  "$WORK/payloads/causal_65536.json"
  "$WORK/payloads/causal_131072.json"
  "$WORK/payloads/causal_262144.json"
  "$WORK/payloads/causal_524288.json"
)
python "$REPO/benchmarks/cute/compare_attention_backends.py" \
  --merge-json "${PAYLOADS[@]}" \
  --output "$WORK/attention_backends_b200_750w.md" \
  --csv-output "$WORK/attention_backends_b200_750w.csv" \
  --plot-output "$WORK/attention_backends_b200_750w.png" \
  --summary-plot-output "$WORK/attention_backends_b200_750w_geomean.png" \
  --plot-impl-label 'kernelagent-1x=KernelAgent Public (1x Helion tuning time; $107 tokens)' \
  --plot-impl-label 'kernelagent-2x=KernelAgent Public (2x Helion tuning time; $245 tokens)' \
  --plot-impl-label 'kernelagent-10x=KernelAgent Public (10x Helion tuning time; $1,368 tokens)' \
  --plot-impl-label 'kernelagent-closed-1x=KernelAgent Closed (1x Helion tuning time; $167 tokens)' \
  --plot-impl-label 'kernelagent-closed-2x=KernelAgent Closed (2x Helion tuning time; $564 tokens)'
```

Absolute TFLOP/s can move with driver, firmware, clocks, thermals, or compiler
changes. A valid comparison must keep the versions, 750 W limit, GPU assignment,
timing protocol, and full-autotune settings fixed. The CSV contains enough
metadata to audit those conditions and to distinguish an autotuner change from
a kernel-performance change.
