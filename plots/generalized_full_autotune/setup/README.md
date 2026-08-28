# Generalized paired Helion+CuTe vs cuDNN SDPA validation

These scripts run and audit the strict full-autotune attention benchmark, then
measure the eight selected Helion+CuTe kernels against cuDNN SDPA. Artifact
roots are supplied at runtime; no script embeds a winner config or source hash.

## What is validated

The strict manifest builder recursively discovers exactly one Helion+CuTe result
for each of the four dense and four causal shapes. It rejects a dirty or
different Helion revision, mixed versions or input seeds, fixed configs, seed
configs supplied by users or the CLI, time or generation budgets, config-cache
reads, noncanonical compiler heuristics, CuTe value priors, incomplete marginal
coverage of reachable structural field values, or, for the current eight
non-enumerated searches, fewer than 100 successful measurements. It requires
the exact ordered canonical CuTe-flash compiler seed policy and proves each
effective seed succeeded or source-deduplicated in generation zero. It also
rejects a non-LFBO search, an unmeasured winner, an inconsistent
effective-source ledger, or anything other than 64 passing final repeatability
and correctness launches. The selected config is read from
`helion_overrides.autotune_provenance.selected_config`. The current schema's
source digest is `selected_source_sha256`; `selected_source_hash` is accepted as
a compatibility spelling only when it agrees with the winning trial.

The manifest binds all three autotuner sidecars. `autotune.csv` and
`autotune.sources.csv` must have identical lifecycle join fields row-for-row;
the single `autotune.meta.jsonl` record must recompute the same content-addressed
run ID, contain every canonical config ID, and map the selected config back to
the measured winning source. Paths and SHA256 digests for all three sidecars are
recorded in each manifest row.
Before the in-process strict audit starts, the benchmark atomically writes
`result.strict-prevalidation.json` beside the requested result. This diagnostic
preserves the completed search provenance if validation fails, but it is never
accepted as a benchmark result and cannot replace `result.json`.
Manifest and recombination outputs are resolved before writing and cannot alias
any result, sidecar, worker payload, generated source, or harness input that the
same command validates.

Coverage is deliberately not a raw Cartesian enumeration. The tuner constructs
a compact normalized design spanning every live structural value, interaction,
ordinary protocol leaf, and compound packet leaf. Generation zero injects the
policy-sized prefix of that design, then fills its 100-candidate population
without a measured winner. The policy derives its budget from the live design
and requested population: it injects the complete compact design when it fits,
even when that uses more than half of the population, and otherwise preserves
the bounded structural prefix required by the production policy. Complete
structural reachability does not depend on a fixed row count. A separate
low-confound pass benchmarks the neutral
normalized anchor for every legal
pipeline-family, noncompound exp2-packet, and softmax-protocol combination.
Pipeline qualification then establishes each active K/V- and S-stage lane; CLC
families qualify every legal `B*H` divisor and the planned depth/divisor matrix.
The normalized prefix contains two witnesses for every nonsingleton ordinary
leaf and one reachability witness for each compound packet; measured compound
transfers provide the compound candidates used for ranking.
LFBO ranks parent families by successful ordinary schedules so a family with
more packet variants receives no promotion advantage.

If more than four parent families remain, full effort gives every family one
measured constrained probe generation before ranking the four parents that
continue through the remaining generations. Every compound leaf and one
unrestricted global path are retained independently of that parent cap.

Continuation capacity is computed from the live leaf catalog, not a fixed path
count. It reserves the unrestricted winner, every ordinary protocol that could
belong to the promoted-family set, a secondary candidate per promoted family
when configured, and every compound leaf. Constrained paths retain their
measured family, packet, and protocol while tuning remaining fields. The
unrestricted path disables patience-based early stopping; constrained diversity
paths retain ordinary early stopping. The strict run records the complete LFBO
generation budget. This makes every structural family reachable and prevents a
competitive compound schedule from being pruned because a sibling won its
parent family, without claiming that a finite stochastic search proves the
global optimum.

For the current eight non-enumerated searches, the artifact audit requires
exactly 100 distinct attempted generation-zero configs and proves that every
injected design row was among them. It then independently reconstructs and
checks the low-confound anchors and every qualification pass. A generation-zero
probe may time out in one context without invalidating the run, but every live
ordinary/compound leaf and required lane or divisor must obtain the successful
measured evidence or bounded repair required by v22 before promotion.

Each GPU worker then binds the current `examples.attention.attention_output` or
`causal_attention_output`, supplies the selected config to `to_triton_code`, and
requires the regenerated source SHA256 to equal the measured winner before
archiving it. It then calls `compile_config` and independently requires the
source hash attached to that exact compiled callable to match before using the
callable. Thus the paired run compiles current source with the exact measured
config; it neither imports saved generated Python nor autotunes again.

## Measurement protocol

- Dense shapes run sequentially on physical GPU 7 while causal shapes run
  sequentially on physical GPU 6. The two lanes run concurrently.
- Every shape and campaign runs in a fresh process with the expected GPU UUID as
  its sole `CUDA_VISIBLE_DEVICES` entry.
- Each worker's `PYTHONPATH` is exactly the detached measured checkout; ambient
  import paths are removed rather than appended.
- Compiler autotuner heuristics remain enabled while reconstructing the source,
  matching the strict search environment and allowing the worker to replay the
  authenticated generation-zero population, although no new autotuning occurs.
- Each worker uses a fresh per-shape, per-campaign `HELION_CACHE_DIR`, so the
  compiled kernel cannot be inherited from an unrelated local cache.
- The default is two independent campaign seeds and 12 balanced randomized
  Helion/cuDNN pairs per shape per campaign, for 24 raw pairs per shape. The
  defaults are the prior protocol's `2026081101` and `2026081102` seeds.
- Inputs are `2x32xSx64` float16. cuDNN SDPA is forced explicitly.
- Before timing, the complete output is checked at `atol=0.05`, `rtol=0.02`,
  finiteness is checked, and three Helion launches must be bitwise identical.
- After all timing, a peaky-logit input formed by scaling Q and K by 2 is
  checked with tighter aggregate error gates and one exact Helion repeat. This
  stress input is never used for a timing observation.
- Each observation is one call. CUDA-event and synchronized host-wall latency
  are both retained because those timers can disagree for CuTe on Blackwell.
- GPU 6 and 7 must be idle NVIDIA B200s at a measured 750 W power limit.
  Competing compute PIDs are checked at every worker's start and end, not only
  at the initial campaign preflight.

The final `all8_paired_raw.json` retains every raw observation, both campaign
payloads, per-shape pooled summaries, and dense/causal/overall geometric means.
`run_manifest.json` records every subprocess command, environment policy, GPU,
exit status, and output digest.

The paired point estimate is the geometric mean of the per-pair SDPA/Helion
latency ratios. Its confidence interval uses 20,000 bootstrap samples, resampling
within each campaign/shape stratum. The recorded base seed defaults to
`2026081103`; each shape and timer derives and records its own deterministic
seed. This removes the unrecorded-bootstrap-seed caveat in the prior artifact.
The interval is conditional on the two observed campaign strata; calls are
resampled within each fixed campaign rather than treating two campaigns as an
estimate of the population of possible campaigns.

The paired log-ratio and interval are not the plotted bar statistic. The
publisher plots the pooled marginal median TFLOPS from the 24 CUDA-event calls
for each implementation. The paired CUDA-event ratio is the primary relative
estimate, and the synchronized wall measurements provide an independent timer
cross-check.

Publication also validates CUDA-event timing against the synchronized wall
interval from the same calls. A shared-pair stratified bootstrap requires the
relative Helion/SDPA timer bias to stay within 0.5%, while per-implementation
checks reject common-mode event failures and tolerate at most two host-wall
outliers among 24 pairs. A sub-0.5% point estimate is classified as directional
only when both timer intervals, both individual campaigns, and the plotted
marginal-median comparison all agree; otherwise it is reported as inconclusive
without blocking publication of the measurements.

## Strict cold search

Restore the measured commit from the retained complete-history bundle, create
a detached sibling worktree, then launch both lanes from the artifact checkout:

```bash
export GENERALIZED_BUNDLE="$ARTIFACT_REPO/plots/generalized_full_autotune/helion-generalized-c3e36b65.bundle"
test "$(sha256sum "$GENERALIZED_BUNDLE" | cut -d' ' -f1)" = \
  a39cfcd01206c36609a178ee483e40194189a00f878e8d48f68a082aec3cdfad
git -C "$ARTIFACT_REPO" bundle verify "$GENERALIZED_BUNDLE"
git -C "$ARTIFACT_REPO" fetch --no-tags "$GENERALIZED_BUNDLE" \
  refs/archive/helion-generalized-c3e36b65:refs/archive/helion-generalized-c3e36b65
export REPO="$REPRO_ROOT/helion-generalized-c3e36b65d"
git -C "$ARTIFACT_REPO" worktree add --detach "$REPO" \
  refs/archive/helion-generalized-c3e36b65
test "$(git -C "$REPO" rev-parse HEAD)" = \
  c3e36b65d69681c23e053042b0bc21e2331bad17
```

Use `nvidia-cutlass-dsl==4.7.0`. The launchers and validators reject any other
CuTe version before accepting campaign evidence.

The script verifies that commit, a `v1.4.0-157-g<commit-prefix>` Git
description resolving to that commit, a clean tree, idle B200 GPUs 6 and 7 at
750 W, and an absent output root. The hexadecimal abbreviation may vary with
the repository object set. The measured checkout must not retain a local
campaign tag because that changes the reported Helion version. During cold
search it rejects every nonignored
untracked path except the four paired-run files in the launcher's allowlist.
The later paired-validation step stages all six files documented below. Every
ignored file is rejected, including
Python bytecode and tool caches, because cached bytecode can execute instead of
tracked source. `OUTPUT_ROOT` is canonicalized and must resolve outside the
measured checkout; all Python and compiler caches are redirected beneath that
external root. Keep `WORK`, `STRICT_ROOT`, and benchmark output outside the
measured checkout. Dense shapes run serially on physical GPU 7 while causal
shapes run serially on physical GPU 6; the two lanes overlap.

The launcher holds a nonblocking lock beside the output root for its complete
lifetime. If the host or supervisor interrupts a run, resume through the public
entrypoint:

```bash
HELION_REPO_ROOT="$REPO" \
STRICT_PYTHON_EXECUTABLE="$STRICT_PYTHON_EXECUTABLE" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/run_strict_all8.sh" \
  --resume "$STRICT_ROOT"
```

Resume delegates to the immutable launcher snapshot. Shapes with an
authenticated post-check completion marker and unchanged `result.json` are
skipped. A partial shape directory is moved in full to the sibling
`${STRICT_ROOT}.quarantine` tree before that shape restarts, so no compiler or
autotuner cache from the interrupted attempt is reused. A changed completion
marker fails closed.

The launcher resolves one canonical Python executable before either lane
starts, passes that absolute path to every shape, and revalidates it before each
invocation. This prevents an ambient `PATH` change from selecting a different
environment. Set `STRICT_PYTHON_EXECUTABLE` to an executable path or command
name to choose the environment explicitly; otherwise `python` is resolved from
the launcher's initial `PATH`. Each shape's `run.log` records the canonical
path and Python version. The launcher repeats the physical-GPU compute-PID
check immediately before and after every shape, after the foreground process
has exited, so the final shape in each lane receives the same contamination
check as every earlier shape.

Before starting either lane, the launcher copies itself to
`$STRICT_ROOT/launcher/run_strict_all8.sh`, changes the snapshot to read-only
mode `0555`, and writes its SHA-256 beside it. Both lane processes execute that
campaign-local snapshot. Later edits to the launcher in the artifact checkout
therefore cannot change a running campaign or the bytes retained with its
output.

```bash
export STRICT_ROOT="$WORK/strict-all8"
export STRICT_PYTHON_EXECUTABLE="$(python -c \
  'import os, sys; print(os.path.realpath(sys.executable))')"
HELION_REPO_ROOT="$REPO" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/run_strict_all8.sh" \
  "$STRICT_ROOT"
```

For these eight non-enumerated search spaces, the command forces standard
`autotune_effort=full`, an exactly 100-candidate cold initial population,
isolated candidate benchmarking, candidate accuracy checks, one trial, uncapped
LFBO neighborhoods, and no wall-time or generation override. It disables
config-cache reads, user/CLI seeds, value priors, fixed fields, custom tuners,
and flash codegen environment overrides. It enables only the canonical
CuTe-flash compiler seed policy generated from the live legal fragments. The
ordered normalized seed IDs and policy digest are recorded, and every effective
compiler seed must have a successful or source-deduplicated generation-zero
measurement. The only per-shape inputs in the launcher are shape, input seed,
and autotuner RNG seed. The CuTe backend scores tuner candidates with
synchronized host-wall measurements. The explicit benchmark timer also makes
the nine final strict result samples use host-wall timing because CUDA events
can mis-time CuTe kernels on Blackwell. The candidate wall scores select the
winner, and the final strict samples audit and report it; the later plotted
measurements are fresh paired CUDA-event samples, with wall measurements
retained as a cross-check.
The benchmark also requires the search to record the full profile's LFBO
generation count; the unrestricted path cannot stop on patience, while the
structurally constrained paths may. It fails unless the selected effective-source digest
appears in the measured ledger and the final kernel passes 64 repeated
correctness launches.

The metadata sidecar records the distinction between the raw setting and the
effective invocation policy: `settings.force_autotune` must remain `false`,
while `settings.effective_cache_read_bypass` must be `true` because the strict
harness requests a cold search at the call site. Isolated rebenchmark timeouts
are counted separately from compile and worker failures. When a previously
successful generated source times out during isolated confirmation, every
config alias for that source must transition to `timeout` or `error` with both
performances cleared. The source stays quarantined for the rest of the search,
and the manifest reconciles the number of distinct timed-out source hashes with
`num_isolated_rebenchmark_timeouts`.

Validate all eight result and three-sidecar sets and create the deterministic
manifest:

```bash
python "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/build_strict_manifest.py" \
  --artifact-root "$STRICT_ROOT" \
  --output "$WORK/strict_manifest.csv"
```

This strict mode uses the normal canonical CuTe-flash compiler seeds while
remaining colder than an ordinary Helion invocation in every other respect: it
bypasses result caches and rejects user seeds, fixed configs, and active value
priors. The standard tuner uses the same legality-driven search surface and, at
`full` effort, the same structural qualification and unrestricted exploration.
The compiler seeds cover legal schedule families and compound packets; they are
measured starting candidates, not preselected winners. Earlier seed-free runs
remain useful historical diagnostics, but they are not the active publication
protocol and must not be relabeled as canonical-seeded evidence.

The v22 qualification phase is hierarchical rather than two equal searches for
every leaf. It independently rebuilds a neutral schedule anchor for every live
ordinary family, softmax protocol, and noncompound exp2 packet. It then plans
enough pipeline passes to establish a witness for every active K/V or
softmax-stage lane and add one conditional child for each lane whose exact
space is not already exhausted. Each pass partitions the historical budget of
200 raw neighbors and at most four evaluated candidates per leaf across its
active lanes. If a lane has no successful attempt or accuracy failure but has a
retryable compiler or worker failure, v22 permits one additional exact-lane
repair child. CLC conditional-parent records carry the exact share of the
200-neighbor budget assigned by that same allocator.
The repair pass is data-dependent, is included in the recorded pass totals, and
cannot repair an accuracy failure or promote a failed candidate. A repair may
record no generated child only when another job in that same global pass proves
that a tracked retryable failure source-deduplicated successfully. The detached
audit replays every pass's ordered parent jobs, decision-time measurement
snapshots including generated-source identity, selection source, and emitted
IDs. A retryable-to-deduplicated state transition requires a same-hash success
introduced in that resolution pass. Source-ledger repairs additionally require
a prior successful config with the same effective-source hash in the repair
row's resolution generation; the original failed attempt may belong to an
earlier generation. CLC families then run a common-context witness and
conditional search for every legal divisor, followed by the complete planned
retained-depth by retained-divisor matrix. Failed cells remain in the
provenance; completion requires the full projection ledger and the complete
planned divisor/depth evidence. Finally, ordinary-family winners are normalized
into compound packet leaves. When more than four live parent families remain,
v22 runs one shared probe generation from the best measured representative of
every ordinary family, every compound leaf, and one unrestricted global
representative. Parent promotion is ranked only by ordinary qualification and
constrained-probe measurements, so a child found by the unrestricted probe
cannot promote its family by itself. Full effort continues the best four
competitive parent families, keeps every compound leaf represented, and
appends one unrestricted winner after the constrained paths. Continuation
capacity is derived from the live normalized leaf catalog and reserves every
ordinary protocol in the largest possible promoted-family set plus one
secondary per promoted family. The unrestricted path runs after constrained
paths and marks only selected candidates visited, so discarded global
neighbors remain available to later constrained or global generations.
Explicit `retained_families=None` custom
profiles continue all successful parent families; finite caps apply the 2x
slowdown filter before the cap. The live producer verifies the v22 decision
snapshots through its bound `ConfigGeneration`; the detached audit reconstructs lane
budgets and reconciles the reported candidate IDs, statuses, measurements, and
retention decisions against the mandatory canonical-config manifest and sparse
pass-by-pass measurement timeline. Compound source candidates are reconstructed
from the successful ordinary-family population; the audit verifies the ranked
combination prefix, attempted prefix, selected projections, stop condition, and
backfill suffix. Compound transfers retain exactly the first target-count
successful projections; retryable failures can trigger one bounded backfill
pass, while later source repairs may make the full success ledger longer than
that retained subset.
For finite-cap custom profiles, the 2x threshold is based only on measurements
from the current invocation.
The exact arithmetic winner remains stochastic. The held-out control below
tests new sequence lengths within the `B=2`, `H=32`, `D=64`, FP16 class. The
broader campaign then covers varied batch/head factorizations, FP16/BF16,
`D=64/128`, dense/causal attention, and every length-legality class. Neither
campaign establishes peak-performance discovery for unsupported tails,
modifiers, output forms, or every possible legal shape. Quick effort has a
smaller budget and is not expected to reproduce every peak-performance winner.
The specialized search currently applies to square self-attention with FP16 or
BF16 inputs, `D` in `{64, 128}`, and sequence lengths divisible by 128; other
forms fall back to the generic CuTe search instead of selecting a flash preset.

## Held-out seed robustness

The held-out launcher applies the same cold-search policy to dense `S=81920`
and causal `S=196608`, neither of which is in the plotted suite. It runs five
independent tuner seeds per shape, serially on the shape's assigned GPU while
the dense and causal lanes overlap. Both shapes use input seed `2026081500`;
dense uses tuner seeds `2026082301` through `2026082305`, and causal uses
`2026082311` through `2026082315`:

```bash
export HELDOUT_ROOT="$WORK/strict-heldout"
HELION_REPO_ROOT="$REPO" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/run_strict_heldout.sh" \
  "$HELDOUT_ROOT"
python "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/build_heldout_manifest.py" \
  --artifact-root "$HELDOUT_ROOT" \
  --all8-artifact-root "$STRICT_ROOT" \
  --output "$WORK/heldout_manifest.csv"
```

Before launching either lane, the runner copies both
`run_strict_heldout.sh` and its adjacent `run_strict_all8.sh` helper into
`$HELDOUT_ROOT/launcher` and changes both campaign-local snapshots to read-only
mode `0555`. All lane and per-seed processes execute only those snapshots. The
runner writes their SHA-256 digests, the exact ten-case campaign, and the
canonical Python executable to `campaign.csv`. The validator requires both
snapshots to be regular nonsymlink files with no write bits and verifies their
digests against every campaign row.

The validator rejects missing, extra, reordered, or pre-campaign results, then
reuses the complete all-eight sidecar and provenance audit for all ten
searches. It rebuilds the all-eight manifest directly from that campaign's raw
result and three-sidecar sets rather than trusting an editable summary CSV. It
requires dense to match all-eight `S=65536/131072`, causal to match all-eight
`S=131072/262144`, and the held-out coverage design to match both bracketing
lengths, the median seed to reach at least 99.8% of that shape's best seed, and
the slowest seed to reach at least 99.5% of the best. Every seed must also reach
at least 99% of the lower measured throughput from the two bracketing all-eight
lengths. The independent bracketing gate prevents five consistently poor
searches from certifying one another.

### Held-out winner adjudication

`run_heldout_adjudication.py` separates search validity from the cross-seed
performance decision. It first runs the complete strict validator on each of
the ten held-out searches, but does not apply the median/best or worst/best
throughput gates while selecting the five winners for each shape.
It still rebuilds the all-eight reference from `--all8-artifact-root`, requires
the held-out and bracketing rows to use the same Helion/CuTe version, structural
coverage design, and canonical compiler-seed policy, and keeps the existing
99% floor against the lower bracketing all-eight throughput. Thus only the two
cross-seed gates under adjudication are deferred.
It then regenerates every selected kernel from the clean measured checkout,
requires its source and compiled-source SHA-256 to match the search artifact,
and archives the regenerated source. The initial contender policy is exactly
the five held-out winners plus explicitly forced cuDNN SDPA. The contender
list is an explicit manifest so a later, separately declared experiment can
add authenticated training-derived configs without changing the measurement
worker; no all-eight or other training-derived config is present in this run.

The runner assigns causal `S=196608` to physical GPU 6 and dense `S=81920` to
physical GPU 7. The lanes run concurrently. Each lane checks FP16 correctness
against cuDNN SDPA with `atol=0.05` and `rtol=0.02`, requires three exact-repeat
launches for every contender and SDPA, performs the existing 10-second thermal
warmup, and calibrates repetitions to at least 20 ms of SDPA work. It then
runs 12 deterministic randomized rotations. With six implementations, each
implementation occupies each order position twice. Every rotation
records both CUDA-event and host-wall time. The report recomputes paired
20,000-sample bootstrap intervals for median/best, worst/best, and the slowest
winner versus SDPA. The existing 99.8% median/best and 99.5% worst/best
thresholds are reported unchanged for both timers. A valid measurement may
therefore produce `performance_gate_status: FAIL`; that is an adjudication
result, not a harness failure.

Run from the artifact checkout while pointing `--repo` at the clean measured
checkout pinned by this setup and `--heldout-root` at the completed ten-search
campaign:

```bash
export ADJUDICATION_ROOT="$WORK/heldout-winner-adjudication"
mkdir -p "$WORK/logs"
nohup setsid "$STRICT_PYTHON_EXECUTABLE" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/run_heldout_adjudication.py" \
  "$ADJUDICATION_ROOT" \
  --repo "$REPO" \
  --heldout-root "$HELDOUT_ROOT" \
  --all8-artifact-root "$STRICT_ROOT" \
  --python "$STRICT_PYTHON_EXECUTABLE" \
  >"$WORK/logs/heldout-winner-adjudication.log" 2>&1 < /dev/null &
echo $!
```

The runner holds a sibling campaign lock for its lifetime. Before GPU work it
atomically creates a read-only `campaign.json`, its digest, read-only snapshots
of the worker and every local imported dependency, snapshots of the measured
benchmark and attention example, the exact result paths, and hashes of every
held-out and all-eight result and each required sidecar. Worker processes
execute the snapshot, use only one visible GPU, recheck the clean measured
commit and source files before and after measurement, and write an
authenticated read-only completion marker only after those postconditions
pass. The runner publishes a case directory only when both the result and
marker validate. Logs, caches, temporary directories, and the append-only event
journal are outside the authenticated result payloads.

An interrupted campaign can be resumed with the same arguments, the additional
`--resume` flag, and preferably the snapshotted runner at
`$ADJUDICATION_ROOT/launcher/run_heldout_adjudication.py`. Valid completed case
directories are revalidated and skipped. A complete authenticated staging
directory left after the worker exited is promoted atomically; an incomplete
non-symlink staging directory is discarded and remeasured. A malformed
completed result, changed dependency, changed held-out artifact, different
Python executable, dirty checkout, or different commit fails closed.
Concurrent supervisors are rejected by the campaign lock. Each worker also
inherits a per-case lock descriptor, so a supervisor killed without cleanup
cannot let a resume delete an active orphan's staging directory or launch a
second process on the same case.

```bash
nohup setsid "$STRICT_PYTHON_EXECUTABLE" \
  "$ADJUDICATION_ROOT/launcher/run_heldout_adjudication.py" \
  "$ADJUDICATION_ROOT" \
  --repo "$REPO" \
  --heldout-root "$HELDOUT_ROOT" \
  --all8-artifact-root "$STRICT_ROOT" \
  --python "$STRICT_PYTHON_EXECUTABLE" \
  --resume \
  >>"$WORK/logs/heldout-winner-adjudication.log" 2>&1 < /dev/null &
```

After completion, independently validate the raw artifacts and write a copy of
the authenticated report outside every evidence tree:

```bash
"$STRICT_PYTHON_EXECUTABLE" \
  "$ADJUDICATION_ROOT/launcher/validate_heldout_adjudication.py" \
  --campaign-root "$ADJUDICATION_ROOT" \
  --output "$WORK/heldout_winner_adjudication_report.json"
```

The validator itself must run from the recorded read-only snapshot. It
revalidates all ten original searches individually, every declaration and
evidence digest, all linked all-eight invariants, the exact randomized order,
raw event/wall measurements, correctness and repeatability records,
regenerated-source archives, bootstrap summary, and final report. It does not
rerun GPU work.

## Minimal cross-shape cold-full validation

`minimal_cross_shape_cases.csv` is the smaller cross-shape check used before
claiming that the standard full tuner generalizes beyond the plotted
`B=2`, `H=32`, FP16 D64 shapes. Dense and causal variants of these three shapes
are included:

- `B=4`, `H=16`, `S=16384`, `D=64`, FP16
- `B=3`, `H=32`, `S=16384`, `D=64`, BF16
- `B=3`, `H=32`, `S=16384`, `D=128`, FP16

Each case uses three unique tuner RNG seeds, for 18 independent searches. The
validator binds the exact ordered matrix, including input and tuner seeds, so a
changed shape or seed is not accepted as the same campaign. All six shapes are
in the divisible-by-four legality class; this is a targeted check of batch/head
factorization, dtype, and head dimension rather than a replacement for the
broader all-legality-class campaign below.

Run it from the same clean detached checkout pinned above, with an output root
outside that checkout:

```bash
export MINIMAL_CROSS_SHAPE_ROOT="$WORK/minimal-cross-shape"
HELION_REPO_ROOT="$REPO" \
STRICT_PYTHON_EXECUTABLE="$STRICT_PYTHON_EXECUTABLE" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/run_minimal_cross_shape_campaign.sh" \
  "$MINIMAL_CROSS_SHAPE_ROOT"
python "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/validate_generalization_campaign.py" \
  --artifact-root "$MINIMAL_CROSS_SHAPE_ROOT" \
  --output "$WORK/minimal_cross_shape_manifest.csv"
```

The launcher accepts only an optional `--resume`, pins commit
`c3e36b65d69681c23e053042b0bc21e2331bad17`, and delegates to
`run_generalization_campaign.py`. That runner executes dense searches serially
on physical GPU 7 and causal searches serially on physical GPU 6, with the two
lanes concurrent. It verifies B200 GPUs at 750 W, gives every search a fresh
cache, forces the random initial population and `autotune_effort=full`, and
forbids fixed configs, user/CLI seed configs, cache reads, noncanonical compiler
heuristics, custom tuners, time budgets, and generation caps through the same
command and artifact gates as the broader campaign. Canonical CuTe-flash
compiler seeds remain enabled and must all be measured in generation zero. No
selected config or generated source is an input to tuning. Afterward, its
existing remeasurement phase regenerates each
of the three winners per shape, verifies source identity and numerics, and
compares them with explicitly forced cuDNN SDPA.

## Cross-shape and cross-seed generalization

The broader generalization campaign exercises fifteen shapes that are not in
the plotted all-eight suite. It covers FP16 and BF16, head dimensions 64 and 128,
dense and causal attention, varied batch/head factorizations, and all four
length-legality classes: one KV tile, odd tile count, even paired, and
divisible-by-four cluster alignment. Three pairs have shape descriptors that
differ only in sequence length, which requires their ordered
structural-coverage digest to remain identical.
The original ten cases run five independent tuner RNG seeds. Five targeted
FP16 D64 paired cases run three seeds each: dense BH=96, BH=120, and BH=180
cases audit CLC factor coverage, while two length-only causal twins audit every
legal aliased K/V-stage lane. BH=180 has 18 legal CLC divisors, more than fit in
the bounded generation-zero anchor set. Full qualification must nevertheless
measure a common-context witness for all 18 in every run. The campaign therefore
performs 65 fresh full searches.

The validator accepts only lane-policy v14 / structural-qualification v22
artifacts. For each targeted causal run, every resource-legal D64 aliased K/V
depth from 2 through 12 must have completed measured evidence and a conditional
continuation, unless the recorded exact effective space proves that lane was
already exhaustively measured. A retryable failed witness/continuation may use
the single recorded exact-lane repair, but a failed candidate is never accepted
as successful evidence; individual staged-family catalogs may omit
depths that normalize away for that family. For each targeted dense run, all
legal divisors must appear as independently recorded `search_values` in all
four ordinary CLC families. The generation-zero ledger must attempt every
bounded coverage anchor. Qualification then measures common-context witnesses
for every legal divisor in one generation, including all 18 BH=180 values.
This keeps qualification independent of both the RNG seed and the remaining
generation budget. These are provenance gates, not names or configs supplied
to the benchmark command.

When the pre-normalization raw Cartesian product fits within the initial
population bound, v22 enumerates it and records the resulting distinct
normalized config IDs plus per-leaf and per-lane counts. Otherwise the exact
provenance values and counts remain unset, with an empty phase ID list. A lane
may omit its conditional child only when every recorded exact ID for that lane
has a successful or source-deduplicated generation-zero measurement. Compound
packet transfers record source and
projected snapshots, canonical IDs, and a content digest of the active
ConfigSpec normalization context. These SHA-256 values bind artifact contents;
they are not cryptographic signatures.

The live benchmark verifies the exact-space enumeration and each projection
through the same bound `ConfigGeneration` normalization used by the search. The
detached verifier checks the recorded context digest, config snapshots, leaf
transition, IDs, preserved pipeline depths, and sidecar measurements, but it
does not independently recreate the raw search space. The all-eight paired
worker binds the actual case and repeats the live `ConfigGeneration` validation.
The broader campaign's remeasurement worker instead regenerates and compiles
each selected source and checks its digest; it relies on the live validation
performed when the search artifact was produced.

Every search uses the same strict policy as the plotted suite: an isolated
cache, forced cache miss, authenticated canonical CuTe-flash compiler seeds,
disabled user seeds and value priors, the legality-complete structural prefix
plus unconstrained random population, candidate accuracy checks, and no config
or generated source supplied by the harness. A non-enumerated search runs all
20 LFBO generations. When the raw
Cartesian product fits within the 100-row enumeration bound, the generator
normalizes every legal raw point and deduplicates aliases, records the ordered
effective config-ID digest, and may use fewer generations only after every
normalized ID is present and successful or source-deduplicated. Dense searches
run serially
on physical GPU 7 and causal searches serially on physical GPU 6; the two lanes
overlap. A failure in either lane terminates its peer and is never retried
implicitly.

Lane policy v14's terminal schema v2 adds a deterministic coordinate refinement
after the ordinary full search. It keeps the selected structural leaf fixed,
enumerates the recorded normalized coordinate surface at radius two, and runs a
width-four beam for two rounds. Each round uses a mirrored wall-time comparison
targeting 200 ms from the recorded repeat reference; the final confirmation
targets 5,000 ms. Fast kernels batch multiple timed calls into at most 64 even,
deterministically ordered samples, and record the sweep count, calls per sample,
and actual total calls. Each timed call is preceded by L2 clearing and device
synchronization; JIT cleanup runs after elapsed-time capture. The validator
checks the exact uncapped call-layout formula, normalized timings, and independent
raw-work floors that prevent an inflated repeat reference from shrinking the
trace. It also checks the
new/reused/prior-failed classifications, the preterminal registry digest, and
every config/source sidecar. The normal 100-candidate full-search floor is
evaluated at the recorded preterminal boundary, so refinement work cannot make
an undersized search appear complete.
The boundary is the first ledger record for a declared terminal-new config;
`preterminal_num_configs_tested` must exactly equal the chronological `started`
rows before the first terminal-new benchmark attempt. A precompile failure has
only its terminal failure row, no `started` row, and does not increment that
counter. Every preterminal registry member must have its first source-lifecycle
evidence before the terminal boundary; later repair/deduplication evidence may
follow it. Detached validation also requires one unique neighbor row for every
recorded active coordinate value; non-enum surfaces may contain no other rows,
while target-only enum base/default rows must be unique. Completeness of those
extra enum rows is part of the mandatory live surface replay. Detached validation
independently applies each one-coordinate request to its parent. Exact direct
projections must name that resulting config; normalization-dependent aliases and
invalid projections are counted separately and remain subject to mandatory live
replay. The paired worker replays every coordinate projection through the
freshly bound `ConfigSpec`, and publication checks that its replay count equals
the authenticated transcript count.
Coordinate-surface hashes must agree across seeds and across sequence lengths in
the same normalization/legality class; distinct causal, dtype, head-dimension,
or legality classes may have different surfaces.
Wall-clock-budgeted searches skip terminal refinement; publication-grade runs
already forbid such budgets and therefore require a complete terminal transcript.

After tuning, a fresh process per case regenerates and compiles every selected
config and requires both source hashes to equal the measured ledger. It checks
full-output numerics against explicitly forced cuDNN SDPA and exact
repeatability, then measures every winner and SDPA in 12 randomized,
position-balanced rounds with both CUDA-event and synchronized wall timers.
The common per-round repetition count is calibrated from three forced-SDPA
event probes to target at least 20 ms, capped at 4,096 calls, so the singleton
cases are not dominated by timer resolution.
For every case, the slowest seed must retain at least 98% of the fastest seed;
the divisible-by-four class uses a 99% gate. The lower bound of a 20,000-sample
paired bootstrap interval must pass the same gate. For the FP16 D64
divisible-by-four dense and causal cases, every seed must also reach 98% of
forced cuDNN SDPA by both timers, including the paired-bootstrap lower bound.

These bootstrap intervals resample the 12 timing rounds for the fixed set of
three or five tuner seeds; they do not estimate the probability that an
arbitrary future tuner seed finds the same performance. Only the FP16 D64
divisible-by-four dense and causal cases have an absolute SDPA floor. The other
cases establish structural reachability and consistency across the sampled
seeds, not absolute performance parity with SDPA.

Run the campaign from the artifact checkout while measuring the clean detached
checkout created above:

```bash
export GENERALIZATION_ROOT="$WORK/generalization-campaign"
python "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/run_generalization_campaign.py" \
  "$GENERALIZATION_ROOT" \
  --repo "$REPO" \
  --python "$STRICT_PYTHON_EXECUTABLE"
python "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/validate_generalization_campaign.py" \
  --artifact-root "$GENERALIZATION_ROOT" \
  --output "$WORK/generalization_manifest.csv"
```

Initialization snapshots the matrix, launcher, validator, strict validator,
remeasurement worker, and benchmark source into the artifact root and makes the
code snapshots read-only. Resume only through that immutable launcher; it
accepts completed attempts with explicit successful terminal events and refuses
partial or failed attempts:

```bash
"$STRICT_PYTHON_EXECUTABLE" \
  "$GENERALIZATION_ROOT/launcher/run_generalization_campaign.py" \
  "$GENERALIZATION_ROOT" \
  --repo "$REPO" \
  --python "$STRICT_PYTHON_EXECUTABLE" \
  --resume
```

## CPU-only validation

Stage exactly these six harness files under the detached measured checkout so
its repository-root resolution, source import, and commit validation all refer
to that checkout. No other nonignored untracked file is permitted; in
particular, startup hooks such as `sitecustomize.py` and additions under
`helion/` fail validation. No ignored file is allowed. Python bytecode must be
redirected to a fresh directory outside the checkout before starting Python.
`ARTIFACT_REPO`, `REPO`, and `WORK` are defined by the source setup in the main
plot README:

```bash
export VALIDATION_DIR="$REPO/.validation/generalized_paired"
export VALIDATION_PYCACHE="$WORK/generalized-paired-python-cache"
test ! -e "$VALIDATION_PYCACHE"
export PYTHONPYCACHEPREFIX="$VALIDATION_PYCACHE"
export PYTHONPATH="$VALIDATION_DIR"
mkdir -p "$VALIDATION_DIR"
cp "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/paired_worker.py" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/build_strict_manifest.py" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/run_all8.py" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/combine_results.py" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/test_build_strict_manifest.py" \
  "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/test_static.py" \
  "$VALIDATION_DIR/"
```

These commands do not query or initialize a GPU:

```bash
cd "$REPO"
python .validation/generalized_paired/test_static.py
python .validation/generalized_paired/paired_worker.py --help
python .validation/generalized_paired/run_all8.py --help
python .validation/generalized_paired/run_all8.py \
  --artifact-root "$STRICT_ROOT" \
  --output-dir "$WORK/generalized-paired-static" \
  --validate-only
```

The final command writes the full provenance audit and all 16 planned worker
commands to `static_validation.json`.

## Full run

After confirming GPUs 6 and 7 are idle and capped at 750 W:

```bash
cd "$REPO"
python .validation/generalized_paired/run_all8.py \
  --artifact-root "$STRICT_ROOT" \
  --output-dir "$WORK/generalized-paired"
```

To choose different recorded campaign seeds, pass exactly two distinct
`--campaign-seed` options. A worker has a 20-minute safety timeout by default;
timeout or failure terminates its process group and cancels the peer lane.

The runner combines results automatically. To reproduce that projection from a
completed run, while rechecking the worker JSON, archived source, manifest, and
strict search artifacts:

```bash
python .validation/generalized_paired/combine_results.py \
  --output-dir "$WORK/generalized-paired" \
  --artifact-root "$STRICT_ROOT"
```

The recombination command reads the bootstrap sample count and base seed from
`static_validation.json`. Supplying either bootstrap option explicitly is
allowed only when it matches that recorded protocol.

Static validation cannot prove the generated-source digest by itself because it
does not create CUDA inputs. That last provenance link, compilation, numerical
check, repeatability check, and timing all happen in the isolated GPU worker.

## Publication

After the paired run passes, publish only the replacement `helion-cute` and
`sdpa` rows into deep copies of the archived all-backend payloads. The publisher
rejects any change to another implementation and revalidates raw paired orders,
timing summaries, FLOP math, bootstrap intervals, strict provenance, and source
digests against the strict manifest and source ledgers before invoking the
renderer. Before publication, independently rerun every retained CuTe-backed
baseline with the repository-pinned CuTe version and update the baseline
payload directory. The publisher rejects a payload when Helion+CuTe,
FlexAttention+CuTe, FA4, or a passing CuTe KernelAgent result reports a
different CuTe version in either its version string or display label. It also
applies the event-versus-wall consistency gates described above:

Set `CUTE470_BASELINE_ROOT` to the completed output root produced by
`plots/cute470_baseline_refresh/setup/run_campaign.sh`. Its validated overlay
is the publication baseline; the historical payloads in the repository still
contain CuTe 4.6.1 rows and are intentionally rejected here.

```bash
export CUTE470_BASELINE_ROOT=/path/to/completed/cute470-baseline-refresh
python "$ARTIFACT_REPO/plots/generalized_full_autotune/setup/publish_results.py" \
  --paired-raw "$WORK/generalized-paired/all8_paired_raw.json" \
  --run-manifest "$WORK/generalized-paired/run_manifest.json" \
  --static-validation "$WORK/generalized-paired/static_validation.json" \
  --strict-artifact-root "$STRICT_ROOT" \
  --strict-manifest "$WORK/strict_manifest.csv" \
  --heldout-artifact-root "$HELDOUT_ROOT" \
  --heldout-manifest "$WORK/heldout_manifest.csv" \
  --generalization-artifact-root "$GENERALIZATION_ROOT" \
  --generalization-manifest "$WORK/generalization_manifest.csv" \
  --baseline-payload-dir "$CUTE470_BASELINE_ROOT/published/payloads" \
  --output-payload-dir "$WORK/generalized-payloads" \
  --raw-artifact-label plots/generalized_full_autotune/all8_paired_raw.json \
  --renderer "$REPO/benchmarks/cute/compare_attention_backends.py" \
  --render-csv "$WORK/attention_backends_b200_750w.csv" \
  --render-plot "$WORK/attention_backends_b200_750w.png" \
  --render-summary-plot "$WORK/attention_backends_b200_750w_geomean.png" \
  --plot-impl-label 'kernelagent-1x=KernelAgent Public (1x Helion tuning time; $107 tokens)' \
  --plot-impl-label 'kernelagent-2x=KernelAgent Public (2x Helion tuning time; $245 tokens)' \
  --plot-impl-label 'kernelagent-10x=KernelAgent Public (10x Helion tuning time; $1,368 tokens)' \
  --plot-impl-label 'kernelagent-closed-1x=KernelAgent Closed (1x Helion tuning time; $167 tokens)' \
  --plot-impl-label 'kernelagent-closed-2x=KernelAgent Closed (2x Helion tuning time; $564 tokens)'
```

The strict, held-out, and 65-search generalization manifests are mandatory
publication prerequisites. The publisher regenerates all three from their raw
artifact roots, requires every held-out row to reference the exact all-eight
strict-manifest digest and version, and snapshots content plus
validator-relevant file metadata before copying any result. The published
Helion rows bind all three manifest digests in their full-autotune provenance.
The smaller 18-search cross-shape matrix is a preflight only and is rejected as
a publication substitute. `combine_results.py` removes absolute checkout,
artifact, generated-module paths from `all8_paired_raw.json`; the publisher
rejects any remaining absolute path while validating the portable projection
against the full path-bearing run and static manifests.
