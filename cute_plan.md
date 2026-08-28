# CuTe Flash Attention Optimization Plan

## Scope

This plan covers FP16/BF16 forward self-attention on NVIDIA B200 at a 750 W
power cap. The plotted performance workload is unbiased, output-only FP16
attention. BF16, LSE, modifiers, causal boundaries, and supported fallback
forms remain correctness requirements rather than performance claims.

The optimization must remain an autotuning problem. Exact plotted sequence
lengths may appear in benchmark case declarations, but B200 production code
must not map a sequence length to a winner, restrict a new length to a measured
winner, or change emitted source through an unrecorded length policy. Compiler
seeds may guide the search when every seed is measured and the same legal seed
families are generated for every shape in the same legality class.

## Historical Baseline

The fully autotuned B200 baseline was:

| Shape | Helion+CuTe | SDPA | Gap |
| --- | ---: | ---: | ---: |
| Dense `2x32x32768x64` | 770.6 TFLOPS | 773.2 TFLOPS | -0.34% |
| Dense `2x32x65536x64` | 765.8 TFLOPS | 770.4 TFLOPS | -0.60% |
| Dense `2x32x131072x64` | 782.0 TFLOPS | 785.6 TFLOPS | -0.45% |
| Dense `2x32x262144x64` | 755.1 TFLOPS | 774.5 TFLOPS | -2.51% |
| Causal `2x32x65536x64` | 734.5 TFLOPS | 760.1 TFLOPS | -3.36% |
| Causal `2x32x131072x64` | 730.5 TFLOPS | 759.8 TFLOPS | -3.86% |
| Causal `2x32x262144x64` | 741.9 TFLOPS | 775.8 TFLOPS | -4.36% |
| Causal `2x32x524288x64` | 728.8 TFLOPS | 743.1 TFLOPS | -1.93% |

The overall geometric-mean gap was 2.185%, with a 0.976% dense gap and a
3.379% causal gap. SDPA resolved to cuDNN 9.20's generated two-CTA
`128x128x64` attention kernel.

## Profile Evidence

- Causal 64K: Helion had 48.6% issue utilization versus 55.2%, 51.2%
  no-eligible cycles versus 44.0%, and 19.0M L2 sectors versus 17.0M.
- Dense 256K: Helion had 47.2% issue utilization versus 56.5%, 0.55 versus
  0.79 eligible warps per scheduler, and 78.2M versus 71.7M L2 sectors.
- cuDNN used Q2/K4/V4/S4, separate K/V storage, two-CTA clustering, 232,448
  bytes of dynamic shared memory, and full-grid scheduling.
- The historical Helion kernel used Q2, aliased KV2, S2, 101,376 bytes, and
  148 persistent CTAs for dense 256K.
- Helion executed fewer instructions but waited more. Synchronization and
  pipeline depth, rather than instruction count, were the primary headroom.
- The integrated two-CTA/deeper pipeline was profile-supported before it was
  experimentally established. It must not be presented as proven solely from
  the profile comparison.

## Experiment Ledger

- Causal predicate removal: +0.08% to +0.37%; constexpr unrolling: -1.76%.
- Independent CGA2: -0.29%; standalone 4D TMA: +0.02%.
- Wait hint zero: causal +0.13%, dense 256K +0.03%; removing the operand:
  -1.83%.
- Conflict-free scale layout: -1.98%; single-slot stats handoff: -0.03%.
- Register reallocations: neutral to -1.63%. Reducing static local operations
  from `49/33` to `13/6` produced no speedup.
- Existing dense scheduler/TMA variants stayed within 0.09%. Experimental role
  chaining, CuTe GEMM, stage-local setup, and unscoped correction regressed.
- Prior dense wins included early split publication, role chaining, and the TMA
  epilogue; that integrated role-chaining path is distinct from the regressing
  experiment above.

## Implemented Search Design

The active B200 search is derived from graph semantics and legality rather than
an exact-length winner table.

1. Effective code generation is fingerprinted after normalization. Candidates
   that emit identical source are deduplicated, while failures quarantine every
   alias of the same source.
2. Inactive fields are canonicalized. Legacy fixed configurations remain
   accepted, but dead Boolean combinations do not inflate the active search.
3. Structural schedules are compound pipeline families. Family selection owns
   topology, CTA cooperation, ring layout, CLC/local-TMA mode, and derived query
   tile count.
4. Compound exp2 packets own their required cadence and structural protocol.
   Independent arithmetic and resource fields remain tunable.
5. Full LFBO first qualifies every live ordinary family and relevant pipeline
   lane, evaluates legal compound transfers, and probes each live parent family
   before retaining the strongest parents. Compound leaves and an unrestricted
   path remain available after parent promotion.
6. The effective search policy is included in cache identity and provenance.
   Registers, shared memory, stack/local traffic, config identity, and generated
   source identity are retained for measured candidates.
7. Generic compiler seeds are generated from the live legal fragments. They
   are measured starting candidates, not compiler defaults or selected winners.
   Publication-grade strict evidence authenticates this canonical seed policy
   and proves every effective seed was measured in generation zero. Earlier
   seed-free runs are retained only as historical diagnostics.

For B200/SM100, sequence length affects structural legality such as tile
alignment, not a chosen winner. Shapes in the same legality class receive the
same ordered search choices and generic seed families. Other architectures may
carry their own target-specific measured policies and are outside this B200
generality claim.

## Correctness And Provenance

Every publication-grade run must:

- force a fresh full autotune and bypass result-cache reads;
- reject fixed configs, config overrides, user winner seeds, value priors,
  custom tuners, and wall-clock/generation caps;
- record the RNG seed and every effective candidate/source identity;
- require the selected config and generated source to have been measured;
- pass full-output correctness, exact repeatability, causal boundaries, and a
  peaky-logits numerical stress case;
- record the exact Helion, PyTorch, CuTe, cuDNN, CUDA, driver, GPU, and power-cap
  versions;
- use only physical GPUs 6 and 7, causal on GPU 6 and dense on GPU 7;
- keep per-shape caches isolated and reject dirty or changed source checkouts.

Each future PR must pass FP16/BF16 attention correctness, LSE and modifiers,
`cute-verify`, and the full test suite with and without `HELION_BACKEND=cute`.
Generic autotuner behavior changes must be CuTe-flash gated unless separately
justified and reviewed.

## Remaining Validation

1. On the final rebased revision and repository-pinned CuTe version, run a new
   cold all-eight campaign.
2. Run five independent seeds on held-out dense 81,920 and causal 196,608
   sequence lengths.
3. Run the broad cross-shape matrix covering batch/head factorizations, FP16 and
   BF16, head dimensions 64 and 128, dense and causal modes, and all supported
   length-legality classes.
4. Remeasure selected Helion and forced-cuDNN SDPA kernels with balanced paired
   event and wall timing.
5. Rerun every plotted CuTe-backed baseline with the same CuTe version. Do not
   publish a chart that mixes CuTe runtime versions.
6. Run exact-source race/synchronization checks for new selected kernels; the
   repeated correctness screen alone is not a complete race proof.
7. Regenerate the CSV, grouped bar chart, and geomean chart from validated,
   portable evidence with no local filesystem paths.

## Acceptance Gates

- Autotuner truthfulness: at least 30% fewer effective candidate evaluations
  than the historical Cartesian search and recovery of the current winner
  within 0.2%.
- Range specialization: at least +0.15% causal geometric mean, with no shape
  below -0.2%.
- One-CTA pipeline: at least +1.0% causal geometric mean, issue utilization
  above 52%, and no-eligible cycles below 47%.
- Cluster multicast: at least +0.75% incremental causal geometric mean and
  causal-64K L2 sectors at or below 18.0M.
- Softmax phase: at least +0.5% incremental geometric mean or 5% fewer
  `MUFU.EX2` instructions without spilling.
- Gap closed: every shape reaches at least 99.5% of SDPA and overall geometric
  mean reaches at least 100%.
- Stretch: overall geometric mean at least 102%, causal geometric mean at least
  101%, and no shape below SDPA.

Gains below 0.5% require randomized paired measurements with nine samples per
side and a confidence interval excluding zero. Full tuning runs have no
wall-clock early termination.
