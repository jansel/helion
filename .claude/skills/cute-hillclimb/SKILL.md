---
name: cute-hillclimb
description: Iteratively optimize a Helion kernel under HELION_BACKEND=cute until it matches or beats the best available baseline — via general backend/autotuner changes (knobs and seeds, not kernel-specific hacks) — then clean up history and report before/after numbers.
---

Your task is to improve the performance of a specific Helion kernel (with `HELION_BACKEND=cute`) until it matches the state of the art. Limit your changes to the Helion CuTe backend and autotuner; do not make changes to the Helion frontend language. Make your changes as general as possible, so optimizations are likely to apply to many different kernels — don't over-specialize to one specific kernel. When you hit a meaningful choice (e.g. A is better for one kernel and B is better for another), add an autotuner knob so we can find the answer automatically. Avoid hardcoding shape- or kernel-specific heuristics. The one place heuristics are OK is autotuner seeds: a seed gives the autotuner a head start without hardcoding the choice. If the CuTe backend's design is a limiting factor, feel free to change the design and refactor.

# General notes

## Shared machine

Unless the prompt says otherwise, assume you are on a shared server with multiple GPUs and play nice with other users. Autotuning, benchmarking, and tests are sensitive to other load on the GPU, so run on an idle GPU by explicitly setting `CUDA_VISIBLE_DEVICES`. Before launching GPU jobs, check `nvidia-smi` for other processes; if a GPU is in use by someone else, avoid that GPU for the rest of the session — the other user is likely to reuse it.

Parallel experiments and tests on different GPUs are fine, but performance varies between GPUs due to thermals and power limits, so keep comparisons apples-to-apples: stick to the same GPU for the same benchmark+shape, and never run a baseline on one GPU and compare its absolute number against one collected on a different GPU. Good ways to parallelize: run `HELION_BACKEND=triton` tests on one GPU and `HELION_BACKEND=cute` tests on another, shard different shapes or kernels across GPUs, or use a spare GPU to try out ideas or re-run autotuning to check robustness (autotuning is randomized).

## Measurement pitfalls

Each of these has produced a convincing-but-fake regression on this hardware; guard against all of them:

- **Use the same timer for every backend.** Time all implementations with the same CUDA-event-based `do_bench`. A wall-clock timer charges per-launch host overhead to the kernel (~0.2 ms/launch for CuTe), which fabricates a deficit of roughly 0.5% on a tens-of-ms kernel while every other backend is event-timed.
- **Control thermal state.** Under a power cap the GPU heat-soaks during long runs (roughly 1% slowdown per minute until it plateaus, on a B200 at 750 W), so a multi-hour autotune followed immediately by measurement makes the "after" number look worse than it is. Let the GPU idle back to near its process-startup temperature before any measurement you intend to compare or report, and interleave A/B runs (ABAB) rather than running all of A then all of B, so drift hits both sides equally.
- **One config per process.** `kernel.bind(args)` memoizes by args: binding the same kernel+args twice returns the same object, so a second `set_config` clobbers the first and an in-process A/B silently measures one config twice. Compare configs in separate processes.

## Commit your work as you go

Group work into logical, complete chunks where everything is working, and commit each chunk to save progress. Use the cute-verify skill to test and review each chunk before committing it.

# Hillclimbing task

## Step 1: Establish a target kernel and baseline

Decide what to optimize and get a good baseline. The prompt should say what the target kernel is; if not, ask the user. Pick a small number of variants of that kernel to test and optimize (use 8 if not otherwise specified). Variants can be different input shapes or minor variations of the kernel.

Find or write a Helion implementation of the target kernel. There may already be one in `examples/` or `benchmarks/cute/` — reuse it where possible. Verify the kernel is correct under both `HELION_BACKEND=triton` and `HELION_BACKEND=cute`; fix bugs if necessary.

Next, find a good baseline to compare performance against. We will optimize the Helion version, but to know whether it is good we need a fair baseline. There may already be one in `benchmarks/` or `examples/`. At minimum you should have a PyTorch baseline and a CuTe baseline; more is fine too. A baseline written in CuTe is extremely useful because you can compare Helion's generated CuTe code against it for optimization ideas. A good source of CuTe kernels is Quack (`./quack` or https://github.com/Dao-AILab/quack.git); if you don't find one there, you may need to search the web or write one yourself. Spend a little time optimizing the baseline to make sure we are comparing against something solid.

Next, select the variants or shapes to test on. Pick realistic shapes — you may need to do some research here. Existing benchmarks, or the shapes used in our baselines' own benchmarking setups, are good sources. Don't pick shapes that are too small: they will be overhead-bound (mainly waiting for kernel launches), and you are looking for memory- or compute-bound shapes. Some variety is good, since you want your optimizations to generalize.

You can create a benchmark harness in `benchmarks/cute/`; reuse code where possible. Report performance in TFLOPS or GB/s, depending on whether we expect the kernel to be compute- or memory-bound.

Measure and record the starting performance. At the end you will need to report the original numbers, so measure and save them now: write the raw benchmark output (JSON) and the winning autotuner configs to files. These artifacts back the "before" column of the final report and belong in the `[noland]` commit later — do not let a history rewrite lose them. Allow a full autotuning run from a cold start — fresh random seed, autotune cache bypassed — which might take an hour or more; never cut autotuning short or reuse a warm cache for any number reported to the user.

If needed, make a commit to save your progress before moving on to the next step.

## Step 2: Optimize the performance of one shape+variant

Pick the shape/variant with the largest performance gap to the best baseline, and optimize it until we are beating that baseline. You will need to come up with optimization ideas. One good way is to experiment with manual modifications to Helion's output code, especially comparing it against the handwritten CuTe baseline — this lets you iterate on ideas quickly without first implementing a general compiler pass. Then implement the winning ideas as generally as possible, adding autotuner knobs to expose choices where needed. If the autotuner is unable to find the best configuration, you may need to improve the autotuner seeding algorithm.

An optimization only counts once a *cold* full autotune finds it: run a fresh full-effort search (new random seed, autotune cache bypassed) and confirm it lands on the improvement. A config that only wins when hand-picked is a hardcoded heuristic in disguise — fix the search space or the seeding until the autotuner finds it on its own.

If needed, make a commit to save your progress before moving on to the next step.

## Step 3: Completion check, or loop again

Have we matched (within measurement noise) or beaten the best baseline on every variant? Define the noise margin operationally rather than eyeballing it: re-run the identical configuration several times back-to-back on the same GPU and use the observed spread (typically around 1% on a power-capped B200). We should also have a geomean speedup of at least 1. If both hold, continue to Step 4. Otherwise, go back to Step 2 and repeat. Keep repeating this optimization loop until the performance goals are met.

## Step 4: Clean up git history

You should now have a git history (relative to `origin/main`) of all the optimization work. Rewrite this history into clear logical chunks with good commit messages:

- All benchmarking/baseline changes go in a single commit at the start of the history.
- Planning files, measurement data/plots, the raw benchmark artifacts and autotuner configs saved in Steps 1–3, and anything else that should not be submitted upstream go in a single `[noland]` commit at the end of the history; these files must not be referenced by other commits.
- Group the remaining work into logical pieces that are easy to review. Avoid numerous small commits, but use your best judgment.
- Prefix CuTe backend commits with `[cutedsl]`.

Use cute-verify as needed to review and test each commit.

## Step 5: Final report

Prepare a final report for the user. It should include a table listing the performance of each variant and each baseline, with Helion+CuTe numbers from both before this hillclimbing process and after. Every number in the table must be traceable to an archived raw artifact from the `[noland]` commit — the "before" column comes from the Step 1 artifacts, and both columns from cold full-autotune runs. Label the numbers with version numbers and the GPU model and power cap.

The report should also summarize the changes made, and flag anything risky or with possible impact on parts of Helion beyond the CuTe backend.
