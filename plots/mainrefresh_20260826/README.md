# 2026-08-26 Helion CuTe + SDPA main-report refresh (superseded)

**Superseded by `plots/mainrefresh_20260827/`**, which fixes two measurement
biases against the Helion rows in this campaign: the wall-clock timer charged
~0.2 ms/launch of host overhead that the baselines' CUDA-event timing does
not, and the final measurement ran on a GPU heat-soaked by the multi-hour
autotune search (worth ~1% under the 750 W cap) while baselines ran on an
idle GPU. See the measurement-bias note in `plots/README.md`.

Raw artifacts for the formerly tracked Helion CuTe and SDPA rows of the
fixed-shape FP16 report (`plots/attention_backends_b200_750w.{csv,png}` and
the geomean chart). Only those two implementations were rerun; every other baseline row
retains its historical measurement. The measurement protocol, seeds, and how
these rows differ from the superseded `0e1113f8` paired campaign are
documented in `plots/README.md`.

- `results/*.json.gz`: one gzipped JSON per implementation and shape, exactly
  as written by `benchmarks/cute/compare_attention_backends.py --json-output`.
  The `helion-cute` files embed the complete strict full-autotune provenance,
  including the per-candidate `trials` ledger and terminal-refinement
  transcript. The `*.strict-prevalidation.json.gz` files are the harness's
  pre-measurement provenance snapshots.
- `manifest.json`: SHA256 and byte counts of every uncompressed artifact, the
  input seed, the autotune RNG seed rule, GPU assignment, and the power cap.

Provenance summary (identical across all eight `helion-cute` artifacts):
Helion checkout `86d5d8cf849acbc5d0fc54bb7db0d57bd9040bed`
(`1.4.0.dev165+g86d5d8cf8`, clean tree), CuTe 4.7.0, strict mode
(`--helion-require-full-autotune 1`) with a cold `FROM_RANDOM` full-effort
search, forced cache bypass, best-of-1, no budget or generation caps, and a
64-launch final correctness plus bitwise-repeatability gate against forced
cuDNN SDPA. Dense shapes ran on physical GPU 7 and causal shapes on physical
GPU 6, both verified at 750 W.

The tracked payload rows under `plots/kernelagent/results/payloads/` no
longer reference this campaign; they now come from
`plots/mainrefresh_20260827/`.
