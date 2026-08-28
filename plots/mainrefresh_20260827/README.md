# 2026-08-27 Helion CuTe + SDPA main-report refresh

Raw artifacts for the Helion CuTe and SDPA rows in the tracked fixed-shape
FP16 report (`plots/attention_backends_b200_750w.{csv,png}` and the geomean
chart). Only those two implementations were rerun; every other baseline row
retains its historical measurement. This refresh supersedes
`plots/mainrefresh_20260826/` by fixing two measurement biases against the
Helion rows: helion-cute is now timed by the same CUDA-event `do_bench` as
every baseline (the wall timer charged ~0.2 ms/launch of host overhead), and
the harness idled the GPU back to 55 C before the measurement phase so the
multi-hour autotune search cannot heat-soak the GPU (worth ~1% under the
750 W cap; the harness has since generalized the cooldown target to the
process-startup temperature plus a small margin, equivalent on these
machines). The protocol, seeds, and the bias analysis are documented in
`plots/README.md`.

- `results/*.json.gz`: one gzipped JSON per implementation and shape, exactly
  as written by `benchmarks/cute/compare_attention_backends.py --json-output`.
  The `helion-cute` files embed the complete strict full-autotune provenance,
  including the per-candidate `trials` ledger and terminal-refinement
  transcript. The `*.strict-prevalidation.json.gz` files are the harness's
  pre-measurement provenance snapshots.
- `manifest.json`: SHA256 and byte counts of every uncompressed artifact, the
  input seed, the autotune RNG seed rule, GPU assignment, power cap, timer,
  and cooldown threshold.

Provenance summary (identical across all eight `helion-cute` artifacts):
Helion checkout `7789ca3269887c687c850a5a1f3e6c6823aac267`
(`1.4.0.dev168+g7789ca326`, clean tree), CuTe 4.7.0, strict mode
(`--helion-require-full-autotune 1`) with a cold `FROM_RANDOM` full-effort
search, forced cache bypass, best-of-1, no budget or generation caps, and a
64-launch final correctness plus bitwise-repeatability gate against forced
cuDNN SDPA. Dense shapes ran on physical GPU 7 and causal shapes on physical
GPU 6, both verified at 750 W. Every row records its pre-measurement
`thermal_cooldown` wait.

The payload rows under `plots/kernelagent/results/payloads/` embed the same
provenance with sub-objects larger than 20 KB replaced by
`{"elided": true, "sha256": ..., "bytes": ..., "entries": ...}` fingerprints;
each fingerprint is the canonical-JSON SHA256 of the corresponding object in
the archived raw artifact named by the row's `raw_artifact` field.
