# KernelAgent Attention Setup

These files preserve the local adapter used for the B200 attention runs. They
are audit artifacts, not part of the Helion package.

## Source

```bash
git clone https://github.com/meta-pytorch/KernelAgent.git
cd KernelAgent
git checkout e0647170da36ef9b059ac0bd3d60103aa4ed378b
git apply /path/to/helion/plots/kernelagent/setup/kernelagent-e0647170.patch
```

The patch adds the Claude Code CLI provider, B200 hardware data, FP16/event
benchmark controls, sandbox launch hooks, atomic result persistence, and the
multi-block response extractor fix. Its SHA256 is
`3846cac469d50365d37d37a6512b07680a0dfbbd83448ef334c982ece51a7cbc`.

No packages were installed for this experiment. `dotenv_shim.py.txt` and
`omegaconf_shim.py.txt` supplied the small import surfaces missing from the
existing environment. Place each at `__init__.py` under corresponding
`dotenv/` and `omegaconf/` directories, then prepend their parent directory
and the patched KernelAgent checkout to `PYTHONPATH`.

Python sources are archived with a `.txt` suffix so generated/audit code is
not treated as Helion package code by repository lint. Remove only that final
`.txt` suffix when recreating the launch environment.

## Launch files

- `kernelagent_attention_driver.py.txt` is the finalized supervisor. SHA256:
  `82b9d1fd0a71cd0dbb7a747bd81ac6c9e11deedc389bf707b729eee7b753e91c`.
- `kernelagent_attention_driver_v1.py.txt` is the exact earlier runner recorded by
  four 1x manifests. SHA256:
  `c6bf360f24d05383d231681b498a80404b89f368710dcb7f2e9322252f86cea8`.
- `sandbox_launcher.sh` creates the no-network namespace and exposes one GPU.
  It is a portability-normalized copy of the measured launcher.
- `run_attention_queue.sh` records the shape order and exact 1x/2x budgets.
- `run_final_attention_measurements.sh` records the final nine-sample command;
  its optional second argument selects one or more KernelAgent implementations.
- `run_kernelagent_10x.sh` runs fresh 10x searches with dense on GPU 7 and
  causal on GPU 6 concurrently, retaining the same isolation and power cap.

The manifests record the exact measured sandbox launcher as SHA256
`4b6f60c20c510e3d62e7b2d30dbb5063062107b1008f416bcb8016535c5e86a4`.
That byte-identical version is retained at
`plots/kernelagent/setup/sandbox_launcher.sh` in measured commit `0e1113f8`.
That commit is retained in `../../cute461_rerun/helion-0e1113f8.bundle`
(SHA256 `5d094bad9ffa47062ffadf56f3e4e9170905f0017d3f6956058ea210534a0d28`,
ref `refs/noland/helion-cute-0e1113f8`). The checked-out launcher replaces
only the machine-specific environment path with `KERNELAGENT_ENV_ROOT`; its
SHA256 is
`ceaa323bb1edc0178fbb610c6e72e33312fc49f197a633d42896d4d1e03b4a80`.
Keep dense runs on physical GPU 7, causal runs on physical GPU 6, and both
power limits at 750 W. Run dense and causal queues concurrently; never run two
processes on the same GPU.

The finalized supervisor rejects a non-idle GPU before starting and kills the
entire phase process group after its deadline. The v1 runner lacked that final
descendant cleanup. One losing causal-64K candidate survived after that run's
selection and held GPU 6's lock; it was killed, the affected causal-128K
attempt was discarded in full, and causal-128K was restarted from an empty
directory under the finalized runner. No published timing or selected kernel
came from the contaminated attempt.
