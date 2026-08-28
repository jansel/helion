from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any
from typing import Callable
from typing import Iterator

import build_strict_manifest as strict
import torch
import validate_generalization_campaign as validator


@contextlib.contextmanager
def scrubbed_argv() -> Iterator[None]:
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        yield
    finally:
        sys.argv = saved


def gpu_record(physical_gpu: int) -> dict[str, object]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-gpu=index,name,uuid,power.limit",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    index, name, uuid, power = (field.strip() for field in query.split(","))
    pids = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    competing = [
        int(line)
        for line in pids.splitlines()
        if line.strip().isdigit() and int(line) != os.getpid()
    ]
    strict.require(not competing, f"GPU {physical_gpu} has competing PIDs {competing}")
    strict.check_equal(int(index), physical_gpu, "physical GPU index")
    strict.check_equal(name, "NVIDIA B200", "GPU model")
    strict.require(abs(float(power) - 750.0) <= 0.5, f"GPU power limit is {power}")
    return {
        "physical_gpu": int(index),
        "name": name,
        "uuid": uuid,
        "power_limit_w": float(power),
        "active_compute_pids": competing,
    }


def validate_repo_identity(repo: Path, root: Path) -> str:
    with (root / "campaign.jsonl").open() as handle:
        header = json.loads(handle.readline())
    strict.require(isinstance(header, dict), "campaign header is not an object")
    strict.check_equal(header.get("repo_root"), str(repo), "measured repository")
    expected_commit = header.get("expected_commit")
    strict.require(isinstance(expected_commit, str), "campaign commit is missing")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    strict.check_equal(commit, expected_commit, "measured repository commit")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    strict.require(not status, f"measured repository is not clean:\n{status}")
    strict.check_equal(
        strict.file_sha256(repo / validator.BENCHMARK_RELATIVE),
        header.get("benchmark_sha256"),
        "measured benchmark source",
    )
    return expected_commit


def tensor_error_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 5e-2,
    rtol: float = 2e-2,
) -> dict[str, object]:
    strict.check_equal(tuple(actual.shape), tuple(expected.shape), "output shape")
    strict.check_equal(actual.dtype, expected.dtype, "output dtype")
    count = actual.numel()
    close = 0
    nonfinite = 0
    max_abs = 0.0
    sum_sq = 0.0
    expected_sum_sq = 0.0
    for start in range(0, actual.shape[-2], 2048):
        stop = min(start + 2048, actual.shape[-2])
        actual_chunk = actual[..., start:stop, :].float()
        expected_chunk = expected[..., start:stop, :].float()
        diff = (actual_chunk - expected_chunk).abs()
        close += int((diff <= atol + rtol * expected_chunk.abs()).sum())
        nonfinite += int((~actual_chunk.isfinite()).sum())
        max_abs = max(max_abs, float(diff.max()))
        sum_sq += float((diff * diff).sum(dtype=torch.float64))
        expected_sum_sq += float(
            (expected_chunk * expected_chunk).sum(dtype=torch.float64)
        )
    rmse = math.sqrt(sum_sq / count)
    expected_rms = math.sqrt(expected_sum_sq / count)
    return {
        "count": count,
        "close_count": close,
        "max_abs": max_abs,
        "rmse": rmse,
        "nrmse": rmse / expected_rms if expected_rms else 0.0,
        "actual_nonfinite": nonfinite,
        "atol": atol,
        "rtol": rtol,
        "passed": close == count and nonfinite == 0,
    }


def exact_repeat_summary(
    actual: torch.Tensor, baseline: torch.Tensor
) -> dict[str, object]:
    different = 0
    for start in range(0, actual.shape[-2], 2048):
        stop = min(start + 2048, actual.shape[-2])
        different += int(
            (actual[..., start:stop, :] != baseline[..., start:stop, :]).sum()
        )
    return {
        "count": actual.numel(),
        "different": different,
        "passed": different == 0,
    }


def timed_call(fn: Callable[[], torch.Tensor], repetitions: int) -> dict[str, float]:
    strict.require(repetitions >= 1, "timing repetitions must be positive")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    for _ in range(repetitions):
        output = fn()
    end.record()
    end.synchronize()
    wall_end = time.perf_counter_ns()
    del output
    return {
        "event_ms": float(start.elapsed_time(end)) / repetitions,
        "wall_ms": (wall_end - wall_start) / (1e6 * repetitions),
    }


def calibrate_timing_repetitions(
    fn: Callable[[], torch.Tensor],
    *,
    target_ms: float,
    maximum: int,
) -> tuple[int, list[float]]:
    strict.require(target_ms > 0, "timing target must be positive")
    strict.require(maximum >= 1, "maximum timing repetitions must be positive")
    probes = [timed_call(fn, 1)["event_ms"] for _ in range(3)]
    estimate = max(statistics.median(probes), 1e-6)
    repetitions = min(maximum, max(1, math.ceil(target_ms / estimate)))
    return repetitions, probes


def thermal_warmup(seconds: float) -> None:
    if seconds <= 0:
        return
    left = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for _ in range(20):
            left = left @ right
        torch.cuda.synchronize()


def balanced_orders(names: list[str], rounds: int, seed: int) -> list[list[str]]:
    strict.require(
        rounds > 0 and rounds % len(names) == 0,
        "rounds must be a multiple of implementation count",
    )
    rng = random.Random(seed)
    result: list[list[str]] = []
    for _block in range(rounds // len(names)):
        base = names.copy()
        rng.shuffle(base)
        rotations = [base[offset:] + base[:offset] for offset in range(len(base))]
        rng.shuffle(rotations)
        result.extend(rotations)
    return result


def live_compiler_seed_policy(bound: object) -> dict[str, object]:
    """Rebuild the canonical seed policy from the freshly bound kernel."""
    from benchmarks.cute import compare_attention_backends

    import helion
    from helion.autotuner.config_generation import ConfigGeneration

    repo = Path(helion.__file__).resolve().parent.parent
    strict.check_equal(
        Path(compare_attention_backends.__file__).resolve(),
        (repo / validator.BENCHMARK_RELATIVE).resolve(),
        "remeasurement benchmark module",
    )
    config_spec = bound.env.config_spec
    policy = compare_attention_backends._compiler_seed_policy(
        config_spec, ConfigGeneration(config_spec)
    )
    strict.check_equal(
        policy.get("kind"),
        strict.EXPECTED_COMPILER_SEED_POLICY_KIND,
        "live compiler seed policy",
    )
    return policy


def build_callables(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    case: validator.CaseSpec,
    runs: list[validator.RunSpec],
    payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, Callable[[], torch.Tensor]], list[dict[str, object]]]:
    from examples.attention import attention_output
    from examples.attention import causal_attention_output

    import helion

    kernel = causal_attention_output if case.causal else attention_output
    bound = kernel.bind(inputs)
    compiler_seed_policy = live_compiler_seed_policy(bound)
    implementations: dict[str, Callable[[], torch.Tensor]] = {}
    sources: list[dict[str, object]] = []
    for run in runs:
        provenance = payloads[run.run_id]["helion_overrides"]["autotune_provenance"]
        strict.check_equal(
            provenance.get("compiler_seed_policy"),
            compiler_seed_policy,
            f"{run.run_id}: compiler seed policy",
        )
        config = helion.Config.from_dict(provenance["selected_config"])
        source = bound.to_triton_code(config, emit_repro_caller=False)
        source_digest = hashlib.sha256(source.encode()).hexdigest()
        strict.check_equal(
            source_digest,
            provenance["selected_source_sha256"],
            f"{run.run_id}: regenerated source",
        )
        compiled = bound.compile_config(config)
        compiled_digest = bound.env.backend.generated_source_hash(compiled)
        strict.check_equal(
            compiled_digest, source_digest, f"{run.run_id}: compiled source"
        )
        name = f"seed_{run.tuner_seed}"
        implementations[name] = lambda compiled=compiled: compiled(*inputs)
        sources.append(
            {
                "name": name,
                "run_id": run.run_id,
                "tuner_seed": run.tuner_seed,
                "selected_config_sha256": strict.canonical_sha256(
                    provenance["selected_config"]
                ),
                "selected_source_sha256": source_digest,
                "compiled_source_sha256": compiled_digest,
            }
        )
    return implementations, sources


def run_remeasurement(
    args: argparse.Namespace, expected_commit: str
) -> dict[str, object]:
    root = args.artifact_root.expanduser().resolve()
    validation = validator.validate_campaign(root, require_remeasurement=False)
    cases = {case.case_id: case for case in validation.cases}
    case = cases.get(args.case_id)
    strict.require(case is not None, f"unknown case {args.case_id!r}")
    strict.check_equal(args.physical_gpu, case.physical_gpu, "requested GPU")
    strict.check_equal(
        os.environ.get("CUDA_VISIBLE_DEVICES"), str(case.physical_gpu), "GPU visibility"
    )
    gpu_start = gpu_record(case.physical_gpu)
    strict.check_equal(torch.cuda.device_count(), 1, "visible CUDA devices")
    torch.cuda.set_device(0)
    dtype = torch.float16 if case.dtype == "float16" else torch.bfloat16
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.protocol_seed)
    shape = (case.z, case.h, case.seq_len, case.head_dim)
    inputs = tuple(
        torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
        for _ in range(3)
    )
    case_runs = [
        run for run in validation.run_specs if run.case.case_id == case.case_id
    ]
    payloads = {
        run.run_id: strict.load_json_object(root / run.result_path) for run in case_runs
    }
    with scrubbed_argv():
        implementations, sources = build_callables(inputs, case, case_runs, payloads)

        def run_sdpa() -> torch.Tensor:
            return torch.nn.functional.scaled_dot_product_attention(
                *inputs, is_causal=case.causal
            )

        implementations["sdpa"] = run_sdpa
        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
        ):
            expected = run_sdpa()
            correctness: dict[str, object] = {}
            for name, implementation in implementations.items():
                if name == "sdpa":
                    continue
                actual = implementation()
                torch.cuda.synchronize()
                numerics = tensor_error_summary(actual, expected)
                repeats = []
                for repeat_index in range(1, args.repeatability_launches):
                    repeated = implementation()
                    torch.cuda.synchronize()
                    repeat = exact_repeat_summary(repeated, actual)
                    repeat["repeat_index"] = repeat_index
                    repeats.append(repeat)
                    del repeated
                strict.require(
                    numerics["passed"] and all(repeat["passed"] for repeat in repeats),
                    f"{case.case_id}/{name}: correctness or repeatability failed",
                )
                correctness[name] = {"numerics": numerics, "repeatability": repeats}
                del actual
            del expected
            torch.cuda.empty_cache()
            thermal_warmup(args.thermal_warmup_seconds)
            for _ in range(args.warmup_calls):
                for implementation in implementations.values():
                    output = implementation()
                    del output
            torch.cuda.synchronize()
            timing_repetitions, calibration_event_ms = calibrate_timing_repetitions(
                implementations["sdpa"],
                target_ms=args.target_timing_ms,
                maximum=args.max_timing_repetitions,
            )
            names = list(implementations)
            orders = balanced_orders(names, args.rounds, args.protocol_seed ^ 0x5A17)
            raw_rounds = []
            for round_index, order in enumerate(orders):
                raw_rounds.append(
                    {
                        "round_index": round_index,
                        "order": order,
                        "times": {
                            name: timed_call(implementations[name], timing_repetitions)
                            for name in order
                        },
                    }
                )
    summary = validator.summarize_remeasurement(
        raw_rounds,
        [name for name in implementations if name != "sdpa"],
        case,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.protocol_seed ^ 0xB007,
    )
    gpu_end = gpu_record(case.physical_gpu)
    return {
        "schema_version": 1,
        "status": "PASS",
        "case_id": case.case_id,
        "shape": {
            "z": case.z,
            "h": case.h,
            "seq_len": case.seq_len,
            "head_dim": case.head_dim,
            "dtype": case.dtype,
            "causal": int(case.causal),
        },
        "physical_gpu": case.physical_gpu,
        "measured_commit": expected_commit,
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
        "protocol": {
            "name": "pooled winners balanced randomized rotation",
            "protocol_seed": args.protocol_seed,
            "rounds": args.rounds,
            "warmup_calls_per_implementation": args.warmup_calls,
            "thermal_warmup_seconds": args.thermal_warmup_seconds,
            "repeatability_launches": args.repeatability_launches,
            "target_timing_ms": args.target_timing_ms,
            "max_timing_repetitions": args.max_timing_repetitions,
            "timing_repetitions": timing_repetitions,
            "sdpa_calibration_event_ms": calibration_event_ms,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.protocol_seed ^ 0xB007,
            "forced_sdpa_backend": "CUDNN_ATTENTION",
            "timers": ["cuda_event", "host_wall"],
        },
        "selected_sources": sources,
        "correctness": correctness,
        "raw_rounds": raw_rounds,
        "summary": summary,
        "worker_sha256": strict.file_sha256(Path(__file__).resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-measure five attention winners")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--protocol-seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--thermal-warmup-seconds", type=float, default=10.0)
    parser.add_argument("--repeatability-launches", type=int, default=3)
    parser.add_argument("--target-timing-ms", type=float, default=20.0)
    parser.add_argument("--max-timing-repetitions", type=int, default=4096)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--expected-worker-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    strict.check_equal(
        strict.file_sha256(Path(__file__).resolve()),
        args.expected_worker_sha256,
        "remeasurement worker digest",
    )
    strict.require(
        args.rounds > 0 and args.rounds % 6 == 0,
        "rounds must be a positive multiple of 6",
    )
    strict.require(
        args.repeatability_launches >= 2, "repeatability launches must be >=2"
    )
    strict.require(args.target_timing_ms > 0, "timing target must be positive")
    strict.require(
        args.max_timing_repetitions >= 1,
        "maximum timing repetitions must be positive",
    )
    strict.require(args.bootstrap_samples > 0, "bootstrap samples must be positive")
    expected_commit = validate_repo_identity(
        repo, args.artifact_root.expanduser().resolve()
    )
    sys.path.insert(0, str(repo))
    payload = run_remeasurement(args, expected_commit)
    strict.check_equal(
        validate_repo_identity(repo, args.artifact_root.expanduser().resolve()),
        expected_commit,
        "remeasurement checkout commit",
    )
    temporary = args.output.with_suffix(".json.tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
