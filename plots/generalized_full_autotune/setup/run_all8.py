from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any

from paired_worker import CASES
from paired_worker import EXPECTED_POWER_CAP_W
from paired_worker import PEAKY_STRESS_THRESHOLDS
from paired_worker import REPO_ROOT
from paired_worker import Case
from paired_worker import atomic_write_json
from paired_worker import nvidia_smi_record
from paired_worker import sha256
from paired_worker import validate_artifact_set

EXPECTED_UUIDS = {
    6: "GPU-b95967ee-1ea7-3bca-894f-5ec74e1c5513",
    7: "GPU-9e1e775d-008f-d915-e4a1-80cde2e60a7e",
}
DEFAULT_CAMPAIGN_SEEDS = (2026081101, 2026081102)
DEFAULT_BOOTSTRAP_SEED = 2026081103
LANES = {
    7: [CASES[("dense", seq_len)] for seq_len in (32768, 65536, 131072, 262144)],
    6: [CASES[("causal", seq_len)] for seq_len in (65536, 131072, 262144, 524288)],
}
WORKER = Path(__file__).with_name("paired_worker.py")
HARNESS_PATHS = {
    "run_all8.py": Path(__file__).resolve(),
    "paired_worker.py": WORKER.resolve(),
    "build_strict_manifest.py": Path(__file__)
    .with_name("build_strict_manifest.py")
    .resolve(),
    "combine_results.py": Path(__file__).with_name("combine_results.py").resolve(),
    "test_build_strict_manifest.py": Path(__file__)
    .with_name("test_build_strict_manifest.py")
    .resolve(),
    "test_static.py": Path(__file__).with_name("test_static.py").resolve(),
}
DEFAULT_WORKER_TIMEOUT_SECONDS = 1200.0
EXPECTED_WORKER_SCHEMA_VERSION = 4
WORKER_POLL_SECONDS = 0.25
WORKER_TERMINATION_GRACE_SECONDS = 10.0
STRICT_ARTIFACT_IDENTITY_FIELDS = (
    "search_result_sha256",
    "selected_config_sha256",
    "selected_source_sha256",
    "compiler_seed_policy_sha256",
    "terminal_refinement_policy_sha256",
    "terminal_coordinate_surface_sha256",
    "terminal_refinement_sha256",
)
LOCAL_PATH_FIELDS = frozenset(
    {
        "artifact_root",
        "attention_example_expected_module_path",
        "attention_example_import_path",
        "helion_expected_package_path",
        "helion_import_path",
        "helion_module",
        "runtime_checkout",
        "search_result_path",
        "worker_pythonpath",
    }
)

# The paired worker gets a controlled code-generation and timing environment.
# In particular, no interactive CuTe or Helion experiment overrides survive.
SCRUBBED_ENV_PREFIXES = ("CUDNN_", "CUTE_DSL_", "HELION_", "TORCH_CUDNN_")
SCRUBBED_ENV_NAMES = {
    "CUDA_DEVICE_ORDER",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MODULE_LOADING",
    "NVIDIA_TF32_OVERRIDE",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
}


def current_harness_sha256() -> dict[str, str]:
    return {name: sha256(path) for name, path in HARNESS_PATHS.items()}


def validate_harness_sha256(expected: object) -> dict[str, str]:
    actual = current_harness_sha256()
    if expected != actual:
        raise RuntimeError(
            f"paired harness identity changed: expected {expected!r}, got {actual!r}"
        )
    return actual


def json_payload_sha256(payload: dict[str, Any]) -> str:
    contents = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(contents.encode()).hexdigest()


def active_compute_pids(gpu_uuid: str) -> list[int]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        int(line.strip()) for line in proc.stdout.splitlines() if line.strip().isdigit()
    ]


def validate_live_gpu(physical_gpu: int, expected_uuid: str) -> dict[str, Any]:
    gpu = nvidia_smi_record(str(physical_gpu))
    if gpu["physical_index"] != physical_gpu:
        raise RuntimeError(
            f"expected physical GPU {physical_gpu}, got {gpu['physical_index']}"
        )
    if gpu["uuid"] != expected_uuid:
        raise RuntimeError(
            f"physical GPU {physical_gpu} UUID changed: expected {expected_uuid}, "
            f"got {gpu['uuid']}"
        )
    if gpu["name"] != "NVIDIA B200":
        raise RuntimeError(f"physical GPU {physical_gpu} is {gpu['name']}, not B200")
    if abs(gpu["power_limit_w"] - EXPECTED_POWER_CAP_W) > 0.5:
        raise RuntimeError(
            f"physical GPU {physical_gpu} power cap is {gpu['power_limit_w']} W, "
            f"not {EXPECTED_POWER_CAP_W} W"
        )
    pids = active_compute_pids(expected_uuid)
    if pids:
        raise RuntimeError(
            f"physical GPU {physical_gpu} is not idle; active compute PIDs: {pids}"
        )
    gpu["active_compute_pids_before_run"] = pids
    return gpu


def worker_seed(campaign_seed: int, case: Case) -> int:
    return campaign_seed + case.seq_len + (1_000_000 if case.causal else 0)


def result_path(output_dir: Path, campaign_seed: int, case: Case) -> Path:
    return (
        output_dir
        / "campaigns"
        / f"seed_{campaign_seed}"
        / "results"
        / f"{case.name}.json"
    )


def logical_output_reference(output_dir: Path, path: Path) -> str:
    """Return a portable reference to a file within the paired output tree."""
    try:
        relative = path.resolve().relative_to(output_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"{path} is outside paired output root {output_dir}"
        ) from exc
    return relative.as_posix()


def logical_artifact_root_reference(output_dir: Path, artifact_root: Path) -> str:
    """Describe the strict tree relative to the paired tree when possible."""
    return Path(
        os.path.relpath(artifact_root.resolve(), start=output_dir.resolve())
    ).as_posix()


def resolve_output_reference(output_dir: Path, reference: str) -> Path:
    """Resolve a manifest file reference without allowing absolute paths or escapes."""
    relative = Path(reference)
    if relative.is_absolute():
        raise RuntimeError(f"paired output reference must be relative: {reference}")
    root = output_dir.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"paired output reference escapes output root: {reference}"
        ) from exc
    return resolved


def strict_artifact_identities(
    validated: dict[Case, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    return {
        case.name: {
            field: provenance[field] for field in STRICT_ARTIFACT_IDENTITY_FIELDS
        }
        for case, provenance in sorted(validated.items(), key=lambda item: item[0].name)
    }


def _without_local_paths(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _without_local_paths(item)
            for key, item in value.items()
            if key not in LOCAL_PATH_FIELDS
        }
    if isinstance(value, list):
        return [_without_local_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_local_paths(item) for item in value)
    return value


def portable_search_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Remove machine-local paths from combined publication provenance."""
    portable = _without_local_paths(provenance)
    assert isinstance(portable, dict)
    return portable


def portable_worker_environment(environment: dict[str, Any]) -> dict[str, Any]:
    """Remove import paths while retaining measured software and GPU identity."""
    portable = _without_local_paths(environment)
    assert isinstance(portable, dict)
    return portable


def require_portable_json(value: object, *, context: str = "combined artifact") -> None:
    """Reject absolute filesystem paths from a publication JSON value."""
    if isinstance(value, dict):
        for key, item in value.items():
            require_portable_json(item, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_portable_json(item, context=f"{context}[{index}]")
    elif isinstance(value, str) and Path(value).is_absolute():
        raise RuntimeError(f"{context}: absolute path is not portable: {value}")


def validate_strict_artifact_identities(
    expected: object,
    validated: dict[Case, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    actual = strict_artifact_identities(validated)
    if expected != actual:
        raise RuntimeError(
            f"strict artifact identity changed: expected {expected!r}, got {actual!r}"
        )
    return actual


def worker_command(
    args: argparse.Namespace,
    campaign_seed: int,
    physical_gpu: int,
    expected_uuid: str,
    case: Case,
    output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(WORKER),
        "--variant",
        case.variant,
        "--seq-len",
        str(case.seq_len),
        "--physical-gpu",
        str(physical_gpu),
        "--expected-gpu-uuid",
        expected_uuid,
        "--artifact-root",
        str(args.artifact_root.resolve()),
        "--output",
        str(output_path),
        "--generated-source-output",
        str(output_path.parents[1] / "generated_sources" / f"{case.name}.py.txt"),
        "--expected-worker-sha256",
        args.harness_sha256["paired_worker.py"],
        "--campaign-seed",
        str(campaign_seed),
        "--pairs",
        str(args.pairs),
        "--warmup-calls",
        str(args.warmup_calls),
        "--thermal-warmup-seconds",
        str(args.thermal_warmup_seconds),
        "--race-runs",
        str(args.race_runs),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--seed",
        str(worker_seed(campaign_seed, case)),
    ]


def worker_environment(
    expected_uuid: str, cache_dir: Path
) -> tuple[dict[str, str], list[str]]:
    env = os.environ.copy()
    scrubbed = sorted(
        key
        for key in env
        if key in SCRUBBED_ENV_NAMES
        or any(key.startswith(prefix) for prefix in SCRUBBED_ENV_PREFIXES)
    )
    for key in scrubbed:
        env.pop(key)
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": expected_uuid,
            "HELION_BACKEND": "cute",
            "HELION_CACHE_DIR": str(cache_dir.resolve()),
            "HELION_DISABLE_AUTOTUNER_HEURISTICS": "0",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(REPO_ROOT),
            "PYTHONPYCACHEPREFIX": str((cache_dir / "pycache").resolve()),
        }
    )
    return env, scrubbed


def terminate_process_group(process: subprocess.Popen[object]) -> str:
    if process.poll() is not None:
        return "already_exited"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return "exited_before_sigterm"
    try:
        process.wait(timeout=WORKER_TERMINATION_GRACE_SECONDS)
        return "sigterm"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return "exited_before_sigkill"
        try:
            process.wait(timeout=WORKER_TERMINATION_GRACE_SECONDS)
            return "sigkill"
        except subprocess.TimeoutExpired:
            return f"sigkill_sent_process_still_present_pid_{process.pid}"


def wait_for_worker(
    process: subprocess.Popen[object],
    *,
    timeout_seconds: float,
    stop_event: threading.Event,
) -> tuple[int, str | None, str | None]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode, None, None
        if stop_event.wait(WORKER_POLL_SECONDS):
            cleanup = terminate_process_group(process)
            return (
                process.returncode if process.returncode is not None else -1,
                "cancelled_after_peer_lane_failure",
                cleanup,
            )
        if time.monotonic() >= deadline:
            cleanup = terminate_process_group(process)
            return (
                process.returncode if process.returncode is not None else -1,
                f"worker_timeout_after_{timeout_seconds:g}_seconds",
                cleanup,
            )


def run_lane(
    args: argparse.Namespace,
    campaign_seeds: tuple[int, int],
    physical_gpu: int,
    expected_uuid: str,
    output_dir: Path,
    records: list[dict[str, Any]],
    records_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    try:
        for campaign_seed in campaign_seeds:
            for case in LANES[physical_gpu]:
                if stop_event.is_set():
                    raise RuntimeError(
                        f"campaign {campaign_seed} {case.name} cancelled before "
                        "launch after peer lane failure"
                    )
                validate_harness_sha256(args.harness_sha256)
                output_path = result_path(output_dir, campaign_seed, case)
                log_dir = output_path.parents[1] / "logs"
                stdout_path = log_dir / f"{case.name}.stdout"
                stderr_path = log_dir / f"{case.name}.stderr"
                generated_source_path = (
                    output_path.parents[1] / "generated_sources" / f"{case.name}.py.txt"
                )
                command = worker_command(
                    args,
                    campaign_seed,
                    physical_gpu,
                    expected_uuid,
                    case,
                    output_path,
                )
                cache_dir = output_path.parents[1] / "cache" / case.name
                env, scrubbed_env = worker_environment(expected_uuid, cache_dir)
                started_ns = time.time_ns()
                with (
                    stdout_path.open("w") as stdout_handle,
                    stderr_path.open("w") as stderr_handle,
                ):
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        start_new_session=True,
                    )
                    returncode, termination_reason, cleanup = wait_for_worker(
                        process,
                        timeout_seconds=args.worker_timeout_seconds,
                        stop_event=stop_event,
                    )
                record = {
                    "campaign_seed": campaign_seed,
                    "case": case.name,
                    "physical_gpu": physical_gpu,
                    "gpu_uuid": expected_uuid,
                    "pid": process.pid,
                    "command": command,
                    "returncode": returncode,
                    "termination_reason": termination_reason,
                    "cleanup": cleanup,
                    "worker_timeout_seconds": args.worker_timeout_seconds,
                    "scrubbed_environment_keys": scrubbed_env,
                    "controlled_environment": {
                        key: env[key]
                        for key in (
                            "CUDA_VISIBLE_DEVICES",
                            "CUDA_DEVICE_ORDER",
                            "HELION_BACKEND",
                            "HELION_CACHE_DIR",
                            "HELION_DISABLE_AUTOTUNER_HEURISTICS",
                            "PYTHONHASHSEED",
                            "PYTHONPATH",
                            "PYTHONPYCACHEPREFIX",
                        )
                    },
                    "started_ns": started_ns,
                    "finished_ns": time.time_ns(),
                    "output": logical_output_reference(output_dir, output_path),
                    "output_sha256": sha256(output_path)
                    if returncode == 0 and output_path.is_file()
                    else None,
                    "generated_source": logical_output_reference(
                        output_dir, generated_source_path
                    ),
                    "generated_source_sha256": sha256(generated_source_path)
                    if returncode == 0 and generated_source_path.is_file()
                    else None,
                    "stdout": logical_output_reference(output_dir, stdout_path),
                    "stderr": logical_output_reference(output_dir, stderr_path),
                }
                with records_lock:
                    records.append(record)
                if termination_reason is not None:
                    raise RuntimeError(
                        f"campaign {campaign_seed} {case.name} "
                        f"{termination_reason}; cleanup={cleanup}; see {stderr_path}"
                    )
                if returncode:
                    raise RuntimeError(
                        f"campaign {campaign_seed} {case.name} failed with "
                        f"rc={returncode}; see {stderr_path}"
                    )
                if not output_path.is_file():
                    raise RuntimeError(
                        f"campaign {campaign_seed} {case.name} exited without "
                        f"{output_path}"
                    )
    except Exception:
        stop_event.set()
        raise


def _case_from_worker_payload(payload: dict[str, Any]) -> Case:
    shape = payload["shape"]
    variant = "causal" if shape["causal"] else "dense"
    case = CASES.get((variant, shape["seq_len"]))
    if case is None:
        raise RuntimeError(f"unexpected worker result shape: {shape}")
    return case


def _validate_post_timing_peaky_logits(
    correctness: dict[str, Any], key: tuple[int, str]
) -> None:
    peaky = correctness.get("post_timing_peaky_logits")
    if not isinstance(peaky, dict):
        raise RuntimeError(f"{key}: missing post-timing peaky-logit validation")
    expected_policy = {
        "performed_after_timing": True,
        "q_scale_in_place": 2.0,
        "k_scale_in_place": 2.0,
        "v_mutated": False,
    }
    actual_policy = {field: peaky.get(field) for field in expected_policy}
    if actual_policy != expected_policy:
        raise RuntimeError(
            f"{key}: invalid post-timing peaky-logit policy: "
            f"expected {expected_policy!r}, got {actual_policy!r}"
        )

    numerics = peaky.get("helion_vs_cudnn_sdpa")
    if not isinstance(numerics, dict):
        raise RuntimeError(f"{key}: missing peaky-logit numerical metrics")
    if numerics.get("thresholds") != PEAKY_STRESS_THRESHOLDS:
        raise RuntimeError(f"{key}: peaky-logit thresholds changed")
    if (
        numerics.get("atol") != PEAKY_STRESS_THRESHOLDS["atol"]
        or numerics.get("rtol") != PEAKY_STRESS_THRESHOLDS["rtol"]
        or numerics.get("finite_outputs") is not True
        or numerics.get("passed") is not True
        or numerics.get("actual_nonfinite") != 0
        or numerics.get("expected_nonfinite") != 0
    ):
        raise RuntimeError(f"{key}: peaky-logit numerical validation failed")
    for metric, threshold in (
        ("max_abs", "max_abs_exclusive"),
        ("nrmse", "nrmse_exclusive"),
        ("mismatch_fraction", "mismatch_fraction_exclusive"),
    ):
        value = numerics.get(metric)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
            or value >= PEAKY_STRESS_THRESHOLDS[threshold]
        ):
            raise RuntimeError(
                f"{key}: peaky-logit {metric} failed exclusive gate "
                f"{PEAKY_STRESS_THRESHOLDS[threshold]}: {value!r}"
            )

    repeat = peaky.get("helion_exact_repeatability")
    if (
        not isinstance(repeat, dict)
        or repeat.get("passed") is not True
        or repeat.get("different") != 0
    ):
        raise RuntimeError(f"{key}: peaky-logit exact repeatability failed")


def _implementation_stats(values: list[float], flops: float) -> dict[str, Any]:
    best_ms = min(values)
    median_ms = statistics.median(values)
    return {
        "best_ms": best_ms,
        "best_tflops": flops / (best_ms * 1e9),
        "mean_ms": statistics.fmean(values),
        "median_ms": median_ms,
        "median_tflops": flops / (median_ms * 1e9),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "runs_ms": values,
    }


def _combined_shape_summary(
    campaigns: list[dict[str, Any]],
    *,
    case: Case,
    flops: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for timer_index, timer in enumerate(("event", "wall")):
        helion_values = [
            pair["times"]["helion"][f"{timer}_ms"]
            for campaign in campaigns
            for pair in campaign["raw_pairs"]
        ]
        sdpa_values = [
            pair["times"]["sdpa"][f"{timer}_ms"]
            for campaign in campaigns
            for pair in campaign["raw_pairs"]
        ]
        strata = [
            [
                math.log(
                    pair["times"]["sdpa"][f"{timer}_ms"]
                    / pair["times"]["helion"][f"{timer}_ms"]
                )
                for pair in campaign["raw_pairs"]
            ]
            for campaign in campaigns
        ]
        log_ratios = [value for stratum in strata for value in stratum]
        paired_log_ratio_pct = 100.0 * math.expm1(statistics.fmean(log_ratios))
        derived_seed = (
            bootstrap_seed
            + case.seq_len * 2
            + int(case.causal)
            + timer_index * 10_000_000
        )
        rng = random.Random(derived_seed)
        bootstrap_values = []
        for _ in range(bootstrap_samples):
            sampled = [
                stratum[rng.randrange(len(stratum))]
                for stratum in strata
                for _ in stratum
            ]
            bootstrap_values.append(100.0 * math.expm1(statistics.fmean(sampled)))
        bootstrap_values.sort()
        helion = _implementation_stats(helion_values, flops)
        sdpa = _implementation_stats(sdpa_values, flops)
        summary[timer] = {
            "helion": helion,
            "sdpa": sdpa,
            "median_throughput_ratio_pct": 100.0
            * (helion["median_tflops"] / sdpa["median_tflops"] - 1.0),
            "paired_log_ratio_pct": paired_log_ratio_pct,
            "paired_log_ratio_stratified_bootstrap_95_ci_pct": [
                bootstrap_values[int(bootstrap_samples * 0.025)],
                bootstrap_values[int(bootstrap_samples * 0.975)],
            ],
            "paired_log_ratio_stratified_bootstrap_seed": derived_seed,
        }
    return summary


def aggregate_results(
    output_dir: Path,
    campaign_seeds: tuple[int, int],
    bootstrap_samples: int,
    bootstrap_seed: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    harness_sha256 = manifest.get("harness_sha256")
    if not isinstance(harness_sha256, dict):
        raise RuntimeError("run manifest is missing paired harness identities")
    strict_artifacts = manifest.get("strict_artifacts")
    if not isinstance(strict_artifacts, dict):
        raise RuntimeError("run manifest is missing strict artifact identities")
    manifest_records: dict[tuple[int, str], dict[str, Any]] = {}
    for record in manifest.get("records", []):
        key = (record.get("campaign_seed"), record.get("case"))
        if key in manifest_records:
            raise RuntimeError(f"duplicate run-manifest record {key}")
        manifest_records[key] = record
    payloads = [
        json.loads(path.read_text())
        for path in sorted((output_dir / "campaigns").glob("seed_*/results/*.json"))
    ]
    expected_keys = {
        (campaign_seed, case.name)
        for campaign_seed in campaign_seeds
        for case in CASES.values()
    }
    if set(manifest_records) != expected_keys:
        raise RuntimeError(
            f"run-manifest record set mismatch: expected {sorted(expected_keys)}, "
            f"got {sorted(manifest_records)}"
        )
    actual_keys: set[tuple[int, str]] = set()
    by_case: dict[Case, list[dict[str, Any]]] = {case: [] for case in CASES.values()}
    for payload in payloads:
        worker_schema_version = payload.get("schema_version")
        if (
            type(worker_schema_version) is not int
            or worker_schema_version != EXPECTED_WORKER_SCHEMA_VERSION
        ):
            raise RuntimeError(
                "worker payload schema mismatch: expected "
                f"{EXPECTED_WORKER_SCHEMA_VERSION}, got {worker_schema_version!r}"
            )
        if payload.get("status") != "PASS":
            raise RuntimeError(f"non-passing worker payload: {payload.get('status')}")
        case = _case_from_worker_payload(payload)
        campaign_seed = payload.get("campaign_seed")
        key = (campaign_seed, case.name)
        if key in actual_keys:
            raise RuntimeError(f"duplicate worker result {key}")
        actual_keys.add(key)
        expected_worker_harness = {
            "paired_worker.py": harness_sha256.get("paired_worker.py")
        }
        if payload.get("harness_sha256") != expected_worker_harness:
            raise RuntimeError(
                f"{key}: worker harness identity changed: expected "
                f"{expected_worker_harness!r}, got {payload.get('harness_sha256')!r}"
            )
        expected_identity = strict_artifacts.get(case.name)
        actual_identity = {
            field: payload.get("provenance", {}).get(field)
            for field in STRICT_ARTIFACT_IDENTITY_FIELDS
        }
        if actual_identity != expected_identity:
            raise RuntimeError(
                f"{key}: worker strict artifact identity changed: "
                f"expected {expected_identity!r}, got {actual_identity!r}"
            )
        if (
            payload.get("regenerated_kernel", {}).get("source_hash_matches_search")
            is not True
        ):
            raise RuntimeError(f"{key}: generated source did not match search winner")
        regenerated_source_sha256 = payload["regenerated_kernel"].get(
            "regenerated_source_sha256"
        )
        if (
            manifest_records[key].get("generated_source_sha256")
            != regenerated_source_sha256
        ):
            raise RuntimeError(
                f"{key}: archived generated source digest does not match worker "
                f"result: expected {regenerated_source_sha256!r}, got "
                f"{manifest_records[key].get('generated_source_sha256')!r}"
            )
        correctness = payload["correctness"]
        if not correctness["helion_vs_cudnn_sdpa"]["passed"]:
            raise RuntimeError(f"{key}: numerical correctness failed")
        if not all(
            repeat["passed"] for repeat in correctness["helion_exact_repeatability"]
        ):
            raise RuntimeError(f"{key}: exact repeatability failed")
        _validate_post_timing_peaky_logits(correctness, key)
        by_case[case].append(payload)
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"result set mismatch: expected {sorted(expected_keys)}, "
            f"got {sorted(actual_keys)}"
        )

    campaign_numbers = {
        campaign_seed: index
        for index, campaign_seed in enumerate(campaign_seeds, start=1)
    }
    shape_results = []
    for case in sorted(
        CASES.values(), key=lambda value: (value.variant, value.seq_len)
    ):
        campaigns = sorted(by_case[case], key=operator.itemgetter("campaign_seed"))
        full_provenance = campaigns[0]["provenance"]
        for campaign in campaigns[1:]:
            if campaign["provenance"] != full_provenance:
                raise RuntimeError(
                    f"{case.name}: search provenance changed by campaign"
                )
        provenance = portable_search_provenance(full_provenance)
        flops = campaigns[0]["flop_model"]["flops"]
        shape_results.append(
            {
                "shape": campaigns[0]["shape"],
                "flop_model": campaigns[0]["flop_model"],
                "provenance": provenance,
                "regenerated_kernels": [
                    {
                        "campaign": campaign_numbers[campaign["campaign_seed"]],
                        **campaign["regenerated_kernel"],
                        "generated_source_path": manifest_records[
                            (campaign["campaign_seed"], case.name)
                        ]["generated_source"],
                    }
                    for campaign in campaigns
                ],
                "correctness": [
                    {
                        "campaign": campaign_numbers[campaign["campaign_seed"]],
                        **campaign["correctness"],
                    }
                    for campaign in campaigns
                ],
                "environment": portable_worker_environment(campaigns[0]["environment"]),
                "campaign_environments": [
                    {
                        "campaign": campaign_numbers[campaign["campaign_seed"]],
                        **portable_worker_environment(campaign["environment"]),
                    }
                    for campaign in campaigns
                ],
                "raw_campaigns": [
                    {
                        "campaign": campaign_numbers[campaign["campaign_seed"]],
                        "orchestrator_seed": campaign["campaign_seed"],
                        "protocol_seed": campaign["protocol"]["input_seed"],
                        "raw_pairs": campaign["raw_pairs"],
                    }
                    for campaign in campaigns
                ],
                "summary": _combined_shape_summary(
                    campaigns,
                    case=case,
                    flops=flops,
                    bootstrap_samples=bootstrap_samples,
                    bootstrap_seed=bootstrap_seed,
                ),
            }
        )

    aggregates: dict[str, Any] = {}
    for timer in ("event", "wall"):
        timer_payload: dict[str, Any] = {}
        for group_name, selected in (
            (
                "dense",
                [result for result in shape_results if not result["shape"]["causal"]],
            ),
            (
                "causal",
                [result for result in shape_results if result["shape"]["causal"]],
            ),
            ("overall", shape_results),
        ):
            helion_tflops = [
                result["summary"][timer]["helion"]["median_tflops"]
                for result in selected
            ]
            sdpa_tflops = [
                result["summary"][timer]["sdpa"]["median_tflops"] for result in selected
            ]
            ratios = [
                helion / sdpa
                for helion, sdpa in zip(helion_tflops, sdpa_tflops, strict=True)
            ]
            timer_payload[group_name] = {
                "shape_count": len(selected),
                "helion_geomean_tflops": statistics.geometric_mean(helion_tflops),
                "sdpa_geomean_tflops": statistics.geometric_mean(sdpa_tflops),
                "helion_vs_sdpa_geomean_ratio": statistics.geometric_mean(ratios),
                "helion_vs_sdpa_geomean_pct": 100.0 * statistics.geometric_mean(ratios),
            }
        aggregates[timer] = timer_payload

    campaign_records = []
    for campaign_seed in campaign_seeds:
        records = [
            record
            for record in manifest["records"]
            if record["campaign_seed"] == campaign_seed
        ]
        campaign_records.append(
            {
                "campaign": campaign_numbers[campaign_seed],
                "orchestrator_seed": campaign_seed,
                "started_ns": min(record["started_ns"] for record in records),
                "finished_ns": max(record["finished_ns"] for record in records),
                "initial_gpu_preflight": manifest["gpu_preflight"],
                "result_sha256": {
                    record["case"]: record["output_sha256"] for record in records
                },
                "generated_source_sha256": {
                    record["case"]: record["generated_source_sha256"]
                    for record in records
                },
            }
        )
    combined = {
        "schema_version": 4,
        "status": "PASS",
        "harness_sha256": harness_sha256,
        "strict_artifacts": strict_artifacts,
        "protocol": {
            "name": "two independent balanced randomized paired raw single-call campaigns",
            "campaign_count": len(campaign_seeds),
            "campaign_seeds": list(campaign_seeds),
            "pairs_per_shape_per_campaign": len(
                shape_results[0]["raw_campaigns"][0]["raw_pairs"]
            ),
            "combined_pairs_per_shape": sum(
                len(campaign["raw_pairs"])
                for campaign in shape_results[0]["raw_campaigns"]
            ),
            "bootstrap_method": (
                "resample paired log ratios within each campaign/shape stratum"
            ),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_base_seed": bootstrap_seed,
            "process_isolation": "one fresh worker per shape and campaign",
            "dense_physical_gpu": 7,
            "causal_physical_gpu": 6,
            "power_cap_w": EXPECTED_POWER_CAP_W,
            "dtype": "float16",
            "timers": ["cuda_event", "host_wall"],
            "forced_sdpa_backend": "CUDNN_ATTENTION",
            "correctness_before_timing": True,
            "post_timing_peaky_logits": {
                "performed_after_timing": True,
                "q_scale_in_place": 2.0,
                "k_scale_in_place": 2.0,
                "v_mutated": False,
                "thresholds": PEAKY_STRESS_THRESHOLDS,
                "exact_repeatability": True,
            },
            "full_output_tolerance": {"atol": 0.05, "rtol": 0.02},
            "helion_exact_repeat_runs_per_campaign": len(
                shape_results[0]["correctness"][0]["helion_exact_repeatability"]
            )
            + 1,
            "thermal_warmup_seconds": payloads[0]["protocol"]["thermal_warmup_seconds"],
            "warmup_calls_per_implementation": payloads[0]["protocol"][
                "warmup_calls_per_implementation"
            ],
        },
        "campaigns": campaign_records,
        "results": shape_results,
        "aggregate": aggregates,
    }
    require_portable_json(combined)
    return combined


def finalize_run(
    output_dir: Path,
    campaign_seeds: tuple[int, int],
    bootstrap_samples: int,
    bootstrap_seed: int,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = output_dir / "run_manifest.json"
    try:
        validate_harness_sha256(manifest.get("harness_sha256"))
        aggregate = aggregate_results(
            output_dir,
            campaign_seeds,
            bootstrap_samples,
            bootstrap_seed,
            manifest,
        )
        final_manifest = {
            **manifest,
            "status": "PASS",
            "finished_ns": time.time_ns(),
        }
        manifest_sha256 = json_payload_sha256(final_manifest)
        aggregate["run_manifest_sha256"] = manifest_sha256
        aggregate["static_validation_sha256"] = sha256(
            output_dir / "static_validation.json"
        )

        # Publish PASS last. An interrupted finalization therefore leaves the
        # manifest in FINALIZING rather than certifying an incomplete aggregate.
        atomic_write_json(output_dir / "all8_paired_raw.json", aggregate)
        atomic_write_json(manifest_path, final_manifest)
        if sha256(manifest_path) != manifest_sha256:
            raise RuntimeError("final run manifest digest changed while publishing")
        return aggregate
    except BaseException as exc:
        failure_manifest = {
            **manifest,
            "status": "FAIL",
            "finished_ns": time.time_ns(),
            "errors": [
                *manifest.get("errors", []),
                f"finalization failed: {exc!r}",
            ],
        }
        atomic_write_json(manifest_path, failure_manifest)
        raise


def planned_commands(
    args: argparse.Namespace,
    output_dir: Path,
    campaign_seeds: tuple[int, int],
) -> list[dict[str, Any]]:
    plans = []
    for campaign_seed in campaign_seeds:
        for physical_gpu, cases in LANES.items():
            uuid = args.gpu6_uuid if physical_gpu == 6 else args.gpu7_uuid
            for case in cases:
                output_path = result_path(output_dir, campaign_seed, case)
                plans.append(
                    {
                        "campaign_seed": campaign_seed,
                        "worker_input_seed": worker_seed(campaign_seed, case),
                        "case": case.name,
                        "physical_gpu": physical_gpu,
                        "gpu_uuid": uuid,
                        "cuda_visible_devices": uuid,
                        "command": worker_command(
                            args,
                            campaign_seed,
                            physical_gpu,
                            uuid,
                            case,
                            output_path,
                        ),
                    }
                )
    return plans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--campaign-seed",
        type=int,
        action="append",
        help="repeat exactly twice; defaults to two recorded campaign seeds",
    )
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--thermal-warmup-seconds", type=float, default=10.0)
    parser.add_argument("--race-runs", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_SECONDS,
    )
    parser.add_argument("--gpu6-uuid", default=EXPECTED_UUIDS[6])
    parser.add_argument("--gpu7-uuid", default=EXPECTED_UUIDS[7])
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.harness_sha256 = current_harness_sha256()
    campaign_seeds = tuple(args.campaign_seed or DEFAULT_CAMPAIGN_SEEDS)
    if len(campaign_seeds) != 2 or len(set(campaign_seeds)) != 2:
        raise SystemExit("exactly two distinct --campaign-seed values are required")
    if args.pairs <= 0 or args.pairs % 2:
        raise SystemExit("--pairs must be a positive even number")
    if args.warmup_calls < 0:
        raise SystemExit("--warmup-calls must be nonnegative")
    if args.thermal_warmup_seconds < 0:
        raise SystemExit("--thermal-warmup-seconds must be nonnegative")
    if args.race_runs < 2:
        raise SystemExit("--race-runs must be at least 2")
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    if args.worker_timeout_seconds <= 0:
        raise SystemExit("--worker-timeout-seconds must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for campaign_seed in campaign_seeds:
        campaign_dir = args.output_dir / "campaigns" / f"seed_{campaign_seed}"
        (campaign_dir / "results").mkdir(parents=True)
        (campaign_dir / "logs").mkdir()
        (campaign_dir / "generated_sources").mkdir()

    static_provenance = validate_artifact_set(args.artifact_root)
    strict_artifacts = strict_artifact_identities(static_provenance)
    artifact_root_reference = logical_artifact_root_reference(
        args.output_dir, args.artifact_root
    )
    plans = planned_commands(args, args.output_dir, campaign_seeds)
    static_payload = {
        "schema_version": 5,
        "status": "VALIDATED_STATIC_ONLY" if args.validate_only else "READY",
        "artifact_root": artifact_root_reference,
        "artifact_root_at_run": str(args.artifact_root.resolve()),
        "strict_artifacts": strict_artifacts,
        "harness_sha256": args.harness_sha256,
        "runtime_checkout": str(REPO_ROOT),
        "campaign_seeds": list(campaign_seeds),
        "parallel_lanes": {
            "dense": {"physical_gpu": 7, "gpu_uuid": args.gpu7_uuid},
            "causal": {"physical_gpu": 6, "gpu_uuid": args.gpu6_uuid},
        },
        "worker_environment_policy": {
            "scrubbed_prefixes": list(SCRUBBED_ENV_PREFIXES),
            "scrubbed_names": sorted(SCRUBBED_ENV_NAMES),
            "controlled_values": {
                "HELION_BACKEND": "cute",
                "HELION_CACHE_DIR": "fresh per shape and campaign",
                "HELION_DISABLE_AUTOTUNER_HEURISTICS": "0",
                "PYTHONHASHSEED": "0",
                "CUDA_VISIBLE_DEVICES": "per-lane expected GPU UUID",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONPATH": str(REPO_ROOT),
                "PYTHONPYCACHEPREFIX": "fresh per shape and campaign",
            },
        },
        "worker_timeout_seconds": args.worker_timeout_seconds,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "base_seed": args.bootstrap_seed,
            "method": "resample paired log ratios within each campaign/shape stratum",
        },
        "planned_commands": plans,
        "provenance": {
            case.name: provenance
            for case, provenance in sorted(
                static_provenance.items(), key=lambda item: item[0].name
            )
        },
    }
    atomic_write_json(args.output_dir / "static_validation.json", static_payload)
    if args.validate_only:
        return

    gpu_preflight = {
        "6": validate_live_gpu(6, args.gpu6_uuid),
        "7": validate_live_gpu(7, args.gpu7_uuid),
    }
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    stop_event = threading.Event()
    termination_signals: list[int] = []

    def handle_termination(signum: int, frame: object) -> None:
        termination_signals.append(signum)
        stop_event.set()

    started_ns = time.time_ns()
    errors = []
    previous_sigterm = signal.signal(signal.SIGTERM, handle_termination)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_lane,
                    args,
                    campaign_seeds,
                    physical_gpu,
                    args.gpu6_uuid if physical_gpu == 6 else args.gpu7_uuid,
                    args.output_dir,
                    records,
                    records_lock,
                    stop_event,
                )
                for physical_gpu in (6, 7)
            ]
            try:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append(repr(exc))
            except BaseException:
                stop_event.set()
                raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    if termination_signals:
        errors.append(f"coordinator received signals: {termination_signals}")
    manifest = {
        "schema_version": 5,
        "status": "FAIL" if errors else "FINALIZING",
        "artifact_root": artifact_root_reference,
        "artifact_root_at_run": str(args.artifact_root.resolve()),
        "strict_artifacts": strict_artifacts,
        "harness_sha256": args.harness_sha256,
        "campaign_seeds": list(campaign_seeds),
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
        "gpu_preflight": gpu_preflight,
        "records": sorted(records, key=operator.itemgetter("campaign_seed", "case")),
        "termination_signals": termination_signals,
        "errors": errors,
    }
    atomic_write_json(args.output_dir / "run_manifest.json", manifest)
    if errors:
        raise SystemExit("; ".join(errors))
    finalize_run(
        args.output_dir,
        campaign_seeds,
        args.bootstrap_samples,
        args.bootstrap_seed,
        manifest,
    )


if __name__ == "__main__":
    main()
