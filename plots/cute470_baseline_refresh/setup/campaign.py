from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import hashlib
from importlib import metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any

EXPECTED_MEASURED_COMMIT = "c3e36b65d69681c23e053042b0bc21e2331bad17"
EXPECTED_CUTE_VERSION = "4.7.0"
EXPECTED_KA_SELECTION_CUTE_VERSION = "4.5.1"
EXPECTED_KA_VERSION = "v3-20260730"
EXPECTED_KA_MODEL = "gpt-5.6-sol"
EXPECTED_KA_MODEL_LABEL = "GPT-5.6"
EXPECTED_FA4_COMMIT = "2409214a03797b168f648ea30df1adbc09ce658a"
EXPECTED_FA4_DESCRIBE = "fa4-v4.0.0.beta23"
EXPECTED_QUACK_COMMIT = "b5b49dae477d39cb8ea8cca2820ef09ba548c72c"
EXPECTED_TORCH_VERSION = "2.13.0.dev20260506+cu130"
EXPECTED_TRITON_VERSION = "3.7.0+git88b227e2"
EXPECTED_CUDNN_RUNTIME_VERSION = "9.20.0"
EXPECTED_CUDNN_PACKAGE_VERSION = "9.20.0.48"
EXPECTED_TVM_FFI_VERSION = "0.1.11"
EXPECTED_CUDA_BINDINGS_VERSION = "13.2.0"
EXPECTED_GPU_NAME = "NVIDIA B200"
EXPECTED_GPU_UUIDS = {
    6: "GPU-b95967ee-1ea7-3bca-894f-5ec74e1c5513",
    7: "GPU-9e1e775d-008f-d915-e4a1-80cde2e60a7e",
}
EXPECTED_POWER_CAP_W = 750
INPUT_SEED = 2026080106
SAMPLE_COUNT = 9
WARMUP_MS = 1000
REP_MS = 500

BASELINE_IMPLEMENTATIONS = ("fa4", "flexattention-cute")
KA_IMPLEMENTATIONS = ("kernelagent-closed-1x", "kernelagent-closed-2x")
EXPECTED_PAYLOAD_IMPLEMENTATIONS = (
    "helion-triton",
    "helion-tileir",
    "flexattention",
    "gluon",
    "tlx",
    "flexattention-cute",
    "fa4",
    "sdpa",
    "helion-cute",
    "kernelagent-1x",
    "kernelagent-2x",
    "kernelagent-10x",
    "kernelagent-closed-1x",
    "kernelagent-closed-2x",
)
DIST_NAMES = (
    "apache-tvm-ffi",
    "torch",
    "triton",
    "nvidia-cutlass-dsl",
    "nvidia-cudnn-cu13",
    "cuda-bindings",
)
TOOL_NAMES = ("nvidia-smi", "ptxas", "nvcc", "gcc", "g++")
MODULE_NAMES = (
    "torch",
    "triton",
    "cutlass",
    "tvm_ffi",
    "torch.backends.cudnn",
    "cuda.bindings",
)


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    order: int
    variant: str
    seq_len: int
    physical_gpu: int

    @property
    def case_id(self) -> str:
        return f"{self.variant}_{self.seq_len}"

    @property
    def causal(self) -> int:
        return int(self.variant == "causal")

    @property
    def shape(self) -> dict[str, object]:
        return {
            "z": 2,
            "h": 32,
            "seq_len": self.seq_len,
            "head_dim": 64,
            "dtype": "float16",
            "causal": self.causal,
            "biased": 0,
        }


@dataclass(frozen=True)
class KernelAgentRun:
    case_id: str
    budget: str
    source_sha256: str

    @property
    def implementation(self) -> str:
        return f"kernelagent-closed-{self.budget}"

    @property
    def run_id(self) -> str:
        return f"{self.case_id}_{self.budget}"


CASES = (
    Case(1, "dense", 32768, 7),
    Case(2, "dense", 65536, 7),
    Case(3, "dense", 131072, 7),
    Case(4, "dense", 262144, 7),
    Case(5, "causal", 65536, 6),
    Case(6, "causal", 131072, 6),
    Case(7, "causal", 262144, 6),
    Case(8, "causal", 524288, 6),
)
CASES_BY_ID = {case.case_id: case for case in CASES}
KA_RUNS = (
    KernelAgentRun(
        "dense_131072",
        "1x",
        "ce1623b644b08820e57aaafc61f4837a2899a8e5496c189d9867457380a44964",
    ),
    KernelAgentRun(
        "dense_131072",
        "2x",
        "074f22a17bd02069240d9a524920f6e2bcf870e8879ae0aa411d499e8c0ffffe",
    ),
    KernelAgentRun(
        "dense_262144",
        "1x",
        "3c73e42060837b2875cefa88d1482798ed558c5ddbb40dd8dcffb98aa4e81582",
    ),
    KernelAgentRun(
        "causal_65536",
        "1x",
        "42344665bc37faf98a7ef6284390dd860ab65bc1bec9e47e9bdbf4b80e0d7c28",
    ),
    KernelAgentRun(
        "causal_65536",
        "2x",
        "1b5f13ed3f51d76b419738753e290efc4790df2a6adbbde8cb8a384e5158cb2d",
    ),
    KernelAgentRun(
        "causal_131072",
        "1x",
        "b0e1db988031c6059ed19b4ad4b8cabc56453b9f35f8389aa0a916c70b4a44ac",
    ),
    KernelAgentRun(
        "causal_262144",
        "1x",
        "20e4d29a206326cdb7bc1e99914d7dcf2c7400c2dfff5bf8d9a1797b98f77ee6",
    ),
    KernelAgentRun(
        "causal_262144",
        "2x",
        "ad81863031d867e68e467aae9b49ed82b9c3b768a9e7b5d5b70847b3d601b5fc",
    ),
    KernelAgentRun(
        "causal_524288",
        "1x",
        "a31f126ded320f9684efe192345314078c87e2d7217e026e8eb8ee0b4d9a6336",
    ),
)
KA_RUNS_BY_KEY = {(run.case_id, run.implementation): run for run in KA_RUNS}


def _require(condition: object, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode())


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise CampaignError(
            f"command failed: {' '.join(command)}\n{stderr.strip()}"
        ) from exc


def _git(root: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(root), *arguments]).stdout.strip()


def _validate_checkout(root: Path, expected_commit: str) -> None:
    _require(root.is_dir(), f"missing checkout: {root}")
    _require(
        _git(root, "rev-parse", "HEAD") == expected_commit,
        f"checkout {root.name} is not at {expected_commit}",
    )
    _require(not _git(root, "diff", "--name-only"), f"{root.name} has changes")
    _require(
        not _git(root, "diff", "--cached", "--name-only"),
        f"{root.name} has staged changes",
    )
    untracked = _git(root, "ls-files", "--others", "--exclude-standard")
    _require(not untracked, f"{root.name} has untracked files: {untracked}")
    ignored = _git(
        root,
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
    )
    _require(not ignored, f"{root.name} has ignored files: {ignored}")


def _ensure_worktree(source: Path, destination: Path, commit: str) -> None:
    if destination.exists():
        _validate_checkout(destination, commit)
        return
    _require(source.is_dir(), f"source repository does not exist: {source}")
    resolved = _git(source, "rev-parse", "--verify", f"{commit}^{{commit}}")
    _require(resolved == commit, f"{source.name} cannot resolve commit {commit}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "-C",
            str(source),
            "worktree",
            "add",
            "--detach",
            str(destination),
            commit,
        ]
    )
    _validate_checkout(destination, commit)


def _validate_external_output(output_root: Path, source: Path) -> None:
    _require(
        output_root != source and source not in output_root.parents,
        f"output root must be outside source repository {source.name}",
    )


def _source_record(root: Path, kind: str) -> dict[str, str]:
    record = {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "describe": _git(root, "describe", "--tags", "--always", "--dirty"),
    }
    if kind == "helion":
        record["benchmark_sha256"] = _sha256_file(
            root / "benchmarks/cute/compare_attention_backends.py"
        )
    elif kind == "fa4":
        record["flash_attn_cute_tree"] = _git(root, "rev-parse", "HEAD:flash_attn/cute")
    elif kind == "quack":
        record["quack_package_tree"] = _git(root, "rev-parse", "HEAD:quack")
    else:
        raise AssertionError(kind)
    return record


def _distribution_record(name: str) -> dict[str, str]:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError as exc:
        raise CampaignError(f"missing required distribution: {name}") from exc
    record_text = distribution.read_text("RECORD")
    if record_text is None:
        raise CampaignError(f"distribution {name} has no RECORD")
    return {
        "version": distribution.version,
        "record_sha256": _sha256_bytes(record_text.encode()),
    }


def _tool_record(name: str) -> dict[str, str] | None:
    executable = shutil.which(name)
    if executable is None:
        return None
    path = Path(executable).resolve()
    return {"name": path.name, "sha256": _sha256_file(path)}


def _module_record(name: str) -> dict[str, str]:
    if name == "torch.backends.cudnn":
        import torch

        origin_value = torch.backends.cudnn.__file__
    else:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise CampaignError(f"required module has no import origin: {name}")
        origin_value = spec.origin
    if origin_value is None:
        raise CampaignError(f"required module has no import origin: {name}")
    origin = Path(origin_value).resolve()
    _require(origin.is_file(), f"required module origin is not a file: {name}")
    prefix = Path(sys.prefix).resolve()
    try:
        relative = origin.relative_to(prefix).as_posix()
    except ValueError as exc:
        raise CampaignError(f"module {name} is outside the Python prefix") from exc
    return {
        "module": name,
        "origin": relative,
        "sha256": _sha256_file(origin),
    }


def _distribution_file_record(distribution_name: str, relative: str) -> dict[str, str]:
    distribution = metadata.distribution(distribution_name)
    path = Path(str(distribution.locate_file(relative))).resolve()
    _require(path.is_file(), f"missing {distribution_name} artifact: {relative}")
    return {
        "distribution": distribution_name,
        "path": relative,
        "sha256": _sha256_file(path),
    }


def _toolchain_record() -> dict[str, object]:
    import torch

    cudnn_value = torch.backends.cudnn.version()
    cudnn_runtime = (
        "unknown"
        if cudnn_value is None
        else f"{cudnn_value // 10000}.{(cudnn_value // 100) % 100}.{cudnn_value % 100}"
    )
    executable = Path(sys.executable).resolve()
    return {
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable_name": executable.name,
            "executable_sha256": _sha256_file(executable),
        },
        "packages": {name: _distribution_record(name) for name in DIST_NAMES},
        "modules": {name: _module_record(name) for name in MODULE_NAMES},
        "cudnn_library": _distribution_file_record(
            "nvidia-cudnn-cu13", "nvidia/cudnn/lib/libcudnn.so.9"
        ),
        "cudnn_runtime_version": cudnn_runtime,
        "tools": {
            name: record
            for name in TOOL_NAMES
            if (record := _tool_record(name)) is not None
        },
    }


def _validate_toolchain(record: dict[str, object]) -> None:
    packages = record["packages"]
    assert isinstance(packages, dict)
    expected = {
        "apache-tvm-ffi": EXPECTED_TVM_FFI_VERSION,
        "torch": EXPECTED_TORCH_VERSION,
        "triton": EXPECTED_TRITON_VERSION,
        "nvidia-cutlass-dsl": EXPECTED_CUTE_VERSION,
        "nvidia-cudnn-cu13": EXPECTED_CUDNN_PACKAGE_VERSION,
        "cuda-bindings": EXPECTED_CUDA_BINDINGS_VERSION,
    }
    for name, version in expected.items():
        _require(packages[name]["version"] == version, f"expected {name} {version}")
    _require(
        record["cudnn_runtime_version"] == EXPECTED_CUDNN_RUNTIME_VERSION,
        f"expected cuDNN runtime {EXPECTED_CUDNN_RUNTIME_VERSION}",
    )
    tools = record["tools"]
    assert isinstance(tools, dict)
    _require(
        set(TOOL_NAMES) <= set(tools),
        f"missing compiler tools: {sorted(set(TOOL_NAMES) - set(tools))}",
    )
    modules = record["modules"]
    assert isinstance(modules, dict)
    _require(set(modules) == set(MODULE_NAMES), "module provenance is incomplete")


def _probe_gpus() -> list[dict[str, object]]:
    result = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,power.limit,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    selected: dict[int, dict[str, object]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        _require(len(fields) == 5, f"unexpected nvidia-smi row: {line}")
        index = int(fields[0])
        if index in EXPECTED_GPU_UUIDS:
            selected[index] = {
                "physical_gpu": index,
                "uuid": fields[1],
                "name": fields[2],
                "power_limit_w": float(fields[3]),
                "driver_version": fields[4],
            }
    _require(set(selected) == set(EXPECTED_GPU_UUIDS), "GPUs 6 and 7 are required")
    for index, record in selected.items():
        _require(record["uuid"] == EXPECTED_GPU_UUIDS[index], f"GPU {index} UUID")
        _require(record["name"] == EXPECTED_GPU_NAME, f"GPU {index} model")
        _require(
            math.isclose(
                _finite(record["power_limit_w"], f"GPU {index} power limit"),
                EXPECTED_POWER_CAP_W,
            ),
            f"GPU {index} is not at {EXPECTED_POWER_CAP_W} W",
        )
    return [selected[index] for index in sorted(selected)]


def _validate_gpu_idle(physical_gpu: int) -> None:
    result = _run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    active = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    _require(not active, f"physical GPU {physical_gpu} is busy: {active}")


def _clean_environment() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "PATH",
        "LANG",
        "LC_ALL",
    ):
        value = os.environ.get(name)
        if value is not None:
            result[name] = value
    result.setdefault("PATH", os.defpath)
    result.setdefault("LANG", "C.UTF-8")
    result["PYTHONHASHSEED"] = "0"
    result["PYTHONNOUSERSITE"] = "1"
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    result["PYTHONUNBUFFERED"] = "1"
    return result


def _probe_import_roots(
    helion_root: Path, fa4_root: Path, quack_root: Path
) -> dict[str, str]:
    code = """
import hashlib
import importlib.util
import json
import os
from pathlib import Path
helion_root = Path(os.environ['BASELINE_HELION_ROOT']).resolve()
fa4_root = Path(os.environ['BASELINE_FA4_ROOT']).resolve()
quack_root = Path(os.environ['BASELINE_QUACK_ROOT']).resolve()
path = helion_root / 'benchmarks/cute/compare_attention_backends.py'
spec = importlib.util.spec_from_file_location('_baseline_import_probe', path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
fa4 = module._import_fa4()
import helion
import quack
helion_path = Path(helion.__file__).resolve()
fa4_path = Path(fa4.__file__).resolve()
quack_path = Path(quack.__file__).resolve()
if helion_root not in helion_path.parents:
    raise SystemExit(f'Helion imported outside measured root: {helion_path}')
if quack_root not in quack_path.parents:
    raise SystemExit(f'Quack imported outside pinned root: {quack_path}')
if fa4_root not in fa4_path.parents:
    raise SystemExit(f'FA4 imported outside pinned root: {fa4_path}')
print(json.dumps({
    'helion_module': helion_path.relative_to(helion_root).as_posix(),
    'helion_module_sha256': hashlib.sha256(helion_path.read_bytes()).hexdigest(),
    'fa4_module': fa4_path.relative_to(fa4_root).as_posix(),
    'fa4_module_sha256': hashlib.sha256(fa4_path.read_bytes()).hexdigest(),
    'quack_module': quack_path.relative_to(quack_root).as_posix(),
    'quack_module_sha256': hashlib.sha256(quack_path.read_bytes()).hexdigest(),
}))
"""
    env = _clean_environment()
    env["PYTHONPATH"] = os.pathsep.join((str(helion_root), str(quack_root)))
    env["HELION_FA4_ROOT"] = str(fa4_root)
    env["BASELINE_HELION_ROOT"] = str(helion_root)
    env["BASELINE_FA4_ROOT"] = str(fa4_root)
    env["BASELINE_QUACK_ROOT"] = str(quack_root)
    output = _run([sys.executable, "-c", code], cwd=helion_root, env=env).stdout
    try:
        record = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise CampaignError("failed to validate Helion and Quack imports") from exc
    _require(isinstance(record, dict), "invalid import-root record")
    _reject_absolute_strings(record)
    return record


def _expected_result_keys(case: Case) -> tuple[tuple[str, str], ...]:
    keys = [(case.case_id, impl) for impl in BASELINE_IMPLEMENTATIONS]
    keys.extend(
        (case.case_id, run.implementation)
        for run in KA_RUNS
        if run.case_id == case.case_id
    )
    return tuple(keys)


def _case_for_id(case_id: str) -> Case:
    try:
        return CASES_BY_ID[case_id]
    except KeyError as exc:
        raise CampaignError(f"unknown case: {case_id}") from exc


def _validate_historical_payload(payload: object, case: Case, context: str) -> None:
    if not isinstance(payload, dict):
        raise CampaignError(f"{context}: payload is not an object")
    _require(payload.get("shape") == case.shape, f"{context}: shape mismatch")
    results = payload.get("results")
    if not isinstance(results, list):
        raise CampaignError(f"{context}: results are missing")
    implementations = [result.get("impl") for result in results]
    _require(
        tuple(implementations) == EXPECTED_PAYLOAD_IMPLEMENTATIONS,
        f"{context}: implementation order changed",
    )
    _require(
        all(
            isinstance(result, dict) and result.get("shape") == case.shape
            for result in results
        ),
        f"{context}: result shapes differ",
    )
    _require(
        all("input_seed" not in result for result in results),
        f"{context}: historical schema unexpectedly contains input_seed",
    )
    expected_pass = {
        run.implementation for run in KA_RUNS if run.case_id == case.case_id
    }
    actual_pass = {
        str(result["impl"])
        for result in results
        if str(result.get("impl", "")).startswith("kernelagent-closed-")
        and result.get("accuracy") == "PASS"
    }
    _require(
        actual_pass == expected_pass,
        f"{context}: KernelAgent Closed PASS set changed: {actual_pass}",
    )


def _validate_ka_input(
    run: KernelAgentRun, directory: Path, prior_result_path: Path
) -> dict[str, object]:
    case = _case_for_id(run.case_id)
    manifest_path = directory / "manifest.json"
    source_path = directory / "selected_kernel.py.txt"
    for label, path in (
        ("manifest", manifest_path),
        ("source", source_path),
        ("prior result", prior_result_path),
    ):
        _require(
            path.is_file() and not path.is_symlink(),
            f"missing or linked KA {label} for {run.run_id}",
        )
    manifest = json.loads(manifest_path.read_text())
    _require(isinstance(manifest, dict), f"invalid KA manifest for {run.run_id}")
    expected_manifest = {
        "budget_label": run.budget,
        "shape": case.shape,
        "physical_gpu": case.physical_gpu,
        "power_cap_w": EXPECTED_POWER_CAP_W,
        "seed": INPUT_SEED,
        "kernelagent_family": "closed_binary",
        "cutlass_dsl_version": EXPECTED_KA_SELECTION_CUTE_VERSION,
        "kernelagent_version": EXPECTED_KA_VERSION,
        "kernelagent_display_version": EXPECTED_KA_VERSION,
        "model": EXPECTED_KA_MODEL,
        "model_display_name": EXPECTED_KA_MODEL_LABEL,
    }
    for key, expected in expected_manifest.items():
        actual = manifest.get(key)
        _require(actual == expected, f"{run.run_id}: manifest {key} mismatch")
    _require(
        manifest.get("status") in {None, "PASS"},
        f"{run.run_id}: failed or unknown manifest status",
    )
    _finite(manifest.get("budget_seconds"), f"{run.run_id}: budget_seconds")
    _finite(manifest.get("elapsed_seconds"), f"{run.run_id}: elapsed_seconds")
    source_hash = _sha256_file(source_path)
    _require(source_hash == run.source_sha256, f"{run.run_id}: source hash mismatch")
    for container_name, container in (
        ("manifest", manifest),
        ("selection", manifest.get("selection")),
        ("posthoc", manifest.get("posthoc_correctness_validation")),
    ):
        _require(isinstance(container, dict), f"{run.run_id}: missing {container_name}")
        _require(
            container.get("source_sha256") == source_hash,
            f"{run.run_id}: {container_name} source hash mismatch",
        )
    posthoc = manifest["posthoc_correctness_validation"]
    selection = manifest["selection"]
    _require(
        type(selection.get("candidate_id")) is int and selection["candidate_id"] >= 0,
        f"{run.run_id}: invalid selected candidate",
    )
    _finite(selection.get("median_ms"), f"{run.run_id}: selected median_ms")
    _finite(selection.get("tflops"), f"{run.run_id}: selected tflops")
    _require(posthoc.get("status") == "PASS", f"{run.run_id}: posthoc failed")
    _require(
        posthoc.get("standard_seed") == INPUT_SEED
        and posthoc.get("stress_seed") == INPUT_SEED + 1,
        f"{run.run_id}: posthoc seeds mismatch",
    )
    prior = json.loads(prior_result_path.read_text())
    _require(isinstance(prior, dict), f"{run.run_id}: invalid prior result")
    _require(prior.get("accuracy") == "PASS", f"{run.run_id}: prior result failed")
    _require(
        prior.get("impl") == run.implementation,
        f"{run.run_id}: prior implementation mismatch",
    )
    _require(prior.get("shape") == case.shape, f"{run.run_id}: prior shape mismatch")
    _require(
        str(prior.get("physical_gpu")) == str(case.physical_gpu)
        and prior.get("power_cap_w") == EXPECTED_POWER_CAP_W,
        f"{run.run_id}: prior hardware mismatch",
    )
    prior_config = prior.get("config")
    if not isinstance(prior_config, dict):
        raise CampaignError(f"{run.run_id}: prior config is missing")
    _require(
        prior_config.get("source_sha256") == source_hash,
        f"{run.run_id}: prior result source mismatch",
    )
    for key in ("budget_label", "budget_seconds", "elapsed_seconds", "selection"):
        _require(
            prior_config.get(key) == manifest.get(key),
            f"{run.run_id}: prior {key} mismatch",
        )
    return {
        "manifest_sha256": _sha256_file(manifest_path),
        "selected_kernel_sha256": source_hash,
        "prior_result_sha256": _sha256_file(prior_result_path),
        "selection_cute_version": EXPECTED_KA_SELECTION_CUTE_VERSION,
        "model": manifest.get("model"),
        "kernelagent_version": manifest.get("kernelagent_version"),
    }


def _tracked_inputs_clean(source_repo: Path, paths: list[Path]) -> None:
    relative = [
        path.resolve().relative_to(source_repo.resolve()).as_posix() for path in paths
    ]
    _run(["git", "-C", str(source_repo), "ls-files", "--error-unmatch", *relative])
    _run(["git", "-C", str(source_repo), "diff", "--quiet", "--", *relative])
    _run(
        [
            "git",
            "-C",
            str(source_repo),
            "diff",
            "--cached",
            "--quiet",
            "--",
            *relative,
        ]
    )


def _quarantine_path(path: Path, output_root: Path, label: str) -> None:
    if not path.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = output_root / "quarantine" / f"{label}-{stamp}"
    destination = base
    index = 1
    while destination.exists():
        destination = Path(f"{base}-{index}")
        index += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.rename(destination)


def _snapshot_inputs(
    output_root: Path,
    source_repo: Path,
    historical_dir: Path,
    ka_runs_dir: Path,
    ka_results_dir: Path,
) -> None:
    destination = output_root / "inputs"
    if destination.exists():
        return
    expected_directories = {
        "historical payload": source_repo / "plots/kernelagent/results/payloads",
        "KernelAgent run": source_repo / "plots/kernelagent_closed/runs",
        "KernelAgent result": source_repo / "plots/kernelagent_closed/results",
    }
    supplied_directories = {
        "historical payload": historical_dir,
        "KernelAgent run": ka_runs_dir,
        "KernelAgent result": ka_results_dir,
    }
    for label, expected in expected_directories.items():
        _require(
            supplied_directories[label].resolve() == expected.resolve(),
            f"{label} directory must be {expected.relative_to(source_repo)}",
        )
    historical_paths = [historical_dir / f"{case.case_id}.json" for case in CASES]
    ka_paths: list[Path] = []
    for run in KA_RUNS:
        ka_paths.extend(
            (
                ka_runs_dir / run.run_id / "manifest.json",
                ka_runs_dir / run.run_id / "selected_kernel.py.txt",
                ka_results_dir / f"{run.case_id}_kernelagent-closed-{run.budget}.json",
            )
        )
    _tracked_inputs_clean(source_repo, [*historical_paths, *ka_paths])
    pending = output_root / "inputs.pending"
    if pending.exists():
        _quarantine_path(pending, output_root, "inputs-pending")
    (pending / "historical_payloads").mkdir(parents=True)
    (pending / "kernelagent_closed").mkdir(parents=True)
    for case, source in zip(CASES, historical_paths, strict=True):
        _require(source.is_file() and not source.is_symlink(), f"bad input: {source}")
        payload = json.loads(source.read_text())
        _validate_historical_payload(payload, case, source.name)
        shutil.copy2(source, pending / "historical_payloads" / source.name)
    for run in KA_RUNS:
        run_destination = pending / "kernelagent_closed" / run.run_id
        run_destination.mkdir()
        source_directory = ka_runs_dir / run.run_id
        manifest_path = source_directory / "manifest.json"
        source_path = source_directory / "selected_kernel.py.txt"
        prior_path = (
            ka_results_dir / f"{run.case_id}_kernelagent-closed-{run.budget}.json"
        )
        _validate_ka_input(run, source_directory, prior_path)
        shutil.copy2(manifest_path, run_destination / "manifest.json")
        shutil.copy2(source_path, run_destination / "selected_kernel.py.txt")
        shutil.copy2(prior_path, run_destination / "previous_result.json")
    _tracked_inputs_clean(source_repo, [*historical_paths, *ka_paths])
    for case, source in zip(CASES, historical_paths, strict=True):
        copied = pending / "historical_payloads" / source.name
        _require(
            _sha256_file(copied) == _sha256_file(source),
            f"historical payload changed while copying: {case.case_id}",
        )
    for run in KA_RUNS:
        copied = pending / "kernelagent_closed" / run.run_id
        source = ka_runs_dir / run.run_id
        prior = ka_results_dir / f"{run.case_id}_kernelagent-closed-{run.budget}.json"
        _require(
            _sha256_file(copied / "manifest.json")
            == _sha256_file(source / "manifest.json"),
            f"KernelAgent manifest changed while copying: {run.run_id}",
        )
        _require(
            _sha256_file(copied / "selected_kernel.py.txt")
            == _sha256_file(source / "selected_kernel.py.txt"),
            f"KernelAgent source changed while copying: {run.run_id}",
        )
        _require(
            _sha256_file(copied / "previous_result.json") == _sha256_file(prior),
            f"KernelAgent result changed while copying: {run.run_id}",
        )
    os.replace(pending, destination)


def _input_record(output_root: Path) -> dict[str, object]:
    inputs = output_root / "inputs"
    historical: dict[str, object] = {}
    for case in CASES:
        path = inputs / "historical_payloads" / f"{case.case_id}.json"
        payload = json.loads(path.read_text())
        _validate_historical_payload(payload, case, path.name)
        historical[path.name] = {"sha256": _sha256_file(path)}
    ka: dict[str, object] = {}
    for run in KA_RUNS:
        directory = inputs / "kernelagent_closed" / run.run_id
        record = _validate_ka_input(run, directory, directory / "previous_result.json")
        ka[run.run_id] = record
    return {"historical_payloads": historical, "kernelagent_closed": ka}


def _probe_versions(
    helion_root: Path,
    fa4_root: Path,
    quack_root: Path,
    ka_root: Path,
) -> dict[str, object]:
    code = """
import importlib.util
import json
import os
from pathlib import Path
path = Path('benchmarks/cute/compare_attention_backends.py').resolve()
spec = importlib.util.spec_from_file_location('_baseline_version_probe', path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
versions = {
    name: module._implementation_version(name)
    for name in ('fa4', 'flexattention-cute')
}
ka_root = Path(os.environ['BASELINE_KA_ROOT'])
ka = {}
for manifest_path in sorted(ka_root.glob('*/manifest.json')):
    manifest = json.loads(manifest_path.read_text())
    impl = 'kernelagent-closed-' + manifest['budget_label']
    ka[manifest_path.parent.name] = module._kernelagent_version_info(
        impl, manifest, evaluation_backend_version='4.7.0')
print(json.dumps({'baselines': versions, 'kernelagent': ka}))
"""
    env = _clean_environment()
    env["PYTHONPATH"] = os.pathsep.join((str(helion_root), str(quack_root)))
    env["HELION_FA4_ROOT"] = str(fa4_root)
    env["BASELINE_KA_ROOT"] = str(ka_root)
    output = _run([sys.executable, "-c", code], cwd=helion_root, env=env).stdout
    try:
        result = json.loads(output.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise CampaignError("failed to resolve implementation versions") from exc
    _require(isinstance(result, dict), "invalid version probe")
    return result


def _reject_absolute_strings(value: object, context: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_absolute_strings(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_strings(child, f"{context}[{index}]")
    elif isinstance(value, str):
        _require(
            not value.startswith("/") and re.match(r"^[A-Za-z]:[\\/]", value) is None,
            f"absolute path leaked at {context}",
        )


def _definition() -> dict[str, object]:
    return {
        "input_seed": INPUT_SEED,
        "sample_count": SAMPLE_COUNT,
        "warmup_ms": WARMUP_MS,
        "rep_ms": REP_MS,
        "power_cap_w": EXPECTED_POWER_CAP_W,
        "cases": [asdict(case) | {"case_id": case.case_id} for case in CASES],
        "kernelagent_runs": [asdict(run) for run in KA_RUNS],
    }


def _manifest_path(output_root: Path) -> Path:
    return output_root / "campaign_manifest.json"


def _load_manifest(output_root: Path) -> dict[str, Any]:
    path = _manifest_path(output_root)
    _require(path.is_file(), f"missing campaign manifest: {path}")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "campaign manifest is not an object")
    return value


def _create_manifest(output_root: Path, source_repo: Path) -> dict[str, object]:
    helion_root = output_root / "checkouts/helion"
    fa4_root = output_root / "checkouts/flash-attention"
    quack_root = output_root / "checkouts/quack"
    toolchain = _toolchain_record()
    _validate_toolchain(toolchain)
    sources = {
        "helion": _source_record(helion_root, "helion"),
        "flash_attention": _source_record(fa4_root, "fa4"),
        "quack": _source_record(quack_root, "quack"),
        "input_repository": {
            "commit": _git(source_repo, "rev-parse", "HEAD"),
            "tree": _git(source_repo, "rev-parse", "HEAD^{tree}"),
        },
        "setup_files": {
            name: _sha256_file(output_root / "launcher" / name)
            for name in ("campaign.py", "run_campaign.sh")
        },
    }
    _require(
        sources["helion"]["commit"] == EXPECTED_MEASURED_COMMIT,
        "measured Helion commit mismatch",
    )
    _require(
        sources["flash_attention"]["commit"] == EXPECTED_FA4_COMMIT,
        "FA4 commit mismatch",
    )
    _require(
        sources["flash_attention"]["describe"] == EXPECTED_FA4_DESCRIBE,
        "FA4 describe mismatch",
    )
    _require(
        sources["quack"]["commit"] == EXPECTED_QUACK_COMMIT,
        "Quack commit mismatch",
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "campaign": "cute470_baseline_refresh",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_measured_commit": EXPECTED_MEASURED_COMMIT,
        "definition": _definition(),
        "sources": sources,
        "inputs": _input_record(output_root),
        "toolchain": toolchain,
        "hardware": _probe_gpus(),
        "import_roots": _probe_import_roots(helion_root, fa4_root, quack_root),
        "versions": _probe_versions(
            helion_root,
            fa4_root,
            quack_root,
            output_root / "inputs/kernelagent_closed",
        ),
    }
    _reject_absolute_strings(manifest)
    return manifest


def _validate_manifest_static(manifest: dict[str, Any], output_root: Path) -> None:
    _require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    _require(manifest.get("campaign") == "cute470_baseline_refresh", "wrong campaign")
    _require(
        manifest.get("expected_measured_commit") == EXPECTED_MEASURED_COMMIT,
        "measured commit pin changed",
    )
    _require(manifest.get("definition") == _definition(), "definition changed")
    _require(manifest.get("inputs") == _input_record(output_root), "inputs changed")
    for name, expected_hash in manifest["sources"]["setup_files"].items():
        path = output_root / "launcher" / name
        _require(path.is_file(), f"missing launcher file {name}")
        _require(_sha256_file(path) == expected_hash, f"launcher file changed: {name}")
    versions = manifest.get("versions")
    if not isinstance(versions, dict):
        raise CampaignError("missing version manifest")
    baselines = versions.get("baselines")
    kernelagent = versions.get("kernelagent")
    if not isinstance(baselines, dict) or not isinstance(kernelagent, dict):
        raise CampaignError("incomplete version manifest")
    for impl in BASELINE_IMPLEMENTATIONS:
        version = baselines[impl]
        _require(isinstance(version, dict), f"invalid {impl} version record")
        _require(
            f"CuTe {EXPECTED_CUTE_VERSION}" in version["version"],
            f"{impl} does not use CuTe {EXPECTED_CUTE_VERSION}",
        )
    for run in KA_RUNS:
        version = kernelagent[run.run_id]
        _require(isinstance(version, dict), f"invalid {run.run_id} version record")
        _require(
            f"CuTe {EXPECTED_CUTE_VERSION}" in version["version"],
            f"{run.run_id} evaluation version is not CuTe {EXPECTED_CUTE_VERSION}",
        )
        _require(
            f"CuTe {EXPECTED_CUTE_VERSION}" in version["version_label"],
            f"{run.run_id} version label is not CuTe {EXPECTED_CUTE_VERSION}",
        )
    _reject_absolute_strings(manifest)


def _validate_live_state(manifest: dict[str, Any], output_root: Path) -> None:
    _require(_load_manifest(output_root) == manifest, "campaign manifest changed")
    _validate_manifest_static(manifest, output_root)
    helion_root = output_root / "checkouts/helion"
    fa4_root = output_root / "checkouts/flash-attention"
    quack_root = output_root / "checkouts/quack"
    _validate_checkout(helion_root, EXPECTED_MEASURED_COMMIT)
    _validate_checkout(fa4_root, EXPECTED_FA4_COMMIT)
    _validate_checkout(quack_root, EXPECTED_QUACK_COMMIT)
    _require(
        _source_record(helion_root, "helion") == manifest["sources"]["helion"],
        "Helion source changed",
    )
    _require(
        _source_record(fa4_root, "fa4") == manifest["sources"]["flash_attention"],
        "FA4 source changed",
    )
    _require(
        _source_record(quack_root, "quack") == manifest["sources"]["quack"],
        "Quack source changed",
    )
    toolchain = _toolchain_record()
    _validate_toolchain(toolchain)
    _require(toolchain == manifest["toolchain"], "toolchain changed")
    _require(_probe_gpus() == manifest["hardware"], "GPU identity or power changed")
    _require(
        _probe_import_roots(helion_root, fa4_root, quack_root)
        == manifest["import_roots"],
        "import roots changed",
    )
    _require(
        _probe_versions(
            helion_root,
            fa4_root,
            quack_root,
            output_root / "inputs/kernelagent_closed",
        )
        == manifest["versions"],
        "implementation versions changed",
    )
    _require(_load_manifest(output_root) == manifest, "campaign manifest changed")


def initialize(
    output_root: Path,
    source_repo: Path | None,
    fa4_source_repo: Path | None,
    quack_source_repo: Path | None,
    historical_dir: Path | None,
    ka_runs_dir: Path | None,
    ka_results_dir: Path | None,
) -> None:
    output_root = output_root.resolve()
    helion_root = output_root / "checkouts/helion"
    fa4_root = output_root / "checkouts/flash-attention"
    quack_root = output_root / "checkouts/quack"
    if not helion_root.exists():
        if source_repo is None:
            raise CampaignError("Helion source repository is required")
        source_repo = source_repo.resolve()
        _validate_external_output(output_root, source_repo)
        _ensure_worktree(source_repo, helion_root, EXPECTED_MEASURED_COMMIT)
    if not fa4_root.exists():
        if fa4_source_repo is None:
            raise CampaignError("FA4 source repository is required")
        fa4_source_repo = fa4_source_repo.resolve()
        _validate_external_output(output_root, fa4_source_repo)
        _ensure_worktree(fa4_source_repo, fa4_root, EXPECTED_FA4_COMMIT)
    if not quack_root.exists():
        if quack_source_repo is None:
            raise CampaignError("Quack source repository is required")
        quack_source_repo = quack_source_repo.resolve()
        _validate_external_output(output_root, quack_source_repo)
        _ensure_worktree(quack_source_repo, quack_root, EXPECTED_QUACK_COMMIT)
    if not (output_root / "inputs").exists():
        if None in (source_repo, historical_dir, ka_runs_dir, ka_results_dir):
            raise CampaignError("historical and KernelAgent input paths are required")
        assert source_repo is not None
        assert historical_dir is not None
        assert ka_runs_dir is not None
        assert ka_results_dir is not None
        _snapshot_inputs(
            output_root,
            source_repo.resolve(),
            historical_dir.resolve(),
            ka_runs_dir.resolve(),
            ka_results_dir.resolve(),
        )
    _validate_checkout(helion_root, EXPECTED_MEASURED_COMMIT)
    _validate_checkout(fa4_root, EXPECTED_FA4_COMMIT)
    _validate_checkout(quack_root, EXPECTED_QUACK_COMMIT)
    manifest_path = _manifest_path(output_root)
    if manifest_path.exists():
        _validate_live_state(_load_manifest(output_root), output_root)
    else:
        if source_repo is None:
            raise CampaignError("source repository is required to create manifest")
        _atomic_write_json(
            manifest_path, _create_manifest(output_root, source_repo.resolve())
        )
    print(f"campaign initialized: {output_root}")


def _expected_implementations(case: Case) -> tuple[str, ...]:
    return tuple(impl for _case_id, impl in _expected_result_keys(case))


def _result_dir(output_root: Path, case: Case, impl: str) -> Path:
    return output_root / "results" / case.case_id / impl


def _flop_count(case: Case) -> int:
    result = 4 * 2 * 32 * case.seq_len * case.seq_len * 64
    return result // 2 if case.causal else result


def _finite(value: object, context: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{context} is not numeric")
    result = float(value)
    _require(math.isfinite(result), f"{context} is not finite")
    _require(result > 0 if positive else result >= 0, f"{context} is out of range")
    return result


def _close(
    actual: object, expected: float, context: str, *, positive: bool = True
) -> None:
    value = _finite(actual, context, positive=positive)
    _require(
        math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{context} is inconsistent",
    )


def _expected_version(
    manifest: dict[str, Any], case: Case, impl: str
) -> dict[str, str]:
    if impl in BASELINE_IMPLEMENTATIONS:
        value = manifest["versions"]["baselines"][impl]
    else:
        run = KA_RUNS_BY_KEY[(case.case_id, impl)]
        value = manifest["versions"]["kernelagent"][run.run_id]
    _require(isinstance(value, dict), f"missing version for {case.case_id}/{impl}")
    return value


def validate_result(
    payload: object,
    case: Case,
    impl: str,
    manifest: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CampaignError("result is not an object")
    _require(impl in _expected_implementations(case), "unexpected result key")
    _require(payload.get("impl") == impl, "implementation mismatch")
    version = _expected_version(manifest, case, impl)
    _require(payload.get("version") == version["version"], "version mismatch")
    _require(
        payload.get("version_label") == version["version_label"],
        "version label mismatch",
    )
    _require(payload.get("shape") == case.shape, "shape mismatch")
    _require(payload.get("gpu") == EXPECTED_GPU_NAME, "GPU model mismatch")
    _require(
        payload.get("physical_gpu") == str(case.physical_gpu),
        "physical GPU mismatch",
    )
    _require(payload.get("power_cap_w") == EXPECTED_POWER_CAP_W, "power mismatch")
    _require(payload.get("input_seed") == INPUT_SEED, "input seed mismatch")
    _require(payload.get("flop_model") == "softmax_attention_forward", "FLOP model")
    _require(payload.get("accuracy") == "PASS", "correctness failed")
    _require("error" not in payload, "passing result contains an error")
    _require(payload.get("benchmark_timer") == "event", "timer mismatch")
    runs = payload.get("runs_ms")
    if not isinstance(runs, list):
        raise CampaignError("timing samples are missing")
    _require(len(runs) == SAMPLE_COUNT, f"expected {SAMPLE_COUNT} timing samples")
    parsed = [_finite(value, f"runs_ms[{index}]") for index, value in enumerate(runs)]
    median_ms = statistics.median(parsed)
    best_ms = min(parsed)
    mean_ms = sum(parsed) / len(parsed)
    _close(payload.get("best_ms"), best_ms, "best_ms")
    _close(payload.get("median_ms"), median_ms, "median_ms")
    _close(payload.get("mom_median_ms"), median_ms, "mom_median_ms")
    _close(payload.get("mean_ms"), mean_ms, "mean_ms")
    _close(
        payload.get("std_ms"),
        statistics.stdev(parsed),
        "std_ms",
        positive=False,
    )
    _close(
        payload.get("best_tflops"),
        _flop_count(case) / best_ms / 1e9,
        "best_tflops",
    )
    expected_tflops = _flop_count(case) / median_ms / 1e9
    _close(payload.get("median_tflops"), expected_tflops, "median_tflops")
    _close(payload.get("mom_median_tflops"), expected_tflops, "mom TFLOPS")
    notes = payload.get("notes", [])
    _require(isinstance(notes, list), "notes are malformed")
    note_text = " ".join(str(note) for note in notes)
    if impl == "flexattention-cute":
        _require("BACKEND='FLASH'" in note_text, "FlexAttention backend mismatch")
    if impl.startswith("kernelagent-closed-"):
        run = KA_RUNS_BY_KEY[(case.case_id, impl)]
        input_record = manifest["inputs"]["kernelagent_closed"][run.run_id]
        retained_manifest_path = (
            output_root / "inputs/kernelagent_closed" / run.run_id / "manifest.json"
        )
        retained_manifest = json.loads(retained_manifest_path.read_text())
        _require(isinstance(retained_manifest, dict), "retained KA manifest is invalid")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise CampaignError("KernelAgent config is missing")
        expected_config = {
            "budget_label": run.budget,
            "source_sha256": run.source_sha256,
            "selection_cute_version": EXPECTED_KA_SELECTION_CUTE_VERSION,
            "evaluation_cute_version": EXPECTED_CUTE_VERSION,
            "standard_correctness_executed": True,
            "repeat_determinism_executed": True,
            "stress_correctness_executed": True,
        }
        for key, expected in expected_config.items():
            _require(config.get(key) == expected, f"KernelAgent {key} mismatch")
        _require(
            config.get("selection") == retained_manifest.get("selection"),
            "KernelAgent selection mismatch",
        )
        _require(
            config.get("budget_seconds") == retained_manifest.get("budget_seconds"),
            "KernelAgent budget mismatch",
        )
        _require(
            config.get("elapsed_seconds") == retained_manifest.get("elapsed_seconds"),
            "KernelAgent elapsed time mismatch",
        )
        _require(
            input_record["selected_kernel_sha256"] == run.source_sha256,
            "KernelAgent retained source changed",
        )
        _require(
            f"selected with CuTe {EXPECTED_KA_SELECTION_CUTE_VERSION}"
            in payload["version"],
            "KernelAgent version omits selection CuTe",
        )
        _require(
            f"CuTe {EXPECTED_CUTE_VERSION}" in payload["version_label"],
            "KernelAgent label omits evaluation CuTe",
        )
        _require(
            "Standard and stress full-output checks passed with exact repeatability."
            in note_text,
            "KernelAgent correctness note is incomplete",
        )
    return payload


def _relative(output_root: Path, path: Path) -> str:
    return path.resolve().relative_to(output_root.resolve()).as_posix()


def _receipt(
    output_root: Path,
    case: Case,
    impl: str,
    result_path: Path,
    invocation_path: Path,
) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "case_id": case.case_id,
        "implementation": impl,
        "campaign_manifest_sha256": _sha256_file(_manifest_path(output_root)),
        "result_file": _relative(output_root, result_path),
        "result_sha256": _sha256_file(result_path),
        "invocation_file": _relative(output_root, invocation_path),
        "invocation_sha256": _sha256_file(invocation_path),
    }
    _reject_absolute_strings(value)
    return value


def _validate_receipt(
    output_root: Path,
    case: Case,
    impl: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    directory = _result_dir(output_root, case, impl)
    result_path = directory / "result.json"
    invocation_path = directory / "invocation.json"
    receipt_path = directory / "receipt.json"
    _require(result_path.is_file(), f"missing result for {case.case_id}/{impl}")
    _require(invocation_path.is_file(), f"missing invocation for {case.case_id}/{impl}")
    _require(receipt_path.is_file(), f"missing receipt for {case.case_id}/{impl}")
    receipt = json.loads(receipt_path.read_text())
    _require(
        receipt == _receipt(output_root, case, impl, result_path, invocation_path),
        f"receipt mismatch for {case.case_id}/{impl}",
    )
    invocation = json.loads(invocation_path.read_text())
    _require(isinstance(invocation, dict), "invocation is not an object")
    _command, _environment, expected_invocation = _invocation(
        output_root, case, impl, directory
    )
    _require(
        invocation == expected_invocation,
        f"invocation mismatch for {case.case_id}/{impl}",
    )
    _reject_absolute_strings(invocation)
    return validate_result(
        json.loads(result_path.read_text()), case, impl, manifest, output_root
    )


def _quarantine_result(path: Path, output_root: Path) -> None:
    _quarantine_path(path, output_root, f"{path.parent.name}--{path.name}")


def _invocation(
    output_root: Path,
    case: Case,
    impl: str,
    result_dir: Path,
) -> tuple[list[str], dict[str, str], dict[str, object]]:
    helion_root = output_root / "checkouts/helion"
    fa4_root = output_root / "checkouts/flash-attention"
    quack_root = output_root / "checkouts/quack"
    cache = result_dir / "cache"
    pending = result_dir / "result.pending.json"
    arguments = [
        "benchmarks/cute/compare_attention_backends.py",
        "--impl",
        impl,
        "--z",
        "2",
        "--h",
        "32",
        "--seq-len",
        str(case.seq_len),
        "--head-dim",
        "64",
        "--dtype",
        "float16",
        "--causal",
        str(case.causal),
        "--biased",
        "0",
        "--epilogue",
        "none",
        "--num-runs",
        str(SAMPLE_COUNT),
        "--warmup-ms",
        str(WARMUP_MS),
        "--rep-ms",
        str(REP_MS),
        "--seed",
        str(INPUT_SEED),
        "--power-cap-w",
        str(EXPECTED_POWER_CAP_W),
        "--skip-correctness",
        "0",
        "--json",
        "--json-output",
        str(pending),
    ]
    if impl.startswith("kernelagent-closed-"):
        arguments.extend(
            [
                "--kernelagent-closed-results-root",
                str(output_root / "inputs/kernelagent_closed"),
            ]
        )
    environment = _clean_environment()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(helion_root), str(quack_root))),
            "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(case.physical_gpu),
            "CUDA_CACHE_DISABLE": "0",
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "CUTE_DSL_CACHE_DIR": str(cache / "cute_dsl"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "TMPDIR": str(cache / "tmp"),
            "HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS": "6,7",
        }
    )
    if impl in BASELINE_IMPLEMENTATIONS:
        environment["HELION_FA4_ROOT"] = str(fa4_root)
    command = [sys.executable, str(helion_root / arguments[0]), *arguments[1:]]
    public_arguments: list[str] = []
    for argument in arguments:
        if argument == str(pending):
            public_arguments.append("result.pending.json")
        elif argument == str(output_root / "inputs/kernelagent_closed"):
            public_arguments.append("inputs/kernelagent_closed")
        else:
            public_arguments.append(argument)
    invocation: dict[str, object] = {
        "schema_version": 1,
        "case_id": case.case_id,
        "implementation": impl,
        "arguments": public_arguments,
        "physical_gpu": case.physical_gpu,
        "input_seed": INPUT_SEED,
        "sample_count": SAMPLE_COUNT,
        "correctness": True,
        "isolated_cache": True,
    }
    _reject_absolute_strings(invocation)
    return command, environment, invocation


def run_one(output_root: Path, case: Case, impl: str) -> None:
    _require(
        impl in _expected_implementations(case),
        f"{impl} is not selected for {case.case_id}",
    )
    manifest = _load_manifest(output_root)
    _validate_live_state(manifest, output_root)
    directory = _result_dir(output_root, case, impl)
    if directory.exists():
        try:
            _validate_receipt(output_root, case, impl, manifest)
        except (CampaignError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"quarantining incomplete {case.case_id}/{impl}: {exc}")
            _quarantine_result(directory, output_root)
        else:
            print(f"SKIP validated {case.case_id}/{impl}")
            return
    _validate_gpu_idle(case.physical_gpu)
    directory.mkdir(parents=True)
    for name in (
        "cuda",
        "cute_dsl",
        "pycache",
        "tmp",
        "torchinductor",
        "triton",
        "xdg",
    ):
        (directory / "cache" / name).mkdir(parents=True)
    command, environment, invocation = _invocation(output_root, case, impl, directory)
    invocation_path = directory / "invocation.json"
    _atomic_write_json(invocation_path, invocation)
    log_path = directory / "run.log"
    print(f"START {case.case_id}/{impl} on physical GPU {case.physical_gpu}")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=output_root / "checkouts/helion",
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    _require(process.returncode == 0, f"benchmark failed; inspect {log_path}")
    _validate_gpu_idle(case.physical_gpu)
    pending = directory / "result.pending.json"
    _require(pending.is_file(), "benchmark did not write its result")
    payload = validate_result(
        json.loads(pending.read_text()), case, impl, manifest, output_root
    )
    result_path = directory / "result.json"
    os.replace(pending, result_path)
    _validate_live_state(manifest, output_root)
    _atomic_write_json(
        directory / "receipt.json",
        _receipt(output_root, case, impl, result_path, invocation_path),
    )
    _validate_receipt(output_root, case, impl, manifest)
    _require(payload.get("accuracy") == "PASS", "accepted result did not pass")
    print(f"PASS {case.case_id}/{impl}")


def run_lane(output_root: Path, lane: str) -> None:
    _require(lane in {"dense", "causal"}, f"unknown lane: {lane}")
    expected_gpu = 7 if lane == "dense" else 6
    for case in CASES:
        if case.variant != lane:
            continue
        _require(case.physical_gpu == expected_gpu, "case GPU assignment changed")
        for impl in _expected_implementations(case):
            run_one(output_root, case, impl)


def _validate_overlay_with_harness(output_root: Path, paths: list[Path]) -> None:
    helion_root = output_root / "checkouts/helion"
    code = """
import importlib.util
import json
import sys
from pathlib import Path
benchmark = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('_baseline_overlay_validator', benchmark)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
payloads = [json.loads(Path(path).read_text()) for path in sys.argv[2:]]
module._validate_report_payloads(payloads)
"""
    env = _clean_environment()
    env["PYTHONPATH"] = str(helion_root)
    _run(
        [
            sys.executable,
            "-c",
            code,
            str(helion_root / "benchmarks/cute/compare_attention_backends.py"),
            *(str(path) for path in paths),
        ],
        cwd=helion_root,
        env=env,
    )


def build_overlay(
    output_root: Path, *, validate_live: bool = True, validate_report: bool = True
) -> list[Path]:
    manifest = _load_manifest(output_root)
    if validate_live:
        _validate_live_state(manifest, output_root)
    else:
        _validate_manifest_static(manifest, output_root)
    published = output_root / "published"
    pending = output_root / "published.pending"
    if pending.exists():
        _quarantine_path(pending, output_root, "published-pending")
    payload_dir = pending / "payloads"
    payload_dir.mkdir(parents=True)
    evidence_results: list[dict[str, object]] = []
    output_paths: list[Path] = []
    for case in CASES:
        historical_path = (
            output_root / "inputs/historical_payloads" / f"{case.case_id}.json"
        )
        historical = json.loads(historical_path.read_text())
        _validate_historical_payload(historical, case, historical_path.name)
        overlay = copy.deepcopy(historical)
        replacements = {
            impl: _validate_receipt(output_root, case, impl, manifest)
            for impl in _expected_implementations(case)
        }
        replacement_count = 0
        for index, old_result in enumerate(overlay["results"]):
            impl = old_result["impl"]
            if impl not in replacements:
                _require(
                    old_result == historical["results"][index],
                    f"unrefreshed row changed: {case.case_id}/{impl}",
                )
                continue
            refreshed = copy.deepcopy(replacements[impl])
            refreshed.pop("input_seed", None)
            overlay["results"][index] = refreshed
            replacement_count += 1
        _require(
            replacement_count == len(replacements),
            f"replacement count mismatch for {case.case_id}",
        )
        _require(
            all("input_seed" not in result for result in overlay["results"]),
            f"overlay schema contains input_seed for {case.case_id}",
        )
        _validate_historical_payload(overlay, case, f"overlay {case.case_id}")
        _reject_absolute_strings(overlay)
        output_path = payload_dir / f"{case.case_id}.json"
        _atomic_write_text(output_path, _canonical_json(overlay) + "\n")
        output_paths.append(output_path)
        for impl, result in replacements.items():
            raw_path = _result_dir(output_root, case, impl) / "result.json"
            receipt_path = raw_path.with_name("receipt.json")
            evidence_results.append(
                {
                    "case_id": case.case_id,
                    "implementation": impl,
                    "raw_result": _relative(output_root, raw_path),
                    "raw_result_sha256": _sha256_file(raw_path),
                    "receipt": _relative(output_root, receipt_path),
                    "receipt_sha256": _sha256_file(receipt_path),
                    "runs_ms": result["runs_ms"],
                    "median_ms": result["median_ms"],
                    "median_tflops": result["median_tflops"],
                    "accuracy": result["accuracy"],
                    "version": result["version"],
                }
            )
    if validate_report:
        _validate_overlay_with_harness(output_root, output_paths)
    if validate_live:
        _validate_live_state(manifest, output_root)
    else:
        _validate_manifest_static(manifest, output_root)
    evidence = {
        "schema_version": 1,
        "campaign": "cute470_baseline_refresh",
        "campaign_manifest_sha256": _sha256_file(_manifest_path(output_root)),
        "historical_inputs": manifest["inputs"]["historical_payloads"],
        "kernelagent_inputs": manifest["inputs"]["kernelagent_closed"],
        "source_hashes": manifest["sources"],
        "toolchain": manifest["toolchain"],
        "toolchain_sha256": _json_sha256(manifest["toolchain"]),
        "refreshed_results": evidence_results,
        "overlay_payloads": {
            path.name: {"sha256": _sha256_file(path)} for path in output_paths
        },
    }
    _reject_absolute_strings(evidence)
    _atomic_write_json(pending / "baseline_refresh_evidence.json", evidence)
    if published.exists():
        _quarantine_path(published, output_root, "published-previous")
    os.replace(pending, published)
    final_paths = [published / "payloads" / path.name for path in output_paths]
    print(f"wrote {len(final_paths)} overlay payloads to {published}")
    return final_paths


def _optional_path(value: str | None) -> Path | None:
    return None if value is None or not value else Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize_parser = subparsers.add_parser("initialize")
    initialize_parser.add_argument("output_root", type=Path)
    initialize_parser.add_argument("--source-repo")
    initialize_parser.add_argument("--fa4-source-repo")
    initialize_parser.add_argument("--quack-source-repo")
    initialize_parser.add_argument("--historical-dir")
    initialize_parser.add_argument("--ka-runs-dir")
    initialize_parser.add_argument("--ka-results-dir")
    lane_parser = subparsers.add_parser("run-lane")
    lane_parser.add_argument("output_root", type=Path)
    lane_parser.add_argument("--lane", choices=("dense", "causal"), required=True)
    one_parser = subparsers.add_parser("run-one")
    one_parser.add_argument("output_root", type=Path)
    one_parser.add_argument("--case", choices=tuple(CASES_BY_ID), required=True)
    one_parser.add_argument(
        "--impl",
        choices=(*BASELINE_IMPLEMENTATIONS, *KA_IMPLEMENTATIONS),
        required=True,
    )
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("output_root", type=Path)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("output_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "initialize":
            initialize(
                args.output_root,
                _optional_path(args.source_repo),
                _optional_path(args.fa4_source_repo),
                _optional_path(args.quack_source_repo),
                _optional_path(args.historical_dir),
                _optional_path(args.ka_runs_dir),
                _optional_path(args.ka_results_dir),
            )
        elif args.command == "run-lane":
            run_lane(args.output_root.resolve(), args.lane)
        elif args.command == "run-one":
            run_one(
                args.output_root.resolve(),
                _case_for_id(args.case),
                str(args.impl),
            )
        elif args.command == "validate":
            root = args.output_root.resolve()
            manifest = _load_manifest(root)
            _validate_live_state(manifest, root)
            for case in CASES:
                for impl in _expected_implementations(case):
                    _validate_receipt(root, case, impl, manifest)
            print("all 25 refreshed results passed validation")
        elif args.command == "build":
            build_overlay(args.output_root.resolve())
        else:
            raise AssertionError(args.command)
    except CampaignError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
