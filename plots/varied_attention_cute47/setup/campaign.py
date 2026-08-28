from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import hashlib
from importlib import metadata
import importlib.util
import io
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
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from types import ModuleType

EXPECTED_MEASURED_COMMIT = "c3e36b65d69681c23e053042b0bc21e2331bad17"
EXPECTED_CUTE_VERSION = "4.7.0"
EXPECTED_FA4_COMMIT = "2409214a03797b168f648ea30df1adbc09ce658a"
EXPECTED_FA4_DESCRIBE = "fa4-v4.0.0.beta23"
EXPECTED_QUACK_COMMIT = "b5b49dae477d39cb8ea8cca2820ef09ba548c72c"
EXPECTED_TORCH_VERSION = "2.13.0.dev20260506+cu130"
EXPECTED_TRITON_VERSION = "3.7.0+git88b227e2"
EXPECTED_CUDNN_RUNTIME_VERSION = "9.20.0"
EXPECTED_CUDNN_PACKAGE_VERSION = "9.20.0.48"
EXPECTED_TVM_FFI_VERSION = "0.1.11"
EXPECTED_GPU_NAME = "NVIDIA B200"
EXPECTED_GPU_UUIDS = {
    6: "GPU-b95967ee-1ea7-3bca-894f-5ec74e1c5513",
    7: "GPU-9e1e775d-008f-d915-e4a1-80cde2e60a7e",
}
EXPECTED_POWER_CAP_W = 750
EXPECTED_FINAL_CORRECTNESS_LAUNCHES = 64
INPUT_SEED = 2026081500
SAMPLE_COUNT = 9
WARMUP_MS = 1000
REP_MS = 500
AUTOTUNE_BENCHMARK_TIMEOUT_SECONDS = 180

IMPLEMENTATIONS = (
    "helion-cute",
    "sdpa",
    "fa4",
    "flexattention-cute",
)
IMPLEMENTATION_LABELS = {
    "helion-cute": "Helion (backend=CuTe)",
    "sdpa": "torch SDPA",
    "fa4": "FA4",
    "flexattention-cute": "FlexAttention (backend=CuTe)",
}
CSV_FIELDS = (
    "shape_order",
    "shape_group",
    "variant",
    "z",
    "h",
    "seq_len",
    "head_dim",
    "dtype",
    "epilogue",
    "implementation",
    "implementation_label",
    "version",
    "tflops",
    "statistic",
    "sample_count",
    "gpu",
    "physical_gpu",
    "power_cap_w",
    "power_cap_recorded",
    "correctness",
    "timing_mode",
    "evidence_file",
    "notes",
)
SOURCE_PATHS = (
    "helion",
    "examples/__init__.py",
    "examples/attention.py",
    "benchmarks/cute/compare_attention_backends.py",
    "sitecustomize.py",
    "usercustomize.py",
)
DIST_NAMES = (
    "apache-tvm-ffi",
    "torch",
    "triton",
    "nvidia-cutlass-dsl",
    "nvidia-cudnn-cu13",
)
TOOL_NAMES = ("nvidia-smi", "ptxas", "nvcc", "gcc", "g++")
MODULE_NAMES = ("torch", "triton", "cutlass", "tvm_ffi", "torch.backends.cudnn")
AUTOTUNE_SIDECAR_NAMES = (
    "autotune.csv",
    "autotune.meta.jsonl",
    "autotune.sources.csv",
)
_STRICT_VALIDATOR_MODULES: dict[str, ModuleType] = {}


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class Case:
    order: int
    group: str
    variant: str
    z: int
    h: int
    seq_len: int
    head_dim: int
    epilogue: str
    physical_gpu: int
    tuner_seed: int

    @property
    def case_id(self) -> str:
        return (
            f"{self.order:02d}-{self.group}-{self.variant}-"
            f"b{self.z}-h{self.h}-s{self.seq_len}-d{self.head_dim}"
            + ("-relu" if self.epilogue == "relu" else "")
        )

    @property
    def causal(self) -> int:
        return int(self.variant == "causal")

    @property
    def shape(self) -> dict[str, object]:
        shape: dict[str, object] = {
            "z": self.z,
            "h": self.h,
            "seq_len": self.seq_len,
            "head_dim": self.head_dim,
            "dtype": "bfloat16",
            "causal": self.causal,
            "biased": 0,
        }
        if self.epilogue != "none":
            shape["epilogue"] = self.epilogue
        return shape

    @property
    def input_spec(self) -> dict[str, object]:
        return {
            "generator": (
                "torch.manual_seed(seed), followed by sequential torch.randn "
                "calls for q, k, and v"
            ),
            "seed": INPUT_SEED,
            "shape_bhsd": [self.z, self.h, self.seq_len, self.head_dim],
            "dtype": "torch.bfloat16",
        }


CASES = (
    Case(1, "head_dim_128", "dense", 2, 32, 262144, 128, "none", 7, 2026081701),
    Case(
        2,
        "head_dim_128",
        "causal",
        2,
        32,
        524288,
        128,
        "none",
        6,
        2026081702,
    ),
    Case(3, "batch_1", "dense", 1, 32, 524288, 64, "none", 7, 2026081703),
    Case(
        4,
        "batch_1",
        "causal",
        1,
        32,
        1048576,
        64,
        "none",
        6,
        2026081704,
    ),
    Case(5, "batch_8", "dense", 8, 32, 524288, 64, "none", 7, 2026081705),
    Case(
        6,
        "batch_8",
        "causal",
        8,
        32,
        786432,
        64,
        "none",
        6,
        2026081706,
    ),
    Case(
        7,
        "relu_epilogue",
        "dense",
        2,
        32,
        524288,
        64,
        "relu",
        7,
        2026081707,
    ),
    Case(
        8,
        "relu_epilogue",
        "causal",
        2,
        32,
        1048576,
        64,
        "relu",
        6,
        2026081708,
    ),
)
CASES_BY_ID = {case.case_id: case for case in CASES}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_sha256(value: object) -> str:
    return _sha256_bytes(_canonical_json(value).encode())


def _atomic_write_text(path: Path, text: str) -> None:
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
        handle.write(text)
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
    actual_commit = _git(root, "rev-parse", "HEAD")
    _require(
        actual_commit == expected_commit,
        f"checkout {root.name} is at {actual_commit}, expected {expected_commit}",
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
    _require(resolved == commit, f"{source.name} cannot resolve exact commit {commit}")
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


def _validate_external_output(output_root: Path, source_repo: Path) -> None:
    _require(
        output_root != source_repo and source_repo not in output_root.parents,
        f"output root must be outside source repository {source_repo.name}",
    )


def _source_subset_snapshot(root: Path) -> dict[str, object]:
    listed = _run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *SOURCE_PATHS,
        ]
    ).stdout
    paths = sorted(set(listed.rstrip("\0").split("\0"))) if listed else []
    digest = hashlib.sha256()
    count = 0
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
        count += 1
    return {"sha256": digest.hexdigest(), "file_count": count}


def _source_record(root: Path, *, kind: str) -> dict[str, object]:
    record: dict[str, object] = {
        "commit": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "describe": _git(root, "describe", "--tags", "--always", "--dirty"),
    }
    if kind == "helion":
        benchmark = root / "benchmarks/cute/compare_attention_backends.py"
        attention = root / "examples/attention.py"
        record.update(
            {
                "benchmark_sha256": _sha256_file(benchmark),
                "attention_sha256": _sha256_file(attention),
                "source_subset": _source_subset_snapshot(root),
            }
        )
    elif kind == "flash_attention":
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
        raise CampaignError(f"required distribution is not installed: {name}") from exc
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
    _require(
        packages["torch"]["version"] == EXPECTED_TORCH_VERSION,
        f"expected torch {EXPECTED_TORCH_VERSION}",
    )
    _require(
        packages["triton"]["version"] == EXPECTED_TRITON_VERSION,
        f"expected Triton {EXPECTED_TRITON_VERSION}",
    )
    _require(
        packages["nvidia-cutlass-dsl"]["version"] == EXPECTED_CUTE_VERSION,
        f"expected CuTe {EXPECTED_CUTE_VERSION}",
    )
    _require(
        packages["nvidia-cudnn-cu13"]["version"] == EXPECTED_CUDNN_PACKAGE_VERSION,
        f"expected nvidia-cudnn-cu13 {EXPECTED_CUDNN_PACKAGE_VERSION}",
    )
    _require(
        packages["apache-tvm-ffi"]["version"] == EXPECTED_TVM_FFI_VERSION,
        f"expected apache-tvm-ffi {EXPECTED_TVM_FFI_VERSION}",
    )
    _require(
        record["cudnn_runtime_version"] == EXPECTED_CUDNN_RUNTIME_VERSION,
        f"expected cuDNN runtime {EXPECTED_CUDNN_RUNTIME_VERSION}",
    )
    tools = record["tools"]
    assert isinstance(tools, dict)
    _require(
        set(TOOL_NAMES) <= set(tools),
        f"required compiler tools are missing: {sorted(set(TOOL_NAMES) - set(tools))}",
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
    records: dict[int, dict[str, object]] = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        _require(len(fields) == 5, f"unexpected nvidia-smi row: {line}")
        index = int(fields[0])
        if index not in EXPECTED_GPU_UUIDS:
            continue
        records[index] = {
            "physical_gpu": index,
            "uuid": fields[1],
            "name": fields[2],
            "power_limit_w": float(fields[3]),
            "driver_version": fields[4],
        }
    _require(set(records) == set(EXPECTED_GPU_UUIDS), "GPUs 6 and 7 are required")
    for index, record in records.items():
        _require(
            record["uuid"] == EXPECTED_GPU_UUIDS[index],
            f"physical GPU {index} UUID changed",
        )
        _require(record["name"] == EXPECTED_GPU_NAME, f"GPU {index} is not B200")
        _require(
            math.isclose(
                _finite_positive(record["power_limit_w"], "GPU power limit"),
                EXPECTED_POWER_CAP_W,
            ),
            f"GPU {index} power limit is not {EXPECTED_POWER_CAP_W} W",
        )
    return [records[index] for index in sorted(records)]


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


def _probe_implementation_versions(
    helion_root: Path, fa4_root: Path, quack_root: Path
) -> dict[str, dict[str, str]]:
    benchmark = helion_root / "benchmarks/cute/compare_attention_backends.py"
    code = """
import importlib.util
import json
from pathlib import Path
path = Path('benchmarks/cute/compare_attention_backends.py').resolve()
spec = importlib.util.spec_from_file_location('_varied_attention_benchmark', path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print(json.dumps({name: module._implementation_version(name) for name in (
    'helion-cute', 'sdpa', 'fa4', 'flexattention-cute')}))
"""
    env = _clean_base_environment()
    env["HELION_FA4_ROOT"] = str(fa4_root)
    env["PYTHONPATH"] = os.pathsep.join((str(helion_root), str(quack_root)))
    result = _run([sys.executable, "-c", code], cwd=helion_root, env=env)
    _require(benchmark.is_file(), "measured benchmark is missing")
    try:
        versions = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise CampaignError("failed to resolve implementation versions") from exc
    _require(set(versions) == set(IMPLEMENTATIONS), "version probe is incomplete")
    for impl, values in versions.items():
        _require(
            isinstance(values, dict) and set(values) == {"version", "version_label"},
            f"invalid version record for {impl}",
        )
    return versions


def _probe_import_roots(
    helion_root: Path, fa4_root: Path, quack_root: Path
) -> dict[str, object]:
    core_code = """
import hashlib
import json
import os
from pathlib import Path
import helion
helion_root = Path(os.environ['VARIED_HELION_ROOT']).resolve()
helion_path = Path(helion.__file__).resolve()
if helion_root not in helion_path.parents:
    raise SystemExit(f'Helion imported outside measured root: {helion_path}')
print(json.dumps({
    'module': helion_path.relative_to(helion_root).as_posix(),
    'module_sha256': hashlib.sha256(helion_path.read_bytes()).hexdigest(),
}))
"""
    fa4_code = """
import hashlib
import importlib.util
import json
import os
from pathlib import Path
helion_root = Path(os.environ['VARIED_HELION_ROOT']).resolve()
fa4_root = Path(os.environ['VARIED_FA4_ROOT']).resolve()
quack_root = Path(os.environ['VARIED_QUACK_ROOT']).resolve()
path = helion_root / 'benchmarks/cute/compare_attention_backends.py'
spec = importlib.util.spec_from_file_location('_varied_attention_import_probe', path)
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
    core_env = _clean_base_environment()
    core_env["PYTHONPATH"] = str(helion_root)
    core_env["VARIED_HELION_ROOT"] = str(helion_root)
    core_result = _run([sys.executable, "-c", core_code], cwd=helion_root, env=core_env)
    fa4_env = _clean_base_environment()
    fa4_env["PYTHONPATH"] = os.pathsep.join((str(helion_root), str(quack_root)))
    fa4_env["HELION_FA4_ROOT"] = str(fa4_root)
    fa4_env["VARIED_HELION_ROOT"] = str(helion_root)
    fa4_env["VARIED_FA4_ROOT"] = str(fa4_root)
    fa4_env["VARIED_QUACK_ROOT"] = str(quack_root)
    fa4_result = _run([sys.executable, "-c", fa4_code], cwd=helion_root, env=fa4_env)
    try:
        record = {
            "core": json.loads(core_result.stdout.strip().splitlines()[-1]),
            "fa4_flex": json.loads(fa4_result.stdout.strip().splitlines()[-1]),
        }
    except (IndexError, json.JSONDecodeError) as exc:
        raise CampaignError("failed to validate Helion and Quack import roots") from exc
    _require(isinstance(record, dict), "invalid import-root record")
    _reject_absolute_strings(record)
    return record


def _reject_absolute_strings(value: object, context: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_strings(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_strings(item, f"{context}[{index}]")
    elif isinstance(value, str):
        _require(
            not value.startswith("/") and re.match(r"^[A-Za-z]:[\\/]", value) is None,
            f"absolute path leaked into published data at {context}",
        )


def _campaign_definition() -> dict[str, object]:
    return {
        "dtype": "bfloat16",
        "input_seed": INPUT_SEED,
        "sample_count": SAMPLE_COUNT,
        "warmup_ms": WARMUP_MS,
        "rep_ms": REP_MS,
        "power_cap_w": EXPECTED_POWER_CAP_W,
        "implementations": list(IMPLEMENTATIONS),
        "cases": [
            {
                **asdict(case),
                "case_id": case.case_id,
                "input_spec_sha256": _json_sha256(case.input_spec),
            }
            for case in CASES
        ],
    }


def _manifest_path(output_root: Path) -> Path:
    return output_root / "campaign_manifest.json"


def _load_manifest(output_root: Path) -> dict[str, Any]:
    path = _manifest_path(output_root)
    _require(path.is_file(), f"missing campaign manifest: {path}")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "campaign manifest is not an object")
    return value


def _create_manifest(output_root: Path) -> dict[str, object]:
    helion_root = output_root / "checkouts/helion"
    fa4_root = output_root / "checkouts/flash-attention"
    quack_root = output_root / "checkouts/quack"
    toolchain = _toolchain_record()
    _validate_toolchain(toolchain)
    helion_source = _source_record(helion_root, kind="helion")
    fa4_source = _source_record(fa4_root, kind="flash_attention")
    quack_source = _source_record(quack_root, kind="quack")
    _require(
        helion_source["commit"] == EXPECTED_MEASURED_COMMIT,
        "measured Helion commit mismatch",
    )
    _require(fa4_source["commit"] == EXPECTED_FA4_COMMIT, "FA4 commit mismatch")
    _require(
        fa4_source["describe"] == EXPECTED_FA4_DESCRIBE,
        "FA4 version description mismatch",
    )
    _require(
        quack_source["commit"] == EXPECTED_QUACK_COMMIT,
        "Quack commit mismatch",
    )
    setup_files = {
        name: _sha256_file(output_root / "launcher" / name)
        for name in (
            "build_strict_manifest.py",
            "campaign.py",
            "run_campaign.sh",
        )
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "campaign": "varied_attention_cute47",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "expected_measured_commit": EXPECTED_MEASURED_COMMIT,
        "definition": _campaign_definition(),
        "sources": {
            "helion": helion_source,
            "flash_attention": fa4_source,
            "quack": quack_source,
            "setup_files": setup_files,
        },
        "toolchain": toolchain,
        "hardware": _probe_gpus(),
        "import_roots": _probe_import_roots(helion_root, fa4_root, quack_root),
        "implementation_versions": _probe_implementation_versions(
            helion_root, fa4_root, quack_root
        ),
    }
    _reject_absolute_strings(manifest)
    return manifest


def _validate_manifest_static(manifest: dict[str, Any], output_root: Path) -> None:
    _require(manifest.get("schema_version") == 1, "unsupported manifest schema")
    _require(
        manifest.get("campaign") == "varied_attention_cute47",
        "wrong campaign manifest",
    )
    _require(
        manifest.get("expected_measured_commit") == EXPECTED_MEASURED_COMMIT,
        "manifest measured commit does not match the launcher",
    )
    _require(
        manifest.get("definition") == _campaign_definition(),
        "campaign definition changed",
    )
    versions = manifest.get("implementation_versions")
    if not isinstance(versions, dict) or set(versions) != set(IMPLEMENTATIONS):
        raise CampaignError("manifest implementation versions are incomplete")
    for impl in ("helion-cute", "fa4", "flexattention-cute"):
        version_record = versions[impl]
        _require(isinstance(version_record, dict), f"invalid {impl} version record")
        _require(
            f"CuTe {EXPECTED_CUTE_VERSION}" in str(version_record.get("version")),
            f"manifest {impl} does not use CuTe {EXPECTED_CUTE_VERSION}",
        )
    _reject_absolute_strings(manifest)
    setup_files = manifest["sources"]["setup_files"]
    for name, expected_hash in setup_files.items():
        path = output_root / "launcher" / name
        _require(path.is_file(), f"missing snapshotted launcher file: {name}")
        _require(_sha256_file(path) == expected_hash, f"launcher {name} changed")


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
        _source_record(helion_root, kind="helion") == manifest["sources"]["helion"],
        "Helion source fingerprint changed",
    )
    _require(
        _source_record(fa4_root, kind="flash_attention")
        == manifest["sources"]["flash_attention"],
        "FA4 source fingerprint changed",
    )
    _require(
        _source_record(quack_root, kind="quack") == manifest["sources"]["quack"],
        "Quack source fingerprint changed",
    )
    current_toolchain = _toolchain_record()
    _validate_toolchain(current_toolchain)
    _require(current_toolchain == manifest["toolchain"], "toolchain changed")
    _require(
        _probe_import_roots(helion_root, fa4_root, quack_root)
        == manifest["import_roots"],
        "Helion or Quack import root changed",
    )
    _require(_probe_gpus() == manifest["hardware"], "GPU identity or power changed")
    _require(_load_manifest(output_root) == manifest, "campaign manifest changed")


def initialize(
    output_root: Path,
    source_repo: Path | None,
    fa4_source_repo: Path | None,
    quack_source_repo: Path | None,
) -> None:
    output_root = output_root.resolve()
    helion_root = output_root / "checkouts/helion"
    fa4_root = output_root / "checkouts/flash-attention"
    quack_root = output_root / "checkouts/quack"
    if not helion_root.exists():
        if source_repo is None:
            raise CampaignError("a Helion source repository is required")
        source_repo = source_repo.resolve()
        _validate_external_output(output_root, source_repo)
        _ensure_worktree(source_repo, helion_root, EXPECTED_MEASURED_COMMIT)
    if not fa4_root.exists():
        if fa4_source_repo is None:
            raise CampaignError("an FA4 source repository is required")
        fa4_source_repo = fa4_source_repo.resolve()
        _validate_external_output(output_root, fa4_source_repo)
        _ensure_worktree(fa4_source_repo, fa4_root, EXPECTED_FA4_COMMIT)
    if not quack_root.exists():
        if quack_source_repo is None:
            raise CampaignError("a Quack source repository is required")
        quack_source_repo = quack_source_repo.resolve()
        _validate_external_output(output_root, quack_source_repo)
        _ensure_worktree(quack_source_repo, quack_root, EXPECTED_QUACK_COMMIT)
    _validate_checkout(helion_root, EXPECTED_MEASURED_COMMIT)
    _validate_checkout(fa4_root, EXPECTED_FA4_COMMIT)
    _validate_checkout(quack_root, EXPECTED_QUACK_COMMIT)
    path = _manifest_path(output_root)
    if path.exists():
        manifest = _load_manifest(output_root)
        _validate_live_state(manifest, output_root)
    else:
        manifest = _create_manifest(output_root)
        _atomic_write_json(path, manifest)
    print(f"campaign initialized: {output_root}")


def _case(case_id: str) -> Case:
    try:
        return CASES_BY_ID[case_id]
    except KeyError as exc:
        raise CampaignError(f"unknown case: {case_id}") from exc


def _result_dir(output_root: Path, case: Case, impl: str) -> Path:
    return output_root / "results" / case.case_id / impl


def _flop_count(case: Case) -> int:
    count = 4 * case.z * case.h * case.seq_len * case.seq_len * case.head_dim
    return count // 2 if case.causal else count


def _finite_positive(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"{context} is not numeric")
    converted = float(value)
    _require(math.isfinite(converted) and converted > 0, f"{context} is invalid")
    return converted


def _check_close(
    actual: object, expected: float, context: str, *, allow_zero: bool = False
) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise CampaignError(f"{context} is not numeric")
    value = float(actual)
    _require(
        math.isfinite(value) and (value >= 0 if allow_zero else value > 0),
        f"{context} is invalid",
    )
    _require(
        math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{context} is inconsistent: {value} != {expected}",
    )


def _validate_common_result(
    payload: dict[str, Any],
    case: Case,
    impl: str,
    manifest: dict[str, Any],
) -> None:
    _require(payload.get("impl") == impl, "result implementation mismatch")
    version = manifest["implementation_versions"][impl]
    _require(payload.get("version") == version["version"], "result version mismatch")
    _require(
        payload.get("version_label") == version["version_label"],
        "result version label mismatch",
    )
    _require(payload.get("shape") == case.shape, "result shape mismatch")
    _require(payload.get("gpu") == EXPECTED_GPU_NAME, "result GPU mismatch")
    _require(
        payload.get("physical_gpu") == str(case.physical_gpu),
        "result physical GPU mismatch",
    )
    _require(
        payload.get("power_cap_w") == EXPECTED_POWER_CAP_W,
        "result power cap mismatch",
    )
    _require(payload.get("input_seed") == INPUT_SEED, "result input seed mismatch")
    _require(
        payload.get("flop_model") == "softmax_attention_forward",
        "result FLOP model mismatch",
    )
    _require(payload.get("accuracy") == "PASS", "result correctness did not pass")
    expected_timer = "wall" if impl == "helion-cute" else "event"
    _require(
        payload.get("benchmark_timer") == expected_timer,
        "result benchmark timer mismatch",
    )
    if case.epilogue == "relu":
        _require(
            payload.get("epilogue_flops_included") is False,
            "ReLU FLOPs must be excluded",
        )
    else:
        _require(
            "epilogue_flops_included" not in payload,
            "identity epilogue unexpectedly has epilogue metadata",
        )

    runs = payload.get("runs_ms")
    if not isinstance(runs, list):
        raise CampaignError("result timings are missing")
    _require(len(runs) == SAMPLE_COUNT, f"expected {SAMPLE_COUNT} timing samples")
    timings = [
        _finite_positive(value, f"runs_ms[{index}]") for index, value in enumerate(runs)
    ]
    median_ms = statistics.median(timings)
    mean_ms = sum(timings) / len(timings)
    std_ms = statistics.stdev(timings)
    _check_close(payload.get("best_ms"), min(timings), "best_ms")
    _check_close(payload.get("median_ms"), median_ms, "median_ms")
    _check_close(payload.get("mom_median_ms"), median_ms, "mom_median_ms")
    _check_close(payload.get("mean_ms"), mean_ms, "mean_ms")
    _check_close(payload.get("std_ms"), std_ms, "std_ms", allow_zero=True)
    expected_tflops = _flop_count(case) / median_ms / 1e9
    _check_close(payload.get("median_tflops"), expected_tflops, "median_tflops")
    _check_close(payload.get("mom_median_tflops"), expected_tflops, "mom_median_tflops")
    _check_close(
        payload.get("best_tflops"),
        _flop_count(case) / min(timings) / 1e9,
        "best_tflops",
    )


def _validate_notes(payload: dict[str, Any], case: Case, impl: str) -> None:
    notes = payload.get("notes", [])
    _require(isinstance(notes, list), "result notes are malformed")
    text = " ".join(str(note) for note in notes)
    if impl == "sdpa":
        _require("CUDNN_ATTENTION" in text, "SDPA was not forced to cuDNN")
    if impl == "flexattention-cute":
        _require("BACKEND='FLASH'" in text, "FlexAttention was not forced to FLASH")
    if case.epilogue == "relu":
        required = {
            "helion-cute": "fused into",
            "sdpa": "eager torch.relu",
            "fa4": "eager torch.relu",
            "flexattention-cute": "captured FlexAttention and ReLU",
        }[impl]
        _require(required in text, f"missing ReLU timing evidence for {impl}")


def _path_is_within(path_value: object, root: Path, context: str) -> None:
    if not isinstance(path_value, str) or not path_value:
        raise CampaignError(f"missing {context}")
    path = Path(path_value).resolve()
    _require(path == root or root in path.parents, f"{context} escaped result cache")


def _validate_helion_result(
    payload: dict[str, Any],
    case: Case,
    manifest: dict[str, Any],
    result_dir: Path,
) -> None:
    codegen = payload.get("codegen")
    if not isinstance(codegen, dict):
        raise CampaignError("missing Helion codegen markers")
    _require(codegen.get("uses_tcgen05") is True, "Helion did not use tcgen05")
    _require(
        codegen.get("uses_relu_epilogue") is (case.epilogue == "relu"),
        "Helion ReLU codegen marker mismatch",
    )
    overrides = payload.get("helion_overrides")
    if not isinstance(overrides, dict):
        raise CampaignError("missing Helion overrides")
    expected_overrides = {
        "autotuned": True,
        "benchmark_timer": "wall",
        "config_overrides": {},
        "seed_config_overrides": {},
        "force_autotune": True,
        "return_lse": False,
    }
    for key, expected in expected_overrides.items():
        _require(overrides.get(key) == expected, f"Helion override {key} mismatch")
    if case.epilogue == "relu":
        _require(overrides.get("epilogue") == "relu", "Helion epilogue mismatch")
    else:
        _require("epilogue" not in overrides, "unexpected Helion epilogue override")
    env = overrides.get("env_overrides")
    if not isinstance(env, dict):
        raise CampaignError("missing Helion environment overrides")
    required_env = {
        "HELION_AUTOTUNE_RANDOM_SEED": str(case.tuner_seed),
        "HELION_DISABLE_AUTOTUNER_HEURISTICS": "0",
        "HELION_AUTOTUNER": "",
        "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "-1",
        "HELION_AUTOTUNE_EFFORT": "full",
        "HELION_AUTOTUNE_BEST_OF_K": "1",
        "HELION_AUTOTUNE_BENCHMARK_TIMEOUT": str(AUTOTUNE_BENCHMARK_TIMEOUT_SECONDS),
        "HELION_AUTOTUNE_ACCURACY_CHECK": "1",
        "HELION_AUTOTUNER_INITIAL_POPULATION": "from_random",
    }
    for key, expected in required_env.items():
        _require(env.get(key) == expected, f"Helion environment {key} mismatch")
    cache_keys = {
        "HELION_CACHE_DIR",
        "CUTE_DSL_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    }
    _require(
        set(env) == set(required_env) | cache_keys,
        "Helion environment override set is not canonical",
    )
    cache_root = (result_dir / "cache").resolve()
    for key in cache_keys:
        _path_is_within(env.get(key), cache_root, key)

    provenance = overrides.get("autotune_provenance")
    if not isinstance(provenance, dict):
        raise CampaignError("missing strict autotune provenance")
    strict_values = {
        "require_full_autotune": True,
        "effort": "full",
        "requested_force_autotune": True,
        "effective_force_autotune": True,
        "fixed_config": False,
        "autotune_budget_seconds": None,
        "autotune_max_generations": None,
        "autotune_best_of_k": 1,
        "autotune_accuracy_check": True,
        "autotune_random_seed": case.tuner_seed,
        "autotuner_initial_population_env": "from_random",
        "user_seed_configs": False,
        "cache_read_policy": "bypass",
        "cache_write_policy": "write",
        "final_correctness_launches": EXPECTED_FINAL_CORRECTNESS_LAUNCHES,
        "final_repeatability_passed": True,
        "final_correctness_passed": True,
        "post_measurement_source_verified": True,
    }
    for key, expected in strict_values.items():
        _require(provenance.get(key) == expected, f"provenance {key} mismatch")
    source = manifest["sources"]["helion"]
    _require(
        provenance.get("helion_checkout_git_commit") == source["commit"],
        "strict provenance commit mismatch",
    )
    _require(
        provenance.get("helion_source_tree_sha256")
        == source["source_subset"]["sha256"],
        "strict provenance source hash mismatch",
    )
    _require(
        provenance.get("helion_source_tree_dirty") is False,
        "strict provenance source was dirty",
    )
    post_source = provenance.get("post_measurement_source")
    if not isinstance(post_source, dict):
        raise CampaignError("missing post-measurement source")
    for key in (
        "helion_checkout_git_commit",
        "helion_source_tree_sha256",
        "helion_source_tree_file_count",
        "helion_source_tree_dirty",
    ):
        _require(
            post_source.get(key) == provenance.get(key),
            f"post-measurement source changed: {key}",
        )
    selected_source = provenance.get("selected_source_sha256")
    _require(
        isinstance(selected_source, str)
        and re.fullmatch(r"[0-9a-f]{64}", selected_source) is not None,
        "invalid selected source hash",
    )
    selected_config = provenance.get("selected_config")
    _require(isinstance(selected_config, dict), "missing selected config")
    trials = provenance.get("trials")
    if not (
        isinstance(trials, list) and len(trials) == 1 and isinstance(trials[0], dict)
    ):
        raise CampaignError("strict run must contain exactly one autotune trial")
    trial = trials[0]
    shape = (case.z, case.h, case.seq_len, case.head_dim)
    expected_trial = {
        "random_seed": case.tuner_seed,
        "search_algorithm": "LFBOTreeSearch",
        "input_shapes": repr([shape, shape, shape]),
        "dtypes": repr(["torch.bfloat16"] * 3),
        "hardware": EXPECTED_GPU_NAME,
        "selected_source_hash": selected_source,
        "selected_config": selected_config,
        "selected_source_was_measured": True,
    }
    for key, expected in expected_trial.items():
        _require(trial.get(key) == expected, f"autotune trial {key} mismatch")
    _require(
        isinstance(trial.get("num_successful_candidate_measurements"), int)
        and trial["num_successful_candidate_measurements"] > 0,
        "autotune trial has no successful measurements",
    )


def validate_result_payload(
    payload: object,
    case: Case,
    impl: str,
    manifest: dict[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    _require(impl in IMPLEMENTATIONS, f"unsupported implementation: {impl}")
    if not isinstance(payload, dict):
        raise CampaignError("result is not an object")
    typed: dict[str, Any] = payload
    _validate_common_result(typed, case, impl, manifest)
    _validate_notes(typed, case, impl)
    if impl == "helion-cute":
        _validate_helion_result(typed, case, manifest, result_dir)
    return typed


def _load_strict_validator(output_root: Path, manifest: dict[str, Any]) -> ModuleType:
    path = output_root / "launcher/build_strict_manifest.py"
    _require(path.is_file(), "pinned strict validator is missing")
    expected_hash = manifest["sources"]["setup_files"]["build_strict_manifest.py"]
    actual_hash = _sha256_file(path)
    _require(actual_hash == expected_hash, "pinned strict validator changed")
    module = _STRICT_VALIDATOR_MODULES.get(actual_hash)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        f"_varied_attention_strict_{actual_hash[:16]}", path
    )
    if spec is None or spec.loader is None:
        raise CampaignError("validator import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _STRICT_VALIDATOR_MODULES[actual_hash] = module
    return module


def _generalized_autotune_sidecars(
    validator: ModuleType,
    result_dir: Path,
    source_rows: list[dict[str, str]],
    source_run_id: str,
    selected_config: dict[str, Any],
    case: Case,
) -> dict[str, object]:
    csv_path = result_dir / "autotune.csv"
    metadata_path = result_dir / "autotune.meta.jsonl"
    _require(csv_path.is_file(), f"missing adjacent autotune CSV: {csv_path}")
    try:
        csv_contents = csv_path.read_bytes()
        reader = csv.DictReader(io.StringIO(csv_contents.decode()))
        validator.check_equal(
            tuple(reader.fieldnames or ()),
            validator.AUTOTUNE_CSV_FIELDS,
            f"{csv_path}: header",
        )
        rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise CampaignError(f"unable to read {csv_path}: {exc}") from exc
    validator.check_equal(
        len(rows), len(source_rows), f"{csv_path}: source ledger row count"
    )
    for line, (row, source_row) in enumerate(zip(rows, source_rows, strict=True), 2):
        validator.require(
            None not in row
            and all(
                row.get(field) is not None for field in validator.AUTOTUNE_CSV_FIELDS
            ),
            f"{csv_path}:{line}: malformed autotune row",
        )
        for field in validator.AUTOTUNE_JOIN_FIELDS:
            validator.check_equal(
                row[field], source_row[field], f"{csv_path}:{line}: {field}"
            )
        successful = row["status"] in {"ok", "deduplicated"}
        validator.check_equal(
            bool(row["perf_ms"]),
            successful,
            f"{csv_path}:{line}: performance/status consistency",
        )
        if successful:
            validator.require(
                validator.csv_float(row["perf_ms"], f"{csv_path}:{line}: perf_ms")
                > 0.0,
                f"{csv_path}:{line}: perf_ms must be positive",
            )
        if row["compile_time_s"]:
            validator.require(
                validator.csv_float(
                    row["compile_time_s"], f"{csv_path}:{line}: compile_time_s"
                )
                >= 0.0,
                f"{csv_path}:{line}: compile_time_s must be nonnegative",
            )
        validator.require(row["config"], f"{csv_path}:{line}: config is empty")

    attempt_by_config: dict[str, dict[str, Any]] = {}
    attempt_history_by_config: dict[str, list[dict[str, Any]]] = {}
    for position, row in enumerate(rows):
        if row["status"] == "started":
            continue
        config_id = row["config_id"]
        previous = attempt_by_config.get(config_id)
        validator.require(
            previous is None
            or (
                previous["status"] in validator.LEDGER_REPAIRABLE_FAILURE_STATUSES
                and row["status"] == "deduplicated"
            ),
            f"{csv_path}: config {config_id} has conflicting terminal attempts",
        )
        attempt = {
            "generation": int(row["generation"]),
            "status": row["status"],
            "source_hash": source_rows[position]["source_hash"],
            "perf_ms": float(row["perf_ms"]) if row["perf_ms"] else None,
            "position": position,
        }
        attempt_by_config[config_id] = attempt
        attempt_history_by_config.setdefault(config_id, []).append(attempt)

    metadata_record, metadata_sha256 = validator.read_metadata_record(metadata_path)
    run_id = validator.metadata_run_id(metadata_record, metadata_path)
    validator.check_equal(
        metadata_record.get("run_id"), run_id, f"{metadata_path}: computed run_id"
    )
    validator.check_equal(
        run_id, source_run_id, f"{metadata_path}: source ledger run_id"
    )
    kernel_name = (
        "causal_attention_relu_output"
        if case.causal and case.epilogue == "relu"
        else (
            "attention_relu_output"
            if case.epilogue == "relu"
            else ("causal_attention_output" if case.causal else "attention_output")
        )
    )
    validator.check_equal(
        metadata_record.get("kernel_name"),
        kernel_name,
        f"{metadata_path}: kernel name",
    )
    shape = (case.z, case.h, case.seq_len, case.head_dim)
    validator.check_equal(
        metadata_record.get("input_shapes"),
        repr([shape, shape, shape]),
        f"{metadata_path}: input shapes",
    )
    validator.check_equal(
        metadata_record.get("dtypes"),
        repr(["torch.bfloat16"] * 3),
        f"{metadata_path}: input dtypes",
    )
    validator.check_equal(
        metadata_record.get("hardware"),
        EXPECTED_GPU_NAME,
        f"{metadata_path}: hardware",
    )
    settings = metadata_record.get("settings")
    validator.require(isinstance(settings, dict), f"{metadata_path}: settings")
    expected_settings = {
        "backend": "cute",
        "force_autotune": False,
        "effective_cache_read_bypass": True,
        "static_shapes": True,
        "autotune_log_details": True,
        "autotune_compile_timeout": 60,
        "autotune_benchmark_subprocess": True,
        "autotune_benchmark_timeout": AUTOTUNE_BENCHMARK_TIMEOUT_SECONDS,
        "autotune_random_seed": case.tuner_seed,
        "autotune_best_of_k": 1,
        "autotune_accuracy_check": True,
        "autotune_max_generations": None,
        "autotune_budget_seconds": None,
        "autotune_ignore_errors": False,
        "disable_autotuner_heuristics": False,
        "autotune_effort": "full",
        "autotune_config_overrides": {},
        "autotune_seed_configs": None,
        "autotune_config_filter": None,
    }
    for key, expected in expected_settings.items():
        validator.check_equal(
            settings.get(key), expected, f"{metadata_path}: settings.{key}"
        )

    configs = metadata_record.get("configs")
    validator.require(isinstance(configs, dict), f"{metadata_path}: configs map")
    observed_config_ids = {row["config_id"] for row in rows}
    validator.check_equal(
        set(configs), observed_config_ids, f"{metadata_path}: config ID set"
    )
    for config_id, config in configs.items():
        validator.require(
            re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
            and isinstance(config, dict),
            f"{metadata_path}: invalid config entry {config_id!r}",
        )
        validator.check_equal(
            validator.canonical_sha256(config)[:16],
            config_id,
            f"{metadata_path}: canonical config ID {config_id}",
        )
    selected_config_id = validator.canonical_sha256(selected_config)[:16]
    validator.check_equal(
        configs.get(selected_config_id),
        selected_config,
        f"{metadata_path}: selected config",
    )
    for line, row in enumerate(rows, 2):
        config = configs[row["config_id"]]
        config_repr = (
            "Config("
            + ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))
            + ")"
        )
        validator.check_equal(
            row["config"], config_repr, f"{csv_path}:{line}: config payload"
        )
    return {
        "run_id": run_id,
        "configs": configs,
        "attempt_by_config": attempt_by_config,
        "attempt_history_by_config": attempt_history_by_config,
        "csv_sha256": _sha256_bytes(csv_contents),
        "metadata_sha256": metadata_sha256,
    }


def _validate_helion_external_evidence(
    output_root: Path,
    result_path: Path,
    payload: dict[str, Any],
    case: Case,
    manifest: dict[str, Any],
) -> dict[str, object]:
    validator = _load_strict_validator(output_root, manifest)
    result_dir = result_path.parent
    overrides = payload["helion_overrides"]
    provenance = overrides["autotune_provenance"]
    expected_baseline = (
        "examples.attention._causal_attention_relu_output_baseline"
        if case.causal and case.epilogue == "relu"
        else (
            "examples.attention._attention_relu_output_baseline"
            if case.epilogue == "relu"
            else (
                "examples.attention._causal_attention_output_baseline"
                if case.causal
                else "examples.attention._attention_output_baseline"
            )
        )
    )
    _require(
        provenance.get("autotune_baseline_fn") == expected_baseline,
        "strict correctness baseline mismatch",
    )
    _require(
        provenance.get("autotune_benchmark_timeout")
        == AUTOTUNE_BENCHMARK_TIMEOUT_SECONDS,
        "strict benchmark timeout mismatch",
    )

    normalized = copy.deepcopy(payload)
    normalized_overrides = normalized["helion_overrides"]
    normalized_env = normalized_overrides["env_overrides"]
    for key in (
        "HELION_CACHE_DIR",
        "CUTE_DSL_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        normalized_env.pop(key)
    normalized_env["HELION_AUTOTUNE_BENCHMARK_TIMEOUT"] = "60"
    normalized_provenance = normalized_overrides["autotune_provenance"]
    normalized_provenance["autotune_benchmark_timeout"] = 60
    normalized_provenance["autotune_baseline_fn"] = (
        "examples.attention._causal_attention_output_baseline"
        if case.causal
        else "examples.attention._attention_output_baseline"
    )
    try:
        _normalized_provenance, _normalized_trial, _count, _member, _distance = (
            validator.validate_strict_provenance(
                result_path,
                normalized,
                case.variant,
                case.seq_len,
                case.tuner_seed,
                expected_input_shape=(
                    case.z,
                    case.h,
                    case.seq_len,
                    case.head_dim,
                ),
                expected_input_dtype="torch.bfloat16",
            )
        )
        selected_config = provenance["selected_config"]
        selected_source = provenance["selected_source_sha256"]
        trial = provenance["trials"][0]
        ledger = validator.read_and_validate_ledger(
            result_dir / "autotune.sources.csv",
            trial,
            selected_config,
            selected_source,
        )
        autotune = _generalized_autotune_sidecars(
            validator,
            result_dir,
            ledger["rows"],
            ledger["run_id"],
            selected_config,
            case,
        )
        structural = validator.validate_structural_prefix_execution(
            result_path,
            provenance,
            trial,
            ledger["rows"],
            autotune["configs"],
            autotune["attempt_by_config"],
            autotune["attempt_history_by_config"],
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CampaignError(f"full strict evidence validation failed: {exc}") from exc
    compiler_policy = structural["compiler_seed_policy"]
    _require(
        compiler_policy == provenance.get("compiler_seed_policy"),
        "validated compiler seed policy differs from provenance",
    )
    return {
        "run_id": ledger["run_id"],
        "autotune_csv_sha256": autotune["csv_sha256"],
        "autotune_metadata_sha256": autotune["metadata_sha256"],
        "source_ledger_sha256": _sha256_file(result_dir / "autotune.sources.csv"),
        "compiler_seed_policy_sha256": _json_sha256(compiler_policy),
    }


def _relative_artifact_path(output_root: Path, path: Path) -> str:
    return path.resolve().relative_to(output_root.resolve()).as_posix()


def _autotune_sidecar_receipt(
    output_root: Path, result_dir: Path
) -> tuple[str, dict[str, dict[str, str]]]:
    records: dict[str, dict[str, str]] = {}
    run_ids: set[str] = set()
    for name in AUTOTUNE_SIDECAR_NAMES:
        path = result_dir / name
        _require(path.is_file(), f"missing strict autotune sidecar: {name}")
        records[name] = {
            "file": _relative_artifact_path(output_root, path),
            "sha256": _sha256_file(path),
        }
        if name.endswith(".csv"):
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            _require(bool(rows), f"strict autotune sidecar is empty: {name}")
            ids = {row.get("run_id") for row in rows}
            _require(
                len(ids) == 1 and None not in ids,
                f"strict autotune sidecar mixes run IDs: {name}",
            )
            run_id = next(iter(ids))
            if not isinstance(run_id, str):
                raise CampaignError(f"invalid run ID in {name}")
            run_ids.add(run_id)
        else:
            lines = [line for line in path.read_text().splitlines() if line.strip()]
            _require(len(lines) == 1, "autotune metadata must have one record")
            metadata_record = json.loads(lines[0])
            _require(isinstance(metadata_record, dict), "invalid autotune metadata")
            run_id = metadata_record.get("run_id")
            if not isinstance(run_id, str):
                raise CampaignError("autotune metadata has no run ID")
            run_ids.add(run_id)
    _require(len(run_ids) == 1, "strict autotune sidecar run IDs differ")
    return next(iter(run_ids)), records


def _receipt_payload(
    output_root: Path,
    case: Case,
    impl: str,
    result_path: Path,
    invocation_path: Path,
    prevalidation_path: Path | None,
) -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": 1,
        "case_id": case.case_id,
        "implementation": impl,
        "input_spec_sha256": _json_sha256(case.input_spec),
        "campaign_manifest_sha256": _sha256_file(_manifest_path(output_root)),
        "result_file": _relative_artifact_path(output_root, result_path),
        "result_sha256": _sha256_file(result_path),
        "invocation_file": _relative_artifact_path(output_root, invocation_path),
        "invocation_sha256": _sha256_file(invocation_path),
    }
    if prevalidation_path is not None:
        receipt["strict_prevalidation_file"] = _relative_artifact_path(
            output_root, prevalidation_path
        )
        receipt["strict_prevalidation_sha256"] = _sha256_file(prevalidation_path)
        run_id, sidecars = _autotune_sidecar_receipt(output_root, result_path.parent)
        receipt["autotune_run_id"] = run_id
        receipt["autotune_sidecars"] = sidecars
    _reject_absolute_strings(receipt)
    return receipt


def _validate_receipt(
    output_root: Path,
    case: Case,
    impl: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    result_dir = _result_dir(output_root, case, impl)
    result_path = result_dir / "result.json"
    invocation_path = result_dir / "invocation.json"
    receipt_path = result_dir / "receipt.json"
    _require(result_path.is_file(), f"missing result for {case.case_id}/{impl}")
    _require(invocation_path.is_file(), f"missing invocation for {case.case_id}/{impl}")
    _require(receipt_path.is_file(), f"missing receipt for {case.case_id}/{impl}")
    receipt = json.loads(receipt_path.read_text())
    _require(isinstance(receipt, dict), "receipt is not an object")
    expected = _receipt_payload(
        output_root,
        case,
        impl,
        result_path,
        invocation_path,
        result_dir / "strict-prevalidation.json" if impl == "helion-cute" else None,
    )
    _require(receipt == expected, f"receipt mismatch for {case.case_id}/{impl}")
    _require(
        receipt["campaign_manifest_sha256"]
        == _sha256_file(_manifest_path(output_root)),
        "receipt refers to another campaign manifest",
    )
    invocation = json.loads(invocation_path.read_text())
    _require(isinstance(invocation, dict), "invocation is not an object")
    _command, _environment, expected_invocation = _invocation(
        output_root, case, impl, result_dir
    )
    _require(
        invocation == expected_invocation,
        f"invocation mismatch for {case.case_id}/{impl}",
    )
    _reject_absolute_strings(invocation)
    prevalidation: dict[str, Any] | None = None
    if impl == "helion-cute":
        prevalidation_path = result_dir / "strict-prevalidation.json"
        loaded_prevalidation = json.loads(prevalidation_path.read_text())
        _require(
            isinstance(loaded_prevalidation, dict)
            and loaded_prevalidation.get("schema_version") == 1
            and loaded_prevalidation.get("status") == "autotune_complete_prevalidation",
            "strict prevalidation sidecar is malformed",
        )
        prevalidation = loaded_prevalidation
    payload = json.loads(result_path.read_text())
    validated = validate_result_payload(payload, case, impl, manifest, result_dir)
    if impl == "helion-cute":
        assert prevalidation is not None
        prevalidation_provenance = prevalidation.get("autotune_provenance")
        final_provenance = validated["helion_overrides"]["autotune_provenance"]
        if not isinstance(prevalidation_provenance, dict):
            raise CampaignError("strict prevalidation provenance is missing")
        for key, value in prevalidation_provenance.items():
            _require(
                final_provenance.get(key) == value,
                f"strict prevalidation changed before final result: {key}",
            )
        strict_summary = _validate_helion_external_evidence(
            output_root, result_path, validated, case, manifest
        )
        _require(
            receipt.get("autotune_run_id") == strict_summary["run_id"],
            "receipt autotune run ID mismatch",
        )
        sidecars = receipt.get("autotune_sidecars")
        _require(isinstance(sidecars, dict), "receipt sidecar hashes are missing")
        expected_hashes = {
            "autotune.csv": strict_summary["autotune_csv_sha256"],
            "autotune.meta.jsonl": strict_summary["autotune_metadata_sha256"],
            "autotune.sources.csv": strict_summary["source_ledger_sha256"],
        }
        for name, expected_hash in expected_hashes.items():
            _require(
                sidecars[name]["sha256"] == expected_hash,
                f"receipt {name} hash mismatch",
            )
    return validated


def _clean_base_environment() -> dict[str, str]:
    environment: dict[str, str] = {}
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
            environment[name] = value
    environment.setdefault("PATH", os.defpath)
    environment.setdefault("LANG", "C.UTF-8")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _invocation(
    output_root: Path, case: Case, impl: str, result_dir: Path
) -> tuple[list[str], dict[str, str], dict[str, object]]:
    helion_root = output_root / "checkouts/helion"
    fa4_root = output_root / "checkouts/flash-attention"
    quack_root = output_root / "checkouts/quack"
    cache = result_dir / "cache"
    temporary_result = result_dir / "result.pending.json"
    arguments = [
        "benchmarks/cute/compare_attention_backends.py",
        "--impl",
        impl,
        "--z",
        str(case.z),
        "--h",
        str(case.h),
        "--seq-len",
        str(case.seq_len),
        "--head-dim",
        str(case.head_dim),
        "--dtype",
        "bfloat16",
        "--causal",
        str(case.causal),
        "--biased",
        "0",
        "--epilogue",
        case.epilogue,
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
        str(temporary_result),
    ]
    environment = _clean_base_environment()
    environment.update(
        {
            "PYTHONPATH": str(helion_root),
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
    if impl == "helion-cute":
        environment.update(
            {
                "HELION_CACHE_DIR": str(cache / "helion"),
                "HELION_AUTOTUNE_LOG": str(result_dir / "autotune"),
                "HELION_AUTOTUNE_LOG_DETAILS": "1",
            }
        )
        arguments.extend(
            [
                "--helion-force-flash-config",
                "0",
                "--helion-force-autotune",
                "1",
                "--helion-require-full-autotune",
                "1",
                "--helion-return-lse",
                "0",
                "--helion-cute-benchmark-timer",
                "wall",
                "--helion-autotune-effort",
                "full",
                "--helion-autotune-best-of-k",
                "1",
                "--helion-autotune-benchmark-timeout",
                str(AUTOTUNE_BENCHMARK_TIMEOUT_SECONDS),
                "--helion-autotune-accuracy-check",
                "1",
                "--helion-autotuner-initial-population",
                "from_random",
                "--helion-env",
                f"HELION_AUTOTUNE_RANDOM_SEED={case.tuner_seed}",
                "--helion-env",
                f"HELION_CACHE_DIR={cache / 'helion'}",
                "--helion-env",
                f"CUTE_DSL_CACHE_DIR={cache / 'cute_dsl'}",
                "--helion-env",
                f"TORCHINDUCTOR_CACHE_DIR={cache / 'torchinductor'}",
                "--helion-env",
                f"TRITON_CACHE_DIR={cache / 'triton'}",
            ]
        )
    elif impl in {"fa4", "flexattention-cute"}:
        environment["HELION_FA4_ROOT"] = str(fa4_root)
        environment["PYTHONPATH"] = os.pathsep.join((str(helion_root), str(quack_root)))

    script_path = str(helion_root / arguments[0])
    if impl == "helion-cute":
        bootstrap = (
            "import os, runpy, sys; "
            "script = sys.argv[1]; sys.argv = sys.argv[1:]; "
            "os.environ.pop('PYTHONPATH', None); "
            "runpy.run_path(script, run_name='__main__')"
        )
    else:
        bootstrap = (
            "import runpy, sys; "
            "script = sys.argv[1]; sys.argv = sys.argv[1:]; "
            "runpy.run_path(script, run_name='__main__')"
        )
    command = [sys.executable, "-c", bootstrap, script_path, *arguments[1:]]
    public_arguments: list[str] = []
    cache_arguments = {
        "HELION_CACHE_DIR": "cache/helion",
        "CUTE_DSL_CACHE_DIR": "cache/cute_dsl",
        "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
        "TRITON_CACHE_DIR": "cache/triton",
    }
    for argument in arguments:
        if argument == str(temporary_result):
            public_arguments.append("result.pending.json")
            continue
        key, separator, _value = argument.partition("=")
        if separator and key in cache_arguments:
            public_arguments.append(f"{key}={cache_arguments[key]}")
            continue
        public_arguments.append(argument)
    invocation_record: dict[str, object] = {
        "schema_version": 1,
        "case_id": case.case_id,
        "implementation": impl,
        "arguments": public_arguments,
        "physical_gpu": case.physical_gpu,
        "input_spec_sha256": _json_sha256(case.input_spec),
        "isolated_cache": True,
        "fresh_directory": True,
        "pythonpath": (
            "measured_helion_then_pinned_quack"
            if impl in {"fa4", "flexattention-cute"}
            else "measured_helion"
        ),
        "pythonpath_scrubbed_before_harness": impl == "helion-cute",
    }
    _reject_absolute_strings(invocation_record)
    return command, environment, invocation_record


def _quarantine(result_dir: Path, output_root: Path) -> None:
    if not result_dir.exists():
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = (
        output_root
        / "quarantine"
        / f"{result_dir.parent.name}--{result_dir.name}--{stamp}"
    )
    destination = base
    counter = 1
    while destination.exists():
        destination = Path(f"{base}-{counter}")
        counter += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    result_dir.rename(destination)


def run_one(output_root: Path, case: Case, impl: str) -> None:
    manifest = _load_manifest(output_root)
    _validate_live_state(manifest, output_root)
    result_dir = _result_dir(output_root, case, impl)
    if result_dir.exists():
        try:
            _validate_receipt(output_root, case, impl, manifest)
        except (CampaignError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"quarantining incomplete {case.case_id}/{impl}: {exc}")
            _quarantine(result_dir, output_root)
        else:
            print(f"SKIP validated {case.case_id}/{impl}")
            return
    _validate_gpu_idle(case.physical_gpu)
    result_dir.mkdir(parents=True)
    for name in (
        "cuda",
        "cute_dsl",
        "helion",
        "pycache",
        "tmp",
        "torchinductor",
        "triton",
        "xdg",
    ):
        (result_dir / "cache" / name).mkdir(parents=True)
    command, environment, invocation = _invocation(output_root, case, impl, result_dir)
    invocation_path = result_dir / "invocation.json"
    _atomic_write_json(invocation_path, invocation)
    log_path = result_dir / "run.log"
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
    temporary_result = result_dir / "result.pending.json"
    _require(temporary_result.is_file(), "benchmark did not write its result")
    payload = json.loads(temporary_result.read_text())
    validate_result_payload(payload, case, impl, manifest, result_dir)
    if impl == "helion-cute":
        _validate_helion_external_evidence(
            output_root, temporary_result, payload, case, manifest
        )
    result_path = result_dir / "result.json"
    os.replace(temporary_result, result_path)
    prevalidation: Path | None = None
    if impl == "helion-cute":
        temporary_prevalidation = (
            result_dir / "result.pending.strict-prevalidation.json"
        )
        _require(temporary_prevalidation.is_file(), "strict prevalidation is missing")
        prevalidation = result_dir / "strict-prevalidation.json"
        os.replace(temporary_prevalidation, prevalidation)
    _validate_live_state(manifest, output_root)
    receipt = _receipt_payload(
        output_root,
        case,
        impl,
        result_path,
        invocation_path,
        prevalidation,
    )
    _atomic_write_json(result_dir / "receipt.json", receipt)
    _validate_receipt(output_root, case, impl, manifest)
    print(f"PASS {case.case_id}/{impl}")


def run_lane(output_root: Path, lane: str) -> None:
    _require(lane in {"dense", "causal"}, f"unknown lane: {lane}")
    expected_gpu = 7 if lane == "dense" else 6
    for case in CASES:
        if case.variant != lane:
            continue
        _require(
            case.physical_gpu == expected_gpu,
            f"{case.case_id} is assigned to the wrong physical GPU",
        )
        for impl in IMPLEMENTATIONS:
            run_one(output_root, case, impl)


def _result_notes(payload: dict[str, Any], case: Case, impl: str) -> str:
    notes = [str(value) for value in payload.get("notes", [])]
    if impl == "helion-cute":
        notes.insert(
            0,
            "Fresh strict full autotune; 64-launch correctness and exact repeat passed.",
        )
    if case.epilogue == "relu":
        notes.append("ReLU FLOPs excluded from the throughput numerator.")
    return " ".join(notes)


def build_outputs(
    output_root: Path, *, validate_live: bool = True
) -> tuple[Path, Path]:
    manifest = _load_manifest(output_root)
    if validate_live:
        _validate_live_state(manifest, output_root)
    else:
        _validate_manifest_static(manifest, output_root)
    rows: list[dict[str, object]] = []
    measurements: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    for case in CASES:
        for impl in IMPLEMENTATIONS:
            payload = _validate_receipt(output_root, case, impl, manifest)
            result_path = _result_dir(output_root, case, impl) / "result.json"
            receipt_path = result_path.with_name("receipt.json")
            artifact = _relative_artifact_path(output_root, result_path)
            runs = [float(value) for value in payload["runs_ms"]]
            median_ms = statistics.median(runs)
            tflops = _flop_count(case) / median_ms / 1e9
            rows.append(
                {
                    "shape_order": case.order,
                    "shape_group": case.group,
                    "variant": case.variant,
                    "z": case.z,
                    "h": case.h,
                    "seq_len": case.seq_len,
                    "head_dim": case.head_dim,
                    "dtype": "bfloat16",
                    "epilogue": "ReLU" if case.epilogue == "relu" else "",
                    "implementation": impl,
                    "implementation_label": IMPLEMENTATION_LABELS[impl],
                    "version": payload["version"],
                    "tflops": tflops,
                    "statistic": "median",
                    "sample_count": SAMPLE_COUNT,
                    "gpu": EXPECTED_GPU_NAME,
                    "physical_gpu": case.physical_gpu,
                    "power_cap_w": EXPECTED_POWER_CAP_W,
                    "power_cap_recorded": "yes",
                    "correctness": "PASS",
                    "timing_mode": (
                        "strict_full_autotune"
                        if impl == "helion-cute"
                        else "fresh_standalone"
                    ),
                    "evidence_file": artifact,
                    "notes": _result_notes(payload, case, impl),
                }
            )
            measurements.append(
                {
                    "shape_order": case.order,
                    "implementation": impl,
                    "shape_bhsd": [case.z, case.h, case.seq_len, case.head_dim],
                    "variant": case.variant,
                    "dtype": "bfloat16",
                    "epilogue": None if case.epilogue == "none" else "ReLU",
                    "gpu": EXPECTED_GPU_NAME,
                    "physical_gpu": case.physical_gpu,
                    "power_cap_w": EXPECTED_POWER_CAP_W,
                    "power_cap_recorded": "yes",
                    "correctness": "PASS",
                    "timing_mode": (
                        "strict_full_autotune"
                        if impl == "helion-cute"
                        else "fresh_standalone"
                    ),
                    "flop_count": _flop_count(case),
                    "flop_model": (
                        "QK+PV; causal uses one-half dense work; ReLU excluded"
                    ),
                    "runs_ms": runs,
                    "median_ms": median_ms,
                    "median_tflops": tflops,
                    "source_artifact": artifact,
                    "source_result_sha256": _sha256_file(result_path),
                    "input_spec_sha256": _json_sha256(case.input_spec),
                }
            )
            artifact_record: dict[str, object] = {
                "result": artifact,
                "result_sha256": _sha256_file(result_path),
                "receipt": _relative_artifact_path(output_root, receipt_path),
                "receipt_sha256": _sha256_file(receipt_path),
            }
            if impl == "helion-cute":
                receipt = json.loads(receipt_path.read_text())
                provenance = payload["helion_overrides"]["autotune_provenance"]
                artifact_record.update(
                    {
                        "autotune_run_id": receipt["autotune_run_id"],
                        "autotune_sidecars": receipt["autotune_sidecars"],
                        "compiler_seed_policy_sha256": _json_sha256(
                            provenance["compiler_seed_policy"]
                        ),
                    }
                )
            artifacts.append(artifact_record)

    published = output_root / "published"
    csv_path = published / "attention_varied_shapes_b200_750w.csv"
    evidence_path = published / "attention_varied_shapes_b200_750w_evidence.json"
    evidence = {
        "schema_version": 2,
        "description": (
            "Compact raw timing evidence for the CuTe 4.7 varied BF16 attention "
            "campaign. TFLOP/s is flop_count / median(runs_ms) / 1e9."
        ),
        "hardware": {"gpu": EXPECTED_GPU_NAME, "power_cap_w": 750},
        "helion_version_scope": (
            "Every Helion result is from the single authenticated measured commit "
            "recorded in campaign_provenance."
        ),
        "implementation_versions": {
            impl: manifest["implementation_versions"][impl]["version"]
            for impl in IMPLEMENTATIONS
        },
        "campaign_provenance": {
            "manifest_sha256": _sha256_file(_manifest_path(output_root)),
            "measured_commit": manifest["sources"]["helion"]["commit"],
            "helion_tree": manifest["sources"]["helion"]["tree"],
            "helion_source_subset_sha256": manifest["sources"]["helion"][
                "source_subset"
            ]["sha256"],
            "benchmark_sha256": manifest["sources"]["helion"]["benchmark_sha256"],
            "attention_sha256": manifest["sources"]["helion"]["attention_sha256"],
            "fa4_commit": manifest["sources"]["flash_attention"]["commit"],
            "fa4_tree": manifest["sources"]["flash_attention"]["tree"],
            "quack_commit": manifest["sources"]["quack"]["commit"],
            "quack_tree": manifest["sources"]["quack"]["tree"],
            "toolchain_sha256": _json_sha256(manifest["toolchain"]),
            "setup_files": manifest["sources"]["setup_files"],
            "input_generator": CASES[0].input_spec["generator"],
            "input_seed": INPUT_SEED,
        },
        "measurements": measurements,
        "source_artifacts": artifacts,
    }
    _reject_absolute_strings(evidence)
    published.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=published,
        prefix=f".{csv_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_csv = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary_csv, csv_path)
    finally:
        temporary_csv.unlink(missing_ok=True)
    _atomic_write_json(evidence_path, evidence)
    print(f"wrote {csv_path}")
    print(f"wrote {evidence_path}")
    return csv_path, evidence_path


def _parse_optional_path(value: str | None) -> Path | None:
    return None if value is None or not value else Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize_parser = subparsers.add_parser("initialize")
    initialize_parser.add_argument("output_root", type=Path)
    initialize_parser.add_argument("--source-repo")
    initialize_parser.add_argument("--fa4-source-repo")
    initialize_parser.add_argument("--quack-source-repo")

    lane_parser = subparsers.add_parser("run-lane")
    lane_parser.add_argument("output_root", type=Path)
    lane_parser.add_argument("--lane", choices=("dense", "causal"), required=True)

    one_parser = subparsers.add_parser("run-one")
    one_parser.add_argument("output_root", type=Path)
    one_parser.add_argument("--case", choices=tuple(CASES_BY_ID), required=True)
    one_parser.add_argument("--impl", choices=IMPLEMENTATIONS, required=True)

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
                _parse_optional_path(args.source_repo),
                _parse_optional_path(args.fa4_source_repo),
                _parse_optional_path(args.quack_source_repo),
            )
        elif args.command == "run-lane":
            run_lane(args.output_root.resolve(), args.lane)
        elif args.command == "run-one":
            run_one(args.output_root.resolve(), _case(args.case), str(args.impl))
        elif args.command == "validate":
            output_root = args.output_root.resolve()
            manifest = _load_manifest(output_root)
            _validate_live_state(manifest, output_root)
            for case in CASES:
                for impl in IMPLEMENTATIONS:
                    _validate_receipt(output_root, case, impl, manifest)
            print("all 32 results passed validation")
        elif args.command == "build":
            build_outputs(args.output_root.resolve())
        else:
            raise AssertionError(args.command)
    except CampaignError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
