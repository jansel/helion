from __future__ import annotations

import argparse
from concurrent.futures import FIRST_EXCEPTION
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from contextlib import contextmanager
from contextlib import suppress
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from typing import Iterator

import build_strict_manifest as strict
import validate_generalization_campaign as validator

SCRIPT_PATH = Path(__file__).resolve()
SETUP_ROOT = SCRIPT_PATH.parent
DEFAULT_MATRIX = SETUP_ROOT / "generalization_cases.csv"
VALIDATOR_PATH = SETUP_ROOT / "validate_generalization_campaign.py"
REMEASUREMENT_PATH = SETUP_ROOT / "remeasure_generalization_winners.py"
STRICT_VALIDATOR_PATH = SETUP_ROOT / "build_strict_manifest.py"
BENCHMARK_RELATIVE = Path("benchmarks/cute/compare_attention_backends.py")
EXPECTED_COMMIT = "c3e36b65d69681c23e053042b0bc21e2331bad17"
SANITIZED_PREFIXES = (
    "HELION_",
    "CUTE_DSL_",
    "CUDA_MPS_",
    "CUDNN_",
    "TORCH_CUDNN_",
    "TRITON_",
    "TORCHINDUCTOR_",
    "PYTORCH_TUNABLEOP_",
    "CUDA_CACHE_",
)
SANITIZED_EXACT = {
    "CUBLAS_FORCE_TF32",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_AUTO_BOOST",
    "CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_DEVICE_ORDER",
    "CUDA_DISABLE_PTX_JIT",
    "CUDA_FORCE_PTX_JIT",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_MANAGED_FORCE_DEVICE_ALLOC",
    "CUDA_MODULE_LOADING",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_TF32_OVERRIDE",
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "XDG_CACHE_HOME",
}
event_lock = threading.Lock()
PROCESS_GROUP_TERM_GRACE_SECONDS = 10.0
PROCESS_GROUP_KILL_GRACE_SECONDS = 5.0
PROCESS_GROUP_POLL_SECONDS = 0.05


def canonical_executable(path: str) -> Path:
    candidate = shutil.which(path) if "/" not in path else path
    strict.require(candidate is not None, f"Python executable not found: {path}")
    result = Path(candidate).resolve(strict=True)
    strict.require(
        result.is_file() and os.access(result, os.X_OK), f"not executable: {result}"
    )
    strict.require(not result.is_symlink(), f"Python executable is a symlink: {result}")
    return result


def require_python_environment(python_executable: Path) -> None:
    actual = subprocess.run(
        [
            str(python_executable),
            "-c",
            (
                "from importlib.metadata import version; "
                "print(version('nvidia-cutlass-dsl'))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    strict.check_equal(actual, strict.EXPECTED_CUTE_VERSION, "CuTe version")


def require_clean_checkout(repo: Path) -> None:
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
    strict.require(not status, f"measured checkout is not clean:\n{status}")


def git_commit(repo: Path) -> str:
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    strict.require(len(commit) == 40, f"invalid measured commit: {commit!r}")
    return commit


def require_commit_contract() -> None:
    strict.check_equal(
        validator.EXPECTED_COMMIT,
        EXPECTED_COMMIT,
        "launcher and validator expected commit",
    )


def require_checkout_identity(
    repo: Path, expected_commit: str, expected_benchmark_sha256: str
) -> None:
    require_commit_contract()
    strict.check_equal(expected_commit, EXPECTED_COMMIT, "declared campaign commit")
    require_clean_checkout(repo)
    strict.check_equal(git_commit(repo), EXPECTED_COMMIT, "measured checkout commit")
    strict.check_equal(
        strict.file_sha256(repo / BENCHMARK_RELATIVE),
        expected_benchmark_sha256,
        "measured benchmark source",
    )


def result_directory(root: Path, run: validator.RunSpec) -> Path:
    return (root / run.result_path).parent


def build_command(
    python_executable: Path,
    repo: Path,
    root: Path,
    run: validator.RunSpec,
) -> list[str]:
    return validator.expected_command(python_executable, repo, root, run)


def sanitized_environment(
    repo: Path, root: Path, run: validator.RunSpec
) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in SANITIZED_EXACT
        and not any(key.startswith(prefix) for prefix in SANITIZED_PREFIXES)
    }
    output = result_directory(root, run)
    cache = output / "cache"
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(run.case.physical_gpu),
            "CUDA_CACHE_DISABLE": "0",
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "CUTE_DSL_CACHE_DIR": str(cache / "cute_dsl"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS": "6,7",
            "HELION_BACKEND": "cute",
            "HELION_CACHE_DIR": str(cache / "helion"),
            "HELION_AUTOTUNE_LOG": str(output / "autotune"),
            "HELION_AUTOTUNE_LOG_DETAILS": "1",
        }
    )
    return env


def gpu_query(gpu: int, field: str) -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            f"--query-gpu={field}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_gpu(gpu: int) -> None:
    strict.check_equal(gpu_query(gpu, "name"), "NVIDIA B200", f"GPU {gpu} model")
    power = float(gpu_query(gpu, "power.limit"))
    strict.require(749.5 <= power <= 750.5, f"GPU {gpu} power limit is {power}")
    active = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    strict.require(
        not any(character.isdigit() for character in active),
        f"GPU {gpu} busy: {active}",
    )


def process_group_live(process_group: int) -> bool:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-g", str(process_group)],
        check=False,
        capture_output=True,
        text=True,
    )
    strict.require(result.returncode in (0, 1), "failed to inspect process group")
    return any(
        state and not state.startswith("Z")
        for line in result.stdout.splitlines()
        if (state := line.strip())
    )


def drain_process_group(process: subprocess.Popen[Any]) -> int:
    returncode = process.poll()
    if process_group_live(process.pid):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + PROCESS_GROUP_TERM_GRACE_SECONDS
        while process_group_live(process.pid) and time.monotonic() < deadline:
            if returncode is None:
                returncode = process.poll()
            time.sleep(PROCESS_GROUP_POLL_SECONDS)
        if process_group_live(process.pid):
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            deadline = time.monotonic() + PROCESS_GROUP_KILL_GRACE_SECONDS
            while process_group_live(process.pid) and time.monotonic() < deadline:
                if returncode is None:
                    returncode = process.poll()
                time.sleep(PROCESS_GROUP_POLL_SECONDS)
    if returncode is None:
        returncode = process.wait()
    else:
        process.wait()
    strict.require(
        not process_group_live(process.pid),
        f"process group {process.pid} survived termination",
    )
    return returncode


def wait_for_process(
    process: subprocess.Popen[Any], stop_event: threading.Event
) -> int:
    while process.poll() is None and not stop_event.wait(0.5):
        pass
    return drain_process_group(process)


def snapshot(source: Path, destination: Path) -> str:
    strict.require(
        source.is_file() and not source.is_symlink(), f"invalid source: {source}"
    )
    strict.require(not destination.exists(), f"snapshot exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o555 if os.access(source, os.X_OK) else 0o444)
    digest = strict.file_sha256(destination)
    destination.with_name(f"{destination.name}.sha256").write_text(f"{digest}\n")
    return digest


def declaration_record(run: validator.RunSpec, command: list[str]) -> dict[str, Any]:
    return {
        "record_type": "run",
        "run_id": run.run_id,
        "case_id": run.case.case_id,
        "tuner_seed": run.tuner_seed,
        "physical_gpu": run.case.physical_gpu,
        "result_path": run.result_path.as_posix(),
        "command": command,
        "command_sha256": strict.canonical_sha256(command),
    }


def remeasurement_declaration_record(
    case: validator.CaseSpec, command: list[str]
) -> dict[str, Any]:
    return {
        "record_type": "remeasure",
        "case_id": case.case_id,
        "physical_gpu": case.physical_gpu,
        "result_path": f"remeasure/{case.case_id}.json",
        "command": command,
        "command_sha256": strict.canonical_sha256(command),
    }


def canonical_lines(records: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(strict.canonical_json(record) for record in records) + "\n"
    ).encode()


def initialize_campaign(
    root: Path,
    repo: Path,
    python_executable: Path,
    matrix: Path,
    cases: tuple[validator.CaseSpec, ...],
) -> tuple[validator.RunSpec, ...]:
    require_commit_contract()
    strict.require(not root.exists(), f"output root exists; use --resume: {root}")
    strict.require(
        repo not in root.parents and root != repo, "output root is inside checkout"
    )
    strict.check_equal(git_commit(repo), EXPECTED_COMMIT, "campaign checkout commit")
    root.mkdir(parents=True)
    launcher_dir = root / "launcher"
    matrix_digest = snapshot(matrix, launcher_dir / "generalization_cases.csv")
    launcher_digest = snapshot(SCRIPT_PATH, launcher_dir / SCRIPT_PATH.name)
    validator_digest = snapshot(VALIDATOR_PATH, launcher_dir / VALIDATOR_PATH.name)
    strict_validator_digest = snapshot(
        STRICT_VALIDATOR_PATH, launcher_dir / STRICT_VALIDATOR_PATH.name
    )
    remeasurement_digest = snapshot(
        REMEASUREMENT_PATH, launcher_dir / REMEASUREMENT_PATH.name
    )
    benchmark_digest = snapshot(
        repo / BENCHMARK_RELATIVE, launcher_dir / BENCHMARK_RELATIVE.name
    )
    runs = validator.expand_runs(cases)
    header = {
        "record_type": "campaign",
        "schema_version": validator.SCHEMA_VERSION,
        "expected_commit": EXPECTED_COMMIT,
        "repo_root": str(repo),
        "python_executable": str(python_executable),
        "matrix_path": "launcher/generalization_cases.csv",
        "matrix_sha256": matrix_digest,
        "launcher_path": f"launcher/{SCRIPT_PATH.name}",
        "launcher_sha256": launcher_digest,
        "validator_path": f"launcher/{VALIDATOR_PATH.name}",
        "validator_sha256": validator_digest,
        "strict_validator_path": f"launcher/{STRICT_VALIDATOR_PATH.name}",
        "strict_validator_sha256": strict_validator_digest,
        "remeasurement_path": f"launcher/{REMEASUREMENT_PATH.name}",
        "remeasurement_sha256": remeasurement_digest,
        "benchmark_path": f"launcher/{BENCHMARK_RELATIVE.name}",
        "benchmark_sha256": benchmark_digest,
    }
    declarations = [
        declaration_record(run, build_command(python_executable, repo, root, run))
        for run in runs
    ]
    remeasurement_declarations = [
        remeasurement_declaration_record(
            case,
            validator.expected_remeasurement_command(
                python_executable,
                repo,
                root,
                case,
                remeasurement_digest,
            ),
        )
        for case in cases
    ]
    contents = canonical_lines([header, *declarations, *remeasurement_declarations])
    (root / "campaign.jsonl").write_bytes(contents)
    (root / "campaign.declarations.sha256").write_text(
        f"{strict.sha256_bytes(contents)}\n"
    )
    return runs


def resume_campaign(
    root: Path,
    repo: Path,
    python_executable: Path,
    matrix: Path,
) -> tuple[validator.RunSpec, ...]:
    require_commit_contract()
    records = validator.load_campaign_records(root)
    header = records[0]
    strict.check_equal(header.get("repo_root"), str(repo), "resume repository")
    strict.check_equal(
        header.get("expected_commit"), EXPECTED_COMMIT, "resume declared commit"
    )
    strict.check_equal(git_commit(repo), EXPECTED_COMMIT, "resume checkout commit")
    strict.check_equal(
        header.get("python_executable"), str(python_executable), "resume Python"
    )
    current_hashes = {
        "matrix_sha256": strict.file_sha256(matrix),
        "launcher_sha256": strict.file_sha256(SCRIPT_PATH),
        "validator_sha256": strict.file_sha256(VALIDATOR_PATH),
        "strict_validator_sha256": strict.file_sha256(STRICT_VALIDATOR_PATH),
        "remeasurement_sha256": strict.file_sha256(REMEASUREMENT_PATH),
        "benchmark_sha256": strict.file_sha256(repo / BENCHMARK_RELATIVE),
    }
    for field, digest in current_hashes.items():
        strict.check_equal(header.get(field), digest, f"resume {field}")
    cases = validator.parse_case_matrix(root / str(header["matrix_path"]))
    runs = validator.expand_runs(cases)
    declarations = validator.declared_runs(records)
    expected = [
        declaration_record(run, build_command(python_executable, repo, root, run))
        for run in runs
    ]
    strict.check_equal(declarations, expected, "resume declarations")
    remeasurement_declarations = validator.declared_remeasurements(records)
    expected_remeasurements = [
        remeasurement_declaration_record(
            case,
            validator.expected_remeasurement_command(
                python_executable,
                repo,
                root,
                case,
                str(header["remeasurement_sha256"]),
            ),
        )
        for case in cases
    ]
    strict.check_equal(
        remeasurement_declarations,
        expected_remeasurements,
        "resume remeasurement declarations",
    )
    return runs


def append_event(root: Path, event: dict[str, Any]) -> None:
    record = {
        "record_type": "event",
        "timestamp_ns": time.time_ns(),
        **event,
    }
    data = f"{strict.canonical_json(record)}\n".encode()
    with event_lock, (root / "campaign.jsonl").open("ab", buffering=0) as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(data)
        os.fsync(handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_UN)


def prior_events(root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for record in validator.load_campaign_records(root):
        if record.get("record_type") == "event" and isinstance(
            record.get("run_id"), str
        ):
            result.setdefault(record["run_id"], []).append(record)
    return result


def completed_result(path: Path) -> bool:
    return all(path.with_name(name).is_file() for name in validator.RESULT_FILENAMES)


def classify_resume(
    root: Path,
    run: validator.RunSpec,
    history: list[dict[str, Any]],
) -> str:
    path = root / run.result_path
    present = [
        name for name in validator.RESULT_FILENAMES if path.with_name(name).exists()
    ]
    success = any(
        event.get("event") == "attempt_finished" and event.get("returncode") == 0
        for event in history
    )
    started = any(event.get("event") == "attempt_started" for event in history)
    failure = any(
        event.get("event") == "attempt_finished" and event.get("returncode") != 0
        for event in history
    )
    if completed_result(path) and success:
        return "accept"
    if present or failure or started:
        raise RuntimeError(
            f"{run.run_id}: incomplete or failed prior attempt; refusing an implicit retry"
        )
    return "run"


def prepare_run_directory(output: Path) -> None:
    strict.require(not output.exists(), f"run output already exists: {output}")
    for relative in (
        "autotune",
        "cache/cuda",
        "cache/cute_dsl",
        "cache/helion",
        "cache/torchinductor",
        "cache/triton",
        "cache/xdg",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)


def execute_run(
    root: Path,
    repo: Path,
    python_executable: Path,
    run: validator.RunSpec,
    expected_commit: str,
    expected_benchmark_sha256: str,
    stop_event: threading.Event,
) -> None:
    gpu = run.case.physical_gpu
    require_checkout_identity(repo, expected_commit, expected_benchmark_sha256)
    validate_gpu(gpu)
    output = result_directory(root, run)
    prepare_run_directory(output)
    command = build_command(python_executable, repo, root, run)
    append_event(root, {"event": "attempt_started", "run_id": run.run_id})
    with (output / "run.log").open("w") as log:
        log.write(
            f"PYTHON executable={python_executable} version={sys.version.replace(chr(10), ' ')}\n"
        )
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=sanitized_environment(repo, root, run),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        returncode = wait_for_process(process, stop_event)
    append_event(
        root,
        {
            "event": "attempt_finished",
            "run_id": run.run_id,
            "returncode": returncode,
        },
    )
    strict.require(returncode == 0, f"{run.run_id}: benchmark failed")
    strict.require(
        completed_result(root / run.result_path), f"{run.run_id}: missing output"
    )
    validate_gpu(gpu)
    require_checkout_identity(repo, expected_commit, expected_benchmark_sha256)


def run_lane(
    root: Path,
    repo: Path,
    python_executable: Path,
    runs: list[validator.RunSpec],
    histories: dict[str, list[dict[str, Any]]],
    expected_commit: str,
    expected_benchmark_sha256: str,
    stop_event: threading.Event,
) -> None:
    try:
        for run in runs:
            strict.require(not stop_event.is_set(), "peer tuning lane failed")
            disposition = classify_resume(root, run, histories.get(run.run_id, []))
            if disposition == "accept":
                continue
            execute_run(
                root,
                repo,
                python_executable,
                run,
                expected_commit,
                expected_benchmark_sha256,
                stop_event,
            )
    except BaseException:
        stop_event.set()
        raise


def remeasurement_environment(
    repo: Path, root: Path, case: validator.CaseSpec
) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in SANITIZED_EXACT
        and not any(key.startswith(prefix) for prefix in SANITIZED_PREFIXES)
    }
    cache = root / "remeasure_cache" / case.case_id
    for relative in (
        "cuda",
        "cute_dsl",
        "helion",
        "pycache",
        "torchinductor",
        "triton",
        "xdg",
    ):
        (cache / relative).mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(repo),
            "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(case.physical_gpu),
            "CUDA_CACHE_DISABLE": "0",
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "CUTE_DSL_CACHE_DIR": str(cache / "cute_dsl"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "HELION_BENCHMARK_ALLOWED_PHYSICAL_GPUS": "6,7",
            "HELION_BACKEND": "cute",
            "HELION_CACHE_DIR": str(cache / "helion"),
            "HELION_DISABLE_AUTOTUNER_HEURISTICS": "0",
        }
    )
    return env


def classify_remeasurement_resume(
    root: Path,
    case: validator.CaseSpec,
    records: list[dict[str, Any]],
) -> str:
    output = root / "remeasure" / f"{case.case_id}.json"
    history = [record for record in records if record.get("case_id") == case.case_id]
    started = any(record.get("event") == "remeasurement_started" for record in history)
    success = any(
        record.get("event") == "remeasurement_finished"
        and record.get("returncode") == 0
        for record in history
    )
    failure = any(
        record.get("event") == "remeasurement_finished"
        and record.get("returncode") != 0
        for record in history
    )
    if output.is_file() and success:
        return "accept"
    if output.exists() or started or failure:
        raise RuntimeError(
            f"{case.case_id}: incomplete or failed remeasurement; refusing implicit retry"
        )
    return "run"


def execute_remeasurement(
    root: Path,
    repo: Path,
    python_executable: Path,
    case: validator.CaseSpec,
    worker_sha256: str,
    expected_commit: str,
    expected_benchmark_sha256: str,
    stop_event: threading.Event,
) -> None:
    require_checkout_identity(repo, expected_commit, expected_benchmark_sha256)
    validate_gpu(case.physical_gpu)
    output = root / "remeasure" / f"{case.case_id}.json"
    strict.require(not output.exists(), f"remeasurement output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = validator.expected_remeasurement_command(
        python_executable, repo, root, case, worker_sha256
    )
    append_event(root, {"event": "remeasurement_started", "case_id": case.case_id})
    log_path = output.with_suffix(".log")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=remeasurement_environment(repo, root, case),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        returncode = wait_for_process(process, stop_event)
    append_event(
        root,
        {
            "event": "remeasurement_finished",
            "case_id": case.case_id,
            "returncode": returncode,
        },
    )
    strict.require(returncode == 0, f"{case.case_id}: remeasurement failed")
    strict.require(output.is_file(), f"{case.case_id}: missing remeasurement output")
    validate_gpu(case.physical_gpu)
    require_checkout_identity(repo, expected_commit, expected_benchmark_sha256)


def run_remeasurement_lane(
    root: Path,
    repo: Path,
    python_executable: Path,
    cases: list[validator.CaseSpec],
    records: list[dict[str, Any]],
    worker_sha256: str,
    expected_commit: str,
    expected_benchmark_sha256: str,
    stop_event: threading.Event,
) -> None:
    try:
        for case in cases:
            strict.require(not stop_event.is_set(), "peer remeasurement lane failed")
            disposition = classify_remeasurement_resume(root, case, records)
            if disposition == "accept":
                continue
            execute_remeasurement(
                root,
                repo,
                python_executable,
                case,
                worker_sha256,
                expected_commit,
                expected_benchmark_sha256,
                stop_event,
            )
    except BaseException:
        stop_event.set()
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generalized strict full-autotune campaign"
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@contextmanager
def campaign_lock(root: Path) -> Iterator[None]:
    lock_path = root.with_name(f"{root.name}.campaign.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another campaign process holds {lock_path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_campaign(
    args: argparse.Namespace,
    root: Path,
    repo: Path,
    matrix: Path,
    python_executable: str,
) -> None:
    require_python_environment(python_executable)
    strict.require((repo / BENCHMARK_RELATIVE).is_file(), f"invalid repository: {repo}")
    require_clean_checkout(repo)
    cases = validator.parse_case_matrix(matrix)
    if args.resume:
        strict.require(root.is_dir(), f"resume root does not exist: {root}")
        runs = resume_campaign(root, repo, python_executable, matrix)
    else:
        runs = initialize_campaign(root, repo, python_executable, matrix, cases)
    header = validator.load_campaign_records(root)[0]
    expected_commit = str(header["expected_commit"])
    expected_benchmark_sha256 = str(header["benchmark_sha256"])
    require_checkout_identity(repo, expected_commit, expected_benchmark_sha256)
    if any(
        record.get("record_type") == "event"
        and record.get("event") == "campaign_validated"
        for record in validator.load_campaign_records(root)
    ):
        validator.validate_campaign(root)
        return
    histories = prior_events(root)
    dense = [run for run in runs if not run.case.causal]
    causal = [run for run in runs if run.case.causal]
    strict.require(dense and causal, "campaign requires both lanes")
    for gpu in (6, 7):
        validate_gpu(gpu)
    tuning_stop = threading.Event()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="attention-lane") as pool:
        futures = [
            pool.submit(
                run_lane,
                root,
                repo,
                python_executable,
                dense,
                histories,
                expected_commit,
                expected_benchmark_sha256,
                tuning_stop,
            ),
            pool.submit(
                run_lane,
                root,
                repo,
                python_executable,
                causal,
                histories,
                expected_commit,
                expected_benchmark_sha256,
                tuning_stop,
            ),
        ]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for future in done:
            future.result()
        for future in pending:
            future.result()
    validation = validator.validate_campaign(root, require_remeasurement=False)
    records = validator.load_campaign_records(root)
    remeasurement_stop = threading.Event()
    with ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="remeasurement-lane"
    ) as pool:
        futures = [
            pool.submit(
                run_remeasurement_lane,
                root,
                repo,
                python_executable,
                [case for case in validation.cases if not case.causal],
                records,
                str(header["remeasurement_sha256"]),
                expected_commit,
                expected_benchmark_sha256,
                remeasurement_stop,
            ),
            pool.submit(
                run_remeasurement_lane,
                root,
                repo,
                python_executable,
                [case for case in validation.cases if case.causal],
                records,
                str(header["remeasurement_sha256"]),
                expected_commit,
                expected_benchmark_sha256,
                remeasurement_stop,
            ),
        ]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for future in done:
            future.result()
        for future in pending:
            future.result()
    validator.validate_campaign(root)
    append_event(root, {"event": "campaign_validated"})
    validator.validate_campaign(root)


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    matrix = args.matrix.expanduser().resolve()
    python_executable = canonical_executable(args.python)
    with campaign_lock(root):
        _run_campaign(args, root, repo, matrix, python_executable)


if __name__ == "__main__":
    main()
