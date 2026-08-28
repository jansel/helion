from __future__ import annotations

import argparse
from concurrent.futures import FIRST_EXCEPTION
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from contextlib import contextmanager
from contextlib import suppress
import csv
import fcntl
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from typing import Iterator

import build_heldout_manifest as heldout
import build_strict_manifest as strict
import heldout_adjudication as adjudication

SCRIPT_PATH = Path(__file__).resolve()
SETUP_ROOT = SCRIPT_PATH.parent
PROCESS_GROUP_TERM_GRACE_SECONDS = 10.0
PROCESS_GROUP_KILL_GRACE_SECONDS = 5.0
PROCESS_GROUP_POLL_SECONDS = 0.05
EVENT_LOCK = threading.Lock()


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
    strict.check_equal(actual, adjudication.EXPECTED_CUTE_VERSION, "CuTe version")


def snapshot(source: Path, destination: Path) -> str:
    adjudication.require_regular_file(source, "snapshot source")
    strict.require(not destination.exists(), f"snapshot exists: {destination}")
    before = strict.file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    after = strict.file_sha256(source)
    strict.check_equal(after, before, f"snapshot source changed: {source}")
    strict.check_equal(
        strict.file_sha256(destination), before, f"snapshot copy changed: {source}"
    )
    destination.chmod(0o444)
    return before


def write_immutable_json(path: Path, value: object) -> str:
    strict.require(not path.exists(), f"output exists: {path}")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    temporary.replace(path)
    path.chmod(0o444)
    return strict.file_sha256(path)


def initialize_campaign(
    root: Path,
    repo: Path,
    heldout_root: Path,
    all8_root: Path,
    python_executable: Path,
) -> dict[str, Any]:
    strict.require(not root.exists(), f"output root exists; use --resume: {root}")
    strict.require(
        root != repo and repo not in root.parents, "output root is in checkout"
    )
    strict.require(
        root != heldout_root and heldout_root not in root.parents,
        "output root is in held-out evidence",
    )
    strict.require(
        root != all8_root and all8_root not in root.parents,
        "output root is in all8 evidence",
    )
    strict.require(
        heldout_root != all8_root
        and heldout_root not in all8_root.parents
        and all8_root not in heldout_root.parents,
        "held-out and all8 evidence roots overlap",
    )
    strict.check_equal(
        adjudication.EXPECTED_COMMIT,
        strict.EXPECTED_MEASURED_COMMIT,
        "adjudication and strict validator commit",
    )
    campaign_path, rows, all8_reference, all8_reference_sha256 = (
        heldout.validate_individual_results(heldout_root, all8_root)
    )
    all8_rows: list[dict[str, object]] = list(all8_reference.values())
    with campaign_path.open(newline="") as handle:
        recorded_python = {row["python_executable"] for row in csv.DictReader(handle)}
    strict.require(
        len(recorded_python) == 1, "held-out campaign changed Python executable"
    )
    strict.check_equal(
        Path(next(iter(recorded_python))).resolve(),
        python_executable,
        "held-out versus adjudication Python executable",
    )
    staging = root.with_name(f".{root.name}.initializing")
    if staging.exists() or staging.is_symlink():
        clear_stale_staging(staging)
    staging.mkdir(parents=True)
    try:
        launcher = staging / "launcher"
        source_snapshots = []
        for name in adjudication.SNAPSHOT_FILENAMES:
            digest = snapshot(SETUP_ROOT / name, launcher / name)
            source_snapshots.append({"name": name, "sha256": digest})
        measured_source_snapshots = []
        for relative, snapshot_name in (
            (
                adjudication.BENCHMARK_RELATIVE,
                "measured_compare_attention_backends.py",
            ),
            (adjudication.ATTENTION_RELATIVE, "measured_attention.py"),
        ):
            digest = snapshot(repo / relative, launcher / snapshot_name)
            measured_source_snapshots.append(
                {
                    "source_path": relative.as_posix(),
                    "snapshot_name": snapshot_name,
                    "sha256": digest,
                }
            )
        campaign = {
            "schema_version": adjudication.SCHEMA_VERSION,
            "expected_commit": adjudication.EXPECTED_COMMIT,
            "expected_cute_version": adjudication.EXPECTED_CUTE_VERSION,
            "campaign_nonce": secrets.token_hex(16),
            "repo_root": str(repo),
            "heldout_root": str(heldout_root),
            "all8_root": str(all8_root),
            "python_executable": str(python_executable),
            "source_snapshots": source_snapshots,
            "measured_source_snapshots": measured_source_snapshots,
            "heldout_evidence": adjudication.evidence_files(heldout_root, rows),
            "all8_evidence": adjudication.all8_evidence_files(all8_root, all8_rows),
            "validated_heldout_rows": rows,
            "validated_heldout_rows_sha256": adjudication.canonical_sha256(rows),
            "validated_all8_rows": all8_rows,
            "validated_all8_rows_sha256": adjudication.canonical_sha256(all8_rows),
            "all8_reference_manifest_sha256": all8_reference_sha256,
            "cases": adjudication.initial_case_records(rows),
            "report_path": "adjudication_report.json",
        }
        declaration_digest = write_immutable_json(
            adjudication.campaign_path(staging), campaign
        )
        digest_path = adjudication.campaign_digest_path(staging)
        digest_path.write_text(f"{declaration_digest}\n")
        digest_path.chmod(0o444)
        validated = adjudication.validate_campaign(
            staging, deep_artifact_validation=True
        )
        adjudication.require_repo_identity(repo, validated)
        staging.replace(root)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return adjudication.validate_campaign(root, deep_artifact_validation=True)


def resume_campaign(
    root: Path,
    repo: Path,
    heldout_root: Path,
    all8_root: Path,
    python_executable: Path,
) -> dict[str, Any]:
    strict.require(
        root.is_dir() and not root.is_symlink(), f"invalid resume root: {root}"
    )
    campaign = adjudication.validate_campaign(root, deep_artifact_validation=True)
    strict.check_equal(campaign["repo_root"], str(repo), "resume repository")
    strict.check_equal(
        campaign["heldout_root"], str(heldout_root), "resume held-out root"
    )
    strict.check_equal(campaign["all8_root"], str(all8_root), "resume all8 root")
    strict.check_equal(
        campaign["python_executable"], str(python_executable), "resume Python"
    )
    snapshots = adjudication.source_snapshot_hashes(campaign)
    for name in adjudication.SNAPSHOT_FILENAMES:
        strict.check_equal(
            strict.file_sha256(SETUP_ROOT / name),
            snapshots[name],
            f"resume source {name}",
        )
    adjudication.require_repo_identity(repo, campaign)
    return campaign


@contextmanager
def campaign_lock(root: Path) -> Iterator[None]:
    lock_path = root.with_name(f"{root.name}.campaign.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another adjudication process holds {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def case_lock(root: Path, case_id: str) -> Iterator[Any]:
    lock_path = root / ".case_locks" / f"{case_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"a prior {case_id} worker still holds {lock_path}"
            ) from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_event(root: Path, event: dict[str, object]) -> None:
    record = {"timestamp_ns": time.time_ns(), **event}
    path = root / "events.jsonl"
    data = f"{adjudication.canonical_json(record)}\n".encode()
    with EVENT_LOCK, path.open("a+b", buffering=0) as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        contents = handle.read()
        if contents and not contents.endswith(b"\n"):
            tail_start = contents.rfind(b"\n") + 1
            tail = contents[tail_start:]
            try:
                tail_record = json.loads(tail)
            except (UnicodeDecodeError, json.JSONDecodeError):
                handle.truncate(tail_start)
            else:
                strict.require(
                    isinstance(tail_record, dict), f"invalid event record in {path}"
                )
                handle.write(b"\n")
        handle.write(data)
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def prior_attempt_count(root: Path, case_id: str) -> int:
    path = root / "events.jsonl"
    count = 0
    if path.is_file():
        with EVENT_LOCK, path.open("rb") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            contents = handle.read()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        lines = contents.splitlines(keepends=True)
        for index, raw_line in enumerate(lines):
            terminated = raw_line.endswith(b"\n")
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if index == len(lines) - 1 and not terminated:
                    break
                raise RuntimeError(f"invalid event record in {path}") from exc
            strict.require(isinstance(record, dict), f"invalid event record in {path}")
            if (
                record.get("event") == "attempt_started"
                and record.get("case_id") == case_id
            ):
                count += 1
    prefix = f"{case_id}.attempt-"
    log_attempts = []
    for log_path in (root / "logs").glob(f"{prefix}*.log"):
        suffix = log_path.name.removeprefix(prefix).removesuffix(".log")
        strict.require(suffix.isdigit(), f"invalid attempt log name: {log_path}")
        log_attempts.append(int(suffix))
    return max([count, *log_attempts])


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


def validate_gpu_idle(gpu: int) -> None:
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
        f"GPU {gpu} is busy: {active}",
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


@contextmanager
def stop_on_termination(stop_event: threading.Event) -> Iterator[None]:
    previous = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    for signum in previous:
        signal.signal(signum, request_stop)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def worker_environment(
    root: Path, repo: Path, case: adjudication.CaseDefinition
) -> dict[str, str]:
    env = adjudication.sanitized_environment(dict(os.environ))
    cache = root / "cache" / case.case_id
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
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.pathsep.join((str(root / "launcher"), str(repo))),
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


def worker_command(
    root: Path,
    repo: Path,
    python_executable: Path,
    case: adjudication.CaseDefinition,
    staging: Path,
    campaign_sha256: str,
    case_lock_path: Path,
) -> list[str]:
    worker = root / "launcher" / "remeasure_heldout_winners.py"
    worker_digest = adjudication.source_snapshot_hashes(
        adjudication.validate_campaign(root, deep_artifact_validation=False)
    )[worker.name]
    return [
        str(python_executable),
        str(worker),
        "--campaign-root",
        str(root),
        "--repo",
        str(repo),
        "--case-id",
        case.case_id,
        "--physical-gpu",
        str(case.physical_gpu),
        "--output-dir",
        str(staging),
        "--expected-campaign-sha256",
        campaign_sha256,
        "--expected-worker-sha256",
        worker_digest,
        "--case-lock-path",
        str(case_lock_path),
    ]


def final_result_path(
    root: Path, campaign: dict[str, Any], case: adjudication.CaseDefinition
) -> Path:
    declaration = adjudication.case_record(campaign, case.case_id)
    return root / adjudication.relative_path(
        declaration["result_path"], f"{case.case_id} result path"
    )


def clear_stale_staging(staging: Path) -> None:
    if not staging.exists() and not staging.is_symlink():
        return
    strict.require(
        staging.is_dir() and not staging.is_symlink(),
        f"invalid stale staging path: {staging}",
    )
    shutil.rmtree(staging)


def recover_staged_result(
    root: Path,
    campaign: dict[str, Any],
    case: adjudication.CaseDefinition,
    staging: Path,
    final_directory: Path,
) -> bool:
    if not staging.exists() and not staging.is_symlink():
        return False
    strict.require(
        staging.is_dir() and not staging.is_symlink(),
        f"invalid stale staging path: {staging}",
    )
    staged_result = staging / adjudication.RESULT_FILENAME
    if staged_result.is_file() and not staged_result.is_symlink():
        try:
            adjudication.validate_case_output(
                root, campaign, case.case_id, output_path=staged_result
            )
        except RuntimeError:
            pass
        else:
            final_directory.parent.mkdir(parents=True, exist_ok=True)
            strict.require(
                not final_directory.exists(), f"result appeared: {final_directory}"
            )
            staging.replace(final_directory)
            adjudication.validate_case_output(root, campaign, case.case_id)
            return True
    clear_stale_staging(staging)
    return False


def run_case(
    root: Path,
    repo: Path,
    python_executable: Path,
    campaign: dict[str, Any],
    case: adjudication.CaseDefinition,
    stop_event: threading.Event,
) -> None:
    try:
        result_path = final_result_path(root, campaign, case)
        final_directory = result_path.parent
        if final_directory.exists():
            adjudication.validate_case_output(root, campaign, case.case_id)
            return
        lock_path = root / ".case_locks" / f"{case.case_id}.lock"
        with case_lock(root, case.case_id) as lock_handle:
            staging = root / ".staging" / case.case_id
            if recover_staged_result(root, campaign, case, staging, final_directory):
                append_event(
                    root,
                    {"event": "staged_result_recovered", "case_id": case.case_id},
                )
                return
            strict.require(not stop_event.is_set(), "peer adjudication lane failed")
            adjudication.require_repo_identity(repo, campaign)
            validate_gpu_idle(case.physical_gpu)
            attempt = prior_attempt_count(root, case.case_id) + 1
            log_path = root / "logs" / f"{case.case_id}.attempt-{attempt}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = worker_command(
                root,
                repo,
                python_executable,
                case,
                staging,
                campaign["_declaration_sha256"],
                lock_path,
            )
            append_event(
                root,
                {
                    "event": "attempt_started",
                    "case_id": case.case_id,
                    "attempt": attempt,
                    "command": command,
                    "command_sha256": adjudication.canonical_sha256(command),
                },
            )
            with log_path.open("x") as log:
                env = worker_environment(root, repo, case)
                env["ADJUDICATION_CASE_LOCK_FD"] = str(lock_handle.fileno())
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(lock_handle.fileno(),),
                )
                returncode = wait_for_process(process, stop_event)
            append_event(
                root,
                {
                    "event": "attempt_finished",
                    "case_id": case.case_id,
                    "attempt": attempt,
                    "returncode": returncode,
                },
            )
        strict.require(returncode == 0, f"{case.case_id}: worker failed")
        staged_result = staging / adjudication.RESULT_FILENAME
        adjudication.validate_case_output(
            root, campaign, case.case_id, output_path=staged_result
        )
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        strict.require(
            not final_directory.exists(), f"result appeared: {final_directory}"
        )
        staging.replace(final_directory)
        adjudication.validate_case_output(root, campaign, case.case_id)
        validate_gpu_idle(case.physical_gpu)
        adjudication.require_repo_identity(repo, campaign)
    except BaseException:
        stop_event.set()
        raise


def write_report(root: Path, campaign: dict[str, Any]) -> dict[str, Any]:
    report = adjudication.build_report(root, campaign)
    path = root / str(campaign["report_path"])
    if path.exists():
        strict.check_equal(
            adjudication.load_json_object(path), report, "existing report"
        )
    else:
        write_immutable_json(path, report)
    strict.check_equal(
        adjudication.validate_complete_campaign(root), report, "final report"
    )
    return report


def _run(args: argparse.Namespace, root: Path) -> None:
    repo = args.repo.expanduser().resolve()
    heldout_root = args.heldout_root.expanduser().resolve()
    all8_root = args.all8_artifact_root.expanduser().resolve()
    python_executable = canonical_executable(args.python)
    require_python_environment(python_executable)
    if args.resume:
        campaign = resume_campaign(
            root, repo, heldout_root, all8_root, python_executable
        )
    else:
        adjudication.require_repo_identity(
            repo,
            {
                "repo_root": str(repo),
                "measured_source_snapshots": [
                    {
                        "source_path": relative.as_posix(),
                        "sha256": strict.file_sha256(repo / relative),
                    }
                    for relative in (
                        adjudication.BENCHMARK_RELATIVE,
                        adjudication.ATTENTION_RELATIVE,
                    )
                ],
            },
        )
        campaign = initialize_campaign(
            root, repo, heldout_root, all8_root, python_executable
        )
    report_path = root / str(campaign["report_path"])
    if report_path.is_file():
        report = adjudication.validate_complete_campaign(root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    for case in adjudication.CASES:
        final = final_result_path(root, campaign, case).parent
        if not final.exists():
            validate_gpu_idle(case.physical_gpu)
    stop_event = threading.Event()
    with (
        stop_on_termination(stop_event),
        ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="heldout-adjudication"
        ) as pool,
    ):
        futures = [
            pool.submit(
                run_case,
                root,
                repo,
                python_executable,
                campaign,
                case,
                stop_event,
            )
            for case in adjudication.CASES
        ]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for future in done:
            future.result()
        for future in pending:
            future.result()
    report = write_report(root, campaign)
    append_event(
        root,
        {
            "event": "campaign_validated",
            "report_sha256": strict.file_sha256(report_path),
            "performance_gate_status": report["performance_gate_status"],
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-measure the five held-out winners and cuDNN SDPA."
    )
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--all8-artifact-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.expanduser().resolve()
    with campaign_lock(root):
        _run(args, root)


if __name__ == "__main__":
    main()
