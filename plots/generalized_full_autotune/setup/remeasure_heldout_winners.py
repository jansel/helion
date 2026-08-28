from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

if TYPE_CHECKING:
    from types import ModuleType

EXPECTED_SNAPSHOT_FILENAMES = (
    "build_heldout_manifest.py",
    "build_strict_manifest.py",
    "heldout_adjudication.py",
    "remeasure_generalization_winners.py",
    "remeasure_heldout_winners.py",
    "run_heldout_adjudication.py",
    "validate_generalization_campaign.py",
    "validate_heldout_adjudication.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_local_modules(
    campaign_root: Path, expected_campaign_sha256: str, expected_worker_sha256: str
) -> tuple[ModuleType, ModuleType, ModuleType, dict[str, Any]]:
    campaign_path = campaign_root / "campaign.json"
    actual_campaign_sha256 = file_sha256(campaign_path)
    if actual_campaign_sha256 != expected_campaign_sha256:
        raise RuntimeError(
            "campaign declaration digest: expected "
            f"{expected_campaign_sha256!r}, got {actual_campaign_sha256!r}"
        )
    campaign = json.loads(campaign_path.read_text())
    if not isinstance(campaign, dict):
        raise RuntimeError("campaign declaration is not an object")
    snapshots = campaign.get("source_snapshots")
    if not isinstance(snapshots, list):
        raise RuntimeError("campaign has no source snapshot set")
    if [record.get("name") for record in snapshots if isinstance(record, dict)] != list(
        EXPECTED_SNAPSHOT_FILENAMES
    ):
        raise RuntimeError("campaign source snapshot set is not canonical")
    expected = {}
    launcher = (campaign_root / "launcher").resolve()
    for record in snapshots:
        if not isinstance(record, dict):
            raise RuntimeError("invalid source snapshot record")
        name = record.get("name")
        digest = record.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise RuntimeError("invalid source snapshot identity")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"invalid source snapshot digest: {digest!r}")
        path = launcher / name
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"invalid immutable source snapshot: {path}")
        actual = file_sha256(path)
        if actual != digest:
            raise RuntimeError(
                f"source snapshot {name}: expected {digest!r}, got {actual!r}"
            )
        expected[name] = digest
    worker = Path(__file__).resolve()
    if worker.parent != launcher:
        raise RuntimeError(f"worker is not running from snapshot directory: {worker}")
    if file_sha256(worker) != expected_worker_sha256:
        raise RuntimeError("worker digest does not match command declaration")
    if expected.get(worker.name) != expected_worker_sha256:
        raise RuntimeError("worker digest does not match campaign declaration")
    sys.path.insert(0, str(launcher))
    import build_heldout_manifest as heldout
    import build_strict_manifest as strict
    import heldout_adjudication as adjudication
    import remeasure_generalization_winners as measurement

    adjudication.validate_runtime_module_paths(
        campaign_root,
        [strict, heldout, adjudication, measurement, measurement.validator],
    )
    return strict, adjudication, measurement, campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-measure five held-out winners against cuDNN SDPA."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-campaign-sha256", required=True)
    parser.add_argument("--expected-worker-sha256", required=True)
    parser.add_argument("--case-lock-path", type=Path, required=True)
    return parser.parse_args()


def atomic_write_text(path: Path, contents: str, *, read_only: bool) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(contents)
    temporary.replace(path)
    if read_only:
        path.chmod(0o444)


def build_implementations(
    inputs: tuple[object, object, object],
    declaration: dict[str, Any],
    repo: Path,
    heldout_root: Path,
    strict: ModuleType,
    measurement: ModuleType,
) -> tuple[dict[str, Callable[[], object]], list[dict[str, object]], list[str]]:
    import examples.attention as attention_example

    import helion

    strict.check_equal(
        Path(helion.__file__).resolve().parent.parent,
        repo,
        "runtime Helion checkout",
    )
    strict.check_equal(
        Path(attention_example.__file__).resolve(),
        repo / "examples/attention.py",
        "runtime attention example",
    )
    causal = bool(declaration["shape"]["causal"])
    kernel = (
        attention_example.causal_attention_output
        if causal
        else attention_example.attention_output
    )
    bound = kernel.bind(inputs)
    live_seed_policy = measurement.live_compiler_seed_policy(bound)
    implementations: dict[str, Callable[[], object]] = {}
    sources = []
    source_texts = []
    for contender in declaration["contenders"]:
        result_path = heldout_root / contender["origin_result_path"]
        strict.check_equal(
            strict.file_sha256(result_path),
            contender["origin_result_sha256"],
            f"{contender['name']}: origin result",
        )
        payload = strict.load_json_object(result_path)
        provenance = payload["helion_overrides"]["autotune_provenance"]
        strict.check_equal(
            provenance.get("compiler_seed_policy"),
            live_seed_policy,
            f"{contender['name']}: compiler seed policy",
        )
        strict.check_equal(
            provenance.get("selected_config"),
            contender["selected_config"],
            f"{contender['name']}: selected config",
        )
        strict.check_equal(
            strict.canonical_sha256(provenance["selected_config"]),
            contender["selected_config_sha256"],
            f"{contender['name']}: selected config digest",
        )
        strict.check_equal(
            provenance.get("selected_source_sha256"),
            contender["origin_selected_source_sha256"],
            f"{contender['name']}: recorded selected source",
        )
        config = helion.Config.from_dict(contender["selected_config"])
        source = bound.to_triton_code(config, emit_repro_caller=False)
        source_digest = hashlib.sha256(source.encode()).hexdigest()
        strict.check_equal(
            source_digest,
            contender["expected_regenerated_source_sha256"],
            f"{contender['name']}: regenerated source",
        )
        compiled = bound.compile_config(config)
        compiled_digest = bound.env.backend.generated_source_hash(compiled)
        strict.check_equal(
            compiled_digest, source_digest, f"{contender['name']}: compiled source"
        )
        name = contender["name"]
        implementations[name] = lambda compiled=compiled: compiled(*inputs)
        sources.append(
            {
                "name": name,
                "origin_kind": contender["origin_kind"],
                "origin_result_sha256": contender["origin_result_sha256"],
                "origin_selected_source_sha256": contender[
                    "origin_selected_source_sha256"
                ],
                "selected_config_sha256": contender["selected_config_sha256"],
                "regenerated_source_sha256": source_digest,
                "compiled_source_sha256": compiled_digest,
                "archive_path": f"generated_sources/{name}.py.txt",
            }
        )
        source_texts.append(source)
    return implementations, sources, source_texts


def run_measurement(
    args: argparse.Namespace,
    strict: ModuleType,
    adjudication: ModuleType,
    measurement: ModuleType,
    campaign: dict[str, Any],
) -> dict[str, object]:
    import torch

    root = args.campaign_root.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    validated = adjudication.validate_campaign(root, deep_artifact_validation=False)
    strict.check_equal(
        validated["_declaration_sha256"],
        args.expected_campaign_sha256,
        "worker campaign digest",
    )
    strict.check_equal(
        Path(sys.executable).resolve(),
        Path(validated["python_executable"]).resolve(),
        "worker Python executable",
    )
    lock_fd_value = os.environ.get("ADJUDICATION_CASE_LOCK_FD")
    strict.require(
        isinstance(lock_fd_value, str) and lock_fd_value.isdigit(),
        "worker did not inherit the case lock descriptor",
    )
    lock_fd = int(lock_fd_value)
    os.fstat(lock_fd)
    lock_path = args.case_lock_path.expanduser().resolve()
    strict.check_equal(
        Path(f"/proc/self/fd/{lock_fd}").resolve(),
        lock_path,
        "worker case lock descriptor",
    )
    adjudication.require_repo_identity(repo, validated)
    case = adjudication.case_by_id(args.case_id)
    declaration = adjudication.case_record(validated, case.case_id)
    strict.check_equal(args.physical_gpu, case.physical_gpu, "requested physical GPU")
    strict.check_equal(
        os.environ.get("CUDA_VISIBLE_DEVICES"), str(case.physical_gpu), "GPU visibility"
    )
    expected_output = root / ".staging" / case.case_id
    strict.check_equal(
        lock_path,
        root / ".case_locks" / f"{case.case_id}.lock",
        "worker case lock path",
    )
    output_dir = args.output_dir.expanduser().resolve()
    strict.check_equal(output_dir, expected_output, "worker output directory")
    strict.require(not output_dir.exists(), f"worker output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    gpu_start = measurement.gpu_record(case.physical_gpu)
    strict.check_equal(torch.cuda.device_count(), 1, "visible CUDA devices")
    torch.cuda.set_device(0)
    protocol = declaration["protocol"]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(protocol["input_seed"]))
    shape = (2, 32, case.seq_len, 64)
    query, key, value = (
        torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
        for _ in range(3)
    )
    inputs = (query, key, value)
    with measurement.scrubbed_argv():
        implementations, sources, source_texts = build_implementations(
            inputs,
            declaration,
            repo,
            Path(validated["heldout_root"]),
            strict,
            measurement,
        )

        def run_sdpa() -> torch.Tensor:
            return torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=case.causal
            )

        implementations["sdpa"] = run_sdpa
        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
        ):
            expected = run_sdpa()
            torch.cuda.synchronize()
            correctness: dict[str, object] = {}
            for name, implementation in implementations.items():
                actual = implementation()
                torch.cuda.synchronize()
                numerics = measurement.tensor_error_summary(actual, expected)
                repeats = []
                for repeat_index in range(1, int(protocol["repeatability_launches"])):
                    repeated = implementation()
                    torch.cuda.synchronize()
                    repeat = measurement.exact_repeat_summary(repeated, actual)
                    repeat["repeat_index"] = repeat_index
                    repeats.append(repeat)
                    del repeated
                strict.require(
                    numerics["passed"] and all(repeat["passed"] for repeat in repeats),
                    f"{case.case_id}/{name}: correctness or repeatability failed",
                )
                correctness[name] = {
                    "numerics": numerics,
                    "repeatability": repeats,
                }
                del actual
            del expected
            torch.cuda.empty_cache()
            measurement.thermal_warmup(float(protocol["thermal_warmup_seconds"]))
            for _ in range(int(protocol["warmup_calls_per_implementation"])):
                for implementation in implementations.values():
                    output = implementation()
                    del output
            torch.cuda.synchronize()
            timing_repetitions, calibration = measurement.calibrate_timing_repetitions(
                implementations["sdpa"],
                target_ms=float(protocol["target_timing_ms"]),
                maximum=int(protocol["max_timing_repetitions"]),
            )
            names = list(implementations)
            orders = adjudication.balanced_orders(
                names, int(protocol["rounds"]), int(protocol["order_seed"])
            )
            raw_rounds = []
            for round_index, order in enumerate(orders):
                raw_rounds.append(
                    {
                        "round_index": round_index,
                        "order": order,
                        "times": {
                            name: measurement.timed_call(
                                implementations[name], timing_repetitions
                            )
                            for name in order
                        },
                    }
                )
    contender_names = [contender["name"] for contender in declaration["contenders"]]
    direct_search_tflops = [
        float(contender["direct_search_tflops"])
        for contender in declaration["contenders"]
    ]
    summary = adjudication.summarize_measurements(
        raw_rounds,
        contender_names,
        case,
        bootstrap_samples=int(protocol["bootstrap_samples"]),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
        direct_tflops=direct_search_tflops,
    )
    source_directory = output_dir / "generated_sources"
    source_directory.mkdir()
    for source_record, source_text in zip(sources, source_texts, strict=True):
        archive = output_dir / source_record["archive_path"]
        atomic_write_text(archive, source_text, read_only=True)
        strict.check_equal(
            strict.file_sha256(archive),
            source_record["regenerated_source_sha256"],
            f"{source_record['name']}: archived regenerated source",
        )
    gpu_end = measurement.gpu_record(case.physical_gpu)
    cudnn_version = torch.backends.cudnn.version()
    strict.require(
        isinstance(cudnn_version, int) and cudnn_version > 0,
        f"invalid cuDNN version: {cudnn_version!r}",
    )
    heldout_versions = {row["version"] for row in validated["validated_heldout_rows"]}
    strict.require(len(heldout_versions) == 1, "held-out versions differ")
    payload = {
        "schema_version": adjudication.RESULT_SCHEMA_VERSION,
        "status": "MEASUREMENT_COMPLETE",
        "case_id": case.case_id,
        "shape": declaration["shape"],
        "physical_gpu": case.physical_gpu,
        "measured_commit": adjudication.EXPECTED_COMMIT,
        "campaign_sha256": validated["_declaration_sha256"],
        "worker_sha256": args.expected_worker_sha256,
        "source_snapshot_sha256": adjudication.source_snapshot_hashes(validated),
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
        "environment": {
            "torch_version": torch.__version__,
            "cudnn_version": cudnn_version,
            "cute_version": version("nvidia-cutlass-dsl"),
            "helion_version": next(iter(heldout_versions)),
        },
        "protocol": {
            **protocol,
            "timing_repetitions": timing_repetitions,
            "sdpa_calibration_event_ms": calibration,
        },
        "selected_sources": sources,
        "correctness": correctness,
        "direct_search_tflops": direct_search_tflops,
        "raw_rounds": raw_rounds,
        "summary": summary,
    }
    result_path = output_dir / adjudication.RESULT_FILENAME
    atomic_write_text(
        result_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        read_only=True,
    )
    adjudication.validate_case_output(
        root,
        validated,
        case.case_id,
        output_path=result_path,
        require_completion=False,
    )
    adjudication.require_repo_identity(repo, validated)
    completion = {
        "schema_version": 1,
        "status": "POSTCONDITIONS_PASSED",
        "case_id": case.case_id,
        "campaign_sha256": validated["_declaration_sha256"],
        "worker_sha256": args.expected_worker_sha256,
        "result_sha256": strict.file_sha256(result_path),
    }
    atomic_write_text(
        output_dir / adjudication.COMPLETION_FILENAME,
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        read_only=True,
    )
    return payload


def main() -> None:
    args = parse_args()
    root = args.campaign_root.expanduser().resolve()
    strict, adjudication, measurement, raw_campaign = bootstrap_local_modules(
        root, args.expected_campaign_sha256, args.expected_worker_sha256
    )
    strict.check_equal(
        raw_campaign.get("expected_commit"),
        adjudication.EXPECTED_COMMIT,
        "bootstrap campaign commit",
    )
    run_measurement(args, strict, adjudication, measurement, raw_campaign)


if __name__ == "__main__":
    main()
