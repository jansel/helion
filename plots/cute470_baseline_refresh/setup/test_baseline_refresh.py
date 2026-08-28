from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any

import pytest

MODULE_PATH = Path(__file__).with_name("campaign.py")
SPEC = importlib.util.spec_from_file_location("cute470_baseline_campaign", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _snapshot_tracked_inputs(root: Path) -> None:
    historical_source = REPO_ROOT / "plots/kernelagent/results/payloads"
    runs_source = REPO_ROOT / "plots/kernelagent_closed/runs"
    results_source = REPO_ROOT / "plots/kernelagent_closed/results"
    historical_destination = root / "inputs/historical_payloads"
    ka_destination = root / "inputs/kernelagent_closed"
    historical_destination.mkdir(parents=True)
    ka_destination.mkdir(parents=True)
    for case in campaign.CASES:
        name = f"{case.case_id}.json"
        shutil.copy2(historical_source / name, historical_destination / name)
    for run in campaign.KA_RUNS:
        destination = ka_destination / run.run_id
        destination.mkdir()
        shutil.copy2(
            runs_source / run.run_id / "manifest.json",
            destination / "manifest.json",
        )
        shutil.copy2(
            runs_source / run.run_id / "selected_kernel.py.txt",
            destination / "selected_kernel.py.txt",
        )
        shutil.copy2(
            results_source / f"{run.case_id}_kernelagent-closed-{run.budget}.json",
            destination / "previous_result.json",
        )


def _versions() -> dict[str, object]:
    baselines = {
        "fa4": {
            "version": "FlashAttention fa4-v4.0.0.beta23; CuTe 4.7.0",
            "version_label": "fa4-v4.0.0.beta23; CuTe 4.7.0",
        },
        "flexattention-cute": {
            "version": (
                "PyTorch 2.13.0.dev20260506+cu130; FA4 fa4-v4.0.0.beta23; CuTe 4.7.0"
            ),
            "version_label": (
                "PyTorch 2.13.0.dev20260506; FA4 fa4-v4.0.0.beta23; CuTe 4.7.0"
            ),
        },
    }
    ka_version = {
        "version": (
            "KernelAgent v3-20260730; model gpt-5.6-sol; CuTe 4.7.0; "
            "selected with CuTe 4.5.1"
        ),
        "version_label": "KernelAgent v3-20260730 / GPT-5.6 / CuTe 4.7.0",
    }
    return {
        "baselines": baselines,
        "kernelagent": {run.run_id: dict(ka_version) for run in campaign.KA_RUNS},
    }


def _write_manifest(root: Path) -> dict[str, Any]:
    launcher = root / "launcher"
    launcher.mkdir(parents=True)
    (launcher / "campaign.py").write_text("campaign fixture\n")
    (launcher / "run_campaign.sh").write_text("launcher fixture\n")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "campaign": "cute470_baseline_refresh",
        "created_utc": "2026-08-20T00:00:00+00:00",
        "expected_measured_commit": campaign.EXPECTED_MEASURED_COMMIT,
        "definition": campaign._definition(),
        "sources": {
            "setup_files": {
                "campaign.py": campaign._sha256_file(launcher / "campaign.py"),
                "run_campaign.sh": campaign._sha256_file(launcher / "run_campaign.sh"),
            }
        },
        "inputs": campaign._input_record(root),
        "toolchain": {"fixture": True},
        "hardware": [],
        "import_roots": {},
        "versions": _versions(),
    }
    campaign._atomic_write_json(root / "campaign_manifest.json", manifest)
    return manifest


def _fixture_root(root: Path) -> dict[str, Any]:
    _snapshot_tracked_inputs(root)
    return _write_manifest(root)


def _payload(
    root: Path,
    case: campaign.Case,
    impl: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    runs = [10.0 + index / 10 for index in range(campaign.SAMPLE_COUNT)]
    median_ms = statistics.median(runs)
    best_ms = min(runs)
    result: dict[str, Any] = {
        "impl": impl,
        **campaign._expected_version(manifest, case, impl),
        "shape": case.shape,
        "gpu": campaign.EXPECTED_GPU_NAME,
        "physical_gpu": str(case.physical_gpu),
        "power_cap_w": campaign.EXPECTED_POWER_CAP_W,
        "input_seed": campaign.INPUT_SEED,
        "flop_model": "softmax_attention_forward",
        "accuracy": "PASS",
        "benchmark_timer": "event",
        "best_ms": best_ms,
        "median_ms": median_ms,
        "mom_median_ms": median_ms,
        "mean_ms": sum(runs) / len(runs),
        "std_ms": statistics.stdev(runs),
        "runs_ms": runs,
        "best_tflops": campaign._flop_count(case) / best_ms / 1e9,
        "median_tflops": campaign._flop_count(case) / median_ms / 1e9,
        "mom_median_tflops": campaign._flop_count(case) / median_ms / 1e9,
    }
    if impl == "flexattention-cute":
        result["notes"] = ["Forced PyTorch FlexAttention BACKEND='FLASH'."]
    if impl.startswith("kernelagent-closed-"):
        run = campaign.KA_RUNS_BY_KEY[(case.case_id, impl)]
        retained = json.loads(
            (
                root / "inputs/kernelagent_closed" / run.run_id / "manifest.json"
            ).read_text()
        )
        result["config"] = {
            "budget_label": run.budget,
            "budget_seconds": retained["budget_seconds"],
            "elapsed_seconds": retained["elapsed_seconds"],
            "selection": retained["selection"],
            "source_sha256": run.source_sha256,
            "standard_correctness_executed": True,
            "repeat_determinism_executed": True,
            "stress_correctness_executed": True,
            "selection_cute_version": campaign.EXPECTED_KA_SELECTION_CUTE_VERSION,
            "evaluation_cute_version": campaign.EXPECTED_CUTE_VERSION,
        }
        result["notes"] = [
            (
                "Source selected with CuTe 4.5.1; recompiled with CuTe 4.7.0. "
                "Performance was measured. Standard and stress full-output checks "
                "passed with exact repeatability."
            )
        ]
    return result


def _write_accepted_result(
    root: Path,
    case: campaign.Case,
    impl: str,
    manifest: dict[str, Any],
) -> None:
    directory = campaign._result_dir(root, case, impl)
    directory.mkdir(parents=True)
    result_path = directory / "result.json"
    invocation_path = directory / "invocation.json"
    campaign._atomic_write_json(result_path, _payload(root, case, impl, manifest))
    _command, _environment, invocation = campaign._invocation(
        root, case, impl, directory
    )
    campaign._atomic_write_json(invocation_path, invocation)
    campaign._atomic_write_json(
        directory / "receipt.json",
        campaign._receipt(root, case, impl, result_path, invocation_path),
    )


def test_case_matrix_and_kernelagent_pass_set_are_exact() -> None:
    assert [(case.case_id, case.physical_gpu) for case in campaign.CASES] == [
        ("dense_32768", 7),
        ("dense_65536", 7),
        ("dense_131072", 7),
        ("dense_262144", 7),
        ("causal_65536", 6),
        ("causal_131072", 6),
        ("causal_262144", 6),
        ("causal_524288", 6),
    ]
    assert {(run.case_id, run.budget) for run in campaign.KA_RUNS} == {
        ("dense_131072", "1x"),
        ("dense_131072", "2x"),
        ("dense_262144", "1x"),
        ("causal_65536", "1x"),
        ("causal_65536", "2x"),
        ("causal_131072", "1x"),
        ("causal_262144", "1x"),
        ("causal_262144", "2x"),
        ("causal_524288", "1x"),
    }
    assert (
        sum(len(campaign._expected_implementations(case)) for case in campaign.CASES)
        == 25
    )


def test_tracked_input_snapshot_validation_fails_closed(tmp_path: Path) -> None:
    _snapshot_tracked_inputs(tmp_path)
    record = campaign._input_record(tmp_path)
    assert len(record["historical_payloads"]) == 8
    assert len(record["kernelagent_closed"]) == 9

    run = campaign.KA_RUNS[0]
    source = (
        tmp_path / "inputs/kernelagent_closed" / run.run_id / "selected_kernel.py.txt"
    )
    source.write_text(source.read_text() + "\n# tampered\n")
    with pytest.raises(campaign.CampaignError, match="source hash mismatch"):
        campaign._input_record(tmp_path)


def test_snapshot_uses_only_canonical_tracked_input_directories(tmp_path: Path) -> None:
    historical = REPO_ROOT / "plots/kernelagent/results/payloads"
    runs = REPO_ROOT / "plots/kernelagent_closed/runs"
    results = REPO_ROOT / "plots/kernelagent_closed/results"
    output = tmp_path / "valid"
    campaign._snapshot_inputs(output, REPO_ROOT, historical, runs, results)
    assert len(campaign._input_record(output)["kernelagent_closed"]) == 9

    with pytest.raises(campaign.CampaignError, match="historical payload directory"):
        campaign._snapshot_inputs(
            tmp_path / "invalid",
            REPO_ROOT,
            REPO_ROOT / "plots/kernelagent/results",
            runs,
            results,
        )


def test_invocation_uses_exact_gpu_correctness_and_isolated_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/untrusted/lib")
    monkeypatch.setenv("LIBRARY_PATH", "/untrusted/link")
    case = campaign.CASES[2]
    impl = "kernelagent-closed-1x"
    directory = campaign._result_dir(tmp_path, case, impl)
    command, environment, record = campaign._invocation(tmp_path, case, impl, directory)
    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "LD_LIBRARY_PATH" not in environment
    assert "LIBRARY_PATH" not in environment
    assert environment["PYTHONPATH"].split(campaign.os.pathsep) == [
        str(tmp_path / "checkouts/helion"),
        str(tmp_path / "checkouts/quack"),
    ]
    assert "--skip-correctness" in command
    assert command[command.index("--skip-correctness") + 1] == "0"
    assert "--num-runs" in command
    assert command[command.index("--num-runs") + 1] == "9"
    assert "--kernelagent-closed-results-root" in command
    assert record["physical_gpu"] == 7
    campaign._reject_absolute_strings(record)


def test_result_validation_checks_stats_versions_and_kernelagent_manifest(
    tmp_path: Path,
) -> None:
    manifest = _fixture_root(tmp_path)
    case = campaign.CASES[2]
    impl = "kernelagent-closed-1x"
    valid = _payload(tmp_path, case, impl, manifest)
    campaign.validate_result(valid, case, impl, manifest, tmp_path)

    wrong_stats = copy.deepcopy(valid)
    wrong_stats["median_tflops"] *= 1.01
    with pytest.raises(campaign.CampaignError, match="median_tflops"):
        campaign.validate_result(wrong_stats, case, impl, manifest, tmp_path)

    wrong_version = copy.deepcopy(valid)
    wrong_version["version_label"] = "CuTe 4.5.1"
    with pytest.raises(campaign.CampaignError, match="version label"):
        campaign.validate_result(wrong_version, case, impl, manifest, tmp_path)

    wrong_selection = copy.deepcopy(valid)
    wrong_selection["config"]["selection"]["candidate_id"] = -1
    with pytest.raises(campaign.CampaignError, match="selection mismatch"):
        campaign.validate_result(wrong_selection, case, impl, manifest, tmp_path)


def test_receipt_rejects_rehashed_noncanonical_invocation(tmp_path: Path) -> None:
    manifest = _fixture_root(tmp_path)
    case = campaign.CASES[0]
    impl = "fa4"
    _write_accepted_result(tmp_path, case, impl, manifest)
    directory = campaign._result_dir(tmp_path, case, impl)
    invocation_path = directory / "invocation.json"
    invocation = json.loads(invocation_path.read_text())
    invocation["arguments"][4] = "99"
    campaign._atomic_write_json(invocation_path, invocation)
    receipt_path = directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["invocation_sha256"] = campaign._sha256_file(invocation_path)
    campaign._atomic_write_json(receipt_path, receipt)
    with pytest.raises(campaign.CampaignError, match="invocation mismatch"):
        campaign._validate_receipt(tmp_path, case, impl, manifest)


def test_overlay_replaces_only_25_rows_and_preserves_schema(tmp_path: Path) -> None:
    manifest = _fixture_root(tmp_path)
    for case in campaign.CASES:
        for impl in campaign._expected_implementations(case):
            _write_accepted_result(tmp_path, case, impl, manifest)

    paths = campaign.build_overlay(tmp_path, validate_live=False, validate_report=False)
    assert len(paths) == 8
    replacement_count = 0
    for case, output_path in zip(campaign.CASES, paths, strict=True):
        historical = json.loads(
            (
                tmp_path / "inputs/historical_payloads" / f"{case.case_id}.json"
            ).read_text()
        )
        overlay = json.loads(output_path.read_text())
        assert [row["impl"] for row in overlay["results"]] == list(
            campaign.EXPECTED_PAYLOAD_IMPLEMENTATIONS
        )
        expected_replacements = set(campaign._expected_implementations(case))
        for old, new in zip(historical["results"], overlay["results"], strict=True):
            if old["impl"] in expected_replacements:
                replacement_count += 1
                assert new["version"] != old["version"]
            else:
                assert new == old
            assert "input_seed" not in new
        campaign._reject_absolute_strings(overlay)
    assert replacement_count == 25

    evidence = json.loads(
        (tmp_path / "published/baseline_refresh_evidence.json").read_text()
    )
    assert len(evidence["refreshed_results"]) == 25
    assert evidence["historical_inputs"] == manifest["inputs"]["historical_payloads"]
    assert evidence["kernelagent_inputs"] == manifest["inputs"]["kernelagent_closed"]
    assert evidence["toolchain"] == manifest["toolchain"]
    assert evidence["toolchain_sha256"] == campaign._json_sha256(manifest["toolchain"])
    campaign._reject_absolute_strings(evidence)

    harness_path = REPO_ROOT / "benchmarks/cute/compare_attention_backends.py"
    harness_spec = importlib.util.spec_from_file_location(
        "_cute470_overlay_schema", harness_path
    )
    assert harness_spec is not None and harness_spec.loader is not None
    harness = importlib.util.module_from_spec(harness_spec)
    harness_spec.loader.exec_module(harness)
    harness._validate_report_payloads([json.loads(path.read_text()) for path in paths])

    generalized_setup = REPO_ROOT / "plots/generalized_full_autotune/setup"
    publisher_spec = importlib.util.spec_from_file_location(
        "_cute470_generalized_publisher", generalized_setup / "publish_results.py"
    )
    assert publisher_spec is not None and publisher_spec.loader is not None
    publisher = importlib.util.module_from_spec(publisher_spec)
    sys.path.insert(0, str(generalized_setup))
    try:
        publisher_spec.loader.exec_module(publisher)
    finally:
        sys.path.remove(str(generalized_setup))
    indexed = publisher.index_baseline_payloads(tmp_path / "published/payloads")
    for path, payload in indexed.values():
        with pytest.raises(RuntimeError, match="helion-cute CuTe version"):
            publisher.validate_cute_backend_versions(payload, context=str(path))
        after_generalized_publish = copy.deepcopy(payload)
        helion = next(
            row
            for row in after_generalized_publish["results"]
            if row["impl"] == "helion-cute"
        )
        helion["version"] = helion["version"].replace("CuTe 4.6.1", "CuTe 4.7.0")
        helion["version_label"] = helion["version_label"].replace(
            "CuTe 4.6.1", "CuTe 4.7.0"
        )
        publisher.validate_cute_backend_versions(
            after_generalized_publish, context=str(path)
        )


def test_launcher_has_valid_bash_syntax_and_bounded_cleanup() -> None:
    launcher = Path(__file__).with_name("run_campaign.sh")
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    command = f"""
set -euo pipefail
source {launcher!s}
TERM_GRACE_SECONDS=1
setsid bash -c 'trap "" TERM; while :; do sleep 10; done' &
lane_pids=("$!")
pid=${{lane_pids[0]}}
cleanup_lanes
[[ ${{#lane_pids[@]}} = 0 ]]
! kill -0 "$pid" 2>/dev/null
"""
    subprocess.run(["bash", "-c", command], check=True, timeout=10)


def test_launcher_cleans_descendants_when_last_lane_fails(tmp_path: Path) -> None:
    launcher = Path(__file__).with_name("run_campaign.sh")
    runner = tmp_path / "runner.sh"
    output_root = tmp_path / "output"
    output_root.mkdir()
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "lane=${5:?missing lane}\n"
        "if [[ $lane = dense ]]; then sleep 0.1; exit 0; fi\n"
        'printf "%s\\n" "$BASHPID" >"$3/failed-group.pid"\n'
        "(trap '' TERM; while :; do sleep 10; done) &\n"
        "sleep 0.5\n"
        "exit 17\n"
    )
    runner.chmod(0o755)
    command = f"""
set -euo pipefail
source {launcher!s}
TERM_GRACE_SECONDS=1
if run_parallel_lanes {runner!s} {output_root!s}; then
  status=0
else
  status=$?
fi
[[ $status = 17 ]]
group_pid=$(cat {output_root!s}/failed-group.pid)
if process_group_live "$group_pid"; then
  kill -KILL -- "-$group_pid" 2>/dev/null || true
  exit 1
fi
"""
    subprocess.run(["bash", "-c", command], check=True, timeout=10)


def test_launcher_campaign_lock_is_nonblocking(tmp_path: Path) -> None:
    launcher = Path(__file__).with_name("run_campaign.sh")
    output_root = tmp_path / "campaign"
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                f"source {launcher!s}; "
                f"acquire_campaign_lock {output_root!s}; "
                "printf ready; sleep 30"
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.read(5) == "ready"
        contender = subprocess.run(
            ["bash", str(launcher), str(output_root)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        assert contender.returncode != 0
        assert "another campaign process holds" in contender.stderr
        assert not output_root.exists()
    finally:
        holder.terminate()
        holder.wait(timeout=5)


def test_launcher_adopts_campaign_lock_across_exec(tmp_path: Path) -> None:
    launcher = Path(__file__).with_name("run_campaign.sh")
    output_root = tmp_path / "campaign"
    command = f"""
set -euo pipefail
source {launcher!s}
acquire_campaign_lock {output_root!s}
export CUTE470_BASELINE_CAMPAIGN_LOCK_FD=$campaign_lock_fd
exec bash -c '
  set -euo pipefail
  source "$1"
  campaign_lock_fd=
  adopt_campaign_lock "$2"
  [[ -n $campaign_lock_fd ]]
  ! flock -n "$2.campaign.lock" -c true
' _ {launcher!s} {output_root!s}
"""
    subprocess.run(["bash", "-c", command], check=True, timeout=5)


def test_all_expected_tflops_are_finite(tmp_path: Path) -> None:
    manifest = _fixture_root(tmp_path)
    for case in campaign.CASES:
        for impl in campaign._expected_implementations(case):
            result = _payload(tmp_path, case, impl, manifest)
            validated = campaign.validate_result(result, case, impl, manifest, tmp_path)
            assert math.isfinite(validated["median_tflops"])
