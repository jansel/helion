from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import pytest

MODULE_PATH = Path(__file__).with_name("campaign.py")
SPEC = importlib.util.spec_from_file_location("varied_attention_campaign", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def _versions() -> dict[str, dict[str, str]]:
    return {
        "helion-cute": {
            "version": "Helion test; CuTe 4.7.0",
            "version_label": "Helion test / CuTe 4.7.0",
        },
        "sdpa": {
            "version": "PyTorch test; cuDNN runtime 9.20.0",
            "version_label": "cuDNN 9.20.0.48",
        },
        "fa4": {
            "version": "FlashAttention fa4-v4.0.0.beta23; CuTe 4.7.0",
            "version_label": "fa4-v4.0.0.beta23; CuTe 4.7.0",
        },
        "flexattention-cute": {
            "version": ("PyTorch test; FA4 fa4-v4.0.0.beta23; CuTe 4.7.0"),
            "version_label": ("PyTorch test; FA4 fa4-v4.0.0.beta23; CuTe 4.7.0"),
        },
    }


def _write_manifest(root: Path) -> dict[str, Any]:
    launcher = root / "launcher"
    launcher.mkdir(parents=True)
    (launcher / "campaign.py").write_text("campaign fixture\n")
    (launcher / "run_campaign.sh").write_text("launcher fixture\n")
    (launcher / "build_strict_manifest.py").write_text("validator fixture\n")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "campaign": "varied_attention_cute47",
        "created_utc": "2026-08-20T00:00:00+00:00",
        "expected_measured_commit": campaign.EXPECTED_MEASURED_COMMIT,
        "definition": campaign._campaign_definition(),
        "sources": {
            "helion": {
                "commit": "1" * 40,
                "tree": "2" * 40,
                "describe": "test",
                "benchmark_sha256": "3" * 64,
                "attention_sha256": "4" * 64,
                "source_subset": {"sha256": "5" * 64, "file_count": 100},
            },
            "flash_attention": {
                "commit": campaign.EXPECTED_FA4_COMMIT,
                "tree": "6" * 40,
                "describe": campaign.EXPECTED_FA4_DESCRIBE,
                "flash_attn_cute_tree": "7" * 40,
            },
            "quack": {
                "commit": campaign.EXPECTED_QUACK_COMMIT,
                "tree": "a" * 40,
                "describe": "b5b49da",
                "quack_package_tree": "b" * 40,
            },
            "setup_files": {
                "build_strict_manifest.py": campaign._sha256_file(
                    launcher / "build_strict_manifest.py"
                ),
                "campaign.py": campaign._sha256_file(launcher / "campaign.py"),
                "run_campaign.sh": campaign._sha256_file(launcher / "run_campaign.sh"),
            },
        },
        "toolchain": {"fixture": True},
        "hardware": [],
        "import_roots": {
            "core": {
                "module": "helion/__init__.py",
                "module_sha256": "d" * 64,
            },
            "fa4_flex": {
                "helion_module": "helion/__init__.py",
                "helion_module_sha256": "d" * 64,
                "quack_module": "quack/__init__.py",
                "quack_module_sha256": "e" * 64,
            },
        },
        "implementation_versions": _versions(),
    }
    campaign._atomic_write_json(root / "campaign_manifest.json", manifest)
    return manifest


def _payload(
    root: Path,
    case: campaign.Case,
    impl: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    runs = [10.0 + index / 10 for index in range(campaign.SAMPLE_COUNT)]
    median_ms = statistics.median(runs)
    mean_ms = sum(runs) / len(runs)
    best_ms = min(runs)
    result: dict[str, Any] = {
        "impl": impl,
        **manifest["implementation_versions"][impl],
        "shape": case.shape,
        "gpu": campaign.EXPECTED_GPU_NAME,
        "physical_gpu": str(case.physical_gpu),
        "power_cap_w": campaign.EXPECTED_POWER_CAP_W,
        "input_seed": campaign.INPUT_SEED,
        "flop_model": "softmax_attention_forward",
        "accuracy": "PASS",
        "benchmark_timer": "wall" if impl == "helion-cute" else "event",
        "best_ms": best_ms,
        "median_ms": median_ms,
        "mom_median_ms": median_ms,
        "mean_ms": mean_ms,
        "std_ms": statistics.stdev(runs),
        "runs_ms": runs,
        "best_tflops": campaign._flop_count(case) / best_ms / 1e9,
        "median_tflops": campaign._flop_count(case) / median_ms / 1e9,
        "mom_median_tflops": campaign._flop_count(case) / median_ms / 1e9,
    }
    notes = {
        "helion-cute": [],
        "sdpa": ["Forced torch SDPBackend.CUDNN_ATTENTION."],
        "fa4": [],
        "flexattention-cute": ["Forced PyTorch FlexAttention BACKEND='FLASH'."],
    }[impl]
    if case.epilogue == "relu":
        result["epilogue_flops_included"] = False
        notes = [
            *notes,
            {
                "helion-cute": "ReLU was fused into the generated epilogue.",
                "sdpa": "Timed eager torch.relu after SDPA.",
                "fa4": "Timed eager torch.relu after FA4.",
                "flexattention-cute": (
                    "torch.compile(fullgraph=True) captured FlexAttention and ReLU."
                ),
            }[impl],
        ]
    if notes:
        result["notes"] = notes
    if impl == "helion-cute":
        result_dir = campaign._result_dir(root, case, impl)
        selected_source = "8" * 64
        selected_config = {"block_sizes": [1, 128, 128]}
        shape = (case.z, case.h, case.seq_len, case.head_dim)
        trial = {
            "random_seed": case.tuner_seed,
            "search_algorithm": "LFBOTreeSearch",
            "input_shapes": repr([shape, shape, shape]),
            "dtypes": repr(["torch.bfloat16"] * 3),
            "hardware": campaign.EXPECTED_GPU_NAME,
            "selected_source_hash": selected_source,
            "selected_config": selected_config,
            "selected_source_was_measured": True,
            "num_successful_candidate_measurements": 100,
        }
        source_values = {
            "helion_checkout_git_commit": manifest["sources"]["helion"]["commit"],
            "helion_source_tree_sha256": manifest["sources"]["helion"]["source_subset"][
                "sha256"
            ],
            "helion_source_tree_file_count": 100,
            "helion_source_tree_dirty": False,
        }
        provenance = {
            **source_values,
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
            "final_correctness_launches": (
                campaign.EXPECTED_FINAL_CORRECTNESS_LAUNCHES
            ),
            "final_repeatability_passed": True,
            "final_correctness_passed": True,
            "post_measurement_source_verified": True,
            "post_measurement_source": source_values,
            "selected_source_sha256": selected_source,
            "selected_config": selected_config,
            "trials": [trial],
            "compiler_seed_policy": {
                "schema_version": 1,
                "kind": "canonical_cute_flash",
            },
        }
        cache = result_dir / "cache"
        result["codegen"] = {
            "uses_tcgen05": True,
            "uses_tcgen05_two_cta": False,
            "uses_tma_umma_pipeline": True,
            "uses_relu_epilogue": case.epilogue == "relu",
        }
        result["helion_overrides"] = {
            "env_overrides": {
                "HELION_AUTOTUNE_RANDOM_SEED": str(case.tuner_seed),
                "HELION_DISABLE_AUTOTUNER_HEURISTICS": "0",
                "HELION_AUTOTUNER": "",
                "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "-1",
                "HELION_AUTOTUNE_EFFORT": "full",
                "HELION_AUTOTUNE_BEST_OF_K": "1",
                "HELION_AUTOTUNE_BENCHMARK_TIMEOUT": str(
                    campaign.AUTOTUNE_BENCHMARK_TIMEOUT_SECONDS
                ),
                "HELION_AUTOTUNE_ACCURACY_CHECK": "1",
                "HELION_AUTOTUNER_INITIAL_POPULATION": "from_random",
                "HELION_CACHE_DIR": str(cache / "helion"),
                "CUTE_DSL_CACHE_DIR": str(cache / "cute_dsl"),
                "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
                "TRITON_CACHE_DIR": str(cache / "triton"),
            },
            "config_overrides": {},
            "seed_config_overrides": {},
            "autotuned": True,
            "benchmark_timer": "wall",
            "force_autotune": True,
            "return_lse": False,
            "autotune_provenance": provenance,
        }
        if case.epilogue == "relu":
            result["helion_overrides"]["epilogue"] = "relu"
    return result


def _write_accepted_result(
    root: Path,
    case: campaign.Case,
    impl: str,
    manifest: dict[str, Any],
) -> None:
    result_dir = campaign._result_dir(root, case, impl)
    result_dir.mkdir(parents=True)
    payload = _payload(root, case, impl, manifest)
    result_path = result_dir / "result.json"
    invocation_path = result_dir / "invocation.json"
    campaign._atomic_write_json(result_path, payload)
    _command, _environment, invocation = campaign._invocation(
        root, case, impl, result_dir
    )
    campaign._atomic_write_json(invocation_path, invocation)
    prevalidation_path = None
    if impl == "helion-cute":
        run_id = "c" * 64
        (result_dir / "autotune.csv").write_text(f"run_id\n{run_id}\n")
        (result_dir / "autotune.sources.csv").write_text(f"run_id\n{run_id}\n")
        (result_dir / "autotune.meta.jsonl").write_text(
            json.dumps({"run_id": run_id}) + "\n"
        )
        prevalidation_path = result_dir / "strict-prevalidation.json"
        campaign._atomic_write_json(
            prevalidation_path,
            {
                "schema_version": 1,
                "status": "autotune_complete_prevalidation",
                "autotune_provenance": payload["helion_overrides"][
                    "autotune_provenance"
                ],
            },
        )
    receipt = campaign._receipt_payload(
        root,
        case,
        impl,
        result_path,
        invocation_path,
        prevalidation_path,
    )
    campaign._atomic_write_json(result_dir / "receipt.json", receipt)


def test_case_matrix_and_gpu_assignment_are_exact() -> None:
    assert len(campaign.CASES) == 8
    assert [case.order for case in campaign.CASES] == list(range(1, 9))
    assert all(
        case.physical_gpu == (7 if case.variant == "dense" else 6)
        for case in campaign.CASES
    )
    assert [case.epilogue for case in campaign.CASES].count("relu") == 2
    assert {
        (case.z, case.h, case.seq_len, case.head_dim, case.variant)
        for case in campaign.CASES
    } == {
        (2, 32, 262144, 128, "dense"),
        (2, 32, 524288, 128, "causal"),
        (1, 32, 524288, 64, "dense"),
        (1, 32, 1048576, 64, "causal"),
        (8, 32, 524288, 64, "dense"),
        (8, 32, 786432, 64, "causal"),
        (2, 32, 524288, 64, "dense"),
        (2, 32, 1048576, 64, "causal"),
    }


def test_input_spec_hash_is_stable_and_sensitive() -> None:
    case = campaign.CASES[0]
    first = campaign._json_sha256(case.input_spec)
    second = campaign._json_sha256(dict(case.input_spec))
    assert first == second
    modified = {**case.input_spec, "seed": campaign.INPUT_SEED + 1}
    assert campaign._json_sha256(modified) != first


def test_invocation_is_strict_isolated_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LD_LIBRARY_PATH", "/untrusted/lib")
    monkeypatch.setenv("LIBRARY_PATH", "/untrusted/link")
    case = campaign.CASES[0]
    result_dir = campaign._result_dir(tmp_path, case, "helion-cute")
    command, environment, record = campaign._invocation(
        tmp_path, case, "helion-cute", result_dir
    )
    assert "--helion-require-full-autotune" in command
    assert "--helion-autotune-effort" in command
    assert "--helion-autotune-budget-seconds" not in command
    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert "HELION_FA4_ROOT" not in environment
    assert environment["PYTHONPATH"].endswith("checkouts/helion")
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "LD_LIBRARY_PATH" not in environment
    assert "LIBRARY_PATH" not in environment
    assert record["pythonpath_scrubbed_before_harness"] is True
    campaign._reject_absolute_strings(record)

    _, flex_environment, _ = campaign._invocation(
        tmp_path,
        case,
        "flexattention-cute",
        campaign._result_dir(tmp_path, case, "flexattention-cute"),
    )
    assert flex_environment["HELION_FA4_ROOT"].endswith("checkouts/flash-attention")
    python_paths = flex_environment["PYTHONPATH"].split(campaign.os.pathsep)
    assert python_paths[0].endswith("checkouts/helion")
    assert python_paths[1].endswith("checkouts/quack")


def test_result_validation_fails_closed(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    case = campaign.CASES[-1]
    result_dir = campaign._result_dir(tmp_path, case, "helion-cute")
    valid = _payload(tmp_path, case, "helion-cute", manifest)
    campaign.validate_result_payload(valid, case, "helion-cute", manifest, result_dir)

    wrong_gpu = json.loads(json.dumps(valid))
    wrong_gpu["physical_gpu"] = "7"
    with pytest.raises(campaign.CampaignError, match="physical GPU"):
        campaign.validate_result_payload(
            wrong_gpu, case, "helion-cute", manifest, result_dir
        )

    short = json.loads(json.dumps(valid))
    short["runs_ms"] = short["runs_ms"][:-1]
    with pytest.raises(campaign.CampaignError, match="9 timing samples"):
        campaign.validate_result_payload(
            short, case, "helion-cute", manifest, result_dir
        )

    seeded = json.loads(json.dumps(valid))
    seeded["helion_overrides"]["seed_config_overrides"] = {"block_sizes": [1]}
    with pytest.raises(campaign.CampaignError, match="seed_config_overrides"):
        campaign.validate_result_payload(
            seeded, case, "helion-cute", manifest, result_dir
        )


def test_strict_adapter_normalizes_only_declared_campaign_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(tmp_path)
    case = campaign.CASES[-1]
    result_dir = campaign._result_dir(tmp_path, case, "helion-cute")
    result_dir.mkdir(parents=True)
    payload = _payload(tmp_path, case, "helion-cute", manifest)
    provenance = payload["helion_overrides"]["autotune_provenance"]
    provenance["autotune_baseline_fn"] = (
        "examples.attention._causal_attention_relu_output_baseline"
    )
    provenance["autotune_benchmark_timeout"] = (
        campaign.AUTOTUNE_BENCHMARK_TIMEOUT_SECONDS
    )
    captured: dict[str, Any] = {}

    class Validator:
        @staticmethod
        def validate_strict_provenance(
            _path: Path,
            normalized: dict[str, Any],
            _variant: str,
            _seq_len: int,
            _tuner_seed: int,
            **kwargs: object,
        ) -> tuple[dict[str, Any], dict[str, Any], int, bool, int]:
            captured["normalized"] = normalized
            captured["kwargs"] = kwargs
            normalized_provenance = normalized["helion_overrides"][
                "autotune_provenance"
            ]
            return normalized_provenance, normalized_provenance["trials"][0], 1, True, 0

        @staticmethod
        def read_and_validate_ledger(
            _path: Path,
            _trial: dict[str, Any],
            _config: dict[str, Any],
            _source: str,
        ) -> dict[str, object]:
            return {"rows": [], "run_id": "c" * 64}

        @staticmethod
        def validate_structural_prefix_execution(*_args: object) -> dict[str, Any]:
            return {"compiler_seed_policy": provenance["compiler_seed_policy"]}

    monkeypatch.setattr(campaign, "_load_strict_validator", lambda *_args: Validator())
    monkeypatch.setattr(
        campaign,
        "_generalized_autotune_sidecars",
        lambda *_args: {
            "configs": {},
            "attempt_by_config": {},
            "attempt_history_by_config": {},
            "csv_sha256": "d" * 64,
            "metadata_sha256": "e" * 64,
        },
    )
    (result_dir / "autotune.sources.csv").write_text("fixture\n")
    summary = campaign._validate_helion_external_evidence(
        tmp_path, result_dir / "result.json", payload, case, manifest
    )
    normalized = captured["normalized"]
    normalized_overrides = normalized["helion_overrides"]
    normalized_env = normalized_overrides["env_overrides"]
    assert "HELION_CACHE_DIR" not in normalized_env
    assert normalized_env["HELION_AUTOTUNE_BENCHMARK_TIMEOUT"] == "60"
    normalized_provenance = normalized_overrides["autotune_provenance"]
    assert normalized_provenance["autotune_benchmark_timeout"] == 60
    assert normalized_provenance["autotune_baseline_fn"].endswith(
        "_causal_attention_output_baseline"
    )
    assert provenance["autotune_benchmark_timeout"] == 180
    assert captured["kwargs"] == {
        "expected_input_shape": (2, 32, 1048576, 64),
        "expected_input_dtype": "torch.bfloat16",
    }
    assert summary["run_id"] == "c" * 64


def test_snapshotted_validator_accepts_authentic_full_evidence_fixture(
    tmp_path: Path,
) -> None:
    manifest_record = _write_manifest(tmp_path)
    generalized_setup = (
        Path(__file__).resolve().parents[2] / "generalized_full_autotune/setup"
    )
    validator_source = generalized_setup / "build_strict_manifest.py"
    validator_snapshot = tmp_path / "launcher/build_strict_manifest.py"
    validator_snapshot.write_bytes(validator_source.read_bytes())
    manifest_record["sources"]["setup_files"]["build_strict_manifest.py"] = (
        campaign._sha256_file(validator_snapshot)
    )
    campaign._atomic_write_json(tmp_path / "campaign_manifest.json", manifest_record)
    validator = campaign._load_strict_validator(tmp_path, manifest_record)

    helper_path = generalized_setup / "test_build_strict_manifest.py"
    helper_spec = importlib.util.spec_from_file_location(
        "_varied_authentic_strict_fixture", helper_path
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    previous = sys.modules.get("build_strict_manifest")
    sys.modules["build_strict_manifest"] = validator
    try:
        helper_spec.loader.exec_module(helper)
    finally:
        if previous is None:
            del sys.modules["build_strict_manifest"]
        else:
            sys.modules["build_strict_manifest"] = previous

    variant = "dense"
    seq_len = 32768
    tuner_seed = 2026081501
    payload, source_rows, autotune_rows, metadata_record = helper.fixture_payload(
        variant, seq_len, 7, tuner_seed, 0
    )
    result_dir = tmp_path / "authentic-fixture"
    result_dir.mkdir()
    result_path = result_dir / "result.json"
    campaign._atomic_write_json(result_path, payload)
    helper.write_ledger(result_dir / "autotune.sources.csv", source_rows)
    helper.write_autotune_csv(result_dir / "autotune.csv", autotune_rows)
    (result_dir / "autotune.meta.jsonl").write_text(
        json.dumps(metadata_record, sort_keys=True) + "\n"
    )

    provenance, trial, *_rest = validator.validate_strict_provenance(
        result_path, payload, variant, seq_len, tuner_seed
    )
    ledger = validator.read_and_validate_ledger(
        result_dir / "autotune.sources.csv",
        trial,
        provenance["selected_config"],
        provenance["selected_source_sha256"],
    )
    autotune = validator.validate_autotune_sidecars(
        result_dir / "autotune.csv",
        result_dir / "autotune.meta.jsonl",
        ledger["rows"],
        ledger["run_id"],
        provenance["selected_config"],
        variant,
        seq_len,
        tuner_seed,
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
    assert structural["compiler_seed_policy"] == provenance["compiler_seed_policy"]


def test_builder_emits_renderer_compatible_csv_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(tmp_path)
    for case in campaign.CASES:
        for impl in campaign.IMPLEMENTATIONS:
            _write_accepted_result(tmp_path, case, impl, manifest)

    strict_calls: list[str] = []

    def validate_strict(
        _root: Path,
        _path: Path,
        payload: dict[str, Any],
        case: campaign.Case,
        _manifest: dict[str, Any],
    ) -> dict[str, object]:
        strict_calls.append(case.case_id)
        return {
            "run_id": "c" * 64,
            "autotune_csv_sha256": campaign._sha256_file(_path.parent / "autotune.csv"),
            "autotune_metadata_sha256": campaign._sha256_file(
                _path.parent / "autotune.meta.jsonl"
            ),
            "source_ledger_sha256": campaign._sha256_file(
                _path.parent / "autotune.sources.csv"
            ),
            "compiler_seed_policy_sha256": campaign._json_sha256(
                payload["helion_overrides"]["autotune_provenance"][
                    "compiler_seed_policy"
                ]
            ),
        }

    monkeypatch.setattr(campaign, "_validate_helion_external_evidence", validate_strict)

    csv_path, evidence_path = campaign.build_outputs(tmp_path, validate_live=False)
    assert strict_calls == [case.case_id for case in campaign.CASES]
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == campaign.CSV_FIELDS
    assert len(rows) == 32
    assert {int(row["physical_gpu"]) for row in rows if row["variant"] == "dense"} == {
        7
    }
    assert {int(row["physical_gpu"]) for row in rows if row["variant"] == "causal"} == {
        6
    }
    assert {int(row["sample_count"]) for row in rows} == {9}
    assert {row["correctness"] for row in rows} == {"PASS"}

    evidence = json.loads(evidence_path.read_text())
    assert len(evidence["measurements"]) == 32
    for item in evidence["measurements"]:
        expected = item["flop_count"] / statistics.median(item["runs_ms"]) / 1e9
        assert math.isclose(item["median_tflops"], expected, rel_tol=1e-15)
        assert not item["source_artifact"].startswith("/")
    campaign._reject_absolute_strings(evidence)


def test_receipt_rejects_modified_result(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    case = campaign.CASES[0]
    _write_accepted_result(tmp_path, case, "sdpa", manifest)
    result_dir = campaign._result_dir(tmp_path, case, "sdpa")
    payload = json.loads((result_dir / "result.json").read_text())
    payload["accuracy"] = "FAIL"
    campaign._atomic_write_json(result_dir / "result.json", payload)
    with pytest.raises(campaign.CampaignError, match="receipt mismatch"):
        campaign._validate_receipt(tmp_path, case, "sdpa", manifest)


def test_receipt_rejects_modified_invocation_arguments(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    case = campaign.CASES[0]
    _write_accepted_result(tmp_path, case, "sdpa", manifest)
    result_dir = campaign._result_dir(tmp_path, case, "sdpa")
    invocation_path = result_dir / "invocation.json"
    invocation = json.loads(invocation_path.read_text())
    invocation["arguments"][5] = "999"
    campaign._atomic_write_json(invocation_path, invocation)
    receipt_path = result_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["invocation_sha256"] = campaign._sha256_file(invocation_path)
    campaign._atomic_write_json(receipt_path, receipt)
    with pytest.raises(campaign.CampaignError, match="invocation mismatch"):
        campaign._validate_receipt(tmp_path, case, "sdpa", manifest)


def test_launcher_has_valid_bash_syntax() -> None:
    subprocess.run(
        ["bash", "-n", str(Path(__file__).with_name("run_campaign.sh"))],
        check=True,
    )


def test_launcher_cleanup_escalates_and_clears_process_groups() -> None:
    launcher = Path(__file__).with_name("run_campaign.sh")
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
export VARIED_ATTENTION_CAMPAIGN_LOCK_FD=$campaign_lock_fd
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
