from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_strict_manifest
import combine_results
from combine_results import recorded_bootstrap_settings
from combine_results import resolve_artifact_root
from combine_results import validate_manifest_files
import paired_worker
from paired_worker import CASES
from paired_worker import PEAKY_STRESS_THRESHOLDS
from paired_worker import SOURCE_LEDGER_FIELDS
from paired_worker import Case
from paired_worker import atomic_write_json
from paired_worker import canonical_sha256
from paired_worker import git_output
from paired_worker import peaky_tensor_error_summary
from paired_worker import sha256
from paired_worker import validate_artifact_set
from paired_worker import validate_checkout
from paired_worker import validate_worker_harness_sha256
from run_all8 import REPO_ROOT as RUNNER_REPO_ROOT
from run_all8 import aggregate_results
from run_all8 import current_harness_sha256
from run_all8 import finalize_run
from run_all8 import json_payload_sha256
from run_all8 import logical_artifact_root_reference
from run_all8 import logical_output_reference
from run_all8 import require_portable_json
from run_all8 import resolve_output_reference
from run_all8 import strict_artifact_identities
from run_all8 import validate_harness_sha256
from run_all8 import validate_strict_artifact_identities
from run_all8 import worker_environment
from test_build_strict_manifest import (
    add_phase_config_identity as shared_add_phase_config_identity,
)


def add_flash_normalization_context(
    provenance: dict[str, Any],
    trial: dict[str, Any],
    *,
    shape: tuple[int, ...],
    dtype: str,
    causal: bool,
) -> None:
    provenance["autotune_baseline_fn"] = (
        "examples.attention._causal_attention_output_baseline"
        if causal
        else "examples.attention._attention_output_baseline"
    )
    context = {
        "schema_version": 1,
        "backend": "cute",
        "config_spec_structural_fingerprint_sha256": "a" * 64,
        "default_config_sha256": provenance["flash_fragment_default_sha256"],
        "dtype": dtype,
        "head_dim": shape[-1],
        "num_kv": (shape[-2] + 127) // 128,
        "num_bh": math.prod(shape[:-2]),
        "tensor_4d_heads": shape[-3],
        "is_causal": causal,
        "has_kv_tile_pruning": False,
        "requires_ws_overlap": False,
        "small_biased_candidate": False,
        "standard_dense_output": not causal,
        "standard_causal_output": causal,
        "output_requires_tma": False,
        "supports_tensor_4d_tma": True,
        "block_size_targets": [[0, 1], [1, 128], [2, 128]],
        "flat_key_layout": [["cute_flash_pipeline_family", 1, False]],
    }
    provenance["flash_normalization_context"] = context
    provenance["flash_normalization_context_sha256"] = canonical_sha256(context)
    trial["input_shapes"] = repr([shape] * 3)
    trial["dtypes"] = repr([dtype] * 3)


def add_clc_lane_catalog(
    provenance: dict[str, Any], entries: list[dict[str, Any]] | None = None
) -> None:
    catalog = [] if entries is None else entries
    provenance["flash_clc_lane_catalog"] = catalog
    provenance["flash_clc_lane_catalog_sha256"] = canonical_sha256(catalog)


def add_phase_config_identity(
    provenance: dict[str, Any],
    phase: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    *,
    attempts: dict[str, dict[str, Any]] | None = None,
) -> None:
    shared_add_phase_config_identity(provenance, phase, configs, attempts=attempts)


class PeakyTensorErrorSummaryTests(unittest.TestCase):
    def test_records_strict_metrics_and_accepts_small_error(self) -> None:
        expected = torch.ones((1, 16, 4), dtype=torch.float32)
        actual = expected + 5e-4

        summary = peaky_tensor_error_summary(actual, expected, chunk_rows=3)

        self.assertTrue(summary["passed"])
        self.assertTrue(summary["finite_outputs"])
        self.assertEqual(summary["thresholds"], PEAKY_STRESS_THRESHOLDS)
        self.assertEqual(summary["atol"], 0.002)
        self.assertEqual(summary["rtol"], 0.01)
        self.assertEqual(summary["mismatch_count"], 0)
        self.assertLess(summary["max_abs"], 0.01)
        self.assertLess(summary["nrmse"], 0.002)
        self.assertEqual(summary["nrmse_normalization"], "rms(cudnn_sdpa_output)")

    def test_rejects_each_exclusive_numerical_gate(self) -> None:
        expected = torch.ones((1, 16, 4), dtype=torch.float32)

        nrmse_failure = peaky_tensor_error_summary(expected + 0.003, expected)
        self.assertLess(nrmse_failure["max_abs"], 0.01)
        self.assertEqual(nrmse_failure["mismatch_fraction"], 0.0)
        self.assertGreater(nrmse_failure["nrmse"], 0.002)
        self.assertFalse(nrmse_failure["passed"])

        large_expected = expected * 10.0
        max_abs_failure = peaky_tensor_error_summary(
            large_expected + 0.011, large_expected
        )
        self.assertLess(max_abs_failure["nrmse"], 0.002)
        self.assertEqual(max_abs_failure["mismatch_fraction"], 0.0)
        self.assertGreater(max_abs_failure["max_abs"], 0.01)
        self.assertFalse(max_abs_failure["passed"])

        expected = torch.ones((1, 200_000, 1), dtype=torch.float32)
        expected[:, :2] = 0.0
        one_mismatch = expected.clone()
        one_mismatch[:, 0] = 0.003
        self.assertTrue(peaky_tensor_error_summary(one_mismatch, expected)["passed"])

        two_mismatches = one_mismatch.clone()
        two_mismatches[:, 1] = 0.003
        mismatch_failure = peaky_tensor_error_summary(two_mismatches, expected)
        self.assertEqual(mismatch_failure["mismatch_count"], 2)
        self.assertEqual(mismatch_failure["mismatch_fraction"], 1e-5)
        self.assertLess(mismatch_failure["max_abs"], 0.01)
        self.assertLess(mismatch_failure["nrmse"], 0.002)
        self.assertFalse(mismatch_failure["passed"])

    def test_rejects_nonfinite_output(self) -> None:
        expected = torch.ones((1, 2, 2), dtype=torch.float32)
        actual = expected.clone()
        actual[0, 0, 0] = torch.nan

        summary = peaky_tensor_error_summary(actual, expected)

        self.assertEqual(summary["actual_nonfinite"], 1)
        self.assertEqual(summary["expected_nonfinite"], 0)
        self.assertFalse(summary["finite_outputs"])
        self.assertFalse(summary["passed"])


class CheckoutValidationTests(unittest.TestCase):
    def test_version_accepts_longer_git_abbreviation(self) -> None:
        commit = "c3e36b65d69681c23e053042b0bc21e2331bad17"
        self.assertEqual(
            paired_worker._validate_version(
                f"Helion 1.4.0.dev157+g{commit[:9]}; CuTe 4.7.0",
                commit,
                Path("result.json"),
            ),
            "4.7.0",
        )

    @staticmethod
    def initialize_checkout(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / ".gitignore").write_text(
            ".pytest_cache/\n.ruff_cache/\n__pycache__/\n/torch/\n"
        )
        (root / "tracked.txt").write_text("tracked\n")
        subprocess.run(
            ["git", "add", ".gitignore", "tracked.txt"], cwd=root, check=True
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Validation Test",
                "-c",
                "user.email=validation@example.com",
                "commit",
                "-qm",
                "initial",
            ],
            cwd=root,
            check=True,
        )

    @staticmethod
    def stage_allowed_harness(root: Path) -> list[str]:
        paths = [
            ".validation/generalized_paired/build_strict_manifest.py",
            ".validation/generalized_paired/combine_results.py",
            ".validation/generalized_paired/paired_worker.py",
            ".validation/generalized_paired/run_all8.py",
            ".validation/generalized_paired/test_build_strict_manifest.py",
            ".validation/generalized_paired/test_static.py",
        ]
        for path in paths:
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("# staged validation harness\n")
        return paths

    def test_accepts_only_staged_harness_in_otherwise_clean_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize_checkout(root)
            harness = self.stage_allowed_harness(root)

            checkout = validate_checkout(root)

        self.assertTrue(checkout["runtime_tracked_clean"])
        self.assertTrue(checkout["runtime_untracked_clean"])
        self.assertTrue(checkout["runtime_ignored_clean"])
        self.assertEqual(checkout["runtime_allowed_untracked_files"], harness)
        self.assertEqual(checkout["runtime_allowed_ignored_file_count"], 0)

    def test_rejects_untracked_startup_hook_and_helion_addition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize_checkout(root)
            self.stage_allowed_harness(root)
            (root / "sitecustomize.py").write_text("raise RuntimeError\n")
            (root / "helion").mkdir()
            (root / "helion/shadow.py").write_text("raise RuntimeError\n")

            with self.assertRaisesRegex(
                RuntimeError, "helion/shadow.py.*sitecustomize.py"
            ):
                validate_checkout(root)

    def test_rejects_checkout_local_bytecode_and_tool_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize_checkout(root)
            self.stage_allowed_harness(root)
            for path in (
                root / ".pytest_cache/v/cache/nodeids",
                root / "helion/__pycache__/module.cpython-312.pyc",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("untrusted cache\n")

            with self.assertRaisesRegex(
                RuntimeError,
                "caches must be external: .*pytest_cache.*helion/__pycache__",
            ):
                validate_checkout(root)

    def test_rejects_ignored_import_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize_checkout(root)
            self.stage_allowed_harness(root)
            shadow = root / "torch/__init__.py"
            shadow.parent.mkdir()
            shadow.write_text("raise RuntimeError\n")

            with self.assertRaisesRegex(
                RuntimeError, "caches must be external: torch/__init__.py"
            ):
                validate_checkout(root)


def make_payload(case: Case, *, legacy_source_key: bool = False) -> dict[str, Any]:
    config = {
        "block_sizes": [1, 128, 128],
        "cute_flash_pipeline_family": "fa4",
        "cute_flash_wait_hint": 10000000,
        "cute_flash_epi_tma": False,
        "cute_flash_epi_stg": True,
        "cute_flash_epi_stg_store": "slice",
        "cute_flash_epi_stg_gmem": "stage",
    }
    alternate_config = {**config, "cute_flash_wait_hint": -1}
    source_hash = canonical_sha256({"case": case.name, "config": config})
    coverage_hash = canonical_sha256(config)
    alternate_coverage_hash = canonical_sha256(alternate_config)
    coverage = [
        {"config": config, "config_sha256": coverage_hash},
        {
            "config": alternate_config,
            "config_sha256": alternate_coverage_hash,
        },
    ]
    compiler_seed_ids = [canonical_sha256(config)[:16]]
    compiler_seed_policy = {
        "schema_version": 1,
        "kind": "canonical_cute_flash",
        "heuristic_names": ["cute_flash_attention"],
        "raw_config_count": 1,
        "effective_config_ids": compiler_seed_ids,
        "effective_config_ids_sha256": canonical_sha256(compiler_seed_ids),
        "timeout_retry_repetitions": 3,
    }
    provenance = {
        "require_full_autotune": True,
        "effort": "full",
        "requested_force_autotune": True,
        "effective_force_autotune": True,
        "fixed_config": False,
        "autotune_budget_seconds": None,
        "autotune_max_generations": None,
        "autotune_lfbo_max_generations": 8,
        "autotune_best_of_k": 1,
        "autotune_config_overrides": {},
        "user_seed_configs": False,
        "disable_autotuner_heuristics": False,
        "compiler_seed_config_count": 1,
        "compiler_seed_policy": compiler_seed_policy,
        "compiler_default_config": False,
        "kernel_declared_config_count": 0,
        "autotune_initial_population_strategy_override": None,
        "autotune_initial_population_size": 100,
        "autotuner_initial_population_env": "from_random",
        "autotuner_env": "",
        "autotune_num_neighbors_cap_env": "-1",
        "autotuner_fn_is_default": True,
        "autotune_baseline_fn_is_expected": True,
        "autotune_baseline_atol": 5e-2,
        "autotune_baseline_rtol": 2e-2,
        "autotune_baseline_accuracy_check_fn": False,
        "autotune_benchmark_fn": False,
        "autotune_rebenchmark_threshold": None,
        "autotune_suspicious_rebenchmark_ratio": None,
        "autotune_accuracy_check": True,
        "autotune_compile_timeout": 60,
        "autotune_benchmark_subprocess": True,
        "autotune_benchmark_timeout": 60,
        "autotune_adaptive_timeout": True,
        "autotune_force_persistent": False,
        "autotune_finishing_rounds_env": "",
        "autotune_ignore_errors": False,
        "autotune_search_acf": [],
        "autotune_config_filter": False,
        "flash_value_prior_keys": [],
        "active_value_prior_keys": [],
        "flash_fragment_default_config": config,
        "flash_fragment_default_sha256": coverage_hash,
        "flash_structural_coverage_active_values": [
            {"key": "cute_flash_pipeline_family", "value": "fa4"},
            {"key": "cute_flash_epi_tma", "value": False},
            {"key": "cute_flash_epi_stg", "value": True},
            {"key": "cute_flash_epi_stg_store", "value": "slice"},
            {"key": "cute_flash_epi_stg_gmem", "value": "stage"},
        ],
        "flash_structural_coverage_design": coverage,
        "flash_structural_coverage_design_count": len(coverage),
        "flash_structural_coverage_design_sha256": canonical_sha256(
            [config, alternate_config]
        ),
        "flash_structural_coverage_uncovered_values": [],
        "flash_structural_coverage_underqualified_values": [],
        "flash_structural_leaf_catalog": [
            {"family": "fa4", "compound_packet": None, "softmax_disc": False}
        ],
        "flash_pipeline_lane_catalog": [
            {
                "family": "fa4",
                "compound_packet": None,
                "softmax_disc": False,
                "pipeline_lanes": [],
            }
        ],
        "flash_structural_coverage_underqualified_leaves": [],
        "flash_structural_coverage_interaction_key_groups": [
            list(group) for group in paired_worker.FLASH_INTERACTION_KEY_GROUPS
        ],
        "flash_structural_coverage_active_interactions": [
            {
                "keys": list(paired_worker.FLASH_INTERACTION_KEY_GROUPS[0]),
                "values": [False, True, "slice", "stage"],
            }
        ],
        "flash_structural_coverage_uncovered_interactions": [],
        "flash_structural_qualification_values": [
            {"key": "cute_flash_pipeline_family", "value": "fa4"}
        ],
        "flash_structural_parent_coverage_prefix_count": 1,
        "flash_structural_qualification_prefix_count": 2,
        "flash_structural_population_budget": 50,
        "flash_structural_injected_design_count": len(coverage),
        "flash_structural_qualification_rounds": 2,
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": 4,
        "flash_structural_retained_candidates_per_leaf": 2,
        "flash_structural_retained_family_cap": None,
        "flash_structural_retained_family_limit": 1,
        "flash_structural_retained_family_slowdown_limit": 2.0,
        "flash_structural_starting_path_limit": 13,
        "flash_structural_unrestricted_path_exhausts_generation_budget": True,
        "cute_flash_env_overrides": {},
        "final_correctness_enabled": True,
        "final_correctness_launches": 64,
        "final_repeatability_passed": True,
        "final_correctness_passed": True,
        "autotune_cache": "LocalAutotuneCache",
        "rebenchmark_env_overrides": {},
        "selected_config": config,
        "selected_config_is_structural_coverage_design_member": True,
        "selected_config_nearest_structural_coverage_design_field_distance": 0,
        "selected_config_nearest_structural_coverage_design_config_sha256": [
            coverage_hash
        ],
        "trials": [
            {
                "input_shapes": repr([(2, 32, case.seq_len, 64)] * 3),
                "dtypes": repr(["torch.float16"] * 3),
                "hardware": "NVIDIA B200",
                "random_seed": 101,
                "search_algorithm": "LFBOTreeSearch",
                "num_configs_tested": 300,
                "num_compile_failures": 3,
                "num_worker_failures": 0,
                "num_isolated_rebenchmark_timeouts": 0,
                "num_accuracy_failures": 0,
                "num_successful_candidate_measurements": 297,
                "num_unique_sources": 300,
                "num_source_deduplications": 2,
                "num_generations": 8,
                "autotune_time": 1234.0,
                "best_perf_ms": 12.0,
                "selected_config": config,
                "selected_source_hash": source_hash,
                "selected_source_was_measured": True,
            }
        ],
    }
    add_clc_lane_catalog(provenance)
    provenance[
        "selected_source_hash" if legacy_source_key else "selected_source_sha256"
    ] = source_hash
    trial = provenance["trials"][0]
    add_flash_normalization_context(
        provenance,
        trial,
        shape=(2, 32, case.seq_len, 64),
        dtype="torch.float16",
        causal=case.causal,
    )
    head = git_output("rev-parse", "HEAD")
    return {
        "impl": "helion-cute",
        "version": f"Helion 1.4.0.dev0+g{head[:8]}; CuTe 4.7.0",
        "version_label": f"Helion dev+g{head[:8]} / CuTe 4.7.0",
        "shape": {
            "z": 2,
            "h": 32,
            "seq_len": case.seq_len,
            "head_dim": 64,
            "dtype": "float16",
            "causal": int(case.causal),
            "biased": 0,
        },
        "physical_gpu": str(case.physical_gpu),
        "gpu": "NVIDIA B200",
        "power_cap_w": 750,
        "input_seed": 2026081500,
        "flop_model": "softmax_attention_forward",
        "accuracy": "PASS",
        "benchmark_timer": "wall",
        "helion_overrides": {
            "env_overrides": {},
            "config_overrides": {},
            "seed_config_overrides": {},
            "autotuned": True,
            "force_autotune": True,
            "return_lse": False,
            "autotune_provenance": provenance,
        },
    }


class StaticArtifactValidationTests(unittest.TestCase):
    def test_paired_worker_requires_protocol_in_pipeline_lane_identity(self) -> None:
        case = CASES[("dense", 32768)]
        provenance = make_payload(case)["helion_overrides"]["autotune_provenance"]

        catalog = paired_worker.flash_pipeline_lane_catalog(
            Path("fixture.json"), provenance
        )
        self.assertEqual(len(catalog), 1)

        malformed = copy.deepcopy(provenance)
        malformed["flash_pipeline_lane_catalog"][0].pop("softmax_disc")
        with self.assertRaisesRegex(
            RuntimeError, "invalid flash pipeline lane catalog entry"
        ):
            paired_worker.flash_pipeline_lane_catalog(Path("fixture.json"), malformed)

    def test_paired_worker_accepts_dynamic_structural_population_budget(self) -> None:
        case = CASES[("dense", 32768)]
        for design_count, qualification_prefix_count, expected_budget in (
            (63, 63, 63),
            (120, 63, 50),
        ):
            with self.subTest(design_count=design_count):
                provenance = make_payload(case)["helion_overrides"][
                    "autotune_provenance"
                ]
                first = provenance["flash_structural_coverage_design"][0]["config"]
                second = provenance["flash_structural_coverage_design"][1]["config"]
                configs = [first, second]
                configs.extend(
                    {**first, "cute_flash_wait_hint": value}
                    for value in range(design_count - len(configs))
                )
                provenance["flash_structural_coverage_design"] = [
                    {"config": config, "config_sha256": canonical_sha256(config)}
                    for config in configs
                ]
                provenance["flash_structural_coverage_design_count"] = len(configs)
                provenance["flash_structural_coverage_design_sha256"] = (
                    canonical_sha256(configs)
                )
                provenance["flash_structural_qualification_prefix_count"] = (
                    qualification_prefix_count
                )
                provenance["flash_structural_population_budget"] = expected_budget
                provenance["flash_structural_injected_design_count"] = min(
                    expected_budget, len(configs)
                )

                self.assertEqual(
                    len(
                        paired_worker._validate_coverage(
                            provenance, Path("fixture.json")
                        )
                    ),
                    design_count,
                )

    def test_normalization_context_is_shape_dependent(self) -> None:
        for causal in (False, True):
            with self.subTest(causal=causal):
                shape = (2, 7, 256, 128)
                trial: dict[str, Any] = {}
                provenance: dict[str, Any] = {"flash_fragment_default_sha256": "b" * 64}
                add_flash_normalization_context(
                    provenance,
                    trial,
                    shape=shape,
                    dtype="torch.bfloat16",
                    causal=causal,
                )
                build_strict_manifest.validate_flash_normalization_context(
                    "fixture", provenance, trial
                )
                paired_worker.validate_flash_normalization_context(
                    "fixture", provenance, trial
                )

                wrong = copy.deepcopy(provenance)
                wrong["flash_normalization_context"]["head_dim"] = 64
                wrong["flash_normalization_context_sha256"] = canonical_sha256(
                    wrong["flash_normalization_context"]
                )
                with self.assertRaisesRegex(RuntimeError, "head_dim"):
                    build_strict_manifest.validate_flash_normalization_context(
                        "fixture", wrong, trial
                    )
                with self.assertRaisesRegex(RuntimeError, "head_dim"):
                    paired_worker.validate_flash_normalization_context(
                        "fixture", wrong, trial
                    )

    def test_paired_worker_replays_bound_d128_compound_projection(self) -> None:
        from types import SimpleNamespace

        from benchmarks.cute.compare_attention_backends import (
            _canonical_flash_projection,
        )
        from benchmarks.cute.compare_attention_backends import (
            _flash_normalization_context,
        )

        from helion._compiler.backend import CuteBackend
        from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
        from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
        from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
        from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config
        from helion.autotuner.config_generation import ConfigGeneration
        from helion.autotuner.config_spec import BlockSizeSpec
        from helion.autotuner.config_spec import ConfigSpec
        from helion.runtime.config import Config

        for causal, packet, family in (
            (False, "deg2_16x6", "fa4_2cta"),
            (
                True,
                "causal_hd128_resident3_013_prefetch2_deg2_early_acquire",
                "fa4",
            ),
        ):
            with self.subTest(causal=causal):
                spec = ConfigSpec(backend=CuteBackend())
                for block_id, target in enumerate((1, 128, 128)):
                    spec.block_sizes.append(
                        BlockSizeSpec(block_id=block_id, size_hint=target)
                    )
                spec.enable_cute_flash_search(
                    head_dim=128,
                    num_kv=256,
                    num_bh=64,
                    tensor_4d_heads=32,
                    dtype=torch.bfloat16,
                    block_size_targets={0: 1, 1: 128, 2: 128},
                    is_causal=causal,
                    standard_dense_output=not causal,
                    standard_causal_output=causal,
                )
                generation = ConfigGeneration(spec)
                source = next(
                    config.config
                    for config in generation.flash_deterministic_population_configs()
                    if (
                        (leaf := flash_structural_leaf_from_config(config.config))
                        is not None
                        and leaf.pipeline_family == family
                        and leaf.compound_exp2_packet is None
                    )
                )
                overrides = {"cute_flash_exp2_packet": packet}
                projected = _canonical_flash_projection(generation, source, overrides)
                context = _flash_normalization_context(spec)
                provenance = {
                    "flash_normalization_context": context,
                    "flash_normalization_context_sha256": canonical_sha256(context),
                    "trials": [
                        {
                            "search_phase_metrics": {
                                "compound_transfers": [
                                    {
                                        "transfers": [
                                            {
                                                "source_config": source,
                                                "projected_config": projected,
                                                "projection_overrides": overrides,
                                                "transferred_config_id": canonical_sha256(
                                                    projected
                                                )[:16],
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    ],
                }
                terminal_policy = (
                    build_strict_manifest.expected_terminal_refinement_policy()
                )
                terminal_surface = generation.flash_terminal_coordinate_surface_catalog(
                    radius=terminal_policy["radius"]
                )
                provenance.update(
                    {
                        "flash_terminal_coordinate_refinement_policy": terminal_policy,
                        "flash_terminal_coordinate_refinement_policy_sha256": (
                            canonical_sha256(terminal_policy)
                        ),
                        "flash_terminal_coordinate_surface_catalog": terminal_surface,
                        "flash_terminal_coordinate_surface_catalog_sha256": (
                            canonical_sha256(terminal_surface)
                        ),
                    }
                )
                source_config = Config.from_dict(source)
                source_leaf = flash_structural_leaf_from_config(source)
                assert source_leaf is not None
                leaf_overrides = {
                    FLASH_PIPELINE_FAMILY_KEY: source_leaf.pipeline_family,
                    FLASH_SOFTMAX_DISC_KEY: source_leaf.softmax_disc,
                }
                if source_leaf.compound_exp2_packet is not None:
                    leaf_overrides[FLASH_EXP2_PACKET_KEY] = (
                        source_leaf.compound_exp2_packet
                    )
                leaf_generation = spec.create_config_generation(
                    overrides=leaf_overrides
                )
                requests = []
                terminal_manifest = {}
                source_id = canonical_sha256(source)[:16]
                terminal_manifest[source_id] = {"config": source}
                for projection in generation.canonicalize_coordinate_projections(
                    leaf_generation.coordinate_neighbor_projections(
                        leaf_generation.flatten(source_config), radius=2
                    ),
                    base_config=source_config,
                ):
                    projected_config = projection.config
                    projected_id = (
                        canonical_sha256(projected_config.config)[:16]
                        if projected_config is not None
                        else None
                    )
                    if projected_config is not None:
                        terminal_manifest[projected_id] = {
                            "config": projected_config.config
                        }
                    outcome = projection.outcome
                    if (
                        outcome == "candidate"
                        and projected_config is not None
                        and flash_structural_leaf_from_config(projected_config.config)
                        != source_leaf
                    ):
                        outcome = "different_leaf"
                    requests.append(
                        {
                            "flat_index": projection.flat_index,
                            "key": projection.key,
                            "sequence_index": projection.sequence_index,
                            "from_value": projection.from_value,
                            "to_value": projection.to_value,
                            "outcome": outcome,
                            "config_id": projected_id,
                        }
                    )
                phase = provenance["trials"][0]["search_phase_metrics"]
                phase["terminal_coordinate_refinement"] = {
                    "initial_incumbent_config_id": source_id,
                    "radius": 2,
                    "projection_attempt_count": len(requests),
                    "config_manifest": terminal_manifest,
                    "rounds": [
                        {
                            "parent_config_ids": [source_id],
                            "parent_projections": [
                                {
                                    "parent_config_id": source_id,
                                    "coordinate_requests": requests,
                                }
                            ],
                        }
                    ],
                }
                self.assertEqual(
                    paired_worker.validate_bound_flash_normalization(
                        SimpleNamespace(config_spec=spec), provenance
                    )["canonical_compound_transfer_count"],
                    1,
                )

                strict = copy.deepcopy(provenance)
                strict["require_full_autotune"] = True
                strict["trials"][0].update(
                    {
                        "input_shapes": repr([(2, 32, 32768, 128)] * 3),
                        "dtypes": repr(["torch.bfloat16"] * 3),
                        "hardware": "NVIDIA B200",
                    }
                )
                with mock.patch(
                    "benchmarks.cute.compare_attention_backends."
                    "_validate_required_full_autotune_trials"
                ) as live_validator:
                    validation = paired_worker.validate_bound_flash_normalization(
                        SimpleNamespace(config_spec=spec), strict
                    )
                self.assertTrue(validation["live_full_autotune_validated"])
                live_validator.assert_called_once()
                self.assertIs(live_validator.call_args.kwargs["config_spec"], spec)

                wrong_terminal = copy.deepcopy(provenance)
                wrong_terminal["trials"][0]["search_phase_metrics"][
                    "terminal_coordinate_refinement"
                ]["rounds"][0]["parent_projections"][0]["coordinate_requests"][0][
                    "to_value"
                ] = "tampered"
                with self.assertRaisesRegex(
                    RuntimeError, "bound terminal coordinate projections"
                ):
                    paired_worker.validate_bound_flash_normalization(
                        SimpleNamespace(config_spec=spec), wrong_terminal
                    )

                wrong = copy.deepcopy(provenance)
                wrong["trials"][0]["search_phase_metrics"]["compound_transfers"][0][
                    "transfers"
                ][0]["projected_config"][
                    "cute_flash_softmax_disc" if not causal else "cute_flash_disc_pipe"
                ] = False if not causal else 5
                with self.assertRaisesRegex(
                    RuntimeError, "bound canonical compound projection"
                ):
                    paired_worker.validate_bound_flash_normalization(
                        SimpleNamespace(config_spec=spec), wrong
                    )

    def test_paired_worker_quarantines_all_isolated_timeout_aliases(self) -> None:
        source_hash = "a" * 64
        config_ids = ["1" * 16, "2" * 16]
        success = {
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "source_hash": source_hash,
        }
        timeout = {
            "attempt_perf": None,
            "selection_perf": None,
            "status": "timeout",
            "source_hash": source_hash,
        }
        phase = {
            "initial_config_ids": config_ids,
            "qualification_passes_completed": 1,
            "measurement_timeline": [
                {
                    "pass_index": 0,
                    "updates": [
                        {"config_id": config_id, **success} for config_id in config_ids
                    ],
                },
                {
                    "pass_index": 1,
                    "updates": [
                        {"config_id": config_id, **timeout} for config_id in config_ids
                    ],
                },
            ],
        }
        configs = {config_id: {} for config_id in config_ids}

        paired_worker.validate_measurement_timeline(
            Path("fixture.json"), phase, configs
        )
        self.assertEqual(
            paired_worker.isolated_rebenchmark_timeout_source_hashes(phase),
            {source_hash},
        )

        partial = copy.deepcopy(phase)
        partial["measurement_timeline"][1]["updates"].pop()
        with self.assertRaisesRegex(RuntimeError, "effective-source invalidation"):
            paired_worker.validate_measurement_timeline(
                Path("fixture.json"), partial, configs
            )

    def test_paired_worker_retains_nonpromoted_compound_leaves(self) -> None:
        lane = ("cute_flash_kv_stage", 2)
        next_id = 1

        def member(perf: float, *, covers_lane: bool = False) -> dict[str, Any]:
            nonlocal next_id
            result = {
                "config_id": f"{next_id:016x}",
                "selection_perf": perf,
                "pipeline_lanes": frozenset((lane,)) if covers_lane else frozenset(),
            }
            next_id += 1
            return result

        qualified_leaves: list[dict[str, Any]] = []
        for family, base_perf in (("family_a", 1.0), ("family_b", 1.1)):
            for softmax_disc, offset in ((False, 0.0), (True, 0.05)):
                qualified_leaves.append(
                    {
                        "family": family,
                        "compound_packet": None,
                        "softmax_disc": softmax_disc,
                        "members": [
                            member(base_perf + offset),
                            member(base_perf + offset + 0.3, covers_lane=True),
                        ],
                        "pipeline_lanes": [lane],
                    }
                )
        qualified_leaves.extend(
            (
                {
                    "family": "family_c",
                    "compound_packet": None,
                    "softmax_disc": False,
                    "members": [member(1.2), member(1.5, covers_lane=True)],
                    "pipeline_lanes": [lane],
                },
                {
                    "family": "family_c",
                    "compound_packet": "compound_1",
                    "softmax_disc": False,
                    "members": [member(1.25)],
                    "pipeline_lanes": [],
                },
                {
                    "family": "family_c",
                    "compound_packet": "compound_2",
                    "softmax_disc": False,
                    "members": [member(1.3)],
                    "pipeline_lanes": [],
                },
            )
        )
        options = {
            "retained_per_leaf": 2,
            "retained_family_cap": 2,
            "retained_family_limit": 2,
            "retained_family_slowdown_limit": 2.0,
            "starting_path_limit": 9,
        }

        expected = build_strict_manifest.expected_structural_retention(
            qualified_leaves, **options
        )
        actual = paired_worker.expected_structural_retention(
            qualified_leaves, **options
        )

        self.assertEqual(actual, expected)
        family_c = next(family for family in actual if family["family"] == "family_c")
        self.assertFalse(family_c["parent_promoted"])
        self.assertEqual(
            {path["compound_packet"] for path in family_c["starting_paths"]},
            {"compound_1", "compound_2"},
        )

    def test_paired_worker_validates_family_probe_evidence_and_promotion(self) -> None:
        source_hash = "a" * 64

        def config(family: str, packet: str = "1x1") -> dict[str, Any]:
            return {
                "cute_flash_pipeline_family": family,
                "cute_flash_exp2_packet": packet,
                "cute_flash_softmax_disc": False,
            }

        def state(perf: float) -> dict[str, Any]:
            return {
                "attempt_perf": perf,
                "selection_perf": perf,
                "status": "ok",
                "source_hash": source_hash,
            }

        families = [f"family_{index}" for index in range(5)]
        ordinary_ids = [f"{index + 1:016x}" for index in range(5)]
        compound_id = "0000000000000006"
        promoted_id = "0000000000000007"
        compound_probe_id = "0000000000000008"
        metadata = {
            config_id: config(family)
            for config_id, family in zip(ordinary_ids, families, strict=True)
        }
        metadata[compound_id] = config(families[-1], "deg2_16x6")
        metadata[promoted_id] = config(families[-1])
        metadata[compound_probe_id] = config(families[-1], "deg2_16x6")
        pre_states = {
            config_id: state(perf)
            for config_id, perf in zip(
                ordinary_ids, (1.0, 1.1, 1.2, 1.3, 3.0), strict=True
            )
        }
        pre_states[compound_id] = state(0.9)
        post_states = {
            **pre_states,
            ordinary_ids[3]: state(0.75),
            promoted_id: state(0.8),
            compound_probe_id: state(0.7),
        }
        first_probe_states = {
            **pre_states,
            promoted_id: state(3.0),
            compound_probe_id: state(1.7),
        }
        leaf_catalog = [
            paired_worker.structural_leaf(metadata[config_id])
            for config_id in ordinary_ids
        ] + [paired_worker.structural_leaf(metadata[compound_id])]
        assert all(leaf is not None for leaf in leaf_catalog)

        def probe_path(
            config_id: str,
            *,
            unrestricted: bool = False,
            candidate_id: str | None = None,
        ) -> dict[str, Any]:
            leaf = paired_worker.structural_leaf(metadata[config_id])
            assert leaf is not None
            candidate_ids = [] if candidate_id is None else [candidate_id]
            return {
                **leaf,
                "starting_config_id": config_id,
                "unrestricted": unrestricted,
                "rounds": [
                    {
                        "probe_generation": 1,
                        "measurement_pass_index": 1,
                        "candidate_ids": candidate_ids,
                        "results": [
                            {
                                "config_id": candidate_id,
                                **first_probe_states[candidate_id],
                                "measurement_pass_index": 1,
                            }
                            for candidate_id in candidate_ids
                        ],
                    },
                    {
                        "probe_generation": 2,
                        "measurement_pass_index": 2,
                        "candidate_ids": [],
                        "results": [],
                    },
                ],
            }

        phase = {
            "family_probe_path_limit": 7,
            "family_probe_required": True,
            "family_probe_complete": True,
            "family_probe_generations": 2,
            "family_probe_candidates_per_path": 20,
            "qualification_passes_completed": 2,
            "exact_space_exhausted": False,
            "compound_transfers": [{"qualified_transfer_config_ids": [compound_id]}],
            "retained_family_limit": 4,
            "retained_family_slowdown_limit": 2.0,
            "family_probe_paths": [
                *(probe_path(config_id) for config_id in ordinary_ids[:-1]),
                probe_path(ordinary_ids[-1], candidate_id=promoted_id),
                probe_path(compound_id, candidate_id=compound_probe_id),
                probe_path(compound_id, unrestricted=True),
            ],
            "retained_families": [
                {
                    "family": family,
                    "score": perf,
                    "score_compound_packet": None,
                    "score_softmax_disc": False,
                    "parent_promoted": True,
                }
                for family, perf in (
                    (families[3], 0.75),
                    (families[-1], 0.8),
                    (families[0], 1.0),
                    (families[1], 1.1),
                )
            ],
        }

        self.assertEqual(
            paired_worker._validate_family_probe_execution(
                Path("fixture.json"),
                phase,
                leaf_catalog,
                metadata,
                [pre_states, first_probe_states, post_states],
            ),
            {promoted_id, compound_probe_id},
        )

        repaired_pre_id = "0000000000000009"
        metadata[repaired_pre_id] = config(families[0])
        pre_with_failed = {
            **pre_states,
            repaired_pre_id: {
                "attempt_perf": None,
                "selection_perf": None,
                "status": "timeout",
                "source_hash": source_hash,
            },
        }
        first_with_repaired = {
            **first_probe_states,
            repaired_pre_id: state(0.1) | {"status": "deduplicated"},
        }
        post_with_repaired = {
            **post_states,
            repaired_pre_id: state(0.1) | {"status": "deduplicated"},
        }
        self.assertEqual(
            paired_worker._validate_family_probe_execution(
                Path("fixture.json"),
                phase,
                leaf_catalog,
                metadata,
                [pre_with_failed, first_with_repaired, post_with_repaired],
            ),
            {promoted_id, compound_probe_id},
        )

        wrong_promotion = copy.deepcopy(phase)
        wrong_promotion["retained_families"][-1]["parent_promoted"] = False
        with self.assertRaisesRegex(RuntimeError, "promotion ranking"):
            paired_worker._validate_family_probe_execution(
                Path("fixture.json"),
                wrong_promotion,
                leaf_catalog,
                metadata,
                [pre_states, first_probe_states, post_states],
            )

        invalidated = copy.deepcopy(post_states)
        invalidated[compound_probe_id] = {
            "attempt_perf": None,
            "selection_perf": None,
            "status": "timeout",
            "source_hash": source_hash,
        }
        self.assertEqual(
            paired_worker._validate_family_probe_execution(
                Path("fixture.json"),
                phase,
                leaf_catalog,
                metadata,
                [pre_states, first_probe_states, invalidated],
            ),
            {promoted_id},
        )

        repaired_phase = copy.deepcopy(phase)
        repaired_result = repaired_phase["family_probe_paths"][-3]["rounds"][0][
            "results"
        ][0]
        repaired_result.update(
            attempt_perf=None,
            selection_perf=None,
            status="timeout",
        )
        repaired_first_states = copy.deepcopy(first_probe_states)
        repaired_first_states[promoted_id] = {
            "attempt_perf": None,
            "selection_perf": None,
            "status": "timeout",
            "source_hash": source_hash,
        }
        repaired_final_states = copy.deepcopy(post_states)
        repaired_final_states[promoted_id]["status"] = "deduplicated"
        self.assertEqual(
            paired_worker._validate_family_probe_execution(
                Path("fixture.json"),
                repaired_phase,
                leaf_catalog,
                metadata,
                [pre_states, repaired_first_states, repaired_final_states],
            ),
            {promoted_id, compound_probe_id},
        )

    @unittest.skip(
        "the retired v16 standalone replay is superseded by the live v22 validator"
    )
    def test_paired_worker_replays_v16_lane_diverse_retention(self) -> None:
        lane_values = (2, 3)
        configs: dict[str, dict[str, Any]] = {}
        attempts: dict[str, dict[str, Any]] = {}
        qualified = []
        ids_by_lane: dict[int, list[str]] = {value: [] for value in lane_values}
        for index, (lane_value, perf) in enumerate(
            ((2, 1.0), (2, 1.1), (3, 1.2), (3, 1.3))
        ):
            config = {
                "cute_flash_pipeline_family": "fa4_clc",
                "cute_flash_exp2_packet": "1x1",
                "cute_flash_kv_stage": lane_value,
                "cute_flash_s_stage": 2,
                "cute_flash_clc_heads_per_batch": 1 if index % 2 == 0 else 2,
                "fixture": index,
            }
            config_id = canonical_sha256(config)[:16]
            configs[config_id] = config
            ids_by_lane[lane_value].append(config_id)
            attempts[config_id] = {
                "generation": 0,
                "status": "ok",
                "perf_ms": perf,
            }
            qualified.append(
                {
                    "config_id": config_id,
                    "status": "ok",
                    "attempt_perf": perf,
                    "selection_perf": perf,
                    "pipeline_lanes": [
                        {"key": "cute_flash_kv_stage", "value": lane_value}
                    ],
                }
            )
        retained_ids = [ids_by_lane[2][0], ids_by_lane[3][0]]
        phase = {
            "starting_path_limit": 5,
            "initial_config_ids": list(configs),
            "config_manifest": {
                config_id: {"config": config} for config_id, config in configs.items()
            },
            "initial_results": [
                {
                    "config_id": config_id,
                    "family": "fa4_clc",
                    "compound_packet": None,
                    "attempt_perf": attempts[config_id]["perf_ms"],
                    "selection_perf": attempts[config_id]["perf_ms"],
                    "status": "ok",
                    "pipeline_lanes": [
                        {
                            "key": "cute_flash_kv_stage",
                            "value": config["cute_flash_kv_stage"],
                        }
                    ],
                }
                for config_id, config in configs.items()
            ],
            "exact_space_enumerated": True,
            "exact_space_exhausted": True,
            "exact_space_raw_budget": len(configs),
            "exact_space_config_ids": list(configs),
            "leaf_results": [
                {
                    "family": "fa4_clc",
                    "compound_packet": None,
                    "initial_config_ids": list(configs),
                    "space_exhausted": True,
                    "space_config_count": len(configs),
                    "ordinary_search_required": False,
                    "rounds": [
                        {
                            "candidate_config_ids": [],
                            "neighbor_generation_limit": 0,
                            "ordinary_neighbor_generation_limit": 0,
                        },
                        {
                            "candidate_config_ids": [],
                            "neighbor_generation_limit": 0,
                            "ordinary_neighbor_generation_limit": 0,
                        },
                    ],
                    "pipeline_lanes": [
                        {
                            "key": "cute_flash_kv_stage",
                            "value": value,
                            "initial_config_ids": ids_by_lane[value],
                            "space_exhausted": True,
                            "space_config_count": len(ids_by_lane[value]),
                            "conditional_required": False,
                            "rounds": [
                                {
                                    "candidate_config_ids": [ids_by_lane[value][0]],
                                    "neighbor_generation_limit": 0,
                                },
                                {
                                    "candidate_config_ids": [],
                                    "neighbor_generation_limit": 0,
                                },
                            ],
                            "witness_attempted": True,
                            "witness_config_id": ids_by_lane[value][0],
                            "witness_succeeded": True,
                            "conditional_candidate_ids": [],
                            "successful_conditional_candidate_ids": [],
                            "repair_candidate_ids": [],
                            "successful_repair_candidate_ids": [],
                            "repair_parent_decisions": [],
                            "complete": True,
                        }
                        for value in lane_values
                    ],
                    "qualified_results": qualified,
                    "retained_config_ids": retained_ids,
                    "complete": True,
                }
            ],
            "retained_families": [
                {
                    "family": "fa4_clc",
                    "score": 1.0,
                    "score_compound_packet": None,
                    "parent_promoted": True,
                    "starting_paths": [
                        {
                            "family": "fa4_clc",
                            "compound_packet": None,
                            "config_id": retained_ids[0],
                            "unrestricted": True,
                            "pipeline_lane": None,
                        },
                        {
                            "family": "fa4_clc",
                            "compound_packet": None,
                            "config_id": retained_ids[1],
                            "unrestricted": False,
                            "pipeline_lane": {
                                "key": "cute_flash_kv_stage",
                                "value": 3,
                            },
                        },
                    ],
                }
            ],
            "retained_path_count": 2,
        }
        provenance = {
            "trials": [],
            "flash_fragment_default_sha256": "b" * 64,
            "autotune_initial_population_size": len(configs),
            "flash_exact_effective_search_space_size": len(configs),
            "flash_exact_effective_search_space_config_ids": list(configs),
            "flash_exact_effective_search_space_sha256": hashlib.sha256(
                json.dumps(list(configs), separators=(",", ":")).encode()
            ).hexdigest(),
            "flash_structural_coverage_design": [
                {"config": config} for config in configs.values()
            ],
            "flash_structural_coverage_active_values": [
                {"key": "cute_flash_clc_heads_per_batch", "value": 1}
            ],
            "flash_structural_injected_design_count": len(configs),
            "flash_structural_leaf_catalog": [
                {"family": "fa4_clc", "compound_packet": None}
            ],
            "flash_pipeline_lane_catalog": [
                {
                    "family": "fa4_clc",
                    "compound_packet": None,
                    "pipeline_lanes": [
                        {"key": "cute_flash_kv_stage", "value": value}
                        for value in lane_values
                    ],
                }
            ],
            "flash_structural_qualification_rounds": 2,
            "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": 4,
            "flash_structural_retained_candidates_per_leaf": 2,
            "flash_structural_retained_family_cap": 4,
            "flash_structural_retained_family_limit": 4,
            "flash_structural_retained_family_slowdown_limit": 2.0,
            "flash_structural_starting_path_limit": 5,
            "flash_structural_unrestricted_path_exhausts_generation_budget": True,
        }
        clc_catalog = [
            {
                "family": "fa4_clc",
                "compound_packet": None,
                "legal_values": [1, 2],
                "search_values": [1, 2],
                "anchor_values": [1],
                "refinement_values": [2],
                "planned_values": [1, 2],
                "witness_config_ids": {
                    "1": ids_by_lane[2][0],
                    "2": ids_by_lane[2][1],
                },
            }
        ]
        add_clc_lane_catalog(provenance, clc_catalog)
        clc_witness_snapshots = [
            {
                "value": value,
                "config_id": config_id,
                "attempt_perf": attempts[config_id]["perf_ms"],
                "selection_perf": attempts[config_id]["perf_ms"],
                "status": "ok",
            }
            for value, config_id in (
                (1, ids_by_lane[2][0]),
                (2, ids_by_lane[2][1]),
            )
        ]
        phase.update(
            {
                "phase": "cute_flash_structural_qualification_v16",
                "cute_flash_lane_policy_version": 6,
                "qualification_failure_retries": 1,
                "completed": True,
                "unrestricted_path_exhausts_generation_budget": True,
                "initial_config_count": len(configs),
                "leaf_count": 1,
                "ordinary_leaf_count": 1,
                "compound_leaf_count": 0,
                "pipeline_qualification_keys": list(
                    paired_worker.FLASH_PIPELINE_QUALIFICATION_KEYS
                ),
                "qualification_rounds": 2,
                "qualification_rounds_started": 3,
                "qualification_rounds_completed": 3,
                "qualification_passes_planned": 3,
                "qualification_passes_started": 3,
                "qualification_passes_completed": 3,
                "budget_exhausted": False,
                "pipeline_candidate_limit_per_leaf_per_round": 4,
                "conditional_candidates_per_pipeline_lane": 1,
                "neighbor_generation_limit_per_leaf_per_round": 200,
                "candidate_count": 0,
                "leaves_with_candidates": 0,
                "retained_candidates_per_leaf": 2,
                "retained_family_cap": 4,
                "retained_family_limit": 4,
                "retained_family_slowdown_limit": 2.0,
                "clc_families": [
                    {
                        "family": "fa4_clc",
                        "space_exhausted": True,
                        "legal_values": [1, 2],
                        "search_values": [1, 2],
                        "anchor_values": [1],
                        "refinement_values": [2],
                        "planned_values": [1, 2],
                        "attempted_values": [1, 2],
                        "witness_config_ids": {
                            "1": ids_by_lane[2][0],
                            "2": ids_by_lane[2][1],
                        },
                        "witness_repair_candidate_ids": {},
                        "witness_repair_parent_decisions": [],
                        "value_space_exhausted": {"1": True, "2": True},
                        "witness_candidate_results": copy.deepcopy(
                            clc_witness_snapshots
                        ),
                        "witness_selection_results": copy.deepcopy(
                            clc_witness_snapshots
                        ),
                        "selected_values": [1, 2],
                        "selected_config_ids": [
                            ids_by_lane[2][0],
                            ids_by_lane[2][1],
                        ],
                        "conditional_values": [],
                        "conditional_parent_decisions": [],
                        "conditional_repair_candidate_ids": {},
                        "conditional_repair_parent_decisions": [],
                        "retained_values": [1, 2],
                        "retained_config_ids": [
                            ids_by_lane[2][0],
                            ids_by_lane[2][1],
                        ],
                        "retained_value_decisions": [
                            {
                                "value": snapshot["value"],
                                "candidate_results": [
                                    {
                                        key: snapshot[key]
                                        for key in (
                                            "config_id",
                                            "attempt_perf",
                                            "selection_perf",
                                            "status",
                                        )
                                    }
                                ],
                                "selected_config_id": snapshot["config_id"],
                            }
                            for snapshot in clc_witness_snapshots
                        ],
                        "retained_ranking_results": copy.deepcopy(
                            clc_witness_snapshots
                        ),
                        "conditional_candidate_ids": {},
                        "combination_required": False,
                        "depth_selection": {
                            "candidate_results": [],
                            "selected_representatives": [],
                        },
                        "combination_candidate_ids": [],
                        "combination_depth_config_ids": [],
                        "combination_divisor_values": [],
                        "combination_cells": [],
                        "combination_projection_complete": True,
                        "successful_combination_depth_config_ids": [],
                        "successful_combination_divisor_values": [],
                        "combination_row_coverage_complete": True,
                        "combination_column_coverage_complete": True,
                        "combination_failure_statuses_allowed": True,
                        "complete": True,
                    }
                ],
                "compound_transfers": [],
            }
        )
        add_phase_config_identity(provenance, phase, configs)
        trial = {"search_phase_metrics": phase}
        add_flash_normalization_context(
            provenance,
            trial,
            shape=(2, 32, 65536, 64),
            dtype="torch.float16",
            causal=False,
        )
        provenance["trials"] = [trial]

        build_strict_manifest.validate_phase_config_identity(
            Path("fixture.json"),
            provenance,
            phase,
            {
                build_strict_manifest.canonical_json(
                    {"family": "fa4_clc", "compound_packet": None}
                ): [("cute_flash_kv_stage", value) for value in lane_values]
            },
        )
        paired_worker._validate_structural_qualification_phase(
            Path("fixture.json"), provenance
        )

        missing_snapshot_pass = copy.deepcopy(provenance)
        missing_snapshot_pass["trials"][0]["search_phase_metrics"]["clc_families"][0][
            "witness_selection_results"
        ][0].pop("measurement_pass_index")
        with self.assertRaisesRegex(RuntimeError, "immutable CLC witness decision"):
            paired_worker._validate_structural_qualification_phase(
                Path("fixture.json"), missing_snapshot_pass
            )

        narrowed_clc_search = copy.deepcopy(phase)
        narrowed_clc_search["clc_families"][0]["search_values"].pop()
        narrowed_trial = copy.deepcopy(trial)
        narrowed_trial["search_phase_metrics"] = narrowed_clc_search
        with self.assertRaisesRegex(
            RuntimeError, "CLC catalog search_values|full search reachability"
        ):
            build_strict_manifest.validate_structural_qualification_phase(
                Path("fixture.json"),
                provenance,
                narrowed_trial,
            )
        narrowed_provenance = copy.deepcopy(provenance)
        narrowed_provenance["trials"][0]["search_phase_metrics"] = narrowed_clc_search
        with self.assertRaisesRegex(
            RuntimeError, "CLC catalog search_values|invalid CLC value selection"
        ):
            paired_worker._validate_structural_qualification_phase(
                Path("fixture.json"), narrowed_provenance
            )

        missing_leaf_child = copy.deepcopy(phase)
        conditional_id = "f" * 16
        lane = missing_leaf_child["leaf_results"][0]["pipeline_lanes"][0]
        lane["space_exhausted"] = False
        lane["conditional_required"] = True
        lane["conditional_candidate_ids"] = [conditional_id]
        lane["successful_conditional_candidate_ids"] = [conditional_id]
        lane["rounds"][1]["candidate_config_ids"] = [conditional_id]
        provenance["trials"] = [{**trial, "search_phase_metrics": missing_leaf_child}]
        with self.assertRaisesRegex(RuntimeError, "leaf/lane pass candidate mismatch"):
            paired_worker._validate_structural_qualification_phase(
                Path("fixture.json"), provenance
            )
        provenance["trials"] = [{**trial, "search_phase_metrics": phase}]

        paired_worker._reconcile_structural_qualification_phase(
            Path("fixture.json"), provenance, phase, attempts, configs
        )

        false_membership = copy.deepcopy(phase)
        false_membership["leaf_results"][0]["qualified_results"][0]["pipeline_lanes"][
            0
        ]["value"] = 3
        with self.assertRaisesRegex(RuntimeError, "actual.*lane membership"):
            paired_worker._reconcile_structural_qualification_phase(
                Path("fixture.json"),
                provenance,
                false_membership,
                attempts,
                configs,
            )

        wrong_retention = copy.deepcopy(phase)
        wrong_retention["leaf_results"][0]["retained_config_ids"] = ids_by_lane[2]
        with self.assertRaisesRegex(RuntimeError, "retained candidates"):
            paired_worker._reconcile_structural_qualification_phase(
                Path("fixture.json"), provenance, wrong_retention, attempts, configs
            )

        wrong_path = copy.deepcopy(phase)
        wrong_path["retained_families"][0]["starting_paths"][1]["pipeline_lane"][
            "value"
        ] = 2
        with self.assertRaisesRegex(RuntimeError, "retained path violates"):
            paired_worker._reconcile_structural_qualification_phase(
                Path("fixture.json"), provenance, wrong_path, attempts, configs
            )

        wrong_budget_provenance = copy.deepcopy(provenance)
        wrong_budget_provenance["trials"][0]["search_phase_metrics"][
            "neighbor_generation_limit_per_leaf_per_round"
        ] = 300
        with self.assertRaisesRegex(RuntimeError, "neighbor generation limit"):
            paired_worker._validate_structural_qualification_phase(
                Path("fixture.json"), wrong_budget_provenance
            )

    def setUp(self) -> None:
        checkout = {
            "runtime_checkout": "/fixture/checkout",
            "runtime_git_head": git_output("rev-parse", "HEAD"),
            "runtime_tracked_clean": True,
            "runtime_untracked_clean": True,
            "runtime_allowed_untracked_files": [],
            "runtime_ignored_clean": True,
            "runtime_allowed_ignored_file_count": 0,
            "runtime_allowed_ignored_files_sha256": canonical_sha256([]),
        }
        patcher = mock.patch("paired_worker.validate_checkout", return_value=checkout)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_artifacts(self, root: Path) -> dict[str, Path]:
        paths = {}
        for index, case in enumerate(CASES.values()):
            path = root / case.name / "result.json"
            path.parent.mkdir(parents=True)
            payload = make_payload(case, legacy_source_key=index == 0)
            path.write_text(json.dumps(payload) + "\n")
            provenance = payload["helion_overrides"]["autotune_provenance"]
            configs = [
                item["config"]
                for item in provenance["flash_structural_coverage_design"]
            ]
            configs.extend(
                {
                    "cute_flash_pipeline_family": "fa4",
                    "cute_flash_exp2_packet": "1x1",
                    "fixture_initial_population_config": candidate,
                }
                for candidate in range(
                    provenance["autotune_initial_population_size"] - len(configs)
                )
            )
            qualification_candidate = {
                **configs[1],
                "cute_flash_wait_hint": -2,
            }
            config_map = {canonical_sha256(config)[:16]: config for config in configs}
            qualification_candidate_id = canonical_sha256(qualification_candidate)[:16]
            config_map[qualification_candidate_id] = qualification_candidate
            settings = {
                "backend": "cute",
                "force_autotune": False,
                "effective_cache_read_bypass": True,
                "static_shapes": True,
                "autotune_log_details": True,
                "autotune_compile_timeout": 60,
                "autotune_benchmark_subprocess": True,
                "autotune_benchmark_timeout": 60,
                "autotune_random_seed": provenance["trials"][0]["random_seed"],
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
            metadata = {
                "kernel_source": "fixture attention kernel",
                "input_shapes": repr([(2, 32, case.seq_len, 64)] * 3),
                "dtypes": repr(["torch.float16"] * 3),
                "hardware": "NVIDIA B200",
                "settings": settings,
                "configs": config_map,
            }
            run_id = paired_worker._metadata_run_id(
                metadata, path.with_name("autotune.meta.jsonl")
            )
            metadata["run_id"] = run_id
            path.with_name("autotune.meta.jsonl").write_text(
                json.dumps(metadata) + "\n"
            )
            ledger_output = io.StringIO(newline="")
            source_writer = csv.DictWriter(
                ledger_output, fieldnames=SOURCE_LEDGER_FIELDS, lineterminator="\n"
            )
            source_writer.writeheader()
            autotune_output = io.StringIO(newline="")
            autotune_writer = csv.DictWriter(
                autotune_output,
                fieldnames=paired_worker.AUTOTUNE_CSV_FIELDS,
                lineterminator="\n",
            )
            autotune_writer.writeheader()
            selected_config_id = canonical_sha256(provenance["selected_config"])[:16]
            phase_attempts = {}
            for source_index, (config_id, config) in enumerate(config_map.items()):
                source_hash = (
                    provenance.get("selected_source_sha256")
                    or provenance.get("selected_source_hash")
                    if config_id == selected_config_id
                    else canonical_sha256(
                        {
                            "case": case.name,
                            "prefix_source": source_index,
                            "config": config,
                        }
                    )
                )
                phase_attempts[config_id] = {
                    "status": "ok",
                    "perf_ms": 1.0,
                    "source_hash": source_hash,
                }
                for offset, status in enumerate(("started", "ok")):
                    common = {
                        "run_id": run_id,
                        "timestamp_s": f"{source_index + offset / 10:.1f}",
                        "config_id": config_id,
                        "generation": (
                            "1" if config_id == qualification_candidate_id else "0"
                        ),
                        "status": status,
                    }
                    source_writer.writerow({**common, "source_hash": source_hash})
                    config_repr = (
                        "Config("
                        + ", ".join(
                            f"{key}={value!r}" for key, value in sorted(config.items())
                        )
                        + ")"
                    )
                    autotune_writer.writerow(
                        {
                            **common,
                            "perf_ms": "1.0" if status == "ok" else "",
                            "compile_time_s": "0.1" if status == "ok" else "",
                            "config": config_repr,
                        }
                    )
            path.with_name("autotune.sources.csv").write_text(ledger_output.getvalue())
            path.with_name("autotune.csv").write_text(autotune_output.getvalue())
            provenance["trials"][0].update(
                {
                    "num_configs_tested": len(config_map),
                    "num_compile_failures": 0,
                    "num_worker_failures": 0,
                    "num_isolated_rebenchmark_timeouts": 0,
                    "num_accuracy_failures": 0,
                    "num_successful_candidate_measurements": len(config_map),
                    "num_unique_sources": len(config_map),
                    "num_source_deduplications": 0,
                    "num_generations": 1,
                }
            )
            provenance["autotune_lfbo_max_generations"] = 1
            initial_config_ids = [canonical_sha256(config)[:16] for config in configs]
            leaf = {
                "family": "fa4",
                "compound_packet": None,
                "softmax_disc": False,
            }
            leaf_initial_ids = [
                config_id
                for config_id in initial_config_ids
                if build_strict_manifest.structural_leaf(config_map[config_id]) == leaf
            ]
            provenance["trials"][0]["search_phase_metrics"] = {
                "phase": "cute_flash_structural_qualification_v22",
                "cute_flash_lane_policy_version": 11,
                "qualification_failure_retries": 1,
                "completed": True,
                "unrestricted_path_exhausts_generation_budget": True,
                "initial_config_count": len(initial_config_ids),
                "initial_config_ids": initial_config_ids,
                "exact_space_enumerated": False,
                "exact_space_exhausted": False,
                "exact_space_raw_budget": 100,
                "exact_space_config_ids": [],
                "leaf_count": 1,
                "ordinary_leaf_count": 1,
                "compound_leaf_count": 0,
                "pipeline_qualification_keys": list(
                    paired_worker.FLASH_PIPELINE_QUALIFICATION_KEYS
                ),
                "leaf_results": [
                    {
                        **leaf,
                        "initial_config_ids": leaf_initial_ids,
                        "space_exhausted": False,
                        "space_config_count": None,
                        "ordinary_search_required": True,
                        "rounds": [
                            {
                                "candidate_config_ids": [qualification_candidate_id],
                                "neighbor_generation_limit": 200,
                                "ordinary_neighbor_generation_limit": 200,
                            },
                            {
                                "candidate_config_ids": [],
                                "neighbor_generation_limit": 200,
                                "ordinary_neighbor_generation_limit": 200,
                            },
                        ],
                        "pipeline_lanes": [],
                        "qualified_results": [
                            {
                                "config_id": config_id,
                                "attempt_perf": 1.0,
                                "selection_perf": 1.0,
                                "status": "ok",
                                "pipeline_lanes": [],
                            }
                            for config_id in [
                                *leaf_initial_ids,
                                qualification_candidate_id,
                            ]
                        ],
                        "retained_config_ids": sorted(
                            [*leaf_initial_ids, qualification_candidate_id]
                        )[:2],
                        "complete": True,
                    }
                ],
                "qualification_rounds": 2,
                "qualification_rounds_started": 2,
                "qualification_rounds_completed": 2,
                "qualification_passes_planned": 2,
                "qualification_passes_started": 2,
                "qualification_passes_completed": 2,
                "budget_exhausted": False,
                "pipeline_candidate_limit_per_leaf_per_round": 4,
                "conditional_candidates_per_pipeline_lane": 1,
                "neighbor_generation_limit_per_leaf_per_round": 200,
                "candidate_count": 1,
                "leaves_with_candidates": 1,
                "retained_candidates_per_leaf": 2,
                "retained_family_cap": None,
                "retained_family_limit": 1,
                "retained_family_slowdown_limit": 2.0,
                "clc_families": [],
                "compound_transfers": [],
                "starting_path_limit": provenance[
                    "flash_structural_starting_path_limit"
                ],
                "retained_families": [
                    {
                        "family": "fa4",
                        "score": 1.0,
                        "score_compound_packet": None,
                        "parent_promoted": True,
                        "starting_paths": [
                            {
                                **leaf,
                                "config_id": config_id,
                                "unrestricted": index == 0,
                                "pipeline_lane": None,
                            }
                            for index, config_id in enumerate(
                                sorted([*leaf_initial_ids, qualification_candidate_id])[
                                    :2
                                ]
                            )
                        ],
                    }
                ],
                "retained_path_count": 2,
            }
            add_phase_config_identity(
                provenance,
                provenance["trials"][0]["search_phase_metrics"],
                config_map,
                attempts=phase_attempts,
            )
            path.write_text(json.dumps(payload) + "\n")
            paths[case.name] = path
        return paths

    def test_accepts_complete_strict_result_set_and_both_source_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root)
            validated = validate_artifact_set(root)
        self.assertEqual(set(validated), set(CASES.values()))
        self.assertTrue(
            all(item["strict_full_autotune_validated"] for item in validated.values())
        )
        execution = next(iter(validated.values()))["structural_design_execution"]
        self.assertEqual(execution["prefix_count"], 2)
        self.assertEqual(execution["attempted_count"], 2)
        self.assertEqual(execution["successful_count"], 2)
        summary = next(iter(validated.values()))["matching_trial_summaries"][0]
        self.assertEqual(summary["num_isolated_rebenchmark_timeouts"], 0)

    def test_rejects_mismatched_live_derived_path_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_artifacts(root)["dense_32768"]
            payload = json.loads(result_path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            phase = provenance["trials"][0]["search_phase_metrics"]
            phase["starting_path_limit"] = (
                provenance["flash_structural_starting_path_limit"] + 1
            )
            result_path.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(RuntimeError, "live-derived"):
                validate_artifact_set(root)

    def test_rejects_missing_or_false_unrestricted_path_exhaustion(self) -> None:
        provenance_key = "flash_structural_unrestricted_path_exhausts_generation_budget"
        phase_key = "unrestricted_path_exhausts_generation_budget"
        for location, key in (("provenance", provenance_key), ("phase", phase_key)):
            for mutation in ("missing", "false"):
                with (
                    self.subTest(location=location, mutation=mutation),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    root = Path(directory)
                    result_path = self.write_artifacts(root)["dense_32768"]
                    payload = json.loads(result_path.read_text())
                    provenance = payload["helion_overrides"]["autotune_provenance"]
                    target = (
                        provenance
                        if location == "provenance"
                        else provenance["trials"][0]["search_phase_metrics"]
                    )
                    if mutation == "missing":
                        target.pop(key)
                    else:
                        target[key] = False
                    result_path.write_text(json.dumps(payload) + "\n")
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "unrestricted.path.*exhaust|unrestricted_path_exhausts",
                    ):
                        validate_artifact_set(root)

    def test_rejects_unrestricted_path_stopping_before_lfbo_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_artifacts(root)["dense_32768"]
            payload = json.loads(result_path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            provenance["autotune_lfbo_max_generations"] += 1
            result_path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(
                RuntimeError, "unrestricted path generation budget"
            ):
                validate_artifact_set(root)

    def test_rejects_incomplete_structural_qualification_budget(self) -> None:
        for key, value, match in (
            (
                "qualification_rounds_started",
                1,
                "qualification_rounds_started",
            ),
            (
                "qualification_rounds_completed",
                1,
                "qualification_rounds_completed",
            ),
            ("budget_exhausted", True, "budget exhaustion"),
        ):
            with (
                self.subTest(key=key),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                result_path = self.write_artifacts(root)["dense_32768"]
                payload = json.loads(result_path.read_text())
                phase = payload["helion_overrides"]["autotune_provenance"]["trials"][0][
                    "search_phase_metrics"
                ]
                phase[key] = value
                result_path.write_text(json.dumps(payload) + "\n")
                with self.assertRaisesRegex(RuntimeError, match):
                    validate_artifact_set(root)

    def test_rejects_structural_prefix_missing_from_generation_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_artifacts(root)
            result_path = paths["dense_32768"]
            payload = json.loads(result_path.read_text())
            config = payload["helion_overrides"]["autotune_provenance"][
                "flash_structural_coverage_design"
            ][-1]["config"]
            config_id = canonical_sha256(config)[:16]
            ledger_path = result_path.with_name("autotune.sources.csv")
            rows = list(csv.DictReader(io.StringIO(ledger_path.read_text())))
            for row in rows:
                if row["config_id"] == config_id:
                    row["generation"] = "1"
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output, fieldnames=SOURCE_LEDGER_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            ledger_path.write_text(output.getvalue())
            autotune_path = result_path.with_name("autotune.csv")
            autotune_rows = list(csv.DictReader(io.StringIO(autotune_path.read_text())))
            for row in autotune_rows:
                if row["config_id"] == config_id:
                    row["generation"] = "1"
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output,
                fieldnames=paired_worker.AUTOTUNE_CSV_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(autotune_rows)
            autotune_path.write_text(output.getvalue())

            with self.assertRaisesRegex(RuntimeError, "first ledger generation"):
                validate_artifact_set(root)

    def test_rejects_reordered_generation_zero_population(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_artifacts(root)["dense_32768"]
            payload = json.loads(result_path.read_text())
            phase = payload["helion_overrides"]["autotune_provenance"]["trials"][0][
                "search_phase_metrics"
            ]
            reordered_ids = list(reversed(phase["initial_config_ids"]))
            rank = {config_id: index for index, config_id in enumerate(reordered_ids)}
            phase["initial_config_ids"] = reordered_ids
            phase["initial_results"].sort(key=lambda result: rank[result["config_id"]])
            for leaf in phase["leaf_results"]:
                leaf["initial_config_ids"].sort(key=rank.__getitem__)
                for lane in leaf["pipeline_lanes"]:
                    lane["initial_config_ids"].sort(key=rank.__getitem__)
            result_path.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(
                RuntimeError, "generation-zero initial population order"
            ):
                validate_artifact_set(root)

    def test_rejects_qualification_status_not_backed_by_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_artifacts(root)["dense_32768"]
            payload = json.loads(result_path.read_text())
            phase = payload["helion_overrides"]["autotune_provenance"]["trials"][0][
                "search_phase_metrics"
            ]
            phase["leaf_results"][0]["qualified_results"][0]["status"] = "deduplicated"
            result_path.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(
                RuntimeError, "sidecar status|qualified measurement snapshot"
            ):
                validate_artifact_set(root)

    def test_paired_sidecars_require_exact_started_count_and_allow_precompile_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_artifacts(root)["dense_32768"]
            payload = json.loads(result_path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            trial = provenance["trials"][0]
            selected_id = canonical_sha256(provenance["selected_config"])[:16]
            ledger_path = result_path.with_name("autotune.sources.csv")
            source_rows = list(csv.DictReader(io.StringIO(ledger_path.read_text())))
            failed_id = next(
                row["config_id"]
                for row in source_rows
                if row["status"] == "ok" and row["config_id"] != selected_id
            )
            source_rows = [
                row
                for row in source_rows
                if not (row["config_id"] == failed_id and row["status"] == "started")
            ]
            next(
                row
                for row in source_rows
                if row["config_id"] == failed_id and row["status"] == "ok"
            )["status"] = "error"
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output, fieldnames=SOURCE_LEDGER_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(source_rows)
            ledger_path.write_text(output.getvalue())

            autotune_path = result_path.with_name("autotune.csv")
            autotune_rows = list(csv.DictReader(io.StringIO(autotune_path.read_text())))
            autotune_rows = [
                row
                for row in autotune_rows
                if not (row["config_id"] == failed_id and row["status"] == "started")
            ]
            failed_autotune = next(
                row
                for row in autotune_rows
                if row["config_id"] == failed_id and row["status"] == "ok"
            )
            failed_autotune["status"] = "error"
            failed_autotune["perf_ms"] = ""
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output,
                fieldnames=paired_worker.AUTOTUNE_CSV_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(autotune_rows)
            autotune_path.write_text(output.getvalue())

            trial["num_configs_tested"] -= 1
            trial["num_successful_candidate_measurements"] -= 1
            trial["num_compile_failures"] += 1
            selected_source = provenance.get(
                "selected_source_sha256", provenance["selected_source_hash"]
            )
            paired_worker._validate_autotune_sidecars(
                result_path,
                provenance,
                CASES[("dense", 32768)],
                provenance["selected_config"],
                selected_source,
            )

            trial["num_configs_tested"] -= 1
            with self.assertRaisesRegex(RuntimeError, "tested count"):
                paired_worker._validate_autotune_sidecars(
                    result_path,
                    provenance,
                    CASES[("dense", 32768)],
                    provenance["selected_config"],
                    selected_source,
                )

    def test_accepts_failed_attempt_repaired_by_source_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_artifacts(root)["dense_32768"]
            payload = json.loads(result_path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            trial = provenance["trials"][0]
            phase = trial["search_phase_metrics"]
            selected_id = canonical_sha256(provenance["selected_config"])[:16]
            retained_ids = set(phase["leaf_results"][0]["retained_config_ids"])
            qualified = next(
                item
                for item in phase["leaf_results"][0]["qualified_results"]
                if item["config_id"] != selected_id
                and item["config_id"] not in retained_ids
            )
            repaired_id = qualified["config_id"]
            qualified["attempt_perf"] = None
            qualified["selection_perf"] = None
            qualified["status"] = "error"
            trial["num_compile_failures"] += 1
            trial["num_successful_candidate_measurements"] -= 1
            trial["num_unique_sources"] -= 1
            trial["num_source_deduplications"] += 1
            post_qualification_configs = [
                {
                    "cute_flash_pipeline_family": "fa4",
                    "cute_flash_exp2_packet": "1x1",
                    "fixture_post_qualification_config": generation,
                }
                for generation in (3, 4)
            ]
            trial["num_configs_tested"] += len(post_qualification_configs)
            trial["num_successful_candidate_measurements"] += len(
                post_qualification_configs
            )
            trial["num_unique_sources"] += len(post_qualification_configs)
            trial["num_generations"] = 4
            provenance["autotune_lfbo_max_generations"] = 4
            result_path.write_text(json.dumps(payload) + "\n")

            metadata_path = result_path.with_name("autotune.meta.jsonl")
            metadata = json.loads(metadata_path.read_text())
            for config in post_qualification_configs:
                metadata["configs"][canonical_sha256(config)[:16]] = config
            metadata_path.write_text(json.dumps(metadata) + "\n")
            add_phase_config_identity(provenance, phase, metadata["configs"])
            result_path.write_text(json.dumps(payload) + "\n")

            ledger_path = result_path.with_name("autotune.sources.csv")
            source_rows = list(csv.DictReader(io.StringIO(ledger_path.read_text())))
            terminal = next(
                row
                for row in source_rows
                if row["config_id"] == repaired_id and row["status"] == "ok"
            )
            repair_source = canonical_sha256(
                {
                    "config": post_qualification_configs[0],
                    "phase": "post_qualification",
                }
            )
            for row in source_rows:
                if row["config_id"] == repaired_id:
                    row["source_hash"] = repair_source
            terminal["status"] = "error"
            post_source_rows = []
            for generation, config in zip(
                (3, 4), post_qualification_configs, strict=True
            ):
                common = {
                    "run_id": terminal["run_id"],
                    "config_id": canonical_sha256(config)[:16],
                    "generation": str(generation),
                    "source_hash": canonical_sha256(
                        {"config": config, "phase": "post_qualification"}
                    ),
                }
                post_source_rows.extend(
                    (
                        {
                            **common,
                            "timestamp_s": f"99{generation}.0",
                            "status": "started",
                        },
                        {
                            **common,
                            "timestamp_s": f"99{generation}.1",
                            "status": "ok",
                        },
                    )
                )
            source_rows.extend(post_source_rows)
            source_rows.append(
                {
                    **terminal,
                    "timestamp_s": "999.0",
                    "generation": "3",
                    "status": "deduplicated",
                }
            )
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output, fieldnames=SOURCE_LEDGER_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(source_rows)
            ledger_path.write_text(output.getvalue())

            autotune_path = result_path.with_name("autotune.csv")
            autotune_rows = list(csv.DictReader(io.StringIO(autotune_path.read_text())))
            terminal_csv = next(
                row
                for row in autotune_rows
                if row["config_id"] == repaired_id and row["status"] == "ok"
            )
            for row in autotune_rows:
                if row["config_id"] == repaired_id:
                    row["status"] = "started" if row["status"] == "started" else "error"
                    row["perf_ms"] = ""
            post_configs_by_id = {
                canonical_sha256(config)[:16]: config
                for config in post_qualification_configs
            }
            for row in post_source_rows:
                config = post_configs_by_id[row["config_id"]]
                autotune_rows.append(
                    {
                        **{key: row[key] for key in paired_worker.AUTOTUNE_JOIN_FIELDS},
                        "perf_ms": "1.0" if row["status"] == "ok" else "",
                        "compile_time_s": "0.1" if row["status"] == "ok" else "",
                        "config": (
                            "Config("
                            + ", ".join(
                                f"{key}={value!r}"
                                for key, value in sorted(config.items())
                            )
                            + ")"
                        ),
                    }
                )
            autotune_rows.append(
                {
                    **terminal_csv,
                    "timestamp_s": "999.0",
                    "generation": "3",
                    "status": "deduplicated",
                    "perf_ms": "1.0",
                    "compile_time_s": "",
                }
            )
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output,
                fieldnames=paired_worker.AUTOTUNE_CSV_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(autotune_rows)
            autotune_path.write_text(output.getvalue())

            validated = validate_artifact_set(root)

            repaired_source_rows = list(
                csv.DictReader(io.StringIO(ledger_path.read_text()))
            )
            next(
                row
                for row in repaired_source_rows
                if row["config_id"] == repaired_id and row["status"] == "deduplicated"
            )["generation"] = "4"
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output, fieldnames=SOURCE_LEDGER_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(repaired_source_rows)
            ledger_path.write_text(output.getvalue())
            repaired_autotune_rows = list(
                csv.DictReader(io.StringIO(autotune_path.read_text()))
            )
            next(
                row
                for row in repaired_autotune_rows
                if row["config_id"] == repaired_id and row["status"] == "deduplicated"
            )["generation"] = "4"
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output,
                fieldnames=paired_worker.AUTOTUNE_CSV_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(repaired_autotune_rows)
            autotune_path.write_text(output.getvalue())
            with self.assertRaisesRegex(RuntimeError, "repair resolution generation"):
                validate_artifact_set(root)

        self.assertTrue(
            validated[CASES[("dense", 32768)]]["strict_full_autotune_validated"]
        )

    def test_accepts_accuracy_failure_and_unexecuted_source_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_artifacts(root)["dense_32768"]
            payload = json.loads(result_path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            trial = provenance["trials"][0]
            phase = trial["search_phase_metrics"]
            qualified_results = phase["leaf_results"][0]["qualified_results"]
            retained_ids = set(phase["leaf_results"][0]["retained_config_ids"])
            mutable = [
                result
                for result in qualified_results
                if result["config_id"] not in retained_ids
            ]
            accuracy_result, rejected_result = mutable[-2:]
            accuracy_id = accuracy_result["config_id"]
            rejected_id = rejected_result["config_id"]
            extra_config = {
                "cute_flash_pipeline_family": "fa4",
                "cute_flash_exp2_packet": "1x1",
                "fixture_qualification_config": 999,
            }
            extra_id = canonical_sha256(extra_config)[:16]
            for result, status in (
                (accuracy_result, "accuracy_error"),
                (rejected_result, "source_rejected"),
            ):
                result["attempt_perf"] = None
                result["selection_perf"] = None
                result["status"] = status

            trial["num_accuracy_failures"] = 1
            trial["num_successful_candidate_measurements"] -= 1
            trial["num_source_deduplications"] = 1
            phase["leaf_results"][0]["rounds"][0]["candidate_config_ids"].append(
                extra_id
            )
            phase["leaf_results"][0]["qualified_results"].append(
                {
                    "config_id": extra_id,
                    "attempt_perf": 2.0,
                    "selection_perf": 2.0,
                    "status": "ok",
                    "pipeline_lanes": [],
                }
            )
            phase["candidate_count"] += 1
            result_path.write_text(json.dumps(payload) + "\n")

            metadata_path = result_path.with_name("autotune.meta.jsonl")
            metadata = json.loads(metadata_path.read_text())
            metadata["configs"][extra_id] = extra_config
            metadata_path.write_text(json.dumps(metadata) + "\n")
            add_phase_config_identity(provenance, phase, metadata["configs"])
            result_path.write_text(json.dumps(payload) + "\n")

            ledger_path = result_path.with_name("autotune.sources.csv")
            source_rows = list(csv.DictReader(io.StringIO(ledger_path.read_text())))
            accuracy_terminal = next(
                row
                for row in source_rows
                if row["config_id"] == accuracy_id and row["status"] == "ok"
            )
            accuracy_terminal["status"] = "accuracy_error"
            accuracy_source = accuracy_terminal["source_hash"]
            source_rows = [
                row
                for row in source_rows
                if not (row["config_id"] == rejected_id and row["status"] == "started")
            ]
            rejected_terminal = next(
                row
                for row in source_rows
                if row["config_id"] == rejected_id and row["status"] == "ok"
            )
            rejected_terminal["status"] = "source_rejected"
            rejected_terminal["source_hash"] = accuracy_source
            extra_source = canonical_sha256(
                {"case": "dense_32768", "extra_config": extra_config}
            )
            common = {
                "run_id": source_rows[0]["run_id"],
                "config_id": extra_id,
                "generation": "1",
            }
            source_rows.extend(
                [
                    {
                        **common,
                        "timestamp_s": "999.0",
                        "status": "started",
                        "source_hash": extra_source,
                    },
                    {
                        **common,
                        "timestamp_s": "999.1",
                        "status": "ok",
                        "source_hash": extra_source,
                    },
                ]
            )
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output, fieldnames=SOURCE_LEDGER_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(source_rows)
            ledger_path.write_text(output.getvalue())

            autotune_path = result_path.with_name("autotune.csv")
            autotune_rows = list(csv.DictReader(io.StringIO(autotune_path.read_text())))
            accuracy_csv = next(
                row
                for row in autotune_rows
                if row["config_id"] == accuracy_id and row["status"] == "ok"
            )
            accuracy_csv["status"] = "accuracy_error"
            accuracy_csv["perf_ms"] = ""
            autotune_rows = [
                row
                for row in autotune_rows
                if not (row["config_id"] == rejected_id and row["status"] == "started")
            ]
            rejected_csv = next(
                row
                for row in autotune_rows
                if row["config_id"] == rejected_id and row["status"] == "ok"
            )
            rejected_csv["status"] = "source_rejected"
            rejected_csv["perf_ms"] = ""
            config_repr = (
                "Config("
                + ", ".join(
                    f"{key}={value!r}" for key, value in sorted(extra_config.items())
                )
                + ")"
            )
            autotune_rows.extend(
                [
                    {
                        **common,
                        "timestamp_s": "999.0",
                        "status": "started",
                        "perf_ms": "",
                        "compile_time_s": "",
                        "config": config_repr,
                    },
                    {
                        **common,
                        "timestamp_s": "999.1",
                        "status": "ok",
                        "perf_ms": "2.0",
                        "compile_time_s": "0.1",
                        "config": config_repr,
                    },
                ]
            )
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output,
                fieldnames=paired_worker.AUTOTUNE_CSV_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(autotune_rows)
            autotune_path.write_text(output.getvalue())

            validated = validate_artifact_set(root)

        self.assertTrue(
            validated[CASES[("dense", 32768)]]["strict_full_autotune_validated"]
        )

    def test_rejects_inconsistent_injected_design_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_artifacts(root)
            result_path = paths["dense_32768"]
            payload = json.loads(result_path.read_text())
            payload["helion_overrides"]["autotune_provenance"][
                "flash_structural_injected_design_count"
            ] = 1
            result_path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(
                RuntimeError, "injected structural design count"
            ):
                validate_artifact_set(root)

    def test_rejects_source_ledger_success_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_artifacts(root)
            ledger_path = paths["dense_32768"].with_name("autotune.sources.csv")
            rows = list(csv.DictReader(io.StringIO(ledger_path.read_text())))
            for row in rows:
                if row["status"] == "ok":
                    row["status"] = "error"
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output, fieldnames=SOURCE_LEDGER_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            ledger_path.write_text(output.getvalue())

            with self.assertRaisesRegex(RuntimeError, "successful count"):
                validate_artifact_set(root)

    def test_rejects_duplicate_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_artifacts(root)
            duplicate = root / "duplicate" / "result.json"
            duplicate.parent.mkdir()
            duplicate.write_text(paths["dense_32768"].read_text())
            with self.assertRaisesRegex(RuntimeError, "duplicate strict results"):
                validate_artifact_set(root)

    def test_rejects_fixed_config_disguised_as_full_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_artifacts(root)
            path = paths["causal_65536"]
            payload = json.loads(path.read_text())
            changed = copy.deepcopy(payload)
            changed["helion_overrides"]["autotune_provenance"]["fixed_config"] = True
            path.write_text(json.dumps(changed) + "\n")
            with self.assertRaisesRegex(RuntimeError, "provenance.fixed_config"):
                validate_artifact_set(root)

    def test_combines_two_campaigns_into_publish_ready_raw_schema(self) -> None:
        campaign_seeds = (2026081101, 2026081102)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            strict_artifacts = {}
            harness_sha256 = current_harness_sha256()
            for case in CASES.values():
                config = {"block_sizes": [1, 128, 128]}
                strict_artifacts[case.name] = {
                    "search_result_sha256": canonical_sha256(
                        {"strict_result": case.name}
                    ),
                    "selected_config_sha256": canonical_sha256(config),
                    "selected_source_sha256": "a" * 64,
                    "compiler_seed_policy_sha256": "b" * 64,
                    "terminal_refinement_policy_sha256": "c" * 64,
                    "terminal_coordinate_surface_sha256": "d" * 64,
                    "terminal_refinement_sha256": "e" * 64,
                }
            for campaign_seed in campaign_seeds:
                for case in CASES.values():
                    path = (
                        root
                        / "campaigns"
                        / f"seed_{campaign_seed}"
                        / "results"
                        / f"{case.name}.json"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    raw_pairs = [
                        {
                            "pair_index": index,
                            "order": ["helion", "sdpa"]
                            if index % 2
                            else ["sdpa", "helion"],
                            "times": {
                                "helion": {"event_ms": 1.0, "wall_ms": 1.1},
                                "sdpa": {"event_ms": 1.1, "wall_ms": 1.21},
                            },
                        }
                        for index in range(12)
                    ]
                    payload = {
                        "schema_version": 4,
                        "status": "PASS",
                        "harness_sha256": {
                            "paired_worker.py": harness_sha256["paired_worker.py"]
                        },
                        "campaign_seed": campaign_seed,
                        "shape": {
                            "z": 2,
                            "h": 32,
                            "seq_len": case.seq_len,
                            "head_dim": 64,
                            "dtype": "float16",
                            "causal": int(case.causal),
                        },
                        "flop_model": {
                            "name": "softmax_attention_forward",
                            "flops": 1000.0,
                        },
                        "protocol": {
                            "input_seed": campaign_seed + case.seq_len,
                            "thermal_warmup_seconds": 10.0,
                            "warmup_calls_per_implementation": 3,
                        },
                        "environment": {
                            "cute_version": build_strict_manifest.EXPECTED_CUTE_VERSION,
                            "helion_module": "/fixture/checkout/helion/__init__.py",
                        },
                        "provenance": {
                            **strict_artifacts[case.name],
                            "selected_config": {"block_sizes": [1, 128, 128]},
                            "runtime_checkout": "/fixture/checkout",
                            "artifact_root": f"/fixture/strict/{case.name}",
                            "search_result_path": (
                                f"/fixture/strict/{case.name}/result.json"
                            ),
                        },
                        "regenerated_kernel": {
                            "source_hash_matches_search": True,
                            "regenerated_source_sha256": "a" * 64,
                        },
                        "correctness": {
                            "helion_vs_cudnn_sdpa": {"passed": True},
                            "helion_exact_repeatability": [
                                {"passed": True},
                                {"passed": True},
                            ],
                            "post_timing_peaky_logits": {
                                "performed_after_timing": True,
                                "q_scale_in_place": 2.0,
                                "k_scale_in_place": 2.0,
                                "v_mutated": False,
                                "helion_vs_cudnn_sdpa": {
                                    "thresholds": PEAKY_STRESS_THRESHOLDS,
                                    "atol": PEAKY_STRESS_THRESHOLDS["atol"],
                                    "rtol": PEAKY_STRESS_THRESHOLDS["rtol"],
                                    "finite_outputs": True,
                                    "passed": True,
                                    "actual_nonfinite": 0,
                                    "expected_nonfinite": 0,
                                    "max_abs": 0.001,
                                    "nrmse": 0.001,
                                    "mismatch_fraction": 0.0,
                                },
                                "helion_exact_repeatability": {
                                    "passed": True,
                                    "different": 0,
                                },
                            },
                        },
                        "raw_pairs": raw_pairs,
                    }
                    path.write_text(json.dumps(payload) + "\n")
                    records.append(
                        {
                            "campaign_seed": campaign_seed,
                            "case": case.name,
                            "started_ns": campaign_seed,
                            "finished_ns": campaign_seed + 1,
                            "output_sha256": "b" * 64,
                            "generated_source": (
                                "campaigns/"
                                f"seed_{campaign_seed}/generated_sources/"
                                f"{case.name}.py.txt"
                            ),
                            "generated_source_sha256": "a" * 64,
                        }
                    )
            combined = aggregate_results(
                root,
                campaign_seeds,
                100,
                2026081103,
                {
                    "records": records,
                    "gpu_preflight": {},
                    "harness_sha256": harness_sha256,
                    "strict_artifacts": strict_artifacts,
                },
            )
            worker_path = (
                root
                / "campaigns"
                / f"seed_{campaign_seeds[0]}"
                / "results"
                / "dense_32768.json"
            )
            worker_payload = json.loads(worker_path.read_text())
            worker_payload["schema_version"] = 3
            worker_path.write_text(json.dumps(worker_payload) + "\n")
            with self.assertRaisesRegex(RuntimeError, "worker payload schema mismatch"):
                aggregate_results(
                    root,
                    campaign_seeds,
                    100,
                    2026081103,
                    {
                        "records": records,
                        "gpu_preflight": {},
                        "harness_sha256": harness_sha256,
                        "strict_artifacts": strict_artifacts,
                    },
                )
            worker_payload["schema_version"] = 4
            worker_path.write_text(json.dumps(worker_payload) + "\n")
            changed_records = copy.deepcopy(records)
            changed_records[0]["generated_source_sha256"] = "c" * 64
            with self.assertRaisesRegex(
                RuntimeError, "archived generated source digest"
            ):
                aggregate_results(
                    root,
                    campaign_seeds,
                    100,
                    2026081103,
                    {
                        "records": changed_records,
                        "gpu_preflight": {},
                        "harness_sha256": harness_sha256,
                        "strict_artifacts": strict_artifacts,
                    },
                )
            worker_payload["correctness"]["post_timing_peaky_logits"][
                "helion_vs_cudnn_sdpa"
            ]["max_abs"] = PEAKY_STRESS_THRESHOLDS["max_abs_exclusive"]
            worker_path.write_text(json.dumps(worker_payload) + "\n")
            with self.assertRaisesRegex(RuntimeError, "max_abs failed exclusive gate"):
                aggregate_results(
                    root,
                    campaign_seeds,
                    100,
                    2026081103,
                    {
                        "records": records,
                        "gpu_preflight": {},
                        "harness_sha256": harness_sha256,
                        "strict_artifacts": strict_artifacts,
                    },
                )
            worker_payload["correctness"]["post_timing_peaky_logits"][
                "helion_vs_cudnn_sdpa"
            ]["max_abs"] = 0.001
            worker_payload["harness_sha256"]["paired_worker.py"] = "0" * 64
            worker_path.write_text(json.dumps(worker_payload) + "\n")
            with self.assertRaisesRegex(
                RuntimeError, "worker harness identity changed"
            ):
                aggregate_results(
                    root,
                    campaign_seeds,
                    100,
                    2026081103,
                    {
                        "records": records,
                        "gpu_preflight": {},
                        "harness_sha256": harness_sha256,
                        "strict_artifacts": strict_artifacts,
                    },
                )
        self.assertEqual(combined["status"], "PASS")
        self.assertEqual(len(combined["campaigns"]), 2)
        self.assertEqual(len(combined["results"]), 8)
        self.assertNotIn("runtime_checkout", combined["results"][0]["provenance"])
        self.assertNotIn("artifact_root", combined["results"][0]["provenance"])
        self.assertNotIn("search_result_path", combined["results"][0]["provenance"])
        self.assertNotIn("helion_module", combined["results"][0]["environment"])
        require_portable_json(combined)
        self.assertEqual(combined["protocol"]["combined_pairs_per_shape"], 24)
        self.assertEqual(len(combined["results"][0]["raw_campaigns"]), 2)
        self.assertAlmostEqual(
            combined["results"][0]["summary"]["event"]["paired_log_ratio_pct"],
            10.0,
        )

    def test_manifest_file_references_survive_output_tree_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "scratch" / "paired"
            output = original / "campaigns/seed_1/results/dense_32768.json"
            source = original / "campaigns/seed_1/generated_sources/dense_32768.py.txt"
            output.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            output.write_text("{}\n")
            source.write_text("# generated\n")
            manifest = {
                "status": "PASS",
                "records": [
                    {
                        "output": logical_output_reference(original, output),
                        "output_sha256": sha256(output),
                        "generated_source": logical_output_reference(original, source),
                        "generated_source_sha256": sha256(source),
                    }
                ],
            }
            relocated = root / "archive" / "paired"
            relocated.parent.mkdir()
            original.rename(relocated)
            validate_manifest_files(manifest, relocated)
            self.assertEqual(
                resolve_output_reference(relocated, manifest["records"][0]["output"]),
                relocated / "campaigns/seed_1/results/dense_32768.json",
            )

    def test_recombination_inherits_recorded_bootstrap_settings(self) -> None:
        static_validation = {
            "bootstrap": {
                "samples": 1234,
                "base_seed": 5678,
                "method": (
                    "resample paired log ratios within each campaign/shape stratum"
                ),
            }
        }
        self.assertEqual(
            recorded_bootstrap_settings(static_validation, None, None),
            (1234, 5678),
        )
        self.assertEqual(
            recorded_bootstrap_settings(static_validation, 1234, 5678),
            (1234, 5678),
        )
        with self.assertRaisesRegex(RuntimeError, "--bootstrap-samples"):
            recorded_bootstrap_settings(static_validation, 20000, None)
        with self.assertRaisesRegex(RuntimeError, "--bootstrap-seed"):
            recorded_bootstrap_settings(static_validation, None, 2026081103)

    def test_combine_output_cannot_overwrite_validated_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strict_root = root / "strict"
            strict_root.mkdir()
            output_dir = root / "paired"
            worker = output_dir / "campaigns/seed_1/results/dense_32768.json"
            source = (
                output_dir / "campaigns/seed_1/generated_sources/dense_32768.py.txt"
            )
            worker.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            worker.write_text("{}\n")
            source.write_text("# generated\n")
            strict_result = strict_root / "result.json"
            strict_result.write_text("{}\n")
            strict_sidecars = tuple(
                strict_result.with_name(filename)
                for filename in (
                    "autotune.csv",
                    "autotune.meta.jsonl",
                    "autotune.sources.csv",
                )
            )
            for sidecar in strict_sidecars:
                sidecar.write_text("fixture\n")
            harness_sha256 = current_harness_sha256()
            manifest = {
                "status": "PASS",
                "harness_sha256": harness_sha256,
                "campaign_seeds": [1, 2],
                "records": [
                    {
                        "output": logical_output_reference(output_dir, worker),
                        "output_sha256": sha256(worker),
                        "generated_source": logical_output_reference(
                            output_dir, source
                        ),
                        "generated_source_sha256": sha256(source),
                    }
                ],
                "strict_artifacts": {},
            }
            static_validation = {
                "harness_sha256": harness_sha256,
                "bootstrap": {
                    "samples": 1234,
                    "base_seed": 5678,
                    "method": (
                        "resample paired log ratios within each campaign/shape stratum"
                    ),
                },
            }
            atomic_write_json(output_dir / "run_manifest.json", manifest)
            atomic_write_json(output_dir / "static_validation.json", static_validation)
            validated = {"fixture": {"search_result_path": str(strict_result)}}
            evidence_paths = (
                worker,
                source,
                output_dir / "run_manifest.json",
                output_dir / "static_validation.json",
                strict_result,
                *strict_sidecars,
                Path(combine_results.__file__),
            )
            for evidence in evidence_paths:
                with (
                    self.subTest(evidence=evidence),
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "combine_results.py",
                            "--output-dir",
                            str(output_dir),
                            "--artifact-root",
                            str(strict_root),
                            "--output",
                            str(evidence),
                        ],
                    ),
                    mock.patch(
                        "combine_results.validate_artifact_set",
                        return_value=validated,
                    ),
                    mock.patch("combine_results.validate_strict_artifact_identities"),
                    mock.patch("combine_results.aggregate_results") as aggregate,
                    self.assertRaisesRegex(RuntimeError, "output collides"),
                ):
                    combine_results.main()
                aggregate.assert_not_called()

    def test_output_references_reject_absolute_paths_and_parent_escapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "must be relative"):
                resolve_output_reference(root, str(root / "result.json"))
            with self.assertRaisesRegex(RuntimeError, "escapes output root"):
                resolve_output_reference(root, "../result.json")

    def test_explicit_artifact_root_replaces_stale_run_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relocated = root / "archive" / "strict"
            relocated.mkdir(parents=True)
            manifest = {"artifact_root_at_run": str(root / "deleted" / "strict")}
            self.assertEqual(
                resolve_artifact_root(relocated, manifest, root), relocated.resolve()
            )
            with self.assertRaisesRegex(RuntimeError, "pass --artifact-root"):
                resolve_artifact_root(None, manifest, root)

    def test_relative_artifact_root_survives_common_tree_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "scratch"
            paired = original / "paired"
            strict = original / "strict"
            paired.mkdir(parents=True)
            strict.mkdir()
            manifest = {
                "artifact_root": logical_artifact_root_reference(paired, strict),
                "artifact_root_at_run": str(strict.resolve()),
            }
            relocated = root / "archive"
            original.rename(relocated)
            self.assertEqual(
                resolve_artifact_root(None, manifest, relocated / "paired"),
                (relocated / "strict").resolve(),
            )

    def test_strict_artifact_identity_survives_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "scratch" / "strict"
            self.write_artifacts(original)
            identity = strict_artifact_identities(validate_artifact_set(original))
            relocated = root / "archive" / "strict"
            relocated.parent.mkdir()
            original.rename(relocated)
            validated = validate_artifact_set(relocated)
            self.assertEqual(
                validate_strict_artifact_identities(identity, validated), identity
            )

            changed = copy.deepcopy(identity)
            changed["dense_32768"]["search_result_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                validate_strict_artifact_identities(changed, validated)

    def test_rejects_changed_runner_and_worker_harnesses(self) -> None:
        expected = current_harness_sha256()
        changed = copy.deepcopy(expected)
        changed["run_all8.py"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "harness identity changed"):
            validate_harness_sha256(changed)
        with self.assertRaisesRegex(RuntimeError, "worker harness SHA256"):
            validate_worker_harness_sha256("0" * 64)

    def test_finalization_failure_cannot_leave_pass_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "schema_version": 5,
                "status": "FINALIZING",
                "errors": [],
                "harness_sha256": current_harness_sha256(),
            }
            with (
                mock.patch(
                    "run_all8.aggregate_results", side_effect=RuntimeError("boom")
                ),
                self.assertRaisesRegex(RuntimeError, "boom"),
            ):
                finalize_run(root, (1, 2), 100, 3, manifest)
            failed = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(failed["status"], "FAIL")
            self.assertIn("finalization failed", failed["errors"][-1])

    def test_predicted_manifest_digest_matches_atomic_json_writer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_manifest.json"
            payload = {"schema_version": 5, "status": "PASS", "records": []}
            atomic_write_json(path, payload)
            self.assertEqual(sha256(path), json_payload_sha256(payload))

    def test_worker_environment_scrubs_cudnn_and_pins_cuda_order(self) -> None:
        ambient = {
            "CUDNN_LOGLEVEL_DBG": "3",
            "PYTHONPATH": "/untrusted/checkout",
            "TORCH_CUDNN_V8_API_DISABLED": "1",
            "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
            "PYTHONPYCACHEPREFIX": "/untrusted/pycache",
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(os.environ, ambient, clear=False),
        ):
            env, scrubbed = worker_environment("GPU-test", Path(directory))
        self.assertTrue(set(ambient) <= set(scrubbed))
        self.assertNotIn("CUDNN_LOGLEVEL_DBG", env)
        self.assertNotIn("TORCH_CUDNN_V8_API_DISABLED", env)
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(env["HELION_DISABLE_AUTOTUNER_HEURISTICS"], "0")
        self.assertEqual(env["PYTHONPATH"], str(RUNNER_REPO_ROOT))
        self.assertEqual(
            env["PYTHONPYCACHEPREFIX"], str((Path(directory) / "pycache").resolve())
        )


if __name__ == "__main__":
    unittest.main()
