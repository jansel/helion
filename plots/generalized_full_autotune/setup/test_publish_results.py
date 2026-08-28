from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import operator

import publish_results
from test_build_strict_manifest import add_terminal_refinement
from test_build_strict_manifest import terminal_refinement_summary

COMMIT = "c3e36b65d69681c23e053042b0bc21e2331bad17"
VERSION = "Helion 1.4.0.dev157+gc3e36b65d; CuTe 4.7.0"
VERSION_LABEL = "Helion 1.4.0.dev157+gc3e36b65d / CuTe 4.7.0"
TORCH_VERSION = "2.13.0.dev20260506+cu130"


def shape_dict(key: tuple[object, ...]) -> dict[str, Any]:
    z, h, seq_len, head_dim, dtype, causal, biased = key
    return {
        "z": z,
        "h": h,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "dtype": dtype,
        "causal": causal,
        "biased": biased,
    }


def add_empty_clc_lane_catalog(provenance: dict[str, Any]) -> None:
    catalog: list[dict[str, Any]] = []
    provenance["flash_clc_lane_catalog"] = catalog
    provenance["flash_clc_lane_catalog_sha256"] = publish_results.canonical_sha256(
        catalog
    )


def add_phase_config_identity(
    phase: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    source_hash_by_config: dict[str, str],
) -> None:
    qualified_by_id = {
        qualified["config_id"]: qualified
        for result in phase["leaf_results"]
        for qualified in result["qualified_results"]
    }
    for config_id, qualified in qualified_by_id.items():
        qualified["source_hash"] = source_hash_by_config[config_id]
    referenced_ids = set(phase["initial_config_ids"]) | set(
        phase["exact_space_config_ids"]
    )
    referenced_ids.update(qualified_by_id)
    phase["config_manifest"] = {
        config_id: {"config": configs[config_id]}
        for config_id in configs
        if config_id in referenced_ids
    }
    phase["initial_results"] = [
        {
            "config_id": config_id,
            "family": "fa4_2cta",
            "compound_packet": None,
            "softmax_disc": False,
            "attempt_perf": qualified_by_id[config_id]["attempt_perf"],
            "selection_perf": qualified_by_id[config_id]["selection_perf"],
            "status": qualified_by_id[config_id]["status"],
            "source_hash": qualified_by_id[config_id]["source_hash"],
            "measurement_pass_index": 0,
            "pipeline_lanes": [],
        }
        for config_id in phase["initial_config_ids"]
    ]
    initial_id_set = set(phase["initial_config_ids"])
    states_by_id = {
        record["config_id"]: {
            key: record[key]
            for key in ("attempt_perf", "selection_perf", "status", "source_hash")
        }
        for record in phase["initial_results"]
    }
    pass_count = phase["qualification_passes_completed"]
    for result in phase["leaf_results"]:
        for qualified in result["qualified_results"]:
            config_id = qualified["config_id"]
            state = {
                key: qualified[key]
                for key in (
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                )
            }
            states_by_id.setdefault(config_id, state)
            qualified["measurement_pass_index"] = pass_count
        available_ids = set(result["initial_config_ids"])
        leaf = {
            "family": result["family"],
            "compound_packet": result["compound_packet"],
            "softmax_disc": result["softmax_disc"],
        }
        for pass_index, round_result in enumerate(result["rounds"]):
            if (
                pass_index < phase["qualification_rounds"]
                and not result["space_exhausted"]
            ):
                candidate_ids = sorted(
                    (
                        config_id
                        for config_id in available_ids
                        if states_by_id[config_id]["status"] in {"ok", "deduplicated"}
                        and publish_results.build_strict_manifest.structural_leaf(
                            configs[config_id]
                        )
                        == leaf
                    ),
                    key=lambda config_id: (
                        states_by_id[config_id]["selection_perf"],
                        config_id,
                    ),
                )
                candidate_results = [
                    {
                        "config_id": config_id,
                        **states_by_id[config_id],
                        "measurement_pass_index": pass_index,
                    }
                    for config_id in candidate_ids
                ]
                round_result["parent_decisions"] = [
                    {
                        "job_index": 0,
                        "kind": "ordinary",
                        "pipeline_lane": None,
                        "selection_kind": "ranked_parent",
                        "candidate_results": candidate_results,
                        "selected_config_id": (
                            candidate_ids[0] if candidate_ids else None
                        ),
                        "generated_config_ids": round_result["candidate_config_ids"],
                    }
                ]
            else:
                round_result["parent_decisions"] = []
            available_ids.update(round_result["candidate_config_ids"])
    phase["measurement_timeline"] = [
        {
            "pass_index": pass_index,
            "updates": (
                [
                    {"config_id": config_id, **states_by_id[config_id]}
                    for config_id in sorted(initial_id_set)
                ]
                if pass_index == 0
                else [
                    {"config_id": config_id, **state}
                    for config_id, state in sorted(states_by_id.items())
                    if config_id not in initial_id_set
                ]
                if pass_index == 1
                else []
            ),
        }
        for pass_index in range(pass_count + 1)
    ]
    anchor = phase["initial_results"][0]
    phase.update(
        {
            "schedule_anchor_design_source": (
                "live family x ordinary packet x softmax protocol from fragment defaults"
            ),
            "schedule_anchor_pass_planned": False,
            "schedule_anchor_pass_started": False,
            "schedule_anchor_count": 1,
            "schedule_anchor_complete": True,
            "schedule_anchor_results": [
                {
                    key: anchor[key]
                    for key in (
                        "config_id",
                        "family",
                        "compound_packet",
                        "softmax_disc",
                        "attempt_perf",
                        "selection_perf",
                        "status",
                        "source_hash",
                        "measurement_pass_index",
                    )
                }
            ],
        }
    )


def generated_source_text(index: int) -> str:
    return f"# generated fixture source {index}\n"


def timing_campaigns(
    *, relative_bias_pct: float = 0.0, pair_count: int = 24
) -> list[dict[str, Any]]:
    helion_wall_ratio = 1.0
    sdpa_wall_ratio = 1.0
    if relative_bias_pct >= 0:
        helion_wall_ratio += relative_bias_pct / 100.0
    else:
        sdpa_wall_ratio = 1.0 / (1.0 + relative_bias_pct / 100.0)
    campaigns = []
    for campaign_index in range(2):
        raw_pairs = []
        for pair_index in range(pair_count // 2):
            helion_event = 1.0 + pair_index * 0.001
            sdpa_event = helion_event * 1.01
            raw_pairs.append(
                {
                    "times": {
                        "helion": {
                            "event_ms": helion_event,
                            "wall_ms": helion_event * helion_wall_ratio,
                        },
                        "sdpa": {
                            "event_ms": sdpa_event,
                            "wall_ms": sdpa_event * sdpa_wall_ratio,
                        },
                    }
                }
            )
        campaigns.append({"campaign": campaign_index + 1, "raw_pairs": raw_pairs})
    return campaigns


def timing_summary(
    *,
    event_ratio: float = 1.0,
    event_ci: tuple[float, float] = (0.8, 1.2),
    wall_ratio: float = 1.0,
    wall_ci: tuple[float, float] = (0.8, 1.2),
    marginal_ratio: float = 1.0,
) -> dict[str, Any]:
    return {
        "event": {
            "paired_log_ratio_pct": event_ratio,
            "paired_log_ratio_stratified_bootstrap_95_ci_pct": list(event_ci),
            "median_throughput_ratio_pct": marginal_ratio,
        },
        "wall": {
            "paired_log_ratio_pct": wall_ratio,
            "paired_log_ratio_stratified_bootstrap_95_ci_pct": list(wall_ci),
        },
    }


def timing_summary_from_campaigns(
    campaigns: list[dict[str, Any]],
    *,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 5678,
) -> dict[str, Any]:
    summary = {}
    for timer_index, timer in enumerate(("event", "wall")):
        ratio, ci = publish_results.recompute_paired_statistics(
            campaigns,
            timer=timer,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + timer_index,
        )
        helion = [
            pair["times"]["helion"][f"{timer}_ms"]
            for campaign in campaigns
            for pair in campaign["raw_pairs"]
        ]
        sdpa = [
            pair["times"]["sdpa"][f"{timer}_ms"]
            for campaign in campaigns
            for pair in campaign["raw_pairs"]
        ]
        summary[timer] = {
            "paired_log_ratio_pct": ratio,
            "paired_log_ratio_stratified_bootstrap_95_ci_pct": ci,
            "median_throughput_ratio_pct": 100.0
            * (statistics.median(sdpa) / statistics.median(helion) - 1.0),
        }
    return summary


def strict_payload(key: tuple[object, ...], index: int) -> dict[str, Any]:
    variant = "causal" if key[5] else "dense"
    case_settings = publish_results.build_strict_manifest.EXPECTED_CASES[
        (variant, int(key[2]))
    ]
    tuner_seed = case_settings["tuner_seed"]
    config = {
        "block_sizes": [1, 128, 128],
        "cute_flash_exp2_packet": "1x1",
        "cute_flash_pipeline_family": "fa4_2cta",
        "cute_flash_wait_hint": -1,
        "cute_flash_epi_tma": False,
        "cute_flash_epi_stg": True,
        "cute_flash_epi_stg_store": "slice",
        "cute_flash_epi_stg_gmem": "stage",
    }
    alternate_config = {**config, "cute_flash_wait_hint": 0}
    source_hash = hashlib.sha256(generated_source_text(index).encode()).hexdigest()
    config_hash = publish_results.canonical_sha256(config)
    alternate_config_hash = publish_results.canonical_sha256(alternate_config)
    coverage = [
        {"config": config, "config_sha256": config_hash},
        {
            "config": alternate_config,
            "config_sha256": alternate_config_hash,
        },
    ]
    compiler_seed_ids = [publish_results.canonical_sha256(config)[:16]]
    compiler_seed_policy = {
        "schema_version": 1,
        "kind": "canonical_cute_flash",
        "heuristic_names": ["cute_flash_attention"],
        "raw_config_count": 1,
        "effective_config_ids": compiler_seed_ids,
        "effective_config_ids_sha256": publish_results.canonical_sha256(
            compiler_seed_ids
        ),
        "timeout_retry_repetitions": 3,
    }
    provenance = {
        "require_full_autotune": True,
        "helion_source_tree_sha256": "b" * 64,
        "helion_checkout_git_commit": COMMIT,
        "post_measurement_source_verified": True,
        "post_measurement_source": {
            "helion_source_tree_sha256": "b" * 64,
            "helion_checkout_git_commit": COMMIT,
            "helion_source_tree_dirty": False,
        },
        "effort": "full",
        "requested_force_autotune": True,
        "effective_force_autotune": True,
        "fixed_config": False,
        "autotune_budget_seconds": None,
        "autotune_max_generations": None,
        "autotune_lfbo_max_generations": 1,
        "autotune_best_of_k": 1,
        "autotune_accuracy_check": True,
        "autotune_compile_timeout": 60,
        "autotune_benchmark_subprocess": True,
        "autotune_benchmark_subprocess_env": "",
        "autotune_benchmark_timeout": 60,
        "autotune_adaptive_timeout": True,
        "autotune_force_persistent": False,
        "autotune_finishing_rounds_env": "",
        "autotune_ignore_errors": False,
        "autotune_random_seed": tuner_seed,
        "autotune_cache": "LocalAutotuneCache",
        "disable_autotuner_heuristics": False,
        "autotune_initial_population_strategy_override": None,
        "autotune_initial_population_size": 100,
        "autotuner_initial_population_env": "from_random",
        "autotuner_env": "",
        "autotune_num_neighbors_cap_env": "-1",
        "autotuner_fn": "helion.runtime.settings.default_autotuner_fn",
        "autotuner_fn_is_default": True,
        "autotune_baseline_fn": (
            "examples.attention._causal_attention_output_baseline"
            if key[5]
            else "examples.attention._attention_output_baseline"
        ),
        "autotune_baseline_fn_is_expected": True,
        "autotune_baseline_atol": 5e-2,
        "autotune_baseline_rtol": 2e-2,
        "autotune_baseline_accuracy_check_fn": False,
        "autotune_benchmark_fn": False,
        "autotune_rebenchmark_threshold": None,
        "autotune_suspicious_rebenchmark_ratio": None,
        "autotune_search_acf": [],
        "autotune_config_filter": False,
        "autotune_config_overrides": {},
        "user_seed_configs": False,
        "compiler_seed_config_count": 1,
        "compiler_seed_policy": compiler_seed_policy,
        "compiler_default_config": False,
        "kernel_declared_config_count": 0,
        "active_value_prior_keys": [],
        "flash_value_prior_keys": [],
        "cute_flash_env_overrides": {},
        "final_correctness_enabled": True,
        "final_correctness_launches": 64,
        "final_repeatability_passed": True,
        "final_correctness_passed": True,
        "cache_read_policy": "bypass",
        "cache_write_policy": "write",
        "skip_cache_env": False,
        "rebenchmark_env_overrides": {},
        "flash_fragment_default_config": config,
        "flash_fragment_default_sha256": config_hash,
        "flash_structural_coverage_active_values": [
            {"key": "cute_flash_pipeline_family", "value": "fa4_2cta"},
            {"key": "cute_flash_exp2_packet", "value": "1x1"},
            {"key": "cute_flash_epi_tma", "value": False},
            {"key": "cute_flash_epi_stg", "value": True},
            {"key": "cute_flash_epi_stg_store", "value": "slice"},
            {"key": "cute_flash_epi_stg_gmem", "value": "stage"},
        ],
        "flash_structural_coverage_design": coverage,
        "flash_structural_coverage_design_count": len(coverage),
        "flash_structural_coverage_design_source": (
            "normalized active ConfigSpec fragments"
        ),
        "flash_structural_coverage_design_sha256": (
            publish_results.canonical_sha256([config, alternate_config])
        ),
        "flash_structural_coverage_uncovered_values": [],
        "flash_structural_coverage_underqualified_values": [],
        "flash_structural_leaf_catalog": [
            {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": False,
            }
        ],
        "flash_pipeline_lane_catalog": [
            {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": False,
                "pipeline_lanes": [],
            }
        ],
        "flash_structural_coverage_underqualified_leaves": [],
        "flash_structural_coverage_interaction_key_groups": [
            list(group)
            for group in publish_results.build_strict_manifest.FLASH_INTERACTION_KEY_GROUPS
        ],
        "flash_structural_coverage_active_interactions": [
            {
                "keys": list(
                    publish_results.build_strict_manifest.FLASH_INTERACTION_KEY_GROUPS[
                        0
                    ]
                ),
                "values": [False, True, "slice", "stage"],
            }
        ],
        "flash_structural_coverage_uncovered_interactions": [],
        "flash_structural_qualification_values": [
            {"key": "cute_flash_pipeline_family", "value": "fa4_2cta"}
        ],
        "flash_structural_parent_coverage_prefix_count": 1,
        "flash_structural_qualification_prefix_count": 2,
        "flash_structural_population_budget": 50,
        "flash_structural_injected_design_count": len(coverage),
        "flash_structural_qualification_rounds": 2,
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": 4,
        "flash_structural_family_probe_generations": 1,
        "flash_structural_family_probe_candidates_per_path": 20,
        "flash_structural_retained_candidates_per_leaf": 2,
        "flash_structural_retained_family_cap": 4,
        "flash_structural_retained_family_limit": 1,
        "flash_structural_retained_family_slowdown_limit": 2.0,
        "flash_structural_starting_path_limit": 14,
        "flash_structural_family_probe_path_limit": 0,
        "flash_structural_maximum_path_capacity": 14,
        "flash_structural_unrestricted_path_exhausts_generation_budget": True,
        "selected_config": config,
        "selected_source_sha256": source_hash,
        "selected_config_is_structural_coverage_design_member": True,
        "selected_config_nearest_structural_coverage_design_field_distance": 0,
        "selected_config_nearest_structural_coverage_design_config_sha256": [
            config_hash
        ],
        "trials": [
            {
                "input_shapes": repr([(2, 32, int(key[2]), 64)] * 3),
                "dtypes": repr(["torch.float16"] * 3),
                "hardware": "NVIDIA B200",
                "random_seed": tuner_seed,
                "search_algorithm": "LFBOTreeSearch",
                "num_configs_tested": 101,
                "num_compile_failures": 0,
                "num_worker_failures": 0,
                "num_isolated_rebenchmark_timeouts": 0,
                "num_accuracy_failures": 0,
                "num_successful_candidate_measurements": 101,
                "num_unique_sources": 101,
                "num_source_deduplications": 0,
                "num_generations": 1,
                "autotune_time": 1234.0,
                "best_perf_ms": 1.0,
                "selected_config": config,
                "selected_source_hash": source_hash,
                "selected_source_was_measured": True,
            }
        ],
    }
    add_empty_clc_lane_catalog(provenance)
    context = {
        "schema_version": 1,
        "backend": "cute",
        "config_spec_structural_fingerprint_sha256": "a" * 64,
        "default_config_sha256": config_hash,
        "dtype": "torch.float16",
        "head_dim": 64,
        "num_kv": (int(key[2]) + 127) // 128,
        "num_bh": 64,
        "tensor_4d_heads": 32,
        "is_causal": bool(key[5]),
        "has_kv_tile_pruning": False,
        "requires_ws_overlap": False,
        "small_biased_candidate": False,
        "standard_dense_output": not bool(key[5]),
        "standard_causal_output": bool(key[5]),
        "output_requires_tma": False,
        "supports_tensor_4d_tma": True,
        "block_size_targets": [[0, 1], [1, 128], [2, 128]],
        "flat_key_layout": [["cute_flash_pipeline_family", 1, False]],
    }
    provenance["flash_normalization_context"] = context
    provenance["flash_normalization_context_sha256"] = publish_results.canonical_sha256(
        context
    )
    runs = [99.0 + offset for offset in range(9)]
    median_ms = statistics.median(runs)
    return {
        "impl": "helion-cute",
        "version": VERSION,
        "version_label": VERSION_LABEL,
        "shape": shape_dict(key),
        "gpu": "NVIDIA B200",
        "physical_gpu": "6" if key[5] else "7",
        "power_cap_w": 750,
        "input_seed": 2026081500,
        "flop_model": "softmax_attention_forward",
        "accuracy": "PASS",
        "benchmark_timer": "wall",
        "best_ms": min(runs),
        "median_ms": median_ms,
        "median_tflops": publish_results.expected_attention_flops(shape_dict(key))
        / (median_ms * 1e9),
        "runs_ms": runs,
        "config": f"helion.Config(block_sizes=[1, 128, 128], test_index={index})",
        "helion_overrides": {
            "autotuned": True,
            "benchmark_timer": "wall",
            "force_autotune": True,
            "config_overrides": {},
            "seed_config_overrides": {},
            "return_lse": False,
            "env_overrides": {
                "HELION_AUTOTUNE_RANDOM_SEED": str(tuner_seed),
                "HELION_DISABLE_AUTOTUNER_HEURISTICS": "0",
                "HELION_AUTOTUNER": "",
                "HELION_CAP_AUTOTUNE_NUM_NEIGHBORS": "-1",
                "HELION_AUTOTUNE_EFFORT": "full",
                "HELION_AUTOTUNE_BEST_OF_K": "1",
                "HELION_AUTOTUNE_BENCHMARK_TIMEOUT": "60",
                "HELION_AUTOTUNE_ACCURACY_CHECK": "1",
                "HELION_AUTOTUNER_INITIAL_POPULATION": "from_random",
            },
            "autotune_provenance": provenance,
        },
    }


def baseline_payload(key: tuple[object, ...]) -> dict[str, Any]:
    shape = shape_dict(key)
    common = {
        "shape": shape,
        "gpu": "NVIDIA B200",
        "physical_gpu": "6" if key[5] else "7",
        "power_cap_w": 750,
        "flop_model": "softmax_attention_forward",
        "accuracy": "PASS",
        "benchmark_timer": "event",
        "best_ms": 2.0,
        "median_ms": 2.1,
        "mom_median_ms": 2.1,
        "best_tflops": 500.0,
        "median_tflops": 490.0,
        "mom_median_tflops": 490.0,
    }
    return {
        "shape": shape,
        "suite_metadata": {"must": "survive"},
        "results": [
            {"impl": "untouched", "version": "other 1", **common},
            {
                "impl": "sdpa",
                "version": (
                    f"PyTorch {TORCH_VERSION}; cuDNN runtime 9.20.0; "
                    "nvidia-cudnn-cu13 9.20.0.48"
                ),
                "version_label": "cuDNN 9.20.0.48",
                "notes": ["stale"],
                **common,
            },
            {
                "impl": "helion-cute",
                "version": "old",
                "version_label": "old",
                "notes": ["stale"],
                **common,
            },
        ],
    }


class PublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(
            publish_results.build_heldout_manifest,
            "build_manifest",
            side_effect=self.regenerate_heldout_manifest,
        )
        self.heldout_manifest_builder = patcher.start()
        self.addCleanup(patcher.stop)
        validator_patcher = mock.patch.object(
            publish_results.validate_generalization_campaign,
            "validate_campaign",
            side_effect=self.validate_generalization_campaign,
        )
        self.generalization_validator = validator_patcher.start()
        self.addCleanup(validator_patcher.stop)
        renderer_patcher = mock.patch.object(
            publish_results.validate_generalization_campaign,
            "render_manifest",
            side_effect=self.render_generalization_manifest,
        )
        self.generalization_manifest_renderer = renderer_patcher.start()
        self.addCleanup(renderer_patcher.stop)

    @staticmethod
    def regenerate_heldout_manifest(
        artifact_root: Path, all8_artifact_root: Path
    ) -> str:
        expected_all8_root = artifact_root.parent / "strict"
        if all8_artifact_root.resolve() != expected_all8_root.resolve():
            raise AssertionError(
                f"heldout builder used {all8_artifact_root}, "
                f"expected {expected_all8_root}"
            )
        return (artifact_root / "canonical_heldout_manifest.csv").read_text()

    @staticmethod
    def validate_generalization_campaign(
        artifact_root: Path, *, require_remeasurement: bool
    ) -> SimpleNamespace:
        if not require_remeasurement:
            raise AssertionError("generalization validation omitted remeasurements")
        return SimpleNamespace(
            artifact_root=artifact_root,
            cases=(None,)
            * publish_results.validate_generalization_campaign.EXPECTED_CASE_COUNT,
            run_specs=(
                (None,)
                * publish_results.validate_generalization_campaign.EXPECTED_BROAD_RUN_COUNT
            ),
        )

    @staticmethod
    def render_generalization_manifest(validation: SimpleNamespace) -> str:
        return (
            validation.artifact_root / "canonical_generalization_manifest.csv"
        ).read_text()

    def validate_timing_campaigns(
        self,
        campaigns: list[dict[str, Any]],
        *,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return publish_results.validate_event_wall_consistency(
            campaigns,
            summary=summary or timing_summary(),
            bootstrap_samples=1000,
            bootstrap_seed=1234,
        )

    def write_fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
        baseline_dir = root / "baseline"
        strict_root = root / "strict"
        paired_root = root / "paired"
        baseline_dir.mkdir()
        strict_root.mkdir()
        paired_root.mkdir()
        campaign_seeds = (2026081101, 2026081102)
        bootstrap_seed = 2026081103
        bootstrap_samples = 100
        run_records = []
        plans = []
        strict_identities = {}
        static_provenance = {}
        runtime_checkout = str((root / "runtime-checkout").resolve())
        gpu_uuids = {6: "fixture-gpu-6", 7: "fixture-gpu-7"}
        setup_dir = Path(publish_results.__file__).resolve().parent
        harness_sha256 = {
            name: publish_results.file_sha256(setup_dir / name)
            for name in (
                "run_all8.py",
                "paired_worker.py",
                "build_strict_manifest.py",
                "combine_results.py",
                "test_build_strict_manifest.py",
                "test_static.py",
            )
        }

        for index, key in enumerate(
            sorted(publish_results.EXPECTED_SHAPES, key=operator.itemgetter(5, 2))
        ):
            shape = shape_dict(key)
            name = f"{'causal' if key[5] else 'dense'}_{key[2]}"
            (baseline_dir / f"{name}.json").write_text(
                json.dumps(baseline_payload(key)) + "\n"
            )
            strict = strict_payload(key, index)
            strict_path = strict_root / name / "result.json"
            strict_path.parent.mkdir()
            strict_path.write_text(json.dumps(strict) + "\n")
            provenance = strict["helion_overrides"]["autotune_provenance"]
            selected_config = provenance["selected_config"]
            selected_source = provenance["selected_source_sha256"]
            coverage_configs = [
                item["config"]
                for item in provenance["flash_structural_coverage_design"]
            ]
            selected_config_sha256 = publish_results.canonical_sha256(selected_config)
            ledger_path = strict_path.with_name("autotune.sources.csv")
            tuner_seed = publish_results.build_strict_manifest.EXPECTED_CASES[
                ("causal" if key[5] else "dense", int(key[2]))
            ]["tuner_seed"]
            tensor_shape = (2, 32, int(key[2]), 64)
            metadata = {
                "kernel_name": (
                    "causal_attention_output" if key[5] else "attention_output"
                ),
                "kernel_source": f"fixture kernel for {name}",
                "input_shapes": repr([tensor_shape] * 3),
                "dtypes": repr(["torch.float16"] * 3),
                "hardware": "NVIDIA B200",
                "settings": {
                    "backend": "cute",
                    "force_autotune": False,
                    "effective_cache_read_bypass": True,
                    "static_shapes": True,
                    "autotune_log_details": True,
                    "autotune_compile_timeout": 60,
                    "autotune_benchmark_subprocess": True,
                    "autotune_benchmark_timeout": 60,
                    "autotune_random_seed": tuner_seed,
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
                },
                "ir_graph": None,
                "configs": {},
            }
            ledger_run_id = publish_results.build_strict_manifest.metadata_run_id(
                metadata, Path("fixture.meta.jsonl")
            )
            metadata["run_id"] = ledger_run_id
            ledger_output = io.StringIO(newline="")
            ledger_writer = csv.DictWriter(
                ledger_output,
                fieldnames=publish_results.SOURCE_LEDGER_FIELDS,
                lineterminator="\n",
            )
            ledger_writer.writeheader()
            ledger_rows = []
            autotune_rows = []
            initial_config_ids = []
            source_hash_by_config: dict[str, str] = {}
            for source_index in range(101):
                source_hash = (
                    selected_source
                    if source_index == 0
                    else publish_results.canonical_sha256(
                        {"case": name, "source": source_index}
                    )
                )
                config = (
                    coverage_configs[source_index]
                    if source_index < len(coverage_configs)
                    else {
                        "case": name,
                        "config": source_index,
                        "cute_flash_pipeline_family": "fa4_2cta",
                        "cute_flash_exp2_packet": "1x1",
                    }
                )
                config_id = publish_results.canonical_sha256(config)[:16]
                metadata["configs"][config_id] = config
                source_hash_by_config[config_id] = source_hash
                if source_index < 100:
                    initial_config_ids.append(config_id)
                common = {
                    "run_id": ledger_run_id,
                    "config_id": config_id,
                    "generation": "0" if source_index < 100 else "1",
                    "source_hash": source_hash,
                }
                for offset, status in enumerate(("started", "ok")):
                    ledger_row = {
                        "timestamp_s": f"{source_index + offset / 10:.1f}",
                        "status": status,
                        **common,
                    }
                    ledger_rows.append(ledger_row)
                    config_repr = (
                        "Config("
                        + ", ".join(
                            f"{field}={value!r}"
                            for field, value in sorted(config.items())
                        )
                        + ")"
                    )
                    autotune_rows.append(
                        {
                            **{
                                field: ledger_row[field]
                                for field in publish_results.build_strict_manifest.AUTOTUNE_JOIN_FIELDS
                            },
                            "perf_ms": "1.0" if status == "ok" else "",
                            "compile_time_s": "0.1",
                            "config": config_repr,
                        }
                    )
            ledger_writer.writerows(ledger_rows)
            ledger_path.write_text(ledger_output.getvalue())
            autotune_output = io.StringIO(newline="")
            autotune_writer = csv.DictWriter(
                autotune_output,
                fieldnames=publish_results.build_strict_manifest.AUTOTUNE_CSV_FIELDS,
                lineterminator="\n",
            )
            autotune_writer.writeheader()
            autotune_writer.writerows(autotune_rows)
            strict_path.with_name("autotune.csv").write_text(autotune_output.getvalue())
            strict_path.with_name("autotune.meta.jsonl").write_text(
                json.dumps(metadata) + "\n"
            )
            generated_config_id = config_id
            leaf = {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": False,
            }
            leaf_initial_ids = [
                config_id
                for config_id in initial_config_ids
                if publish_results.build_strict_manifest.structural_leaf(
                    metadata["configs"][config_id]
                )
                == leaf
            ]
            retention_ids = list(
                dict.fromkeys([*leaf_initial_ids, generated_config_id])
            )
            retained_families = (
                publish_results.build_strict_manifest.expected_structural_retention(
                    [
                        {
                            **leaf,
                            "members": [
                                {
                                    "config_id": config_id,
                                    "selection_perf": 1.0,
                                    "pipeline_lanes": frozenset(),
                                }
                                for config_id in retention_ids
                            ],
                            "pipeline_lanes": [],
                        }
                    ],
                    retained_per_leaf=2,
                    retained_family_cap=4,
                    retained_family_limit=1,
                    retained_family_slowdown_limit=2.0,
                    starting_path_limit=14,
                )
            )
            provenance["trials"][0]["search_phase_metrics"] = {
                "phase": "cute_flash_structural_qualification_v22",
                "cute_flash_lane_policy_version": (
                    publish_results.build_strict_manifest.EXPECTED_LANE_POLICY_VERSION
                ),
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
                    publish_results.build_strict_manifest.FLASH_PIPELINE_QUALIFICATION_KEYS
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
                                "candidate_config_ids": [generated_config_id],
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
                            for config_id in [*leaf_initial_ids, generated_config_id]
                        ],
                        "retained_config_ids": sorted(leaf_initial_ids)[:2],
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
                "family_probe_generations": 1,
                "family_probe_generations_started": 0,
                "family_probe_generations_completed": 0,
                "family_probe_candidates_per_path": 20,
                "family_probe_required": False,
                "family_probe_complete": True,
                "family_probe_path_limit": 0,
                "family_probe_paths": [],
                "neighbor_generation_limit_per_leaf_per_round": 200,
                "candidate_count": 1,
                "leaves_with_candidates": 1,
                "retained_candidates_per_leaf": 2,
                "retained_family_cap": 4,
                "retained_family_limit": 1,
                "retained_family_slowdown_limit": 2.0,
                "clc_families": [],
                "compound_catalog_complete": True,
                "compound_catalog_errors": [],
                "compound_transfers": [],
                "starting_path_limit": 14,
                "maximum_path_capacity": 14,
                "retained_families": retained_families,
                "retained_path_count": sum(
                    len(family["starting_paths"]) for family in retained_families
                ),
            }
            add_phase_config_identity(
                provenance["trials"][0]["search_phase_metrics"],
                metadata["configs"],
                source_hash_by_config,
            )
            phase = provenance["trials"][0]["search_phase_metrics"]
            add_terminal_refinement(
                provenance,
                provenance["trials"][0],
                phase,
                metadata["configs"],
                {
                    config_id: {
                        "status": "ok",
                        "perf_ms": 1.0,
                        "source_hash": source_hash,
                    }
                    for config_id, source_hash in source_hash_by_config.items()
                },
                default_perf=1.0,
            )
            strict_path.write_text(json.dumps(strict) + "\n")

            worker_provenance = {
                "strict_full_autotune_validated": True,
                "search_result_sha256": publish_results.file_sha256(strict_path),
                "search_version": VERSION,
                "runtime_git_head": COMMIT,
                "runtime_tracked_clean": True,
                "runtime_checkout": runtime_checkout,
                "artifact_root": str(strict_path.parent.resolve()),
                "search_result_path": str(strict_path.resolve()),
                "strict_runtime_environment": {
                    "worker_pythonpath": runtime_checkout,
                },
                "selected_config": selected_config,
                "selected_config_sha256": selected_config_sha256,
                "selected_source_sha256": selected_source,
                "input_seed": 2026081500,
                "compiler_seed_policy": provenance["compiler_seed_policy"],
                "compiler_seed_policy_sha256": publish_results.canonical_sha256(
                    provenance["compiler_seed_policy"]
                ),
                "terminal_refinement_policy_sha256": provenance[
                    "flash_terminal_coordinate_refinement_policy_sha256"
                ],
                "terminal_coordinate_surface_sha256": provenance[
                    "flash_terminal_coordinate_surface_catalog_sha256"
                ],
            }
            terminal_summary = terminal_refinement_summary(provenance, phase)
            terminal_summary["preterminal_successful_measurement_count"] = provenance[
                "trials"
            ][0]["num_successful_candidate_measurements"]
            worker_provenance["terminal_refinement_sha256"] = (
                publish_results.canonical_sha256(terminal_summary)
            )
            worker_provenance["structural_design_execution"] = {
                "terminal_refinement": terminal_summary
            }
            strict_identities[name] = {
                field: worker_provenance[field]
                for field in (
                    "search_result_sha256",
                    "selected_config_sha256",
                    "selected_source_sha256",
                    "compiler_seed_policy_sha256",
                    "terminal_refinement_policy_sha256",
                    "terminal_coordinate_surface_sha256",
                    "terminal_refinement_sha256",
                )
            }
            static_provenance[name] = worker_provenance
            flops = publish_results.expected_attention_flops(shape)
            helion_runs = [1.0, 1.1, 1.2, 1.3]
            sdpa_runs = [1.1, 1.2, 1.3, 1.4]
            for campaign, campaign_seed in enumerate(campaign_seeds, 1):
                campaign_dir = paired_root / "campaigns" / f"seed_{campaign_seed}"
                result_path = campaign_dir / "results" / f"{name}.json"
                source_path = campaign_dir / "generated_sources" / f"{name}.py.txt"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.parent.mkdir(parents=True, exist_ok=True)
                source_path.write_text(generated_source_text(index))
                raw_pairs = [
                    {
                        "pair_index": pair_index,
                        "order": ["helion", "sdpa"]
                        if pair_index == campaign - 1
                        else ["sdpa", "helion"],
                        "times": {
                            "helion": {
                                "event_ms": helion_runs[
                                    (campaign - 1) * 2 + pair_index
                                ],
                                "wall_ms": helion_runs[(campaign - 1) * 2 + pair_index]
                                + 0.01,
                            },
                            "sdpa": {
                                "event_ms": sdpa_runs[(campaign - 1) * 2 + pair_index],
                                "wall_ms": sdpa_runs[(campaign - 1) * 2 + pair_index]
                                + 0.01,
                            },
                        },
                    }
                    for pair_index in range(2)
                ]
                worker_payload = {
                    "schema_version": 4,
                    "status": "PASS",
                    "harness_sha256": {
                        "paired_worker.py": harness_sha256["paired_worker.py"]
                    },
                    "campaign_seed": campaign_seed,
                    "protocol": {
                        "input_seed": campaign_seed + int(key[2]),
                        "thermal_warmup_seconds": 1.0,
                        "warmup_calls_per_implementation": 1,
                    },
                    "shape": {
                        field: value
                        for field, value in shape.items()
                        if field != "biased"
                    },
                    "flop_model": {
                        "name": publish_results.FLOP_MODEL_NAME,
                        "formula": publish_results.FLOP_MODEL_FORMULA,
                        "flops": flops,
                    },
                    "environment": {
                        "cute_version": (
                            publish_results.build_strict_manifest.EXPECTED_CUTE_VERSION
                        ),
                        "torch_version": TORCH_VERSION,
                        "cudnn_version": 92000,
                        "helion_module": f"{runtime_checkout}/helion/__init__.py",
                    },
                    "provenance": worker_provenance,
                    "regenerated_kernel": {
                        "regenerated_source_sha256": selected_source,
                        "compiled_source_sha256": selected_source,
                        "expected_source_sha256": selected_source,
                        "source_hash_matches_search": True,
                        "compiled_from_current_examples_attention": True,
                        "terminal_refinement_policy_sha256": worker_provenance[
                            "terminal_refinement_policy_sha256"
                        ],
                        "terminal_coordinate_surface_sha256": worker_provenance[
                            "terminal_coordinate_surface_sha256"
                        ],
                        "terminal_refinement_transcript_sha256": terminal_summary[
                            "transcript_sha256"
                        ],
                        "terminal_projection_request_count": terminal_summary[
                            "projection_attempt_count"
                        ],
                        "live_full_autotune_validated": True,
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
                                "count": 1000,
                                "close_count": 1000,
                                "close_fraction": 1.0,
                                "mismatch_count": 0,
                                "mismatch_fraction": 0.0,
                                "max_abs": 0.001,
                                "mean_abs": 0.0001,
                                "rmse": 0.0001,
                                "expected_rms": 1.0,
                                "nrmse": 0.0001,
                                "nrmse_normalization": "rms(cudnn_sdpa_output)",
                                "actual_nonfinite": 0,
                                "expected_nonfinite": 0,
                                "atol": 0.002,
                                "rtol": 0.01,
                                "finite_outputs": True,
                                "thresholds": {
                                    "atol": 0.002,
                                    "rtol": 0.01,
                                    "max_abs_exclusive": 0.01,
                                    "nrmse_exclusive": 0.002,
                                    "mismatch_fraction_exclusive": 1e-5,
                                },
                                "passed": True,
                            },
                            "helion_exact_repeatability": {
                                "count": 1000,
                                "different": 0,
                                "different_fraction": 0.0,
                                "passed": True,
                            },
                        },
                    },
                    "raw_pairs": raw_pairs,
                }
                result_path.write_text(json.dumps(worker_payload) + "\n")
                command = [
                    sys.executable,
                    "paired_worker.py",
                    "--campaign-seed",
                    str(campaign_seed),
                    "--expected-worker-sha256",
                    harness_sha256["paired_worker.py"],
                    "--case",
                    name,
                ]
                gpu_uuid = gpu_uuids[6 if key[5] else 7]
                cache_dir = (
                    root / "worker-cache" / f"{campaign_seed}-{name}"
                ).resolve()
                run_records.append(
                    {
                        "campaign_seed": campaign_seed,
                        "case": name,
                        "physical_gpu": 6 if key[5] else 7,
                        "gpu_uuid": gpu_uuid,
                        "command": command,
                        "returncode": 0,
                        "termination_reason": None,
                        "cleanup": None,
                        "worker_timeout_seconds": 1200.0,
                        "scrubbed_environment_keys": [],
                        "controlled_environment": {
                            "CUDA_VISIBLE_DEVICES": gpu_uuid,
                            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                            "HELION_BACKEND": "cute",
                            "HELION_CACHE_DIR": str(cache_dir),
                            "HELION_DISABLE_AUTOTUNER_HEURISTICS": "0",
                            "PYTHONHASHSEED": "0",
                            "PYTHONPATH": runtime_checkout,
                            "PYTHONPYCACHEPREFIX": str(cache_dir / "pycache"),
                        },
                        "started_ns": campaign * 1000 + index,
                        "finished_ns": campaign * 1000 + index + 1,
                        "output": result_path.relative_to(paired_root).as_posix(),
                        "output_sha256": publish_results.file_sha256(result_path),
                        "generated_source": source_path.relative_to(
                            paired_root
                        ).as_posix(),
                        "generated_source_sha256": publish_results.file_sha256(
                            source_path
                        ),
                    }
                )
                plans.append(
                    {
                        "campaign_seed": campaign_seed,
                        "worker_input_seed": campaign_seed + int(key[2]),
                        "case": name,
                        "physical_gpu": 6 if key[5] else 7,
                        "gpu_uuid": gpu_uuid,
                        "cuda_visible_devices": gpu_uuid,
                        "command": command,
                    }
                )

        strict_manifest = root / "strict_manifest.csv"
        strict_manifest.write_text(
            publish_results.build_strict_manifest.build_manifest(strict_root)
        )
        heldout_root = root / "heldout"
        heldout_root.mkdir()
        heldout_rows = []
        for (
            variant,
            seq_len,
            _physical_gpu,
            tuner_seed,
        ) in publish_results.build_heldout_manifest.CASES:
            row = dict.fromkeys(publish_results.HELDOUT_MANIFEST_FIELDS, "")
            row.update(
                {
                    "case": f"{variant}_{seq_len}_seed_{tuner_seed}",
                    "variant": variant,
                    "seq_len": str(seq_len),
                    "version": VERSION,
                    "all8_reference_manifest_sha256": (
                        publish_results.file_sha256(strict_manifest)
                    ),
                }
            )
            heldout_rows.append(row)
        heldout_output = io.StringIO(newline="")
        heldout_writer = csv.DictWriter(
            heldout_output,
            fieldnames=publish_results.HELDOUT_MANIFEST_FIELDS,
            lineterminator="\n",
        )
        heldout_writer.writeheader()
        heldout_writer.writerows(heldout_rows)
        heldout_contents = heldout_output.getvalue()
        (heldout_root / "canonical_heldout_manifest.csv").write_text(heldout_contents)
        (root / "heldout_manifest.csv").write_text(heldout_contents)
        generalization_root = root / "generalization"
        generalization_root.mkdir()
        generalization_contents = "case_id,tuner_seed\nfixture,2026081601\n"
        (generalization_root / "canonical_generalization_manifest.csv").write_text(
            generalization_contents
        )
        (root / "generalization_manifest.csv").write_text(generalization_contents)
        run_manifest = paired_root / "run_manifest.json"
        run_payload = {
            "schema_version": 5,
            "status": "PASS",
            "artifact_root": "../strict",
            "artifact_root_at_run": str(strict_root),
            "strict_artifacts": strict_identities,
            "harness_sha256": harness_sha256,
            "campaign_seeds": list(campaign_seeds),
            "gpu_preflight": {"6": {}, "7": {}},
            "records": sorted(
                run_records, key=operator.itemgetter("campaign_seed", "case")
            ),
            "termination_signals": [],
            "errors": [],
        }
        run_manifest.write_text(json.dumps(run_payload, sort_keys=True) + "\n")
        static_validation = paired_root / "static_validation.json"
        static_payload = {
            "schema_version": 5,
            "status": "READY",
            "artifact_root": "../strict",
            "artifact_root_at_run": str(strict_root),
            "strict_artifacts": strict_identities,
            "harness_sha256": harness_sha256,
            "campaign_seeds": list(campaign_seeds),
            "runtime_checkout": runtime_checkout,
            "parallel_lanes": {
                "dense": {"physical_gpu": 7, "gpu_uuid": gpu_uuids[7]},
                "causal": {"physical_gpu": 6, "gpu_uuid": gpu_uuids[6]},
            },
            "worker_environment_policy": {
                "scrubbed_prefixes": [
                    "CUDNN_",
                    "CUTE_DSL_",
                    "HELION_",
                    "TORCH_CUDNN_",
                ],
                "scrubbed_names": ["CUDA_DEVICE_ORDER", "PYTHONPYCACHEPREFIX"],
                "controlled_values": {
                    "HELION_BACKEND": "cute",
                    "HELION_CACHE_DIR": "fresh per shape and campaign",
                    "HELION_DISABLE_AUTOTUNER_HEURISTICS": "0",
                    "PYTHONHASHSEED": "0",
                    "CUDA_VISIBLE_DEVICES": "per-lane expected GPU UUID",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "PYTHONPATH": runtime_checkout,
                    "PYTHONPYCACHEPREFIX": "fresh per shape and campaign",
                },
            },
            "worker_timeout_seconds": 1200.0,
            "bootstrap": {
                "samples": bootstrap_samples,
                "base_seed": bootstrap_seed,
                "method": "resample paired log ratios within each campaign/shape stratum",
            },
            "planned_commands": sorted(
                plans, key=operator.itemgetter("campaign_seed", "case")
            ),
            "provenance": static_provenance,
        }
        static_validation.write_text(json.dumps(static_payload, sort_keys=True) + "\n")
        paired_raw = paired_root / "all8_paired_raw.json"
        paired_payload = publish_results.combine_results.aggregate_results(
            paired_root,
            campaign_seeds,
            bootstrap_samples,
            bootstrap_seed,
            run_payload,
        )
        paired_payload["run_manifest_sha256"] = publish_results.file_sha256(
            run_manifest
        )
        paired_payload["static_validation_sha256"] = publish_results.file_sha256(
            static_validation
        )
        paired_raw.write_text(json.dumps(paired_payload, sort_keys=True) + "\n")
        return (
            baseline_dir,
            strict_root,
            strict_manifest,
            paired_raw,
            run_manifest,
            static_validation,
        )

    def publication_args(
        self,
        fixture: tuple[Path, Path, Path, Path, Path, Path],
        output_dir: Path,
        *,
        overwrite: bool,
    ) -> argparse.Namespace:
        (
            baseline_dir,
            strict_root,
            strict_manifest,
            paired_raw,
            run_manifest,
            static_validation,
        ) = fixture
        renderer = output_dir.parent / "renderer.py"
        renderer.write_text("# renderer fixture\n")
        return argparse.Namespace(
            baseline_payload_dir=baseline_dir,
            strict_artifact_root=strict_root,
            strict_manifest=strict_manifest,
            heldout_artifact_root=strict_root.parent / "heldout",
            heldout_manifest=strict_root.parent / "heldout_manifest.csv",
            generalization_artifact_root=strict_root.parent / "generalization",
            generalization_manifest=strict_root.parent / "generalization_manifest.csv",
            paired_raw=paired_raw,
            run_manifest=run_manifest,
            static_validation=static_validation,
            output_payload_dir=output_dir,
            raw_artifact_label="plots/generalized_full_autotune/raw.json",
            overwrite=overwrite,
            renderer=renderer,
            plot_impl_label=[],
        )

    def assert_no_publication_staging(self, root: Path) -> None:
        leftovers = [path for path in root.rglob(".*") if ".publish-" in path.name]
        self.assertEqual(leftovers, [])

    def test_publishes_all_shapes_and_preserves_non_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            output_dir = root / "output"
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=output_dir,
                raw_artifact_label=(
                    "plots/generalized_full_autotune/all8_paired_raw.json"
                ),
                overwrite=False,
            )
            paths = publish_results.publish(args)
            self.heldout_manifest_builder.assert_called_once_with(
                args.heldout_artifact_root,
                args.strict_artifact_root,
            )
            self.generalization_validator.assert_called_once_with(
                args.generalization_artifact_root, require_remeasurement=True
            )
            self.generalization_manifest_renderer.assert_called_once()
            validation = self.generalization_manifest_renderer.call_args.args[0]
            self.assertEqual(
                validation.artifact_root, args.generalization_artifact_root
            )
            self.assertEqual(len(paths), 8)
            for path in paths:
                original = json.loads((baseline_dir / path.name).read_text())
                published = json.loads(path.read_text())
                publish_results.validate_non_target_results(
                    original, published, context=path.name
                )
                targets = {
                    result["impl"]: result
                    for result in published["results"]
                    if result["impl"] in publish_results.TARGET_IMPLS
                }
                self.assertNotIn("input_seed", targets["helion-cute"])
                self.assertNotIn("input_seed", targets["sdpa"])
                self.assertEqual(
                    targets["helion-cute"]["paired_protocol"]["campaign_seeds"],
                    [2026081101, 2026081102],
                )
                self.assertEqual(
                    len(targets["helion-cute"]["paired_protocol"]["input_seeds"]),
                    2,
                )
                self.assertEqual(
                    targets["helion-cute"]["paired_protocol"]["raw_artifact_sha256"],
                    publish_results.file_sha256(paired_raw),
                )
                self.assertEqual(targets["helion-cute"]["version"], VERSION)
                self.assertTrue(
                    targets["helion-cute"]["full_autotune_provenance"][
                        "selected_source_was_measured"
                    ]
                )
                self.assertEqual(
                    targets["helion-cute"]["full_autotune_provenance"][
                        "strict_manifest_sha256"
                    ],
                    publish_results.file_sha256(strict_manifest),
                )
                self.assertEqual(
                    targets["helion-cute"]["full_autotune_provenance"][
                        "heldout_manifest_sha256"
                    ],
                    publish_results.file_sha256(args.heldout_manifest),
                )
                self.assertEqual(
                    targets["helion-cute"]["full_autotune_provenance"][
                        "generalization_manifest_sha256"
                    ],
                    publish_results.file_sha256(args.generalization_manifest),
                )
                self.assertEqual(targets["helion-cute"]["benchmark_timer"], "event")
                self.assertEqual(
                    targets["helion-cute"]["helion_overrides"]["benchmark_timer"],
                    "wall",
                )
                self.assertEqual(targets["sdpa"]["benchmark_timer"], "event")
                self.assertEqual(targets["helion-cute"]["median_ms"], 1.15)
                self.assertEqual(targets["sdpa"]["median_ms"], 1.25)

    def test_event_wall_consistency_rejects_relative_timer_bias(self) -> None:
        campaigns = timing_campaigns()
        for campaign in campaigns:
            for pair in campaign["raw_pairs"]:
                pair["times"]["helion"]["event_ms"] *= 0.99
        with self.assertRaisesRegex(RuntimeError, "relative ratio bias"):
            self.validate_timing_campaigns(campaigns)

    def test_event_wall_consistency_rejects_common_mode_timer_failure(self) -> None:
        campaigns = timing_campaigns()
        for campaign in campaigns:
            for pair in campaign["raw_pairs"]:
                for implementation in ("helion", "sdpa"):
                    pair["times"][implementation]["event_ms"] *= 0.95
        with self.assertRaisesRegex(RuntimeError, "helion absolute sanity"):
            self.validate_timing_campaigns(campaigns)

    def test_event_wall_consistency_allows_at_most_two_outliers(self) -> None:
        campaigns = timing_campaigns()
        pairs = [pair for campaign in campaigns for pair in campaign["raw_pairs"]]
        for pair in pairs[:2]:
            pair["times"]["helion"]["wall_ms"] *= 1.03
        result = self.validate_timing_campaigns(campaigns)
        self.assertEqual(result["relative_ratio_bias"]["inlier_count"], 22)
        self.assertEqual(
            result["implementation_wall_over_event"]["helion"]["inlier_count"],
            22,
        )

        pairs[2]["times"]["helion"]["wall_ms"] *= 1.03
        with self.assertRaisesRegex(RuntimeError, "only 21/24 inlier pairs"):
            self.validate_timing_campaigns(campaigns)

    def test_event_wall_consistency_relative_bias_boundaries(self) -> None:
        for bias in (-0.49, 0.49):
            with self.subTest(bias=bias):
                self.validate_timing_campaigns(timing_campaigns(relative_bias_pct=bias))
        for bias in (-0.51, 0.51):
            with (
                self.subTest(bias=bias),
                self.assertRaisesRegex(RuntimeError, "relative ratio bias"),
            ):
                self.validate_timing_campaigns(timing_campaigns(relative_bias_pct=bias))

    def test_event_wall_consistency_is_deterministic(self) -> None:
        campaigns = timing_campaigns()
        pairs = [pair for campaign in campaigns for pair in campaign["raw_pairs"]]
        for index, pair in enumerate(pairs):
            pair["times"]["helion"]["wall_ms"] *= 1.0 + index / 100_000
            pair["times"]["sdpa"]["wall_ms"] *= 1.0 + index / 200_000
        first = self.validate_timing_campaigns(campaigns)
        second = self.validate_timing_campaigns(campaigns)
        self.assertEqual(first, second)
        self.assertEqual(
            first["method"],
            "stratified bootstrap median with shared pair resampling",
        )

    def test_event_wall_consistency_classifies_stable_small_directions(self) -> None:
        stable = timing_summary(
            event_ratio=0.4,
            event_ci=(0.1, 0.7),
            wall_ratio=0.3,
            wall_ci=(0.05, 0.6),
        )
        result = self.validate_timing_campaigns(timing_campaigns(), summary=stable)
        directional = result["sub_half_percent_directional"]
        self.assertTrue(directional["in_scope"])
        self.assertEqual(directional["classification"], "gain")
        self.assertTrue(all(directional["agreement_checks"].values()))

        unstable = (
            timing_summary(
                event_ratio=0.4,
                event_ci=(0.1, 0.7),
                wall_ratio=-0.3,
                wall_ci=(-0.6, -0.05),
            ),
            timing_summary(
                event_ratio=0.4,
                event_ci=(-0.1, 0.7),
                wall_ratio=0.3,
                wall_ci=(0.05, 0.6),
            ),
            timing_summary(
                event_ratio=0.4,
                event_ci=(0.1, 0.7),
                wall_ratio=0.3,
                wall_ci=(-0.05, 0.6),
            ),
        )
        for summary in unstable:
            with self.subTest(summary=summary):
                result = self.validate_timing_campaigns(
                    timing_campaigns(), summary=summary
                )
                self.assertEqual(
                    result["sub_half_percent_directional"]["classification"],
                    "inconclusive",
                )

    def test_event_wall_consistency_admitted_outlier_is_inconclusive(self) -> None:
        campaigns = timing_campaigns()
        for campaign in campaigns:
            for pair in campaign["raw_pairs"]:
                helion = pair["times"]["helion"]
                sdpa = pair["times"]["sdpa"]
                helion["wall_ms"] = helion["event_ms"]
                sdpa["event_ms"] = helion["event_ms"] * 1.004
                sdpa["wall_ms"] = sdpa["event_ms"]
        campaigns[0]["raw_pairs"][0]["times"]["helion"]["wall_ms"] *= 1.10
        result = self.validate_timing_campaigns(
            campaigns, summary=timing_summary_from_campaigns(campaigns)
        )
        self.assertEqual(result["relative_ratio_bias"]["inlier_count"], 23)
        directional = result["sub_half_percent_directional"]
        self.assertEqual(directional["classification"], "inconclusive")
        self.assertFalse(directional["agreement_checks"]["wall_confidence_interval"])

    def test_event_wall_consistency_classifies_parity_as_inconclusive(self) -> None:
        campaigns = timing_campaigns()
        for campaign in campaigns:
            for pair in campaign["raw_pairs"]:
                helion = pair["times"]["helion"]
                sdpa = pair["times"]["sdpa"]
                sdpa["event_ms"] = helion["event_ms"]
                sdpa["wall_ms"] = helion["wall_ms"]
        result = self.validate_timing_campaigns(
            campaigns, summary=timing_summary_from_campaigns(campaigns)
        )
        directional = result["sub_half_percent_directional"]
        self.assertTrue(directional["in_scope"])
        self.assertEqual(directional["classification"], "inconclusive")

    def test_event_wall_consistency_classifies_small_regression(self) -> None:
        campaigns = timing_campaigns()
        for campaign in campaigns:
            for pair in campaign["raw_pairs"]:
                helion = pair["times"]["helion"]
                sdpa = pair["times"]["sdpa"]
                sdpa["event_ms"] = helion["event_ms"] * 0.997
                sdpa["wall_ms"] = helion["wall_ms"] * 0.997
        result = self.validate_timing_campaigns(
            campaigns, summary=timing_summary_from_campaigns(campaigns)
        )
        directional = result["sub_half_percent_directional"]
        self.assertEqual(directional["classification"], "regression")
        self.assertTrue(all(directional["agreement_checks"].values()))

    def test_event_wall_consistency_campaign_disagreement_is_inconclusive(
        self,
    ) -> None:
        campaigns = timing_campaigns()
        for campaign_index, campaign in enumerate(campaigns):
            ratio = 1.01 if campaign_index == 0 else 0.998
            for pair in campaign["raw_pairs"]:
                helion = pair["times"]["helion"]
                sdpa = pair["times"]["sdpa"]
                helion["event_ms"] = helion["wall_ms"] = 1.0
                sdpa["event_ms"] = sdpa["wall_ms"] = ratio
        result = self.validate_timing_campaigns(
            campaigns, summary=timing_summary_from_campaigns(campaigns)
        )
        directional = result["sub_half_percent_directional"]
        self.assertEqual(directional["classification"], "inconclusive")
        self.assertEqual(
            directional["campaign_directions"]["event"], ["gain", "regression"]
        )
        self.assertFalse(directional["agreement_checks"]["event_campaigns"])

    def test_event_wall_consistency_marginal_disagreement_is_inconclusive(
        self,
    ) -> None:
        campaigns = timing_campaigns()
        for campaign in campaigns:
            for pair_index, pair in enumerate(campaign["raw_pairs"]):
                if pair_index < 7:
                    helion_ms, sdpa_ms = 100.0, 99.999
                else:
                    helion_ms, sdpa_ms = 1.0, 1.0096
                for timer in ("event_ms", "wall_ms"):
                    pair["times"]["helion"][timer] = helion_ms
                    pair["times"]["sdpa"][timer] = sdpa_ms
        result = self.validate_timing_campaigns(
            campaigns, summary=timing_summary_from_campaigns(campaigns)
        )
        directional = result["sub_half_percent_directional"]
        self.assertEqual(directional["classification"], "inconclusive")
        self.assertEqual(
            directional["evidence_directions"]["event_confidence_interval"],
            "gain",
        )
        self.assertEqual(
            directional["evidence_directions"]["plotted_marginal_median"],
            "regression",
        )
        self.assertFalse(directional["agreement_checks"]["plotted_marginal_median"])

    def test_inconclusive_note_reports_only_point_estimate(self) -> None:
        notes = publish_results.paired_estimate_notes(
            0.1,
            [-0.2, 0.4],
            {"classification": "inconclusive"},
        )
        self.assertIn("point estimate +0.100%", notes[0])
        self.assertIn("conditional on the two observed campaign strata", notes[0])
        self.assertEqual(
            notes[1],
            "Sub-0.5% directional classification: inconclusive; only the point "
            "estimate and interval are reported.",
        )

    def test_write_or_reload_failure_preserves_all_existing_payloads(self) -> None:
        real_load_object = publish_results.load_object
        for failure in ("write", "reload"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                fixture = self.write_fixture(root)
                baseline_dir = fixture[0]
                output_dir = root / "output"
                output_dir.mkdir()
                expected: dict[Path, bytes] = {}
                for baseline_path in baseline_dir.glob("*.json"):
                    output = output_dir / baseline_path.name
                    output.write_bytes(f"old:{output.name}".encode())
                    expected[output] = output.read_bytes()
                args = self.publication_args(fixture, output_dir, overwrite=True)

                def fail_staged_reload(path: Path) -> dict[str, Any]:
                    if ".publish-stage-" in path.name:
                        raise RuntimeError("injected staged reload failure")
                    return real_load_object(path)

                patch = (
                    mock.patch.object(
                        publish_results,
                        "atomic_write_json",
                        side_effect=OSError("injected staged write failure"),
                    )
                    if failure == "write"
                    else mock.patch.object(
                        publish_results,
                        "load_object",
                        side_effect=fail_staged_reload,
                    )
                )
                with (
                    patch,
                    self.assertRaisesRegex(
                        (OSError, RuntimeError), f"injected staged {failure} failure"
                    ),
                ):
                    publish_results.publish(args)
                self.assertEqual(
                    {path: path.read_bytes() for path in expected}, expected
                )
                self.assert_no_publication_staging(root)

    def test_renderer_failure_preserves_payloads_and_render_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.write_fixture(root)
            baseline_dir = fixture[0]
            output_dir = root / "output"
            output_dir.mkdir()
            expected: dict[Path, bytes] = {}
            for baseline_path in baseline_dir.glob("*.json"):
                output = output_dir / baseline_path.name
                output.write_bytes(f"old:{output.name}".encode())
                expected[output] = output.read_bytes()
            render_outputs = (
                root / "render" / "results.csv",
                root / "render" / "results.png",
                root / "render" / "summary.png",
            )
            render_outputs[0].parent.mkdir()
            for output in render_outputs:
                output.write_bytes(f"old:{output.name}".encode())
                expected[output] = output.read_bytes()
            args = self.publication_args(fixture, output_dir, overwrite=True)
            with (
                mock.patch.object(
                    publish_results.subprocess,
                    "run",
                    side_effect=subprocess.CalledProcessError(1, ["renderer"]),
                ),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                publish_results.publish(args, render_outputs=render_outputs)
            self.assertEqual({path: path.read_bytes() for path in expected}, expected)
            self.assert_no_publication_staging(root)

    def test_success_replaces_payloads_and_render_outputs_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.write_fixture(root)
            baseline_dir = fixture[0]
            output_dir = root / "output"
            output_dir.mkdir()
            payload_paths = []
            for baseline_path in baseline_dir.glob("*.json"):
                output = output_dir / baseline_path.name
                output.write_text("old payload")
                payload_paths.append(output)
            render_outputs = (
                root / "render" / "results.csv",
                root / "render" / "results.png",
                root / "render" / "summary.png",
            )
            render_outputs[0].parent.mkdir()
            for output in render_outputs:
                output.write_text("old render")
            args = self.publication_args(fixture, output_dir, overwrite=True)
            rendered_payload_paths: list[Path] = []

            def render(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess:
                merge_index = command.index("--merge-json") + 1
                csv_index = command.index("--csv-output")
                rendered_payload_paths.extend(
                    Path(value) for value in command[merge_index:csv_index]
                )
                for flag, contents in (
                    ("--csv-output", b"new csv"),
                    ("--plot-output", b"new plot"),
                    ("--summary-plot-output", b"new summary"),
                ):
                    Path(command[command.index(flag) + 1]).write_bytes(contents)
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(
                publish_results.subprocess, "run", side_effect=render
            ):
                published = publish_results.publish(args, render_outputs=render_outputs)
            self.assertEqual(set(published), set(payload_paths))
            self.assertTrue(
                all(".publish-stage-" in path.name for path in rendered_payload_paths)
            )
            for output in payload_paths:
                self.assertEqual(
                    json.loads(output.read_text())["shape"],
                    json.loads((baseline_dir / output.name).read_text())["shape"],
                )
            self.assertEqual(render_outputs[0].read_bytes(), b"new csv")
            self.assertEqual(render_outputs[1].read_bytes(), b"new plot")
            self.assertEqual(render_outputs[2].read_bytes(), b"new summary")
            self.assert_no_publication_staging(root)

    def test_commit_failure_rolls_back_every_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destinations = [root / f"output-{index}.txt" for index in range(3)]
            staged = {
                destination: root / f"stage-{index}.txt"
                for index, destination in enumerate(destinations)
            }
            for index, destination in enumerate(destinations):
                destination.write_text(f"old-{index}")
                staged[destination].write_text(f"new-{index}")
            real_replace = Path.replace
            failed = False

            def fail_second_install(source: Path, target: Path) -> Path:
                nonlocal failed
                if source == staged[destinations[1]] and not failed:
                    failed = True
                    raise OSError("injected commit failure")
                return real_replace(source, target)

            with (
                mock.patch.object(Path, "replace", new=fail_second_install),
                self.assertRaisesRegex(OSError, "injected commit failure"),
            ):
                publish_results.commit_staged_outputs(staged, overwrite=True)
            self.assertEqual(
                [path.read_text() for path in destinations],
                ["old-0", "old-1", "old-2"],
            )
            self.assertEqual(
                list(root.glob(".*.publish-backup-*")),
                [],
            )

    def test_rejects_overwrite_aliases_and_evidence_tree_destinations(self) -> None:
        for collision in (
            "paired_raw",
            "renderer",
            "heldout_manifest",
            "generalization_manifest",
            "strict_tree",
            "heldout_tree",
            "generalization_tree",
        ):
            with (
                self.subTest(collision=collision),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                fixture = self.write_fixture(root)
                output_dir = root / "output"
                args = self.publication_args(fixture, output_dir, overwrite=True)
                protected = {
                    fixture[3]: fixture[3].read_bytes(),
                    args.renderer: args.renderer.read_bytes(),
                }
                if collision in {
                    "strict_tree",
                    "heldout_tree",
                    "generalization_tree",
                }:
                    evidence_root = {
                        "strict_tree": fixture[1],
                        "heldout_tree": args.heldout_artifact_root,
                        "generalization_tree": args.generalization_artifact_root,
                    }[collision]
                    args.output_payload_dir = evidence_root / "published"
                    with self.assertRaisesRegex(
                        RuntimeError, "inside an evidence tree"
                    ):
                        publish_results.publish(args)
                else:
                    collided = {
                        "paired_raw": fixture[3],
                        "renderer": args.renderer,
                        "heldout_manifest": args.heldout_manifest,
                        "generalization_manifest": args.generalization_manifest,
                    }[collision]
                    render_outputs = (
                        collided,
                        root / "render" / "results.png",
                        root / "render" / "summary.png",
                    )
                    with self.assertRaisesRegex(RuntimeError, "aliases evidence input"):
                        publish_results.publish(args, render_outputs=render_outputs)
                self.assertEqual(
                    {path: path.read_bytes() for path in protected}, protected
                )
                self.assert_no_publication_staging(root)

    def test_rejects_missing_heldout_evidence(self) -> None:
        for missing in ("manifest", "artifact_root"):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                fixture = self.write_fixture(root)
                args = self.publication_args(fixture, root / "output", overwrite=False)
                if missing == "manifest":
                    args.heldout_manifest.unlink()
                    message = "publication evidence files are missing"
                else:
                    (
                        args.heldout_artifact_root / "canonical_heldout_manifest.csv"
                    ).unlink()
                    args.heldout_artifact_root.rmdir()
                    message = "publication evidence roots are missing"
                with self.assertRaisesRegex(RuntimeError, message):
                    publish_results.publish(args)

    def test_rejects_missing_generalization_evidence(self) -> None:
        for missing in ("manifest", "artifact_root"):
            with (
                self.subTest(missing=missing),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                fixture = self.write_fixture(root)
                args = self.publication_args(fixture, root / "output", overwrite=False)
                if missing == "manifest":
                    args.generalization_manifest.unlink()
                    message = "publication evidence files are missing"
                else:
                    (
                        args.generalization_artifact_root
                        / "canonical_generalization_manifest.csv"
                    ).unlink()
                    args.generalization_artifact_root.rmdir()
                    message = "publication evidence roots are missing"
                with self.assertRaisesRegex(RuntimeError, message):
                    publish_results.publish(args)

    def test_rejects_tampered_heldout_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.write_fixture(root)
            args = self.publication_args(fixture, root / "output", overwrite=False)
            args.heldout_manifest.write_text(args.heldout_manifest.read_text() + "\n")
            with self.assertRaisesRegex(
                RuntimeError, "regenerated heldout manifest differs"
            ):
                publish_results.publish(args)

    def test_rejects_generalization_manifest_byte_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.write_fixture(root)
            args = self.publication_args(fixture, root / "output", overwrite=False)
            args.generalization_manifest.write_bytes(
                args.generalization_manifest.read_bytes() + b"\n"
            )
            with self.assertRaisesRegex(
                RuntimeError, "regenerated generalization manifest differs"
            ):
                publish_results.publish(args)

    def test_rejects_minimal_generalization_campaign_for_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.write_fixture(root)
            args = self.publication_args(fixture, root / "output", overwrite=False)
            minimal = SimpleNamespace(
                artifact_root=args.generalization_artifact_root,
                cases=(None,) * 6,
                run_specs=(None,) * 18,
            )
            with (
                mock.patch.object(
                    publish_results.validate_generalization_campaign,
                    "validate_campaign",
                    return_value=minimal,
                ),
                self.assertRaisesRegex(RuntimeError, "broad 65-search"),
            ):
                publish_results.publish(args)

    def test_generalization_validator_and_worker_are_immutable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.write_fixture(root)
            args = self.publication_args(fixture, root / "output", overwrite=False)
            paths = publish_results.publication_evidence_paths(
                args, include_renderer=False
            )
            setup_dir = Path(publish_results.__file__).resolve().parent
            self.assertIn(
                (setup_dir / "validate_generalization_campaign.py").resolve(), paths
            )
            self.assertIn(
                (setup_dir / "remeasure_generalization_winners.py").resolve(), paths
            )
            self.assertIn(args.generalization_manifest.resolve(), paths)
            self.assertIn(
                (
                    args.generalization_artifact_root
                    / "canonical_generalization_manifest.csv"
                ).resolve(),
                paths,
            )

    def test_rejects_heldout_manifest_with_wrong_all8_linkage(self) -> None:
        for field, value, message in (
            (
                "all8_reference_manifest_sha256",
                "f" * 64,
                "all8 strict manifest digest",
            ),
            ("version", "wrong-version", "all8 strict version"),
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                fixture = self.write_fixture(root)
                args = self.publication_args(fixture, root / "output", overwrite=False)
                with args.heldout_manifest.open(newline="") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    row[field] = value
                output = io.StringIO(newline="")
                writer = csv.DictWriter(
                    output,
                    fieldnames=publish_results.HELDOUT_MANIFEST_FIELDS,
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
                contents = output.getvalue()
                args.heldout_manifest.write_text(contents)
                (
                    args.heldout_artifact_root / "canonical_heldout_manifest.csv"
                ).write_text(contents)
                with self.assertRaisesRegex(RuntimeError, message):
                    publish_results.publish(args)

    def test_evidence_metadata_mutation_before_commit_preserves_destinations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.write_fixture(root)
            output_dir = root / "output"
            output_dir.mkdir()
            expected: dict[Path, bytes] = {}
            for baseline_path in fixture[0].glob("*.json"):
                output = output_dir / baseline_path.name
                output.write_bytes(f"old:{output.name}".encode())
                expected[output] = output.read_bytes()
            render_outputs = (
                root / "render" / "results.csv",
                root / "render" / "results.png",
                root / "render" / "summary.png",
            )
            render_outputs[0].parent.mkdir()
            for output in render_outputs:
                output.write_bytes(f"old:{output.name}".encode())
                expected[output] = output.read_bytes()
            args = self.publication_args(fixture, output_dir, overwrite=True)

            def mutate_evidence(
                command: list[str], **_kwargs: object
            ) -> subprocess.CompletedProcess:
                for flag in (
                    "--csv-output",
                    "--plot-output",
                    "--summary-plot-output",
                ):
                    Path(command[command.index(flag) + 1]).write_text("rendered")
                heldout_artifact = (
                    args.heldout_artifact_root / "canonical_heldout_manifest.csv"
                )
                stat = heldout_artifact.stat()
                os.utime(
                    heldout_artifact,
                    ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
                )
                return subprocess.CompletedProcess(command, 0)

            with (
                mock.patch.object(
                    publish_results.subprocess, "run", side_effect=mutate_evidence
                ),
                self.assertRaisesRegex(
                    RuntimeError, "publication evidence changed during validation"
                ),
            ):
                publish_results.publish(args, render_outputs=render_outputs)
            self.assertEqual({path: path.read_bytes() for path in expected}, expected)
            self.assert_no_publication_staging(root)

    def test_generalization_mutation_before_commit_preserves_destinations(self) -> None:
        for evidence_kind in ("artifact", "manifest"):
            with (
                self.subTest(evidence_kind=evidence_kind),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                fixture = self.write_fixture(root)
                output_dir = root / "output"
                output_dir.mkdir()
                expected: dict[Path, bytes] = {}
                for baseline_path in fixture[0].glob("*.json"):
                    output = output_dir / baseline_path.name
                    output.write_bytes(f"old:{output.name}".encode())
                    expected[output] = output.read_bytes()
                render_outputs = (
                    root / "render" / "results.csv",
                    root / "render" / "results.png",
                    root / "render" / "summary.png",
                )
                render_outputs[0].parent.mkdir()
                for output in render_outputs:
                    output.write_bytes(f"old:{output.name}".encode())
                    expected[output] = output.read_bytes()
                args = self.publication_args(fixture, output_dir, overwrite=True)
                evidence = (
                    args.generalization_artifact_root
                    / "canonical_generalization_manifest.csv"
                    if evidence_kind == "artifact"
                    else args.generalization_manifest
                )

                def mutate_evidence(
                    command: list[str], evidence: Path = evidence, **_kwargs: object
                ) -> subprocess.CompletedProcess:
                    for flag in (
                        "--csv-output",
                        "--plot-output",
                        "--summary-plot-output",
                    ):
                        Path(command[command.index(flag) + 1]).write_text("rendered")
                    evidence.write_bytes(evidence.read_bytes() + b"mutated")
                    return subprocess.CompletedProcess(command, 0)

                with (
                    mock.patch.object(
                        publish_results.subprocess,
                        "run",
                        side_effect=mutate_evidence,
                    ),
                    self.assertRaisesRegex(
                        RuntimeError,
                        "publication evidence changed during validation",
                    ),
                ):
                    publish_results.publish(args, render_outputs=render_outputs)
                self.assertEqual(
                    {path: path.read_bytes() for path in expected}, expected
                )
                self.assert_no_publication_staging(root)

    def test_rejects_unpinned_pythonpath_policy_and_record(self) -> None:
        for location in ("policy", "record"):
            with (
                self.subTest(location=location),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                fixture = self.write_fixture(root)
                paired_raw = fixture[3]
                if location == "policy":
                    evidence = fixture[5]
                    payload = json.loads(evidence.read_text())
                    payload["worker_environment_policy"]["controlled_values"][
                        "PYTHONPATH"
                    ] = "/ambient"
                    digest_field = "static_validation_sha256"
                    message = "worker PYTHONPATH policy"
                else:
                    evidence = fixture[4]
                    payload = json.loads(evidence.read_text())
                    payload["records"][0]["controlled_environment"]["PYTHONPATH"] = (
                        "/ambient"
                    )
                    digest_field = "run_manifest_sha256"
                    message = "run record .*: PYTHONPATH"
                evidence.write_text(json.dumps(payload, sort_keys=True) + "\n")
                raw = json.loads(paired_raw.read_text())
                raw[digest_field] = publish_results.file_sha256(evidence)
                paired_raw.write_text(json.dumps(raw, sort_keys=True) + "\n")
                args = self.publication_args(fixture, root / "output", overwrite=False)
                with self.assertRaisesRegex(RuntimeError, message):
                    publish_results.publish(args)

    def test_rejects_invalid_python_pycache_policy_and_records(self) -> None:
        cases = (
            ("policy_missing", "worker PYTHONPYCACHEPREFIX policy"),
            ("policy_tamper", "worker PYTHONPYCACHEPREFIX policy"),
            ("record_prefix_missing", "invalid PYTHONPYCACHEPREFIX"),
            ("record_cache_missing", "invalid HELION_CACHE_DIR"),
            ("record_mismatch", "PYTHONPYCACHEPREFIX location"),
            (
                "record_inside_checkout",
                "PYTHONPYCACHEPREFIX must be outside runtime checkout",
            ),
        )
        for case, message in cases:
            with (
                self.subTest(case=case),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                fixture = self.write_fixture(root)
                paired_raw = fixture[3]
                if case.startswith("policy_"):
                    evidence = fixture[5]
                    payload = json.loads(evidence.read_text())
                    controlled = payload["worker_environment_policy"][
                        "controlled_values"
                    ]
                    if case == "policy_missing":
                        controlled.pop("PYTHONPYCACHEPREFIX")
                    else:
                        controlled["PYTHONPYCACHEPREFIX"] = "shared"
                    digest_field = "static_validation_sha256"
                else:
                    evidence = fixture[4]
                    payload = json.loads(evidence.read_text())
                    controlled = payload["records"][0]["controlled_environment"]
                    if case == "record_prefix_missing":
                        controlled.pop("PYTHONPYCACHEPREFIX")
                    elif case == "record_cache_missing":
                        controlled.pop("HELION_CACHE_DIR")
                    elif case == "record_mismatch":
                        controlled["PYTHONPYCACHEPREFIX"] = "/ambient/pycache"
                    else:
                        cache_dir = (
                            Path(
                                payload["records"][0]["controlled_environment"][
                                    "PYTHONPATH"
                                ]
                            )
                            / "cache"
                        )
                        controlled["HELION_CACHE_DIR"] = str(cache_dir)
                        controlled["PYTHONPYCACHEPREFIX"] = str(cache_dir / "pycache")
                    digest_field = "run_manifest_sha256"
                evidence.write_text(json.dumps(payload, sort_keys=True) + "\n")
                raw = json.loads(paired_raw.read_text())
                raw[digest_field] = publish_results.file_sha256(evidence)
                paired_raw.write_text(json.dumps(raw, sort_keys=True) + "\n")
                args = self.publication_args(fixture, root / "output", overwrite=False)
                with self.assertRaisesRegex(RuntimeError, message):
                    publish_results.publish(args)

    def test_rejects_stale_manifest_result_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            strict_path = min(strict_root.rglob("result.json"))
            strict_path.write_text(strict_path.read_text() + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "regenerated strict manifest"):
                publish_results.publish(args)

    def test_strict_manifest_authenticates_terminal_refinement_fields(self) -> None:
        for field, expected in (
            (
                "terminal_refinement_policy_sha256",
                "terminal refinement policy digest",
            ),
            (
                "terminal_coordinate_surface_sha256",
                "terminal coordinate surface digest",
            ),
            ("terminal_refinement_sha256", "terminal refinement digest"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.write_fixture(root)
                strict_root = fixture[1]
                strict_manifest = fixture[2]
                with strict_manifest.open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                    fieldnames = reader.fieldnames
                assert fieldnames is not None
                rows[0][field] = "0" * 64
                with strict_manifest.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

                strict_results = publish_results.index_strict_results(strict_root)
                with self.assertRaisesRegex(RuntimeError, expected):
                    publish_results.index_strict_manifest(
                        strict_manifest, strict_root, strict_results
                    )

    def test_rejects_stale_manifest_source_ledger_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            ledger_path = min(strict_root.rglob("autotune.sources.csv"))
            ledger_path.write_text(ledger_path.read_text() + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "regenerated strict manifest"):
                publish_results.publish(args)

    def test_rejects_paired_source_that_does_not_match_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            raw = json.loads(paired_raw.read_text())
            raw["results"][0]["provenance"]["selected_source_sha256"] = "f" * 64
            paired_raw.write_text(json.dumps(raw) + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "portable provenance"):
                publish_results.publish(args)

    def test_rejects_incorrect_flop_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            raw = json.loads(paired_raw.read_text())
            raw["results"][0]["flop_model"]["flops"] *= 2
            paired_raw.write_text(json.dumps(raw) + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "paired raw regeneration"):
                publish_results.publish(args)

    def test_rejects_reversed_top_level_campaign_seed_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            raw = json.loads(paired_raw.read_text())
            raw["protocol"]["campaign_seeds"].reverse()
            paired_raw.write_text(json.dumps(raw) + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "run campaign seeds"):
                publish_results.publish(args)

    def test_rejects_one_config_disguised_as_full_tuning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            strict_path = min(strict_root.rglob("result.json"))
            strict = json.loads(strict_path.read_text())
            trial = strict["helion_overrides"]["autotune_provenance"]["trials"][0]
            trial["num_configs_tested"] = 1
            trial["num_successful_candidate_measurements"] = 1
            trial["num_unique_sources"] = 1
            strict_path.write_text(json.dumps(strict) + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "fewer than 100"):
                publish_results.publish(args)

    def test_rejects_invalid_bootstrap_confidence_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            raw = json.loads(paired_raw.read_text())
            raw["results"][0]["summary"]["event"][
                "paired_log_ratio_stratified_bootstrap_95_ci_pct"
            ] = [10.0, 8.0]
            paired_raw.write_text(json.dumps(raw) + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "paired raw regeneration"):
                publish_results.publish(args)

    def test_rejects_unvalidated_paired_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            raw = json.loads(paired_raw.read_text())
            raw["results"][0]["provenance"]["strict_full_autotune_validated"] = False
            paired_raw.write_text(json.dumps(raw) + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "portable provenance"):
                publish_results.publish(args)

    def test_rejects_machine_local_path_in_paired_raw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.write_fixture(root)
            args = self.publication_args(fixture, root / "output", overwrite=False)
            raw = json.loads(args.paired_raw.read_text())
            raw["results"][0]["environment"]["helion_module"] = (
                "/local/checkout/helion/__init__.py"
            )
            args.paired_raw.write_text(json.dumps(raw) + "\n")

            with self.assertRaisesRegex(RuntimeError, "absolute path is not portable"):
                publish_results.publish(args)

    def test_rejects_paired_environment_not_backed_by_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.write_fixture(root)
            args = self.publication_args(fixture, root / "output", overwrite=False)
            raw = json.loads(args.paired_raw.read_text())
            result = raw["results"][0]
            result["environment"]["torch_version"] = "tampered"
            for environment in result["campaign_environments"]:
                environment["torch_version"] = "tampered"
            args.paired_raw.write_text(json.dumps(raw) + "\n")

            with self.assertRaisesRegex(RuntimeError, "portable environment"):
                publish_results.publish(args)

    def test_rejects_dummy_run_and_static_hashes(self) -> None:
        for field, message in (
            ("run_manifest_sha256", "run manifest digest"),
            ("static_validation_sha256", "static validation digest"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (
                    baseline_dir,
                    strict_root,
                    strict_manifest,
                    paired_raw,
                    run_manifest,
                    static_validation,
                ) = self.write_fixture(root)
                raw = json.loads(paired_raw.read_text())
                raw[field] = "0" * 64
                paired_raw.write_text(json.dumps(raw) + "\n")
                args = argparse.Namespace(
                    baseline_payload_dir=baseline_dir,
                    strict_artifact_root=strict_root,
                    strict_manifest=strict_manifest,
                    heldout_artifact_root=strict_root.parent / "heldout",
                    heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                    generalization_artifact_root=strict_root.parent / "generalization",
                    generalization_manifest=strict_root.parent
                    / "generalization_manifest.csv",
                    paired_raw=paired_raw,
                    run_manifest=run_manifest,
                    static_validation=static_validation,
                    output_payload_dir=root / "output",
                    raw_artifact_label="plots/generalized_full_autotune/raw.json",
                    overwrite=False,
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    publish_results.publish(args)

    def test_rejects_duplicate_regeneration_and_correctness_campaigns(self) -> None:
        for field in ("regenerated_kernels", "correctness"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (
                    baseline_dir,
                    strict_root,
                    strict_manifest,
                    paired_raw,
                    run_manifest,
                    static_validation,
                ) = self.write_fixture(root)
                raw = json.loads(paired_raw.read_text())
                raw["results"][0][field][1]["campaign"] = 1
                paired_raw.write_text(json.dumps(raw) + "\n")
                args = argparse.Namespace(
                    baseline_payload_dir=baseline_dir,
                    strict_artifact_root=strict_root,
                    strict_manifest=strict_manifest,
                    heldout_artifact_root=strict_root.parent / "heldout",
                    heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                    generalization_artifact_root=strict_root.parent / "generalization",
                    generalization_manifest=strict_root.parent
                    / "generalization_manifest.csv",
                    paired_raw=paired_raw,
                    run_manifest=run_manifest,
                    static_validation=static_validation,
                    output_payload_dir=root / "output",
                    raw_artifact_label="plots/generalized_full_autotune/raw.json",
                    overwrite=False,
                )
                with self.assertRaisesRegex(RuntimeError, "duplicate campaign 1"):
                    publish_results.publish(args)

    def test_rejects_invalid_source_ledger_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            ledger_path = min(strict_root.rglob("autotune.sources.csv"))
            with ledger_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["status"] = "ok"
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output,
                fieldnames=publish_results.SOURCE_LEDGER_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            ledger_path.write_text(output.getvalue())
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "selected source"):
                publish_results.publish(args)

    def test_rejects_fabricated_full_tuning_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            ledger_path = min(strict_root.rglob("autotune.sources.csv"))
            with ledger_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            output = io.StringIO(newline="")
            writer = csv.DictWriter(
                output,
                fieldnames=publish_results.SOURCE_LEDGER_FIELDS,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows[:2])
            ledger_path.write_text(output.getvalue())
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "tested count"):
                publish_results.publish(args)

    def test_rejects_changed_generated_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            source_path = min(run_manifest.parent.rglob("*.py.txt"))
            source_path.write_text(source_path.read_text() + "# changed\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "digest"):
                publish_results.publish(args)

    def test_rejects_coordinated_static_plan_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            static = json.loads(static_validation.read_text())
            static["planned_commands"][0]["command"].append("--changed")
            static_validation.write_text(json.dumps(static) + "\n")
            raw = json.loads(paired_raw.read_text())
            raw["static_validation_sha256"] = publish_results.file_sha256(
                static_validation
            )
            paired_raw.write_text(json.dumps(raw) + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "static plan .* command"):
                publish_results.publish(args)

    def test_rejects_changed_harness_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            raw = json.loads(paired_raw.read_text())
            raw["harness_sha256"]["paired_worker.py"] = "0" * 64
            paired_raw.write_text(json.dumps(raw) + "\n")
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=root / "output",
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            with self.assertRaisesRegex(RuntimeError, "raw harness identity"):
                publish_results.publish(args)

    def test_rejects_invalid_peaky_stress_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _baseline, _strict, _manifest, paired_raw, _run, _static = (
                self.write_fixture(root)
            )
            raw = json.loads(paired_raw.read_text())
            valid = raw["results"][0]["correctness"][0]["post_timing_peaky_logits"]
        for message, update in (
            (
                "max_abs failed exclusive gate",
                lambda value: value["helion_vs_cudnn_sdpa"].update({"max_abs": 0.01}),
            ),
            (
                "mismatch fraction does not match counts",
                lambda value: value["helion_vs_cudnn_sdpa"].update(
                    {"mismatch_count": 1}
                ),
            ),
            (
                "repeat differences",
                lambda value: value["helion_exact_repeatability"].update(
                    {"different": 1}
                ),
            ),
        ):
            with self.subTest(message=message):
                changed = copy.deepcopy(valid)
                update(changed)
                with self.assertRaisesRegex(RuntimeError, message):
                    publish_results.validate_peaky_stress(changed, context="test")

    def test_non_target_validator_detects_mutation(self) -> None:
        key = next(iter(publish_results.EXPECTED_SHAPES))
        original = baseline_payload(key)
        changed = copy.deepcopy(original)
        changed["results"][0]["version"] = "mutated"
        with self.assertRaisesRegex(RuntimeError, "non-target baseline results"):
            publish_results.validate_non_target_results(
                original, changed, context="test"
            )

    def test_cute_backend_versions_must_match_publication_runtime(self) -> None:
        key = next(iter(publish_results.EXPECTED_SHAPES))
        payload = baseline_payload(key)
        next(
            result for result in payload["results"] if result["impl"] == "helion-cute"
        ).update({"version": VERSION, "version_label": VERSION_LABEL})
        payload["results"].append(
            {
                "impl": "fa4",
                "accuracy": "PASS",
                "version": "FlashAttention fa4-v4.0.0; CuTe 4.6.1",
                "version_label": "fa4-v4.0.0 / CuTe 4.6.1",
            }
        )

        with self.assertRaisesRegex(RuntimeError, "fa4 CuTe version"):
            publish_results.validate_cute_backend_versions(payload, context="test")

        payload["results"][-1]["version"] = (
            "FlashAttention fa4-v4.0.0; CuTe "
            + publish_results.build_strict_manifest.EXPECTED_CUTE_VERSION
        )
        with self.assertRaisesRegex(RuntimeError, "fa4 CuTe version label"):
            publish_results.validate_cute_backend_versions(payload, context="test")

        payload["results"][-1]["version_label"] = (
            "fa4-v4.0.0 / CuTe "
            + publish_results.build_strict_manifest.EXPECTED_CUTE_VERSION
        )
        payload["results"].append(
            {
                "impl": "kernelagent-closed-1x",
                "accuracy": "FAIL",
                "version": "KernelAgent v3; CuTe 4.5.1",
                "version_label": "KernelAgent v3 / CuTe 4.5.1",
            }
        )
        publish_results.validate_cute_backend_versions(payload, context="test")

    def test_cute_backend_versions_accept_harness_fa4_and_flex_labels(self) -> None:
        key = next(iter(publish_results.EXPECTED_SHAPES))
        expected = publish_results.build_strict_manifest.EXPECTED_CUTE_VERSION
        payload = baseline_payload(key)
        next(
            result for result in payload["results"] if result["impl"] == "helion-cute"
        ).update({"version": VERSION, "version_label": VERSION_LABEL})
        payload["results"].extend(
            [
                {
                    "impl": "fa4",
                    "accuracy": "PASS",
                    "version": f"FlashAttention fa4-v4.0.0; CuTe {expected}",
                    "version_label": f"fa4-v4.0.0; CuTe {expected}",
                },
                {
                    "impl": "flexattention-cute",
                    "accuracy": "PASS",
                    "version": (
                        f"PyTorch {TORCH_VERSION}; FA4 fa4-v4.0.0; CuTe {expected}"
                    ),
                    "version_label": (
                        f"PyTorch 2.13.0.dev20260506; FA4 fa4-v4.0.0; CuTe {expected}"
                    ),
                },
            ]
        )

        publish_results.validate_cute_backend_versions(payload, context="test")

        for impl in ("fa4", "flexattention-cute"):
            with self.subTest(impl=impl, field="version"):
                changed = copy.deepcopy(payload)
                result = next(
                    row for row in changed["results"] if row.get("impl") == impl
                )
                result["version"] = result["version"].replace(expected, "4.6.1")
                with self.assertRaisesRegex(RuntimeError, f"{impl} CuTe version"):
                    publish_results.validate_cute_backend_versions(
                        changed, context="test"
                    )
            with self.subTest(impl=impl, field="version_label"):
                changed = copy.deepcopy(payload)
                result = next(
                    row for row in changed["results"] if row.get("impl") == impl
                )
                result["version_label"] = result["version_label"].replace(
                    expected, "4.6.1"
                )
                with self.assertRaisesRegex(RuntimeError, f"{impl} CuTe version label"):
                    publish_results.validate_cute_backend_versions(
                        changed, context="test"
                    )

    def test_known_cute_backend_must_declare_cute_version(self) -> None:
        key = next(iter(publish_results.EXPECTED_SHAPES))
        payload = baseline_payload(key)
        next(
            result for result in payload["results"] if result["impl"] == "helion-cute"
        ).update({"version": VERSION, "version_label": VERSION_LABEL})
        payload["results"].append(
            {
                "impl": "flexattention-cute",
                "accuracy": "PASS",
                "version": "PyTorch dev",
                "version_label": "PyTorch dev",
            }
        )

        with self.assertRaisesRegex(RuntimeError, "flexattention-cute CuTe version"):
            publish_results.validate_cute_backend_versions(payload, context="test")

    def test_renderer_command_uses_published_payloads_and_labels(self) -> None:
        command = publish_results.renderer_command(
            Path("renderer.py"),
            [Path("a.json"), Path("b.json")],
            Path("out.csv"),
            Path("out.png"),
            Path("summary.png"),
            ["helion-cute=Helion"],
        )
        self.assertIn("--merge-json", command)
        self.assertIn(str(Path("a.json").resolve()), command)
        self.assertIn("--summary-plot-output", command)
        self.assertEqual(command[-2:], ["--plot-impl-label", "helion-cute=Helion"])

    def test_published_payloads_pass_existing_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                baseline_dir,
                strict_root,
                strict_manifest,
                paired_raw,
                run_manifest,
                static_validation,
            ) = self.write_fixture(root)
            output_dir = root / "output"
            args = argparse.Namespace(
                baseline_payload_dir=baseline_dir,
                strict_artifact_root=strict_root,
                strict_manifest=strict_manifest,
                heldout_artifact_root=strict_root.parent / "heldout",
                heldout_manifest=strict_root.parent / "heldout_manifest.csv",
                generalization_artifact_root=strict_root.parent / "generalization",
                generalization_manifest=strict_root.parent
                / "generalization_manifest.csv",
                paired_raw=paired_raw,
                run_manifest=run_manifest,
                static_validation=static_validation,
                output_payload_dir=output_dir,
                raw_artifact_label="plots/generalized_full_autotune/raw.json",
                overwrite=False,
            )
            payloads = publish_results.publish(args)
            with contextlib.chdir(root):
                command = publish_results.renderer_command(
                    publish_results.DEFAULT_RENDERER,
                    [path.relative_to(root) for path in payloads],
                    Path("out.csv"),
                    Path("out.png"),
                    Path("summary.png"),
                    [],
                )
            subprocess.run(
                command,
                cwd=publish_results.REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue((root / "out.csv").is_file())
            self.assertTrue((root / "out.png").is_file())
            self.assertTrue((root / "summary.png").is_file())


if __name__ == "__main__":
    unittest.main()
