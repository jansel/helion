from __future__ import annotations

import copy
import csv
import io
import json
import math
from operator import itemgetter
from pathlib import Path
import sys
import tempfile
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_strict_manifest as manifest

if TYPE_CHECKING:
    import pytest


class CurrentVerifier:
    """Expose the standalone v22 verifier through the old worker test interface."""

    def __getattr__(self, name: str) -> object:
        return getattr(manifest, name)

    @staticmethod
    def _validate_structural_qualification_phase(
        path: Path, provenance: dict[str, Any]
    ) -> dict[str, Any]:
        return manifest.validate_structural_qualification_phase(
            path, provenance, provenance["trials"][0]
        )

    _reconcile_structural_qualification_phase = staticmethod(
        manifest.reconcile_structural_qualification_phase
    )


paired_worker = CurrentVerifier()

FIXTURE_COMMIT = manifest.EXPECTED_MEASURED_COMMIT


def attention_flops(seq_len: int, causal: bool) -> float:
    flops = 4.0 * 2 * 32 * seq_len**2 * 64
    return flops * (0.5 if causal else 1.0)


def add_flash_normalization_context(
    provenance: dict[str, Any],
    trial: dict[str, Any],
    *,
    seq_len: int,
    causal: bool,
) -> None:
    shape = (2, 32, seq_len, 64)
    default_sha256 = provenance.setdefault("flash_fragment_default_sha256", "b" * 64)
    provenance.setdefault(
        "autotune_baseline_fn",
        (
            "examples.attention._causal_attention_output_baseline"
            if causal
            else "examples.attention._attention_output_baseline"
        ),
    )
    context = {
        "schema_version": 1,
        "backend": "cute",
        "config_spec_structural_fingerprint_sha256": "a" * 64,
        "default_config_sha256": default_sha256,
        "dtype": "torch.float16",
        "head_dim": 64,
        "num_kv": (seq_len + 127) // 128,
        "num_bh": 64,
        "tensor_4d_heads": 32,
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
    provenance["flash_normalization_context_sha256"] = manifest.canonical_sha256(
        context
    )
    trial.setdefault("input_shapes", repr([shape] * 3))
    trial.setdefault("dtypes", repr(["torch.float16"] * 3))


def add_clc_lane_catalog(
    provenance: dict[str, Any], entries: list[dict[str, Any]] | None = None
) -> None:
    catalog = [] if entries is None else entries
    provenance["flash_clc_lane_catalog"] = catalog
    provenance["flash_clc_lane_catalog_sha256"] = manifest.canonical_sha256(catalog)


def terminal_measurement(
    config_ids: list[str],
    timing_by_id: dict[str, float],
    *,
    target_ms: float,
    repeat_reference_perf_ms: float,
) -> dict[str, Any]:
    elapsed_ms = []
    indices = list(range(len(config_ids)))
    desired_calls = min(20_000, max(3, int(target_ms / repeat_reference_perf_ms)))
    desired_calls = max(2, desired_calls + desired_calls % 2)
    calls_per_sample = max(1, math.ceil(desired_calls / 64))
    sweep_count = math.ceil(desired_calls / calls_per_sample)
    if sweep_count % 2:
        sweep_count += 1
    total_calls = sweep_count * calls_per_sample
    for sweep_index in range(sweep_count):
        offset = (sweep_index // 2) % len(indices)
        rotated = indices[offset:] + indices[:offset]
        order = rotated if sweep_index % 2 == 0 else list(reversed(rotated))
        elapsed_ms.append([timing_by_id[config_ids[index]] for index in order])
    return {
        "base_order": config_ids,
        "target_ms": target_ms,
        "repeat_reference_perf_ms": repeat_reference_perf_ms,
        "sweep_count": sweep_count,
        "calls_per_sample": calls_per_sample,
        "total_calls": total_calls,
        "elapsed_ms": elapsed_ms,
        "median_ms": [
            {"config_id": config_id, "value": timing_by_id[config_id]}
            for config_id in config_ids
        ],
    }


def add_terminal_refinement(
    provenance: dict[str, Any],
    trial: dict[str, Any],
    phase: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    attempts: dict[str, dict[str, Any]] | None,
    *,
    default_perf: float,
) -> None:
    policy = manifest.expected_terminal_refinement_policy()
    provenance["flash_terminal_coordinate_refinement_policy"] = policy
    provenance["flash_terminal_coordinate_refinement_policy_sha256"] = (
        manifest.canonical_sha256(policy)
    )

    def succeeded(config_id: str) -> bool:
        attempt = attempts.get(config_id) if attempts is not None else None
        return attempt is None or attempt.get("status") in {"ok", "deduplicated"}

    selected = trial.get("selected_config")
    selected_id = (
        manifest.canonical_sha256(selected)[:16]
        if isinstance(selected, dict) and selected in configs.values()
        else None
    )
    initial_id = (
        selected_id if selected_id is not None and succeeded(selected_id) else None
    )
    if initial_id is None:
        initial_id = next(
            (config_id for config_id in configs if succeeded(config_id)),
            next(iter(configs)),
        )
        trial["selected_config"] = configs[initial_id]
        provenance["selected_config"] = configs[initial_id]
    initial_config = configs[initial_id]
    initial_leaf = manifest.structural_leaf(initial_config)
    assert initial_leaf is not None
    pipeline_key = manifest.FLASH_PIPELINE_QUALIFICATION_KEYS[0]
    pipeline_candidates = [
        (config_id, config)
        for config_id, config in configs.items()
        if config_id != initial_id
        and manifest.structural_leaf(config) == initial_leaf
        and pipeline_key in initial_config
        and config.get(pipeline_key) in {2, 3}
        and config[pipeline_key] != initial_config[pipeline_key]
    ]
    if pipeline_candidates:
        coordinate_key = pipeline_key
        candidate_id = next(
            (
                config_id
                for config_id, _config in pipeline_candidates
                if succeeded(config_id)
            ),
            pipeline_candidates[0][0],
        )
        active_values = [2, 3]
    else:
        candidate_id = next(
            (
                config_id
                for config_id, config in configs.items()
                if config_id != initial_id
                and manifest.structural_leaf(config) == initial_leaf
                and succeeded(config_id)
                and any(
                    key in initial_config
                    and initial_config[key] != value
                    and not isinstance(value, (list, dict))
                    for key, value in config.items()
                )
            ),
            next(
                config_id
                for config_id, config in configs.items()
                if config_id != initial_id
                and manifest.structural_leaf(config) == initial_leaf
                and any(
                    key in initial_config
                    and initial_config[key] != value
                    and not isinstance(value, (list, dict))
                    for key, value in config.items()
                )
            ),
        )
        candidate_config = configs[candidate_id]
        coordinate_key = next(
            key
            for key, value in candidate_config.items()
            if key in initial_config
            and initial_config[key] != value
            and not isinstance(value, (list, dict))
        )
        active_values = [
            initial_config[coordinate_key],
            candidate_config[coordinate_key],
        ]
    candidate_config = configs[candidate_id]
    surface = {
        "schema_version": manifest.EXPECTED_TERMINAL_SURFACE_SCHEMA_VERSION,
        "radius": manifest.EXPECTED_TERMINAL_REFINEMENT_RADIUS,
        "leaves": [
            {
                "leaf": leaf,
                "coordinates": [
                    {
                        "flat_index": 0,
                        "key": coordinate_key,
                        "sequence_index": None,
                        "fragment_type": "EnumFragment",
                        "overridden": False,
                        "active_values": active_values,
                        "neighbors_by_value": [
                            {
                                "from_value": active_values[0],
                                "to_values": [active_values[1]],
                            },
                            {
                                "from_value": active_values[1],
                                "to_values": [active_values[0]],
                            },
                        ],
                    }
                ],
            }
            for leaf in provenance["flash_structural_leaf_catalog"]
        ],
    }
    provenance["flash_terminal_coordinate_surface_catalog"] = surface
    provenance["flash_terminal_coordinate_surface_catalog_sha256"] = (
        manifest.canonical_sha256(surface)
    )

    initial_attempt = attempts.get(initial_id) if attempts is not None else None
    request = {
        "flat_index": 0,
        "key": coordinate_key,
        "sequence_index": None,
        "from_value": initial_config[coordinate_key],
        "to_value": candidate_config[coordinate_key],
        "outcome": "incumbent_alias",
        "config_id": initial_id,
    }
    terminal_manifest = {initial_id: {"config": initial_config}}
    preterminal_ids = sorted(configs)
    trial.setdefault(
        "num_configs_tested",
        sum(
            attempt.get("status") not in manifest.LEDGER_ALIAS_STATUSES
            for attempt in (attempts or {}).values()
        )
        or len(configs),
    )
    trial.setdefault("num_generations", 2)
    trial.setdefault(
        "selected_source_hash",
        (
            initial_attempt.get("source_hash")
            if initial_attempt is not None
            else manifest.canonical_sha256({"measurement_source": initial_id})
        ),
    )
    rounds = [
        {
            "round_index": 1,
            "incumbent_config_id": initial_id,
            "leaf": initial_leaf,
            "parent_config_ids": [initial_id],
            "parent_projections": [
                {"parent_config_id": initial_id, "coordinate_requests": [request]}
            ],
            "candidate_config_ids": [],
            "new_candidate_ids": [],
            "reused_candidate_ids": [],
            "intra_terminal_reused_candidate_ids": [],
            "prior_failed_candidate_ids": [],
            "candidate_results": [],
            "comparison_config_ids": [],
            "measurement": None,
            "round_best_config_id": initial_id,
            "selected_config_id": initial_id,
            "accepted": False,
            "improvement_fraction": 0.0,
            "beam_config_ids": [initial_id],
        }
    ]
    confirmation = {
        "candidate_config_ids": [initial_id],
        "measurement": None,
        "best_config_id": initial_id,
        "selected_config_id": initial_id,
        "accepted": False,
        "improvement_fraction": 0.0,
        "skipped_reason": "single_candidate",
    }
    transcript = {
        "schema_version": policy["schema_version"],
        "policy_version": policy["policy_version"],
        "lane_policy_version": policy["lane_policy_version"],
        "coordinate_policy": policy["coordinate_policy"],
        "measurement_policy": policy["measurement_policy"],
        "rounds_planned": policy["rounds"],
        "beam_width": policy["beam_width"],
        "maximum_projection_parent_count": 5,
        "projection_parent_count": 1,
        "rounds_started": len(rounds),
        "rounds_completed": len(rounds),
        "completed": True,
        "budget_exhausted": False,
        "termination_reason": "no_candidates",
        "search_generation": trial["num_generations"],
        "preterminal_num_configs_tested": trial["num_configs_tested"],
        "preterminal_registry_config_count": len(preterminal_ids),
        "preterminal_registry_config_ids_hash_policy": (
            manifest.EXPECTED_PRETERMINAL_REGISTRY_HASH_POLICY
        ),
        "preterminal_registry_config_ids_sha256": manifest.sha256_bytes(
            json.dumps(preterminal_ids, separators=(",", ":")).encode()
        ),
        "radius": policy["radius"],
        "minimum_improvement_fraction": policy["minimum_improvement_fraction"],
        "initial_incumbent_config_id": initial_id,
        "refined_config_id": initial_id,
        "final_config_id": initial_id,
        "projection_attempt_count": 1,
        "unique_candidate_count": 0,
        "new_candidate_count": 0,
        "reused_candidate_count": 0,
        "intra_terminal_reused_candidate_count": 0,
        "prior_failed_candidate_count": 0,
        "accepted_config_ids": [],
        "config_manifest_sha256": manifest.canonical_sha256(terminal_manifest),
        "config_manifest": terminal_manifest,
        "rounds": rounds,
        "confirmation": confirmation,
    }
    phase["terminal_coordinate_refinement"] = transcript


def terminal_refinement_summary(
    provenance: dict[str, Any], phase: dict[str, Any]
) -> dict[str, Any]:
    terminal = phase["terminal_coordinate_refinement"]
    summary = {
        key: terminal[key]
        for key in (
            "schema_version",
            "policy_version",
            "lane_policy_version",
            "coordinate_policy",
            "measurement_policy",
            "rounds_planned",
            "beam_width",
            "radius",
            "preterminal_num_configs_tested",
            "preterminal_registry_config_count",
            "preterminal_registry_config_ids_sha256",
            "projection_parent_count",
            "projection_attempt_count",
            "unique_candidate_count",
            "new_candidate_count",
            "reused_candidate_count",
            "intra_terminal_reused_candidate_count",
            "prior_failed_candidate_count",
            "initial_incumbent_config_id",
            "refined_config_id",
            "final_config_id",
            "accepted_config_ids",
            "config_manifest_sha256",
        )
    }
    summary.update(
        {
            "required_preterminal_candidate_count": 100,
            "preterminal_effective_candidate_count": terminal[
                "preterminal_num_configs_tested"
            ],
            "preterminal_successful_measurement_count": 100,
            "detached_direct_projection_count": 0,
            "paired_live_projection_required_count": terminal[
                "projection_attempt_count"
            ],
            "policy_sha256": provenance[
                "flash_terminal_coordinate_refinement_policy_sha256"
            ],
            "coordinate_surface_sha256": provenance[
                "flash_terminal_coordinate_surface_catalog_sha256"
            ],
            "rounds_sha256": manifest.canonical_sha256(terminal["rounds"]),
            "confirmation_sha256": manifest.canonical_sha256(terminal["confirmation"]),
            "transcript_sha256": manifest.canonical_sha256(terminal),
        }
    )
    return summary


def add_phase_config_identity(
    provenance: dict[str, Any],
    phase: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    *,
    attempts: dict[str, dict[str, Any]] | None = None,
    default_perf: float = 1.0,
) -> None:
    leaf_catalog = provenance["flash_structural_leaf_catalog"]
    for leaf in leaf_catalog:
        leaf.setdefault("softmax_disc", False)
    for entry in provenance["flash_pipeline_lane_catalog"]:
        entry.setdefault("softmax_disc", False)
    for entry in provenance.get("flash_clc_lane_catalog", []):
        entry.setdefault("softmax_disc", False)
    retained_family_cap = provenance.get("flash_structural_retained_family_cap")
    if retained_family_cap is None:
        retained_family_cap = manifest.EXPECTED_FULL_FLASH_RETAINED_FAMILIES
        provenance["flash_structural_retained_family_cap"] = retained_family_cap
    retained_family_limit = manifest.expected_retained_family_limit(
        leaf_catalog, retained_family_cap
    )
    provenance["flash_structural_retained_family_limit"] = retained_family_limit
    starting_path_limit = manifest.expected_starting_path_limit(
        leaf_catalog,
        retained_per_leaf=provenance["flash_structural_retained_candidates_per_leaf"],
        retained_family_limit=retained_family_limit,
    )
    provenance["flash_structural_starting_path_limit"] = starting_path_limit
    family_probe_path_limit = manifest.expected_family_probe_path_limit(
        leaf_catalog,
        retained_family_cap,
        manifest.EXPECTED_FAMILY_PROBE_GENERATIONS,
    )
    provenance["flash_structural_family_probe_generations"] = (
        manifest.EXPECTED_FAMILY_PROBE_GENERATIONS
    )
    provenance["flash_structural_family_probe_candidates_per_path"] = (
        manifest.EXPECTED_FAMILY_PROBE_CANDIDATES_PER_PATH
    )
    provenance["flash_structural_family_probe_path_limit"] = family_probe_path_limit
    provenance["flash_structural_maximum_path_capacity"] = max(
        starting_path_limit, family_probe_path_limit
    )
    phase["phase"] = "cute_flash_structural_qualification_v22"
    phase["cute_flash_lane_policy_version"] = manifest.EXPECTED_LANE_POLICY_VERSION
    phase["retained_family_cap"] = retained_family_cap
    phase["retained_family_limit"] = retained_family_limit
    phase["starting_path_limit"] = starting_path_limit
    phase["maximum_path_capacity"] = max(starting_path_limit, family_probe_path_limit)
    phase["family_probe_generations"] = manifest.EXPECTED_FAMILY_PROBE_GENERATIONS
    phase["family_probe_candidates_per_path"] = (
        manifest.EXPECTED_FAMILY_PROBE_CANDIDATES_PER_PATH
    )
    phase["family_probe_path_limit"] = family_probe_path_limit
    family_probe_required = bool(
        family_probe_path_limit and not phase.get("exact_space_exhausted")
    )
    phase["family_probe_required"] = family_probe_required
    phase["family_probe_complete"] = True
    phase["family_probe_generations_started"] = (
        manifest.EXPECTED_FAMILY_PROBE_GENERATIONS if family_probe_required else 0
    )
    phase["family_probe_generations_completed"] = (
        manifest.EXPECTED_FAMILY_PROBE_GENERATIONS if family_probe_required else 0
    )
    phase.setdefault("family_probe_paths", [])
    phase.setdefault("compound_catalog_complete", True)
    phase.setdefault("compound_catalog_errors", [])
    for key in ("leaf_results", "compound_transfers", "clc_families"):
        for result in phase.get(key, []):
            result.setdefault("softmax_disc", False)
    for family in phase.get("retained_families", []):
        family.setdefault("score_softmax_disc", False)
        for path in family["starting_paths"]:
            path.setdefault("softmax_disc", False)
    phase.pop("measurement_timeline", None)
    existing_terminal = phase.pop("terminal_coordinate_refinement", None)
    lanes_by_leaf = manifest.flash_pipeline_lane_catalog(
        Path("fixture.json"), provenance
    )
    qualified_by_id = {
        qualified["config_id"]: qualified
        for result in phase["leaf_results"]
        for qualified in result["qualified_results"]
    }

    def source_hash_for(config_id: str) -> str:
        attempt = attempts.get(config_id) if attempts is not None else None
        if attempt is not None and isinstance(attempt.get("source_hash"), str):
            return attempt["source_hash"]
        return manifest.canonical_sha256({"measurement_source": config_id})

    referenced_ids = set(phase["initial_config_ids"]) | set(
        phase["exact_space_config_ids"]
    )
    referenced_ids.update(qualified_by_id)
    for result in phase["compound_transfers"]:
        for transfer in result["transfers"]:
            referenced_ids.add(transfer["source_config_id"])
            referenced_ids.add(transfer["transferred_config_id"])
    for probe_path in phase["family_probe_paths"]:
        for round_record in probe_path["rounds"]:
            referenced_ids.update(round_record["candidate_ids"])
    phase["config_manifest"] = {
        config_id: {"config": configs[config_id]}
        for config_id in configs
        if config_id in referenced_ids
    }
    initial_results = []
    for config_id in phase["initial_config_ids"]:
        config = configs[config_id]
        leaf = manifest.structural_leaf(config)
        assert leaf is not None
        qualified = qualified_by_id.get(config_id)
        attempt = attempts.get(config_id) if attempts is not None else None
        status = (
            attempt["status"]
            if attempt is not None
            else qualified["status"]
            if qualified is not None
            else "ok"
        )
        attempt_perf = (
            attempt["perf_ms"]
            if attempt is not None
            else qualified["attempt_perf"]
            if qualified is not None
            else default_perf
        )
        selection_perf = (
            qualified["selection_perf"] if qualified is not None else attempt_perf
        )
        lanes = lanes_by_leaf[manifest.canonical_json(leaf)]
        initial_results.append(
            {
                "config_id": config_id,
                **leaf,
                "attempt_perf": attempt_perf,
                "selection_perf": selection_perf,
                "status": status,
                "source_hash": source_hash_for(config_id),
                "measurement_pass_index": 0,
                "pipeline_lanes": [
                    manifest.pipeline_lane_metric(lane)
                    for lane in lanes
                    if config.get(lane[0]) == lane[1]
                ],
            }
        )
    phase["initial_results"] = initial_results
    successful_initial_ids = [
        result["config_id"]
        for result in initial_results
        if result["status"] in {"ok", "deduplicated"}
    ]
    stable_coverage_seed_id = next(
        (
            config_id
            for config_id in successful_initial_ids
            if configs[config_id].get("cute_flash_pipeline_family") == "fa4_2cta"
            and configs[config_id].get("cute_flash_exp2_packet") == "1x1"
            and configs[config_id].get("cute_flash_kv_stage") == 2
            and configs[config_id].get("cute_flash_s_stage") == 2
            and configs[config_id].get("cute_flash_wait_hint") == -1
        ),
        None,
    )
    compiler_seed_ids = [
        stable_coverage_seed_id or successful_initial_ids[-1]
        if successful_initial_ids
        else phase["initial_config_ids"][-1]
    ]
    provenance["compiler_seed_config_count"] = 1
    provenance["compiler_seed_policy"] = {
        "schema_version": 1,
        "kind": "canonical_cute_flash",
        "heuristic_names": ["cute_flash_attention"],
        "raw_config_count": 1,
        "effective_config_ids": compiler_seed_ids,
        "effective_config_ids_sha256": manifest.canonical_sha256(compiler_seed_ids),
        "timeout_retry_repetitions": 3,
    }
    ordinary_leaves = [leaf for leaf in leaf_catalog if leaf["compound_packet"] is None]
    anchor_results = []
    for leaf in ordinary_leaves:
        anchor = next(
            result
            for result in initial_results
            if {
                "family": result["family"],
                "compound_packet": result["compound_packet"],
                "softmax_disc": result["softmax_disc"],
            }
            == leaf
        )
        anchor_results.append(
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
        )
    phase.update(
        {
            "schedule_anchor_design_source": (
                "live family x ordinary packet x softmax protocol from fragment defaults"
            ),
            "schedule_anchor_pass_planned": False,
            "schedule_anchor_pass_started": False,
            "schedule_anchor_count": len(anchor_results),
            "schedule_anchor_complete": True,
            "schedule_anchor_results": anchor_results,
        }
    )
    initial_id_set = set(phase["initial_config_ids"])
    states_by_id = {
        record["config_id"]: {
            key: record[key]
            for key in ("attempt_perf", "selection_perf", "status", "source_hash")
        }
        for record in initial_results
    }
    for config_id, qualified in qualified_by_id.items():
        states_by_id.setdefault(
            config_id,
            {
                key: qualified[key]
                for key in ("attempt_perf", "selection_perf", "status")
            }
            | {
                "source_hash": source_hash_for(config_id),
            },
        )
    pass_count = phase["qualification_passes_completed"]
    candidate_limit = phase["pipeline_candidate_limit_per_leaf_per_round"]
    pipeline_pass_count = max(
        (len(result["rounds"]) for result in phase["leaf_results"]), default=0
    )
    clc_families = phase["clc_families"]
    clc_witness_passes = int(any(result["planned_values"] for result in clc_families))
    clc_witness_repair_passes = max(
        (
            math.ceil(len(result["witness_repair_parent_decisions"]) / candidate_limit)
            for result in clc_families
        ),
        default=0,
    )
    clc_conditional_passes = int(
        any(result["conditional_values"] for result in clc_families)
    )
    clc_conditional_repair_passes = max(
        (
            math.ceil(
                len(result["conditional_repair_parent_decisions"]) / candidate_limit
            )
            for result in clc_families
        ),
        default=0,
    )
    clc_combination_passes = int(
        any(result["combination_required"] for result in clc_families)
    )
    witness_repair_start = pipeline_pass_count + clc_witness_passes
    post_witness_pass = witness_repair_start + clc_witness_repair_passes
    conditional_repair_start = post_witness_pass + clc_conditional_passes
    post_conditional_pass = conditional_repair_start + clc_conditional_repair_passes
    compound_source_pass = post_conditional_pass + clc_combination_passes
    introduction_pass_by_id = dict.fromkeys(initial_id_set, 0)
    for result in phase["leaf_results"]:
        for completion_pass, round_result in enumerate(result["rounds"], start=1):
            for config_id in round_result["candidate_config_ids"]:
                introduction_pass_by_id.setdefault(config_id, completion_pass)
    witness_completion_pass = pipeline_pass_count + clc_witness_passes
    for result in clc_families:
        for config_id in result["witness_config_ids"].values():
            introduction_pass_by_id.setdefault(config_id, witness_completion_pass)
        for offset, decision in enumerate(result["witness_repair_parent_decisions"]):
            completion_pass = witness_repair_start + offset // candidate_limit + 1
            for config_id in decision["generated_config_ids"]:
                introduction_pass_by_id.setdefault(config_id, completion_pass)
        for decision in result["conditional_parent_decisions"]:
            for config_id in decision["generated_config_ids"]:
                introduction_pass_by_id.setdefault(config_id, post_witness_pass + 1)
        for offset, decision in enumerate(
            result["conditional_repair_parent_decisions"]
        ):
            completion_pass = conditional_repair_start + offset // candidate_limit + 1
            for config_id in decision["generated_config_ids"]:
                introduction_pass_by_id.setdefault(config_id, completion_pass)
        if result["combination_required"]:
            for config_id in result["combination_candidate_ids"]:
                introduction_pass_by_id.setdefault(config_id, post_conditional_pass + 1)
    for result in phase["compound_transfers"]:
        for config_id in result["primary_transfer_config_ids"]:
            introduction_pass_by_id.setdefault(config_id, compound_source_pass + 1)
        for backfill_index, backfill in enumerate(result["backfill_rounds"]):
            for config_id in backfill["generated_config_ids"]:
                introduction_pass_by_id.setdefault(
                    config_id, compound_source_pass + 2 + backfill_index
                )

    successful_statuses = {"ok", "deduplicated"}
    retryable_statuses = {"error", "timeout", "peer_compilation_fail"}

    def decision_snapshot(config_id: str, pass_index: int) -> dict[str, Any]:
        if introduction_pass_by_id[config_id] > pass_index:
            return {
                "config_id": config_id,
                "attempt_perf": None,
                "selection_perf": None,
                "status": "unknown",
                "source_hash": None,
            }
        return {"config_id": config_id, **states_by_id[config_id]}

    def ranked_snapshots(config_ids: set[str], pass_index: int) -> list[dict[str, Any]]:
        return sorted(
            (decision_snapshot(config_id, pass_index) for config_id in config_ids),
            key=lambda snapshot: (
                (
                    snapshot["selection_perf"]
                    if snapshot["status"] in successful_statuses
                    else math.inf
                ),
                snapshot["config_id"],
            ),
        )

    clc_by_family = {result["family"]: result for result in clc_families}
    for transfer_result in phase["compound_transfers"]:
        family = transfer_result["family"]
        ordinary_leaf = {
            "family": family,
            "compound_packet": None,
            "softmax_disc": transfer_result["softmax_disc"],
        }
        candidate_ids = {
            config_id
            for config_id, state in states_by_id.items()
            if state["status"] in successful_statuses
            and introduction_pass_by_id[config_id] <= compound_source_pass
            and manifest.structural_leaf(configs[config_id]) == ordinary_leaf
        }
        clc_result = clc_by_family.get(family)
        combination_ids = (
            []
            if clc_result is None
            else [
                config_id
                for config_id in clc_result["combination_candidate_ids"]
                if any(
                    cell["config_id"] == config_id
                    and cell["status"] in successful_statuses
                    for cell in clc_result["combination_cells"]
                )
            ]
        )
        combination_set = set(combination_ids)
        candidate_snapshots = [
            *ranked_snapshots(combination_set, compound_source_pass),
            *ranked_snapshots(candidate_ids - combination_set, compound_source_pass),
        ]
        ordered_ids = [snapshot["config_id"] for snapshot in candidate_snapshots]
        selected_ids = [
            transfer["source_config_id"] for transfer in transfer_result["transfers"]
        ]
        selected_positions = [
            ordered_ids.index(config_id) for config_id in selected_ids
        ]
        assert selected_positions == sorted(selected_positions)
        attempted_ids = (
            ordered_ids[: selected_positions[-1] + 1]
            if len(selected_ids) == transfer_result["limit"]
            else ordered_ids
        )
        transfer_result["source_selection"] = {
            "candidate_results": candidate_snapshots,
            "combination_prefix_count": len(combination_ids),
            "attempted_config_ids": attempted_ids,
            "selected_config_ids": selected_ids,
        }
        primary_attempted_end = (
            attempted_ids.index(
                selected_ids[transfer_result["transfer_target_count"] - 1]
            )
            + 1
        )
        suffix = attempted_ids[primary_attempted_end:]
        for backfill_index, backfill in enumerate(transfer_result["backfill_rounds"]):
            backfill["attempted_source_config_ids"] = (
                suffix if backfill_index == 0 else []
            )

    for result in phase["leaf_results"]:
        lanes = [(lane["key"], lane["value"]) for lane in result["pipeline_lanes"]]
        lane_records = {
            (lane["key"], lane["value"]): lane for lane in result["pipeline_lanes"]
        }
        repair_pass_count = math.ceil(
            sum(len(lane["repair_parent_decisions"]) for lane in lane_records.values())
            / candidate_limit
        )
        baseline_pass_count = len(result["rounds"]) - repair_pass_count
        witness_jobs = [("witness", lane, None) for lane in lanes]
        conditional_jobs = [
            ("conditional", lane, None)
            for lane in lanes
            if lane_records[lane]["conditional_required"]
        ]
        jobs_by_pass = [
            jobs[offset : offset + candidate_limit]
            for jobs in (witness_jobs, conditional_jobs)
            for offset in range(0, len(jobs), candidate_limit)
        ]
        if not lanes:
            jobs_by_pass.extend(
                [("ordinary", None, None)]
                for _ in range(
                    0 if result["space_exhausted"] else phase["qualification_rounds"]
                )
            )
        jobs_by_pass.extend([] for _ in range(baseline_pass_count - len(jobs_by_pass)))
        repair_jobs = [
            ("failure_repair", lane, repair_index)
            for repair_index in range(phase["qualification_failure_retries"])
            for lane in lanes
            if len(lane_records[lane]["repair_parent_decisions"]) > repair_index
        ]
        repair_passes = [
            repair_jobs[offset : offset + candidate_limit]
            for offset in range(0, len(repair_jobs), candidate_limit)
        ]
        jobs_by_pass.extend(repair_passes)
        jobs_by_pass.extend([] for _ in range(repair_pass_count - len(repair_passes)))
        available_ids = set(result["initial_config_ids"])
        leaf = {
            "family": result["family"],
            "compound_packet": result["compound_packet"],
            "softmax_disc": result["softmax_disc"],
        }
        for pass_index, (round_result, jobs) in enumerate(
            zip(result["rounds"], jobs_by_pass, strict=True)
        ):
            decisions = []
            emitted_ids: list[str] = []
            for job_index, (kind, lane, repair_index) in enumerate(jobs):
                scoped_available = {
                    config_id
                    for config_id in available_ids
                    if manifest.structural_leaf(configs[config_id]) == leaf
                    and (lane is None or configs[config_id].get(lane[0]) == lane[1])
                }
                if kind == "failure_repair":
                    assert lane is not None and repair_index is not None
                    lane_record = lane_records[lane]
                    candidate_ids = {
                        lane_record["witness_config_id"],
                        *lane_record["conditional_candidate_ids"],
                        *lane_record["repair_candidate_ids"][:repair_index],
                    }
                    assert all(
                        states_by_id[config_id]["status"] in retryable_statuses
                        for config_id in candidate_ids
                    )
                    selection_kind = "ranked_failed_parent"
                    generated_ids = lane_record["repair_parent_decisions"][
                        repair_index
                    ]["generated_config_ids"]
                elif kind == "ordinary":
                    candidate_ids = {
                        config_id
                        for config_id in scoped_available
                        if states_by_id[config_id]["status"] in successful_statuses
                    }
                    selection_kind = "ranked_parent"
                    generated_ids = round_result["candidate_config_ids"]
                elif kind == "conditional":
                    assert lane is not None
                    candidate_ids = scoped_available
                    selection_kind = "ranked_parent"
                    generated_ids = lane_records[lane]["conditional_candidate_ids"]
                elif successful_scoped := {
                    config_id
                    for config_id in scoped_available
                    if states_by_id[config_id]["status"] in successful_statuses
                }:
                    candidate_ids = successful_scoped
                    selection_kind = "ranked_existing"
                    generated_ids = []
                else:
                    assert lane is not None
                    candidate_ids = {lane_records[lane]["witness_config_id"]}
                    selection_kind = "catalog_witness"
                    generated_ids = [lane_records[lane]["witness_config_id"]]
                candidates = ranked_snapshots(candidate_ids, pass_index)
                decision = {
                    "job_index": job_index,
                    "kind": kind,
                    "pipeline_lane": (
                        None if lane is None else manifest.pipeline_lane_metric(lane)
                    ),
                    "selection_kind": selection_kind,
                    "candidate_results": candidates,
                    "selected_config_id": (
                        candidates[0]["config_id"] if candidates else None
                    ),
                    "generated_config_ids": generated_ids,
                }
                if kind == "failure_repair":
                    assert lane is not None and repair_index is not None
                    decision["repair_index"] = repair_index
                    lane_records[lane]["repair_parent_decisions"][repair_index] = {
                        key: decision[key]
                        for key in (
                            "repair_index",
                            "candidate_results",
                            "selected_config_id",
                            "generated_config_ids",
                        )
                    }
                decisions.append(decision)
                emitted_ids.extend(generated_ids)
            assert emitted_ids == round_result["candidate_config_ids"], (
                emitted_ids,
                round_result["candidate_config_ids"],
                jobs_by_pass,
            )
            round_result["parent_decisions"] = decisions
            available_ids.update(round_result["candidate_config_ids"])

    def annotate_measurements(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                annotate_measurements(item)
            return
        if not isinstance(value, dict):
            return
        config_id = value.get("config_id")
        if (
            config_id is None
            and {
                "transferred_config_id",
                "attempt_perf",
                "selection_perf",
                "status",
            }
            <= value.keys()
        ):
            config_id = value["transferred_config_id"]
        if (
            isinstance(config_id, str)
            and {"attempt_perf", "selection_perf", "status"} <= value.keys()
        ):
            if value["status"] == "unknown":
                value["source_hash"] = None
                value["measurement_pass_index"] = None
            else:
                value["source_hash"] = source_hash_for(config_id)
                state = {
                    key: value[key]
                    for key in (
                        "attempt_perf",
                        "selection_perf",
                        "status",
                        "source_hash",
                    )
                }
                states_by_id.setdefault(config_id, state)
                value["measurement_pass_index"] = introduction_pass_by_id[config_id]
        elif value.get("status") == "projection_rejected":
            value["source_hash"] = None
        for item in value.values():
            annotate_measurements(item)

    annotate_measurements(phase)
    for result in phase["leaf_results"]:
        for pass_index, round_result in enumerate(result["rounds"]):
            for decision in round_result.get("parent_decisions", []):
                for snapshot in decision["candidate_results"]:
                    snapshot["measurement_pass_index"] = (
                        None if snapshot["status"] == "unknown" else pass_index
                    )
        for qualified in result["qualified_results"]:
            qualified["measurement_pass_index"] = pass_count
    for result in clc_families:
        for field in ("witness_candidate_results", "witness_selection_results"):
            for snapshot in result[field]:
                snapshot["measurement_pass_index"] = post_witness_pass
        for decision in result["conditional_parent_decisions"]:
            for snapshot in decision["candidate_results"]:
                snapshot["measurement_pass_index"] = post_witness_pass
        for field in ("retained_value_decisions",):
            for decision in result[field]:
                for snapshot in decision["candidate_results"]:
                    snapshot["measurement_pass_index"] = post_conditional_pass
        for snapshot in result["retained_ranking_results"]:
            snapshot["measurement_pass_index"] = post_conditional_pass
        for snapshot in result["depth_selection"]["candidate_results"]:
            snapshot["measurement_pass_index"] = post_conditional_pass
        for cell in result["combination_cells"]:
            if cell["status"] != "projection_rejected":
                cell["measurement_pass_index"] = post_conditional_pass + 1
    for result in phase["compound_transfers"]:
        for snapshot in result["source_selection"]["candidate_results"]:
            snapshot["measurement_pass_index"] = compound_source_pass
        for transfer in result["transfers"]:
            transfer["measurement_pass_index"] = pass_count - (
                phase.get("family_probe_generations", 0)
                if phase.get("family_probe_required")
                else 0
            )
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
                    {"config_id": config_id, **states_by_id[config_id]}
                    for config_id in sorted(states_by_id)
                    if introduction_pass_by_id[config_id] == pass_index
                ]
            ),
        }
        for pass_index in range(pass_count + 1)
    ]
    for trial in provenance.get("trials", []):
        if trial.get("search_phase_metrics") is phase:
            if attempts is None and isinstance(existing_terminal, dict):
                terminal = existing_terminal
                new_ids = {
                    config_id
                    for round_value in terminal["rounds"]
                    for config_id in round_value["new_candidate_ids"]
                }
                preterminal_ids = sorted(set(configs) - new_ids)
                terminal["search_generation"] = trial["num_generations"]
                assert not new_ids
                terminal["preterminal_num_configs_tested"] = trial["num_configs_tested"]
                terminal["preterminal_registry_config_count"] = len(preterminal_ids)
                terminal["preterminal_registry_config_ids_sha256"] = (
                    manifest.sha256_bytes(
                        json.dumps(preterminal_ids, separators=(",", ":")).encode()
                    )
                )
                phase["terminal_coordinate_refinement"] = terminal
            else:
                add_terminal_refinement(
                    provenance,
                    trial,
                    phase,
                    configs,
                    attempts,
                    default_perf=default_perf,
                )
            break


def fixture_payload(
    variant: str, seq_len: int, physical_gpu: int, tuner_seed: int, index: int
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    list[dict[str, str]],
    dict[str, Any],
]:
    causal = variant == "causal"
    selected_config = {
        "block_sizes": [1, 128, 128],
        "cute_flash_pipeline_family": "fa4_2cta",
        "cute_flash_exp2_packet": "1x1",
        "cute_flash_kv_stage": 2,
        "cute_flash_s_stage": 2,
        "cute_flash_wait_hint": index,
        "cute_flash_epi_tma": False,
        "cute_flash_epi_stg": True,
        "cute_flash_epi_stg_store": "slice",
        "cute_flash_epi_stg_gmem": "stage",
    }
    coverage_config = {
        "block_sizes": [1, 128, 128],
        "cute_flash_pipeline_family": "fa4_2cta",
        "cute_flash_exp2_packet": "1x1",
        "cute_flash_kv_stage": 2,
        "cute_flash_s_stage": 2,
        "cute_flash_wait_hint": -1,
        "cute_flash_epi_tma": False,
        "cute_flash_epi_stg": True,
        "cute_flash_epi_stg_store": "slice",
        "cute_flash_epi_stg_gmem": "stage",
    }
    alternate_coverage_config = {
        **coverage_config,
        "cute_flash_exp2_packet": "deg1_16x8",
        "cute_flash_kv_stage": 2,
        "cute_flash_wait_hint": -2,
    }
    second_compound_coverage_config = {
        **alternate_coverage_config,
        "cute_flash_kv_stage": 3,
        "cute_flash_wait_hint": -3,
    }
    selected_source = manifest.canonical_sha256(
        {"case": [variant, seq_len], "source": 0}
    )
    coverage_hash = manifest.canonical_sha256(coverage_config)
    alternate_coverage_hash = manifest.canonical_sha256(alternate_coverage_config)
    second_compound_coverage_hash = manifest.canonical_sha256(
        second_compound_coverage_config
    )
    design = [
        {"config": coverage_config, "config_sha256": coverage_hash},
        {
            "config": alternate_coverage_config,
            "config_sha256": alternate_coverage_hash,
        },
        {
            "config": second_compound_coverage_config,
            "config_sha256": second_compound_coverage_hash,
        },
    ]
    compiler_seed_ids = [manifest.canonical_sha256(selected_config)[:16]]
    compiler_seed_policy = {
        "schema_version": 1,
        "kind": "canonical_cute_flash",
        "heuristic_names": ["cute_flash_attention"],
        "raw_config_count": 1,
        "effective_config_ids": compiler_seed_ids,
        "effective_config_ids_sha256": manifest.canonical_sha256(compiler_seed_ids),
        "timeout_retry_repetitions": 3,
    }
    trial = {
        "input_shapes": repr([(2, 32, seq_len, 64)] * 3),
        "dtypes": repr(["torch.float16"] * 3),
        "hardware": "NVIDIA B200",
        "random_seed": tuner_seed,
        "search_algorithm": "LFBOTreeSearch",
        "num_configs_tested": 120,
        "num_compile_failures": 3,
        "num_worker_failures": 1,
        "num_isolated_rebenchmark_timeouts": 0,
        "num_accuracy_failures": 1,
        "num_successful_candidate_measurements": 115,
        "num_unique_sources": 120,
        "num_source_deduplications": 2,
        "num_generations": 4,
        "autotune_time": 1200.0 + index,
        "best_perf_ms": 9.5 + index,
        "selected_config": selected_config,
        "selected_source_hash": selected_source,
        "selected_source_was_measured": True,
    }
    provenance = {
        "helion_checkout_git_commit": FIXTURE_COMMIT,
        "require_full_autotune": True,
        "effort": "full",
        "requested_force_autotune": True,
        "effective_force_autotune": True,
        "fixed_config": False,
        "autotune_budget_seconds": None,
        "autotune_max_generations": None,
        "autotune_lfbo_max_generations": trial["num_generations"],
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
            if causal
            else "examples.attention._attention_output_baseline"
        ),
        "autotune_baseline_fn_is_expected": True,
        "autotune_baseline_atol": 5e-2,
        "autotune_baseline_rtol": 2e-2,
        "autotune_baseline_accuracy_check_fn": False,
        "autotune_benchmark_fn": False,
        "autotune_rebenchmark_threshold": None,
        "autotune_suspicious_rebenchmark_ratio": None,
        "autotune_config_overrides": {},
        "autotune_search_acf": [],
        "autotune_config_filter": False,
        "user_seed_configs": False,
        "compiler_seed_config_count": 1,
        "compiler_seed_policy": compiler_seed_policy,
        "compiler_default_config": False,
        "kernel_declared_config_count": 0,
        "active_value_prior_keys": [],
        "flash_value_prior_keys": [],
        "flash_fragment_default_config": coverage_config,
        "flash_fragment_default_sha256": coverage_hash,
        "flash_structural_coverage_design_source": (
            "normalized active ConfigSpec fragments"
        ),
        "flash_structural_coverage_active_values": [
            {"key": "cute_flash_pipeline_family", "value": "fa4_2cta"},
            {"key": "cute_flash_exp2_packet", "value": "1x1"},
            {"key": "cute_flash_exp2_packet", "value": "deg1_16x8"},
            {"key": "cute_flash_wait_hint", "value": -1},
            {"key": "cute_flash_epi_tma", "value": False},
            {"key": "cute_flash_epi_stg", "value": True},
            {"key": "cute_flash_epi_stg_store", "value": "slice"},
            {"key": "cute_flash_epi_stg_gmem", "value": "stage"},
        ],
        "flash_structural_coverage_design": design,
        "flash_structural_coverage_design_count": len(design),
        "flash_structural_coverage_design_sha256": manifest.canonical_sha256(
            [
                coverage_config,
                alternate_coverage_config,
                second_compound_coverage_config,
            ]
        ),
        "flash_structural_coverage_uncovered_values": [],
        "flash_structural_coverage_underqualified_values": [],
        "flash_structural_leaf_catalog": [
            {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": False,
            },
            {
                "family": "fa4_2cta",
                "compound_packet": "deg1_16x8",
                "softmax_disc": False,
            },
        ],
        "flash_pipeline_lane_catalog": [
            {
                "family": "fa4_2cta",
                "compound_packet": compound_packet,
                "softmax_disc": False,
                "pipeline_lanes": (
                    [{"key": "cute_flash_kv_stage", "value": value} for value in (2, 3)]
                    if compound_packet is None
                    else []
                ),
            }
            for compound_packet in (None, "deg1_16x8")
        ],
        "flash_structural_coverage_underqualified_leaves": [],
        "flash_structural_coverage_interaction_key_groups": [
            list(group) for group in manifest.FLASH_INTERACTION_KEY_GROUPS
        ],
        "flash_structural_coverage_active_interactions": [
            {
                "keys": list(manifest.FLASH_INTERACTION_KEY_GROUPS[0]),
                "values": [False, True, "slice", "stage"],
            }
        ],
        "flash_structural_coverage_uncovered_interactions": [],
        "flash_structural_qualification_values": [
            {"key": "cute_flash_pipeline_family", "value": "fa4_2cta"},
            {"key": "cute_flash_exp2_packet", "value": "deg1_16x8"},
        ],
        "flash_structural_parent_coverage_prefix_count": 2,
        "flash_structural_qualification_prefix_count": 3,
        "flash_structural_population_budget": 50,
        "flash_structural_injected_design_count": len(design),
        "flash_structural_qualification_rounds": 2,
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": 4,
        "flash_structural_retained_candidates_per_leaf": 2,
        "flash_structural_retained_family_cap": None,
        "flash_structural_retained_family_limit": 1,
        "flash_structural_retained_family_slowdown_limit": 2.0,
        "flash_structural_starting_path_limit": 14,
        "flash_structural_unrestricted_path_exhausts_generation_budget": True,
        "cute_flash_env_overrides": {},
        "final_correctness_enabled": True,
        "final_correctness_launches": 64,
        "final_repeatability_passed": True,
        "final_correctness_passed": True,
        "cache_read_policy": "bypass",
        "cache_write_policy": "write",
        "skip_cache_env": False,
        "rebenchmark_env_overrides": {},
        "selected_config": selected_config,
        "selected_source_sha256": selected_source,
        "selected_config_is_structural_coverage_design_member": False,
        "selected_config_nearest_structural_coverage_design_field_distance": 1,
        "selected_config_nearest_structural_coverage_design_config_sha256": [
            coverage_hash,
        ],
        "trials": [trial],
    }
    add_clc_lane_catalog(provenance)
    add_flash_normalization_context(provenance, trial, seq_len=seq_len, causal=causal)
    runs = [10.0 + index + offset / 100.0 for offset in range(9)]
    median_ms = runs[4]
    payload = {
        "impl": "helion-cute",
        "version": "Helion 1.4.0.dev157+gc3e36b65d; CuTe 4.7.0",
        "version_label": "generalized fixture",
        "shape": {
            "z": 2,
            "h": 32,
            "seq_len": seq_len,
            "head_dim": 64,
            "dtype": "float16",
            "causal": int(causal),
            "biased": 0,
        },
        "gpu": "NVIDIA B200",
        "physical_gpu": str(physical_gpu),
        "power_cap_w": 750,
        "input_seed": manifest.EXPECTED_INPUT_SEED,
        "flop_model": "softmax_attention_forward",
        "accuracy": "PASS",
        "benchmark_timer": "wall",
        "median_ms": median_ms,
        "median_tflops": attention_flops(seq_len, causal) / (median_ms * 1e9),
        "runs_ms": runs,
        "helion_overrides": {
            "autotuned": True,
            "benchmark_timer": "wall",
            "config_overrides": {},
            "seed_config_overrides": {},
            "force_autotune": True,
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

    shape = (2, 32, seq_len, 64)
    metadata = {
        "kernel_name": ("causal_attention_output" if causal else "attention_output"),
        "kernel_source": f"fixture kernel for {variant}",
        "input_shapes": repr([shape, shape, shape]),
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
    run_id = manifest.metadata_run_id(metadata, Path("fixture.meta.jsonl"))
    metadata["run_id"] = run_id
    ledger = []
    autotune_rows = []
    initial_config_ids = []
    qualification_candidate_ids = []
    lane_conditional_candidate_ids = []
    for source_index in range(120):
        source_hash = manifest.canonical_sha256(
            {"case": [variant, seq_len], "source": source_index}
        )
        if source_index == 0:
            config = selected_config
        elif source_index <= len(design):
            config = design[source_index - 1]["config"]
        elif source_index == 100:
            config = {
                **coverage_config,
                "cute_flash_kv_stage": 3,
                "cute_flash_wait_hint": 100,
                "cute_flash_e2e_offset": 25,
                "cute_flash_e2e_offset0": 34,
            }
        elif source_index == 101:
            config = {
                **coverage_config,
                "cute_flash_exp2_packet": "deg1_16x8",
                "cute_flash_kv_stage": 3,
                "cute_flash_wait_hint": 100,
                "cute_flash_q_tile_count": 2,
                "cute_flash_p_store_rep": 16,
                "cute_flash_s_load_rep": 32,
                "cute_flash_e2e_schedule": "16/8",
                "cute_flash_e2e_offset": 9,
                "cute_flash_e2e_offset0": 2,
            }
        else:
            config = {
                "block_sizes": [1, 128, 128],
                "cute_flash_pipeline_family": "fa4_2cta",
                "cute_flash_exp2_packet": "1x1",
                "cute_flash_kv_stage": (
                    2 if source_index < 103 or source_index % 2 == 0 else 3
                ),
                "cute_flash_s_stage": 2,
                "fixture_case": f"{variant}_{seq_len}",
                "fixture_source": source_index,
            }
        config_id = manifest.canonical_sha256(config)[:16]
        metadata["configs"][config_id] = config
        if source_index < 100:
            initial_config_ids.append(config_id)
        elif source_index < 102:
            qualification_candidate_ids.append(config_id)
        elif source_index < 104:
            lane_conditional_candidate_ids.append(config_id)
        if source_index < provenance["autotune_initial_population_size"]:
            generation = "0"
        elif source_index == 100:
            generation = "1"
        elif source_index in {102, 103}:
            generation = "2"
        elif source_index == 101:
            generation = "3"
        else:
            generation = str(trial["num_generations"])
        common = {
            "run_id": run_id,
            "config_id": config_id,
            "generation": generation,
            "source_hash": source_hash,
        }
        entries = [
            {
                "timestamp_s": f"{source_index + 0.1:.1f}",
                "status": "started",
                **common,
            },
            {
                "timestamp_s": f"{source_index + 0.2:.1f}",
                "status": (
                    "ok"
                    if source_index < 115
                    else "accuracy_error"
                    if source_index == 115
                    else "error"
                ),
                **common,
            },
        ]
        ledger.extend(entries)
        config_repr = (
            "Config("
            + ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))
            + ")"
        )
        for entry in entries:
            autotune_rows.append(
                {
                    **{key: entry[key] for key in manifest.AUTOTUNE_JOIN_FIELDS},
                    "perf_ms": "10.0" if entry["status"] == "ok" else "",
                    "compile_time_s": "",
                    "config": config_repr,
                }
            )
    alias_ids = []
    for alias in range(2):
        config = {
            "block_sizes": [1, 128, 128],
            "cute_flash_pipeline_family": "fa4_2cta",
            "cute_flash_exp2_packet": "1x1",
            "cute_flash_kv_stage": 2 + alias,
            "cute_flash_s_stage": 2,
            "fixture_alias": alias,
            "fixture_case": f"{variant}_{seq_len}",
        }
        config_id = manifest.canonical_sha256(config)[:16]
        alias_ids.append(config_id)
        metadata["configs"][config_id] = config
        entry = {
            "run_id": run_id,
            "timestamp_s": f"{121 + alias:.1f}",
            "config_id": config_id,
            "generation": str(trial["num_generations"]),
            "status": "deduplicated",
            "source_hash": manifest.canonical_sha256(
                {"case": [variant, seq_len], "source": alias + 1}
            ),
        }
        ledger.append(entry)
        config_repr = (
            "Config("
            + ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))
            + ")"
        )
        autotune_rows.append(
            {
                **{key: entry[key] for key in manifest.AUTOTUNE_JOIN_FIELDS},
                "perf_ms": "10.0",
                "compile_time_s": "",
                "config": config_repr,
            }
        )
    ordinary_leaf = {
        "family": "fa4_2cta",
        "compound_packet": None,
        "softmax_disc": False,
    }
    compound_leaf = {
        "family": "fa4_2cta",
        "compound_packet": "deg1_16x8",
        "softmax_disc": False,
    }
    ordinary_initial_ids = [
        config_id
        for config_id in initial_config_ids
        if manifest.structural_leaf(metadata["configs"][config_id]) == ordinary_leaf
    ]
    ordinary_qualified_ids = [
        *ordinary_initial_ids,
        qualification_candidate_ids[0],
    ]
    pipeline_lanes = [("cute_flash_kv_stage", 2), ("cute_flash_kv_stage", 3)]

    def retained_ids(config_ids: list[str]) -> list[str]:
        members = [
            {
                "config_id": config_id,
                "selection_perf": (
                    20.0 if config_id == qualification_candidate_ids[0] else 10.0
                ),
                "pipeline_lanes": manifest.config_pipeline_lanes(
                    metadata["configs"][config_id], pipeline_lanes
                ),
            }
            for config_id in config_ids
        ]
        return [
            member["config_id"]
            for member, _lane in manifest.lane_diverse_members(
                members, pipeline_lanes, limit=2
            )
        ]

    ordinary_retained_ids = retained_ids(ordinary_qualified_ids)

    lane_witness_ids = {
        2: min(
            config_id
            for config_id in ordinary_initial_ids
            if metadata["configs"][config_id].get("cute_flash_kv_stage") == 2
        ),
        3: qualification_candidate_ids[0],
    }
    lane_conditional_ids = {
        value: next(
            config_id
            for config_id in lane_conditional_candidate_ids
            if metadata["configs"][config_id].get("cute_flash_kv_stage") == value
        )
        for value in (2, 3)
    }
    ordinary_qualified_ids = [
        *ordinary_initial_ids,
        qualification_candidate_ids[0],
        *lane_conditional_candidate_ids,
    ]
    ordinary_retained_ids = retained_ids(ordinary_qualified_ids)
    compound_path_ids = [qualification_candidate_ids[1]]
    expected_retained_families = manifest.expected_structural_retention(
        [
            {
                **ordinary_leaf,
                "members": [
                    {
                        "config_id": config_id,
                        "selection_perf": (
                            20.0
                            if config_id == qualification_candidate_ids[0]
                            else 10.0
                        ),
                        "pipeline_lanes": manifest.config_pipeline_lanes(
                            metadata["configs"][config_id], pipeline_lanes
                        ),
                    }
                    for config_id in ordinary_qualified_ids
                ],
                "pipeline_lanes": pipeline_lanes,
            },
            {
                **compound_leaf,
                "members": [
                    {
                        "config_id": compound_path_ids[0],
                        "selection_perf": 10.0,
                        "pipeline_lanes": frozenset(),
                    }
                ],
                "pipeline_lanes": [],
            },
        ],
        retained_per_leaf=2,
        retained_family_cap=None,
        retained_family_limit=1,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=14,
    )
    trial["search_phase_metrics"] = {
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
        "leaf_count": 2,
        "ordinary_leaf_count": 1,
        "compound_leaf_count": 1,
        "pipeline_qualification_keys": list(manifest.FLASH_PIPELINE_QUALIFICATION_KEYS),
        "leaf_results": [
            {
                **ordinary_leaf,
                "initial_config_ids": ordinary_initial_ids,
                "space_exhausted": False,
                "space_config_count": None,
                "ordinary_search_required": False,
                "rounds": [
                    {
                        "candidate_config_ids": [qualification_candidate_ids[0]],
                        "neighbor_generation_limit": 0,
                        "ordinary_neighbor_generation_limit": 0,
                    },
                    {
                        "candidate_config_ids": lane_conditional_candidate_ids,
                        "neighbor_generation_limit": 200,
                        "ordinary_neighbor_generation_limit": 0,
                    },
                ],
                "pipeline_lanes": [
                    {
                        "key": key,
                        "value": value,
                        "initial_config_ids": [
                            config_id
                            for config_id in ordinary_initial_ids
                            if metadata["configs"][config_id].get(key) == value
                        ],
                        "space_exhausted": False,
                        "space_config_count": None,
                        "conditional_required": True,
                        "rounds": [
                            {
                                "candidate_config_ids": [lane_witness_ids[value]],
                                "neighbor_generation_limit": 0,
                            },
                            {
                                "candidate_config_ids": [lane_conditional_ids[value]],
                                "neighbor_generation_limit": 100,
                            },
                        ],
                        "witness_attempted": True,
                        "witness_config_id": lane_witness_ids[value],
                        "witness_succeeded": True,
                        "conditional_candidate_ids": [lane_conditional_ids[value]],
                        "successful_conditional_candidate_ids": [
                            lane_conditional_ids[value]
                        ],
                        "repair_candidate_ids": [],
                        "successful_repair_candidate_ids": [],
                        "repair_parent_decisions": [],
                        "terminal_failure_exhausted": False,
                        "complete": True,
                    }
                    for key, value in pipeline_lanes
                ],
                "qualified_results": [
                    {
                        "config_id": config_id,
                        "attempt_perf": 10.0,
                        "selection_perf": (
                            20.0
                            if config_id == qualification_candidate_ids[0]
                            else 10.0
                        ),
                        "status": ("deduplicated" if config_id in alias_ids else "ok"),
                        "pipeline_lanes": [
                            manifest.pipeline_lane_metric(lane)
                            for lane in pipeline_lanes
                            if metadata["configs"][config_id].get(lane[0]) == lane[1]
                        ],
                    }
                    for config_id in ordinary_qualified_ids
                ],
                "retained_config_ids": ordinary_retained_ids,
                "complete": True,
            }
        ],
        "qualification_rounds": 2,
        "qualification_rounds_started": 3,
        "qualification_rounds_completed": 3,
        "qualification_passes_planned": 3,
        "qualification_passes_started": 3,
        "qualification_passes_completed": 3,
        "budget_exhausted": False,
        "schedule_anchor_design_source": (
            "live family x ordinary packet x softmax protocol from fragment defaults"
        ),
        "schedule_anchor_pass_planned": False,
        "schedule_anchor_pass_started": False,
        "schedule_anchor_count": 1,
        "schedule_anchor_complete": True,
        "schedule_anchor_results": [],
        "pipeline_candidate_limit_per_leaf_per_round": 4,
        "conditional_candidates_per_pipeline_lane": 1,
        "neighbor_generation_limit_per_leaf_per_round": 200,
        "candidate_count": 4,
        "leaves_with_candidates": 2,
        "retained_candidates_per_leaf": 2,
        "retained_family_cap": None,
        "retained_family_limit": 1,
        "retained_family_slowdown_limit": 2.0,
        "clc_families": [],
        "compound_catalog_complete": True,
        "compound_catalog_errors": [],
        "compound_transfers": [
            {
                **compound_leaf,
                "limit": 2,
                "transfer_target_count": 1,
                "transfer_count": 1,
                "primary_transfer_config_ids": [qualification_candidate_ids[1]],
                "backfill_rounds": [],
                "successful_transfer_config_ids": [qualification_candidate_ids[1]],
                "qualified_transfer_config_ids": [qualification_candidate_ids[1]],
                "failure_statuses_allowed": True,
                "source_selection": {
                    "candidate_results": [
                        {
                            "config_id": qualification_candidate_ids[0],
                            "attempt_perf": 10.0,
                            "selection_perf": 10.0,
                            "status": "ok",
                        }
                    ],
                    "combination_prefix_count": 0,
                    "attempted_config_ids": [qualification_candidate_ids[0]],
                    "selected_config_ids": [qualification_candidate_ids[0]],
                },
                "transfers": [
                    {
                        "source_config_id": qualification_candidate_ids[0],
                        "source_config": metadata["configs"][
                            qualification_candidate_ids[0]
                        ],
                        "transferred_config_id": qualification_candidate_ids[1],
                        "projected_config": metadata["configs"][
                            qualification_candidate_ids[1]
                        ],
                        "attempt_perf": 10.0,
                        "selection_perf": 10.0,
                        "status": "ok",
                        "projection_overrides": {"cute_flash_exp2_packet": "deg1_16x8"},
                        "projected_config_id": qualification_candidate_ids[1],
                        "preserved_pipeline_values": {
                            "cute_flash_kv_stage": 3,
                            "cute_flash_s_stage": 2,
                        },
                    }
                ],
                "complete": True,
            }
        ],
        "starting_path_limit": 14,
        "retained_families": expected_retained_families,
        "retained_path_count": sum(
            len(family["starting_paths"]) for family in expected_retained_families
        ),
    }
    phase_attempts = {
        row["config_id"]: {
            "status": row["status"],
            "perf_ms": float(row["perf_ms"]) if row["perf_ms"] else None,
            "source_hash": ledger[position]["source_hash"],
        }
        for position, row in enumerate(autotune_rows)
        if row["status"] != "started"
    }
    add_phase_config_identity(
        provenance,
        trial["search_phase_metrics"],
        metadata["configs"],
        attempts=phase_attempts,
        default_perf=10.0,
    )
    phase = trial["search_phase_metrics"]
    anchor = next(
        result
        for result in phase["initial_results"]
        if {
            "family": result["family"],
            "compound_packet": result["compound_packet"],
            "softmax_disc": result["softmax_disc"],
        }
        == ordinary_leaf
    )
    phase["schedule_anchor_results"] = [
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
    ]
    return payload, ledger, autotune_rows, metadata


def write_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=manifest.LEDGER_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(output.getvalue())


def test_phase_identity_records_compound_transfer_before_family_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, _ledger, _autotune_rows, metadata = fixture_payload(
        "dense", 32768, 7, 2026081501, 0
    )
    provenance = payload["helion_overrides"]["autotune_provenance"]
    phase = provenance["trials"][0]["search_phase_metrics"]
    monkeypatch.setattr(
        manifest,
        "expected_family_probe_path_limit",
        lambda *_args: 1,
    )

    add_phase_config_identity(
        provenance,
        phase,
        metadata["configs"],
        default_perf=10.0,
    )

    assert phase["family_probe_required"] is True
    assert phase["compound_transfers"][0]["transfers"][0]["measurement_pass_index"] == (
        phase["qualification_passes_completed"] - 1
    )


def write_autotune_csv(path: Path, rows: list[dict[str, str]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=manifest.AUTOTUNE_CSV_FIELDS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(output.getvalue())


def structural_execution_fixture(
    *,
    fail_prefix: bool = False,
    successful_family_outside_prefix: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    design_configs = [
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_exp2_packet": "1x1",
            "cute_flash_wait_hint": 0,
        },
        {
            "cute_flash_pipeline_family": "fa4",
            "cute_flash_exp2_packet": "1x1",
            "cute_flash_wait_hint": 1,
        },
    ]
    provenance = {
        "autotune_initial_population_size": 100,
        "flash_structural_population_budget": len(design_configs),
        "flash_structural_coverage_design": [
            {"config": config} for config in design_configs
        ],
        "flash_structural_injected_design_count": len(design_configs),
        "flash_structural_coverage_active_values": [
            {"key": "cute_flash_pipeline_family", "value": "fa4"},
            {"key": "cute_flash_exp2_packet", "value": "1x1"},
            {"key": "cute_flash_wait_hint", "value": 0},
        ],
        "flash_structural_coverage_active_interactions": [],
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
        "flash_structural_qualification_values": [
            {"key": "cute_flash_pipeline_family", "value": "fa4"}
        ],
        "flash_structural_qualification_rounds": 2,
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": 4,
        "flash_structural_retained_candidates_per_leaf": 2,
        "flash_structural_retained_family_cap": None,
        "flash_structural_retained_family_limit": 1,
        "flash_structural_retained_family_slowdown_limit": 2.0,
        "flash_structural_starting_path_limit": 14,
        "flash_structural_unrestricted_path_exhausts_generation_budget": True,
    }
    add_clc_lane_catalog(provenance)
    source_rows: list[dict[str, str]] = []
    metadata_configs: dict[str, dict[str, Any]] = {}
    for index in range(100):
        if index < len(design_configs):
            config = design_configs[index]
        elif index == 2 and successful_family_outside_prefix:
            config = {
                "cute_flash_pipeline_family": "fa4",
                "cute_flash_exp2_packet": "1x1",
                "fixture_candidate": index,
            }
        else:
            config = {
                "cute_flash_pipeline_family": "fa4",
                "cute_flash_exp2_packet": "1x1",
                "fixture_candidate": index,
            }
        config_id = manifest.canonical_sha256(config)[:16]
        source_hash = manifest.canonical_sha256({"measurement_source": config_id})
        metadata_configs[config_id] = config
        source_rows.extend(
            (
                {
                    "config_id": config_id,
                    "generation": "0",
                    "status": "started",
                    "source_hash": source_hash,
                },
                {
                    "config_id": config_id,
                    "generation": "0",
                    "status": (
                        "error"
                        if fail_prefix
                        and not (successful_family_outside_prefix and index == 2)
                        else "ok"
                    ),
                    "source_hash": source_hash,
                },
            )
        )
    initial_ids = [
        row["config_id"] for row in source_rows if row["status"] == "started"
    ]
    leaf = {"family": "fa4", "compound_packet": None, "softmax_disc": False}
    leaf_initial_ids = [
        config_id
        for config_id in initial_ids
        if manifest.structural_leaf(metadata_configs[config_id]) == leaf
    ]
    attempt_by_config = {
        row["config_id"]: {
            "generation": int(row["generation"]),
            "status": row["status"],
            "source_hash": row["source_hash"],
            "perf_ms": 1.0 if row["status"] == "ok" else None,
        }
        for row in source_rows
        if row["status"] != "started"
    }
    attempt_history_by_config = {
        row["config_id"]: [
            {
                **attempt_by_config[row["config_id"]],
                "position": position,
            }
        ]
        for position, row in enumerate(source_rows)
        if row["status"] != "started"
    }
    successful_leaf_ids = sorted(
        config_id
        for config_id in leaf_initial_ids
        if attempt_by_config[config_id]["status"] == "ok"
    )
    retained_leaf_ids = successful_leaf_ids[:2]
    retained_families = manifest.expected_structural_retention(
        [
            {
                **leaf,
                "members": [
                    {
                        "config_id": config_id,
                        "selection_perf": 1.0,
                        "pipeline_lanes": frozenset(),
                    }
                    for config_id in successful_leaf_ids
                ],
                "pipeline_lanes": [],
            }
        ],
        retained_per_leaf=2,
        retained_family_cap=manifest.EXPECTED_FULL_FLASH_RETAINED_FAMILIES,
        retained_family_limit=1,
        retained_family_slowdown_limit=2.0,
        starting_path_limit=14,
    )
    trial = {
        "num_isolated_rebenchmark_timeouts": 0,
        "search_phase_metrics": {
            "phase": "cute_flash_structural_qualification_v22",
            "cute_flash_lane_policy_version": 11,
            "qualification_failure_retries": 1,
            "completed": True,
            "unrestricted_path_exhausts_generation_budget": True,
            "initial_config_count": len(initial_ids),
            "initial_config_ids": initial_ids,
            "exact_space_enumerated": False,
            "exact_space_exhausted": False,
            "exact_space_raw_budget": 100,
            "exact_space_config_ids": [],
            "leaf_count": 1,
            "ordinary_leaf_count": 1,
            "compound_leaf_count": 0,
            "pipeline_qualification_keys": list(
                manifest.FLASH_PIPELINE_QUALIFICATION_KEYS
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
                            "candidate_config_ids": [],
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
                            "attempt_perf": attempt_by_config[config_id]["perf_ms"],
                            "selection_perf": attempt_by_config[config_id]["perf_ms"],
                            "status": attempt_by_config[config_id]["status"],
                            "pipeline_lanes": [],
                        }
                        for config_id in leaf_initial_ids
                    ],
                    "retained_config_ids": retained_leaf_ids,
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
            "candidate_count": 0,
            "leaves_with_candidates": 0,
            "retained_candidates_per_leaf": 2,
            "retained_family_cap": None,
            "retained_family_limit": 1,
            "retained_family_slowdown_limit": 2.0,
            "clc_families": [],
            "compound_transfers": [],
            "starting_path_limit": 14,
            "retained_families": retained_families,
            "retained_path_count": sum(
                len(family["starting_paths"]) for family in retained_families
            ),
        },
    }
    add_flash_normalization_context(provenance, trial, seq_len=32768, causal=False)
    provenance["trials"] = [trial]
    add_phase_config_identity(
        provenance,
        trial["search_phase_metrics"],
        metadata_configs,
        attempts=attempt_by_config,
    )
    return (
        provenance,
        trial,
        source_rows,
        metadata_configs,
        attempt_by_config,
        attempt_history_by_config,
    )


def exact_structural_execution_fixture(
    *, deduplicate_last: bool = False
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    (
        provenance,
        trial,
        source_rows,
        metadata_configs,
        _attempt_by_config,
        _attempt_history_by_config,
    ) = structural_execution_fixture()
    phase = trial["search_phase_metrics"]
    exact_ids = phase["initial_config_ids"][:4]
    exact_id_set = set(exact_ids)
    source_rows = [row for row in source_rows if row["config_id"] in exact_id_set]
    if deduplicate_last:
        deduplicated_id = exact_ids[-1]
        source_rows = [
            row
            for row in source_rows
            if not (row["config_id"] == deduplicated_id and row["status"] == "started")
        ]
        terminal = next(
            row for row in source_rows if row["config_id"] == deduplicated_id
        )
        terminal["status"] = "deduplicated"

    metadata_configs = {
        config_id: metadata_configs[config_id] for config_id in exact_ids
    }
    attempt_by_config = {
        row["config_id"]: {
            "generation": int(row["generation"]),
            "status": row["status"],
            "source_hash": row["source_hash"],
            "perf_ms": 1.0 if row["status"] in {"ok", "deduplicated"} else None,
        }
        for row in source_rows
        if row["status"] != "started"
    }
    attempt_history_by_config = {
        row["config_id"]: [
            {
                **attempt_by_config[row["config_id"]],
                "position": position,
            }
        ]
        for position, row in enumerate(source_rows)
        if row["status"] != "started"
    }

    provenance.update(
        {
            "flash_exact_effective_search_space_size": len(exact_ids),
            "flash_exact_effective_search_space_config_ids": exact_ids,
            "flash_exact_effective_search_space_sha256": manifest.canonical_sha256(
                exact_ids
            ),
        }
    )
    phase.update(
        {
            "initial_config_count": len(exact_ids),
            "initial_config_ids": exact_ids,
            "exact_space_enumerated": True,
            "exact_space_exhausted": True,
            "exact_space_config_ids": exact_ids,
            "candidate_count": 0,
            "leaves_with_candidates": 0,
            "qualification_rounds_started": 0,
            "qualification_rounds_completed": 0,
            "qualification_passes_planned": 0,
            "qualification_passes_started": 0,
            "qualification_passes_completed": 0,
        }
    )
    leaf = phase["leaf_results"][0]
    leaf.update(
        {
            "initial_config_ids": exact_ids,
            "space_exhausted": True,
            "space_config_count": len(exact_ids),
            "ordinary_search_required": False,
            "rounds": [],
            "qualified_results": [
                {
                    "config_id": config_id,
                    "attempt_perf": attempt_by_config[config_id]["perf_ms"],
                    "selection_perf": attempt_by_config[config_id]["perf_ms"],
                    "status": attempt_by_config[config_id]["status"],
                    "pipeline_lanes": [],
                }
                for config_id in exact_ids
            ],
        }
    )
    members = [
        {
            "config_id": config_id,
            "selection_perf": 1.0,
            "pipeline_lanes": frozenset(),
        }
        for config_id in exact_ids
    ]
    retained_families = manifest.expected_structural_retention(
        [
            {
                "family": leaf["family"],
                "compound_packet": leaf["compound_packet"],
                "members": members,
                "pipeline_lanes": [],
            }
        ],
        retained_per_leaf=provenance["flash_structural_retained_candidates_per_leaf"],
        retained_family_cap=provenance["flash_structural_retained_family_cap"],
        retained_family_limit=provenance["flash_structural_retained_family_limit"],
        retained_family_slowdown_limit=provenance[
            "flash_structural_retained_family_slowdown_limit"
        ],
        starting_path_limit=provenance["flash_structural_starting_path_limit"],
    )
    leaf["retained_config_ids"] = sorted(exact_ids)[:2]
    phase["retained_families"] = retained_families
    phase["retained_path_count"] = sum(
        len(family["starting_paths"]) for family in retained_families
    )
    trial["num_configs_tested"] = sum(row["status"] == "started" for row in source_rows)
    trial["num_successful_candidate_measurements"] = sum(
        row["status"] == "ok" for row in source_rows
    )
    add_phase_config_identity(
        provenance,
        phase,
        metadata_configs,
        attempts=attempt_by_config,
    )
    return (
        provenance,
        trial,
        source_rows,
        metadata_configs,
        attempt_by_config,
        attempt_history_by_config,
    )


def terminal_validation_fixture() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    leaf = {"family": "fa4", "compound_packet": None, "softmax_disc": False}
    configs = {
        manifest.canonical_sha256(config)[:16]: config
        for config in (
            {
                "cute_flash_pipeline_family": "fa4",
                "cute_flash_exp2_packet": "1x1",
                "fixture_coordinate": value,
            }
            for value in range(102)
        )
    }
    ids_by_value = {
        config["fixture_coordinate"]: config_id for config_id, config in configs.items()
    }
    initial_id = ids_by_value[0]
    reused_id = ids_by_value[1]
    failed_id = ids_by_value[2]
    compile_failed_id = ids_by_value[100]
    new_id = ids_by_value[101]
    source_rows: list[dict[str, str]] = []
    attempt_by_config: dict[str, dict[str, Any]] = {}
    attempt_history_by_config: dict[str, list[dict[str, Any]]] = {}
    for config_id in (ids_by_value[value] for value in range(101)):
        source_hash = manifest.canonical_sha256({"source": config_id})
        status = "error" if config_id in {failed_id, compile_failed_id} else "ok"
        if config_id != compile_failed_id:
            source_rows.append(
                {
                    "config_id": config_id,
                    "generation": "0",
                    "status": "started",
                    "source_hash": source_hash,
                }
            )
        source_rows.append(
            {
                "config_id": config_id,
                "generation": "0",
                "status": status,
                "source_hash": source_hash,
            }
        )
        attempt = {
            "generation": 0,
            "status": status,
            "source_hash": source_hash,
            "perf_ms": 1.0 if status == "ok" else None,
            "position": len(source_rows) - 1,
        }
        attempt_by_config[config_id] = attempt
        attempt_history_by_config[config_id] = [attempt]
    new_source_hash = manifest.canonical_sha256({"source": new_id})
    source_rows.extend(
        (
            {
                "config_id": new_id,
                "generation": "20",
                "status": "started",
                "source_hash": new_source_hash,
            },
            {
                "config_id": new_id,
                "generation": "20",
                "status": "ok",
                "source_hash": new_source_hash,
            },
        )
    )
    new_attempt = {
        "generation": 20,
        "status": "ok",
        "source_hash": new_source_hash,
        "perf_ms": 9.0,
        "position": len(source_rows) - 1,
    }
    attempt_by_config[new_id] = new_attempt
    attempt_history_by_config[new_id] = [new_attempt]
    policy = manifest.expected_terminal_refinement_policy()
    surface = {
        "schema_version": 1,
        "radius": 2,
        "leaves": [
            {
                "leaf": leaf,
                "coordinates": [
                    {
                        "flat_index": 0,
                        "key": "fixture_coordinate",
                        "sequence_index": None,
                        "fragment_type": "IntegerFragment",
                        "overridden": False,
                        "active_values": [0, 1, 2, 101],
                        "neighbors_by_value": [
                            {"from_value": 0, "to_values": [1, 2, 101]},
                            {"from_value": 1, "to_values": [101]},
                            {"from_value": 2, "to_values": [0]},
                            {"from_value": 101, "to_values": [1]},
                        ],
                    }
                ],
            }
        ],
    }
    provenance = {
        "flash_structural_leaf_catalog": [leaf],
        "flash_terminal_coordinate_refinement_policy": policy,
        "flash_terminal_coordinate_refinement_policy_sha256": (
            manifest.canonical_sha256(policy)
        ),
        "flash_terminal_coordinate_surface_catalog": surface,
        "flash_terminal_coordinate_surface_catalog_sha256": (
            manifest.canonical_sha256(surface)
        ),
    }

    def request(
        parent_value: int,
        to_value: int,
        outcome: str,
        config_id: str,
    ) -> dict[str, Any]:
        return {
            "flat_index": 0,
            "key": "fixture_coordinate",
            "sequence_index": None,
            "from_value": parent_value,
            "to_value": to_value,
            "outcome": outcome,
            "config_id": config_id,
        }

    round_one_ids = [initial_id, reused_id, new_id]
    round_one_times = {initial_id: 10.0, reused_id: 11.0, new_id: 9.0}
    round_two_ids = [new_id, initial_id, reused_id]
    round_two_times = {new_id: 9.2, initial_id: 10.0, reused_id: 11.0}
    confirmation_ids = [initial_id, new_id, reused_id]
    confirmation_times = {initial_id: 10.0, new_id: 9.5, reused_id: 11.0}
    terminal_manifest = dict(
        sorted(
            {
                config_id: {"config": configs[config_id]}
                for config_id in (initial_id, reused_id, failed_id, new_id)
            }.items()
        )
    )
    rounds = [
        {
            "round_index": 1,
            "incumbent_config_id": initial_id,
            "leaf": leaf,
            "parent_config_ids": [initial_id],
            "parent_projections": [
                {
                    "parent_config_id": initial_id,
                    "coordinate_requests": [
                        request(0, 1, "candidate", reused_id),
                        request(0, 2, "candidate", failed_id),
                        request(0, 101, "candidate", new_id),
                    ],
                }
            ],
            "candidate_config_ids": [reused_id, failed_id, new_id],
            "new_candidate_ids": [new_id],
            "reused_candidate_ids": [reused_id],
            "intra_terminal_reused_candidate_ids": [],
            "prior_failed_candidate_ids": [failed_id],
            "candidate_results": [
                {
                    "config_id": reused_id,
                    "attempt_perf": 1.0,
                    "selection_perf": 11.0,
                    "status": "ok",
                    "source_hash": attempt_by_config[reused_id]["source_hash"],
                },
                {
                    "config_id": failed_id,
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "error",
                    "source_hash": attempt_by_config[failed_id]["source_hash"],
                },
                {
                    "config_id": new_id,
                    "attempt_perf": 9.0,
                    "selection_perf": 9.0,
                    "status": "ok",
                    "source_hash": new_source_hash,
                },
            ],
            "comparison_config_ids": round_one_ids,
            "measurement": terminal_measurement(
                round_one_ids,
                round_one_times,
                target_ms=200.0,
                repeat_reference_perf_ms=9.0,
            ),
            "round_best_config_id": new_id,
            "selected_config_id": new_id,
            "accepted": True,
            "improvement_fraction": 0.1,
            "beam_config_ids": [new_id, initial_id, reused_id],
        },
        {
            "round_index": 2,
            "incumbent_config_id": new_id,
            "leaf": leaf,
            "parent_config_ids": [new_id, initial_id, reused_id],
            "parent_projections": [
                {
                    "parent_config_id": new_id,
                    "coordinate_requests": [request(101, 1, "beam_alias", reused_id)],
                },
                {
                    "parent_config_id": initial_id,
                    "coordinate_requests": [
                        request(0, 1, "beam_alias", reused_id),
                        request(0, 2, "candidate", failed_id),
                        request(0, 101, "beam_alias", new_id),
                    ],
                },
                {
                    "parent_config_id": reused_id,
                    "coordinate_requests": [request(1, 101, "beam_alias", new_id)],
                },
            ],
            "candidate_config_ids": [failed_id],
            "new_candidate_ids": [],
            "reused_candidate_ids": [],
            "intra_terminal_reused_candidate_ids": [],
            "prior_failed_candidate_ids": [failed_id],
            "candidate_results": [
                {
                    "config_id": failed_id,
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "error",
                    "source_hash": attempt_by_config[failed_id]["source_hash"],
                }
            ],
            "comparison_config_ids": round_two_ids,
            "measurement": terminal_measurement(
                round_two_ids,
                round_two_times,
                target_ms=200.0,
                repeat_reference_perf_ms=11.0,
            ),
            "round_best_config_id": new_id,
            "selected_config_id": new_id,
            "accepted": False,
            "improvement_fraction": 0.0,
            "beam_config_ids": round_two_ids,
        },
    ]
    transcript = {
        "schema_version": policy["schema_version"],
        "policy_version": policy["policy_version"],
        "lane_policy_version": policy["lane_policy_version"],
        "coordinate_policy": policy["coordinate_policy"],
        "measurement_policy": policy["measurement_policy"],
        "rounds_planned": policy["rounds"],
        "beam_width": policy["beam_width"],
        "maximum_projection_parent_count": 5,
        "projection_parent_count": 4,
        "rounds_started": 2,
        "rounds_completed": 2,
        "completed": True,
        "budget_exhausted": False,
        "termination_reason": "round_limit",
        "search_generation": 20,
        "preterminal_num_configs_tested": 100,
        "preterminal_registry_config_count": 101,
        "preterminal_registry_config_ids_hash_policy": (
            "sorted_compact_json_sha256_v1"
        ),
        "preterminal_registry_config_ids_sha256": manifest.sha256_bytes(
            json.dumps(sorted(set(configs) - {new_id}), separators=(",", ":")).encode()
        ),
        "radius": 2,
        "minimum_improvement_fraction": 0.001,
        "initial_incumbent_config_id": initial_id,
        "refined_config_id": new_id,
        "final_config_id": new_id,
        "projection_attempt_count": 8,
        "unique_candidate_count": 3,
        "new_candidate_count": 1,
        "reused_candidate_count": 1,
        "intra_terminal_reused_candidate_count": 0,
        "prior_failed_candidate_count": 1,
        "accepted_config_ids": [new_id],
        "config_manifest_sha256": manifest.canonical_sha256(terminal_manifest),
        "config_manifest": terminal_manifest,
        "rounds": rounds,
        "confirmation": {
            "candidate_config_ids": confirmation_ids,
            "measurement": terminal_measurement(
                confirmation_ids,
                confirmation_times,
                target_ms=5000.0,
                repeat_reference_perf_ms=11.0,
            ),
            "best_config_id": new_id,
            "selected_config_id": new_id,
            "accepted": True,
            "improvement_fraction": 0.05,
            "skipped_reason": None,
        },
    }
    phase = {
        "exact_space_config_ids": [],
        "measurement_timeline": [],
        "terminal_coordinate_refinement": transcript,
    }
    trial = {
        "num_configs_tested": 101,
        "num_generations": 20,
        "selected_config": configs[new_id],
        "search_phase_metrics": phase,
    }
    provenance["trials"] = [trial]
    return (
        provenance,
        trial,
        phase,
        source_rows,
        configs,
        attempt_by_config,
        attempt_history_by_config,
    )


class StrictManifestTests(unittest.TestCase):
    def test_validates_terminal_boundary_with_unstarted_precompile_failure(
        self,
    ) -> None:
        args = terminal_validation_fixture()

        summary = manifest.validate_terminal_coordinate_refinement(
            Path("fixture.json"), *args
        )

        terminal_new_ids = {
            config_id
            for round_value in args[2]["terminal_coordinate_refinement"]["rounds"]
            for config_id in round_value["new_candidate_ids"]
        }
        first_terminal_row = next(
            index
            for index, row in enumerate(args[3])
            if row["config_id"] in terminal_new_ids
        )
        self.assertEqual(
            sum(row["status"] == "started" for row in args[3][:first_terminal_row]),
            100,
        )
        self.assertEqual(summary["preterminal_num_configs_tested"], 100)
        self.assertEqual(summary["preterminal_registry_config_count"], 101)
        self.assertEqual(summary["new_candidate_count"], 1)
        self.assertEqual(summary["reused_candidate_count"], 1)
        self.assertEqual(summary["prior_failed_candidate_count"], 1)
        self.assertEqual(
            summary["final_config_id"],
            manifest.canonical_sha256(args[1]["selected_config"])[:16],
        )
        confirmation_measurement = args[2]["terminal_coordinate_refinement"][
            "confirmation"
        ]["measurement"]
        self.assertEqual(confirmation_measurement["sweep_count"], 58)
        self.assertEqual(confirmation_measurement["calls_per_sample"], 8)
        self.assertEqual(confirmation_measurement["total_calls"], 464)
        self.assertEqual(len(confirmation_measurement["elapsed_ms"]), 58)

    def test_rejects_preterminal_first_evidence_after_terminal_boundary(self) -> None:
        args = list(terminal_validation_fixture())
        compile_failed_id = next(
            config_id
            for config_id, config in args[4].items()
            if config["fixture_coordinate"] == 100
        )
        position = next(
            index
            for index, row in enumerate(args[3])
            if row["config_id"] == compile_failed_id
        )
        args[3].append(args[3].pop(position))

        with self.assertRaisesRegex(
            RuntimeError, "preterminal registry configs lack source evidence"
        ):
            manifest.validate_terminal_coordinate_refinement(
                Path("fixture.json"), *args
            )

    def test_rejects_invalid_terminal_refinement_boundary_and_claims(self) -> None:
        def force_four_sweeps(terminal: dict[str, Any], reference: float) -> None:
            measurement = terminal["rounds"][0]["measurement"]
            measurement["repeat_reference_perf_ms"] = reference
            measurement["sweep_count"] = 4
            measurement["calls_per_sample"] = 1
            measurement["total_calls"] = 4
            del measurement["elapsed_ms"][4:]

        mutations = {
            "preterminal tested count versus chronological started rows": lambda terminal: (
                terminal.update(preterminal_num_configs_tested=101)
            ),
            "preterminal registry digest": lambda terminal: terminal.update(
                preterminal_registry_config_ids_sha256="0" * 64
            ),
            "preterminal tested count|terminal tested-count delta|started-row partition|new_candidate_ids": lambda terminal: (
                terminal["rounds"][0].update(new_candidate_ids=[])
            ),
            "reused_candidate_ids": lambda terminal: terminal["rounds"][0].update(
                reused_candidate_ids=[]
            ),
            "prior_failed_candidate_ids": lambda terminal: terminal["rounds"][0].update(
                prior_failed_candidate_ids=[]
            ),
            "coordinate request versus recorded surface": lambda terminal: terminal[
                "rounds"
            ][0]["parent_projections"][0]["coordinate_requests"][0].update(to_value=99),
            "measurement target": lambda terminal: terminal["rounds"][0][
                "measurement"
            ].update(target_ms=199.0),
            "sweep-count formula": lambda terminal: terminal["rounds"][0][
                "measurement"
            ].update(repeat_reference_perf_ms=10.0),
            "calls-per-sample formula": lambda terminal: terminal["confirmation"][
                "measurement"
            ].update(calls_per_sample=7),
            "total-call formula": lambda terminal: terminal["confirmation"][
                "measurement"
            ].update(total_calls=463),
            "elapsed row count does not match sweep count": lambda terminal: terminal[
                "rounds"
            ][0]["measurement"]["elapsed_ms"].pop(),
            "confirmation measurement target": lambda terminal: terminal[
                "confirmation"
            ]["measurement"].update(target_ms=200.0),
            "repeat reference is inconsistent with raw timings": lambda terminal: (
                force_four_sweeps(terminal, 45.0)
            ),
            "insufficient raw timing work": lambda terminal: force_four_sweeps(
                terminal, 44.0
            ),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected=expected):
                args = list(terminal_validation_fixture())
                terminal = args[2]["terminal_coordinate_refinement"]
                mutate(terminal)
                with self.assertRaisesRegex(RuntimeError, expected):
                    manifest.validate_terminal_coordinate_refinement(
                        Path("fixture.json"), *args
                    )

    def test_rejects_incomplete_or_duplicate_terminal_surface_rows(self) -> None:
        for expected, mutate in (
            (
                "neighbor rows omit an active value",
                lambda rows: rows.pop(2),
            ),
            (
                "duplicate terminal coordinate neighbor row",
                lambda rows: rows.append(copy.deepcopy(rows[0])),
            ),
        ):
            with self.subTest(expected=expected):
                args = list(terminal_validation_fixture())
                provenance = args[0]
                surface = provenance["flash_terminal_coordinate_surface_catalog"]
                rows = surface["leaves"][0]["coordinates"][0]["neighbors_by_value"]
                mutate(rows)
                provenance["flash_terminal_coordinate_surface_catalog_sha256"] = (
                    manifest.canonical_sha256(surface)
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    manifest.validate_terminal_coordinate_refinement(
                        Path("fixture.json"), *args
                    )

    def test_rejects_projection_config_that_does_not_apply_to_value(self) -> None:
        args = list(terminal_validation_fixture())
        terminal = args[2]["terminal_coordinate_refinement"]
        reused_id = terminal["rounds"][0]["reused_candidate_ids"][0]
        replacement_id = next(
            config_id
            for config_id, config in args[4].items()
            if config["fixture_coordinate"] == 3
        )

        def replace(value: object) -> object:
            if isinstance(value, dict):
                return {
                    replacement_id if key == reused_id else key: replace(item)
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [replace(item) for item in value]
            return replacement_id if value == reused_id else value

        replaced = replace(terminal)
        assert isinstance(replaced, dict)
        args[2]["terminal_coordinate_refinement"] = replaced
        replaced["config_manifest"][replacement_id]["config"] = args[4][replacement_id]
        replaced["config_manifest_sha256"] = manifest.canonical_sha256(
            replaced["config_manifest"]
        )
        with self.assertRaisesRegex(
            RuntimeError, "terminal projected config coordinate value"
        ):
            manifest.validate_terminal_coordinate_refinement(
                Path("fixture.json"), *args
            )

    def test_terminal_candidate_accepts_null_perf_failure_statuses(self) -> None:
        for status in ("accuracy_error", "source_rejected"):
            with self.subTest(status=status):
                args = list(terminal_validation_fixture())
                terminal = args[2]["terminal_coordinate_refinement"]
                failed_result = next(
                    result
                    for result in terminal["rounds"][0]["candidate_results"]
                    if result["status"] == "error"
                )
                failed_id = failed_result["config_id"]
                failed_result["status"] = status
                terminal["rounds"][1]["candidate_results"][0]["status"] = status
                args[5][failed_id]["status"] = status
                args[6][failed_id][0]["status"] = status
                failed_row = next(
                    row
                    for row in args[3]
                    if row["config_id"] == failed_id and row["status"] == "error"
                )
                failed_row["status"] = status

                manifest.validate_terminal_coordinate_refinement(
                    Path("fixture.json"), *args
                )

    def test_family_probe_capacity_is_zero_when_probes_are_disabled(self) -> None:
        leaf_catalog = [
            {"family": f"family_{index}", "compound_packet": None} for index in range(5)
        ]
        self.assertEqual(
            manifest.expected_family_probe_path_limit(leaf_catalog, 4, 0), 0
        )

    def test_sparse_search_generations_follow_structural_timeline(self) -> None:
        trial = {
            "num_generations": 6,
            "search_phase_metrics": {
                "schedule_anchor_pass_started": True,
                "qualification_passes_completed": 4,
                "measurement_timeline": [
                    {"pass_index": 0, "updates": [{"config_id": "initial"}]},
                    {"pass_index": 1, "updates": [{"config_id": "anchor"}]},
                    {"pass_index": 2, "updates": [{"config_id": "first"}]},
                    {"pass_index": 3, "updates": []},
                    {"pass_index": 4, "updates": [{"config_id": "qualified"}]},
                ],
            },
        }
        rows = [
            {"config_id": config_id, "generation": str(generation)}
            for config_id, generation in (
                ("initial", 0),
                ("anchor", 0),
                ("first", 1),
                ("qualified", 3),
                ("main-a", 5),
                ("main-b", 6),
            )
        ]

        manifest.validate_search_generations(Path("ledger.csv"), rows, trial)

        with self.assertRaisesRegex(RuntimeError, "structural config generations"):
            manifest.validate_search_generations(
                Path("ledger.csv"),
                [*rows, {"config_id": "unexpected", "generation": "2"}],
                trial,
            )
        with self.assertRaisesRegex(RuntimeError, "recorded final generation"):
            manifest.validate_search_generations(
                Path("ledger.csv"),
                rows[:-1],
                trial,
            )
        swapped = [dict(row) for row in rows]
        swapped[2]["generation"] = "3"
        swapped[3]["generation"] = "1"
        with self.assertRaisesRegex(RuntimeError, "first ledger generation"):
            manifest.validate_search_generations(Path("ledger.csv"), swapped, trial)

    def test_no_anchor_first_pass_maps_to_generation_one(self) -> None:
        trial = {
            "num_generations": 3,
            "search_phase_metrics": {
                "schedule_anchor_pass_started": False,
                "qualification_passes_completed": 2,
                "measurement_timeline": [
                    {"pass_index": 0, "updates": [{"config_id": "initial"}]},
                    {"pass_index": 1, "updates": [{"config_id": "first"}]},
                    {"pass_index": 2, "updates": []},
                ],
            },
        }
        rows = [
            {"config_id": "initial", "generation": "0"},
            {"config_id": "first", "generation": "1"},
            {"config_id": "main", "generation": "3"},
        ]

        manifest.validate_search_generations(Path("ledger.csv"), rows, trial)

    def test_generation_mapping_uses_earliest_lifecycle_attempt(self) -> None:
        trial = {
            "num_generations": 4,
            "search_phase_metrics": {
                "schedule_anchor_pass_started": False,
                "qualification_passes_completed": 2,
                "measurement_timeline": [
                    {"pass_index": 0, "updates": [{"config_id": "initial"}]},
                    {"pass_index": 1, "updates": [{"config_id": "repair"}]},
                    {
                        "pass_index": 2,
                        "updates": [
                            {"config_id": "repair"},
                            {"config_id": "new"},
                        ],
                    },
                ],
            },
        }
        rows = [
            {"config_id": "initial", "generation": "0"},
            {"config_id": "repair", "generation": "1"},
            {"config_id": "repair", "generation": "2"},
            {"config_id": "new", "generation": "2"},
            {"config_id": "main", "generation": "4"},
        ]

        manifest.validate_search_generations(Path("ledger.csv"), rows, trial)

    def test_search_generation_bounds_are_enforced(self) -> None:
        trial = {
            "num_generations": 2,
            "search_phase_metrics": {
                "schedule_anchor_pass_started": False,
                "qualification_passes_completed": 1,
                "measurement_timeline": [
                    {"pass_index": 0, "updates": [{"config_id": "initial"}]},
                    {"pass_index": 1, "updates": [{"config_id": "first"}]},
                ],
            },
        }
        rows = [
            {"config_id": "initial", "generation": "0"},
            {"config_id": "first", "generation": "1"},
            {"config_id": "main", "generation": "2"},
        ]

        with self.assertRaisesRegex(RuntimeError, "omit generation zero"):
            manifest.validate_search_generations(Path("ledger.csv"), rows[1:], trial)
        with self.assertRaisesRegex(RuntimeError, "exceeds the recorded final"):
            manifest.validate_search_generations(
                Path("ledger.csv"),
                [*rows, {"config_id": "too-late", "generation": "3"}],
                trial,
            )

    def test_flash_structural_population_budget_matches_producer_policy(self) -> None:
        cases = (
            ("design smaller than half", 100, 32, 8, 10, 50),
            ("complete design above half", 100, 63, 30, 57, 63),
            ("design fills population", 100, 100, 30, 57, 100),
            (
                "oversized design with qualification prefix fitting",
                100,
                120,
                30,
                50,
                50,
            ),
            ("oversized design preserving parent prefix", 100, 120, 60, 80, 60),
        )
        for (
            name,
            population_size,
            coverage_design_count,
            parent_prefix_count,
            qualification_prefix_count,
            expected,
        ) in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    manifest.flash_structural_population_budget(
                        population_size=population_size,
                        coverage_design_count=coverage_design_count,
                        parent_coverage_prefix_count=parent_prefix_count,
                        qualification_prefix_count=qualification_prefix_count,
                    ),
                    expected,
                )

    def qualification_fixture(
        self,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        payload, ledger, autotune_rows, metadata = fixture_payload(
            "dense",
            32768,
            7,
            manifest.EXPECTED_CASES[("dense", 32768)]["tuner_seed"],
            0,
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        trial = provenance["trials"][0]
        attempts = {
            row["config_id"]: {
                "generation": int(row["generation"]),
                "status": row["status"],
                "source_hash": ledger[position]["source_hash"],
                "perf_ms": float(row["perf_ms"]) if row["perf_ms"] else None,
            }
            for position, row in enumerate(autotune_rows)
            if row["status"] != "started"
        }
        return provenance, trial, attempts, metadata["configs"]

    def test_phase_identity_includes_family_probe_only_configs(self) -> None:
        provenance, trial, _attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        probe_config = copy.deepcopy(next(iter(metadata.values())))
        probe_config["cute_flash_wait_hint"] = 999
        probe_id = manifest.canonical_sha256(probe_config)[:16]
        phase["config_manifest"][probe_id] = {"config": probe_config}
        final_pass = phase["qualification_passes_completed"]
        source_hash = manifest.canonical_sha256({"probe": probe_id})
        phase["measurement_timeline"][final_pass]["updates"].append(
            {
                "config_id": probe_id,
                "attempt_perf": 9.0,
                "selection_perf": 9.0,
                "status": "ok",
                "source_hash": source_hash,
            }
        )
        phase["family_probe_paths"] = [
            {
                "starting_config_id": phase["initial_config_ids"][0],
                "rounds": [
                    {
                        "candidate_ids": [probe_id],
                        "results": [{"config_id": probe_id}],
                    }
                ],
            }
        ]
        lanes_by_leaf = manifest.flash_pipeline_lane_catalog(
            Path("fixture.json"), provenance
        )

        manifest.validate_phase_config_identity(
            Path("fixture.json"), provenance, phase, lanes_by_leaf
        )

        phase["family_probe_paths"] = []
        with self.assertRaisesRegex(RuntimeError, "canonical config manifest keys"):
            manifest.validate_phase_config_identity(
                Path("fixture.json"), provenance, phase, lanes_by_leaf
            )

    def test_strict_provenance_accepts_explicit_cross_shape_identity(self) -> None:
        tuner_seed = manifest.EXPECTED_CASES[("dense", 32768)]["tuner_seed"]
        payload, _ledger, _rows, _metadata = fixture_payload(
            "dense", 32768, 7, tuner_seed, 0
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        trial = provenance["trials"][0]
        shape = (3, 5, 32768, 128)
        dtype = "torch.bfloat16"
        trial["input_shapes"] = repr([shape] * 3)
        trial["dtypes"] = repr([dtype] * 3)
        context = provenance["flash_normalization_context"]
        context.update(
            dtype=dtype,
            head_dim=shape[3],
            num_bh=shape[0] * shape[1],
            tensor_4d_heads=shape[1],
        )
        provenance["flash_normalization_context_sha256"] = manifest.canonical_sha256(
            context
        )

        manifest.validate_strict_provenance(
            Path("fixture.json"),
            payload,
            "dense",
            shape[2],
            tuner_seed,
            expected_input_shape=shape,
            expected_input_dtype=dtype,
        )

        with self.assertRaisesRegex(RuntimeError, "trial input shapes"):
            manifest.validate_strict_provenance(
                Path("fixture.json"), payload, "dense", shape[2], tuner_seed
            )

    def test_v16_accepts_bounded_lane_repair_and_rejects_tampering(self) -> None:
        provenance, trial, _attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        leaf = phase["leaf_results"][0]
        lane = leaf["pipeline_lanes"][0]
        witness_id = lane["witness_config_id"]
        conditional_id = lane["conditional_candidate_ids"][0]
        repair_config = {
            **metadata[conditional_id],
            "cute_flash_wait_hint": 987654,
        }
        repair_id = manifest.canonical_sha256(repair_config)[:16]
        metadata[repair_id] = repair_config

        qualified_by_id = {
            result["config_id"]: result for result in leaf["qualified_results"]
        }
        for config_id in (witness_id, conditional_id):
            qualified_by_id[config_id].update(
                {
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "error",
                }
            )
        leaf["qualified_results"].append(
            {
                "config_id": repair_id,
                "attempt_perf": 9.0,
                "selection_perf": 9.0,
                "status": "ok",
                "pipeline_lanes": [{"key": lane["key"], "value": lane["value"]}],
            }
        )
        failed_results = [
            {
                "config_id": config_id,
                "attempt_perf": None,
                "selection_perf": None,
                "status": "error",
            }
            for config_id in sorted((witness_id, conditional_id))
        ]
        repair_decision = {
            "repair_index": 0,
            "candidate_results": failed_results,
            "selected_config_id": failed_results[0]["config_id"],
            "generated_config_ids": [repair_id],
        }
        lane.update(
            {
                "witness_succeeded": False,
                "successful_conditional_candidate_ids": [],
                "repair_candidate_ids": [repair_id],
                "successful_repair_candidate_ids": [repair_id],
                "repair_parent_decisions": [repair_decision],
            }
        )
        leaf["rounds"].append(
            {
                "candidate_config_ids": [repair_id],
                "neighbor_generation_limit": 200,
                "ordinary_neighbor_generation_limit": 0,
                "parent_decisions": [
                    {
                        "job_index": 0,
                        "kind": "failure_repair",
                        "pipeline_lane": {
                            "key": lane["key"],
                            "value": lane["value"],
                        },
                        "selection_kind": "ranked_failed_parent",
                        **repair_decision,
                    }
                ],
            }
        )
        for lane_result in leaf["pipeline_lanes"]:
            lane_result["rounds"].append(
                {
                    "candidate_config_ids": (
                        [repair_id] if lane_result is lane else []
                    ),
                    "neighbor_generation_limit": 200 if lane_result is lane else 0,
                }
            )
        successful_ids = [
            result["config_id"]
            for result in leaf["qualified_results"]
            if result["status"] in {"ok", "deduplicated"}
        ]
        leaf["retained_config_ids"] = successful_ids[:2]
        phase["candidate_count"] += 1
        for key in (
            "qualification_rounds_started",
            "qualification_rounds_completed",
            "qualification_passes_planned",
            "qualification_passes_started",
            "qualification_passes_completed",
        ):
            phase[key] += 1
        add_phase_config_identity(provenance, phase, metadata, default_perf=10.0)

        manifest.validate_structural_qualification_phase(
            Path("fixture.json"), provenance, trial
        )
        paired_worker._validate_structural_qualification_phase(
            Path("fixture.json"), provenance
        )

        for field, value, error in (
            ("qualification_failure_retries", 2, "failure_retries"),
            (
                "successful_repair_candidate_ids",
                [],
                "no successful evidence|lacks successful evidence",
            ),
        ):
            mutated_trial = copy.deepcopy(trial)
            if field == "qualification_failure_retries":
                mutated_trial["search_phase_metrics"][field] = value
            else:
                mutated_trial["search_phase_metrics"]["leaf_results"][0][
                    "pipeline_lanes"
                ][0][field] = value
            with self.assertRaisesRegex(RuntimeError, error):
                manifest.validate_structural_qualification_phase(
                    Path("fixture.json"), provenance, mutated_trial
                )

    def test_v16_rejects_measurement_introduced_on_wrong_pass(self) -> None:
        provenance, trial, _attempts, _metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        config_id = phase["leaf_results"][0]["rounds"][0]["candidate_config_ids"][0]
        update = next(
            update
            for update in phase["measurement_timeline"][1]["updates"]
            if update["config_id"] == config_id
        )
        phase["measurement_timeline"][1]["updates"].remove(update)
        phase["measurement_timeline"][2]["updates"].append(update)
        phase["measurement_timeline"][2]["updates"].sort(key=itemgetter("config_id"))

        with self.assertRaisesRegex(
            RuntimeError,
            "measurement introduction timeline|immutable pipeline parent decision",
        ):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), provenance, trial
            )
        with self.assertRaisesRegex(
            RuntimeError,
            "measurement introduction timeline|immutable pipeline parent decision",
        ):
            paired_worker._validate_structural_qualification_phase(
                Path("fixture.json"), provenance
            )

    def test_v16_timeline_source_repair_requires_same_pass_success(self) -> None:
        failed_id = "1" * 16
        successful_id = "2" * 16
        source_hash = "a" * 64
        phase = {
            "initial_config_ids": [failed_id],
            "qualification_passes_completed": 1,
            "measurement_timeline": [
                {
                    "pass_index": 0,
                    "updates": [
                        {
                            "config_id": failed_id,
                            "attempt_perf": None,
                            "selection_perf": None,
                            "status": "error",
                            "source_hash": source_hash,
                        }
                    ],
                },
                {
                    "pass_index": 1,
                    "updates": [
                        {
                            "config_id": failed_id,
                            "attempt_perf": 1.0,
                            "selection_perf": 1.0,
                            "status": "deduplicated",
                            "source_hash": source_hash,
                        },
                        {
                            "config_id": successful_id,
                            "attempt_perf": 1.0,
                            "selection_perf": 1.0,
                            "status": "ok",
                            "source_hash": source_hash,
                        },
                    ],
                },
            ],
        }
        configs = {failed_id: {}, successful_id: {}}
        for validator in (
            manifest.validate_measurement_timeline,
            paired_worker.validate_measurement_timeline,
        ):
            validator(Path("fixture.json"), phase, configs)

            wrong_source = copy.deepcopy(phase)
            wrong_source["measurement_timeline"][1]["updates"][1]["source_hash"] = (
                "b" * 64
            )
            with self.assertRaisesRegex(RuntimeError, "unproven.*source repair"):
                validator(Path("fixture.json"), wrong_source, configs)

            missing_source = copy.deepcopy(phase)
            missing_source["measurement_timeline"][1]["updates"][1].pop("source_hash")
            with self.assertRaisesRegex(RuntimeError, "timeline update"):
                validator(Path("fixture.json"), missing_source, configs)

    def test_v22_isolated_rebenchmark_timeout_invalidates_all_source_aliases(
        self,
    ) -> None:
        source_hash = "a" * 64
        first_id = "1" * 16
        second_id = "2" * 16
        successful = {
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "source_hash": source_hash,
        }
        failed = {
            "attempt_perf": None,
            "selection_perf": None,
            "status": "timeout",
            "source_hash": source_hash,
        }
        phase = {
            "initial_config_ids": [first_id, second_id],
            "qualification_passes_completed": 1,
            "measurement_timeline": [
                {
                    "pass_index": 0,
                    "updates": [
                        {"config_id": first_id, **successful},
                        {"config_id": second_id, **successful},
                    ],
                },
                {
                    "pass_index": 1,
                    "updates": [
                        {"config_id": first_id, **failed},
                        {"config_id": second_id, **failed},
                    ],
                },
            ],
        }
        configs = {first_id: {}, second_id: {}}

        states = manifest.validate_measurement_timeline(
            Path("fixture.json"), phase, configs
        )
        self.assertEqual(states[-1][first_id]["status"], "timeout")
        self.assertEqual(
            manifest.isolated_rebenchmark_timeout_source_hashes(phase),
            {source_hash},
        )

        partial = copy.deepcopy(phase)
        partial["measurement_timeline"][1]["updates"].pop()
        with self.assertRaisesRegex(RuntimeError, "effective-source invalidation"):
            manifest.validate_measurement_timeline(
                Path("fixture.json"), partial, configs
            )

        wrong_source = copy.deepcopy(phase)
        wrong_source["measurement_timeline"][1]["updates"][0]["source_hash"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "state transition"):
            manifest.validate_measurement_timeline(
                Path("fixture.json"), wrong_source, configs
            )

    def test_structural_execution_reconciles_timeout_metric(self) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        ) = structural_execution_fixture()
        with (
            mock.patch.object(
                manifest,
                "isolated_rebenchmark_timeout_source_hashes",
                return_value={"a" * 64},
            ),
            self.assertRaisesRegex(RuntimeError, "fewer isolated rebenchmark"),
        ):
            manifest.validate_structural_prefix_execution(
                Path("fixture.json"),
                provenance,
                trial,
                source_rows,
                metadata_configs,
                attempt_by_config,
                attempt_history_by_config,
            )

    def test_v16_rejects_missing_or_mutated_pipeline_parent_decisions(self) -> None:
        mutations = (
            lambda phase: phase["leaf_results"][0]["rounds"][0].pop("parent_decisions"),
            lambda phase: phase["leaf_results"][0]["rounds"][0]["parent_decisions"][
                0
            ].__setitem__("job_index", 99),
            lambda phase: phase["leaf_results"][0]["rounds"][0]["parent_decisions"][
                0
            ].__setitem__("selection_kind", "ranked_parent"),
            lambda phase: phase["leaf_results"][0]["rounds"][0]["parent_decisions"][0][
                "candidate_results"
            ][0].__setitem__("measurement_pass_index", 1),
            lambda phase: phase["leaf_results"][0]["rounds"][0]["parent_decisions"][
                1
            ].__setitem__("generated_config_ids", []),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                provenance, trial, _attempts, _metadata = self.qualification_fixture()
                mutation(trial["search_phase_metrics"])
                with self.assertRaisesRegex(
                    RuntimeError,
                    "pipeline parent decision|pipeline qualification round|malformed leaf|emitted candidate order",
                ):
                    manifest.validate_structural_qualification_phase(
                        Path("fixture.json"), provenance, trial
                    )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "pipeline parent decision|pipeline qualification round|malformed leaf|emitted candidate order",
                ):
                    paired_worker._validate_structural_qualification_phase(
                        Path("fixture.json"), provenance
                    )

    def test_v16_rejects_compound_source_and_lane_schema_tampering(self) -> None:
        mutations = (
            lambda phase: phase["compound_transfers"][0][
                "source_selection"
            ].__setitem__("attempted_config_ids", []),
            lambda phase: phase["compound_transfers"][0][
                "source_selection"
            ].__setitem__("combination_prefix_count", 1),
            lambda phase: phase["compound_transfers"][0].__setitem__("limit", 3),
            lambda phase: phase["leaf_results"][0]["pipeline_lanes"][0].pop(
                "initial_config_ids"
            ),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                provenance, trial, _attempts, _metadata = self.qualification_fixture()
                mutation(trial["search_phase_metrics"])
                with self.assertRaisesRegex(
                    RuntimeError,
                    "compound source|compound transfer|pipeline lane",
                ):
                    manifest.validate_structural_qualification_phase(
                        Path("fixture.json"), provenance, trial
                    )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "compound source|compound transfer|pipeline lane",
                ):
                    paired_worker._validate_structural_qualification_phase(
                        Path("fixture.json"), provenance
                    )

    def test_v16_clc_repair_requires_retryable_novel_same_lane_child(self) -> None:
        primary_id = "1" * 16
        repair_id = "2" * 16
        phase_configs = {
            primary_id: {
                "cute_flash_pipeline_family": "fa4_clc",
                "cute_flash_clc_heads_per_batch": 2,
            },
            repair_id: {
                "cute_flash_pipeline_family": "fa4_clc",
                "cute_flash_clc_heads_per_batch": 2,
            },
        }
        failed_state = {
            "attempt_perf": None,
            "selection_perf": None,
            "status": "error",
            "source_hash": "a" * 64,
        }
        measurement_states = [
            {primary_id: failed_state},
            {
                primary_id: failed_state,
                repair_id: {
                    "attempt_perf": 1.0,
                    "selection_perf": 1.0,
                    "status": "ok",
                    "source_hash": "b" * 64,
                },
            },
        ]
        decision = {
            "kind": "witness_failure_repair",
            "value": 2,
            "repair_index": 0,
            "candidate_results": [
                {
                    "config_id": primary_id,
                    **failed_state,
                    "measurement_pass_index": 0,
                }
            ],
            "selected_config_id": primary_id,
            "generated_config_ids": [repair_id],
            "neighbor_generation_limit": 200,
        }
        kwargs = {
            "repair_ids_by_value": {"2": [repair_id]},
            "primary_ids_by_value": {"2": primary_id},
            "values": [2],
            "expected_kind": "witness_failure_repair",
            "expected_leaf": {
                "family": "fa4_clc",
                "compound_packet": None,
                "softmax_disc": False,
            },
            "candidate_limit": 4,
            "phase_configs": phase_configs,
            "measurement_states": measurement_states,
            "label": "invalid immutable CLC witness repair",
        }
        for validator in (
            manifest.validate_failure_repair_decisions,
            paired_worker.validate_failure_repair_decisions,
        ):
            validator(Path("fixture.json"), [decision], **kwargs)

            wrong_budget = copy.deepcopy(decision)
            wrong_budget["neighbor_generation_limit"] = 199
            with self.assertRaisesRegex(RuntimeError, "CLC witness repair"):
                validator(Path("fixture.json"), [wrong_budget], **kwargs)

            wrong_status = copy.deepcopy(measurement_states)
            wrong_status[0][primary_id] = {
                "attempt_perf": 1.0,
                "selection_perf": 1.0,
                "status": "ok",
                "source_hash": "a" * 64,
            }
            with self.assertRaisesRegex(RuntimeError, "CLC witness repair"):
                validator(
                    Path("fixture.json"),
                    [decision],
                    **{**kwargs, "measurement_states": wrong_status},
                )

            wrong_lane = copy.deepcopy(phase_configs)
            wrong_lane[repair_id]["cute_flash_clc_heads_per_batch"] = 4
            with self.assertRaisesRegex(RuntimeError, "CLC witness repair"):
                validator(
                    Path("fixture.json"),
                    [decision],
                    **{**kwargs, "phase_configs": wrong_lane},
                )

    def test_v16_clc_conditional_decisions_record_exact_neighbor_split(self) -> None:
        witness_ids = {"1": "1" * 16, "2": "2" * 16}
        conditional_ids = {"1": ["3" * 16], "2": ["4" * 16]}
        source_hashes = {
            config_id: str(index) * 64
            for index, config_id in enumerate(
                [*witness_ids.values(), *conditional_ids["1"], *conditional_ids["2"]],
                start=1,
            )
        }
        pass_zero = {
            witness_ids["1"]: {
                "attempt_perf": 1.0,
                "selection_perf": 1.0,
                "status": "ok",
                "source_hash": source_hashes[witness_ids["1"]],
            },
            witness_ids["2"]: {
                "attempt_perf": 2.0,
                "selection_perf": 2.0,
                "status": "ok",
                "source_hash": source_hashes[witness_ids["2"]],
            },
        }
        pass_one = {
            **pass_zero,
            conditional_ids["1"][0]: {
                "attempt_perf": 0.5,
                "selection_perf": 0.5,
                "status": "ok",
                "source_hash": source_hashes[conditional_ids["1"][0]],
            },
            conditional_ids["2"][0]: {
                "attempt_perf": 0.6,
                "selection_perf": 0.6,
                "status": "ok",
                "source_hash": source_hashes[conditional_ids["2"][0]],
            },
        }

        def snapshot(config_id: str, pass_index: int) -> dict[str, Any]:
            states = (pass_zero, pass_one)[pass_index]
            return {
                "config_id": config_id,
                **states[config_id],
                "measurement_pass_index": pass_index,
            }

        witness_candidates = [
            {"value": value, **snapshot(config_id, 0)}
            for value, config_id in ((1, witness_ids["1"]), (2, witness_ids["2"]))
        ]
        retained_candidates = {
            value: sorted(
                [
                    snapshot(conditional_ids[str(value)][0], 1),
                    snapshot(witness_ids[str(value)], 1),
                ],
                key=itemgetter("selection_perf", "config_id"),
            )
            for value in (1, 2)
        }
        result = {
            "conditional_neighbor_generation_limit": 200,
            "witness_candidate_results": witness_candidates,
            "witness_selection_results": copy.deepcopy(witness_candidates),
            "selected_config_ids": list(witness_ids.values()),
            "conditional_parent_decisions": [
                {
                    "value": value,
                    "candidate_results": [snapshot(witness_ids[str(value)], 0)],
                    "selected_config_id": witness_ids[str(value)],
                    "generated_config_ids": conditional_ids[str(value)],
                    "neighbor_generation_limit": 100,
                }
                for value in (1, 2)
            ],
            "retained_value_decisions": [
                {
                    "value": value,
                    "candidate_results": retained_candidates[value],
                    "selected_config_id": conditional_ids[str(value)][0],
                }
                for value in (1, 2)
            ],
            "retained_ranking_results": [
                {"value": value, **snapshot(conditional_ids[str(value)][0], 1)}
                for value in (1, 2)
            ],
            "retained_config_ids": [
                conditional_ids["1"][0],
                conditional_ids["2"][0],
            ],
            "depth_selection": {
                "candidate_results": [
                    snapshot(conditional_ids["1"][0], 1),
                    snapshot(conditional_ids["2"][0], 1),
                ],
                "selected_representatives": [],
            },
        }
        kwargs = {
            "planned_values": [1, 2],
            "selected_values": [1, 2],
            "conditional_values": [1, 2],
            "retained_values": [1, 2],
            "witness_ids": witness_ids,
            "witness_repair_ids": {},
            "conditional_ids": conditional_ids,
            "conditional_repair_ids": {},
            "phase_configs": {config_id: {} for config_id in source_hashes},
            "measurement_states": [pass_zero, pass_one],
            "qualification_neighbor_limit": 200,
        }
        for validator in (
            manifest.validate_clc_decision_evidence,
            paired_worker.validate_clc_decision_evidence,
        ):
            validator(Path("fixture.json"), result, **kwargs)
            mutated = copy.deepcopy(result)
            mutated["conditional_parent_decisions"][1]["neighbor_generation_limit"] = 99
            with self.assertRaisesRegex(RuntimeError, "conditional-parent decision"):
                validator(Path("fixture.json"), mutated, **kwargs)

    def test_v16_timeline_source_repair_requires_matching_config_history(self) -> None:
        config_id = "1" * 16
        source_hash = "a" * 64
        phase = {
            "measurement_timeline": [
                {
                    "pass_index": 0,
                    "updates": [
                        {
                            "config_id": config_id,
                            "attempt_perf": None,
                            "selection_perf": None,
                            "status": "error",
                            "source_hash": source_hash,
                        }
                    ],
                },
                {
                    "pass_index": 1,
                    "updates": [
                        {
                            "config_id": config_id,
                            "attempt_perf": 1.0,
                            "selection_perf": 1.0,
                            "status": "deduplicated",
                            "source_hash": source_hash,
                        }
                    ],
                },
            ]
        }
        correct_history = {
            config_id: [
                {"generation": 1, "status": "error", "source_hash": source_hash},
                {
                    "generation": 2,
                    "status": "deduplicated",
                    "source_hash": source_hash,
                },
            ]
        }
        for validator in (
            manifest.validate_timeline_source_repairs,
            paired_worker.validate_timeline_source_repairs,
        ):
            validator(Path("fixture.json"), phase, correct_history)
            with self.assertRaisesRegex(RuntimeError, "matching same-source"):
                validator(Path("fixture.json"), phase, {})
            wrong_source = copy.deepcopy(correct_history)
            wrong_source[config_id][-1]["source_hash"] = "b" * 64
            with self.assertRaisesRegex(RuntimeError, "matching same-source"):
                validator(Path("fixture.json"), phase, wrong_source)

    def test_v16_compound_success_overshoot_retains_exact_target(self) -> None:
        provenance, trial, _attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        transfer_result = phase["compound_transfers"][0]
        primary_transfer = transfer_result["transfers"][0]
        primary_id = primary_transfer["transferred_config_id"]
        primary_transfer.update(
            {
                "status": "deduplicated",
                "attempt_perf": 10.0,
                "selection_perf": 10.0,
            }
        )
        source_id = next(
            config_id
            for config_id in phase["initial_config_ids"]
            if manifest.structural_leaf(metadata[config_id])
            == {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": False,
            }
            and config_id != primary_transfer["source_config_id"]
        )
        source_result = next(
            qualified
            for qualified in phase["leaf_results"][0]["qualified_results"]
            if qualified["config_id"] == source_id
        )
        source_result["selection_perf"] = 30.0
        target_config = {
            **metadata[source_id],
            "cute_flash_exp2_packet": "deg1_16x8",
        }
        target_id = manifest.canonical_sha256(target_config)[:16]
        metadata[target_id] = target_config
        transfer_result["source_selection"]["candidate_results"].append(
            {
                "config_id": source_id,
                "attempt_perf": 10.0,
                "selection_perf": 10.0,
                "status": "ok",
            }
        )
        transfer_result["source_selection"]["attempted_config_ids"].append(source_id)
        transfer_result["source_selection"]["selected_config_ids"].append(source_id)
        transfer_result["transfers"].append(
            {
                "source_config_id": source_id,
                "source_config": metadata[source_id],
                "transferred_config_id": target_id,
                "projected_config": target_config,
                "attempt_perf": 9.0,
                "selection_perf": 9.0,
                "status": "ok",
                "projection_overrides": {"cute_flash_exp2_packet": "deg1_16x8"},
                "projected_config_id": target_id,
                "preserved_pipeline_values": {
                    key: metadata[source_id][key]
                    for key in manifest.FLASH_PIPELINE_QUALIFICATION_KEYS
                    if key in metadata[source_id]
                },
            }
        )
        transfer_result.update(
            {
                "transfer_count": 2,
                "backfill_rounds": [
                    {
                        "repair_index": 0,
                        "required_successes": 1,
                        "failed_transfer_config_ids": [primary_id],
                        "attempted_source_config_ids": [source_id],
                        "generated_config_ids": [target_id],
                    }
                ],
                "successful_transfer_config_ids": [primary_id, target_id],
                "qualified_transfer_config_ids": [primary_id],
            }
        )
        phase["candidate_count"] += 1
        for key in (
            "qualification_rounds_started",
            "qualification_rounds_completed",
            "qualification_passes_planned",
            "qualification_passes_started",
            "qualification_passes_completed",
        ):
            phase[key] += 1
        add_phase_config_identity(provenance, phase, metadata, default_perf=10.0)
        repaired_source_hash = next(
            update["source_hash"]
            for update in phase["measurement_timeline"][3]["updates"]
            if update["config_id"] == primary_id
        )

        def share_repaired_source(value: object) -> None:
            if isinstance(value, dict):
                measured_config_id = value.get(
                    "config_id", value.get("transferred_config_id")
                )
                if measured_config_id == target_id and "source_hash" in value:
                    value["source_hash"] = repaired_source_hash
                for child in value.values():
                    share_repaired_source(child)
            elif isinstance(value, list):
                for child in value:
                    share_repaired_source(child)

        share_repaired_source(phase)
        pass_three = phase["measurement_timeline"][3]["updates"]
        primary_update = next(
            update for update in pass_three if update["config_id"] == primary_id
        )
        primary_update.update(
            {"attempt_perf": None, "selection_perf": None, "status": "error"}
        )
        phase["measurement_timeline"][4]["updates"].append(
            {
                "config_id": primary_id,
                "attempt_perf": 10.0,
                "selection_perf": 10.0,
                "status": "deduplicated",
                "source_hash": repaired_source_hash,
            }
        )
        phase["measurement_timeline"][4]["updates"].sort(key=itemgetter("config_id"))

        manifest.validate_structural_qualification_phase(
            Path("fixture.json"), provenance, trial
        )
        paired_worker._validate_structural_qualification_phase(
            Path("fixture.json"), provenance
        )

        fabricated = copy.deepcopy(trial)
        fabricated["search_phase_metrics"]["compound_transfers"][0][
            "qualified_transfer_config_ids"
        ].append(target_id)
        with self.assertRaisesRegex(RuntimeError, "qualified compound transfer"):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), provenance, fabricated
            )

    def test_v16_clc_failed_cell_uses_successful_axis_coverage(self) -> None:
        depth_ids = ["1" * 16, "2" * 16]
        divisors = [2, 4]
        projected_ids = iter(("a" * 16, "b" * 16, "c" * 16, "d" * 16))
        cells = []
        for depth_id in depth_ids:
            for divisor in divisors:
                config_id = next(projected_ids)
                failed = depth_id == depth_ids[0] and divisor == 4
                cells.append(
                    {
                        "depth_config_id": depth_id,
                        "divisor_value": divisor,
                        "projected_config_id": config_id,
                        "status": "error" if failed else "ok",
                    }
                )
        result = {
            "combination_depth_config_ids": depth_ids,
            "combination_divisor_values": divisors,
            "combination_cells": cells,
        }

        for verifier in (manifest, paired_worker):
            with self.subTest(verifier=verifier.__name__):
                coverage = verifier.clc_combination_coverage(result)
                self.assertEqual(
                    coverage["combination_candidate_ids"],
                    [cell["projected_config_id"] for cell in cells],
                )
                self.assertEqual(
                    coverage["successful_combination_depth_config_ids"], depth_ids
                )
                self.assertEqual(
                    coverage["successful_combination_divisor_values"], divisors
                )
                self.assertTrue(coverage["combination_projection_complete"])
                self.assertTrue(coverage["combination_row_coverage_complete"])
                self.assertTrue(coverage["combination_column_coverage_complete"])

                uncovered = copy.deepcopy(result)
                uncovered["combination_cells"][0]["status"] = "error"
                uncovered_coverage = verifier.clc_combination_coverage(uncovered)
                self.assertFalse(
                    uncovered_coverage["combination_row_coverage_complete"]
                )
                self.assertTrue(
                    uncovered_coverage["combination_column_coverage_complete"]
                )

    def test_v16_requires_historical_neighbor_and_evaluated_budgets(self) -> None:
        provenance, trial, _attempts, _metadata = self.qualification_fixture()
        for field, value, error in (
            (
                "neighbor_generation_limit_per_leaf_per_round",
                300,
                "neighbor generation limit",
            ),
            ("pipeline_candidate_limit_per_leaf_per_round", 5, "candidate.*limit"),
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(trial)
                mutated["search_phase_metrics"][field] = value
                with self.assertRaisesRegex(RuntimeError, error):
                    manifest.validate_structural_qualification_phase(
                        Path("fixture.json"), provenance, mutated
                    )

    def test_v16_requires_canonical_config_identity_evidence(self) -> None:
        provenance, _trial, _attempts, _metadata = self.qualification_fixture()
        for field in ("config_manifest", "initial_results"):
            for verifier in ("manifest", "worker"):
                with self.subTest(field=field, verifier=verifier):
                    mutated = copy.deepcopy(provenance)
                    mutated["trials"][0]["search_phase_metrics"].pop(field)
                    with self.assertRaisesRegex(
                        RuntimeError, "incomplete canonical config identity evidence"
                    ):
                        if verifier == "manifest":
                            manifest.validate_structural_qualification_phase(
                                Path("fixture.json"), mutated, mutated["trials"][0]
                            )
                        else:
                            paired_worker._validate_structural_qualification_phase(
                                Path("fixture.json"), mutated
                            )

    def test_v16_requires_complete_clc_catalog_evidence(self) -> None:
        provenance, _trial, _attempts, _metadata = self.qualification_fixture()
        mutations = (
            ("flash_clc_lane_catalog", None, "invalid flash CLC lane catalog"),
            ("flash_clc_lane_catalog_sha256", None, "CLC lane catalog digest"),
            ("flash_clc_lane_catalog_sha256", "0" * 64, "CLC lane catalog digest"),
        )
        for field, value, error in mutations:
            for verifier in ("manifest", "worker"):
                with self.subTest(field=field, value=value, verifier=verifier):
                    mutated = copy.deepcopy(provenance)
                    if value is None:
                        mutated.pop(field)
                    else:
                        mutated[field] = value
                    with self.assertRaisesRegex(RuntimeError, error):
                        if verifier == "manifest":
                            manifest.validate_structural_qualification_phase(
                                Path("fixture.json"), mutated, mutated["trials"][0]
                            )
                        else:
                            paired_worker._validate_structural_qualification_phase(
                                Path("fixture.json"), mutated
                            )

    def test_v16_requires_phase_lanes_to_equal_provenance_catalog(self) -> None:
        provenance, trial, _attempts, metadata = self.qualification_fixture()
        mutated_provenance = copy.deepcopy(provenance)
        mutated_provenance["flash_pipeline_lane_catalog"][0]["pipeline_lanes"][1][
            "value"
        ] = 4
        mutated_trial = copy.deepcopy(trial)
        add_phase_config_identity(
            mutated_provenance,
            mutated_trial["search_phase_metrics"],
            metadata,
            default_perf=10.0,
        )
        with self.assertRaisesRegex(RuntimeError, "phase lanes"):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), mutated_provenance, mutated_trial
            )

        malformed = copy.deepcopy(provenance)
        malformed["flash_pipeline_lane_catalog"].reverse()
        with self.assertRaisesRegex(RuntimeError, "lane catalog entry"):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), malformed, trial
            )

    def test_v16_rejects_incorrect_lane_round_budget_and_assignment(self) -> None:
        provenance, trial, _attempts, _metadata = self.qualification_fixture()

        wrong_budget = copy.deepcopy(trial)
        wrong_budget["search_phase_metrics"]["leaf_results"][0]["pipeline_lanes"][0][
            "rounds"
        ][0]["neighbor_generation_limit"] = 99
        with self.assertRaisesRegex(RuntimeError, "per-lane qualification budget"):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), provenance, wrong_budget
            )

        wrong_assignment = copy.deepcopy(trial)
        lanes = wrong_assignment["search_phase_metrics"]["leaf_results"][0][
            "pipeline_lanes"
        ]
        for lane in lanes:
            lane["rounds"][0]["candidate_config_ids"] = []
        with self.assertRaisesRegex(
            RuntimeError, "evidence is absent|unassigned candidates"
        ):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), provenance, wrong_assignment
            )

        omitted_generated_witness = copy.deepcopy(trial)
        omitted_generated_witness["search_phase_metrics"]["leaf_results"][0]["rounds"][
            0
        ]["candidate_config_ids"] = []
        with self.assertRaisesRegex(RuntimeError, "omits generated candidates"):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), provenance, omitted_generated_witness
            )

    def test_v16_reconciles_actual_lane_membership_and_completion(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        leaf = phase["leaf_results"][0]

        false_membership = copy.deepcopy(phase)
        membership = false_membership["leaf_results"][0]["qualified_results"][0][
            "pipeline_lanes"
        ][0]
        membership["value"] = 2 if membership["value"] == 3 else 3
        with self.assertRaisesRegex(RuntimeError, "actual pipeline membership"):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"), provenance, false_membership, attempts, metadata
            )

        incomplete = copy.deepcopy(phase)
        incomplete["leaf_results"][0]["pipeline_lanes"][0]["complete"] = False
        incomplete_trial = copy.deepcopy(trial)
        incomplete_trial["search_phase_metrics"] = incomplete
        with self.assertRaisesRegex(RuntimeError, "lane.*completion"):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), provenance, incomplete_trial
            )

        candidate_id = leaf["rounds"][0]["candidate_config_ids"][0]
        wrong_lane = copy.deepcopy(phase)
        lane_results = wrong_lane["leaf_results"][0]["pipeline_lanes"]
        nonmember_lane = next(
            lane
            for lane in lane_results
            if metadata[candidate_id].get(lane["key"]) != lane["value"]
        )
        nonmember_lane["conditional_candidate_ids"] = [candidate_id]
        nonmember_lane["successful_conditional_candidate_ids"] = [candidate_id]
        with self.assertRaisesRegex(RuntimeError, "invalid conditional child"):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"), provenance, wrong_lane, attempts, metadata
            )

    def test_v16_recomputes_candidate_and_leaf_accounting(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        for field, value in (
            ("candidate_count", phase["candidate_count"] + 1),
            ("leaves_with_candidates", 0),
        ):
            for reconcile in (
                manifest.reconcile_structural_qualification_phase,
                paired_worker._reconcile_structural_qualification_phase,
            ):
                with self.subTest(field=field, reconcile=reconcile.__module__):
                    mutated = copy.deepcopy(phase)
                    mutated[field] = value
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "structural qualification (candidate count|leaves with candidates)",
                    ):
                        reconcile(
                            Path("fixture.json"),
                            provenance,
                            mutated,
                            attempts,
                            metadata,
                        )

    def test_v16_recomputes_lane_diverse_retention_and_path_constraints(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        leaf = phase["leaf_results"][0]
        same_lane_ids = [
            qualified["config_id"]
            for qualified in leaf["qualified_results"]
            if metadata[qualified["config_id"]].get("cute_flash_kv_stage") == 2
        ][:2]
        mutated = copy.deepcopy(phase)
        mutated["leaf_results"][0]["retained_config_ids"] = same_lane_ids
        with self.assertRaisesRegex(
            RuntimeError, "retained candidates|witness success"
        ):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"), provenance, mutated, attempts, metadata
            )

        mutated = copy.deepcopy(phase)
        constrained = next(
            path
            for path in mutated["retained_families"][0]["starting_paths"]
            if path["pipeline_lane"] is not None
        )
        constrained["pipeline_lane"]["value"] = (
            2 if constrained["pipeline_lane"]["value"] == 3 else 3
        )
        with self.assertRaisesRegex(
            RuntimeError, "exact retained structural family ranking"
        ):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"), provenance, mutated, attempts, metadata
            )

    def test_exact_effective_search_space_provenance_is_self_consistent(self) -> None:
        config_ids = ["a" * 16, "b" * 16]
        provenance = {
            "autotune_initial_population_size": 100,
            "flash_exact_effective_search_space_size": len(config_ids),
            "flash_exact_effective_search_space_config_ids": config_ids,
            "flash_exact_effective_search_space_sha256": manifest.canonical_sha256(
                config_ids
            ),
        }

        self.assertEqual(
            manifest.exact_effective_search_space_ids(Path("result.json"), provenance),
            config_ids,
        )

        for key, value in (
            ("flash_exact_effective_search_space_size", 3),
            ("flash_exact_effective_search_space_config_ids", [config_ids[0]] * 2),
            ("flash_exact_effective_search_space_sha256", "0" * 64),
        ):
            with self.subTest(key=key):
                invalid = {**provenance, key: value}
                with self.assertRaisesRegex(RuntimeError, "invalid exact effective"):
                    manifest.exact_effective_search_space_ids(
                        Path("result.json"), invalid
                    )

    def write_fixture(self, root: Path) -> dict[str, Path]:
        paths = {}
        for index, (variant, seq_len, gpu, tuner_seed) in enumerate(manifest.CASES):
            payload, ledger, autotune_rows, metadata = fixture_payload(
                variant, seq_len, gpu, tuner_seed, index
            )
            directory = root / f"{variant}_s{seq_len}"
            directory.mkdir(parents=True)
            result_path = directory / "result.json"
            result_path.write_text(json.dumps(payload) + "\n")
            write_ledger(directory / "autotune.sources.csv", ledger)
            write_autotune_csv(directory / "autotune.csv", autotune_rows)
            (directory / "autotune.meta.jsonl").write_text(json.dumps(metadata) + "\n")
            paths[f"{variant}_{seq_len}"] = result_path
        return paths

    def test_builds_stable_complete_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            first = manifest.build_manifest(root)
            second = manifest.build_manifest(root)
            rows = list(csv.DictReader(io.StringIO(first)))
            dense_result_sha256 = manifest.file_sha256(paths["dense_32768"])
            dense_autotune_sha256 = manifest.file_sha256(
                paths["dense_32768"].with_name("autotune.csv")
            )
            dense_metadata_sha256 = manifest.file_sha256(
                paths["dense_32768"].with_name("autotune.meta.jsonl")
            )

        self.assertEqual(first, second)
        self.assertEqual(tuple(rows[0]), manifest.MANIFEST_FIELDS)
        self.assertEqual(
            [row["case"] for row in rows],
            [
                f"{variant}_{seq_len}"
                for variant, seq_len, _gpu, _seed in manifest.CASES
            ],
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["num_configs_tested"], "120")
        self.assertEqual(rows[0]["num_unique_sources"], "120")
        self.assertEqual(rows[0]["num_isolated_rebenchmark_timeouts"], "0")
        self.assertEqual(rows[0]["winner_is_coverage_design_member"], "false")
        self.assertEqual(rows[0]["winner_to_coverage_field_distance"], "1")
        selected = json.loads(rows[0]["selected_config_json"])
        self.assertEqual(
            rows[0]["selected_config_sha256"], manifest.canonical_sha256(selected)
        )
        self.assertEqual(rows[0]["result_sha256"], dense_result_sha256)
        self.assertEqual(rows[0]["autotune_csv_sha256"], dense_autotune_sha256)
        self.assertEqual(rows[0]["autotune_metadata_sha256"], dense_metadata_sha256)
        self.assertEqual(rows[0]["autotune_run_id"], rows[0]["source_ledger_run_id"])
        self.assertEqual(rows[0]["autotune_metadata_config_count"], "122")
        compiler_seed_policy = json.loads(rows[0]["compiler_seed_policy_json"])
        self.assertEqual(compiler_seed_policy["kind"], "canonical_cute_flash")
        self.assertEqual(compiler_seed_policy["raw_config_count"], 1)
        self.assertEqual(rows[0]["coverage_design_prefix_count"], "3")
        self.assertEqual(rows[0]["coverage_design_prefix_gen0_attempted_count"], "3")
        self.assertEqual(rows[0]["coverage_design_prefix_gen0_successful_count"], "3")
        active_counts = json.loads(
            rows[0]["coverage_active_value_successful_witness_counts_json"]
        )
        qualification_counts = json.loads(
            rows[0]["coverage_qualification_successful_witness_counts_json"]
        )
        self.assertEqual(
            [(item["key"], item["value"]) for item in qualification_counts],
            [
                ("cute_flash_pipeline_family", "fa4_2cta"),
                ("cute_flash_exp2_packet", "deg1_16x8"),
            ],
        )
        self.assertIn(
            {
                "key": "cute_flash_exp2_packet",
                "value": "1x1",
                "successful_generation_zero_witness_count": 98,
            },
            active_counts,
        )

    def test_build_rejects_tampered_retention_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_fixture(root)["dense_32768"]
            payload = json.loads(path.read_text())
            phase = payload["helion_overrides"]["autotune_provenance"]["trials"][0][
                "search_phase_metrics"
            ]
            phase["retained_families"][0]["score"] += 0.01
            path.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(
                RuntimeError, "exact retained structural family ranking"
            ):
                manifest.build_manifest(root)

    def test_accepts_sidecar_success_quarantined_by_isolated_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_fixture(root)["dense_32768"]
            payload = json.loads(path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            trial = provenance["trials"][0]
            phase = trial["search_phase_metrics"]
            retained_ids = {
                starting_path["config_id"]
                for family in phase["retained_families"]
                for starting_path in family["starting_paths"]
            }
            protected_ids = {
                result["config_id"] for result in phase["schedule_anchor_results"]
            }
            protected_ids.update(
                provenance["compiler_seed_policy"]["effective_config_ids"]
            )
            for leaf in phase["leaf_results"]:
                for lane in leaf["pipeline_lanes"]:
                    protected_ids.add(lane["witness_config_id"])
                    protected_ids.update(lane["conditional_candidate_ids"])
            target = next(
                result
                for result in phase["leaf_results"][0]["qualified_results"]
                if result["config_id"] in phase["initial_config_ids"]
                and result["config_id"] not in retained_ids | protected_ids
            )
            target_id = target["config_id"]
            source_hash = target["source_hash"]
            target.update(
                attempt_perf=None,
                selection_perf=None,
                status="timeout",
                measurement_pass_index=phase["qualification_passes_completed"],
            )
            final_updates = phase["measurement_timeline"][-1]["updates"]
            final_updates.append(
                {
                    "config_id": target_id,
                    "attempt_perf": None,
                    "selection_perf": None,
                    "status": "timeout",
                    "source_hash": source_hash,
                }
            )
            final_updates.sort(key=itemgetter("config_id"))
            trial["num_isolated_rebenchmark_timeouts"] = 1
            path.write_text(json.dumps(payload) + "\n")

            rows = list(csv.DictReader(io.StringIO(manifest.build_manifest(root))))

        self.assertEqual(rows[0]["num_isolated_rebenchmark_timeouts"], "1")

    def test_failed_source_aliases_do_not_satisfy_successful_candidate_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_fixture(root)["dense_32768"]
            payload = json.loads(result_path.read_text())
            trial = payload["helion_overrides"]["autotune_provenance"]["trials"][0]
            trial["num_successful_candidate_measurements"] = 99
            trial["num_source_deduplications"] = 200
            result_path.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(
                RuntimeError, "fewer than 100 actual successful"
            ):
                manifest.build_manifest(root)

    def test_plain_packet_is_active_but_only_compound_packet_is_qualified(self) -> None:
        active = [
            {"key": "cute_flash_pipeline_family", "value": "fa4_2cta"},
            {"key": "cute_flash_exp2_packet", "value": "1x1"},
            {"key": "cute_flash_exp2_packet", "value": "deg1_16x8"},
        ]
        self.assertEqual(
            manifest.structural_qualification_values(active),
            [active[0], active[2]],
        )

    def test_reconciles_qualification_phase_with_sidecars(self) -> None:
        mutations = ("status", "attempt_perf", "missing", "extra")
        for mutation in mutations:
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                path = self.write_fixture(root)["dense_32768"]
                payload = json.loads(path.read_text())
                phase = payload["helion_overrides"]["autotune_provenance"]["trials"][0][
                    "search_phase_metrics"
                ]
                qualified = phase["leaf_results"][0]["qualified_results"]
                if mutation == "status":
                    qualified[0]["status"] = "deduplicated"
                elif mutation == "attempt_perf":
                    qualified[0]["attempt_perf"] += 0.01
                elif mutation == "missing":
                    qualified.pop()
                else:
                    qualified.append(copy.deepcopy(qualified[0]))
                path.write_text(json.dumps(payload) + "\n")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "sidecar status|does not match the CSV|qualified membership|"
                    "canonical config manifest|invalid qualified measurement",
                ):
                    manifest.build_manifest(root)

    def test_replays_retention_parent_ranking_and_start_policy(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        manifest.reconcile_structural_qualification_phase(
            Path("fixture.json"), provenance, phase, attempts, metadata
        )

        def reject(mutated: dict[str, Any], pattern: str) -> None:
            with self.assertRaisesRegex(RuntimeError, pattern):
                manifest.reconcile_structural_qualification_phase(
                    Path("fixture.json"),
                    provenance,
                    mutated,
                    attempts,
                    metadata,
                )

        mutated = copy.deepcopy(phase)
        mutated["retained_families"][0]["score"] += 0.01
        reject(mutated, "exact retained structural family ranking")

        mutated = copy.deepcopy(phase)
        compound_path = next(
            path
            for path in mutated["retained_families"][0]["starting_paths"]
            if path["compound_packet"] is not None
        )
        compound_path["config_id"] = mutated["initial_config_ids"][0]
        reject(mutated, "exact retained structural family ranking")

        mutated = copy.deepcopy(phase)
        mutated["retained_families"][0]["starting_paths"].pop()
        reject(mutated, "exact retained structural family ranking")

    def test_v16_rejects_skipped_conditional_without_exact_space_proof(self) -> None:
        provenance, trial, _attempts, _metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        lane = phase["leaf_results"][0]["pipeline_lanes"][0]
        lane["space_exhausted"] = True
        lane["conditional_required"] = False
        with self.assertRaisesRegex(RuntimeError, "without exact-space proof"):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), provenance, trial
            )

    def test_v16_rejects_non_novel_conditional_candidate(self) -> None:
        provenance, trial, _attempts, _metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        leaf = phase["leaf_results"][0]
        lane = leaf["pipeline_lanes"][0]
        old_id = lane["conditional_candidate_ids"][0]
        initial_id = lane["initial_config_ids"][0]
        lane["conditional_candidate_ids"] = [initial_id]
        lane["successful_conditional_candidate_ids"] = [initial_id]
        for lane_round in lane["rounds"]:
            lane_round["candidate_config_ids"] = [
                initial_id if config_id == old_id else config_id
                for config_id in lane_round["candidate_config_ids"]
            ]
        with self.assertRaisesRegex(RuntimeError, "not novel"):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), provenance, trial
            )

    def test_v16_replays_clc_depth_snapshot_before_combinations(self) -> None:
        leaf = {
            "family": "fa4_clc",
            "compound_packet": None,
            "softmax_disc": False,
        }
        lanes = [
            ("cute_flash_kv_stage", 2),
            ("cute_flash_kv_stage", 3),
            ("cute_flash_s_stage", 2),
            ("cute_flash_s_stage", 4),
        ]
        initial_id = "1" * 16
        pipeline_id = "2" * 16
        witness_id = "3" * 16
        conditional_id = "4" * 16
        novel_combination_id = "5" * 16
        metadata = {
            initial_id: {
                "cute_flash_pipeline_family": "fa4_clc",
                "cute_flash_exp2_packet": "1x1",
                "cute_flash_kv_stage": 2,
                "cute_flash_s_stage": 2,
                "cute_flash_clc_heads_per_batch": 1,
            },
            pipeline_id: {
                "cute_flash_pipeline_family": "fa4_clc",
                "cute_flash_exp2_packet": "1x1",
                "cute_flash_kv_stage": 3,
                "cute_flash_s_stage": 4,
                "cute_flash_clc_heads_per_batch": 1,
            },
            witness_id: {
                "cute_flash_pipeline_family": "fa4_clc",
                "cute_flash_exp2_packet": "1x1",
                "cute_flash_kv_stage": 2,
                "cute_flash_s_stage": 4,
                "cute_flash_clc_heads_per_batch": 26,
            },
            conditional_id: {
                "cute_flash_pipeline_family": "fa4_clc",
                "cute_flash_exp2_packet": "1x1",
                "cute_flash_kv_stage": 3,
                "cute_flash_s_stage": 2,
                "cute_flash_clc_heads_per_batch": 26,
            },
            novel_combination_id: {
                "cute_flash_pipeline_family": "fa4_clc",
                "cute_flash_exp2_packet": "1x1",
                "cute_flash_kv_stage": 11,
                "cute_flash_s_stage": 2,
                "cute_flash_clc_heads_per_batch": 26,
            },
        }
        perfs = {
            initial_id: 1.0,
            pipeline_id: 4.0,
            witness_id: 2.0,
            conditional_id: 3.0,
            novel_combination_id: 0.5,
        }
        successful_results = {
            config_id: {
                "config_id": config_id,
                "attempt_perf": perf,
                "selection_perf": perf,
                "status": "ok",
            }
            for config_id, perf in perfs.items()
        }
        pre_combination_ids = {
            initial_id,
            pipeline_id,
            witness_id,
            conditional_id,
        }
        expected_candidates = sorted(
            (successful_results[config_id] for config_id in pre_combination_ids),
            key=itemgetter("selection_perf", "config_id"),
        )

        for verifier in (manifest, paired_worker):
            with self.subTest(verifier=verifier.__name__):
                members = [
                    {
                        "config_id": result["config_id"],
                        "selection_perf": result["selection_perf"],
                        "pipeline_lanes": verifier.config_pipeline_lanes(
                            metadata[result["config_id"]], lanes
                        ),
                    }
                    for result in expected_candidates
                ]
                representatives = [
                    {
                        "config_id": member["config_id"],
                        "assigned_pipeline_lane": verifier.pipeline_lane_metric(lane),
                    }
                    for member, lane in verifier.lane_diverse_members(
                        members, lanes, limit=2
                    )
                ]
                result = {
                    "family": "fa4_clc",
                    "combination_required": True,
                    "depth_selection": {
                        "candidate_results": expected_candidates,
                        "selected_representatives": representatives,
                    },
                    # A canonical projection may reuse an earlier depth candidate;
                    # only the second ID was introduced by the combination pass.
                    "combination_candidate_ids": [
                        initial_id,
                        novel_combination_id,
                    ],
                }
                verifier.reconcile_clc_depth_selection(
                    Path("fixture.json"),
                    result,
                    leaf=leaf,
                    pre_combination_ids=pre_combination_ids,
                    successful_ids=set(successful_results),
                    successful_results_by_id=successful_results,
                    metadata_configs=metadata,
                    pipeline_lanes=lanes,
                    retained_limit=2,
                )

                missing = copy.deepcopy(result)
                missing["depth_selection"]["candidate_results"].pop()
                with self.assertRaisesRegex(RuntimeError, "depth candidates"):
                    verifier.reconcile_clc_depth_selection(
                        Path("fixture.json"),
                        missing,
                        leaf=leaf,
                        pre_combination_ids=pre_combination_ids,
                        successful_ids=set(successful_results),
                        successful_results_by_id=successful_results,
                        metadata_configs=metadata,
                        pipeline_lanes=lanes,
                        retained_limit=2,
                    )

                leaked_combination = copy.deepcopy(result)
                leaked_combination["depth_selection"]["candidate_results"].insert(
                    0, successful_results[novel_combination_id]
                )
                with self.assertRaisesRegex(RuntimeError, "depth candidates"):
                    verifier.reconcile_clc_depth_selection(
                        Path("fixture.json"),
                        leaked_combination,
                        leaf=leaf,
                        pre_combination_ids=pre_combination_ids,
                        successful_ids=set(successful_results),
                        successful_results_by_id=successful_results,
                        metadata_configs=metadata,
                        pipeline_lanes=lanes,
                        retained_limit=2,
                    )

                tampered = copy.deepcopy(result)
                tampered["depth_selection"]["selected_representatives"][0][
                    "config_id"
                ] = novel_combination_id
                with self.assertRaisesRegex(RuntimeError, "depth representatives"):
                    verifier.reconcile_clc_depth_selection(
                        Path("fixture.json"),
                        tampered,
                        leaf=leaf,
                        pre_combination_ids=pre_combination_ids,
                        successful_ids=set(successful_results),
                        successful_results_by_id=successful_results,
                        metadata_configs=metadata,
                        pipeline_lanes=lanes,
                        retained_limit=2,
                    )

                legacy = {
                    "family": "fa4_clc",
                    "combination_required": True,
                    "combination_candidate_ids": [initial_id],
                }
                verifier.reconcile_clc_depth_selection(
                    Path("fixture.json"),
                    legacy,
                    leaf=leaf,
                    pre_combination_ids=pre_combination_ids,
                    successful_ids=set(successful_results),
                    successful_results_by_id=successful_results,
                    metadata_configs=metadata,
                    pipeline_lanes=lanes,
                    retained_limit=2,
                )

    def test_v16_replays_compound_projection_contract(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]

        def reject(
            mutated_phase: dict[str, Any],
            mutated_metadata: dict[str, dict[str, Any]],
            pattern: str,
        ) -> None:
            with self.assertRaisesRegex(RuntimeError, pattern):
                manifest.reconcile_structural_qualification_phase(
                    Path("fixture.json"),
                    provenance,
                    mutated_phase,
                    attempts,
                    mutated_metadata,
                )

        mutated = copy.deepcopy(phase)
        transfer = mutated["compound_transfers"][0]["transfers"][0]
        transfer["projected_config_id"] = "f" * 16
        reject(mutated, metadata, "compound target canonical ID")

        mutated = copy.deepcopy(phase)
        transfer = mutated["compound_transfers"][0]["transfers"][0]
        transfer["preserved_pipeline_values"]["cute_flash_kv_stage"] = 2
        reject(mutated, metadata, "preserved pipeline values")

        mutated_metadata = copy.deepcopy(metadata)
        transfer = phase["compound_transfers"][0]["transfers"][0]
        mutated_metadata[transfer["transferred_config_id"]]["fixture_leak"] = True
        reject(
            phase,
            mutated_metadata,
            "canonical ID|projected snapshot|inserted a non-owned field|"
            "manifest/sidecar config",
        )

    def test_v16_replays_compound_selection_performance(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        transfer = phase["compound_transfers"][0]["transfers"][0]
        transfer["selection_perf"] = 0.1
        with self.assertRaisesRegex(
            RuntimeError, "compound target selection performance"
        ):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"), provenance, phase, attempts, metadata
            )

    def test_v22_compound_retention_uses_final_post_probe_performance(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        lanes_by_leaf = manifest.flash_pipeline_lane_catalog(
            Path("fixture.json"), provenance
        )
        phase_configs, measurement_states = manifest.validate_phase_config_identity(
            Path("fixture.json"), provenance, phase, lanes_by_leaf
        )
        transfer_result = phase["compound_transfers"][0]
        transfer = transfer_result["transfers"][0]
        target_id = transfer["transferred_config_id"]
        transfer["measurement_pass_index"] = 2
        phase["family_probe_required"] = True
        phase["family_probe_generations"] = 1
        pre_probe_state = {
            key: transfer[key]
            for key in ("attempt_perf", "selection_perf", "status", "source_hash")
        }
        final_state = {**pre_probe_state, "selection_perf": 0.1}
        measurement_states[2][target_id] = pre_probe_state
        measurement_states[3][target_id] = final_state
        final_update = next(
            update
            for update in phase["measurement_timeline"][3]["updates"]
            if update["config_id"] == target_id
        )
        final_update["selection_perf"] = 0.1

        ordinary = phase["leaf_results"][0]
        ordinary_lanes = [
            (lane["key"], lane["value"]) for lane in ordinary["pipeline_lanes"]
        ]
        ordinary_members = [
            {
                "config_id": qualified["config_id"],
                "selection_perf": qualified["selection_perf"],
                "pipeline_lanes": manifest.config_pipeline_lanes(
                    metadata[qualified["config_id"]], ordinary_lanes
                ),
            }
            for qualified in ordinary["qualified_results"]
            if qualified["status"] in {"ok", "deduplicated"}
        ]
        phase["retained_families"] = manifest.expected_structural_retention(
            [
                {
                    "family": ordinary["family"],
                    "compound_packet": ordinary["compound_packet"],
                    "softmax_disc": ordinary["softmax_disc"],
                    "members": ordinary_members,
                    "pipeline_lanes": ordinary_lanes,
                },
                {
                    "family": transfer_result["family"],
                    "compound_packet": transfer_result["compound_packet"],
                    "softmax_disc": transfer_result["softmax_disc"],
                    "members": [
                        {
                            "config_id": target_id,
                            "selection_perf": 0.1,
                            "pipeline_lanes": frozenset(),
                        }
                    ],
                    "pipeline_lanes": [],
                },
            ],
            retained_per_leaf=provenance[
                "flash_structural_retained_candidates_per_leaf"
            ],
            retained_family_cap=provenance["flash_structural_retained_family_cap"],
            retained_family_limit=provenance["flash_structural_retained_family_limit"],
            retained_family_slowdown_limit=provenance[
                "flash_structural_retained_family_slowdown_limit"
            ],
            starting_path_limit=provenance["flash_structural_starting_path_limit"],
            parent_score_config_ids={
                config_id
                for config_id, state in measurement_states[2].items()
                if state["status"] in {"ok", "deduplicated"}
                and manifest.structural_leaf(metadata[config_id])["compound_packet"]
                is None
            },
        )
        phase["retained_path_count"] = sum(
            len(family["starting_paths"]) for family in phase["retained_families"]
        )

        with mock.patch.object(
            manifest,
            "validate_phase_config_identity",
            return_value=(phase_configs, measurement_states),
        ):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"), provenance, phase, attempts, metadata
            )

    def test_compound_global_uses_ordinary_leaf_lane_as_parent_path(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        manifest.reconcile_structural_qualification_phase(
            Path("fixture.json"), provenance, phase, attempts, metadata
        )
        paired_worker._reconcile_structural_qualification_phase(
            Path("fixture.json"), provenance, phase, attempts, metadata
        )

    def test_reconciles_generation_leaf_and_failure_semantics(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        candidate_id = phase["leaf_results"][0]["rounds"][1]["candidate_config_ids"][0]

        mutated_attempts = copy.deepcopy(attempts)
        mutated_attempts[candidate_id]["generation"] = 4
        with self.assertRaisesRegex(RuntimeError, "generation"):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"),
                provenance,
                phase,
                mutated_attempts,
                metadata,
            )

        mutated_metadata = copy.deepcopy(metadata)
        mutated_metadata[candidate_id]["cute_flash_pipeline_family"] = "fa4"
        with self.assertRaisesRegex(RuntimeError, "exact leaf|manifest/sidecar config"):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"),
                provenance,
                phase,
                attempts,
                mutated_metadata,
            )

        failed_phase = copy.deepcopy(phase)
        failed_attempts = copy.deepcopy(attempts)
        failed = next(
            item
            for item in failed_phase["leaf_results"][0]["qualified_results"]
            if item["config_id"] == phase["leaf_results"][0]["retained_config_ids"][0]
        )
        failed.update(status="error", attempt_perf=None, selection_perf=None)
        failed_attempts[failed["config_id"]].update(status="error", perf_ms=None)
        add_phase_config_identity(
            provenance,
            failed_phase,
            metadata,
            attempts=failed_attempts,
            default_perf=10.0,
        )
        with self.assertRaisesRegex(
            RuntimeError, "retained candidates|witness success"
        ):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"),
                provenance,
                failed_phase,
                failed_attempts,
                metadata,
            )

        deduplicated_phase = copy.deepcopy(phase)
        deduplicated = deduplicated_phase["leaf_results"][0]["qualified_results"][0]
        deduplicated["status"] = "deduplicated"
        with self.assertRaisesRegex(
            RuntimeError, "sidecar status|qualified measurement snapshot"
        ):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"),
                provenance,
                deduplicated_phase,
                attempts,
                metadata,
            )

    def test_equal_selection_perf_uses_config_id_tiebreak(self) -> None:
        provenance, trial, attempts, metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        leaf = phase["leaf_results"][0]
        for qualified in leaf["qualified_results"]:
            config_id = qualified["config_id"]
            qualified["attempt_perf"] = 1.0
            qualified["selection_perf"] = 1.0
            attempts[config_id]["perf_ms"] = 1.0
        lanes = [(lane["key"], lane["value"]) for lane in leaf["pipeline_lanes"]]
        members = [
            {
                "config_id": qualified["config_id"],
                "selection_perf": qualified["selection_perf"],
                "pipeline_lanes": manifest.config_pipeline_lanes(
                    metadata[qualified["config_id"]], lanes
                ),
            }
            for qualified in leaf["qualified_results"]
            if qualified["status"] in {"ok", "deduplicated"}
        ]
        leaf["retained_config_ids"] = [
            member["config_id"]
            for member, _lane in manifest.lane_diverse_members(
                members,
                lanes,
                limit=provenance["flash_structural_retained_candidates_per_leaf"],
            )
        ]
        transfer_result = phase["compound_transfers"][0]
        compound_members = [
            {
                "config_id": transfer["transferred_config_id"],
                "selection_perf": transfer["selection_perf"],
                "pipeline_lanes": frozenset(),
            }
            for transfer in transfer_result["transfers"]
            if transfer["transferred_config_id"]
            in transfer_result["qualified_transfer_config_ids"]
        ]
        phase["retained_families"] = manifest.expected_structural_retention(
            [
                {
                    "family": leaf["family"],
                    "compound_packet": None,
                    "members": members,
                    "pipeline_lanes": lanes,
                },
                {
                    "family": transfer_result["family"],
                    "compound_packet": transfer_result["compound_packet"],
                    "members": compound_members,
                    "pipeline_lanes": [],
                },
            ],
            retained_per_leaf=provenance[
                "flash_structural_retained_candidates_per_leaf"
            ],
            retained_family_cap=provenance["flash_structural_retained_family_cap"],
            retained_family_limit=provenance["flash_structural_retained_family_limit"],
            retained_family_slowdown_limit=provenance[
                "flash_structural_retained_family_slowdown_limit"
            ],
            starting_path_limit=provenance["flash_structural_starting_path_limit"],
        )
        phase["retained_path_count"] = sum(
            len(family["starting_paths"]) for family in phase["retained_families"]
        )
        add_phase_config_identity(
            provenance, phase, metadata, attempts=attempts, default_perf=1.0
        )
        manifest.reconcile_structural_qualification_phase(
            Path("fixture.json"), provenance, phase, attempts, metadata
        )
        leaf["retained_config_ids"].reverse()
        with self.assertRaisesRegex(RuntimeError, "retained candidates"):
            manifest.reconcile_structural_qualification_phase(
                Path("fixture.json"), provenance, phase, attempts, metadata
            )

    def test_rejects_spliced_main_autotune_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            dense_csv = paths["dense_32768"].with_name("autotune.csv")
            causal_csv = paths["causal_65536"].with_name("autotune.csv")
            dense_csv.write_bytes(causal_csv.read_bytes())
            with self.assertRaisesRegex(
                RuntimeError, "source ledger (row count|run_id)"
            ):
                manifest.build_manifest(root)

    def test_rejects_spliced_or_stale_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            dense_metadata = paths["dense_32768"].with_name("autotune.meta.jsonl")
            causal_metadata = paths["causal_65536"].with_name("autotune.meta.jsonl")
            dense_metadata.write_bytes(causal_metadata.read_bytes())
            with self.assertRaisesRegex(RuntimeError, "source ledger run_id"):
                manifest.build_manifest(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            metadata_path = paths["dense_32768"].with_name("autotune.meta.jsonl")
            metadata = json.loads(metadata_path.read_text())
            metadata["run_id"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata) + "\n")
            with self.assertRaisesRegex(RuntimeError, "computed run_id"):
                manifest.build_manifest(root)

    def test_rejects_false_effective_cache_provenance(self) -> None:
        for key, value in (
            ("force_autotune", True),
            ("effective_cache_read_bypass", False),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = self.write_fixture(root)["dense_32768"]
                metadata_path = path.with_name("autotune.meta.jsonl")
                metadata = json.loads(metadata_path.read_text())
                metadata["settings"][key] = value
                metadata_path.write_text(json.dumps(metadata) + "\n")
                with self.assertRaisesRegex(RuntimeError, rf"settings\.{key}"):
                    manifest.build_manifest(root)

    def test_rejects_config_id_not_grounded_in_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            metadata_path = paths["dense_32768"].with_name("autotune.meta.jsonl")
            metadata = json.loads(metadata_path.read_text())
            config_id = next(iter(metadata["configs"]))
            metadata["configs"][config_id]["tampered"] = True
            metadata_path.write_text(json.dumps(metadata) + "\n")
            with self.assertRaisesRegex(RuntimeError, "canonical config ID"):
                manifest.build_manifest(root)

    def test_manifest_output_cannot_overwrite_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            result = paths["dense_32768"]
            for evidence in (
                result,
                result.with_name("autotune.csv"),
                result.with_name("autotune.meta.jsonl"),
                result.with_name("autotune.sources.csv"),
            ):
                with (
                    self.subTest(evidence=evidence.name),
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "build_strict_manifest.py",
                            "--artifact-root",
                            str(root),
                            "--output",
                            str(evidence),
                        ],
                    ),
                    self.assertRaisesRegex(RuntimeError, "output collides"),
                ):
                    manifest.main()

    def test_rejects_missing_and_duplicate_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            paths["dense_32768"].unlink()
            with self.assertRaisesRegex(RuntimeError, "strict result set mismatch"):
                manifest.build_manifest(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            duplicate = root / "duplicate" / "result.json"
            duplicate.parent.mkdir()
            duplicate.write_text(paths["dense_32768"].read_text())
            with self.assertRaisesRegex(RuntimeError, "duplicate strict result"):
                manifest.build_manifest(root)

    def test_rejects_seeded_or_truncated_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            path = paths["dense_32768"]
            payload = json.loads(path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            provenance["user_seed_configs"] = True
            path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(RuntimeError, "user_seed_configs"):
                manifest.build_manifest(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            path = paths["dense_32768"]
            payload = json.loads(path.read_text())
            trial = payload["helion_overrides"]["autotune_provenance"]["trials"][0]
            trial["num_unique_sources"] = 99
            path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(RuntimeError, "unique source count"):
                manifest.build_manifest(root)

    def test_rejects_terminal_repeat_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_fixture(root)["dense_32768"]
            payload = json.loads(path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            provenance["cute_flash_env_overrides"] = {
                "HELION_CAP_REBENCHMARK_REPEAT": "4"
            }
            path.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(
                RuntimeError, "provenance.cute_flash_env_overrides"
            ):
                manifest.build_manifest(root)

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
                    path = self.write_fixture(root)["dense_32768"]
                    payload = json.loads(path.read_text())
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
                    path.write_text(json.dumps(payload) + "\n")
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "unrestricted.path.*exhaust|unrestricted_path_exhausts",
                    ):
                        manifest.build_manifest(root)

    def test_rejects_unrestricted_path_stopping_before_lfbo_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_fixture(root)["dense_32768"]
            payload = json.loads(path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            provenance["autotune_lfbo_max_generations"] += 1
            path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(
                RuntimeError, "unrestricted path generation budget"
            ):
                manifest.build_manifest(root)

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
                path = self.write_fixture(root)["dense_32768"]
                payload = json.loads(path.read_text())
                phase = payload["helion_overrides"]["autotune_provenance"]["trials"][0][
                    "search_phase_metrics"
                ]
                phase[key] = value
                path.write_text(json.dumps(payload) + "\n")
                with self.assertRaisesRegex(RuntimeError, match):
                    manifest.build_manifest(root)

    def test_rejects_shape_specific_structural_coverage_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            path = paths["dense_65536"]
            payload = json.loads(path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            design = provenance["flash_structural_coverage_design"]
            design[1], design[2] = design[2], design[1]
            configs = [item["config"] for item in design]
            provenance["flash_structural_coverage_design_sha256"] = (
                manifest.canonical_sha256(configs)
            )
            path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(
                RuntimeError,
                "length-dependent structural coverage designs within one legality class",
            ):
                manifest.build_manifest(root)

    def test_rejects_shape_specific_compiler_seed_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            path = paths["dense_65536"]
            payload = json.loads(path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            phase = provenance["trials"][0]["search_phase_metrics"]
            policy = provenance["compiler_seed_policy"]
            replacement_id = next(
                record["config_id"]
                for record in phase["initial_results"]
                if record["status"] in {"ok", "deduplicated"}
                and record["config_id"] not in policy["effective_config_ids"]
            )
            policy["effective_config_ids"] = [replacement_id]
            policy["effective_config_ids_sha256"] = manifest.canonical_sha256(
                [replacement_id]
            )
            path.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(
                RuntimeError,
                "length-dependent compiler seed policies within one legality class",
            ):
                manifest.build_manifest(root)

    def test_rejects_shape_specific_terminal_policy_or_surface(self) -> None:
        for field, expected in (
            (
                "terminal_refinement_policy_sha256",
                "length-dependent terminal refinement policies",
            ),
            (
                "terminal_coordinate_surface_sha256",
                "length-dependent terminal coordinate surfaces",
            ),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_fixture(root)
                validate_result = manifest.validate_result

                def validate_with_mismatch(
                    artifact_root: Path,
                    path: Path,
                    variant: str,
                    seq_len: int,
                    *,
                    _validate_result: Callable[
                        [Path, Path, str, int], dict[str, Any]
                    ] = validate_result,
                    _field: str = field,
                ) -> dict[str, Any]:
                    row = _validate_result(artifact_root, path, variant, seq_len)
                    if variant == "dense" and seq_len == 65536:
                        row[_field] = "0" * 64
                    return row

                with (
                    mock.patch.object(
                        manifest,
                        "validate_result",
                        side_effect=validate_with_mismatch,
                    ),
                    self.assertRaisesRegex(RuntimeError, expected),
                ):
                    manifest.build_manifest(root)

    def test_rejects_malformed_active_structural_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            path = paths["dense_32768"]
            payload = json.loads(path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            provenance["flash_structural_coverage_active_values"] = [{"key": "absent"}]
            path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(
                RuntimeError, "invalid active structural value"
            ):
                manifest.build_manifest(root)

    def test_rejects_incomplete_structural_qualification_prefix(self) -> None:
        payload, _ledger, _autotune_rows, _metadata = fixture_payload(
            "dense",
            32768,
            7,
            manifest.EXPECTED_CASES[("dense", 32768)]["tuner_seed"],
            0,
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        design = provenance["flash_structural_coverage_design"]
        second_ordinary = {
            **design[0]["config"],
            "cute_flash_kv_stage": 3,
            "cute_flash_wait_hint": -4,
        }
        design.append(
            {
                "config": second_ordinary,
                "config_sha256": manifest.canonical_sha256(second_ordinary),
            }
        )
        provenance["flash_structural_coverage_design_count"] = len(design)
        provenance["flash_structural_coverage_design_sha256"] = (
            manifest.canonical_sha256([item["config"] for item in design])
        )
        provenance["flash_structural_injected_design_count"] = len(design)
        with self.assertRaisesRegex(RuntimeError, "underrepresents exact leaf"):
            manifest.validate_coverage(
                Path("dense_s32768/result.json"),
                provenance,
                provenance["selected_config"],
            )

    def test_rejects_incorrect_structural_population_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            path = paths["dense_32768"]
            payload = json.loads(path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            provenance["flash_structural_population_budget"] = 49
            path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(RuntimeError, "structural population budget"):
                manifest.build_manifest(root)

    def test_rejects_structural_prefix_missing_from_generation_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            result_path = paths["dense_32768"]
            payload = json.loads(result_path.read_text())
            prefix_config = payload["helion_overrides"]["autotune_provenance"][
                "flash_structural_coverage_design"
            ][-1]["config"]
            prefix_id = manifest.canonical_sha256(prefix_config)[:16]
            for sidecar_name, writer_fn in (
                ("autotune.sources.csv", write_ledger),
                ("autotune.csv", write_autotune_csv),
            ):
                sidecar_path = result_path.with_name(sidecar_name)
                rows = list(csv.DictReader(io.StringIO(sidecar_path.read_text())))
                for row in rows:
                    if row["config_id"] == prefix_id:
                        row["generation"] = "1"
                writer_fn(sidecar_path, rows)
            with self.assertRaisesRegex(RuntimeError, "first ledger generation"):
                manifest.build_manifest(root)

    def test_rejects_initial_population_reordered_after_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_fixture(root)["dense_32768"]
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
                manifest.build_manifest(root)

    def test_rejects_inconsistent_injected_design_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            result_path = paths["dense_32768"]
            payload = json.loads(result_path.read_text())
            payload["helion_overrides"]["autotune_provenance"][
                "flash_structural_injected_design_count"
            ] = 2
            result_path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(
                RuntimeError, "injected structural design count"
            ):
                manifest.build_manifest(root)

    def test_reports_zero_successful_active_child_without_rejecting(self) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        ) = structural_execution_fixture()
        first_id = manifest.canonical_sha256(
            provenance["flash_structural_coverage_design"][0]["config"]
        )[:16]
        for row in source_rows:
            if row["config_id"] == first_id and row["status"] == "ok":
                row["status"] = "error"
        attempt_by_config[first_id].update(status="error", perf_ms=None)
        attempt_history_by_config[first_id][-1].update(status="error", perf_ms=None)
        phase = trial["search_phase_metrics"]
        leaf_result = phase["leaf_results"][0]
        qualified = next(
            item
            for item in leaf_result["qualified_results"]
            if item["config_id"] == first_id
        )
        qualified.update(status="error", attempt_perf=None, selection_perf=None)
        retained_ids = sorted(
            item["config_id"]
            for item in leaf_result["qualified_results"]
            if item["status"] == "ok"
        )[:2]
        leaf_result["retained_config_ids"] = retained_ids
        phase["retained_families"][0]["starting_paths"] = [
            {
                "family": leaf_result["family"],
                "compound_packet": leaf_result["compound_packet"],
                "config_id": config_id,
                "unrestricted": False,
                "pipeline_lane": None,
            }
            for config_id in retained_ids
        ]
        phase["retained_families"][0]["starting_paths"].append(
            {
                "family": leaf_result["family"],
                "compound_packet": leaf_result["compound_packet"],
                "config_id": retained_ids[0],
                "unrestricted": True,
                "pipeline_lane": None,
            }
        )
        phase["retained_path_count"] = len(retained_ids) + 1
        add_phase_config_identity(
            provenance,
            phase,
            metadata_configs,
            attempts=attempt_by_config,
        )

        result = manifest.validate_structural_prefix_execution(
            Path("fixture.json"),
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        )

        child = next(
            item
            for item in result["active_value_successful_witness_counts"]
            if item["key"] == "cute_flash_wait_hint" and item["value"] == 0
        )
        self.assertEqual(child["successful_generation_zero_witness_count"], 0)
        self.assertEqual(result["active_value_count"], 3)
        self.assertEqual(result["active_value_successful_witness_count"], 2)

    def test_accepts_qualified_family_success_outside_design_prefix(self) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        ) = structural_execution_fixture(
            fail_prefix=True, successful_family_outside_prefix=True
        )

        result = manifest.validate_structural_prefix_execution(
            Path("fixture.json"),
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        )

        self.assertEqual(result["successful_count"], 0)
        self.assertEqual(
            result["qualification_successful_witness_counts"][0][
                "successful_generation_zero_witness_count"
            ],
            1,
        )

    def test_rejects_qualified_family_without_successful_generation_zero_config(
        self,
    ) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        ) = structural_execution_fixture(fail_prefix=True)

        with self.assertRaisesRegex(
            RuntimeError,
            "compiler seed .* lacks a successful|invalid retained starting paths|"
            "qualified structural values lack a successful",
        ):
            manifest.validate_structural_prefix_execution(
                Path("fixture.json"),
                provenance,
                trial,
                source_rows,
                metadata_configs,
                attempt_by_config,
                attempt_history_by_config,
            )

    def test_rejects_wrong_generation_zero_population_size(self) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        ) = structural_execution_fixture()
        filler_id = manifest.canonical_sha256(
            {
                "cute_flash_pipeline_family": "fa4",
                "cute_flash_exp2_packet": "1x1",
                "fixture_candidate": 99,
            }
        )[:16]
        for row in source_rows:
            if row["config_id"] == filler_id:
                row["generation"] = "1"

        with self.assertRaisesRegex(
            RuntimeError, "generation-zero ledger has an incomplete initial population"
        ):
            manifest.validate_structural_prefix_execution(
                Path("fixture.json"),
                provenance,
                trial,
                source_rows,
                metadata_configs,
                attempt_by_config,
                attempt_history_by_config,
            )

    def test_accepts_exact_generation_zero_population_smaller_than_nominal(
        self,
    ) -> None:
        for deduplicate_last in (False, True):
            with self.subTest(deduplicate_last=deduplicate_last):
                (
                    provenance,
                    trial,
                    source_rows,
                    metadata_configs,
                    attempt_by_config,
                    attempt_history_by_config,
                ) = exact_structural_execution_fixture(
                    deduplicate_last=deduplicate_last
                )

                result = manifest.validate_structural_prefix_execution(
                    Path("fixture.json"),
                    provenance,
                    trial,
                    source_rows,
                    metadata_configs,
                    attempt_by_config,
                    attempt_history_by_config,
                )

                self.assertEqual(result["prefix_count"], 2)
                self.assertEqual(result["attempted_count"], 2)

    def test_compiler_seed_policy_accepts_normalized_duplicate_raw_seeds(self) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        ) = structural_execution_fixture()
        provenance["compiler_seed_config_count"] = 2
        provenance["compiler_seed_policy"]["raw_config_count"] = 2

        result = manifest.validate_structural_prefix_execution(
            Path("fixture.json"),
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        )

        self.assertEqual(result["compiler_seed_policy"]["raw_config_count"], 2)
        self.assertEqual(len(result["compiler_seed_policy"]["effective_config_ids"]), 1)

    def test_compiler_seed_policy_rejects_unmeasured_generation_zero_seed(
        self,
    ) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            _attempt_by_config,
            _attempt_history_by_config,
        ) = structural_execution_fixture()
        missing_id = "f" * 16
        policy = provenance["compiler_seed_policy"]
        policy["effective_config_ids"] = [missing_id]
        policy["effective_config_ids_sha256"] = manifest.canonical_sha256([missing_id])

        with self.assertRaisesRegex(RuntimeError, "missing from generation zero"):
            manifest.validate_compiler_seed_policy(
                Path("fixture.json"),
                provenance,
                phase=trial["search_phase_metrics"],
                metadata_configs=metadata_configs,
                source_rows=source_rows,
            )

    def test_compiler_seed_policy_rejects_failed_generation_zero_seed(self) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            _attempt_by_config,
            _attempt_history_by_config,
        ) = structural_execution_fixture(
            fail_prefix=True, successful_family_outside_prefix=True
        )
        failed_id = trial["search_phase_metrics"]["initial_config_ids"][0]
        policy = provenance["compiler_seed_policy"]
        policy["effective_config_ids"] = [failed_id]
        policy["effective_config_ids_sha256"] = manifest.canonical_sha256([failed_id])

        with self.assertRaisesRegex(
            RuntimeError, "lacks a successful generation-zero measurement"
        ):
            manifest.validate_compiler_seed_policy(
                Path("fixture.json"),
                provenance,
                phase=trial["search_phase_metrics"],
                metadata_configs=metadata_configs,
                source_rows=source_rows,
            )

    def test_compiler_seed_policy_rejects_schema_tampering(self) -> None:
        provenance, *_rest = structural_execution_fixture()
        mutations = (
            ("schema_version", 2),
            ("schema_version", True),
            ("kind", "arbitrary"),
            ("heuristic_names", ["other"]),
            ("raw_config_count", 0),
            ("effective_config_ids_sha256", "0" * 64),
            ("timeout_retry_repetitions", 2),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(provenance)
                changed["compiler_seed_policy"][field] = value
                with self.assertRaises(RuntimeError):
                    manifest.validate_compiler_seed_policy(
                        Path("fixture.json"), changed
                    )

        changed = copy.deepcopy(provenance)
        changed["compiler_seed_policy"]["unexpected"] = True
        with self.assertRaisesRegex(RuntimeError, "malformed compiler seed policy"):
            manifest.validate_compiler_seed_policy(Path("fixture.json"), changed)

    def test_rejects_unsuccessful_exact_generation_zero_member(self) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        ) = exact_structural_execution_fixture()
        phase = trial["search_phase_metrics"]
        retained_ids = set(phase["leaf_results"][0]["retained_config_ids"])
        failed_id = next(
            config_id
            for config_id in phase["exact_space_config_ids"]
            if config_id not in retained_ids
        )
        terminal = next(
            row
            for row in source_rows
            if row["config_id"] == failed_id and row["status"] == "ok"
        )
        terminal["status"] = "error"
        attempt_by_config[failed_id].update(status="error", perf_ms=None)
        attempt_history_by_config[failed_id][-1].update(status="error", perf_ms=None)
        qualified = next(
            result
            for result in phase["leaf_results"][0]["qualified_results"]
            if result["config_id"] == failed_id
        )
        qualified.update(status="error", attempt_perf=None, selection_perf=None)
        add_phase_config_identity(
            provenance,
            phase,
            metadata_configs,
            attempts=attempt_by_config,
        )

        with self.assertRaisesRegex(RuntimeError, "not successfully exhausted"):
            manifest.validate_structural_prefix_execution(
                Path("fixture.json"),
                provenance,
                trial,
                source_rows,
                metadata_configs,
                attempt_by_config,
                attempt_history_by_config,
            )

    def test_rejects_same_size_wrong_exact_generation_zero_id(self) -> None:
        (
            provenance,
            trial,
            source_rows,
            metadata_configs,
            attempt_by_config,
            attempt_history_by_config,
        ) = exact_structural_execution_fixture()
        phase = trial["search_phase_metrics"]
        replaced_id = phase["exact_space_config_ids"][-1]
        replacement_config = {
            **metadata_configs[replaced_id],
            "fixture_wrong_exact_member": True,
        }
        replacement_id = manifest.canonical_sha256(replacement_config)[:16]
        metadata_configs[replacement_id] = replacement_config
        for row in source_rows:
            if row["config_id"] == replaced_id:
                row["config_id"] = replacement_id

        with self.assertRaisesRegex(
            RuntimeError, "generation-zero initial population order"
        ):
            manifest.validate_structural_prefix_execution(
                Path("fixture.json"),
                provenance,
                trial,
                source_rows,
                metadata_configs,
                attempt_by_config,
                attempt_history_by_config,
            )

    def test_rejects_winner_missing_from_successful_ledger_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            result_path = paths["causal_65536"]
            payload = json.loads(result_path.read_text())
            selected = payload["helion_overrides"]["autotune_provenance"][
                "selected_source_sha256"
            ]
            ledger_path = result_path.with_name("autotune.sources.csv")
            with ledger_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            winner = next(
                row
                for row in rows
                if row["source_hash"] == selected and row["status"] == "ok"
            )
            winner["status"] = "error"
            write_ledger(ledger_path, rows)
            with self.assertRaisesRegex(RuntimeError, "selected source is not present"):
                manifest.build_manifest(root)

    def test_rejects_inconsistent_source_ledger_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            ledger_path = paths["dense_32768"].with_name("autotune.sources.csv")
            with ledger_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            terminal = next(row for row in rows if row["status"] == "ok")
            terminal["generation"] = str(int(terminal["generation"]) + 1)
            write_ledger(ledger_path, rows)
            with self.assertRaisesRegex(
                RuntimeError, "malformed lifecycle|inconsistent lifecycle provenance"
            ):
                manifest.build_manifest(root)

    def test_accepts_failed_attempt_repaired_by_source_alias(self) -> None:
        payload, rows, _autotune_rows, _metadata = fixture_payload(
            "dense", 32768, 7, 2026081507, 0
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        trial = provenance["trials"][0]
        selected_config = provenance["selected_config"]
        selected_source = provenance["selected_source_sha256"]
        aliased_sources = {
            row["source_hash"] for row in rows if row["status"] == "deduplicated"
        }
        failed = next(
            row
            for row in rows
            if row["status"] == "ok"
            and row["source_hash"] != selected_source
            and row["source_hash"] not in aliased_sources
        )
        failed_config_id = failed["config_id"]
        failed_position = rows.index(failed)
        repairing_row = next(
            row
            for row in rows[failed_position + 1 :]
            if row["status"] == "ok" and int(row["generation"]) > 0
        )
        successful_source = repairing_row["source_hash"]
        repair_generation = repairing_row["generation"]
        failure_generation = str(int(repair_generation) - 1)
        for row in rows:
            if row["config_id"] == failed_config_id:
                row["source_hash"] = successful_source
                row["generation"] = failure_generation
                if row["status"] == "ok":
                    row["status"] = "error"
        rows.append(
            {
                **failed,
                "timestamp_s": "999.0",
                "generation": repair_generation,
                "status": "deduplicated",
                "source_hash": successful_source,
            }
        )
        trial["num_unique_sources"] -= 1
        trial["num_successful_candidate_measurements"] -= 1
        trial["num_compile_failures"] += 1
        trial["num_source_deduplications"] += 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, rows)
            ledger = manifest.read_and_validate_ledger(
                path, trial, selected_config, selected_source
            )

        repaired = [
            row for row in ledger["rows"] if row["config_id"] == failed_config_id
        ]
        self.assertEqual(
            [row["status"] for row in repaired],
            ["started", "error", "deduplicated"],
        )

        mismatched_generation = copy.deepcopy(rows)
        repaired_terminal = next(
            row
            for row in mismatched_generation
            if row["config_id"] == failed_config_id and row["status"] == "deduplicated"
        )
        repaired_terminal["generation"] = str(int(repair_generation) + 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, mismatched_generation)
            with self.assertRaisesRegex(RuntimeError, "repair resolution generation"):
                manifest.read_and_validate_ledger(
                    path, trial, selected_config, selected_source
                )

        mismatched_source = copy.deepcopy(rows)
        mismatched_repairing_row = next(
            row
            for row in mismatched_source
            if row["source_hash"] == successful_source and row["status"] == "ok"
        )
        mismatched_repairing_row["source_hash"] = manifest.canonical_sha256(
            {"mismatched_repair_source": failed_config_id}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, mismatched_source)
            with self.assertRaisesRegex(RuntimeError, "prior successful source"):
                manifest.read_and_validate_ledger(
                    path, trial, selected_config, selected_source
                )

    def test_accepts_source_inferred_accuracy_rejection_without_execution(
        self,
    ) -> None:
        payload, rows, _autotune_rows, _metadata = fixture_payload(
            "dense", 32768, 7, 2026081507, 0
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        trial = provenance["trials"][0]
        failed = next(row for row in rows if row["status"] == "accuracy_error")
        alias = next(row for row in rows if row["status"] == "deduplicated")
        alias["status"] = "source_rejected"
        alias["source_hash"] = failed["source_hash"]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, rows)
            manifest.read_and_validate_ledger(
                path,
                trial,
                provenance["selected_config"],
                provenance["selected_source_sha256"],
            )

    def test_rejects_source_rejection_without_prior_accuracy_failure(self) -> None:
        payload, rows, _autotune_rows, _metadata = fixture_payload(
            "dense", 32768, 7, 2026081507, 0
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        alias = next(row for row in rows if row["status"] == "deduplicated")
        alias["status"] = "source_rejected"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, rows)
            with self.assertRaisesRegex(RuntimeError, "no prior accuracy failure"):
                manifest.read_and_validate_ledger(
                    path,
                    provenance["trials"][0],
                    provenance["selected_config"],
                    provenance["selected_source_sha256"],
                )

    def test_rejects_accuracy_count_mismatch(self) -> None:
        payload, rows, _autotune_rows, _metadata = fixture_payload(
            "dense", 32768, 7, 2026081507, 0
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        accuracy = next(row for row in rows if row["status"] == "accuracy_error")
        accuracy["status"] = "error"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, rows)
            with self.assertRaisesRegex(RuntimeError, "accuracy failure count"):
                manifest.read_and_validate_ledger(
                    path,
                    provenance["trials"][0],
                    provenance["selected_config"],
                    provenance["selected_source_sha256"],
                )

    def test_rejects_source_with_success_and_accuracy_failure(self) -> None:
        payload, rows, _autotune_rows, _metadata = fixture_payload(
            "dense", 32768, 7, 2026081507, 0
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        successful_source = next(
            row["source_hash"] for row in rows if row["status"] == "ok"
        )
        accuracy = next(row for row in rows if row["status"] == "accuracy_error")
        accuracy["source_hash"] = successful_source

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, rows)
            with self.assertRaisesRegex(RuntimeError, "definitive outcome"):
                manifest.read_and_validate_ledger(
                    path,
                    provenance["trials"][0],
                    provenance["selected_config"],
                    provenance["selected_source_sha256"],
                )

    def test_rejects_new_attempt_after_successful_source_outcome(self) -> None:
        payload, rows, _autotune_rows, _metadata = fixture_payload(
            "dense", 32768, 7, 2026081507, 0
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        successful_source = next(
            row["source_hash"] for row in rows if row["status"] == "ok"
        )
        failed_config_id = next(
            row["config_id"] for row in rows if row["status"] == "error"
        )
        for row in rows:
            if row["config_id"] == failed_config_id:
                row["source_hash"] = successful_source
        provenance["trials"][0]["num_unique_sources"] -= 1

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, rows)
            with self.assertRaisesRegex(
                RuntimeError, "attempted outcome after its definitive outcome"
            ):
                manifest.read_and_validate_ledger(
                    path,
                    provenance["trials"][0],
                    provenance["selected_config"],
                    provenance["selected_source_sha256"],
                )

    def test_rejects_duplicate_success_for_effective_source(self) -> None:
        payload, rows, _autotune_rows, _metadata = fixture_payload(
            "dense", 32768, 7, 2026081507, 0
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        selected_source = provenance["selected_source_sha256"]
        successful = [
            row
            for row in rows
            if row["status"] == "ok" and row["source_hash"] != selected_source
        ]
        successful[1]["source_hash"] = successful[0]["source_hash"]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, rows)
            with self.assertRaisesRegex(RuntimeError, "definitive outcome"):
                manifest.read_and_validate_ledger(
                    path,
                    provenance["trials"][0],
                    provenance["selected_config"],
                    provenance["selected_source_sha256"],
                )

    def test_rejects_deduplication_before_source_measurement(self) -> None:
        payload, rows, _autotune_rows, _metadata = fixture_payload(
            "dense", 32768, 7, 2026081507, 0
        )
        provenance = payload["helion_overrides"]["autotune_provenance"]
        alias_index = next(
            index for index, row in enumerate(rows) if row["status"] == "deduplicated"
        )
        rows.insert(0, rows.pop(alias_index))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotune.sources.csv"
            write_ledger(path, rows)
            with self.assertRaisesRegex(RuntimeError, "no prior successful source"):
                manifest.read_and_validate_ledger(
                    path,
                    provenance["trials"][0],
                    provenance["selected_config"],
                    provenance["selected_source_sha256"],
                )

    def test_rejects_selected_config_without_source_ledger_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            ledger_path = paths["dense_32768"].with_name("autotune.sources.csv")
            with ledger_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            selected_config_id = rows[0]["config_id"]
            replacement = "f" * 16
            for row in rows:
                if row["config_id"] == selected_config_id:
                    row["config_id"] = replacement
            write_ledger(ledger_path, rows)
            with self.assertRaisesRegex(RuntimeError, "selected config is not linked"):
                manifest.build_manifest(root)

    def test_accepts_selected_config_as_deduplicated_source_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            result_path = paths["dense_32768"]
            payload = json.loads(result_path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            selected_source = provenance["selected_source_sha256"]
            selected_config_id = manifest.canonical_sha256(
                provenance["selected_config"]
            )[:16]
            ledger_path = result_path.with_name("autotune.sources.csv")
            with ledger_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            alias = next(row for row in rows if row["status"] == "deduplicated")
            replacement_config_id = alias["config_id"]
            metadata_path = result_path.with_name("autotune.meta.jsonl")
            metadata = json.loads(metadata_path.read_text())
            for row in rows:
                if row["source_hash"] == selected_source:
                    row["config_id"] = replacement_config_id
            alias["config_id"] = selected_config_id
            alias["source_hash"] = selected_source
            phase = provenance["trials"][0]["search_phase_metrics"]

            def replace_phase_config_id(value: object) -> object:
                if isinstance(value, dict):
                    return {
                        key: replace_phase_config_id(item)
                        for key, item in value.items()
                    }
                if isinstance(value, list):
                    return [replace_phase_config_id(item) for item in value]
                return replacement_config_id if value == selected_config_id else value

            replaced_phase = replace_phase_config_id(phase)
            assert isinstance(replaced_phase, dict)
            provenance["trials"][0]["search_phase_metrics"] = replaced_phase
            phase = replaced_phase
            ordinary = phase["leaf_results"][0]
            for qualified in ordinary["qualified_results"]:
                if qualified["config_id"] == replacement_config_id:
                    qualified["selection_perf"] = 0.5

            def retain(
                result: dict[str, Any],
            ) -> list[tuple[str, dict[str, Any] | None]]:
                lanes = [
                    (lane["key"], lane["value"]) for lane in result["pipeline_lanes"]
                ]
                members = [
                    {
                        "config_id": qualified["config_id"],
                        "selection_perf": qualified["selection_perf"],
                        "pipeline_lanes": manifest.config_pipeline_lanes(
                            metadata["configs"][qualified["config_id"]], lanes
                        ),
                    }
                    for qualified in result["qualified_results"]
                ]
                return [
                    (member["config_id"], manifest.pipeline_lane_metric(lane))
                    for member, lane in manifest.lane_diverse_members(
                        members, lanes, limit=2
                    )
                ]

            ordinary_retained = retain(ordinary)
            ordinary["retained_config_ids"] = [
                config_id for config_id, _lane in ordinary_retained
            ]
            transfer = phase["compound_transfers"][0]["transfers"][0]
            compound_config_id = transfer["transferred_config_id"]
            phase["retained_families"][0]["score"] = 0.5
            phase["retained_families"][0]["starting_paths"] = [
                {
                    "family": ordinary["family"],
                    "compound_packet": ordinary["compound_packet"],
                    "config_id": ordinary["retained_config_ids"][0],
                    "unrestricted": False,
                    "pipeline_lane": None,
                },
                {
                    "family": ordinary["family"],
                    "compound_packet": ordinary["compound_packet"],
                    "config_id": ordinary["retained_config_ids"][1],
                    "unrestricted": False,
                    "pipeline_lane": ordinary_retained[1][1],
                },
                {
                    "family": phase["compound_transfers"][0]["family"],
                    "compound_packet": phase["compound_transfers"][0][
                        "compound_packet"
                    ],
                    "config_id": compound_config_id,
                    "unrestricted": False,
                    "pipeline_lane": None,
                },
                {
                    "family": ordinary["family"],
                    "compound_packet": ordinary["compound_packet"],
                    "config_id": ordinary["retained_config_ids"][0],
                    "unrestricted": True,
                    "pipeline_lane": None,
                },
            ]
            phase["retained_path_count"] = 4
            attempts = {
                row["config_id"]: {
                    "generation": int(row["generation"]),
                    "status": row["status"],
                    "source_hash": row["source_hash"],
                    "perf_ms": (
                        10.0 if row["status"] in {"ok", "deduplicated"} else None
                    ),
                }
                for row in rows
                if row["status"] != "started"
            }
            add_phase_config_identity(
                provenance,
                phase,
                metadata["configs"],
                attempts=attempts,
                default_perf=10.0,
            )
            result_path.write_text(json.dumps(payload) + "\n")
            write_ledger(ledger_path, rows)
            autotune_path = result_path.with_name("autotune.csv")
            with autotune_path.open(newline="") as handle:
                autotune_rows = list(csv.DictReader(handle))
            for autotune_row, source_row in zip(autotune_rows, rows, strict=True):
                for field in manifest.AUTOTUNE_JOIN_FIELDS:
                    autotune_row[field] = source_row[field]
                config = json.loads(
                    result_path.with_name("autotune.meta.jsonl").read_text()
                )["configs"].get(autotune_row["config_id"])
                if config is not None:
                    autotune_row["config"] = (
                        "Config("
                        + ", ".join(
                            f"{key}={value!r}" for key, value in sorted(config.items())
                        )
                        + ")"
                    )
            metadata_path.write_text(json.dumps(metadata) + "\n")
            write_autotune_csv(autotune_path, autotune_rows)

            manifest_rows = list(
                csv.DictReader(io.StringIO(manifest.build_manifest(root)))
            )

        self.assertEqual(
            manifest_rows[0]["selected_source_ledger_config_id"], selected_config_id
        )

    def test_rejects_deduplicated_row_without_successful_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            ledger_path = paths["dense_32768"].with_name("autotune.sources.csv")
            with ledger_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            failed_source = next(
                row["source_hash"] for row in rows if row["status"] == "error"
            )
            deduplicated = next(row for row in rows if row["status"] == "deduplicated")
            deduplicated["source_hash"] = failed_source
            write_ledger(ledger_path, rows)
            with self.assertRaisesRegex(RuntimeError, "has no prior successful source"):
                manifest.build_manifest(root)

    def test_rejects_malformed_source_ledger_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            ledger_path = paths["dense_32768"].with_name("autotune.sources.csv")
            lines = ledger_path.read_text().splitlines()
            lines[1] += ",unexpected"
            ledger_path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(RuntimeError, "malformed source ledger row"):
                manifest.build_manifest(root)

    def test_source_ledger_allows_unstarted_precompile_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = self.write_fixture(root)["dense_32768"]
            payload = json.loads(result_path.read_text())
            provenance = payload["helion_overrides"]["autotune_provenance"]
            trial = copy.deepcopy(provenance["trials"][0])
            trial["num_configs_tested"] -= 1
            ledger_path = result_path.with_name("autotune.sources.csv")
            with ledger_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            failed = next(row for row in rows if row["status"] == "error")
            rows = [
                row
                for row in rows
                if not (
                    row["config_id"] == failed["config_id"]
                    and row["status"] == "started"
                )
            ]
            write_ledger(ledger_path, rows)

            validated = manifest.read_and_validate_ledger(
                ledger_path,
                trial,
                provenance["selected_config"],
                provenance["selected_source_sha256"],
            )

            self.assertEqual(
                sum(row["status"] == "started" for row in validated["rows"]),
                trial["num_configs_tested"],
            )

    def test_rejects_filtered_source_from_unfiltered_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            ledger_path = paths["dense_32768"].with_name("autotune.sources.csv")
            with ledger_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            failed = next(row for row in rows if row["status"] == "error")
            failed["status"] = "filtered"
            write_ledger(ledger_path, rows)
            with self.assertRaisesRegex(RuntimeError, "unexpected status 'filtered'"):
                manifest.build_manifest(root)

    def test_rejects_wrong_correctness_gpu_and_version(self) -> None:
        mutations = (
            (
                "correctness",
                lambda payload: payload["helion_overrides"][
                    "autotune_provenance"
                ].__setitem__("final_correctness_launches", 63),
                "final_correctness_launches",
            ),
            (
                "gpu",
                lambda payload: payload.__setitem__("physical_gpu", "6"),
                "GPU",
            ),
            (
                "version",
                lambda payload: payload.__setitem__(
                    "version", "Helion 1.4.0.dev157+gc3e36b65d; CuTe 4.7.0.dirty"
                ),
                "dirty or invalid version",
            ),
        )
        for label, mutate, expected_error in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.write_fixture(root)
                path = paths["dense_32768"]
                payload = json.loads(path.read_text())
                mutate(payload)
                path.write_text(json.dumps(payload) + "\n")
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    manifest.build_manifest(root)

    def test_version_accepts_any_unambiguous_git_abbreviation(self) -> None:
        commit = manifest.EXPECTED_MEASURED_COMMIT
        self.assertTrue(
            manifest.helion_cute_version_matches_commit(
                f"Helion 1.4.0.dev0+g{commit[:9]}; CuTe 4.7.0", commit
            )
        )
        self.assertFalse(
            manifest.helion_cute_version_matches_commit(
                "Helion 1.4.0.dev0+gdeadbee; CuTe 4.7.0", commit
            )
        )
        self.assertFalse(
            manifest.helion_cute_version_matches_commit(
                f"Helion 1.4.0.dev0+g{commit[:9]}; CuTe 4.6.1", commit
            )
        )

    def test_v22_starting_path_capacity_is_derived_from_live_catalog(self) -> None:
        catalog = [
            {
                "family": family,
                "compound_packet": packet,
                "softmax_disc": softmax_disc,
            }
            for family, packet, softmax_disc in (
                ("a", None, False),
                ("a", None, True),
                ("a", "deg1_16x8", False),
                ("b", None, False),
                ("c", None, False),
                ("d", None, False),
            )
        ]
        self.assertEqual(manifest.expected_retained_family_limit(catalog, None), 4)
        self.assertEqual(manifest.expected_retained_family_limit(catalog, 10), 4)
        self.assertEqual(manifest.expected_retained_family_limit(catalog, 2), 2)
        self.assertEqual(
            manifest.expected_starting_path_limit(
                catalog, retained_per_leaf=2, retained_family_limit=4
            ),
            14,
        )
        expanded = [
            *catalog,
            {
                "family": "b",
                "compound_packet": None,
                "softmax_disc": True,
            },
            {
                "family": "c",
                "compound_packet": "deg2_16x6",
                "softmax_disc": False,
            },
            {
                "family": "c",
                "compound_packet": None,
                "softmax_disc": True,
            },
            {
                "family": "d",
                "compound_packet": None,
                "softmax_disc": True,
            },
            {
                "family": "d",
                "compound_packet": "hybrid_deg1_16x8",
                "softmax_disc": False,
            },
        ]
        self.assertEqual(
            manifest.expected_starting_path_limit(
                expanded, retained_per_leaf=2, retained_family_limit=4
            ),
            16,
        )

    def test_v22_starting_path_capacity_does_not_import_ambient_helion(self) -> None:
        original_import = __import__

        def reject_helion_import(
            name: str,
            globals_: dict[str, object] | None = None,
            locals_: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "helion" or name.startswith("helion."):
                raise AssertionError(f"unexpected ambient import: {name}")
            return original_import(name, globals_, locals_, fromlist, level)

        catalog = [
            {
                "family": "fa4_2cta",
                "compound_packet": None,
                "softmax_disc": False,
            }
        ]
        with mock.patch("builtins.__import__", side_effect=reject_helion_import):
            self.assertEqual(
                manifest.expected_starting_path_limit(
                    catalog, retained_per_leaf=2, retained_family_limit=4
                ),
                manifest.EXPECTED_FULL_FLASH_STARTING_PATHS,
            )

    def test_rejects_self_consistent_foreign_campaign_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.write_fixture(root)
            path = paths["dense_32768"]
            payload = json.loads(path.read_text())
            foreign_commit = "d" * 40
            payload["version"] = "Helion 1.4.0.dev0+gdddddddd; CuTe 4.7.0"
            payload["helion_overrides"]["autotune_provenance"][
                "helion_checkout_git_commit"
            ] = foreign_commit
            path.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(RuntimeError, "measured campaign commit"):
                manifest.build_manifest(root)

    def test_v22_requires_complete_schedule_anchor_evidence(self) -> None:
        provenance, trial, _attempts, _metadata = self.qualification_fixture()
        for field, value in (
            ("schedule_anchor_results", []),
            ("schedule_anchor_complete", False),
            ("schedule_anchor_design_source", "fixture"),
        ):
            mutated = copy.deepcopy(trial)
            mutated["search_phase_metrics"][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(RuntimeError, "schedule-anchor"),
            ):
                manifest.validate_structural_qualification_phase(
                    Path("fixture.json"), provenance, mutated
                )

    def test_v22_requires_complete_owned_compound_catalog(self) -> None:
        provenance, trial, _attempts, _metadata = self.qualification_fixture()
        phase = trial["search_phase_metrics"]
        for field, value in (
            ("compound_catalog_complete", False),
            ("compound_catalog_errors", [{"error": "orphan"}]),
        ):
            mutated = copy.deepcopy(trial)
            mutated["search_phase_metrics"][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(RuntimeError, "compound catalog"),
            ):
                manifest.validate_structural_qualification_phase(
                    Path("fixture.json"), provenance, mutated
                )
        orphaned = copy.deepcopy(provenance)
        orphaned["flash_structural_leaf_catalog"] = [
            leaf
            for leaf in orphaned["flash_structural_leaf_catalog"]
            if leaf["compound_packet"] is not None
        ]
        with self.assertRaisesRegex(RuntimeError, "ordinary family/protocol owner"):
            manifest.validate_structural_qualification_phase(
                Path("fixture.json"), orphaned, trial
            )
        self.assertTrue(phase["compound_catalog_complete"])

    def test_v22_repairs_empty_clc_conditional_generation(self) -> None:
        parent_id = "1" * 16
        repair_id = "2" * 16
        config = {
            "cute_flash_pipeline_family": "fa4_clc",
            "cute_flash_clc_heads_per_batch": 2,
        }
        source_hash = "a" * 64
        parent_state = {
            "attempt_perf": 1.0,
            "selection_perf": 1.0,
            "status": "ok",
            "source_hash": source_hash,
        }
        decision = {
            "kind": "conditional_failure_repair",
            "value": 2,
            "repair_index": 0,
            "candidate_results": [
                {
                    "config_id": parent_id,
                    **parent_state,
                    "measurement_pass_index": 0,
                }
            ],
            "selected_config_id": parent_id,
            "generated_config_ids": [repair_id],
            "neighbor_generation_limit": 200,
        }
        kwargs = {
            "repair_ids_by_value": {"2": [repair_id]},
            "primary_ids_by_value": {"2": []},
            "values": [2],
            "expected_kind": "conditional_failure_repair",
            "expected_leaf": manifest.structural_leaf(config),
            "candidate_limit": 4,
            "phase_configs": {parent_id: config, repair_id: config},
            "measurement_states": [
                {parent_id: parent_state},
                {
                    parent_id: parent_state,
                    repair_id: {**parent_state, "source_hash": "b" * 64},
                },
            ],
            "label": "invalid immutable CLC conditional repair",
            "missing_generation_parent_ids_by_value": {"2": parent_id},
        }
        manifest.validate_failure_repair_decisions(
            Path("fixture.json"), [decision], **kwargs
        )
        decision["generated_config_ids"] = []
        with self.assertRaisesRegex(RuntimeError, "CLC conditional repair"):
            manifest.validate_failure_repair_decisions(
                Path("fixture.json"), [decision], **kwargs
            )


if __name__ == "__main__":
    unittest.main()
