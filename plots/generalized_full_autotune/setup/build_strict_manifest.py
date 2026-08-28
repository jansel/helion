from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import io
import json
import math
from operator import itemgetter
from pathlib import Path
import re
import statistics
from typing import Any

EXPECTED_INPUT_SEED = 2026081500
EXPECTED_POWER_CAP_W = 750.0
EXPECTED_FINAL_CORRECTNESS_LAUNCHES = 64
EXPECTED_MEASURED_COMMIT = "c3e36b65d69681c23e053042b0bc21e2331bad17"
EXPECTED_CUTE_VERSION = "4.7.0"
# Immutable v22 full-effort policy floors. Catalog-derived structural capacity
# below may exceed these values, but artifact validation must not depend on the
# Helion checkout imported by the publishing process.
EXPECTED_FULL_LFBO_COPIES = 5
EXPECTED_FULL_FLASH_STARTING_PATHS = 14
EXPECTED_FULL_FLASH_RETAINED_FAMILIES = 4
EXPECTED_FAMILY_PROBE_GENERATIONS = 1
EXPECTED_FAMILY_PROBE_CANDIDATES_PER_PATH = 20
EXPECTED_NEIGHBOR_GENERATION_LIMIT = 200
EXPECTED_LANE_POLICY_VERSION = 14
EXPECTED_TERMINAL_REFINEMENT_SCHEMA_VERSION = 2
EXPECTED_TERMINAL_REFINEMENT_POLICY_VERSION = 2
EXPECTED_TERMINAL_REFINEMENT_ROUNDS = 2
EXPECTED_TERMINAL_REFINEMENT_BEAM_WIDTH = 4
EXPECTED_TERMINAL_REFINEMENT_RADIUS = 2
EXPECTED_TERMINAL_REFINEMENT_MINIMUM_IMPROVEMENT = 0.001
EXPECTED_TERMINAL_REFINEMENT_ROUND_TARGET_MS = 200.0
EXPECTED_TERMINAL_REFINEMENT_CONFIRMATION_TARGET_MS = 5000.0
EXPECTED_TERMINAL_REFINEMENT_REPEAT_MAX = 20_000
EXPECTED_TERMINAL_REFINEMENT_MAX_SWEEPS = 64
EXPECTED_TERMINAL_REFINEMENT_COORDINATE_POLICY = (
    "same_leaf_full_surface_normalized_coordinate_v2"
)
EXPECTED_TERMINAL_REFINEMENT_MEASUREMENT_POLICY = "mirrored_rotating_batched_wall_v2"
EXPECTED_TERMINAL_SURFACE_SCHEMA_VERSION = 1
EXPECTED_PRETERMINAL_REGISTRY_HASH_POLICY = "sorted_compact_json_sha256_v1"
EXPECTED_CONDITIONAL_CANDIDATES_PER_PIPELINE_LANE = 1
EXPECTED_QUALIFICATION_FAILURE_RETRIES = 1
EXPECTED_COMPILER_SEED_POLICY_SCHEMA_VERSION = 1
EXPECTED_COMPILER_SEED_POLICY_KIND = "canonical_cute_flash"
EXPECTED_COMPILER_SEED_HEURISTICS = ("cute_flash_attention",)
EXPECTED_COMPILER_SEED_TIMEOUT_RETRY_REPETITIONS = 3
FLASH_PIPELINE_FAMILY_KEY = "cute_flash_pipeline_family"
FLASH_EXP2_PACKET_KEY = "cute_flash_exp2_packet"
FLASH_SOFTMAX_DISC_KEY = "cute_flash_softmax_disc"
FLASH_CLC_HEADS_PER_BATCH_KEY = "cute_flash_clc_heads_per_batch"
FLASH_PIPELINE_QUALIFICATION_KEYS = (
    "cute_flash_kv_stage",
    "cute_flash_s_stage",
)


def attention_trial_context(
    provenance: dict[str, Any], trial: dict[str, Any]
) -> tuple[list[tuple[int, ...]], list[str], bool]:
    shapes = ast.literal_eval(trial["input_shapes"])
    dtypes = ast.literal_eval(trial["dtypes"])
    require(
        isinstance(shapes, list)
        and shapes
        and isinstance(shapes[0], tuple)
        and shapes[0]
        and isinstance(dtypes, list)
        and dtypes
        and isinstance(dtypes[0], str),
        "invalid attention trial shape/dtype identity",
    )
    return (
        shapes,
        dtypes,
        "causal_attention" in str(provenance.get("autotune_baseline_fn", "")),
    )


def validate_flash_normalization_context(
    path: str, provenance: dict[str, Any], trial: dict[str, Any]
) -> dict[str, Any]:
    """Validate the recorded ConfigSpec identity without replaying normalization."""
    context = provenance.get("flash_normalization_context")
    require(isinstance(context, dict), f"{path}: flash normalization context")
    check_equal(
        provenance.get("flash_normalization_context_sha256"),
        canonical_sha256(context),
        f"{path}: flash normalization context digest",
    )
    shapes, dtypes, causal = attention_trial_context(provenance, trial)
    shape = shapes[0]
    dtype = dtypes[0]
    require(
        len(shape) >= 2
        and all(type(dimension) is int and dimension > 0 for dimension in shape)
        and all(item == shape for item in shapes)
        and all(isinstance(item, str) and item == dtype for item in dtypes),
        f"{path}: inconsistent attention trial inputs",
    )
    expected = {
        "schema_version": 1,
        "backend": "cute",
        "dtype": dtype,
        "head_dim": shape[-1],
        "num_kv": (shape[-2] + 127) // 128,
        "num_bh": math.prod(shape[:-2]),
        "is_causal": causal,
        "standard_dense_output": not causal,
        "standard_causal_output": causal,
        "default_config_sha256": provenance.get("flash_fragment_default_sha256"),
    }
    for key, value in expected.items():
        check_equal(context.get(key), value, f"{path}: normalization context {key}")
    require(
        isinstance(context.get("config_spec_structural_fingerprint_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", context["config_spec_structural_fingerprint_sha256"]
        )
        is not None,
        f"{path}: ConfigSpec structural fingerprint",
    )
    require(
        isinstance(context.get("flat_key_layout"), list) and context["flat_key_layout"],
        f"{path}: normalization flat-key layout",
    )
    require(
        isinstance(context.get("block_size_targets"), list)
        and context["block_size_targets"],
        f"{path}: normalization block-size targets",
    )
    return context


# Keep this standalone verifier synchronized with cute_flash.py.
FLASH_INTERACTION_KEY_GROUPS = (
    (
        "cute_flash_epi_tma",
        "cute_flash_epi_stg",
        "cute_flash_epi_stg_store",
        "cute_flash_epi_stg_gmem",
    ),
)
# Keep this standalone verifier synchronized with cute_flash.py. These packet
# names own additional schedule fields; ordinary packets such as 1x1 do not.
COMPOUND_EXP2_PACKETS = frozenset(
    {
        "deg2_16x6",
        "hybrid_deg1_16x8",
        "deg1_16x8",
        "deg1_8x2_corr10",
        "causal_hd128_resident3_013_prefetch2_deg2_early_acquire",
    }
)
ORDINARY_CLC_FAMILIES = frozenset(
    {
        "fa4_clc",
        "fa4_clc_tma_4d",
        "fa4_clc_local_tma",
        "fa4_clc_local_tma_4d",
    }
)

CASES = (
    ("dense", 32768, 7, 2026081501),
    ("dense", 65536, 7, 2026081502),
    ("dense", 131072, 7, 2026081503),
    ("dense", 262144, 7, 2026081504),
    ("causal", 65536, 6, 2026081511),
    ("causal", 131072, 6, 2026081512),
    ("causal", 262144, 6, 2026081513),
    ("causal", 524288, 6, 2026081514),
)
EXPECTED_CASES = {
    (variant, seq_len): {
        "physical_gpu": physical_gpu,
        "tuner_seed": tuner_seed,
    }
    for variant, seq_len, physical_gpu, tuner_seed in CASES
}

LEDGER_FIELDS = (
    "run_id",
    "timestamp_s",
    "config_id",
    "generation",
    "status",
    "source_hash",
)
AUTOTUNE_CSV_FIELDS = (
    "run_id",
    "timestamp_s",
    "config_id",
    "generation",
    "status",
    "perf_ms",
    "compile_time_s",
    "config",
)
AUTOTUNE_JOIN_FIELDS = LEDGER_FIELDS[:-1]
CODEGEN_SETTINGS = (
    "allow_warp_specialize",
    "backend",
    "debug_dtype_asserts",
    "dot_precision",
    "fast_math",
    "index_dtype",
    "pallas_collective_id",
    "pallas_interpret",
    "pallas_topk_recall_target",
    "persistent_reserved_sms",
    "static_shapes",
    "triton_do_not_specialize",
)
LEDGER_TERMINAL_STATUSES = frozenset(
    {"ok", "error", "timeout", "peer_compilation_fail", "accuracy_error"}
)
LEDGER_REPAIRABLE_FAILURE_STATUSES = frozenset(
    {"error", "timeout", "peer_compilation_fail"}
)
LEDGER_ALIAS_STATUSES = frozenset({"deduplicated", "source_rejected"})
LEDGER_STATUSES = LEDGER_TERMINAL_STATUSES | LEDGER_ALIAS_STATUSES | {"started"}

MANIFEST_FIELDS = (
    "case",
    "variant",
    "seq_len",
    "causal",
    "dtype",
    "z",
    "h",
    "head_dim",
    "version",
    "gpu",
    "physical_gpu",
    "power_cap_w",
    "input_seed",
    "tuner_seed",
    "benchmark_timer",
    "result_path",
    "result_sha256",
    "autotune_csv_path",
    "autotune_csv_sha256",
    "autotune_metadata_path",
    "autotune_metadata_sha256",
    "autotune_run_id",
    "autotune_metadata_config_count",
    "compiler_seed_policy_json",
    "source_ledger_path",
    "source_ledger_sha256",
    "source_ledger_run_id",
    "median_ms",
    "median_tflops",
    "search_algorithm",
    "num_configs_tested",
    "num_successful_candidate_measurements",
    "num_unique_sources",
    "num_source_deduplications",
    "num_compile_failures",
    "num_worker_failures",
    "num_isolated_rebenchmark_timeouts",
    "num_accuracy_failures",
    "num_generations",
    "autotune_time_seconds",
    "coverage_design_count",
    "coverage_design_sha256",
    "coverage_design_prefix_count",
    "coverage_design_prefix_gen0_attempted_count",
    "coverage_design_prefix_gen0_successful_count",
    "coverage_active_value_count",
    "coverage_active_value_successful_witness_count",
    "coverage_active_value_successful_witness_counts_json",
    "coverage_qualification_value_count",
    "coverage_qualification_successful_witness_count",
    "coverage_qualification_successful_witness_counts_json",
    "coverage_interaction_count",
    "coverage_interaction_successful_witness_count",
    "coverage_interaction_successful_witness_counts_json",
    "structural_qualification_leaf_count",
    "structural_qualification_leaves_with_candidates",
    "structural_qualification_candidate_count",
    "structural_qualification_successful_candidate_count",
    "structural_qualification_retained_family_count",
    "structural_qualification_retained_path_count",
    "structural_qualification_leaf_results_json",
    "terminal_refinement_policy_sha256",
    "terminal_coordinate_surface_sha256",
    "terminal_refinement_json",
    "terminal_refinement_sha256",
    "winner_is_coverage_design_member",
    "winner_to_coverage_field_distance",
    "selected_config_json",
    "selected_config_sha256",
    "selected_source_sha256",
    "selected_source_ledger_config_id",
    "selected_source_ledger_generation",
    "final_correctness_launches",
    "final_repeatability_passed",
)


def require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def helion_cute_version_matches_commit(version: object, commit: str) -> bool:
    if not isinstance(version, str) or ".dirty" in version:
        return False
    match = re.fullmatch(r"Helion [^;]*\+g([0-9a-f]{7,40}); CuTe ([^;\s]+)", version)
    return (
        match is not None
        and commit.startswith(match.group(1))
        and match.group(2) == EXPECTED_CUTE_VERSION
    )


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def exact_effective_search_space_ids(
    path: Path, provenance: dict[str, Any]
) -> list[str] | None:
    size = provenance.get("flash_exact_effective_search_space_size")
    config_ids = provenance.get("flash_exact_effective_search_space_config_ids")
    digest = provenance.get("flash_exact_effective_search_space_sha256")
    if size is None and config_ids is None and digest is None:
        return None
    require(
        isinstance(size, int)
        and not isinstance(size, bool)
        and 0 < size <= provenance.get("autotune_initial_population_size", 0)
        and isinstance(config_ids, list)
        and len(config_ids) == size
        and len(set(config_ids)) == size
        and all(
            isinstance(config_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
            for config_id in config_ids
        )
        and digest
        == hashlib.sha256(
            json.dumps(config_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        f"{path}: invalid exact effective search-space provenance",
    )
    return config_ids


def structural_qualification_values(
    active_values: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        active
        for active in active_values
        if active["key"] == FLASH_PIPELINE_FAMILY_KEY
        or (
            active["key"] == FLASH_EXP2_PACKET_KEY
            and active["value"] in COMPOUND_EXP2_PACKETS
        )
    ]


def structural_leaf(config: dict[str, Any]) -> dict[str, Any] | None:
    family = config.get(FLASH_PIPELINE_FAMILY_KEY)
    if not isinstance(family, str):
        return None
    packet = config.get(FLASH_EXP2_PACKET_KEY)
    return {
        "family": family,
        "compound_packet": packet if packet in COMPOUND_EXP2_PACKETS else None,
        "softmax_disc": bool(config.get(FLASH_SOFTMAX_DISC_KEY)),
    }


def expected_starting_path_limit(
    leaf_catalog: list[dict[str, Any]],
    *,
    retained_per_leaf: int,
    retained_family_limit: int,
) -> int:
    """Rebuild v22 path capacity from immutable floors and the leaf catalog."""
    ordinary_widths: dict[str, int] = {}
    compound_count = 0
    for leaf in leaf_catalog:
        if leaf["compound_packet"] is None:
            family = leaf["family"]
            ordinary_widths[family] = ordinary_widths.get(family, 0) + 1
        else:
            compound_count += 1
    promoted_count = min(retained_family_limit, len(ordinary_widths))
    promoted_protocol_count = sum(
        sorted(ordinary_widths.values(), reverse=True)[:promoted_count]
    )
    return max(
        EXPECTED_FULL_LFBO_COPIES,
        EXPECTED_FULL_FLASH_STARTING_PATHS,
        1
        + promoted_protocol_count
        + (promoted_count if retained_per_leaf > 1 else 0)
        + compound_count,
    )


def expected_retained_family_limit(
    leaf_catalog: list[dict[str, Any]], retained_family_cap: int | None
) -> int:
    """Resolve the configured cap against ordinary families in the live catalog."""
    live_families = {
        leaf["family"] for leaf in leaf_catalog if leaf["compound_packet"] is None
    }
    return (
        len(live_families)
        if retained_family_cap is None
        else min(retained_family_cap, len(live_families))
    )


def expected_family_probe_path_limit(
    leaf_catalog: list[dict[str, Any]],
    retained_family_cap: int | None,
    family_probe_generations: int,
) -> int:
    """Size the measured pre-promotion probe from the live catalog."""
    if family_probe_generations <= 0:
        return 0
    live_families = {
        leaf["family"] for leaf in leaf_catalog if leaf["compound_packet"] is None
    }
    if retained_family_cap is None or len(live_families) <= retained_family_cap:
        return 0
    compound_count = sum(leaf["compound_packet"] is not None for leaf in leaf_catalog)
    return len(live_families) + compound_count + 1


def flash_pipeline_lane_catalog(
    path: Path, provenance: dict[str, Any]
) -> dict[str, list[tuple[str, int]]]:
    leaf_catalog = provenance.get("flash_structural_leaf_catalog")
    catalog = provenance.get("flash_pipeline_lane_catalog")
    require(
        isinstance(leaf_catalog, list)
        and isinstance(catalog, list)
        and len(catalog) == len(leaf_catalog),
        f"{path}: invalid flash pipeline lane catalog",
    )
    result: dict[str, list[tuple[str, int]]] = {}
    for expected_leaf, entry in zip(leaf_catalog, catalog, strict=True):
        require(
            isinstance(expected_leaf, dict)
            and isinstance(entry, dict)
            and set(entry)
            == {"family", "compound_packet", "softmax_disc", "pipeline_lanes"}
            and {
                "family": entry["family"],
                "compound_packet": entry["compound_packet"],
                "softmax_disc": entry["softmax_disc"],
            }
            == expected_leaf
            and isinstance(entry["pipeline_lanes"], list),
            f"{path}: invalid flash pipeline lane catalog entry",
        )
        lanes: list[tuple[str, int]] = []
        previous_key_index = -1
        lane_counts = dict.fromkeys(FLASH_PIPELINE_QUALIFICATION_KEYS, 0)
        for metric in entry["pipeline_lanes"]:
            require(
                isinstance(metric, dict)
                and set(metric) == {"key", "value"}
                and metric["key"] in FLASH_PIPELINE_QUALIFICATION_KEYS
                and isinstance(metric["value"], int)
                and not isinstance(metric["value"], bool)
                and metric["value"] > 0,
                f"{path}: invalid flash pipeline lane metric",
            )
            lane = (metric["key"], metric["value"])
            key_index = FLASH_PIPELINE_QUALIFICATION_KEYS.index(metric["key"])
            require(
                lane not in lanes and key_index >= previous_key_index,
                f"{path}: duplicate or out-of-order flash pipeline lane",
            )
            previous_key_index = key_index
            lanes.append(lane)
            lane_counts[metric["key"]] += 1
        require(
            all(count != 1 for count in lane_counts.values()),
            f"{path}: singleton flash pipeline lane",
        )
        leaf_key = canonical_json(expected_leaf)
        require(leaf_key not in result, f"{path}: duplicate pipeline lane leaf")
        result[leaf_key] = lanes
    return result


def flash_clc_lane_catalog(
    path: Path, provenance: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate the immutable CLC lane catalog recorded by the live producer."""
    leaf_catalog = provenance.get("flash_structural_leaf_catalog")
    catalog = provenance.get("flash_clc_lane_catalog")
    require(
        isinstance(leaf_catalog, list) and isinstance(catalog, list),
        f"{path}: invalid flash CLC lane catalog",
    )
    check_equal(
        provenance.get("flash_clc_lane_catalog_sha256"),
        canonical_sha256(catalog),
        f"{path}: flash CLC lane catalog digest",
    )
    expected_leaves = [
        leaf
        for leaf in leaf_catalog
        if isinstance(leaf, dict)
        and leaf.get("compound_packet") is None
        and leaf.get("family") in ORDINARY_CLC_FAMILIES
    ]
    check_equal(
        [
            {
                "family": entry.get("family"),
                "compound_packet": entry.get("compound_packet"),
                "softmax_disc": entry.get("softmax_disc"),
            }
            for entry in catalog
            if isinstance(entry, dict)
        ],
        expected_leaves,
        f"{path}: flash CLC lane catalog order",
    )
    validated: list[dict[str, Any]] = []
    for expected_leaf, entry in zip(expected_leaves, catalog, strict=True):
        require(
            isinstance(entry, dict)
            and set(entry)
            == {
                "family",
                "compound_packet",
                "softmax_disc",
                "legal_values",
                "search_values",
                "anchor_values",
                "refinement_values",
                "planned_values",
                "witness_config_ids",
            }
            and {
                "family": entry["family"],
                "compound_packet": entry["compound_packet"],
                "softmax_disc": entry["softmax_disc"],
            }
            == expected_leaf,
            f"{path}: invalid flash CLC lane catalog entry",
        )
        legal = _positive_int_list(
            entry["legal_values"], f"{path}: CLC catalog legal values"
        )
        search = _positive_int_list(
            entry["search_values"], f"{path}: CLC catalog search values"
        )
        anchors = _positive_int_list(
            entry["anchor_values"], f"{path}: CLC catalog anchor values"
        )
        refinements = _positive_int_list(
            entry["refinement_values"], f"{path}: CLC catalog refinement values"
        )
        planned = _positive_int_list(
            entry["planned_values"], f"{path}: CLC catalog planned values"
        )
        witnesses = entry["witness_config_ids"]
        require(
            search == legal
            and set(anchors) <= set(search)
            and set(refinements) <= set(search)
            and not (set(anchors) & set(refinements))
            and planned == [*anchors, *refinements]
            and set(planned) == set(legal)
            and isinstance(witnesses, dict)
            and list(witnesses) == [str(value) for value in planned]
            and all(
                isinstance(config_id, str)
                and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
                for config_id in witnesses.values()
            ),
            f"{path}: invalid flash CLC lane catalog values",
        )
        validated.append(entry)
    return validated


def pipeline_lane_metric(
    lane: tuple[str, int] | None,
) -> dict[str, Any] | None:
    if lane is None:
        return None
    return {"key": lane[0], "value": lane[1]}


def config_pipeline_lanes(
    config: dict[str, Any], lanes: list[tuple[str, int]]
) -> frozenset[tuple[str, int]]:
    return frozenset(lane for lane in lanes if config.get(lane[0]) == lane[1])


def lane_diverse_members(
    members: list[dict[str, Any]],
    lanes: list[tuple[str, int]],
    *,
    limit: int,
) -> list[tuple[dict[str, Any], tuple[str, int] | None]]:
    remaining = sorted(members, key=itemgetter("selection_perf", "config_id"))
    if limit <= 0 or not remaining:
        return []
    selected = [(remaining.pop(0), None)]
    covered = set(selected[0][0]["pipeline_lanes"]) & set(lanes)
    while remaining and len(selected) < limit:

        def rank(member: dict[str, Any]) -> tuple[Any, ...]:
            newly_covered = (set(member["pipeline_lanes"]) & set(lanes)) - covered
            return (
                *(
                    -sum(lane[0] == key for lane in newly_covered)
                    for key in FLASH_PIPELINE_QUALIFICATION_KEYS
                ),
                member["selection_perf"],
                member["config_id"],
            )

        member = min(remaining, key=rank)
        remaining.remove(member)
        newly_covered = [
            lane
            for lane in lanes
            if lane not in covered and lane in member["pipeline_lanes"]
        ]
        selected.append((member, newly_covered[0] if newly_covered else None))
        covered.update(set(member["pipeline_lanes"]) & set(lanes))
    return selected


def clc_combination_coverage(result: dict[str, Any]) -> dict[str, Any]:
    """Replay v22's CLC matrix identity and successful axis coverage."""
    depth_ids = result["combination_depth_config_ids"]
    divisors = result["combination_divisor_values"]
    cells = result["combination_cells"]
    projected_ids = [
        cell["projected_config_id"]
        for cell in cells
        if isinstance(cell["projected_config_id"], str)
    ]
    successful_depth_ids = {
        cell["depth_config_id"]
        for cell in cells
        if cell["status"] in {"ok", "deduplicated"}
    }
    successful_divisors = {
        cell["divisor_value"]
        for cell in cells
        if cell["status"] in {"ok", "deduplicated"}
    }
    covered_depth_ids = [
        config_id for config_id in depth_ids if config_id in successful_depth_ids
    ]
    covered_divisors = [value for value in divisors if value in successful_divisors]
    projection_complete = bool(
        len(cells) == len(depth_ids) * len(divisors)
        and len({(cell["depth_config_id"], cell["divisor_value"]) for cell in cells})
        == len(cells)
    )
    return {
        "combination_candidate_ids": list(dict.fromkeys(projected_ids)),
        "successful_combination_depth_config_ids": covered_depth_ids,
        "successful_combination_divisor_values": covered_divisors,
        "combination_projection_complete": projection_complete,
        "combination_row_coverage_complete": covered_depth_ids == depth_ids,
        "combination_column_coverage_complete": covered_divisors == divisors,
    }


def reconcile_clc_depth_selection(
    path: Path,
    result: dict[str, Any],
    *,
    leaf: dict[str, Any],
    pre_combination_ids: set[str],
    successful_ids: set[str],
    successful_results_by_id: dict[str, dict[str, Any]],
    metadata_configs: dict[str, dict[str, Any]],
    pipeline_lanes: list[tuple[str, int]],
    retained_limit: int,
) -> None:
    """Replay a v22 CLC depth decision at its pre-combination boundary."""
    depth_selection = result.get("depth_selection")
    if depth_selection is None:
        return
    require(
        isinstance(depth_selection, dict)
        and set(depth_selection) == {"candidate_results", "selected_representatives"},
        f"{path}: invalid immutable CLC depth decision for {result['family']}",
    )
    expected_ids = {
        config_id
        for config_id in successful_ids & pre_combination_ids
        if structural_leaf(metadata_configs[config_id]) == leaf
    }
    if not result["combination_required"]:
        expected_ids.clear()
    check_equal(
        {snapshot["config_id"] for snapshot in depth_selection["candidate_results"]},
        expected_ids,
        f"{path}: immutable CLC depth candidates for {result['family']}",
    )
    members = [
        {
            "config_id": snapshot["config_id"],
            "selection_perf": snapshot["selection_perf"],
            "pipeline_lanes": config_pipeline_lanes(
                metadata_configs[snapshot["config_id"]], pipeline_lanes
            ),
        }
        for snapshot in depth_selection["candidate_results"]
    ]
    expected_representatives = [
        {
            "config_id": member["config_id"],
            "assigned_pipeline_lane": pipeline_lane_metric(lane),
        }
        for member, lane in lane_diverse_members(
            members, pipeline_lanes, limit=retained_limit
        )
    ]
    check_equal(
        depth_selection["selected_representatives"],
        expected_representatives,
        f"{path}: immutable CLC depth representatives for {result['family']}",
    )


def expected_structural_retention(
    qualified_leaves: list[dict[str, Any]],
    *,
    retained_per_leaf: int,
    retained_family_cap: int | None,
    retained_family_limit: int,
    retained_family_slowdown_limit: float,
    starting_path_limit: int,
    parent_score_config_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Replay the autotuner's deterministic family and starting-path policy."""
    family_leaves: dict[str, list[dict[str, Any]]] = {}
    retained_by_leaf: dict[
        tuple[str, str | None, bool],
        list[tuple[dict[str, Any], tuple[str, int] | None]],
    ] = {}
    for leaf in qualified_leaves:
        members = leaf["members"]
        if not members:
            continue
        family = leaf["family"]
        packet = leaf["compound_packet"]
        softmax_disc = bool(leaf.get("softmax_disc", False))
        leaf = {**leaf, "softmax_disc": softmax_disc}
        lanes = leaf["pipeline_lanes"]
        sorted_members = sorted(members, key=itemgetter("selection_perf", "config_id"))
        family_leaves.setdefault(family, []).append({**leaf, "members": sorted_members})
        retained_by_leaf[(family, packet, softmax_disc)] = lane_diverse_members(
            sorted_members, lanes, limit=retained_per_leaf
        )

    family_queues: dict[str, list[dict[str, Any]]] = {}
    family_scores: dict[str, dict[str, Any]] = {}
    parent_score_families: set[str] = set()
    for family, leaves in family_leaves.items():
        queue: list[dict[str, Any]] = []
        for rank in range(retained_per_leaf):
            layer = [
                {
                    "compound_packet": leaf["compound_packet"],
                    "softmax_disc": leaf["softmax_disc"],
                    "member": retained[rank][0],
                    "pipeline_lane": retained[rank][1],
                }
                for leaf in leaves
                for retained in [
                    retained_by_leaf[
                        (family, leaf["compound_packet"], leaf["softmax_disc"])
                    ]
                ]
                if rank < len(retained)
            ]
            queue.extend(
                sorted(
                    layer,
                    key=lambda item: (
                        item["member"]["selection_perf"],
                        item["compound_packet"] or "",
                        item["softmax_disc"],
                        item["member"]["config_id"],
                    ),
                )
            )
        family_queues[family] = queue
        ordinary = []
        for leaf in leaves:
            if leaf["compound_packet"] is not None:
                continue
            score_member = next(
                (
                    member
                    for member in leaf["members"]
                    if parent_score_config_ids is None
                    or member["config_id"] in parent_score_config_ids
                ),
                None,
            )
            if score_member is not None:
                ordinary.append(
                    {
                        "compound_packet": None,
                        "softmax_disc": leaf["softmax_disc"],
                        "member": score_member,
                    }
                )
        if ordinary:
            parent_score_families.add(family)
        family_scores[family] = min(
            ordinary or [queue[0]],
            key=lambda item: (
                item["member"]["selection_perf"],
                item["member"]["config_id"],
            ),
        )

    if not family_queues:
        return []
    ranked_families = sorted(
        parent_score_families,
        key=lambda family: (
            family_scores[family]["member"]["selection_perf"],
            family,
        ),
    )
    competitive_families: list[str] = []
    if retained_family_cap is None:
        competitive_families = ranked_families
    elif ranked_families:
        best_family_perf = family_scores[ranked_families[0]]["member"]["selection_perf"]
        competitive_families = [
            family
            for family in ranked_families
            if family_scores[family]["member"]["selection_perf"]
            <= best_family_perf * retained_family_slowdown_limit
        ]

    best_family, best_leaf = min(
        ((family, leaf) for family, leaves in family_leaves.items() for leaf in leaves),
        key=lambda item: (
            item[1]["members"][0]["selection_perf"],
            item[0],
            item[1]["compound_packet"] or "",
            item[1]["softmax_disc"],
            item[1]["members"][0]["config_id"],
        ),
    )
    best_packet = best_leaf["compound_packet"]
    best_softmax_disc = best_leaf["softmax_disc"]
    best_member = best_leaf["members"][0]
    family_score = family_scores[best_family]
    best_leaf_key = (best_family, best_packet, best_softmax_disc)
    family_score_leaf = (
        best_family,
        family_score["compound_packet"],
        family_score["softmax_disc"],
    )
    alternate_leaf_order = [best_leaf_key]
    if family_score_leaf != best_leaf_key:
        alternate_leaf_order.append(family_score_leaf)
    alternate_leaf_order.extend(
        sorted(
            (
                (best_family, leaf["compound_packet"], leaf["softmax_disc"])
                for leaf in family_leaves[best_family]
                if (best_family, leaf["compound_packet"], leaf["softmax_disc"])
                not in alternate_leaf_order
            ),
            key=lambda leaf: (
                retained_by_leaf[leaf][0][0]["selection_perf"],
                retained_by_leaf[leaf][0][0]["config_id"],
            ),
        )
    )
    best_lane_alternate = next(
        (
            (member, leaf, lane)
            for leaf in alternate_leaf_order
            for member, lane in retained_by_leaf[leaf]
            if member["config_id"] != best_member["config_id"] and lane is not None
        ),
        None,
    )
    constrained_limit = max(0, starting_path_limit - 1)
    retained_parent_families = competitive_families[
        : min(retained_family_limit, constrained_limit)
    ]
    selected: list[dict[str, Any]] = []
    for family in retained_parent_families:
        score = family_scores[family]
        if len(selected) >= constrained_limit:
            break
        selected.append(
            {
                "family": family,
                "compound_packet": score["compound_packet"],
                "softmax_disc": score["softmax_disc"],
                "config_id": score["member"]["config_id"],
                "unrestricted": False,
                "pipeline_lane": None,
            }
        )
    selected_ids = {item["config_id"] for item in selected}
    selected_leaves = {
        (item["family"], item["compound_packet"], item["softmax_disc"])
        for item in selected
    }

    ordinary_leaf_candidates = sorted(
        (
            {
                "family": family,
                "compound_packet": leaf["compound_packet"],
                "softmax_disc": leaf["softmax_disc"],
                "config_id": leaf["members"][0]["config_id"],
                "selection_perf": leaf["members"][0]["selection_perf"],
            }
            for family in retained_parent_families
            for leaf in family_leaves[family]
            if leaf["compound_packet"] is None
            if (family, leaf["compound_packet"], leaf["softmax_disc"])
            not in selected_leaves
        ),
        key=itemgetter("selection_perf", "family", "softmax_disc", "config_id"),
    )
    for item in ordinary_leaf_candidates:
        if len(selected) >= constrained_limit:
            break
        if item["config_id"] in selected_ids:
            continue
        selected.append(
            {
                **{
                    key: item[key]
                    for key in (
                        "family",
                        "compound_packet",
                        "softmax_disc",
                        "config_id",
                    )
                },
                "unrestricted": False,
                "pipeline_lane": None,
            }
        )
        selected_ids.add(item["config_id"])
        selected_leaves.add(
            (item["family"], item["compound_packet"], item["softmax_disc"])
        )

    if (
        best_lane_alternate is not None
        and best_family in retained_parent_families
        and len(selected) < constrained_limit
    ):
        member, leaf, lane = best_lane_alternate
        if member["config_id"] not in selected_ids:
            selected.append(
                {
                    "family": leaf[0],
                    "compound_packet": leaf[1],
                    "softmax_disc": leaf[2],
                    "config_id": member["config_id"],
                    "unrestricted": False,
                    "pipeline_lane": pipeline_lane_metric(lane),
                }
            )
            selected_ids.add(member["config_id"])
            selected_leaves.add(leaf)

    leaf_best_ids = {
        (family, leaf["compound_packet"], leaf["softmax_disc"]): leaf["members"][0][
            "config_id"
        ]
        for family, leaves in family_leaves.items()
        for leaf in leaves
    }
    families_with_ordinary_secondary = {
        item["family"]
        for item in selected
        if item["compound_packet"] is None
        and item["config_id"]
        != leaf_best_ids[(item["family"], None, item["softmax_disc"])]
    }
    for family in retained_parent_families:
        if (
            len(selected) >= constrained_limit
            or family in families_with_ordinary_secondary
        ):
            continue
        secondary = next(
            (
                item
                for item in family_queues[family]
                if item["compound_packet"] is None
                and item["member"]["config_id"]
                != leaf_best_ids[(family, None, item["softmax_disc"])]
                and item["member"]["config_id"] not in selected_ids
            ),
            None,
        )
        if secondary is None:
            continue
        member = secondary["member"]
        selected.append(
            {
                "family": family,
                "compound_packet": None,
                "softmax_disc": secondary["softmax_disc"],
                "config_id": member["config_id"],
                "unrestricted": False,
                "pipeline_lane": pipeline_lane_metric(secondary["pipeline_lane"]),
            }
        )
        selected_ids.add(member["config_id"])
        selected_leaves.add((family, None, secondary["softmax_disc"]))
        families_with_ordinary_secondary.add(family)

    compound_leaf_candidates = sorted(
        (
            {
                "family": family,
                "compound_packet": leaf["compound_packet"],
                "softmax_disc": leaf["softmax_disc"],
                "config_id": leaf["members"][0]["config_id"],
                "selection_perf": leaf["members"][0]["selection_perf"],
            }
            for family, leaves in family_leaves.items()
            for leaf in leaves
            if leaf["compound_packet"] is not None
            and (family, leaf["compound_packet"], leaf["softmax_disc"])
            not in selected_leaves
        ),
        key=lambda item: (
            item["selection_perf"],
            item["family"],
            item["compound_packet"] or "",
            item["softmax_disc"],
            item["config_id"],
        ),
    )
    for item in compound_leaf_candidates:
        if len(selected) >= constrained_limit:
            break
        if item["config_id"] in selected_ids:
            continue
        selected.append(
            {
                "family": item["family"],
                "compound_packet": item["compound_packet"],
                "softmax_disc": item["softmax_disc"],
                "config_id": item["config_id"],
                "unrestricted": False,
                "pipeline_lane": None,
            }
        )
        selected_ids.add(item["config_id"])
        selected_leaves.add(
            (item["family"], item["compound_packet"], item["softmax_disc"])
        )

    offsets = dict.fromkeys(retained_parent_families, 0)
    while len(selected) < constrained_limit:
        added = False
        for family in retained_parent_families:
            queue = family_queues[family]
            offset = offsets[family]
            while (
                offset < len(queue)
                and queue[offset]["member"]["config_id"] in selected_ids
            ):
                offset += 1
            offsets[family] = offset
            if offset >= len(queue):
                continue
            item = queue[offset]
            offsets[family] += 1
            selected.append(
                {
                    "family": family,
                    "compound_packet": item["compound_packet"],
                    "softmax_disc": item["softmax_disc"],
                    "config_id": item["member"]["config_id"],
                    "unrestricted": False,
                    "pipeline_lane": pipeline_lane_metric(item["pipeline_lane"]),
                }
            )
            selected_ids.add(item["member"]["config_id"])
            added = True
            if len(selected) >= constrained_limit:
                break
        if not added:
            break

    if starting_path_limit > 0:
        selected.append(
            {
                "family": best_family,
                "compound_packet": best_packet,
                "softmax_disc": best_softmax_disc,
                "config_id": best_member["config_id"],
                "unrestricted": True,
                "pipeline_lane": None,
            }
        )

    reported_families = list(
        dict.fromkeys(
            (
                best_family,
                *retained_parent_families,
                *(item["family"] for item in selected),
            )
        )
    )
    return [
        {
            "family": family,
            "score": family_scores[family]["member"]["selection_perf"],
            "score_compound_packet": family_scores[family]["compound_packet"],
            "score_softmax_disc": family_scores[family]["softmax_disc"],
            "parent_promoted": family in retained_parent_families,
            "starting_paths": [item for item in selected if item["family"] == family],
        }
        for family in reported_families
    ]


def optional_positive_float(value: object) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: result must be a JSON object")
    return value


def strict_int(value: object, label: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"{label}: expected an integer, got {value!r}")
    if minimum is not None:
        require(value >= minimum, f"{label}: expected at least {minimum}, got {value}")
    return value


def finite_float(value: object, label: str, *, positive: bool = False) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"{label}: expected a number, got {value!r}")
    result = float(value)
    require(math.isfinite(result), f"{label}: expected a finite number, got {value!r}")
    if positive:
        require(result > 0.0, f"{label}: expected a positive number, got {value!r}")
    return result


def csv_int(value: str, label: str, *, minimum: int | None = None) -> int:
    require(re.fullmatch(r"-?[0-9]+", value) is not None, f"{label}: invalid integer")
    return strict_int(int(value), label, minimum=minimum)


def csv_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{label}: invalid number {value!r}") from exc
    require(math.isfinite(result), f"{label}: expected a finite number")
    return result


def case_from_payload(path: Path, payload: dict[str, Any]) -> tuple[str, int]:
    shape = payload.get("shape")
    if not isinstance(shape, dict):
        raise RuntimeError(f"{path}: missing shape")
    causal = shape.get("causal")
    require(causal in (0, 1, False, True), f"{path}: invalid causal flag")
    variant = "causal" if bool(causal) else "dense"
    seq_len = strict_int(shape.get("seq_len"), f"{path}: sequence length", minimum=1)
    case = (variant, seq_len)
    require(case in EXPECTED_CASES, f"{path}: unexpected all8 shape {case}")
    return case


def discover_results(artifact_root: Path) -> dict[tuple[str, int], Path]:
    root = artifact_root.expanduser().resolve()
    require(root.is_dir(), f"artifact root is not a directory: {root}")
    results: dict[tuple[str, int], Path] = {}
    for path in sorted(root.rglob("result.json")):
        payload = load_json_object(path)
        if payload.get("impl") != "helion-cute":
            continue
        case = case_from_payload(path, payload)
        require(
            case not in results,
            f"duplicate strict result for {case}: {results.get(case)} and {path}",
        )
        results[case] = path
    missing = set(EXPECTED_CASES) - set(results)
    extra = set(results) - set(EXPECTED_CASES)
    require(
        not missing and not extra and len(results) == 8,
        "strict result set mismatch: "
        f"missing={sorted(missing)}, extra={sorted(extra)}, count={len(results)}",
    )
    return results


def validate_shape_and_timing(
    path: Path, payload: dict[str, Any], variant: str, seq_len: int
) -> tuple[float, float]:
    shape = {
        "z": 2,
        "h": 32,
        "seq_len": seq_len,
        "head_dim": 64,
        "dtype": "float16",
        "causal": int(variant == "causal"),
        "biased": 0,
    }
    check_equal(payload.get("shape"), shape, f"{path}: shape")
    check_equal(
        payload.get("flop_model"), "softmax_attention_forward", f"{path}: flop model"
    )
    check_equal(payload.get("accuracy"), "PASS", f"{path}: accuracy")
    check_equal(payload.get("benchmark_timer"), "wall", f"{path}: timer")

    runs = payload.get("runs_ms")
    if not isinstance(runs, list) or len(runs) != 9:
        raise RuntimeError(f"{path}: expected 9 timing runs")
    parsed_runs = [
        finite_float(value, f"{path}: runs_ms[{index}]", positive=True)
        for index, value in enumerate(runs)
    ]
    median_ms = finite_float(
        payload.get("median_ms"), f"{path}: median_ms", positive=True
    )
    require(
        math.isclose(median_ms, statistics.median(parsed_runs), rel_tol=1e-12),
        f"{path}: median_ms does not match runs_ms",
    )
    median_tflops = finite_float(
        payload.get("median_tflops"), f"{path}: median_tflops", positive=True
    )
    flops = 4.0 * shape["z"] * shape["h"] * seq_len**2 * shape["head_dim"]
    if variant == "causal":
        flops *= 0.5
    expected_tflops = flops / (median_ms * 1e9)
    require(
        math.isclose(median_tflops, expected_tflops, rel_tol=1e-12),
        f"{path}: median_tflops does not match shape and median_ms",
    )
    return median_ms, median_tflops


def flash_structural_population_budget(
    *,
    population_size: int,
    coverage_design_count: int,
    parent_coverage_prefix_count: int,
    qualification_prefix_count: int,
) -> int:
    """Mirror the production budget for normalized flash coverage rows."""
    half_population = population_size // 2
    if coverage_design_count <= population_size:
        return max(half_population, coverage_design_count)
    if qualification_prefix_count <= half_population:
        return half_population
    return min(
        population_size,
        max(half_population, parent_coverage_prefix_count),
    )


def validate_coverage(
    path: Path, provenance: dict[str, Any], selected_config: dict[str, Any]
) -> tuple[int, bool, int]:
    fragment_default = provenance.get("flash_fragment_default_config")
    if not isinstance(fragment_default, dict):
        raise RuntimeError(f"{path}: missing fragment default")
    check_equal(
        provenance.get("flash_fragment_default_sha256"),
        canonical_sha256(fragment_default),
        f"{path}: fragment default digest",
    )

    design = provenance.get("flash_structural_coverage_design")
    if not isinstance(design, list) or not design:
        raise RuntimeError(f"{path}: missing coverage design")
    check_equal(
        provenance.get("flash_structural_coverage_design_count"),
        len(design),
        f"{path}: coverage design count",
    )
    configs: list[dict[str, Any]] = []
    for index, item in enumerate(design):
        require(isinstance(item, dict), f"{path}: coverage entry {index} is invalid")
        config = item.get("config")
        require(isinstance(config, dict), f"{path}: coverage config {index} is invalid")
        check_equal(
            item.get("config_sha256"),
            canonical_sha256(config),
            f"{path}: coverage config {index} digest",
        )
        configs.append(config)
    require(
        len({canonical_sha256(config) for config in configs}) == len(configs),
        f"{path}: coverage design contains duplicate configs",
    )
    check_equal(
        provenance.get("flash_structural_coverage_design_sha256"),
        canonical_sha256(configs),
        f"{path}: coverage design digest",
    )
    check_equal(
        provenance.get("flash_structural_coverage_uncovered_values"),
        [],
        f"{path}: uncovered structural values",
    )
    check_equal(
        provenance.get("flash_structural_coverage_underqualified_values"),
        [],
        f"{path}: underqualified structural values",
    )
    check_equal(
        provenance.get("flash_structural_coverage_underqualified_leaves"),
        [],
        f"{path}: underqualified exact structural leaves",
    )
    leaf_catalog = provenance.get("flash_structural_leaf_catalog")
    require(
        isinstance(leaf_catalog, list)
        and leaf_catalog
        and all(
            isinstance(leaf, dict)
            and set(leaf) == {"family", "compound_packet", "softmax_disc"}
            and isinstance(leaf["family"], str)
            and isinstance(leaf["softmax_disc"], bool)
            and (
                leaf["compound_packet"] is None
                or isinstance(leaf["compound_packet"], str)
            )
            for leaf in leaf_catalog
        ),
        f"{path}: invalid exact structural leaf catalog",
    )
    catalog_keys = [canonical_json(leaf) for leaf in leaf_catalog]
    ordinary_owner_keys = {
        (leaf["family"], leaf["softmax_disc"])
        for leaf in leaf_catalog
        if leaf["compound_packet"] is None
    }
    require(
        all(
            leaf["compound_packet"] is None
            or (leaf["family"], leaf["softmax_disc"]) in ordinary_owner_keys
            for leaf in leaf_catalog
        ),
        f"{path}: compound leaf has no ordinary family/protocol owner",
    )
    design_leaf_keys = {
        canonical_json(leaf)
        for config in configs
        if (leaf := structural_leaf(config)) is not None
    }
    compound_owners = [
        leaf["compound_packet"]
        for leaf in leaf_catalog
        if leaf["compound_packet"] is not None
    ]
    require(
        len(catalog_keys) == len(set(catalog_keys))
        and set(catalog_keys) == design_leaf_keys
        and len(compound_owners) == len(set(compound_owners)),
        f"{path}: exact leaf catalog is inconsistent with family/packet owners",
    )
    check_equal(
        provenance.get("flash_structural_coverage_uncovered_interactions"),
        [],
        f"{path}: uncovered structural interactions",
    )
    active_values = provenance.get("flash_structural_coverage_active_values")
    if not isinstance(active_values, list) or not active_values:
        raise RuntimeError(f"{path}: no active values")
    for active in active_values:
        require(
            isinstance(active, dict)
            and set(active) == {"key", "value"}
            and isinstance(active["key"], str),
            f"{path}: invalid active structural value {active!r}",
        )
        require(
            any(config.get(active["key"]) == active.get("value") for config in configs),
            f"{path}: coverage design omits {active!r}",
        )
    active_interactions = provenance.get(
        "flash_structural_coverage_active_interactions"
    )
    check_equal(
        provenance.get("flash_structural_coverage_interaction_key_groups"),
        [list(group) for group in FLASH_INTERACTION_KEY_GROUPS],
        f"{path}: declared structural interaction groups",
    )
    if not isinstance(active_interactions, list) or not active_interactions:
        raise RuntimeError(f"{path}: no active structural interactions")

    def interaction_matches(
        config: dict[str, Any], interaction: dict[str, Any]
    ) -> bool:
        return all(
            config.get(key) == value
            for key, value in zip(
                interaction["keys"], interaction["values"], strict=True
            )
        )

    for interaction in active_interactions:
        require(
            isinstance(interaction, dict)
            and set(interaction) == {"keys", "values"}
            and isinstance(interaction["keys"], list)
            and isinstance(interaction["values"], list)
            and interaction["keys"]
            and len(interaction["keys"]) == len(interaction["values"])
            and all(isinstance(key, str) for key in interaction["keys"])
            and tuple(interaction["keys"]) in FLASH_INTERACTION_KEY_GROUPS,
            f"{path}: invalid structural interaction {interaction!r}",
        )
        require(
            any(interaction_matches(config, interaction) for config in configs),
            f"{path}: coverage design omits interaction {interaction!r}",
        )
    recorded_interactions = {canonical_json(item) for item in active_interactions}
    require(
        len(recorded_interactions) == len(active_interactions),
        f"{path}: duplicate active structural interactions",
    )
    active_value_keys = {item["key"] for item in active_values}
    expected_interactions = {
        canonical_json(
            {
                "keys": list(group),
                "values": [config.get(key) for key in group],
            }
        )
        for group in FLASH_INTERACTION_KEY_GROUPS
        if any(key in active_value_keys for key in group)
        for config in configs
    }
    check_equal(
        recorded_interactions,
        expected_interactions,
        f"{path}: active structural interaction completeness",
    )

    parent_prefix_count = strict_int(
        provenance.get("flash_structural_parent_coverage_prefix_count"),
        f"{path}: parent coverage prefix count",
        minimum=1,
    )
    qualification_prefix_count = strict_int(
        provenance.get("flash_structural_qualification_prefix_count"),
        f"{path}: qualification prefix count",
        minimum=parent_prefix_count,
    )
    initial_population_size = strict_int(
        provenance.get("autotune_initial_population_size"),
        f"{path}: initial population size",
        minimum=1,
    )
    population_budget = strict_int(
        provenance.get("flash_structural_population_budget"),
        f"{path}: structural population budget",
        minimum=1,
    )
    check_equal(
        population_budget,
        flash_structural_population_budget(
            population_size=initial_population_size,
            coverage_design_count=len(configs),
            parent_coverage_prefix_count=parent_prefix_count,
            qualification_prefix_count=qualification_prefix_count,
        ),
        f"{path}: structural population budget",
    )
    injected_design_count = strict_int(
        provenance.get("flash_structural_injected_design_count"),
        f"{path}: injected structural design count",
        minimum=1,
    )
    check_equal(
        injected_design_count,
        min(population_budget, len(configs)),
        f"{path}: injected structural design count",
    )
    for interaction in active_interactions:
        require(
            any(
                interaction_matches(config, interaction)
                for config in configs[:injected_design_count]
            ),
            f"{path}: injected coverage prefix omits interaction {interaction!r}",
        )
    require(
        qualification_prefix_count <= len(configs),
        f"{path}: qualification prefix exceeds coverage design",
    )
    parent_values = [
        active
        for active in active_values
        if active["key"] in (FLASH_PIPELINE_FAMILY_KEY, FLASH_EXP2_PACKET_KEY)
    ]
    for active in parent_values:
        require(
            any(
                config.get(active["key"]) == active["value"]
                for config in configs[:parent_prefix_count]
            ),
            f"{path}: parent prefix omits {active!r}",
        )
    qualification_values = provenance.get("flash_structural_qualification_values")
    expected_qualification_values = structural_qualification_values(active_values)
    check_equal(
        qualification_values,
        expected_qualification_values,
        f"{path}: structural qualification manifest",
    )
    require(
        expected_qualification_values,
        f"{path}: no active structural qualification values",
    )
    for qualified in expected_qualification_values:
        witnesses = {
            canonical_sha256(config)
            for config in configs[:qualification_prefix_count]
            if config.get(qualified["key"]) == qualified["value"]
        }
        required_witnesses = 2 if qualified["key"] == FLASH_PIPELINE_FAMILY_KEY else 1
        require(
            len(witnesses) >= required_witnesses,
            f"{path}: qualification prefix has fewer than {required_witnesses} "
            f"distinct {qualified!r}",
        )
    for leaf in leaf_catalog:
        total_witnesses = sum(structural_leaf(config) == leaf for config in configs)
        prefix_witnesses = sum(
            structural_leaf(config) == leaf
            for config in configs[:qualification_prefix_count]
        )
        required_witnesses = (
            1 if leaf["compound_packet"] is not None else min(2, total_witnesses)
        )
        require(
            prefix_witnesses >= required_witnesses,
            f"{path}: qualification prefix underrepresents exact leaf {leaf!r}",
        )

    distances = [
        sum(
            key not in selected_config
            or key not in config
            or selected_config[key] != config[key]
            for key in selected_config.keys() | config.keys()
        )
        for config in configs
    ]
    distance = min(distances)
    nearest = sorted(
        canonical_sha256(config)
        for config, config_distance in zip(configs, distances, strict=True)
        if config_distance == distance
    )
    check_equal(
        provenance.get(
            "selected_config_nearest_structural_coverage_design_field_distance"
        ),
        distance,
        f"{path}: winner coverage distance",
    )
    check_equal(
        sorted(
            provenance.get(
                "selected_config_nearest_structural_coverage_design_config_sha256", []
            )
        ),
        nearest,
        f"{path}: nearest coverage configs",
    )
    member = distance == 0
    check_equal(
        provenance.get("selected_config_is_structural_coverage_design_member"),
        member,
        f"{path}: winner coverage membership",
    )
    return len(configs), member, distance


def validate_strict_provenance(
    path: Path,
    payload: dict[str, Any],
    variant: str,
    seq_len: int,
    tuner_seed: int,
    *,
    expected_input_shape: tuple[int, int, int, int] | None = None,
    expected_input_dtype: str = "torch.float16",
) -> tuple[dict[str, Any], dict[str, Any], int, bool, int]:
    overrides = payload.get("helion_overrides")
    if not isinstance(overrides, dict):
        raise RuntimeError(f"{path}: missing helion_overrides")
    expected_overrides = {
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
    }
    for key, expected in expected_overrides.items():
        check_equal(overrides.get(key), expected, f"{path}: helion_overrides.{key}")

    provenance = overrides.get("autotune_provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError(f"{path}: missing autotune provenance")
    expected_baseline = (
        "examples.attention._causal_attention_output_baseline"
        if variant == "causal"
        else "examples.attention._attention_output_baseline"
    )
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
        "autotune_baseline_fn": expected_baseline,
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
        "compiler_default_config": False,
        "kernel_declared_config_count": 0,
        "active_value_prior_keys": [],
        "flash_value_prior_keys": [],
        "flash_structural_coverage_design_source": (
            "normalized active ConfigSpec fragments"
        ),
        "flash_structural_coverage_uncovered_values": [],
        "flash_structural_coverage_underqualified_values": [],
        "flash_structural_coverage_underqualified_leaves": [],
        "flash_structural_coverage_uncovered_interactions": [],
        "flash_structural_coverage_interaction_key_groups": [
            list(group) for group in FLASH_INTERACTION_KEY_GROUPS
        ],
        "flash_structural_qualification_rounds": 2,
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": 4,
        "flash_structural_family_probe_generations": (
            EXPECTED_FAMILY_PROBE_GENERATIONS
        ),
        "flash_structural_family_probe_candidates_per_path": (
            EXPECTED_FAMILY_PROBE_CANDIDATES_PER_PATH
        ),
        "flash_structural_retained_candidates_per_leaf": 2,
        "flash_structural_retained_family_cap": (EXPECTED_FULL_FLASH_RETAINED_FAMILIES),
        "flash_structural_retained_family_slowdown_limit": 2.0,
        "flash_structural_unrestricted_path_exhausts_generation_budget": True,
        "cute_flash_env_overrides": {},
        "final_correctness_enabled": True,
        "final_correctness_launches": EXPECTED_FINAL_CORRECTNESS_LAUNCHES,
        "final_repeatability_passed": True,
        "final_correctness_passed": True,
        "cache_read_policy": "bypass",
        "cache_write_policy": "write",
        "skip_cache_env": False,
        "rebenchmark_env_overrides": {},
    }
    for key, expected in strict_values.items():
        check_equal(provenance.get(key), expected, f"{path}: provenance.{key}")
    validate_compiler_seed_policy(path, provenance)
    flash_clc_lane_catalog(path, provenance)
    exact_config_ids = exact_effective_search_space_ids(path, provenance)
    require(
        not provenance.get("dense_d64_2cta_performance_anchor_present", False),
        f"{path}: legacy dense shape anchor is active",
    )

    selected_config = provenance.get("selected_config")
    if not isinstance(selected_config, dict):
        raise RuntimeError(f"{path}: missing selected config")
    coverage = validate_coverage(path, provenance, selected_config)

    selected_source = provenance.get("selected_source_sha256")
    require(
        isinstance(selected_source, str)
        and re.fullmatch(r"[0-9a-f]{64}", selected_source) is not None,
        f"{path}: invalid selected source SHA256",
    )
    trials = provenance.get("trials")
    if not isinstance(trials, list) or len(trials) != 1:
        raise RuntimeError(f"{path}: expected one trial")
    trial = trials[0]
    require(isinstance(trial, dict), f"{path}: trial is not an object")
    check_equal(trial.get("random_seed"), tuner_seed, f"{path}: trial tuner seed")
    check_equal(trial.get("search_algorithm"), "LFBOTreeSearch", f"{path}: search")
    shape = expected_input_shape or (2, 32, seq_len, 64)
    check_equal(
        trial.get("input_shapes"),
        repr([shape, shape, shape]),
        f"{path}: trial input shapes",
    )
    check_equal(
        trial.get("dtypes"),
        repr([expected_input_dtype] * 3),
        f"{path}: trial input dtypes",
    )
    check_equal(trial.get("hardware"), "NVIDIA B200", f"{path}: trial hardware")
    num_configs_tested = strict_int(
        trial.get("num_configs_tested"),
        f"{path}: trial.num_configs_tested",
        minimum=1,
    )
    num_successful = strict_int(
        trial.get("num_successful_candidate_measurements"),
        f"{path}: trial.num_successful_candidate_measurements",
        minimum=1,
    )
    strict_int(
        trial.get("num_unique_sources"),
        f"{path}: trial.num_unique_sources",
        minimum=1,
    )
    for field in (
        "num_compile_failures",
        "num_worker_failures",
        "num_isolated_rebenchmark_timeouts",
        "num_accuracy_failures",
    ):
        strict_int(trial.get(field), f"{path}: trial.{field}", minimum=0)
    num_source_deduplications = strict_int(
        trial.get("num_source_deduplications"),
        f"{path}: trial.num_source_deduplications",
        minimum=0,
    )
    required_candidates = 100 if exact_config_ids is None else len(exact_config_ids)
    require(
        num_configs_tested + num_source_deduplications >= required_candidates,
        f"{path}: trial covered fewer than {required_candidates} effective candidates",
    )
    if exact_config_ids is None:
        require(
            num_successful >= required_candidates,
            f"{path}: trial has fewer than {required_candidates} actual successful "
            "candidate measurements",
        )
    num_generations = strict_int(
        trial.get("num_generations"), f"{path}: trial.num_generations", minimum=0
    )
    lfbo_max_generations = strict_int(
        provenance.get("autotune_lfbo_max_generations"),
        f"{path}: provenance.autotune_lfbo_max_generations",
        minimum=1,
    )
    if exact_config_ids is None:
        check_equal(
            num_generations,
            lfbo_max_generations,
            f"{path}: unrestricted path generation budget",
        )
    else:
        require(
            num_generations <= lfbo_max_generations,
            f"{path}: exact-space search exceeded its generation budget",
        )
    finite_float(
        trial.get("autotune_time"), f"{path}: trial.autotune_time", positive=True
    )
    finite_float(
        trial.get("best_perf_ms"), f"{path}: trial.best_perf_ms", positive=True
    )
    check_equal(trial.get("selected_config"), selected_config, f"{path}: trial winner")
    check_equal(
        trial.get("selected_source_hash"), selected_source, f"{path}: trial source"
    )
    check_equal(
        trial.get("selected_source_was_measured"),
        True,
        f"{path}: selected source measured flag",
    )
    return provenance, trial, *coverage


def _config_id_list(value: object, label: str) -> list[str]:
    require(
        isinstance(value, list)
        and all(
            isinstance(config_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
            for config_id in value
        )
        and len(value) == len(set(value)),
        f"{label}: invalid config ID list",
    )
    return value


def validate_compiler_seed_policy(
    path: Path | str,
    provenance: dict[str, Any],
    *,
    phase: dict[str, Any] | None = None,
    metadata_configs: dict[str, dict[str, Any]] | None = None,
    source_rows: list[dict[str, str]] | None = None,
    invalidated_config_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate the only compiler-owned seed policy allowed in strict runs."""
    policy = provenance.get("compiler_seed_policy")
    expected_fields = {
        "schema_version",
        "kind",
        "heuristic_names",
        "raw_config_count",
        "effective_config_ids",
        "effective_config_ids_sha256",
        "timeout_retry_repetitions",
    }
    require(
        isinstance(policy, dict) and set(policy) == expected_fields,
        f"{path}: missing or malformed compiler seed policy",
    )
    require(
        type(policy.get("schema_version")) is int
        and policy["schema_version"] == EXPECTED_COMPILER_SEED_POLICY_SCHEMA_VERSION,
        f"{path}: invalid compiler seed policy schema",
    )
    check_equal(
        policy.get("kind"),
        EXPECTED_COMPILER_SEED_POLICY_KIND,
        f"{path}: compiler seed policy kind",
    )
    check_equal(
        policy.get("heuristic_names"),
        list(EXPECTED_COMPILER_SEED_HEURISTICS),
        f"{path}: compiler seed heuristic names",
    )
    raw_count = strict_int(
        policy.get("raw_config_count"),
        f"{path}: compiler seed raw config count",
        minimum=1,
    )
    effective_ids = _config_id_list(
        policy.get("effective_config_ids"),
        f"{path}: compiler seed effective config IDs",
    )
    require(effective_ids, f"{path}: compiler seed policy is empty")
    require(
        raw_count >= len(effective_ids),
        f"{path}: compiler seed raw count is below its normalized count",
    )
    check_equal(
        provenance.get("compiler_seed_config_count"),
        raw_count,
        f"{path}: compiler seed raw count",
    )
    check_equal(
        policy.get("effective_config_ids_sha256"),
        canonical_sha256(effective_ids),
        f"{path}: compiler seed effective config digest",
    )
    check_equal(
        policy.get("timeout_retry_repetitions"),
        EXPECTED_COMPILER_SEED_TIMEOUT_RETRY_REPETITIONS,
        f"{path}: compiler seed timeout retry policy",
    )

    if phase is None:
        return policy

    initial_ids = _config_id_list(
        phase.get("initial_config_ids"), f"{path}: initial config IDs"
    )
    initial_id_set = set(initial_ids)
    missing_initial = [
        config_id for config_id in effective_ids if config_id not in initial_id_set
    ]
    require(
        not missing_initial,
        f"{path}: compiler seeds missing from generation zero: {missing_initial}",
    )
    config_manifest = phase.get("config_manifest")
    require(isinstance(config_manifest, dict), f"{path}: missing config manifest")
    initial_results = phase.get("initial_results")
    require(isinstance(initial_results, list), f"{path}: missing initial results")
    initial_result_by_id = {
        record.get("config_id"): record
        for record in initial_results
        if isinstance(record, dict)
    }
    invalidated = invalidated_config_ids or set()
    for config_id in effective_ids:
        entry = config_manifest.get(config_id)
        require(
            isinstance(entry, dict) and isinstance(entry.get("config"), dict),
            f"{path}: compiler seed {config_id} is absent from the config manifest",
        )
        config = entry["config"]
        check_equal(
            canonical_sha256(config)[:16],
            config_id,
            f"{path}: compiler seed {config_id} canonical ID",
        )
        if metadata_configs is not None:
            check_equal(
                metadata_configs.get(config_id),
                config,
                f"{path}: compiler seed {config_id} metadata config",
            )
        result = initial_result_by_id.get(config_id)
        require(
            isinstance(result, dict)
            and result.get("measurement_pass_index") == 0
            and result.get("status") in {"ok", "deduplicated"}
            and terminal_measurement_is_valid(result)
            and config_id not in invalidated,
            f"{path}: compiler seed {config_id} lacks a successful generation-zero "
            "measurement",
        )
        if source_rows is not None:
            require(
                any(
                    row.get("config_id") == config_id
                    and row.get("generation") == "0"
                    and row.get("status") in {"ok", "deduplicated"}
                    for row in source_rows
                ),
                f"{path}: compiler seed {config_id} lacks successful generation-zero "
                "sidecar evidence",
            )
    return policy


def _positive_int_list(value: object, label: str) -> list[int]:
    require(
        isinstance(value, (list, tuple))
        and len(value) == len(set(value))
        and all(type(item) is int and item > 0 for item in value),
        f"{label}: invalid positive integer list",
    )
    return list(value)


def terminal_measurement_is_valid(
    record: dict[str, Any], *, allow_projection_rejected: bool = False
) -> bool:
    status = record.get("status")
    attempt_perf = record.get("attempt_perf")
    selection_perf = record.get("selection_perf")
    if status in {"ok", "deduplicated"}:
        return bool(
            optional_positive_float(attempt_perf)
            and attempt_perf is not None
            and optional_positive_float(selection_perf)
            and selection_perf is not None
        )
    if status in {"error", "timeout", "peer_compilation_fail"}:
        return attempt_perf is None and selection_perf is None
    return bool(
        allow_projection_rejected
        and status == "projection_rejected"
        and record.get("config_id") is None
        and record.get("projected_config_id") is None
        and attempt_perf is None
        and selection_perf is None
    )


def generation_zero_initial_config_order(
    path: Path,
    source_rows: list[dict[str, str]],
    expected_count: int,
) -> list[str]:
    """Recover generation-zero admission order before later phase candidates."""
    ordered: list[str] = []
    seen: set[str] = set()
    for row in source_rows:
        config_id = row["config_id"]
        if (
            row["generation"] != "0"
            or row["status"] not in ({"started"} | LEDGER_ALIAS_STATUSES)
            or config_id in seen
        ):
            continue
        seen.add(config_id)
        ordered.append(config_id)
        if len(ordered) == expected_count:
            break
    require(
        len(ordered) == expected_count,
        f"{path}: generation-zero ledger has an incomplete initial population",
    )
    return ordered


def valid_source_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _repair_id_mapping(
    value: object,
    allowed_values: list[int],
    label: str,
) -> dict[str, list[str]]:
    require(isinstance(value, dict), f"{label}: invalid repair ID mapping")
    allowed_keys = [str(item) for item in allowed_values]
    parsed: dict[str, list[str]] = {}
    for key, raw_ids in value.items():
        require(key in allowed_keys, f"{label}: invalid repair value")
        ids = _config_id_list(raw_ids, label)
        require(
            len(ids) <= EXPECTED_QUALIFICATION_FAILURE_RETRIES,
            f"{label}: oversized repair ID list",
        )
        parsed[key] = ids
    check_equal(
        list(parsed),
        [key for key in allowed_keys if key in parsed],
        f"{label}: repair value order",
    )
    flat_ids = [config_id for ids in parsed.values() for config_id in ids]
    require(
        len(flat_ids) == len(set(flat_ids)),
        f"{label}: duplicate repair config ID",
    )
    return parsed


def validate_measurement_timeline(
    path: Path,
    phase: dict[str, Any],
    configs: dict[str, dict[str, Any]],
) -> list[dict[str, dict[str, Any]]]:
    """Validate and replay the immutable v22 measurement state timeline."""
    timeline = phase.get("measurement_timeline")
    pass_count = phase.get("qualification_passes_completed")
    require(
        isinstance(timeline, list)
        and type(pass_count) is int
        and len(timeline) == pass_count + 1,
        f"{path}: malformed v22 measurement timeline",
    )
    initial_ids = set(phase["initial_config_ids"])
    successful_statuses = {"ok", "deduplicated"}
    known_statuses = successful_statuses | {
        "error",
        "timeout",
        "peer_compilation_fail",
        "accuracy_error",
        "source_rejected",
        "filtered",
    }
    retryable_statuses = {"error", "timeout", "peer_compilation_fail"}
    current: dict[str, dict[str, Any]] = {}
    states_by_pass: list[dict[str, dict[str, Any]]] = []
    isolated_invalidated_source_hashes: set[str] = set()
    for expected_pass, record in enumerate(timeline):
        require(
            isinstance(record, dict)
            and set(record) == {"pass_index", "updates"}
            and record["pass_index"] == expected_pass
            and isinstance(record["updates"], list),
            f"{path}: malformed v22 measurement timeline",
        )
        update_ids: list[str] = []
        next_states = dict(current)
        introduced_success_hashes: set[str] = set()
        source_repair_hashes: list[str] = []
        for update in record["updates"]:
            require(
                isinstance(update, dict)
                and set(update)
                == {
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                }
                and isinstance(update["config_id"], str)
                and re.fullmatch(r"[0-9a-f]{16}", update["config_id"]) is not None
                and update["config_id"] in configs
                and update["status"] in known_statuses,
                f"{path}: malformed v22 measurement timeline update",
            )
            config_id = update["config_id"]
            state = {
                key: update[key]
                for key in ("attempt_perf", "selection_perf", "status", "source_hash")
            }
            succeeded = state["status"] in successful_statuses
            require(
                config_id not in update_ids
                and (
                    state["source_hash"] is None
                    or valid_source_hash(state["source_hash"])
                )
                and (not succeeded or valid_source_hash(state["source_hash"]))
                and optional_positive_float(state["attempt_perf"])
                and optional_positive_float(state["selection_perf"])
                and (
                    state["attempt_perf"] is not None
                    and state["selection_perf"] is not None
                )
                is succeeded
                and current.get(config_id) != state,
                f"{path}: invalid v22 measurement timeline update",
            )
            previous = current.get(config_id)
            if previous is not None:
                previous_succeeded = previous["status"] in successful_statuses
                rebenchmarked = (
                    previous_succeeded
                    and succeeded
                    and state["status"] == previous["status"]
                    and state["attempt_perf"] == previous["attempt_perf"]
                    and state["source_hash"] == previous["source_hash"]
                )
                source_repaired = (
                    previous["status"] in retryable_statuses
                    and state["status"] == "deduplicated"
                    and succeeded
                    and valid_source_hash(previous["source_hash"])
                    and state["source_hash"] == previous["source_hash"]
                )
                isolated_rebenchmark_invalidated = (
                    previous_succeeded
                    and state["status"] in {"error", "timeout"}
                    and state["attempt_perf"] is None
                    and state["selection_perf"] is None
                    and state["source_hash"] == previous["source_hash"]
                )
                require(
                    rebenchmarked
                    or source_repaired
                    or isolated_rebenchmark_invalidated,
                    f"{path}: invalid v22 measurement state transition",
                )
                if source_repaired:
                    source_repair_hashes.append(state["source_hash"])
                if isolated_rebenchmark_invalidated:
                    isolated_invalidated_source_hashes.add(state["source_hash"])
            elif succeeded:
                introduced_success_hashes.add(state["source_hash"])
            update_ids.append(config_id)
            next_states[config_id] = state
        require(
            set(source_repair_hashes) <= introduced_success_hashes,
            f"{path}: unproven v22 effective-source repair",
        )
        require(
            not any(
                state["status"] in successful_statuses
                and state["source_hash"] in isolated_invalidated_source_hashes
                for state in next_states.values()
            ),
            f"{path}: incomplete v22 effective-source invalidation",
        )
        check_equal(
            update_ids,
            sorted(update_ids),
            f"{path}: reordered v22 measurement timeline update",
        )
        if expected_pass == 0:
            check_equal(
                set(update_ids),
                initial_ids,
                f"{path}: v22 measurement timeline initial population",
            )
        current = next_states
        states_by_pass.append(current)
    return states_by_pass


def isolated_rebenchmark_invalidations(
    phase: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return configs invalidated after an isolated successful rebenchmark."""
    successful_statuses = {"ok", "deduplicated"}
    current: dict[str, dict[str, Any]] = {}
    invalidations: dict[str, dict[str, Any]] = {}
    for pass_record in phase["measurement_timeline"]:
        for update in pass_record["updates"]:
            previous = current.get(update["config_id"])
            if (
                previous is not None
                and previous["status"] in successful_statuses
                and update["status"] in {"error", "timeout"}
                and update["attempt_perf"] is None
                and update["selection_perf"] is None
                and update["source_hash"] == previous["source_hash"]
            ):
                invalidations[update["config_id"]] = {
                    key: update[key]
                    for key in (
                        "attempt_perf",
                        "selection_perf",
                        "status",
                        "source_hash",
                    )
                }
            current[update["config_id"]] = update
    return invalidations


def isolated_rebenchmark_timeout_source_hashes(
    phase: dict[str, Any],
) -> set[str]:
    """Return distinct sources invalidated by an isolated rebenchmark timeout."""
    return {
        state["source_hash"]
        for state in isolated_rebenchmark_invalidations(phase).values()
        if state["status"] == "timeout"
    }


def validate_measurement_snapshot(
    path: Path,
    states_by_pass: list[dict[str, dict[str, Any]]],
    record: dict[str, Any],
    *,
    config_id: str,
    label: str,
    allow_unmeasured: bool = False,
    expected_pass_index: int | None = None,
) -> None:
    pass_index = record.get("measurement_pass_index")
    snapshot = {
        key: record.get(key)
        for key in ("attempt_perf", "selection_perf", "status", "source_hash")
    }
    if pass_index is None:
        require(
            allow_unmeasured
            and snapshot
            == {
                "attempt_perf": None,
                "selection_perf": None,
                "status": "unknown",
                "source_hash": None,
            }
            and (
                expected_pass_index is None
                or config_id not in states_by_pass[expected_pass_index]
            ),
            f"{path}: {label}",
        )
        return
    require(
        type(pass_index) is int
        and 0 <= pass_index < len(states_by_pass)
        and (expected_pass_index is None or pass_index == expected_pass_index)
        and states_by_pass[pass_index].get(config_id) == snapshot,
        f"{path}: {label}",
    )


def validate_measurement_introductions(
    path: Path,
    phase: dict[str, Any],
    states_by_pass: list[dict[str, dict[str, Any]]],
    scheduled_ids_by_completion_pass: list[set[str]],
) -> None:
    require(
        len(scheduled_ids_by_completion_pass) == len(states_by_pass),
        f"{path}: inconsistent v22 qualification pass accounting",
    )
    timeline = phase["measurement_timeline"]
    for pass_index in range(1, len(states_by_pass)):
        previous_ids = set(states_by_pass[pass_index - 1])
        actual_new_ids = set(states_by_pass[pass_index]) - previous_ids
        expected_new_ids = scheduled_ids_by_completion_pass[pass_index] - previous_ids
        updates = timeline[pass_index]["updates"]
        require(
            actual_new_ids == expected_new_ids
            and (bool(expected_new_ids) or not updates),
            f"{path}: inconsistent v22 measurement introduction timeline",
        )


def validate_timeline_source_repairs(
    path: Path,
    phase: dict[str, Any],
    attempt_history_by_config: dict[str, list[dict[str, Any]]],
) -> None:
    """Bind each timeline source repair to that config's sidecar lifecycle."""
    retryable_statuses = {"error", "timeout", "peer_compilation_fail"}
    current_states: dict[str, dict[str, Any]] = {}
    for pass_record in phase["measurement_timeline"]:
        for update in pass_record["updates"]:
            config_id = update["config_id"]
            previous = current_states.get(config_id)
            if (
                previous is not None
                and previous["status"] in retryable_statuses
                and update["status"] == "deduplicated"
            ):
                history = attempt_history_by_config.get(config_id)
                require(
                    isinstance(history, list)
                    and [attempt["status"] for attempt in history]
                    == [previous["status"], "deduplicated"]
                    and len({attempt["source_hash"] for attempt in history}) == 1
                    and history[0]["source_hash"] == previous["source_hash"]
                    and history[0]["source_hash"] == update["source_hash"],
                    f"{path}: timeline source repair lacks a matching same-source "
                    "sidecar lifecycle",
                )
            current_states[config_id] = update


def validate_failure_repair_decisions(
    path: Path,
    raw_decisions: object,
    repair_ids_by_value: dict[str, list[str]],
    primary_ids_by_value: dict[str, object],
    values: list[int],
    *,
    expected_kind: str,
    expected_leaf: dict[str, Any],
    candidate_limit: int,
    phase_configs: dict[str, dict[str, Any]],
    measurement_states: list[dict[str, dict[str, Any]]],
    label: str,
    missing_generation_parent_ids_by_value: dict[str, str] | None = None,
) -> set[int]:
    require(isinstance(raw_decisions, list), f"{path}: {label}")
    expected_values = [value for value in values if str(value) in repair_ids_by_value]
    require(len(raw_decisions) == len(expected_values), f"{path}: {label}")
    decision_passes: set[int] = set()
    local_pass_counts: dict[int, int] = {}
    neighbor_limits_by_pass: dict[int, list[int]] = {}
    ordered_decision_passes: list[int] = []
    for value, decision in zip(expected_values, raw_decisions, strict=True):
        require(
            isinstance(decision, dict)
            and set(decision)
            == {
                "kind",
                "value",
                "repair_index",
                "candidate_results",
                "selected_config_id",
                "generated_config_ids",
                "neighbor_generation_limit",
            }
            and decision["kind"] == expected_kind
            and decision["value"] == value
            and type(decision["repair_index"]) is int
            and 0 <= decision["repair_index"] < EXPECTED_QUALIFICATION_FAILURE_RETRIES
            and isinstance(decision["candidate_results"], list)
            and type(decision["neighbor_generation_limit"]) is int
            and 0 < decision["neighbor_generation_limit"] <= 200,
            f"{path}: {label}",
        )
        candidate_ids: list[str] = []
        candidate_passes: set[int] = set()
        for snapshot in decision["candidate_results"]:
            require(
                isinstance(snapshot, dict)
                and set(snapshot)
                == {
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                }
                and snapshot["config_id"] in phase_configs
                and snapshot["status"]
                in {
                    "ok",
                    "deduplicated",
                    "error",
                    "timeout",
                    "peer_compilation_fail",
                }
                and terminal_measurement_is_valid(snapshot),
                f"{path}: {label}",
            )
            validate_measurement_snapshot(
                path,
                measurement_states,
                snapshot,
                config_id=snapshot["config_id"],
                label=label,
            )
            candidate_ids.append(snapshot["config_id"])
            candidate_passes.add(snapshot["measurement_pass_index"])
        require(
            len(candidate_passes) == 1 and candidate_ids == sorted(set(candidate_ids)),
            f"{path}: {label}",
        )
        decision_pass = next(iter(candidate_passes))
        primary = primary_ids_by_value.get(str(value))
        primary_ids = [primary] if isinstance(primary, str) else list(primary or ())
        repair_index = decision["repair_index"]
        tracked_ids = [
            *primary_ids,
            *repair_ids_by_value[str(value)][:repair_index],
        ]
        states = measurement_states[decision_pass]
        retryable_statuses = {"error", "timeout", "peer_compilation_fail"}
        retryable_ids = sorted(
            config_id
            for config_id in tracked_ids
            if states.get(config_id, {}).get("status") in retryable_statuses
        )
        fallback_parent = (
            None
            if missing_generation_parent_ids_by_value is None
            else missing_generation_parent_ids_by_value.get(str(value))
        )
        missing_generation = not tracked_ids and fallback_parent is not None
        expected_parent_ids = (
            [fallback_parent]
            if missing_generation
            and states.get(fallback_parent, {}).get("status") in {"ok", "deduplicated"}
            else retryable_ids
        )
        generated_ids = _config_id_list(decision["generated_config_ids"], label)
        expected_generated = repair_ids_by_value[str(value)][
            repair_index : repair_index + 1
        ]
        require(
            expected_parent_ids == candidate_ids
            and expected_parent_ids
            and (
                missing_generation
                or all(
                    states.get(config_id, {}).get("status") in retryable_statuses
                    for config_id in tracked_ids
                )
            )
            and decision["selected_config_id"] == expected_parent_ids[0]
            and generated_ids == expected_generated
            and len(generated_ids) <= 1,
            f"{path}: {label}",
        )
        require(
            not missing_generation or bool(generated_ids),
            f"{path}: {label}",
        )
        if not generated_ids:
            require(
                decision_pass + 1 < len(measurement_states)
                and any(
                    measurement_states[decision_pass + 1]
                    .get(config_id, {})
                    .get("status")
                    == "deduplicated"
                    for config_id in candidate_ids
                ),
                f"{path}: {label}",
            )
        else:
            generated_id = generated_ids[0]
            require(
                generated_id not in tracked_ids
                and generated_id not in states
                and generated_id in phase_configs
                and structural_leaf(phase_configs[generated_id]) == expected_leaf
                and phase_configs[generated_id].get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                == value,
                f"{path}: {label}",
            )
        decision_passes.add(decision_pass)
        ordered_decision_passes.append(decision_pass)
        local_pass_counts[decision_pass] = local_pass_counts.get(decision_pass, 0) + 1
        neighbor_limits_by_pass.setdefault(decision_pass, []).append(
            decision["neighbor_generation_limit"]
        )
    require(
        all(count <= candidate_limit for count in local_pass_counts.values())
        and (
            not ordered_decision_passes
            or ordered_decision_passes
            == [
                ordered_decision_passes[0] + index // candidate_limit
                for index in range(len(ordered_decision_passes))
            ]
        )
        and all(
            limits
            == [
                (index + 1) * EXPECTED_NEIGHBOR_GENERATION_LIMIT // len(limits)
                - index * EXPECTED_NEIGHBOR_GENERATION_LIMIT // len(limits)
                for index in range(len(limits))
            ]
            for limits in neighbor_limits_by_pass.values()
        ),
        f"{path}: {label}",
    )
    return decision_passes


def validate_decision_results(
    path: Path,
    value: object,
    *,
    phase_configs: dict[str, dict[str, Any]],
    measurement_states: list[dict[str, dict[str, Any]]],
    label: str,
    value_key: bool = False,
    allow_failed: bool = False,
) -> tuple[list[dict[str, Any]], set[int]]:
    require(isinstance(value, list), f"{path}: {label}")
    expected_keys = {
        "config_id",
        "attempt_perf",
        "selection_perf",
        "status",
        "source_hash",
        "measurement_pass_index",
    }
    if value_key:
        expected_keys.add("value")
    parsed: list[dict[str, Any]] = []
    passes: set[int] = set()
    seen: set[object] = set()
    for raw_snapshot in value:
        require(
            isinstance(raw_snapshot, dict)
            and set(raw_snapshot) == expected_keys
            and raw_snapshot["config_id"] in phase_configs
            and (
                raw_snapshot["source_hash"] is None
                or valid_source_hash(raw_snapshot["source_hash"])
            )
            and (
                raw_snapshot["status"] not in {"ok", "deduplicated"}
                or valid_source_hash(raw_snapshot["source_hash"])
            )
            and terminal_measurement_is_valid(raw_snapshot)
            and (allow_failed or raw_snapshot["status"] in {"ok", "deduplicated"}),
            f"{path}: {label}",
        )
        identity: object = (
            (raw_snapshot["value"], raw_snapshot["config_id"])
            if value_key
            else raw_snapshot["config_id"]
        )
        require(identity not in seen, f"{path}: {label}")
        validate_measurement_snapshot(
            path,
            measurement_states,
            raw_snapshot,
            config_id=raw_snapshot["config_id"],
            label=label,
        )
        seen.add(identity)
        parsed.append(raw_snapshot)
        passes.add(raw_snapshot["measurement_pass_index"])
    require(len(passes) <= 1, f"{path}: {label}")
    return parsed, passes


def validate_clc_decision_evidence(
    path: Path,
    result: dict[str, Any],
    *,
    planned_values: list[int],
    selected_values: list[int],
    conditional_values: list[int],
    retained_values: list[int],
    witness_ids: dict[str, object],
    witness_repair_ids: dict[str, list[str]],
    conditional_ids: dict[str, object],
    conditional_repair_ids: dict[str, list[str]],
    phase_configs: dict[str, dict[str, Any]],
    measurement_states: list[dict[str, dict[str, Any]]],
    qualification_neighbor_limit: int,
) -> tuple[set[int], set[int], set[int], set[int]]:
    witness_candidates, witness_passes = validate_decision_results(
        path,
        result["witness_candidate_results"],
        phase_configs=phase_configs,
        measurement_states=measurement_states,
        label="invalid immutable CLC witness candidate snapshot",
        value_key=True,
        allow_failed=True,
    )
    witness_selection, selection_passes = validate_decision_results(
        path,
        result["witness_selection_results"],
        phase_configs=phase_configs,
        measurement_states=measurement_states,
        label="invalid immutable CLC witness decision",
        value_key=True,
    )
    expected_witness_selection: list[dict[str, Any]] = []
    for value in planned_values:
        value_candidates = [
            snapshot for snapshot in witness_candidates if snapshot["value"] == value
        ]
        require(
            value_candidates
            == sorted(
                value_candidates,
                key=lambda snapshot: (
                    snapshot["selection_perf"]
                    if snapshot["status"] in {"ok", "deduplicated"}
                    else math.inf,
                    snapshot["config_id"],
                ),
            ),
            f"{path}: invalid immutable CLC witness candidate snapshot",
        )
        successful = [
            snapshot
            for snapshot in value_candidates
            if snapshot["status"] in {"ok", "deduplicated"}
        ]
        if successful:
            expected_witness_selection.append(successful[0])
    expected_witness_selection.sort(
        key=itemgetter("selection_perf", "config_id", "value")
    )
    require(
        all(
            type(snapshot["value"]) is int and snapshot["value"] in planned_values
            for snapshot in witness_candidates
        )
        and all(
            {
                snapshot["config_id"]
                for snapshot in witness_candidates
                if snapshot["value"] == value
            }
            == {
                witness_ids[str(value)],
                *witness_repair_ids.get(str(value), []),
            }
            for value in planned_values
        )
        and witness_selection == expected_witness_selection
        and [snapshot["value"] for snapshot in witness_selection][
            : len(selected_values)
        ]
        == selected_values
        and [snapshot["config_id"] for snapshot in witness_selection][
            : len(selected_values)
        ]
        == _config_id_list(
            result["selected_config_ids"],
            f"{path}: selected CLC config IDs",
        ),
        f"{path}: invalid immutable CLC witness decision",
    )
    conditional_passes: set[int] = set()
    witness_selection_by_value = {
        snapshot["value"]: snapshot for snapshot in witness_selection
    }
    decisions = result["conditional_parent_decisions"]
    require(
        isinstance(decisions, list) and len(decisions) == len(conditional_values),
        f"{path}: invalid immutable CLC conditional-parent decision",
    )
    conditional_neighbor_limit = (
        max(qualification_neighbor_limit, len(conditional_values))
        if conditional_values
        else 0
    )
    check_equal(
        result.get("conditional_neighbor_generation_limit"),
        conditional_neighbor_limit,
        f"{path}: CLC conditional neighbor budget",
    )
    conditional_neighbor_limits = [
        (index + 1) * conditional_neighbor_limit // len(conditional_values)
        - index * conditional_neighbor_limit // len(conditional_values)
        for index in range(len(conditional_values))
    ]
    for value, decision, expected_neighbor_limit in zip(
        conditional_values,
        decisions,
        conditional_neighbor_limits,
        strict=True,
    ):
        require(
            isinstance(decision, dict)
            and set(decision)
            == {
                "value",
                "candidate_results",
                "selected_config_id",
                "generated_config_ids",
                "neighbor_generation_limit",
            }
            and decision["value"] == value,
            f"{path}: invalid immutable CLC conditional-parent decision",
        )
        candidates, passes = validate_decision_results(
            path,
            decision["candidate_results"],
            phase_configs=phase_configs,
            measurement_states=measurement_states,
            label="invalid immutable CLC conditional-parent decision",
        )
        generated = _config_id_list(
            decision["generated_config_ids"],
            f"{path}: conditional CLC generated IDs",
        )
        require(
            len(candidates) == 1
            and candidates[0]
            == {
                key: witness_selection_by_value[value][key]
                for key in (
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                )
            }
            and decision["selected_config_id"] == candidates[0]["config_id"]
            and generated == conditional_ids[str(value)]
            and type(decision["neighbor_generation_limit"]) is int
            and decision["neighbor_generation_limit"] == expected_neighbor_limit,
            f"{path}: invalid immutable CLC conditional-parent decision",
        )
        conditional_passes.update(passes)
    retained_passes: set[int] = set()
    value_decisions = result["retained_value_decisions"]
    require(
        isinstance(value_decisions, list)
        and len(value_decisions) == len(selected_values),
        f"{path}: invalid immutable CLC retained-value decision",
    )
    selected_by_value: dict[int, str] = {}
    for value, decision in zip(selected_values, value_decisions, strict=True):
        require(
            isinstance(decision, dict)
            and set(decision) == {"value", "candidate_results", "selected_config_id"}
            and decision["value"] == value,
            f"{path}: invalid immutable CLC retained-value decision",
        )
        candidates, passes = validate_decision_results(
            path,
            decision["candidate_results"],
            phase_configs=phase_configs,
            measurement_states=measurement_states,
            label="invalid immutable CLC retained-value decision",
            allow_failed=True,
        )
        successful = [
            candidate
            for candidate in candidates
            if candidate["status"] in {"ok", "deduplicated"}
        ]
        require(
            {candidate["config_id"] for candidate in candidates}
            == {
                witness_ids[str(value)],
                *witness_repair_ids.get(str(value), []),
                *conditional_ids.get(str(value), []),
                *conditional_repair_ids.get(str(value), []),
            }
            and candidates
            == sorted(
                candidates,
                key=lambda candidate: (
                    candidate["selection_perf"]
                    if candidate["status"] in {"ok", "deduplicated"}
                    else math.inf,
                    candidate["config_id"],
                ),
            )
            and successful
            and decision["selected_config_id"] == successful[0]["config_id"],
            f"{path}: invalid immutable CLC retained-value decision",
        )
        selected_by_value[value] = successful[0]["config_id"]
        retained_passes.update(passes)
    retained_ranking, ranking_passes = validate_decision_results(
        path,
        result["retained_ranking_results"],
        phase_configs=phase_configs,
        measurement_states=measurement_states,
        label="invalid immutable CLC retained ranking",
        value_key=True,
    )
    require(
        len(retained_ranking) == len(selected_values)
        and retained_ranking
        == sorted(
            retained_ranking,
            key=itemgetter("selection_perf", "config_id", "value"),
        )
        and all(
            snapshot["value"] in selected_by_value
            and snapshot["config_id"] == selected_by_value[snapshot["value"]]
            for snapshot in retained_ranking
        )
        and [snapshot["value"] for snapshot in retained_ranking][: len(retained_values)]
        == retained_values
        and [snapshot["config_id"] for snapshot in retained_ranking][
            : len(retained_values)
        ]
        == _config_id_list(
            result["retained_config_ids"],
            f"{path}: retained CLC config IDs",
        ),
        f"{path}: invalid immutable CLC retained ranking",
    )
    retained_passes.update(ranking_passes)
    depth_selection = result["depth_selection"]
    depth_candidates, depth_passes = validate_decision_results(
        path,
        depth_selection["candidate_results"],
        phase_configs=phase_configs,
        measurement_states=measurement_states,
        label="invalid immutable CLC depth decision",
    )
    representative_ids = [
        representative["config_id"]
        for representative in depth_selection["selected_representatives"]
        if isinstance(representative, dict)
        and set(representative) == {"config_id", "assigned_pipeline_lane"}
    ]
    require(
        len(representative_ids) == len(depth_selection["selected_representatives"])
        and set(representative_ids)
        <= {candidate["config_id"] for candidate in depth_candidates},
        f"{path}: invalid immutable CLC depth decision",
    )
    return (
        witness_passes | selection_passes,
        conditional_passes,
        retained_passes,
        depth_passes,
    )


def validate_phase_config_identity(
    path: Path,
    provenance: dict[str, Any],
    phase: dict[str, Any],
    lanes_by_leaf: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, dict[str, Any]]]]:
    """Validate the mandatory v22 canonical config and measurement evidence."""
    manifest = phase.get("config_manifest")
    initial_results = phase.get("initial_results")
    require(
        isinstance(manifest, dict) and isinstance(initial_results, list),
        f"{path}: incomplete canonical config identity evidence",
    )
    configs: dict[str, dict[str, Any]] = {}
    leaf_catalog = provenance["flash_structural_leaf_catalog"]
    for config_id, entry in manifest.items():
        require(
            isinstance(config_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
            and isinstance(entry, dict)
            and set(entry) == {"config"}
            and isinstance(entry["config"], dict)
            and canonical_sha256(entry["config"])[:16] == config_id
            and structural_leaf(entry["config"]) in leaf_catalog,
            f"{path}: invalid canonical config manifest entry {config_id!r}",
        )
        configs[config_id] = entry["config"]
    initial_ids = phase["initial_config_ids"]
    states_by_pass = validate_measurement_timeline(path, phase, configs)
    check_equal(
        [
            record.get("config_id") if isinstance(record, dict) else None
            for record in initial_results
        ],
        initial_ids,
        f"{path}: generation-zero result order",
    )
    successful_statuses = {"ok", "deduplicated"}
    known_statuses = successful_statuses | {
        "error",
        "timeout",
        "peer_compilation_fail",
        "accuracy_error",
        "source_rejected",
        "filtered",
    }
    for record in initial_results:
        require(
            isinstance(record, dict)
            and set(record)
            == {
                "config_id",
                "family",
                "compound_packet",
                "softmax_disc",
                "attempt_perf",
                "selection_perf",
                "status",
                "source_hash",
                "measurement_pass_index",
                "pipeline_lanes",
            }
            and record["config_id"] in configs
            and record["status"] in known_statuses
            and isinstance(record["pipeline_lanes"], list),
            f"{path}: invalid generation-zero result",
        )
        config = configs[record["config_id"]]
        validate_measurement_snapshot(
            path,
            states_by_pass,
            record,
            config_id=record["config_id"],
            label="inconsistent generation-zero measurement snapshot",
            expected_pass_index=0,
        )
        leaf = structural_leaf(config)
        check_equal(
            {
                "family": record["family"],
                "compound_packet": record["compound_packet"],
                "softmax_disc": record["softmax_disc"],
            },
            leaf,
            f"{path}: generation-zero structural leaf",
        )
        expected_lanes = lanes_by_leaf[canonical_json(leaf)]
        check_equal(
            record["pipeline_lanes"],
            [
                pipeline_lane_metric(lane)
                for lane in expected_lanes
                if config.get(lane[0]) == lane[1]
            ],
            f"{path}: generation-zero pipeline membership",
        )
        succeeded = record["status"] in successful_statuses
        require(
            (
                optional_positive_float(record["attempt_perf"])
                and record["attempt_perf"] is not None
                and optional_positive_float(record["selection_perf"])
                and record["selection_perf"] is not None
            )
            is succeeded,
            f"{path}: generation-zero status/performance mismatch",
        )

    referenced_ids = set(initial_ids) | set(phase["exact_space_config_ids"])
    schedule_anchor_results = phase.get("schedule_anchor_results")
    leaf_results = phase.get("leaf_results")
    compound_transfers = phase.get("compound_transfers")
    require(
        isinstance(leaf_results, list)
        and all(isinstance(result, dict) for result in leaf_results)
        and isinstance(compound_transfers, list)
        and all(isinstance(result, dict) for result in compound_transfers),
        f"{path}: invalid canonical config manifest membership",
    )
    require(
        isinstance(schedule_anchor_results, list)
        and all(isinstance(result, dict) for result in schedule_anchor_results),
        f"{path}: invalid schedule anchor manifest membership",
    )
    referenced_ids.update(result.get("config_id") for result in schedule_anchor_results)
    for result in leaf_results:
        qualified_results = result.get("qualified_results")
        require(
            isinstance(qualified_results, list)
            and all(isinstance(qualified, dict) for qualified in qualified_results),
            f"{path}: invalid canonical config manifest membership",
        )
        for qualified in qualified_results:
            referenced_ids.add(qualified.get("config_id"))
    for result in compound_transfers:
        transfers = result.get("transfers")
        require(
            isinstance(transfers, list)
            and all(isinstance(transfer, dict) for transfer in transfers),
            f"{path}: invalid canonical config manifest membership",
        )
        for transfer in transfers:
            referenced_ids.add(transfer.get("source_config_id"))
            referenced_ids.add(transfer.get("transferred_config_id"))
    family_probe_paths = phase.get("family_probe_paths")
    require(
        isinstance(family_probe_paths, list)
        and all(isinstance(probe_path, dict) for probe_path in family_probe_paths),
        f"{path}: invalid family probe manifest membership",
    )
    for probe_path in family_probe_paths:
        starting_config_id = probe_path.get("starting_config_id")
        require(
            isinstance(starting_config_id, str),
            f"{path}: invalid family probe manifest membership",
        )
        referenced_ids.add(starting_config_id)
        rounds = probe_path.get("rounds")
        require(
            isinstance(rounds, list)
            and all(isinstance(round_record, dict) for round_record in rounds),
            f"{path}: invalid family probe manifest membership",
        )
        for round_record in rounds:
            candidate_ids = round_record.get("candidate_ids")
            results = round_record.get("results")
            require(
                isinstance(candidate_ids, list)
                and all(isinstance(config_id, str) for config_id in candidate_ids)
                and isinstance(results, list)
                and all(
                    isinstance(result, dict)
                    and isinstance(result.get("config_id"), str)
                    for result in results
                ),
                f"{path}: invalid family probe manifest membership",
            )
            referenced_ids.update(candidate_ids)
            referenced_ids.update(result.get("config_id") for result in results)
    check_equal(set(configs), referenced_ids, f"{path}: canonical config manifest keys")
    return configs, states_by_pass


def validate_pipeline_parent_decisions(
    path: Path,
    leaf: dict[str, Any],
    result: dict[str, Any],
    expected_lanes: list[tuple[str, int]],
    phase_configs: dict[str, dict[str, Any]],
    measurement_states: list[dict[str, dict[str, Any]]],
    *,
    qualification_rounds: int,
    candidate_limit: int,
    baseline_pass_count: int,
    repair_pass_count: int,
    pass_offset: int = 0,
    prior_config_ids: set[str] | None = None,
) -> None:
    """Replay the immutable v22 pipeline-parent decisions pass by pass."""
    successful_statuses = {"ok", "deduplicated"}
    retryable_statuses = {"error", "timeout", "peer_compilation_fail"}
    known_statuses = (
        successful_statuses
        | retryable_statuses
        | {
            "accuracy_error",
            "source_rejected",
            "filtered",
        }
    )
    pipeline_pass_count = baseline_pass_count + repair_pass_count
    rounds = result["rounds"]
    require(
        isinstance(rounds, list) and len(rounds) == pipeline_pass_count,
        f"{path}: invalid pipeline parent-decision pass count",
    )
    lane_records = {
        (lane_result["key"], lane_result["value"]): lane_result
        for lane_result in result["pipeline_lanes"]
    }

    def decision_results(
        value: object,
        *,
        pass_index: int,
        allow_unmeasured: bool,
    ) -> list[dict[str, Any]]:
        label = "invalid immutable pipeline parent decision"
        require(isinstance(value, list), f"{path}: {label}")
        parsed: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for snapshot in value:
            require(
                isinstance(snapshot, dict)
                and set(snapshot)
                == {
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                }
                and isinstance(snapshot["config_id"], str)
                and snapshot["config_id"] in phase_configs
                and snapshot["config_id"] not in seen_ids
                and snapshot["status"] in known_statuses | {"unknown"},
                f"{path}: {label}",
            )
            config_id = snapshot["config_id"]
            status = snapshot["status"]
            if status in successful_statuses:
                require(
                    optional_positive_float(snapshot["attempt_perf"])
                    and snapshot["attempt_perf"] is not None
                    and optional_positive_float(snapshot["selection_perf"])
                    and snapshot["selection_perf"] is not None,
                    f"{path}: {label}",
                )
            else:
                require(
                    snapshot["attempt_perf"] is None
                    and snapshot["selection_perf"] is None
                    and (status != "unknown" or allow_unmeasured),
                    f"{path}: {label}",
                )
            validate_measurement_snapshot(
                path,
                measurement_states,
                snapshot,
                config_id=config_id,
                label=label,
                allow_unmeasured=allow_unmeasured,
                expected_pass_index=pass_index,
            )
            seen_ids.add(config_id)
            parsed.append(snapshot)
        require(
            parsed
            == sorted(
                parsed,
                key=lambda snapshot: (
                    (
                        snapshot["selection_perf"]
                        if snapshot["status"] in successful_statuses
                        else math.inf
                    ),
                    snapshot["config_id"],
                ),
            ),
            f"{path}: {label}",
        )
        return parsed

    witness_jobs = [("witness", lane, None) for lane in expected_lanes]
    conditional_jobs = [
        ("conditional", lane, None)
        for lane in expected_lanes
        if lane_records[lane]["conditional_required"] is True
    ]
    expected_jobs_by_pass: list[
        list[tuple[str, tuple[str, int] | None, int | None]]
    ] = [
        jobs[offset : offset + candidate_limit]
        for jobs in (witness_jobs, conditional_jobs)
        for offset in range(0, len(jobs), candidate_limit)
    ]
    if not expected_lanes:
        expected_jobs_by_pass.extend(
            [("ordinary", None, None)]
            for _ in range(0 if result["space_exhausted"] else qualification_rounds)
        )
    require(
        len(expected_jobs_by_pass) <= baseline_pass_count,
        f"{path}: inconsistent v22 pipeline parent-decision accounting",
    )
    expected_jobs_by_pass.extend(
        [] for _ in range(baseline_pass_count - len(expected_jobs_by_pass))
    )
    repair_jobs = [
        ("failure_repair", lane, repair_index)
        for repair_index in range(EXPECTED_QUALIFICATION_FAILURE_RETRIES)
        for lane in expected_lanes
        if len(lane_records[lane]["repair_parent_decisions"]) > repair_index
    ]
    repair_passes = [
        repair_jobs[offset : offset + candidate_limit]
        for offset in range(0, len(repair_jobs), candidate_limit)
    ]
    require(
        len(repair_passes) <= repair_pass_count,
        f"{path}: inconsistent v22 pipeline repair accounting",
    )
    expected_jobs_by_pass.extend(repair_passes)
    expected_jobs_by_pass.extend(
        [] for _ in range(repair_pass_count - len(repair_passes))
    )
    require(
        len(expected_jobs_by_pass) == pipeline_pass_count,
        f"{path}: inconsistent v22 pipeline parent-decision accounting",
    )

    for pass_index, expected_jobs in enumerate(expected_jobs_by_pass):
        repair_lanes = [
            lane
            for kind, lane, _repair_index in expected_jobs
            if kind == "failure_repair" and lane is not None
        ]
        if repair_lanes:
            repair_limits = [
                (index + 1) * EXPECTED_NEIGHBOR_GENERATION_LIMIT // len(repair_lanes)
                - index * EXPECTED_NEIGHBOR_GENERATION_LIMIT // len(repair_lanes)
                for index in range(len(repair_lanes))
            ]
            for lane, expected_limit in zip(repair_lanes, repair_limits, strict=True):
                check_equal(
                    lane_records[lane]["rounds"][pass_index][
                        "neighbor_generation_limit"
                    ],
                    expected_limit,
                    f"{path}: invalid per-lane pipeline repair budget",
                )

    available_ids = set(result["initial_config_ids"])
    available_ids.update(prior_config_ids or ())
    for pass_index, (pass_result, expected_jobs) in enumerate(
        zip(rounds, expected_jobs_by_pass, strict=True)
    ):
        require(
            isinstance(pass_result, dict)
            and set(pass_result)
            == {
                "candidate_config_ids",
                "neighbor_generation_limit",
                "ordinary_neighbor_generation_limit",
                "parent_decisions",
            },
            f"{path}: invalid v22 pipeline qualification round",
        )
        pass_ids = _config_id_list(
            pass_result["candidate_config_ids"],
            f"{path}: pipeline qualification round candidates",
        )
        raw_decisions = pass_result["parent_decisions"]
        require(
            isinstance(raw_decisions, list)
            and len(raw_decisions) == len(expected_jobs),
            f"{path}: invalid immutable pipeline parent decision",
        )
        emitted_ids: list[str] = []
        global_pass_index = pass_offset + pass_index
        state_at_decision = measurement_states[global_pass_index]
        for job_index, (raw_decision, expected_job) in enumerate(
            zip(raw_decisions, expected_jobs, strict=True)
        ):
            require(
                isinstance(raw_decision, dict),
                f"{path}: invalid immutable pipeline parent decision",
            )
            kind, lane, repair_index = expected_job
            expected_keys = {
                "job_index",
                "kind",
                "pipeline_lane",
                "selection_kind",
                "candidate_results",
                "selected_config_id",
                "generated_config_ids",
            }
            if kind == "failure_repair":
                expected_keys.add("repair_index")
            require(
                set(raw_decision) == expected_keys
                and raw_decision["job_index"] == job_index
                and raw_decision["kind"] == kind
                and raw_decision["pipeline_lane"]
                == (None if lane is None else pipeline_lane_metric(lane))
                and raw_decision.get("repair_index") == repair_index,
                f"{path}: invalid immutable pipeline parent decision",
            )
            allow_unmeasured = (
                kind == "witness"
                and raw_decision["selection_kind"] == "catalog_witness"
            )
            candidates = decision_results(
                raw_decision["candidate_results"],
                pass_index=global_pass_index,
                allow_unmeasured=allow_unmeasured,
            )
            candidate_ids = [snapshot["config_id"] for snapshot in candidates]
            generated_ids = _config_id_list(
                raw_decision["generated_config_ids"],
                f"{path}: invalid immutable pipeline parent decision",
            )
            selected_id = raw_decision["selected_config_id"]
            require(
                (selected_id is None or selected_id in phase_configs)
                and selected_id == (candidate_ids[0] if candidate_ids else None)
                and (bool(candidates) or not generated_ids)
                and all(
                    structural_leaf(phase_configs[config_id]) == leaf
                    and (
                        lane is None or phase_configs[config_id].get(lane[0]) == lane[1]
                    )
                    for config_id in [*candidate_ids, *generated_ids]
                ),
                f"{path}: invalid immutable pipeline parent decision",
            )

            scoped_available = {
                config_id
                for config_id in available_ids
                if structural_leaf(phase_configs[config_id]) == leaf
                and (lane is None or phase_configs[config_id].get(lane[0]) == lane[1])
            }
            expected_candidate_order: list[str] | None = None
            if kind == "failure_repair":
                assert lane is not None and repair_index is not None
                lane_record = lane_records[lane]
                tracked_ids = {
                    lane_record["witness_config_id"],
                    *lane_record["conditional_candidate_ids"],
                    *lane_record["repair_candidate_ids"][:repair_index],
                }
                require(
                    bool(tracked_ids)
                    and all(
                        state_at_decision.get(config_id, {}).get("status")
                        in retryable_statuses
                        for config_id in tracked_ids
                    ),
                    f"{path}: illegitimate pipeline failure repair",
                )
                expected_candidate_order = sorted(tracked_ids)
                expected_candidate_ids = tracked_ids
                expected_selection_kind = "ranked_failed_parent"
            elif kind == "ordinary":
                expected_candidate_ids = {
                    config_id
                    for config_id in scoped_available
                    if state_at_decision.get(config_id, {}).get("status")
                    in successful_statuses
                }
                expected_selection_kind = "ranked_parent"
            elif kind == "conditional":
                expected_candidate_ids = scoped_available
                expected_selection_kind = "ranked_parent"
            elif successful_scoped := {
                config_id
                for config_id in scoped_available
                if state_at_decision.get(config_id, {}).get("status")
                in successful_statuses
            }:
                expected_candidate_ids = successful_scoped
                expected_selection_kind = "ranked_existing"
            else:
                assert lane is not None
                witness_id = lane_records[lane]["witness_config_id"]
                expected_candidate_ids = {witness_id}
                expected_selection_kind = "catalog_witness"
            require(
                set(candidate_ids) == expected_candidate_ids
                and (
                    expected_candidate_order is None
                    or candidate_ids == expected_candidate_order
                )
                and raw_decision["selection_kind"] == expected_selection_kind,
                f"{path}: incomplete immutable pipeline parent decision",
            )
            if kind == "witness":
                require(
                    generated_ids in ([], [selected_id])
                    and not (
                        raw_decision["selection_kind"] == "ranked_existing"
                        and generated_ids
                    ),
                    f"{path}: invalid immutable pipeline parent decision",
                )
            elif kind == "conditional":
                assert lane is not None
                check_equal(
                    generated_ids,
                    lane_records[lane]["conditional_candidate_ids"],
                    f"{path}: immutable pipeline conditional decision",
                )
            elif kind == "failure_repair":
                assert lane is not None and repair_index is not None
                lane_decision = lane_records[lane]["repair_parent_decisions"][
                    repair_index
                ]
                check_equal(
                    lane_decision,
                    {
                        key: raw_decision[key]
                        for key in (
                            "repair_index",
                            "candidate_results",
                            "selected_config_id",
                            "generated_config_ids",
                        )
                    },
                    f"{path}: immutable pipeline repair decision",
                )
            emitted_ids.extend(generated_ids)
        check_equal(
            emitted_ids,
            pass_ids,
            f"{path}: immutable pipeline emitted candidate order",
        )
        available_ids.update(pass_ids)


def _validate_structural_qualification_phase_v22(
    path: Path, provenance: dict[str, Any], trial: dict[str, Any]
) -> dict[str, Any]:
    phase = trial.get("search_phase_metrics")
    require(isinstance(phase, dict), f"{path}: missing structural qualification phase")
    validate_flash_normalization_context(str(path), provenance, trial)
    check_equal(
        phase.get("phase"),
        "cute_flash_structural_qualification_v22",
        f"{path}: qualification phase name",
    )
    check_equal(
        phase.get("cute_flash_lane_policy_version"),
        EXPECTED_LANE_POLICY_VERSION,
        f"{path}: lane policy version",
    )
    check_equal(phase.get("completed"), True, f"{path}: qualification completion")
    check_equal(
        phase.get("budget_exhausted"),
        False,
        f"{path}: qualification budget exhaustion",
    )

    rounds = provenance.get("flash_structural_qualification_rounds")
    candidate_limit = provenance.get(
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round"
    )
    family_probe_generations = provenance.get(
        "flash_structural_family_probe_generations"
    )
    family_probe_candidates_per_path = provenance.get(
        "flash_structural_family_probe_candidates_per_path"
    )
    retained_per_leaf = provenance.get("flash_structural_retained_candidates_per_leaf")
    retained_family_cap = provenance.get("flash_structural_retained_family_cap")
    retained_family_limit = provenance.get("flash_structural_retained_family_limit")
    retained_family_slowdown_limit = provenance.get(
        "flash_structural_retained_family_slowdown_limit"
    )
    starting_path_limit = provenance.get("flash_structural_starting_path_limit")
    family_probe_path_limit = provenance.get("flash_structural_family_probe_path_limit")
    maximum_path_capacity = provenance.get("flash_structural_maximum_path_capacity")
    check_equal(rounds, 2, f"{path}: configured qualification rounds")
    check_equal(candidate_limit, 4, f"{path}: qualification candidate limit")
    check_equal(
        family_probe_generations,
        EXPECTED_FAMILY_PROBE_GENERATIONS,
        f"{path}: family probe generations",
    )
    check_equal(
        family_probe_candidates_per_path,
        EXPECTED_FAMILY_PROBE_CANDIDATES_PER_PATH,
        f"{path}: family probe candidates per path",
    )
    check_equal(retained_per_leaf, 2, f"{path}: retained candidates per leaf")
    check_equal(
        retained_family_cap,
        EXPECTED_FULL_FLASH_RETAINED_FAMILIES,
        f"{path}: retained family cap",
    )
    require(
        type(retained_family_limit) is int and retained_family_limit > 0,
        f"{path}: retained family limit",
    )
    check_equal(
        retained_family_slowdown_limit,
        2.0,
        f"{path}: retained family slowdown limit",
    )
    for key, expected in (
        ("qualification_rounds", rounds),
        ("pipeline_candidate_limit_per_leaf_per_round", candidate_limit),
        ("family_probe_generations", family_probe_generations),
        (
            "family_probe_candidates_per_path",
            family_probe_candidates_per_path,
        ),
        (
            "conditional_candidates_per_pipeline_lane",
            EXPECTED_CONDITIONAL_CANDIDATES_PER_PIPELINE_LANE,
        ),
        ("retained_candidates_per_leaf", retained_per_leaf),
        ("retained_family_cap", retained_family_cap),
        ("retained_family_limit", retained_family_limit),
        ("retained_family_slowdown_limit", retained_family_slowdown_limit),
        ("starting_path_limit", starting_path_limit),
        ("family_probe_path_limit", family_probe_path_limit),
        ("maximum_path_capacity", maximum_path_capacity),
        (
            "unrestricted_path_exhausts_generation_budget",
            provenance.get(
                "flash_structural_unrestricted_path_exhausts_generation_budget"
            ),
        ),
    ):
        check_equal(phase.get(key), expected, f"{path}: phase.{key}")
    check_equal(
        phase.get("qualification_failure_retries"),
        EXPECTED_QUALIFICATION_FAILURE_RETRIES,
        f"{path}: phase.qualification_failure_retries",
    )
    check_equal(
        phase.get("unrestricted_path_exhausts_generation_budget"),
        True,
        f"{path}: unrestricted path exhaustion requirement",
    )
    check_equal(
        phase.get("pipeline_qualification_keys"),
        list(FLASH_PIPELINE_QUALIFICATION_KEYS),
        f"{path}: pipeline qualification keys",
    )
    check_equal(
        phase.get("neighbor_generation_limit_per_leaf_per_round"),
        EXPECTED_NEIGHBOR_GENERATION_LIMIT,
        f"{path}: qualification neighbor generation limit",
    )

    initial_ids = _config_id_list(
        phase.get("initial_config_ids"), f"{path}: initial population"
    )
    exact_config_ids = exact_effective_search_space_ids(path, provenance)
    phase_exact_ids = _config_id_list(
        phase.get("exact_space_config_ids"), f"{path}: phase exact search space"
    )
    check_equal(
        phase.get("exact_space_enumerated"),
        exact_config_ids is not None,
        f"{path}: exact-space enumeration flag",
    )
    check_equal(
        phase_exact_ids,
        exact_config_ids or [],
        f"{path}: exact-space config IDs",
    )
    exact_ids_measured = exact_config_ids is not None and set(exact_config_ids) <= set(
        initial_ids
    )
    require(
        isinstance(phase.get("exact_space_exhausted"), bool)
        and (not phase["exact_space_exhausted"] or exact_ids_measured),
        f"{path}: exact-space exhaustion flag",
    )
    check_equal(
        phase.get("exact_space_raw_budget"),
        max(1, provenance["autotune_initial_population_size"], len(initial_ids)),
        f"{path}: exact-space raw enumeration budget",
    )
    expected_initial_count = (
        provenance["autotune_initial_population_size"]
        if exact_config_ids is None
        else len(exact_config_ids)
    )
    check_equal(
        phase.get("initial_config_count"),
        len(initial_ids),
        f"{path}: initial config ID count",
    )
    check_equal(
        len(initial_ids), expected_initial_count, f"{path}: initial population metric"
    )
    injected_ids = {
        canonical_sha256(item["config"])[:16]
        for item in provenance["flash_structural_coverage_design"][
            : provenance["flash_structural_injected_design_count"]
        ]
    }
    if exact_config_ids is not None:
        injected_ids.update(exact_config_ids)
    require(
        injected_ids <= set(initial_ids),
        f"{path}: measured initial population omits injected structural configs",
    )

    leaf_catalog = provenance.get("flash_structural_leaf_catalog")
    require(isinstance(leaf_catalog, list) and leaf_catalog, f"{path}: leaf catalog")
    require(
        all(
            isinstance(leaf, dict)
            and set(leaf) == {"family", "compound_packet", "softmax_disc"}
            and isinstance(leaf["family"], str)
            and isinstance(leaf["softmax_disc"], bool)
            and (
                leaf["compound_packet"] is None
                or leaf["compound_packet"] in COMPOUND_EXP2_PACKETS
            )
            for leaf in leaf_catalog
        )
        and len({canonical_json(leaf) for leaf in leaf_catalog}) == len(leaf_catalog),
        f"{path}: malformed or duplicate structural leaf catalog",
    )
    ordinary_leaves = [leaf for leaf in leaf_catalog if leaf["compound_packet"] is None]
    compound_leaves = [
        leaf for leaf in leaf_catalog if leaf["compound_packet"] is not None
    ]
    ordinary_leaf_keys = {
        (leaf["family"], leaf["softmax_disc"]) for leaf in ordinary_leaves
    }
    require(
        all(
            (leaf["family"], leaf["softmax_disc"]) in ordinary_leaf_keys
            for leaf in compound_leaves
        ),
        f"{path}: compound leaf has no ordinary family/protocol owner",
    )
    check_equal(
        retained_family_limit,
        expected_retained_family_limit(leaf_catalog, retained_family_cap),
        f"{path}: live-derived retained family limit",
    )
    check_equal(
        starting_path_limit,
        expected_starting_path_limit(
            leaf_catalog,
            retained_per_leaf=retained_per_leaf,
            retained_family_limit=retained_family_limit,
        ),
        f"{path}: live-derived starting path limit",
    )
    expected_probe_path_limit = expected_family_probe_path_limit(
        leaf_catalog, retained_family_cap, family_probe_generations
    )
    check_equal(
        family_probe_path_limit,
        expected_probe_path_limit,
        f"{path}: live-derived family probe path limit",
    )
    check_equal(
        maximum_path_capacity,
        max(starting_path_limit, expected_probe_path_limit),
        f"{path}: live-derived maximum path capacity",
    )
    family_probe_required = bool(
        expected_probe_path_limit and not phase["exact_space_exhausted"]
    )
    check_equal(
        phase.get("family_probe_required"),
        family_probe_required,
        f"{path}: family probe requirement",
    )
    check_equal(
        phase.get("family_probe_complete"),
        True,
        f"{path}: family probe completion",
    )
    check_equal(
        phase.get("family_probe_generations_started"),
        family_probe_generations if family_probe_required else 0,
        f"{path}: family probe generations started",
    )
    check_equal(
        phase.get("family_probe_generations_completed"),
        family_probe_generations if family_probe_required else 0,
        f"{path}: family probe generations completed",
    )
    family_probe_paths = phase.get("family_probe_paths")
    require(
        isinstance(family_probe_paths, list)
        and len(family_probe_paths)
        == (expected_probe_path_limit if family_probe_required else 0),
        f"{path}: family probe path catalog",
    )
    check_equal(phase.get("leaf_count"), len(leaf_catalog), f"{path}: leaf count")
    check_equal(
        phase.get("ordinary_leaf_count"),
        len(ordinary_leaves),
        f"{path}: ordinary leaf count",
    )
    check_equal(
        phase.get("compound_leaf_count"),
        len(compound_leaves),
        f"{path}: compound leaf count",
    )

    lanes_by_leaf = flash_pipeline_lane_catalog(path, provenance)
    phase_configs, measurement_states = validate_phase_config_identity(
        path, provenance, phase, lanes_by_leaf
    )
    qualification_pass_count = phase.get("qualification_passes_completed")
    require(
        type(qualification_pass_count) is int,
        f"{path}: qualification pass count",
    )
    pre_probe_pass = qualification_pass_count - (
        family_probe_generations if family_probe_required else 0
    )
    require(
        0 <= pre_probe_pass < len(measurement_states),
        f"{path}: pre-probe measurement boundary",
    )
    expected_clc_catalog = flash_clc_lane_catalog(path, provenance)
    active_values = provenance.get("flash_structural_coverage_active_values")
    require(isinstance(active_values, list), f"{path}: active structural values")
    family_choices = [
        item["value"]
        for item in active_values
        if item.get("key") == FLASH_PIPELINE_FAMILY_KEY
    ]
    ordinary_packet_choices = [
        item["value"]
        for item in active_values
        if item.get("key") == FLASH_EXP2_PACKET_KEY
        and item.get("value") not in COMPOUND_EXP2_PACKETS
    ]
    softmax_choices = [
        item["value"]
        for item in active_values
        if item.get("key") == FLASH_SOFTMAX_DISC_KEY
    ]
    if not softmax_choices:
        softmax_choices = list(
            dict.fromkeys(leaf["softmax_disc"] for leaf in ordinary_leaves)
        )
    require(
        family_choices and ordinary_packet_choices and softmax_choices,
        f"{path}: incomplete low-confound schedule axes",
    )
    schedule_protocol_order = [
        (family, packet, softmax_disc)
        for family in family_choices
        for packet in ordinary_packet_choices
        for softmax_disc in softmax_choices
    ]
    schedule_protocol_rank = {
        protocol: index for index, protocol in enumerate(schedule_protocol_order)
    }
    schedule_anchor_results = phase.get("schedule_anchor_results")
    require(
        phase.get("schedule_anchor_design_source")
        == "live family x ordinary packet x softmax protocol from fragment defaults"
        and isinstance(phase.get("schedule_anchor_pass_planned"), bool)
        and isinstance(phase.get("schedule_anchor_pass_started"), bool)
        and phase["schedule_anchor_pass_started"]
        is phase["schedule_anchor_pass_planned"]
        and phase.get("schedule_anchor_complete") is True
        and isinstance(schedule_anchor_results, list)
        and phase.get("schedule_anchor_count") == len(schedule_anchor_results)
        and len(ordinary_leaves)
        <= len(schedule_anchor_results)
        <= len(schedule_protocol_order),
        f"{path}: incomplete low-confound schedule-anchor design",
    )
    schedule_anchor_ids: list[str] = []
    schedule_anchor_leaves: list[dict[str, Any]] = []
    schedule_anchor_protocols: list[tuple[object, object, object]] = []
    schedule_anchor_pass_count = int(phase["schedule_anchor_pass_started"])
    for result in schedule_anchor_results:
        require(
            isinstance(result, dict)
            and set(result)
            == {
                "config_id",
                "family",
                "compound_packet",
                "softmax_disc",
                "attempt_perf",
                "selection_perf",
                "status",
                "source_hash",
                "measurement_pass_index",
            },
            f"{path}: malformed low-confound schedule-anchor result",
        )
        config_id = result["config_id"]
        config = phase_configs.get(config_id)
        leaf = {
            "family": result["family"],
            "compound_packet": result["compound_packet"],
            "softmax_disc": result["softmax_disc"],
        }
        protocol = (
            result["family"],
            config.get(FLASH_EXP2_PACKET_KEY) if isinstance(config, dict) else None,
            result["softmax_disc"],
        )
        require(
            isinstance(config_id, str)
            and isinstance(config, dict)
            and structural_leaf(config) == leaf
            and leaf in ordinary_leaves
            and protocol in schedule_protocol_rank
            and terminal_measurement_is_valid(result),
            f"{path}: invalid low-confound schedule-anchor result",
        )
        validate_measurement_snapshot(
            path,
            measurement_states,
            result,
            config_id=config_id,
            label="inconsistent low-confound schedule-anchor result",
            expected_pass_index=schedule_anchor_pass_count,
        )
        schedule_anchor_ids.append(config_id)
        schedule_anchor_leaves.append(leaf)
        schedule_anchor_protocols.append(protocol)
    require(
        len(schedule_anchor_ids) == len(set(schedule_anchor_ids))
        and len(schedule_anchor_protocols) == len(set(schedule_anchor_protocols)),
        f"{path}: duplicate low-confound schedule anchor",
    )
    require(
        [schedule_protocol_rank[item] for item in schedule_anchor_protocols]
        == sorted(schedule_protocol_rank[item] for item in schedule_anchor_protocols),
        f"{path}: low-confound schedule-anchor order",
    )
    check_equal(
        {(leaf["family"], leaf["softmax_disc"]) for leaf in schedule_anchor_leaves},
        {(leaf["family"], leaf["softmax_disc"]) for leaf in ordinary_leaves},
        f"{path}: low-confound schedule-anchor ordinary-leaf coverage",
    )
    check_equal(
        {protocol[1] for protocol in schedule_anchor_protocols},
        set(ordinary_packet_choices),
        f"{path}: low-confound schedule-anchor packet coverage",
    )
    check_equal(
        phase["schedule_anchor_pass_planned"],
        any(config_id not in initial_ids for config_id in schedule_anchor_ids),
        f"{path}: low-confound schedule-anchor pass requirement",
    )
    for leaf in compound_leaves:
        check_equal(
            lanes_by_leaf[canonical_json(leaf)],
            [],
            f"{path}: compound leaf pipeline lanes",
        )
    leaf_results = phase.get("leaf_results")
    require(isinstance(leaf_results, list), f"{path}: invalid ordinary leaf results")
    check_equal(
        [
            {
                "family": result.get("family"),
                "compound_packet": result.get("compound_packet"),
                "softmax_disc": result.get("softmax_disc"),
            }
            for result in leaf_results
            if isinstance(result, dict)
        ],
        ordinary_leaves,
        f"{path}: ordinary leaf result order",
    )

    baseline_pipeline_pass_count = 0
    leaf_pass_counts: dict[str, int] = {}
    for leaf, result in zip(ordinary_leaves, leaf_results, strict=True):
        require(isinstance(result, dict), f"{path}: invalid ordinary leaf result")
        lane_results = result.get("pipeline_lanes")
        require(isinstance(lane_results, list), f"{path}: invalid pipeline lanes")
        lane_count = len(lanes_by_leaf[canonical_json(leaf)])
        conditional_lane_count = sum(
            lane.get("conditional_required") is True
            for lane in lane_results
            if isinstance(lane, dict)
        )
        leaf_pass_count = max(
            0 if not lane_count and result.get("space_exhausted") is True else rounds,
            math.ceil(lane_count / candidate_limit)
            + math.ceil(
                conditional_lane_count
                * EXPECTED_CONDITIONAL_CANDIDATES_PER_PIPELINE_LANE
                / candidate_limit
            ),
        )
        leaf_pass_counts[canonical_json(leaf)] = leaf_pass_count
        baseline_pipeline_pass_count = max(
            baseline_pipeline_pass_count, leaf_pass_count
        )

    repair_pass_count = 0
    for leaf, result in zip(ordinary_leaves, leaf_results, strict=True):
        lane_results = result["pipeline_lanes"]
        require(
            all(
                isinstance(lane, dict)
                and isinstance(lane.get("repair_parent_decisions"), list)
                for lane in lane_results
            ),
            f"{path}: missing v22 repair provenance for leaf {leaf!r}",
        )
        repair_jobs = sum(len(lane["repair_parent_decisions"]) for lane in lane_results)
        require(
            repair_jobs <= len(lane_results) * EXPECTED_QUALIFICATION_FAILURE_RETRIES,
            f"{path}: too many repair jobs for leaf {leaf!r}",
        )
        repair_pass_count = max(
            repair_pass_count,
            math.ceil(repair_jobs / candidate_limit),
        )
    pipeline_pass_count = baseline_pipeline_pass_count + repair_pass_count

    known_statuses = {
        "ok",
        "deduplicated",
        "error",
        "timeout",
        "peer_compilation_fail",
        "accuracy_error",
        "source_rejected",
        "filtered",
    }
    for leaf, result in zip(ordinary_leaves, leaf_results, strict=True):
        require(
            isinstance(result, dict)
            and set(result)
            == {
                "family",
                "compound_packet",
                "softmax_disc",
                "initial_config_ids",
                "space_exhausted",
                "space_config_count",
                "ordinary_search_required",
                "rounds",
                "pipeline_lanes",
                "qualified_results",
                "retained_config_ids",
                "complete",
            },
            f"{path}: malformed ordinary qualification leaf {leaf!r}",
        )
        check_equal(result["complete"], True, f"{path}: ordinary leaf completion")
        leaf_initial_ids = _config_id_list(
            result["initial_config_ids"], f"{path}: {leaf!r} initial IDs"
        )
        require(
            isinstance(result["space_exhausted"], bool)
            and (
                result["space_config_count"] is None
                if exact_config_ids is None
                else type(result["space_config_count"]) is int
                and result["space_config_count"] >= 0
            ),
            f"{path}: invalid exact-space evidence for leaf {leaf!r}",
        )
        require(
            not result["space_exhausted"]
            or (exact_config_ids is not None and result["space_config_count"] > 0),
            f"{path}: leaf {leaf!r} claims exhaustion without exact-space proof",
        )
        check_equal(
            result["ordinary_search_required"],
            not lanes_by_leaf[canonical_json(leaf)] and not result["space_exhausted"],
            f"{path}: ordinary-search requirement for leaf {leaf!r}",
        )
        require(
            set(leaf_initial_ids) <= set(initial_ids),
            f"{path}: {leaf!r} initial IDs are outside the initial population",
        )
        leaf_rounds = result["rounds"]
        require(
            isinstance(leaf_rounds, list) and len(leaf_rounds) == pipeline_pass_count,
            f"{path}: invalid qualification pass count for leaf {leaf!r}",
        )
        round_ids: list[set[str]] = []
        seen_leaf_round_ids: set[str] = set()
        for pass_index, pass_result in enumerate(leaf_rounds):
            require(
                isinstance(pass_result, dict)
                and set(pass_result)
                == {
                    "candidate_config_ids",
                    "neighbor_generation_limit",
                    "ordinary_neighbor_generation_limit",
                    "parent_decisions",
                }
                and type(pass_result["neighbor_generation_limit"]) is int
                and type(pass_result["ordinary_neighbor_generation_limit"]) is int
                and 0
                <= pass_result["neighbor_generation_limit"]
                <= EXPECTED_NEIGHBOR_GENERATION_LIMIT,
                f"{path}: malformed leaf {leaf!r} pass {pass_index}",
            )
            pass_ids = set(
                _config_id_list(
                    pass_result["candidate_config_ids"],
                    f"{path}: {leaf!r} pass {pass_index}",
                )
            )
            require(
                len(pass_ids) <= candidate_limit
                and not (seen_leaf_round_ids & pass_ids),
                f"{path}: repeated qualification ID for leaf {leaf!r}",
            )
            seen_leaf_round_ids.update(pass_ids)
            round_ids.append(pass_ids)

        expected_lanes = lanes_by_leaf[canonical_json(leaf)]
        lane_results = result["pipeline_lanes"]
        require(isinstance(lane_results, list), f"{path}: invalid lanes for {leaf!r}")
        check_equal(
            [(lane.get("key"), lane.get("value")) for lane in lane_results],
            expected_lanes,
            f"{path}: phase lanes for exact structural leaf {leaf!r}",
        )
        lane_round_union = [set() for _ in range(pipeline_pass_count)]
        lane_neighbor_limits = [0 for _ in range(pipeline_pass_count)]
        for lane, lane_result in zip(expected_lanes, lane_results, strict=True):
            require(
                isinstance(lane_result, dict)
                and set(lane_result)
                == {
                    "key",
                    "value",
                    "initial_config_ids",
                    "space_exhausted",
                    "space_config_count",
                    "conditional_required",
                    "rounds",
                    "witness_attempted",
                    "witness_config_id",
                    "witness_succeeded",
                    "conditional_candidate_ids",
                    "successful_conditional_candidate_ids",
                    "repair_candidate_ids",
                    "successful_repair_candidate_ids",
                    "repair_parent_decisions",
                    "terminal_failure_exhausted",
                    "complete",
                },
                f"{path}: malformed pipeline lane {lane!r} for leaf {leaf!r}",
            )
            lane_initial_ids = _config_id_list(
                lane_result["initial_config_ids"],
                f"{path}: {leaf!r} lane {lane!r} initial IDs",
            )
            require(
                set(lane_initial_ids) <= set(leaf_initial_ids),
                f"{path}: invalid initial membership for lane {lane!r}",
            )
            check_equal(
                lane_result["witness_attempted"],
                True,
                f"{path}: lane {lane!r} witness attempted",
            )
            witness_id = lane_result["witness_config_id"]
            require(
                isinstance(witness_id, str)
                and re.fullmatch(r"[0-9a-f]{16}", witness_id) is not None,
                f"{path}: lane {lane!r} missing deterministic witness",
            )
            conditional_ids = _config_id_list(
                lane_result["conditional_candidate_ids"],
                f"{path}: lane {lane!r} conditional IDs",
            )
            require(
                isinstance(lane_result["space_exhausted"], bool)
                and (
                    lane_result["space_config_count"] is None
                    if exact_config_ids is None
                    else type(lane_result["space_config_count"]) is int
                    and lane_result["space_config_count"] >= 0
                )
                and isinstance(lane_result["conditional_required"], bool),
                f"{path}: invalid exact-space evidence for lane {lane!r}",
            )
            require(
                not lane_result["space_exhausted"]
                or (
                    exact_config_ids is not None
                    and lane_result["space_config_count"] > 0
                ),
                f"{path}: lane {lane!r} claims exhaustion without exact-space proof",
            )
            check_equal(
                lane_result["conditional_required"],
                not lane_result["space_exhausted"],
                f"{path}: conditional-search requirement for lane {lane!r}",
            )
            check_equal(
                len(conditional_ids),
                (
                    EXPECTED_CONDITIONAL_CANDIDATES_PER_PIPELINE_LANE
                    if lane_result["conditional_required"]
                    else 0
                ),
                f"{path}: lane {lane!r} conditional candidate count",
            )
            require(
                not (set(conditional_ids) & set(initial_ids)),
                f"{path}: lane {lane!r} conditional candidates are not novel",
            )
            successful_conditional_ids = _config_id_list(
                lane_result["successful_conditional_candidate_ids"],
                f"{path}: lane {lane!r} successful conditional IDs",
            )
            repair_ids = _config_id_list(
                lane_result["repair_candidate_ids"],
                f"{path}: lane {lane!r} repair IDs",
            )
            successful_repair_ids = _config_id_list(
                lane_result["successful_repair_candidate_ids"],
                f"{path}: lane {lane!r} successful repair IDs",
            )
            repair_decisions = lane_result["repair_parent_decisions"]
            require(
                set(successful_conditional_ids) <= set(conditional_ids)
                and set(successful_repair_ids) <= set(repair_ids)
                and not (set(repair_ids) & ({*initial_ids, *conditional_ids}))
                and isinstance(repair_decisions, list)
                and len(repair_decisions) <= EXPECTED_QUALIFICATION_FAILURE_RETRIES
                and isinstance(lane_result["terminal_failure_exhausted"], bool)
                and isinstance(lane_result["witness_succeeded"], bool),
                f"{path}: invalid successful evidence for lane {lane!r}",
            )
            generated_repair_ids: list[str] = []
            retryable_failure_ids = {witness_id, *conditional_ids}
            for repair_index, decision in enumerate(repair_decisions):
                require(
                    isinstance(decision, dict)
                    and set(decision)
                    == {
                        "repair_index",
                        "candidate_results",
                        "selected_config_id",
                        "generated_config_ids",
                    }
                    and decision["repair_index"] == repair_index
                    and isinstance(decision["candidate_results"], list)
                    and bool(decision["candidate_results"]),
                    f"{path}: malformed repair decision for lane {lane!r}",
                )
                snapshots = decision["candidate_results"]
                snapshot_ids: list[str] = []
                snapshot_passes: set[int] = set()
                for snapshot in snapshots:
                    require(
                        isinstance(snapshot, dict)
                        and set(snapshot)
                        == {
                            "config_id",
                            "attempt_perf",
                            "selection_perf",
                            "status",
                            "source_hash",
                            "measurement_pass_index",
                        }
                        and snapshot["config_id"] in retryable_failure_ids
                        and snapshot["attempt_perf"] is None
                        and snapshot["selection_perf"] is None
                        and snapshot["status"]
                        in {"error", "timeout", "peer_compilation_fail"},
                        f"{path}: invalid failed repair parent for lane {lane!r}",
                    )
                    validate_measurement_snapshot(
                        path,
                        measurement_states,
                        snapshot,
                        config_id=snapshot["config_id"],
                        label=f"invalid failed repair parent for lane {lane!r}",
                    )
                    snapshot_ids.append(snapshot["config_id"])
                    snapshot_passes.add(snapshot["measurement_pass_index"])
                check_equal(
                    snapshot_ids,
                    sorted(set(snapshot_ids)),
                    f"{path}: ranked repair parents for lane {lane!r}",
                )
                require(
                    len(snapshot_passes) == 1,
                    f"{path}: inconsistent repair decision pass for lane {lane!r}",
                )
                decision_pass = next(iter(snapshot_passes))
                tracked_ids = [witness_id, *conditional_ids, *generated_repair_ids]
                require(
                    snapshot_ids == sorted(tracked_ids)
                    and all(
                        measurement_states[decision_pass]
                        .get(config_id, {})
                        .get("status")
                        in {"error", "timeout", "peer_compilation_fail"}
                        for config_id in tracked_ids
                    ),
                    f"{path}: illegitimate repair decision for lane {lane!r}",
                )
                check_equal(
                    decision["selected_config_id"],
                    snapshot_ids[0],
                    f"{path}: selected repair parent for lane {lane!r}",
                )
                generated = _config_id_list(
                    decision["generated_config_ids"],
                    f"{path}: generated repairs for lane {lane!r}",
                )
                require(
                    len(generated) <= 1,
                    f"{path}: oversized repair decision for lane {lane!r}",
                )
                if not generated:
                    require(
                        decision_pass + 1 < len(measurement_states)
                        and any(
                            measurement_states[decision_pass + 1]
                            .get(config_id, {})
                            .get("status")
                            == "deduplicated"
                            for config_id in snapshot_ids
                        ),
                        f"{path}: unproven empty repair for lane {lane!r}",
                    )
                generated_repair_ids.extend(generated)
                retryable_failure_ids.update(generated)
            check_equal(
                generated_repair_ids,
                repair_ids,
                f"{path}: repair decision IDs for lane {lane!r}",
            )
            check_equal(
                lane_result["complete"],
                True,
                f"{path}: lane {lane!r} completion",
            )
            require(
                lane_result["witness_succeeded"]
                or successful_conditional_ids
                or successful_repair_ids,
                f"{path}: lane {lane!r} has no successful evidence",
            )
            lane_rounds = lane_result["rounds"]
            require(
                isinstance(lane_rounds, list)
                and len(lane_rounds) == pipeline_pass_count,
                f"{path}: invalid lane pass count for {lane!r}",
            )
            seen_lane_ids: set[str] = set()
            witness_pass: int | None = None
            conditional_passes: list[int] = []
            repair_passes: list[int] = []
            for pass_index, lane_pass in enumerate(lane_rounds):
                require(
                    isinstance(lane_pass, dict)
                    and set(lane_pass)
                    == {"candidate_config_ids", "neighbor_generation_limit"}
                    and type(lane_pass["neighbor_generation_limit"]) is int
                    and 0
                    <= lane_pass["neighbor_generation_limit"]
                    <= EXPECTED_NEIGHBOR_GENERATION_LIMIT,
                    f"{path}: malformed lane {lane!r} pass {pass_index}",
                )
                ids = set(
                    _config_id_list(
                        lane_pass["candidate_config_ids"],
                        f"{path}: lane {lane!r} pass {pass_index}",
                    )
                )
                require(
                    not (seen_lane_ids & ids),
                    f"{path}: repeated qualification ID for lane {lane!r}",
                )
                seen_lane_ids.update(ids)
                lane_round_union[pass_index].update(ids)
                lane_neighbor_limits[pass_index] += lane_pass[
                    "neighbor_generation_limit"
                ]
                if witness_id in ids:
                    witness_pass = pass_index
                if set(conditional_ids) & ids:
                    conditional_passes.append(pass_index)
                if set(repair_ids) & ids:
                    repair_passes.append(pass_index)
            require(
                seen_lane_ids == {witness_id, *conditional_ids, *repair_ids},
                f"{path}: lane {lane!r} evidence is absent from qualification passes",
            )
            require(
                witness_pass is not None
                and all(witness_pass < pass_index for pass_index in conditional_passes),
                f"{path}: lane {lane!r} conditional search precedes its witness",
            )
            require(
                all(
                    pass_index >= baseline_pipeline_pass_count
                    for pass_index in repair_passes
                ),
                f"{path}: lane {lane!r} repair precedes ordinary qualification",
            )
        for pass_index, ids in enumerate(round_ids):
            if expected_lanes:
                require(
                    ids <= lane_round_union[pass_index],
                    f"{path}: leaf {leaf!r} pass {pass_index} has unassigned candidates",
                )
                require(
                    lane_round_union[pass_index] - ids
                    <= set(measurement_states[schedule_anchor_pass_count + pass_index]),
                    f"{path}: leaf {leaf!r} pass {pass_index} omits generated candidates",
                )
                check_equal(
                    leaf_rounds[pass_index]["neighbor_generation_limit"],
                    lane_neighbor_limits[pass_index],
                    f"{path}: leaf {leaf!r} pass {pass_index} per-lane qualification budget",
                )
                check_equal(
                    leaf_rounds[pass_index]["ordinary_neighbor_generation_limit"],
                    0,
                    f"{path}: leaf {leaf!r} pass {pass_index} ordinary budget",
                )
            else:
                expected_ordinary_limit = (
                    EXPECTED_NEIGHBOR_GENERATION_LIMIT
                    if pass_index < leaf_pass_counts[canonical_json(leaf)]
                    else 0
                )
                check_equal(
                    leaf_rounds[pass_index]["ordinary_neighbor_generation_limit"],
                    expected_ordinary_limit,
                    f"{path}: leaf {leaf!r} pass {pass_index} ordinary budget",
                )
                check_equal(
                    leaf_rounds[pass_index]["neighbor_generation_limit"],
                    expected_ordinary_limit,
                    f"{path}: leaf {leaf!r} pass {pass_index} total budget",
                )

        validate_pipeline_parent_decisions(
            path,
            leaf,
            result,
            expected_lanes,
            phase_configs,
            measurement_states,
            qualification_rounds=rounds,
            candidate_limit=candidate_limit,
            baseline_pass_count=baseline_pipeline_pass_count,
            repair_pass_count=repair_pass_count,
            pass_offset=schedule_anchor_pass_count,
            prior_config_ids=set(schedule_anchor_ids),
        )

        qualified_results = result["qualified_results"]
        require(
            isinstance(qualified_results, list), f"{path}: invalid results for {leaf!r}"
        )
        qualified_ids: list[str] = []
        successful_ids: set[str] = set()
        for qualified in qualified_results:
            require(
                isinstance(qualified, dict)
                and set(qualified)
                == {
                    "config_id",
                    "status",
                    "attempt_perf",
                    "selection_perf",
                    "source_hash",
                    "measurement_pass_index",
                    "pipeline_lanes",
                }
                and isinstance(qualified["config_id"], str)
                and re.fullmatch(r"[0-9a-f]{16}", qualified["config_id"]) is not None
                and qualified["status"] in known_statuses
                and optional_positive_float(qualified["attempt_perf"])
                and optional_positive_float(qualified["selection_perf"])
                and isinstance(qualified["pipeline_lanes"], list),
                f"{path}: malformed qualified result for leaf {leaf!r}",
            )
            memberships = [
                (membership.get("key"), membership.get("value"))
                for membership in qualified["pipeline_lanes"]
                if isinstance(membership, dict) and set(membership) == {"key", "value"}
            ]
            check_equal(
                memberships,
                [lane for lane in expected_lanes if lane in memberships],
                f"{path}: qualified pipeline membership order",
            )
            require(
                len(memberships) == len(qualified["pipeline_lanes"])
                and len(memberships) == len(set(memberships)),
                f"{path}: malformed qualified pipeline memberships",
            )
            config_id = qualified["config_id"]
            require(
                config_id in phase_configs,
                f"{path}: qualified result missing canonical config",
            )
            validate_measurement_snapshot(
                path,
                measurement_states,
                qualified,
                config_id=config_id,
                label=f"invalid qualified measurement for leaf {leaf!r}",
                expected_pass_index=phase["qualification_passes_completed"],
            )
            qualified_ids.append(config_id)
            if qualified["status"] in {"ok", "deduplicated"}:
                require(
                    qualified["attempt_perf"] is not None
                    and qualified["selection_perf"] is not None,
                    f"{path}: successful qualified result lacks performance",
                )
                successful_ids.add(config_id)
            else:
                require(
                    qualified["attempt_perf"] is None
                    and qualified["selection_perf"] is None,
                    f"{path}: failed qualified result records performance",
                )
        require(
            len(qualified_ids) == len(set(qualified_ids))
            and set(leaf_initial_ids) <= set(qualified_ids),
            f"{path}: inconsistent qualified membership for leaf {leaf!r}",
        )
        retained_ids = _config_id_list(
            result["retained_config_ids"], f"{path}: {leaf!r} retained IDs"
        )
        require(
            len(retained_ids) <= retained_per_leaf
            and set(retained_ids) <= successful_ids,
            f"{path}: retained candidates are not successful for leaf {leaf!r}",
        )

    active_clc_values = {
        active["value"]
        for active in provenance.get("flash_structural_coverage_active_values", [])
        if isinstance(active, dict)
        and active.get("key") == FLASH_CLC_HEADS_PER_BATCH_KEY
        and type(active.get("value")) is int
        and active["value"] > 0
    }
    expected_clc_leaves = [
        {
            "family": record["family"],
            "compound_packet": record["compound_packet"],
            "softmax_disc": record["softmax_disc"],
        }
        for record in expected_clc_catalog
    ]
    clc_families = phase.get("clc_families")
    require(isinstance(clc_families, list), f"{path}: invalid CLC family results")
    check_equal(
        [
            {
                "family": result.get("family"),
                "compound_packet": None,
                "softmax_disc": result.get("softmax_disc"),
            }
            for result in clc_families
            if isinstance(result, dict)
        ],
        expected_clc_leaves,
        f"{path}: CLC family result order",
    )
    max_clc_planned = 0
    clc_witness_repair_decision_passes: set[int] = set()
    clc_conditional_repair_decision_passes: set[int] = set()
    clc_combination_snapshot_passes: set[int] = set()
    clc_witness_snapshot_passes: set[int] = set()
    clc_conditional_parent_passes: set[int] = set()
    clc_retained_snapshot_passes: set[int] = set()
    clc_depth_snapshot_passes: set[int] = set()
    for leaf, expected_clc, result in zip(
        expected_clc_leaves, expected_clc_catalog, clc_families, strict=True
    ):
        require(
            isinstance(result, dict)
            and set(result)
            == {
                "family",
                "softmax_disc",
                "space_exhausted",
                "legal_values",
                "search_values",
                "anchor_values",
                "refinement_values",
                "planned_values",
                "attempted_values",
                "witness_config_ids",
                "witness_repair_candidate_ids",
                "witness_repair_parent_decisions",
                "value_space_exhausted",
                "witness_candidate_results",
                "witness_selection_results",
                "selected_values",
                "selected_config_ids",
                "conditional_values",
                "conditional_neighbor_generation_limit",
                "conditional_parent_decisions",
                "conditional_repair_candidate_ids",
                "conditional_repair_parent_decisions",
                "retained_values",
                "retained_config_ids",
                "retained_value_decisions",
                "retained_ranking_results",
                "conditional_candidate_ids",
                "combination_required",
                "depth_selection",
                "combination_candidate_ids",
                "combination_depth_config_ids",
                "combination_divisor_values",
                "combination_cells",
                "combination_projection_complete",
                "successful_combination_depth_config_ids",
                "successful_combination_divisor_values",
                "combination_row_coverage_complete",
                "combination_column_coverage_complete",
                "combination_failure_statuses_allowed",
                "complete",
            },
            f"{path}: malformed CLC family result for {leaf!r}",
        )
        require(
            all(
                key in result
                for key in (
                    "depth_selection",
                    "combination_depth_config_ids",
                    "combination_divisor_values",
                    "combination_cells",
                    "combination_projection_complete",
                    "successful_combination_depth_config_ids",
                    "successful_combination_divisor_values",
                    "combination_row_coverage_complete",
                    "combination_column_coverage_complete",
                )
            ),
            f"{path}: missing v22 CLC combination evidence for {leaf!r}",
        )
        legal = _positive_int_list(result["legal_values"], f"{path}: CLC legal values")
        search = _positive_int_list(
            result["search_values"], f"{path}: CLC search values"
        )
        anchors = _positive_int_list(result["anchor_values"], f"{path}: CLC anchors")
        refinements = _positive_int_list(
            result["refinement_values"], f"{path}: CLC refinements"
        )
        planned = _positive_int_list(result["planned_values"], f"{path}: CLC planned")
        attempted = _positive_int_list(
            result["attempted_values"], f"{path}: CLC attempted"
        )
        selected = _positive_int_list(
            result["selected_values"], f"{path}: CLC selected"
        )
        conditional = _positive_int_list(
            result["conditional_values"], f"{path}: CLC conditional"
        )
        retained = _positive_int_list(
            result["retained_values"], f"{path}: CLC retained"
        )
        for field in (
            "family",
            "legal_values",
            "search_values",
            "anchor_values",
            "refinement_values",
            "planned_values",
            "witness_config_ids",
        ):
            check_equal(
                result[field], expected_clc[field], f"{path}: CLC catalog {field}"
            )
        check_equal(set(anchors), active_clc_values, f"{path}: CLC active anchors")
        check_equal(search, legal, f"{path}: CLC full search reachability")
        require(
            set(anchors) <= set(search)
            and set(refinements) <= set(search)
            and not (set(anchors) & set(refinements)),
            f"{path}: invalid CLC anchor/refinement partition",
        )
        check_equal(planned, [*anchors, *refinements], f"{path}: CLC planned values")
        check_equal(set(planned), set(legal), f"{path}: CLC full qualification")
        check_equal(attempted, planned, f"{path}: CLC attempted values")
        check_equal(
            len(selected),
            len(planned),
            f"{path}: CLC selected value count",
        )
        require(
            set(selected) == set(planned),
            f"{path}: invalid CLC selected values",
        )
        value_space_exhausted = result["value_space_exhausted"]
        require(
            isinstance(value_space_exhausted, dict)
            and set(value_space_exhausted) == {str(value) for value in planned}
            and all(
                type(exhausted) is bool for exhausted in value_space_exhausted.values()
            ),
            f"{path}: invalid CLC value-space exhaustion map",
        )
        check_equal(
            conditional,
            [value for value in selected if not value_space_exhausted[str(value)]],
            f"{path}: CLC conditional values",
        )
        require(
            isinstance(result["space_exhausted"], bool)
            and isinstance(result["combination_required"], bool)
            and result["combination_required"] is (not result["space_exhausted"]),
            f"{path}: invalid CLC exhaustion policy",
        )
        require(
            set(conditional) <= set(selected),
            f"{path}: invalid CLC conditional values",
        )
        check_equal(
            len(retained),
            len(planned),
            f"{path}: CLC retained value count",
        )
        require(set(retained) == set(planned), f"{path}: invalid retained CLC values")
        witnesses = result["witness_config_ids"]
        require(
            isinstance(witnesses, dict)
            and set(witnesses) == {str(value) for value in planned}
            and all(
                isinstance(config_id, str)
                and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
                for config_id in witnesses.values()
            ),
            f"{path}: invalid CLC witness IDs",
        )
        witness_repair_ids = _repair_id_mapping(
            result["witness_repair_candidate_ids"],
            planned,
            f"{path}: invalid immutable CLC witness repair",
        )
        conditional_ids = result["conditional_candidate_ids"]
        require(
            isinstance(conditional_ids, dict)
            and set(conditional_ids) == {str(value) for value in conditional},
            f"{path}: invalid CLC conditional candidate map",
        )
        for value in conditional:
            require(
                len(
                    _config_id_list(
                        conditional_ids[str(value)],
                        f"{path}: CLC {value} conditional IDs",
                    )
                )
                <= 1,
                f"{path}: CLC {value} conditional candidate count",
            )
        conditional_repair_ids = _repair_id_mapping(
            result["conditional_repair_candidate_ids"],
            conditional,
            f"{path}: invalid immutable CLC conditional repair",
        )
        require(
            all(
                conditional_ids[str(value)] or conditional_repair_ids.get(str(value))
                for value in conditional
            ),
            f"{path}: empty CLC conditional generation lacks repair telemetry",
        )
        clc_witness_repair_decision_passes.update(
            validate_failure_repair_decisions(
                path,
                result["witness_repair_parent_decisions"],
                witness_repair_ids,
                witnesses,
                planned,
                expected_kind="witness_failure_repair",
                expected_leaf=leaf,
                candidate_limit=candidate_limit,
                phase_configs=phase_configs,
                measurement_states=measurement_states,
                label="invalid immutable CLC witness repair",
            )
        )
        clc_conditional_repair_decision_passes.update(
            validate_failure_repair_decisions(
                path,
                result["conditional_repair_parent_decisions"],
                conditional_repair_ids,
                conditional_ids,
                conditional,
                expected_kind="conditional_failure_repair",
                expected_leaf=leaf,
                candidate_limit=candidate_limit,
                phase_configs=phase_configs,
                measurement_states=measurement_states,
                label="invalid immutable CLC conditional repair",
                missing_generation_parent_ids_by_value={
                    str(value): config_id
                    for value, config_id in zip(
                        selected,
                        _config_id_list(
                            result["selected_config_ids"],
                            f"{path}: selected CLC config IDs",
                        ),
                        strict=True,
                    )
                },
            )
        )
        (
            witness_snapshot_passes,
            conditional_parent_passes,
            retained_snapshot_passes,
            depth_snapshot_passes,
        ) = validate_clc_decision_evidence(
            path,
            result,
            planned_values=planned,
            selected_values=selected,
            conditional_values=conditional,
            retained_values=retained,
            witness_ids=witnesses,
            witness_repair_ids=witness_repair_ids,
            conditional_ids=conditional_ids,
            conditional_repair_ids=conditional_repair_ids,
            phase_configs=phase_configs,
            measurement_states=measurement_states,
            qualification_neighbor_limit=EXPECTED_NEIGHBOR_GENERATION_LIMIT,
        )
        clc_witness_snapshot_passes.update(witness_snapshot_passes)
        clc_conditional_parent_passes.update(conditional_parent_passes)
        clc_retained_snapshot_passes.update(retained_snapshot_passes)
        clc_depth_snapshot_passes.update(depth_snapshot_passes)
        combination_ids = _config_id_list(
            result["combination_candidate_ids"],
            f"{path}: CLC combination candidate IDs",
        )
        combination_depth_ids = _config_id_list(
            result["combination_depth_config_ids"],
            f"{path}: CLC combination depth IDs",
        )
        combination_divisors = _positive_int_list(
            result["combination_divisor_values"],
            f"{path}: CLC combination divisors",
        )
        depth_selection = result["depth_selection"]
        require(
            isinstance(depth_selection, dict)
            and set(depth_selection)
            == {"candidate_results", "selected_representatives"}
            and isinstance(depth_selection["selected_representatives"], list),
            f"{path}: invalid CLC depth selection",
        )
        check_equal(
            combination_depth_ids,
            [
                representative.get("config_id")
                for representative in depth_selection["selected_representatives"]
                if isinstance(representative, dict)
            ],
            f"{path}: CLC combination depth axes",
        )
        check_equal(
            combination_divisors,
            retained if result["combination_required"] else [],
            f"{path}: CLC combination divisor axes",
        )
        cells = result["combination_cells"]
        require(
            isinstance(cells, list)
            and len(cells) == len(combination_depth_ids) * len(combination_divisors),
            f"{path}: incomplete CLC combination cell ledger",
        )
        check_equal(
            [
                (cell.get("depth_config_id"), cell.get("divisor_value"))
                for cell in cells
                if isinstance(cell, dict)
            ],
            [
                (depth_id, divisor)
                for depth_id in combination_depth_ids
                for divisor in combination_divisors
            ],
            f"{path}: CLC combination cell axes",
        )
        for cell in cells:
            require(
                isinstance(cell, dict)
                and set(cell)
                == {
                    "depth_config_id",
                    "divisor_value",
                    "projected_config_id",
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                },
                f"{path}: malformed CLC combination cell",
            )
            projected_id = cell["projected_config_id"]
            if projected_id is None:
                require(
                    cell["config_id"] is None
                    and cell["attempt_perf"] is None
                    and cell["selection_perf"] is None
                    and cell["status"] == "projection_rejected"
                    and cell["source_hash"] is None
                    and cell["measurement_pass_index"] is None,
                    f"{path}: invalid rejected CLC projection",
                )
                continue
            require(
                isinstance(projected_id, str)
                and re.fullmatch(r"[0-9a-f]{16}", projected_id) is not None
                and cell["config_id"] == projected_id
                and cell["status"] in known_statuses
                and optional_positive_float(cell["attempt_perf"])
                and optional_positive_float(cell["selection_perf"]),
                f"{path}: invalid measured CLC projection",
            )
            succeeded = cell["status"] in {"ok", "deduplicated"}
            require(
                (
                    cell["attempt_perf"] is not None
                    and cell["selection_perf"] is not None
                )
                is succeeded,
                f"{path}: CLC projection status/performance mismatch",
            )
            validate_measurement_snapshot(
                path,
                measurement_states,
                cell,
                config_id=projected_id,
                label="inconsistent measured CLC projection",
            )
            clc_combination_snapshot_passes.add(cell["measurement_pass_index"])
        coverage = clc_combination_coverage(result)
        check_equal(
            combination_ids,
            coverage["combination_candidate_ids"],
            f"{path}: CLC effective combination IDs",
        )
        check_equal(
            result["successful_combination_depth_config_ids"],
            coverage["successful_combination_depth_config_ids"],
            f"{path}: successful CLC depth coverage",
        )
        check_equal(
            result["successful_combination_divisor_values"],
            coverage["successful_combination_divisor_values"],
            f"{path}: successful CLC divisor coverage",
        )
        projection_complete = coverage["combination_projection_complete"]
        row_coverage_complete = coverage["combination_row_coverage_complete"]
        column_coverage_complete = coverage["combination_column_coverage_complete"]
        check_equal(
            result["combination_projection_complete"],
            projection_complete,
            f"{path}: CLC projection completion",
        )
        check_equal(
            result["combination_row_coverage_complete"],
            row_coverage_complete,
            f"{path}: CLC row coverage",
        )
        check_equal(
            result["combination_column_coverage_complete"],
            column_coverage_complete,
            f"{path}: CLC column coverage",
        )
        require(
            (
                0
                < len(combination_ids)
                <= len(combination_depth_ids) * len(combination_divisors)
                and bool(combination_depth_ids)
                and bool(combination_divisors)
                and projection_complete
                and row_coverage_complete
                and column_coverage_complete
                if result["combination_required"]
                else not combination_ids
                and not combination_depth_ids
                and not combination_divisors
                and not cells
                and projection_complete
                and row_coverage_complete
                and column_coverage_complete
            ),
            f"{path}: invalid CLC combination candidate count",
        )
        check_equal(
            result["combination_failure_statuses_allowed"],
            all(
                terminal_measurement_is_valid(cell, allow_projection_rejected=True)
                for cell in cells
            ),
            f"{path}: CLC combination terminal statuses",
        )
        check_equal(
            result["combination_failure_statuses_allowed"],
            True,
            f"{path}: CLC combination terminal statuses",
        )
        check_equal(result["complete"], True, f"{path}: CLC family completion")
        max_clc_planned = max(max_clc_planned, len(planned))

    check_equal(
        phase["exact_space_exhausted"],
        exact_ids_measured
        and all(result["space_exhausted"] for result in clc_families),
        f"{path}: hierarchical exact-space exhaustion flag",
    )

    compound_transfers = phase.get("compound_transfers")
    require(isinstance(compound_transfers, list), f"{path}: invalid compound transfers")
    check_equal(
        phase.get("compound_catalog_complete"),
        True,
        f"{path}: compound catalog completion",
    )
    check_equal(
        phase.get("compound_catalog_errors"),
        [],
        f"{path}: compound catalog errors",
    )
    check_equal(
        [
            {
                "family": result.get("family"),
                "compound_packet": result.get("compound_packet"),
                "softmax_disc": result.get("softmax_disc"),
            }
            for result in compound_transfers
            if isinstance(result, dict)
        ],
        compound_leaves,
        f"{path}: compound transfer order",
    )
    compound_backfill_pass_count = 0
    compound_source_snapshot_passes: set[int] = set()
    qualified_compound_phase_ids: set[str] = set()
    clc_results_by_family = {result["family"]: result for result in clc_families}
    for leaf, result in zip(compound_leaves, compound_transfers, strict=True):
        require(
            isinstance(result, dict)
            and set(result)
            == {
                "family",
                "compound_packet",
                "softmax_disc",
                "limit",
                "transfer_target_count",
                "transfer_count",
                "primary_transfer_config_ids",
                "backfill_rounds",
                "successful_transfer_config_ids",
                "qualified_transfer_config_ids",
                "failure_statuses_allowed",
                "source_selection",
                "transfers",
                "complete",
            }
            and result["limit"] == retained_per_leaf
            and type(result["transfer_target_count"]) is int
            and 0 < result["transfer_target_count"] <= result["limit"]
            and type(result["transfer_count"]) is int
            and 0
            < result["transfer_count"]
            <= result["limit"] * (1 + EXPECTED_QUALIFICATION_FAILURE_RETRIES)
            and isinstance(result["transfers"], list)
            and len(result["transfers"]) == result["transfer_count"]
            and isinstance(result["backfill_rounds"], list)
            and len(result["backfill_rounds"]) <= EXPECTED_QUALIFICATION_FAILURE_RETRIES
            and result["failure_statuses_allowed"] is True
            and result["complete"] is True,
            f"{path}: malformed or incomplete compound transfer for {leaf!r}",
        )
        primary_ids = _config_id_list(
            result["primary_transfer_config_ids"],
            f"{path}: primary compound transfer IDs",
        )
        successful_transfer_ids = _config_id_list(
            result["successful_transfer_config_ids"],
            f"{path}: successful compound transfer IDs",
        )
        qualified_transfer_ids = _config_id_list(
            result["qualified_transfer_config_ids"],
            f"{path}: qualified compound transfer IDs",
        )
        require(
            len(primary_ids) == result["transfer_target_count"]
            and len(successful_transfer_ids) >= result["transfer_target_count"]
            and len(qualified_transfer_ids) == result["transfer_target_count"]
            and qualified_transfer_ids
            == successful_transfer_ids[: result["transfer_target_count"]],
            f"{path}: incomplete qualified compound transfer set",
        )
        qualified_compound_phase_ids.update(qualified_transfer_ids)
        source_selection = result["source_selection"]
        require(
            isinstance(source_selection, dict)
            and set(source_selection)
            == {
                "candidate_results",
                "combination_prefix_count",
                "attempted_config_ids",
                "selected_config_ids",
            }
            and isinstance(source_selection["candidate_results"], list)
            and type(source_selection["combination_prefix_count"]) is int,
            f"{path}: invalid immutable compound source decision",
        )
        source_candidate_ids: list[str] = []
        source_snapshot_passes: set[int] = set()
        for snapshot in source_selection["candidate_results"]:
            require(
                isinstance(snapshot, dict)
                and set(snapshot)
                == {
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                }
                and snapshot["config_id"] in phase_configs
                and snapshot["status"] in {"ok", "deduplicated"},
                f"{path}: invalid immutable compound source decision",
            )
            validate_measurement_snapshot(
                path,
                measurement_states,
                snapshot,
                config_id=snapshot["config_id"],
                label="invalid immutable compound source decision",
            )
            source_candidate_ids.append(snapshot["config_id"])
            source_snapshot_passes.add(snapshot["measurement_pass_index"])
        require(
            len(source_snapshot_passes) == 1,
            f"{path}: invalid immutable compound source decision",
        )
        source_snapshot_pass = next(iter(source_snapshot_passes))
        compound_source_snapshot_passes.add(source_snapshot_pass)
        attempted_source_ids = _config_id_list(
            source_selection["attempted_config_ids"],
            f"{path}: attempted compound source IDs",
        )
        selected_source_ids = _config_id_list(
            source_selection["selected_config_ids"],
            f"{path}: selected compound source IDs",
        )
        clc_result = clc_results_by_family.get(leaf["family"])
        expected_combination_source_ids = (
            []
            if clc_result is None
            else [
                config_id
                for config_id in clc_result["combination_candidate_ids"]
                if any(
                    cell["config_id"] == config_id
                    and cell["status"] in {"ok", "deduplicated"}
                    for cell in clc_result["combination_cells"]
                )
            ]
        )
        combination_prefix_count = source_selection["combination_prefix_count"]
        require(
            combination_prefix_count == len(expected_combination_source_ids)
            and set(source_candidate_ids[:combination_prefix_count])
            == set(expected_combination_source_ids)
            and attempted_source_ids
            == source_candidate_ids[: len(attempted_source_ids)]
            and len(selected_source_ids) == result["transfer_count"],
            f"{path}: invalid immutable compound source decision",
        )
        for segment in (
            source_selection["candidate_results"][:combination_prefix_count],
            source_selection["candidate_results"][combination_prefix_count:],
        ):
            require(
                segment
                == sorted(
                    segment,
                    key=itemgetter("selection_perf", "config_id"),
                ),
                f"{path}: invalid immutable compound source decision",
            )
        expected_source_ids = {
            config_id
            for config_id, state in measurement_states[source_snapshot_pass].items()
            if state["status"] in {"ok", "deduplicated"}
            and structural_leaf(phase_configs[config_id])
            == {
                "family": leaf["family"],
                "compound_packet": None,
                "softmax_disc": leaf["softmax_disc"],
            }
        }
        require(
            set(source_candidate_ids) == expected_source_ids,
            f"{path}: incomplete immutable compound source decision",
        )
        sources: list[str] = []
        targets: list[str] = []
        for transfer in result["transfers"]:
            require(
                isinstance(transfer, dict)
                and set(transfer)
                == {
                    "source_config_id",
                    "source_config",
                    "transferred_config_id",
                    "projected_config",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                    "projection_overrides",
                    "projected_config_id",
                    "preserved_pipeline_values",
                },
                f"{path}: malformed compound transfer entry",
            )
            source = transfer["source_config_id"]
            target = transfer["transferred_config_id"]
            source_config = transfer["source_config"]
            projected_config = transfer["projected_config"]
            require(
                isinstance(source, str)
                and isinstance(target, str)
                and isinstance(source_config, dict)
                and isinstance(projected_config, dict)
                and re.fullmatch(r"[0-9a-f]{16}", source) is not None
                and re.fullmatch(r"[0-9a-f]{16}", target) is not None
                and source != target,
                f"{path}: invalid compound transfer IDs",
            )
            check_equal(
                canonical_sha256(source_config)[:16],
                source,
                f"{path}: compound source config ID",
            )
            check_equal(
                canonical_sha256(projected_config)[:16],
                target,
                f"{path}: compound projected config ID",
            )
            require(
                structural_leaf(source_config)
                == {
                    "family": leaf["family"],
                    "compound_packet": None,
                    "softmax_disc": leaf["softmax_disc"],
                }
                and structural_leaf(projected_config) == leaf,
                f"{path}: compound snapshots change the wrong structural leaf",
            )
            check_equal(
                transfer["projected_config_id"],
                target,
                f"{path}: compound projected config ID",
            )
            check_equal(
                transfer["projection_overrides"],
                {FLASH_EXP2_PACKET_KEY: leaf["compound_packet"]},
                f"{path}: compound projection overrides",
            )
            validate_measurement_snapshot(
                path,
                measurement_states,
                transfer,
                config_id=target,
                label="inconsistent compound transfer measurement snapshot",
                expected_pass_index=pre_probe_pass,
            )
            succeeded = transfer["status"] in {"ok", "deduplicated"}
            require(
                terminal_measurement_is_valid(transfer)
                and succeeded
                is (
                    transfer["attempt_perf"] is not None
                    and transfer["selection_perf"] is not None
                )
                and isinstance(transfer["preserved_pipeline_values"], dict)
                and set(transfer["preserved_pipeline_values"])
                <= set(FLASH_PIPELINE_QUALIFICATION_KEYS),
                f"{path}: invalid measured compound transfer evidence",
            )
            check_equal(
                transfer["preserved_pipeline_values"],
                {
                    key: source_config[key]
                    for key in FLASH_PIPELINE_QUALIFICATION_KEYS
                    if key in source_config
                },
                f"{path}: compound preserved pipeline values",
            )
            sources.append(source)
            targets.append(target)
        require(
            len(sources) == len(set(sources)) and len(targets) == len(set(targets)),
            f"{path}: duplicate compound transfer IDs",
        )
        check_equal(
            selected_source_ids,
            sources,
            f"{path}: immutable compound source selection",
        )
        require(
            all(source_id in attempted_source_ids for source_id in selected_source_ids)
            and [
                source_id
                for source_id in attempted_source_ids
                if source_id in set(selected_source_ids)
            ]
            == selected_source_ids,
            f"{path}: invalid immutable compound source decision",
        )
        require(
            (
                len(selected_source_ids) == result["limit"]
                and attempted_source_ids
                and attempted_source_ids[-1] == selected_source_ids[-1]
            )
            or (
                len(selected_source_ids) < result["limit"]
                and attempted_source_ids == source_candidate_ids
            ),
            f"{path}: invalid immutable compound source decision stop condition",
        )
        check_equal(
            primary_ids,
            targets[: result["transfer_target_count"]],
            f"{path}: primary compound transfer IDs",
        )
        check_equal(
            successful_transfer_ids,
            [
                transfer["transferred_config_id"]
                for transfer in result["transfers"]
                if transfer["status"] in {"ok", "deduplicated"}
            ],
            f"{path}: successful compound transfer IDs",
        )
        check_equal(
            result["failure_statuses_allowed"],
            all(
                terminal_measurement_is_valid(transfer)
                for transfer in result["transfers"]
            ),
            f"{path}: compound terminal statuses",
        )
        raw_backfills = result["backfill_rounds"]
        compound_backfill_pass_count = max(
            compound_backfill_pass_count, len(raw_backfills)
        )
        consumed_transfer_count = len(primary_ids)
        backfill_attempted_source_ids: list[str] = []
        for backfill_index, backfill in enumerate(raw_backfills):
            require(
                isinstance(backfill, dict)
                and set(backfill)
                == {
                    "repair_index",
                    "required_successes",
                    "failed_transfer_config_ids",
                    "attempted_source_config_ids",
                    "generated_config_ids",
                }
                and backfill["repair_index"] == backfill_index
                and type(backfill["required_successes"]) is int,
                f"{path}: invalid compound transfer backfill",
            )
            failed_ids = _config_id_list(
                backfill["failed_transfer_config_ids"],
                f"{path}: failed compound transfer IDs",
            )
            backfill_source_ids = _config_id_list(
                backfill["attempted_source_config_ids"],
                f"{path}: backfill source IDs",
            )
            generated_ids = _config_id_list(
                backfill["generated_config_ids"],
                f"{path}: generated compound transfer IDs",
            )
            decision_pass = source_snapshot_pass + 1 + backfill_index
            require(
                decision_pass + 1 < len(measurement_states),
                f"{path}: invalid compound transfer backfill",
            )
            attempted_targets = targets[:consumed_transfer_count]
            decision_states = measurement_states[decision_pass]
            expected_failed_ids = [
                config_id
                for config_id in attempted_targets
                if decision_states.get(config_id, {}).get("status")
                not in {"ok", "deduplicated"}
            ]
            successful_count = sum(
                decision_states.get(config_id, {}).get("status")
                in {"ok", "deduplicated"}
                for config_id in attempted_targets
            )
            missing = result["transfer_target_count"] - successful_count
            require(
                missing > 0
                and backfill["required_successes"] == missing
                and failed_ids == expected_failed_ids
                and failed_ids
                and all(
                    decision_states.get(config_id, {}).get("status")
                    in {"error", "timeout", "peer_compilation_fail"}
                    for config_id in failed_ids
                )
                and generated_ids
                == targets[
                    consumed_transfer_count : consumed_transfer_count
                    + len(generated_ids)
                ]
                and len(generated_ids) <= missing,
                f"{path}: illegitimate compound transfer backfill",
            )
            consumed_transfer_count += len(generated_ids)
            backfill_attempted_source_ids.extend(backfill_source_ids)
            completed_states = measurement_states[decision_pass + 1]
            require(
                sum(
                    completed_states.get(config_id, {}).get("status")
                    in {"ok", "deduplicated"}
                    for config_id in targets[:consumed_transfer_count]
                )
                >= result["transfer_target_count"],
                f"{path}: incomplete compound transfer backfill",
            )
        require(
            consumed_transfer_count == len(targets),
            f"{path}: invalid compound transfer backfill",
        )
        primary_attempted_end = (
            attempted_source_ids.index(selected_source_ids[len(primary_ids) - 1]) + 1
        )
        require(
            attempted_source_ids[primary_attempted_end:]
            == backfill_attempted_source_ids,
            f"{path}: invalid compound transfer backfill source suffix",
        )

    clc_witness_passes = int(max_clc_planned > 0)
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
    compound_primary_passes = int(bool(compound_leaves))
    expected_passes_without_probe = (
        schedule_anchor_pass_count
        + pipeline_pass_count
        + clc_witness_passes
        + clc_witness_repair_passes
        + clc_conditional_passes
        + clc_conditional_repair_passes
        + clc_combination_passes
        + compound_primary_passes
        + compound_backfill_pass_count
    )
    expected_passes = expected_passes_without_probe + (
        family_probe_generations if family_probe_required else 0
    )
    check_equal(
        pre_probe_pass,
        expected_passes_without_probe,
        f"{path}: pre-probe pass accounting",
    )
    for key in (
        "qualification_passes_planned",
        "qualification_passes_started",
        "qualification_passes_completed",
        "qualification_rounds_started",
        "qualification_rounds_completed",
    ):
        check_equal(phase.get(key), expected_passes, f"{path}: phase.{key}")
    witness_repair_start = (
        schedule_anchor_pass_count + pipeline_pass_count + clc_witness_passes
    )
    conditional_repair_start = (
        witness_repair_start + clc_witness_repair_passes + clc_conditional_passes
    )
    check_equal(
        clc_witness_repair_decision_passes,
        set(
            range(
                witness_repair_start, witness_repair_start + clc_witness_repair_passes
            )
        ),
        f"{path}: CLC witness repair pass accounting",
    )
    check_equal(
        clc_conditional_repair_decision_passes,
        set(
            range(
                conditional_repair_start,
                conditional_repair_start + clc_conditional_repair_passes,
            )
        ),
        f"{path}: CLC conditional repair pass accounting",
    )
    post_witness_pass = witness_repair_start + clc_witness_repair_passes
    post_conditional_pass = conditional_repair_start + clc_conditional_repair_passes
    check_equal(
        clc_witness_snapshot_passes,
        {post_witness_pass} if max_clc_planned > 0 else set(),
        f"{path}: CLC witness snapshot pass",
    )
    check_equal(
        clc_conditional_parent_passes,
        {post_witness_pass} if clc_conditional_passes else set(),
        f"{path}: CLC conditional-parent snapshot pass",
    )
    check_equal(
        clc_retained_snapshot_passes,
        {post_conditional_pass} if max_clc_planned > 0 else set(),
        f"{path}: CLC retained snapshot pass",
    )
    check_equal(
        clc_depth_snapshot_passes,
        {post_conditional_pass} if clc_combination_passes else set(),
        f"{path}: CLC depth snapshot pass",
    )
    compound_source_pass = post_conditional_pass + clc_combination_passes
    require(
        all(
            pass_index == post_conditional_pass + 1
            for pass_index in clc_combination_snapshot_passes
        ),
        f"{path}: inconsistent CLC combination snapshot pass",
    )
    check_equal(
        compound_source_snapshot_passes,
        {compound_source_pass} if compound_leaves else set(),
        f"{path}: immutable compound source snapshot pass",
    )
    family_probe_candidate_ids: set[str] = set()
    if family_probe_required:
        pre_probe_states = measurement_states[pre_probe_pass]
        ordinary_by_family: dict[str, list[str]] = {}
        for config_id, state in pre_probe_states.items():
            leaf = structural_leaf(phase_configs[config_id])
            if state["status"] in {"ok", "deduplicated"} and leaf is not None:
                if leaf["compound_packet"] is None:
                    ordinary_by_family.setdefault(leaf["family"], []).append(config_id)

        def probe_rank(config_id: str) -> tuple[float, str]:
            state = pre_probe_states[config_id]
            selection_perf = state.get("selection_perf")
            require(
                isinstance(selection_perf, (int, float))
                and not isinstance(selection_perf, bool)
                and math.isfinite(selection_perf)
                and selection_perf > 0,
                f"{path}: invalid family probe starting performance",
            )
            return float(selection_perf), config_id

        expected_probe_starts: list[tuple[str, dict[str, Any], bool]] = []
        family_starts = [
            min(config_ids, key=probe_rank)
            for config_ids in ordinary_by_family.values()
        ]
        for config_id in sorted(
            family_starts,
            key=lambda item: (
                probe_rank(item),
                structural_leaf(phase_configs[item])["family"],
            ),
        ):
            leaf = structural_leaf(phase_configs[config_id])
            assert leaf is not None
            expected_probe_starts.append((config_id, leaf, False))
        for leaf in compound_leaves:
            leaf_ids = [
                config_id
                for config_id in qualified_compound_phase_ids
                if structural_leaf(phase_configs[config_id]) == leaf
                and pre_probe_states.get(config_id, {}).get("status")
                in {"ok", "deduplicated"}
            ]
            require(
                bool(leaf_ids),
                f"{path}: compound leaf has no measured family probe start",
            )
            config_id = min(leaf_ids, key=probe_rank)
            expected_probe_starts.append((config_id, leaf, False))
        expected_probe_starts[len(family_starts) :] = sorted(
            expected_probe_starts[len(family_starts) :],
            key=lambda item: (
                probe_rank(item[0]),
                item[1]["family"],
                item[1]["compound_packet"],
                item[1]["softmax_disc"],
            ),
        )
        probe_eligible_ids = {
            config_id
            for config_ids in ordinary_by_family.values()
            for config_id in config_ids
        } | {
            config_id
            for config_id in qualified_compound_phase_ids
            if pre_probe_states.get(config_id, {}).get("status")
            in {"ok", "deduplicated"}
        }
        require(probe_eligible_ids, f"{path}: empty family probe population")
        global_start = min(probe_eligible_ids, key=probe_rank)
        global_leaf = structural_leaf(phase_configs[global_start])
        assert global_leaf is not None
        expected_probe_starts.append((global_start, global_leaf, True))
        check_equal(
            len(expected_probe_starts),
            expected_probe_path_limit,
            f"{path}: family probe start count",
        )

        for _path_index, (raw_path, expected_start) in enumerate(
            zip(family_probe_paths, expected_probe_starts, strict=True)
        ):
            require(
                isinstance(raw_path, dict)
                and set(raw_path)
                == {
                    "family",
                    "compound_packet",
                    "softmax_disc",
                    "starting_config_id",
                    "unrestricted",
                    "rounds",
                },
                f"{path}: malformed family probe path",
            )
            start_id, start_leaf, unrestricted = expected_start
            check_equal(
                {
                    "family": raw_path["family"],
                    "compound_packet": raw_path["compound_packet"],
                    "softmax_disc": raw_path["softmax_disc"],
                },
                start_leaf,
                f"{path}: family probe starting leaf",
            )
            check_equal(
                raw_path["starting_config_id"],
                start_id,
                f"{path}: family probe starting config",
            )
            check_equal(
                raw_path["unrestricted"],
                unrestricted,
                f"{path}: family probe path scope",
            )
            rounds_record = raw_path["rounds"]
            require(
                isinstance(rounds_record, list)
                and len(rounds_record) == family_probe_generations,
                f"{path}: family probe rounds",
            )
            for generation_index, round_record in enumerate(rounds_record, start=1):
                expected_pass = expected_passes_without_probe + generation_index
                require(
                    isinstance(round_record, dict)
                    and set(round_record)
                    == {
                        "probe_generation",
                        "measurement_pass_index",
                        "candidate_ids",
                        "results",
                    }
                    and round_record["probe_generation"] == generation_index
                    and round_record["measurement_pass_index"] == expected_pass,
                    f"{path}: malformed family probe round",
                )
                candidate_ids = _config_id_list(
                    round_record["candidate_ids"],
                    f"{path}: family probe candidate IDs",
                )
                results = round_record["results"]
                require(
                    isinstance(results, list)
                    and len(candidate_ids) <= family_probe_candidates_per_path - 1
                    and not (set(candidate_ids) & family_probe_candidate_ids)
                    and not (set(candidate_ids) & set(pre_probe_states)),
                    f"{path}: invalid family probe candidate set",
                )
                result_ids: list[str] = []
                for result in results:
                    require(
                        isinstance(result, dict)
                        and set(result)
                        == {
                            "config_id",
                            "attempt_perf",
                            "selection_perf",
                            "status",
                            "source_hash",
                            "measurement_pass_index",
                        }
                        and result["config_id"] in candidate_ids
                        and terminal_measurement_is_valid(result),
                        f"{path}: invalid family probe result",
                    )
                    config_id = result["config_id"]
                    validate_measurement_snapshot(
                        path,
                        measurement_states,
                        result,
                        config_id=config_id,
                        label="invalid family probe result",
                        expected_pass_index=expected_pass,
                    )
                    candidate_leaf = structural_leaf(phase_configs[config_id])
                    require(
                        candidate_leaf is not None
                        and (unrestricted or candidate_leaf == start_leaf),
                        f"{path}: family probe changed a constrained leaf",
                    )
                    if candidate_leaf["compound_packet"] is not None and result[
                        "status"
                    ] in {"ok", "deduplicated"}:
                        qualified_compound_phase_ids.add(config_id)
                    result_ids.append(config_id)
                check_equal(
                    result_ids,
                    candidate_ids,
                    f"{path}: family probe result order",
                )
                family_probe_candidate_ids.update(candidate_ids)
    scheduled_ids_by_completion_pass = [set() for _ in range(expected_passes + 1)]
    if schedule_anchor_pass_count:
        scheduled_ids_by_completion_pass[1].update(
            set(schedule_anchor_ids) - set(initial_ids)
        )
    for leaf_result in leaf_results:
        for pass_index, round_result in enumerate(
            leaf_result["rounds"], start=schedule_anchor_pass_count + 1
        ):
            scheduled_ids_by_completion_pass[pass_index].update(
                _config_id_list(
                    round_result["candidate_config_ids"],
                    f"{path}: qualification round candidates",
                )
            )
    witness_completion_pass = (
        schedule_anchor_pass_count + pipeline_pass_count + clc_witness_passes
    )
    for clc_result in clc_families:
        scheduled_ids_by_completion_pass[witness_completion_pass].update(
            clc_result["witness_config_ids"].values()
        )
        for decision_key in (
            "witness_repair_parent_decisions",
            "conditional_repair_parent_decisions",
        ):
            for decision in clc_result[decision_key]:
                decision_pass = decision["candidate_results"][0][
                    "measurement_pass_index"
                ]
                scheduled_ids_by_completion_pass[decision_pass + 1].update(
                    decision["generated_config_ids"]
                )
        for decision in clc_result["conditional_parent_decisions"]:
            scheduled_ids_by_completion_pass[post_witness_pass + 1].update(
                decision["generated_config_ids"]
            )
        if clc_result["combination_required"]:
            scheduled_ids_by_completion_pass[post_conditional_pass + 1].update(
                clc_result["combination_candidate_ids"]
            )
    for transfer_result in compound_transfers:
        scheduled_ids_by_completion_pass[compound_source_pass + 1].update(
            transfer_result["primary_transfer_config_ids"]
        )
        for backfill_index, backfill in enumerate(transfer_result["backfill_rounds"]):
            scheduled_ids_by_completion_pass[
                compound_source_pass + 2 + backfill_index
            ].update(backfill["generated_config_ids"])
    if family_probe_required:
        for raw_path in family_probe_paths:
            for round_record in raw_path["rounds"]:
                scheduled_ids_by_completion_pass[
                    round_record["measurement_pass_index"]
                ].update(round_record["candidate_ids"])
    validate_measurement_introductions(
        path,
        phase,
        measurement_states,
        scheduled_ids_by_completion_pass,
    )
    candidate_count = phase.get("candidate_count")
    leaves_with_candidates = phase.get("leaves_with_candidates")
    require(
        type(candidate_count) is int
        and candidate_count >= 0
        and type(leaves_with_candidates) is int
        and 0 <= leaves_with_candidates <= len(leaf_catalog),
        f"{path}: invalid qualification candidate counts",
    )

    retained_families = phase.get("retained_families")
    require(isinstance(retained_families, list), f"{path}: retained families")
    starting_ids: list[str] = []
    starting_identities: set[str] = set()
    unrestricted_count = 0
    retained_names: set[str] = set()
    for retained in retained_families:
        require(
            isinstance(retained, dict)
            and set(retained)
            == {
                "family",
                "score",
                "score_compound_packet",
                "score_softmax_disc",
                "parent_promoted",
                "starting_paths",
            }
            and isinstance(retained["family"], str)
            and optional_positive_float(retained["score"])
            and isinstance(retained["parent_promoted"], bool)
            and isinstance(retained["starting_paths"], list)
            and retained["starting_paths"],
            f"{path}: malformed retained structural family",
        )
        family = retained["family"]
        require(family not in retained_names, f"{path}: duplicate retained family")
        retained_names.add(family)
        for starting_path in retained["starting_paths"]:
            require(
                isinstance(starting_path, dict)
                and set(starting_path)
                == {
                    "family",
                    "compound_packet",
                    "softmax_disc",
                    "config_id",
                    "unrestricted",
                    "pipeline_lane",
                }
                and starting_path["family"] == family
                and {
                    "family": family,
                    "compound_packet": starting_path["compound_packet"],
                    "softmax_disc": starting_path["softmax_disc"],
                }
                in leaf_catalog
                and isinstance(starting_path["unrestricted"], bool),
                f"{path}: malformed retained starting path",
            )
            config_id = starting_path["config_id"]
            require(
                isinstance(config_id, str)
                and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None,
                f"{path}: invalid retained starting-path ID",
            )
            require(
                starting_path["compound_packet"] is None
                or config_id in qualified_compound_phase_ids,
                f"{path}: retained compound path is outside the qualified subset",
            )
            lane = starting_path["pipeline_lane"]
            require(
                lane is None
                or (
                    isinstance(lane, dict)
                    and set(lane) == {"key", "value"}
                    and (lane["key"], lane["value"])
                    in lanes_by_leaf[
                        canonical_json(
                            {
                                "family": family,
                                "compound_packet": starting_path["compound_packet"],
                                "softmax_disc": starting_path["softmax_disc"],
                            }
                        )
                    ]
                ),
                f"{path}: invalid retained starting-path pipeline lane",
            )
            starting_ids.append(config_id)
            starting_identity = canonical_json(starting_path)
            require(
                starting_identity not in starting_identities,
                f"{path}: duplicate retained starting path",
            )
            starting_identities.add(starting_identity)
            unrestricted_count += int(starting_path["unrestricted"])
    require(
        len(starting_ids) <= starting_path_limit
        and unrestricted_count == int(bool(starting_ids)),
        f"{path}: invalid retained starting paths",
    )
    check_equal(
        phase.get("retained_path_count"),
        len(starting_ids),
        f"{path}: retained structural path count",
    )
    require(
        sum(retained["parent_promoted"] for retained in retained_families)
        <= retained_family_limit,
        f"{path}: too many parent-promoted structural families",
    )
    return phase


def validate_structural_qualification_phase(
    path: Path, provenance: dict[str, Any], trial: dict[str, Any]
) -> dict[str, Any]:
    return _validate_structural_qualification_phase_v22(path, provenance, trial)


def _reconcile_structural_qualification_phase_v22(
    path: Path,
    provenance: dict[str, Any],
    phase: dict[str, Any],
    attempt_by_config: dict[str, dict[str, Any]],
    metadata_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trials = provenance.get("trials")
    require(
        isinstance(trials, list) and len(trials) == 1 and isinstance(trials[0], dict),
        f"{path}: expected one autotune trial for projection replay",
    )
    validate_flash_normalization_context(str(path), provenance, trials[0])
    initial_ids = set(phase["initial_config_ids"])
    successful_statuses = {"ok", "deduplicated"}
    retryable_statuses = {"error", "timeout", "peer_compilation_fail"}
    passes = phase["qualification_passes_completed"]
    lanes_by_leaf = flash_pipeline_lane_catalog(path, provenance)
    _phase_configs, measurement_states = validate_phase_config_identity(
        path, provenance, phase, lanes_by_leaf
    )
    phase_manifest = phase.get("config_manifest")
    if isinstance(phase_manifest, dict):
        for config_id, entry in phase_manifest.items():
            check_equal(
                metadata_configs.get(config_id),
                entry["config"],
                f"{path}: manifest/sidecar config {config_id}",
            )
    successful_phase_ids: set[str] = set()
    successful_candidate_ids: set[str] = set()
    selection_perf_by_id: dict[str, float] = {}
    explicit_candidate_ids: set[str] = set()
    leaves_with_candidate_keys: set[str] = set()
    qualified_leaves: list[dict[str, Any]] = []
    successful_results_by_id: dict[str, dict[str, Any]] = {}
    final_measurement_by_id: dict[str, dict[str, Any]] = {}
    for pass_record in phase["measurement_timeline"]:
        for update in pass_record["updates"]:
            final_measurement_by_id[update["config_id"]] = update
    schedule_anchor_ids = {
        result["config_id"] for result in phase["schedule_anchor_results"]
    }
    invalidations = isolated_rebenchmark_invalidations(phase)

    family_probe_candidate_ids: set[str] = set()
    family_probe_candidate_ids_by_leaf: dict[str, set[str]] = {}
    family_probe_successful_compound_members: dict[str, list[dict[str, Any]]] = {}
    family_probe_required = phase["family_probe_required"]
    pre_probe_pass = passes - (
        phase["family_probe_generations"] if family_probe_required else 0
    )
    parent_score_config_ids = {
        config_id
        for config_id, state in measurement_states[pre_probe_pass].items()
        if state["status"] in successful_statuses
        and (leaf := structural_leaf(metadata_configs[config_id])) is not None
        and leaf["compound_packet"] is None
    }

    def final_phase_attempt(
        config_id: str, attempt: dict[str, Any]
    ) -> tuple[str, float | None]:
        invalidation = invalidations.get(config_id)
        if invalidation is None:
            return attempt["status"], attempt["perf_ms"]
        require(
            attempt.get("status") in successful_statuses
            and attempt.get("source_hash") == invalidation["source_hash"],
            f"{path}: isolated rebenchmark invalidation lacks a matching "
            "successful sidecar source",
        )
        return invalidation["status"], None

    def require_attempt(config_id: str, *, initial: bool) -> dict[str, Any]:
        attempt = attempt_by_config.get(config_id)
        require(
            isinstance(attempt, dict),
            f"{path}: qualification config {config_id} has no sidecar attempt",
        )
        generation = attempt.get("generation")
        generation_zero = initial or config_id in schedule_anchor_ids
        require(
            type(generation) is int
            and ((generation == 0) if generation_zero else (1 <= generation <= passes)),
            f"{path}: qualification config {config_id} generation",
        )
        final_status, _final_perf = final_phase_attempt(config_id, attempt)
        if attempt.get("status") in successful_statuses:
            require(
                optional_positive_float(attempt.get("perf_ms"))
                and attempt.get("perf_ms") is not None,
                f"{path}: successful qualification config {config_id} lacks performance",
            )
            if final_status in successful_statuses:
                successful_phase_ids.add(config_id)
                if not initial:
                    successful_candidate_ids.add(config_id)
        else:
            check_equal(
                attempt.get("perf_ms"),
                None,
                f"{path}: failed qualification config {config_id} performance",
            )
        return attempt

    initial_by_leaf: dict[str, set[str]] = {}
    initial_result_by_id = {
        record["config_id"]: record
        for record in phase.get("initial_results", [])
        if isinstance(record, dict) and isinstance(record.get("config_id"), str)
    }
    for config_id in initial_ids:
        config = metadata_configs.get(config_id)
        require(
            isinstance(config, dict),
            f"{path}: initial config {config_id} is absent from metadata",
        )
        leaf = structural_leaf(config)
        if leaf is not None:
            initial_by_leaf.setdefault(canonical_json(leaf), set()).add(config_id)
        attempt = require_attempt(config_id, initial=True)
        if initial_result_by_id:
            record = initial_result_by_id[config_id]
            check_equal(
                record["source_hash"],
                attempt.get("source_hash"),
                f"{path}: generation-zero source {config_id}",
            )
            initial_status = attempt.get(
                "pre_isolated_rebenchmark_status", attempt.get("status")
            )
            repaired_initial = (
                "pre_isolated_rebenchmark_status" not in attempt
                and record["status"] in retryable_statuses
                and attempt.get("status") in successful_statuses
            )
            require(
                repaired_initial or record["status"] == initial_status,
                f"{path}: generation-zero status {config_id}: expected "
                f"{initial_status!r}, got {record['status']!r}",
            )
            initial_perf = (
                None
                if repaired_initial
                else attempt.get(
                    "pre_isolated_rebenchmark_perf_ms", attempt.get("perf_ms")
                )
            )
            attempt_perf = initial_perf
            if attempt_perf is None:
                check_equal(
                    record["attempt_perf"],
                    None,
                    f"{path}: generation-zero performance {config_id}",
                )
            else:
                require(
                    isinstance(record["attempt_perf"], (int, float))
                    and not isinstance(record["attempt_perf"], bool)
                    and abs(record["attempt_perf"] - attempt_perf) <= 0.500001e-6,
                    f"{path}: generation-zero performance {config_id}",
                )

    for probe_path in phase["family_probe_paths"]:
        constrained_ordinary = (
            not probe_path["unrestricted"] and probe_path["compound_packet"] is None
        )
        for round_record in probe_path["rounds"]:
            results_by_id = {
                result["config_id"]: result for result in round_record["results"]
            }
            for config_id in round_record["candidate_ids"]:
                family_probe_candidate_ids.add(config_id)
                explicit_candidate_ids.add(config_id)
                config = metadata_configs.get(config_id)
                require(
                    isinstance(config, dict),
                    f"{path}: family probe config {config_id} is absent from metadata",
                )
                leaf = structural_leaf(config)
                require(
                    leaf is not None,
                    f"{path}: family probe config {config_id} has no structural leaf",
                )
                leaf_key = canonical_json(leaf)
                family_probe_candidate_ids_by_leaf.setdefault(leaf_key, set()).add(
                    config_id
                )
                leaves_with_candidate_keys.add(leaf_key)
                if constrained_ordinary:
                    parent_score_config_ids.add(config_id)
                require_attempt(config_id, initial=False)
                result = results_by_id[config_id]
                if result["status"] in successful_statuses:
                    final_measurement = final_measurement_by_id.get(config_id)
                    selection_perf = (
                        final_measurement.get("selection_perf")
                        if isinstance(final_measurement, dict)
                        and final_measurement.get("status") in successful_statuses
                        else None
                    )
                    if selection_perf is None:
                        continue
                    require(
                        isinstance(selection_perf, (int, float))
                        and not isinstance(selection_perf, bool)
                        and math.isfinite(selection_perf)
                        and selection_perf > 0,
                        f"{path}: family probe selection performance {config_id}",
                    )
                    selection_perf_by_id[config_id] = selection_perf
                    if leaf["compound_packet"] is not None:
                        family_probe_successful_compound_members.setdefault(
                            leaf_key, []
                        ).append(
                            {
                                "config_id": config_id,
                                "selection_perf": selection_perf,
                                "pipeline_lanes": frozenset(),
                            }
                        )

    exact_config_ids = phase["exact_space_config_ids"]
    exact_by_leaf: dict[str, list[str]] = {}
    for config_id in exact_config_ids:
        config = metadata_configs.get(config_id)
        require(
            isinstance(config, dict),
            f"{path}: exact-space config {config_id} is absent from metadata",
        )
        leaf = structural_leaf(config)
        if leaf is not None:
            exact_by_leaf.setdefault(canonical_json(leaf), []).append(config_id)
    unsuccessful_exact_ids = {
        config_id: final_phase_attempt(config_id, attempt_by_config[config_id])[0]
        for config_id in exact_config_ids
        if final_phase_attempt(config_id, attempt_by_config[config_id])[0]
        not in successful_statuses
    }
    require(
        not unsuccessful_exact_ids,
        f"{path}: exact effective search space was not successfully exhausted: "
        f"{unsuccessful_exact_ids}",
    )
    clc_by_leaf = {
        (result["family"], result["softmax_disc"]): result
        for result in phase["clc_families"]
    }

    def hierarchical_clc_values_covered(
        family: str, softmax_disc: bool, config_ids: list[str]
    ) -> bool:
        clc_result = clc_by_leaf.get((family, softmax_disc))
        if clc_result is None:
            return True
        present_values = {
            metadata_configs[config_id].get(FLASH_CLC_HEADS_PER_BATCH_KEY)
            for config_id in config_ids
        }
        return set(clc_result["planned_values"]) <= present_values

    exact_space_exhausted = (
        bool(exact_config_ids)
        and all(
            config_id in initial_ids and config_id in successful_phase_ids
            for config_id in exact_config_ids
        )
        and all(
            hierarchical_clc_values_covered(
                family,
                softmax_disc,
                [
                    config_id
                    for config_id in exact_config_ids
                    if structural_leaf(metadata_configs[config_id])
                    == {
                        "family": family,
                        "compound_packet": None,
                        "softmax_disc": softmax_disc,
                    }
                ],
            )
            for family, softmax_disc in clc_by_leaf
        )
    )
    check_equal(
        phase["exact_space_exhausted"],
        exact_space_exhausted,
        f"{path}: measured exact-space exhaustion",
    )

    reconciled_leaves: list[dict[str, Any]] = []
    for result in phase["leaf_results"]:
        leaf = {
            "family": result["family"],
            "compound_packet": result["compound_packet"],
            "softmax_disc": result["softmax_disc"],
        }
        leaf_key = canonical_json(leaf)
        check_equal(
            set(result["initial_config_ids"]),
            initial_by_leaf.get(leaf_key, set()),
            f"{path}: exact initial membership for leaf {leaf!r}",
        )
        leaf_exact_ids = exact_by_leaf.get(leaf_key, [])
        leaf_space_exhausted = (
            bool(leaf_exact_ids)
            and all(
                config_id in initial_ids and config_id in successful_phase_ids
                for config_id in leaf_exact_ids
            )
            and hierarchical_clc_values_covered(
                result["family"], result["softmax_disc"], leaf_exact_ids
            )
        )
        check_equal(
            result["space_config_count"],
            len(leaf_exact_ids) if phase["exact_space_enumerated"] else None,
            f"{path}: exact-space config count for leaf {leaf!r}",
        )
        check_equal(
            result["space_exhausted"],
            leaf_space_exhausted,
            f"{path}: exact-space exhaustion for leaf {leaf!r}",
        )
        check_equal(
            result["ordinary_search_required"],
            not lanes_by_leaf[leaf_key] and not leaf_space_exhausted,
            f"{path}: ordinary-search requirement for leaf {leaf!r}",
        )
        explicit_ids = set(result["initial_config_ids"])
        explicit_ids.update(
            config_id
            for config_id in schedule_anchor_ids
            if structural_leaf(metadata_configs[config_id]) == leaf
        )
        leaf_round_candidate_ids: set[str] = set()
        for pass_result in result["rounds"]:
            explicit_ids.update(pass_result["candidate_config_ids"])
            leaf_round_candidate_ids.update(pass_result["candidate_config_ids"])
        for lane_result in result["pipeline_lanes"]:
            explicit_ids.add(lane_result["witness_config_id"])
            explicit_ids.update(lane_result["conditional_candidate_ids"])
            explicit_ids.update(lane_result["repair_candidate_ids"])
        clc_result = clc_by_leaf.get((result["family"], result["softmax_disc"]))
        if clc_result is not None:
            explicit_ids.update(clc_result["witness_config_ids"].values())
            for ids in clc_result["conditional_candidate_ids"].values():
                explicit_ids.update(ids)
            for repair_key in (
                "witness_repair_candidate_ids",
                "conditional_repair_candidate_ids",
            ):
                for ids in clc_result[repair_key].values():
                    explicit_ids.update(ids)
            explicit_ids.update(clc_result["combination_candidate_ids"])
        explicit_ids.update(family_probe_candidate_ids_by_leaf.get(leaf_key, set()))
        explicit_candidate_ids.update(explicit_ids - initial_ids)
        clc_witness_ids = (
            set(clc_result["witness_config_ids"].values())
            if clc_result is not None
            else set()
        )
        if (
            any(
                config_id not in initial_ids
                and structural_leaf(metadata_configs[config_id]) == leaf
                for config_id in schedule_anchor_ids
            )
            or leaf_round_candidate_ids
            or (clc_witness_ids - initial_ids)
        ):
            leaves_with_candidate_keys.add(leaf_key)

        qualified_by_id = {
            qualified["config_id"]: qualified
            for qualified in result["qualified_results"]
        }
        check_equal(
            set(qualified_by_id),
            explicit_ids,
            f"{path}: exact qualified membership for leaf {leaf!r}",
        )
        pipeline_lanes = lanes_by_leaf[leaf_key]
        successful_members: list[dict[str, Any]] = []
        for config_id, qualified in qualified_by_id.items():
            config = metadata_configs.get(config_id)
            require(
                isinstance(config, dict),
                f"{path}: qualified config {config_id} is absent from metadata",
            )
            check_equal(
                structural_leaf(config),
                leaf,
                f"{path}: qualified config {config_id} exact leaf",
            )
            actual_memberships = config_pipeline_lanes(config, pipeline_lanes)
            check_equal(
                qualified["pipeline_lanes"],
                [
                    pipeline_lane_metric(lane)
                    for lane in pipeline_lanes
                    if lane in actual_memberships
                ],
                f"{path}: qualified config {config_id} actual pipeline membership",
            )
            attempt = require_attempt(config_id, initial=config_id in initial_ids)
            phase_status, phase_perf = final_phase_attempt(config_id, attempt)
            check_equal(
                qualified["status"],
                phase_status,
                f"{path}: qualified config {config_id} sidecar status",
            )
            recorded_attempt_perf = qualified["attempt_perf"]
            attempt_perf = phase_perf
            if attempt_perf is None:
                check_equal(
                    recorded_attempt_perf,
                    None,
                    f"{path}: qualified config {config_id} attempt performance",
                )
            else:
                require(
                    isinstance(recorded_attempt_perf, (int, float))
                    and not isinstance(recorded_attempt_perf, bool)
                    and abs(recorded_attempt_perf - attempt_perf) <= 0.500001e-6,
                    f"{path}: qualified config {config_id} attempt performance "
                    "does not match the CSV",
                )
            if qualified["status"] in successful_statuses:
                selection_perf = qualified["selection_perf"]
                require(
                    isinstance(selection_perf, (int, float))
                    and not isinstance(selection_perf, bool)
                    and math.isfinite(selection_perf)
                    and selection_perf > 0,
                    f"{path}: successful qualified config {config_id} lacks "
                    "selection performance",
                )
                selection_perf_by_id[config_id] = selection_perf
                successful_results_by_id[config_id] = {
                    "config_id": config_id,
                    "attempt_perf": qualified["attempt_perf"],
                    "selection_perf": selection_perf,
                    "status": qualified["status"],
                }
                successful_members.append(
                    {
                        "config_id": config_id,
                        "selection_perf": selection_perf,
                        "pipeline_lanes": actual_memberships,
                    }
                )
            else:
                check_equal(
                    qualified["selection_perf"],
                    None,
                    f"{path}: failed qualified config {config_id} selection performance",
                )

        for lane_result in result["pipeline_lanes"]:
            lane = (lane_result["key"], lane_result["value"])
            lane_exact_ids = [
                config_id
                for config_id in leaf_exact_ids
                if metadata_configs[config_id].get(lane[0]) == lane[1]
            ]
            lane_space_exhausted = (
                bool(lane_exact_ids)
                and all(
                    config_id in initial_ids and config_id in successful_phase_ids
                    for config_id in lane_exact_ids
                )
                and hierarchical_clc_values_covered(result["family"], lane_exact_ids)
            )
            check_equal(
                lane_result["space_config_count"],
                len(lane_exact_ids) if phase["exact_space_enumerated"] else None,
                f"{path}: exact-space config count for pipeline lane {lane!r}",
            )
            check_equal(
                lane_result["space_exhausted"],
                lane_space_exhausted,
                f"{path}: exact-space exhaustion for pipeline lane {lane!r}",
            )
            check_equal(
                lane_result["conditional_required"],
                not lane_space_exhausted,
                f"{path}: conditional-search requirement for pipeline lane {lane!r}",
            )
            exact_initial_lane_ids = [
                config_id
                for config_id in result["initial_config_ids"]
                if metadata_configs[config_id].get(lane[0]) == lane[1]
            ]
            check_equal(
                lane_result["initial_config_ids"],
                exact_initial_lane_ids,
                f"{path}: exact initial membership for pipeline lane {lane!r}",
            )
            witness_id = lane_result["witness_config_id"]
            require(
                metadata_configs[witness_id].get(lane[0]) == lane[1],
                f"{path}: pipeline lane {lane!r} has an invalid witness",
            )
            witness_succeeded = witness_id in successful_phase_ids
            check_equal(
                lane_result["witness_succeeded"],
                witness_succeeded,
                f"{path}: pipeline lane {lane!r} witness success",
            )
            conditional_ids = lane_result["conditional_candidate_ids"]
            require(
                not (set(conditional_ids) & initial_ids)
                and len(conditional_ids)
                == (
                    EXPECTED_CONDITIONAL_CANDIDATES_PER_PIPELINE_LANE
                    if lane_result["conditional_required"]
                    else 0
                )
                and all(
                    metadata_configs[config_id].get(lane[0]) == lane[1]
                    for config_id in conditional_ids
                ),
                f"{path}: pipeline lane {lane!r} has an invalid conditional child",
            )
            successful_conditional_ids = [
                config_id
                for config_id in conditional_ids
                if config_id in successful_phase_ids
            ]
            check_equal(
                lane_result["successful_conditional_candidate_ids"],
                successful_conditional_ids,
                f"{path}: pipeline lane {lane!r} conditional successes",
            )
            repair_ids = lane_result["repair_candidate_ids"]
            require(
                not (set(repair_ids) & initial_ids)
                and not (set(repair_ids) & set(conditional_ids))
                and all(
                    metadata_configs[config_id].get(lane[0]) == lane[1]
                    for config_id in repair_ids
                ),
                f"{path}: pipeline lane {lane!r} has an invalid repair child",
            )
            tracked_failure_ids = [witness_id, *conditional_ids]
            generated_repair_ids: list[str] = []
            for repair_index, decision in enumerate(
                lane_result["repair_parent_decisions"]
            ):
                check_equal(
                    decision["repair_index"],
                    repair_index,
                    f"{path}: pipeline lane {lane!r} repair index",
                )
                generated = decision["generated_config_ids"]
                generated_repair_ids.extend(generated)
                tracked_failure_ids.extend(generated)
            check_equal(
                generated_repair_ids,
                repair_ids,
                f"{path}: pipeline lane {lane!r} repair children",
            )
            successful_repair_ids = [
                config_id
                for config_id in repair_ids
                if config_id in successful_phase_ids
            ]
            check_equal(
                lane_result["successful_repair_candidate_ids"],
                successful_repair_ids,
                f"{path}: pipeline lane {lane!r} repair successes",
            )
            require(
                witness_succeeded
                or successful_conditional_ids
                or successful_repair_ids,
                f"{path}: pipeline lane {lane!r} lacks successful evidence",
            )
            check_equal(
                lane_result["complete"],
                True,
                f"{path}: pipeline lane {lane!r} completion",
            )

        retained = lane_diverse_members(
            successful_members,
            pipeline_lanes,
            limit=provenance["flash_structural_retained_candidates_per_leaf"],
        )
        check_equal(
            result["retained_config_ids"],
            [member["config_id"] for member, _lane in retained],
            f"{path}: retained candidates for leaf {leaf!r}",
        )
        check_equal(result["complete"], True, f"{path}: ordinary leaf completion")
        reconciled_leaves.append(
            {
                **leaf,
                "retained_config_ids": result["retained_config_ids"],
                "retained_pipeline_lanes": [
                    pipeline_lane_metric(lane) for _member, lane in retained
                ],
            }
        )
        qualified_leaves.append(
            {
                **leaf,
                "members": successful_members,
                "pipeline_lanes": pipeline_lanes,
            }
        )

    for result in phase["clc_families"]:
        leaf = {
            "family": result["family"],
            "compound_packet": None,
            "softmax_disc": result["softmax_disc"],
        }
        leaf_exact_ids = exact_by_leaf.get(canonical_json(leaf), [])
        matching_leaf_result = next(
            leaf_result
            for leaf_result in phase["leaf_results"]
            if leaf_result["family"] == result["family"]
        )
        check_equal(
            result["space_exhausted"],
            matching_leaf_result["space_exhausted"],
            f"{path}: CLC leaf exhaustion",
        )
        expected_value_exhaustion: dict[str, bool] = {}
        best_witness_by_value: dict[int, str] = {}
        for value in result["planned_values"]:
            candidate_ids = [
                result["witness_config_ids"][str(value)],
                *result["witness_repair_candidate_ids"].get(str(value), []),
            ]
            require(
                all(
                    isinstance(metadata_configs.get(config_id), dict)
                    and structural_leaf(metadata_configs[config_id]) == leaf
                    and metadata_configs[config_id].get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                    == value
                    for config_id in candidate_ids
                )
                and any(
                    config_id in successful_phase_ids for config_id in candidate_ids
                ),
                f"{path}: CLC {result['family']} value {value} lacks a successful witness",
            )
            best_witness_by_value[value] = min(
                (
                    config_id
                    for config_id in candidate_ids
                    if config_id in successful_phase_ids
                ),
                key=lambda config_id: (selection_perf_by_id[config_id], config_id),
            )
            value_exact_ids = [
                exact_id
                for exact_id in leaf_exact_ids
                if metadata_configs[exact_id].get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                == value
            ]
            expected_value_exhaustion[str(value)] = bool(value_exact_ids) and all(
                exact_id in initial_ids and exact_id in successful_phase_ids
                for exact_id in value_exact_ids
            )
        check_equal(
            result["value_space_exhausted"],
            expected_value_exhaustion,
            f"{path}: CLC value-space exhaustion",
        )
        expected_selected = [
            selection["value"] for selection in result["witness_selection_results"]
        ]
        check_equal(
            result["selected_values"],
            expected_selected,
            f"{path}: selected CLC values",
        )
        for value in result["conditional_values"]:
            ids = [
                *result["conditional_candidate_ids"][str(value)],
                *result["conditional_repair_candidate_ids"].get(str(value), []),
            ]
            require(
                all(
                    structural_leaf(metadata_configs[config_id]) == leaf
                    and metadata_configs[config_id].get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                    == value
                    for config_id in ids
                )
                and any(config_id in successful_phase_ids for config_id in ids),
                f"{path}: CLC {result['family']} value {value} lacks a successful child",
            )
        check_equal(
            result["retained_values"],
            [ranking["value"] for ranking in result["retained_ranking_results"]],
            f"{path}: retained CLC values",
        )
        pipeline_generated_ids = {
            config_id
            for leaf_result in phase["leaf_results"]
            for round_result in leaf_result["rounds"]
            for config_id in round_result["candidate_config_ids"]
        }
        pre_combination_ids = (
            initial_ids
            | schedule_anchor_ids
            | pipeline_generated_ids
            | set(result["witness_config_ids"].values())
            | {
                config_id
                for candidate_ids in result["witness_repair_candidate_ids"].values()
                for config_id in candidate_ids
            }
            | {
                config_id
                for candidate_ids in result["conditional_candidate_ids"].values()
                for config_id in candidate_ids
            }
            | {
                config_id
                for candidate_ids in result["conditional_repair_candidate_ids"].values()
                for config_id in candidate_ids
            }
        )
        reconcile_clc_depth_selection(
            path,
            result,
            leaf=leaf,
            pre_combination_ids=pre_combination_ids,
            successful_ids=successful_phase_ids,
            successful_results_by_id=successful_results_by_id,
            metadata_configs=metadata_configs,
            pipeline_lanes=lanes_by_leaf[canonical_json(leaf)],
            retained_limit=provenance["flash_structural_retained_candidates_per_leaf"],
        )
        combination_ids = result["combination_candidate_ids"]
        cells = result["combination_cells"]
        projected_ids = [
            cell["projected_config_id"]
            for cell in cells
            if isinstance(cell["projected_config_id"], str)
        ]
        check_equal(
            combination_ids,
            list(dict.fromkeys(projected_ids)),
            f"{path}: exact CLC combination ID ledger",
        )
        successful_cell_depth_ids: set[str] = set()
        successful_cell_divisors: set[int] = set()
        for cell in cells:
            depth_id = cell["depth_config_id"]
            divisor = cell["divisor_value"]
            require(
                depth_id in successful_phase_ids
                and structural_leaf(metadata_configs[depth_id]) == leaf,
                f"{path}: CLC cell has an invalid depth source",
            )
            projected_id = cell["projected_config_id"]
            if projected_id is None:
                continue
            projected = metadata_configs.get(projected_id)
            require(
                isinstance(projected, dict)
                and canonical_sha256(projected)[:16] == projected_id
                and structural_leaf(projected) == leaf
                and projected.get(FLASH_CLC_HEADS_PER_BATCH_KEY) == divisor,
                f"{path}: CLC cell projection changes the wrong schedule axis",
            )
            if cell["status"] in successful_statuses:
                successful_cell_depth_ids.add(depth_id)
                successful_cell_divisors.add(divisor)
        expected_successful_depth_ids = [
            config_id
            for config_id in result["combination_depth_config_ids"]
            if config_id in successful_cell_depth_ids
        ]
        expected_successful_divisors = [
            value
            for value in result["combination_divisor_values"]
            if value in successful_cell_divisors
        ]
        check_equal(
            result["successful_combination_depth_config_ids"],
            expected_successful_depth_ids,
            f"{path}: exact CLC row coverage",
        )
        check_equal(
            result["successful_combination_divisor_values"],
            expected_successful_divisors,
            f"{path}: exact CLC column coverage",
        )
        require(
            all(
                structural_leaf(metadata_configs[config_id]) == leaf
                and metadata_configs[config_id].get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                in result["retained_values"]
                for config_id in combination_ids
            ),
            f"{path}: invalid CLC depth/divisor combination evidence",
        )
        check_equal(result["complete"], True, f"{path}: CLC family completion")

    qualified_compound_ids: set[str] = set()
    for result in phase["compound_transfers"]:
        result_qualified_ids = set(result["qualified_transfer_config_ids"])
        ordinary_leaf = {
            "family": result["family"],
            "compound_packet": None,
            "softmax_disc": result["softmax_disc"],
        }
        compound_leaf = {
            "family": result["family"],
            "compound_packet": result["compound_packet"],
            "softmax_disc": result["softmax_disc"],
        }
        compound_leaf_key = canonical_json(compound_leaf)
        if result["primary_transfer_config_ids"] or any(
            backfill["generated_config_ids"] for backfill in result["backfill_rounds"]
        ):
            leaves_with_candidate_keys.add(compound_leaf_key)
        compound_members: list[dict[str, Any]] = []
        for transfer in result["transfers"]:
            source_id = transfer["source_config_id"]
            target_id = transfer["transferred_config_id"]
            source = metadata_configs.get(source_id)
            target = metadata_configs.get(target_id)
            require(
                isinstance(source, dict)
                and isinstance(target, dict)
                and structural_leaf(source) == ordinary_leaf
                and structural_leaf(target) == compound_leaf,
                f"{path}: compound transfer changes the wrong structural leaf",
            )
            check_equal(
                transfer["source_config"],
                source,
                f"{path}: compound source snapshot",
            )
            check_equal(
                transfer["projected_config"],
                target,
                f"{path}: compound projected snapshot",
            )
            check_equal(
                transfer["projected_config_id"],
                canonical_sha256(target)[:16],
                f"{path}: compound target canonical ID",
            )
            check_equal(
                transfer["projection_overrides"],
                {FLASH_EXP2_PACKET_KEY: result["compound_packet"]},
                f"{path}: compound projection overrides",
            )
            expected_preserved_pipeline = {
                key: source[key]
                for key in FLASH_PIPELINE_QUALIFICATION_KEYS
                if key in source
            }
            check_equal(
                transfer["preserved_pipeline_values"],
                expected_preserved_pipeline,
                f"{path}: compound preserved pipeline values",
            )
            require_attempt(source_id, initial=source_id in initial_ids)
            require_attempt(target_id, initial=target_id in initial_ids)
            transfer_measurement = measurement_states[pre_probe_pass].get(target_id)
            require(
                source_id in successful_phase_ids,
                f"{path}: compound transfer lacks a successful source",
            )
            require(
                isinstance(transfer_measurement, dict),
                f"{path}: compound target lacks its pre-probe measurement",
            )
            check_equal(
                transfer["status"],
                transfer_measurement["status"],
                f"{path}: compound target status",
            )
            target_perf = transfer_measurement["attempt_perf"]
            if target_perf is None:
                check_equal(
                    transfer["attempt_perf"],
                    None,
                    f"{path}: compound target attempt performance",
                )
            else:
                require(
                    abs(transfer["attempt_perf"] - target_perf) <= 0.500001e-6,
                    f"{path}: compound target attempt performance",
                )
            transfer_selection_perf = transfer_measurement["selection_perf"]
            if transfer_selection_perf is None:
                check_equal(
                    transfer["selection_perf"],
                    None,
                    f"{path}: compound target selection performance",
                )
            else:
                require(
                    isinstance(transfer["selection_perf"], (int, float))
                    and not isinstance(transfer["selection_perf"], bool)
                    and abs(transfer["selection_perf"] - transfer_selection_perf)
                    <= 0.500001e-6,
                    f"{path}: compound target selection performance",
                )
            if target_id in result_qualified_ids:
                final_measurement = final_measurement_by_id.get(target_id)
                if (
                    target_id in successful_phase_ids
                    and isinstance(final_measurement, dict)
                    and final_measurement.get("status") in successful_statuses
                    and optional_positive_float(final_measurement.get("selection_perf"))
                ):
                    selection_perf = final_measurement["selection_perf"]
                    selection_perf_by_id[target_id] = selection_perf
                    compound_members.append(
                        {
                            "config_id": target_id,
                            "selection_perf": selection_perf,
                            "pipeline_lanes": frozenset(),
                        }
                    )
                    qualified_compound_ids.add(target_id)
            if target_id not in initial_ids:
                explicit_candidate_ids.add(target_id)
        check_equal(result["complete"], True, f"{path}: compound transfer completion")
        check_equal(
            [member["config_id"] for member in compound_members],
            [
                config_id
                for config_id in result["qualified_transfer_config_ids"]
                if measurement_states[-1].get(config_id, {}).get("status")
                in successful_statuses
            ],
            f"{path}: qualified compound retention pool",
        )
        probe_members = family_probe_successful_compound_members.get(
            compound_leaf_key, []
        )
        require(
            not (
                {member["config_id"] for member in compound_members}
                & {member["config_id"] for member in probe_members}
            ),
            f"{path}: duplicate compound family probe member",
        )
        compound_members.extend(probe_members)
        qualified_compound_ids.update(member["config_id"] for member in probe_members)
        qualified_leaves.append(
            {
                **compound_leaf,
                "members": compound_members,
                "pipeline_lanes": [],
            }
        )

    for config_id in initial_ids:
        config = metadata_configs[config_id]
        leaf = structural_leaf(config)
        if leaf is not None and leaf["compound_packet"] is not None:
            attempt = attempt_by_config[config_id]
            if attempt["status"] in successful_statuses:
                selection_perf_by_id.setdefault(config_id, attempt["perf_ms"])
    for result in phase["compound_transfers"]:
        for transfer in result["transfers"]:
            config_id = transfer["transferred_config_id"]
            selection_perf_by_id.setdefault(
                config_id, attempt_by_config[config_id]["perf_ms"]
            )

    expected_retained_families = expected_structural_retention(
        qualified_leaves,
        retained_per_leaf=provenance["flash_structural_retained_candidates_per_leaf"],
        retained_family_cap=provenance["flash_structural_retained_family_cap"],
        retained_family_limit=provenance["flash_structural_retained_family_limit"],
        retained_family_slowdown_limit=provenance[
            "flash_structural_retained_family_slowdown_limit"
        ],
        starting_path_limit=provenance["flash_structural_starting_path_limit"],
        parent_score_config_ids=parent_score_config_ids,
    )

    def order_independent_starting_paths(
        retained_families: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                **family,
                "starting_paths": sorted(family["starting_paths"], key=canonical_json),
            }
            for family in retained_families
        ]

    check_equal(
        order_independent_starting_paths(phase["retained_families"]),
        order_independent_starting_paths(expected_retained_families),
        f"{path}: exact retained structural family ranking",
    )

    retained_path_ids: list[str] = []
    for family_result in phase["retained_families"]:
        family = family_result["family"]
        ordinary_scores = [
            selection_perf
            for config_id, selection_perf in selection_perf_by_id.items()
            if config_id in parent_score_config_ids
            if structural_leaf(metadata_configs[config_id])
            in [
                leaf
                for leaf in provenance["flash_structural_leaf_catalog"]
                if leaf["family"] == family and leaf["compound_packet"] is None
            ]
        ]
        require(
            ordinary_scores, f"{path}: retained family {family} has no ordinary score"
        )
        require(
            abs(family_result["score"] - min(ordinary_scores)) <= 0.500001e-6,
            f"{path}: retained family {family} score",
        )
        check_equal(
            family_result["score_compound_packet"],
            None,
            f"{path}: retained family score leaf",
        )
        score_leaf = next(
            leaf
            for leaf in provenance["flash_structural_leaf_catalog"]
            if leaf["family"] == family
            and leaf["compound_packet"] is None
            and leaf["softmax_disc"] == family_result["score_softmax_disc"]
            and any(
                structural_leaf(metadata_configs[config_id]) == leaf
                and config_id in parent_score_config_ids
                and selection_perf == family_result["score"]
                for config_id, selection_perf in selection_perf_by_id.items()
            )
        )
        require(score_leaf is not None, f"{path}: retained family score protocol")
        for starting_path in family_result["starting_paths"]:
            config_id = starting_path["config_id"]
            config = metadata_configs.get(config_id)
            require(
                isinstance(config, dict)
                and config_id in successful_phase_ids
                and structural_leaf(config)
                == {
                    "family": starting_path["family"],
                    "compound_packet": starting_path["compound_packet"],
                    "softmax_disc": starting_path["softmax_disc"],
                },
                f"{path}: retained starting path is not a successful phase config",
            )
            if starting_path["compound_packet"] is not None:
                require(
                    config_id in qualified_compound_ids,
                    f"{path}: retained compound path was not transferred",
                )
            lane = starting_path["pipeline_lane"]
            require(
                lane is None or config.get(lane["key"]) == lane["value"],
                f"{path}: retained starting path violates its pipeline lane",
            )
            retained_path_ids.append(config_id)
    check_equal(
        len(retained_path_ids),
        phase["retained_path_count"],
        f"{path}: retained path count",
    )
    check_equal(
        phase["candidate_count"],
        len(explicit_candidate_ids),
        f"{path}: exact structural qualification candidate count",
    )
    check_equal(
        phase["leaves_with_candidates"],
        len(leaves_with_candidate_keys),
        f"{path}: exact structural qualification leaves with candidates",
    )
    return {
        "successful_candidate_ids": successful_candidate_ids,
        "leaf_results": reconciled_leaves,
    }


def reconcile_structural_qualification_phase(
    path: Path,
    provenance: dict[str, Any],
    phase: dict[str, Any],
    attempt_by_config: dict[str, dict[str, Any]],
    metadata_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return _reconcile_structural_qualification_phase_v22(
        path, provenance, phase, attempt_by_config, metadata_configs
    )


def validate_search_generations(
    path: Path,
    rows: list[dict[str, str]],
    trial: dict[str, Any],
) -> None:
    """Validate sparse generation IDs against the structural pass timeline."""
    generations = {int(row["generation"]) for row in rows}
    final_generation = trial["num_generations"]
    require(0 in generations, f"{path}: search generations omit generation zero")
    require(
        all(generation <= final_generation for generation in generations),
        f"{path}: search generation exceeds the recorded final generation",
    )
    require(
        max(generations) == final_generation,
        f"{path}: search generations do not reach the recorded final generation",
    )

    phase = trial.get("search_phase_metrics")
    require(isinstance(phase, dict), f"{path}: missing structural search phase")
    timeline = phase.get("measurement_timeline")
    pass_count = phase.get("qualification_passes_completed")
    anchor_started = phase.get("schedule_anchor_pass_started")
    require(
        isinstance(timeline, list)
        and type(pass_count) is int
        and type(anchor_started) is bool
        and len(timeline) == pass_count + 1,
        f"{path}: malformed structural generation timeline",
    )

    # Generation IDs are budget slots, not a dense count of benchmark batches.
    # The optional schedule-anchor pass extends generation zero. A counted
    # qualification pass with no new candidates legitimately leaves no ledger
    # row for its generation, while later search generations may still run.
    anchor_offset = int(anchor_started)
    structural_generation_limit = pass_count - anchor_offset
    require(
        structural_generation_limit >= 0,
        f"{path}: malformed structural generation limit",
    )
    earliest_generation_by_config: dict[str, int] = {}
    for row in rows:
        earliest_generation_by_config.setdefault(
            row["config_id"], int(row["generation"])
        )
    expected_config_ids_by_generation: dict[int, set[str]] = {}
    introduced_config_ids: set[str] = set()
    for pass_index, record in enumerate(timeline):
        require(
            isinstance(record, dict) and isinstance(record.get("updates"), list),
            f"{path}: malformed structural generation timeline",
        )
        update_ids: set[str] = set()
        for update in record["updates"]:
            config_id = update.get("config_id") if isinstance(update, dict) else None
            require(
                isinstance(config_id, str),
                f"{path}: malformed structural generation update",
            )
            update_ids.add(config_id)
        newly_introduced = update_ids - introduced_config_ids
        expected_generation = (
            0 if pass_index <= anchor_offset else pass_index - anchor_offset
        )
        for config_id in newly_introduced:
            check_equal(
                earliest_generation_by_config.get(config_id),
                expected_generation,
                f"{path}: config {config_id} first ledger generation",
            )
        expected_config_ids_by_generation.setdefault(expected_generation, set()).update(
            newly_introduced
        )
        introduced_config_ids.update(update_ids)
    observed_config_ids_by_generation: dict[int, set[str]] = {}
    for config_id, generation in earliest_generation_by_config.items():
        if generation <= structural_generation_limit:
            observed_config_ids_by_generation.setdefault(generation, set()).add(
                config_id
            )
    expected_config_ids_by_generation = {
        generation: config_ids
        for generation, config_ids in expected_config_ids_by_generation.items()
        if config_ids
    }
    check_equal(
        observed_config_ids_by_generation,
        expected_config_ids_by_generation,
        f"{path}: structural config generations",
    )
    expected_structural_generations = {
        generation
        for generation, config_ids in expected_config_ids_by_generation.items()
        if generation > 0 and config_ids
    }
    observed_structural_generations = {
        generation
        for generation in generations
        if 0 < generation <= structural_generation_limit
    }
    check_equal(
        observed_structural_generations,
        expected_structural_generations,
        f"{path}: measured structural search generations",
    )


def read_and_validate_ledger(
    path: Path,
    trial: dict[str, Any],
    selected_config: dict[str, Any],
    selected_source: str,
) -> dict[str, Any]:
    require(path.is_file(), f"missing adjacent source ledger: {path}")
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            check_equal(
                tuple(reader.fieldnames or ()), LEDGER_FIELDS, f"{path}: header"
            )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"unable to read {path}: {exc}") from exc
    require(rows, f"{path}: source ledger is empty")

    for line, row in enumerate(rows, 2):
        require(
            None not in row
            and all(row.get(field) is not None for field in LEDGER_FIELDS),
            f"{path}:{line}: malformed source ledger row",
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", row["run_id"]) is not None,
            f"{path}:{line}: invalid run_id",
        )
        csv_float(row["timestamp_s"], f"{path}:{line}: timestamp_s")
        require(
            re.fullmatch(r"[0-9a-f]{16}", row["config_id"]) is not None,
            f"{path}:{line}: invalid config_id",
        )
        csv_int(row["generation"], f"{path}:{line}: generation", minimum=0)
        require(
            row["status"] in LEDGER_STATUSES,
            f"{path}:{line}: unexpected status {row['status']!r}",
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", row["source_hash"]) is not None,
            f"{path}:{line}: invalid source hash",
        )

    run_ids = {row["run_id"] for row in rows}
    require(len(run_ids) == 1, f"{path}: source ledger mixes run IDs")
    selected_ok = [
        row
        for row in rows
        if row["source_hash"] == selected_source and row["status"] == "ok"
    ]
    require(
        len(selected_ok) == 1,
        f"{path}: selected source is not present exactly once as a successful measurement",
    )
    selected_config_id = canonical_sha256(selected_config)[:16]
    selected_config_rows = [
        row
        for row in rows
        if row["config_id"] == selected_config_id
        and row["source_hash"] == selected_source
        and row["status"] in {"ok", "deduplicated"}
    ]
    require(
        len(selected_config_rows) == 1,
        f"{path}: selected config is not linked to its successful source",
    )

    prior_ok_sources: set[str] = set()
    prior_ok_source_rows: dict[str, dict[str, str]] = {}
    accuracy_failure_sources: set[str] = set()
    pending_transient_configs: dict[str, set[str]] = {}
    accuracy_failure_position: dict[str, int] = {}
    for position, row in enumerate(rows):
        source_hash = row["source_hash"]
        if row["status"] in ({"started"} | LEDGER_REPAIRABLE_FAILURE_STATUSES):
            require(
                source_hash not in prior_ok_sources
                and source_hash not in accuracy_failure_sources,
                f"{path}: source has an attempted outcome after its definitive outcome",
            )
            if row["status"] in LEDGER_REPAIRABLE_FAILURE_STATUSES:
                pending_transient_configs.setdefault(source_hash, set()).add(
                    row["config_id"]
                )
        if row["status"] == "accuracy_error":
            require(
                source_hash not in prior_ok_sources
                and source_hash not in accuracy_failure_sources,
                f"{path}: source has more than one definitive outcome",
            )
            accuracy_failure_sources.add(source_hash)
            accuracy_failure_position.setdefault(source_hash, position)
        elif row["status"] == "ok":
            require(
                source_hash not in accuracy_failure_sources
                and source_hash not in prior_ok_sources,
                f"{path}: source has more than one definitive outcome",
            )
            prior_ok_sources.add(source_hash)
            prior_ok_source_rows[source_hash] = row
        elif row["status"] == "deduplicated":
            repairing_row = prior_ok_source_rows.get(source_hash)
            repairs_transient = row["config_id"] in pending_transient_configs.get(
                source_hash, set()
            )
            require(
                source_hash in prior_ok_sources
                and (
                    not repairs_transient
                    or (
                        repairing_row is not None
                        and repairing_row["generation"] == row["generation"]
                        and repairing_row["config_id"] != row["config_id"]
                    )
                ),
                f"{path}: deduplicated config {row['config_id']} has no prior "
                "successful source in its repair resolution generation",
            )
            pending_transient_configs.get(source_hash, set()).discard(row["config_id"])
        elif row["status"] == "source_rejected":
            require(
                accuracy_failure_position.get(source_hash, len(rows)) < position,
                f"{path}: source-rejected config {row['config_id']} has no prior "
                "accuracy failure for its source",
            )
    unrepaired = {
        source_hash: sorted(pending_transient_configs.get(source_hash, set()))
        for source_hash in prior_ok_sources
        if pending_transient_configs.get(source_hash)
    }
    require(
        not unrepaired,
        f"{path}: successful sources have unrepaired transient attempts: {unrepaired}",
    )

    by_config: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_config.setdefault(row["config_id"], []).append(row)
    for config_id, config_rows in by_config.items():
        statuses = [row["status"] for row in config_rows]
        standalone = len(statuses) == 1 and statuses[0] in LEDGER_ALIAS_STATUSES
        standalone_failure = (
            len(statuses) == 1 and statuses[0] in LEDGER_REPAIRABLE_FAILURE_STATUSES
        )
        attempted = (
            len(statuses) == 2
            and statuses[0] == "started"
            and statuses[1] in LEDGER_TERMINAL_STATUSES
        )
        repaired = (
            len(statuses) == 3
            and statuses[0] == "started"
            and statuses[1] in LEDGER_REPAIRABLE_FAILURE_STATUSES
            and statuses[2] == "deduplicated"
        )
        unstarted_repaired = (
            len(statuses) == 2
            and statuses[0] in LEDGER_REPAIRABLE_FAILURE_STATUSES
            and statuses[1] == "deduplicated"
        )
        generations = [row["generation"] for row in config_rows]
        require(
            (
                standalone
                or standalone_failure
                or attempted
                or repaired
                or unstarted_repaired
            )
            and (
                standalone
                or standalone_failure
                or (attempted and len(set(generations)) == 1)
                or (repaired and generations[0] == generations[1])
                or unstarted_repaired
            ),
            f"{path}: malformed lifecycle for config {config_id}: {statuses}",
        )
        require(
            len({row["source_hash"] for row in config_rows}) == 1,
            f"{path}: inconsistent lifecycle provenance for config {config_id}",
        )

    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in LEDGER_STATUSES
    }
    num_configs_tested = strict_int(
        trial.get("num_configs_tested"), f"{path}: tested count", minimum=0
    )
    check_equal(counts["started"], num_configs_tested, f"{path}: tested count")
    check_equal(
        counts["ok"],
        trial["num_successful_candidate_measurements"],
        f"{path}: successful count",
    )
    check_equal(
        counts["deduplicated"] + counts["source_rejected"],
        trial["num_source_deduplications"],
        f"{path}: source alias count",
    )
    check_equal(
        counts["accuracy_error"],
        trial["num_accuracy_failures"],
        f"{path}: accuracy failure count",
    )
    other_failures = sum(
        counts[status] for status in LEDGER_REPAIRABLE_FAILURE_STATUSES
    )
    check_equal(
        other_failures,
        trial["num_compile_failures"] + trial["num_worker_failures"],
        f"{path}: non-accuracy failure count",
    )
    unique_sources = {row["source_hash"] for row in rows}
    check_equal(
        len(unique_sources), trial["num_unique_sources"], f"{path}: unique source count"
    )
    validate_search_generations(path, rows, trial)
    return {
        "run_id": next(iter(run_ids)),
        "selected_config_id": selected_config_rows[0]["config_id"],
        "selected_generation": int(selected_config_rows[0]["generation"]),
        "rows": rows,
    }


def metadata_run_id(metadata: dict[str, Any], path: Path) -> str:
    settings = metadata.get("settings")
    require(isinstance(settings, dict), f"{path}: metadata settings are invalid")
    identity_fields = ("kernel_source", "input_shapes", "dtypes", "hardware")
    for field in identity_fields:
        require(
            isinstance(metadata.get(field), str),
            f"{path}: metadata {field} is invalid",
        )
    codegen_signature = ", ".join(
        f"{name}={settings.get(name)}" for name in CODEGEN_SETTINGS
    )
    payload = (
        f"{metadata['kernel_source']}\x00{codegen_signature}\x00"
        f"{metadata['input_shapes']}\x00{metadata['dtypes']}\x00"
        f"{metadata['hardware']}"
    )
    return sha256_bytes(payload.encode())


def read_metadata_record(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"missing adjacent autotune metadata: {path}")
    try:
        contents = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"unable to read {path}: {exc}") from exc
    lines = [line for line in contents.splitlines() if line.strip()]
    require(len(lines) == 1, f"{path}: expected exactly one metadata record")
    try:
        metadata = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{path}: invalid metadata JSON: {exc}") from exc
    require(isinstance(metadata, dict), f"{path}: metadata must be a JSON object")
    return metadata, sha256_bytes(contents)


def validate_autotune_sidecars(
    csv_path: Path,
    metadata_path: Path,
    source_rows: list[dict[str, str]],
    source_run_id: str,
    selected_config: dict[str, Any],
    variant: str,
    seq_len: int,
    tuner_seed: int,
) -> dict[str, object]:
    require(csv_path.is_file(), f"missing adjacent autotune CSV: {csv_path}")
    try:
        csv_contents = csv_path.read_bytes()
        reader = csv.DictReader(io.StringIO(csv_contents.decode()))
        check_equal(
            tuple(reader.fieldnames or ()), AUTOTUNE_CSV_FIELDS, f"{csv_path}: header"
        )
        rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise RuntimeError(f"unable to read {csv_path}: {exc}") from exc
    check_equal(len(rows), len(source_rows), f"{csv_path}: source ledger row count")
    for line, (row, source_row) in enumerate(zip(rows, source_rows, strict=True), 2):
        require(
            None not in row
            and all(row.get(field) is not None for field in AUTOTUNE_CSV_FIELDS),
            f"{csv_path}:{line}: malformed autotune row",
        )
        for field in AUTOTUNE_JOIN_FIELDS:
            check_equal(
                row[field],
                source_row[field],
                f"{csv_path}:{line}: source ledger {field}",
            )
        successful = row["status"] in {"ok", "deduplicated"}
        check_equal(
            bool(row["perf_ms"]),
            successful,
            f"{csv_path}:{line}: performance/status consistency",
        )
        if successful:
            require(
                csv_float(row["perf_ms"], f"{csv_path}:{line}: perf_ms") > 0.0,
                f"{csv_path}:{line}: perf_ms must be positive",
            )
        if row["compile_time_s"]:
            require(
                csv_float(row["compile_time_s"], f"{csv_path}:{line}: compile_time_s")
                >= 0.0,
                f"{csv_path}:{line}: compile_time_s must be nonnegative",
            )
        require(row["config"], f"{csv_path}:{line}: config is empty")

    attempt_by_config: dict[str, dict[str, Any]] = {}
    attempt_history_by_config: dict[str, list[dict[str, Any]]] = {}
    for position, row in enumerate(rows):
        if row["status"] == "started":
            continue
        config_id = row["config_id"]
        previous = attempt_by_config.get(config_id)
        require(
            previous is None
            or (
                previous["status"] in LEDGER_REPAIRABLE_FAILURE_STATUSES
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

    metadata, metadata_sha256 = read_metadata_record(metadata_path)
    run_id = metadata_run_id(metadata, metadata_path)
    check_equal(metadata.get("run_id"), run_id, f"{metadata_path}: computed run_id")
    check_equal(run_id, source_run_id, f"{metadata_path}: source ledger run_id")
    check_equal(
        metadata.get("kernel_name"),
        "causal_attention_output" if variant == "causal" else "attention_output",
        f"{metadata_path}: kernel name",
    )
    shape = (2, 32, seq_len, 64)
    check_equal(
        metadata.get("input_shapes"),
        repr([shape, shape, shape]),
        f"{metadata_path}: input shapes",
    )
    check_equal(
        metadata.get("dtypes"),
        repr(["torch.float16"] * 3),
        f"{metadata_path}: input dtypes",
    )
    check_equal(metadata.get("hardware"), "NVIDIA B200", f"{metadata_path}: hardware")
    settings = metadata["settings"]
    expected_settings = {
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
    }
    for key, expected in expected_settings.items():
        check_equal(settings.get(key), expected, f"{metadata_path}: settings.{key}")

    configs = metadata.get("configs")
    require(isinstance(configs, dict), f"{metadata_path}: configs map is invalid")
    observed_config_ids = {row["config_id"] for row in rows}
    check_equal(set(configs), observed_config_ids, f"{metadata_path}: config ID set")
    for config_id, config in configs.items():
        require(
            re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
            and isinstance(config, dict),
            f"{metadata_path}: invalid config entry {config_id!r}",
        )
        check_equal(
            canonical_sha256(config)[:16],
            config_id,
            f"{metadata_path}: canonical config ID {config_id}",
        )
    selected_config_id = canonical_sha256(selected_config)[:16]
    check_equal(
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
        check_equal(row["config"], config_repr, f"{csv_path}:{line}: config payload")

    return {
        "run_id": run_id,
        "config_count": len(configs),
        "configs": configs,
        "attempt_by_config": attempt_by_config,
        "attempt_history_by_config": attempt_history_by_config,
        "csv_sha256": sha256_bytes(csv_contents),
        "metadata_sha256": metadata_sha256,
    }


def _terminal_config_ids(value: object, label: str) -> list[str]:
    config_ids = _config_id_list(value, label)
    require(len(config_ids) == len(set(config_ids)), f"{label}: duplicate config ID")
    return config_ids


def _terminal_float(value: object, label: str, *, nonnegative: bool = False) -> float:
    result = finite_float(value, label)
    if nonnegative:
        require(result >= 0.0, f"{label}: expected a nonnegative value")
    return result


def _check_terminal_float(actual: object, expected: float, label: str) -> None:
    value = _terminal_float(actual, label)
    require(
        math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12),
        f"{label}: expected {expected!r}, got {value!r}",
    )


def expected_terminal_refinement_policy() -> dict[str, object]:
    return {
        "schema_version": EXPECTED_TERMINAL_REFINEMENT_SCHEMA_VERSION,
        "policy_version": EXPECTED_TERMINAL_REFINEMENT_POLICY_VERSION,
        "lane_policy_version": EXPECTED_LANE_POLICY_VERSION,
        "coordinate_policy": EXPECTED_TERMINAL_REFINEMENT_COORDINATE_POLICY,
        "measurement_policy": EXPECTED_TERMINAL_REFINEMENT_MEASUREMENT_POLICY,
        "rounds": EXPECTED_TERMINAL_REFINEMENT_ROUNDS,
        "beam_width": EXPECTED_TERMINAL_REFINEMENT_BEAM_WIDTH,
        "radius": EXPECTED_TERMINAL_REFINEMENT_RADIUS,
        "minimum_improvement_fraction": (
            EXPECTED_TERMINAL_REFINEMENT_MINIMUM_IMPROVEMENT
        ),
        "round_target_ms": EXPECTED_TERMINAL_REFINEMENT_ROUND_TARGET_MS,
        "confirmation_target_ms": (EXPECTED_TERMINAL_REFINEMENT_CONFIRMATION_TARGET_MS),
    }


def validate_terminal_refinement_provenance(
    path: Path, provenance: dict[str, Any]
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate the immutable policy and recorded ConfigSpec coordinate surface."""
    policy = provenance.get("flash_terminal_coordinate_refinement_policy")
    check_equal(
        canonical_json(policy),
        canonical_json(expected_terminal_refinement_policy()),
        f"{path}: terminal refinement policy",
    )
    assert isinstance(policy, dict)
    check_equal(
        provenance.get("flash_terminal_coordinate_refinement_policy_sha256"),
        canonical_sha256(policy),
        f"{path}: terminal refinement policy digest",
    )

    surface = provenance.get("flash_terminal_coordinate_surface_catalog")
    require(isinstance(surface, dict), f"{path}: terminal coordinate surface")
    check_equal(
        set(surface),
        {"schema_version", "radius", "leaves"},
        f"{path}: terminal coordinate surface fields",
    )
    check_equal(
        surface.get("schema_version"),
        EXPECTED_TERMINAL_SURFACE_SCHEMA_VERSION,
        f"{path}: terminal coordinate surface schema",
    )
    require(
        type(surface.get("schema_version")) is int,
        f"{path}: terminal coordinate surface schema type",
    )
    check_equal(
        surface.get("radius"),
        EXPECTED_TERMINAL_REFINEMENT_RADIUS,
        f"{path}: terminal coordinate surface radius",
    )
    require(
        type(surface.get("radius")) is int,
        f"{path}: terminal coordinate surface radius type",
    )
    check_equal(
        provenance.get("flash_terminal_coordinate_surface_catalog_sha256"),
        canonical_sha256(surface),
        f"{path}: terminal coordinate surface digest",
    )
    leaves = surface.get("leaves")
    require(
        isinstance(leaves, list) and leaves,
        f"{path}: terminal coordinate surface leaves",
    )
    expected_leaves = []
    for raw_leaf in provenance.get("flash_structural_leaf_catalog", []):
        require(isinstance(raw_leaf, dict), f"{path}: structural leaf catalog")
        expected_leaves.append(
            {
                "family": raw_leaf.get("family"),
                "compound_packet": raw_leaf.get("compound_packet"),
                "softmax_disc": raw_leaf.get("softmax_disc", False),
            }
        )
    observed_leaves: list[dict[str, object]] = []
    seen_leaf_keys: set[str] = set()
    for leaf_index, leaf_record in enumerate(leaves):
        require(
            isinstance(leaf_record, dict)
            and set(leaf_record) == {"leaf", "coordinates"},
            f"{path}: malformed terminal coordinate surface leaf {leaf_index}",
        )
        leaf = leaf_record.get("leaf")
        require(
            isinstance(leaf, dict)
            and set(leaf) == {"family", "compound_packet", "softmax_disc"}
            and isinstance(leaf.get("family"), str)
            and (
                leaf.get("compound_packet") is None
                or isinstance(leaf.get("compound_packet"), str)
            )
            and isinstance(leaf.get("softmax_disc"), bool),
            f"{path}: malformed terminal coordinate surface leaf identity",
        )
        leaf_key = canonical_json(leaf)
        require(
            leaf_key not in seen_leaf_keys,
            f"{path}: duplicate terminal coordinate surface leaf",
        )
        seen_leaf_keys.add(leaf_key)
        observed_leaves.append(leaf)
        coordinates = leaf_record.get("coordinates")
        require(
            isinstance(coordinates, list) and coordinates,
            f"{path}: empty terminal coordinate surface leaf",
        )
        seen_coordinates: set[tuple[str, int | None]] = set()
        for flat_index, coordinate in enumerate(coordinates):
            require(
                isinstance(coordinate, dict)
                and set(coordinate)
                == {
                    "flat_index",
                    "key",
                    "sequence_index",
                    "fragment_type",
                    "overridden",
                    "active_values",
                    "neighbors_by_value",
                },
                f"{path}: malformed terminal coordinate surface entry",
            )
            sequence_index = coordinate.get("sequence_index")
            require(
                type(coordinate.get("flat_index")) is int
                and coordinate.get("flat_index") == flat_index
                and isinstance(coordinate.get("key"), str)
                and (
                    sequence_index is None
                    or (type(sequence_index) is int and sequence_index >= 0)
                )
                and isinstance(coordinate.get("fragment_type"), str)
                and bool(coordinate.get("fragment_type"))
                and isinstance(coordinate.get("overridden"), bool)
                and isinstance(coordinate.get("active_values"), list)
                and bool(coordinate.get("active_values")),
                f"{path}: invalid terminal coordinate surface entry",
            )
            coordinate_key = (coordinate["key"], sequence_index)
            require(
                coordinate_key not in seen_coordinates,
                f"{path}: duplicate terminal coordinate surface coordinate",
            )
            seen_coordinates.add(coordinate_key)
            neighbors = coordinate.get("neighbors_by_value")
            require(
                isinstance(neighbors, list)
                and all(
                    isinstance(row, dict)
                    and set(row) == {"from_value", "to_values"}
                    and isinstance(row.get("to_values"), list)
                    for row in neighbors
                ),
                f"{path}: malformed terminal coordinate surface neighbors",
            )
            active_value_keys = [
                canonical_json(value) for value in coordinate["active_values"]
            ]
            require(
                len(active_value_keys) == len(set(active_value_keys)),
                f"{path}: duplicate terminal coordinate active value",
            )
            neighbor_value_keys = [
                canonical_json(row["from_value"]) for row in neighbors
            ]
            require(
                len(neighbor_value_keys) == len(set(neighbor_value_keys)),
                f"{path}: duplicate terminal coordinate neighbor row",
            )
            require(
                set(active_value_keys) <= set(neighbor_value_keys),
                f"{path}: terminal coordinate neighbor rows omit an active value",
            )
            if coordinate["fragment_type"] != "EnumFragment":
                check_equal(
                    set(neighbor_value_keys),
                    set(active_value_keys),
                    f"{path}: terminal coordinate neighbor row values",
                )
            for row in neighbors:
                to_value_keys = [canonical_json(value) for value in row["to_values"]]
                require(
                    len(to_value_keys) == len(set(to_value_keys))
                    and set(to_value_keys) <= set(active_value_keys)
                    and canonical_json(row["from_value"]) not in set(to_value_keys),
                    f"{path}: invalid terminal coordinate neighbor values",
                )
    check_equal(
        observed_leaves,
        expected_leaves,
        f"{path}: terminal coordinate surface leaf catalog",
    )
    return policy, surface


def _terminal_surface_requests(
    path: Path,
    leaf_record: dict[str, object],
    parent_config: dict[str, Any],
) -> list[dict[str, object]]:
    expected: list[dict[str, object]] = []
    coordinates = leaf_record["coordinates"]
    assert isinstance(coordinates, list)
    for coordinate in coordinates:
        assert isinstance(coordinate, dict)
        if coordinate["overridden"]:
            continue
        key = coordinate["key"]
        sequence_index = coordinate["sequence_index"]
        require(
            isinstance(key, str) and key in parent_config,
            f"{path}: terminal parent lacks surface coordinate {key!r}",
        )
        parent_value = parent_config[key]
        if sequence_index is not None:
            require(
                isinstance(parent_value, list)
                and type(sequence_index) is int
                and sequence_index < len(parent_value),
                f"{path}: invalid terminal parent sequence coordinate",
            )
            parent_value = parent_value[sequence_index]
        matching_rows = [
            row
            for row in coordinate["neighbors_by_value"]
            if row["from_value"] == parent_value
        ]
        require(
            len(matching_rows) == 1,
            f"{path}: terminal surface has no unique parent value",
        )
        for to_value in matching_rows[0]["to_values"]:
            expected.append(
                {
                    "flat_index": coordinate["flat_index"],
                    "key": key,
                    "sequence_index": sequence_index,
                    "from_value": parent_value,
                    "to_value": to_value,
                }
            )
    return expected


def _terminal_direct_projection(
    path: Path,
    parent_config: dict[str, Any],
    *,
    key: str,
    sequence_index: int | None,
    to_value: object,
) -> dict[str, Any]:
    """Apply one recorded coordinate without reproducing ConfigSpec normalization."""
    require(key in parent_config, f"{path}: terminal request key {key}")
    projected = copy.deepcopy(parent_config)
    if sequence_index is None:
        projected[key] = copy.deepcopy(to_value)
    else:
        sequence = projected[key]
        require(
            isinstance(sequence, list) and sequence_index < len(sequence),
            f"{path}: terminal request sequence coordinate",
        )
        sequence[sequence_index] = copy.deepcopy(to_value)
    return projected


def _terminal_coordinate_value(
    path: Path,
    config: dict[str, Any],
    *,
    key: str,
    sequence_index: int | None,
) -> object:
    require(key in config, f"{path}: projected terminal config lacks key {key!r}")
    value = config[key]
    if sequence_index is None:
        return value
    require(
        isinstance(value, list) and sequence_index < len(value),
        f"{path}: projected terminal config has an invalid sequence coordinate",
    )
    return value[sequence_index]


def _validate_terminal_measurement(
    path: Path,
    value: object,
    expected_config_ids: list[str],
    *,
    label: str,
    expected_target_ms: float,
) -> dict[str, float]:
    require(isinstance(value, dict), f"{path}: {label} is not an object")
    check_equal(
        set(value),
        {
            "base_order",
            "target_ms",
            "repeat_reference_perf_ms",
            "sweep_count",
            "calls_per_sample",
            "total_calls",
            "elapsed_ms",
            "median_ms",
        },
        f"{path}: {label} fields",
    )
    base_order = _terminal_config_ids(
        value.get("base_order"), f"{path}: {label} base order"
    )
    require(base_order, f"{path}: {label} base order is empty")
    check_equal(base_order, expected_config_ids, f"{path}: {label} base order")
    target_ms = _terminal_float(value.get("target_ms"), f"{path}: {label} target")
    check_equal(
        target_ms,
        expected_target_ms,
        f"{path}: {label} target",
    )
    repeat_reference = _terminal_float(
        value.get("repeat_reference_perf_ms"),
        f"{path}: {label} repeat reference",
    )
    require(
        repeat_reference > 0.0,
        f"{path}: {label} repeat reference must be positive",
    )
    raw_sweep_count = value.get("sweep_count")
    raw_calls_per_sample = value.get("calls_per_sample")
    raw_total_calls = value.get("total_calls")
    require(
        type(raw_sweep_count) is int
        and type(raw_calls_per_sample) is int
        and type(raw_total_calls) is int,
        f"{path}: {label} call-layout fields must be integers",
    )
    base_repeat_float = expected_target_ms / repeat_reference
    base_repeat = (
        EXPECTED_TERMINAL_REFINEMENT_REPEAT_MAX
        if not math.isfinite(base_repeat_float)
        else int(base_repeat_float)
    )
    desired_calls = min(EXPECTED_TERMINAL_REFINEMENT_REPEAT_MAX, max(3, base_repeat))
    desired_calls = max(2, desired_calls + desired_calls % 2)
    calls_per_sample = max(
        1, math.ceil(desired_calls / EXPECTED_TERMINAL_REFINEMENT_MAX_SWEEPS)
    )
    sweep_count = math.ceil(desired_calls / calls_per_sample)
    if sweep_count % 2:
        sweep_count += 1
    total_calls = sweep_count * calls_per_sample
    check_equal(raw_sweep_count, sweep_count, f"{path}: {label} sweep-count formula")
    check_equal(
        raw_calls_per_sample,
        calls_per_sample,
        f"{path}: {label} calls-per-sample formula",
    )
    check_equal(raw_total_calls, total_calls, f"{path}: {label} total-call formula")
    require(
        2 <= sweep_count <= EXPECTED_TERMINAL_REFINEMENT_MAX_SWEEPS
        and sweep_count % 2 == 0
        and total_calls >= desired_calls,
        f"{path}: {label} invalid call layout",
    )
    elapsed_rows = value.get("elapsed_ms")
    require(
        isinstance(elapsed_rows, list) and len(elapsed_rows) == sweep_count,
        f"{path}: {label} elapsed row count does not match sweep count",
    )
    sample_by_config: dict[str, list[float]] = {
        config_id: [] for config_id in base_order
    }
    indices = list(range(len(base_order)))
    for sweep_index, elapsed in enumerate(elapsed_rows):
        offset = (sweep_index // 2) % len(indices)
        rotated = indices[offset:] + indices[:offset]
        expected_order = rotated if sweep_index % 2 == 0 else list(reversed(rotated))
        require(
            isinstance(elapsed, list) and len(elapsed) == len(base_order),
            f"{path}: {label} malformed elapsed row",
        )
        for position, raw_timing in enumerate(elapsed):
            timing = _terminal_float(
                raw_timing,
                f"{path}: {label} sweep {sweep_index} timing {position}",
            )
            require(timing > 0.0, f"{path}: {label} timing must be positive")
            sample_by_config[base_order[expected_order[position]]].append(timing)

    raw_medians_by_config = [
        statistics.median(sample_by_config[config_id]) for config_id in base_order
    ]
    require(
        repeat_reference <= 4.0 * max(raw_medians_by_config),
        f"{path}: {label} repeat reference is inconsistent with raw timings",
    )
    require(
        max(sum(values) for values in sample_by_config.values()) * calls_per_sample
        >= 0.25 * expected_target_ms,
        f"{path}: {label} has insufficient raw timing work",
    )

    medians = value.get("median_ms")
    require(
        isinstance(medians, list) and len(medians) == len(base_order),
        f"{path}: {label} malformed medians",
    )
    result: dict[str, float] = {}
    for position, (config_id, record) in enumerate(
        zip(base_order, medians, strict=True)
    ):
        require(
            isinstance(record, dict) and set(record) == {"config_id", "value"},
            f"{path}: {label} malformed median {position}",
        )
        check_equal(
            record.get("config_id"),
            config_id,
            f"{path}: {label} median config order",
        )
        expected = raw_medians_by_config[position]
        _check_terminal_float(
            record.get("value"), expected, f"{path}: {label} median {config_id}"
        )
        result[config_id] = expected
    return result


def validate_terminal_coordinate_refinement(
    path: Path,
    provenance: dict[str, Any],
    trial: dict[str, Any],
    phase: dict[str, Any],
    source_rows: list[dict[str, str]],
    metadata_configs: dict[str, dict[str, Any]],
    attempt_by_config: dict[str, dict[str, Any]],
    attempt_history_by_config: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Authenticate the deterministic post-search CuTe flash refinement."""
    policy, surface = validate_terminal_refinement_provenance(path, provenance)
    value = phase.get("terminal_coordinate_refinement")
    require(isinstance(value, dict), f"{path}: missing terminal refinement")
    expected_fields = {
        "schema_version",
        "policy_version",
        "lane_policy_version",
        "coordinate_policy",
        "measurement_policy",
        "rounds_planned",
        "beam_width",
        "maximum_projection_parent_count",
        "projection_parent_count",
        "rounds_started",
        "rounds_completed",
        "completed",
        "budget_exhausted",
        "termination_reason",
        "search_generation",
        "preterminal_num_configs_tested",
        "preterminal_registry_config_count",
        "preterminal_registry_config_ids_hash_policy",
        "preterminal_registry_config_ids_sha256",
        "radius",
        "minimum_improvement_fraction",
        "initial_incumbent_config_id",
        "refined_config_id",
        "final_config_id",
        "projection_attempt_count",
        "unique_candidate_count",
        "new_candidate_count",
        "reused_candidate_count",
        "intra_terminal_reused_candidate_count",
        "prior_failed_candidate_count",
        "accepted_config_ids",
        "config_manifest_sha256",
        "config_manifest",
        "rounds",
        "confirmation",
    }
    check_equal(set(value), expected_fields, f"{path}: terminal refinement fields")
    for field in (
        "schema_version",
        "policy_version",
        "lane_policy_version",
        "rounds_planned",
        "beam_width",
        "maximum_projection_parent_count",
        "projection_parent_count",
        "rounds_started",
        "rounds_completed",
        "search_generation",
        "preterminal_num_configs_tested",
        "preterminal_registry_config_count",
        "radius",
        "projection_attempt_count",
        "unique_candidate_count",
        "new_candidate_count",
        "reused_candidate_count",
        "intra_terminal_reused_candidate_count",
        "prior_failed_candidate_count",
    ):
        require(
            type(value.get(field)) is int,
            f"{path}: terminal {field} must be an integer",
        )
    require(
        type(value.get("completed")) is bool
        and type(value.get("budget_exhausted")) is bool,
        f"{path}: terminal completion fields must be Boolean",
    )
    for field, expected in (
        ("schema_version", policy["schema_version"]),
        ("policy_version", policy["policy_version"]),
        ("lane_policy_version", policy["lane_policy_version"]),
        ("coordinate_policy", policy["coordinate_policy"]),
        ("measurement_policy", policy["measurement_policy"]),
        ("rounds_planned", policy["rounds"]),
        ("beam_width", policy["beam_width"]),
        ("radius", policy["radius"]),
        (
            "maximum_projection_parent_count",
            1 + int(policy["beam_width"]) * (int(policy["rounds"]) - 1),
        ),
        ("completed", True),
        ("budget_exhausted", False),
        ("search_generation", trial.get("num_generations")),
    ):
        check_equal(value.get(field), expected, f"{path}: terminal {field}")
    require(
        value.get("termination_reason") in {"round_limit", "no_candidates"},
        f"{path}: invalid terminal termination reason",
    )
    _check_terminal_float(
        value.get("minimum_improvement_fraction"),
        float(policy["minimum_improvement_fraction"]),
        f"{path}: terminal minimum improvement",
    )

    manifest = value.get("config_manifest")
    require(
        isinstance(manifest, dict) and manifest and list(manifest) == sorted(manifest),
        f"{path}: terminal config manifest",
    )
    check_equal(
        value.get("config_manifest_sha256"),
        canonical_sha256(manifest),
        f"{path}: terminal config manifest digest",
    )
    for config_id, record in manifest.items():
        require(
            isinstance(config_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
            and isinstance(record, dict)
            and set(record) == {"config"}
            and isinstance(record["config"], dict),
            f"{path}: malformed terminal config manifest entry {config_id!r}",
        )
        check_equal(
            canonical_sha256(record["config"])[:16],
            config_id,
            f"{path}: terminal config manifest ID {config_id}",
        )

    rounds = value.get("rounds")
    require(
        isinstance(rounds, list) and 0 < len(rounds) <= int(policy["rounds"]),
        f"{path}: terminal refinement rounds",
    )
    check_equal(
        value.get("rounds_started"), len(rounds), f"{path}: terminal rounds started"
    )
    check_equal(
        value.get("rounds_completed"),
        len(rounds),
        f"{path}: terminal rounds completed",
    )
    round_fields = {
        "round_index",
        "incumbent_config_id",
        "leaf",
        "parent_config_ids",
        "parent_projections",
        "candidate_config_ids",
        "new_candidate_ids",
        "reused_candidate_ids",
        "intra_terminal_reused_candidate_ids",
        "prior_failed_candidate_ids",
        "candidate_results",
        "comparison_config_ids",
        "measurement",
        "round_best_config_id",
        "selected_config_id",
        "accepted",
        "improvement_fraction",
        "beam_config_ids",
    }
    projection_fields = {
        "flat_index",
        "key",
        "sequence_index",
        "from_value",
        "to_value",
        "outcome",
        "config_id",
    }
    result_fields = {
        "config_id",
        "attempt_perf",
        "selection_perf",
        "status",
        "source_hash",
    }
    allowed_projection_outcomes = {
        "candidate",
        "incumbent_alias",
        "candidate_alias",
        "invalid",
        "different_leaf",
        "beam_alias",
        "round_candidate_alias",
    }

    declared_new_order: list[str] = []
    for round_index, round_value in enumerate(rounds, 1):
        require(
            isinstance(round_value, dict),
            f"{path}: malformed terminal round {round_index}",
        )
        declared_new_order.extend(
            _terminal_config_ids(
                round_value.get("new_candidate_ids"),
                f"{path}: terminal round {round_index} new candidates",
            )
        )
    require(
        len(declared_new_order) == len(set(declared_new_order)),
        f"{path}: terminal candidate is classified as new more than once",
    )
    declared_new_ids = set(declared_new_order)
    metadata_ids = set(metadata_configs)
    require(
        declared_new_ids <= metadata_ids,
        f"{path}: terminal new candidate is absent from autotune metadata",
    )
    require(
        all(
            int(row["generation"]) == trial.get("num_generations")
            for row in source_rows
            if row["config_id"] in declared_new_ids
        ),
        f"{path}: terminal new candidate has the wrong search generation",
    )
    preterminal_ids = metadata_ids - declared_new_ids
    preterminal_tested = strict_int(
        value.get("preterminal_num_configs_tested"),
        f"{path}: preterminal tested count",
        minimum=1,
    )
    total_tested = strict_int(
        trial.get("num_configs_tested"), f"{path}: total tested count", minimum=1
    )
    require(
        preterminal_tested <= total_tested,
        f"{path}: preterminal tested count exceeds total",
    )
    terminal_row_positions = [
        index
        for index, row in enumerate(source_rows)
        if row["config_id"] in declared_new_ids
    ]
    terminal_benchmark_positions = [
        index
        for index, row in enumerate(source_rows)
        if row["config_id"] in declared_new_ids and row["status"] == "started"
    ]
    observed_terminal_new_ids = {
        row["config_id"] for row in source_rows if row["config_id"] in declared_new_ids
    }
    check_equal(
        observed_terminal_new_ids,
        declared_new_ids,
        f"{path}: terminal new candidate sidecar coverage",
    )
    first_terminal_position = min(terminal_row_positions, default=len(source_rows))
    preterminal_rows = source_rows[:first_terminal_position]
    terminal_rows = source_rows[first_terminal_position:]
    require(
        all(row["config_id"] in preterminal_ids for row in preterminal_rows),
        f"{path}: terminal evidence appears before its declared boundary",
    )
    missing_preterminal_evidence = sorted(
        preterminal_ids - {row["config_id"] for row in preterminal_rows}
    )
    require(
        not missing_preterminal_evidence,
        f"{path}: preterminal registry configs lack source evidence before terminal: "
        f"{missing_preterminal_evidence}",
    )
    first_terminal_benchmark_position = min(
        terminal_benchmark_positions, default=len(source_rows)
    )
    chronological_preterminal_started = sum(
        row["status"] == "started"
        for row in source_rows[:first_terminal_benchmark_position]
    )
    terminal_started = [
        row
        for row in terminal_rows
        if row["config_id"] in declared_new_ids and row["status"] == "started"
    ]
    expected_terminal_started = [
        config_id
        for config_id in declared_new_order
        if any(
            row["config_id"] == config_id and row["status"] == "started"
            for row in source_rows
        )
    ]
    check_equal(
        [row["config_id"] for row in terminal_started],
        expected_terminal_started,
        f"{path}: terminal started-row partition",
    )
    check_equal(
        preterminal_tested,
        chronological_preterminal_started,
        f"{path}: preterminal tested count versus chronological started rows",
    )
    terminal_tested = total_tested - preterminal_tested
    check_equal(
        terminal_tested,
        len(terminal_started),
        f"{path}: terminal tested-count delta versus chronological started rows",
    )
    preterminal_id_list = sorted(preterminal_ids)
    check_equal(
        value.get("preterminal_registry_config_count"),
        len(preterminal_id_list),
        f"{path}: preterminal registry count",
    )
    check_equal(
        value.get("preterminal_registry_config_ids_hash_policy"),
        EXPECTED_PRETERMINAL_REGISTRY_HASH_POLICY,
        f"{path}: preterminal registry hash policy",
    )
    check_equal(
        value.get("preterminal_registry_config_ids_sha256"),
        hashlib.sha256(
            json.dumps(preterminal_id_list, separators=(",", ":")).encode()
        ).hexdigest(),
        f"{path}: preterminal registry digest",
    )
    preterminal_aliases = sum(
        row["status"] in LEDGER_ALIAS_STATUSES for row in preterminal_rows
    )
    preterminal_successes = sum(row["status"] == "ok" for row in preterminal_rows)
    exact_ids = phase.get("exact_space_config_ids")
    required_preterminal = (
        len(exact_ids) if isinstance(exact_ids, list) and exact_ids else 100
    )
    preterminal_effective_candidate_count = preterminal_tested + preterminal_aliases

    invalidations = isolated_rebenchmark_invalidations(phase)

    def attempt_before(config_id: str, position: int) -> dict[str, Any] | None:
        history = attempt_history_by_config.get(config_id, [])
        eligible = [attempt for attempt in history if attempt["position"] < position]
        return eligible[-1] if eligible else None

    def effective_attempt(
        config_id: str, *, before_position: int | None = None
    ) -> dict[str, Any]:
        attempt = (
            attempt_by_config.get(config_id)
            if before_position is None
            else attempt_before(config_id, before_position)
        )
        require(
            isinstance(attempt, dict),
            f"{path}: terminal config {config_id} lacks a sidecar outcome",
        )
        invalidation = invalidations.get(config_id)
        return invalidation if invalidation is not None else attempt

    initial_id = value.get("initial_incumbent_config_id")
    require(
        isinstance(initial_id, str)
        and initial_id in preterminal_ids
        and initial_id in manifest,
        f"{path}: invalid terminal initial incumbent",
    )
    initial_leaf = structural_leaf(manifest[initial_id]["config"])
    require(initial_leaf is not None, f"{path}: terminal incumbent has no leaf")
    surface_leaves = surface["leaves"]
    assert isinstance(surface_leaves, list)
    matching_surface_leaves = [
        leaf_record
        for leaf_record in surface_leaves
        if isinstance(leaf_record, dict) and leaf_record.get("leaf") == initial_leaf
    ]
    require(
        len(matching_surface_leaves) == 1,
        f"{path}: terminal incumbent is absent from the coordinate surface",
    )
    surface_leaf = matching_surface_leaves[0]
    expected_manifest_ids = {initial_id}
    terminal_seen_new: set[str] = set()
    unique_candidate_ids: set[str] = set()
    reused_ids: set[str] = set()
    intra_reused_ids: set[str] = set()
    prior_failed_ids: set[str] = set()
    accepted_ids: list[str] = []
    incumbent_id = initial_id
    expected_parent_ids = [initial_id]
    projection_parent_count = 0
    detached_direct_projection_count = 0
    paired_live_projection_required_count = 0
    final_beam = [initial_id]
    stopped_for_no_candidates = False

    raw_round_start_positions: list[int | None] = []
    for round_index, round_value in enumerate(rounds, 1):
        round_new_ids = set(round_value["new_candidate_ids"])
        positions = [
            index
            for index, row in enumerate(source_rows)
            if row["config_id"] in round_new_ids
        ]
        if round_new_ids:
            require(
                positions,
                f"{path}: terminal round {round_index} new candidates lack sidecars",
            )
        raw_round_start_positions.append(min(positions) if positions else None)
    next_start = len(source_rows)
    round_start_positions = [len(source_rows)] * len(rounds)
    for index in range(len(rounds) - 1, -1, -1):
        raw_start = raw_round_start_positions[index]
        if raw_start is not None:
            next_start = raw_start
        round_start_positions[index] = next_start
    require(
        round_start_positions == sorted(round_start_positions),
        f"{path}: terminal round sidecars are out of order",
    )
    initial_state = attempt_before(initial_id, round_start_positions[0])
    require(
        isinstance(initial_state, dict)
        and initial_state.get("status") in {"ok", "deduplicated"},
        f"{path}: terminal initial incumbent was not reusable at the boundary",
    )

    for expected_round, round_value in enumerate(rounds, 1):
        require(
            isinstance(round_value, dict) and set(round_value) == round_fields,
            f"{path}: malformed terminal round {expected_round}",
        )
        require(
            type(round_value.get("round_index")) is int
            and type(round_value.get("accepted")) is bool,
            f"{path}: invalid terminal round field types",
        )
        check_equal(
            round_value.get("round_index"),
            expected_round,
            f"{path}: terminal round index",
        )
        check_equal(
            round_value.get("incumbent_config_id"),
            incumbent_id,
            f"{path}: terminal round incumbent",
        )
        parent_ids = _terminal_config_ids(
            round_value.get("parent_config_ids"),
            f"{path}: terminal round {expected_round} parents",
        )
        check_equal(parent_ids, expected_parent_ids, f"{path}: terminal beam parents")
        require(
            len(parent_ids) <= EXPECTED_TERMINAL_REFINEMENT_BEAM_WIDTH,
            f"{path}: terminal beam exceeds its width",
        )
        projection_parent_count += len(parent_ids)
        check_equal(
            round_value.get("leaf"), initial_leaf, f"{path}: terminal round leaf"
        )
        for parent_id in parent_ids:
            require(
                parent_id in manifest
                and structural_leaf(manifest[parent_id]["config"]) == initial_leaf,
                f"{path}: terminal parent leaves the initial structural leaf",
            )
            expected_manifest_ids.add(parent_id)

        parent_projections = round_value.get("parent_projections")
        require(
            isinstance(parent_projections, list)
            and len(parent_projections) == len(parent_ids),
            f"{path}: terminal parent projection count",
        )
        expected_candidates: list[str] = []
        round_candidate_seen: set[str] = set()
        for parent_id, parent_projection in zip(
            parent_ids, parent_projections, strict=True
        ):
            require(
                isinstance(parent_projection, dict)
                and set(parent_projection)
                == {"parent_config_id", "coordinate_requests"},
                f"{path}: malformed terminal parent projection",
            )
            check_equal(
                parent_projection.get("parent_config_id"),
                parent_id,
                f"{path}: terminal projection parent",
            )
            requests = parent_projection.get("coordinate_requests")
            require(
                isinstance(requests, list),
                f"{path}: terminal parent coordinate requests",
            )
            parent_config = manifest[parent_id]["config"]
            expected_surface_requests = _terminal_surface_requests(
                path, surface_leaf, parent_config
            )
            require(
                len(requests) == len(expected_surface_requests),
                f"{path}: incomplete terminal coordinate projection enumeration",
            )
            local_projection_seen = {parent_id}
            for request, expected_surface_request in zip(
                requests, expected_surface_requests, strict=True
            ):
                require(
                    isinstance(request, dict) and set(request) == projection_fields,
                    f"{path}: malformed terminal coordinate request",
                )
                flat_index = request.get("flat_index")
                key = request.get("key")
                sequence_index = request.get("sequence_index")
                outcome = request.get("outcome")
                config_id = request.get("config_id")
                require(
                    type(flat_index) is int
                    and flat_index >= 0
                    and isinstance(key, str)
                    and key
                    and (
                        sequence_index is None
                        or (type(sequence_index) is int and sequence_index >= 0)
                    )
                    and outcome in allowed_projection_outcomes,
                    f"{path}: invalid terminal coordinate request",
                )
                check_equal(
                    {
                        field: request[field]
                        for field in (
                            "flat_index",
                            "key",
                            "sequence_index",
                            "from_value",
                            "to_value",
                        )
                    },
                    expected_surface_request,
                    f"{path}: terminal coordinate request versus recorded surface",
                )
                require(key in parent_config, f"{path}: terminal request key {key}")
                parent_value = parent_config[key]
                if sequence_index is None:
                    expected_from = parent_value
                else:
                    require(
                        isinstance(parent_value, list)
                        and sequence_index < len(parent_value),
                        f"{path}: terminal request sequence coordinate",
                    )
                    expected_from = parent_value[sequence_index]
                check_equal(
                    request.get("from_value"),
                    expected_from,
                    f"{path}: terminal request from value",
                )
                directly_projected_config = _terminal_direct_projection(
                    path,
                    parent_config,
                    key=key,
                    sequence_index=sequence_index,
                    to_value=request.get("to_value"),
                )
                if outcome == "invalid":
                    check_equal(
                        config_id, None, f"{path}: invalid projection config ID"
                    )
                    paired_live_projection_required_count += 1
                    continue
                require(
                    isinstance(config_id, str) and config_id in manifest,
                    f"{path}: terminal projection config is absent from manifest",
                )
                expected_manifest_ids.add(config_id)
                projected_config = manifest[config_id]["config"]
                if projected_config == directly_projected_config:
                    detached_direct_projection_count += 1
                else:
                    paired_live_projection_required_count += 1
                    if outcome in {"candidate", "different_leaf"}:
                        check_equal(
                            _terminal_coordinate_value(
                                path,
                                projected_config,
                                key=key,
                                sequence_index=sequence_index,
                            ),
                            request.get("to_value"),
                            f"{path}: terminal projected config coordinate value",
                        )
                projected_leaf = structural_leaf(projected_config)
                if outcome == "incumbent_alias":
                    check_equal(
                        config_id, parent_id, f"{path}: terminal incumbent alias"
                    )
                elif outcome == "candidate_alias":
                    require(
                        config_id in local_projection_seen and config_id != parent_id,
                        f"{path}: terminal candidate alias precedes its candidate",
                    )
                elif outcome == "different_leaf":
                    require(
                        projected_leaf != initial_leaf,
                        f"{path}: terminal different-leaf projection stayed in leaf",
                    )
                elif outcome == "beam_alias":
                    require(
                        config_id in set(parent_ids),
                        f"{path}: terminal beam alias is not a parent",
                    )
                elif outcome == "round_candidate_alias":
                    require(
                        config_id in round_candidate_seen,
                        f"{path}: terminal round alias precedes its candidate",
                    )
                else:
                    require(
                        projected_leaf == initial_leaf
                        and config_id not in set(parent_ids)
                        and config_id not in round_candidate_seen,
                        f"{path}: invalid terminal candidate projection",
                    )
                    expected_candidates.append(config_id)
                    round_candidate_seen.add(config_id)
                local_projection_seen.add(config_id)

        candidate_ids = _terminal_config_ids(
            round_value.get("candidate_config_ids"),
            f"{path}: terminal round {expected_round} candidates",
        )
        check_equal(
            candidate_ids,
            expected_candidates,
            f"{path}: terminal candidate projection order",
        )
        unique_candidate_ids.update(candidate_ids)
        expected_new: list[str] = []
        expected_reused: list[str] = []
        expected_intra_reused: list[str] = []
        expected_prior_failed: list[str] = []
        round_start_position = round_start_positions[expected_round - 1]
        round_end_position = (
            round_start_positions[expected_round]
            if expected_round < len(round_start_positions)
            else len(source_rows)
        )
        for config_id in candidate_ids:
            state_before = attempt_before(config_id, round_start_position)
            succeeded_before = state_before is not None and state_before.get(
                "status"
            ) in {"ok", "deduplicated"}
            if config_id in terminal_seen_new:
                (
                    expected_intra_reused if succeeded_before else expected_prior_failed
                ).append(config_id)
            elif config_id in preterminal_ids:
                (expected_reused if succeeded_before else expected_prior_failed).append(
                    config_id
                )
            else:
                require(
                    state_before is None,
                    f"{path}: terminal new candidate has preexisting sidecar evidence",
                )
                expected_new.append(config_id)
                terminal_seen_new.add(config_id)
        for field, expected in (
            ("new_candidate_ids", expected_new),
            ("reused_candidate_ids", expected_reused),
            ("intra_terminal_reused_candidate_ids", expected_intra_reused),
            ("prior_failed_candidate_ids", expected_prior_failed),
        ):
            check_equal(
                round_value.get(field),
                expected,
                f"{path}: terminal round {expected_round} {field}",
            )
        reused_ids.update(expected_reused)
        intra_reused_ids.update(expected_intra_reused)
        prior_failed_ids.update(expected_prior_failed)

        candidate_results = round_value.get("candidate_results")
        require(
            isinstance(candidate_results, list)
            and len(candidate_results) == len(candidate_ids),
            f"{path}: terminal candidate result count",
        )
        comparison_ids = _terminal_config_ids(
            round_value.get("comparison_config_ids"),
            f"{path}: terminal round {expected_round} comparison configs",
        )
        expected_comparison_ids = list(parent_ids)
        for config_id, result in zip(candidate_ids, candidate_results, strict=True):
            require(
                isinstance(result, dict) and set(result) == result_fields,
                f"{path}: malformed terminal candidate result",
            )
            check_equal(
                result.get("config_id"),
                config_id,
                f"{path}: terminal candidate result order",
            )
            state = effective_attempt(
                config_id,
                before_position=round_end_position,
            )
            check_equal(
                result.get("status"),
                state.get("status"),
                f"{path}: terminal candidate status",
            )
            if state.get("status") in {"ok", "deduplicated"}:
                _check_terminal_float(
                    result.get("attempt_perf"),
                    float(state["perf_ms"]),
                    f"{path}: terminal candidate attempt performance",
                )
                check_equal(
                    result.get("source_hash"),
                    state.get("source_hash"),
                    f"{path}: terminal candidate source hash",
                )
                if config_id not in expected_comparison_ids:
                    expected_comparison_ids.append(config_id)
            else:
                check_equal(
                    result.get("attempt_perf"),
                    None,
                    f"{path}: failed terminal candidate attempt performance",
                )
                check_equal(
                    result.get("selection_perf"),
                    None,
                    f"{path}: failed terminal candidate selection performance",
                )
                check_equal(
                    result.get("source_hash"),
                    state.get("source_hash"),
                    f"{path}: failed terminal candidate source hash",
                )
        beam_ids = _terminal_config_ids(
            round_value.get("beam_config_ids"),
            f"{path}: terminal round {expected_round} beam",
        )
        if len(expected_comparison_ids) < 2:
            check_equal(comparison_ids, [], f"{path}: no-candidate comparison")
            check_equal(
                round_value.get("measurement"),
                None,
                f"{path}: no-candidate measurement",
            )
            check_equal(
                round_value.get("round_best_config_id"),
                incumbent_id,
                f"{path}: no-candidate round best",
            )
            check_equal(
                round_value.get("selected_config_id"),
                incumbent_id,
                f"{path}: no-candidate selection",
            )
            check_equal(
                round_value.get("accepted"), False, f"{path}: no-candidate acceptance"
            )
            check_equal(
                round_value.get("improvement_fraction"),
                0.0,
                f"{path}: no-candidate improvement",
            )
            check_equal(beam_ids, parent_ids, f"{path}: no-candidate beam")
            check_equal(
                expected_round,
                len(rounds),
                f"{path}: no-candidate terminal round must be last",
            )
            stopped_for_no_candidates = True
        else:
            check_equal(
                comparison_ids,
                expected_comparison_ids,
                f"{path}: terminal comparison config order",
            )
            medians = _validate_terminal_measurement(
                path,
                round_value.get("measurement"),
                comparison_ids,
                label=f"terminal round {expected_round} measurement",
                expected_target_ms=float(policy["round_target_ms"]),
            )
            for result in candidate_results:
                config_id = result["config_id"]
                if config_id in medians:
                    _check_terminal_float(
                        result.get("selection_perf"),
                        medians[config_id],
                        f"{path}: terminal candidate selection performance",
                    )

            ranked = sorted(
                comparison_ids,
                key=lambda config_id: (medians[config_id], config_id),
            )
            round_best_id = ranked[0]
            check_equal(
                round_value.get("round_best_config_id"),
                round_best_id,
                f"{path}: terminal round best",
            )
            improvement = 1.0 - medians[round_best_id] / medians[incumbent_id]
            _check_terminal_float(
                round_value.get("improvement_fraction"),
                improvement,
                f"{path}: terminal round improvement",
            )
            accepted = round_best_id != incumbent_id and improvement >= float(
                policy["minimum_improvement_fraction"]
            )
            check_equal(
                round_value.get("accepted"), accepted, f"{path}: terminal acceptance"
            )
            if accepted:
                incumbent_id = round_best_id
                if incumbent_id not in accepted_ids:
                    accepted_ids.append(incumbent_id)
            check_equal(
                round_value.get("selected_config_id"),
                incumbent_id,
                f"{path}: terminal selected config",
            )
            expected_beam = [incumbent_id]
            for config_id in ranked:
                if config_id not in expected_beam:
                    expected_beam.append(config_id)
                if len(expected_beam) >= int(policy["beam_width"]):
                    break
            check_equal(beam_ids, expected_beam, f"{path}: terminal ranked beam")
        expected_parent_ids = beam_ids
        final_beam = beam_ids

    if stopped_for_no_candidates:
        check_equal(
            value.get("termination_reason"),
            "no_candidates",
            f"{path}: no-candidate termination",
        )
    else:
        check_equal(len(rounds), int(policy["rounds"]), f"{path}: terminal round count")
        check_equal(
            value.get("termination_reason"),
            "round_limit",
            f"{path}: terminal round-limit termination",
        )

    check_equal(
        value.get("projection_parent_count"),
        projection_parent_count,
        f"{path}: terminal projection parent count",
    )
    check_equal(
        value.get("projection_attempt_count"),
        sum(
            len(parent["coordinate_requests"])
            for round_value in rounds
            for parent in round_value["parent_projections"]
        ),
        f"{path}: terminal projection attempt count",
    )
    check_equal(
        detached_direct_projection_count + paired_live_projection_required_count,
        value.get("projection_attempt_count"),
        f"{path}: detached terminal projection accounting",
    )
    check_equal(
        value.get("accepted_config_ids"),
        accepted_ids,
        f"{path}: terminal accepted config order",
    )
    check_equal(
        value.get("refined_config_id"),
        incumbent_id,
        f"{path}: terminal refined config",
    )

    declared_new_union = {
        config_id
        for round_value in rounds
        for config_id in round_value["new_candidate_ids"]
    }
    check_equal(declared_new_union, declared_new_ids, f"{path}: terminal new IDs")
    check_equal(
        value.get("unique_candidate_count"),
        len(unique_candidate_ids),
        f"{path}: terminal unique candidate count",
    )
    for field, observed in (
        ("new_candidate_count", declared_new_ids),
        ("reused_candidate_count", reused_ids),
        ("intra_terminal_reused_candidate_count", intra_reused_ids),
        ("prior_failed_candidate_count", prior_failed_ids),
    ):
        check_equal(value.get(field), len(observed), f"{path}: terminal {field}")

    confirmation = value.get("confirmation")
    require(
        isinstance(confirmation, dict)
        and set(confirmation)
        == {
            "candidate_config_ids",
            "measurement",
            "best_config_id",
            "selected_config_id",
            "accepted",
            "improvement_fraction",
            "skipped_reason",
        },
        f"{path}: malformed terminal confirmation",
    )
    expected_confirmation_ids: list[str] = []
    for config_id in (initial_id, *accepted_ids, *final_beam):
        if config_id not in expected_confirmation_ids:
            expected_confirmation_ids.append(config_id)
    confirmation_ids = _terminal_config_ids(
        confirmation.get("candidate_config_ids"),
        f"{path}: terminal confirmation candidates",
    )
    require(
        type(confirmation.get("accepted")) is bool,
        f"{path}: terminal confirmation acceptance must be Boolean",
    )
    check_equal(
        confirmation_ids,
        expected_confirmation_ids,
        f"{path}: terminal confirmation candidate order",
    )
    if len(confirmation_ids) == 1:
        final_id = initial_id
        check_equal(
            confirmation.get("measurement"),
            None,
            f"{path}: single-candidate confirmation measurement",
        )
        check_equal(
            confirmation.get("best_config_id"),
            initial_id,
            f"{path}: single-candidate confirmation best",
        )
        check_equal(
            confirmation.get("selected_config_id"),
            initial_id,
            f"{path}: single-candidate confirmation selection",
        )
        check_equal(
            confirmation.get("accepted"),
            False,
            f"{path}: single-candidate confirmation acceptance",
        )
        check_equal(
            confirmation.get("improvement_fraction"),
            0.0,
            f"{path}: single-candidate confirmation improvement",
        )
        check_equal(
            confirmation.get("skipped_reason"),
            "single_candidate",
            f"{path}: single-candidate confirmation reason",
        )
    else:
        check_equal(
            confirmation.get("skipped_reason"),
            None,
            f"{path}: terminal confirmation skip",
        )
        confirmation_medians = _validate_terminal_measurement(
            path,
            confirmation.get("measurement"),
            confirmation_ids,
            label="terminal confirmation measurement",
            expected_target_ms=float(policy["confirmation_target_ms"]),
        )
        confirmed_best_id = min(
            confirmation_ids,
            key=lambda config_id: (confirmation_medians[config_id], config_id),
        )
        check_equal(
            confirmation.get("best_config_id"),
            confirmed_best_id,
            f"{path}: terminal confirmed best",
        )
        confirmation_improvement = (
            1.0
            - confirmation_medians[confirmed_best_id] / confirmation_medians[initial_id]
        )
        _check_terminal_float(
            confirmation.get("improvement_fraction"),
            confirmation_improvement,
            f"{path}: terminal confirmation improvement",
        )
        confirmation_accepted = (
            confirmed_best_id != initial_id
            and confirmation_improvement
            >= float(policy["minimum_improvement_fraction"])
        )
        check_equal(
            confirmation.get("accepted"),
            confirmation_accepted,
            f"{path}: terminal confirmation acceptance",
        )
        final_id = confirmed_best_id if confirmation_accepted else initial_id
        check_equal(
            confirmation.get("selected_config_id"),
            final_id,
            f"{path}: terminal confirmation selection",
        )
    check_equal(value.get("final_config_id"), final_id, f"{path}: terminal winner")
    selected_config = trial.get("selected_config")
    require(isinstance(selected_config, dict), f"{path}: missing trial winner")
    check_equal(
        canonical_sha256(selected_config)[:16],
        final_id,
        f"{path}: terminal winner versus trial winner",
    )
    final_state = effective_attempt(final_id)
    require(
        final_state.get("status") in {"ok", "deduplicated"},
        f"{path}: terminal winner lacks a successful source",
    )

    expected_manifest_ids.update(
        config_id
        for round_value in rounds
        for parent in round_value["parent_projections"]
        for request in parent["coordinate_requests"]
        if (config_id := request["config_id"]) is not None
    )
    expected_manifest_ids.update(confirmation_ids)
    check_equal(
        set(manifest), expected_manifest_ids, f"{path}: terminal config manifest set"
    )

    summary = {
        key: value[key]
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
    summary["required_preterminal_candidate_count"] = required_preterminal
    summary["preterminal_effective_candidate_count"] = (
        preterminal_effective_candidate_count
    )
    summary["preterminal_successful_measurement_count"] = preterminal_successes
    summary["detached_direct_projection_count"] = detached_direct_projection_count
    summary["paired_live_projection_required_count"] = (
        paired_live_projection_required_count
    )
    summary["policy_sha256"] = canonical_sha256(policy)
    summary["coordinate_surface_sha256"] = canonical_sha256(surface)
    summary["rounds_sha256"] = canonical_sha256(rounds)
    summary["confirmation_sha256"] = canonical_sha256(confirmation)
    summary["transcript_sha256"] = canonical_sha256(value)
    return summary


def validate_structural_prefix_execution(
    path: Path,
    provenance: dict[str, Any],
    trial: dict[str, Any],
    source_rows: list[dict[str, str]],
    metadata_configs: dict[str, dict[str, Any]],
    attempt_by_config: dict[str, dict[str, Any]],
    attempt_history_by_config: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    phase = validate_structural_qualification_phase(path, provenance, trial)
    invalidations = isolated_rebenchmark_invalidations(phase)
    timed_out_sources = isolated_rebenchmark_timeout_source_hashes(phase)
    require(
        trial["num_isolated_rebenchmark_timeouts"] >= len(timed_out_sources),
        f"{path}: fewer isolated rebenchmark timeouts than distinct timed-out "
        "generated sources",
    )
    validate_timeline_source_repairs(path, phase, attempt_history_by_config)
    for config_id, invalidation in invalidations.items():
        attempt = attempt_by_config.get(config_id)
        require(
            isinstance(attempt, dict)
            and attempt.get("status") in {"ok", "deduplicated"}
            and attempt.get("source_hash") == invalidation["source_hash"],
            f"{path}: isolated rebenchmark invalidation lacks a matching "
            "successful sidecar source",
        )
    reconciled_phase = reconcile_structural_qualification_phase(
        path,
        provenance,
        phase,
        attempt_by_config,
        metadata_configs,
    )
    design = provenance["flash_structural_coverage_design"]
    design_configs = [item["config"] for item in design]
    injected_design_count = provenance["flash_structural_injected_design_count"]
    prefix_configs = design_configs[:injected_design_count]
    require(
        0
        < len(prefix_configs)
        <= provenance["flash_structural_population_budget"]
        <= provenance["autotune_initial_population_size"],
        f"{path}: injected structural design prefix exceeds its dynamic budget",
    )
    prefix_ids = [canonical_sha256(config)[:16] for config in prefix_configs]
    require(
        len(set(prefix_ids)) == len(prefix_ids),
        f"{path}: structural design prefix has a config ID collision",
    )

    attempted_ids: set[str] = set()
    successful_prefix_ids: set[str] = set()
    for config_id, config in zip(prefix_ids, prefix_configs, strict=True):
        check_equal(
            metadata_configs.get(config_id),
            config,
            f"{path}: structural prefix metadata config {config_id}",
        )
        generation_zero = [
            row
            for row in source_rows
            if row["config_id"] == config_id and row["generation"] == "0"
        ]
        require(
            any(
                row["status"] in ({"started"} | LEDGER_ALIAS_STATUSES)
                for row in generation_zero
            ),
            f"{path}: structural prefix config {config_id} was not attempted at generation 0",
        )
        attempted_ids.add(config_id)
        if config_id not in invalidations and any(
            row["status"] in {"ok", "deduplicated"} for row in generation_zero
        ):
            successful_prefix_ids.add(config_id)

    generation_zero_attempted_ids = {
        row["config_id"]
        for row in source_rows
        if row["generation"] == "0"
        and row["status"] in ({"started"} | LEDGER_ALIAS_STATUSES)
    }
    initial_population_size = strict_int(
        provenance.get("autotune_initial_population_size"),
        f"{path}: initial population size",
        minimum=1,
    )
    expected_generation_zero_count = (
        len(phase["exact_space_config_ids"])
        if phase["exact_space_enumerated"]
        else initial_population_size
    )
    initial_population_order = phase["initial_config_ids"]
    check_equal(
        initial_population_order,
        generation_zero_initial_config_order(
            path, source_rows, expected_generation_zero_count
        ),
        f"{path}: generation-zero initial population order",
    )
    initial_population_ids = set(initial_population_order)
    check_equal(
        len(initial_population_ids),
        expected_generation_zero_count,
        f"{path}: distinct initial-population config count",
    )
    require(
        initial_population_ids <= generation_zero_attempted_ids,
        f"{path}: initial population is absent from generation-0 sidecars",
    )
    successful_generation_zero_ids = {
        row["config_id"]
        for row in source_rows
        if row["generation"] == "0"
        and row["status"] in {"ok", "deduplicated"}
        and row["config_id"] not in invalidations
        and row["config_id"] in initial_population_ids
    }
    compiler_seed_policy = validate_compiler_seed_policy(
        path,
        provenance,
        phase=phase,
        metadata_configs=metadata_configs,
        source_rows=source_rows,
        invalidated_config_ids=set(invalidations),
    )
    if phase["exact_space_enumerated"]:
        require(
            set(phase["exact_space_config_ids"]) <= successful_generation_zero_ids,
            f"{path}: exact structural search space was not successfully exhausted",
        )
    successful_generation_zero_configs = [
        metadata_configs[config_id] for config_id in successful_generation_zero_ids
    ]

    active_values = provenance["flash_structural_coverage_active_values"]
    qualification_values = structural_qualification_values(active_values)

    def witness_counts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **value,
                "successful_generation_zero_witness_count": sum(
                    config.get(value["key"]) == value["value"]
                    for config in successful_generation_zero_configs
                ),
            }
            for value in values
        ]

    active_counts = witness_counts(active_values)
    qualification_counts = witness_counts(qualification_values)
    active_interactions = provenance["flash_structural_coverage_active_interactions"]
    interaction_counts = [
        {
            **interaction,
            "successful_generation_zero_witness_count": sum(
                all(
                    config.get(key) == value
                    for key, value in zip(
                        interaction["keys"], interaction["values"], strict=True
                    )
                )
                for config in successful_generation_zero_configs
            ),
        }
        for interaction in active_interactions
    ]
    missing_qualification = [
        item
        for item in qualification_counts
        if item["successful_generation_zero_witness_count"] < 1
    ]
    require(
        not missing_qualification,
        f"{path}: qualified structural values lack a successful generation-0 "
        f"representative: {missing_qualification!r}",
    )
    missing_interactions = [
        item
        for item in interaction_counts
        if item["successful_generation_zero_witness_count"] < 1
    ]
    require(
        not missing_interactions,
        f"{path}: structural interactions lack a successful generation-0 "
        f"representative: {missing_interactions!r}",
    )
    terminal_refinement = validate_terminal_coordinate_refinement(
        path,
        provenance,
        trial,
        phase,
        source_rows,
        metadata_configs,
        attempt_by_config,
        attempt_history_by_config,
    )
    return {
        "compiler_seed_policy": compiler_seed_policy,
        "prefix_count": len(prefix_configs),
        "attempted_count": len(attempted_ids),
        "successful_count": len(successful_prefix_ids),
        "active_value_count": len(active_values),
        "active_value_successful_witness_count": sum(
            item["successful_generation_zero_witness_count"] > 0
            for item in active_counts
        ),
        "active_value_successful_witness_counts": active_counts,
        "qualification_value_count": len(qualification_values),
        "qualification_successful_witness_count": sum(
            item["successful_generation_zero_witness_count"] > 0
            for item in qualification_counts
        ),
        "qualification_successful_witness_counts": qualification_counts,
        "interaction_count": len(active_interactions),
        "interaction_successful_witness_count": sum(
            item["successful_generation_zero_witness_count"] > 0
            for item in interaction_counts
        ),
        "interaction_successful_witness_counts": interaction_counts,
        "structural_qualification_leaf_count": phase["leaf_count"],
        "structural_qualification_leaves_with_candidates": phase[
            "leaves_with_candidates"
        ],
        "structural_qualification_candidate_count": phase["candidate_count"],
        "structural_qualification_successful_candidate_count": len(
            reconciled_phase["successful_candidate_ids"]
        ),
        "structural_qualification_retained_family_count": len(
            phase["retained_families"]
        ),
        "structural_qualification_retained_path_count": phase["retained_path_count"],
        "structural_qualification_leaf_results": reconciled_phase["leaf_results"],
        "terminal_refinement": terminal_refinement,
    }


def validate_result(
    artifact_root: Path, path: Path, variant: str, seq_len: int
) -> dict[str, object]:
    payload = load_json_object(path)
    case_settings = EXPECTED_CASES[(variant, seq_len)]
    physical_gpu = case_settings["physical_gpu"]
    tuner_seed = case_settings["tuner_seed"]
    check_equal(payload.get("impl"), "helion-cute", f"{path}: implementation")
    version = payload.get("version")
    require(
        isinstance(version, str) and ".dirty" not in version,
        f"{path}: dirty or invalid version",
    )
    check_equal(payload.get("gpu"), "NVIDIA B200", f"{path}: GPU model")
    check_equal(str(payload.get("physical_gpu")), str(physical_gpu), f"{path}: GPU")
    power_cap = finite_float(payload.get("power_cap_w"), f"{path}: power cap")
    check_equal(power_cap, EXPECTED_POWER_CAP_W, f"{path}: power cap")
    input_seed = strict_int(payload.get("input_seed"), f"{path}: input seed")
    check_equal(input_seed, EXPECTED_INPUT_SEED, f"{path}: input seed")
    median_ms, median_tflops = validate_shape_and_timing(
        path, payload, variant, seq_len
    )
    provenance, trial, coverage_count, coverage_member, coverage_distance = (
        validate_strict_provenance(path, payload, variant, seq_len, tuner_seed)
    )
    checkout_commit = provenance.get("helion_checkout_git_commit")
    require(
        isinstance(checkout_commit, str)
        and re.fullmatch(r"[0-9a-f]{40}", checkout_commit) is not None
        and checkout_commit == EXPECTED_MEASURED_COMMIT
        and helion_cute_version_matches_commit(version, checkout_commit),
        f"{path}: version or Git revision does not identify the measured campaign commit",
    )
    selected_config = provenance["selected_config"]
    selected_source = provenance["selected_source_sha256"]
    ledger_path = path.with_name("autotune.sources.csv")
    ledger = read_and_validate_ledger(
        ledger_path, trial, selected_config, selected_source
    )
    autotune_csv_path = path.with_name("autotune.csv")
    autotune_metadata_path = path.with_name("autotune.meta.jsonl")
    autotune = validate_autotune_sidecars(
        autotune_csv_path,
        autotune_metadata_path,
        ledger["rows"],
        ledger["run_id"],
        selected_config,
        variant,
        seq_len,
        tuner_seed,
    )
    structural_execution = validate_structural_prefix_execution(
        path,
        provenance,
        trial,
        ledger["rows"],
        autotune["configs"],
        autotune["attempt_by_config"],
        autotune["attempt_history_by_config"],
    )
    terminal_refinement = structural_execution["terminal_refinement"]
    required_preterminal = terminal_refinement["required_preterminal_candidate_count"]
    require(
        terminal_refinement["preterminal_effective_candidate_count"]
        >= required_preterminal,
        f"{path}: terminal work masks an undersized full-search candidate set",
    )
    if not trial["search_phase_metrics"]["exact_space_config_ids"]:
        require(
            terminal_refinement["preterminal_successful_measurement_count"]
            >= required_preterminal,
            f"{path}: terminal work masks fewer than 100 measured preterminal successes",
        )
    root = artifact_root.expanduser().resolve()
    return {
        "case": f"{variant}_{seq_len}",
        "variant": variant,
        "seq_len": seq_len,
        "causal": int(variant == "causal"),
        "dtype": "float16",
        "z": 2,
        "h": 32,
        "head_dim": 64,
        "version": payload["version"],
        "gpu": payload["gpu"],
        "physical_gpu": physical_gpu,
        "power_cap_w": power_cap,
        "input_seed": input_seed,
        "tuner_seed": tuner_seed,
        "benchmark_timer": payload["benchmark_timer"],
        "result_path": path.resolve().relative_to(root).as_posix(),
        "result_sha256": file_sha256(path),
        "autotune_csv_path": autotune_csv_path.resolve().relative_to(root).as_posix(),
        "autotune_csv_sha256": autotune["csv_sha256"],
        "autotune_metadata_path": autotune_metadata_path.resolve()
        .relative_to(root)
        .as_posix(),
        "autotune_metadata_sha256": autotune["metadata_sha256"],
        "autotune_run_id": autotune["run_id"],
        "autotune_metadata_config_count": autotune["config_count"],
        "compiler_seed_policy_json": json.dumps(
            structural_execution["compiler_seed_policy"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        "source_ledger_path": ledger_path.resolve().relative_to(root).as_posix(),
        "source_ledger_sha256": file_sha256(ledger_path),
        "source_ledger_run_id": ledger["run_id"],
        "median_ms": median_ms,
        "median_tflops": median_tflops,
        "search_algorithm": trial["search_algorithm"],
        "num_configs_tested": trial["num_configs_tested"],
        "num_successful_candidate_measurements": trial[
            "num_successful_candidate_measurements"
        ],
        "num_unique_sources": trial["num_unique_sources"],
        "num_source_deduplications": trial["num_source_deduplications"],
        "num_compile_failures": trial["num_compile_failures"],
        "num_worker_failures": trial["num_worker_failures"],
        "num_isolated_rebenchmark_timeouts": trial["num_isolated_rebenchmark_timeouts"],
        "num_accuracy_failures": trial["num_accuracy_failures"],
        "num_generations": trial["num_generations"],
        "autotune_time_seconds": trial["autotune_time"],
        "coverage_design_count": coverage_count,
        "coverage_design_sha256": provenance["flash_structural_coverage_design_sha256"],
        "coverage_design_prefix_count": structural_execution["prefix_count"],
        "coverage_design_prefix_gen0_attempted_count": structural_execution[
            "attempted_count"
        ],
        "coverage_design_prefix_gen0_successful_count": structural_execution[
            "successful_count"
        ],
        "coverage_active_value_count": structural_execution["active_value_count"],
        "coverage_active_value_successful_witness_count": structural_execution[
            "active_value_successful_witness_count"
        ],
        "coverage_active_value_successful_witness_counts_json": canonical_json(
            structural_execution["active_value_successful_witness_counts"]
        ),
        "coverage_qualification_value_count": structural_execution[
            "qualification_value_count"
        ],
        "coverage_qualification_successful_witness_count": structural_execution[
            "qualification_successful_witness_count"
        ],
        "coverage_qualification_successful_witness_counts_json": canonical_json(
            structural_execution["qualification_successful_witness_counts"]
        ),
        "coverage_interaction_count": structural_execution["interaction_count"],
        "coverage_interaction_successful_witness_count": structural_execution[
            "interaction_successful_witness_count"
        ],
        "coverage_interaction_successful_witness_counts_json": canonical_json(
            structural_execution["interaction_successful_witness_counts"]
        ),
        "structural_qualification_leaf_count": structural_execution[
            "structural_qualification_leaf_count"
        ],
        "structural_qualification_leaves_with_candidates": structural_execution[
            "structural_qualification_leaves_with_candidates"
        ],
        "structural_qualification_candidate_count": structural_execution[
            "structural_qualification_candidate_count"
        ],
        "structural_qualification_successful_candidate_count": structural_execution[
            "structural_qualification_successful_candidate_count"
        ],
        "structural_qualification_retained_family_count": structural_execution[
            "structural_qualification_retained_family_count"
        ],
        "structural_qualification_retained_path_count": structural_execution[
            "structural_qualification_retained_path_count"
        ],
        "structural_qualification_leaf_results_json": canonical_json(
            structural_execution["structural_qualification_leaf_results"]
        ),
        "terminal_refinement_policy_sha256": structural_execution[
            "terminal_refinement"
        ]["policy_sha256"],
        "terminal_coordinate_surface_sha256": structural_execution[
            "terminal_refinement"
        ]["coordinate_surface_sha256"],
        "terminal_refinement_json": canonical_json(
            structural_execution["terminal_refinement"]
        ),
        "terminal_refinement_sha256": canonical_sha256(
            structural_execution["terminal_refinement"]
        ),
        "winner_is_coverage_design_member": str(coverage_member).lower(),
        "winner_to_coverage_field_distance": coverage_distance,
        "selected_config_json": canonical_json(selected_config),
        "selected_config_sha256": canonical_sha256(selected_config),
        "selected_source_sha256": selected_source,
        "selected_source_ledger_config_id": ledger["selected_config_id"],
        "selected_source_ledger_generation": ledger["selected_generation"],
        "final_correctness_launches": provenance["final_correctness_launches"],
        "final_repeatability_passed": str(
            provenance["final_repeatability_passed"]
        ).lower(),
    }


def build_manifest(artifact_root: Path) -> str:
    root = artifact_root.expanduser().resolve()
    results = discover_results(root)
    rows = [
        validate_result(root, results[(variant, seq_len)], variant, seq_len)
        for variant, seq_len, _physical_gpu, _tuner_seed in CASES
    ]
    versions = {row["version"] for row in rows}
    input_seeds = {row["input_seed"] for row in rows}
    require(len(versions) == 1, f"all8 versions differ: {sorted(versions)}")
    check_equal(input_seeds, {EXPECTED_INPUT_SEED}, "all8 input seeds")
    for variant in ("dense", "causal"):
        coverage_hashes = {
            str(row["coverage_design_sha256"])
            for row in rows
            if row["variant"] == variant
        }
        require(
            len(coverage_hashes) == 1,
            f"{variant} shapes use length-dependent structural coverage designs "
            "within one legality class: "
            f"{sorted(coverage_hashes)}",
        )
        compiler_seed_policies = {
            str(row["compiler_seed_policy_json"])
            for row in rows
            if row["variant"] == variant
        }
        require(
            len(compiler_seed_policies) == 1,
            f"{variant} shapes use length-dependent compiler seed policies "
            "within one legality class",
        )
        terminal_policies = {
            str(row["terminal_refinement_policy_sha256"])
            for row in rows
            if row["variant"] == variant
        }
        require(
            len(terminal_policies) == 1,
            f"{variant} shapes use length-dependent terminal refinement policies "
            "within one legality class",
        )
        terminal_surfaces = {
            str(row["terminal_coordinate_surface_sha256"])
            for row in rows
            if row["variant"] == variant
        }
        require(
            len(terminal_surfaces) == 1,
            f"{variant} shapes use length-dependent terminal coordinate surfaces "
            "within one legality class",
        )

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=MANIFEST_FIELDS, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate strict generalized all8 artifacts and write a CSV manifest."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def strict_evidence_paths(artifact_root: Path) -> set[Path]:
    results = discover_results(artifact_root)
    paths = set()
    for result_path in results.values():
        paths.add(result_path.resolve())
        paths.update(
            result_path.with_name(filename).resolve()
            for filename in (
                "autotune.csv",
                "autotune.meta.jsonl",
                "autotune.sources.csv",
            )
        )
    return paths


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    require(
        output not in strict_evidence_paths(args.artifact_root),
        f"manifest output collides with validated strict evidence: {output}",
    )
    contents = build_manifest(args.artifact_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(contents)
    print(f"wrote {len(CASES)} validated rows to {output}")


if __name__ == "__main__":
    main()
