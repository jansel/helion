from __future__ import annotations

import argparse
import ast
import contextlib
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from operator import itemgetter
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Protocol

import build_strict_manifest as strict
import torch

if TYPE_CHECKING:
    from helion.autotuner.config_spec import ConfigSpec


class _BoundWithConfigSpec(Protocol):
    @property
    def config_spec(self) -> ConfigSpec: ...


EXPECTED_POWER_CAP_W = 750.0
EXPECTED_FINAL_CORRECTNESS_LAUNCHES = 64
EXPECTED_NEIGHBOR_GENERATION_LIMIT = 200
EXPECTED_LANE_POLICY_VERSION = 14
EXPECTED_RETAINED_FAMILY_CAP = 4
EXPECTED_FAMILY_PROBE_GENERATIONS = 1
EXPECTED_FAMILY_PROBE_CANDIDATES_PER_PATH = 20
EXPECTED_CONDITIONAL_CANDIDATES_PER_PIPELINE_LANE = 1
EXPECTED_QUALIFICATION_FAILURE_RETRIES = 1
FLASH_PIPELINE_FAMILY_KEY = "cute_flash_pipeline_family"
FLASH_EXP2_PACKET_KEY = "cute_flash_exp2_packet"
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
SOURCE_LEDGER_FIELDS = (
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
AUTOTUNE_JOIN_FIELDS = SOURCE_LEDGER_FIELDS[:-1]
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
SOURCE_TERMINAL_STATUSES = frozenset(
    {"ok", "error", "timeout", "peer_compilation_fail", "accuracy_error"}
)
SOURCE_REPAIRABLE_FAILURE_STATUSES = frozenset(
    {"error", "timeout", "peer_compilation_fail"}
)
SOURCE_ALIAS_STATUSES = frozenset({"deduplicated", "source_rejected"})
SOURCE_STATUSES = SOURCE_TERMINAL_STATUSES | SOURCE_ALIAS_STATUSES | {"started"}
PEAKY_STRESS_THRESHOLDS = {
    "atol": 0.002,
    "rtol": 0.01,
    "max_abs_exclusive": 0.01,
    "nrmse_exclusive": 0.002,
    "mismatch_fraction_exclusive": 1e-5,
}
REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_UNTRACKED_HARNESS_PATHS = frozenset(
    {
        ".validation/generalized_paired/build_strict_manifest.py",
        ".validation/generalized_paired/combine_results.py",
        ".validation/generalized_paired/paired_worker.py",
        ".validation/generalized_paired/run_all8.py",
        ".validation/generalized_paired/test_build_strict_manifest.py",
        ".validation/generalized_paired/test_static.py",
    }
)


@dataclass(frozen=True)
class Case:
    variant: str
    seq_len: int
    physical_gpu: int

    @property
    def causal(self) -> bool:
        return self.variant == "causal"

    @property
    def name(self) -> str:
        return f"{self.variant}_{self.seq_len}"


CASES = {
    (variant, seq_len): Case(variant, seq_len, physical_gpu)
    for variant, physical_gpu, seq_lens in (
        ("dense", 7, (32768, 65536, 131072, 262144)),
        ("causal", 6, (65536, 131072, 262144, 524288)),
    )
    for seq_len in seq_lens
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_worker_harness_sha256(expected: str) -> str:
    actual = sha256(Path(__file__).resolve())
    check_equal(actual, expected, "paired worker harness SHA256")
    return actual


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


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
        type(size) is int
        and 0 < size <= provenance.get("autotune_initial_population_size", 0)
        and isinstance(config_ids, list)
        and len(config_ids) == size
        and len(config_ids) == len(set(config_ids))
        and all(
            isinstance(config_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
            for config_id in config_ids
        )
        and digest
        == hashlib.sha256(
            json.dumps(config_ids, separators=(",", ":")).encode()
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
        "softmax_disc": bool(config.get("cute_flash_softmax_disc")),
    }


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
        leaf_key = json.dumps(expected_leaf, sort_keys=True, separators=(",", ":"))
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
    """Replay v16's CLC matrix identity and successful axis coverage."""
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
    result_path: Path,
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
    """Replay a v16 CLC depth decision at its pre-combination boundary."""
    depth_selection = result.get("depth_selection")
    if depth_selection is None:
        return
    require(
        isinstance(depth_selection, dict)
        and set(depth_selection) == {"candidate_results", "selected_representatives"},
        f"{result_path}: invalid immutable CLC depth decision for {result['family']}",
    )
    expected_ids = {
        config_id
        for config_id in successful_ids & pre_combination_ids
        if structural_leaf(metadata_configs[config_id]) == leaf
    }
    if not result["combination_required"]:
        expected_ids.clear()
    expected_results = sorted(
        (successful_results_by_id[config_id] for config_id in expected_ids),
        key=itemgetter("selection_perf", "config_id"),
    )
    check_equal(
        depth_selection["candidate_results"],
        expected_results,
        f"{result_path}: immutable CLC depth candidates for {result['family']}",
    )
    members = [
        {
            "config_id": snapshot["config_id"],
            "selection_perf": snapshot["selection_perf"],
            "pipeline_lanes": config_pipeline_lanes(
                metadata_configs[snapshot["config_id"]], pipeline_lanes
            ),
        }
        for snapshot in expected_results
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
        f"{result_path}: immutable CLC depth representatives for {result['family']}",
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


def check_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git_output(*args: str, repo_root: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_untracked_paths(repo_root: Path, *, ignored: bool) -> list[str]:
    command = ["git", "ls-files", "--others", "--exclude-standard", "-z"]
    if ignored:
        command.insert(3, "--ignored")
    output = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(path for path in output.split("\0") if path)


def validate_checkout(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    head = git_output("rev-parse", "HEAD", repo_root=repo_root)
    subprocess.run(["git", "diff", "--quiet"], cwd=repo_root, check=True)
    subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_root, check=True)
    untracked = _git_untracked_paths(repo_root, ignored=False)
    unexpected = sorted(set(untracked) - ALLOWED_UNTRACKED_HARNESS_PATHS)
    if unexpected:
        raise RuntimeError(
            "checkout has unexpected untracked paths: " + ", ".join(unexpected)
        )
    for path in untracked:
        candidate = repo_root / path
        require(
            candidate.is_file() and not candidate.is_symlink(),
            f"allowed staged harness path is not a regular file: {path}",
        )
    ignored = _git_untracked_paths(repo_root, ignored=True)
    if ignored:
        raise RuntimeError(
            "checkout has ignored paths; caches must be external: " + ", ".join(ignored)
        )
    return {
        "runtime_checkout": str(repo_root),
        "runtime_git_head": head,
        "runtime_tracked_clean": True,
        "runtime_untracked_clean": True,
        "runtime_allowed_untracked_files": untracked,
        "runtime_ignored_clean": True,
        "runtime_allowed_ignored_file_count": 0,
        "runtime_allowed_ignored_files_sha256": canonical_sha256([]),
    }


def _case_from_result(path: Path, payload: dict[str, Any]) -> Case:
    shape = payload.get("shape")
    require(isinstance(shape, dict), f"{path}: missing shape object")
    causal = shape.get("causal")
    require(causal in (0, 1, False, True), f"{path}: invalid causal value")
    variant = "causal" if bool(causal) else "dense"
    seq_len = shape.get("seq_len")
    require(
        isinstance(seq_len, int) and not isinstance(seq_len, bool),
        f"{path}: invalid sequence length",
    )
    case = CASES.get((variant, seq_len))
    require(case is not None, f"{path}: unsupported Helion+CuTe shape {shape!r}")
    return case


def discover_result_paths(artifact_root: Path) -> dict[Case, Path]:
    artifact_root = artifact_root.expanduser().resolve()
    require(
        artifact_root.is_dir(), f"artifact root is not a directory: {artifact_root}"
    )
    found: dict[Case, Path] = {}
    for path in sorted(artifact_root.rglob("result.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unable to read {path}: {exc}") from exc
        require(isinstance(payload, dict), f"{path}: result must be a JSON object")
        if payload.get("impl") != "helion-cute":
            continue
        case = _case_from_result(path, payload)
        if case in found:
            raise RuntimeError(
                f"duplicate strict results for {case.name}: {found[case]} and {path}"
            )
        found[case] = path
    missing = set(CASES.values()) - set(found)
    extra = set(found) - set(CASES.values())
    if missing or extra:
        raise RuntimeError(
            "strict result set mismatch: "
            f"missing={[case.name for case in sorted(missing, key=lambda x: x.name)]}, "
            f"extra={[case.name for case in sorted(extra, key=lambda x: x.name)]}"
        )
    return found


def _validate_version(version: object, checkout_head: str, path: Path) -> str:
    require(isinstance(version, str), f"{path}: missing version string")
    require(".dirty" not in version, f"{path}: search checkout was dirty: {version}")
    match = re.fullmatch(r"Helion [^;]*\+g([0-9a-f]{7,40}); CuTe ([^;\s]+)", version)
    require(match is not None, f"{path}: unexpected version string {version!r}")
    require(
        checkout_head.startswith(match.group(1)),
        f"{path}: Helion git revision: {match.group(1)!r} is not a prefix of "
        f"{checkout_head!r}",
    )
    check_equal(match.group(2), strict.EXPECTED_CUTE_VERSION, f"{path}: CuTe version")
    return match.group(2)


def _validate_coverage(provenance: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    design = provenance.get("flash_structural_coverage_design")
    require(isinstance(design, list) and design, f"{path}: missing coverage design")
    check_equal(
        provenance.get("flash_structural_coverage_design_count"),
        len(design),
        f"{path}: coverage design count",
    )
    configs: list[dict[str, Any]] = []
    for index, candidate in enumerate(design):
        require(isinstance(candidate, dict), f"{path}: coverage entry {index} invalid")
        config = candidate.get("config")
        require(isinstance(config, dict), f"{path}: coverage config {index} invalid")
        check_equal(
            candidate.get("config_sha256"),
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
    catalog_keys = [
        json.dumps(leaf, sort_keys=True, separators=(",", ":")) for leaf in leaf_catalog
    ]
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
        json.dumps(leaf, sort_keys=True, separators=(",", ":"))
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
    check_equal(
        provenance.get("flash_structural_coverage_interaction_key_groups"),
        [list(group) for group in FLASH_INTERACTION_KEY_GROUPS],
        f"{path}: declared structural interaction groups",
    )
    active_values = provenance.get("flash_structural_coverage_active_values")
    require(
        isinstance(active_values, list) and active_values,
        f"{path}: missing active structural values",
    )
    for active in active_values:
        require(
            isinstance(active, dict)
            and set(active) == {"key", "value"}
            and isinstance(active["key"], str),
            f"{path}: malformed active structural value {active!r}",
        )
        require(
            any(config.get(active["key"]) == active.get("value") for config in configs),
            f"{path}: coverage design omits {active!r}",
        )
    active_interactions = provenance.get(
        "flash_structural_coverage_active_interactions"
    )
    require(
        isinstance(active_interactions, list) and active_interactions,
        f"{path}: missing active structural interactions",
    )
    for interaction in active_interactions:
        require(
            isinstance(interaction, dict)
            and set(interaction) == {"keys", "values"}
            and isinstance(interaction["keys"], list)
            and isinstance(interaction["values"], list)
            and tuple(interaction["keys"]) in FLASH_INTERACTION_KEY_GROUPS
            and len(interaction["keys"]) == len(interaction["values"]),
            f"{path}: malformed active structural interaction {interaction!r}",
        )
    recorded_interactions = {
        json.dumps(interaction, sort_keys=True, separators=(",", ":"))
        for interaction in active_interactions
    }
    require(
        len(recorded_interactions) == len(active_interactions),
        f"{path}: duplicate active structural interactions",
    )
    active_value_keys = {active["key"] for active in active_values}
    expected_interactions = {
        json.dumps(
            {
                "keys": list(group),
                "values": [config.get(key) for key in group],
            },
            sort_keys=True,
            separators=(",", ":"),
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
    parent_prefix_count = provenance.get(
        "flash_structural_parent_coverage_prefix_count"
    )
    qualification_prefix_count = provenance.get(
        "flash_structural_qualification_prefix_count"
    )
    population_budget = provenance.get("flash_structural_population_budget")
    require(
        isinstance(parent_prefix_count, int)
        and not isinstance(parent_prefix_count, bool)
        and parent_prefix_count >= 1,
        f"{path}: invalid parent coverage prefix count",
    )
    require(
        isinstance(qualification_prefix_count, int)
        and not isinstance(qualification_prefix_count, bool)
        and qualification_prefix_count >= parent_prefix_count,
        f"{path}: invalid qualification prefix count",
    )
    require(
        isinstance(population_budget, int)
        and not isinstance(population_budget, bool)
        and population_budget >= 1,
        f"{path}: invalid structural population budget",
    )
    initial_population_size = provenance.get("autotune_initial_population_size")
    require(
        isinstance(initial_population_size, int)
        and not isinstance(initial_population_size, bool)
        and initial_population_size > 0,
        f"{path}: invalid initial population size",
    )
    half_population = initial_population_size // 2
    if len(configs) <= initial_population_size:
        expected_population_budget = max(half_population, len(configs))
    elif qualification_prefix_count <= half_population:
        expected_population_budget = half_population
    else:
        expected_population_budget = min(
            initial_population_size,
            max(half_population, parent_prefix_count),
        )
    check_equal(
        population_budget,
        expected_population_budget,
        f"{path}: structural population budget",
    )
    injected_design_count = provenance.get("flash_structural_injected_design_count")
    require(
        isinstance(injected_design_count, int)
        and not isinstance(injected_design_count, bool)
        and injected_design_count >= 1,
        f"{path}: invalid injected structural design count",
    )
    check_equal(
        injected_design_count,
        min(population_budget, len(configs)),
        f"{path}: injected structural design count",
    )
    for interaction in active_interactions:
        require(
            any(
                all(
                    config.get(key) == value
                    for key, value in zip(
                        interaction["keys"], interaction["values"], strict=True
                    )
                )
                for config in configs[:injected_design_count]
            ),
            f"{path}: injected prefix omits structural interaction {interaction!r}",
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
    require(parent_values, f"{path}: no active structural parent values")
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
    fragment_default = provenance.get("flash_fragment_default_config")
    require(isinstance(fragment_default, dict), f"{path}: missing fragment default")
    check_equal(
        provenance.get("flash_fragment_default_sha256"),
        canonical_sha256(fragment_default),
        f"{path}: fragment default digest",
    )
    return configs


def _config_id_list(value: object, label: str) -> list[str]:
    require(
        isinstance(value, list)
        and len(value) == len(set(value))
        and all(
            isinstance(config_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
            for config_id in value
        ),
        f"{label}: invalid config ID list",
    )
    return value


def _positive_int_list(value: object, label: str) -> list[int]:
    require(
        isinstance(value, list)
        and len(value) == len(set(value))
        and all(type(item) is int and item > 0 for item in value),
        f"{label}: invalid positive integer list",
    )
    return value


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
    """Validate and replay the immutable v16 measurement state timeline."""
    timeline = phase.get("measurement_timeline")
    pass_count = phase.get("qualification_passes_completed")
    require(
        isinstance(timeline, list)
        and type(pass_count) is int
        and len(timeline) == pass_count + 1,
        f"{path}: malformed v16 measurement timeline",
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
            f"{path}: malformed v16 measurement timeline",
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
                f"{path}: malformed v16 measurement timeline update",
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
                f"{path}: invalid v16 measurement timeline update",
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
                    f"{path}: invalid v16 measurement state transition",
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
            f"{path}: unproven v16 effective-source repair",
        )
        require(
            not any(
                state["status"] in successful_statuses
                and state["source_hash"] in isolated_invalidated_source_hashes
                for state in next_states.values()
            ),
            f"{path}: incomplete v16 effective-source invalidation",
        )
        check_equal(
            update_ids,
            sorted(update_ids),
            f"{path}: reordered v16 measurement timeline update",
        )
        if expected_pass == 0:
            check_equal(
                set(update_ids),
                initial_ids,
                f"{path}: v16 measurement timeline initial population",
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
        f"{path}: inconsistent v16 qualification pass accounting",
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
            f"{path}: inconsistent v16 measurement introduction timeline",
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
                and snapshot["status"] in {"error", "timeout", "peer_compilation_fail"}
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
        retryable_ids = sorted(tracked_ids)
        generated_ids = _config_id_list(decision["generated_config_ids"], label)
        expected_generated = repair_ids_by_value[str(value)][
            repair_index : repair_index + 1
        ]
        require(
            retryable_ids == candidate_ids
            and retryable_ids
            and all(
                states.get(config_id, {}).get("status") in retryable_statuses
                for config_id in tracked_ids
            )
            and decision["selected_config_id"] == retryable_ids[0]
            and generated_ids == expected_generated
            and len(generated_ids) <= 1,
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
    conditional_neighbor_limits = [
        (index + 1) * qualification_neighbor_limit // len(conditional_values)
        - index * qualification_neighbor_limit // len(conditional_values)
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
    result_path: Path,
    provenance: dict[str, Any],
    phase: dict[str, Any],
    lanes_by_leaf: dict[str, list[tuple[str, int]]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, dict[str, Any]]]]:
    """Validate the mandatory v16 canonical config and measurement evidence."""
    manifest = phase.get("config_manifest")
    initial_results = phase.get("initial_results")
    require(
        isinstance(manifest, dict) and isinstance(initial_results, list),
        f"{result_path}: incomplete canonical config identity evidence",
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
            f"{result_path}: invalid canonical config manifest entry {config_id!r}",
        )
        configs[config_id] = entry["config"]
    initial_ids = phase["initial_config_ids"]
    states_by_pass = validate_measurement_timeline(result_path, phase, configs)
    check_equal(
        [
            record.get("config_id") if isinstance(record, dict) else None
            for record in initial_results
        ],
        initial_ids,
        f"{result_path}: generation-zero result order",
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
            f"{result_path}: invalid generation-zero result",
        )
        config = configs[record["config_id"]]
        validate_measurement_snapshot(
            result_path,
            states_by_pass,
            record,
            config_id=record["config_id"],
            label="inconsistent generation-zero measurement snapshot",
            expected_pass_index=0,
        )
        leaf = structural_leaf(config)
        check_equal(
            {"family": record["family"], "compound_packet": record["compound_packet"]},
            leaf,
            f"{result_path}: generation-zero structural leaf",
        )
        expected_lanes = lanes_by_leaf[canonical_json(leaf)]
        check_equal(
            record["pipeline_lanes"],
            [
                pipeline_lane_metric(lane)
                for lane in expected_lanes
                if config.get(lane[0]) == lane[1]
            ],
            f"{result_path}: generation-zero pipeline membership",
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
            f"{result_path}: generation-zero status/performance mismatch",
        )
    referenced_ids = set(initial_ids) | set(phase["exact_space_config_ids"])
    leaf_results = phase.get("leaf_results")
    compound_transfers = phase.get("compound_transfers")
    require(
        isinstance(leaf_results, list)
        and all(isinstance(result, dict) for result in leaf_results)
        and isinstance(compound_transfers, list)
        and all(isinstance(result, dict) for result in compound_transfers),
        f"{result_path}: invalid canonical config manifest membership",
    )
    for result in leaf_results:
        qualified_results = result.get("qualified_results")
        require(
            isinstance(qualified_results, list)
            and all(isinstance(qualified, dict) for qualified in qualified_results),
            f"{result_path}: invalid canonical config manifest membership",
        )
        for qualified in qualified_results:
            referenced_ids.add(qualified.get("config_id"))
    for result in compound_transfers:
        transfers = result.get("transfers")
        require(
            isinstance(transfers, list)
            and all(isinstance(transfer, dict) for transfer in transfers),
            f"{result_path}: invalid canonical config manifest membership",
        )
        for transfer in transfers:
            referenced_ids.add(transfer.get("source_config_id"))
            referenced_ids.add(transfer.get("transferred_config_id"))
    family_probe_paths = phase.get("family_probe_paths")
    require(
        isinstance(family_probe_paths, list)
        and all(isinstance(probe_path, dict) for probe_path in family_probe_paths),
        f"{result_path}: invalid family probe manifest membership",
    )
    for probe_path in family_probe_paths:
        starting_config_id = probe_path.get("starting_config_id")
        require(
            isinstance(starting_config_id, str),
            f"{result_path}: invalid family probe manifest membership",
        )
        referenced_ids.add(starting_config_id)
        rounds = probe_path.get("rounds")
        require(
            isinstance(rounds, list)
            and all(isinstance(round_record, dict) for round_record in rounds),
            f"{result_path}: invalid family probe manifest membership",
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
                f"{result_path}: invalid family probe manifest membership",
            )
            referenced_ids.update(candidate_ids)
            referenced_ids.update(result.get("config_id") for result in results)
    check_equal(
        set(configs), referenced_ids, f"{result_path}: canonical config manifest keys"
    )
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
) -> None:
    """Replay the immutable v16 pipeline-parent decisions pass by pass."""
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
        f"{path}: inconsistent v16 pipeline parent-decision accounting",
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
        f"{path}: inconsistent v16 pipeline repair accounting",
    )
    expected_jobs_by_pass.extend(repair_passes)
    expected_jobs_by_pass.extend(
        [] for _ in range(repair_pass_count - len(repair_passes))
    )
    require(
        len(expected_jobs_by_pass) == pipeline_pass_count,
        f"{path}: inconsistent v16 pipeline parent-decision accounting",
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
            f"{path}: invalid v16 pipeline qualification round",
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
        state_at_decision = measurement_states[pass_index]
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
                pass_index=pass_index,
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
            elif scoped_available:
                expected_candidate_ids = scoped_available
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


def _validate_structural_qualification_phase_v16(
    result_path: Path, provenance: dict[str, Any], phase: dict[str, Any]
) -> dict[str, Any]:
    trial = provenance["trials"][0]
    validate_flash_normalization_context(str(result_path), provenance, trial)
    check_equal(
        phase.get("phase"),
        "cute_flash_structural_qualification_v16",
        f"{result_path}: qualification phase name",
    )
    check_equal(
        phase.get("cute_flash_lane_policy_version"),
        EXPECTED_LANE_POLICY_VERSION,
        f"{result_path}: lane policy version",
    )
    check_equal(
        phase.get("completed"), True, f"{result_path}: qualification completion"
    )
    check_equal(
        phase.get("budget_exhausted"),
        False,
        f"{result_path}: qualification budget exhaustion",
    )
    check_equal(
        phase.get("conditional_candidates_per_pipeline_lane"),
        EXPECTED_CONDITIONAL_CANDIDATES_PER_PIPELINE_LANE,
        f"{result_path}: conditional candidates per pipeline lane",
    )
    check_equal(
        phase.get("qualification_failure_retries"),
        EXPECTED_QUALIFICATION_FAILURE_RETRIES,
        f"{result_path}: qualification failure retries",
    )
    planned = phase.get("qualification_passes_planned")
    require(type(planned) is int and planned >= 0, f"{result_path}: pass count")
    for key in (
        "qualification_passes_started",
        "qualification_passes_completed",
        "qualification_rounds_started",
        "qualification_rounds_completed",
    ):
        check_equal(phase.get(key), planned, f"{result_path}: {key}")
    check_equal(
        phase.get("qualification_rounds"),
        provenance.get("flash_structural_qualification_rounds"),
        f"{result_path}: configured qualification rounds",
    )
    check_equal(
        phase.get("pipeline_candidate_limit_per_leaf_per_round"),
        provenance.get(
            "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round"
        ),
        f"{result_path}: qualification candidate limit",
    )
    for key, provenance_key in (
        (
            "retained_candidates_per_leaf",
            "flash_structural_retained_candidates_per_leaf",
        ),
        ("retained_family_cap", "flash_structural_retained_family_cap"),
        ("retained_family_limit", "flash_structural_retained_family_limit"),
        (
            "retained_family_slowdown_limit",
            "flash_structural_retained_family_slowdown_limit",
        ),
        ("starting_path_limit", "flash_structural_starting_path_limit"),
        (
            "unrestricted_path_exhausts_generation_budget",
            "flash_structural_unrestricted_path_exhausts_generation_budget",
        ),
    ):
        check_equal(
            phase.get(key), provenance.get(provenance_key), f"{result_path}: {key}"
        )
    check_equal(
        phase.get("pipeline_qualification_keys"),
        list(FLASH_PIPELINE_QUALIFICATION_KEYS),
        f"{result_path}: pipeline qualification keys",
    )
    check_equal(
        phase.get("neighbor_generation_limit_per_leaf_per_round"),
        EXPECTED_NEIGHBOR_GENERATION_LIMIT,
        f"{result_path}: neighbor generation limit",
    )

    initial_ids = _config_id_list(
        phase.get("initial_config_ids"), f"{result_path}: initial population"
    )
    exact_config_ids = exact_effective_search_space_ids(result_path, provenance)
    phase_exact_ids = _config_id_list(
        phase.get("exact_space_config_ids"), f"{result_path}: phase exact space"
    )
    check_equal(
        phase.get("exact_space_enumerated"),
        exact_config_ids is not None,
        f"{result_path}: exact-space enumeration flag",
    )
    check_equal(
        phase_exact_ids,
        exact_config_ids or [],
        f"{result_path}: exact-space config IDs",
    )
    exact_ids_measured = exact_config_ids is not None and set(exact_config_ids) <= set(
        initial_ids
    )
    require(
        isinstance(phase.get("exact_space_exhausted"), bool)
        and (not phase["exact_space_exhausted"] or exact_ids_measured),
        f"{result_path}: exact-space exhaustion flag",
    )
    check_equal(
        phase.get("exact_space_raw_budget"),
        max(1, provenance["autotune_initial_population_size"], len(initial_ids)),
        f"{result_path}: exact-space raw enumeration budget",
    )
    check_equal(
        phase.get("initial_config_count"),
        len(initial_ids),
        f"{result_path}: initial config count",
    )
    check_equal(
        len(initial_ids),
        (
            provenance["autotune_initial_population_size"]
            if exact_config_ids is None
            else len(exact_config_ids)
        ),
        f"{result_path}: initial population metric",
    )
    injected_ids = {
        canonical_sha256(item["config"])[:16]
        for item in provenance["flash_structural_coverage_design"][
            : provenance["flash_structural_injected_design_count"]
        ]
    }
    require(
        injected_ids <= set(initial_ids),
        f"{result_path}: initial population omits injected structural configs",
    )

    leaf_catalog = provenance.get("flash_structural_leaf_catalog")
    leaf_results = phase.get("leaf_results")
    compound_transfers = phase.get("compound_transfers")
    require(
        isinstance(leaf_catalog, list)
        and isinstance(leaf_results, list)
        and isinstance(compound_transfers, list),
        f"{result_path}: missing v16 structural records",
    )
    ordinary_leaves = [
        leaf
        for leaf in leaf_catalog
        if isinstance(leaf, dict) and leaf.get("compound_packet") is None
    ]
    compound_leaves = [
        leaf
        for leaf in leaf_catalog
        if isinstance(leaf, dict) and leaf.get("compound_packet") is not None
    ]
    check_equal(phase.get("leaf_count"), len(leaf_catalog), f"{result_path}: leaves")
    check_equal(
        phase.get("ordinary_leaf_count"),
        len(ordinary_leaves),
        f"{result_path}: ordinary leaves",
    )
    check_equal(
        phase.get("compound_leaf_count"),
        len(compound_leaves),
        f"{result_path}: compound leaves",
    )
    check_equal(
        [
            {
                "family": result.get("family"),
                "compound_packet": result.get("compound_packet"),
            }
            for result in leaf_results
            if isinstance(result, dict)
        ],
        ordinary_leaves,
        f"{result_path}: ordinary leaf order",
    )
    lanes_by_leaf = flash_pipeline_lane_catalog(result_path, provenance)
    expected_clc_catalog = flash_clc_lane_catalog(result_path, provenance)
    phase_configs, measurement_states = validate_phase_config_identity(
        result_path, provenance, phase, lanes_by_leaf
    )
    for leaf_result in phase["leaf_results"]:
        for qualified in leaf_result["qualified_results"]:
            require(
                isinstance(qualified, dict)
                and set(qualified)
                == {
                    "config_id",
                    "attempt_perf",
                    "selection_perf",
                    "status",
                    "source_hash",
                    "measurement_pass_index",
                    "pipeline_lanes",
                }
                and qualified["config_id"] in phase_configs,
                f"{result_path}: malformed v16 qualified result",
            )
            validate_measurement_snapshot(
                result_path,
                measurement_states,
                qualified,
                config_id=qualified["config_id"],
                label="invalid v16 qualified measurement snapshot",
                expected_pass_index=phase["qualification_passes_completed"],
            )
    for leaf, result in zip(ordinary_leaves, leaf_results, strict=True):
        require(
            isinstance(result, dict)
            and set(result)
            == {
                "family",
                "compound_packet",
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
            f"{result_path}: malformed ordinary leaf",
        )
        check_equal(
            result.get("complete"), True, f"{result_path}: ordinary leaf completion"
        )
        lanes = result.get("pipeline_lanes")
        require(isinstance(lanes, list), f"{result_path}: malformed pipeline lanes")
        require(
            isinstance(result.get("space_exhausted"), bool)
            and (
                result.get("space_config_count") is None
                if exact_config_ids is None
                else type(result.get("space_config_count")) is int
                and result["space_config_count"] >= 0
            )
            and result.get("ordinary_search_required")
            is (not lanes and not result.get("space_exhausted")),
            f"{result_path}: invalid leaf exact-space evidence",
        )
        require(
            not result["space_exhausted"]
            or (exact_config_ids is not None and result["space_config_count"] > 0),
            f"{result_path}: leaf claims exhaustion without exact-space proof",
        )
        check_equal(
            [(lane.get("key"), lane.get("value")) for lane in lanes],
            lanes_by_leaf[json.dumps(leaf, sort_keys=True, separators=(",", ":"))],
            f"{result_path}: pipeline lane catalog",
        )
        for lane in lanes:
            require(
                isinstance(lane, dict)
                and set(lane)
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
                    "complete",
                }
                and lane.get("witness_attempted") is True
                and isinstance(lane.get("witness_config_id"), str)
                and re.fullmatch(r"[0-9a-f]{16}", lane["witness_config_id"]) is not None
                and isinstance(lane.get("witness_succeeded"), bool)
                and lane.get("complete") is True,
                f"{result_path}: incomplete pipeline lane",
            )
            lane_initial_ids = _config_id_list(
                lane["initial_config_ids"],
                f"{result_path}: pipeline lane initial IDs",
            )
            require(
                set(lane_initial_ids) <= set(result["initial_config_ids"]),
                f"{result_path}: invalid pipeline lane initial membership",
            )
            conditional_ids = _config_id_list(
                lane.get("conditional_candidate_ids"),
                f"{result_path}: pipeline conditional candidates",
            )
            require(
                isinstance(lane.get("space_exhausted"), bool)
                and (
                    lane.get("space_config_count") is None
                    if exact_config_ids is None
                    else type(lane.get("space_config_count")) is int
                    and lane["space_config_count"] >= 0
                )
                and lane.get("conditional_required") is not None
                and lane["conditional_required"] is (not lane["space_exhausted"]),
                f"{result_path}: invalid lane exact-space evidence",
            )
            require(
                not lane["space_exhausted"]
                or (exact_config_ids is not None and lane["space_config_count"] > 0),
                f"{result_path}: lane claims exhaustion without exact-space proof",
            )
            check_equal(
                len(conditional_ids),
                (
                    EXPECTED_CONDITIONAL_CANDIDATES_PER_PIPELINE_LANE
                    if lane["conditional_required"]
                    else 0
                ),
                f"{result_path}: pipeline conditional candidate count",
            )
            require(
                not (set(conditional_ids) & set(initial_ids)),
                f"{result_path}: pipeline conditional candidates are not novel",
            )
            successful_conditional_ids = _config_id_list(
                lane.get("successful_conditional_candidate_ids"),
                f"{result_path}: successful pipeline conditional candidates",
            )
            repair_ids = _config_id_list(
                lane.get("repair_candidate_ids"),
                f"{result_path}: pipeline repair candidates",
            )
            successful_repair_ids = _config_id_list(
                lane.get("successful_repair_candidate_ids"),
                f"{result_path}: successful pipeline repair candidates",
            )
            repair_decisions = lane.get("repair_parent_decisions")
            require(
                set(successful_conditional_ids) <= set(conditional_ids)
                and set(successful_repair_ids) <= set(repair_ids)
                and not (set(repair_ids) & ({*initial_ids, *conditional_ids}))
                and isinstance(repair_decisions, list)
                and len(repair_decisions) <= EXPECTED_QUALIFICATION_FAILURE_RETRIES
                and (
                    lane["witness_succeeded"]
                    or successful_conditional_ids
                    or successful_repair_ids
                ),
                f"{result_path}: pipeline lane lacks successful evidence",
            )
            generated_repair_ids: list[str] = []
            retryable_failure_ids = {lane["witness_config_id"], *conditional_ids}
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
                    f"{result_path}: malformed pipeline repair decision",
                )
                snapshot_ids: list[str] = []
                snapshot_passes: set[int] = set()
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
                        and snapshot["config_id"] in retryable_failure_ids
                        and snapshot["attempt_perf"] is None
                        and snapshot["selection_perf"] is None
                        and snapshot["status"]
                        in {"error", "timeout", "peer_compilation_fail"},
                        f"{result_path}: invalid pipeline repair parent",
                    )
                    validate_measurement_snapshot(
                        result_path,
                        measurement_states,
                        snapshot,
                        config_id=snapshot["config_id"],
                        label="invalid pipeline repair parent",
                    )
                    snapshot_ids.append(snapshot["config_id"])
                    snapshot_passes.add(snapshot["measurement_pass_index"])
                check_equal(
                    snapshot_ids,
                    sorted(set(snapshot_ids)),
                    f"{result_path}: ranked pipeline repair parents",
                )
                require(
                    len(snapshot_passes) == 1,
                    f"{result_path}: inconsistent pipeline repair decision pass",
                )
                decision_pass = next(iter(snapshot_passes))
                tracked_ids = [
                    lane["witness_config_id"],
                    *conditional_ids,
                    *generated_repair_ids,
                ]
                require(
                    snapshot_ids == sorted(tracked_ids)
                    and all(
                        measurement_states[decision_pass]
                        .get(config_id, {})
                        .get("status")
                        in {"error", "timeout", "peer_compilation_fail"}
                        for config_id in tracked_ids
                    ),
                    f"{result_path}: illegitimate pipeline repair decision",
                )
                check_equal(
                    decision["selected_config_id"],
                    snapshot_ids[0],
                    f"{result_path}: selected pipeline repair parent",
                )
                generated = _config_id_list(
                    decision["generated_config_ids"],
                    f"{result_path}: generated pipeline repairs",
                )
                require(
                    len(generated) <= 1,
                    f"{result_path}: oversized pipeline repair decision",
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
                        f"{result_path}: unproven empty pipeline repair",
                    )
                generated_repair_ids.extend(generated)
                retryable_failure_ids.update(generated)
            check_equal(
                generated_repair_ids,
                repair_ids,
                f"{result_path}: pipeline repair decision IDs",
            )

    clc_families = phase.get("clc_families")
    require(isinstance(clc_families, list), f"{result_path}: CLC family records")
    check_equal(
        len(clc_families),
        len(expected_clc_catalog),
        f"{result_path}: CLC family count",
    )
    active_clc_values = {
        active.get("value")
        for active in provenance.get("flash_structural_coverage_active_values", [])
        if isinstance(active, dict)
        and active.get("key") == FLASH_CLC_HEADS_PER_BATCH_KEY
        and type(active.get("value")) is int
    }
    max_clc_planned = 0
    clc_witness_repair_decision_passes: set[int] = set()
    clc_conditional_repair_decision_passes: set[int] = set()
    clc_combination_snapshot_passes: set[int] = set()
    clc_witness_snapshot_passes: set[int] = set()
    clc_conditional_parent_passes: set[int] = set()
    clc_retained_snapshot_passes: set[int] = set()
    clc_depth_snapshot_passes: set[int] = set()
    clc_families_seen: set[str] = set()
    for expected_clc, result in zip(expected_clc_catalog, clc_families, strict=True):
        require(
            isinstance(result, dict)
            and set(result)
            == {
                "family",
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
            f"{result_path}: malformed CLC family record",
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
            f"{result_path}: missing v16 CLC combination evidence",
        )
        family = result["family"]
        require(
            isinstance(family, str)
            and family not in clc_families_seen
            and family in {leaf["family"] for leaf in ordinary_leaves},
            f"{result_path}: invalid CLC family",
        )
        clc_families_seen.add(family)
        legal = _positive_int_list(
            result["legal_values"], f"{result_path}: CLC legal values"
        )
        search_values = _positive_int_list(
            result["search_values"], f"{result_path}: CLC search values"
        )
        anchors = _positive_int_list(
            result["anchor_values"], f"{result_path}: CLC anchor values"
        )
        refinements = _positive_int_list(
            result["refinement_values"], f"{result_path}: CLC refinement values"
        )
        planned_values = _positive_int_list(
            result["planned_values"], f"{result_path}: CLC planned values"
        )
        attempted_values = _positive_int_list(
            result["attempted_values"], f"{result_path}: CLC attempted values"
        )
        selected_values = _positive_int_list(
            result["selected_values"], f"{result_path}: CLC selected values"
        )
        conditional_values = _positive_int_list(
            result["conditional_values"], f"{result_path}: CLC conditional values"
        )
        retained_values = _positive_int_list(
            result["retained_values"], f"{result_path}: CLC retained values"
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
                result[field],
                expected_clc[field],
                f"{result_path}: CLC catalog {field}",
            )
        retained_limit = provenance["flash_structural_retained_candidates_per_leaf"]
        require(
            search_values == legal
            and set(anchors) <= set(search_values)
            and set(refinements) <= set(search_values)
            and not (set(anchors) & set(refinements))
            and set(anchors) == active_clc_values
            and planned_values == [*anchors, *refinements]
            and set(planned_values) == set(legal)
            and attempted_values == planned_values
            and len(selected_values) == min(retained_limit, len(planned_values))
            and set(selected_values) <= set(planned_values)
            and len(retained_values) == len(selected_values)
            and set(retained_values) <= set(selected_values),
            f"{result_path}: invalid CLC value selection",
        )
        witnesses = result["witness_config_ids"]
        require(
            isinstance(witnesses, dict)
            and set(witnesses) == {str(value) for value in planned_values}
            and all(
                isinstance(config_id, str)
                and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
                for config_id in witnesses.values()
            ),
            f"{result_path}: invalid CLC witnesses",
        )
        witness_repair_ids = _repair_id_mapping(
            result["witness_repair_candidate_ids"],
            planned_values,
            f"{result_path}: invalid immutable CLC witness repair",
        )
        value_space_exhausted = result["value_space_exhausted"]
        require(
            isinstance(value_space_exhausted, dict)
            and set(value_space_exhausted) == {str(value) for value in planned_values}
            and all(type(value) is bool for value in value_space_exhausted.values()),
            f"{result_path}: invalid CLC value exhaustion",
        )
        check_equal(
            conditional_values,
            [
                value
                for value in selected_values
                if not value_space_exhausted[str(value)]
            ],
            f"{result_path}: CLC conditional values",
        )
        conditional_ids = result["conditional_candidate_ids"]
        require(
            isinstance(conditional_ids, dict)
            and set(conditional_ids) == {str(value) for value in conditional_values}
            and all(
                len(_config_id_list(ids, f"{result_path}: CLC conditional IDs")) == 1
                for ids in conditional_ids.values()
            ),
            f"{result_path}: incomplete CLC conditional records",
        )
        conditional_repair_ids = _repair_id_mapping(
            result["conditional_repair_candidate_ids"],
            conditional_values,
            f"{result_path}: invalid immutable CLC conditional repair",
        )
        clc_witness_repair_decision_passes.update(
            validate_failure_repair_decisions(
                result_path,
                result["witness_repair_parent_decisions"],
                witness_repair_ids,
                witnesses,
                planned_values,
                expected_kind="witness_failure_repair",
                expected_leaf={
                    "family": result["family"],
                    "compound_packet": None,
                },
                candidate_limit=phase["pipeline_candidate_limit_per_leaf_per_round"],
                phase_configs=phase_configs,
                measurement_states=measurement_states,
                label="invalid immutable CLC witness repair",
            )
        )
        clc_conditional_repair_decision_passes.update(
            validate_failure_repair_decisions(
                result_path,
                result["conditional_repair_parent_decisions"],
                conditional_repair_ids,
                conditional_ids,
                conditional_values,
                expected_kind="conditional_failure_repair",
                expected_leaf={
                    "family": result["family"],
                    "compound_packet": None,
                },
                candidate_limit=phase["pipeline_candidate_limit_per_leaf_per_round"],
                phase_configs=phase_configs,
                measurement_states=measurement_states,
                label="invalid immutable CLC conditional repair",
            )
        )
        (
            witness_snapshot_passes,
            conditional_parent_passes,
            retained_snapshot_passes,
            depth_snapshot_passes,
        ) = validate_clc_decision_evidence(
            result_path,
            result,
            planned_values=planned_values,
            selected_values=selected_values,
            conditional_values=conditional_values,
            retained_values=retained_values,
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
            result.get("combination_candidate_ids"),
            f"{result_path}: CLC combination IDs",
        )
        combination_depth_ids = _config_id_list(
            result.get("combination_depth_config_ids"),
            f"{result_path}: CLC combination depth IDs",
        )
        combination_divisors = _positive_int_list(
            result.get("combination_divisor_values"),
            f"{result_path}: CLC combination divisors",
        )
        depth_selection = result["depth_selection"]
        require(
            isinstance(depth_selection, dict)
            and set(depth_selection)
            == {"candidate_results", "selected_representatives"}
            and isinstance(depth_selection["selected_representatives"], list),
            f"{result_path}: invalid CLC depth selection",
        )
        check_equal(
            combination_depth_ids,
            [
                representative.get("config_id")
                for representative in depth_selection["selected_representatives"]
                if isinstance(representative, dict)
            ],
            f"{result_path}: CLC depth axes",
        )
        check_equal(
            combination_divisors,
            retained_values if result["combination_required"] else [],
            f"{result_path}: CLC divisor axes",
        )
        cells = result["combination_cells"]
        require(
            isinstance(cells, list)
            and len(cells) == len(combination_depth_ids) * len(combination_divisors),
            f"{result_path}: incomplete CLC combination cell ledger",
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
            f"{result_path}: CLC combination cell axes",
        )
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
                f"{result_path}: malformed CLC combination cell",
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
                    f"{result_path}: invalid rejected CLC projection",
                )
                continue
            require(
                isinstance(projected_id, str)
                and re.fullmatch(r"[0-9a-f]{16}", projected_id) is not None
                and cell["config_id"] == projected_id
                and cell["status"] in known_statuses
                and optional_positive_float(cell["attempt_perf"])
                and optional_positive_float(cell["selection_perf"]),
                f"{result_path}: invalid measured CLC projection",
            )
            succeeded = cell["status"] in {"ok", "deduplicated"}
            require(
                (
                    cell["attempt_perf"] is not None
                    and cell["selection_perf"] is not None
                )
                is succeeded,
                f"{result_path}: CLC projection status/performance mismatch",
            )
            validate_measurement_snapshot(
                result_path,
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
            f"{result_path}: CLC effective combination IDs",
        )
        check_equal(
            result["successful_combination_depth_config_ids"],
            coverage["successful_combination_depth_config_ids"],
            f"{result_path}: successful CLC depth coverage",
        )
        check_equal(
            result["successful_combination_divisor_values"],
            coverage["successful_combination_divisor_values"],
            f"{result_path}: successful CLC divisor coverage",
        )
        projection_complete = coverage["combination_projection_complete"]
        row_coverage_complete = coverage["combination_row_coverage_complete"]
        column_coverage_complete = coverage["combination_column_coverage_complete"]
        check_equal(
            result["combination_projection_complete"],
            projection_complete,
            f"{result_path}: CLC projection completion",
        )
        check_equal(
            result["combination_row_coverage_complete"],
            row_coverage_complete,
            f"{result_path}: CLC row coverage",
        )
        check_equal(
            result["combination_column_coverage_complete"],
            column_coverage_complete,
            f"{result_path}: CLC column coverage",
        )
        require(
            isinstance(result["space_exhausted"], bool)
            and isinstance(result["combination_required"], bool)
            and result["combination_required"] is (not result["space_exhausted"])
            and (
                0 < len(combination_ids) <= retained_limit**2
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
            )
            and result["complete"] is True,
            f"{result_path}: invalid CLC combination count",
        )
        check_equal(
            result["combination_failure_statuses_allowed"],
            all(
                terminal_measurement_is_valid(cell, allow_projection_rejected=True)
                for cell in cells
            ),
            f"{result_path}: CLC combination terminal statuses",
        )
        check_equal(
            result["combination_failure_statuses_allowed"],
            True,
            f"{result_path}: CLC combination terminal statuses",
        )
        matching_leaf = next(
            leaf_result
            for leaf_result in leaf_results
            if leaf_result["family"] == family
        )
        check_equal(
            result["space_exhausted"],
            matching_leaf["space_exhausted"],
            f"{result_path}: CLC leaf exhaustion",
        )
        max_clc_planned = max(max_clc_planned, len(planned_values))

    check_equal(
        phase["exact_space_exhausted"],
        exact_ids_measured
        and all(result["space_exhausted"] for result in clc_families),
        f"{result_path}: hierarchical exact-space exhaustion flag",
    )

    qualification_rounds = provenance["flash_structural_qualification_rounds"]
    candidate_limit = provenance[
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round"
    ]
    baseline_pipeline_pass_count = max(
        (
            max(
                (
                    0
                    if not result["pipeline_lanes"] and result["space_exhausted"]
                    else qualification_rounds
                ),
                math.ceil(len(result["pipeline_lanes"]) / candidate_limit)
                + math.ceil(
                    sum(
                        lane["conditional_required"]
                        for lane in result["pipeline_lanes"]
                    )
                    / candidate_limit
                ),
            )
            for result in leaf_results
        ),
        default=0,
    )
    repair_pass_count = max(
        (
            math.ceil(
                sum(
                    len(lane["repair_parent_decisions"])
                    for lane in result["pipeline_lanes"]
                )
                / candidate_limit
            )
            for result in leaf_results
        ),
        default=0,
    )
    pipeline_pass_count = baseline_pipeline_pass_count + repair_pass_count
    for result in leaf_results:
        leaf_initial_ids = set(result["initial_config_ids"])
        require(
            isinstance(result["rounds"], list)
            and len(result["rounds"]) == pipeline_pass_count,
            f"{result_path}: invalid leaf pass count",
        )
        leaf_ids_by_pass: list[set[str]] = []
        seen_leaf_ids: set[str] = set()
        for pass_result in result["rounds"]:
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
                f"{result_path}: malformed leaf pass",
            )
            ids = set(
                _config_id_list(
                    pass_result["candidate_config_ids"],
                    f"{result_path}: leaf pass candidates",
                )
            )
            require(
                len(ids) <= candidate_limit and not (seen_leaf_ids & ids),
                f"{result_path}: repeated or oversized leaf pass",
            )
            seen_leaf_ids.update(ids)
            leaf_ids_by_pass.append(ids)
        lane_ids_by_pass = [set() for _ in range(pipeline_pass_count)]
        lane_neighbor_by_pass = [0 for _ in range(pipeline_pass_count)]
        for lane in result["pipeline_lanes"]:
            lane_rounds = lane.get("rounds")
            require(
                isinstance(lane_rounds, list)
                and len(lane_rounds) == pipeline_pass_count,
                f"{result_path}: invalid lane pass count",
            )
            witness_id = lane["witness_config_id"]
            conditional_ids = set(lane["conditional_candidate_ids"])
            repair_ids = set(lane["repair_candidate_ids"])
            seen: set[str] = set()
            witness_pass: int | None = None
            conditional_passes: list[int] = []
            repair_passes: list[int] = []
            for pass_index, lane_round in enumerate(lane_rounds):
                require(
                    isinstance(lane_round, dict)
                    and set(lane_round)
                    == {"candidate_config_ids", "neighbor_generation_limit"},
                    f"{result_path}: malformed lane pass",
                )
                require(
                    type(lane_round["neighbor_generation_limit"]) is int
                    and 0
                    <= lane_round["neighbor_generation_limit"]
                    <= EXPECTED_NEIGHBOR_GENERATION_LIMIT,
                    f"{result_path}: invalid lane neighbor budget",
                )
                ids = set(
                    _config_id_list(
                        lane_round["candidate_config_ids"],
                        f"{result_path}: lane pass candidates",
                    )
                )
                require(not (seen & ids), f"{result_path}: repeated lane candidate")
                seen.update(ids)
                lane_ids_by_pass[pass_index].update(ids)
                lane_neighbor_by_pass[pass_index] += lane_round[
                    "neighbor_generation_limit"
                ]
                if witness_id in ids:
                    witness_pass = pass_index
                if conditional_ids & ids:
                    conditional_passes.append(pass_index)
                if repair_ids & ids:
                    repair_passes.append(pass_index)
            require(
                seen == {witness_id, *conditional_ids, *repair_ids}
                and witness_pass is not None
                and all(witness_pass < index for index in conditional_passes)
                and all(
                    index >= baseline_pipeline_pass_count for index in repair_passes
                ),
                f"{result_path}: conditional search precedes its witness",
            )
        for pass_index, leaf_ids in enumerate(leaf_ids_by_pass):
            pass_result = result["rounds"][pass_index]
            if result["pipeline_lanes"]:
                require(
                    leaf_ids <= lane_ids_by_pass[pass_index]
                    and lane_ids_by_pass[pass_index] - leaf_ids <= leaf_initial_ids,
                    f"{result_path}: leaf/lane pass candidate mismatch",
                )
                check_equal(
                    pass_result["neighbor_generation_limit"],
                    lane_neighbor_by_pass[pass_index],
                    f"{result_path}: leaf/lane pass neighbor budget",
                )
                check_equal(
                    pass_result["ordinary_neighbor_generation_limit"],
                    0,
                    f"{result_path}: lane leaf ordinary neighbor budget",
                )
            else:
                ordinary_pass_count = (
                    0 if result["space_exhausted"] else qualification_rounds
                )
                expected_limit = (
                    EXPECTED_NEIGHBOR_GENERATION_LIMIT
                    if pass_index < ordinary_pass_count
                    else 0
                )
                check_equal(
                    pass_result["neighbor_generation_limit"],
                    expected_limit,
                    f"{result_path}: ordinary leaf neighbor budget",
                )
                check_equal(
                    pass_result["ordinary_neighbor_generation_limit"],
                    expected_limit,
                    f"{result_path}: ordinary leaf direct neighbor budget",
                )
        validate_pipeline_parent_decisions(
            result_path,
            {
                "family": result["family"],
                "compound_packet": result["compound_packet"],
            },
            result,
            lanes_by_leaf[
                canonical_json(
                    {
                        "family": result["family"],
                        "compound_packet": result["compound_packet"],
                    }
                )
            ],
            phase_configs,
            measurement_states,
            qualification_rounds=qualification_rounds,
            candidate_limit=candidate_limit,
            baseline_pass_count=baseline_pipeline_pass_count,
            repair_pass_count=repair_pass_count,
        )
    check_equal(
        [
            {
                "family": result.get("family"),
                "compound_packet": result.get("compound_packet"),
            }
            for result in compound_transfers
            if isinstance(result, dict)
        ],
        compound_leaves,
        f"{result_path}: compound transfer order",
    )
    qualified_compound_ids: set[str] = set()
    compound_backfill_pass_count = 0
    compound_source_snapshot_passes: set[int] = set()
    retained_per_leaf = provenance["flash_structural_retained_candidates_per_leaf"]
    clc_results_by_family = {result["family"]: result for result in clc_families}
    for result in compound_transfers:
        require(
            isinstance(result, dict)
            and set(result)
            == {
                "family",
                "compound_packet",
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
            and result.get("complete") is True
            and type(result.get("limit")) is int
            and result["limit"] == retained_per_leaf
            and type(result.get("transfer_target_count")) is int
            and 0 < result["transfer_target_count"] <= result["limit"]
            and type(result.get("transfer_count")) is int
            and isinstance(result.get("transfers"), list)
            and result["transfer_count"] == len(result["transfers"])
            and 0
            < result["transfer_count"]
            <= result["limit"] * (1 + EXPECTED_QUALIFICATION_FAILURE_RETRIES)
            and isinstance(result["backfill_rounds"], list)
            and len(result["backfill_rounds"]) <= EXPECTED_QUALIFICATION_FAILURE_RETRIES
            and result["failure_statuses_allowed"] is True,
            f"{result_path}: incomplete compound transfer",
        )
        primary_ids = _config_id_list(
            result["primary_transfer_config_ids"],
            f"{result_path}: primary compound transfer IDs",
        )
        successful_ids = _config_id_list(
            result["successful_transfer_config_ids"],
            f"{result_path}: successful compound transfer IDs",
        )
        qualified_ids = _config_id_list(
            result["qualified_transfer_config_ids"],
            f"{result_path}: qualified compound transfer IDs",
        )
        require(
            len(primary_ids) == result["transfer_target_count"]
            and len(successful_ids) >= result["transfer_target_count"]
            and len(qualified_ids) == result["transfer_target_count"]
            and qualified_ids == successful_ids[: result["transfer_target_count"]],
            f"{result_path}: incomplete qualified compound transfer set",
        )
        qualified_compound_ids.update(qualified_ids)
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
            f"{result_path}: invalid immutable compound source decision",
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
                f"{result_path}: invalid immutable compound source decision",
            )
            validate_measurement_snapshot(
                result_path,
                measurement_states,
                snapshot,
                config_id=snapshot["config_id"],
                label="invalid immutable compound source decision",
            )
            source_candidate_ids.append(snapshot["config_id"])
            source_snapshot_passes.add(snapshot["measurement_pass_index"])
        require(
            len(source_snapshot_passes) == 1,
            f"{result_path}: invalid immutable compound source decision",
        )
        source_snapshot_pass = next(iter(source_snapshot_passes))
        compound_source_snapshot_passes.add(source_snapshot_pass)
        attempted_source_ids = _config_id_list(
            source_selection["attempted_config_ids"],
            f"{result_path}: attempted compound source IDs",
        )
        selected_source_ids = _config_id_list(
            source_selection["selected_config_ids"],
            f"{result_path}: selected compound source IDs",
        )
        clc_result = clc_results_by_family.get(result["family"])
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
            f"{result_path}: invalid immutable compound source decision",
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
                f"{result_path}: invalid immutable compound source decision",
            )
        expected_source_ids = {
            config_id
            for config_id, state in measurement_states[source_snapshot_pass].items()
            if state["status"] in {"ok", "deduplicated"}
            and structural_leaf(phase_configs[config_id])
            == {"family": result["family"], "compound_packet": None}
        }
        require(
            set(source_candidate_ids) == expected_source_ids,
            f"{result_path}: incomplete immutable compound source decision",
        )
        transfer_sources: list[str] = []
        transfer_targets: list[str] = []
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
                }
                and isinstance(transfer["source_config_id"], str)
                and re.fullmatch(r"[0-9a-f]{16}", transfer["source_config_id"])
                is not None
                and isinstance(transfer["transferred_config_id"], str)
                and re.fullmatch(r"[0-9a-f]{16}", transfer["transferred_config_id"])
                is not None
                and isinstance(transfer["source_config"], dict)
                and isinstance(transfer["projected_config"], dict)
                and transfer["projected_config_id"] == transfer["transferred_config_id"]
                and transfer["projection_overrides"]
                == {FLASH_EXP2_PACKET_KEY: result["compound_packet"]}
                and terminal_measurement_is_valid(transfer),
                f"{result_path}: malformed compound transfer",
            )
            validate_measurement_snapshot(
                result_path,
                measurement_states,
                transfer,
                config_id=transfer["transferred_config_id"],
                label="inconsistent compound transfer measurement snapshot",
                expected_pass_index=phase["qualification_passes_completed"]
                - (
                    phase.get("family_probe_generations", 0)
                    if phase.get("family_probe_required")
                    else 0
                ),
            )
            check_equal(
                canonical_sha256(transfer["source_config"])[:16],
                transfer["source_config_id"],
                f"{result_path}: compound source config ID",
            )
            check_equal(
                canonical_sha256(transfer["projected_config"])[:16],
                transfer["transferred_config_id"],
                f"{result_path}: compound projected config ID",
            )
            require(
                structural_leaf(transfer["source_config"])
                == {"family": result["family"], "compound_packet": None}
                and structural_leaf(transfer["projected_config"])
                == {
                    "family": result["family"],
                    "compound_packet": result["compound_packet"],
                },
                f"{result_path}: compound snapshots change the wrong structural leaf",
            )
            check_equal(
                transfer["preserved_pipeline_values"],
                {
                    key: transfer["source_config"][key]
                    for key in FLASH_PIPELINE_QUALIFICATION_KEYS
                    if key in transfer["source_config"]
                },
                f"{result_path}: compound preserved pipeline values",
            )
            transfer_sources.append(transfer["source_config_id"])
            transfer_targets.append(transfer["transferred_config_id"])
        check_equal(
            selected_source_ids,
            transfer_sources,
            f"{result_path}: immutable compound source selection",
        )
        require(
            all(source_id in attempted_source_ids for source_id in selected_source_ids)
            and [
                source_id
                for source_id in attempted_source_ids
                if source_id in set(selected_source_ids)
            ]
            == selected_source_ids,
            f"{result_path}: invalid immutable compound source decision",
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
            f"{result_path}: invalid immutable compound source decision stop condition",
        )
        check_equal(
            primary_ids,
            transfer_targets[: result["transfer_target_count"]],
            f"{result_path}: primary compound transfer IDs",
        )
        check_equal(
            successful_ids,
            [
                transfer["transferred_config_id"]
                for transfer in result["transfers"]
                if transfer["status"] in {"ok", "deduplicated"}
            ],
            f"{result_path}: successful compound transfer IDs",
        )
        check_equal(
            result["failure_statuses_allowed"],
            all(
                terminal_measurement_is_valid(transfer)
                for transfer in result["transfers"]
            ),
            f"{result_path}: compound terminal statuses",
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
                f"{result_path}: invalid compound transfer backfill",
            )
            failed_ids = _config_id_list(
                backfill["failed_transfer_config_ids"],
                f"{result_path}: failed compound transfer IDs",
            )
            backfill_source_ids = _config_id_list(
                backfill["attempted_source_config_ids"],
                f"{result_path}: backfill source IDs",
            )
            generated_ids = _config_id_list(
                backfill["generated_config_ids"],
                f"{result_path}: generated compound transfer IDs",
            )
            decision_pass = source_snapshot_pass + 1 + backfill_index
            require(
                decision_pass + 1 < len(measurement_states),
                f"{result_path}: invalid compound transfer backfill",
            )
            attempted_targets = transfer_targets[:consumed_transfer_count]
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
                == transfer_targets[
                    consumed_transfer_count : consumed_transfer_count
                    + len(generated_ids)
                ]
                and len(generated_ids) <= missing,
                f"{result_path}: illegitimate compound transfer backfill",
            )
            consumed_transfer_count += len(generated_ids)
            backfill_attempted_source_ids.extend(backfill_source_ids)
            completed_states = measurement_states[decision_pass + 1]
            require(
                sum(
                    completed_states.get(config_id, {}).get("status")
                    in {"ok", "deduplicated"}
                    for config_id in transfer_targets[:consumed_transfer_count]
                )
                >= result["transfer_target_count"],
                f"{result_path}: incomplete compound transfer backfill",
            )
        require(
            consumed_transfer_count == len(transfer_targets),
            f"{result_path}: invalid compound transfer backfill",
        )
        primary_attempted_end = (
            attempted_source_ids.index(selected_source_ids[len(primary_ids) - 1]) + 1
        )
        require(
            attempted_source_ids[primary_attempted_end:]
            == backfill_attempted_source_ids,
            f"{result_path}: invalid compound transfer backfill source suffix",
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
    expected_passes = (
        pipeline_pass_count
        + clc_witness_passes
        + clc_witness_repair_passes
        + clc_conditional_passes
        + clc_conditional_repair_passes
        + clc_combination_passes
        + compound_primary_passes
        + compound_backfill_pass_count
    )
    for key in (
        "qualification_passes_planned",
        "qualification_passes_started",
        "qualification_passes_completed",
        "qualification_rounds_started",
        "qualification_rounds_completed",
    ):
        check_equal(phase.get(key), expected_passes, f"{result_path}: {key}")
    witness_repair_start = pipeline_pass_count + clc_witness_passes
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
        f"{result_path}: CLC witness repair pass accounting",
    )
    check_equal(
        clc_conditional_repair_decision_passes,
        set(
            range(
                conditional_repair_start,
                conditional_repair_start + clc_conditional_repair_passes,
            )
        ),
        f"{result_path}: CLC conditional repair pass accounting",
    )
    post_witness_pass = witness_repair_start + clc_witness_repair_passes
    post_conditional_pass = conditional_repair_start + clc_conditional_repair_passes
    check_equal(
        clc_witness_snapshot_passes,
        {post_witness_pass} if max_clc_planned > 0 else set(),
        f"{result_path}: CLC witness snapshot pass",
    )
    check_equal(
        clc_conditional_parent_passes,
        {post_witness_pass} if clc_conditional_passes else set(),
        f"{result_path}: CLC conditional-parent snapshot pass",
    )
    check_equal(
        clc_retained_snapshot_passes,
        {post_conditional_pass} if max_clc_planned > 0 else set(),
        f"{result_path}: CLC retained snapshot pass",
    )
    check_equal(
        clc_depth_snapshot_passes,
        {post_conditional_pass} if clc_combination_passes else set(),
        f"{result_path}: CLC depth snapshot pass",
    )
    compound_source_pass = post_conditional_pass + clc_combination_passes
    require(
        all(
            pass_index == post_conditional_pass + 1
            for pass_index in clc_combination_snapshot_passes
        ),
        f"{result_path}: inconsistent CLC combination snapshot pass",
    )
    check_equal(
        compound_source_snapshot_passes,
        {compound_source_pass} if compound_leaves else set(),
        f"{result_path}: immutable compound source snapshot pass",
    )
    scheduled_ids_by_completion_pass = [set() for _ in range(expected_passes + 1)]
    for leaf_result in leaf_results:
        for pass_index, round_result in enumerate(leaf_result["rounds"], start=1):
            scheduled_ids_by_completion_pass[pass_index].update(
                _config_id_list(
                    round_result["candidate_config_ids"],
                    f"{result_path}: qualification round candidates",
                )
            )
    witness_completion_pass = pipeline_pass_count + clc_witness_passes
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
    validate_measurement_introductions(
        result_path,
        phase,
        measurement_states,
        scheduled_ids_by_completion_pass,
    )
    retained = phase.get("retained_families")
    require(isinstance(retained, list), f"{result_path}: retained families")
    retained_ids = [
        starting_path.get("config_id")
        for family in retained
        if isinstance(family, dict) and isinstance(family.get("starting_paths"), list)
        for starting_path in family["starting_paths"]
        if isinstance(starting_path, dict)
    ]
    require(
        len(retained_ids) == len(set(retained_ids))
        and all(
            isinstance(config_id, str)
            and re.fullmatch(r"[0-9a-f]{16}", config_id) is not None
            for config_id in retained_ids
        ),
        f"{result_path}: invalid retained paths",
    )
    require(
        all(
            starting_path.get("compound_packet") is None
            or starting_path.get("config_id") in qualified_compound_ids
            for family in retained
            if isinstance(family, dict)
            and isinstance(family.get("starting_paths"), list)
            for starting_path in family["starting_paths"]
            if isinstance(starting_path, dict)
        ),
        f"{result_path}: retained compound path is outside the qualified subset",
    )
    check_equal(
        phase.get("retained_path_count"),
        len(retained_ids),
        f"{result_path}: retained path count",
    )
    return phase


def _validate_structural_qualification_phase(
    result_path: Path, provenance: dict[str, Any]
) -> dict[str, Any]:
    trials = provenance.get("trials")
    require(
        isinstance(trials, list) and len(trials) == 1 and isinstance(trials[0], dict),
        f"{result_path}: expected one autotune trial",
    )
    phase = trials[0].get("search_phase_metrics")
    require(isinstance(phase, dict), f"{result_path}: missing qualification phase")
    return _validate_structural_qualification_phase_v16(result_path, provenance, phase)


def _reconcile_structural_qualification_phase_v16(
    result_path: Path,
    provenance: dict[str, Any],
    phase: dict[str, Any],
    attempt_by_config: dict[str, dict[str, Any]],
    metadata_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trial = provenance["trials"][0]
    validate_flash_normalization_context(str(result_path), provenance, trial)
    initial_ids = set(phase["initial_config_ids"])
    successful_statuses = {"ok", "deduplicated"}
    successful_ids: set[str] = set()
    measured_ids: set[str] = set()
    successful_candidate_ids: set[str] = set()
    selection_perf_by_id: dict[str, float] = {}
    passes = phase["qualification_passes_completed"]
    explicit_candidate_ids: set[str] = set()
    leaves_with_candidates = 0
    qualified_leaves: list[dict[str, Any]] = []
    successful_results_by_id: dict[str, dict[str, Any]] = {}
    phase_manifest = phase.get("config_manifest")
    if isinstance(phase_manifest, dict):
        for config_id, entry in phase_manifest.items():
            check_equal(
                metadata_configs.get(config_id),
                entry["config"],
                f"{result_path}: manifest/sidecar config {config_id}",
            )

    def reconcile_attempt(config_id: str) -> dict[str, Any]:
        attempt = attempt_by_config.get(config_id)
        require(
            isinstance(attempt, dict),
            f"{result_path}: qualification config {config_id} has no sidecar attempt",
        )
        generation = attempt.get("generation")
        require(
            type(generation) is int
            and (
                generation == 0
                if config_id in initial_ids
                else 1 <= generation <= passes
            ),
            f"{result_path}: qualification config {config_id} generation",
        )
        measured_ids.add(config_id)
        if attempt.get("status") in successful_statuses:
            require(
                optional_positive_float(attempt.get("perf_ms"))
                and attempt.get("perf_ms") is not None,
                f"{result_path}: successful config {config_id} lacks performance",
            )
            successful_ids.add(config_id)
            if config_id not in initial_ids:
                successful_candidate_ids.add(config_id)
        else:
            check_equal(
                attempt.get("perf_ms"),
                None,
                f"{result_path}: failed config {config_id} performance",
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
            f"{result_path}: initial config {config_id} is absent from metadata",
        )
        leaf = structural_leaf(config)
        if leaf is not None:
            leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
            initial_by_leaf.setdefault(leaf_key, set()).add(config_id)
        attempt = reconcile_attempt(config_id)
        if initial_result_by_id:
            record = initial_result_by_id[config_id]
            check_equal(
                record["status"],
                attempt.get("status"),
                f"{result_path}: generation-zero status {config_id}",
            )
            attempt_perf = attempt.get("perf_ms")
            if attempt_perf is None:
                check_equal(
                    record["attempt_perf"],
                    None,
                    f"{result_path}: generation-zero performance {config_id}",
                )
            else:
                require(
                    isinstance(record["attempt_perf"], (int, float))
                    and not isinstance(record["attempt_perf"], bool)
                    and abs(record["attempt_perf"] - attempt_perf) <= 0.500001e-6,
                    f"{result_path}: generation-zero performance {config_id}",
                )

    exact_config_ids = phase["exact_space_config_ids"]
    exact_by_leaf: dict[str, list[str]] = {}
    for config_id in exact_config_ids:
        config = metadata_configs.get(config_id)
        require(
            isinstance(config, dict),
            f"{result_path}: exact-space config {config_id} is absent from metadata",
        )
        leaf = structural_leaf(config)
        if leaf is not None:
            leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
            exact_by_leaf.setdefault(leaf_key, []).append(config_id)
    clc_by_family = {result["family"]: result for result in phase["clc_families"]}

    def hierarchical_clc_values_covered(family: str, config_ids: list[str]) -> bool:
        clc_result = clc_by_family.get(family)
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
            config_id in initial_ids and config_id in measured_ids
            for config_id in exact_config_ids
        )
        and all(
            hierarchical_clc_values_covered(
                family,
                [
                    config_id
                    for config_id in exact_config_ids
                    if structural_leaf(metadata_configs[config_id])
                    == {"family": family, "compound_packet": None}
                ],
            )
            for family in clc_by_family
        )
    )
    check_equal(
        phase["exact_space_exhausted"],
        exact_space_exhausted,
        f"{result_path}: measured exact-space exhaustion",
    )

    reconciled_leaves: list[dict[str, Any]] = []
    qualified_ids: set[str] = set()
    for result in phase["leaf_results"]:
        leaf = {
            "family": result["family"],
            "compound_packet": result["compound_packet"],
        }
        leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        check_equal(
            set(result["initial_config_ids"]),
            initial_by_leaf.get(leaf_key, set()),
            f"{result_path}: exact initial membership for leaf {leaf!r}",
        )
        lanes = [
            (lane_result["key"], lane_result["value"])
            for lane_result in result["pipeline_lanes"]
        ]
        leaf_exact_ids = exact_by_leaf.get(leaf_key, [])
        leaf_space_exhausted = (
            bool(leaf_exact_ids)
            and all(
                config_id in initial_ids and config_id in measured_ids
                for config_id in leaf_exact_ids
            )
            and hierarchical_clc_values_covered(result["family"], leaf_exact_ids)
        )
        check_equal(
            result["space_config_count"],
            len(leaf_exact_ids) if phase["exact_space_enumerated"] else None,
            f"{result_path}: exact-space config count for leaf {leaf!r}",
        )
        check_equal(
            result["space_exhausted"],
            leaf_space_exhausted,
            f"{result_path}: exact-space exhaustion for leaf {leaf!r}",
        )
        check_equal(
            result["ordinary_search_required"],
            not lanes and not leaf_space_exhausted,
            f"{result_path}: ordinary-search requirement for leaf {leaf!r}",
        )
        explicit_ids = set(result["initial_config_ids"])
        leaf_round_candidate_ids: set[str] = set()
        for pass_result in result["rounds"]:
            explicit_ids.update(pass_result["candidate_config_ids"])
            leaf_round_candidate_ids.update(pass_result["candidate_config_ids"])
        for lane_result in result["pipeline_lanes"]:
            explicit_ids.add(lane_result["witness_config_id"])
            explicit_ids.update(lane_result["conditional_candidate_ids"])
            explicit_ids.update(lane_result["repair_candidate_ids"])
        clc_result = clc_by_family.get(result["family"])
        if clc_result is not None:
            explicit_ids.update(clc_result["witness_config_ids"].values())
            for ids in clc_result["conditional_candidate_ids"].values():
                explicit_ids.update(ids)
            explicit_ids.update(clc_result["combination_candidate_ids"])
        explicit_candidate_ids.update(explicit_ids - initial_ids)
        clc_witness_ids = (
            set(clc_result["witness_config_ids"].values())
            if clc_result is not None
            else set()
        )
        leaves_with_candidates += int(
            bool(leaf_round_candidate_ids or (clc_witness_ids - initial_ids))
        )
        successful_members: list[dict[str, Any]] = []
        leaf_qualified_ids: set[str] = set()
        for qualified in result["qualified_results"]:
            config_id = qualified["config_id"]
            require(
                config_id not in qualified_ids,
                f"{result_path}: duplicate qualified config {config_id}",
            )
            qualified_ids.add(config_id)
            leaf_qualified_ids.add(config_id)
            config = metadata_configs.get(config_id)
            require(
                isinstance(config, dict) and structural_leaf(config) == leaf,
                f"{result_path}: qualified config {config_id} exact leaf",
            )
            memberships = config_pipeline_lanes(config, lanes)
            check_equal(
                qualified["pipeline_lanes"],
                [pipeline_lane_metric(lane) for lane in lanes if lane in memberships],
                f"{result_path}: qualified config {config_id} actual lane membership",
            )
            attempt = reconcile_attempt(config_id)
            check_equal(
                qualified["status"],
                attempt.get("status"),
                f"{result_path}: qualified config {config_id} sidecar status",
            )
            if qualified["status"] in successful_statuses:
                require(
                    optional_positive_float(qualified["attempt_perf"])
                    and qualified["attempt_perf"] is not None
                    and optional_positive_float(qualified["selection_perf"])
                    and qualified["selection_perf"] is not None,
                    f"{result_path}: successful qualified config lacks performance",
                )
                require(
                    abs(qualified["attempt_perf"] - attempt["perf_ms"]) <= 0.500001e-6,
                    f"{result_path}: qualified attempt performance differs from CSV",
                )
                successful_members.append(
                    {
                        "config_id": config_id,
                        "selection_perf": qualified["selection_perf"],
                        "pipeline_lanes": memberships,
                    }
                )
                selection_perf_by_id[config_id] = qualified["selection_perf"]
                successful_results_by_id[config_id] = {
                    "config_id": config_id,
                    "attempt_perf": qualified["attempt_perf"],
                    "selection_perf": qualified["selection_perf"],
                    "status": qualified["status"],
                }
            else:
                require(
                    qualified["attempt_perf"] is None
                    and qualified["selection_perf"] is None,
                    f"{result_path}: failed qualified config records performance",
                )
        check_equal(
            leaf_qualified_ids,
            explicit_ids,
            f"{result_path}: exact qualified membership for leaf {leaf!r}",
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
                    config_id in initial_ids and config_id in measured_ids
                    for config_id in lane_exact_ids
                )
                and hierarchical_clc_values_covered(result["family"], lane_exact_ids)
            )
            check_equal(
                lane_result["space_config_count"],
                len(lane_exact_ids) if phase["exact_space_enumerated"] else None,
                f"{result_path}: exact-space count for lane {lane!r}",
            )
            check_equal(
                lane_result["space_exhausted"],
                lane_space_exhausted,
                f"{result_path}: exact-space exhaustion for lane {lane!r}",
            )
            check_equal(
                lane_result["conditional_required"],
                not lane_space_exhausted,
                f"{result_path}: conditional-search requirement for lane {lane!r}",
            )
            witness_id = lane_result["witness_config_id"]
            require(
                metadata_configs[witness_id].get(lane[0]) == lane[1],
                f"{result_path}: lane {lane!r} has an invalid witness",
            )
            witness_succeeded = witness_id in successful_ids
            check_equal(
                lane_result["witness_succeeded"],
                witness_succeeded,
                f"{result_path}: lane {lane!r} witness success",
            )
            require(
                not (set(lane_result["conditional_candidate_ids"]) & initial_ids)
                and len(lane_result["conditional_candidate_ids"])
                == (
                    EXPECTED_CONDITIONAL_CANDIDATES_PER_PIPELINE_LANE
                    if lane_result["conditional_required"]
                    else 0
                )
                and all(
                    metadata_configs[config_id].get(lane[0]) == lane[1]
                    for config_id in lane_result["conditional_candidate_ids"]
                ),
                f"{result_path}: lane {lane!r} has an invalid conditional child",
            )
            successful_conditional_ids = [
                config_id
                for config_id in lane_result["conditional_candidate_ids"]
                if config_id in successful_ids
            ]
            check_equal(
                lane_result["successful_conditional_candidate_ids"],
                successful_conditional_ids,
                f"{result_path}: lane {lane!r} conditional successes",
            )
            repair_ids = lane_result["repair_candidate_ids"]
            require(
                not (set(repair_ids) & initial_ids)
                and not (
                    set(repair_ids) & set(lane_result["conditional_candidate_ids"])
                )
                and all(
                    metadata_configs[config_id].get(lane[0]) == lane[1]
                    for config_id in repair_ids
                ),
                f"{result_path}: lane {lane!r} has an invalid repair child",
            )
            tracked_failure_ids = [
                witness_id,
                *lane_result["conditional_candidate_ids"],
            ]
            generated_repair_ids: list[str] = []
            for repair_index, decision in enumerate(
                lane_result["repair_parent_decisions"]
            ):
                check_equal(
                    decision["repair_index"],
                    repair_index,
                    f"{result_path}: lane {lane!r} repair index",
                )
                generated = decision["generated_config_ids"]
                generated_repair_ids.extend(generated)
                tracked_failure_ids.extend(generated)
            check_equal(
                generated_repair_ids,
                repair_ids,
                f"{result_path}: lane {lane!r} repair children",
            )
            successful_repair_ids = [
                config_id for config_id in repair_ids if config_id in successful_ids
            ]
            check_equal(
                lane_result["successful_repair_candidate_ids"],
                successful_repair_ids,
                f"{result_path}: lane {lane!r} repair successes",
            )
            require(
                witness_succeeded
                or successful_conditional_ids
                or successful_repair_ids,
                f"{result_path}: lane {lane!r} lacks successful evidence",
            )
        retained = lane_diverse_members(
            successful_members,
            lanes,
            limit=provenance["flash_structural_retained_candidates_per_leaf"],
        )
        check_equal(
            result["retained_config_ids"],
            [member["config_id"] for member, _lane in retained],
            f"{result_path}: retained candidates for leaf {leaf!r}",
        )
        check_equal(
            result["complete"], True, f"{result_path}: ordinary leaf completion"
        )
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
                "pipeline_lanes": lanes,
            }
        )

    for result in phase["clc_families"]:
        leaf = {"family": result["family"], "compound_packet": None}
        leaf_key = json.dumps(leaf, sort_keys=True, separators=(",", ":"))
        leaf_exact_ids = exact_by_leaf.get(leaf_key, [])
        matching_leaf_result = next(
            leaf_result
            for leaf_result in phase["leaf_results"]
            if leaf_result["family"] == result["family"]
        )
        check_equal(
            result["space_exhausted"],
            matching_leaf_result["space_exhausted"],
            f"{result_path}: CLC leaf exhaustion",
        )
        expected_value_exhaustion: dict[str, bool] = {}
        best_witness_by_value: dict[int, str] = {}
        for value in result["planned_values"]:
            candidate_ids = [
                result["witness_config_ids"][str(value)],
                *result["witness_repair_candidate_ids"].get(str(value), []),
            ]
            for config_id in candidate_ids:
                reconcile_attempt(config_id)
            require(
                all(
                    isinstance(metadata_configs.get(config_id), dict)
                    and structural_leaf(metadata_configs[config_id]) == leaf
                    and metadata_configs[config_id].get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                    == value
                    for config_id in candidate_ids
                )
                and any(config_id in successful_ids for config_id in candidate_ids),
                f"{result_path}: CLC value {value} lacks a successful witness",
            )
            best_witness_by_value[value] = min(
                (
                    config_id
                    for config_id in candidate_ids
                    if config_id in successful_ids
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
                exact_id in initial_ids and exact_id in measured_ids
                for exact_id in value_exact_ids
            )
        check_equal(
            result["value_space_exhausted"],
            expected_value_exhaustion,
            f"{result_path}: CLC value-space exhaustion",
        )
        expected_selected = [
            value
            for value, _config_id in sorted(
                (
                    (value, best_witness_by_value[value])
                    for value in result["planned_values"]
                ),
                key=lambda item: (
                    selection_perf_by_id[item[1]],
                    item[1],
                    item[0],
                ),
            )[: provenance["flash_structural_retained_candidates_per_leaf"]]
        ]
        check_equal(
            result["selected_values"],
            expected_selected,
            f"{result_path}: selected CLC values",
        )
        for value in result["conditional_values"]:
            conditional_candidate_ids = [
                *result["conditional_candidate_ids"][str(value)],
                *result["conditional_repair_candidate_ids"].get(str(value), []),
            ]
            for config_id in conditional_candidate_ids:
                config = metadata_configs.get(config_id)
                reconcile_attempt(config_id)
                require(
                    isinstance(config, dict)
                    and structural_leaf(config) == leaf
                    and config.get(FLASH_CLC_HEADS_PER_BATCH_KEY) == value,
                    f"{result_path}: CLC value {value} lacks a successful child",
                )
            require(
                any(
                    config_id in successful_ids
                    for config_id in conditional_candidate_ids
                ),
                f"{result_path}: CLC value {value} lacks a successful child",
            )
        ranked_retained: list[tuple[float, str, int]] = []
        for value in expected_selected:
            candidate_ids = [
                result["witness_config_ids"][str(value)],
                *result["witness_repair_candidate_ids"].get(str(value), []),
                *result["conditional_candidate_ids"].get(str(value), []),
                *result["conditional_repair_candidate_ids"].get(str(value), []),
            ]
            best_id = min(
                (
                    config_id
                    for config_id in candidate_ids
                    if config_id in successful_ids
                ),
                key=lambda config_id: (selection_perf_by_id[config_id], config_id),
            )
            ranked_retained.append((selection_perf_by_id[best_id], best_id, value))
        check_equal(
            result["retained_values"],
            [value for _perf, _config_id, value in sorted(ranked_retained)],
            f"{result_path}: retained CLC values",
        )
        pipeline_generated_ids = {
            config_id
            for leaf_result in phase["leaf_results"]
            for round_result in leaf_result["rounds"]
            for config_id in round_result["candidate_config_ids"]
        }
        pre_combination_ids = (
            initial_ids
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
            result_path,
            result,
            leaf=leaf,
            pre_combination_ids=pre_combination_ids,
            successful_ids=successful_ids,
            successful_results_by_id=successful_results_by_id,
            metadata_configs=metadata_configs,
            pipeline_lanes=[
                (lane_result["key"], lane_result["value"])
                for lane_result in matching_leaf_result["pipeline_lanes"]
            ],
            retained_limit=provenance["flash_structural_retained_candidates_per_leaf"],
        )
        combination_ids = result["combination_candidate_ids"]
        cells = result["combination_cells"]
        check_equal(
            combination_ids,
            list(
                dict.fromkeys(
                    cell["projected_config_id"]
                    for cell in cells
                    if isinstance(cell["projected_config_id"], str)
                )
            ),
            f"{result_path}: exact CLC combination ID ledger",
        )
        successful_cell_depth_ids: set[str] = set()
        successful_cell_divisors: set[int] = set()
        qualified_by_id = {
            qualified["config_id"]: qualified
            for qualified in matching_leaf_result["qualified_results"]
        }
        for cell in cells:
            depth_id = cell["depth_config_id"]
            divisor = cell["divisor_value"]
            require(
                depth_id in successful_ids
                and structural_leaf(metadata_configs[depth_id]) == leaf,
                f"{result_path}: CLC cell has an invalid depth source",
            )
            projected_id = cell["projected_config_id"]
            if projected_id is None:
                continue
            projected = metadata_configs.get(projected_id)
            attempt = reconcile_attempt(projected_id)
            require(
                isinstance(projected, dict)
                and canonical_sha256(projected)[:16] == projected_id
                and structural_leaf(projected) == leaf
                and projected.get(FLASH_CLC_HEADS_PER_BATCH_KEY) == divisor,
                f"{result_path}: CLC cell projection changes the wrong schedule axis",
            )
            qualified = qualified_by_id[projected_id]
            check_equal(
                {
                    key: cell[key]
                    for key in (
                        "config_id",
                        "attempt_perf",
                        "selection_perf",
                        "status",
                    )
                },
                {
                    key: qualified[key]
                    for key in (
                        "config_id",
                        "attempt_perf",
                        "selection_perf",
                        "status",
                    )
                },
                f"{result_path}: CLC cell result snapshot",
            )
            check_equal(
                cell["status"],
                attempt["status"],
                f"{result_path}: CLC cell sidecar status",
            )
            if projected_id in successful_ids:
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
            f"{result_path}: exact CLC row coverage",
        )
        check_equal(
            result["successful_combination_divisor_values"],
            expected_successful_divisors,
            f"{result_path}: exact CLC column coverage",
        )
        require(
            all(
                structural_leaf(metadata_configs[config_id]) == leaf
                and metadata_configs[config_id].get(FLASH_CLC_HEADS_PER_BATCH_KEY)
                in result["retained_values"]
                for config_id in combination_ids
            ),
            f"{result_path}: invalid CLC combination candidate",
        )

    qualified_compound_ids: set[str] = set()
    for result in phase["compound_transfers"]:
        result_qualified_ids = set(result["qualified_transfer_config_ids"])
        compound_members: list[dict[str, Any]] = []
        for transfer in result["transfers"]:
            source_id = transfer["source_config_id"]
            target_id = transfer["transferred_config_id"]
            source = metadata_configs.get(source_id)
            target = metadata_configs.get(target_id)
            reconcile_attempt(source_id)
            target_attempt = reconcile_attempt(target_id)
            require(
                isinstance(source, dict)
                and structural_leaf(source)
                == {"family": result["family"], "compound_packet": None}
                and isinstance(target, dict)
                and structural_leaf(target)
                == {
                    "family": result["family"],
                    "compound_packet": result["compound_packet"],
                }
                and source_id in successful_ids,
                f"{result_path}: compound transfer lacks a successful source",
            )
            check_equal(
                transfer["source_config"],
                source,
                f"{result_path}: compound source snapshot",
            )
            check_equal(
                transfer["projected_config"],
                target,
                f"{result_path}: compound projected snapshot",
            )
            check_equal(
                transfer["projected_config_id"],
                canonical_sha256(target)[:16],
                f"{result_path}: compound target canonical ID",
            )
            check_equal(
                transfer["projection_overrides"],
                {FLASH_EXP2_PACKET_KEY: result["compound_packet"]},
                f"{result_path}: compound projection overrides",
            )
            check_equal(
                transfer["preserved_pipeline_values"],
                {
                    key: source[key]
                    for key in FLASH_PIPELINE_QUALIFICATION_KEYS
                    if key in source
                },
                f"{result_path}: compound preserved pipeline values",
            )
            check_equal(
                transfer["status"],
                target_attempt["status"],
                f"{result_path}: compound target status",
            )
            if target_attempt["perf_ms"] is None:
                check_equal(
                    transfer["attempt_perf"],
                    None,
                    f"{result_path}: compound target attempt performance",
                )
            else:
                require(
                    abs(transfer["attempt_perf"] - target_attempt["perf_ms"])
                    <= 0.500001e-6,
                    f"{result_path}: compound target attempt performance",
                )
            if target_id in result_qualified_ids:
                require(
                    target_id in successful_ids,
                    f"{result_path}: qualified compound target lacks success",
                )
                qualified_compound_ids.add(target_id)
                compound_members.append(
                    {
                        "config_id": target_id,
                        "selection_perf": transfer["selection_perf"],
                        "pipeline_lanes": frozenset(),
                    }
                )
            if target_id not in initial_ids:
                explicit_candidate_ids.add(target_id)
        check_equal(
            [member["config_id"] for member in compound_members],
            result["qualified_transfer_config_ids"],
            f"{result_path}: qualified compound retention pool",
        )
        qualified_leaves.append(
            {
                "family": result["family"],
                "compound_packet": result["compound_packet"],
                "members": compound_members,
                "pipeline_lanes": [],
            }
        )

    for family in phase["retained_families"]:
        for starting_path in family["starting_paths"]:
            config_id = starting_path["config_id"]
            config = metadata_configs.get(config_id)
            require(
                isinstance(config, dict)
                and config_id in successful_ids
                and structural_leaf(config)
                == {
                    "family": starting_path["family"],
                    "compound_packet": starting_path["compound_packet"],
                },
                f"{result_path}: retained path is not a successful phase config",
            )
            require(
                starting_path["compound_packet"] is None
                or config_id in qualified_compound_ids,
                f"{result_path}: retained compound path is outside qualified subset",
            )
            lane = starting_path["pipeline_lane"]
            require(
                lane is None or config.get(lane["key"]) == lane["value"],
                f"{result_path}: retained path violates its pipeline lane",
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
    )
    check_equal(
        phase["retained_families"],
        expected_retained_families,
        f"{result_path}: retained structural family ranking",
    )
    check_equal(
        phase["retained_path_count"],
        sum(len(family["starting_paths"]) for family in expected_retained_families),
        f"{result_path}: retained structural path count",
    )
    check_equal(
        phase["candidate_count"],
        len(explicit_candidate_ids),
        f"{result_path}: exact structural qualification candidate count",
    )
    check_equal(
        phase["leaves_with_candidates"],
        leaves_with_candidates,
        f"{result_path}: exact structural qualification leaves with candidates",
    )
    return {
        "successful_candidate_ids": successful_candidate_ids,
        "leaf_results": reconciled_leaves,
    }


def _reconcile_structural_qualification_phase(
    result_path: Path,
    provenance: dict[str, Any],
    phase: dict[str, Any],
    attempt_by_config: dict[str, dict[str, Any]],
    metadata_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return _reconcile_structural_qualification_phase_v16(
        result_path, provenance, phase, attempt_by_config, metadata_configs
    )


def _csv_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(f"{label}: invalid float {value!r}") from exc
    require(math.isfinite(parsed), f"{label}: nonfinite float")
    return parsed


def _metadata_run_id(metadata: dict[str, Any], path: Path) -> str:
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
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_autotune_sidecars(
    result_path: Path,
    provenance: dict[str, Any],
    case: Case,
    selected_config: dict[str, Any],
    selected_source: str,
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    trial = provenance["trials"][0]
    ledger_path = result_path.with_name("autotune.sources.csv")
    require(ledger_path.is_file(), f"{result_path}: missing source ledger")
    try:
        with ledger_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            check_equal(
                tuple(reader.fieldnames or ()),
                SOURCE_LEDGER_FIELDS,
                f"{ledger_path}: header",
            )
            source_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"unable to read {ledger_path}: {exc}") from exc
    require(source_rows, f"{ledger_path}: source ledger is empty")
    for line, row in enumerate(source_rows, 2):
        require(
            None not in row
            and all(row.get(field) is not None for field in SOURCE_LEDGER_FIELDS),
            f"{ledger_path}:{line}: malformed source ledger row",
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", row["run_id"]) is not None,
            f"{ledger_path}:{line}: invalid run ID",
        )
        _csv_float(row["timestamp_s"], f"{ledger_path}:{line}: timestamp_s")
        require(
            re.fullmatch(r"[0-9a-f]{16}", row["config_id"]) is not None,
            f"{ledger_path}:{line}: invalid config ID",
        )
        require(
            row["generation"].isdigit(),
            f"{ledger_path}:{line}: invalid generation",
        )
        require(
            row["status"] in SOURCE_STATUSES,
            f"{ledger_path}:{line}: invalid status",
        )
        require(
            re.fullmatch(r"[0-9a-f]{64}", row["source_hash"]) is not None,
            f"{ledger_path}:{line}: invalid source hash",
        )

    run_ids = {row["run_id"] for row in source_rows}
    require(len(run_ids) == 1, f"{ledger_path}: source ledger mixes run IDs")
    source_run_id = next(iter(run_ids))
    by_config: dict[str, list[dict[str, str]]] = {}
    for row in source_rows:
        by_config.setdefault(row["config_id"], []).append(row)
    for config_id, rows in by_config.items():
        statuses = [row["status"] for row in rows]
        standalone = len(statuses) == 1 and statuses[0] in SOURCE_ALIAS_STATUSES
        standalone_failure = (
            len(statuses) == 1 and statuses[0] in SOURCE_REPAIRABLE_FAILURE_STATUSES
        )
        attempted = (
            len(statuses) == 2
            and statuses[0] == "started"
            and statuses[1] in SOURCE_TERMINAL_STATUSES
        )
        repaired = (
            len(statuses) == 3
            and statuses[0] == "started"
            and statuses[1] in SOURCE_REPAIRABLE_FAILURE_STATUSES
            and statuses[2] == "deduplicated"
        )
        unstarted_repaired = (
            len(statuses) == 2
            and statuses[0] in SOURCE_REPAIRABLE_FAILURE_STATUSES
            and statuses[1] == "deduplicated"
        )
        generations = [row["generation"] for row in rows]
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
            f"{ledger_path}: malformed lifecycle for config {config_id}: {statuses}",
        )
        require(
            len({row["source_hash"] for row in rows}) == 1,
            f"{ledger_path}: inconsistent lifecycle for config {config_id}",
        )
    prior_ok_sources: set[str] = set()
    prior_ok_source_rows: dict[str, dict[str, str]] = {}
    accuracy_failure_sources: set[str] = set()
    pending_transient_configs: dict[str, set[str]] = {}
    accuracy_failure_position: dict[str, int] = {}
    for position, row in enumerate(source_rows):
        source_hash = row["source_hash"]
        if row["status"] in ({"started"} | SOURCE_REPAIRABLE_FAILURE_STATUSES):
            require(
                source_hash not in prior_ok_sources
                and source_hash not in accuracy_failure_sources,
                f"{ledger_path}: source has an attempted outcome after its "
                "definitive outcome",
            )
            if row["status"] in SOURCE_REPAIRABLE_FAILURE_STATUSES:
                pending_transient_configs.setdefault(source_hash, set()).add(
                    row["config_id"]
                )
        if row["status"] == "accuracy_error":
            require(
                source_hash not in prior_ok_sources
                and source_hash not in accuracy_failure_sources,
                f"{ledger_path}: source has more than one definitive outcome",
            )
            accuracy_failure_sources.add(source_hash)
            accuracy_failure_position.setdefault(source_hash, position)
        elif row["status"] == "ok":
            require(
                source_hash not in accuracy_failure_sources
                and source_hash not in prior_ok_sources,
                f"{ledger_path}: source has more than one definitive outcome",
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
                f"{ledger_path}: deduplicated config has no prior successful source "
                "in its repair resolution generation",
            )
            pending_transient_configs.get(source_hash, set()).discard(row["config_id"])
        elif row["status"] == "source_rejected":
            require(
                accuracy_failure_position.get(source_hash, len(source_rows)) < position,
                f"{ledger_path}: source-rejected config has no prior accuracy "
                "failure for its source",
            )
    unrepaired = {
        source_hash: sorted(pending_transient_configs.get(source_hash, set()))
        for source_hash in prior_ok_sources
        if pending_transient_configs.get(source_hash)
    }
    require(
        not unrepaired,
        f"{ledger_path}: successful sources have unrepaired transient attempts: "
        f"{unrepaired}",
    )

    counts = {
        status: sum(row["status"] == status for row in source_rows)
        for status in SOURCE_STATUSES
    }
    num_configs_tested = strict.strict_int(
        trial.get("num_configs_tested"), f"{ledger_path}: tested count", minimum=0
    )
    check_equal(counts["started"], num_configs_tested, f"{ledger_path}: tested count")
    check_equal(
        counts["ok"],
        trial["num_successful_candidate_measurements"],
        f"{ledger_path}: successful count",
    )
    check_equal(
        counts["deduplicated"] + counts["source_rejected"],
        trial["num_source_deduplications"],
        f"{ledger_path}: source alias count",
    )
    check_equal(
        counts["accuracy_error"],
        trial["num_accuracy_failures"],
        f"{ledger_path}: accuracy failure count",
    )
    other_failures = sum(
        counts[status] for status in SOURCE_REPAIRABLE_FAILURE_STATUSES
    )
    check_equal(
        other_failures,
        trial["num_compile_failures"] + trial["num_worker_failures"],
        f"{ledger_path}: non-accuracy failure count",
    )
    check_equal(
        len({row["source_hash"] for row in source_rows}),
        trial["num_unique_sources"],
        f"{ledger_path}: unique source count",
    )
    generations = {int(row["generation"]) for row in source_rows}
    final_generation = trial["num_generations"]
    require(
        0 in generations,
        f"{ledger_path}: search generations omit generation zero",
    )
    require(
        max(generations) == final_generation,
        f"{ledger_path}: search generations do not reach the recorded final generation",
    )
    require(
        all(generation <= final_generation for generation in generations),
        f"{ledger_path}: search generation exceeds the recorded final generation",
    )
    phase = trial.get("search_phase_metrics")
    require(isinstance(phase, dict), f"{ledger_path}: missing structural search phase")
    timeline = phase.get("measurement_timeline")
    pass_count = phase.get("qualification_passes_completed")
    anchor_started = phase.get("schedule_anchor_pass_started")
    require(
        isinstance(timeline, list)
        and type(pass_count) is int
        and type(anchor_started) is bool
        and len(timeline) == pass_count + 1,
        f"{ledger_path}: malformed structural generation timeline",
    )
    anchor_offset = int(anchor_started)
    structural_generation_limit = pass_count - anchor_offset
    require(
        structural_generation_limit >= 0,
        f"{ledger_path}: malformed structural generation limit",
    )
    earliest_generation_by_config: dict[str, int] = {}
    for row in source_rows:
        earliest_generation_by_config.setdefault(
            row["config_id"], int(row["generation"])
        )
    expected_config_ids_by_generation: dict[int, set[str]] = {}
    introduced_config_ids: set[str] = set()
    for pass_index, record in enumerate(timeline):
        require(
            isinstance(record, dict) and isinstance(record.get("updates"), list),
            f"{ledger_path}: malformed structural generation timeline",
        )
        update_ids: set[str] = set()
        for update in record["updates"]:
            config_id = update.get("config_id") if isinstance(update, dict) else None
            require(
                isinstance(config_id, str),
                f"{ledger_path}: malformed structural generation update",
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
                f"{ledger_path}: config {config_id} first ledger generation",
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
        f"{ledger_path}: structural config generations",
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
        f"{ledger_path}: measured structural search generations",
    )
    selected_config_id = canonical_sha256(selected_config)[:16]
    selected_rows = [
        row
        for row in source_rows
        if row["config_id"] == selected_config_id
        and row["source_hash"] == selected_source
        and row["status"] in {"ok", "deduplicated"}
    ]
    require(
        len(selected_rows) == 1,
        f"{ledger_path}: selected config is not linked to its successful source",
    )

    csv_path = result_path.with_name("autotune.csv")
    require(csv_path.is_file(), f"{result_path}: missing autotune CSV")
    try:
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            check_equal(
                tuple(reader.fieldnames or ()),
                AUTOTUNE_CSV_FIELDS,
                f"{csv_path}: header",
            )
            csv_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"unable to read {csv_path}: {exc}") from exc
    check_equal(len(csv_rows), len(source_rows), f"{csv_path}: source row count")
    for line, (row, source_row) in enumerate(
        zip(csv_rows, source_rows, strict=True), 2
    ):
        require(
            None not in row
            and all(row.get(field) is not None for field in AUTOTUNE_CSV_FIELDS),
            f"{csv_path}:{line}: malformed row",
        )
        for field in AUTOTUNE_JOIN_FIELDS:
            check_equal(row[field], source_row[field], f"{csv_path}:{line}: {field}")
        successful = row["status"] in {"ok", "deduplicated"}
        check_equal(bool(row["perf_ms"]), successful, f"{csv_path}:{line}: perf")
        if successful:
            require(
                _csv_float(row["perf_ms"], f"{csv_path}:{line}: perf_ms") > 0,
                f"{csv_path}:{line}: nonpositive performance",
            )
        if row["compile_time_s"]:
            require(
                _csv_float(row["compile_time_s"], f"{csv_path}:{line}: compile_time_s")
                >= 0,
                f"{csv_path}:{line}: negative compile time",
            )

    attempt_by_config: dict[str, dict[str, Any]] = {}
    attempt_history_by_config: dict[str, list[dict[str, Any]]] = {}
    for position, row in enumerate(csv_rows):
        if row["status"] == "started":
            continue
        config_id = row["config_id"]
        previous = attempt_by_config.get(config_id)
        require(
            previous is None
            or (
                previous["status"] in SOURCE_REPAIRABLE_FAILURE_STATUSES
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

    metadata_path = result_path.with_name("autotune.meta.jsonl")
    require(metadata_path.is_file(), f"{result_path}: missing autotune metadata")
    try:
        records = [
            json.loads(line)
            for line in metadata_path.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to read {metadata_path}: {exc}") from exc
    require(len(records) == 1, f"{metadata_path}: expected one metadata record")
    metadata = records[0]
    require(isinstance(metadata, dict), f"{metadata_path}: metadata is not an object")
    check_equal(
        metadata.get("run_id"),
        _metadata_run_id(metadata, metadata_path),
        f"{metadata_path}: computed run ID",
    )
    check_equal(metadata.get("run_id"), source_run_id, f"{metadata_path}: run ID")
    shape = (2, 32, case.seq_len, 64)
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
    check_equal(
        settings.get("autotune_random_seed"),
        trial["random_seed"],
        f"{metadata_path}: tuner seed",
    )
    metadata_configs = metadata.get("configs")
    require(isinstance(metadata_configs, dict), f"{metadata_path}: configs map invalid")
    check_equal(set(metadata_configs), set(by_config), f"{metadata_path}: config IDs")
    for config_id, config in metadata_configs.items():
        require(
            isinstance(config, dict), f"{metadata_path}: invalid config {config_id}"
        )
        check_equal(
            canonical_sha256(config)[:16],
            config_id,
            f"{metadata_path}: canonical config ID",
        )
    check_equal(
        metadata_configs.get(selected_config_id),
        selected_config,
        f"{metadata_path}: selected config",
    )
    for line, row in enumerate(csv_rows, 2):
        config = metadata_configs[row["config_id"]]
        config_repr = (
            "Config("
            + ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))
            + ")"
        )
        check_equal(row["config"], config_repr, f"{csv_path}:{line}: config payload")
    return (
        source_rows,
        metadata_configs,
        attempt_by_config,
        attempt_history_by_config,
    )


def _validate_structural_prefix_execution(
    result_path: Path,
    provenance: dict[str, Any],
    case: Case,
    selected_config: dict[str, Any],
    selected_source: str,
) -> dict[str, Any]:
    trials = provenance.get("trials")
    require(
        isinstance(trials, list) and len(trials) == 1 and isinstance(trials[0], dict),
        f"{result_path}: expected one autotune trial",
    )
    phase = trials[0].get("search_phase_metrics")
    require(
        isinstance(phase, dict),
        f"{result_path}: missing structural qualification phase",
    )
    require(
        phase.get("phase") == "cute_flash_structural_qualification_v22"
        and phase.get("cute_flash_lane_policy_version") == EXPECTED_LANE_POLICY_VERSION,
        f"{result_path}: expected v22 structural qualification evidence",
    )
    leaf_catalog = provenance.get("flash_structural_leaf_catalog")
    require(
        isinstance(leaf_catalog, list) and leaf_catalog,
        f"{result_path}: missing live structural leaf catalog",
    )
    retained_family_cap = provenance.get("flash_structural_retained_family_cap")
    retained_family_limit = provenance.get("flash_structural_retained_family_limit")
    live_family_count = len(
        {
            leaf.get("family")
            for leaf in leaf_catalog
            if isinstance(leaf, dict) and leaf.get("compound_packet") is None
        }
    )
    expected_family_limit = min(EXPECTED_RETAINED_FAMILY_CAP, live_family_count)
    check_equal(
        retained_family_cap,
        EXPECTED_RETAINED_FAMILY_CAP,
        f"{result_path}: full-search retained family cap",
    )
    check_equal(
        retained_family_limit,
        expected_family_limit,
        f"{result_path}: live-derived retained family limit",
    )
    check_equal(
        phase.get("retained_family_cap"),
        retained_family_cap,
        f"{result_path}: phase retained family cap",
    )
    check_equal(
        phase.get("retained_family_limit"),
        retained_family_limit,
        f"{result_path}: phase retained family limit",
    )
    require(
        phase.get("completed") is True,
        f"{result_path}: structural qualification is incomplete",
    )
    require(
        phase.get("budget_exhausted") is False,
        f"{result_path}: structural qualification ended with budget exhaustion",
    )
    qualification_passes = phase.get("qualification_passes_planned")
    require(
        type(qualification_passes) is int
        and qualification_passes > 0
        and phase.get("qualification_passes_started") == qualification_passes
        and phase.get("qualification_passes_completed") == qualification_passes,
        f"{result_path}: inconsistent qualification pass accounting",
    )
    require(
        phase.get("qualification_rounds_started") == qualification_passes,
        f"{result_path}: inconsistent qualification_rounds_started",
    )
    require(
        phase.get("qualification_rounds_completed") == qualification_passes,
        f"{result_path}: inconsistent qualification_rounds_completed",
    )
    starting_path_limit = provenance.get("flash_structural_starting_path_limit")
    family_probe_path_limit = provenance.get("flash_structural_family_probe_path_limit")
    maximum_path_capacity = provenance.get("flash_structural_maximum_path_capacity")
    require(
        type(starting_path_limit) is int
        and starting_path_limit > 0
        and phase.get("starting_path_limit") == starting_path_limit,
        f"{result_path}: invalid live-derived starting path capacity",
    )
    compound_count = sum(
        isinstance(leaf, dict) and leaf.get("compound_packet") is not None
        for leaf in leaf_catalog
    )
    expected_probe_path_limit = (
        live_family_count + compound_count + 1
        if live_family_count > EXPECTED_RETAINED_FAMILY_CAP
        else 0
    )
    check_equal(
        family_probe_path_limit,
        expected_probe_path_limit,
        f"{result_path}: live-derived family probe path capacity",
    )
    check_equal(
        phase.get("family_probe_path_limit"),
        family_probe_path_limit,
        f"{result_path}: phase family probe path capacity",
    )
    check_equal(
        maximum_path_capacity,
        max(starting_path_limit, family_probe_path_limit),
        f"{result_path}: maximum path capacity",
    )
    check_equal(
        phase.get("maximum_path_capacity"),
        maximum_path_capacity,
        f"{result_path}: phase maximum path capacity",
    )
    check_equal(
        phase.get("family_probe_generations"),
        EXPECTED_FAMILY_PROBE_GENERATIONS,
        f"{result_path}: family probe generations",
    )
    check_equal(
        phase.get("family_probe_candidates_per_path"),
        EXPECTED_FAMILY_PROBE_CANDIDATES_PER_PATH,
        f"{result_path}: family probe candidates per path",
    )
    require(
        phase.get("unrestricted_path_exhausts_generation_budget") is True,
        f"{result_path}: unrestricted path does not exhaust the generation budget",
    )
    (
        source_rows,
        metadata_configs,
        attempt_by_config,
        attempt_history_by_config,
    ) = _validate_autotune_sidecars(
        result_path, provenance, case, selected_config, selected_source
    )
    terminal_refinement = strict.validate_terminal_coordinate_refinement(
        result_path,
        provenance,
        trials[0],
        phase,
        source_rows,
        metadata_configs,
        attempt_by_config,
        attempt_history_by_config,
    )
    required_preterminal = terminal_refinement["required_preterminal_candidate_count"]
    require(
        terminal_refinement["preterminal_effective_candidate_count"]
        >= required_preterminal,
        f"{result_path}: terminal work masks an undersized full-search candidate set",
    )
    if not phase["exact_space_config_ids"]:
        require(
            terminal_refinement["preterminal_successful_measurement_count"]
            >= required_preterminal,
            f"{result_path}: terminal work masks fewer than 100 measured "
            "preterminal successes",
        )
    measurement_states = validate_measurement_timeline(
        result_path, phase, metadata_configs
    )
    successful_probe_candidate_ids = _validate_family_probe_execution(
        result_path,
        phase,
        leaf_catalog,
        metadata_configs,
        measurement_states,
    )
    invalidations = isolated_rebenchmark_invalidations(phase)
    timed_out_sources = isolated_rebenchmark_timeout_source_hashes(phase)
    require(
        provenance["trials"][0]["num_isolated_rebenchmark_timeouts"]
        >= len(timed_out_sources),
        f"{result_path}: fewer isolated rebenchmark timeouts than distinct "
        "timed-out generated sources",
    )
    for config_id, invalidation in invalidations.items():
        attempt = attempt_by_config.get(config_id)
        require(
            isinstance(attempt, dict)
            and attempt.get("status") in {"ok", "deduplicated"}
            and attempt.get("source_hash") == invalidation["source_hash"],
            f"{result_path}: isolated rebenchmark invalidation lacks a matching "
            "successful sidecar source",
        )
    validate_timeline_source_repairs(result_path, phase, attempt_history_by_config)
    for leaf in phase.get("leaf_results", []):
        if not isinstance(leaf, dict):
            continue
        for result in leaf.get("qualified_results", []):
            require(
                isinstance(result, dict) and isinstance(result.get("config_id"), str),
                f"{result_path}: malformed qualified measurement snapshot",
            )
            validate_measurement_snapshot(
                result_path,
                measurement_states,
                result,
                config_id=result["config_id"],
                label="qualified measurement snapshot is not backed by the timeline",
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
        f"{result_path}: injected structural design prefix exceeds its dynamic budget",
    )
    prefix_ids = [canonical_sha256(config)[:16] for config in prefix_configs]
    require(
        len(set(prefix_ids)) == len(prefix_ids),
        f"{result_path}: structural design prefix has a config ID collision",
    )
    attempted_ids: set[str] = set()
    successful_prefix_ids: set[str] = set()
    for config_id, config in zip(prefix_ids, prefix_configs, strict=True):
        check_equal(
            metadata_configs.get(config_id),
            config,
            f"{result_path}: structural prefix metadata config {config_id}",
        )
        generation_zero = [
            row
            for row in source_rows
            if row["config_id"] == config_id and row["generation"] == "0"
        ]
        require(
            any(
                row["status"] in ({"started"} | SOURCE_ALIAS_STATUSES)
                for row in generation_zero
            ),
            f"{result_path}: structural prefix config {config_id} was not attempted "
            "at generation 0",
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
        and row["status"] in ({"started"} | SOURCE_ALIAS_STATUSES)
    }
    initial_population_size = provenance["autotune_initial_population_size"]
    require(
        isinstance(initial_population_size, int)
        and not isinstance(initial_population_size, bool)
        and initial_population_size > 0,
        f"{result_path}: invalid initial population size",
    )
    expected_generation_zero_count = (
        len(phase["exact_space_config_ids"])
        if phase["exact_space_enumerated"]
        else initial_population_size
    )
    initial_population_order = phase["initial_config_ids"]
    check_equal(
        initial_population_order,
        strict.generation_zero_initial_config_order(
            result_path, source_rows, expected_generation_zero_count
        ),
        f"{result_path}: generation-zero initial population order",
    )
    initial_population_ids = set(initial_population_order)
    check_equal(
        len(initial_population_ids),
        expected_generation_zero_count,
        f"{result_path}: distinct initial-population config count",
    )
    require(
        initial_population_ids <= generation_zero_attempted_ids,
        f"{result_path}: initial population is absent from generation-0 sidecars",
    )
    successful_generation_zero_ids = {
        row["config_id"]
        for row in source_rows
        if row["generation"] == "0"
        and row["status"] in {"ok", "deduplicated"}
        and row["config_id"] not in invalidations
        and row["config_id"] in initial_population_ids
    }
    compiler_seed_policy = strict.validate_compiler_seed_policy(
        result_path,
        provenance,
        phase=phase,
        metadata_configs=metadata_configs,
        source_rows=source_rows,
        invalidated_config_ids=set(invalidations),
    )
    if phase["exact_space_enumerated"]:
        require(
            set(phase["exact_space_config_ids"]) <= successful_generation_zero_ids,
            f"{result_path}: exact structural search space was not successfully exhausted",
        )
    successful_generation_zero_configs = []
    for config_id in successful_generation_zero_ids:
        config = metadata_configs.get(config_id)
        require(
            isinstance(config, dict),
            f"{result_path}: successful generation-0 config {config_id} is absent "
            "from metadata",
        )
        successful_generation_zero_configs.append(config)

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
        f"{result_path}: qualified structural values lack a successful generation-0 "
        f"representative: {missing_qualification!r}",
    )
    missing_interactions = [
        item
        for item in interaction_counts
        if item["successful_generation_zero_witness_count"] < 1
    ]
    require(
        not missing_interactions,
        f"{result_path}: structural interactions lack a successful generation-0 "
        f"representative: {missing_interactions!r}",
    )
    successful_candidate_ids = {
        config_id
        for leaf in phase.get("leaf_results", [])
        if isinstance(leaf, dict)
        for result in leaf.get("qualified_results", [])
        if isinstance(result, dict)
        and result.get("status") in {"ok", "deduplicated"}
        and isinstance((config_id := result.get("config_id")), str)
    }
    successful_candidate_ids.update(successful_probe_candidate_ids)
    require(
        successful_candidate_ids <= metadata_configs.keys(),
        f"{result_path}: qualified config is absent from autotune metadata",
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
            successful_candidate_ids
        ),
        "structural_qualification_retained_family_count": len(
            phase["retained_families"]
        ),
        "structural_qualification_retained_path_count": phase["retained_path_count"],
        "structural_qualification_leaf_results": phase["leaf_results"],
        "terminal_refinement": terminal_refinement,
    }


def _validate_family_probe_execution(
    result_path: Path,
    phase: dict[str, Any],
    leaf_catalog: list[dict[str, Any]],
    metadata_configs: dict[str, dict[str, Any]],
    measurement_states: list[dict[str, dict[str, Any]]],
) -> set[str]:
    """Validate the measured pre-promotion family probe and its ranking."""
    path_limit = phase["family_probe_path_limit"]
    required = bool(path_limit and not phase["exact_space_exhausted"])
    check_equal(
        phase["family_probe_required"],
        required,
        f"{result_path}: family probe requirement",
    )
    check_equal(
        phase["family_probe_complete"],
        True,
        f"{result_path}: family probe completion",
    )
    paths = phase["family_probe_paths"]
    require(
        isinstance(paths, list) and len(paths) == (path_limit if required else 0),
        f"{result_path}: family probe path catalog",
    )
    if not required:
        return set()

    generations = phase["family_probe_generations"]
    pre_probe_pass = phase["qualification_passes_completed"] - generations
    require(
        type(generations) is int
        and generations > 0
        and 0 <= pre_probe_pass < len(measurement_states),
        f"{result_path}: family probe pass accounting",
    )
    pre_probe_states = measurement_states[pre_probe_pass]
    successful_statuses = {"ok", "deduplicated"}
    qualified_compound_ids = {
        config_id
        for transfer in phase["compound_transfers"]
        for config_id in transfer["qualified_transfer_config_ids"]
    }
    ordinary_by_family: dict[str, list[str]] = {}
    for config_id, state in pre_probe_states.items():
        leaf = structural_leaf(metadata_configs[config_id])
        if (
            state["status"] in successful_statuses
            and leaf is not None
            and leaf["compound_packet"] is None
        ):
            ordinary_by_family.setdefault(leaf["family"], []).append(config_id)

    def rank(config_id: str) -> tuple[float, str]:
        selection_perf = pre_probe_states[config_id].get("selection_perf")
        require(
            isinstance(selection_perf, (int, float))
            and not isinstance(selection_perf, bool)
            and math.isfinite(selection_perf)
            and selection_perf > 0,
            f"{result_path}: invalid family probe starting performance",
        )
        return float(selection_perf), config_id

    expected_starts: list[tuple[str, dict[str, Any], bool]] = []
    ordinary_starts = [
        min(config_ids, key=rank) for config_ids in ordinary_by_family.values()
    ]
    for config_id in sorted(
        ordinary_starts,
        key=lambda item: (
            rank(item),
            structural_leaf(metadata_configs[item])["family"],
        ),
    ):
        leaf = structural_leaf(metadata_configs[config_id])
        assert leaf is not None
        expected_starts.append((config_id, leaf, False))

    compound_starts: list[tuple[str, dict[str, Any], bool]] = []
    for leaf in leaf_catalog:
        if leaf["compound_packet"] is None:
            continue
        candidates = [
            config_id
            for config_id in qualified_compound_ids
            if structural_leaf(metadata_configs[config_id]) == leaf
            and pre_probe_states.get(config_id, {}).get("status") in successful_statuses
        ]
        require(candidates, f"{result_path}: compound family probe start")
        config_id = min(candidates, key=rank)
        compound_starts.append((config_id, leaf, False))
    expected_starts.extend(
        sorted(
            compound_starts,
            key=lambda item: (
                rank(item[0]),
                item[1]["family"],
                item[1]["compound_packet"],
                item[1]["softmax_disc"],
            ),
        )
    )
    eligible = {
        config_id
        for config_ids in ordinary_by_family.values()
        for config_id in config_ids
    } | {
        config_id
        for config_id in qualified_compound_ids
        if pre_probe_states.get(config_id, {}).get("status") in successful_statuses
    }
    require(eligible, f"{result_path}: empty family probe population")
    global_start = min(eligible, key=rank)
    global_leaf = structural_leaf(metadata_configs[global_start])
    assert global_leaf is not None
    expected_starts.append((global_start, global_leaf, True))
    check_equal(
        len(expected_starts),
        path_limit,
        f"{result_path}: family probe start count",
    )

    seen_candidate_ids: set[str] = set()
    successful_candidate_ids: set[str] = set()
    parent_scores: dict[str, list[tuple[float, str, dict[str, Any]]]] = {}
    final_probe_states = measurement_states[phase["qualification_passes_completed"]]
    for config_id, pre_probe_state in pre_probe_states.items():
        state = final_probe_states[config_id]
        leaf = structural_leaf(metadata_configs[config_id])
        selection_perf = state.get("selection_perf")
        if (
            pre_probe_state["status"] in successful_statuses
            and state["status"] in successful_statuses
            and leaf is not None
            and leaf["compound_packet"] is None
            and optional_positive_float(selection_perf)
        ):
            parent_scores.setdefault(leaf["family"], []).append(
                (float(selection_perf), config_id, leaf)
            )

    for raw_path, (start_id, start_leaf, unrestricted) in zip(
        paths, expected_starts, strict=True
    ):
        require(
            isinstance(raw_path, dict)
            and raw_path["starting_config_id"] == start_id
            and raw_path["unrestricted"] is unrestricted
            and {
                "family": raw_path["family"],
                "compound_packet": raw_path["compound_packet"],
                "softmax_disc": raw_path["softmax_disc"],
            }
            == start_leaf
            and isinstance(raw_path["rounds"], list)
            and len(raw_path["rounds"]) == generations,
            f"{result_path}: malformed family probe path",
        )
        for generation_index, round_record in enumerate(raw_path["rounds"], start=1):
            expected_pass = pre_probe_pass + generation_index
            candidate_ids = _config_id_list(
                round_record.get("candidate_ids"),
                f"{result_path}: family probe candidate IDs",
            )
            results = round_record.get("results")
            require(
                round_record.get("probe_generation") == generation_index
                and round_record.get("measurement_pass_index") == expected_pass
                and isinstance(results, list)
                and len(candidate_ids) <= phase["family_probe_candidates_per_path"] - 1
                and not (set(candidate_ids) & seen_candidate_ids)
                and not (set(candidate_ids) & set(pre_probe_states)),
                f"{result_path}: invalid family probe round",
            )
            result_ids: list[str] = []
            for result in results:
                require(
                    isinstance(result, dict)
                    and isinstance(result.get("config_id"), str)
                    and result["config_id"] in candidate_ids
                    and terminal_measurement_is_valid(result),
                    f"{result_path}: invalid family probe result",
                )
                config_id = result["config_id"]
                candidate_leaf = structural_leaf(metadata_configs[config_id])
                require(
                    candidate_leaf is not None
                    and (unrestricted or candidate_leaf == start_leaf),
                    f"{result_path}: family probe changed a constrained leaf",
                )
                validate_measurement_snapshot(
                    result_path,
                    measurement_states,
                    result,
                    config_id=config_id,
                    label="family probe result is not backed by the timeline",
                    expected_pass_index=expected_pass,
                )
                final_state = final_probe_states.get(config_id, {})
                if final_state.get("status") in successful_statuses:
                    successful_candidate_ids.add(config_id)
                    if not unrestricted and candidate_leaf["compound_packet"] is None:
                        selection_perf = final_state.get("selection_perf")
                        require(
                            optional_positive_float(selection_perf),
                            f"{result_path}: family probe parent score",
                        )
                        parent_scores.setdefault(candidate_leaf["family"], []).append(
                            (float(selection_perf), config_id, candidate_leaf)
                        )
                result_ids.append(config_id)
            check_equal(
                result_ids,
                candidate_ids,
                f"{result_path}: family probe result order",
            )
            seen_candidate_ids.update(candidate_ids)

    ranked_families = sorted(
        parent_scores,
        key=lambda family: (min(parent_scores[family])[0], family),
    )
    best_perf = min(parent_scores[ranked_families[0]])[0]
    expected_promoted = set(
        [
            family
            for family in ranked_families
            if min(parent_scores[family])[0]
            <= best_perf * phase["retained_family_slowdown_limit"]
        ][: phase["retained_family_limit"]]
    )
    retained = phase["retained_families"]
    actual_promoted = {
        family["family"] for family in retained if family["parent_promoted"]
    }
    check_equal(
        actual_promoted,
        expected_promoted,
        f"{result_path}: family probe promotion ranking",
    )
    for family in retained:
        score, _score_id, score_leaf = min(parent_scores[family["family"]])
        require(
            abs(family["score"] - score) <= 0.500001e-6
            and family["score_compound_packet"] is None
            and family["score_softmax_disc"] == score_leaf["softmax_disc"],
            f"{result_path}: retained family score",
        )
    return successful_candidate_ids


def _selected_source_sha256(
    provenance: dict[str, Any], selected_config: dict[str, Any], path: Path, case: Case
) -> tuple[str, list[dict[str, Any]]]:
    declared = {
        value
        for key in ("selected_source_sha256", "selected_source_hash")
        if isinstance((value := provenance.get(key)), str)
    }
    trials = provenance.get("trials")
    require(isinstance(trials, list) and trials, f"{path}: missing autotune trials")
    expected_trials = provenance.get("autotune_best_of_k")
    require(
        isinstance(expected_trials, int) and expected_trials > 0,
        f"{path}: invalid autotune_best_of_k",
    )
    check_equal(len(trials), expected_trials, f"{path}: autotune trial count")
    lfbo_max_generations = provenance.get("autotune_lfbo_max_generations")
    require(
        isinstance(lfbo_max_generations, int)
        and not isinstance(lfbo_max_generations, bool)
        and lfbo_max_generations > 0,
        f"{path}: provenance.autotune_lfbo_max_generations must be a positive integer",
    )
    matching_trials = []
    shape = (2, 32, case.seq_len, 64)
    for index, trial in enumerate(trials, 1):
        require(isinstance(trial, dict), f"{path}: trial {index} is not an object")
        check_equal(
            trial.get("input_shapes"),
            repr([shape, shape, shape]),
            f"{path}: trial {index} input shapes",
        )
        check_equal(
            trial.get("dtypes"),
            repr(["torch.float16"] * 3),
            f"{path}: trial {index} input dtypes",
        )
        check_equal(
            trial.get("hardware"),
            "NVIDIA B200",
            f"{path}: trial {index} hardware",
        )
        tested = trial.get("num_configs_tested")
        successful = trial.get("num_successful_candidate_measurements")
        isolated_rebenchmark_timeouts = trial.get("num_isolated_rebenchmark_timeouts")
        require(
            isinstance(tested, int) and tested >= 100,
            f"{path}: trial {index} tested fewer than 100 candidates",
        )
        require(
            isinstance(successful, int) and successful >= 100,
            f"{path}: trial {index} has fewer than 100 successful measurements",
        )
        require(
            type(isolated_rebenchmark_timeouts) is int
            and isolated_rebenchmark_timeouts >= 0,
            f"{path}: trial {index} has an invalid isolated rebenchmark timeout count",
        )
        require(
            isinstance(trial.get("num_unique_sources"), int)
            and trial["num_unique_sources"] > 0,
            f"{path}: trial {index} has no unique generated sources",
        )
        require(
            isinstance(trial.get("num_generations"), int)
            and trial["num_generations"] > 0,
            f"{path}: trial {index} has no LFBO generations",
        )
        check_equal(
            trial.get("num_generations"),
            lfbo_max_generations,
            f"{path}: trial {index} unrestricted path generation budget",
        )
        check_equal(
            trial.get("search_algorithm"),
            "LFBOTreeSearch",
            f"{path}: trial {index} search algorithm",
        )
        check_equal(
            trial.get("selected_source_was_measured"),
            True,
            f"{path}: trial {index} measured-source link",
        )
        if trial.get("selected_config") == selected_config:
            source_hash = trial.get("selected_source_hash")
            require(
                isinstance(source_hash, str),
                f"{path}: matching trial {index} lacks selected source hash",
            )
            declared.add(source_hash)
            matching_trials.append(trial)
    require(matching_trials, f"{path}: selected config was not a measured trial winner")
    require(
        len(declared) == 1, f"{path}: inconsistent selected source hashes: {declared}"
    )
    selected_source = next(iter(declared))
    require(
        re.fullmatch(r"[0-9a-f]{64}", selected_source) is not None,
        f"{path}: invalid selected source SHA256 {selected_source!r}",
    )
    return selected_source, matching_trials


def validate_strict_result(
    path: Path, case: Case, checkout: dict[str, object]
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    check_equal(payload.get("impl"), "helion-cute", f"{path}: implementation")
    expected_shape = {
        "z": 2,
        "h": 32,
        "seq_len": case.seq_len,
        "head_dim": 64,
        "dtype": "float16",
        "causal": int(case.causal),
        "biased": 0,
    }
    check_equal(payload.get("shape"), expected_shape, f"{path}: shape")
    check_equal(payload.get("gpu"), "NVIDIA B200", f"{path}: search GPU model")
    check_equal(
        payload.get("flop_model"),
        "softmax_attention_forward",
        f"{path}: flop model",
    )
    check_equal(payload.get("accuracy"), "PASS", f"{path}: search accuracy")
    check_equal(payload.get("benchmark_timer"), "wall", f"{path}: benchmark timer")
    check_equal(
        str(payload.get("physical_gpu")), str(case.physical_gpu), f"{path}: GPU"
    )
    check_equal(
        float(payload.get("power_cap_w")), EXPECTED_POWER_CAP_W, f"{path}: power cap"
    )
    input_seed = payload.get("input_seed")
    require(
        isinstance(input_seed, int) and not isinstance(input_seed, bool),
        f"{path}: strict result lacks an integer input_seed",
    )
    cute_version = _validate_version(
        payload.get("version"), str(checkout["runtime_git_head"]), path
    )

    overrides = payload.get("helion_overrides")
    require(isinstance(overrides, dict), f"{path}: missing helion_overrides")
    expected_override_values = {
        "config_overrides": {},
        "seed_config_overrides": {},
        "autotuned": True,
        "force_autotune": True,
        "return_lse": False,
    }
    for key, expected in expected_override_values.items():
        check_equal(overrides.get(key), expected, f"{path}: helion_overrides.{key}")
    provenance = overrides.get("autotune_provenance")
    require(isinstance(provenance, dict), f"{path}: missing autotune provenance")

    strict_values = {
        "require_full_autotune": True,
        "effort": "full",
        "requested_force_autotune": True,
        "effective_force_autotune": True,
        "fixed_config": False,
        "autotune_budget_seconds": None,
        "autotune_max_generations": None,
        "autotune_best_of_k": 1,
        "autotune_config_overrides": {},
        "user_seed_configs": False,
        "disable_autotuner_heuristics": False,
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
        "autotune_benchmark_subprocess": True,
        "autotune_adaptive_timeout": True,
        "autotune_force_persistent": False,
        "autotune_finishing_rounds_env": "",
        "autotune_ignore_errors": False,
        "autotune_search_acf": [],
        "autotune_config_filter": False,
        "flash_value_prior_keys": [],
        "flash_structural_qualification_rounds": 2,
        "flash_structural_qualification_pipeline_candidate_limit_per_leaf_per_round": 4,
        "flash_structural_family_probe_generations": (
            EXPECTED_FAMILY_PROBE_GENERATIONS
        ),
        "flash_structural_family_probe_candidates_per_path": (
            EXPECTED_FAMILY_PROBE_CANDIDATES_PER_PATH
        ),
        "flash_structural_retained_candidates_per_leaf": 2,
        "flash_structural_retained_family_cap": EXPECTED_RETAINED_FAMILY_CAP,
        "flash_structural_retained_family_slowdown_limit": 2.0,
        "flash_structural_unrestricted_path_exhausts_generation_budget": True,
        "cute_flash_env_overrides": {},
        "final_correctness_enabled": True,
        "final_correctness_launches": EXPECTED_FINAL_CORRECTNESS_LAUNCHES,
        "final_repeatability_passed": True,
        "final_correctness_passed": True,
        "autotune_cache": "LocalAutotuneCache",
        "rebenchmark_env_overrides": {},
    }
    for key, expected in strict_values.items():
        check_equal(provenance.get(key), expected, f"{path}: provenance.{key}")
    strict.validate_compiler_seed_policy(path, provenance)
    starting_path_limit = provenance.get("flash_structural_starting_path_limit")
    require(
        type(starting_path_limit) is int and starting_path_limit > 0,
        f"{path}: provenance.flash_structural_starting_path_limit must be a "
        "positive live-derived capacity",
    )
    require(
        isinstance(provenance.get("autotune_compile_timeout"), int)
        and provenance["autotune_compile_timeout"] >= 60,
        f"{path}: compile timeout below 60 seconds",
    )
    require(
        isinstance(provenance.get("autotune_benchmark_timeout"), int)
        and provenance["autotune_benchmark_timeout"] >= 30,
        f"{path}: benchmark timeout below 30 seconds",
    )
    require(
        not provenance.get("dense_d64_2cta_performance_anchor_present", False),
        f"{path}: legacy dense shape anchor is active",
    )
    check_equal(
        provenance.get("active_value_prior_keys", []),
        [],
        f"{path}: active config value priors",
    )
    coverage_configs = _validate_coverage(provenance, path)

    selected_config = provenance.get("selected_config")
    require(isinstance(selected_config, dict), f"{path}: missing selected config")
    selected_source, matching_trials = _selected_source_sha256(
        provenance, selected_config, path, case
    )
    structural_execution = _validate_structural_prefix_execution(
        path, provenance, case, selected_config, selected_source
    )
    distance = provenance.get(
        "selected_config_nearest_structural_coverage_design_field_distance"
    )
    nearest = provenance.get(
        "selected_config_nearest_structural_coverage_design_config_sha256"
    )
    require(
        isinstance(distance, int) and distance >= 0,
        f"{path}: invalid winner-to-coverage distance",
    )
    require(
        isinstance(nearest, list)
        and nearest
        and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in nearest),
        f"{path}: invalid nearest coverage digests",
    )
    computed_distances = [
        sum(
            key not in selected_config
            or key not in candidate
            or selected_config[key] != candidate[key]
            for key in selected_config.keys() | candidate.keys()
        )
        for candidate in coverage_configs
    ]
    computed_distance = min(computed_distances)
    computed_nearest = sorted(
        canonical_sha256(candidate)
        for candidate, candidate_distance in zip(
            coverage_configs, computed_distances, strict=True
        )
        if candidate_distance == computed_distance
    )
    check_equal(distance, computed_distance, f"{path}: winner-to-coverage distance")
    check_equal(
        sorted(nearest), computed_nearest, f"{path}: nearest coverage config digests"
    )
    check_equal(
        provenance.get("selected_config_is_structural_coverage_design_member"),
        distance == 0,
        f"{path}: coverage membership flag",
    )

    return {
        **checkout,
        "artifact_root": str(path.parent),
        "search_result_path": str(path.resolve()),
        "search_result_sha256": sha256(path),
        "search_version": payload["version"],
        "cute_version": cute_version,
        "input_seed": input_seed,
        "search_physical_gpu": case.physical_gpu,
        "search_power_cap_w": float(payload["power_cap_w"]),
        "search_benchmark_timer": payload.get("benchmark_timer"),
        "selected_config": selected_config,
        "selected_config_sha256": canonical_sha256(selected_config),
        "selected_source_sha256": selected_source,
        "winner_to_coverage_field_distance": distance,
        "winner_is_coverage_design_member": provenance.get(
            "selected_config_is_structural_coverage_design_member"
        ),
        "compiler_seed_policy": structural_execution["compiler_seed_policy"],
        "compiler_seed_policy_sha256": canonical_sha256(
            structural_execution["compiler_seed_policy"]
        ),
        "terminal_refinement_sha256": canonical_sha256(
            structural_execution["terminal_refinement"]
        ),
        "terminal_refinement_policy_sha256": structural_execution[
            "terminal_refinement"
        ]["policy_sha256"],
        "terminal_coordinate_surface_sha256": structural_execution[
            "terminal_refinement"
        ]["coordinate_surface_sha256"],
        "structural_design_execution": structural_execution,
        "matching_trial_summaries": [
            {
                key: trial.get(key)
                for key in (
                    "random_seed",
                    "search_algorithm",
                    "num_configs_tested",
                    "num_compile_failures",
                    "num_worker_failures",
                    "num_isolated_rebenchmark_timeouts",
                    "num_accuracy_failures",
                    "num_successful_candidate_measurements",
                    "num_unique_sources",
                    "num_source_deduplications",
                    "num_generations",
                    "autotune_time",
                    "best_perf_ms",
                    "selected_source_was_measured",
                )
            }
            for trial in matching_trials
        ],
        "strict_full_autotune_validated": True,
    }


def validate_artifact_set(artifact_root: Path) -> dict[Case, dict[str, Any]]:
    checkout = validate_checkout()
    paths = discover_result_paths(artifact_root)
    validated = {
        case: validate_strict_result(path, case, checkout)
        for case, path in paths.items()
    }
    input_seeds = {item["input_seed"] for item in validated.values()}
    require(
        len(input_seeds) == 1,
        f"strict search results mix input seeds: {sorted(input_seeds)}",
    )
    versions = {item["search_version"] for item in validated.values()}
    require(len(versions) == 1, f"strict search results mix versions: {versions}")
    return validated


def nvidia_smi_record(identifier: str) -> dict[str, Any]:
    fields = (
        "index,name,uuid,pci.bus_id,power.limit,driver_version,"
        "temperature.gpu,clocks.sm"
    )
    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            identifier,
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rows = [part.strip() for part in output.split(",")]
    if len(rows) != 8:
        raise RuntimeError(f"unexpected nvidia-smi output: {output!r}")
    return {
        "physical_index": int(rows[0]),
        "name": rows[1],
        "uuid": rows[2],
        "pci_bus_id": rows[3],
        "power_limit_w": float(rows[4]),
        "driver_version": rows[5],
        "temperature_c": float(rows[6]),
        "sm_clock_mhz": float(rows[7]),
    }


def active_compute_pids(gpu_uuid: str) -> list[int]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted(
        int(line.strip()) for line in output.splitlines() if line.strip().isdigit()
    )


def validate_gpu_pin(
    case: Case, expected_uuid: str, *, allowed_compute_pids: set[int]
) -> dict[str, Any]:
    check_equal(os.environ.get("CUDA_VISIBLE_DEVICES"), expected_uuid, "GPU visibility")
    gpu = nvidia_smi_record(str(case.physical_gpu))
    check_equal(gpu["physical_index"], case.physical_gpu, "physical GPU index")
    check_equal(gpu["uuid"], expected_uuid, "physical GPU UUID")
    check_equal(gpu["name"], "NVIDIA B200", "GPU model")
    if abs(gpu["power_limit_w"] - EXPECTED_POWER_CAP_W) > 0.5:
        raise RuntimeError(
            f"GPU {case.physical_gpu} power cap is {gpu['power_limit_w']} W, "
            f"expected {EXPECTED_POWER_CAP_W} W"
        )
    pids = active_compute_pids(expected_uuid)
    unexpected_pids = sorted(set(pids) - allowed_compute_pids)
    if unexpected_pids:
        raise RuntimeError(
            f"GPU {case.physical_gpu} has competing compute PIDs: {unexpected_pids}"
        )
    gpu["active_compute_pids"] = pids
    gpu["allowed_compute_pids"] = sorted(allowed_compute_pids)
    return gpu


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


@contextlib.contextmanager
def scrubbed_argv() -> Iterator[None]:
    saved = sys.argv
    sys.argv = sys.argv[:1]
    try:
        yield
    finally:
        sys.argv = saved


def validate_bound_flash_normalization(
    bound: _BoundWithConfigSpec, provenance: dict[str, Any]
) -> dict[str, Any]:
    """Replay strict search evidence through the bound case's ConfigGeneration."""
    from benchmarks.cute.compare_attention_backends import _canonical_flash_projection
    from benchmarks.cute.compare_attention_backends import _flash_normalization_context
    from benchmarks.cute.compare_attention_backends import (
        _validate_required_full_autotune_trials,
    )

    from helion.autotuner.config_generation import ConfigGeneration
    from helion.runtime.config import Config

    context = _flash_normalization_context(bound.config_spec)
    check_equal(
        provenance.get("flash_normalization_context"),
        context,
        "bound flash normalization context",
    )
    check_equal(
        provenance.get("flash_normalization_context_sha256"),
        canonical_sha256(context),
        "bound flash normalization context digest",
    )
    generation = ConfigGeneration(bound.config_spec)
    phase = provenance["trials"][0]["search_phase_metrics"]
    live_full_autotune_validated = False
    if provenance.get("require_full_autotune") is True:
        trial = provenance["trials"][0]
        _validate_required_full_autotune_trials(
            provenance,
            provenance["trials"],
            config_spec=bound.config_spec,
            expected_input_shapes=trial["input_shapes"],
            expected_dtypes=trial["dtypes"],
            expected_hardware=trial["hardware"],
            config_generation=generation,
        )
        live_full_autotune_validated = True
    transfer_count = 0
    for result in phase["compound_transfers"]:
        for transfer in result["transfers"]:
            expected = _canonical_flash_projection(
                generation,
                transfer["source_config"],
                transfer["projection_overrides"],
            )
            check_equal(
                transfer["projected_config"],
                expected,
                "bound canonical compound projection",
            )
            check_equal(
                canonical_sha256(expected)[:16],
                transfer["transferred_config_id"],
                "bound canonical compound projection ID",
            )
            transfer_count += 1
    terminal = phase.get("terminal_coordinate_refinement")
    if not isinstance(terminal, dict):
        return {
            "normalization_context_sha256": canonical_sha256(context),
            "canonical_compound_transfer_count": transfer_count,
            "live_full_autotune_validated": live_full_autotune_validated,
        }
    terminal_manifest = terminal["config_manifest"]
    initial_config = terminal_manifest[terminal["initial_incumbent_config_id"]][
        "config"
    ]
    from helion._compiler.cute.cute_flash import FLASH_EXP2_PACKET_KEY
    from helion._compiler.cute.cute_flash import FLASH_PIPELINE_FAMILY_KEY
    from helion._compiler.cute.cute_flash import FLASH_SOFTMAX_DISC_KEY
    from helion._compiler.cute.cute_flash import flash_structural_leaf_from_config

    initial_leaf = flash_structural_leaf_from_config(initial_config)
    require(initial_leaf is not None, "bound terminal incumbent has no leaf")
    overrides = {
        FLASH_PIPELINE_FAMILY_KEY: initial_leaf.pipeline_family,
        FLASH_SOFTMAX_DISC_KEY: initial_leaf.softmax_disc,
    }
    if initial_leaf.compound_exp2_packet is not None:
        overrides[FLASH_EXP2_PACKET_KEY] = initial_leaf.compound_exp2_packet
    leaf_generation = bound.config_spec.create_config_generation(overrides=overrides)
    terminal_projection_count = 0
    for round_value in terminal["rounds"]:
        parent_ids = round_value["parent_config_ids"]
        parent_configs = {
            Config.from_dict(terminal_manifest[config_id]["config"])
            for config_id in parent_ids
        }
        round_seen_configs: set[Config] = set()
        expected_parent_projections = []
        for parent_id in parent_ids:
            parent = Config.from_dict(terminal_manifest[parent_id]["config"])
            requests = []
            for projection in generation.canonicalize_coordinate_projections(
                leaf_generation.coordinate_neighbor_projections(
                    leaf_generation.flatten(parent), radius=terminal["radius"]
                ),
                base_config=parent,
            ):
                terminal_projection_count += 1
                outcome = projection.outcome
                config = projection.config
                if (
                    outcome == "candidate"
                    and config is not None
                    and flash_structural_leaf_from_config(config.config) != initial_leaf
                ):
                    outcome = "different_leaf"
                elif outcome == "candidate" and config in parent_configs:
                    outcome = "beam_alias"
                elif outcome == "candidate" and config in round_seen_configs:
                    outcome = "round_candidate_alias"
                if outcome == "candidate" and config is not None:
                    round_seen_configs.add(config)
                requests.append(
                    {
                        "flat_index": projection.flat_index,
                        "key": projection.key,
                        "sequence_index": projection.sequence_index,
                        "from_value": projection.from_value,
                        "to_value": projection.to_value,
                        "outcome": outcome,
                        "config_id": (
                            canonical_sha256(config.config)[:16]
                            if config is not None
                            else None
                        ),
                    }
                )
            expected_parent_projections.append(
                {
                    "parent_config_id": parent_id,
                    "coordinate_requests": requests,
                }
            )
        check_equal(
            round_value["parent_projections"],
            expected_parent_projections,
            "bound terminal coordinate projections",
        )
    check_equal(
        terminal["projection_attempt_count"],
        terminal_projection_count,
        "bound terminal projection count",
    )
    return {
        "normalization_context_sha256": canonical_sha256(context),
        "canonical_compound_transfer_count": transfer_count,
        "terminal_refinement_policy_sha256": canonical_sha256(
            provenance["flash_terminal_coordinate_refinement_policy"]
        ),
        "terminal_coordinate_surface_sha256": canonical_sha256(
            provenance["flash_terminal_coordinate_surface_catalog"]
        ),
        "terminal_refinement_transcript_sha256": canonical_sha256(terminal),
        "terminal_projection_request_count": terminal_projection_count,
        "live_full_autotune_validated": live_full_autotune_validated,
    }


def build_helion_callable(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    provenance: dict[str, Any],
    case: Case,
    generated_source_output: Path,
) -> tuple[Callable[[], torch.Tensor], dict[str, object]]:
    from examples.attention import attention_output
    from examples.attention import causal_attention_output

    import helion

    kernel = causal_attention_output if case.causal else attention_output
    config = helion.Config.from_dict(provenance["selected_config"])
    bound = kernel.bind(inputs)
    normalization_validation = validate_bound_flash_normalization(bound, provenance)
    generated_source = bound.to_triton_code(config, emit_repro_caller=False)
    generated_sha256 = hashlib.sha256(generated_source.encode()).hexdigest()
    check_equal(
        generated_sha256,
        provenance["selected_source_sha256"],
        "regenerated selected source SHA256",
    )
    generated_source_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = generated_source_output.with_suffix(
        generated_source_output.suffix + ".tmp"
    )
    temporary.write_text(generated_source)
    temporary.replace(generated_source_output)
    compiled = bound.compile_config(config)
    compiled_source_sha256 = bound.env.backend.generated_source_hash(compiled)
    check_equal(
        compiled_source_sha256,
        provenance["selected_source_sha256"],
        "compiled selected source SHA256",
    )

    def run() -> torch.Tensor:
        return compiled(*inputs)

    return run, {
        "regenerated_source_sha256": generated_sha256,
        "compiled_source_sha256": compiled_source_sha256,
        "expected_source_sha256": provenance["selected_source_sha256"],
        "source_hash_matches_search": True,
        "generated_source_bytes": len(generated_source.encode()),
        "generated_source_path": str(generated_source_output.resolve()),
        "compiled_from_current_examples_attention": True,
        **normalization_validation,
    }


def attention_flops(case: Case) -> float:
    flops = 4.0 * 2 * 32 * case.seq_len * case.seq_len * 64
    return flops * (0.5 if case.causal else 1.0)


def tensor_error_summary(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    chunk_rows: int = 2048,
    atol: float = 5e-2,
    rtol: float = 2e-2,
) -> dict[str, Any]:
    check_equal(tuple(actual.shape), tuple(expected.shape), "accuracy output shape")
    check_equal(actual.dtype, expected.dtype, "accuracy output dtype")
    check_equal(actual.device, expected.device, "accuracy output device")
    count = actual.numel()
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    expected_sum_sq = 0.0
    close_count = 0
    actual_nonfinite = 0
    expected_nonfinite = 0
    for start in range(0, actual.shape[-2], chunk_rows):
        stop = min(start + chunk_rows, actual.shape[-2])
        actual_chunk = actual[..., start:stop, :].float()
        expected_chunk = expected[..., start:stop, :].float()
        diff = (actual_chunk - expected_chunk).abs()
        max_abs = max(max_abs, float(diff.max()))
        sum_abs += float(diff.sum(dtype=torch.float64))
        sum_sq += float((diff * diff).sum(dtype=torch.float64))
        expected_sum_sq += float(
            (expected_chunk * expected_chunk).sum(dtype=torch.float64)
        )
        close_count += int((diff <= atol + rtol * expected_chunk.abs()).sum().item())
        actual_nonfinite += int((~actual_chunk.isfinite()).sum().item())
        expected_nonfinite += int((~expected_chunk.isfinite()).sum().item())
    rmse = math.sqrt(sum_sq / count)
    expected_rms = math.sqrt(expected_sum_sq / count)
    nrmse = rmse / expected_rms if expected_rms else (0.0 if rmse == 0.0 else math.inf)
    mismatch_count = count - close_count
    return {
        "count": count,
        "close_count": close_count,
        "close_fraction": close_count / count,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": mismatch_count / count,
        "max_abs": max_abs,
        "mean_abs": sum_abs / count,
        "rmse": rmse,
        "expected_rms": expected_rms,
        "nrmse": nrmse,
        "nrmse_normalization": "rms(cudnn_sdpa_output)",
        "actual_nonfinite": actual_nonfinite,
        "expected_nonfinite": expected_nonfinite,
        "atol": atol,
        "rtol": rtol,
        "passed": (
            close_count == count and actual_nonfinite == 0 and expected_nonfinite == 0
        ),
    }


def peaky_tensor_error_summary(
    actual: torch.Tensor, expected: torch.Tensor, *, chunk_rows: int = 2048
) -> dict[str, Any]:
    thresholds = PEAKY_STRESS_THRESHOLDS
    summary = tensor_error_summary(
        actual,
        expected,
        chunk_rows=chunk_rows,
        atol=thresholds["atol"],
        rtol=thresholds["rtol"],
    )
    finite_outputs = (
        summary["actual_nonfinite"] == 0 and summary["expected_nonfinite"] == 0
    )
    summary.update(
        {
            "finite_outputs": finite_outputs,
            "thresholds": dict(thresholds),
            "passed": (
                finite_outputs
                and summary["max_abs"] < thresholds["max_abs_exclusive"]
                and summary["nrmse"] < thresholds["nrmse_exclusive"]
                and summary["mismatch_fraction"]
                < thresholds["mismatch_fraction_exclusive"]
            ),
        }
    )
    return summary


def exact_repeat_summary(
    actual: torch.Tensor, baseline: torch.Tensor, *, chunk_rows: int = 2048
) -> dict[str, Any]:
    check_equal(tuple(actual.shape), tuple(baseline.shape), "repeat output shape")
    check_equal(actual.dtype, baseline.dtype, "repeat output dtype")
    check_equal(actual.device, baseline.device, "repeat output device")
    different = 0
    count = actual.numel()
    for start in range(0, actual.shape[-2], chunk_rows):
        stop = min(start + chunk_rows, actual.shape[-2])
        different += int(
            (actual[..., start:stop, :] != baseline[..., start:stop, :]).sum().item()
        )
    return {
        "count": count,
        "different": different,
        "different_fraction": different / count,
        "passed": different == 0,
    }


def thermal_warmup(seconds: float) -> None:
    if seconds <= 0:
        return
    left = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    right = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for _ in range(20):
            left = left @ right
        torch.cuda.synchronize()


def timed_single_call(fn: Callable[[], torch.Tensor]) -> dict[str, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start_ns = time.perf_counter_ns()
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    wall_end_ns = time.perf_counter_ns()
    event_ms = float(start.elapsed_time(end))
    wall_ms = (wall_end_ns - wall_start_ns) / 1e6
    del result
    return {"event_ms": event_ms, "wall_ms": wall_ms}


def bootstrap_ci(values: list[float], *, samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    means = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    means.sort()
    return [means[int(samples * 0.025)], means[int(samples * 0.975)]]


def summarize_pairs(
    raw_pairs: list[dict[str, Any]], *, flops: float, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for timer in ("event", "wall"):
        helion_ms = [pair["times"]["helion"][f"{timer}_ms"] for pair in raw_pairs]
        sdpa_ms = [pair["times"]["sdpa"][f"{timer}_ms"] for pair in raw_pairs]
        speedups = [
            100.0 * (sdpa / helion - 1.0)
            for helion, sdpa in zip(helion_ms, sdpa_ms, strict=True)
        ]
        helion_median = statistics.median(helion_ms)
        sdpa_median = statistics.median(sdpa_ms)
        result[timer] = {
            "helion": {
                "median_ms": helion_median,
                "mean_ms": statistics.fmean(helion_ms),
                "median_tflops": flops / (helion_median * 1e9),
                "raw_ms": helion_ms,
            },
            "sdpa": {
                "median_ms": sdpa_median,
                "mean_ms": statistics.fmean(sdpa_ms),
                "median_tflops": flops / (sdpa_median * 1e9),
                "raw_ms": sdpa_ms,
            },
            "paired_helion_speedup_pct": {
                "mean": statistics.fmean(speedups),
                "median": statistics.median(speedups),
                "geomean": 100.0
                * (
                    statistics.geometric_mean(1.0 + value / 100.0 for value in speedups)
                    - 1.0
                ),
                "bootstrap_95_ci_of_mean": bootstrap_ci(
                    speedups, samples=bootstrap_samples, seed=seed
                ),
                "raw": speedups,
            },
        }
    return result


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_gpu(
    args: argparse.Namespace, case: Case, provenance: dict[str, Any]
) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    import helion

    gpu_start = validate_gpu_pin(
        case, args.expected_gpu_uuid, allowed_compute_pids=set()
    )
    check_equal(torch.cuda.device_count(), 1, "visible CUDA device count")
    torch.cuda.set_device(0)
    check_equal(torch.cuda.current_device(), 0, "logical CUDA device")
    check_equal(torch.cuda.get_device_name(0), "NVIDIA B200", "CUDA device name")
    check_equal(
        Path(helion.__file__).resolve().parents[1], REPO_ROOT, "Helion import root"
    )
    cute_version = package_version("nvidia-cutlass-dsl")
    check_equal(cute_version, provenance["cute_version"], "CuTe version")

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    shape = (2, 32, case.seq_len, 64)
    inputs = tuple(
        torch.randn(shape, device="cuda", dtype=torch.float16, generator=generator)
        for _ in range(3)
    )
    search_payload = json.loads(Path(provenance["search_result_path"]).read_text())
    recorded_provenance = search_payload.get("helion_overrides", {}).get(
        "autotune_provenance"
    )
    require(
        isinstance(recorded_provenance, dict),
        "search result lost recorded autotune provenance before live rebind",
    )

    with scrubbed_argv():
        run_helion, regenerated = build_helion_callable(
            inputs, recorded_provenance, case, args.generated_source_output
        )
        check_equal(
            regenerated["terminal_refinement_transcript_sha256"],
            provenance["structural_design_execution"]["terminal_refinement"][
                "transcript_sha256"
            ],
            "offline/live terminal refinement transcript",
        )
        check_equal(
            regenerated["terminal_refinement_policy_sha256"],
            provenance["terminal_refinement_policy_sha256"],
            "offline/live terminal refinement policy",
        )
        check_equal(
            regenerated["terminal_coordinate_surface_sha256"],
            provenance["terminal_coordinate_surface_sha256"],
            "offline/live terminal coordinate surface",
        )
        check_equal(
            regenerated["terminal_projection_request_count"],
            provenance["structural_design_execution"]["terminal_refinement"][
                "projection_attempt_count"
            ],
            "offline/live terminal projection request count",
        )

        def run_sdpa() -> torch.Tensor:
            q, k, v = inputs
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=case.causal
            )

        with torch.nn.attention.sdpa_kernel(
            [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
        ):
            expected = run_sdpa()
            actual = run_helion()
            torch.cuda.synchronize()
            numerics = tensor_error_summary(actual, expected)
            if not numerics["passed"]:
                raise RuntimeError(f"accuracy check failed: {numerics}")
            actual = actual.clone()

            race_checks = []
            for repeat_index in range(1, args.race_runs):
                repeated = run_helion()
                torch.cuda.synchronize()
                repeat = exact_repeat_summary(repeated, actual)
                repeat["repeat_index"] = repeat_index
                race_checks.append(repeat)
                del repeated
                if not repeat["passed"]:
                    raise RuntimeError(f"repeatability check failed: {repeat}")
            del expected
            del actual
            torch.cuda.empty_cache()

            thermal_warmup(args.thermal_warmup_seconds)
            for warmup_index in range(args.warmup_calls):
                warmup_order = [run_helion, run_sdpa]
                if warmup_index % 2:
                    warmup_order.reverse()
                for fn in warmup_order:
                    warmup_result = fn()
                    del warmup_result
            torch.cuda.synchronize()

            orders = [["helion", "sdpa"] for _ in range(args.pairs // 2)]
            orders += [["sdpa", "helion"] for _ in range(args.pairs // 2)]
            random.Random(args.seed ^ 0x5A17).shuffle(orders)
            implementations = {"helion": run_helion, "sdpa": run_sdpa}
            raw_pairs = []
            for pair_index, order in enumerate(orders):
                times = {
                    name: timed_single_call(implementations[name]) for name in order
                }
                raw_pairs.append(
                    {"pair_index": pair_index, "order": order, "times": times}
                )

            inputs[0].mul_(2.0)
            inputs[1].mul_(2.0)
            peaky_expected = run_sdpa()
            peaky_actual = run_helion()
            torch.cuda.synchronize()
            peaky_numerics = peaky_tensor_error_summary(peaky_actual, peaky_expected)
            peaky_actual = peaky_actual.clone()
            del peaky_expected

            peaky_repeated = run_helion()
            torch.cuda.synchronize()
            peaky_repeatability = exact_repeat_summary(peaky_repeated, peaky_actual)
            del peaky_repeated
            del peaky_actual
            if not peaky_numerics["passed"]:
                raise RuntimeError(
                    f"post-timing peaky-logit accuracy check failed: {peaky_numerics}"
                )
            if not peaky_repeatability["passed"]:
                raise RuntimeError(
                    "post-timing peaky-logit repeatability check failed: "
                    f"{peaky_repeatability}"
                )

    gpu_end = validate_gpu_pin(
        case, args.expected_gpu_uuid, allowed_compute_pids={os.getpid()}
    )
    flops = attention_flops(case)
    return {
        "schema_version": 4,
        "status": "PASS",
        "campaign_seed": args.campaign_seed,
        "protocol": {
            "name": "balanced randomized paired raw single-call attention",
            "pairs": args.pairs,
            "order_balance": {
                "helion_first": args.pairs // 2,
                "sdpa_first": args.pairs // 2,
            },
            "timers": ["cuda_event", "host_wall"],
            "warmup_calls_per_implementation": args.warmup_calls,
            "thermal_warmup_seconds": args.thermal_warmup_seconds,
            "correctness_before_timing": True,
            "forced_sdpa_backend": "CUDNN_ATTENTION",
            "race_repeat_runs": args.race_runs,
            "bootstrap_samples": args.bootstrap_samples,
            "input_seed": args.seed,
        },
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
            "flops": flops,
            "formula": "4*z*h*seq_len^2*head_dim; multiplied by 0.5 for causal",
        },
        "environment": {
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_start": gpu_start,
            "gpu_end": gpu_end,
            "logical_cuda_device": 0,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cute_version": cute_version,
            "cute_dsl_arch": os.environ.get("CUTE_DSL_ARCH"),
            "helion_module": helion.__file__,
        },
        "provenance": provenance,
        "regenerated_kernel": regenerated,
        "correctness": {
            "helion_vs_cudnn_sdpa": numerics,
            "helion_exact_repeatability": race_checks,
            "post_timing_peaky_logits": {
                "performed_after_timing": True,
                "q_scale_in_place": 2.0,
                "k_scale_in_place": 2.0,
                "v_mutated": False,
                "helion_vs_cudnn_sdpa": peaky_numerics,
                "helion_exact_repeatability": peaky_repeatability,
            },
        },
        "raw_pairs": raw_pairs,
        "summary": summarize_pairs(
            raw_pairs,
            flops=flops,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed ^ 0xB007,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("dense", "causal"), required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generated-source-output", type=Path, required=True)
    parser.add_argument("--expected-worker-sha256", required=True)
    parser.add_argument("--campaign-seed", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--warmup-calls", type=int, default=3)
    parser.add_argument("--thermal-warmup-seconds", type=float, default=10.0)
    parser.add_argument("--race-runs", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_sha256 = validate_worker_harness_sha256(args.expected_worker_sha256)
    if args.pairs <= 0 or args.pairs % 2:
        raise SystemExit("--pairs must be a positive even number")
    if args.warmup_calls < 0:
        raise SystemExit("--warmup-calls must be nonnegative")
    if args.thermal_warmup_seconds < 0:
        raise SystemExit("--thermal-warmup-seconds must be nonnegative")
    if args.race_runs < 2:
        raise SystemExit("--race-runs must be at least 2")
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    case = CASES.get((args.variant, args.seq_len))
    if case is None:
        raise SystemExit(f"unsupported case: {(args.variant, args.seq_len)!r}")
    check_equal(args.physical_gpu, case.physical_gpu, "requested physical GPU")
    provenance = validate_artifact_set(args.artifact_root)[case]
    if args.validate_only:
        payload = {
            "schema_version": 4,
            "status": "VALIDATED_STATIC_ONLY",
            "case": case.name,
            "provenance": provenance,
        }
    else:
        payload = run_gpu(args, case, provenance)
    worker_sha256 = validate_worker_harness_sha256(args.expected_worker_sha256)
    payload["harness_sha256"] = {"paired_worker.py": worker_sha256}
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
