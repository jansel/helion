from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING
from typing import Any

import build_heldout_manifest
import build_strict_manifest
import combine_results
import run_all8
import validate_generalization_campaign

if TYPE_CHECKING:
    from collections.abc import Iterable

TARGET_IMPLS = frozenset({"helion-cute", "sdpa"})
CUTE_BACKED_IMPLS = frozenset(
    {
        "helion-cute",
        "flexattention-cute",
        "fa4",
        "kernelagent-closed-1x",
        "kernelagent-closed-2x",
    }
)
EXPECTED_SHAPES = frozenset(
    {
        (2, 32, seq_len, 64, "float16", causal, 0)
        for causal, seq_lens in (
            (0, (32768, 65536, 131072, 262144)),
            (1, (65536, 131072, 262144, 524288)),
        )
        for seq_len in seq_lens
    }
)
TIMING_FIELDS = frozenset(
    {
        "best_ms",
        "median_ms",
        "mom_median_ms",
        "mean_ms",
        "std_ms",
        "runs_ms",
        "best_tflops",
        "median_tflops",
        "mom_median_tflops",
    }
)
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RENDERER = REPO_ROOT / "benchmarks" / "cute" / "compare_attention_backends.py"
FLOP_MODEL_NAME = "softmax_attention_forward"
FLOP_MODEL_FORMULA = "4*z*h*seq_len^2*head_dim; multiplied by 0.5 for causal"
PEAKY_STRESS_THRESHOLDS = {
    "atol": 0.002,
    "rtol": 0.01,
    "max_abs_exclusive": 0.01,
    "nrmse_exclusive": 0.002,
    "mismatch_fraction_exclusive": 1e-5,
}
EVENT_WALL_RELATIVE_BIAS_BOUNDS_PCT = (-0.5, 0.5)
EVENT_WALL_IMPLEMENTATION_BOUNDS_PCT = (-0.5, 2.0)
EVENT_WALL_MIN_INLIER_FRACTION = 0.9
EVENT_WALL_DECISION_THRESHOLD_PCT = 0.5
STRATIFIED_BOOTSTRAP_INTERPRETATION = (
    "conditional on the two observed campaign strata; paired calls are resampled "
    "within each fixed campaign"
)
STRICT_MANIFEST_FIELDS = build_strict_manifest.MANIFEST_FIELDS
SOURCE_LEDGER_FIELDS = build_strict_manifest.LEDGER_FIELDS
HELDOUT_MANIFEST_FIELDS = build_heldout_manifest.HELDOUT_MANIFEST_FIELDS
EvidenceIdentity = tuple[str, int, int, int]


def require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_identity(path: Path) -> EvidenceIdentity:
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    before_metadata = (before.st_mode, before.st_mtime_ns, before.st_size)
    after_metadata = (after.st_mode, after.st_mtime_ns, after.st_size)
    check_equal(
        after_metadata, before_metadata, f"evidence changed while hashing {path}"
    )
    return (digest, *after_metadata)


def publication_evidence_paths(
    args: argparse.Namespace, *, include_renderer: bool
) -> set[Path]:
    paths = {
        Path(path).resolve()
        for path in (
            args.paired_raw,
            args.run_manifest,
            args.static_validation,
            args.strict_manifest,
            args.heldout_manifest,
            args.generalization_manifest,
        )
    }
    paths.update(path.resolve() for path in args.baseline_payload_dir.glob("*.json"))
    evidence_roots = (
        args.strict_artifact_root,
        args.heldout_artifact_root,
        args.generalization_artifact_root,
        args.run_manifest.resolve().parent,
    )
    missing_roots = [root for root in evidence_roots if not root.is_dir()]
    require(
        not missing_roots, f"publication evidence roots are missing: {missing_roots}"
    )
    for root in evidence_roots:
        paths.update(path.resolve() for path in root.rglob("*") if path.is_file())
    setup_dir = Path(__file__).resolve().parent
    paths.update(
        (setup_dir / name).resolve()
        for name in (
            "build_strict_manifest.py",
            "build_heldout_manifest.py",
            "run_all8.py",
            "paired_worker.py",
            "combine_results.py",
            "publish_results.py",
            "remeasure_generalization_winners.py",
            "run_strict_all8.sh",
            "run_strict_heldout.sh",
            "validate_generalization_campaign.py",
        )
    )
    if include_renderer:
        paths.add(Path(getattr(args, "renderer", DEFAULT_RENDERER)).resolve())
    missing = [path for path in paths if not path.is_file()]
    require(not missing, f"publication evidence files are missing: {missing}")
    return paths


def snapshot_publication_evidence(
    args: argparse.Namespace, *, include_renderer: bool
) -> dict[Path, EvidenceIdentity]:
    return {
        path: evidence_identity(path)
        for path in publication_evidence_paths(args, include_renderer=include_renderer)
    }


def validate_publication_evidence_unchanged(
    args: argparse.Namespace,
    snapshot: dict[Path, EvidenceIdentity],
    *,
    include_renderer: bool,
) -> None:
    current_paths = publication_evidence_paths(args, include_renderer=include_renderer)
    check_equal(current_paths, set(snapshot), "publication evidence file set")
    changed = [
        path
        for path, identity in snapshot.items()
        if evidence_identity(path) != identity
    ]
    require(not changed, f"publication evidence changed during validation: {changed}")


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to load {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def shape_key(shape: object, *, context: str) -> tuple[object, ...]:
    require(isinstance(shape, dict), f"{context}: missing shape object")
    required = ("z", "h", "seq_len", "head_dim", "dtype", "causal")
    missing = [field for field in required if field not in shape]
    require(not missing, f"{context}: shape is missing fields {missing}")
    return (
        shape["z"],
        shape["h"],
        shape["seq_len"],
        shape["head_dim"],
        shape["dtype"],
        int(shape["causal"]),
        int(shape.get("biased", 0)),
    )


def expected_attention_flops(shape: dict[str, Any]) -> float:
    flops = (
        4.0
        * shape["z"]
        * shape["h"]
        * shape["seq_len"]
        * shape["seq_len"]
        * shape["head_dim"]
    )
    return flops * (0.5 if shape["causal"] else 1.0)


def index_baseline_payloads(
    payload_dir: Path,
) -> dict[tuple[object, ...], tuple[Path, dict[str, Any]]]:
    paths = sorted(payload_dir.glob("*.json"))
    require(
        len(paths) == 8,
        f"expected 8 baseline payloads in {payload_dir}, got {len(paths)}",
    )
    indexed: dict[tuple[object, ...], tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        payload = load_object(path)
        key = shape_key(payload.get("shape"), context=str(path))
        require(key not in indexed, f"duplicate baseline shape {key}")
        results = payload.get("results")
        require(isinstance(results, list), f"{path}: missing results list")
        target_counts = {
            impl: sum(
                isinstance(result, dict) and result.get("impl") == impl
                for result in results
            )
            for impl in TARGET_IMPLS
        }
        check_equal(target_counts, {"helion-cute": 1, "sdpa": 1}, f"{path}: targets")
        indexed[key] = (path, payload)
    check_equal(frozenset(indexed), EXPECTED_SHAPES, "baseline shape set")
    return indexed


def index_strict_results(
    artifact_root: Path,
) -> dict[tuple[object, ...], tuple[Path, dict[str, Any]]]:
    indexed: dict[tuple[object, ...], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(artifact_root.rglob("result.json")):
        payload = load_object(path)
        if payload.get("impl") != "helion-cute":
            continue
        key = shape_key(payload.get("shape"), context=str(path))
        require(key not in indexed, f"duplicate strict result shape {key}")
        indexed[key] = (path, payload)
    check_equal(frozenset(indexed), EXPECTED_SHAPES, "strict result shape set")
    return indexed


def validate_regenerated_strict_manifest(
    manifest_path: Path, artifact_root: Path
) -> None:
    try:
        supplied = manifest_path.read_text()
    except OSError as exc:
        raise RuntimeError(f"unable to load {manifest_path}: {exc}") from exc
    regenerated = build_strict_manifest.build_manifest(artifact_root)
    if supplied != regenerated:
        raise RuntimeError(
            f"{manifest_path}: regenerated strict manifest differs: "
            f"supplied_sha256={hashlib.sha256(supplied.encode()).hexdigest()}, "
            f"regenerated_sha256={hashlib.sha256(regenerated.encode()).hexdigest()}"
        )


def validate_regenerated_heldout_manifest(
    manifest_path: Path,
    artifact_root: Path,
    all8_artifact_root: Path,
    strict_manifest_path: Path,
    strict_manifest: dict[tuple[object, ...], dict[str, str]],
) -> None:
    try:
        supplied = manifest_path.read_text()
    except OSError as exc:
        raise RuntimeError(f"unable to load {manifest_path}: {exc}") from exc
    regenerated = build_heldout_manifest.build_manifest(
        artifact_root, all8_artifact_root
    )
    if supplied != regenerated:
        raise RuntimeError(
            f"{manifest_path}: regenerated heldout manifest differs: "
            f"supplied_sha256={hashlib.sha256(supplied.encode()).hexdigest()}, "
            f"regenerated_sha256={hashlib.sha256(regenerated.encode()).hexdigest()}"
        )

    try:
        reader = csv.DictReader(supplied.splitlines())
        check_equal(
            tuple(reader.fieldnames or ()),
            HELDOUT_MANIFEST_FIELDS,
            f"{manifest_path}: header",
        )
        rows = list(reader)
    except csv.Error as exc:
        raise RuntimeError(f"unable to load {manifest_path}: {exc}") from exc
    check_equal(
        len(rows),
        len(build_heldout_manifest.CASES),
        f"{manifest_path}: row count",
    )
    expected_cases = {
        f"{variant}_{seq_len}_seed_{tuner_seed}"
        for variant, seq_len, _physical_gpu, tuner_seed in build_heldout_manifest.CASES
    }
    check_equal(
        {row["case"] for row in rows},
        expected_cases,
        f"{manifest_path}: heldout cases",
    )
    strict_manifest_sha256 = file_sha256(strict_manifest_path)
    check_equal(
        {row["all8_reference_manifest_sha256"] for row in rows},
        {strict_manifest_sha256},
        f"{manifest_path}: all8 strict manifest digest",
    )
    check_equal(
        {row["version"] for row in rows},
        {row["version"] for row in strict_manifest.values()},
        f"{manifest_path}: all8 strict version",
    )


def validate_regenerated_generalization_manifest(
    manifest_path: Path, artifact_root: Path
) -> None:
    try:
        supplied = manifest_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"unable to load {manifest_path}: {exc}") from exc
    validation = validate_generalization_campaign.validate_campaign(
        artifact_root, require_remeasurement=True
    )
    require(
        len(validation.cases) == validate_generalization_campaign.EXPECTED_CASE_COUNT
        and len(validation.run_specs)
        == validate_generalization_campaign.EXPECTED_BROAD_RUN_COUNT,
        f"{artifact_root}: publication requires the broad 65-search "
        "generalization campaign",
    )
    regenerated = validate_generalization_campaign.render_manifest(validation).encode()
    if supplied != regenerated:
        raise RuntimeError(
            f"{manifest_path}: regenerated generalization manifest differs: "
            f"supplied_sha256={hashlib.sha256(supplied).hexdigest()}, "
            f"regenerated_sha256={hashlib.sha256(regenerated).hexdigest()}"
        )


def manifest_int(value: str, *, context: str) -> int:
    require(re.fullmatch(r"-?[0-9]+", value) is not None, f"{context}: bad integer")
    return int(value)


def manifest_path(artifact_root: Path, value: str, *, context: str) -> Path:
    relative = Path(value)
    require(
        value != "" and not relative.is_absolute() and ".." not in relative.parts,
        f"{context}: expected a safe artifact-relative path",
    )
    root = artifact_root.resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"{context}: path escapes artifact root")
    return path


def validate_manifest_source_ledger(
    path: Path,
    row: dict[str, str],
    *,
    selected_config_sha256: str,
    selected_source_sha256: str,
    context: str,
) -> None:
    require(path.is_file(), f"{context}: source ledger does not exist: {path}")
    check_equal(
        file_sha256(path), row["source_ledger_sha256"], f"{context}: ledger digest"
    )
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            check_equal(
                tuple(reader.fieldnames or ()),
                SOURCE_LEDGER_FIELDS,
                f"{context}: ledger header",
            )
            ledger_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"{context}: unable to read source ledger: {exc}") from exc
    require(ledger_rows, f"{context}: source ledger is empty")
    require(
        all(
            None not in ledger_row
            and all(ledger_row.get(field) is not None for field in SOURCE_LEDGER_FIELDS)
            for ledger_row in ledger_rows
        ),
        f"{context}: malformed source ledger row",
    )
    run_ids = {ledger_row["run_id"] for ledger_row in ledger_rows}
    check_equal(len(run_ids), 1, f"{context}: source ledger run count")
    check_equal(
        next(iter(run_ids)), row["source_ledger_run_id"], f"{context}: ledger run ID"
    )
    successful = [
        ledger_row
        for ledger_row in ledger_rows
        if ledger_row["status"] == "ok"
        and ledger_row["source_hash"] == selected_source_sha256
    ]
    require(
        len(successful) == 1,
        f"{context}: selected source is not exactly one successful ledger row",
    )
    selected_config_id = selected_config_sha256[:16]
    check_equal(
        row["selected_source_ledger_config_id"],
        selected_config_id,
        f"{context}: selected config ledger ID",
    )
    linked = [
        ledger_row
        for ledger_row in ledger_rows
        if ledger_row["config_id"] == selected_config_id
        and ledger_row["source_hash"] == selected_source_sha256
        and ledger_row["status"] in {"ok", "deduplicated"}
    ]
    require(
        len(linked) == 1,
        f"{context}: selected config is not linked to its measured source",
    )
    check_equal(
        manifest_int(
            row["selected_source_ledger_generation"],
            context=f"{context}: selected source generation",
        ),
        manifest_int(linked[0]["generation"], context=f"{context}: ledger generation"),
        f"{context}: selected source generation",
    )


def index_strict_manifest(
    manifest_path_value: Path,
    artifact_root: Path,
    strict_results: dict[tuple[object, ...], tuple[Path, dict[str, Any]]],
) -> dict[tuple[object, ...], dict[str, str]]:
    manifest_path_value = manifest_path_value.resolve()
    try:
        with manifest_path_value.open(newline="") as handle:
            reader = csv.DictReader(handle)
            check_equal(
                tuple(reader.fieldnames or ()),
                STRICT_MANIFEST_FIELDS,
                f"{manifest_path_value}: header",
            )
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"unable to load {manifest_path_value}: {exc}") from exc
    check_equal(len(rows), 8, f"{manifest_path_value}: row count")

    indexed: dict[tuple[object, ...], dict[str, str]] = {}
    for row_number, row in enumerate(rows, 2):
        context = f"{manifest_path_value}:{row_number}"
        require(
            None not in row
            and all(row.get(field) is not None for field in STRICT_MANIFEST_FIELDS),
            f"{context}: malformed manifest row",
        )
        causal = manifest_int(row["causal"], context=f"{context}: causal")
        key = (
            manifest_int(row["z"], context=f"{context}: z"),
            manifest_int(row["h"], context=f"{context}: h"),
            manifest_int(row["seq_len"], context=f"{context}: seq_len"),
            manifest_int(row["head_dim"], context=f"{context}: head_dim"),
            row["dtype"],
            causal,
            0,
        )
        require(key in EXPECTED_SHAPES, f"{context}: unexpected shape {key}")
        require(key not in indexed, f"{context}: duplicate shape {key}")
        variant = "causal" if causal else "dense"
        check_equal(row["variant"], variant, f"{context}: variant")
        check_equal(row["case"], f"{variant}_{key[2]}", f"{context}: case")

        strict_path, strict = strict_results[key]
        result_path = manifest_path(
            artifact_root, row["result_path"], context=f"{context}: result_path"
        )
        check_equal(result_path, strict_path.resolve(), f"{context}: result path")
        check_equal(
            file_sha256(result_path), row["result_sha256"], f"{context}: result digest"
        )
        check_equal(row["version"], strict.get("version"), f"{context}: version")
        check_equal(row["gpu"], strict.get("gpu"), f"{context}: GPU")
        check_equal(
            row["physical_gpu"], str(strict.get("physical_gpu")), f"{context}: GPU"
        )
        check_equal(
            manifest_int(row["input_seed"], context=f"{context}: input seed"),
            strict.get("input_seed"),
            f"{context}: input seed",
        )
        check_equal(
            row["benchmark_timer"],
            strict.get("benchmark_timer"),
            f"{context}: benchmark timer",
        )

        overrides = strict.get("helion_overrides")
        require(isinstance(overrides, dict), f"{context}: missing Helion overrides")
        provenance = overrides.get("autotune_provenance")
        require(isinstance(provenance, dict), f"{context}: missing provenance")
        compiler_seed_policy = build_strict_manifest.validate_compiler_seed_policy(
            context, provenance
        )
        try:
            manifest_compiler_seed_policy = json.loads(row["compiler_seed_policy_json"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{context}: invalid compiler seed policy JSON") from exc
        check_equal(
            manifest_compiler_seed_policy,
            compiler_seed_policy,
            f"{context}: compiler seed policy",
        )
        check_equal(
            row["compiler_seed_policy_json"],
            json.dumps(compiler_seed_policy, sort_keys=True, separators=(",", ":")),
            f"{context}: canonical compiler seed policy JSON",
        )
        selected_config = provenance.get("selected_config")
        require(isinstance(selected_config, dict), f"{context}: missing config")
        try:
            manifest_config = json.loads(row["selected_config_json"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{context}: invalid selected config JSON") from exc
        check_equal(manifest_config, selected_config, f"{context}: selected config")
        selected_config_sha256 = canonical_sha256(selected_config)
        check_equal(
            row["selected_config_json"],
            json.dumps(selected_config, sort_keys=True, separators=(",", ":")),
            f"{context}: selected config canonical JSON",
        )
        check_equal(
            row["selected_config_sha256"],
            selected_config_sha256,
            f"{context}: selected config digest",
        )
        selected_source = selected_source_sha256(provenance, context=context)
        check_equal(
            row["selected_source_sha256"],
            selected_source,
            f"{context}: selected source digest",
        )
        ledger_path = manifest_path(
            artifact_root,
            row["source_ledger_path"],
            context=f"{context}: source_ledger_path",
        )
        validate_manifest_source_ledger(
            ledger_path,
            row,
            selected_config_sha256=selected_config_sha256,
            selected_source_sha256=selected_source,
            context=context,
        )
        try:
            terminal_refinement = json.loads(row["terminal_refinement_json"])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{context}: invalid terminal refinement JSON") from exc
        check_equal(
            row["terminal_refinement_json"],
            json.dumps(terminal_refinement, sort_keys=True, separators=(",", ":")),
            f"{context}: canonical terminal refinement JSON",
        )
        check_equal(
            row["terminal_refinement_sha256"],
            canonical_sha256(terminal_refinement),
            f"{context}: terminal refinement digest",
        )
        check_equal(
            row["terminal_refinement_policy_sha256"],
            terminal_refinement.get("policy_sha256"),
            f"{context}: terminal refinement policy digest",
        )
        check_equal(
            row["terminal_coordinate_surface_sha256"],
            terminal_refinement.get("coordinate_surface_sha256"),
            f"{context}: terminal coordinate surface digest",
        )
        indexed[key] = row
    check_equal(frozenset(indexed), EXPECTED_SHAPES, "strict manifest shape set")
    return indexed


def index_paired_results(
    raw: dict[str, Any],
) -> dict[tuple[object, ...], dict[str, Any]]:
    check_equal(raw.get("schema_version"), 4, "paired raw schema version")
    check_equal(raw.get("status"), "PASS", "paired raw status")
    protocol = raw.get("protocol")
    require(isinstance(protocol, dict), "paired raw is missing protocol")
    check_equal(protocol.get("campaign_count"), 2, "paired campaign count")
    seeds = protocol.get("campaign_seeds")
    require(
        isinstance(seeds, list)
        and len(seeds) == 2
        and len(set(seeds)) == 2
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds),
        "paired raw must record two distinct integer campaign seeds",
    )
    check_equal(protocol.get("dtype"), "float16", "paired dtype")
    check_equal(protocol.get("dense_physical_gpu"), 7, "paired dense GPU")
    check_equal(protocol.get("causal_physical_gpu"), 6, "paired causal GPU")
    check_equal(protocol.get("power_cap_w"), 750.0, "paired power cap")
    check_equal(protocol.get("forced_sdpa_backend"), "CUDNN_ATTENTION", "SDPA backend")
    check_equal(protocol.get("correctness_before_timing"), True, "correctness policy")
    check_equal(
        protocol.get("post_timing_peaky_logits"),
        {
            "performed_after_timing": True,
            "q_scale_in_place": 2.0,
            "k_scale_in_place": 2.0,
            "v_mutated": False,
            "thresholds": PEAKY_STRESS_THRESHOLDS,
            "exact_repeatability": True,
        },
        "peaky-logit protocol",
    )
    pairs = protocol.get("pairs_per_shape_per_campaign")
    require(
        isinstance(pairs, int)
        and not isinstance(pairs, bool)
        and pairs > 0
        and pairs % 2 == 0,
        "paired raw must use a positive even pair count",
    )
    check_equal(protocol.get("combined_pairs_per_shape"), pairs * 2, "combined pairs")
    bootstrap_samples = protocol.get("bootstrap_samples")
    require(
        isinstance(bootstrap_samples, int)
        and not isinstance(bootstrap_samples, bool)
        and bootstrap_samples > 0,
        "paired raw has invalid bootstrap sample count",
    )
    require(
        isinstance(protocol.get("bootstrap_base_seed"), int)
        and not isinstance(protocol["bootstrap_base_seed"], bool),
        "paired raw has invalid bootstrap seed",
    )
    for field in ("run_manifest_sha256", "static_validation_sha256"):
        require(
            isinstance(raw.get(field), str)
            and re.fullmatch(r"[0-9a-f]{64}", raw[field]) is not None,
            f"paired raw has invalid {field}",
        )

    results = raw.get("results")
    require(isinstance(results, list), "paired raw is missing results")
    indexed: dict[tuple[object, ...], dict[str, Any]] = {}
    for result in results:
        require(isinstance(result, dict), "paired result must be an object")
        key = shape_key(result.get("shape"), context="paired result")
        require(key not in indexed, f"duplicate paired result shape {key}")
        case = f"{'causal' if key[5] else 'dense'}_{key[2]}"
        for field in ("regenerated_kernels", "correctness", "raw_campaigns"):
            index_campaign_entries(
                result.get(field), context=f"{case}: {field.replace('_', ' ')}"
            )
        indexed[key] = result
    check_equal(frozenset(indexed), EXPECTED_SHAPES, "paired result shape set")
    return indexed


def index_campaign_entries(value: object, *, context: str) -> dict[int, dict[str, Any]]:
    require(isinstance(value, list) and len(value) == 2, f"{context}: expected 2")
    indexed: dict[int, dict[str, Any]] = {}
    for entry in value:
        require(isinstance(entry, dict), f"{context}: entry is not an object")
        campaign = entry.get("campaign")
        require(
            isinstance(campaign, int) and not isinstance(campaign, bool),
            f"{context}: invalid campaign ID",
        )
        require(campaign not in indexed, f"{context}: duplicate campaign {campaign}")
        indexed[campaign] = entry
    check_equal(set(indexed), {1, 2}, f"{context}: campaign IDs")
    return indexed


def validate_peaky_stress(value: object, *, context: str) -> None:
    require(isinstance(value, dict), f"{context}: missing peaky-logit validation")
    check_equal(
        {
            field: value.get(field)
            for field in (
                "performed_after_timing",
                "q_scale_in_place",
                "k_scale_in_place",
                "v_mutated",
            )
        },
        {
            "performed_after_timing": True,
            "q_scale_in_place": 2.0,
            "k_scale_in_place": 2.0,
            "v_mutated": False,
        },
        f"{context}: policy",
    )
    numerics = value.get("helion_vs_cudnn_sdpa")
    require(isinstance(numerics, dict), f"{context}: missing numerical metrics")
    check_equal(
        numerics.get("thresholds"),
        PEAKY_STRESS_THRESHOLDS,
        f"{context}: thresholds",
    )
    check_equal(numerics.get("atol"), 0.002, f"{context}: atol")
    check_equal(numerics.get("rtol"), 0.01, f"{context}: rtol")
    check_equal(
        numerics.get("nrmse_normalization"),
        "rms(cudnn_sdpa_output)",
        f"{context}: NRMSE normalization",
    )
    check_equal(numerics.get("finite_outputs"), True, f"{context}: finite outputs")
    check_equal(numerics.get("actual_nonfinite"), 0, f"{context}: Helion nonfinite")
    check_equal(numerics.get("expected_nonfinite"), 0, f"{context}: SDPA nonfinite")
    check_equal(numerics.get("passed"), True, f"{context}: numerical result")
    count = numerics.get("count")
    mismatch_count = numerics.get("mismatch_count")
    require(
        isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
        and isinstance(mismatch_count, int)
        and not isinstance(mismatch_count, bool)
        and 0 <= mismatch_count <= count,
        f"{context}: invalid mismatch counts",
    )
    mismatch_fraction = numerics.get("mismatch_fraction")
    require(
        isinstance(mismatch_fraction, (int, float))
        and not isinstance(mismatch_fraction, bool)
        and math.isclose(
            mismatch_fraction, mismatch_count / count, rel_tol=0.0, abs_tol=1e-15
        ),
        f"{context}: mismatch fraction does not match counts",
    )
    for metric, threshold in (
        ("max_abs", "max_abs_exclusive"),
        ("nrmse", "nrmse_exclusive"),
        ("mismatch_fraction", "mismatch_fraction_exclusive"),
    ):
        metric_value = numerics.get(metric)
        require(
            isinstance(metric_value, (int, float))
            and not isinstance(metric_value, bool)
            and math.isfinite(metric_value)
            and 0.0 <= metric_value < PEAKY_STRESS_THRESHOLDS[threshold],
            f"{context}: {metric} failed exclusive gate",
        )
    repeat = value.get("helion_exact_repeatability")
    require(isinstance(repeat, dict), f"{context}: missing exact repeatability")
    require(
        isinstance(repeat.get("count"), int)
        and not isinstance(repeat["count"], bool)
        and repeat["count"] > 0,
        f"{context}: invalid repeatability count",
    )
    check_equal(repeat.get("passed"), True, f"{context}: repeatability result")
    check_equal(repeat.get("different"), 0, f"{context}: repeat differences")
    check_equal(
        repeat.get("different_fraction"), 0.0, f"{context}: repeat difference rate"
    )


def expected_strict_artifact_identities(
    manifest: dict[tuple[object, ...], dict[str, str]],
) -> dict[str, dict[str, str]]:
    return {
        row["case"]: {
            "search_result_sha256": row["result_sha256"],
            "selected_config_sha256": row["selected_config_sha256"],
            "selected_source_sha256": row["selected_source_sha256"],
            "compiler_seed_policy_sha256": canonical_sha256(
                json.loads(row["compiler_seed_policy_json"])
            ),
            "terminal_refinement_policy_sha256": row[
                "terminal_refinement_policy_sha256"
            ],
            "terminal_coordinate_surface_sha256": row[
                "terminal_coordinate_surface_sha256"
            ],
            "terminal_refinement_sha256": row["terminal_refinement_sha256"],
        }
        for row in manifest.values()
    }


def validate_paired_artifacts(
    raw: dict[str, Any],
    paired: dict[tuple[object, ...], dict[str, Any]],
    strict_manifest: dict[tuple[object, ...], dict[str, str]],
    run_manifest_path: Path,
    static_validation_path: Path,
) -> None:
    run_all8.require_portable_json(raw, context="paired raw artifact")
    run_manifest_path = run_manifest_path.resolve()
    static_validation_path = static_validation_path.resolve()
    check_equal(
        static_validation_path.parent,
        run_manifest_path.parent,
        "paired run/static artifact directory",
    )
    check_equal(
        file_sha256(run_manifest_path),
        raw["run_manifest_sha256"],
        "paired run manifest digest",
    )
    check_equal(
        file_sha256(static_validation_path),
        raw["static_validation_sha256"],
        "paired static validation digest",
    )
    run_manifest = load_object(run_manifest_path)
    static_validation = load_object(static_validation_path)
    check_equal(run_manifest.get("schema_version"), 5, "run manifest schema")
    check_equal(run_manifest.get("status"), "PASS", "run manifest status")
    check_equal(run_manifest.get("errors"), [], "run manifest errors")
    check_equal(
        run_manifest.get("termination_signals"), [], "run manifest termination signals"
    )
    check_equal(static_validation.get("schema_version"), 5, "static schema")
    check_equal(static_validation.get("status"), "READY", "static status")
    harness_sha256 = run_manifest.get("harness_sha256")
    check_equal(
        static_validation.get("harness_sha256"),
        harness_sha256,
        "static/run harness identity",
    )
    check_equal(raw.get("harness_sha256"), harness_sha256, "raw harness identity")
    combine_results.validate_harness_sha256(harness_sha256)

    environment_policy = static_validation.get("worker_environment_policy")
    require(isinstance(environment_policy, dict), "worker environment policy missing")
    scrubbed_prefixes = environment_policy.get("scrubbed_prefixes")
    require(
        isinstance(scrubbed_prefixes, list)
        and {"CUDNN_", "TORCH_CUDNN_", "CUTE_DSL_", "HELION_"}
        <= set(scrubbed_prefixes),
        "worker environment policy does not scrub CuDNN/CuTe/Helion prefixes",
    )
    controlled_values = environment_policy.get("controlled_values")
    require(
        isinstance(controlled_values, dict), "controlled environment policy missing"
    )
    runtime_checkout = static_validation.get("runtime_checkout")
    require(
        isinstance(runtime_checkout, str)
        and runtime_checkout
        and Path(runtime_checkout).is_absolute(),
        "static runtime checkout must be an absolute path",
    )
    check_equal(
        controlled_values.get("CUDA_DEVICE_ORDER"),
        "PCI_BUS_ID",
        "worker CUDA device order policy",
    )
    check_equal(
        controlled_values.get("PYTHONPATH"),
        runtime_checkout,
        "worker PYTHONPATH policy",
    )
    check_equal(
        controlled_values.get("PYTHONPYCACHEPREFIX"),
        "fresh per shape and campaign",
        "worker PYTHONPYCACHEPREFIX policy",
    )

    protocol = raw["protocol"]
    campaign_seeds = protocol["campaign_seeds"]
    check_equal(
        run_manifest.get("campaign_seeds"), campaign_seeds, "run campaign seeds"
    )
    check_equal(
        static_validation.get("campaign_seeds"),
        campaign_seeds,
        "static campaign seeds",
    )
    expected_identities = expected_strict_artifact_identities(strict_manifest)
    check_equal(
        raw.get("strict_artifacts"), expected_identities, "raw strict identities"
    )
    check_equal(
        run_manifest.get("strict_artifacts"),
        expected_identities,
        "run strict identities",
    )
    check_equal(
        static_validation.get("strict_artifacts"),
        expected_identities,
        "static strict identities",
    )
    static_provenance = static_validation.get("provenance")
    require(isinstance(static_provenance, dict), "static provenance is missing")
    check_equal(set(static_provenance), set(expected_identities), "static cases")
    for case, identity in expected_identities.items():
        provenance = static_provenance[case]
        require(isinstance(provenance, dict), f"static {case}: bad provenance")
        check_equal(
            {field: provenance.get(field) for field in identity},
            identity,
            f"static {case}: strict identity",
        )

    expected_workers: dict[tuple[int, str], dict[str, Any]] = {}
    for key, result in paired.items():
        case = strict_manifest[key]["case"]
        check_equal(
            run_all8.portable_search_provenance(static_provenance[case]),
            result.get("provenance"),
            f"static {case}: portable provenance",
        )
        campaigns = index_campaign_entries(
            result.get("raw_campaigns"), context=f"{case}: raw campaigns"
        )
        campaign_environments = index_campaign_entries(
            result.get("campaign_environments"),
            context=f"{case}: campaign environments",
        )
        first_environment = {
            field: value
            for field, value in campaign_environments[1].items()
            if field != "campaign"
        }
        check_equal(
            result.get("environment"),
            first_environment,
            f"{case}: primary portable environment",
        )
        for campaign, entry in campaigns.items():
            seed = entry.get("orchestrator_seed")
            protocol_seed = entry.get("protocol_seed")
            require(
                isinstance(seed, int)
                and not isinstance(seed, bool)
                and isinstance(protocol_seed, int)
                and not isinstance(protocol_seed, bool),
                f"{case}: invalid campaign seeds",
            )
            check_equal(
                seed,
                campaign_seeds[campaign - 1],
                f"{case}: campaign {campaign} seed",
            )
            expected_workers[(seed, case)] = {
                "worker_input_seed": protocol_seed,
                "physical_gpu": 6 if key[5] else 7,
                "full_provenance": static_provenance[case],
                "environment": {
                    field: value
                    for field, value in campaign_environments[campaign].items()
                    if field != "campaign"
                },
            }

    records = run_manifest.get("records")
    require(isinstance(records, list), "run manifest records are missing")
    indexed_records: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        require(isinstance(record, dict), "run manifest record is not an object")
        key = (record.get("campaign_seed"), record.get("case"))
        require(key not in indexed_records, f"duplicate run record {key}")
        indexed_records[key] = record
        check_equal(record.get("returncode"), 0, f"run record {key}: return code")
        check_equal(
            record.get("termination_reason"), None, f"run record {key}: termination"
        )
        check_equal(record.get("cleanup"), None, f"run record {key}: cleanup")
        started_ns = record.get("started_ns")
        finished_ns = record.get("finished_ns")
        require(
            isinstance(started_ns, int)
            and not isinstance(started_ns, bool)
            and isinstance(finished_ns, int)
            and not isinstance(finished_ns, bool)
            and started_ns <= finished_ns,
            f"run record {key}: invalid timestamps",
        )
    check_equal(set(indexed_records), set(expected_workers), "run worker record set")
    for key, expected in expected_workers.items():
        check_equal(
            indexed_records[key].get("physical_gpu"),
            expected["physical_gpu"],
            f"run record {key}: physical GPU",
        )
        controlled = indexed_records[key].get("controlled_environment")
        require(isinstance(controlled, dict), f"run record {key}: missing environment")
        check_equal(
            controlled.get("CUDA_VISIBLE_DEVICES"),
            indexed_records[key].get("gpu_uuid"),
            f"run record {key}: CUDA visibility",
        )
        check_equal(
            controlled.get("CUDA_DEVICE_ORDER"),
            "PCI_BUS_ID",
            f"run record {key}: CUDA device order",
        )
        check_equal(
            controlled.get("HELION_BACKEND"), "cute", f"run record {key}: backend"
        )
        check_equal(
            controlled.get("HELION_DISABLE_AUTOTUNER_HEURISTICS"),
            "0",
            f"run record {key}: heuristic policy",
        )
        check_equal(
            controlled.get("PYTHONHASHSEED"),
            "0",
            f"run record {key}: hash seed",
        )
        check_equal(
            controlled.get("PYTHONPATH"),
            runtime_checkout,
            f"run record {key}: PYTHONPATH",
        )
        cache_dir_value = controlled.get("HELION_CACHE_DIR")
        pycache_value = controlled.get("PYTHONPYCACHEPREFIX")
        require(
            isinstance(cache_dir_value, str)
            and cache_dir_value
            and Path(cache_dir_value).is_absolute(),
            f"run record {key}: invalid HELION_CACHE_DIR",
        )
        require(
            isinstance(pycache_value, str)
            and pycache_value
            and Path(pycache_value).is_absolute(),
            f"run record {key}: invalid PYTHONPYCACHEPREFIX",
        )
        cache_dir = Path(cache_dir_value)
        pycache_prefix = Path(pycache_value)
        check_equal(
            pycache_prefix,
            cache_dir / "pycache",
            f"run record {key}: PYTHONPYCACHEPREFIX location",
        )
        require(
            not pycache_prefix.resolve().is_relative_to(
                Path(runtime_checkout).resolve()
            ),
            f"run record {key}: PYTHONPYCACHEPREFIX must be outside runtime checkout",
        )
        check_equal(
            indexed_records[key].get("worker_timeout_seconds"),
            static_validation.get("worker_timeout_seconds"),
            f"run record {key}: worker timeout",
        )
        scrubbed = indexed_records[key].get("scrubbed_environment_keys")
        require(
            isinstance(scrubbed, list)
            and all(isinstance(name, str) for name in scrubbed),
            f"run record {key}: invalid scrubbed environment record",
        )

    plans = static_validation.get("planned_commands")
    require(isinstance(plans, list), "static planned commands are missing")
    indexed_plans: dict[tuple[int, str], dict[str, Any]] = {}
    for plan in plans:
        require(isinstance(plan, dict), "static plan is not an object")
        key = (plan.get("campaign_seed"), plan.get("case"))
        require(key not in indexed_plans, f"duplicate static plan {key}")
        indexed_plans[key] = plan
    check_equal(set(indexed_plans), set(expected_workers), "static worker plan set")
    for key, expected in expected_workers.items():
        plan = indexed_plans[key]
        check_equal(
            plan.get("worker_input_seed"),
            expected["worker_input_seed"],
            f"static plan {key}: worker seed",
        )
        check_equal(
            plan.get("physical_gpu"),
            expected["physical_gpu"],
            f"static plan {key}: physical GPU",
        )
        check_equal(
            plan.get("gpu_uuid"),
            indexed_records[key].get("gpu_uuid"),
            f"static plan {key}: GPU UUID",
        )
        check_equal(
            plan.get("cuda_visible_devices"),
            indexed_records[key].get("gpu_uuid"),
            f"static plan {key}: CUDA visibility",
        )
        check_equal(
            plan.get("command"),
            indexed_records[key].get("command"),
            f"static plan {key}: command",
        )
        command = plan.get("command")
        require(isinstance(command, list), f"static plan {key}: invalid command")
        worker_sha_flag = [
            index
            for index, value in enumerate(command)
            if value == "--expected-worker-sha256"
        ]
        require(
            len(worker_sha_flag) == 1 and worker_sha_flag[0] + 1 < len(command),
            f"static plan {key}: missing worker harness digest",
        )
        check_equal(
            command[worker_sha_flag[0] + 1],
            harness_sha256["paired_worker.py"],
            f"static plan {key}: worker harness digest",
        )
    check_equal(
        static_validation.get("bootstrap"),
        {
            "samples": protocol["bootstrap_samples"],
            "base_seed": protocol["bootstrap_base_seed"],
            "method": protocol["bootstrap_method"],
        },
        "static bootstrap protocol",
    )

    output_dir = run_manifest_path.parent
    combine_results.validate_manifest_files(run_manifest, output_dir)
    for key, record in indexed_records.items():
        worker_path = combine_results.resolve_output_reference(
            output_dir, record["output"]
        )
        worker = load_object(worker_path)
        check_equal(worker.get("status"), "PASS", f"worker {key}: status")
        check_equal(worker.get("campaign_seed"), key[0], f"worker {key}: campaign seed")
        check_equal(
            worker.get("harness_sha256"),
            {"paired_worker.py": harness_sha256["paired_worker.py"]},
            f"worker {key}: harness identity",
        )
        check_equal(
            worker.get("provenance"),
            expected_workers[key]["full_provenance"],
            f"worker {key}: full provenance",
        )
        worker_environment = worker.get("environment")
        require(
            isinstance(worker_environment, dict),
            f"worker {key}: missing measured environment",
        )
        check_equal(
            run_all8.portable_worker_environment(worker_environment),
            expected_workers[key]["environment"],
            f"worker {key}: portable environment",
        )
        selected_source = expected_identities[key[1]]["selected_source_sha256"]
        regenerated_kernel = worker.get("regenerated_kernel")
        require(
            isinstance(regenerated_kernel, dict),
            f"worker {key}: missing regenerated kernel",
        )
        for field in (
            "regenerated_source_sha256",
            "compiled_source_sha256",
            "expected_source_sha256",
        ):
            check_equal(
                regenerated_kernel.get(field),
                selected_source,
                f"worker {key}: {field}",
            )
        check_equal(
            regenerated_kernel.get("source_hash_matches_search"),
            True,
            f"worker {key}: source match",
        )
        check_equal(
            regenerated_kernel.get("compiled_from_current_examples_attention"),
            True,
            f"worker {key}: source origin",
        )
    regenerated = combine_results.aggregate_results(
        output_dir,
        tuple(campaign_seeds),
        protocol["bootstrap_samples"],
        protocol["bootstrap_base_seed"],
        run_manifest,
    )
    regenerated["run_manifest_sha256"] = file_sha256(run_manifest_path)
    regenerated["static_validation_sha256"] = file_sha256(static_validation_path)
    if raw != regenerated:
        raise RuntimeError(
            "paired raw regeneration differs: "
            f"supplied_sha256={canonical_sha256(raw)}, "
            f"regenerated_sha256={canonical_sha256(regenerated)}"
        )


def selected_source_sha256(provenance: dict[str, Any], *, context: str) -> str:
    values = {
        value
        for key in ("selected_source_sha256", "selected_source_hash")
        if isinstance((value := provenance.get(key)), str)
    }
    require(
        len(values) == 1, f"{context}: missing or inconsistent selected source hash"
    )
    value = next(iter(values))
    require(
        re.fullmatch(r"[0-9a-f]{64}", value) is not None, f"{context}: bad source hash"
    )
    return value


def validate_coverage_design(
    provenance: dict[str, Any], *, context: str
) -> list[dict[str, Any]]:
    design = provenance.get("flash_structural_coverage_design")
    require(isinstance(design, list) and design, f"{context}: missing coverage design")
    check_equal(
        provenance.get("flash_structural_coverage_design_count"),
        len(design),
        f"{context}: coverage design count",
    )
    configs = []
    for index, candidate in enumerate(design):
        require(isinstance(candidate, dict), f"{context}: bad coverage entry {index}")
        config = candidate.get("config")
        require(isinstance(config, dict), f"{context}: bad coverage config {index}")
        check_equal(
            candidate.get("config_sha256"),
            canonical_sha256(config),
            f"{context}: coverage config {index} digest",
        )
        configs.append(config)
    check_equal(
        provenance.get("flash_structural_coverage_design_sha256"),
        canonical_sha256(configs),
        f"{context}: coverage design digest",
    )
    check_equal(
        provenance.get("flash_structural_coverage_uncovered_values"),
        [],
        f"{context}: uncovered structural values",
    )
    active_values = provenance.get("flash_structural_coverage_active_values")
    require(
        isinstance(active_values, list) and active_values,
        f"{context}: missing active structural values",
    )
    for active in active_values:
        require(
            isinstance(active, dict) and isinstance(active.get("key"), str),
            f"{context}: malformed active structural value {active!r}",
        )
        require(
            any(config.get(active["key"]) == active.get("value") for config in configs),
            f"{context}: coverage design omits {active!r}",
        )
    fragment_default = provenance.get("flash_fragment_default_config")
    require(isinstance(fragment_default, dict), f"{context}: missing fragment default")
    check_equal(
        provenance.get("flash_fragment_default_sha256"),
        canonical_sha256(fragment_default),
        f"{context}: fragment default digest",
    )
    return configs


def validate_strict_result(
    path: Path,
    strict: dict[str, Any],
    paired: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    check_equal(strict.get("accuracy"), "PASS", f"{path}: accuracy")
    check_equal(strict.get("benchmark_timer"), "wall", f"{path}: timer")
    check_equal(strict.get("flop_model"), FLOP_MODEL_NAME, f"{path}: FLOP model")
    check_equal(
        strict.get("physical_gpu"),
        "6" if strict["shape"]["causal"] else "7",
        f"{path}: GPU",
    )
    check_equal(strict.get("power_cap_w"), 750, f"{path}: power cap")
    require(
        isinstance(strict.get("input_seed"), int)
        and not isinstance(strict["input_seed"], bool),
        f"{path}: missing search input seed",
    )
    version = strict.get("version")
    require(
        isinstance(version, str) and ".dirty" not in version, f"{path}: dirty version"
    )
    match = re.fullmatch(r"Helion ([^;]*\+g([0-9a-f]{7,40})); CuTe ([^;\s]+)", version)
    require(match is not None, f"{path}: unexpected version {version!r}")
    check_equal(
        match.group(3),
        build_strict_manifest.EXPECTED_CUTE_VERSION,
        f"{path}: CuTe version",
    )
    check_equal(
        strict.get("version_label"),
        f"Helion {match.group(1)} / CuTe {match.group(3)}",
        f"{path}: version label",
    )

    overrides = strict.get("helion_overrides")
    require(isinstance(overrides, dict), f"{path}: missing Helion overrides")
    check_equal(overrides.get("autotuned"), True, f"{path}: autotuned")
    check_equal(overrides.get("force_autotune"), True, f"{path}: force autotune")
    check_equal(overrides.get("config_overrides"), {}, f"{path}: config overrides")
    check_equal(overrides.get("seed_config_overrides"), {}, f"{path}: seed overrides")
    check_equal(overrides.get("return_lse"), False, f"{path}: return LSE")
    provenance = overrides.get("autotune_provenance")
    require(isinstance(provenance, dict), f"{path}: missing autotune provenance")
    expected = {
        "require_full_autotune": True,
        "post_measurement_source_verified": True,
        "effort": "full",
        "requested_force_autotune": True,
        "effective_force_autotune": True,
        "fixed_config": False,
        "autotune_budget_seconds": None,
        "autotune_max_generations": None,
        "autotune_best_of_k": 1,
        "autotune_accuracy_check": True,
        "disable_autotuner_heuristics": False,
        "autotune_initial_population_strategy_override": None,
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
        "autotune_benchmark_subprocess": True,
        "autotune_adaptive_timeout": True,
        "autotune_force_persistent": False,
        "autotune_finishing_rounds_env": "",
        "autotune_ignore_errors": False,
        "autotune_search_acf": [],
        "autotune_config_filter": False,
        "autotune_config_overrides": {},
        "user_seed_configs": False,
        "compiler_default_config": False,
        "kernel_declared_config_count": 0,
        "active_value_prior_keys": [],
        "flash_value_prior_keys": [],
        "cute_flash_env_overrides": {},
        "final_correctness_enabled": True,
        "final_correctness_launches": 64,
        "final_repeatability_passed": True,
        "final_correctness_passed": True,
        "autotune_cache": "LocalAutotuneCache",
        "rebenchmark_env_overrides": {},
    }
    for field, value in expected.items():
        check_equal(provenance.get(field), value, f"{path}: provenance.{field}")
    compiler_seed_policy = build_strict_manifest.validate_compiler_seed_policy(
        path, provenance
    )
    post_measurement_source = provenance.get("post_measurement_source")
    require(
        isinstance(post_measurement_source, dict),
        f"{path}: missing post-measurement source provenance",
    )
    check_equal(
        post_measurement_source.get("helion_source_tree_sha256"),
        provenance.get("helion_source_tree_sha256"),
        f"{path}: post-measurement source digest",
    )
    check_equal(
        post_measurement_source.get("helion_checkout_git_commit"),
        provenance.get("helion_checkout_git_commit"),
        f"{path}: post-measurement source commit",
    )
    check_equal(
        post_measurement_source.get("helion_source_tree_dirty"),
        False,
        f"{path}: post-measurement source cleanliness",
    )
    require(
        not provenance.get("dense_d64_2cta_performance_anchor_present", False),
        f"{path}: legacy dense shape anchor is active",
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
    coverage_configs = validate_coverage_design(provenance, context=str(path))
    selected_config = provenance.get("selected_config")
    require(isinstance(selected_config, dict), f"{path}: missing selected config")
    source_sha256 = selected_source_sha256(provenance, context=str(path))
    trials = provenance.get("trials")
    require(isinstance(trials, list) and trials, f"{path}: missing trials")
    check_equal(len(trials), provenance["autotune_best_of_k"], f"{path}: trial count")
    matching_trials = []
    for index, trial in enumerate(trials, 1):
        require(isinstance(trial, dict), f"{path}: trial {index} is not an object")
        require(
            isinstance(trial.get("num_configs_tested"), int)
            and trial["num_configs_tested"] >= 100,
            f"{path}: trial {index} tested fewer than 100 candidates",
        )
        require(
            isinstance(trial.get("num_successful_candidate_measurements"), int)
            and trial["num_successful_candidate_measurements"] >= 100,
            f"{path}: trial {index} has fewer than 100 successful measurements",
        )
        require(
            isinstance(trial.get("num_unique_sources"), int)
            and trial["num_unique_sources"] >= 100,
            f"{path}: trial {index} has fewer than 100 unique sources",
        )
        require(
            isinstance(trial.get("num_generations"), int)
            and trial["num_generations"] > 0,
            f"{path}: trial {index} has no LFBO generations",
        )
        check_equal(
            trial.get("search_algorithm"),
            "LFBOTreeSearch",
            f"{path}: trial {index} search algorithm",
        )
        check_equal(
            trial.get("num_worker_failures"),
            0,
            f"{path}: trial {index} worker failures",
        )
        require(
            type(trial.get("num_isolated_rebenchmark_timeouts")) is int
            and trial["num_isolated_rebenchmark_timeouts"] >= 0,
            f"{path}: trial {index} isolated rebenchmark timeout count",
        )
        check_equal(
            trial.get("num_accuracy_failures"),
            0,
            f"{path}: trial {index} accuracy failures",
        )
        check_equal(
            trial.get("selected_source_was_measured"),
            True,
            f"{path}: trial {index} measured source",
        )
        if (
            trial.get("selected_config") == selected_config
            and trial.get("selected_source_hash") == source_sha256
        ):
            matching_trials.append(trial)
    require(
        matching_trials,
        f"{path}: selected source was not measured by the full search",
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
    check_equal(sorted(nearest), computed_nearest, f"{path}: nearest coverage digests")
    check_equal(
        provenance.get("selected_config_is_structural_coverage_design_member"),
        distance == 0,
        f"{path}: coverage membership flag",
    )

    paired_provenance = paired.get("provenance")
    require(isinstance(paired_provenance, dict), "paired result is missing provenance")
    check_equal(
        paired_provenance.get("strict_full_autotune_validated"),
        True,
        f"{path}: paired strict validation",
    )
    check_equal(
        paired_provenance.get("search_result_sha256"),
        file_sha256(path),
        f"{path}: paired search result digest",
    )
    check_equal(
        paired_provenance.get("search_version"), version, f"{path}: paired version"
    )
    runtime_git_head = paired_provenance.get("runtime_git_head")
    require(
        isinstance(runtime_git_head, str)
        and re.fullmatch(r"[0-9a-f]{40}", runtime_git_head) is not None
        and runtime_git_head.startswith(match.group(2)),
        f"{path}: paired runtime revision does not match version",
    )
    check_equal(
        paired_provenance.get("runtime_tracked_clean"),
        True,
        f"{path}: paired runtime checkout cleanliness",
    )
    check_equal(
        paired_provenance.get("selected_config"),
        selected_config,
        f"{path}: paired selected config",
    )
    check_equal(
        paired_provenance.get("selected_config_sha256"),
        canonical_sha256(selected_config),
        f"{path}: paired selected config digest",
    )
    check_equal(
        paired_provenance.get("selected_source_sha256"),
        source_sha256,
        f"{path}: paired source digest",
    )
    check_equal(
        paired_provenance.get("compiler_seed_policy"),
        compiler_seed_policy,
        f"{path}: paired compiler seed policy",
    )
    check_equal(
        paired_provenance.get("compiler_seed_policy_sha256"),
        canonical_sha256(compiler_seed_policy),
        f"{path}: paired compiler seed policy digest",
    )
    terminal_refinement = paired_provenance.get("structural_design_execution", {}).get(
        "terminal_refinement"
    )
    require(
        isinstance(terminal_refinement, dict),
        f"{path}: paired terminal refinement validation",
    )
    check_equal(
        paired_provenance.get("terminal_refinement_sha256"),
        canonical_sha256(terminal_refinement),
        f"{path}: paired terminal refinement digest",
    )
    check_equal(
        terminal_refinement.get("transcript_sha256"),
        canonical_sha256(
            matching_trials[0]["search_phase_metrics"]["terminal_coordinate_refinement"]
        ),
        f"{path}: paired/search terminal refinement transcript",
    )
    for field, provenance_field in (
        (
            "terminal_refinement_policy_sha256",
            "flash_terminal_coordinate_refinement_policy_sha256",
        ),
        (
            "terminal_coordinate_surface_sha256",
            "flash_terminal_coordinate_surface_catalog_sha256",
        ),
    ):
        check_equal(
            paired_provenance.get(field),
            provenance.get(provenance_field),
            f"{path}: paired {field}",
        )
    check_equal(
        paired_provenance.get("input_seed"),
        strict["input_seed"],
        f"{path}: paired search input seed",
    )
    environment = paired.get("environment")
    require(isinstance(environment, dict), "paired result is missing environment")
    check_equal(
        environment.get("cute_version"), match.group(3), f"{path}: CuTe version"
    )
    regenerated = paired.get("regenerated_kernels")
    regenerated_by_campaign = index_campaign_entries(
        regenerated, context=f"{path}: regenerated kernels"
    )
    correctness = paired.get("correctness")
    correctness_by_campaign = index_campaign_entries(
        correctness, context=f"{path}: correctness"
    )
    raw_by_campaign = index_campaign_entries(
        paired.get("raw_campaigns"), context=f"{path}: raw campaigns"
    )
    for campaign in (1, 2):
        regenerated_record = regenerated_by_campaign[campaign]
        require(
            regenerated_record.get("source_hash_matches_search") is True
            and regenerated_record.get("regenerated_source_sha256") == source_sha256
            and regenerated_record.get("compiled_source_sha256") == source_sha256
            and regenerated_record.get("expected_source_sha256") == source_sha256
            and regenerated_record.get("compiled_from_current_examples_attention")
            is True,
            f"{path}: campaign {campaign} did not reproduce selected source",
        )
        check_equal(
            regenerated_record.get("terminal_refinement_transcript_sha256"),
            terminal_refinement.get("transcript_sha256"),
            f"{path}: campaign {campaign} live terminal refinement replay",
        )
        check_equal(
            regenerated_record.get("terminal_refinement_policy_sha256"),
            paired_provenance["terminal_refinement_policy_sha256"],
            f"{path}: campaign {campaign} live terminal refinement policy",
        )
        check_equal(
            regenerated_record.get("terminal_coordinate_surface_sha256"),
            paired_provenance["terminal_coordinate_surface_sha256"],
            f"{path}: campaign {campaign} live terminal coordinate surface",
        )
        check_equal(
            regenerated_record.get("terminal_projection_request_count"),
            terminal_refinement.get("projection_attempt_count"),
            f"{path}: campaign {campaign} live terminal projection count",
        )
        require(
            regenerated_record.get("live_full_autotune_validated") is True,
            f"{path}: campaign {campaign} did not live-validate full autotune",
        )
        correctness_record = correctness_by_campaign[campaign]
        repeats = correctness_record.get("helion_exact_repeatability")
        require(
            correctness_record.get("helion_vs_cudnn_sdpa", {}).get("passed") is True
            and isinstance(repeats, list)
            and repeats
            and all(repeat.get("passed") is True for repeat in repeats),
            f"{path}: campaign {campaign} correctness or repeatability failed",
        )
        validate_peaky_stress(
            correctness_record.get("post_timing_peaky_logits"),
            context=f"{path}: campaign {campaign} peaky logits",
        )
        require(
            isinstance(raw_by_campaign[campaign].get("orchestrator_seed"), int),
            f"{path}: campaign {campaign} has no orchestrator seed",
        )
    return provenance, source_sha256, paired_provenance


def validate_summary_stats(
    stats: object,
    *,
    flops: float,
    context: str,
) -> dict[str, Any]:
    require(isinstance(stats, dict), f"{context}: missing event statistics")
    runs = stats.get("runs_ms")
    require(
        isinstance(runs, list)
        and runs
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value > 0
            for value in runs
        ),
        f"{context}: invalid runs_ms",
    )
    computed = {
        "best_ms": min(runs),
        "median_ms": statistics.median(runs),
        "mean_ms": statistics.fmean(runs),
        "std_ms": statistics.stdev(runs) if len(runs) > 1 else 0.0,
    }
    computed["best_tflops"] = flops / (computed["best_ms"] * 1e9)
    computed["median_tflops"] = flops / (computed["median_ms"] * 1e9)
    for field, expected in computed.items():
        actual = stats.get(field)
        require(
            isinstance(actual, (int, float))
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{context}: inconsistent {field}: expected {expected!r}, got {actual!r}",
        )
    return stats


def timing_fields(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "best_ms": stats["best_ms"],
        "median_ms": stats["median_ms"],
        "mom_median_ms": stats["median_ms"],
        "mean_ms": stats["mean_ms"],
        "std_ms": stats["std_ms"],
        "runs_ms": stats["runs_ms"],
        "best_tflops": stats["best_tflops"],
        "median_tflops": stats["median_tflops"],
        "mom_median_tflops": stats["median_tflops"],
    }


def recompute_paired_statistics(
    campaigns: list[dict[str, Any]],
    *,
    timer: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[float, list[float]]:
    strata = [
        [
            math.log(
                pair["times"]["sdpa"][f"{timer}_ms"]
                / pair["times"]["helion"][f"{timer}_ms"]
            )
            for pair in campaign["raw_pairs"]
        ]
        for campaign in campaigns
    ]
    log_ratios = [value for stratum in strata for value in stratum]
    ratio = 100.0 * math.expm1(statistics.fmean(log_ratios))
    rng = random.Random(bootstrap_seed)
    bootstrap_values = []
    for _ in range(bootstrap_samples):
        sampled = [
            stratum[rng.randrange(len(stratum))] for stratum in strata for _ in stratum
        ]
        bootstrap_values.append(100.0 * math.expm1(statistics.fmean(sampled)))
    bootstrap_values.sort()
    return ratio, [
        bootstrap_values[int(bootstrap_samples * 0.025)],
        bootstrap_values[int(bootstrap_samples * 0.975)],
    ]


def paired_ratio_pct(campaign: dict[str, Any], *, timer: str) -> float:
    log_ratios = [
        math.log(
            pair["times"]["sdpa"][f"{timer}_ms"]
            / pair["times"]["helion"][f"{timer}_ms"]
        )
        for pair in campaign["raw_pairs"]
    ]
    return 100.0 * math.expm1(statistics.fmean(log_ratios))


def point_direction(value: float) -> str:
    if value > 0:
        return "gain"
    if value < 0:
        return "regression"
    return "parity"


def confidence_interval_direction(interval: list[float]) -> str:
    if interval[0] > 0:
        return "gain"
    if interval[1] < 0:
        return "regression"
    return "inconclusive"


def classify_sub_half_percent_direction(
    campaigns: list[dict[str, Any]], summary: dict[str, Any]
) -> dict[str, Any]:
    event = summary["event"]
    wall = summary["wall"]
    event_ratio = event["paired_log_ratio_pct"]
    wall_ratio = wall["paired_log_ratio_pct"]
    event_ci = event["paired_log_ratio_stratified_bootstrap_95_ci_pct"]
    wall_ci = wall["paired_log_ratio_stratified_bootstrap_95_ci_pct"]
    event_campaign_ratios = [
        paired_ratio_pct(campaign, timer="event") for campaign in campaigns
    ]
    wall_campaign_ratios = [
        paired_ratio_pct(campaign, timer="wall") for campaign in campaigns
    ]
    helion_event_values = [
        pair["times"]["helion"]["event_ms"]
        for campaign in campaigns
        for pair in campaign["raw_pairs"]
    ]
    sdpa_event_values = [
        pair["times"]["sdpa"]["event_ms"]
        for campaign in campaigns
        for pair in campaign["raw_pairs"]
    ]
    plotted_marginal_ratio = 100.0 * (
        statistics.median(sdpa_event_values) / statistics.median(helion_event_values)
        - 1.0
    )
    reported_marginal_ratio = event.get("median_throughput_ratio_pct")
    require(
        isinstance(reported_marginal_ratio, (int, float))
        and not isinstance(reported_marginal_ratio, bool)
        and math.isfinite(reported_marginal_ratio)
        and math.isclose(
            reported_marginal_ratio,
            plotted_marginal_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
        "event plotted marginal-median ratio does not match raw calls",
    )

    event_direction = point_direction(event_ratio)
    evidence_directions = {
        "event_point": event_direction,
        "event_confidence_interval": confidence_interval_direction(event_ci),
        "wall_point": point_direction(wall_ratio),
        "wall_confidence_interval": confidence_interval_direction(wall_ci),
        "plotted_marginal_median": point_direction(plotted_marginal_ratio),
    }
    campaign_directions = {
        "event": [point_direction(value) for value in event_campaign_ratios],
        "wall": [point_direction(value) for value in wall_campaign_ratios],
    }
    in_scope = abs(event_ratio) < EVENT_WALL_DECISION_THRESHOLD_PCT
    checks = {
        name: direction == event_direction
        for name, direction in evidence_directions.items()
        if name != "event_point"
    }
    checks["event_campaigns"] = all(
        direction == event_direction for direction in campaign_directions["event"]
    )
    checks["wall_campaigns"] = all(
        direction == event_direction for direction in campaign_directions["wall"]
    )
    directional = event_direction in {"gain", "regression"} and all(checks.values())
    classification = event_direction if in_scope and directional else "inconclusive"
    if not in_scope:
        classification = "not_applicable"
    return {
        "threshold_pct": EVENT_WALL_DECISION_THRESHOLD_PCT,
        "in_scope": in_scope,
        "classification": classification,
        "event_point_estimate_pct": event_ratio,
        "wall_point_estimate_pct": wall_ratio,
        "event_campaign_point_estimates_pct": event_campaign_ratios,
        "wall_campaign_point_estimates_pct": wall_campaign_ratios,
        "plotted_marginal_median_ratio_pct": plotted_marginal_ratio,
        "evidence_directions": evidence_directions,
        "campaign_directions": campaign_directions,
        "agreement_checks": checks,
    }


def validate_event_wall_consistency(
    campaigns: list[dict[str, Any]],
    *,
    summary: dict[str, Any],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    require(bootstrap_samples > 0, "event/wall bootstrap sample count must be positive")
    strata = []
    for campaign in campaigns:
        stratum = []
        for pair in campaign["raw_pairs"]:
            helion = pair["times"]["helion"]
            sdpa = pair["times"]["sdpa"]
            helion_log_bias = math.log(helion["wall_ms"] / helion["event_ms"])
            sdpa_log_bias = math.log(sdpa["wall_ms"] / sdpa["event_ms"])
            stratum.append(
                {
                    "helion": helion_log_bias,
                    "sdpa": sdpa_log_bias,
                    "relative": helion_log_bias - sdpa_log_bias,
                }
            )
        require(stratum, "event/wall validation requires non-empty campaigns")
        strata.append(stratum)

    metrics = ("relative", "helion", "sdpa")
    raw_logs = {
        metric: [row[metric] for stratum in strata for row in stratum]
        for metric in metrics
    }
    bootstrap_logs = {metric: [] for metric in metrics}
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_samples):
        sampled = [
            stratum[rng.randrange(len(stratum))] for stratum in strata for _ in stratum
        ]
        for metric in metrics:
            bootstrap_logs[metric].append(
                statistics.median(row[metric] for row in sampled)
            )

    def summarize(metric: str, bounds: tuple[float, float]) -> dict[str, Any]:
        values = [100.0 * math.expm1(value) for value in raw_logs[metric]]
        bootstrap_values = [
            100.0 * math.expm1(value) for value in bootstrap_logs[metric]
        ]
        bootstrap_values.sort()
        ci = [
            bootstrap_values[int(bootstrap_samples * 0.025)],
            bootstrap_values[int(bootstrap_samples * 0.975)],
        ]
        inlier_count = sum(bounds[0] <= value <= bounds[1] for value in values)
        required_inliers = math.ceil(EVENT_WALL_MIN_INLIER_FRACTION * len(values))
        return {
            "median_pct": 100.0 * math.expm1(statistics.median(raw_logs[metric])),
            "stratified_bootstrap_95_ci_pct": ci,
            "bounds_pct": list(bounds),
            "inlier_count": inlier_count,
            "required_inlier_count": required_inliers,
            "pair_count": len(values),
        }

    relative = summarize("relative", EVENT_WALL_RELATIVE_BIAS_BOUNDS_PCT)
    require(
        relative["stratified_bootstrap_95_ci_pct"][0]
        >= EVENT_WALL_RELATIVE_BIAS_BOUNDS_PCT[0]
        and relative["stratified_bootstrap_95_ci_pct"][1]
        <= EVENT_WALL_RELATIVE_BIAS_BOUNDS_PCT[1],
        "event/wall relative ratio bias bootstrap confidence interval exceeds "
        f"{list(EVENT_WALL_RELATIVE_BIAS_BOUNDS_PCT)}%: "
        f"{relative['stratified_bootstrap_95_ci_pct']}",
    )
    require(
        relative["inlier_count"] >= relative["required_inlier_count"],
        "event/wall relative ratio bias has only "
        f"{relative['inlier_count']}/{relative['pair_count']} inlier pairs; "
        f"at least {relative['required_inlier_count']} are required",
    )

    implementations = {}
    for implementation in ("helion", "sdpa"):
        result = summarize(implementation, EVENT_WALL_IMPLEMENTATION_BOUNDS_PCT)
        require(
            result["stratified_bootstrap_95_ci_pct"][0]
            >= EVENT_WALL_IMPLEMENTATION_BOUNDS_PCT[0]
            and result["stratified_bootstrap_95_ci_pct"][1]
            <= EVENT_WALL_IMPLEMENTATION_BOUNDS_PCT[1],
            f"event/wall {implementation} absolute sanity bootstrap confidence "
            f"interval exceeds {list(EVENT_WALL_IMPLEMENTATION_BOUNDS_PCT)}%: "
            f"{result['stratified_bootstrap_95_ci_pct']}",
        )
        require(
            result["inlier_count"] >= result["required_inlier_count"],
            f"event/wall {implementation} absolute sanity has only "
            f"{result['inlier_count']}/{result['pair_count']} inlier calls; "
            f"at least {result['required_inlier_count']} are required",
        )
        implementations[implementation] = result

    directional = classify_sub_half_percent_direction(campaigns, summary)

    return {
        "method": "stratified bootstrap median with shared pair resampling",
        "bootstrap_interpretation": STRATIFIED_BOOTSTRAP_INTERPRETATION,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "relative_ratio_bias": relative,
        "implementation_wall_over_event": implementations,
        "sub_half_percent_directional": directional,
    }


def paired_protocol(
    raw: dict[str, Any],
    paired: dict[str, Any],
    raw_artifact_label: str,
    raw_artifact_sha256: str,
) -> dict[str, Any]:
    protocol = raw["protocol"]
    summary = paired["summary"]
    event = summary["event"]
    campaigns = paired.get("raw_campaigns")
    require(
        isinstance(campaigns, list) and len(campaigns) == 2, "missing raw campaigns"
    )
    campaign_seeds = [campaign["orchestrator_seed"] for campaign in campaigns]
    input_seeds = [campaign["protocol_seed"] for campaign in campaigns]
    check_equal(
        {campaign["campaign"] for campaign in campaigns},
        {1, 2},
        "paired campaign identifiers",
    )
    check_equal(
        sorted(campaign_seeds),
        sorted(protocol["campaign_seeds"]),
        "paired campaign seed set",
    )
    require(len(set(input_seeds)) == 2, "paired input seeds must be distinct")
    pair_counts = [len(campaign.get("raw_pairs", [])) for campaign in campaigns]
    check_equal(
        pair_counts,
        [protocol["pairs_per_shape_per_campaign"]] * 2,
        "paired per-campaign call counts",
    )
    check_equal(
        sum(pair_counts),
        protocol["combined_pairs_per_shape"],
        "paired combined call count",
    )
    expected_order_counts = {
        ("helion", "sdpa"): protocol["pairs_per_shape_per_campaign"] // 2,
        ("sdpa", "helion"): protocol["pairs_per_shape_per_campaign"] // 2,
    }
    for campaign in campaigns:
        pairs = campaign["raw_pairs"]
        check_equal(
            [pair.get("pair_index") for pair in pairs],
            list(range(len(pairs))),
            f"campaign {campaign['campaign']} pair indices",
        )
        order_counts = {
            order: sum(tuple(pair.get("order", ())) == order for pair in pairs)
            for order in expected_order_counts
        }
        check_equal(
            order_counts,
            expected_order_counts,
            f"campaign {campaign['campaign']} balanced order",
        )
        for pair in pairs:
            check_equal(
                set(pair.get("times", {})),
                {"helion", "sdpa"},
                f"campaign {campaign['campaign']} pair timing implementations",
            )
            for implementation in ("helion", "sdpa"):
                for timer in ("event", "wall"):
                    value = pair["times"][implementation].get(f"{timer}_ms")
                    require(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and value > 0,
                        f"campaign {campaign['campaign']} has invalid {timer} timing",
                    )
    for timer_index, timer in enumerate(("event", "wall")):
        timer_summary = summary.get(timer)
        require(isinstance(timer_summary, dict), f"missing {timer} summary")
        for implementation in ("helion", "sdpa"):
            raw_values = [
                pair["times"][implementation][f"{timer}_ms"]
                for campaign in campaigns
                for pair in campaign["raw_pairs"]
            ]
            check_equal(
                raw_values,
                timer_summary[implementation]["runs_ms"],
                f"{implementation} raw {timer} calls",
            )
        derived_seed = (
            protocol["bootstrap_base_seed"]
            + paired["shape"]["seq_len"] * 2
            + int(paired["shape"]["causal"])
            + timer_index * 10_000_000
        )
        check_equal(
            timer_summary.get("paired_log_ratio_stratified_bootstrap_seed"),
            derived_seed,
            f"{timer} bootstrap seed",
        )
        expected_ratio, expected_ci = recompute_paired_statistics(
            campaigns,
            timer=timer,
            bootstrap_samples=protocol["bootstrap_samples"],
            bootstrap_seed=derived_seed,
        )
        require(
            math.isclose(
                timer_summary.get("paired_log_ratio_pct"),
                expected_ratio,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ),
            f"{timer} paired log-ratio does not match raw calls",
        )
        ci = timer_summary.get("paired_log_ratio_stratified_bootstrap_95_ci_pct")
        require(
            isinstance(ci, list)
            and len(ci) == 2
            and all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in ci
            )
            and ci[0] <= ci[1],
            f"{timer} bootstrap confidence interval is invalid",
        )
        require(
            all(
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
                for actual, expected in zip(ci, expected_ci, strict=True)
            ),
            f"{timer} bootstrap confidence interval does not match raw calls",
        )
    event_wall_validation = validate_event_wall_consistency(
        campaigns,
        summary=summary,
        bootstrap_samples=protocol["bootstrap_samples"],
        bootstrap_seed=(
            protocol["bootstrap_base_seed"]
            + paired["shape"]["seq_len"] * 2
            + int(paired["shape"]["causal"])
            + 20_000_000
        ),
    )
    return {
        "campaign_count": protocol["campaign_count"],
        "campaign_seeds": campaign_seeds,
        "input_seeds": input_seeds,
        "pairs_per_campaign": protocol["pairs_per_shape_per_campaign"],
        "combined_pairs": protocol["combined_pairs_per_shape"],
        "paired_log_ratio_pct": event["paired_log_ratio_pct"],
        "paired_log_ratio_stratified_bootstrap_95_ci_pct": event[
            "paired_log_ratio_stratified_bootstrap_95_ci_pct"
        ],
        "bootstrap_samples": protocol["bootstrap_samples"],
        "bootstrap_base_seed": protocol["bootstrap_base_seed"],
        "bootstrap_shape_seed": event["paired_log_ratio_stratified_bootstrap_seed"],
        "bootstrap_interpretation": STRATIFIED_BOOTSTRAP_INTERPRETATION,
        "event_wall_validation": event_wall_validation,
        "raw_artifact": raw_artifact_label,
        "raw_artifact_sha256": raw_artifact_sha256,
        "run_manifest_sha256": raw["run_manifest_sha256"],
        "static_validation_sha256": raw["static_validation_sha256"],
    }


def paired_estimate_notes(
    ratio: float, ci: list[float], directional: dict[str, Any]
) -> list[str]:
    notes = [
        (
            f"Paired log-ratio point estimate {ratio:+.3f}% with a stratified "
            f"bootstrap 95% CI [{ci[0]:+.3f}%, {ci[1]:+.3f}%]; the interval is "
            f"{STRATIFIED_BOOTSTRAP_INTERPRETATION}."
        )
    ]
    classification = directional["classification"]
    if classification in {"gain", "regression"}:
        notes.append(
            f"Sub-{EVENT_WALL_DECISION_THRESHOLD_PCT}% directional classification: "
            f"{classification}; event and wall intervals, both campaigns, and the "
            "plotted marginal median agree."
        )
    elif classification == "inconclusive":
        notes.append(
            f"Sub-{EVENT_WALL_DECISION_THRESHOLD_PCT}% directional classification: "
            "inconclusive; only the point estimate and interval are reported."
        )
    else:
        check_equal(
            classification,
            "not_applicable",
            "sub-half-percent directional classification",
        )
    return notes


def without_stale_measurements(result: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(result)
    for field in TIMING_FIELDS | {"input_seed", "notes", "paired_protocol"}:
        updated.pop(field, None)
    return updated


def build_target_results(
    baseline_payload: dict[str, Any],
    strict_path: Path,
    strict: dict[str, Any],
    manifest_row: dict[str, str],
    strict_manifest_sha256: str,
    heldout_manifest_sha256: str,
    generalization_manifest_sha256: str,
    paired: dict[str, Any],
    raw: dict[str, Any],
    raw_artifact_label: str,
    raw_artifact_sha256: str,
) -> dict[str, dict[str, Any]]:
    targets = {
        result["impl"]: result
        for result in baseline_payload["results"]
        if result.get("impl") in TARGET_IMPLS
    }
    provenance, source_sha256, paired_provenance = validate_strict_result(
        strict_path, strict, paired
    )
    flop_model = paired.get("flop_model")
    require(isinstance(flop_model, dict), "paired result is missing FLOP model")
    check_equal(flop_model.get("name"), FLOP_MODEL_NAME, "paired FLOP model name")
    check_equal(
        flop_model.get("formula"), FLOP_MODEL_FORMULA, "paired FLOP model formula"
    )
    flops = flop_model.get("flops")
    require(
        isinstance(flops, (int, float))
        and not isinstance(flops, bool)
        and math.isfinite(flops)
        and flops > 0,
        "paired result has invalid FLOP count",
    )
    check_equal(
        flops,
        expected_attention_flops(paired["shape"]),
        "paired FLOP model count",
    )
    event = paired.get("summary", {}).get("event")
    require(isinstance(event, dict), "paired result is missing event summary")
    helion_stats = validate_summary_stats(
        event.get("helion"), flops=flops, context="Helion event"
    )
    sdpa_stats = validate_summary_stats(
        event.get("sdpa"), flops=flops, context="SDPA event"
    )
    protocol = paired_protocol(raw, paired, raw_artifact_label, raw_artifact_sha256)
    check_equal(
        len(helion_stats["runs_ms"]),
        protocol["combined_pairs"],
        "Helion event call count",
    )
    check_equal(
        len(sdpa_stats["runs_ms"]),
        protocol["combined_pairs"],
        "SDPA event call count",
    )
    ratio = event.get("paired_log_ratio_pct")
    ci = event.get("paired_log_ratio_stratified_bootstrap_95_ci_pct")
    require(
        isinstance(ratio, (int, float))
        and isinstance(ci, list)
        and len(ci) == 2
        and all(isinstance(value, (int, float)) for value in ci),
        "paired event ratio is invalid",
    )
    directional = protocol["event_wall_validation"]["sub_half_percent_directional"]

    helion = without_stale_measurements(strict)
    helion.update(timing_fields(helion_stats))
    helion["benchmark_timer"] = "event"
    helion["paired_protocol"] = copy.deepcopy(protocol)
    helion["full_autotune_provenance"] = {
        "strict_manifest_sha256": strict_manifest_sha256,
        "heldout_manifest_sha256": heldout_manifest_sha256,
        "generalization_manifest_sha256": generalization_manifest_sha256,
        "strict_result_sha256": file_sha256(strict_path),
        "source_ledger_sha256": manifest_row["source_ledger_sha256"],
        "search_input_seed": strict["input_seed"],
        "selected_config_sha256": paired_provenance["selected_config_sha256"],
        "selected_source_sha256": source_sha256,
        "selected_source_was_measured": True,
        "compiler_seed_policy": copy.deepcopy(provenance["compiler_seed_policy"]),
        "compiler_seed_policy_sha256": canonical_sha256(
            provenance["compiler_seed_policy"]
        ),
        "winner_is_structural_coverage_design_member": provenance[
            "selected_config_is_structural_coverage_design_member"
        ],
        "winner_to_structural_coverage_field_distance": provenance[
            "selected_config_nearest_structural_coverage_design_field_distance"
        ],
    }
    helion["notes"] = [
        (
            "Winner selected by a cold, uncapped full-effort autotune with no fixed "
            "config, user seed config, compiler default, or active value prior; "
            "all canonical CuTe-flash compiler seeds were measured in generation "
            "zero."
        ),
        (
            f"Plotted value is the median of {protocol['combined_pairs']} CUDA-event "
            "calls from two independent balanced randomized paired campaigns "
            "against forced cuDNN SDPA."
        ),
        *paired_estimate_notes(ratio, ci, directional),
        "Both campaigns passed full-output correctness and exact repeatability.",
    ]

    sdpa = without_stale_measurements(targets["sdpa"])
    environment = paired["environment"]
    torch_version = environment.get("torch_version")
    cudnn_version = environment.get("cudnn_version")
    require(
        isinstance(torch_version, str)
        and f"PyTorch {torch_version};" in sdpa["version"],
        "baseline SDPA PyTorch version does not match paired environment",
    )
    require(
        cudnn_version == 92000 and "cuDNN runtime 9.20.0" in sdpa["version"],
        "baseline SDPA cuDNN version does not match paired environment",
    )
    sdpa.update(timing_fields(sdpa_stats))
    sdpa["benchmark_timer"] = "event"
    sdpa["paired_protocol"] = copy.deepcopy(protocol)
    sdpa["notes"] = [
        "Forced torch SDPBackend.CUDNN_ATTENTION.",
        (
            f"Plotted value is the median of the same {protocol['combined_pairs']} "
            "raw CUDA-event calls used in the paired Helion campaigns."
        ),
    ]
    return {"helion-cute": helion, "sdpa": sdpa}


def non_target_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        result
        for result in payload["results"]
        if isinstance(result, dict) and result.get("impl") not in TARGET_IMPLS
    ]


def validate_non_target_results(
    original: dict[str, Any], updated: dict[str, Any], *, context: str
) -> None:
    check_equal(
        non_target_results(updated),
        non_target_results(original),
        f"{context}: non-target baseline results",
    )
    original_without_results = {
        key: value for key, value in original.items() if key != "results"
    }
    updated_without_results = {
        key: value for key, value in updated.items() if key != "results"
    }
    check_equal(
        updated_without_results,
        original_without_results,
        f"{context}: payload metadata",
    )


def validate_cute_backend_versions(payload: dict[str, Any], *, context: str) -> None:
    expected = build_strict_manifest.EXPECTED_CUTE_VERSION
    for result in payload["results"]:
        if not isinstance(result, dict):
            continue
        impl = result.get("impl")
        version = result.get("version")
        version_label = result.get("version_label")
        declares_cute = (
            impl in CUTE_BACKED_IMPLS
            or (isinstance(version, str) and "CuTe " in version)
            or (isinstance(version_label, str) and "CuTe " in version_label)
        )
        if not declares_cute:
            continue
        if result.get("accuracy") != "PASS":
            continue
        require(
            isinstance(version, str),
            f"{context}: {impl} is missing a version",
        )
        matches = re.findall(r"(?:^|; )CuTe ([^;\s]+)", version)
        check_equal(
            matches,
            [expected],
            f"{context}: {impl} CuTe version",
        )
        require(
            isinstance(version_label, str),
            f"{context}: {impl} is missing a version label",
        )
        check_equal(
            re.findall(r"(?:^|; | / )CuTe ([^;/\s]+)", version_label),
            [expected],
            f"{context}: {impl} CuTe version label",
        )


def replace_targets(
    original: dict[str, Any], replacements: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    updated = copy.deepcopy(original)
    updated["results"] = [
        copy.deepcopy(replacements[result["impl"]])
        if result.get("impl") in TARGET_IMPLS
        else result
        for result in updated["results"]
    ]
    validate_non_target_results(original, updated, context="in-memory publication")
    return updated


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def allocate_sibling_path(path: Path, *, kind: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.publish-{kind}-",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    allocated = Path(name)
    allocated.unlink()
    return allocated


def prepare_staging_paths(
    destinations: list[Path], *, overwrite: bool
) -> tuple[dict[Path, Path], list[Path]]:
    resolved = [path.resolve() for path in destinations]
    require(
        len(set(resolved)) == len(resolved),
        "publication destinations must be distinct",
    )
    existing = [path for path in destinations if path.exists()]
    require(
        overwrite or not existing,
        f"outputs already exist (use --overwrite): {existing}",
    )

    created_directories: list[Path] = []
    known_created: set[Path] = set()
    staged: dict[Path, Path] = {}
    try:
        for destination in destinations:
            missing: list[Path] = []
            parent = destination.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            destination.parent.mkdir(parents=True, exist_ok=True)
            for directory in reversed(missing):
                if directory not in known_created:
                    created_directories.append(directory)
                    known_created.add(directory)
            staged[destination] = allocate_sibling_path(destination, kind="stage")
    except BaseException:
        cleanup_staging_paths(staged.values(), created_directories)
        raise
    return staged, created_directories


def cleanup_staging_paths(
    paths: Iterable[Path], created_directories: list[Path]
) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
    for directory in reversed(created_directories):
        with contextlib.suppress(OSError):
            directory.rmdir()


def commit_staged_outputs(staged: dict[Path, Path], *, overwrite: bool) -> None:
    missing = [path for path in staged.values() if not path.is_file()]
    require(not missing, f"staged outputs are missing: {missing}")
    existing = [path for path in staged if path.exists()]
    require(
        overwrite or not existing,
        f"outputs already exist (use --overwrite): {existing}",
    )

    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination in existing:
            backup = allocate_sibling_path(destination, kind="backup")
            backups[destination] = backup
            shutil.copy2(destination, backup, follow_symlinks=False)
        for destination, staged_path in staged.items():
            staged_path.replace(destination)
            installed.append(destination)
    except BaseException:
        rollback_errors: list[str] = []
        for destination in reversed(installed):
            try:
                backup = backups.get(destination)
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    backup.replace(destination)
            except OSError as exc:
                rollback_errors.append(f"{destination}: {exc}")
        if rollback_errors:
            raise RuntimeError(
                "publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from None
        raise
    finally:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def renderer_command(
    renderer: Path,
    payload_paths: list[Path],
    csv_output: Path,
    plot_output: Path,
    summary_plot_output: Path,
    labels: list[str],
) -> list[str]:
    command = [
        sys.executable,
        str(renderer.resolve()),
        "--merge-json",
        *(str(path.resolve()) for path in payload_paths),
        "--csv-output",
        str(csv_output.resolve()),
        "--plot-output",
        str(plot_output.resolve()),
        "--summary-plot-output",
        str(summary_plot_output.resolve()),
    ]
    for label in labels:
        command.extend(("--plot-impl-label", label))
    return command


def validate_publication_destinations(
    args: argparse.Namespace,
    destinations: list[Path],
) -> None:
    evidence_files = publication_evidence_paths(args, include_renderer=True)
    evidence_roots = {
        args.strict_artifact_root.resolve(),
        args.heldout_artifact_root.resolve(),
        args.generalization_artifact_root.resolve(),
        args.run_manifest.resolve().parent,
        args.baseline_payload_dir.resolve(),
    }
    for destination in destinations:
        resolved = destination.resolve()
        require(
            resolved not in evidence_files,
            f"publication destination aliases evidence input: {destination}",
        )
        containing_roots = [
            root for root in evidence_roots if resolved.is_relative_to(root)
        ]
        require(
            not containing_roots,
            f"publication destination is inside an evidence tree: {destination} "
            f"under {containing_roots}",
        )


def publish(
    args: argparse.Namespace,
    *,
    render_outputs: tuple[Path, Path, Path] | None = None,
) -> list[Path]:
    require(
        not Path(args.raw_artifact_label).is_absolute()
        and ".." not in Path(args.raw_artifact_label).parts,
        "--raw-artifact-label must be a repository-relative logical path",
    )
    require(
        args.baseline_payload_dir.resolve() != args.output_payload_dir.resolve(),
        "baseline and output payload directories must be distinct",
    )
    evidence_snapshot = snapshot_publication_evidence(
        args, include_renderer=render_outputs is not None
    )
    baseline = index_baseline_payloads(args.baseline_payload_dir)
    validate_regenerated_strict_manifest(
        args.strict_manifest, args.strict_artifact_root
    )
    strict = index_strict_results(args.strict_artifact_root)
    strict_manifest = index_strict_manifest(
        args.strict_manifest, args.strict_artifact_root, strict
    )
    strict_manifest_sha256 = file_sha256(args.strict_manifest)
    validate_regenerated_heldout_manifest(
        args.heldout_manifest,
        args.heldout_artifact_root,
        args.strict_artifact_root,
        args.strict_manifest,
        strict_manifest,
    )
    heldout_manifest_sha256 = file_sha256(args.heldout_manifest)
    validate_regenerated_generalization_manifest(
        args.generalization_manifest, args.generalization_artifact_root
    )
    generalization_manifest_sha256 = file_sha256(args.generalization_manifest)
    raw = load_object(args.paired_raw)
    paired = index_paired_results(raw)
    validate_paired_artifacts(
        raw,
        paired,
        strict_manifest,
        args.run_manifest,
        args.static_validation,
    )
    raw_artifact_sha256 = file_sha256(args.paired_raw)

    versions: set[str] = set()
    search_input_seeds: set[int] = set()
    pending: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for key in sorted(EXPECTED_SHAPES, key=operator.itemgetter(5, 2)):
        baseline_path, original = baseline[key]
        strict_path, strict_payload = strict[key]
        replacements = build_target_results(
            original,
            strict_path,
            strict_payload,
            strict_manifest[key],
            strict_manifest_sha256,
            heldout_manifest_sha256,
            generalization_manifest_sha256,
            paired[key],
            raw,
            args.raw_artifact_label,
            raw_artifact_sha256,
        )
        versions.add(replacements["helion-cute"]["version"])
        search_input_seeds.add(
            replacements["helion-cute"]["full_autotune_provenance"]["search_input_seed"]
        )
        updated = replace_targets(original, replacements)
        validate_cute_backend_versions(updated, context=str(baseline_path))
        output_path = args.output_payload_dir / baseline_path.name
        pending.append((output_path, original, updated))
    require(len(versions) == 1, f"strict results mix Helion/CuTe versions: {versions}")
    require(
        len(search_input_seeds) == 1,
        f"strict results mix search input seeds: {search_input_seeds}",
    )
    output_paths = [path for path, _, _ in pending]
    destinations = list(output_paths)
    if render_outputs is not None:
        destinations.extend(render_outputs)
    validate_publication_destinations(args, destinations)
    staged, created_directories = prepare_staging_paths(
        destinations, overwrite=args.overwrite
    )
    try:
        for output_path, original, updated in pending:
            staged_path = staged[output_path]
            atomic_write_json(staged_path, updated)
            reloaded = load_object(staged_path)
            validate_non_target_results(original, reloaded, context=str(output_path))
        if render_outputs is not None:
            csv_output, plot_output, summary_plot_output = render_outputs
            command = renderer_command(
                args.renderer,
                [staged[path] for path in output_paths],
                staged[csv_output],
                staged[plot_output],
                staged[summary_plot_output],
                args.plot_impl_label,
            )
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            for output in render_outputs:
                rendered = staged[output]
                require(
                    rendered.is_file() and rendered.stat().st_size > 0,
                    f"renderer did not produce a nonempty output for {output}",
                )
        validate_publication_evidence_unchanged(
            args,
            evidence_snapshot,
            include_renderer=render_outputs is not None,
        )
        commit_staged_outputs(staged, overwrite=args.overwrite)
    finally:
        cleanup_staging_paths(staged.values(), created_directories)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish generalized paired Helion+CuTe/SDPA measurements into copies "
            "of the eight existing all-backend payloads."
        )
    )
    parser.add_argument("--paired-raw", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--static-validation", type=Path, required=True)
    parser.add_argument("--strict-artifact-root", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--heldout-artifact-root", type=Path, required=True)
    parser.add_argument("--heldout-manifest", type=Path, required=True)
    parser.add_argument("--generalization-artifact-root", type=Path, required=True)
    parser.add_argument("--generalization-manifest", type=Path, required=True)
    parser.add_argument("--baseline-payload-dir", type=Path, required=True)
    parser.add_argument("--output-payload-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-artifact-label",
        default="plots/generalized_full_autotune/all8_paired_raw.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--renderer", type=Path, default=DEFAULT_RENDERER)
    parser.add_argument("--render-csv", type=Path)
    parser.add_argument("--render-plot", type=Path)
    parser.add_argument("--render-summary-plot", type=Path)
    parser.add_argument("--plot-impl-label", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_outputs = (args.render_csv, args.render_plot, args.render_summary_plot)
    require(
        all(output is None for output in render_outputs)
        or all(output is not None for output in render_outputs),
        "specify all three render outputs or none of them",
    )
    render_outputs = (
        (args.render_csv, args.render_plot, args.render_summary_plot)
        if args.render_csv is not None
        else None
    )
    output_paths = publish(args, render_outputs=render_outputs)
    print(
        json.dumps(
            {
                "status": "PASS",
                "payloads": [str(path) for path in output_paths],
                "non_target_baselines_preserved": True,
                "rendered": args.render_csv is not None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
