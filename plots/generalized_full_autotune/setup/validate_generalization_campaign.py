from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any

import build_strict_manifest as strict

SCHEMA_VERSION = 3
TUNER_SEED_FIELDS = tuple(f"tuner_seed_{index}" for index in range(1, 6))
CASE_FIELDS = (
    "schema_version",
    "case_id",
    "z",
    "h",
    "seq_len",
    "head_dim",
    "dtype",
    "causal",
    "legality_class",
    "input_seed",
    *TUNER_SEED_FIELDS,
)
RESULT_FILENAMES = frozenset(
    {"result.json", "autotune.csv", "autotune.meta.jsonl", "autotune.sources.csv"}
)
EXPECTED_GENERATIONS = 20
EXPECTED_RUNS = 9
EXPECTED_CASE_COUNT = 15
EXPECTED_CAUSAL_CASE_COUNT = 7
EXPECTED_BROAD_RUN_COUNT = 65
EXPECTED_COMMIT = "c3e36b65d69681c23e053042b0bc21e2331bad17"
EXPECTED_QUALIFICATION_PHASE = "cute_flash_structural_qualification_v22"
EXPECTED_LANE_POLICY_VERSION = 14
ALLOWED_TUNER_SEED_COUNTS = frozenset({3, 5})
TARGETED_PIPELINE_LANE_CASES = frozenset(
    {
        "causal_lane_paired_d64",
        "causal_lane_paired_d64_twin",
    }
)
TARGETED_CLC_ANCHOR_VALUES = {
    "dense_clc_bh96_paired_d64": frozenset({1, 2, 3, 6, 12, 32, 48, 96}),
    "dense_clc_bh120_paired_d64": frozenset({1, 2, 3, 5, 10, 24, 60, 120}),
    "dense_clc_bh180_paired_d64": frozenset({1, 2, 5, 12, 20, 36, 90, 180}),
}
TARGETED_CLC_LEGAL_VALUES = {
    "dense_clc_bh96_paired_d64": frozenset({1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96}),
    "dense_clc_bh120_paired_d64": frozenset(
        {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120}
    ),
    "dense_clc_bh180_paired_d64": frozenset(
        {1, 2, 3, 4, 5, 6, 9, 10, 12, 15, 18, 20, 30, 36, 45, 60, 90, 180}
    ),
}
MINIMAL_CROSS_SHAPE_CASES = (
    (
        "dense_b4_h16_s16384_d64_fp16",
        4,
        16,
        16384,
        64,
        "float16",
        False,
        "div4",
        2026082001,
        (20260820101, 20260820102, 20260820103),
    ),
    (
        "causal_b4_h16_s16384_d64_fp16",
        4,
        16,
        16384,
        64,
        "float16",
        True,
        "div4",
        2026082002,
        (20260820201, 20260820202, 20260820203),
    ),
    (
        "dense_b3_h32_s16384_d64_bf16",
        3,
        32,
        16384,
        64,
        "bfloat16",
        False,
        "div4",
        2026082003,
        (20260820301, 20260820302, 20260820303),
    ),
    (
        "causal_b3_h32_s16384_d64_bf16",
        3,
        32,
        16384,
        64,
        "bfloat16",
        True,
        "div4",
        2026082004,
        (20260820401, 20260820402, 20260820403),
    ),
    (
        "dense_b3_h32_s16384_d128_fp16",
        3,
        32,
        16384,
        128,
        "float16",
        False,
        "div4",
        2026082005,
        (20260820501, 20260820502, 20260820503),
    ),
    (
        "causal_b3_h32_s16384_d128_fp16",
        3,
        32,
        16384,
        128,
        "float16",
        True,
        "div4",
        2026082006,
        (20260820601, 20260820602, 20260820603),
    ),
)
MINIMAL_CROSS_SHAPE_CASE_IDS = frozenset(case[0] for case in MINIMAL_CROSS_SHAPE_CASES)
EXPECTED_D64_ALIASED_KV_LANES = frozenset(range(2, 13))
EXPECTED_ORDINARY_CLC_FAMILIES = frozenset(
    {
        "fa4_clc",
        "fa4_clc_tma_4d",
        "fa4_clc_local_tma",
        "fa4_clc_local_tma_4d",
    }
)
BENCHMARK_RELATIVE = Path("benchmarks/cute/compare_attention_backends.py")
REMEASUREMENT_RELATIVE = Path(
    "plots/generalized_full_autotune/setup/remeasure_generalization_winners.py"
)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    z: int
    h: int
    seq_len: int
    head_dim: int
    dtype: str
    causal: bool
    legality_class: str
    input_seed: int
    tuner_seeds: tuple[int, ...]

    @property
    def physical_gpu(self) -> int:
        return 6 if self.causal else 7

    @property
    def torch_dtype(self) -> str:
        return f"torch.{self.dtype}"

    @property
    def surface_key(self) -> tuple[object, ...]:
        return (
            self.dtype,
            self.head_dim,
            self.causal,
            self.legality_class,
            self.z,
            self.h,
        )


@dataclass(frozen=True)
class RunSpec:
    case: CaseSpec
    tuner_seed: int
    run_id: str
    result_path: Path


@dataclass(frozen=True)
class ValidatedRun:
    run: RunSpec
    median_tflops: float
    num_configs_tested: int
    num_successful_candidate_measurements: int
    num_source_deduplications: int
    num_isolated_rebenchmark_timeouts: int
    num_generations: int
    exact_effective_search_space_size: int | None
    coverage_design_sha256: str
    compiler_seed_policy_sha256: str
    terminal_refinement_policy_sha256: str
    terminal_coordinate_surface_sha256: str
    terminal_refinement_sha256: str
    result_sha256: str
    autotune_csv_sha256: str
    autotune_metadata_sha256: str
    source_ledger_sha256: str
    selected_source_sha256: str


@dataclass(frozen=True)
class ValidatedRemeasurement:
    case: CaseSpec
    path: Path
    sha256: str
    selected_source_sha256s: tuple[str, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class CampaignValidation:
    cases: tuple[CaseSpec, ...]
    run_specs: tuple[RunSpec, ...]
    runs: tuple[ValidatedRun, ...]
    remeasurements: tuple[ValidatedRemeasurement, ...]


def _case_contract(case: CaseSpec) -> tuple[object, ...]:
    return (
        case.case_id,
        case.z,
        case.h,
        case.seq_len,
        case.head_dim,
        case.dtype,
        case.causal,
        case.legality_class,
        case.input_seed,
        case.tuner_seeds,
    )


def _validate_minimal_cross_shape_matrix(cases: list[CaseSpec]) -> None:
    strict.check_equal(
        tuple(_case_contract(case) for case in cases),
        MINIMAL_CROSS_SHAPE_CASES,
        "minimal cross-shape case matrix",
    )


def _validate_broad_generalization_matrix(cases: list[CaseSpec]) -> None:
    strict.check_equal(
        {case.legality_class for case in cases},
        {"singleton", "odd", "paired", "div4"},
        "legality-class coverage",
    )
    strict.check_equal(
        {case.causal for case in cases}, {False, True}, "dense/causal coverage"
    )
    strict.check_equal(len(cases), EXPECTED_CASE_COUNT, "generalization case count")
    strict.check_equal(
        sum(case.causal for case in cases),
        EXPECTED_CAUSAL_CASE_COUNT,
        "balanced dense/causal case count",
    )
    strict.check_equal(
        sum(len(case.tuner_seeds) for case in cases),
        EXPECTED_BROAD_RUN_COUNT,
        "broad generalization search count",
    )
    strict.check_equal(
        {case.dtype for case in cases}, {"float16", "bfloat16"}, "dtype coverage"
    )
    strict.check_equal(
        {case.head_dim for case in cases}, {64, 128}, "head-dimension coverage"
    )
    surface_counts: dict[tuple[object, ...], int] = {}
    for case in cases:
        surface_counts[case.surface_key] = surface_counts.get(case.surface_key, 0) + 1
    strict.check_equal(
        sorted(count for count in surface_counts.values() if count > 1),
        [2, 2, 2],
        "length-only comparison pairs",
    )
    by_id = {case.case_id: case for case in cases}
    targeted_cases = set(TARGETED_PIPELINE_LANE_CASES) | set(TARGETED_CLC_ANCHOR_VALUES)
    strict.check_equal(
        targeted_cases,
        targeted_cases & set(by_id),
        "targeted coverage cases",
    )
    for case_id in targeted_cases:
        strict.check_equal(
            len(by_id[case_id].tuner_seeds), 3, f"{case_id}: targeted seed count"
        )


def _csv_int(value: str, label: str, *, minimum: int = 1) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise RuntimeError(f"{label}: expected a positive integer, got {value!r}")
    result = int(value)
    strict.require(result >= minimum, f"{label}: expected at least {minimum}")
    return result


def legality_class(seq_len: int) -> str:
    strict.require(
        seq_len % 128 == 0,
        f"sequence length {seq_len} does not exercise the 128-tile flash path",
    )
    num_kv = seq_len // 128
    if num_kv == 1:
        return "singleton"
    if num_kv % 2:
        return "odd"
    if num_kv % 4:
        return "paired"
    return "div4"


def parse_case_matrix(path: Path) -> tuple[CaseSpec, ...]:
    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            strict.check_equal(
                tuple(reader.fieldnames or ()), CASE_FIELDS, "case fields"
            )
            raw_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise RuntimeError(f"unable to read case matrix {path}: {exc}") from exc
    strict.require(raw_rows, f"case matrix is empty: {path}")

    cases: list[CaseSpec] = []
    all_seeds: set[int] = set()
    for line, row in enumerate(raw_rows, 2):
        context = f"{path}:{line}"
        strict.check_equal(row["schema_version"], str(SCHEMA_VERSION), context)
        case_id = row["case_id"]
        strict.require(
            re.fullmatch(r"[a-z0-9][a-z0-9_]*", case_id) is not None,
            f"{context}: invalid case_id {case_id!r}",
        )
        dtype = row["dtype"]
        strict.require(
            dtype in {"float16", "bfloat16"}, f"{context}: unsupported dtype {dtype!r}"
        )
        head_dim = _csv_int(row["head_dim"], f"{context}: head_dim")
        strict.require(
            head_dim in {64, 128}, f"{context}: unsupported head_dim {head_dim}"
        )
        causal_value = row["causal"]
        strict.require(causal_value in {"0", "1"}, f"{context}: invalid causal flag")
        seq_len = _csv_int(row["seq_len"], f"{context}: seq_len")
        expected_class = legality_class(seq_len)
        strict.check_equal(
            row["legality_class"], expected_class, f"{context}: legality class"
        )
        raw_seeds = tuple(row[field] for field in TUNER_SEED_FIELDS)
        first_empty = next(
            (index for index, value in enumerate(raw_seeds) if not value),
            len(raw_seeds),
        )
        strict.require(
            not any(raw_seeds[first_empty:]),
            f"{context}: tuner seed columns must be a contiguous prefix",
        )
        seeds = tuple(
            _csv_int(value, f"{context}: {TUNER_SEED_FIELDS[index]}")
            for index, value in enumerate(raw_seeds[:first_empty])
        )
        strict.require(
            len(seeds) in ALLOWED_TUNER_SEED_COUNTS,
            f"{context}: expected three or five tuner seeds",
        )
        strict.require(
            len(set(seeds)) == len(seeds), f"{context}: tuner seeds are not unique"
        )
        strict.require(
            not (set(seeds) & all_seeds), f"{context}: tuner seed reused across cases"
        )
        all_seeds.update(seeds)
        cases.append(
            CaseSpec(
                case_id=case_id,
                z=_csv_int(row["z"], f"{context}: z"),
                h=_csv_int(row["h"], f"{context}: h"),
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=dtype,
                causal=causal_value == "1",
                legality_class=expected_class,
                input_seed=_csv_int(row["input_seed"], f"{context}: input_seed"),
                tuner_seeds=seeds,
            )
        )
    strict.require(
        len({case.case_id for case in cases}) == len(cases),
        "duplicate case_id in matrix",
    )
    case_ids = {case.case_id for case in cases}
    if case_ids == MINIMAL_CROSS_SHAPE_CASE_IDS:
        _validate_minimal_cross_shape_matrix(cases)
    else:
        _validate_broad_generalization_matrix(cases)
    return tuple(cases)


def expand_runs(cases: tuple[CaseSpec, ...]) -> tuple[RunSpec, ...]:
    result: list[RunSpec] = []
    for case in cases:
        for seed in case.tuner_seeds:
            identity = {
                "case_id": case.case_id,
                "tuner_seed": seed,
                "shape": [case.z, case.h, case.seq_len, case.head_dim],
                "dtype": case.dtype,
                "causal": case.causal,
            }
            run_id = strict.canonical_sha256(identity)[:16]
            result.append(
                RunSpec(
                    case=case,
                    tuner_seed=seed,
                    run_id=run_id,
                    result_path=Path("results")
                    / case.case_id
                    / f"seed_{seed}"
                    / "result.json",
                )
            )
    strict.require(
        len({run.run_id for run in result}) == len(result), "run ID collision"
    )
    return tuple(result)


def expected_command(
    python_executable: Path,
    repo: Path,
    root: Path,
    run: RunSpec,
) -> list[str]:
    case = run.case
    output = (root / run.result_path).parent
    return [
        str(python_executable),
        str(repo / BENCHMARK_RELATIVE),
        "--impl",
        "helion-cute",
        "--z",
        str(case.z),
        "--h",
        str(case.h),
        "--seq-len",
        str(case.seq_len),
        "--head-dim",
        str(case.head_dim),
        "--dtype",
        case.dtype,
        "--causal",
        str(int(case.causal)),
        "--num-runs",
        "9",
        "--warmup-ms",
        "1000",
        "--rep-ms",
        "500",
        "--seed",
        str(case.input_seed),
        "--power-cap-w",
        "750",
        "--helion-force-flash-config",
        "0",
        "--helion-force-autotune",
        "1",
        "--helion-require-full-autotune",
        "1",
        "--helion-return-lse",
        "0",
        "--helion-cute-benchmark-timer",
        "wall",
        "--helion-autotune-effort",
        "full",
        "--helion-autotune-best-of-k",
        "1",
        "--helion-autotune-benchmark-timeout",
        "60",
        "--helion-autotune-accuracy-check",
        "1",
        "--helion-autotuner-initial-population",
        "from_random",
        "--helion-env",
        f"HELION_AUTOTUNE_RANDOM_SEED={run.tuner_seed}",
        "--json",
        "--json-output",
        str(output / "result.json"),
    ]


def expected_remeasurement_command(
    python_executable: Path,
    repo: Path,
    root: Path,
    case: CaseSpec,
    worker_sha256: str,
) -> list[str]:
    return [
        str(python_executable),
        str(root / "launcher" / REMEASUREMENT_RELATIVE.name),
        "--repo",
        str(repo),
        "--artifact-root",
        str(root),
        "--case-id",
        case.case_id,
        "--physical-gpu",
        str(case.physical_gpu),
        "--protocol-seed",
        str(case.input_seed ^ 0xC205),
        "--rounds",
        "12",
        "--warmup-calls",
        "3",
        "--thermal-warmup-seconds",
        "10",
        "--repeatability-launches",
        "3",
        "--target-timing-ms",
        "20",
        "--max-timing-repetitions",
        "4096",
        "--bootstrap-samples",
        "20000",
        "--expected-worker-sha256",
        worker_sha256,
        "--output",
        str(root / "remeasure" / f"{case.case_id}.json"),
    ]


def load_campaign_records(root: Path) -> list[dict[str, Any]]:
    path = root / "campaign.jsonl"
    strict.require(
        path.is_file() and not path.is_symlink(), f"missing campaign: {path}"
    )
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        strict.require(isinstance(record, dict), f"{path}:{line_number}: not an object")
        records.append(record)
    strict.require(records, f"empty campaign: {path}")
    return records


def _declared_records(
    records: list[dict[str, Any]], record_type: str
) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
    saw_event = False
    for record in records:
        kind = record.get("record_type")
        if kind == "event":
            saw_event = True
        elif kind in {"run", "remeasure"}:
            strict.require(
                not saw_event, "run declaration appears after campaign events"
            )
            if kind == record_type:
                declarations.append(record)
        elif kind == "campaign":
            strict.require(
                not declarations and not saw_event, "campaign header is misplaced"
            )
        else:
            raise RuntimeError(f"unknown campaign record type: {kind!r}")
    return declarations


def declared_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _declared_records(records, "run")


def declared_remeasurements(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _declared_records(records, "remeasure")


def validate_result_set(root: Path, runs: tuple[RunSpec, ...]) -> None:
    expected = {root / run.result_path for run in runs}
    discovered = set(root.rglob("result.json"))
    strict.check_equal(discovered, expected, "campaign result set")
    for path in expected:
        strict.require(not path.is_symlink(), f"result is a symlink: {path}")
        invalid = [
            name
            for name in RESULT_FILENAMES
            if not path.with_name(name).is_file() or path.with_name(name).is_symlink()
        ]
        strict.require(
            not invalid, f"{path}: missing or symlinked sidecars {sorted(invalid)}"
        )


def validate_payload_identity(
    path: Path,
    payload: dict[str, Any],
    run: RunSpec,
    expected_commit: str,
) -> tuple[float, float]:
    case = run.case
    strict.check_equal(payload.get("impl"), "helion-cute", f"{path}: implementation")
    strict.check_equal(payload.get("gpu"), "NVIDIA B200", f"{path}: GPU")
    strict.check_equal(
        str(payload.get("physical_gpu")),
        str(case.physical_gpu),
        f"{path}: physical GPU",
    )
    strict.check_equal(
        strict.finite_float(payload.get("power_cap_w"), f"{path}: power cap"),
        750.0,
        f"{path}: power cap",
    )
    version = payload.get("version")
    strict.require(isinstance(version, str), f"{path}: missing version")
    strict.require(".dirty" not in version, f"{path}: dirty version")
    strict.require(
        strict.helion_cute_version_matches_commit(version, expected_commit),
        f"{path}: version {version!r} does not identify {expected_commit}",
    )
    strict.check_equal(
        payload.get("input_seed"), case.input_seed, f"{path}: input seed"
    )
    expected_shape = {
        "z": case.z,
        "h": case.h,
        "seq_len": case.seq_len,
        "head_dim": case.head_dim,
        "dtype": case.dtype,
        "causal": int(case.causal),
        "biased": 0,
    }
    strict.check_equal(payload.get("shape"), expected_shape, f"{path}: shape")
    strict.check_equal(payload.get("accuracy"), "PASS", f"{path}: accuracy")
    strict.check_equal(payload.get("benchmark_timer"), "wall", f"{path}: timer")
    strict.check_equal(
        payload.get("flop_model"), "softmax_attention_forward", f"{path}: flop model"
    )
    runs_ms = payload.get("runs_ms")
    strict.require(
        isinstance(runs_ms, list) and len(runs_ms) == EXPECTED_RUNS,
        f"{path}: expected {EXPECTED_RUNS} timing runs",
    )
    parsed_runs = [
        strict.finite_float(value, f"{path}: runs_ms[{index}]", positive=True)
        for index, value in enumerate(runs_ms)
    ]
    median_ms = strict.finite_float(
        payload.get("median_ms"), f"{path}: median_ms", positive=True
    )
    strict.require(
        math.isclose(median_ms, statistics.median(parsed_runs), rel_tol=1e-12),
        f"{path}: median_ms does not match runs_ms",
    )
    median_tflops = strict.finite_float(
        payload.get("median_tflops"), f"{path}: median_tflops", positive=True
    )
    flops = 4.0 * case.z * case.h * case.seq_len**2 * case.head_dim
    if case.causal:
        flops *= 0.5
    strict.require(
        math.isclose(median_tflops, flops / (median_ms * 1e9), rel_tol=1e-12),
        f"{path}: median_tflops does not match shape and latency",
    )
    return median_ms, median_tflops


def _validate_provenance(
    path: Path, payload: dict[str, Any], run: RunSpec
) -> tuple[dict[str, Any], dict[str, Any]]:
    case = run.case
    overrides = payload.get("helion_overrides")
    strict.require(isinstance(overrides, dict), f"{path}: missing helion overrides")
    provenance = overrides.get("autotune_provenance")
    strict.require(isinstance(provenance, dict), f"{path}: missing provenance")
    trials = provenance.get("trials")
    strict.require(
        isinstance(trials, list) and len(trials) == 1 and isinstance(trials[0], dict),
        f"{path}: expected one trial",
    )
    trial = trials[0]
    shape = (case.z, case.h, case.seq_len, case.head_dim)
    strict.check_equal(
        trial.get("input_shapes"), repr([shape, shape, shape]), f"{path}: trial shapes"
    )
    strict.check_equal(
        trial.get("dtypes"), repr([case.torch_dtype] * 3), f"{path}: trial dtypes"
    )
    phase = trial.get("search_phase_metrics")
    strict.require(isinstance(phase, dict), f"{path}: missing qualification phase")
    strict.check_equal(
        phase.get("phase"),
        EXPECTED_QUALIFICATION_PHASE,
        f"{path}: qualification phase",
    )
    strict.check_equal(
        phase.get("cute_flash_lane_policy_version"),
        EXPECTED_LANE_POLICY_VERSION,
        f"{path}: lane policy",
    )
    validate_live_starting_path_capacity(path, provenance, phase)
    strict.validate_strict_provenance(
        path,
        payload,
        "causal" if case.causal else "dense",
        case.seq_len,
        run.tuner_seed,
        expected_input_shape=shape,
        expected_input_dtype=case.torch_dtype,
    )
    exact_config_ids = strict.exact_effective_search_space_ids(path, provenance)
    generations = strict.strict_int(
        trial.get("num_generations"), f"{path}: generations", minimum=0
    )
    if exact_config_ids is None:
        strict.check_equal(generations, EXPECTED_GENERATIONS, f"{path}: generations")
    else:
        strict.require(
            generations <= EXPECTED_GENERATIONS,
            f"{path}: exhausted search exceeded generation budget",
        )
    strict.check_equal(
        provenance.get("autotune_lfbo_max_generations"),
        EXPECTED_GENERATIONS,
        f"{path}: LFBO generation budget",
    )
    successful = strict.strict_int(
        trial.get("num_successful_candidate_measurements"),
        f"{path}: successful measurements",
        minimum=1,
    )
    if exact_config_ids is None:
        strict.require(
            successful >= 100,
            f"{path}: fewer than 100 actual successful candidate measurements",
        )
    strict.check_equal(
        provenance.get("final_correctness_launches"), 64, f"{path}: repeat launches"
    )
    strict.check_equal(
        trial.get("selected_source_was_measured"), True, f"{path}: measured winner"
    )
    validate_targeted_coverage(path, case, provenance, trial)
    return provenance, trial


def validate_targeted_coverage(
    path: Path,
    case: CaseSpec,
    provenance: dict[str, Any],
    trial: dict[str, Any],
) -> None:
    if (
        case.case_id not in TARGETED_PIPELINE_LANE_CASES
        and case.case_id not in TARGETED_CLC_ANCHOR_VALUES
    ):
        return
    phase = trial.get("search_phase_metrics")
    strict.require(isinstance(phase, dict), f"{path}: missing qualification phase")
    strict.check_equal(
        phase.get("phase"),
        EXPECTED_QUALIFICATION_PHASE,
        f"{path}: targeted qualification phase",
    )
    strict.check_equal(
        phase.get("cute_flash_lane_policy_version"),
        EXPECTED_LANE_POLICY_VERSION,
        f"{path}: targeted lane policy",
    )
    strict.check_equal(
        phase.get("qualification_failure_retries"),
        1,
        f"{path}: targeted qualification failure retries",
    )

    active_values = provenance.get("flash_structural_coverage_active_values")
    strict.require(isinstance(active_values, list), f"{path}: active values")
    if case.case_id in TARGETED_PIPELINE_LANE_CASES:
        active_kv_lanes = {
            item.get("value")
            for item in active_values
            if isinstance(item, dict)
            and item.get("key") == "cute_flash_kv_stage"
            and type(item.get("value")) is int
        }
        strict.check_equal(
            active_kv_lanes,
            set(EXPECTED_D64_ALIASED_KV_LANES),
            f"{path}: targeted active K/V lanes",
        )
        leaf_results = phase.get("leaf_results")
        strict.require(isinstance(leaf_results, list), f"{path}: leaf results")
        recorded_lanes: set[int] = set()
        for leaf_result in leaf_results:
            strict.require(isinstance(leaf_result, dict), f"{path}: invalid leaf")
            lanes = leaf_result.get("pipeline_lanes")
            strict.require(isinstance(lanes, list), f"{path}: invalid leaf lanes")
            for lane in lanes:
                if (
                    isinstance(lane, dict)
                    and lane.get("key") == "cute_flash_kv_stage"
                    and lane.get("value") in EXPECTED_D64_ALIASED_KV_LANES
                ):
                    strict.check_equal(
                        lane.get("witness_attempted"),
                        True,
                        f"{path}: targeted lane witness",
                    )
                    strict.check_equal(
                        lane.get("complete"), True, f"{path}: targeted lane completion"
                    )
                    conditional_ids = lane.get("conditional_candidate_ids")
                    successful_conditional_ids = lane.get(
                        "successful_conditional_candidate_ids"
                    )
                    repair_ids = lane.get("repair_candidate_ids")
                    successful_repair_ids = lane.get("successful_repair_candidate_ids")
                    repair_decisions = lane.get("repair_parent_decisions")
                    conditional_required = lane.get("conditional_required")
                    strict.require(
                        isinstance(conditional_required, bool)
                        and isinstance(conditional_ids, list)
                        and isinstance(successful_conditional_ids, list)
                        and set(successful_conditional_ids) <= set(conditional_ids)
                        and isinstance(repair_ids, list)
                        and isinstance(successful_repair_ids, list)
                        and set(successful_repair_ids) <= set(repair_ids)
                        and isinstance(repair_decisions, list)
                        and len(repair_decisions) <= 1
                        and len(conditional_ids) == (1 if conditional_required else 0)
                        and (
                            lane.get("witness_succeeded") is True
                            or successful_conditional_ids
                            or successful_repair_ids
                        )
                        and (
                            conditional_required
                            or (
                                lane.get("space_exhausted") is True
                                and phase.get("exact_space_enumerated") is True
                            )
                        ),
                        f"{path}: targeted lane conditional child",
                    )
                    recorded_lanes.add(lane["value"])
        strict.check_equal(
            recorded_lanes,
            set(EXPECTED_D64_ALIASED_KV_LANES),
            f"{path}: targeted measured K/V lanes",
        )

    expected_clc_anchors = TARGETED_CLC_ANCHOR_VALUES.get(case.case_id)
    if expected_clc_anchors is None:
        return
    expected_clc_legal = TARGETED_CLC_LEGAL_VALUES[case.case_id]
    active_clc_values = {
        item.get("value")
        for item in active_values
        if isinstance(item, dict)
        and item.get("key") == "cute_flash_clc_heads_per_batch"
        and type(item.get("value")) is int
        and item["value"] > 0
    }
    strict.check_equal(
        active_clc_values,
        set(expected_clc_anchors),
        f"{path}: targeted active CLC values",
    )
    clc_families = phase.get("clc_families")
    strict.require(isinstance(clc_families, list), f"{path}: CLC family results")
    strict.check_equal(
        {result.get("family") for result in clc_families if isinstance(result, dict)},
        set(EXPECTED_ORDINARY_CLC_FAMILIES),
        f"{path}: targeted ordinary CLC families",
    )
    for result in clc_families:
        strict.require(isinstance(result, dict), f"{path}: invalid CLC family")
        strict.check_equal(
            set(result.get("anchor_values", [])),
            set(expected_clc_anchors),
            f"{path}: {result.get('family')} CLC anchors",
        )
        strict.check_equal(
            set(result.get("legal_values", [])),
            set(expected_clc_legal),
            f"{path}: {result.get('family')} legal CLC values",
        )
        strict.check_equal(
            set(result.get("search_values", [])),
            set(expected_clc_legal),
            f"{path}: {result.get('family')} searchable CLC values",
        )
        strict.check_equal(
            set(result.get("planned_values", [])),
            set(expected_clc_legal),
            f"{path}: {result.get('family')} planned CLC values",
        )
        strict.check_equal(
            set(result.get("attempted_values", [])),
            set(expected_clc_legal),
            f"{path}: {result.get('family')} attempted CLC values",
        )
        strict.check_equal(
            set(result.get("witness_config_ids", {})),
            {str(value) for value in expected_clc_legal},
            f"{path}: {result.get('family')} CLC witness values",
        )
        strict.check_equal(
            set(result.get("selected_values", [])),
            set(expected_clc_legal),
            f"{path}: {result.get('family')} selected CLC values",
        )
        strict.check_equal(
            set(result.get("retained_values", [])),
            set(expected_clc_legal),
            f"{path}: {result.get('family')} retained CLC values",
        )
        value_space_exhausted = result.get("value_space_exhausted")
        strict.require(
            isinstance(value_space_exhausted, dict)
            and set(value_space_exhausted)
            == {str(value) for value in expected_clc_legal}
            and all(type(value) is bool for value in value_space_exhausted.values()),
            f"{path}: {result.get('family')} invalid CLC exhaustion map",
        )
        expected_conditional = {
            value
            for value in expected_clc_legal
            if not value_space_exhausted[str(value)]
        }
        strict.check_equal(
            set(result.get("conditional_values", [])),
            expected_conditional,
            f"{path}: {result.get('family')} conditional CLC values",
        )
        strict.check_equal(
            set(result.get("conditional_candidate_ids", {})),
            {str(value) for value in expected_conditional},
            f"{path}: {result.get('family')} conditional CLC children",
        )
        strict.require(
            len(result.get("anchor_values", [])) <= 8
            and set(result.get("planned_values", []))
            == set(result.get("legal_values", [])),
            f"{path}: {result.get('family')} incomplete CLC qualification plan",
        )
        strict.check_equal(
            result.get("complete"),
            True,
            f"{path}: {result.get('family')} CLC completion",
        )


def validate_live_starting_path_capacity(
    path: Path,
    provenance: dict[str, Any],
    phase: dict[str, Any],
) -> None:
    """Require enough continuations for the live v22 structural leaf catalog."""
    leaf_results = phase.get("leaf_results")
    compound_transfers = phase.get("compound_transfers")
    strict.require(
        isinstance(leaf_results, list)
        and leaf_results
        and all(isinstance(result, dict) for result in leaf_results),
        f"{path}: invalid ordinary structural leaf catalog",
    )
    strict.require(
        isinstance(compound_transfers, list)
        and all(isinstance(result, dict) for result in compound_transfers),
        f"{path}: invalid compound structural leaf catalog",
    )
    ordinary_widths: dict[str, int] = {}
    for result in leaf_results:
        family = result.get("family")
        strict.require(
            isinstance(family, str) and family,
            f"{path}: ordinary structural leaf without a family",
        )
        ordinary_widths[family] = ordinary_widths.get(family, 0) + 1

    retained_family_cap = phase.get("retained_family_cap")
    retained_family_limit = phase.get("retained_family_limit")
    retained_per_leaf = phase.get("retained_candidates_per_leaf")
    starting_path_limit = phase.get("starting_path_limit")
    family_probe_generations = phase.get("family_probe_generations")
    family_probe_path_limit = phase.get("family_probe_path_limit")
    maximum_path_capacity = phase.get("maximum_path_capacity")
    strict.require(
        type(retained_family_limit) is int and retained_family_limit > 0,
        f"{path}: invalid retained family limit",
    )
    strict.require(
        retained_family_cap is None
        or (type(retained_family_cap) is int and retained_family_cap > 0),
        f"{path}: invalid retained family cap",
    )
    strict.check_equal(
        provenance.get("flash_structural_retained_family_cap"),
        retained_family_cap,
        f"{path}: provenance retained family cap",
    )
    expected_family_limit = (
        len(ordinary_widths)
        if retained_family_cap is None
        else min(retained_family_cap, len(ordinary_widths))
    )
    strict.check_equal(
        retained_family_limit,
        expected_family_limit,
        f"{path}: live-derived retained family limit",
    )
    strict.require(
        type(retained_per_leaf) is int and retained_per_leaf > 0,
        f"{path}: invalid retained candidates per leaf",
    )
    strict.require(
        type(starting_path_limit) is int and starting_path_limit > 0,
        f"{path}: invalid live-derived starting path limit",
    )
    strict.require(
        type(family_probe_path_limit) is int and family_probe_path_limit >= 0,
        f"{path}: invalid live-derived family probe path limit",
    )
    strict.require(
        type(family_probe_generations) is int and family_probe_generations >= 0,
        f"{path}: invalid family probe generation count",
    )
    strict.require(
        type(maximum_path_capacity) is int and maximum_path_capacity > 0,
        f"{path}: invalid maximum path capacity",
    )
    strict.check_equal(
        provenance.get("flash_structural_starting_path_limit"),
        starting_path_limit,
        f"{path}: provenance starting path limit",
    )
    strict.check_equal(
        provenance.get("flash_structural_family_probe_path_limit"),
        family_probe_path_limit,
        f"{path}: provenance family probe path limit",
    )
    strict.check_equal(
        provenance.get("flash_structural_maximum_path_capacity"),
        maximum_path_capacity,
        f"{path}: provenance maximum path capacity",
    )
    promoted_count = min(retained_family_limit, len(ordinary_widths))
    ordinary_protocol_capacity = sum(
        sorted(ordinary_widths.values(), reverse=True)[:promoted_count]
    )
    secondary_capacity = promoted_count if retained_per_leaf > 1 else 0
    required_capacity = (
        1 + ordinary_protocol_capacity + secondary_capacity + len(compound_transfers)
    )
    strict.require(
        starting_path_limit >= required_capacity,
        f"{path}: starting path limit {starting_path_limit} is below live-derived "
        f"capacity {required_capacity}",
    )
    expected_probe_path_limit = (
        len(ordinary_widths) + len(compound_transfers) + 1
        if family_probe_generations > 0
        and retained_family_cap is not None
        and len(ordinary_widths) > retained_family_cap
        else 0
    )
    strict.check_equal(
        family_probe_path_limit,
        expected_probe_path_limit,
        f"{path}: live-derived family probe path limit",
    )
    strict.check_equal(
        maximum_path_capacity,
        max(starting_path_limit, family_probe_path_limit),
        f"{path}: live-derived maximum path capacity",
    )
    retained_path_count = phase.get("retained_path_count")
    strict.require(
        type(retained_path_count) is int
        and 0 < retained_path_count <= starting_path_limit,
        f"{path}: invalid retained structural path count",
    )


def validate_targeted_clc_generation_zero_attempts(
    path: Path,
    case: CaseSpec,
    source_rows: list[dict[str, str]],
    configs: dict[str, Any],
) -> None:
    expected_anchors = TARGETED_CLC_ANCHOR_VALUES.get(case.case_id)
    if expected_anchors is None:
        return
    attempted_ids = {
        row["config_id"]
        for row in source_rows
        if row["generation"] == "0"
        and row["status"] in ({"started"} | strict.LEDGER_ALIAS_STATUSES)
    }
    attempted_anchors = {
        config.get(strict.FLASH_CLC_HEADS_PER_BATCH_KEY)
        for config_id in attempted_ids
        if isinstance((config := configs.get(config_id)), dict)
        and config.get(strict.FLASH_PIPELINE_FAMILY_KEY)
        in EXPECTED_ORDINARY_CLC_FAMILIES
    }
    strict.check_equal(
        attempted_anchors & set(expected_anchors),
        set(expected_anchors),
        f"{path}: generation-0 CLC anchor attempts",
    )


def _validate_sidecars(
    path: Path,
    run: RunSpec,
    provenance: dict[str, Any],
    trial: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any]]:
    selected_config = provenance["selected_config"]
    selected_source = provenance["selected_source_sha256"]
    phase = trial.get("search_phase_metrics")
    strict.require(isinstance(phase, dict), f"{path}: missing search phase metrics")
    ledger_path = path.with_name("autotune.sources.csv")
    ledger = strict.read_and_validate_ledger(
        ledger_path, trial, selected_config, selected_source
    )
    csv_path = path.with_name("autotune.csv")
    metadata_path = path.with_name("autotune.meta.jsonl")
    try:
        csv_contents = csv_path.read_bytes()
        reader = csv.DictReader(io.StringIO(csv_contents.decode()))
        strict.check_equal(
            tuple(reader.fieldnames or ()),
            strict.AUTOTUNE_CSV_FIELDS,
            f"{csv_path}: header",
        )
        rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise RuntimeError(f"unable to read {csv_path}: {exc}") from exc
    source_rows = ledger["rows"]
    strict.check_equal(len(rows), len(source_rows), f"{csv_path}: ledger row count")
    for line, (row, source_row) in enumerate(zip(rows, source_rows, strict=True), 2):
        strict.require(
            None not in row
            and all(row.get(field) is not None for field in strict.AUTOTUNE_CSV_FIELDS),
            f"{csv_path}:{line}: malformed row",
        )
        for field in strict.AUTOTUNE_JOIN_FIELDS:
            strict.check_equal(
                row.get(field), source_row[field], f"{csv_path}:{line}: {field}"
            )
        successful = row.get("status") in {"ok", "deduplicated"}
        strict.check_equal(
            bool(row.get("perf_ms")), successful, f"{csv_path}:{line}: perf"
        )
        if successful:
            strict.require(
                strict.csv_float(row["perf_ms"], f"{csv_path}:{line}: perf") > 0,
                f"{csv_path}:{line}: nonpositive performance",
            )

    metadata, metadata_digest = strict.read_metadata_record(metadata_path)
    run_id = strict.metadata_run_id(metadata, metadata_path)
    strict.check_equal(metadata.get("run_id"), run_id, f"{metadata_path}: run ID")
    strict.check_equal(run_id, ledger["run_id"], f"{metadata_path}: ledger run ID")
    expected_kernel = (
        "causal_attention_output" if run.case.causal else "attention_output"
    )
    strict.check_equal(
        metadata.get("kernel_name"), expected_kernel, f"{metadata_path}: kernel"
    )
    shape = (run.case.z, run.case.h, run.case.seq_len, run.case.head_dim)
    strict.check_equal(
        metadata.get("input_shapes"),
        repr([shape, shape, shape]),
        f"{metadata_path}: shapes",
    )
    strict.check_equal(
        metadata.get("dtypes"),
        repr([run.case.torch_dtype] * 3),
        f"{metadata_path}: dtypes",
    )
    strict.check_equal(metadata.get("hardware"), "NVIDIA B200", f"{metadata_path}: GPU")
    settings = metadata.get("settings")
    strict.require(isinstance(settings, dict), f"{metadata_path}: missing settings")
    expected_settings = {
        "backend": "cute",
        "force_autotune": False,
        "effective_cache_read_bypass": True,
        "static_shapes": True,
        "autotune_log_details": True,
        "autotune_compile_timeout": 60,
        "autotune_benchmark_subprocess": True,
        "autotune_benchmark_timeout": 60,
        "autotune_random_seed": run.tuner_seed,
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
        strict.check_equal(
            settings.get(key), expected, f"{metadata_path}: settings.{key}"
        )
    configs = metadata.get("configs")
    strict.require(isinstance(configs, dict), f"{metadata_path}: missing configs")
    strict.check_equal(
        set(configs), {row["config_id"] for row in rows}, f"{metadata_path}: IDs"
    )
    for config_id, config in configs.items():
        strict.require(isinstance(config, dict), f"{metadata_path}: invalid config")
        strict.check_equal(
            strict.canonical_sha256(config)[:16],
            config_id,
            f"{metadata_path}: config ID",
        )
    for line, row in enumerate(rows, 2):
        config = configs[row["config_id"]]
        config_repr = (
            "Config("
            + ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))
            + ")"
        )
        strict.check_equal(row["config"], config_repr, f"{csv_path}:{line}: config")
    selected_id = strict.canonical_sha256(selected_config)[:16]
    strict.check_equal(
        configs.get(selected_id), selected_config, f"{metadata_path}: winner"
    )
    validate_targeted_clc_generation_zero_attempts(path, run.case, source_rows, configs)
    attempt_by_config: dict[str, dict[str, Any]] = {}
    attempt_history_by_config: dict[str, list[dict[str, Any]]] = {}
    for position, row in enumerate(rows):
        if row["status"] == "started":
            continue
        attempt = {
            "generation": int(row["generation"]),
            "status": row["status"],
            "source_hash": source_rows[position]["source_hash"],
            "perf_ms": float(row["perf_ms"]) if row["perf_ms"] else None,
            "position": position,
        }
        attempt_by_config[row["config_id"]] = attempt
        attempt_history_by_config.setdefault(row["config_id"], []).append(attempt)
    structural_execution = strict.validate_structural_prefix_execution(
        path,
        provenance,
        trial,
        source_rows,
        configs,
        attempt_by_config,
        attempt_history_by_config,
    )
    terminal_refinement = structural_execution["terminal_refinement"]
    required_preterminal = terminal_refinement["required_preterminal_candidate_count"]
    strict.require(
        terminal_refinement["preterminal_effective_candidate_count"]
        >= required_preterminal,
        f"{path}: terminal work masks an undersized full-search candidate set",
    )
    if not phase["exact_space_config_ids"]:
        strict.require(
            terminal_refinement["preterminal_successful_measurement_count"]
            >= required_preterminal,
            f"{path}: terminal work masks fewer than 100 measured preterminal successes",
        )
    return (
        strict.sha256_bytes(csv_contents),
        metadata_digest,
        strict.file_sha256(ledger_path),
        terminal_refinement,
    )


def validate_run_result(root: Path, run: RunSpec, expected_commit: str) -> ValidatedRun:
    path = (root / run.result_path).resolve()
    payload = strict.load_json_object(path)
    _median_ms, median_tflops = validate_payload_identity(
        path, payload, run, expected_commit
    )
    provenance, trial = _validate_provenance(path, payload, run)
    exact_config_ids = strict.exact_effective_search_space_ids(path, provenance)
    (
        autotune_csv_digest,
        metadata_digest,
        ledger_digest,
        terminal_refinement,
    ) = _validate_sidecars(path, run, provenance, trial)
    return ValidatedRun(
        run=run,
        median_tflops=median_tflops,
        num_configs_tested=strict.strict_int(
            trial.get("num_configs_tested"), f"{path}: tested configs", minimum=1
        ),
        num_successful_candidate_measurements=strict.strict_int(
            trial.get("num_successful_candidate_measurements"),
            f"{path}: successful configs",
            minimum=1,
        ),
        num_source_deduplications=strict.strict_int(
            trial.get("num_source_deduplications"),
            f"{path}: deduplicated sources",
            minimum=0,
        ),
        num_isolated_rebenchmark_timeouts=strict.strict_int(
            trial.get("num_isolated_rebenchmark_timeouts"),
            f"{path}: isolated rebenchmark timeouts",
            minimum=0,
        ),
        num_generations=strict.strict_int(
            trial.get("num_generations"), f"{path}: generations", minimum=0
        ),
        exact_effective_search_space_size=(
            len(exact_config_ids) if exact_config_ids is not None else None
        ),
        coverage_design_sha256=provenance["flash_structural_coverage_design_sha256"],
        compiler_seed_policy_sha256=strict.canonical_sha256(
            provenance["compiler_seed_policy"]
        ),
        terminal_refinement_policy_sha256=terminal_refinement["policy_sha256"],
        terminal_coordinate_surface_sha256=terminal_refinement[
            "coordinate_surface_sha256"
        ],
        terminal_refinement_sha256=strict.canonical_sha256(terminal_refinement),
        result_sha256=strict.file_sha256(path),
        autotune_csv_sha256=autotune_csv_digest,
        autotune_metadata_sha256=metadata_digest,
        source_ledger_sha256=ledger_digest,
        selected_source_sha256=provenance["selected_source_sha256"],
    )


def validate_group_gates(rows: list[ValidatedRun]) -> None:
    by_case: dict[str, list[ValidatedRun]] = {}
    by_surface: dict[tuple[object, ...], list[ValidatedRun]] = {}
    for row in rows:
        by_case.setdefault(row.run.case.case_id, []).append(row)
        by_surface.setdefault(row.run.case.surface_key, []).append(row)
    for case_id, case_rows in by_case.items():
        strict.check_equal(
            len(case_rows),
            len(case_rows[0].run.case.tuner_seeds),
            f"{case_id}: seed count",
        )
        digests = {row.coverage_design_sha256 for row in case_rows}
        strict.require(
            len(digests) == 1, f"{case_id}: seed-dependent coverage {digests}"
        )
        seed_policy_digests = {row.compiler_seed_policy_sha256 for row in case_rows}
        strict.require(
            len(seed_policy_digests) == 1,
            f"{case_id}: tuner-seed-dependent compiler seed policy "
            f"{seed_policy_digests}",
        )
        terminal_policy_digests = {
            row.terminal_refinement_policy_sha256 for row in case_rows
        }
        strict.require(
            len(terminal_policy_digests) == 1,
            f"{case_id}: tuner-seed-dependent terminal refinement policy "
            f"{terminal_policy_digests}",
        )
        terminal_surface_digests = {
            row.terminal_coordinate_surface_sha256 for row in case_rows
        }
        strict.require(
            len(terminal_surface_digests) == 1,
            f"{case_id}: tuner-seed-dependent terminal coordinate surface "
            f"{terminal_surface_digests}",
        )
    for surface, surface_rows in by_surface.items():
        digests = {row.coverage_design_sha256 for row in surface_rows}
        strict.require(
            len(digests) == 1,
            f"identical search surface {surface!r} changed coverage: {digests}",
        )
        seed_policy_digests = {row.compiler_seed_policy_sha256 for row in surface_rows}
        strict.require(
            len(seed_policy_digests) == 1,
            f"identical search surface {surface!r} changed compiler seed policy: "
            f"{seed_policy_digests}",
        )
        terminal_policy_digests = {
            row.terminal_refinement_policy_sha256 for row in surface_rows
        }
        strict.require(
            len(terminal_policy_digests) == 1,
            f"identical search surface {surface!r} changed terminal refinement "
            f"policy: {terminal_policy_digests}",
        )
        terminal_surface_digests = {
            row.terminal_coordinate_surface_sha256 for row in surface_rows
        }
        strict.require(
            len(terminal_surface_digests) == 1,
            f"identical search surface {surface!r} changed terminal coordinate "
            f"surface: {terminal_surface_digests}",
        )


def attention_flops(case: CaseSpec) -> float:
    flops = 4.0 * case.z * case.h * case.seq_len**2 * case.head_dim
    return flops * (0.5 if case.causal else 1.0)


def summarize_remeasurement(
    raw_rounds: list[dict[str, Any]],
    config_names: list[str],
    case: CaseSpec,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    strict.require(raw_rounds and config_names, "remeasurement data is empty")
    names = [*config_names, "sdpa"]
    flops = attention_flops(case)
    timers: dict[str, Any] = {}
    for timer in ("event", "wall"):
        implementation: dict[str, Any] = {}
        for name in names:
            values = [float(row["times"][name][f"{timer}_ms"]) for row in raw_rounds]
            median_ms = statistics.median(values)
            implementation[name] = {
                "median_ms": median_ms,
                "median_tflops": flops / (median_ms * 1e9),
                "raw_ms": values,
            }
        performances = [implementation[name]["median_tflops"] for name in config_names]
        point_fraction = min(performances) / max(performances)
        point_sdpa_fraction = (
            min(performances) / implementation["sdpa"]["median_tflops"]
        )
        rng = random.Random(bootstrap_seed ^ (0xE7E7 if timer == "event" else 0xA11))
        fractions = []
        sdpa_fractions = []
        for _ in range(bootstrap_samples):
            sampled = [rng.randrange(len(raw_rounds)) for _ in raw_rounds]
            sampled_performances = []
            for name in config_names:
                median_ms = statistics.median(
                    raw_rounds[index]["times"][name][f"{timer}_ms"] for index in sampled
                )
                sampled_performances.append(flops / (median_ms * 1e9))
            fractions.append(min(sampled_performances) / max(sampled_performances))
            sdpa_median_ms = statistics.median(
                raw_rounds[index]["times"]["sdpa"][f"{timer}_ms"] for index in sampled
            )
            sdpa_performance = flops / (sdpa_median_ms * 1e9)
            sdpa_fractions.append(min(sampled_performances) / sdpa_performance)
        fractions.sort()
        sdpa_fractions.sort()
        timers[timer] = {
            "implementations": implementation,
            "seed_robustness_fraction": point_fraction,
            "paired_bootstrap_95_ci": [
                fractions[int(bootstrap_samples * 0.025)],
                fractions[int(bootstrap_samples * 0.975)],
            ],
            "minimum_seed_vs_sdpa_fraction": point_sdpa_fraction,
            "minimum_seed_vs_sdpa_paired_bootstrap_95_ci": [
                sdpa_fractions[int(bootstrap_samples * 0.025)],
                sdpa_fractions[int(bootstrap_samples * 0.975)],
            ],
        }
    return {"flops": flops, "timers": timers}


def validate_seed_robustness(
    summary: dict[str, Any], case: CaseSpec, *, context: str
) -> None:
    gate = 0.99 if case.legality_class == "div4" else 0.98
    for timer in ("event", "wall"):
        record = summary["timers"][timer]
        strict.require(
            record["seed_robustness_fraction"] >= gate,
            f"{context}: {timer} cross-measured seed robustness below {gate:.0%}",
        )
        strict.require(
            record["paired_bootstrap_95_ci"][0] >= gate,
            f"{context}: {timer} paired-bootstrap robustness below {gate:.0%}",
        )
        if (
            case.dtype == "float16"
            and case.head_dim == 64
            and case.legality_class == "div4"
        ):
            sdpa_gate = 0.98
            strict.require(
                record["minimum_seed_vs_sdpa_fraction"] >= sdpa_gate,
                f"{context}: {timer} minimum seed is below {sdpa_gate:.0%} of SDPA",
            )
            strict.require(
                record["minimum_seed_vs_sdpa_paired_bootstrap_95_ci"][0] >= sdpa_gate,
                f"{context}: {timer} paired minimum seed is below "
                f"{sdpa_gate:.0%} of SDPA",
            )


def _validate_remeasurement_payload(
    path: Path,
    payload: dict[str, Any],
    case: CaseSpec,
    case_rows: list[ValidatedRun],
    expected_worker_sha256: str,
    expected_commit: str,
) -> ValidatedRemeasurement:
    strict.check_equal(payload.get("schema_version"), 1, f"{path}: schema")
    strict.check_equal(payload.get("status"), "PASS", f"{path}: status")
    strict.check_equal(payload.get("case_id"), case.case_id, f"{path}: case")
    strict.check_equal(payload.get("physical_gpu"), case.physical_gpu, f"{path}: GPU")
    strict.check_equal(
        payload.get("measured_commit"), expected_commit, f"{path}: measured commit"
    )
    strict.check_equal(
        payload.get("worker_sha256"), expected_worker_sha256, f"{path}: worker"
    )
    strict.check_equal(
        payload.get("shape"),
        {
            "z": case.z,
            "h": case.h,
            "seq_len": case.seq_len,
            "head_dim": case.head_dim,
            "dtype": case.dtype,
            "causal": int(case.causal),
        },
        f"{path}: shape",
    )
    for endpoint in ("gpu_start", "gpu_end"):
        gpu = payload.get(endpoint)
        strict.require(isinstance(gpu, dict), f"{path}: missing {endpoint}")
        strict.check_equal(
            gpu.get("physical_gpu"), case.physical_gpu, f"{path}: {endpoint}"
        )
        strict.check_equal(gpu.get("name"), "NVIDIA B200", f"{path}: {endpoint}")
        strict.require(
            abs(strict.finite_float(gpu.get("power_limit_w"), f"{path}: power") - 750)
            <= 0.5,
            f"{path}: invalid power cap",
        )
        strict.check_equal(
            gpu.get("active_compute_pids"), [], f"{path}: competing PIDs"
        )
    protocol = payload.get("protocol")
    strict.require(isinstance(protocol, dict), f"{path}: missing protocol")
    strict.check_equal(
        protocol.get("forced_sdpa_backend"), "CUDNN_ATTENTION", f"{path}: SDPA"
    )
    strict.check_equal(
        protocol.get("protocol_seed"), case.input_seed ^ 0xC205, f"{path}: seed"
    )
    rounds = strict.strict_int(protocol.get("rounds"), f"{path}: rounds", minimum=6)
    strict.require(rounds % 6 == 0, f"{path}: rounds are not balanced blocks")
    repeatability = strict.strict_int(
        protocol.get("repeatability_launches"), f"{path}: repeats", minimum=2
    )
    strict.check_equal(protocol.get("target_timing_ms"), 20.0, f"{path}: target time")
    maximum_repetitions = strict.strict_int(
        protocol.get("max_timing_repetitions"),
        f"{path}: maximum timing repetitions",
        minimum=1,
    )
    strict.check_equal(maximum_repetitions, 4096, f"{path}: repetition cap")
    timing_repetitions = strict.strict_int(
        protocol.get("timing_repetitions"),
        f"{path}: timing repetitions",
        minimum=1,
    )
    strict.require(
        timing_repetitions <= maximum_repetitions,
        f"{path}: timing repetitions exceed cap",
    )
    calibration = protocol.get("sdpa_calibration_event_ms")
    strict.require(
        isinstance(calibration, list) and len(calibration) == 3,
        f"{path}: invalid timing calibration",
    )
    calibration_values = [
        strict.finite_float(value, f"{path}: calibration", positive=True)
        for value in calibration
    ]
    expected_repetitions = min(
        maximum_repetitions,
        max(1, math.ceil(20.0 / max(statistics.median(calibration_values), 1e-6))),
    )
    strict.check_equal(
        timing_repetitions, expected_repetitions, f"{path}: timing calibration"
    )
    bootstrap_samples = strict.strict_int(
        protocol.get("bootstrap_samples"), f"{path}: bootstrap samples", minimum=1
    )
    bootstrap_seed = strict.strict_int(
        protocol.get("bootstrap_seed"), f"{path}: bootstrap seed", minimum=1
    )
    ordered_rows = sorted(
        case_rows, key=lambda row: case.tuner_seeds.index(row.run.tuner_seed)
    )
    expected_names = [f"seed_{row.run.tuner_seed}" for row in ordered_rows]
    selected_sources = payload.get("selected_sources")
    strict.require(isinstance(selected_sources, list), f"{path}: missing sources")
    strict.check_equal(
        [source.get("name") for source in selected_sources],
        expected_names,
        f"{path}: names",
    )
    expected_source_digests = tuple(row.selected_source_sha256 for row in ordered_rows)
    strict.check_equal(
        tuple(source.get("selected_source_sha256") for source in selected_sources),
        expected_source_digests,
        f"{path}: selected source linkage",
    )
    for source, row in zip(selected_sources, ordered_rows, strict=True):
        strict.check_equal(source.get("run_id"), row.run.run_id, f"{path}: source run")
        strict.check_equal(
            source.get("compiled_source_sha256"),
            row.selected_source_sha256,
            f"{path}: compiled source",
        )
        strict.check_equal(
            source.get("selected_config_sha256"),
            strict.canonical_sha256(
                strict.load_json_object(path.parents[1] / row.run.result_path)[
                    "helion_overrides"
                ]["autotune_provenance"]["selected_config"]
            ),
            f"{path}: selected config",
        )
    correctness = payload.get("correctness")
    strict.require(isinstance(correctness, dict), f"{path}: missing correctness")
    strict.check_equal(
        set(correctness), set(expected_names), f"{path}: correctness names"
    )
    for name, record in correctness.items():
        strict.require(isinstance(record, dict), f"{path}: invalid correctness {name}")
        numerics = record.get("numerics")
        repeats = record.get("repeatability")
        strict.require(
            isinstance(numerics, dict) and numerics.get("passed") is True,
            f"{path}: {name} failed numerics",
        )
        strict.require(
            isinstance(repeats, list)
            and len(repeats) == repeatability - 1
            and all(
                isinstance(item, dict) and item.get("passed") is True
                for item in repeats
            ),
            f"{path}: {name} failed repeatability",
        )
    raw_rounds = payload.get("raw_rounds")
    strict.require(
        isinstance(raw_rounds, list) and len(raw_rounds) == rounds,
        f"{path}: raw round count",
    )
    all_names = [*expected_names, "sdpa"]
    position_counts = {name: [0] * len(all_names) for name in all_names}
    for index, row in enumerate(raw_rounds):
        strict.require(isinstance(row, dict), f"{path}: invalid raw round")
        strict.check_equal(row.get("round_index"), index, f"{path}: round index")
        order = row.get("order")
        times = row.get("times")
        strict.require(isinstance(order, list), f"{path}: invalid order")
        strict.check_equal(len(order), len(all_names), f"{path}: order length")
        strict.check_equal(set(order), set(all_names), f"{path}: round implementations")
        strict.require(isinstance(times, dict), f"{path}: invalid times")
        strict.check_equal(
            set(times), set(all_names), f"{path}: timing implementations"
        )
        for position, name in enumerate(order):
            position_counts[name][position] += 1
            timing = times[name]
            strict.require(isinstance(timing, dict), f"{path}: timing entry")
            for timer in ("event_ms", "wall_ms"):
                strict.finite_float(
                    timing.get(timer), f"{path}: {name} {timer}", positive=True
                )
    expected_position_count = rounds // len(all_names)
    strict.require(
        all(
            count == expected_position_count
            for counts in position_counts.values()
            for count in counts
        ),
        f"{path}: implementation order is not position-balanced",
    )
    computed = summarize_remeasurement(
        raw_rounds,
        expected_names,
        case,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    strict.check_equal(payload.get("summary"), computed, f"{path}: summary")
    validate_seed_robustness(computed, case, context=str(path))
    return ValidatedRemeasurement(
        case=case,
        path=path,
        sha256=strict.file_sha256(path),
        selected_source_sha256s=expected_source_digests,
        summary=computed,
    )


def validate_remeasurements(
    root: Path,
    cases: tuple[CaseSpec, ...],
    rows: list[ValidatedRun],
    expected_worker_sha256: str,
    expected_commit: str,
) -> tuple[ValidatedRemeasurement, ...]:
    directory = root / "remeasure"
    expected = {directory / f"{case.case_id}.json" for case in cases}
    discovered = set(directory.glob("*.json")) if directory.is_dir() else set()
    strict.check_equal(discovered, expected, "remeasurement result set")
    declaration_mtime_ns = (root / "campaign.declarations.sha256").stat().st_mtime_ns
    for path in expected:
        strict.require(
            path.is_file() and not path.is_symlink(),
            f"invalid remeasurement result: {path}",
        )
        strict.require(
            path.stat().st_mtime_ns >= declaration_mtime_ns,
            f"{path}: remeasurement predates campaign declaration",
        )
    result = []
    for case in cases:
        case_rows = [row for row in rows if row.run.case.case_id == case.case_id]
        path = directory / f"{case.case_id}.json"
        result.append(
            _validate_remeasurement_payload(
                path,
                strict.load_json_object(path),
                case,
                case_rows,
                expected_worker_sha256,
                expected_commit,
            )
        )
    return tuple(result)


def _validate_snapshot(root: Path, relative: str, expected_digest: str) -> None:
    path = root / relative
    strict.require(
        path.is_file() and not path.is_symlink(), f"missing snapshot: {path}"
    )
    strict.require(path.stat().st_mode & 0o222 == 0, f"writable snapshot: {path}")
    strict.check_equal(
        strict.file_sha256(path), expected_digest, f"snapshot {relative}"
    )
    digest_path = path.with_name(f"{path.name}.sha256")
    strict.check_equal(
        digest_path.read_text().strip(), expected_digest, f"snapshot digest {relative}"
    )


def validate_event_history(
    records: list[dict[str, Any]], runs: tuple[RunSpec, ...]
) -> None:
    expected_ids = {run.run_id for run in runs}
    by_run: dict[str, list[dict[str, Any]]] = {run_id: [] for run_id in expected_ids}
    for record in records:
        if record.get("record_type") != "event":
            continue
        event = record.get("event")
        if event in {
            "remeasurement_started",
            "remeasurement_finished",
        }:
            continue
        strict.require(
            event
            in {
                "attempt_started",
                "attempt_finished",
                "campaign_validated",
            },
            f"unknown campaign event {event!r}",
        )
        run_id = record.get("run_id")
        if event == "campaign_validated":
            strict.require(run_id is None, "campaign validation event has a run ID")
            continue
        strict.require(run_id in expected_ids, f"event has unknown run ID {run_id!r}")
        by_run[run_id].append(record)
    for run_id, history in by_run.items():
        started = [
            record for record in history if record.get("event") == "attempt_started"
        ]
        finished = [
            record for record in history if record.get("event") == "attempt_finished"
        ]
        strict.check_equal(len(started), 1, f"{run_id}: attempt count")
        strict.check_equal(len(finished), 1, f"{run_id}: finish count")
        strict.require(
            history.index(started[0]) < history.index(finished[0]),
            f"{run_id}: finish predates start",
        )
        strict.check_equal(
            finished[0].get("returncode"), 0, f"{run_id}: attempt result"
        )


def validate_remeasurement_event_history(
    records: list[dict[str, Any]], cases: tuple[CaseSpec, ...]
) -> None:
    expected_ids = {case.case_id for case in cases}
    by_case: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in expected_ids}
    allowed = {
        "remeasurement_started",
        "remeasurement_finished",
    }
    for record in records:
        if record.get("record_type") != "event" or record.get("event") not in allowed:
            continue
        case_id = record.get("case_id")
        strict.require(
            case_id in expected_ids, f"remeasurement has unknown case {case_id!r}"
        )
        by_case[case_id].append(record)
    for case_id, history in by_case.items():
        started = [
            item for item in history if item.get("event") == "remeasurement_started"
        ]
        finished = [
            item for item in history if item.get("event") == "remeasurement_finished"
        ]
        strict.check_equal(len(started), 1, f"{case_id}: remeasurement attempt count")
        strict.check_equal(len(finished), 1, f"{case_id}: remeasurement finish count")
        strict.require(
            history.index(started[0]) < history.index(finished[0]),
            f"{case_id}: remeasurement finish predates start",
        )
        strict.check_equal(
            finished[0].get("returncode"), 0, f"{case_id}: remeasurement result"
        )


def validate_campaign(
    root: Path, *, require_remeasurement: bool = True
) -> CampaignValidation:
    root = root.expanduser().resolve()
    records = load_campaign_records(root)
    validation_events = [
        record
        for record in records
        if record.get("record_type") == "event"
        and record.get("event") == "campaign_validated"
    ]
    strict.require(
        len(validation_events) <= 1, "campaign has multiple validation events"
    )
    if validation_events:
        strict.check_equal(records[-1], validation_events[0], "final campaign event")
    header = records[0]
    strict.check_equal(header.get("record_type"), "campaign", "campaign header")
    strict.check_equal(header.get("schema_version"), SCHEMA_VERSION, "campaign schema")
    strict.check_equal(
        header.get("expected_commit"), EXPECTED_COMMIT, "campaign expected commit"
    )
    for name in (
        "matrix",
        "launcher",
        "validator",
        "strict_validator",
        "benchmark",
        "remeasurement",
    ):
        relative = header.get(f"{name}_path")
        digest = header.get(f"{name}_sha256")
        strict.require(isinstance(relative, str), f"campaign missing {name} path")
        strict.require(
            isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            f"campaign invalid {name} digest",
        )
        _validate_snapshot(root, relative, digest)
    matrix_path = root / str(header["matrix_path"])
    cases = parse_case_matrix(matrix_path)
    runs = expand_runs(cases)
    declarations = declared_runs(records)
    strict.check_equal(len(declarations), len(runs), "campaign declaration count")
    declaration_by_id = {record.get("run_id"): record for record in declarations}
    strict.check_equal(set(declaration_by_id), {run.run_id for run in runs}, "run IDs")
    repo_root = header.get("repo_root")
    python_executable = header.get("python_executable")
    strict.require(
        isinstance(repo_root, str) and Path(repo_root).is_absolute(),
        "campaign repository is not absolute",
    )
    strict.require(
        isinstance(python_executable, str) and Path(python_executable).is_absolute(),
        "campaign Python executable is not absolute",
    )
    for run in runs:
        declaration = declaration_by_id[run.run_id]
        strict.check_equal(
            declaration.get("case_id"), run.case.case_id, "case identity"
        )
        strict.check_equal(declaration.get("tuner_seed"), run.tuner_seed, "tuner seed")
        strict.check_equal(
            declaration.get("physical_gpu"), run.case.physical_gpu, "physical GPU"
        )
        strict.check_equal(
            declaration.get("result_path"), run.result_path.as_posix(), "result path"
        )
        command = declaration.get("command")
        strict.check_equal(
            command,
            expected_command(Path(python_executable), Path(repo_root), root, run),
            f"{run.run_id}: strict command",
        )
        strict.check_equal(
            declaration.get("command_sha256"),
            strict.canonical_sha256(command),
            f"{run.run_id}: command digest",
        )
    validate_event_history(records, runs)
    remeasurement_declarations = declared_remeasurements(records)
    strict.check_equal(
        len(remeasurement_declarations), len(cases), "remeasurement declaration count"
    )
    by_case = {
        declaration.get("case_id"): declaration
        for declaration in remeasurement_declarations
    }
    strict.check_equal(
        set(by_case), {case.case_id for case in cases}, "remeasure cases"
    )
    for case in cases:
        declaration = by_case[case.case_id]
        command = expected_remeasurement_command(
            Path(python_executable),
            Path(repo_root),
            root,
            case,
            str(header["remeasurement_sha256"]),
        )
        strict.check_equal(
            declaration.get("command"), command, f"{case.case_id}: remeasure command"
        )
        strict.check_equal(
            declaration.get("command_sha256"),
            strict.canonical_sha256(command),
            f"{case.case_id}: remeasure command digest",
        )
    declaration_lines = [
        strict.canonical_json(record)
        for record in records
        if record.get("record_type") != "event"
    ]
    declarations_digest = strict.sha256_bytes(
        ("\n".join(declaration_lines) + "\n").encode()
    )
    strict.check_equal(
        (root / "campaign.declarations.sha256").read_text().strip(),
        declarations_digest,
        "campaign declaration digest",
    )
    validate_result_set(root, runs)
    declaration_mtime_ns = (root / "campaign.declarations.sha256").stat().st_mtime_ns
    for run in runs:
        result = root / run.result_path
        for name in RESULT_FILENAMES:
            evidence = result.with_name(name)
            strict.require(
                evidence.stat().st_mtime_ns >= declaration_mtime_ns,
                f"{evidence}: evidence predates campaign declaration",
            )
    expected_commit = EXPECTED_COMMIT
    validated = [validate_run_result(root, run, expected_commit) for run in runs]
    validate_group_gates(validated)
    remeasurements: tuple[ValidatedRemeasurement, ...] = ()
    if require_remeasurement:
        validate_remeasurement_event_history(records, cases)
        remeasurements = validate_remeasurements(
            root,
            cases,
            validated,
            str(header["remeasurement_sha256"]),
            expected_commit,
        )
    return CampaignValidation(cases, runs, tuple(validated), remeasurements)


def render_manifest(validation: CampaignValidation) -> str:
    output = io.StringIO()
    fields = (
        "case_id",
        "tuner_seed",
        "physical_gpu",
        "z",
        "h",
        "seq_len",
        "head_dim",
        "dtype",
        "causal",
        "legality_class",
        "median_tflops",
        "num_configs_tested",
        "num_successful_candidate_measurements",
        "num_source_deduplications",
        "num_isolated_rebenchmark_timeouts",
        "num_generations",
        "exact_effective_search_space_size",
        "coverage_design_sha256",
        "compiler_seed_policy_sha256",
        "terminal_refinement_policy_sha256",
        "terminal_coordinate_surface_sha256",
        "terminal_refinement_sha256",
        "selected_source_sha256",
        "result_sha256",
        "autotune_csv_sha256",
        "autotune_metadata_sha256",
        "source_ledger_sha256",
        "remeasurement_sha256",
        "cross_measured_tflops",
        "cross_measured_sdpa_tflops",
        "seed_robustness_fraction",
        "seed_robustness_bootstrap_low",
        "seed_robustness_bootstrap_high",
        "minimum_seed_vs_sdpa_fraction",
        "minimum_seed_vs_sdpa_bootstrap_low",
        "minimum_seed_vs_sdpa_bootstrap_high",
    )
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    remeasurements = {item.case.case_id: item for item in validation.remeasurements}
    for row in validation.runs:
        case = row.run.case
        remeasurement = remeasurements.get(case.case_id)
        event = (
            remeasurement.summary["timers"]["event"]
            if remeasurement is not None
            else None
        )
        implementation = f"seed_{row.run.tuner_seed}"
        writer.writerow(
            {
                "case_id": case.case_id,
                "tuner_seed": row.run.tuner_seed,
                "physical_gpu": case.physical_gpu,
                "z": case.z,
                "h": case.h,
                "seq_len": case.seq_len,
                "head_dim": case.head_dim,
                "dtype": case.dtype,
                "causal": int(case.causal),
                "legality_class": case.legality_class,
                "median_tflops": f"{row.median_tflops:.12g}",
                "num_configs_tested": row.num_configs_tested,
                "num_successful_candidate_measurements": (
                    row.num_successful_candidate_measurements
                ),
                "num_source_deduplications": row.num_source_deduplications,
                "num_isolated_rebenchmark_timeouts": (
                    row.num_isolated_rebenchmark_timeouts
                ),
                "num_generations": row.num_generations,
                "exact_effective_search_space_size": (
                    row.exact_effective_search_space_size
                    if row.exact_effective_search_space_size is not None
                    else ""
                ),
                "coverage_design_sha256": row.coverage_design_sha256,
                "compiler_seed_policy_sha256": row.compiler_seed_policy_sha256,
                "terminal_refinement_policy_sha256": (
                    row.terminal_refinement_policy_sha256
                ),
                "terminal_coordinate_surface_sha256": (
                    row.terminal_coordinate_surface_sha256
                ),
                "terminal_refinement_sha256": row.terminal_refinement_sha256,
                "selected_source_sha256": row.selected_source_sha256,
                "result_sha256": row.result_sha256,
                "autotune_csv_sha256": row.autotune_csv_sha256,
                "autotune_metadata_sha256": row.autotune_metadata_sha256,
                "source_ledger_sha256": row.source_ledger_sha256,
                "remeasurement_sha256": (
                    remeasurement.sha256 if remeasurement is not None else ""
                ),
                "cross_measured_tflops": (
                    event["implementations"][implementation]["median_tflops"]
                    if event is not None
                    else ""
                ),
                "cross_measured_sdpa_tflops": (
                    event["implementations"]["sdpa"]["median_tflops"]
                    if event is not None
                    else ""
                ),
                "seed_robustness_fraction": (
                    event["seed_robustness_fraction"] if event is not None else ""
                ),
                "seed_robustness_bootstrap_low": (
                    event["paired_bootstrap_95_ci"][0] if event is not None else ""
                ),
                "seed_robustness_bootstrap_high": (
                    event["paired_bootstrap_95_ci"][1] if event is not None else ""
                ),
                "minimum_seed_vs_sdpa_fraction": (
                    event["minimum_seed_vs_sdpa_fraction"] if event is not None else ""
                ),
                "minimum_seed_vs_sdpa_bootstrap_low": (
                    event["minimum_seed_vs_sdpa_paired_bootstrap_95_ci"][0]
                    if event is not None
                    else ""
                ),
                "minimum_seed_vs_sdpa_bootstrap_high": (
                    event["minimum_seed_vs_sdpa_paired_bootstrap_95_ci"][1]
                    if event is not None
                    else ""
                ),
            }
        )
    return output.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generalized full-autotune campaign"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contents = render_manifest(validate_campaign(args.artifact_root))
    if args.output is None:
        print(contents, end="")
    else:
        output = args.output.expanduser().resolve()
        root = args.artifact_root.expanduser().resolve()
        strict.require(
            output != root and root not in output.parents,
            "validation output must be outside the artifact root",
        )
        output.write_text(contents)


if __name__ == "__main__":
    main()
