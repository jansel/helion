from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
from typing import TYPE_CHECKING
from typing import Any

import build_heldout_manifest as heldout
import build_strict_manifest as strict

if TYPE_CHECKING:
    from types import ModuleType

SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
EXPECTED_COMMIT = heldout.EXPECTED_COMMIT
EXPECTED_CUTE_VERSION = strict.EXPECTED_CUTE_VERSION
BENCHMARK_RELATIVE = Path("benchmarks/cute/compare_attention_backends.py")
ATTENTION_RELATIVE = Path("examples/attention.py")
ROUNDS = 12
WARMUP_CALLS = 3
THERMAL_WARMUP_SECONDS = 10.0
REPEATABILITY_LAUNCHES = 3
TARGET_TIMING_MS = 20.0
MAX_TIMING_REPETITIONS = 4096
BOOTSTRAP_SAMPLES = 20000
MEDIAN_BEST_GATE = heldout.MINIMUM_MEDIAN_BEST_SEED_FRACTION
WORST_BEST_GATE = heldout.MINIMUM_WORST_BEST_SEED_FRACTION
SNAPSHOT_FILENAMES = (
    "build_heldout_manifest.py",
    "build_strict_manifest.py",
    "heldout_adjudication.py",
    "remeasure_generalization_winners.py",
    "remeasure_heldout_winners.py",
    "run_heldout_adjudication.py",
    "validate_generalization_campaign.py",
    "validate_heldout_adjudication.py",
)
RESULT_FILENAME = "result.json"
COMPLETION_FILENAME = "completion.json"
CAMPAIGN_FIELDS = {
    "schema_version",
    "expected_commit",
    "expected_cute_version",
    "campaign_nonce",
    "repo_root",
    "heldout_root",
    "all8_root",
    "python_executable",
    "source_snapshots",
    "measured_source_snapshots",
    "heldout_evidence",
    "all8_evidence",
    "validated_heldout_rows",
    "validated_heldout_rows_sha256",
    "validated_all8_rows",
    "validated_all8_rows_sha256",
    "all8_reference_manifest_sha256",
    "cases",
    "report_path",
}
RESULT_FIELDS = {
    "schema_version",
    "status",
    "case_id",
    "shape",
    "physical_gpu",
    "measured_commit",
    "campaign_sha256",
    "worker_sha256",
    "source_snapshot_sha256",
    "gpu_start",
    "gpu_end",
    "environment",
    "protocol",
    "selected_sources",
    "correctness",
    "direct_search_tflops",
    "raw_rounds",
    "summary",
}


@dataclass(frozen=True)
class CaseDefinition:
    case_id: str
    variant: str
    seq_len: int
    physical_gpu: int
    tuner_seeds: tuple[int, ...]

    @property
    def causal(self) -> bool:
        return self.variant == "causal"

    @property
    def shape(self) -> dict[str, object]:
        return {
            "z": 2,
            "h": 32,
            "seq_len": self.seq_len,
            "head_dim": 64,
            "dtype": "float16",
            "causal": int(self.causal),
        }


CASES = (
    CaseDefinition(
        "dense_s81920",
        "dense",
        81920,
        7,
        (2026082301, 2026082302, 2026082303, 2026082304, 2026082305),
    ),
    CaseDefinition(
        "causal_s196608",
        "causal",
        196608,
        6,
        (2026082311, 2026082312, 2026082313, 2026082314, 2026082315),
    ),
)


def case_by_id(case_id: str) -> CaseDefinition:
    matches = [case for case in CASES if case.case_id == case_id]
    strict.require(len(matches) == 1, f"unknown adjudication case: {case_id!r}")
    return matches[0]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def require_sha256(value: object, label: str) -> str:
    strict.require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a lowercase SHA-256: {value!r}",
    )
    return value


def relative_path(value: object, label: str) -> Path:
    strict.require(isinstance(value, str) and value, f"{label} is not a path")
    path = Path(value)
    strict.require(
        not path.is_absolute() and ".." not in path.parts,
        f"{label} must be a contained relative path: {value!r}",
    )
    return path


def require_regular_file(path: Path, label: str, *, read_only: bool = False) -> None:
    strict.require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular file: {path}",
    )
    if read_only:
        strict.require(
            path.stat().st_mode & 0o222 == 0,
            f"{label} is writable: {path}",
        )


def load_json_object(path: Path) -> dict[str, Any]:
    require_regular_file(path, "JSON input")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON object: {path}") from exc
    strict.require(isinstance(value, dict), f"JSON value is not an object: {path}")
    return value


def campaign_digest_path(root: Path) -> Path:
    return root / "campaign.sha256"


def campaign_path(root: Path) -> Path:
    return root / "campaign.json"


def read_campaign(root: Path) -> tuple[dict[str, Any], str]:
    path = campaign_path(root)
    digest_path = campaign_digest_path(root)
    require_regular_file(path, "campaign declaration", read_only=True)
    require_regular_file(digest_path, "campaign digest", read_only=True)
    expected_digest = digest_path.read_text().strip()
    require_sha256(expected_digest, "campaign digest")
    actual_digest = strict.file_sha256(path)
    strict.check_equal(actual_digest, expected_digest, "campaign declaration digest")
    return load_json_object(path), actual_digest


def protocol_for_case(case: CaseDefinition) -> dict[str, object]:
    protocol_seed = strict.EXPECTED_INPUT_SEED ^ case.seq_len ^ 0xAD1D1CA7
    return {
        "name": "held-out winners balanced randomized adjudication",
        "input_seed": strict.EXPECTED_INPUT_SEED,
        "order_seed": protocol_seed ^ 0x5A17,
        "bootstrap_seed": protocol_seed ^ 0xB007,
        "rounds": ROUNDS,
        "warmup_calls_per_implementation": WARMUP_CALLS,
        "thermal_warmup_seconds": THERMAL_WARMUP_SECONDS,
        "repeatability_launches": REPEATABILITY_LAUNCHES,
        "target_timing_ms": TARGET_TIMING_MS,
        "max_timing_repetitions": MAX_TIMING_REPETITIONS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "forced_sdpa_backend": "CUDNN_ATTENTION",
        "timers": ["cuda_event", "host_wall"],
    }


def evidence_files(root: Path, rows: list[dict[str, object]]) -> list[dict[str, str]]:
    paths = {
        root / "campaign.csv",
        root / "launcher" / "run_strict_all8.sh",
        root / "launcher" / "run_strict_heldout.sh",
    }
    for row in rows:
        paths.update(
            root / relative_path(row[field], f"held-out {field}")
            for field in (
                "result_path",
                "autotune_csv_path",
                "autotune_metadata_path",
                "source_ledger_path",
            )
        )
    result = []
    for path in sorted(paths):
        require_regular_file(path, "held-out evidence")
        result.append(
            {
                "path": path.resolve().relative_to(root).as_posix(),
                "sha256": strict.file_sha256(path),
            }
        )
    return result


def expected_evidence_paths(rows: list[dict[str, object]]) -> set[Path]:
    result = {
        Path("campaign.csv"),
        Path("launcher/run_strict_all8.sh"),
        Path("launcher/run_strict_heldout.sh"),
    }
    for row in rows:
        result.update(
            relative_path(row[field], f"held-out {field}")
            for field in (
                "result_path",
                "autotune_csv_path",
                "autotune_metadata_path",
                "source_ledger_path",
            )
        )
    return result


def strict_result_evidence_paths(rows: list[dict[str, object]]) -> set[Path]:
    result = set()
    for row in rows:
        result.update(
            relative_path(row[field], f"strict {field}")
            for field in (
                "result_path",
                "autotune_csv_path",
                "autotune_metadata_path",
                "source_ledger_path",
            )
        )
    return result


def all8_evidence_files(
    root: Path, rows: list[dict[str, object]]
) -> list[dict[str, str]]:
    result = []
    for relative in sorted(strict_result_evidence_paths(rows)):
        path = root / relative
        require_regular_file(path, "all8 evidence")
        result.append({"path": relative.as_posix(), "sha256": strict.file_sha256(path)})
    return result


def validate_evidence_files(
    heldout_root: Path,
    records: object,
    *,
    expected_paths: set[Path],
    label: str = "held-out evidence",
) -> None:
    strict.require(isinstance(records, list) and records, f"missing {label} files")
    seen: set[Path] = set()
    for record in records:
        strict.require(isinstance(record, dict), f"invalid {label} record")
        strict.check_equal(set(record), {"path", "sha256"}, f"{label} record fields")
        relative = relative_path(record.get("path"), f"{label} path")
        strict.require(relative not in seen, f"duplicate {label} path: {relative}")
        seen.add(relative)
        path = heldout_root / relative
        require_regular_file(path, label)
        strict.check_equal(
            strict.file_sha256(path),
            require_sha256(record.get("sha256"), f"{label} digest"),
            f"{label} {relative}",
        )
    strict.check_equal(seen, expected_paths, f"{label} path set")


def rows_by_case(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for case in CASES:
        case_rows = [
            row
            for row in rows
            if row.get("variant") == case.variant and row.get("seq_len") == case.seq_len
        ]
        case_rows.sort(key=lambda row: case.tuner_seeds.index(int(row["tuner_seed"])))
        strict.check_equal(
            [int(row["tuner_seed"]) for row in case_rows],
            list(case.tuner_seeds),
            f"{case.case_id} tuner seeds",
        )
        result[case.case_id] = case_rows
    strict.check_equal(
        sum(len(case_rows) for case_rows in result.values()),
        len(rows),
        "held-out row assignment",
    )
    return result


def contender_from_row(row: dict[str, object]) -> dict[str, object]:
    selected_config = json.loads(str(row["selected_config_json"]))
    strict.require(
        isinstance(selected_config, dict), "selected config is not an object"
    )
    selected_config_sha256 = canonical_sha256(selected_config)
    strict.check_equal(
        selected_config_sha256,
        row["selected_config_sha256"],
        "selected config digest",
    )
    tuner_seed = int(row["tuner_seed"])
    return {
        "name": f"seed_{tuner_seed}",
        "origin_kind": "heldout_strict_result",
        "origin_variant": row["variant"],
        "origin_seq_len": row["seq_len"],
        "origin_tuner_seed": tuner_seed,
        "origin_result_path": row["result_path"],
        "origin_result_sha256": row["result_sha256"],
        "selected_config": selected_config,
        "selected_config_sha256": selected_config_sha256,
        "origin_selected_source_sha256": row["selected_source_sha256"],
        "expected_regenerated_source_sha256": row["selected_source_sha256"],
        "direct_search_tflops": float(row["median_tflops"]),
    }


def initial_case_records(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = rows_by_case(rows)
    result = []
    for case in CASES:
        contenders = [contender_from_row(row) for row in grouped[case.case_id]]
        implementation_count = len(contenders) + 1
        strict.require(
            ROUNDS % implementation_count == 0,
            f"{case.case_id}: rounds do not balance {implementation_count} implementations",
        )
        result.append(
            {
                "case_id": case.case_id,
                "variant": case.variant,
                "shape": case.shape,
                "physical_gpu": case.physical_gpu,
                "contender_policy": {
                    "schema_version": 1,
                    "kind": "heldout_winners",
                    "training_data": "none",
                },
                "contenders": contenders,
                "protocol": protocol_for_case(case),
                "result_path": f"results/{case.case_id}/{RESULT_FILENAME}",
                "completion_path": (f"results/{case.case_id}/{COMPLETION_FILENAME}"),
            }
        )
    return result


def validate_snapshot_set(root: Path, campaign: dict[str, Any]) -> None:
    records = campaign.get("source_snapshots")
    strict.require(isinstance(records, list), "missing source snapshots")
    strict.check_equal(
        [record.get("name") for record in records if isinstance(record, dict)],
        list(SNAPSHOT_FILENAMES),
        "source snapshot names",
    )
    for record in records:
        strict.require(isinstance(record, dict), "invalid source snapshot record")
        strict.check_equal(
            set(record), {"name", "sha256"}, "source snapshot record fields"
        )
        name = record["name"]
        path = root / "launcher" / name
        require_regular_file(path, f"source snapshot {name}", read_only=True)
        strict.check_equal(
            strict.file_sha256(path),
            require_sha256(record.get("sha256"), f"source snapshot {name}"),
            f"source snapshot {name}",
        )


def validate_measured_source_snapshots(root: Path, campaign: dict[str, Any]) -> None:
    records = campaign.get("measured_source_snapshots")
    expected = {
        BENCHMARK_RELATIVE.as_posix(): "measured_compare_attention_backends.py",
        ATTENTION_RELATIVE.as_posix(): "measured_attention.py",
    }
    strict.require(isinstance(records, list), "missing measured source snapshots")
    strict.require(
        all(isinstance(record, dict) for record in records),
        "invalid measured source snapshot record",
    )
    strict.check_equal(
        {record.get("source_path"): record.get("snapshot_name") for record in records},
        expected,
        "measured source snapshot set",
    )
    for record in records:
        strict.require(isinstance(record, dict), "invalid measured source snapshot")
        strict.check_equal(
            set(record),
            {"source_path", "snapshot_name", "sha256"},
            "measured source snapshot fields",
        )
        path = root / "launcher" / str(record["snapshot_name"])
        require_regular_file(path, "measured source snapshot", read_only=True)
        strict.check_equal(
            strict.file_sha256(path),
            require_sha256(record.get("sha256"), "measured source snapshot digest"),
            f"measured source snapshot {record['source_path']}",
        )


def validate_case_record(record: object, expected: dict[str, object]) -> None:
    strict.require(isinstance(record, dict), "invalid adjudication case record")
    strict.check_equal(record, expected, f"{expected['case_id']} declaration")


def validate_campaign(root: Path, *, deep_artifact_validation: bool) -> dict[str, Any]:
    campaign, declaration_sha256 = read_campaign(root)
    strict.check_equal(set(campaign), CAMPAIGN_FIELDS, "campaign fields")
    strict.check_equal(
        campaign.get("schema_version"), SCHEMA_VERSION, "campaign schema"
    )
    strict.check_equal(
        campaign.get("expected_commit"), EXPECTED_COMMIT, "campaign commit"
    )
    strict.check_equal(
        campaign.get("expected_cute_version"),
        EXPECTED_CUTE_VERSION,
        "campaign CuTe version",
    )
    campaign_nonce = campaign.get("campaign_nonce")
    strict.require(
        isinstance(campaign_nonce, str)
        and len(campaign_nonce) == 32
        and all(character in "0123456789abcdef" for character in campaign_nonce),
        f"invalid campaign nonce: {campaign_nonce!r}",
    )
    repo_root_value = campaign.get("repo_root")
    heldout_root_value = campaign.get("heldout_root")
    all8_root_value = campaign.get("all8_root")
    python_value = campaign.get("python_executable")
    for value, label in (
        (repo_root_value, "repository"),
        (heldout_root_value, "held-out root"),
        (all8_root_value, "all8 root"),
        (python_value, "Python executable"),
    ):
        strict.require(
            isinstance(value, str) and Path(value).is_absolute(),
            f"campaign {label} is not absolute: {value!r}",
        )
    validate_snapshot_set(root, campaign)
    validate_measured_source_snapshots(root, campaign)
    heldout_root = Path(heldout_root_value).resolve()
    all8_root = Path(all8_root_value).resolve()
    strict.require(
        heldout_root != all8_root
        and heldout_root not in all8_root.parents
        and all8_root not in heldout_root.parents,
        "held-out and all8 evidence roots overlap",
    )
    declared_rows = campaign.get("validated_heldout_rows")
    strict.require(
        isinstance(declared_rows, list) and len(declared_rows) == len(heldout.CASES),
        "invalid validated held-out row set",
    )
    validate_evidence_files(
        heldout_root,
        campaign.get("heldout_evidence"),
        expected_paths=expected_evidence_paths(declared_rows),
    )
    strict.check_equal(
        canonical_sha256(declared_rows),
        campaign.get("validated_heldout_rows_sha256"),
        "validated held-out row digest",
    )
    declared_all8_rows = campaign.get("validated_all8_rows")
    strict.require(
        isinstance(declared_all8_rows, list)
        and len(declared_all8_rows) == len(strict.CASES),
        "invalid validated all8 row set",
    )
    strict.check_equal(
        [
            (row.get("variant"), int(row.get("seq_len", -1)))
            for row in declared_all8_rows
            if isinstance(row, dict)
        ],
        [(variant, seq_len) for variant, seq_len, _gpu, _seed in strict.CASES],
        "validated all8 row order",
    )
    validate_evidence_files(
        all8_root,
        campaign.get("all8_evidence"),
        expected_paths=strict_result_evidence_paths(declared_all8_rows),
        label="all8 evidence",
    )
    strict.check_equal(
        canonical_sha256(declared_all8_rows),
        campaign.get("validated_all8_rows_sha256"),
        "validated all8 row digest",
    )
    require_sha256(
        campaign.get("all8_reference_manifest_sha256"),
        "all8 reference manifest digest",
    )
    if deep_artifact_validation:
        _campaign_path, live_rows, live_reference, live_reference_sha256 = (
            heldout.validate_individual_results(heldout_root, all8_root)
        )
        strict.check_equal(live_rows, declared_rows, "validated held-out rows")
        strict.check_equal(
            list(live_reference.values()),
            declared_all8_rows,
            "validated all8 rows",
        )
        strict.check_equal(
            live_reference_sha256,
            campaign["all8_reference_manifest_sha256"],
            "all8 reference manifest digest",
        )
    expected_cases = initial_case_records(declared_rows)
    cases = campaign.get("cases")
    strict.require(isinstance(cases, list), "missing adjudication cases")
    strict.check_equal(len(cases), len(expected_cases), "adjudication case count")
    for record, expected in zip(cases, expected_cases, strict=True):
        validate_case_record(record, expected)
    strict.check_equal(
        campaign.get("report_path"), "adjudication_report.json", "report path"
    )
    campaign["_declaration_sha256"] = declaration_sha256
    return campaign


def require_repo_identity(repo: Path, campaign: dict[str, Any]) -> None:
    strict.check_equal(str(repo), campaign["repo_root"], "measured repository")
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    strict.check_equal(commit, EXPECTED_COMMIT, "measured repository commit")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    strict.require(not status, f"measured repository is not clean:\n{status}")
    snapshots = {
        record["source_path"]: record
        for record in campaign["measured_source_snapshots"]
    }
    for relative in (BENCHMARK_RELATIVE, ATTENTION_RELATIVE):
        strict.check_equal(
            strict.file_sha256(repo / relative),
            snapshots[relative.as_posix()]["sha256"],
            f"measured source {relative}",
        )


def balanced_orders(names: list[str], rounds: int, seed: int) -> list[list[str]]:
    strict.require(names, "implementation list is empty")
    strict.require(
        rounds > 0 and rounds % len(names) == 0,
        "rounds must be a positive multiple of implementation count",
    )
    rng = random.Random(seed)
    result: list[list[str]] = []
    for _block in range(rounds // len(names)):
        base = names.copy()
        rng.shuffle(base)
        rotations = [base[offset:] + base[:offset] for offset in range(len(base))]
        rng.shuffle(rotations)
        result.extend(rotations)
    return result


def attention_flops(case: CaseDefinition) -> float:
    flops = 4.0 * 2 * 32 * case.seq_len**2 * 64
    return flops * (0.5 if case.causal else 1.0)


def percentile(values: list[float], fraction: float) -> float:
    strict.require(values, "percentile input is empty")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _performance_ratios(
    performance: dict[str, float], contender_names: list[str]
) -> tuple[float, float, float]:
    contenders = [performance[name] for name in contender_names]
    best = max(contenders)
    return (
        statistics.median(contenders) / best,
        min(contenders) / best,
        min(contenders) / performance["sdpa"],
    )


def summarize_measurements(
    raw_rounds: list[dict[str, Any]],
    contender_names: list[str],
    case: CaseDefinition,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    direct_tflops: list[float],
) -> dict[str, Any]:
    strict.require(raw_rounds and contender_names, "adjudication data is empty")
    strict.require(bootstrap_samples > 0, "bootstrap sample count must be positive")
    names = [*contender_names, "sdpa"]
    flops = attention_flops(case)
    timers: dict[str, Any] = {}
    for timer in ("event", "wall"):
        implementation: dict[str, Any] = {}
        performance: dict[str, float] = {}
        for name in names:
            values = [float(row["times"][name][f"{timer}_ms"]) for row in raw_rounds]
            median_ms = statistics.median(values)
            median_tflops = flops / (median_ms * 1e9)
            implementation[name] = {
                "median_ms": median_ms,
                "median_tflops": median_tflops,
                "raw_ms": values,
            }
            performance[name] = median_tflops
        median_best, worst_best, minimum_sdpa = _performance_ratios(
            performance, contender_names
        )
        rng = random.Random(bootstrap_seed ^ (0xE7E7 if timer == "event" else 0xA11))
        median_best_samples = []
        worst_best_samples = []
        minimum_sdpa_samples = []
        for _ in range(bootstrap_samples):
            sampled = [rng.randrange(len(raw_rounds)) for _row in raw_rounds]
            sampled_performance = {
                name: flops
                / (
                    statistics.median(
                        raw_rounds[index]["times"][name][f"{timer}_ms"]
                        for index in sampled
                    )
                    * 1e9
                )
                for name in names
            }
            sampled_ratios = _performance_ratios(sampled_performance, contender_names)
            median_best_samples.append(sampled_ratios[0])
            worst_best_samples.append(sampled_ratios[1])
            minimum_sdpa_samples.append(sampled_ratios[2])
        timers[timer] = {
            "implementations": implementation,
            "median_best_fraction": median_best,
            "median_best_paired_bootstrap_95_ci": [
                percentile(median_best_samples, 0.025),
                percentile(median_best_samples, 0.975),
            ],
            "worst_best_fraction": worst_best,
            "worst_best_paired_bootstrap_95_ci": [
                percentile(worst_best_samples, 0.025),
                percentile(worst_best_samples, 0.975),
            ],
            "minimum_seed_vs_sdpa_fraction": minimum_sdpa,
            "minimum_seed_vs_sdpa_paired_bootstrap_95_ci": [
                percentile(minimum_sdpa_samples, 0.025),
                percentile(minimum_sdpa_samples, 0.975),
            ],
            "performance_gate": {
                "median_best_threshold": MEDIAN_BEST_GATE,
                "median_best_passed": median_best >= MEDIAN_BEST_GATE,
                "worst_best_threshold": WORST_BEST_GATE,
                "worst_best_passed": worst_best >= WORST_BEST_GATE,
                "passed": median_best >= MEDIAN_BEST_GATE
                and worst_best >= WORST_BEST_GATE,
            },
        }
    direct_best = max(direct_tflops)
    direct_median_best = statistics.median(direct_tflops) / direct_best
    direct_worst_best = min(direct_tflops) / direct_best
    return {
        "flops": flops,
        "direct_search_measurements": {
            "tflops": direct_tflops,
            "median_best_fraction": direct_median_best,
            "worst_best_fraction": direct_worst_best,
            "performance_gate": {
                "median_best_threshold": MEDIAN_BEST_GATE,
                "median_best_passed": direct_median_best >= MEDIAN_BEST_GATE,
                "worst_best_threshold": WORST_BEST_GATE,
                "worst_best_passed": direct_worst_best >= WORST_BEST_GATE,
                "passed": direct_median_best >= MEDIAN_BEST_GATE
                and direct_worst_best >= WORST_BEST_GATE,
            },
        },
        "timers": timers,
        "performance_gate_passed": all(
            timer["performance_gate"]["passed"] for timer in timers.values()
        ),
    }


def case_record(campaign: dict[str, Any], case_id: str) -> dict[str, Any]:
    records = [record for record in campaign["cases"] if record["case_id"] == case_id]
    strict.require(len(records) == 1, f"missing campaign case: {case_id}")
    return records[0]


def validate_gpu_record(record: object, case: CaseDefinition, label: str) -> None:
    strict.require(isinstance(record, dict), f"missing {label} GPU record")
    strict.check_equal(
        set(record),
        {
            "physical_gpu",
            "name",
            "uuid",
            "power_limit_w",
            "active_compute_pids",
        },
        f"{label} GPU fields",
    )
    strict.check_equal(record.get("physical_gpu"), case.physical_gpu, f"{label} GPU")
    strict.check_equal(record.get("name"), "NVIDIA B200", f"{label} GPU model")
    strict.require(
        isinstance(record.get("uuid"), str) and record["uuid"].startswith("GPU-"),
        f"{label} GPU UUID is invalid",
    )
    power = strict.finite_float(record.get("power_limit_w"), f"{label} power")
    strict.require(abs(power - 750.0) <= 0.5, f"{label} power cap is {power}")
    strict.check_equal(record.get("active_compute_pids"), [], f"{label} competing PIDs")


def validate_case_output(
    root: Path,
    campaign: dict[str, Any],
    case_id: str,
    *,
    output_path: Path | None = None,
    require_completion: bool = True,
) -> dict[str, Any]:
    declaration = case_record(campaign, case_id)
    case = case_by_id(case_id)
    path = output_path or root / relative_path(
        declaration["result_path"], f"{case_id} result path"
    )
    require_regular_file(path, "adjudication result", read_only=True)
    payload = load_json_object(path)
    strict.check_equal(set(payload), RESULT_FIELDS, f"{path}: result fields")
    strict.check_equal(
        payload.get("schema_version"), RESULT_SCHEMA_VERSION, f"{path}: schema"
    )
    strict.check_equal(payload.get("status"), "MEASUREMENT_COMPLETE", f"{path}: status")
    strict.check_equal(payload.get("case_id"), case_id, f"{path}: case")
    strict.check_equal(payload.get("shape"), declaration["shape"], f"{path}: shape")
    strict.check_equal(
        payload.get("physical_gpu"), case.physical_gpu, f"{path}: physical GPU"
    )
    strict.check_equal(
        payload.get("measured_commit"), EXPECTED_COMMIT, f"{path}: measured commit"
    )
    strict.check_equal(
        payload.get("campaign_sha256"),
        campaign["_declaration_sha256"],
        f"{path}: campaign",
    )
    worker_hash = {
        record["name"]: record["sha256"] for record in campaign["source_snapshots"]
    }["remeasure_heldout_winners.py"]
    strict.check_equal(payload.get("worker_sha256"), worker_hash, f"{path}: worker")
    strict.check_equal(
        payload.get("source_snapshot_sha256"),
        {record["name"]: record["sha256"] for record in campaign["source_snapshots"]},
        f"{path}: source snapshots",
    )
    validate_gpu_record(payload.get("gpu_start"), case, f"{path}: start")
    validate_gpu_record(payload.get("gpu_end"), case, f"{path}: end")
    strict.check_equal(
        payload["gpu_start"]["uuid"],
        payload["gpu_end"]["uuid"],
        f"{path}: GPU UUID changed",
    )
    environment = payload.get("environment")
    strict.require(isinstance(environment, dict), f"{path}: missing environment")
    strict.check_equal(
        set(environment),
        {"torch_version", "cudnn_version", "cute_version", "helion_version"},
        f"{path}: environment fields",
    )
    strict.check_equal(
        environment.get("cute_version"), EXPECTED_CUTE_VERSION, f"{path}: CuTe version"
    )
    heldout_versions = {row["version"] for row in campaign["validated_heldout_rows"]}
    strict.require(len(heldout_versions) == 1, f"{path}: held-out versions differ")
    strict.check_equal(
        environment.get("helion_version"),
        next(iter(heldout_versions)),
        f"{path}: Helion version",
    )
    strict.require(
        all(
            isinstance(environment.get(field), str) and environment[field]
            for field in ("torch_version", "helion_version")
        ),
        f"{path}: incomplete software environment",
    )
    strict.strict_int(
        environment.get("cudnn_version"), f"{path}: cuDNN version", minimum=1
    )
    protocol = payload.get("protocol")
    strict.require(isinstance(protocol, dict), f"{path}: missing protocol")
    declared_protocol = declaration["protocol"]
    strict.check_equal(
        set(protocol),
        {*declared_protocol, "timing_repetitions", "sdpa_calibration_event_ms"},
        f"{path}: protocol fields",
    )
    for field, expected in declared_protocol.items():
        strict.check_equal(protocol.get(field), expected, f"{path}: protocol {field}")
    repetitions = strict.strict_int(
        protocol.get("timing_repetitions"), f"{path}: timing repetitions", minimum=1
    )
    strict.require(
        repetitions <= MAX_TIMING_REPETITIONS,
        f"{path}: timing repetitions exceed cap",
    )
    calibration = protocol.get("sdpa_calibration_event_ms")
    strict.require(
        isinstance(calibration, list) and len(calibration) == 3,
        f"{path}: invalid timing calibration",
    )
    calibration_values = [
        strict.finite_float(value, f"{path}: timing calibration", positive=True)
        for value in calibration
    ]
    expected_repetitions = min(
        MAX_TIMING_REPETITIONS,
        max(
            1,
            math.ceil(
                TARGET_TIMING_MS / max(statistics.median(calibration_values), 1e-6)
            ),
        ),
    )
    strict.check_equal(repetitions, expected_repetitions, f"{path}: timing calibration")
    contenders = declaration["contenders"]
    contender_names = [contender["name"] for contender in contenders]
    selected_sources = payload.get("selected_sources")
    strict.require(
        isinstance(selected_sources, list), f"{path}: missing selected sources"
    )
    strict.check_equal(
        [source.get("name") for source in selected_sources],
        contender_names,
        f"{path}: selected source names",
    )
    source_directory = path.parent / "generated_sources"
    strict.require(
        source_directory.is_dir() and not source_directory.is_symlink(),
        f"{path}: invalid generated source directory",
    )
    expected_archives = set()
    for source, contender in zip(selected_sources, contenders, strict=True):
        strict.require(isinstance(source, dict), f"{path}: invalid selected source")
        strict.check_equal(
            set(source),
            {
                "name",
                "origin_kind",
                "origin_result_sha256",
                "origin_selected_source_sha256",
                "selected_config_sha256",
                "regenerated_source_sha256",
                "compiled_source_sha256",
                "archive_path",
            },
            f"{path}: selected source fields",
        )
        strict.check_equal(
            source.get("origin_kind"),
            contender["origin_kind"],
            f"{path}: origin kind",
        )
        for field in (
            "origin_result_sha256",
            "selected_config_sha256",
            "regenerated_source_sha256",
            "compiled_source_sha256",
        ):
            require_sha256(source.get(field), f"{path}: {field}")
        strict.check_equal(
            source.get("origin_result_sha256"),
            contender["origin_result_sha256"],
            f"{path}: origin result",
        )
        strict.check_equal(
            source.get("origin_selected_source_sha256"),
            contender["origin_selected_source_sha256"],
            f"{path}: origin selected source",
        )
        strict.check_equal(
            source.get("selected_config_sha256"),
            contender["selected_config_sha256"],
            f"{path}: selected config",
        )
        expected_source = contender["expected_regenerated_source_sha256"]
        strict.check_equal(
            source.get("regenerated_source_sha256"),
            expected_source,
            f"{path}: regenerated source",
        )
        strict.check_equal(
            source.get("compiled_source_sha256"),
            expected_source,
            f"{path}: compiled source",
        )
        archive_relative = relative_path(
            source.get("archive_path"), f"{path}: source archive"
        )
        strict.check_equal(
            archive_relative,
            Path("generated_sources") / f"{contender['name']}.py.txt",
            f"{path}: source archive path",
        )
        archive = path.parent / archive_relative
        require_regular_file(archive, "regenerated source archive", read_only=True)
        expected_archives.add(archive)
        strict.check_equal(
            strict.file_sha256(archive),
            expected_source,
            f"{path}: regenerated source archive",
        )
    discovered_archives = set(source_directory.iterdir())
    strict.check_equal(
        discovered_archives, expected_archives, f"{path}: source archive set"
    )
    correctness = payload.get("correctness")
    all_names = [*contender_names, "sdpa"]
    strict.require(isinstance(correctness, dict), f"{path}: missing correctness")
    strict.check_equal(set(correctness), set(all_names), f"{path}: correctness names")
    element_count = 2 * 32 * case.seq_len * 64
    for name in all_names:
        record = correctness[name]
        strict.require(
            isinstance(record, dict), f"{path}: invalid correctness for {name}"
        )
        numerics = record.get("numerics")
        repeats = record.get("repeatability")
        strict.require(
            isinstance(numerics, dict)
            and numerics.get("passed") is True
            and numerics.get("count") == element_count,
            f"{path}: {name} failed numerics",
        )
        strict.check_equal(
            set(numerics),
            {
                "count",
                "close_count",
                "max_abs",
                "rmse",
                "nrmse",
                "actual_nonfinite",
                "atol",
                "rtol",
                "passed",
            },
            f"{path}: {name} numeric fields",
        )
        strict.check_equal(
            numerics.get("close_count"), element_count, f"{path}: {name} close count"
        )
        strict.check_equal(
            numerics.get("actual_nonfinite"), 0, f"{path}: {name} nonfinite output"
        )
        strict.check_equal(numerics.get("atol"), 5e-2, f"{path}: {name} atol")
        strict.check_equal(numerics.get("rtol"), 2e-2, f"{path}: {name} rtol")
        for metric in ("max_abs", "rmse", "nrmse"):
            value = strict.finite_float(
                numerics.get(metric), f"{path}: {name} {metric}"
            )
            strict.require(value >= 0, f"{path}: {name} negative {metric}")
        strict.require(
            isinstance(repeats, list)
            and len(repeats) == REPEATABILITY_LAUNCHES - 1
            and all(
                isinstance(repeat, dict)
                and repeat.get("passed") is True
                and repeat.get("different") == 0
                and repeat.get("count") == element_count
                for repeat in repeats
            ),
            f"{path}: {name} failed exact repeatability",
        )
        for repeat_index, repeat in enumerate(repeats, 1):
            strict.check_equal(
                set(repeat),
                {"count", "different", "passed", "repeat_index"},
                f"{path}: {name} repeatability fields",
            )
            strict.check_equal(
                repeat.get("repeat_index"),
                repeat_index,
                f"{path}: {name} repeatability index",
            )
    raw_rounds = payload.get("raw_rounds")
    rounds = int(protocol["rounds"])
    strict.require(
        isinstance(raw_rounds, list) and len(raw_rounds) == rounds,
        f"{path}: raw round count",
    )
    expected_orders = balanced_orders(all_names, rounds, int(protocol["order_seed"]))
    for round_index, (record, expected_order) in enumerate(
        zip(raw_rounds, expected_orders, strict=True)
    ):
        strict.require(isinstance(record, dict), f"{path}: invalid raw round")
        strict.check_equal(
            set(record),
            {"round_index", "order", "times"},
            f"{path}: raw round fields",
        )
        strict.check_equal(
            record.get("round_index"), round_index, f"{path}: round index"
        )
        strict.check_equal(record.get("order"), expected_order, f"{path}: round order")
        times = record.get("times")
        strict.require(isinstance(times, dict), f"{path}: missing round timings")
        strict.check_equal(set(times), set(all_names), f"{path}: timed implementations")
        for name in all_names:
            timing = times[name]
            strict.require(
                isinstance(timing, dict), f"{path}: invalid timing for {name}"
            )
            strict.check_equal(
                set(timing), {"event_ms", "wall_ms"}, f"{path}: timing fields"
            )
            strict.finite_float(
                timing.get("event_ms"), f"{path}: event time", positive=True
            )
            strict.finite_float(
                timing.get("wall_ms"), f"{path}: wall time", positive=True
            )
    direct_tflops = [float(row) for row in payload.get("direct_search_tflops", [])]
    strict.check_equal(
        direct_tflops,
        [float(contender["direct_search_tflops"]) for contender in contenders],
        f"{path}: direct search measurements",
    )
    computed = summarize_measurements(
        raw_rounds,
        contender_names,
        case,
        bootstrap_samples=int(protocol["bootstrap_samples"]),
        bootstrap_seed=int(protocol["bootstrap_seed"]),
        direct_tflops=direct_tflops,
    )
    strict.check_equal(payload.get("summary"), computed, f"{path}: summary")
    completion_path = path.with_name(COMPLETION_FILENAME)
    if output_path is None:
        strict.check_equal(
            completion_path,
            root / relative_path(declaration["completion_path"], "completion path"),
            f"{path}: completion path",
        )
    if require_completion:
        require_regular_file(
            completion_path, "adjudication completion marker", read_only=True
        )
        completion = load_json_object(completion_path)
        strict.check_equal(
            set(completion),
            {
                "schema_version",
                "status",
                "case_id",
                "campaign_sha256",
                "worker_sha256",
                "result_sha256",
            },
            f"{path}: completion marker fields",
        )
        strict.check_equal(
            completion.get("schema_version"), 1, f"{path}: completion schema"
        )
        strict.check_equal(
            completion.get("status"),
            "POSTCONDITIONS_PASSED",
            f"{path}: completion status",
        )
        strict.check_equal(
            completion.get("case_id"), case_id, f"{path}: completion case"
        )
        strict.check_equal(
            completion.get("campaign_sha256"),
            campaign["_declaration_sha256"],
            f"{path}: completion campaign",
        )
        strict.check_equal(
            completion.get("worker_sha256"), worker_hash, f"{path}: completion worker"
        )
        strict.check_equal(
            completion.get("result_sha256"),
            strict.file_sha256(path),
            f"{path}: completed result digest",
        )
        expected_case_files = {path, source_directory, completion_path}
    else:
        strict.require(
            not completion_path.exists() and not completion_path.is_symlink(),
            f"{path}: premature completion marker",
        )
        expected_case_files = {path, source_directory}
    strict.check_equal(
        set(path.parent.iterdir()),
        expected_case_files,
        f"{path}: case output file set",
    )
    return payload


def build_report(root: Path, campaign: dict[str, Any]) -> dict[str, Any]:
    expected_results = {
        root / relative_path(record["result_path"], "campaign result path")
        for record in campaign["cases"]
    }
    discovered_results = set((root / "results").rglob(RESULT_FILENAME))
    strict.check_equal(discovered_results, expected_results, "adjudication result set")
    results: list[dict[str, Any]] = []
    for case in CASES:
        payload = validate_case_output(root, campaign, case.case_id)
        results.append(
            {
                "case_id": case.case_id,
                "result_path": case_record(campaign, case.case_id)["result_path"],
                "result_sha256": strict.file_sha256(
                    root / case_record(campaign, case.case_id)["result_path"]
                ),
                "completion_sha256": strict.file_sha256(
                    root / case_record(campaign, case.case_id)["completion_path"]
                ),
                "summary": payload["summary"],
            }
        )
    direct_passed = all(
        result["summary"]["direct_search_measurements"]["performance_gate"]["passed"]
        for result in results
    )
    cross_measured_passed = all(
        result["summary"]["performance_gate_passed"] for result in results
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "MEASUREMENT_COMPLETE",
        "direct_search_gate_status": "PASS" if direct_passed else "FAIL",
        "cross_measured_gate_status": "PASS" if cross_measured_passed else "FAIL",
        "performance_gate_status": "PASS" if cross_measured_passed else "FAIL",
        "campaign_sha256": campaign["_declaration_sha256"],
        "expected_commit": EXPECTED_COMMIT,
        "expected_cute_version": EXPECTED_CUTE_VERSION,
        "all8_linked_input_validation": "PASS",
        "all8_reference_manifest_sha256": campaign["all8_reference_manifest_sha256"],
        "thresholds": {
            "median_best_fraction": MEDIAN_BEST_GATE,
            "worst_best_fraction": WORST_BEST_GATE,
        },
        "results": results,
    }


def validate_complete_campaign(root: Path) -> dict[str, Any]:
    campaign = validate_campaign(root, deep_artifact_validation=True)
    report_path = root / str(campaign["report_path"])
    require_regular_file(report_path, "adjudication report", read_only=True)
    report = load_json_object(report_path)
    strict.check_equal(report, build_report(root, campaign), "adjudication report")
    return report


def source_snapshot_hashes(campaign: dict[str, Any]) -> dict[str, str]:
    return {
        str(record["name"]): str(record["sha256"])
        for record in campaign["source_snapshots"]
    }


def validate_runtime_module_paths(root: Path, modules: list[ModuleType]) -> None:
    launcher = (root / "launcher").resolve()
    for module in modules:
        module_path = Path(str(module.__file__)).resolve()
        strict.check_equal(
            module_path.parent, launcher, f"runtime module {module.__name__} location"
        )


def sanitized_environment(base: dict[str, str]) -> dict[str, str]:
    prefixes = (
        "HELION_",
        "CUTE_DSL_",
        "CUDA_MPS_",
        "CUDNN_",
        "TORCH_CUDNN_",
        "TRITON_",
        "TORCHINDUCTOR_",
        "PYTORCH_TUNABLEOP_",
        "CUDA_CACHE_",
    )
    exact = {
        "ADJUDICATION_CASE_LOCK_FD",
        "CUBLAS_FORCE_TF32",
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_AUTO_BOOST",
        "CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "CUDA_DEVICE_ORDER",
        "CUDA_DISABLE_PTX_JIT",
        "CUDA_FORCE_PTX_JIT",
        "CUDA_LAUNCH_BLOCKING",
        "CUDA_MANAGED_FORCE_DEVICE_ALLOC",
        "CUDA_MODULE_LOADING",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_TF32_OVERRIDE",
        "PYTHONPATH",
        "PYTHONPYCACHEPREFIX",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
        "XDG_CACHE_HOME",
    }
    return {
        key: value
        for key, value in base.items()
        if key not in exact and not any(key.startswith(prefix) for prefix in prefixes)
    }
