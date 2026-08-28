from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paired_worker import atomic_write_json
from paired_worker import sha256
from paired_worker import validate_artifact_set
from run_all8 import HARNESS_PATHS
from run_all8 import aggregate_results
from run_all8 import resolve_output_reference
from run_all8 import validate_harness_sha256
from run_all8 import validate_strict_artifact_identities


def check_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: expected {expected!r}, got {actual!r}")


def validate_manifest_files(manifest: dict[str, Any], output_dir: Path) -> None:
    check_equal(manifest.get("status"), "PASS", "run manifest status")
    for record in manifest.get("records", []):
        output = resolve_output_reference(output_dir, record["output"])
        source = resolve_output_reference(output_dir, record["generated_source"])
        check_equal(sha256(output), record["output_sha256"], f"{output} digest")
        check_equal(
            sha256(source),
            record["generated_source_sha256"],
            f"{source} digest",
        )


def resolve_artifact_root(
    requested: Path | None, manifest: dict[str, Any], output_dir: Path
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    candidates = []
    logical = manifest.get("artifact_root")
    if isinstance(logical, str):
        logical_path = Path(logical).expanduser()
        candidates.append(
            logical_path if logical_path.is_absolute() else output_dir / logical_path
        )
    original = manifest.get("artifact_root_at_run")
    if isinstance(original, str):
        candidates.append(Path(original).expanduser())
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_dir():
            return resolved
    locations = ", ".join(str(path.resolve()) for path in candidates) or "none"
    raise RuntimeError(
        f"strict artifacts were not found at recorded locations ({locations}); "
        "pass --artifact-root with their relocated directory"
    )


def recorded_bootstrap_settings(
    static_validation: dict[str, Any],
    requested_samples: int | None,
    requested_seed: int | None,
) -> tuple[int, int]:
    bootstrap = static_validation.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise RuntimeError("static validation lacks a bootstrap protocol")
    samples = bootstrap.get("samples")
    seed = bootstrap.get("base_seed")
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise RuntimeError("static validation has an invalid bootstrap sample count")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise RuntimeError("static validation has an invalid bootstrap base seed")
    check_equal(
        bootstrap.get("method"),
        "resample paired log ratios within each campaign/shape stratum",
        "static bootstrap method",
    )
    if requested_samples is not None:
        check_equal(
            requested_samples,
            samples,
            "--bootstrap-samples/static validation",
        )
    if requested_seed is not None:
        check_equal(requested_seed, seed, "--bootstrap-seed/static validation")
    return samples, seed


def validated_evidence_paths(
    output_dir: Path,
    manifest_path: Path,
    static_validation_path: Path,
    manifest: dict[str, Any],
    validated_strict: dict[object, dict[str, Any]],
) -> set[Path]:
    evidence = {
        manifest_path.resolve(),
        static_validation_path.resolve(),
        *(path.resolve() for path in HARNESS_PATHS.values()),
    }
    for record in manifest.get("records", []):
        evidence.add(resolve_output_reference(output_dir, record["output"]))
        evidence.add(resolve_output_reference(output_dir, record["generated_source"]))
    for provenance in validated_strict.values():
        result_path = Path(provenance["search_result_path"])
        evidence.add(result_path.resolve())
        evidence.update(
            result_path.with_name(filename).resolve()
            for filename in (
                "autotune.csv",
                "autotune.meta.jsonl",
                "autotune.sources.csv",
            )
        )
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--campaign-seed", type=int, action="append")
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    manifest_path = args.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validate_manifest_files(manifest, args.output_dir)
    static_validation_path = args.output_dir / "static_validation.json"
    static_validation = json.loads(static_validation_path.read_text())
    check_equal(
        static_validation.get("harness_sha256"),
        manifest.get("harness_sha256"),
        "static/run harness identity",
    )
    validate_harness_sha256(manifest.get("harness_sha256"))
    bootstrap_samples, bootstrap_seed = recorded_bootstrap_settings(
        static_validation,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    campaign_seeds = tuple(args.campaign_seed or manifest["campaign_seeds"])
    if len(campaign_seeds) != 2 or len(set(campaign_seeds)) != 2:
        raise SystemExit("exactly two distinct campaign seeds are required")
    artifact_root = resolve_artifact_root(args.artifact_root, manifest, args.output_dir)
    validated = validate_artifact_set(artifact_root)
    validate_strict_artifact_identities(manifest.get("strict_artifacts"), validated)
    output = (args.output or args.output_dir / "all8_paired_raw.json").resolve()
    if output in validated_evidence_paths(
        args.output_dir,
        manifest_path,
        static_validation_path,
        manifest,
        validated,
    ):
        raise RuntimeError(
            f"combined output collides with validated evidence input: {output}"
        )
    combined = aggregate_results(
        args.output_dir,
        campaign_seeds,
        bootstrap_samples,
        bootstrap_seed,
        manifest,
    )
    combined["run_manifest_sha256"] = sha256(manifest_path)
    combined["static_validation_sha256"] = sha256(static_validation_path)
    atomic_write_json(output, combined)


if __name__ == "__main__":
    main()
