from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import statistics

import build_strict_manifest as strict

CASES = (
    ("dense", 81920, 7, 2026082301),
    ("dense", 81920, 7, 2026082302),
    ("dense", 81920, 7, 2026082303),
    ("dense", 81920, 7, 2026082304),
    ("dense", 81920, 7, 2026082305),
    ("causal", 196608, 6, 2026082311),
    ("causal", 196608, 6, 2026082312),
    ("causal", 196608, 6, 2026082313),
    ("causal", 196608, 6, 2026082314),
    ("causal", 196608, 6, 2026082315),
)
MINIMUM_MEDIAN_BEST_SEED_FRACTION = 0.998
MINIMUM_WORST_BEST_SEED_FRACTION = 0.995
MINIMUM_BRACKETING_REFERENCE_FRACTION = 0.99
EXPECTED_COMMIT = "c3e36b65d69681c23e053042b0bc21e2331bad17"
CAMPAIGN_SCHEMA_VERSION = 3
CAMPAIGN_FIELDS = (
    "schema_version",
    "expected_commit",
    "python_executable",
    "all8_launcher_sha256",
    "heldout_launcher_sha256",
    "variant",
    "seq_len",
    "physical_gpu",
    "input_seed",
    "tuner_seed",
    "result_path",
)
HELDOUT_MANIFEST_FIELDS = (
    *strict.MANIFEST_FIELDS,
    "campaign_manifest_path",
    "campaign_manifest_sha256",
    "all8_reference_manifest_sha256",
    "bracketing_reference_tflops",
    "minimum_required_tflops",
)
BRACKETING_CASES = {
    ("dense", 81920): (("dense", 65536), ("dense", 131072)),
    ("causal", 196608): (("causal", 131072), ("causal", 262144)),
}


def validate_versions(rows: list[dict[str, str]], label: str) -> str:
    versions = {row["version"] for row in rows}
    strict.require(len(versions) == 1, f"{label} changed versions: {sorted(versions)}")
    version = next(iter(versions))
    strict.require(
        strict.helion_cute_version_matches_commit(version, EXPECTED_COMMIT),
        f"{label} version {version!r} does not identify {EXPECTED_COMMIT}",
    )
    return version


def result_path(root: Path, variant: str, seq_len: int, tuner_seed: int) -> Path:
    return (
        root / variant / f"seed_{tuner_seed}" / f"{variant}_s{seq_len}" / "result.json"
    )


def expected_campaign_rows(
    python_executable: str,
    all8_launcher_sha256: str,
    heldout_launcher_sha256: str,
) -> list[dict[str, str]]:
    return [
        {
            "schema_version": str(CAMPAIGN_SCHEMA_VERSION),
            "expected_commit": EXPECTED_COMMIT,
            "python_executable": python_executable,
            "all8_launcher_sha256": all8_launcher_sha256,
            "heldout_launcher_sha256": heldout_launcher_sha256,
            "variant": variant,
            "seq_len": str(seq_len),
            "physical_gpu": str(physical_gpu),
            "input_seed": str(strict.EXPECTED_INPUT_SEED),
            "tuner_seed": str(tuner_seed),
            "result_path": result_path(
                Path("."), variant, seq_len, tuner_seed
            ).as_posix(),
        }
        for variant, seq_len, physical_gpu, tuner_seed in CASES
    ]


def validate_campaign(root: Path) -> tuple[Path, set[Path]]:
    campaign = root / "campaign.csv"
    strict.require(
        campaign.is_file() and not campaign.is_symlink(),
        f"missing regular held-out campaign manifest: {campaign}",
    )
    with campaign.open(newline="") as handle:
        reader = csv.DictReader(handle)
        strict.check_equal(
            tuple(reader.fieldnames or ()), CAMPAIGN_FIELDS, "campaign fields"
        )
        rows = list(reader)

    python_executables = {row["python_executable"] for row in rows}
    strict.require(
        len(python_executables) == 1,
        f"campaign changed Python executable: {sorted(python_executables)}",
    )
    python_executable = next(iter(python_executables))
    strict.require(
        bool(python_executable) and Path(python_executable).is_absolute(),
        f"campaign Python executable is not absolute: {python_executable!r}",
    )
    launcher_hashes: dict[str, str] = {}
    for field in ("all8_launcher_sha256", "heldout_launcher_sha256"):
        values = {row[field] for row in rows}
        strict.require(
            len(values) == 1,
            f"campaign changed {field}: {sorted(values)}",
        )
        value = next(iter(values))
        strict.require(
            len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"campaign {field} is not a lowercase SHA-256: {value!r}",
        )
        launcher_hashes[field] = value
    strict.check_equal(
        rows,
        expected_campaign_rows(
            python_executable,
            launcher_hashes["all8_launcher_sha256"],
            launcher_hashes["heldout_launcher_sha256"],
        ),
        "predeclared held-out campaign",
    )

    for field, filename in (
        ("all8_launcher_sha256", "run_strict_all8.sh"),
        ("heldout_launcher_sha256", "run_strict_heldout.sh"),
    ):
        launcher = root / "launcher" / filename
        strict.require(
            launcher.is_file() and not launcher.is_symlink(),
            f"missing regular held-out launcher snapshot: {launcher}",
        )
        strict.require(
            launcher.stat().st_mode & 0o222 == 0,
            f"held-out launcher snapshot is writable: {launcher}",
        )
        strict.check_equal(
            strict.file_sha256(launcher),
            launcher_hashes[field],
            f"held-out launcher snapshot {filename}",
        )

    expected_results = {root / row["result_path"] for row in rows}
    discovered_results = set(root.rglob("result.json"))
    symlink_results = sorted(path for path in discovered_results if path.is_symlink())
    strict.require(
        not symlink_results,
        f"held-out result set contains symlinks: {symlink_results}",
    )
    strict.check_equal(
        discovered_results,
        expected_results,
        "held-out result set",
    )
    campaign_mtime_ns = campaign.stat().st_mtime_ns
    for path in expected_results:
        strict.require(
            path.stat().st_mtime_ns >= campaign_mtime_ns,
            f"result predates predeclared campaign: {path}",
        )
    return campaign, expected_results


def load_all8_reference(
    artifact_root: Path,
) -> tuple[dict[tuple[str, int], dict[str, str]], str]:
    contents = strict.build_manifest(artifact_root)
    reader = csv.DictReader(io.StringIO(contents))
    strict.check_equal(
        tuple(reader.fieldnames or ()), strict.MANIFEST_FIELDS, "all8 manifest fields"
    )
    rows = list(reader)

    expected_cases = {
        (variant, seq_len) for variant, seq_len, _gpu, _seed in strict.CASES
    }
    by_case: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        case = (row["variant"], int(row["seq_len"]))
        strict.require(case not in by_case, f"duplicate all8 manifest case: {case}")
        by_case[case] = row
    strict.check_equal(set(by_case), expected_cases, "all8 manifest cases")
    validate_versions(rows, "all8 reference")
    strict.check_equal(
        {int(row["input_seed"]) for row in rows},
        {strict.EXPECTED_INPUT_SEED},
        "all8 reference input seeds",
    )
    for case, row in by_case.items():
        try:
            median_tflops = float(row["median_tflops"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"all8 reference {case} median_tflops is not numeric: "
                f"{row['median_tflops']!r}"
            ) from exc
        strict.finite_float(
            median_tflops, f"all8 reference {case} median_tflops", positive=True
        )
    return by_case, strict.sha256_bytes(contents.encode())


def validate_individual_results(
    artifact_root: Path,
    all8_artifact_root: Path,
) -> tuple[
    Path,
    list[dict[str, object]],
    dict[tuple[str, int], dict[str, str]],
    str,
]:
    """Validate linked evidence while deferring only cross-seed throughput gates."""
    root = artifact_root.expanduser().resolve()
    campaign, _expected_results = validate_campaign(root)
    reference, all8_reference_sha256 = load_all8_reference(all8_artifact_root)
    rows: list[dict[str, object]] = []
    saved_cases = strict.EXPECTED_CASES
    try:
        for variant, seq_len, physical_gpu, tuner_seed in CASES:
            strict.EXPECTED_CASES = {
                (variant, seq_len): {
                    "physical_gpu": physical_gpu,
                    "tuner_seed": tuner_seed,
                }
            }
            row = strict.validate_result(
                root,
                result_path(root, variant, seq_len, tuner_seed),
                variant,
                seq_len,
            )
            row["case"] = f"{variant}_{seq_len}_seed_{tuner_seed}"
            rows.append(row)
    finally:
        strict.EXPECTED_CASES = saved_cases

    heldout_version = validate_versions(rows, "held-out")
    strict.check_equal(
        heldout_version,
        validate_versions(list(reference.values()), "all8 reference"),
        "held-out versus all8 version",
    )
    strict.check_equal(
        {row["input_seed"] for row in rows},
        {strict.EXPECTED_INPUT_SEED},
        "held-out input seeds",
    )
    for variant, seq_len in BRACKETING_CASES:
        shape_rows = [
            row
            for row in rows
            if row["variant"] == variant and row["seq_len"] == seq_len
        ]
        coverage_hashes = {str(row["coverage_design_sha256"]) for row in shape_rows}
        seed_policies = {str(row["compiler_seed_policy_json"]) for row in shape_rows}
        terminal_policy_hashes = {
            str(row["terminal_refinement_policy_sha256"]) for row in shape_rows
        }
        terminal_surface_hashes = {
            str(row["terminal_coordinate_surface_sha256"]) for row in shape_rows
        }
        bracketing_rows = [
            reference[case] for case in BRACKETING_CASES[(variant, seq_len)]
        ]
        reference_coverage_hashes = {
            row["coverage_design_sha256"] for row in bracketing_rows
        }
        reference_seed_policies = {
            row["compiler_seed_policy_json"] for row in bracketing_rows
        }
        reference_terminal_policy_hashes = {
            row["terminal_refinement_policy_sha256"] for row in bracketing_rows
        }
        reference_terminal_surface_hashes = {
            row["terminal_coordinate_surface_sha256"] for row in bracketing_rows
        }
        strict.require(
            len(coverage_hashes) == 1,
            f"{variant} held-out seeds changed structural coverage design: "
            f"{sorted(coverage_hashes)}",
        )
        strict.require(
            len(seed_policies) == 1,
            f"{variant} held-out tuner seeds changed compiler seed policy: "
            f"{sorted(seed_policies)}",
        )
        strict.require(
            len(terminal_policy_hashes) == 1,
            f"{variant} held-out tuner seeds changed terminal refinement policy: "
            f"{sorted(terminal_policy_hashes)}",
        )
        strict.require(
            len(terminal_surface_hashes) == 1,
            f"{variant} held-out tuner seeds changed terminal coordinate surface: "
            f"{sorted(terminal_surface_hashes)}",
        )
        strict.check_equal(
            coverage_hashes,
            reference_coverage_hashes,
            f"{variant} held-out versus bracketing all8 structural coverage",
        )
        strict.check_equal(
            seed_policies,
            reference_seed_policies,
            f"{variant} held-out versus bracketing all8 compiler seed policy",
        )
        strict.check_equal(
            terminal_policy_hashes,
            reference_terminal_policy_hashes,
            f"{variant} held-out versus bracketing all8 terminal refinement policy",
        )
        strict.check_equal(
            terminal_surface_hashes,
            reference_terminal_surface_hashes,
            f"{variant} held-out versus bracketing all8 terminal coordinate surface",
        )
        throughputs = [float(row["median_tflops"]) for row in shape_rows]
        bracketing_reference = min(
            float(row["median_tflops"]) for row in bracketing_rows
        )
        minimum_required = bracketing_reference * MINIMUM_BRACKETING_REFERENCE_FRACTION
        strict.require(
            min(throughputs) >= minimum_required,
            f"{variant} held-out searches below independent bracketing reference: "
            f"throughputs={throughputs}, reference={bracketing_reference}, "
            f"minimum={minimum_required}",
        )
        for row in shape_rows:
            row["campaign_manifest_path"] = "campaign.csv"
            row["campaign_manifest_sha256"] = strict.file_sha256(campaign)
            row["all8_reference_manifest_sha256"] = all8_reference_sha256
            row["bracketing_reference_tflops"] = bracketing_reference
            row["minimum_required_tflops"] = minimum_required
    return campaign, rows, reference, all8_reference_sha256


def build_manifest(artifact_root: Path, all8_artifact_root: Path) -> str:
    _campaign, rows, _reference, _all8_reference_sha256 = validate_individual_results(
        artifact_root, all8_artifact_root
    )
    for variant, seq_len in BRACKETING_CASES:
        shape_rows = [
            row
            for row in rows
            if row["variant"] == variant and row["seq_len"] == seq_len
        ]
        throughputs = [float(row["median_tflops"]) for row in shape_rows]
        best_throughput = max(throughputs)
        median_throughput = statistics.median(throughputs)
        strict.require(
            median_throughput >= best_throughput * MINIMUM_MEDIAN_BEST_SEED_FRACTION,
            f"{variant} held-out median seed robustness below "
            f"{MINIMUM_MEDIAN_BEST_SEED_FRACTION:.1%} of best: {throughputs}",
        )
        strict.require(
            min(throughputs) >= best_throughput * MINIMUM_WORST_BEST_SEED_FRACTION,
            f"{variant} held-out worst seed robustness below "
            f"{MINIMUM_WORST_BEST_SEED_FRACTION:.1%} of best: {throughputs}",
        )

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=HELDOUT_MANIFEST_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate five-seed held-out full-autotune artifacts."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--all8-artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def validate_output_path(output: Path, roots: tuple[tuple[str, Path], ...]) -> None:
    for label, root in roots:
        strict.require(
            output != root and root not in output.parents,
            f"manifest output must be outside the {label} artifact root: {output}",
        )


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    heldout_root = args.artifact_root.expanduser().resolve()
    all8_root = args.all8_artifact_root.expanduser().resolve()
    validate_output_path(output, (("held-out", heldout_root), ("all8", all8_root)))
    contents = build_manifest(heldout_root, all8_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(contents)
    print(f"wrote {len(CASES)} validated rows to {output}")


if __name__ == "__main__":
    main()
