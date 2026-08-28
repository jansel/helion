from __future__ import annotations

import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_heldout_manifest as heldout
import build_strict_manifest as strict

TEST_VERSION = "Helion 1.4.0.dev157+gc3e36b65d; CuTe 4.7.0"


def compiler_seed_policy_json(variant: str) -> str:
    seed_ids = [("a" if variant == "dense" else "b") * 16]
    return strict.canonical_json(
        {
            "schema_version": 1,
            "kind": "canonical_cute_flash",
            "heuristic_names": ["cute_flash_attention"],
            "raw_config_count": 1,
            "effective_config_ids": seed_ids,
            "effective_config_ids_sha256": strict.canonical_sha256(seed_ids),
            "timeout_retry_repetitions": 3,
        }
    )


class HeldoutManifestTests(unittest.TestCase):
    def fake_reference(
        self,
        *,
        reference_tflops: dict[tuple[str, int], float] | None = None,
        reference_coverage: dict[str, str] | None = None,
        reference_seed_policy: dict[str, str] | None = None,
        reference_terminal_policy: dict[str, str] | None = None,
        reference_terminal_surface: dict[str, str] | None = None,
        version: str = TEST_VERSION,
    ) -> dict[tuple[str, int], dict[str, str]]:
        return {
            (variant, seq_len): {
                "variant": variant,
                "seq_len": str(seq_len),
                "version": version,
                "input_seed": str(heldout.strict.EXPECTED_INPUT_SEED),
                "coverage_design_sha256": (reference_coverage or {}).get(
                    variant, f"{variant}-coverage"
                ),
                "compiler_seed_policy_json": (reference_seed_policy or {}).get(
                    variant, compiler_seed_policy_json(variant)
                ),
                "terminal_refinement_policy_sha256": (
                    reference_terminal_policy or {}
                ).get(variant, f"{variant}-terminal-policy"),
                "terminal_coordinate_surface_sha256": (
                    reference_terminal_surface or {}
                ).get(variant, f"{variant}-surface"),
                "median_tflops": str(
                    (reference_tflops or {}).get((variant, seq_len), 750.0)
                ),
            }
            for variant, seq_len, _gpu, _seed in heldout.strict.CASES
        }

    def test_versions_accept_longer_git_abbreviation(self) -> None:
        version = (
            f"Helion 1.4.0.dev1+g{heldout.EXPECTED_COMMIT[:9]}; "
            f"CuTe {strict.EXPECTED_CUTE_VERSION}"
        )
        self.assertEqual(
            heldout.validate_versions([{"version": version}], "heldout"), version
        )

    def test_campaign_rows_pin_current_generalization_commit(self) -> None:
        rows = heldout.expected_campaign_rows("/python", "a" * 64, "b" * 64)

        self.assertEqual(
            heldout.CASES,
            (
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
            ),
        )
        self.assertEqual(
            heldout.BRACKETING_CASES,
            {
                ("dense", 81920): (("dense", 65536), ("dense", 131072)),
                ("causal", 196608): (("causal", 131072), ("causal", 262144)),
            },
        )
        self.assertEqual({row["schema_version"] for row in rows}, {"3"})
        self.assertEqual({row["input_seed"] for row in rows}, {"2026081500"})
        self.assertEqual(
            {row["expected_commit"] for row in rows},
            {"c3e36b65d69681c23e053042b0bc21e2331bad17"},
        )

    def fake_result(
        self,
        _root: Path,
        _path: Path,
        variant: str,
        seq_len: int,
        *,
        throughput_by_seed: dict[int, float] | None = None,
        coverage_by_seed: dict[int, str] | None = None,
        terminal_policy_by_seed: dict[int, str] | None = None,
        terminal_surface_by_seed: dict[int, str] | None = None,
    ) -> dict[str, object]:
        settings = heldout.strict.EXPECTED_CASES[(variant, seq_len)]
        tuner_seed = int(settings["tuner_seed"])
        return {
            "case": f"{variant}_{seq_len}",
            "variant": variant,
            "seq_len": seq_len,
            "version": TEST_VERSION,
            "input_seed": heldout.strict.EXPECTED_INPUT_SEED,
            "tuner_seed": tuner_seed,
            "median_tflops": (throughput_by_seed or {}).get(tuner_seed, 750.0),
            "coverage_design_sha256": (coverage_by_seed or {}).get(
                tuner_seed, f"{variant}-coverage"
            ),
            "compiler_seed_policy_json": compiler_seed_policy_json(variant),
            "terminal_refinement_policy_sha256": (terminal_policy_by_seed or {}).get(
                tuner_seed, f"{variant}-terminal-policy"
            ),
            "terminal_coordinate_surface_sha256": (terminal_surface_by_seed or {}).get(
                tuner_seed, f"{variant}-surface"
            ),
        }

    def validate_individual_with_values(
        self,
        *,
        throughput_by_seed: dict[int, float] | None = None,
        coverage_by_seed: dict[int, str] | None = None,
        reference_tflops: dict[tuple[str, int], float] | None = None,
        reference_coverage: dict[str, str] | None = None,
        reference_seed_policy: dict[str, str] | None = None,
        reference_terminal_policy: dict[str, str] | None = None,
        reference_terminal_surface: dict[str, str] | None = None,
        terminal_policy_by_seed: dict[int, str] | None = None,
        terminal_surface_by_seed: dict[int, str] | None = None,
        reference_version: str = TEST_VERSION,
    ) -> tuple[
        Path,
        list[dict[str, object]],
        dict[tuple[str, int], dict[str, str]],
        str,
    ]:
        def validate(
            root: Path, path: Path, variant: str, seq_len: int
        ) -> dict[str, object]:
            return self.fake_result(
                root,
                path,
                variant,
                seq_len,
                throughput_by_seed=throughput_by_seed,
                coverage_by_seed=coverage_by_seed,
                terminal_policy_by_seed=terminal_policy_by_seed,
                terminal_surface_by_seed=terminal_surface_by_seed,
            )

        reference = self.fake_reference(
            reference_tflops=reference_tflops,
            reference_coverage=reference_coverage,
            reference_seed_policy=reference_seed_policy,
            reference_terminal_policy=reference_terminal_policy,
            reference_terminal_surface=reference_terminal_surface,
            version=reference_version,
        )
        with (
            mock.patch.object(
                heldout,
                "validate_campaign",
                return_value=(Path("/heldout/campaign.csv"), set()),
            ),
            mock.patch.object(heldout.strict, "validate_result", side_effect=validate),
            mock.patch.object(heldout.strict, "file_sha256", return_value="c" * 64),
            mock.patch.object(
                heldout,
                "load_all8_reference",
                return_value=(reference, "a" * 64),
            ),
        ):
            return heldout.validate_individual_results(Path("/heldout"), Path("/all8"))

    def build_with_values(
        self,
        *,
        throughput_by_seed: dict[int, float] | None = None,
        coverage_by_seed: dict[int, str] | None = None,
        reference_tflops: dict[tuple[str, int], float] | None = None,
        reference_coverage: dict[str, str] | None = None,
        reference_seed_policy: dict[str, str] | None = None,
        reference_terminal_policy: dict[str, str] | None = None,
        reference_terminal_surface: dict[str, str] | None = None,
        terminal_policy_by_seed: dict[int, str] | None = None,
        terminal_surface_by_seed: dict[int, str] | None = None,
        reference_version: str = TEST_VERSION,
    ) -> str:
        def validate(
            root: Path, path: Path, variant: str, seq_len: int
        ) -> dict[str, object]:
            return self.fake_result(
                root,
                path,
                variant,
                seq_len,
                throughput_by_seed=throughput_by_seed,
                coverage_by_seed=coverage_by_seed,
                terminal_policy_by_seed=terminal_policy_by_seed,
                terminal_surface_by_seed=terminal_surface_by_seed,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "heldout"
            root.mkdir()
            all8_root = Path(directory) / "all8"
            all8_root.mkdir()
            launcher_dir = root / "launcher"
            launcher_dir.mkdir()
            all8_launcher = launcher_dir / "run_strict_all8.sh"
            heldout_launcher = launcher_dir / "run_strict_heldout.sh"
            all8_launcher.write_text("#!/usr/bin/env bash\n# all8 fixture\n")
            heldout_launcher.write_text("#!/usr/bin/env bash\n# heldout fixture\n")
            all8_launcher.chmod(0o555)
            heldout_launcher.chmod(0o555)
            all8_launcher_sha256 = heldout.strict.file_sha256(all8_launcher)
            heldout_launcher_sha256 = heldout.strict.file_sha256(heldout_launcher)
            campaign = root / "campaign.csv"
            with campaign.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=heldout.CAMPAIGN_FIELDS)
                writer.writeheader()
                writer.writerows(
                    heldout.expected_campaign_rows(
                        "/canonical/python",
                        all8_launcher_sha256,
                        heldout_launcher_sha256,
                    )
                )
            for variant, seq_len, _physical_gpu, tuner_seed in heldout.CASES:
                path = heldout.result_path(root, variant, seq_len, tuner_seed)
                path.parent.mkdir(parents=True)
                path.write_text('{"impl":"helion-cute"}\n')

            all8_output = io.StringIO(newline="")
            writer = csv.DictWriter(
                all8_output, fieldnames=heldout.strict.MANIFEST_FIELDS
            )
            writer.writeheader()
            for row in self.fake_reference(
                reference_tflops=reference_tflops,
                reference_coverage=reference_coverage,
                reference_seed_policy=reference_seed_policy,
                reference_terminal_policy=reference_terminal_policy,
                reference_terminal_surface=reference_terminal_surface,
                version=reference_version,
            ).values():
                writer.writerow(row)
            with (
                mock.patch.object(
                    heldout.strict, "validate_result", side_effect=validate
                ),
                mock.patch.object(
                    heldout.strict,
                    "build_manifest",
                    return_value=all8_output.getvalue(),
                ),
            ):
                return heldout.build_manifest(root, all8_root)

    def test_emits_all_independent_seeds(self) -> None:
        rows = list(csv.DictReader(io.StringIO(self.build_with_values())))
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            {int(row["tuner_seed"]) for row in rows},
            {
                2026082301,
                2026082302,
                2026082303,
                2026082304,
                2026082305,
                2026082311,
                2026082312,
                2026082313,
                2026082314,
                2026082315,
            },
        )
        self.assertEqual(len({row["case"] for row in rows}), 10)

    def test_individual_validation_does_not_apply_cross_seed_performance_gate(
        self,
    ) -> None:
        throughputs = {
            2026082301: 740.0,
            2026082302: 750.0,
            2026082303: 745.0,
            2026082304: 748.0,
            2026082305: 747.0,
            2026082311: 740.0,
            2026082312: 750.0,
            2026082313: 745.0,
            2026082314: 748.0,
            2026082315: 747.0,
        }

        _campaign, rows, _reference, _digest = self.validate_individual_with_values(
            throughput_by_seed=throughputs,
            reference_tflops={
                (variant, seq_len): 700.0
                for variant, seq_len, _gpu, _seed in heldout.strict.CASES
            },
        )

        self.assertEqual(
            [row["median_tflops"] for row in rows], list(throughputs.values())
        )

    def test_individual_validation_keeps_all8_bracketing_floor(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "independent bracketing reference"):
            self.validate_individual_with_values(
                throughput_by_seed={seed: 700.0 for *_prefix, seed in heldout.CASES},
                reference_tflops={
                    (dense_or_causal, seq_len): 750.0
                    for dense_or_causal, seq_len, _gpu, _seed in heldout.strict.CASES
                },
            )

    def test_individual_validation_keeps_all8_seed_policy_link(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "compiler seed policy"):
            self.validate_individual_with_values(
                reference_seed_policy={"causal": "different"}
            )

    def test_individual_validation_keeps_all8_terminal_policy_link(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "terminal refinement policy"):
            self.validate_individual_with_values(
                reference_terminal_policy={"causal": "different"}
            )

    def test_rejects_seed_dependent_terminal_surface(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "terminal coordinate surface"):
            self.validate_individual_with_values(
                terminal_surface_by_seed={2026082315: "different"}
            )

    def test_individual_validation_keeps_all8_terminal_surface_link(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "terminal coordinate surface"):
            self.validate_individual_with_values(
                reference_terminal_surface={"causal": "different"}
            )

    def test_individual_validation_keeps_all8_coverage_link(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "structural coverage"):
            self.validate_individual_with_values(
                reference_coverage={"causal": "different"}
            )

    def test_individual_validation_keeps_all8_version_link(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "held-out versus all8 version"):
            self.validate_individual_with_values(
                reference_version="Helion 9.9.dev0+gc3e36b65d; CuTe 4.7.0"
            )

    def test_rejects_median_seed_performance_instability(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "median seed robustness below"):
            self.build_with_values(
                throughput_by_seed={
                    2026082301: 750.0,
                    2026082302: 748.0,
                    2026082303: 747.0,
                    2026082304: 746.0,
                    2026082305: 745.0,
                }
            )

    def test_rejects_worst_seed_performance_instability(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "worst seed robustness below"):
            self.build_with_values(
                throughput_by_seed={2026082301: 750.0, 2026082302: 743.0}
            )

    def test_accepts_median_and_worst_seed_threshold_boundaries(self) -> None:
        self.build_with_values(
            throughput_by_seed={
                2026082301: 750.0,
                2026082302: 749.0,
                2026082303: 748.5,
                2026082304: 747.0,
                2026082305: 746.25,
            }
        )

    def test_rejects_seed_dependent_coverage(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "changed structural coverage"):
            self.build_with_values(
                coverage_by_seed={2026082315: "different-causal-coverage"}
            )

    def test_rejects_coverage_that_differs_from_all8(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "bracketing all8"):
            self.build_with_values(reference_coverage={"causal": "different"})

    def test_rejects_consistently_slow_seeds(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "independent bracketing reference"):
            self.build_with_values(
                throughput_by_seed={seed: 700.0 for *_prefix, seed in heldout.CASES},
                reference_tflops={
                    ("dense", 65536): 750.0,
                    ("dense", 131072): 760.0,
                    ("causal", 131072): 740.0,
                    ("causal", 262144): 750.0,
                },
            )

    def test_rejects_nonpositive_reference(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "median_tflops"):
            self.build_with_values(
                reference_tflops={
                    (variant, seq_len): -1.0
                    for variant, seq_len, _gpu, _seed in heldout.strict.CASES
                }
            )

    def test_rejects_extra_result(self) -> None:
        original = heldout.validate_campaign

        def add_extra(root: Path) -> tuple[Path, set[Path]]:
            extra = root / "rerun" / "result.json"
            extra.parent.mkdir()
            extra.write_text('{"impl":"helion-cute"}\n')
            return original(root)

        with (
            mock.patch.object(heldout, "validate_campaign", side_effect=add_extra),
            self.assertRaisesRegex(RuntimeError, "held-out result set"),
        ):
            self.build_with_values()

    def test_restores_all8_case_map_after_failure(self) -> None:
        original = heldout.strict.EXPECTED_CASES
        with (
            mock.patch.object(
                heldout,
                "validate_campaign",
                return_value=(Path("/unused/campaign.csv"), set()),
            ),
            mock.patch.object(heldout, "load_all8_reference", return_value=({}, "sha")),
            mock.patch.object(
                heldout.strict,
                "validate_result",
                side_effect=RuntimeError("expected failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "expected failure"),
        ):
            heldout.build_manifest(Path("/unused"), Path("/unused/all8.csv"))
        self.assertIs(heldout.strict.EXPECTED_CASES, original)

    def test_output_must_not_overwrite_artifact_roots(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "outside the held-out"):
            heldout.validate_output_path(
                Path("/artifacts/heldout/campaign.csv"),
                (
                    ("held-out", Path("/artifacts/heldout")),
                    ("all8", Path("/artifacts/all8")),
                ),
            )

    def test_rejects_result_symlink_alias(self) -> None:
        original = heldout.validate_campaign

        def add_alias(root: Path) -> tuple[Path, set[Path]]:
            expected = heldout.result_path(root, "dense", 81920, 2026082301)
            alias = root / "alias" / "result.json"
            alias.parent.mkdir()
            alias.symlink_to(expected)
            return original(root)

        with (
            mock.patch.object(heldout, "validate_campaign", side_effect=add_alias),
            self.assertRaisesRegex(RuntimeError, "contains symlinks"),
        ):
            self.build_with_values()


if __name__ == "__main__":
    unittest.main()
