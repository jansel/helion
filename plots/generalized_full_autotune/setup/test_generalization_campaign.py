from __future__ import annotations

from contextlib import suppress
import copy
import csv
import io
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import build_strict_manifest as strict
import remeasure_generalization_winners as remeasure
import run_generalization_campaign as runner
import validate_generalization_campaign as validator


def make_case(
    *,
    case_id: str = "dense_odd",
    seq_len: int = 6272,
    causal: bool = False,
    legality: str = "odd",
    tuner_seeds: tuple[int, ...] = (201, 202, 203, 204, 205),
) -> validator.CaseSpec:
    return validator.CaseSpec(
        case_id=case_id,
        z=3,
        h=5,
        seq_len=seq_len,
        head_dim=64,
        dtype="float16",
        causal=causal,
        legality_class=legality,
        input_seed=101,
        tuner_seeds=tuner_seeds,
    )


def make_runs(case: validator.CaseSpec) -> tuple[validator.RunSpec, ...]:
    return validator.expand_runs((case,))


def valid_payload(run: validator.RunSpec, commit: str) -> dict[str, object]:
    case = run.case
    flops = 4.0 * case.z * case.h * case.seq_len**2 * case.head_dim
    if case.causal:
        flops *= 0.5
    median_ms = 2.0
    return {
        "impl": "helion-cute",
        "version": (
            f"Helion 1.4.0.dev1+g{commit[:8]}; CuTe {strict.EXPECTED_CUTE_VERSION}"
        ),
        "gpu": "NVIDIA B200",
        "physical_gpu": str(case.physical_gpu),
        "power_cap_w": 750,
        "input_seed": case.input_seed,
        "shape": {
            "z": case.z,
            "h": case.h,
            "seq_len": case.seq_len,
            "head_dim": case.head_dim,
            "dtype": case.dtype,
            "causal": int(case.causal),
            "biased": 0,
        },
        "accuracy": "PASS",
        "benchmark_timer": "wall",
        "flop_model": "softmax_attention_forward",
        "runs_ms": [median_ms] * 9,
        "median_ms": median_ms,
        "median_tflops": flops / (median_ms * 1e9),
    }


def validated_rows(
    case: validator.CaseSpec,
    throughputs: tuple[float, ...] | None = None,
    digests: tuple[str, ...] | None = None,
    seed_policy_digests: tuple[str, ...] | None = None,
    terminal_policy_digests: tuple[str, ...] | None = None,
    terminal_surface_digests: tuple[str, ...] | None = None,
) -> list[validator.ValidatedRun]:
    if throughputs is None:
        throughputs = (100.0,) * len(case.tuner_seeds)
    if digests is None:
        digests = ("a",) * len(case.tuner_seeds)
    if seed_policy_digests is None:
        seed_policy_digests = ("p",) * len(case.tuner_seeds)
    if terminal_policy_digests is None:
        terminal_policy_digests = ("t",) * len(case.tuner_seeds)
    if terminal_surface_digests is None:
        terminal_surface_digests = ("u",) * len(case.tuner_seeds)
    return [
        validator.ValidatedRun(
            run=run,
            median_tflops=throughput,
            num_configs_tested=100,
            num_successful_candidate_measurements=100,
            num_source_deduplications=0,
            num_isolated_rebenchmark_timeouts=0,
            num_generations=20,
            exact_effective_search_space_size=None,
            coverage_design_sha256=digest,
            compiler_seed_policy_sha256=seed_policy_digest,
            terminal_refinement_policy_sha256=terminal_policy_digest,
            terminal_coordinate_surface_sha256=terminal_surface_digest,
            terminal_refinement_sha256="v",
            result_sha256="r",
            autotune_csv_sha256="c",
            autotune_metadata_sha256="m",
            source_ledger_sha256="l",
            selected_source_sha256="s",
        )
        for (
            run,
            throughput,
            digest,
            seed_policy_digest,
            terminal_policy_digest,
            terminal_surface_digest,
        ) in zip(
            make_runs(case),
            throughputs,
            digests,
            seed_policy_digests,
            terminal_policy_digests,
            terminal_surface_digests,
            strict=True,
        )
    ]


def clc_family_result(
    family: str,
    anchors: frozenset[int],
    legal: frozenset[int],
    *,
    exhausted: frozenset[int] = frozenset(),
) -> dict[str, object]:
    conditional = legal - exhausted
    return {
        "family": family,
        "anchor_values": list(anchors),
        "legal_values": list(legal),
        "search_values": list(legal),
        "planned_values": list(legal),
        "attempted_values": list(legal),
        "witness_config_ids": {str(value): f"{value:016x}" for value in legal},
        "selected_values": list(legal),
        "retained_values": list(legal),
        "value_space_exhausted": {str(value): value in exhausted for value in legal},
        "conditional_values": list(conditional),
        "conditional_candidate_ids": {
            str(value): [f"{value + 1000:016x}"] for value in conditional
        },
        "complete": True,
    }


class GeneralizationCampaignTests(unittest.TestCase):
    def test_campaign_commit_is_pinned_across_launcher_and_validator(self) -> None:
        expected = "c3e36b65d69681c23e053042b0bc21e2331bad17"

        self.assertEqual(runner.EXPECTED_COMMIT, expected)
        self.assertEqual(validator.EXPECTED_COMMIT, expected)
        runner.require_commit_contract()

    def test_campaign_lock_rejects_concurrent_supervisors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            lock = runner.campaign_lock(root)
            lock.__enter__()
            try:
                with (
                    self.assertRaisesRegex(
                        RuntimeError, "another campaign process holds"
                    ),
                    runner.campaign_lock(root),
                ):
                    self.fail("concurrent campaign lock unexpectedly succeeded")
            finally:
                lock.__exit__(None, None, None)

    def _spawn_process_group_with_term_ignoring_child(
        self, directory: Path, *, parent_exit: int | None
    ) -> tuple[subprocess.Popen[str], int]:
        child_pid_path = directory / "child.pid"
        if parent_exit is None:
            parent_body = "trap 'exit 0' TERM; while true; do sleep 1; done"
        else:
            parent_body = f"sleep 0.1; exit {parent_exit}"
        script = (
            "(trap '' TERM; "
            f"echo $BASHPID > {child_pid_path!s}; "
            "while true; do sleep 1; done) & "
            f"{parent_body}"
        )
        process = subprocess.Popen(
            ["bash", "-c", script],
            start_new_session=True,
            text=True,
        )
        deadline = time.monotonic() + 5
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(child_pid_path.is_file())
        return process, int(child_pid_path.read_text())

    def _cleanup_process_group(self, process: subprocess.Popen[str]) -> None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def test_wait_for_process_drains_descendant_after_peer_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process, child_pid = self._spawn_process_group_with_term_ignoring_child(
                Path(directory), parent_exit=None
            )
            stop_event = threading.Event()
            stop_event.set()
            try:
                with mock.patch.object(runner, "PROCESS_GROUP_TERM_GRACE_SECONDS", 0.1):
                    self.assertEqual(runner.wait_for_process(process, stop_event), 0)
                self.assertFalse(runner.process_group_live(process.pid))
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                self._cleanup_process_group(process)

    def test_wait_for_process_preserves_failure_and_drains_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process, child_pid = self._spawn_process_group_with_term_ignoring_child(
                Path(directory), parent_exit=17
            )
            try:
                with mock.patch.object(runner, "PROCESS_GROUP_TERM_GRACE_SECONDS", 0.1):
                    self.assertEqual(
                        runner.wait_for_process(process, threading.Event()), 17
                    )
                self.assertFalse(runner.process_group_live(process.pid))
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                self._cleanup_process_group(process)

    def test_python_environment_rejects_wrong_cute_version(self) -> None:
        completed = runner.subprocess.CompletedProcess(
            args=["python"], returncode=0, stdout="4.6.1\n"
        )
        with (
            mock.patch.object(runner.subprocess, "run", return_value=completed),
            self.assertRaisesRegex(RuntimeError, "CuTe version"),
        ):
            runner.require_python_environment(Path("/python"))

    def test_initialize_rejects_checkout_at_another_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "campaign"
            repo = base / "repo"
            repo.mkdir()
            with (
                mock.patch.object(runner, "git_commit", return_value="b" * 40),
                self.assertRaisesRegex(RuntimeError, "campaign checkout commit"),
            ):
                runner.initialize_campaign(
                    root,
                    repo,
                    Path("/python"),
                    Path("/matrix.csv"),
                    (make_case(),),
                )
            self.assertFalse(root.exists())

    def test_initialize_records_only_the_pinned_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "campaign"
            repo = base / "repo"
            repo.mkdir()
            with (
                mock.patch.object(
                    runner, "git_commit", return_value=runner.EXPECTED_COMMIT
                ),
                mock.patch.object(runner, "snapshot", return_value="a" * 64),
            ):
                runner.initialize_campaign(
                    root,
                    repo,
                    Path("/python"),
                    Path("/matrix.csv"),
                    (make_case(),),
                )

            header = validator.load_campaign_records(root)[0]
            self.assertEqual(header["expected_commit"], runner.EXPECTED_COMMIT)

    def test_launch_identity_rejects_declared_or_checked_out_other_commit(self) -> None:
        with (
            mock.patch.object(runner, "require_clean_checkout"),
            mock.patch.object(
                runner, "git_commit", return_value=runner.EXPECTED_COMMIT
            ),
            self.assertRaisesRegex(RuntimeError, "declared campaign commit"),
        ):
            runner.require_checkout_identity(Path("/repo"), "b" * 40, "a" * 64)

        with (
            mock.patch.object(runner, "require_clean_checkout"),
            mock.patch.object(runner, "git_commit", return_value="b" * 40),
            self.assertRaisesRegex(RuntimeError, "measured checkout commit"),
        ):
            runner.require_checkout_identity(
                Path("/repo"), runner.EXPECTED_COMMIT, "a" * 64
            )

    def test_detached_validation_rejects_another_declared_commit(self) -> None:
        records = [
            {
                "record_type": "campaign",
                "schema_version": validator.SCHEMA_VERSION,
                "expected_commit": "b" * 40,
            }
        ]
        with (
            mock.patch.object(validator, "load_campaign_records", return_value=records),
            self.assertRaisesRegex(RuntimeError, "campaign expected commit"),
        ):
            validator.validate_campaign(Path("/detached/campaign"))

    def test_resume_rejects_campaign_from_another_commit(self) -> None:
        repo = Path("/repo")
        records = [
            {
                "record_type": "campaign",
                "repo_root": str(repo),
                "expected_commit": "b" * 40,
            }
        ]
        with (
            mock.patch.object(validator, "load_campaign_records", return_value=records),
            self.assertRaisesRegex(RuntimeError, "resume declared commit"),
        ):
            runner.resume_campaign(
                Path("/campaign"), repo, Path("/python"), Path("/matrix.csv")
            )

    def test_checked_in_matrix_parses_all_classes_and_targeted_seeds(self) -> None:
        cases = validator.parse_case_matrix(
            Path(__file__).with_name("generalization_cases.csv")
        )

        self.assertEqual(len(cases), 15)
        self.assertEqual(
            {case.legality_class for case in cases},
            {"singleton", "odd", "paired", "div4"},
        )
        self.assertEqual(
            [len(case.tuner_seeds) for case in cases].count(5),
            10,
        )
        self.assertEqual(
            [len(case.tuner_seeds) for case in cases].count(3),
            5,
        )
        self.assertEqual([case.physical_gpu for case in cases].count(6), 7)
        self.assertEqual([case.physical_gpu for case in cases].count(7), 8)
        self.assertEqual(sum(len(case.tuner_seeds) for case in cases), 65)
        surface_counts: dict[tuple[object, ...], int] = {}
        for case in cases:
            surface_counts[case.surface_key] = (
                surface_counts.get(case.surface_key, 0) + 1
            )
        self.assertEqual(
            sorted(count for count in surface_counts.values() if count > 1),
            [2, 2, 2],
        )
        main_cases = [
            case
            for case in cases
            if case.dtype == "float16"
            and case.head_dim == 64
            and case.legality_class == "div4"
        ]
        self.assertEqual({case.causal for case in main_cases}, {False, True})

    def test_minimal_cross_shape_matrix_is_exact_and_expands_to_18_runs(self) -> None:
        cases = validator.parse_case_matrix(
            Path(__file__).with_name("minimal_cross_shape_cases.csv")
        )
        runs = validator.expand_runs(cases)

        self.assertEqual(
            tuple(validator._case_contract(case) for case in cases),
            validator.MINIMAL_CROSS_SHAPE_CASES,
        )
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(runs), 18)
        self.assertEqual(sum(not case.causal for case in cases), 3)
        self.assertEqual(sum(case.causal for case in cases), 3)
        self.assertEqual(
            {
                (case.z, case.h, case.seq_len, case.head_dim, case.dtype)
                for case in cases
            },
            {
                (4, 16, 16384, 64, "float16"),
                (3, 32, 16384, 64, "bfloat16"),
                (3, 32, 16384, 128, "float16"),
            },
        )
        self.assertEqual(
            {run.case.physical_gpu for run in runs if run.case.causal}, {6}
        )
        self.assertEqual(
            {run.case.physical_gpu for run in runs if not run.case.causal}, {7}
        )
        self.assertEqual(len({run.tuner_seed for run in runs}), 18)

    def test_minimal_cross_shape_matrix_rejects_any_profile_change(self) -> None:
        source = Path(__file__).with_name("minimal_cross_shape_cases.csv").read_text()
        mutations = (
            source.replace(
                "dense_b4_h16_s16384_d64_fp16,4,16,16384",
                "dense_b4_h16_s16384_d64_fp16,5,16,16384",
                1,
            ),
            source.replace("20260820103,,", "20260820104,,", 1),
            source.replace(
                "dense_b3_h32_s16384_d64_bf16",
                "dense_b3_h32_s16384_d64_bf16_changed",
                1,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            for mutation in mutations:
                path.write_text(mutation)
                with self.assertRaises(RuntimeError):
                    validator.parse_case_matrix(path)

    def test_minimal_cross_shape_runs_use_strict_cold_full_commands(self) -> None:
        cases = validator.parse_case_matrix(
            Path(__file__).with_name("minimal_cross_shape_cases.csv")
        )
        for run in validator.expand_runs(cases):
            command = validator.expected_command(
                Path("/env/python"), Path("/repo"), Path("/artifacts"), run
            )
            joined = " ".join(command)
            for required in (
                "--helion-force-flash-config 0",
                "--helion-force-autotune 1",
                "--helion-require-full-autotune 1",
                "--helion-autotune-effort full",
                "--helion-autotune-best-of-k 1",
                "--helion-autotune-accuracy-check 1",
                "--helion-autotuner-initial-population from_random",
                f"HELION_AUTOTUNE_RANDOM_SEED={run.tuner_seed}",
            ):
                self.assertIn(required, joined)
            for forbidden in (
                "--helion-config",
                "--helion-autotune-budget-seconds",
                "--helion-autotune-max-generations",
                "--helion-seed-config",
            ):
                self.assertNotIn(forbidden, joined)

    def test_minimal_cross_shape_launcher_pins_commit_and_matrix(self) -> None:
        launcher = (
            Path(__file__).with_name("run_minimal_cross_shape_campaign.sh").read_text()
        )

        self.assertIn(
            "EXPECTED_COMMIT=c3e36b65d69681c23e053042b0bc21e2331bad17",
            launcher,
        )
        self.assertIn('MATRIX="$SETUP_ROOT/minimal_cross_shape_cases.csv"', launcher)
        self.assertIn('RUNNER="$SETUP_ROOT/run_generalization_campaign.py"', launcher)
        self.assertIn('--matrix "$MATRIX"', launcher)

    def test_matrix_rejects_noncontiguous_or_legacy_seed_rows(self) -> None:
        source = Path(__file__).with_name("generalization_cases.csv").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            path.write_text(
                source.replace(",20260819102,20260819103,,", ",,20260819103,,")
            )
            with self.assertRaisesRegex(RuntimeError, "contiguous prefix"):
                validator.parse_case_matrix(path)

            path.write_text(source.replace("\n3,", "\n2,", 1))
            with self.assertRaisesRegex(RuntimeError, "expected .*3"):
                validator.parse_case_matrix(path)

    def test_matrix_rejects_mislabeled_legality_class(self) -> None:
        source = Path(__file__).with_name("generalization_cases.csv").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            path.write_text(source.replace(",singleton,", ",odd,", 1))
            with self.assertRaisesRegex(RuntimeError, "legality class"):
                validator.parse_case_matrix(path)

    def test_matrix_rejects_unaligned_fallback_shape(self) -> None:
        source = Path(__file__).with_name("generalization_cases.csv").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.csv"
            path.write_text(source.replace(",6272,", ",6271,", 1))
            with self.assertRaisesRegex(RuntimeError, "does not exercise.*flash path"):
                validator.parse_case_matrix(path)

    def test_command_is_strict_full_cold_search(self) -> None:
        run = make_runs(make_case())[0]
        command = runner.build_command(
            Path("/env/python"), Path("/repo"), Path("/artifacts"), run
        )

        self.assertEqual(
            command,
            validator.expected_command(
                Path("/env/python"), Path("/repo"), Path("/artifacts"), run
            ),
        )
        joined = " ".join(command)
        for required in (
            "--helion-require-full-autotune 1",
            "--helion-autotune-effort full",
            "--helion-autotuner-initial-population from_random",
            "--helion-force-flash-config 0",
            "--helion-force-autotune 1",
            "--helion-autotune-accuracy-check 1",
        ):
            self.assertIn(required, joined)
        self.assertNotIn("--helion-config", command)
        self.assertNotIn("--helion-seed-config", command)

    def test_remeasurement_command_is_predeclared_and_forces_one_case(self) -> None:
        case = make_case()
        command = validator.expected_remeasurement_command(
            Path("/env/python"),
            Path("/repo"),
            Path("/artifacts"),
            case,
            "f" * 64,
        )

        self.assertIn("--case-id", command)
        self.assertEqual(command[command.index("--case-id") + 1], case.case_id)
        self.assertEqual(command[command.index("--physical-gpu") + 1], "7")
        self.assertEqual(command[command.index("--rounds") + 1], "12")
        self.assertEqual(command[command.index("--target-timing-ms") + 1], "20")
        self.assertEqual(command[command.index("--max-timing-repetitions") + 1], "4096")
        self.assertEqual(command[command.index("--bootstrap-samples") + 1], "20000")

    def test_payload_identity_is_dynamic_and_rejects_wrong_gpu(self) -> None:
        commit = "a" * 40
        run = make_runs(make_case(causal=True))[0]
        payload = valid_payload(run, commit)
        validator.validate_payload_identity(Path("result.json"), payload, run, commit)

        longer_version = copy.deepcopy(payload)
        longer_version["version"] = (
            f"Helion 1.4.0.dev1+g{commit[:9]}; CuTe {strict.EXPECTED_CUTE_VERSION}"
        )
        validator.validate_payload_identity(
            Path("result.json"), longer_version, run, commit
        )

        bad = copy.deepcopy(payload)
        bad["physical_gpu"] = "7"
        with self.assertRaisesRegex(RuntimeError, "physical GPU"):
            validator.validate_payload_identity(Path("result.json"), bad, run, commit)

        bad_version = copy.deepcopy(payload)
        bad_version["version"] = (
            f"Helion 1.4.0.dev1+g{'b' * 8}; CuTe {strict.EXPECTED_CUTE_VERSION}"
        )
        with self.assertRaisesRegex(RuntimeError, "does not identify"):
            validator.validate_payload_identity(
                Path("result.json"), bad_version, run, commit
            )

    def test_coverage_grouping_rejects_seed_dependent_design(self) -> None:
        rows = validated_rows(make_case(), digests=("a", "a", "a", "a", "b"))
        with self.assertRaisesRegex(RuntimeError, "seed-dependent coverage"):
            validator.validate_group_gates(rows)

    def test_coverage_grouping_accepts_declared_three_seed_case(self) -> None:
        case = make_case(tuner_seeds=(201, 202, 203))
        validator.validate_group_gates(validated_rows(case))

    def test_coverage_grouping_rejects_seed_dependent_compiler_policy(self) -> None:
        rows = validated_rows(
            make_case(), seed_policy_digests=("a", "a", "a", "a", "b")
        )
        with self.assertRaisesRegex(RuntimeError, "compiler seed policy"):
            validator.validate_group_gates(rows)

    def test_coverage_grouping_rejects_seed_dependent_terminal_policy(self) -> None:
        rows = validated_rows(
            make_case(), terminal_policy_digests=("a", "a", "a", "a", "b")
        )
        with self.assertRaisesRegex(RuntimeError, "terminal refinement policy"):
            validator.validate_group_gates(rows)

    def test_coverage_grouping_rejects_seed_dependent_terminal_surface(self) -> None:
        rows = validated_rows(
            make_case(), terminal_surface_digests=("a", "a", "a", "a", "b")
        )
        with self.assertRaisesRegex(RuntimeError, "terminal coordinate surface"):
            validator.validate_group_gates(rows)

    def test_coverage_grouping_rejects_length_dependent_terminal_surface(self) -> None:
        first = make_case(
            case_id="dense_odd_first",
            seq_len=6272,
            tuner_seeds=(201, 202, 203),
        )
        second = make_case(
            case_id="dense_odd_second",
            seq_len=6528,
            tuner_seeds=(211, 212, 213),
        )
        rows = [
            *validated_rows(first, terminal_surface_digests=("a", "a", "a")),
            *validated_rows(second, terminal_surface_digests=("b", "b", "b")),
        ]

        with self.assertRaisesRegex(
            RuntimeError, "identical search surface.*terminal coordinate surface"
        ):
            validator.validate_group_gates(rows)

    def test_coverage_grouping_allows_distinct_legality_surfaces(self) -> None:
        dense = make_case(
            case_id="dense_odd",
            causal=False,
            tuner_seeds=(201, 202, 203),
        )
        causal = make_case(
            case_id="causal_odd",
            causal=True,
            tuner_seeds=(211, 212, 213),
        )
        rows = [
            *validated_rows(dense, terminal_surface_digests=("a", "a", "a")),
            *validated_rows(causal, terminal_surface_digests=("b", "b", "b")),
        ]

        validator.validate_group_gates(rows)

    def test_remeasurement_rebuilds_compiler_policy_from_bound_spec(self) -> None:
        bound = mock.Mock()
        config_spec = object()
        config_generation = object()
        bound.env.config_spec = config_spec
        policy = {
            "schema_version": 1,
            "kind": "canonical_cute_flash",
        }

        with (
            mock.patch(
                "helion.autotuner.config_generation.ConfigGeneration",
                return_value=config_generation,
            ) as generation_constructor,
            mock.patch(
                "benchmarks.cute.compare_attention_backends._compiler_seed_policy",
                return_value=policy,
            ) as policy_builder,
        ):
            self.assertIs(remeasure.live_compiler_seed_policy(bound), policy)

        generation_constructor.assert_called_once_with(config_spec)
        policy_builder.assert_called_once_with(config_spec, config_generation)

    def test_remeasurement_rejects_recorded_compiler_policy_not_live(self) -> None:
        case = make_case(tuner_seeds=(201,))
        run = make_runs(case)[0]
        recorded_policy = {
            "schema_version": 1,
            "kind": "canonical_cute_flash",
            "effective_config_ids": ["a" * 16],
        }
        live_policy = {
            **recorded_policy,
            "effective_config_ids": ["b" * 16],
        }
        payloads = {
            run.run_id: {
                "helion_overrides": {
                    "autotune_provenance": {
                        "compiler_seed_policy": recorded_policy,
                    }
                }
            }
        }
        bound = mock.Mock()
        kernel = mock.Mock()
        kernel.bind.return_value = bound

        with (
            mock.patch("examples.attention.attention_output", kernel),
            mock.patch.object(
                remeasure,
                "live_compiler_seed_policy",
                return_value=live_policy,
            ),
            self.assertRaisesRegex(RuntimeError, "compiler seed policy"),
        ):
            remeasure.build_callables((mock.Mock(),) * 3, case, [run], payloads)

    def test_bh180_group_plans_every_legal_clc_value(self) -> None:
        case = make_case(
            case_id="dense_clc_bh180_paired_d64",
            tuner_seeds=(201, 202, 203),
        )
        validator.validate_group_gates(validated_rows(case))

    def test_targeted_lane_gate_requires_every_d64_depth(self) -> None:
        case = make_case(case_id="causal_lane_paired_d64", causal=True)
        lanes = [
            {
                "key": "cute_flash_kv_stage",
                "value": value,
                "witness_attempted": True,
                "witness_succeeded": True,
                "space_exhausted": False,
                "conditional_required": True,
                "conditional_candidate_ids": [f"{value:016x}"],
                "successful_conditional_candidate_ids": [f"{value:016x}"],
                "repair_candidate_ids": [],
                "successful_repair_candidate_ids": [],
                "repair_parent_decisions": [],
                "complete": True,
            }
            for value in validator.EXPECTED_D64_ALIASED_KV_LANES
        ]
        provenance = {
            "flash_structural_coverage_active_values": [
                {"key": "cute_flash_kv_stage", "value": value}
                for value in validator.EXPECTED_D64_ALIASED_KV_LANES
            ]
        }
        trial = {
            "search_phase_metrics": {
                "phase": validator.EXPECTED_QUALIFICATION_PHASE,
                "cute_flash_lane_policy_version": (
                    validator.EXPECTED_LANE_POLICY_VERSION
                ),
                "qualification_failure_retries": 1,
                "leaf_results": [{"pipeline_lanes": lanes}],
            }
        }

        validator.validate_targeted_coverage(
            Path("result.json"), case, provenance, trial
        )
        lanes.pop()
        with self.assertRaisesRegex(RuntimeError, "measured K/V lanes"):
            validator.validate_targeted_coverage(
                Path("result.json"), case, provenance, trial
            )

    def test_targeted_clc_gate_requires_each_ordinary_family(self) -> None:
        case = make_case(case_id="dense_clc_bh96_paired_d64")
        expected = validator.TARGETED_CLC_ANCHOR_VALUES[case.case_id]
        legal = validator.TARGETED_CLC_LEGAL_VALUES[case.case_id]
        provenance = {
            "flash_structural_coverage_active_values": [
                {"key": "cute_flash_clc_heads_per_batch", "value": value}
                for value in expected
            ]
        }
        clc_families = [
            clc_family_result(family, expected, legal)
            for family in validator.EXPECTED_ORDINARY_CLC_FAMILIES
        ]
        trial = {
            "search_phase_metrics": {
                "phase": validator.EXPECTED_QUALIFICATION_PHASE,
                "cute_flash_lane_policy_version": (
                    validator.EXPECTED_LANE_POLICY_VERSION
                ),
                "qualification_failure_retries": 1,
                "clc_families": clc_families,
            }
        }

        validator.validate_targeted_coverage(
            Path("result.json"), case, provenance, trial
        )
        clc_families.pop()
        with self.assertRaisesRegex(RuntimeError, "ordinary CLC families"):
            validator.validate_targeted_coverage(
                Path("result.json"), case, provenance, trial
            )

    def test_bh180_targeted_clc_gate_plans_every_search_value(self) -> None:
        case = make_case(case_id="dense_clc_bh180_paired_d64")
        anchors = validator.TARGETED_CLC_ANCHOR_VALUES[case.case_id]
        legal = validator.TARGETED_CLC_LEGAL_VALUES[case.case_id]
        provenance = {
            "flash_structural_coverage_active_values": [
                {"key": "cute_flash_clc_heads_per_batch", "value": value}
                for value in anchors
            ]
        }
        clc_families = [
            clc_family_result(family, anchors, legal)
            for family in validator.EXPECTED_ORDINARY_CLC_FAMILIES
        ]
        trial = {
            "search_phase_metrics": {
                "phase": validator.EXPECTED_QUALIFICATION_PHASE,
                "cute_flash_lane_policy_version": (
                    validator.EXPECTED_LANE_POLICY_VERSION
                ),
                "qualification_failure_retries": 1,
                "clc_families": clc_families,
            }
        }

        validator.validate_targeted_coverage(
            Path("result.json"), case, provenance, trial
        )
        clc_families[0]["search_values"].remove(18)
        with self.assertRaisesRegex(RuntimeError, "searchable CLC values"):
            validator.validate_targeted_coverage(
                Path("result.json"), case, provenance, trial
            )

    def test_targeted_clc_gate_conditionally_searches_every_live_divisor(self) -> None:
        case = make_case(case_id="dense_clc_bh120_paired_d64")
        anchors = validator.TARGETED_CLC_ANCHOR_VALUES[case.case_id]
        legal = validator.TARGETED_CLC_LEGAL_VALUES[case.case_id]
        exhausted = frozenset({1, 120})
        provenance = {
            "flash_structural_coverage_active_values": [
                {"key": "cute_flash_clc_heads_per_batch", "value": value}
                for value in anchors
            ]
        }
        clc_families = [
            clc_family_result(family, anchors, legal, exhausted=exhausted)
            for family in validator.EXPECTED_ORDINARY_CLC_FAMILIES
        ]
        trial = {
            "search_phase_metrics": {
                "phase": validator.EXPECTED_QUALIFICATION_PHASE,
                "cute_flash_lane_policy_version": (
                    validator.EXPECTED_LANE_POLICY_VERSION
                ),
                "qualification_failure_retries": 1,
                "clc_families": clc_families,
            }
        }

        validator.validate_targeted_coverage(
            Path("result.json"), case, provenance, trial
        )
        missing = next(iter(legal - exhausted))
        clc_families[0]["conditional_values"].remove(missing)
        with self.assertRaisesRegex(RuntimeError, "conditional CLC values"):
            validator.validate_targeted_coverage(
                Path("result.json"), case, provenance, trial
            )

    def test_starting_path_capacity_is_derived_from_live_leaf_catalog(self) -> None:
        phase = {
            "leaf_results": [
                {"family": "wide"},
                {"family": "wide"},
                {"family": "narrow"},
            ],
            "compound_transfers": [{"family": "wide"}, {"family": "compound"}],
            "retained_family_cap": 2,
            "retained_family_limit": 2,
            "retained_candidates_per_leaf": 2,
            "starting_path_limit": 8,
            "family_probe_generations": 0,
            "family_probe_path_limit": 0,
            "maximum_path_capacity": 8,
            "retained_path_count": 7,
        }
        provenance = {
            "flash_structural_retained_family_cap": 2,
            "flash_structural_starting_path_limit": 8,
            "flash_structural_family_probe_path_limit": 0,
            "flash_structural_maximum_path_capacity": 8,
        }

        validator.validate_live_starting_path_capacity(
            Path("result.json"), provenance, phase
        )
        phase["starting_path_limit"] = 7
        phase["maximum_path_capacity"] = 7
        provenance["flash_structural_starting_path_limit"] = 7
        provenance["flash_structural_maximum_path_capacity"] = 7
        with self.assertRaisesRegex(RuntimeError, "below live-derived capacity 8"):
            validator.validate_live_starting_path_capacity(
                Path("result.json"), provenance, phase
            )

    def test_targeted_clc_gate_requires_generation_zero_anchor_attempts(self) -> None:
        case = make_case(case_id="dense_clc_bh96_paired_d64")
        anchors = sorted(validator.TARGETED_CLC_ANCHOR_VALUES[case.case_id])
        configs = {
            f"{index:016x}": {
                "cute_flash_pipeline_family": "fa4_clc",
                "cute_flash_clc_heads_per_batch": value,
            }
            for index, value in enumerate(anchors, 1)
        }
        source_rows = [
            {"config_id": config_id, "generation": "0", "status": "started"}
            for config_id in configs
        ]

        validator.validate_targeted_clc_generation_zero_attempts(
            Path("result.json"), case, source_rows, configs
        )
        source_rows.pop()
        with self.assertRaisesRegex(RuntimeError, "generation-0 CLC anchor attempts"):
            validator.validate_targeted_clc_generation_zero_attempts(
                Path("result.json"), case, source_rows, configs
            )

    def test_seed_robustness_uses_cross_measured_bootstrap_gate(self) -> None:
        record = {
            "seed_robustness_fraction": 0.979,
            "paired_bootstrap_95_ci": [0.978, 0.981],
        }
        summary = {
            "timers": {
                "event": copy.deepcopy(record),
                "wall": copy.deepcopy(record),
            }
        }
        with self.assertRaisesRegex(RuntimeError, "below 98%"):
            validator.validate_seed_robustness(
                summary, make_case(), context="odd fixture"
            )

        div4_case = make_case(case_id="dense_div4", seq_len=6144, legality="div4")
        summary["timers"]["event"]["seed_robustness_fraction"] = 0.999
        summary["timers"]["event"]["paired_bootstrap_95_ci"] = [0.989, 1.0]
        with self.assertRaisesRegex(RuntimeError, "below 99%"):
            validator.validate_seed_robustness(
                summary, div4_case, context="div4 fixture"
            )

    def test_main_semantic_class_requires_sdpa_competitiveness(self) -> None:
        record = {
            "seed_robustness_fraction": 1.0,
            "paired_bootstrap_95_ci": [1.0, 1.0],
            "minimum_seed_vs_sdpa_fraction": 0.979,
            "minimum_seed_vs_sdpa_paired_bootstrap_95_ci": [0.979, 0.981],
        }
        summary = {
            "timers": {
                "event": copy.deepcopy(record),
                "wall": copy.deepcopy(record),
            }
        }
        case = make_case(case_id="dense_div4", seq_len=6144, legality="div4")

        with self.assertRaisesRegex(RuntimeError, "below 98% of SDPA"):
            validator.validate_seed_robustness(summary, case, context="main fixture")

    def test_timing_repetition_calibration_uses_common_bounded_batch(self) -> None:
        calls = iter((0.01, 0.02, 0.03))
        with mock.patch.object(
            remeasure,
            "timed_call",
            side_effect=lambda _fn, repetitions: {
                "event_ms": next(calls),
                "wall_ms": float(repetitions),
            },
        ):
            repetitions, probes = remeasure.calibrate_timing_repetitions(
                lambda: None, target_ms=20.0, maximum=512
            )

        self.assertEqual(probes, [0.01, 0.02, 0.03])
        self.assertEqual(repetitions, 512)

    def test_tuning_environment_scrubs_scheduling_overrides(self) -> None:
        run = make_runs(make_case())[0]
        with mock.patch.dict(
            runner.os.environ,
            {
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "CUDA_LAUNCH_BLOCKING": "1",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50",
                "NVIDIA_TF32_OVERRIDE": "1",
                "PYTHONPATH": "/ambient/python",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                "HELION_CONFIG": "should-disappear",
            },
            clear=False,
        ):
            environment = runner.sanitized_environment(
                Path("/repo"), Path("/artifacts"), run
            )

        for name in (
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "CUDA_LAUNCH_BLOCKING",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "NVIDIA_TF32_OVERRIDE",
            "PYTHONPATH",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "HELION_CONFIG",
        ):
            self.assertNotIn(name, environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "7")

    def test_result_discovery_rejects_missing_and_extra(self) -> None:
        runs = make_runs(make_case())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in runs:
                path = root / run.result_path
                path.parent.mkdir(parents=True)
                for name in validator.RESULT_FILENAMES:
                    path.with_name(name).write_text("fixture")
            validator.validate_result_set(root, runs)

            (root / runs[0].result_path).unlink()
            with self.assertRaisesRegex(RuntimeError, "result set"):
                validator.validate_result_set(root, runs)

    def test_result_discovery_rejects_symlinked_sidecar(self) -> None:
        runs = make_runs(make_case())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in runs:
                path = root / run.result_path
                path.parent.mkdir(parents=True)
                for name in validator.RESULT_FILENAMES:
                    path.with_name(name).write_text("fixture")
            sidecar = (root / runs[0].result_path).with_name("autotune.csv")
            target = sidecar.with_suffix(".real")
            sidecar.rename(target)
            sidecar.symlink_to(target.name)

            with self.assertRaisesRegex(RuntimeError, "symlinked sidecars"):
                validator.validate_result_set(root, runs)
            (root / runs[0].result_path).write_text("fixture")
            extra = root / "extra" / "result.json"
            extra.parent.mkdir()
            extra.write_text("fixture")
            with self.assertRaisesRegex(RuntimeError, "result set"):
                validator.validate_result_set(root, runs)

    def test_event_history_rejects_retry(self) -> None:
        runs = make_runs(make_case())
        records: list[dict[str, object]] = []
        for run in runs:
            records.extend(
                (
                    {
                        "record_type": "event",
                        "event": "attempt_started",
                        "run_id": run.run_id,
                    },
                    {
                        "record_type": "event",
                        "event": "attempt_finished",
                        "run_id": run.run_id,
                        "returncode": 0,
                    },
                )
            )
        validator.validate_event_history(records, runs)
        records.append(records[0])
        with self.assertRaisesRegex(RuntimeError, "attempt count"):
            validator.validate_event_history(records, runs)

    def test_resume_rejects_started_only_result(self) -> None:
        run = make_runs(make_case())[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / run.result_path
            path.parent.mkdir(parents=True)
            for name in validator.RESULT_FILENAMES:
                path.with_name(name).write_text("fixture")
            history = [
                {
                    "record_type": "event",
                    "event": "attempt_started",
                    "run_id": run.run_id,
                }
            ]

            with self.assertRaisesRegex(RuntimeError, "refusing an implicit retry"):
                runner.classify_resume(root, run, history)

            history.append(
                {
                    "record_type": "event",
                    "event": "attempt_finished",
                    "run_id": run.run_id,
                    "returncode": 0,
                }
            )
            self.assertEqual(runner.classify_resume(root, run, history), "accept")

    def test_remeasurement_resume_rejects_started_only_result(self) -> None:
        case = make_case()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "remeasure" / f"{case.case_id}.json"
            output.parent.mkdir(parents=True)
            output.write_text("fixture")
            records = [
                {
                    "record_type": "event",
                    "event": "remeasurement_started",
                    "case_id": case.case_id,
                }
            ]

            with self.assertRaisesRegex(RuntimeError, "refusing implicit retry"):
                runner.classify_remeasurement_resume(root, case, records)

            records.append(
                {
                    "record_type": "event",
                    "event": "remeasurement_finished",
                    "case_id": case.case_id,
                    "returncode": 0,
                }
            )
            self.assertEqual(
                runner.classify_remeasurement_resume(root, case, records), "accept"
            )

    def test_remeasurement_orders_are_randomized_and_position_balanced(self) -> None:
        names = ["seed_1", "seed_2", "seed_3", "seed_4", "seed_5", "sdpa"]
        orders = remeasure.balanced_orders(names, 12, 1234)

        self.assertEqual(len(orders), 12)
        self.assertNotEqual(orders[0], names)
        for name in names:
            self.assertEqual(
                [
                    sum(order[position] == name for order in orders)
                    for position in range(6)
                ],
                [2] * 6,
            )

    def test_paired_remeasurement_summary_uses_common_round_resamples(self) -> None:
        case = make_case()
        config_names = [f"seed_{seed}" for seed in case.tuner_seeds]
        all_names = [*config_names, "sdpa"]
        raw = [
            {
                "times": {
                    name: {
                        "event_ms": 1.0 + name_index * 0.001 + round_index * 0.0001,
                        "wall_ms": 1.1 + name_index * 0.001 + round_index * 0.0001,
                    }
                    for name_index, name in enumerate(all_names)
                }
            }
            for round_index in range(12)
        ]

        summary = validator.summarize_remeasurement(
            raw,
            config_names,
            case,
            bootstrap_samples=100,
            bootstrap_seed=99,
        )

        self.assertGreater(summary["timers"]["event"]["seed_robustness_fraction"], 0.98)
        validator.validate_seed_robustness(summary, case, context="fixture")

    def test_render_manifest_is_valid_csv(self) -> None:
        rows = validated_rows(make_case())
        case = rows[0].run.case
        validation = validator.CampaignValidation(
            (case,), tuple(row.run for row in rows), tuple(rows), ()
        )
        parsed = list(
            csv.DictReader(io.StringIO(validator.render_manifest(validation)))
        )
        self.assertEqual(len(parsed), 5)
        self.assertEqual({row["case_id"] for row in parsed}, {"dense_odd"})


if __name__ == "__main__":
    unittest.main()
