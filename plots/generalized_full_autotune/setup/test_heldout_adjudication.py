from __future__ import annotations

from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import build_heldout_manifest as heldout
import build_strict_manifest as strict
import heldout_adjudication as adjudication
import remeasure_heldout_winners as worker
import run_heldout_adjudication as runner
import validate_heldout_adjudication as validator


class HeldoutAdjudicationTests(unittest.TestCase):
    def test_cases_match_predeclared_heldout_matrix(self) -> None:
        self.assertEqual(
            tuple(
                (case.variant, case.seq_len, case.physical_gpu, tuner_seed)
                for case in adjudication.CASES
                for tuner_seed in case.tuner_seeds
            ),
            heldout.CASES,
        )

    def make_fixture(
        self, base: Path
    ) -> tuple[
        Path,
        Path,
        Path,
        list[dict[str, object]],
        list[dict[str, object]],
        str,
    ]:
        repo = base / "repo"
        heldout_root = base / "heldout"
        all8_root = base / "all8"
        root = base / "adjudication"
        for relative in (
            adjudication.BENCHMARK_RELATIVE,
            adjudication.ATTENTION_RELATIVE,
        ):
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"# fixture {relative}\n")
        (heldout_root / "launcher").mkdir(parents=True)
        for name in ("run_strict_all8.sh", "run_strict_heldout.sh"):
            (heldout_root / "launcher" / name).write_text("#!/bin/sh\n")
        (heldout_root / "campaign.csv").write_text(
            "python_executable\n/canonical/python\n"
        )
        source_text = "# regenerated fixture kernel\n"
        source_sha256 = strict.sha256_bytes(source_text.encode())
        rows = []
        for variant, seq_len, _gpu, tuner_seed in heldout.CASES:
            directory = (
                heldout_root / variant / f"seed_{tuner_seed}" / f"{variant}_s{seq_len}"
            )
            directory.mkdir(parents=True)
            paths = {}
            for filename, contents in (
                ("result.json", "{}\n"),
                ("autotune.csv", "fixture\n"),
                ("autotune.meta.jsonl", "{}\n"),
                ("autotune.sources.csv", "fixture\n"),
            ):
                path = directory / filename
                path.write_text(contents)
                paths[filename] = path.relative_to(heldout_root).as_posix()
            config = {"block_sizes": [1, 128, 128], "seed_marker": tuner_seed}
            rows.append(
                {
                    "case": f"{variant}_{seq_len}_seed_{tuner_seed}",
                    "variant": variant,
                    "seq_len": seq_len,
                    "tuner_seed": tuner_seed,
                    "version": (
                        f"Helion 1.4.0.dev1+g{adjudication.EXPECTED_COMMIT[:9]}; "
                        f"CuTe {adjudication.EXPECTED_CUTE_VERSION}"
                    ),
                    "median_tflops": 100.0 if tuner_seed % 3 else 98.0,
                    "result_path": paths["result.json"],
                    "result_sha256": strict.file_sha256(directory / "result.json"),
                    "autotune_csv_path": paths["autotune.csv"],
                    "autotune_metadata_path": paths["autotune.meta.jsonl"],
                    "source_ledger_path": paths["autotune.sources.csv"],
                    "selected_config_json": strict.canonical_json(config),
                    "selected_config_sha256": strict.canonical_sha256(config),
                    "selected_source_sha256": source_sha256,
                }
            )
        all8_rows = []
        for variant, seq_len, _gpu, tuner_seed in strict.CASES:
            directory = all8_root / f"{variant}_s{seq_len}"
            directory.mkdir(parents=True)
            paths = {}
            for filename, contents in (
                ("result.json", "{}\n"),
                ("autotune.csv", "fixture\n"),
                ("autotune.meta.jsonl", "{}\n"),
                ("autotune.sources.csv", "fixture\n"),
            ):
                path = directory / filename
                path.write_text(contents)
                paths[filename] = path.relative_to(all8_root).as_posix()
            all8_rows.append(
                {
                    "case": f"{variant}_{seq_len}",
                    "variant": variant,
                    "seq_len": str(seq_len),
                    "tuner_seed": str(tuner_seed),
                    "version": (
                        f"Helion 1.4.0.dev1+g{adjudication.EXPECTED_COMMIT[:9]}; "
                        f"CuTe {adjudication.EXPECTED_CUTE_VERSION}"
                    ),
                    "result_path": paths["result.json"],
                    "result_sha256": strict.file_sha256(directory / "result.json"),
                    "autotune_csv_path": paths["autotune.csv"],
                    "autotune_metadata_path": paths["autotune.meta.jsonl"],
                    "source_ledger_path": paths["autotune.sources.csv"],
                }
            )
        all8_reference = {
            (str(row["variant"]), int(row["seq_len"])): row for row in all8_rows
        }
        with (
            mock.patch.object(
                heldout,
                "validate_individual_results",
                return_value=(
                    heldout_root / "campaign.csv",
                    rows,
                    all8_reference,
                    "f" * 64,
                ),
            ),
            mock.patch.object(adjudication, "require_repo_identity"),
        ):
            runner.initialize_campaign(
                root,
                repo.resolve(),
                heldout_root.resolve(),
                all8_root.resolve(),
                Path("/canonical/python"),
            )
        return (
            root,
            repo.resolve(),
            heldout_root.resolve(),
            rows,
            all8_rows,
            source_text,
        )

    def write_case_result(
        self,
        root: Path,
        campaign: dict[str, object],
        case: adjudication.CaseDefinition,
        source_text: str,
        *,
        complete: bool = True,
    ) -> dict[str, object]:
        declaration = adjudication.case_record(campaign, case.case_id)
        output = root / declaration["result_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        source_directory = output.parent / "generated_sources"
        source_directory.mkdir()
        selected_sources = []
        for contender in declaration["contenders"]:
            archive = source_directory / f"{contender['name']}.py.txt"
            archive.write_text(source_text)
            archive.chmod(0o444)
            selected_sources.append(
                {
                    "name": contender["name"],
                    "origin_kind": contender["origin_kind"],
                    "origin_result_sha256": contender["origin_result_sha256"],
                    "origin_selected_source_sha256": contender[
                        "origin_selected_source_sha256"
                    ],
                    "selected_config_sha256": contender["selected_config_sha256"],
                    "regenerated_source_sha256": contender[
                        "expected_regenerated_source_sha256"
                    ],
                    "compiled_source_sha256": contender[
                        "expected_regenerated_source_sha256"
                    ],
                    "archive_path": f"generated_sources/{contender['name']}.py.txt",
                }
            )
        names = [contender["name"] for contender in declaration["contenders"]]
        all_names = [*names, "sdpa"]
        protocol = declaration["protocol"]
        orders = adjudication.balanced_orders(
            all_names, protocol["rounds"], protocol["order_seed"]
        )
        raw_rounds = []
        for round_index, order in enumerate(orders):
            raw_rounds.append(
                {
                    "round_index": round_index,
                    "order": order,
                    "times": {
                        name: {
                            "event_ms": 1.02 if name == names[1] else 1.0,
                            "wall_ms": 1.021 if name == names[1] else 1.001,
                        }
                        for name in order
                    },
                }
            )
        element_count = 2 * 32 * case.seq_len * 64
        correctness = {
            name: {
                "numerics": {
                    "count": element_count,
                    "close_count": element_count,
                    "max_abs": 0.0,
                    "rmse": 0.0,
                    "nrmse": 0.0,
                    "actual_nonfinite": 0,
                    "atol": 5e-2,
                    "rtol": 2e-2,
                    "passed": True,
                },
                "repeatability": [
                    {
                        "repeat_index": repeat_index,
                        "passed": True,
                        "count": element_count,
                        "different": 0,
                    }
                    for repeat_index in range(1, adjudication.REPEATABILITY_LAUNCHES)
                ],
            }
            for name in all_names
        }
        direct = [
            float(contender["direct_search_tflops"])
            for contender in declaration["contenders"]
        ]
        summary = adjudication.summarize_measurements(
            raw_rounds,
            names,
            case,
            bootstrap_samples=adjudication.BOOTSTRAP_SAMPLES,
            bootstrap_seed=protocol["bootstrap_seed"],
            direct_tflops=direct,
        )
        gpu = {
            "physical_gpu": case.physical_gpu,
            "name": "NVIDIA B200",
            "uuid": "GPU-fixture",
            "power_limit_w": 750.0,
            "active_compute_pids": [],
        }
        payload = {
            "schema_version": adjudication.RESULT_SCHEMA_VERSION,
            "status": "MEASUREMENT_COMPLETE",
            "case_id": case.case_id,
            "shape": declaration["shape"],
            "physical_gpu": case.physical_gpu,
            "measured_commit": adjudication.EXPECTED_COMMIT,
            "campaign_sha256": campaign["_declaration_sha256"],
            "worker_sha256": adjudication.source_snapshot_hashes(campaign)[
                "remeasure_heldout_winners.py"
            ],
            "source_snapshot_sha256": adjudication.source_snapshot_hashes(campaign),
            "gpu_start": gpu,
            "gpu_end": gpu,
            "environment": {
                "torch_version": "fixture",
                "cudnn_version": 92000,
                "cute_version": adjudication.EXPECTED_CUTE_VERSION,
                "helion_version": campaign["validated_heldout_rows"][0]["version"],
            },
            "protocol": {
                **protocol,
                "timing_repetitions": 20,
                "sdpa_calibration_event_ms": [1.0, 1.0, 1.0],
            },
            "selected_sources": selected_sources,
            "correctness": correctness,
            "direct_search_tflops": direct,
            "raw_rounds": raw_rounds,
            "summary": summary,
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        output.chmod(0o444)
        if complete:
            completion = {
                "schema_version": 1,
                "status": "POSTCONDITIONS_PASSED",
                "case_id": case.case_id,
                "campaign_sha256": campaign["_declaration_sha256"],
                "worker_sha256": adjudication.source_snapshot_hashes(campaign)[
                    "remeasure_heldout_winners.py"
                ],
                "result_sha256": strict.file_sha256(output),
            }
            completion_path = output.with_name(adjudication.COMPLETION_FILENAME)
            completion_path.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n"
            )
            completion_path.chmod(0o444)
        return payload

    def test_balanced_orders_are_deterministic_and_position_balanced(self) -> None:
        names = ["a", "b", "c", "d", "e", "sdpa"]
        orders = adjudication.balanced_orders(names, 12, 123)

        self.assertEqual(orders, adjudication.balanced_orders(names, 12, 123))
        for name in names:
            self.assertEqual(
                [
                    sum(order[position] == name for order in orders)
                    for position in range(6)
                ],
                [2, 2, 2, 2, 2, 2],
            )

    def test_bootstraps_pin_the_complete_snapshot_set(self) -> None:
        self.assertEqual(
            worker.EXPECTED_SNAPSHOT_FILENAMES,
            adjudication.SNAPSHOT_FILENAMES,
        )
        self.assertEqual(
            validator.EXPECTED_SNAPSHOT_FILENAMES,
            adjudication.SNAPSHOT_FILENAMES,
        )

    def test_campaign_lock_rejects_concurrent_supervisors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            lock = runner.campaign_lock(root)
            lock.__enter__()
            try:
                with (
                    self.assertRaisesRegex(
                        RuntimeError, "another adjudication process holds"
                    ),
                    runner.campaign_lock(root),
                ):
                    self.fail("concurrent campaign lock unexpectedly succeeded")
            finally:
                lock.__exit__(None, None, None)

    def test_case_lock_rejects_an_orphaned_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "campaign"
            lock = runner.case_lock(root, "dense_s81920")
            lock.__enter__()
            try:
                with (
                    self.assertRaisesRegex(RuntimeError, "prior dense_s81920 worker"),
                    runner.case_lock(root, "dense_s81920"),
                ):
                    self.fail("concurrent case lock unexpectedly succeeded")
            finally:
                lock.__exit__(None, None, None)

    def test_attempt_history_ignores_only_a_torn_trailing_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "events.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "event": "attempt_started",
                        "case_id": "dense_s81920",
                    }
                )
                + "\n"
                + '{"event":"attempt_started"'
            )
            logs = root / "logs"
            logs.mkdir()
            (logs / "dense_s81920.attempt-2.log").write_text("partial\n")

            self.assertEqual(runner.prior_attempt_count(root, "dense_s81920"), 2)

    def test_attempt_history_rejects_a_committed_malformed_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "events.jsonl").write_text('{"event": broken}\n')

            with self.assertRaisesRegex(RuntimeError, "invalid event record"):
                runner.prior_attempt_count(root, "dense_s81920")

    def test_attempt_history_counts_a_valid_unterminated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "events.jsonl").write_text(
                json.dumps(
                    {
                        "event": "attempt_started",
                        "case_id": "dense_s81920",
                    }
                )
            )

            self.assertEqual(runner.prior_attempt_count(root, "dense_s81920"), 1)

    def test_event_append_repairs_a_torn_trailing_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "events.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "event": "attempt_started",
                        "case_id": "dense_s81920",
                    }
                )
                + "\n"
                + '{"event":"attempt_finished"'
            )

            runner.append_event(
                root,
                {
                    "event": "attempt_finished",
                    "case_id": "dense_s81920",
                    "returncode": 0,
                },
            )

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                [record["event"] for record in records],
                [
                    "attempt_started",
                    "attempt_finished",
                ],
            )

    def test_campaign_snapshot_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _repo, _heldout_root, rows, all8_rows, _source = self.make_fixture(
                Path(directory)
            )
            with mock.patch.object(
                heldout,
                "validate_individual_results",
                return_value=(
                    Path("/campaign.csv"),
                    rows,
                    {
                        (str(row["variant"]), int(row["seq_len"])): row
                        for row in all8_rows
                    },
                    "f" * 64,
                ),
            ):
                adjudication.validate_campaign(root, deep_artifact_validation=True)
                snapshot = root / "launcher" / "remeasure_heldout_winners.py"
                snapshot.chmod(0o644)
                snapshot.write_text(snapshot.read_text() + "# changed\n")
                with self.assertRaisesRegex(RuntimeError, "source snapshot"):
                    adjudication.validate_campaign(root, deep_artifact_validation=False)

    def test_campaign_all8_evidence_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _repo, _heldout_root, _rows, _all8_rows, _source = self.make_fixture(
                Path(directory)
            )
            campaign = adjudication.validate_campaign(
                root, deep_artifact_validation=False
            )
            evidence = (
                Path(campaign["all8_root"]) / campaign["all8_evidence"][0]["path"]
            )
            evidence.write_text(evidence.read_text() + "changed\n")

            with self.assertRaisesRegex(RuntimeError, "all8 evidence"):
                adjudication.validate_campaign(root, deep_artifact_validation=False)

    def test_campaign_records_the_linked_all8_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _repo, _heldout_root, _rows, all8_rows, _source = self.make_fixture(
                Path(directory)
            )
            campaign = adjudication.validate_campaign(
                root, deep_artifact_validation=False
            )

        self.assertEqual(campaign["validated_all8_rows"], all8_rows)
        self.assertEqual(campaign["all8_reference_manifest_sha256"], "f" * 64)
        self.assertEqual(len(campaign["all8_evidence"]), 4 * len(strict.CASES))

    def test_output_validation_accepts_a_failed_performance_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _repo, _heldout_root, rows, all8_rows, source = self.make_fixture(
                Path(directory)
            )
            campaign = adjudication.validate_campaign(
                root, deep_artifact_validation=False
            )
            for case in adjudication.CASES:
                self.write_case_result(root, campaign, case, source)
            report = adjudication.build_report(root, campaign)
            self.assertEqual(report["performance_gate_status"], "FAIL")
            self.assertEqual(report["direct_search_gate_status"], "FAIL")
            self.assertEqual(report["cross_measured_gate_status"], "FAIL")
            runner.write_immutable_json(root / "adjudication_report.json", report)
            with mock.patch.object(
                heldout,
                "validate_individual_results",
                return_value=(
                    Path("/campaign.csv"),
                    rows,
                    {
                        (str(row["variant"]), int(row["seq_len"])): row
                        for row in all8_rows
                    },
                    "f" * 64,
                ),
            ):
                self.assertEqual(adjudication.validate_complete_campaign(root), report)

    def test_output_validation_rejects_changed_randomized_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _repo, _heldout_root, _rows, _all8_rows, source = self.make_fixture(
                Path(directory)
            )
            campaign = adjudication.validate_campaign(
                root, deep_artifact_validation=False
            )
            case = adjudication.CASES[0]
            payload = self.write_case_result(root, campaign, case, source)
            path = (
                root / adjudication.case_record(campaign, case.case_id)["result_path"]
            )
            path.chmod(0o644)
            payload["raw_rounds"][0]["order"] = list(
                reversed(payload["raw_rounds"][0]["order"])
            )
            path.write_text(json.dumps(payload) + "\n")
            path.chmod(0o444)

            with self.assertRaisesRegex(RuntimeError, "round order"):
                adjudication.validate_case_output(root, campaign, case.case_id)

    def test_output_validation_rejects_changed_regenerated_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _repo, _heldout_root, _rows, _all8_rows, source = self.make_fixture(
                Path(directory)
            )
            campaign = adjudication.validate_campaign(
                root, deep_artifact_validation=False
            )
            case = adjudication.CASES[0]
            self.write_case_result(root, campaign, case, source)
            result = (
                root / adjudication.case_record(campaign, case.case_id)["result_path"]
            )
            archive = result.parent / "generated_sources" / "seed_2026082301.py.txt"
            archive.chmod(0o644)
            archive.write_text("# changed\n")
            archive.chmod(0o444)

            with self.assertRaisesRegex(RuntimeError, "source archive"):
                adjudication.validate_case_output(root, campaign, case.case_id)

    def test_snapshot_worker_bootstraps_only_authenticated_local_modules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _repo, _heldout_root, _rows, _all8_rows, _source = self.make_fixture(
                Path(directory)
            )
            campaign = adjudication.validate_campaign(
                root, deep_artifact_validation=False
            )
            worker = root / "launcher" / "remeasure_heldout_winners.py"
            code = (
                "import importlib.util, pathlib; "
                f"p=pathlib.Path({str(worker)!r}); "
                "s=importlib.util.spec_from_file_location('snapshot_worker', p); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                f"m.bootstrap_local_modules(pathlib.Path({str(root)!r}), "
                f"{campaign['_declaration_sha256']!r}, "
                f"{adjudication.source_snapshot_hashes(campaign)[worker.name]!r}); "
                "print('authenticated')"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                check=True,
                capture_output=True,
                text=True,
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key != "PYTHONPATH"
                },
            )

        self.assertEqual(completed.stdout.strip(), "authenticated")

    def test_resume_removes_only_stale_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / ".staging" / "dense_s81920"
            staging.mkdir(parents=True)
            (staging / "partial").write_text("partial")

            runner.clear_stale_staging(staging)

            self.assertFalse(staging.exists())
            invalid = root / ".staging" / "causal_s196608"
            invalid.parent.mkdir(exist_ok=True)
            invalid.symlink_to(root)
            with self.assertRaisesRegex(RuntimeError, "invalid stale staging"):
                runner.clear_stale_staging(invalid)

    def test_resume_promotes_a_complete_authenticated_staging_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _repo, _heldout_root, _rows, _all8_rows, source = self.make_fixture(
                Path(directory)
            )
            campaign = adjudication.validate_campaign(
                root, deep_artifact_validation=False
            )
            case = adjudication.CASES[0]
            self.write_case_result(root, campaign, case, source)
            final_directory = (
                root / adjudication.case_record(campaign, case.case_id)["result_path"]
            ).parent
            staging = root / ".staging" / case.case_id
            staging.parent.mkdir()
            final_directory.replace(staging)

            self.assertTrue(
                runner.recover_staged_result(
                    root, campaign, case, staging, final_directory
                )
            )
            self.assertTrue((final_directory / "result.json").is_file())
            self.assertFalse(staging.exists())

    def test_resume_rejects_valid_result_without_postcondition_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, _repo, _heldout_root, _rows, _all8_rows, source = self.make_fixture(
                Path(directory)
            )
            campaign = adjudication.validate_campaign(
                root, deep_artifact_validation=False
            )
            case = adjudication.CASES[0]
            self.write_case_result(root, campaign, case, source, complete=False)
            final_directory = (
                root / adjudication.case_record(campaign, case.case_id)["result_path"]
            ).parent
            staging = root / ".staging" / case.case_id
            staging.parent.mkdir()
            final_directory.replace(staging)

            self.assertFalse(
                runner.recover_staged_result(
                    root, campaign, case, staging, final_directory
                )
            )
            self.assertFalse(final_directory.exists())
            self.assertFalse(staging.exists())

    def test_worker_environment_scrubs_gpu_and_compiler_overrides(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(
                os.environ,
                {
                    "ADJUDICATION_CASE_LOCK_FD": "99",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "CUDA_LAUNCH_BLOCKING": "1",
                    "CUDNN_LOGLEVEL_DBG": "3",
                    "HELION_CONFIG": "untrusted",
                    "TRITON_CACHE_DIR": "/untrusted",
                },
                clear=False,
            ),
        ):
            root = Path(directory)
            env = runner.worker_environment(root, Path("/repo"), adjudication.CASES[0])

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(env["HELION_BACKEND"], "cute")
        self.assertEqual(env["HELION_DISABLE_AUTOTUNER_HEURISTICS"], "0")
        self.assertNotIn("CUDNN_LOGLEVEL_DBG", env)
        self.assertNotIn("CUDA_LAUNCH_BLOCKING", env)
        self.assertNotIn("ADJUDICATION_CASE_LOCK_FD", env)
        self.assertNotIn("HELION_CONFIG", env)
        self.assertEqual(
            env["PYTHONPATH"],
            os.pathsep.join((str(root / "launcher"), "/repo")),
        )

    def test_wait_for_process_drains_term_ignoring_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            child_pid_path = Path(directory) / "child.pid"
            process = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    (
                        f"(trap '' TERM; echo $BASHPID > {child_pid_path}; "
                        "while true; do sleep 1; done) & "
                        "trap 'exit 0' TERM; while true; do sleep 1; done"
                    ),
                ],
                start_new_session=True,
            )
            deadline = time.monotonic() + 5
            while not child_pid_path.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(child_pid_path.is_file())
            child_pid = int(child_pid_path.read_text())
            stop = threading.Event()
            stop.set()
            try:
                with mock.patch.object(runner, "PROCESS_GROUP_TERM_GRACE_SECONDS", 0.1):
                    self.assertEqual(runner.wait_for_process(process, stop), 0)
                self.assertFalse(runner.process_group_live(process.pid))
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)

    def test_wait_for_process_preserves_failure_while_draining_descendant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            child_pid_path = Path(directory) / "child.pid"
            process = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    (
                        f"(trap '' TERM; echo $BASHPID > {child_pid_path}; "
                        "while true; do sleep 1; done) & sleep 0.1; exit 17"
                    ),
                ],
                start_new_session=True,
            )
            deadline = time.monotonic() + 5
            while not child_pid_path.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(child_pid_path.is_file())
            child_pid = int(child_pid_path.read_text())
            try:
                with mock.patch.object(runner, "PROCESS_GROUP_TERM_GRACE_SECONDS", 0.1):
                    self.assertEqual(
                        runner.wait_for_process(process, threading.Event()), 17
                    )
                self.assertFalse(runner.process_group_live(process.pid))
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
