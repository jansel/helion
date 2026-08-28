from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import unittest

import build_heldout_manifest as heldout

SCRIPT = Path(__file__).with_name("run_strict_all8.sh")
HELDOUT_SCRIPT = Path(__file__).with_name("run_strict_heldout.sh")
ACTIVE_BUNDLE = SCRIPT.parent.parent / "helion-generalized-c3e36b65.bundle"
ACTIVE_BUNDLE_SHA256 = (
    "a39cfcd01206c36609a178ee483e40194189a00f878e8d48f68a082aec3cdfad"
)
HISTORICAL_EDD79764_BUNDLE = SCRIPT.parent.parent / "helion-generalized-edd79764.bundle"
REPO_ROOT = SCRIPT.parents[3]


def run_bash(
    source: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", source],
        check=False,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )


class StrictRunnerTests(unittest.TestCase):
    @staticmethod
    def write_fake_python(path: Path) -> None:
        path.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ ${1-} = -c ]]; then\n"
            "  if [[ ${2-} = *nvidia-cutlass-dsl* ]]; then\n"
            "    printf '4.7.0\\n'\n"
            "    exit 0\n"
            "  fi\n"
            "  printf '3.12.0 strict-runner fixture\\n'\n"
            "  exit 0\n"
            "fi\n"
            'env > "$STRICT_RUNNER_CAPTURE"\n'
            'printf "%s\\n" "$@" > "$STRICT_RUNNER_ARGS"\n'
            "if [[ -n ${STRICT_RUNNER_INVOCATIONS-} ]]; then\n"
            '  printf "%s\\n" "$*" >> "$STRICT_RUNNER_INVOCATIONS"\n'
            "fi\n"
            "previous=\n"
            'for argument in "$@"; do\n'
            "  if [[ $previous = --json-output ]]; then\n"
            '    printf "{}\\n" > "$argument"\n'
            "    break\n"
            "  fi\n"
            "  previous=$argument\n"
            "done\n"
        )
        path.chmod(0o755)

    @staticmethod
    def write_fake_nvidia_smi(path: Path) -> None:
        path.write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "$*" >> "$STRICT_NVIDIA_LOG"\n'
            "if [[ -n ${STRICT_NVIDIA_BUSY_ON_QUERY-} ]]; then\n"
            '  count=$(wc -l < "$STRICT_NVIDIA_LOG")\n'
            "  if [[ $count = $STRICT_NVIDIA_BUSY_ON_QUERY ]]; then\n"
            "    printf '4242\\n'\n"
            "  fi\n"
            "fi\n"
        )
        path.chmod(0o755)

    @staticmethod
    def initialize_checkout(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        (root / ".gitignore").write_text(
            ".pytest_cache/\n.ruff_cache/\n__pycache__/\n/torch/\n"
        )
        (root / "tracked.txt").write_text("tracked\n")
        subprocess.run(
            ["git", "add", ".gitignore", "tracked.txt"], cwd=root, check=True
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Validation Test",
                "-c",
                "user.email=validation@example.com",
                "commit",
                "-qm",
                "initial",
            ],
            cwd=root,
            check=True,
        )

    @staticmethod
    def stage_allowed_harness(root: Path) -> None:
        for path in (
            ".validation/generalized_paired/combine_results.py",
            ".validation/generalized_paired/paired_worker.py",
            ".validation/generalized_paired/run_all8.py",
            ".validation/generalized_paired/test_static.py",
        ):
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("# staged validation harness\n")

    def test_checkout_allows_only_staged_harness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize_checkout(root)
            self.stage_allowed_harness(root)
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                validate_checkout_clean {shlex.quote(str(root))}
                """
            )

            result = run_bash(command)

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_checkout_rejects_startup_hook_and_helion_addition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize_checkout(root)
            self.stage_allowed_harness(root)
            (root / "sitecustomize.py").write_text("raise RuntimeError\n")
            (root / "helion").mkdir()
            (root / "helion/shadow.py").write_text("raise RuntimeError\n")
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                validate_checkout_clean {shlex.quote(str(root))}
                """
            )

            result = run_bash(command)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("helion/shadow.py", result.stderr)
        self.assertIn("sitecustomize.py", result.stderr)

    def test_checkout_rejects_checkout_local_bytecode_and_tool_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize_checkout(root)
            self.stage_allowed_harness(root)
            for path in (
                root / ".pytest_cache/v/cache/nodeids",
                root / "helion/__pycache__/module.cpython-312.pyc",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("untrusted cache\n")
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                validate_checkout_clean {shlex.quote(str(root))}
                """
            )

            result = run_bash(command)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("caches must be external", result.stderr)
        self.assertIn(".pytest_cache", result.stderr)
        self.assertIn("helion/__pycache__", result.stderr)

    def test_checkout_rejects_ignored_import_shadow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize_checkout(root)
            self.stage_allowed_harness(root)
            shadow = root / "torch/__init__.py"
            shadow.parent.mkdir()
            shadow.write_text("raise RuntimeError\n")
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                validate_checkout_clean {shlex.quote(str(root))}
                """
            )

            result = run_bash(command)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("caches must be external", result.stderr)
        self.assertIn("torch/__init__.py", result.stderr)

    def test_checkout_rejects_unexpected_git_describe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.initialize_checkout(root)
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                EXPECTED_COMMIT=$(git -C {shlex.quote(str(root))} rev-parse HEAD)
                EXPECTED_GIT_DESCRIBE_PREFIX=
                validate_checkout_identity {shlex.quote(str(root))}
                git -C {shlex.quote(str(root))} tag helion-generalized-fixture
                ! validate_checkout_identity {shlex.quote(str(root))}
                """
            )

            result = run_bash(command)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("measured repository version changed", result.stderr)

    def test_historical_bundle_archive_preserves_its_git_describe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_ref = "refs/archive/helion-generalized-edd79764"
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--no-tags",
                    str(REPO_ROOT),
                    "refs/tags/v1.4.0:refs/tags/v1.4.0",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--no-tags",
                    str(HISTORICAL_EDD79764_BUNDLE),
                    f"{archive_ref}:{archive_ref}",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "checkout",
                    "--detach",
                    archive_ref,
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            describe = subprocess.check_output(
                ["git", "describe", "--tags", "--always", "--dirty"],
                cwd=root,
                text=True,
            ).strip()
            campaign_tag = subprocess.run(
                [
                    "git",
                    "show-ref",
                    "--verify",
                    "refs/tags/helion-generalized-edd79764",
                ],
                cwd=root,
                check=False,
                capture_output=True,
            )

        prefix = "v1.4.0-141-g"
        self.assertTrue(describe.startswith(prefix), describe)
        abbreviation = describe.removeprefix(prefix)
        self.assertGreaterEqual(len(abbreviation), 7)
        self.assertTrue(
            "edd79764349bdbd43dfb6afbcf4e620128aecc11".startswith(abbreviation)
        )
        self.assertNotEqual(campaign_tag.returncode, 0)

    def test_active_bundle_archive_preserves_its_git_describe(self) -> None:
        self.assertEqual(
            hashlib.sha256(ACTIVE_BUNDLE.read_bytes()).hexdigest(),
            ACTIVE_BUNDLE_SHA256,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_ref = "refs/archive/helion-generalized-c3e36b65"
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--no-tags",
                    str(REPO_ROOT),
                    "refs/tags/v1.4.0:refs/tags/v1.4.0",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "fetch",
                    "--no-tags",
                    str(ACTIVE_BUNDLE),
                    f"{archive_ref}:{archive_ref}",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "checkout", "--detach", archive_ref],
                cwd=root,
                check=True,
                capture_output=True,
            )
            describe = subprocess.check_output(
                ["git", "describe", "--tags", "--always", "--dirty"],
                cwd=root,
                text=True,
            ).strip()
            campaign_tag = subprocess.run(
                [
                    "git",
                    "show-ref",
                    "--verify",
                    "refs/tags/helion-generalized-c3e36b65",
                ],
                cwd=root,
                check=False,
                capture_output=True,
            )

        prefix = "v1.4.0-157-g"
        self.assertTrue(describe.startswith(prefix), describe)
        abbreviation = describe.removeprefix(prefix)
        self.assertGreaterEqual(len(abbreviation), 7)
        self.assertTrue(
            "c3e36b65d69681c23e053042b0bc21e2331bad17".startswith(abbreviation)
        )
        self.assertNotEqual(campaign_tag.returncode, 0)

    def test_output_root_must_be_outside_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root.parent / f"{root.name}-output"
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                validate_output_root_outside_checkout \
                  {shlex.quote(str(root))} {shlex.quote(str(outside))}
                ! validate_output_root_outside_checkout \
                  {shlex.quote(str(root))} {shlex.quote(str(root / "output"))}
                """
            )

            result = run_bash(command)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("must be outside", result.stderr)

    def test_campaign_lock_is_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory) / "campaign"
            holder = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    (
                        f"source {shlex.quote(str(SCRIPT))}; "
                        f"acquire_campaign_lock {shlex.quote(str(output_root))}; "
                        "printf ready; sleep 30"
                    ),
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertIsNotNone(holder.stdout)
                assert holder.stdout is not None
                self.assertEqual(holder.stdout.read(5), "ready")
                contender = run_bash(
                    f"source {shlex.quote(str(SCRIPT))}; "
                    f"acquire_campaign_lock {shlex.quote(str(output_root))}"
                )
                self.assertNotEqual(contender.returncode, 0)
                self.assertIn("another strict campaign process holds", contender.stderr)
            finally:
                holder.terminate()
                holder.wait(timeout=5)

    def test_completion_marker_detects_changes_and_partial_is_quarantined(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "campaign"
            output = output_root / "dense_s32768"
            output.mkdir(parents=True)
            (output / "result.json").write_text('{"status":"ok"}\n')
            python = root / "python"
            python.write_text("#!/usr/bin/env bash\n")
            python.chmod(0o755)
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                write_shape_completion_marker \
                  {shlex.quote(str(output))} 99 7 0 32768 101 \
                  {shlex.quote(str(python))}
                validate_shape_completion_marker \
                  {shlex.quote(str(output))} 99 7 0 32768 101 \
                  {shlex.quote(str(python))}
                printf 'tampered\n' >> {shlex.quote(str(output / "result.json"))}
                ! validate_shape_completion_marker \
                  {shlex.quote(str(output))} 99 7 0 32768 101 \
                  {shlex.quote(str(python))}
                mv -- {shlex.quote(str(output / ".launcher-complete"))} \
                  {shlex.quote(str(root / "saved-marker"))}
                quarantine_incomplete_shape \
                  {shlex.quote(str(output))} {shlex.quote(str(output_root))}
                [[ ! -e {shlex.quote(str(output))} ]]
                find {shlex.quote(str(Path(f"{output_root}.quarantine")))} \
                  -name result.json -print -quit | grep -q .
                """
            )

            result = run_bash(command)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("completed shape evidence changed", result.stderr)

    def test_lane_skips_authenticated_completed_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            self.initialize_checkout(repo)
            output_root = root / "output"
            output = output_root / "dense_s32768"
            output.mkdir(parents=True)
            (output / "result.json").write_text('{"status":"ok"}\n')
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            self.write_fake_python(fake_python)
            self.write_fake_nvidia_smi(fake_bin / "nvidia-smi")
            invocations = root / "invocations"
            nvidia_log = root / "nvidia.log"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "STRICT_RUNNER_CAPTURE": str(root / "env"),
                    "STRICT_RUNNER_ARGS": str(root / "args"),
                    "STRICT_RUNNER_INVOCATIONS": str(invocations),
                    "STRICT_NVIDIA_LOG": str(nvidia_log),
                }
            )
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                EXPECTED_COMMIT=$(git -C {shlex.quote(str(repo))} rev-parse HEAD)
                EXPECTED_GIT_DESCRIBE_PREFIX=
                python_executable=$(resolve_python_executable)
                write_shape_completion_marker \
                  {shlex.quote(str(output))} 99 7 0 32768 101 \
                  "$python_executable"
                run_lane {shlex.quote(str(repo))} {shlex.quote(str(output_root))} \
                  99 7 0 "$python_executable" "32768 101"
                [[ ! -e {shlex.quote(str(invocations))} ]]
                """
            )

            result = run_bash(command, env=env)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SKIP completed", result.stdout)

    def test_public_resume_delegation_restores_repo_and_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            self.initialize_checkout(repo)
            outside = root / "outside"
            outside.mkdir()
            output_root = root / "output"
            launcher = output_root / "launcher/run_strict_all8.sh"
            launcher.parent.mkdir(parents=True)
            capture = root / "delegated"
            launcher.write_text(
                "#!/usr/bin/env bash\n"
                f'printf "%s\\n" "$HELION_REPO_ROOT" > {shlex.quote(str(capture))}\n'
                f'printf "%s\\n" "$STRICT_PYTHON_EXECUTABLE" >> {shlex.quote(str(capture))}\n'
                f'printf "%s\\n" "$*" >> {shlex.quote(str(capture))}\n'
            )
            launcher.chmod(0o555)
            (launcher.parent / "run_strict_all8.sh.sha256").write_text(
                subprocess.check_output(
                    ["sha256sum", str(launcher)], text=True
                ).split()[0]
                + "\n"
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            self.write_fake_python(fake_python)
            real_git = shutil.which("git")
            self.assertIsNotNone(real_git)
            (fake_bin / "git").write_text(
                "#!/usr/bin/env bash\n"
                "if [[ $* = *'rev-parse HEAD'* ]]; then\n"
                "  printf 'c3e36b65d69681c23e053042b0bc21e2331bad17\\n'\n"
                "  exit 0\n"
                "fi\n"
                "if [[ $* = *'describe --tags --always --dirty'* ]]; then\n"
                "  printf 'v1.4.0-157-gc3e36b65d\\n'\n"
                "  exit 0\n"
                "fi\n"
                f'exec {shlex.quote(str(real_git))} "$@"\n'
            )
            (fake_bin / "git").chmod(0o755)
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "HELION_REPO_ROOT": str(repo),
                    "STRICT_PYTHON_EXECUTABLE": str(fake_python),
                }
            )

            result = subprocess.run(
                [str(SCRIPT), "--resume", str(output_root)],
                cwd=outside,
                env=env,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            capture_lines = capture.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            capture_lines,
            [str(repo), str(fake_python), f"--resume {output_root}"],
        )

    def test_sanitizes_search_and_cache_environment(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "HELION_AUTOTUNE_BUDGET_SECONDS": "1",
                "HELION_CUTE_FLASH_PIPELINE_FAMILY": "forced",
                "CUTE_DSL_COMPILER_OPT": "forced",
                "TRITON_CACHE_DIR": "/ambient/triton",
                "TORCHINDUCTOR_CACHE_DIR": "/ambient/torchinductor",
                "PYTORCH_TUNABLEOP_ENABLED": "1",
                "CUDA_CACHE_PATH": "/ambient/cuda",
                "CUDA_DEVICE_MAX_CONNECTIONS": "32",
                "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
                "CUDA_LAUNCH_BLOCKING": "1",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50",
                "CUDA_MODULE_LOADING": "LAZY",
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_TF32_OVERRIDE": "1",
                "PYTHONPATH": "/ambient/python",
                "PYTHONPYCACHEPREFIX": "/ambient/pycache",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64",
                "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
                "CUDNN_LOGLEVEL_DBG": "3",
                "TORCH_CUDNN_V8_API_DISABLED": "1",
                "XDG_CACHE_HOME": "/ambient/xdg",
                "STRICT_RUNNER_SENTINEL": "preserved",
            }
        )
        command = textwrap.dedent(
            f"""
            source {shlex.quote(str(SCRIPT))}
            sanitize_search_environment
            env
            """
        )

        result = run_bash(command, env=env)

        self.assertEqual(result.returncode, 0, result.stderr)
        child_env = dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )
        self.assertEqual(child_env["STRICT_RUNNER_SENTINEL"], "preserved")
        for variable in (
            "HELION_AUTOTUNE_BUDGET_SECONDS",
            "HELION_CUTE_FLASH_PIPELINE_FAMILY",
            "CUTE_DSL_COMPILER_OPT",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "PYTORCH_TUNABLEOP_ENABLED",
            "CUDA_CACHE_PATH",
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "CUDA_DEVICE_ORDER",
            "CUDA_LAUNCH_BLOCKING",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "CUDA_MODULE_LOADING",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_TF32_OVERRIDE",
            "PYTHONPATH",
            "PYTHONPYCACHEPREFIX",
            "PYTORCH_ALLOC_CONF",
            "PYTORCH_CUDA_ALLOC_CONF",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
            "CUDNN_LOGLEVEL_DBG",
            "TORCH_CUDNN_V8_API_DISABLED",
            "XDG_CACHE_HOME",
        ):
            self.assertNotIn(variable, child_env)

    def test_resolves_and_revalidates_one_canonical_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            real_python = fake_bin / "strict-python"
            self.write_fake_python(real_python)
            python_link = fake_bin / "python"
            python_link.symlink_to(real_python)
            env = os.environ.copy()
            env["PATH"] = f"{fake_bin}:{env['PATH']}"
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                default_python=$(resolve_python_executable)
                [[ $default_python = {shlex.quote(str(real_python))} ]]
                STRICT_PYTHON_EXECUTABLE=python
                override_python=$(resolve_python_executable)
                [[ $override_python = "$default_python" ]]
                PATH=/usr/bin:/bin
                validate_python_executable "$default_python"
                ! validate_python_executable {shlex.quote(str(python_link))}
                printf '%s\\n' "$default_python"
                """
            )

            result = run_bash(command, env=env)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(real_python))
        self.assertIn("changed or is not canonical", result.stderr)

    def test_rejects_wrong_cute_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ ${1-} = -c ]]; then printf '4.6.1\\n'; exit 0; fi\n"
            )
            python.chmod(0o755)
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                validate_python_executable {shlex.quote(str(python))}
                """
            )

            result = run_bash(command)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 4.7.0, got 4.6.1", result.stderr)

    def test_lane_uses_per_shape_compiler_caches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            self.initialize_checkout(repo)
            output_root = root / "output"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            self.write_fake_python(fake_python)
            self.write_fake_nvidia_smi(fake_bin / "nvidia-smi")
            capture = root / "env"
            args = root / "args"
            nvidia_log = root / "nvidia.log"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "HELION_AUTOTUNE_BUDGET_SECONDS": "1",
                    "CUTE_DSL_CACHE_DIR": "/ambient/cute",
                    "CUDA_CACHE_PATH": "/ambient/cuda",
                    "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
                    "CUDNN_LOGLEVEL_DBG": "3",
                    "TORCH_CUDNN_V8_API_DISABLED": "1",
                    "STRICT_RUNNER_CAPTURE": str(capture),
                    "STRICT_RUNNER_ARGS": str(args),
                    "STRICT_NVIDIA_LOG": str(nvidia_log),
                }
            )
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                EXPECTED_COMMIT=$(git -C {shlex.quote(str(repo))} rev-parse HEAD)
                EXPECTED_GIT_DESCRIBE_PREFIX=
                python_executable=$(resolve_python_executable)
                sanitize_search_environment
                run_lane {shlex.quote(str(repo))} {shlex.quote(str(output_root))} \
                  99 7 0 "$python_executable" "32768 101"
                """
            )

            result = run_bash(command, env=env)

            self.assertEqual(result.returncode, 0, result.stderr)
            child_env = dict(
                line.split("=", 1)
                for line in capture.read_text().splitlines()
                if "=" in line
            )
            cache_root = output_root / "dense_s32768" / "cache"
            self.assertEqual(child_env["CUDA_VISIBLE_DEVICES"], "7")
            self.assertEqual(child_env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertNotIn("PYTHONPATH", child_env)
            self.assertEqual(
                child_env["PYTHONPYCACHEPREFIX"], str(cache_root / "pycache")
            )
            self.assertEqual(
                child_env["CUTE_DSL_CACHE_DIR"], str(cache_root / "cute_dsl")
            )
            self.assertEqual(child_env["HELION_CACHE_DIR"], str(cache_root / "helion"))
            self.assertEqual(child_env["TRITON_CACHE_DIR"], str(cache_root / "triton"))
            self.assertEqual(child_env["XDG_CACHE_HOME"], str(cache_root / "xdg"))
            self.assertNotIn("HELION_AUTOTUNE_BUDGET_SECONDS", child_env)
            self.assertNotIn("CUDNN_LOGLEVEL_DBG", child_env)
            self.assertNotIn("TORCH_CUDNN_V8_API_DISABLED", child_env)
            self.assertIn("HELION_AUTOTUNE_RANDOM_SEED=101", args.read_text())
            self.assertEqual(
                nvidia_log.read_text().splitlines(),
                [
                    "-i 7 --query-compute-apps=pid --format=csv,noheader,nounits",
                    "-i 7 --query-compute-apps=pid --format=csv,noheader,nounits",
                ],
            )
            run_log = output_root / "dense_s32768" / "run.log"
            self.assertIn(f"PYTHON executable={fake_python}", run_log.read_text())

    def test_lane_rechecks_gpu_idleness_after_every_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            self.initialize_checkout(repo)
            output_root = root / "output"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            self.write_fake_python(fake_python)
            self.write_fake_nvidia_smi(fake_bin / "nvidia-smi")
            capture = root / "env"
            args = root / "args"
            invocations = root / "invocations"
            nvidia_log = root / "nvidia.log"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "STRICT_RUNNER_CAPTURE": str(capture),
                    "STRICT_RUNNER_ARGS": str(args),
                    "STRICT_RUNNER_INVOCATIONS": str(invocations),
                    "STRICT_NVIDIA_LOG": str(nvidia_log),
                    "STRICT_NVIDIA_BUSY_ON_QUERY": "2",
                }
            )
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                EXPECTED_COMMIT=$(git -C {shlex.quote(str(repo))} rev-parse HEAD)
                EXPECTED_GIT_DESCRIBE_PREFIX=
                python_executable=$(resolve_python_executable)
                run_lane {shlex.quote(str(repo))} {shlex.quote(str(output_root))} \
                  99 7 0 "$python_executable" "32768 101" "65536 102"
                """
            )

            result = run_bash(command, env=env)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("physical GPU 7 is not idle", result.stderr)
            self.assertEqual(len(nvidia_log.read_text().splitlines()), 2)
            self.assertEqual(len(invocations.read_text().splitlines()), 1)
            self.assertTrue((output_root / "dense_s32768").exists())
            self.assertFalse((output_root / "dense_s65536").exists())

    def test_lane_rechecks_checkout_identity_after_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            self.initialize_checkout(repo)
            output_root = root / "output"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ ${1-} = -c ]]; then\n"
                "  if [[ ${2-} = *nvidia-cutlass-dsl* ]]; then\n"
                "    printf '4.7.0\\n'\n"
                "  else\n"
                "    printf '3.12 fixture\\n'\n"
                "  fi\n"
                "  exit 0\n"
                "fi\n"
                f"printf 'changed\\n' >> {shlex.quote(str(repo / 'tracked.txt'))}\n"
            )
            fake_python.chmod(0o755)
            self.write_fake_nvidia_smi(fake_bin / "nvidia-smi")
            nvidia_log = root / "nvidia.log"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "STRICT_NVIDIA_LOG": str(nvidia_log),
                }
            )
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                EXPECTED_COMMIT=$(git -C {shlex.quote(str(repo))} rev-parse HEAD)
                EXPECTED_GIT_DESCRIBE_PREFIX=
                python_executable=$(resolve_python_executable)
                run_lane {shlex.quote(str(repo))} {shlex.quote(str(output_root))} \
                  99 7 0 "$python_executable" "32768 101"
                """
            )

            result = run_bash(command, env=env)

        self.assertNotEqual(result.returncode, 0)

    def test_lane_rejects_changed_head_before_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            self.initialize_checkout(repo)
            output_root = root / "output"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            self.write_fake_python(fake_python)
            self.write_fake_nvidia_smi(fake_bin / "nvidia-smi")
            nvidia_log = root / "nvidia.log"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "STRICT_RUNNER_CAPTURE": str(root / "env"),
                    "STRICT_RUNNER_ARGS": str(root / "args"),
                    "STRICT_NVIDIA_LOG": str(nvidia_log),
                }
            )
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                EXPECTED_COMMIT=0000000000000000000000000000000000000000
                python_executable=$(resolve_python_executable)
                run_lane {shlex.quote(str(repo))} {shlex.quote(str(output_root))} \
                  99 7 0 "$python_executable" "32768 101"
                """
            )

            result = run_bash(command, env=env)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("measured repository HEAD changed", result.stderr)

    def test_heldout_campaign_is_written_before_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(HELDOUT_SCRIPT))}
                write_heldout_campaign_manifest \
                  {shlex.quote(str(root))} /canonical/python \
                  {"a" * 64} {"b" * 64}
                """
            )

            result = run_bash(command)
            with (root / "campaign.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            rows,
            heldout.expected_campaign_rows(
                "/canonical/python",
                "a" * 64,
                "b" * 64,
            ),
        )

    def test_launcher_snapshot_is_read_only_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sh"
            destination = root / "snapshot" / "launcher.sh"
            source_contents = "#!/usr/bin/env bash\necho fixture\n"
            source.write_text(source_contents)
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(SCRIPT))}
                snapshot_launcher \
                  {shlex.quote(str(source))} {shlex.quote(str(destination))}
                [[ $(cat {shlex.quote(str(destination))}.sha256) = \
                  $(sha256sum {shlex.quote(str(destination))} | cut -d ' ' -f 1) ]]
                """
            )

            result = run_bash(command)
            snapshot = destination.read_text()
            mode = destination.stat().st_mode & 0o777

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot, source_contents)
        self.assertEqual(mode, 0o555)

    def test_heldout_lane_isolates_every_tuner_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            self.initialize_checkout(repo)
            output_root = root / "output"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python"
            self.write_fake_python(fake_python)
            self.write_fake_nvidia_smi(fake_bin / "nvidia-smi")
            real_git = shutil.which("git")
            self.assertIsNotNone(real_git)
            (fake_bin / "git").write_text(
                "#!/usr/bin/env bash\n"
                "if [[ $* = *'rev-parse HEAD'* ]]; then\n"
                "  printf 'c3e36b65d69681c23e053042b0bc21e2331bad17\\n'\n"
                "  exit 0\n"
                "fi\n"
                "if [[ $* = *'describe --tags --always --dirty'* ]]; then\n"
                "  printf 'v1.4.0-157-gc3e36b65d\\n'\n"
                "  exit 0\n"
                "fi\n"
                f'exec {shlex.quote(str(real_git))} "$@"\n'
            )
            (fake_bin / "git").chmod(0o755)
            capture = root / "env"
            args = root / "args"
            invocations = root / "invocations"
            nvidia_log = root / "nvidia.log"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{fake_bin}:{env['PATH']}",
                    "STRICT_RUNNER_CAPTURE": str(capture),
                    "STRICT_RUNNER_ARGS": str(args),
                    "STRICT_RUNNER_INVOCATIONS": str(invocations),
                    "STRICT_NVIDIA_LOG": str(nvidia_log),
                }
            )
            command = textwrap.dedent(
                f"""
                source {shlex.quote(str(HELDOUT_SCRIPT))}
                EXPECTED_COMMIT=$(git -C {shlex.quote(str(repo))} rev-parse HEAD)
                EXPECTED_GIT_DESCRIBE_PREFIX=
                python_executable=$(resolve_python_executable)
                run_heldout_lane {shlex.quote(str(repo))} {shlex.quote(str(output_root))} \
                  99 7 0 "$python_executable" "81920 101" "81920 102"
                """
            )

            result = run_bash(command, env=env)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(invocations.read_text().splitlines()), 2)
            for tuner_seed in (101, 102):
                run_log = (
                    output_root
                    / "dense"
                    / f"seed_{tuner_seed}"
                    / "dense_s81920"
                    / "run.log"
                )
                self.assertTrue(run_log.is_file())

    def test_lane_failure_terminates_peer_process_group(self) -> None:
        command = textwrap.dedent(
            f"""
            source {shlex.quote(str(SCRIPT))}
            ready=$(mktemp)
            rm -f "$ready"
            setsid bash -c 'trap "exit 0" TERM; touch "$1"; while :; do sleep 10; done' \
              peer "$ready" &
            peer_pid=$!
            while [[ ! -e $ready ]]; do sleep 0.01; done
            setsid bash -c 'sleep 0.1; exit 23' &
            failed_pid=$!
            active_lane_pids=("$failed_pid" "$peer_pid")
            set +e
            wait_for_lanes "$failed_pid" "$peer_pid"
            status=$?
            set -e
            rm -f "$ready"
            [[ $status = 23 ]]
            ! kill -0 -- "-$peer_pid" 2>/dev/null
            [[ ${{#active_lane_pids[@]}} = 0 ]]
            """
        )

        result = run_bash(command)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("failed with status 23", result.stderr)


if __name__ == "__main__":
    unittest.main()
