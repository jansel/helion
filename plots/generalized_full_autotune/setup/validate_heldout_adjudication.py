from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

EXPECTED_SNAPSHOT_FILENAMES = (
    "build_heldout_manifest.py",
    "build_strict_manifest.py",
    "heldout_adjudication.py",
    "remeasure_generalization_winners.py",
    "remeasure_heldout_winners.py",
    "run_heldout_adjudication.py",
    "validate_generalization_campaign.py",
    "validate_heldout_adjudication.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(campaign_root: Path) -> tuple[Any, Any]:
    campaign_path = campaign_root / "campaign.json"
    digest_path = campaign_root / "campaign.sha256"
    expected_campaign_sha256 = digest_path.read_text().strip()
    actual_campaign_sha256 = file_sha256(campaign_path)
    if actual_campaign_sha256 != expected_campaign_sha256:
        raise RuntimeError("campaign declaration digest mismatch")
    campaign = json.loads(campaign_path.read_text())
    if not isinstance(campaign, dict):
        raise RuntimeError("campaign declaration is not an object")
    snapshots = campaign.get("source_snapshots")
    if not isinstance(snapshots, list):
        raise RuntimeError("campaign has no source snapshot set")
    if [record.get("name") for record in snapshots if isinstance(record, dict)] != list(
        EXPECTED_SNAPSHOT_FILENAMES
    ):
        raise RuntimeError("campaign source snapshot set is not canonical")
    launcher = (campaign_root / "launcher").resolve()
    identities = {}
    for record in snapshots:
        if not isinstance(record, dict):
            raise RuntimeError("invalid source snapshot record")
        name = record.get("name")
        digest = record.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            raise RuntimeError("invalid source snapshot identity")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"invalid source snapshot digest: {digest!r}")
        path = launcher / name
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"invalid immutable source snapshot: {path}")
        if file_sha256(path) != digest:
            raise RuntimeError(f"source snapshot changed: {name}")
        identities[name] = digest
    current = Path(__file__).resolve()
    if current.parent != launcher:
        raise RuntimeError(
            f"validator is not running from snapshot directory: {current}"
        )
    if file_sha256(current) != identities.get(current.name):
        raise RuntimeError("validator source does not match campaign declaration")
    sys.path.insert(0, str(launcher))
    import build_heldout_manifest as heldout
    import build_strict_manifest as strict
    import heldout_adjudication as adjudication

    adjudication.validate_runtime_module_paths(
        campaign_root, [strict, heldout, adjudication]
    )
    return strict, adjudication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a completed held-out winner adjudication campaign."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.campaign_root.expanduser().resolve()
    strict, adjudication = bootstrap(root)
    campaign = adjudication.validate_campaign(root, deep_artifact_validation=False)
    strict.check_equal(
        Path(sys.executable).resolve(),
        Path(campaign["python_executable"]).resolve(),
        "validator Python executable",
    )
    report = adjudication.validate_complete_campaign(root)
    contents = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        forbidden_roots = (
            root,
            Path(campaign["repo_root"]),
            Path(campaign["heldout_root"]),
            Path(campaign["all8_root"]),
        )
        strict.require(
            all(
                output != item and item not in output.parents
                for item in forbidden_roots
            ),
            f"validation output collides with evidence: {output}",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        temporary.write_text(contents)
        temporary.replace(output)
    print(contents, end="")


if __name__ == "__main__":
    main()
