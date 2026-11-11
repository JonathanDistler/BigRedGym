"""Checksum preflight to run inside the target runtime; never starts training."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BALANCE = "logs/go2balance/Oct31_23-29-27_/model_1500.pt"
TRAIN = "datasets/go2kickup_recovery_train_v2/train.npz"
KICK = (
    "logs/go2frontyield_fresh_200_20261110_034101_1da5fd/Nov10_03-42-51_/model_200.pt"
)


def checksum(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def build_manifest(root, dataset_dir=None):
    root = Path(root)
    groups = {
        "training": [
            BALANCE,
            TRAIN,
            "datasets/go2kickup_recovery_train_v2/labels.json",
            "datasets/go2kickup_recovery_train_v2/provenance.json",
        ],
        "evaluation": [
            KICK,
            "output/handoff_reference_search/candidate_reference.json",
            "output/handoff_cpg_comparison/atlas.json",
            "gym/envs/go2/references/posture_transition_adaptive.json",
        ],
        "protected": [
            "gym/envs/go2/go2balance.py",
            "gym/envs/go2/go2balance_config.py",
            "gym/envs/go2/contact_cpg.py",
            "gym/envs/go2/references/kickup_front_yield.json",
        ],
    }
    dataset = TRAIN
    if dataset_dir is not None:
        directory = (root / dataset_dir).resolve()
        dataset = str((directory / "train.npz").relative_to(root.resolve()))
        groups["training"] = [BALANCE] + [
            str(path.relative_to(root.resolve()))
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix in (".npz", ".json")
        ]
    # Checkpoint loading uses its adjacent saved source, not today's defaults.
    for checkpoint in (BALANCE, KICK):
        category = "training" if checkpoint == BALANCE else "evaluation"
        saved = root / checkpoint
        sources = sorted((saved.parent / "files").rglob("*.py"))
        if not sources:
            raise FileNotFoundError(f"No saved source beside {checkpoint}")
        groups[category].extend(str(p.relative_to(root)) for p in sources)
    groups["implementation"] = [
        str(p.relative_to(root)) for p in sorted((root / "gym").rglob("*.py"))
    ]
    groups["implementation"] += [
        str(p.relative_to(root)) for p in sorted((root / "learning").rglob("*.py"))
    ]
    groups["implementation"] += [
        str(p.relative_to(root)) for p in sorted((root / "scripts").glob("*.py"))
    ]
    groups["implementation"] += ["pyproject.toml"]
    groups["training"] += [
        str(p.relative_to(root))
        for p in sorted((root / "resources/robots/go2").rglob("*"))
        if p.is_file()
    ]
    files = {}
    for category, paths in groups.items():
        for name in paths:
            path = root / name
            files[name] = dict(
                category=category, bytes=path.stat().st_size, sha256=checksum(path)
            )
    return dict(
        version=1,
        branch="ppo_kickup",
        checkpoint=BALANCE,
        dataset=dataset,
        created_utc=datetime.now(timezone.utc).isoformat(),
        files=files,
    )


def verify_manifest(root, manifest):
    root = Path(root).resolve()
    results = {}
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path escapes repository")
        actual = checksum(path) if path.is_file() else None
        results[name] = dict(
            exists=actual is not None,
            matches=actual == expected["sha256"],
            category=expected["category"],
            sha256=actual,
        )
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return dict(
        verified=all(r["matches"] for r in results.values())
        and branch == manifest["branch"],
        branch=branch,
        repository=str(root),
        files=results,
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/recovery_assets_manifest_v1.json"),
    )
    p.add_argument(
        "--write-manifest",
        action="store_true",
        help="Create expected checksums locally, exclusively.",
    )
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--environment",
        choices=("local", "molab"),
        default="local",
        help="User-supplied runtime label; run in Molab for remote evidence.",
    )
    p.add_argument("--verify-interface", action="store_true")
    p.add_argument(
        "--dataset-dir",
        type=Path,
        help="Expanded bank directory when creating a new manifest",
    )
    a = p.parse_args()
    if a.write_manifest:
        with a.manifest.open("x") as f:
            json.dump(build_manifest(ROOT, a.dataset_dir), f, indent=2)
        print(f"Created {a.manifest}; no remote verification performed.")
        return
    if a.output is None:
        p.error("--output is required for verification")
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads(a.manifest.read_text())
    report = verify_manifest(ROOT, manifest)
    report.update(
        environment_label=a.environment,
        hostname=socket.gethostname(),
        python=sys.executable,
        python_version=platform.python_version(),
        verified_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=checksum(a.manifest),
    )
    if a.verify_interface and report["verified"]:
        from scripts.audit_recovery_dataset import audit_interface

        report["interface"] = audit_interface(ROOT / Path(manifest["dataset"]).parent)
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)
    missing = [name for name, value in report["files"].items() if not value["matches"]]
    print(
        json.dumps(
            dict(
                verified=report["verified"],
                environment=a.environment,
                hostname=report["hostname"],
                mismatched_or_missing=missing,
            ),
            indent=2,
        )
    )
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
