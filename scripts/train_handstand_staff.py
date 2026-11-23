"""Prepare public research sources, train from BC, evaluate and export a candidate.

Requires no private logs or pre-existing BC/PPO checkpoint. The public reference,
balance teacher and feedback assets remain necessary. No robot/SDK is imported.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import urllib.request

SOURCE_SHA = "2cf3a3afe6db87827b60b73c741617f64d054eb1c39a46b5858eb6ab678a77a4"
SOURCE_URL = (
    "https://github.com/JonathanDistler/BigRedGym/releases/download/"
    "handstand-capture-67of100-v1/handstand_capture_bundle.tar.gz"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(workspace, archive=None):
    source = workspace / "source"
    if source.exists():
        raise FileExistsError(
            "Source already exists; use it or choose a fresh workspace"
        )
    workspace.mkdir(parents=True, exist_ok=True)
    archive = archive or workspace / "handstand_capture_bundle.tar.gz"
    if not archive.exists():
        print(f"Downloading public source/reference assets: {SOURCE_URL}", flush=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            archive.write_bytes(response.read())
    if digest(archive) != SOURCE_SHA:
        raise ValueError("Public source archive SHA256 mismatch")
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            p = PurePosixPath(member.name)
            if (
                p.is_absolute()
                or ".." in p.parts
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe source archive member: {member.name}")
        source.mkdir()
        for member in members:
            path_parts = PurePosixPath(member.name).parts
            code = path_parts[0] in {
                "experiments",
                "gym",
                "scripts",
                "configs",
                "resources",
            }
            dependency = member.name in {"pyproject.toml", "uv.lock"}
            teacher = member.name in {
                "artifacts/balance.pt",
                "artifacts/takeoff_teacher.npz",
                "artifacts/handstand_reference/trajectory.npz",
            }
            if member.isfile() and (code or dependency or teacher):
                path = source / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(member).read())
    # All dependencies are public assets, not a user's old logs.
    for name in (
        "balance.pt",
        "takeoff_teacher.npz",
        "handstand_reference/trajectory.npz",
    ):
        if not (source / "artifacts" / name).is_file():
            raise FileNotFoundError(f"Missing public training dependency: {name}")
    return source


def commands(workspace, name, seed, trials):
    source = workspace / "source"
    run = workspace / "runs" / name
    py = sys.executable
    assets = source / "artifacts"
    return [
        [
            py,
            str(source / "scripts/handstand_capture.py"),
            "bc",
            "--seed",
            str(seed),
            "--demonstrations",
            "12",
            "--epochs",
            "100",
            "--trials",
            "4",
            "--trajectory",
            str(assets / "handstand_reference/trajectory.npz"),
            "--balance",
            str(assets / "balance.pt"),
            "--feedback",
            str(assets / "takeoff_teacher.npz"),
            "--output",
            str(run / "bc"),
        ],
        [
            py,
            str(source / "scripts/handstand_takeoff.py"),
            "train",
            "--checkpoint",
            str(run / "bc/bc.pt"),
            "--updates",
            "2",
            "--envs",
            "1",
            "--rollout-steps",
            "128",
            "--trials",
            "2",
            "--seed",
            str(seed),
            "--trajectory",
            str(assets / "handstand_reference/trajectory.npz"),
            "--output",
            str(run / "ppo"),
        ],
        [
            py,
            str(source / "scripts/handstand_capture.py"),
            "full",
            "--checkpoint",
            str(run / "ppo/latest.pt"),
            "--controllers",
            "student",
            "--evaluation-seed",
            "301000",
            "--trials",
            str(trials),
            "--trajectory",
            str(assets / "handstand_reference/trajectory.npz"),
            "--balance",
            str(assets / "balance.pt"),
            "--feedback",
            str(assets / "takeoff_teacher.npz"),
            "--output",
            str(run / "evaluation"),
        ],
        [
            py,
            str(
                Path(__file__).resolve().parents[1]
                / "go2_deploy/custom_implementation/go2handstand.py"
            ),
            "--checkpoint",
            str(run / "ppo/latest.pt"),
            "--reference",
            str(assets / "handstand_reference/trajectory.npz"),
            "--candidate",
            "--export-bundle",
            str(run / "bundle"),
        ],
    ]


def train(workspace, name, seed, trials, dry_run=False):
    source = workspace / "source"
    if not source.is_dir():
        raise FileNotFoundError("Run prepare first")
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a simple directory name")
    plan = commands(workspace, name, seed, trials)
    if dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    import mujoco

    if sys.version_info[:2] != (3, 11) or mujoco.__version__ != "3.11.0":
        raise ValueError("Fresh training requires Python 3.11 and MuJoCo 3.11.0")
    run = workspace / "runs" / name
    run.mkdir(parents=True, exist_ok=False)
    files = run / "files"
    for root in ("experiments", "gym", "scripts", "configs", "resources"):
        shutil.copytree(source / root, files / root)
    manifest = {
        str(p.relative_to(files)): digest(p)
        for p in sorted(files.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    (run / "source_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run / "commands.json").write_text(json.dumps(plan, indent=2) + "\n")
    (run / "provenance.json").write_text(
        json.dumps(
            {
                "source_archive_sha256": SOURCE_SHA,
                "source_origin": (
                    "published research release, not recovered historical CLI"
                ),
                "snapshot_origin": "automatic snapshot BEFORE this new training run",
                "training_environment": (
                    "CaptureTrainingEnvironment (custom research trainer)"
                ),
                "seed": seed,
                "evaluation_trials": trials,
            },
            indent=2,
        )
        + "\n"
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for stage, command in zip(("bc", "ppo", "evaluation", "export"), plan):
        print(f"Running {stage}; log: {run / (stage + '.log')}", flush=True)
        with (run / f"{stage}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=source,
                env=env,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "train"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--run-name", default="capture_seed4110")
    parser.add_argument("--seed", type=int, default=4110)
    parser.add_argument("--evaluation-trials", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.evaluation_trials < 1:
        parser.error("At least one held-out evaluation trial is required")
    workspace = args.workspace.resolve()
    if args.action == "prepare":
        prepare(
            workspace, args.source_archive.resolve() if args.source_archive else None
        )
        print(f"Prepared {workspace / 'source'}; no training performed")
    else:
        train(workspace, args.run_name, args.seed, args.evaluation_trials, args.dry_run)
