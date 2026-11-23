"""Stage Cornell's policy/log submission from the two immutable research releases.

No training, SDK imports or robot connection. The snapshot is reconstructed from
the published release, not falsely described as a standard Gym training run.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2_deploy.custom_implementation.go2handstand import export_runtime_bundle

SOURCE_SHA = "2cf3a3afe6db87827b60b73c741617f64d054eb1c39a46b5858eb6ab678a77a4"
RESULTS_SHA = "69e4d2c391292c9b80a5e2232b71d8a9cffd9f19dcef0bd51f4624bcffff709e"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validated_archive(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Release archive SHA256 mismatch: {path}")
    return tarfile.open(path, "r:gz")


def copy_member(archive, name, destination):
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"Expected regular archive member: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(archive.extractfile(member).read())


def package(source_archive, results_archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a fresh submission directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stage atomically: corrupt/missing assets never leave a usable-looking log.
    with (
        validated_archive(source_archive, SOURCE_SHA) as source,
        validated_archive(results_archive, RESULTS_SHA) as results,
        tempfile.TemporaryDirectory(dir=destination.parent) as temporary,
    ):
        staged = Path(temporary) / "run"
        checkpoint = Path(temporary) / "latest.pt"
        reference = Path(temporary) / "trajectory.npz"
        copy_member(results, "output/capture_ppo_conservative/latest.pt", checkpoint)
        copy_member(source, "artifacts/handstand_reference/trajectory.npz", reference)
        export_runtime_bundle(checkpoint, reference, staged)
        # The actual research sources/configuration, retaining their real paths.
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isfile() and path.parts[0] in {
                "experiments",
                "gym",
                "scripts",
                "resources",
                "configs",
            }:
                copy_member(source, member.name, staged / "files" / member.name)
        for name in ("train.json", "best.pt"):
            copy_member(
                results,
                f"output/capture_ppo_conservative/{name}",
                staged / "training" / name,
            )
        for name in ("bc.json", "bc.pt"):
            copy_member(
                results,
                f"output/capture_bc_seed4110/{name}",
                staged / "training" / "bc" / name,
            )
        log = json.loads((staged / "training/train.json").read_text())
        recorded_sources = {}
        for original, expected in log["source_sha256"].items():
            relative = (
                original
                if not original.startswith("/")
                else "scripts/handstand_takeoff.py"
            )
            actual = digest(staged / "files" / relative)
            recorded_sources[relative] = {
                "recorded_training_sha256": expected,
                "released_source_sha256": actual,
                "matches_training_record": expected == actual,
            }
        provenance = {
            "task_name": "go2handstand",
            "task_name_origin": (
                "deployment identifier; research trainer was not a registered Gym task"
            ),
            "training_environment": (
                "experiments.handstand_robustness.capture.CaptureTrainingEnvironment"
            ),
            "checkpoint_kind": "capture_residual_v1",
            "training_configuration": log["configuration"],
            "source_release_sha256": SOURCE_SHA,
            "results_release_sha256": RESULTS_SHA,
            "recorded_sources": recorded_sources,
            "snapshot_origin": (
                "reconstructed from published release; "
                "NOT an automatic training-time snapshot"
            ),
            "standard_gym_snapshot": False,
            "snapshot_limitation": (
                "Original custom trainer did not save "
                "go2handstand.py/go2handstand_config.py. Released handstand_takeoff.py "
                "differs from its recorded training hash; exact historical CLI "
                "source is unavailable. Core and takeoff environment "
                "source hashes match."
            ),
            "loading": (
                "Use policy_bundle.load_bundle on this directory; standard "
                "setup_actor/model_*.pt loading does not implement "
                "this hybrid controller."
            ),
        }
        (staged / "training_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        manifest = {
            str(p.relative_to(staged)): digest(p)
            for p in sorted(staged.rglob("*"))
            if p.is_file()
        }
        (staged / "submission_sha256.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        staged.rename(destination)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--results-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = package(args.source_archive, args.results_archive, args.output)
    print(
        f"Prepared {args.output}: full policy.pt, original checkpoint/logs, "
        "source and checksums"
    )
    print(provenance["snapshot_limitation"])
