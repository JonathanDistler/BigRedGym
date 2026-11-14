"""Package source, locked dependencies and hash-checked reference for remote use."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--feedback", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--capture-checkpoint", type=Path)
    parser.add_argument("--takeoff-checkpoint", type=Path)
    parser.add_argument("--capture-validation", type=Path)
    parser.add_argument("--capture-demonstrations", type=Path)
    parser.add_argument("--capture-trained-checkpoint", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    expected = json.loads((root / "configs/handstand_reference.json").read_text())
    if digest(args.reference) != expected["trajectory_sha256"]:
        parser.error("Reference hash mismatch")
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        text=True,
    ).splitlines()
    selected = {
        p: root / p
        for p in files
        if p.startswith(
            (
                "gym/",
                "learning/",
                "resources/",
                "experiments/handstand_robustness/",
                "configs/handstand_",
                "tests/",
            )
        )
        or p.startswith("scripts/")
        and "handstand" in p
        or p
        in (
            "pyproject.toml",
            "uv.lock",
            "docs/handstand_robustness.md",
            "docs/handstand_takeoff_molab.md",
            "docs/handstand_capture.md",
            "notebooks/molab_handstand_takeoff.py",
        )
    }
    selected["artifacts/handstand_reference/trajectory.npz"] = args.reference
    if args.checkpoint:
        selected["artifacts/student.pt"] = args.checkpoint
    if args.teacher:
        if digest(args.teacher) != expected["balance_sha256"]:
            parser.error("Teacher hash mismatch")
        selected["artifacts/balance.pt"] = args.teacher
    if args.feedback:
        selected["artifacts/takeoff_teacher.npz"] = args.feedback
    if args.baseline_checkpoint:
        selected["artifacts/baseline_student.pt"] = args.baseline_checkpoint
    for source, name in [
        (args.capture_checkpoint, "capture_student.pt"),
        (args.capture_trained_checkpoint, "capture_ppo_latest.pt"),
        (args.takeoff_checkpoint, "takeoff_latest.pt"),
        (args.capture_validation, "capture_teacher_validation.json"),
        (args.capture_demonstrations, "capture_demonstrations.npz"),
    ]:
        if source:
            selected["artifacts/" + name] = source
    manifest = dict(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        files={name: digest(path) for name, path in selected.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for name, path in selected.items():
            archive.add(path, arcname=name)
        contents = json.dumps(manifest, indent=2).encode()
        entry = tarfile.TarInfo("bundle_manifest.json")
        entry.size = len(contents)
        archive.addfile(entry, io.BytesIO(contents))
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        digest(args.output) + "  " + args.output.name + "\n"
    )
    print(
        json.dumps(
            dict(
                path=str(args.output),
                sha256=digest(args.output),
                bytes=args.output.stat().st_size,
            )
        )
    )


if __name__ == "__main__":
    main()
