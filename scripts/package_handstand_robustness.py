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
        or p in ("pyproject.toml", "uv.lock", "docs/handstand_robustness.md")
    }
    selected["artifacts/handstand_reference/trajectory.npz"] = args.reference
    if args.checkpoint:
        selected["artifacts/student.pt"] = args.checkpoint
    if args.teacher:
        if digest(args.teacher) != expected["balance_sha256"]:
            parser.error("Teacher hash mismatch")
        selected["artifacts/balance.pt"] = args.teacher
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
