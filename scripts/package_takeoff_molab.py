"""Package current working source and exact BC initialization."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--extra-root", type=Path, action="append", default=[])
    a = p.parse_args()
    files = set()
    for root in [
        "experiments",
        "gym",
        "learning",
        "scripts",
        "notebooks",
        "tests/unit_tests",
    ]:
        for f in Path(root).rglob("*"):
            if (
                f.is_file()
                and f.suffix in (".py", ".json", ".md")
                and "__pycache__" not in f.parts
            ):
                files.add(f)
    for root in ["resources/robots/go2"]:
        files.update(f for f in Path(root).rglob("*") if f.is_file())
    for name in [
        "output/takeoff_bc_investigation_v1/bc_actor.pt",
        "output/takeoff_learning_certified/validation.json",
        "output/takeoff_learning_certified/configuration.json",
        "pyproject.toml",
    ]:
        files.add(Path(name))
    files.update(Path("docs").glob("*takeoff*.md"))
    files.update(Path("docs").glob("*rotation*.md"))
    files.update(Path("docs").glob("*morphology*.md"))
    files.add(Path("output/handoff_reference_search/candidate_reference.json"))
    for root in a.extra_root:
        files.update(
            f
            for f in root.rglob("*")
            if f.is_file()
            and f.suffix
            in (".json", ".npz", ".pt", ".csv", ".png", ".pdf", ".md", ".log", ".py")
        )
    # Retain teacher provenance and the exact offline data when present.
    files.update(
        f
        for f in Path("output/takeoff_bc_investigation_v1").rglob("*")
        if f.is_file() and f.suffix in (".json", ".npz", ".csv")
    )
    manifest = dict(
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        branch="ppo_kickup_e2e",
        files={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)
        },
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f))
        z.writestr("molab_manifest.json", json.dumps(manifest, indent=2))
    print(a.output, hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
