"""Create an exclusive, hashed Molab source/data bundle for the bounded diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    files = set()
    for root in (
        "experiments",
        "gym",
        "learning",
        "scripts",
        "tests/unit_tests",
        "resources/robots/go2",
    ):
        files.update(
            f
            for f in Path(root).rglob("*")
            if f.is_file()
            and "__pycache__" not in f.parts
            and (
                f.suffix in (".py", ".json", ".xml", ".stl", ".obj", ".png", ".md")
                or root.startswith("resources/")
            )
        )
    for name in (
        "pyproject.toml",
        "notebooks/molab_com_energy_phase.ipynb",
        "docs/com_energy_phase.md",
        "output/multimorph_bc_20261111_v2/bc_actor.pt",
        "output/multimorph_bc_20261111_v2/bc_dataset.npz",
        "output/recoverability_maps_20261112/recoverable_states.npz",
        "output/recoverability_maps_20261112/validated_region.json",
    ):
        files.add(Path(name))
    manifest = dict(
        branch="experiment/com_energy_phase_ppo",
        parent=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        files={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)
        },
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f))
        z.writestr("com_energy_bundle_manifest.json", json.dumps(manifest, indent=2))
    print(a.output, hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
