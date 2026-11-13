"""Exclusive portable bundle for the bounded parallel front-catching pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--preparation",
        type=Path,
        default=Path("output/front_catch_preparation_20261112_v3"),
    )
    a = p.parse_args()
    manifest = json.loads((a.preparation / "manifest.json").read_text())
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
                f.suffix in (".py", ".md", ".json", ".txt")
                or root.startswith("resources/")
            )
        )
    files.update(f for f in a.preparation.iterdir() if f.is_file())
    files.update(
        [
            Path(manifest["initial"]),
            Path("pyproject.toml"),
            Path("docs/front_catch_parallel.md"),
            Path("docs/front_catch_anticipatory.md"),
            Path("notebooks/molab_front_catch_parallel.ipynb"),
        ]
    )
    for name in [
        "output/recoverability_maps_20261112/recoverable_states.npz",
        "output/recoverability_maps_20261112/validated_region.json",
        "output/multimorph_bc_20261111_v2/bc_dataset.npz",
    ]:
        files.add(Path(name))
    contents = {
        str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f))
        z.writestr("front_catch_bundle_manifest.json", json.dumps(contents, indent=2))
    print(a.output, hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
