"""Exclusive Molab bundle with exact initialization and validated offline data."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--validation",
        type=Path,
        default=Path("output/momentum_phase_validation_20261112_v2"),
    )
    a = p.parse_args()
    gate = json.loads((a.validation / "validation.json").read_text())
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
            and (f.suffix in (".py", ".json", ".md") or root.startswith("resources/"))
        )
    files.update(
        [
            Path("pyproject.toml"),
            Path("docs/momentum_phase.md"),
            Path("notebooks/molab_momentum_phase.ipynb"),
            Path(gate["initial_checkpoint"]),
        ]
    )
    files.update(
        f
        for f in a.validation.iterdir()
        if f.is_file()
        and (
            f.suffix == ".json"
            or f.name.endswith("demonstrations.npz")
            or f.name == "tests.txt"
        )
    )
    # Include the original balance bank so the archived target can be remeasured.
    for name in ("recoverable_states.npz", "validated_region.json"):
        files.add(Path("output/recoverability_maps_20261112") / name)
    manifest = {
        str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)
    }
    for file, digest in gate["source_sha256"].items():
        if manifest[file] != digest:
            raise ValueError("Gate no longer matches source: " + file)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f))
        z.writestr("momentum_bundle_manifest.json", json.dumps(manifest, indent=2))
    print(a.output, hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
