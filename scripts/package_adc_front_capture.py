"""Portable exclusive ZIP with source/data/model hashes; no external repo paths."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    a = p.parse_args()
    manifest = json.loads((a.preparation / "manifest.json").read_text())
    files = set()
    for artifact_root in (
        Path("output/adc_front_capture_report_v1"),
        Path("output/adc_front_capture_videos_v2"),
        Path("output/adc_front_capture_baseline_video_v2"),
        Path("output/adc_front_capture_update10_video_v1"),
    ):
        if artifact_root.exists():
            files.update(
                f
                for f in artifact_root.rglob("*")
                if f.is_file() and "assets" not in f.parts
            )
    for root in (
        "experiments",
        "gym",
        "learning",
        "scripts",
        "tests/unit_tests",
        "resources/robots/go2",
        "deployment/adc_front_capture",
    ):
        files.update(
            f
            for f in Path(root).rglob("*")
            if f.is_file()
            and "__pycache__" not in f.parts
            and (
                f.suffix in (".py", ".md", ".json", ".txt", ".lock")
                or root.startswith("resources/")
            )
        )
    files.update(f for f in a.preparation.iterdir() if f.is_file())
    # Preserve the original prepared bank used by baseline regression tests.
    files.update(
        f
        for f in Path("output/front_catch_preparation_20261112_v3").iterdir()
        if f.is_file()
    )
    files.update(
        [
            Path(manifest["initial"]),
            a.probe,
            Path("pyproject.toml"),
            Path("docs/adc_front_capture.md"),
            Path("notebooks/molab_adc_front_capture.md"),
            Path("notebooks/molab_adc_front_capture.ipynb"),
            Path("output/multimorph_bc_20261111_v2/bc_actor.pt"),
        ]
    )
    if a.checkpoint:
        files.add(a.checkpoint)
    for path in files:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Portable relative paths required")
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f))
        z.writestr("adc_bundle_manifest.json", json.dumps(hashes, indent=2))
    print(a.output, hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
