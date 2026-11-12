"""Export the new source overlay and all new experimental artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


SOURCES = [
    "experiments/morphology/learning/pendulum_reference.py",
    "experiments/morphology/learning/pendulum_models.py",
    "gym/envs/go2/go2kickup_pendulum.py",
    "scripts/run_pendulum_reference.py",
    "scripts/learn_pendulum_reference.py",
    "scripts/diagnose_pendulum_candidates.py",
    "scripts/report_pendulum_reference.py",
    "scripts/package_pendulum_reference.py",
    "notebooks/molab_pendulum_guided_rotation.py",
    "tests/unit_tests/test_pendulum_reference.py",
    "docs/go2_pendulum_guided_rotation.md",
]


def archive(path, files, manifest_name):
    manifest = dict(
        branch="experiment/pendulum_guided_rotation",
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_snapshot="Uncommitted snapshot; cloning HEAD alone is insufficient.",
        required_original_bundle_sha256="1779e4669007a481aeba9973a037c3907739d77fde2c76876997f0f153091b42",
        files={
            str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)
        },
    )
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f))
        z.writestr(manifest_name, json.dumps(manifest, indent=2))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "  " + path.name + "\n")
    print(path, digest)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--output-prefix", type=Path, default=Path("output/pendulum_guided_rotation")
    )
    a = p.parse_args()
    files = {Path(f) for f in SOURCES}
    source = Path(str(a.output_prefix) + "_source.zip")
    source.parent.mkdir(parents=True, exist_ok=True)
    archive(source, files, "pendulum_manifest.json")
    for root in [
        "output/pendulum_guided_rotation_v1",
        "output/pendulum_guided_rotation_v2",
        "output/pendulum_guided_rotation_v3",
        "output/pendulum_guided_learning_gate_v1",
        "output/pendulum_molab_gate_smoke",
    ]:
        files.update(
            f
            for f in Path(root).rglob("*")
            if f.is_file() and f.suffix not in (".zip", ".pyc")
        )
    archive(
        Path(str(a.output_prefix) + "_artifacts.zip"), files, "pendulum_manifest.json"
    )


if __name__ == "__main__":
    main()
