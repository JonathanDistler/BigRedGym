"""Create a portable Molab ZIP with sources, checkpoint copies, and exact hashes."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import zipfile
from experiments.handstand_v2.run import ORIGINAL, OLD


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/handstand_v2_molab_20261111.zip")
    )
    dest = parser.parse_args().output
    if dest.exists():
        raise FileExistsError(dest)
    dependencies = ["numpy", "torch", "mujoco", "matplotlib", "pytest", "pillow"]
    requirements = (
        "\n".join(f"{k}=={importlib.metadata.version(k)}" for k in dependencies) + "\n"
    )
    files = []
    for root in ("experiments", "scripts", "gym", "resources", "learning"):
        files.extend(
            f
            for f in Path(root).rglob("*")
            if f.is_file() and "__pycache__" not in str(f) and f.suffix != ".pyc"
        )
    files += [
        Path("curriculum.py"),
        Path("pyproject.toml"),
        Path("docs/go2_handstand_curriculum_v2.md"),
        ORIGINAL,
    ]
    files += [next(OLD.glob(f"policy_update_{n:04d}_*.pt")) for n in (4, 20, 44)]
    files += list(Path("notebooks").glob("molab_handstand_v2.ipynb"))
    manifest = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
    )
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f)
        z.writestr("handstand_v2_requirements.txt", requirements)
        z.writestr("handstand_v2_source_manifest.json", json.dumps(manifest, indent=2))
    print(dest)


if __name__ == "__main__":
    main()
