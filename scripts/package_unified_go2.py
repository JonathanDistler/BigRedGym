"""Portable unified Go2 ZIP with explicit data roots and complete hash manifest."""

import argparse, hashlib, json, zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    files = set()
    for root in [
        "experiments",
        "gym",
        "learning",
        "scripts",
        "tests/unit_tests",
        "deployment/unified_go2",
        "resources/robots/go2",
    ]:
        files.update(
            f
            for f in Path(root).rglob("*")
            if f.is_file()
            and "__pycache__" not in f.parts
            and (
                f.suffix in [".py", ".md", ".txt", ".json"]
                or root.startswith("resources")
            )
        )
    for root in [
        "output/unified_go2_preparation_v1",
        "output/unified_go2_initial_v1",
        "output/unified_go2_distillation_v3",
    ]:
        files.update(f for f in Path(root).iterdir() if f.is_file())
    for root in [
        "output/unified_go2_evaluation_initial_v1",
        "output/unified_go2_evaluation_initial_v2",
        "output/unified_go2_evaluation_smoke_v2",
    ]:
        if Path(root).exists():
            files.update(
                f
                for f in Path(root).rglob("*")
                if f.is_file() and "assets" not in f.parts
            )
    files.update(
        [
            Path("output/unified_go2_smoke_v2/checkpoint_000002.pt"),
            Path("docs/unified_go2.md"),
            Path("notebooks/molab_unified_go2.py"),
            Path("pyproject.toml"),
        ]
    )
    values = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
    with zipfile.ZipFile(a.output, "x", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f))
        z.writestr("unified_manifest.json", json.dumps(values, indent=2))
    digest = hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_suffix(".zip.sha256").write_text(digest + "  " + a.output.name + "\n")
    print(a.output, "files", len(files), "SHA256", digest)


if __name__ == "__main__":
    main()
