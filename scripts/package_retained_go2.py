"""Exclusive self-contained retained-skill Go2 source/model/data bundle."""

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
        "resources/robots/go2",
        "deployment/retained_go2",
    ]:
        files.update(
            f
            for f in Path(root).rglob("*")
            if f.is_file()
            and "__pycache__" not in f.parts
            and (
                f.suffix in [".py", ".json", ".txt", ".md"]
                or root.startswith("resources")
            )
        )
    for root in [
        "output/unified_go2_preparation_v1",
        "output/retained_go2_preparation_v1",
        "output/retained_go2_preparation_v2",
        "output/retained_go2_preparation_v3",
        "output/retained_go2_distillation_v3",
        "output/retained_go2_validation_v3",
        "output/retained_go2_validation_v4",
        "output/retained_go2_post_smoke_v1",
        "output/retained_go2_smoke_v4",
        "output/retained_go2_analysis_v1",
        "output/retained_go2_comparison_v1",
        "output/retained_go2_bridge_baseline_v1",
    ]:
        files.update(
            f
            for f in Path(root).rglob("*")
            if f.is_file()
            and not any(x in f.parts for x in ["assets", "source", "__pycache__"])
        )
    files.add(Path("output/retained_go2_validation_v1/evaluation.json"))
    files.update(
        Path("output/retained_go2_validation_v1/videos").glob("adc_standing_0*")
    )
    files.add(Path("output/unified_go2_initial_v1/student.pt"))
    files.update(
        Path("output/unified_go2_evaluation_initial_v2/videos").glob(
            "student_standing_0*"
        )
    )
    files.update(
        [
            Path("docs/retained_go2.md"),
            Path("notebooks/molab_retained_go2.py"),
            Path("pyproject.toml"),
        ]
    )
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
    with zipfile.ZipFile(a.output, "x", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(files):
            z.write(f, str(f))
        z.writestr("retained_manifest.json", json.dumps(hashes, indent=2))
    h = hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_suffix(".zip.sha256").write_text(h + "  " + a.output.name + "\n")
    print(
        a.output, len(files), "files", a.output.stat().st_size, "bytes", h, flush=True
    )


if __name__ == "__main__":
    main()
