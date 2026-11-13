"""Extend the verified retained ZIP without changing it or storing binaries in Git."""

import argparse, hashlib, json, zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--base", type=Path, default=Path("output/retained_go2_molab.zip"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    additions = {
        p
        for root in [
            "experiments/dynamic_bridge",
            "output/dynamic_bridge_pipeline_smoke_v2",
            "output/dynamic_bridge_diagnosis_v2",
            "output/dynamic_bridge_eval_smoke_v1",
        ]
        for p in Path(root).rglob("*")
        if p.is_file()
        and not any(x in p.parts for x in ["assets", "source", "__pycache__"])
    }
    additions.update(Path("scripts").glob("*dynamic_bridge*.py"))
    additions.update(
        [
            Path("notebooks/molab_dynamic_bridge_go2.py"),
            Path("docs/dynamic_bridge_go2.md"),
            Path("tests/unit_tests/test_dynamic_bridge.py"),
        ]
    )
    manifest = {}
    with (
        zipfile.ZipFile(a.base) as old,
        zipfile.ZipFile(a.output, "x", zipfile.ZIP_DEFLATED) as new,
    ):
        for name in old.namelist():
            if name == "retained_manifest.json" or Path(name) in additions:
                continue
            blob = old.read(name)
            new.writestr(name, blob)
            manifest[name] = hashlib.sha256(blob).hexdigest()
        for f in sorted(additions):
            new.write(f, str(f))
            manifest[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
        new.writestr("retained_manifest.json", json.dumps(manifest, indent=2))
    sha = hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_suffix(".zip.sha256").write_text(sha + "  " + a.output.name + "\n")
    print(a.output, len(manifest), "files", a.output.stat().st_size, "bytes", sha)


if __name__ == "__main__":
    main()
