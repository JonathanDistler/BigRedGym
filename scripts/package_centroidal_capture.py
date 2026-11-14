"""New portable capture archive, with result/state/source hashes and CRC checks."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    additions = list(Path("experiments/centroidal_capture").glob("*.py"))
    additions += list(Path("scripts").glob("*centroidal_capture.py"))
    additions += [
        Path("scripts/test_centroidal_handoff.py"),
        Path("tests/unit_tests/test_centroidal_capture.py"),
        Path("docs/centroidal_capture.md"),
    ]
    for root in [
        "output/centroidal_capture_diagnosis_v1",
        "output/centroidal_capture_61_v1",
        "output/centroidal_capture_61_v2",
        "output/centroidal_capture_83_v2",
        "output/centroidal_capture_83_v3",
        "output/centroidal_capture_83_v4",
        "output/centroidal_capture_83_continuation_v1",
        "output/centroidal_capture_best_audit_v1",
        "output/centroidal_capture_best_audit_v2",
        "output/centroidal_capture_original_audit_v1",
        "output/centroidal_capture_handoff61_v1",
        "output/centroidal_capture_handoff83_v1",
    ]:
        additions.extend(
            p
            for p in Path(root).rglob("*")
            if p.is_file()
            and not any(
                x in p.parts
                for x in [
                    "assets",
                    "replay_assets_best",
                    "replay_assets_baseline",
                    "__pycache__",
                ]
            )
        )
    additions.extend(
        p for p in Path("output/retained_go2_preparation_v3").glob("*") if p.is_file()
    )
    names = {str(p) for p in additions}
    manifest = {}
    with (
        zipfile.ZipFile("output/centroidal_bridge_molab_v1.zip") as old,
        zipfile.ZipFile(a.output, "x", zipfile.ZIP_DEFLATED) as new,
    ):
        for name in old.namelist():
            if name in names or name in {
                "centroidal_manifest.json",
                "centroidal_capture_manifest.json",
                "retained_manifest.json",
            }:
                continue
            blob = old.read(name)
            new.writestr(name, blob)
            manifest[name] = hashlib.sha256(blob).hexdigest()
        for path in additions:
            blob = path.read_bytes()
            new.writestr(str(path), blob)
            manifest[str(path)] = hashlib.sha256(blob).hexdigest()
        new.writestr("centroidal_capture_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(a.output) as z:
        assert z.testzip() is None
        for name, sha in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == sha
    a.output.with_suffix(".zip.sha256").write_text(
        hashlib.sha256(a.output.read_bytes()).hexdigest() + "  " + a.output.name + "\n"
    )
    print(a.output, len(manifest), "members verified")


if __name__ == "__main__":
    main()
