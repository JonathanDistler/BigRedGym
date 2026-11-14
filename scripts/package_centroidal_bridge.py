"""Create a separate self-contained continuation archive; keep prior archives intact."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    additions = [
        Path(x)
        for x in [
            "scripts/search_centroidal_bridge.py",
            "scripts/report_centroidal_bridge.py",
            "scripts/package_centroidal_bridge.py",
            "scripts/validate_centroidal_capture.py",
            "experiments/centroidal_bridge/physics.py",
            "tests/unit_tests/test_centroidal_bridge.py",
            "docs/centroidal_bridge.md",
        ]
    ]
    additions += [
        p for p in Path("output/centroidal_bridge_capture_v2").glob("*") if p.is_file()
    ]
    manifest = {}
    with (
        zipfile.ZipFile("output/dynamic_bridge_go2_molab.zip") as old,
        zipfile.ZipFile(a.output, "x", zipfile.ZIP_DEFLATED) as new,
    ):
        for name in old.namelist():
            if name.endswith("manifest.json") or name in [str(p) for p in additions]:
                continue
            blob = old.read(name)
            new.writestr(name, blob)
            manifest[name] = hashlib.sha256(blob).hexdigest()
        for path in additions:
            new.write(path, str(path))
            manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        new.writestr("centroidal_manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(a.output) as archive:
        assert archive.testzip() is None
        for name, sha in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
    print(a.output, len(manifest), "verified files")


if __name__ == "__main__":
    main()
