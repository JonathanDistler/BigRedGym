"""Atomic, exclusive recovery/full archives for limited-duration notebook sessions."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--all", action="store_true")
    p.add_argument("--extra", type=Path, action="append", default=[])
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    checkpoint = a.checkpoint or sorted(a.directory.glob("checkpoint_*.pt"))[-1]
    entries = [a.bundle]
    if a.all:
        entries.append(a.directory)
    else:
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        entries.extend(
            [checkpoint, a.directory / "protocol.json", a.directory / "source"]
        )
        for name in saved["history"]["trace_files"]:
            path = Path(name)
            if not path.exists():
                path = a.directory / Path(*path.parts[path.parts.index("sessions") :])
            entries.append(path)
    entries.extend(path for path in a.extra if path.exists())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=a.output.parent, prefix=".backup_", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for entry in entries:
                arcname = entry.name if entry == a.bundle else str(entry)
                archive.add(
                    entry,
                    arcname=arcname,
                    filter=lambda info: None if info.name.endswith(".tmp") else info,
                )
            metadata = dict(
                checkpoint=str(checkpoint),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_bundle=a.bundle.name,
                source_bundle_sha256=hashlib.sha256(a.bundle.read_bytes()).hexdigest(),
                full_backup=a.all,
                scope="Committed checkpoint; older files remain untouched",
            )
            content = json.dumps(metadata, indent=2).encode()
            info = tarfile.TarInfo("front_catch_recovery_manifest.json")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.link(temporary, a.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            dict(
                path=str(a.output),
                bytes=a.output.stat().st_size,
                sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main()
