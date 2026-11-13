"""Import one explicitly named combined checkpoint from a trusted Molab backup.

This initializes a NEW experiment; it is never described as exact training resume.
Only the chosen regular archive member is copied; no archive paths are extracted.
"""

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("backup", type=Path)
    p.add_argument("--member", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if zipfile.is_zipfile(a.backup):
        with zipfile.ZipFile(a.backup) as z:
            if a.member not in z.namelist():
                raise ValueError("Member absent")
            data = z.read(a.member)
    else:
        with tarfile.open(a.backup) as t:
            item = t.getmember(a.member)
            if not item.isfile():
                raise ValueError("Regular checkpoint member required")
            data = t.extractfile(item).read()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("xb") as f:
        f.write(data)
    a.output.with_suffix(".import.json").write_text(
        json.dumps(
            dict(
                backup=str(a.backup),
                member=a.member,
                sha256=hashlib.sha256(data).hexdigest(),
                note="Explicit lane selection required at initialization",
            ),
            indent=2,
        )
    )
    print(a.output)


if __name__ == "__main__":
    main()
