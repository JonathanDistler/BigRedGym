def _bigredgym_setup():
    # Paste this entire cell into Molab. Its notebook Python may be 3.13.
    from pathlib import Path
    import hashlib
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import zipfile

    PROJECT = Path(
        os.environ.get("BIGREDGYM_PROJECT", "/marimo/bigredgym_molab")
    ).resolve()
    ZIP_NAME = "bigredgym_molab_complete.zip"
    explicit_zip = os.environ.get("BIGREDGYM_BUNDLE")
    candidates = [Path(explicit_zip)] if explicit_zip else []
    if not candidates:
        for folder in dict.fromkeys([Path.cwd(), Path("/marimo"), Path("/tmp")]):
            if not folder.exists():
                continue
            for parent, dirs, files in os.walk(folder):
                dirs[:] = [
                    d
                    for d in dirs
                    if d
                    not in {
                        ".venv",
                        ".venv311",
                        ".python",
                        ".cache",
                        ".git",
                        "node_modules",
                    }
                ]
                if ZIP_NAME in files:
                    candidates.append(Path(parent) / ZIP_NAME)
                if len(Path(parent).relative_to(folder).parts) >= 4:
                    dirs[:] = []
    candidates = sorted({p.resolve() for p in candidates if p.is_file()})
    if not candidates:
        raise FileNotFoundError(
            "Upload "
            + ZIP_NAME
            + " to Molab, or set BIGREDGYM_BUNDLE to its full path."
        )
    if len({hashlib.sha256(p.read_bytes()).hexdigest() for p in candidates}) > 1:
        raise RuntimeError(
            "Multiple different bundles found. "
            "Set BIGREDGYM_BUNDLE to the intended uploaded ZIP."
        )
    BUNDLE = candidates[0]
    PROJECT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE) as archive:
        manifest_name = "deployment/molab/manifest.json"
        manifest = json.loads(archive.read(manifest_name))
        for name, expected in dict(
            manifest["files"],
            **{manifest_name: hashlib.sha256(archive.read(manifest_name)).hexdigest()},
        ).items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive path: " + name)
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("ZIP checksum mismatch: " + name)
            destination = PROJECT / relative
            if not destination.resolve().is_relative_to(PROJECT):
                raise ValueError("Archive destination escapes project: " + name)
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                    raise RuntimeError(
                        "Existing source differs: "
                        + str(destination)
                        + ". Select a new BIGREDGYM_PROJECT; data was preserved."
                    )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, delete=False
            ) as stream:
                stream.write(content)
                temporary = Path(stream.name)
            try:
                os.link(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        environment.pop(key, None)
    subprocess.run(
        [sys.executable, str(PROJECT / "deployment/molab/bootstrap.py")],
        cwd=PROJECT,
        env=environment,
        check=True,
    )
    print("READY:", PROJECT, "| Training interpreter:", PROJECT / ".venv311/bin/python")


_bigredgym_setup()
