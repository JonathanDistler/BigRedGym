"""Deployment-only helpers; no policy, reward or native dynamics changes."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv311/bin/python"
MANIFEST = ROOT / "deployment/molab/manifest.json"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def stamp():
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as file:
        json.dump(value, file, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
        temp = Path(file.name)
    os.replace(temp, path)


def verify_manifest():
    manifest = load(MANIFEST)
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError("Required packaged file missing: " + name)
        if digest(path) != expected:
            raise ValueError(
                "Packaged file changed: "
                + name
                + ". Use a new project directory for a different ZIP."
            )
    return manifest


def child_env():
    env = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "UV_INDEX",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_DEFAULT_INDEX",
    ):
        env.pop(name, None)
    env.update(
        WANDB_DISABLED="true",
        WANDB_MODE="disabled",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"),
        XDG_CACHE_HOME=str(ROOT / ".cache"),
        UV_CACHE_DIR=str(ROOT / ".cache/uv"),
        UV_PYTHON_INSTALL_DIR=str(ROOT / ".python"),
        UV_NO_CONFIG="1",
        UV_LINK_MODE="copy",
        PYTHONNOUSERSITE="1",
    )
    return env


def run(arguments, **kwargs):
    command = [str(item) for item in arguments]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=child_env(), check=True, **kwargs)
