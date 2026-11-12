"""Self-contained Molab CLI entry point. Importing this file launches nothing.

Upload the verified original bundle and pendulum_guided_rotation_source.zip.
--setup clones/pins a fresh repo, verifies both overlays and installs CPU deps.
--mode all runs tests, offline search, diagnostics/report, then the hard gate.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ORIGINAL_SHA = "1779e4669007a481aeba9973a037c3907739d77fde2c76876997f0f153091b42"


def overlay(archive, root, original=False):
    if original and hashlib.sha256(archive.read_bytes()).hexdigest() != ORIGINAL_SHA:
        raise ValueError("Original verified bundle hash mismatch")
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(
            z.read("molab_manifest.json" if original else "pendulum_manifest.json")
        )
        for name, digest in manifest["files"].items():
            target = (root / name).resolve()
            if not target.is_relative_to(root.resolve()) or ".git" in Path(name).parts:
                raise ValueError(name)
            data = z.read(name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return manifest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--setup", action="store_true")
    p.add_argument("--repo", type=Path, default=Path("BigRedGym_pendulum"))
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument(
        "--mode", choices=["all", "offline", "learning", "report"], default="all"
    )
    p.add_argument("--offline-run", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use Python 3.11 for verified dependency parity")
    bundle = a.bundle.resolve()
    repo = a.repo.resolve()
    env = dict(
        os.environ,
        WANDB_MODE="disabled",
        MPLCONFIGDIR="/tmp/pendulum-mpl",
        XDG_CACHE_HOME="/tmp/pendulum-cache",
    )
    env.pop("PYTHONPATH", None)

    def command(args):
        subprocess.run(list(map(str, args)), cwd=repo, env=env, check=True)

    if a.setup:
        if repo.exists():
            raise FileExistsError("Setup requires a fresh checkout directory")
        if a.source is None:
            raise ValueError("--source is required with --setup")
        subprocess.run(
            [
                "git",
                "clone",
                "--branch",
                "ppo_kickup_e2e",
                "--single-branch",
                "https://github.com/JonathanDistler/BigRedGym.git",
                str(repo),
            ],
            env=env,
            check=True,
        )
        with zipfile.ZipFile(bundle) as z:
            base = json.loads(z.read("molab_manifest.json"))["base_commit"]
        command(["git", "checkout", "--detach", base])
        command(["git", "switch", "-c", "experiment/pendulum_guided_rotation"])
        overlay(bundle, repo, True)
        overlay(a.source.resolve(), repo)
        command([sys.executable, "-m", "venv", repo / ".venv"])
        command(
            [
                repo / ".venv/bin/python",
                "-m",
                "pip",
                "install",
                "torch==2.13.0",
                "tensordict==0.13.0",
                "mujoco==3.11.0",
                "numpy==2.4.6",
                "pandas==3.0.5",
                "matplotlib==3.11.1",
                "pytest",
                "ruff",
            ]
        )
    python = repo / ".venv/bin/python"
    output = a.output or Path("output") / (
        "pendulum_molab_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    )
    out = output if output.is_absolute() else repo / output
    out.mkdir(parents=True, exist_ok=False)
    version_code = (
        "import sys,json,torch,mujoco,numpy,pandas,matplotlib; "
        "assert sys.version_info[:2]==(3,11); "
        'assert torch.__version__.split("+")[0]=="2.13.0"; '
        'assert mujoco.__version__=="3.11.0"; '
        'print(json.dumps(dict(python=sys.version,torch=torch.__version__,mujoco=mujoco.__version__,numpy=numpy.__version__,pandas=pandas.__version__,matplotlib=matplotlib.__version__,backend="cpu_mujoco",wandb="disabled"),indent=2))'
    )
    versions = subprocess.check_output(
        [str(python), "-c", version_code], cwd=repo, env=env, text=True
    )
    (out / "environment.json").write_text(versions)
    offline = (
        (a.offline_run if a.offline_run.is_absolute() else repo / a.offline_run)
        if a.offline_run
        else out / "offline"
    )
    if a.mode in ("all", "offline"):
        command(
            [
                python,
                "-m",
                "pytest",
                "tests/unit_tests/test_pendulum_reference.py",
                "tests/unit_tests/test_rotation_learning.py",
                "tests/unit_tests/test_takeoff_learning.py",
                "tests/unit_tests/test_takeoff_bc_regularization.py",
                "tests/unit_tests/test_physical_morphology_ppo.py",
                "-q",
            ]
        )
        if a.offline_run is None:
            command(
                [
                    python,
                    "scripts/run_pendulum_reference.py",
                    "--bundle",
                    bundle,
                    "--output",
                    offline,
                ]
            )
        command([python, "scripts/diagnose_pendulum_candidates.py", "--run", offline])
    if a.mode in ("all", "offline", "report"):
        command([python, "scripts/report_pendulum_reference.py", "--run", offline])
    if a.mode in ("all", "learning"):
        command(
            [
                python,
                "scripts/learn_pendulum_reference.py",
                "--offline-run",
                offline,
                "--output",
                out / "learning",
            ]
        )
    # Export complete experiment files (including checkpoints if learning ran).
    backup = out / "complete_artifacts.zip"
    with zipfile.ZipFile(backup, "x", compression=zipfile.ZIP_DEFLATED) as z:
        roots = [out] if offline.is_relative_to(out) else [offline, out]
        for root in roots:
            for f in root.rglob("*"):
                if f.is_file() and f != backup and "__pycache__" not in f.parts:
                    z.write(
                        f,
                        str(f.relative_to(repo))
                        if f.is_relative_to(repo)
                        else str(Path("external") / f.relative_to(root)),
                    )
        for folder in (
            "experiments/morphology/learning",
            "scripts",
            "gym/envs/go2",
            "docs",
            "notebooks",
        ):
            for f in (repo / folder).glob("*"):
                if f.is_file() and f.suffix in (".py", ".json", ".md"):
                    z.write(f, str(f.relative_to(repo)))
    print("Complete backup:", backup)


if __name__ == "__main__":
    main()
