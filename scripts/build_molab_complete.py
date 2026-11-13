"""Build one exclusive, audited source/data deployment ZIP, no old run archives."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("output/bigredgym_molab_complete.zip")
    )
    args = parser.parse_args()
    files = set()
    # gym.envs eagerly loads the task registry and shared learning classes.
    # Include this import infrastructure intact, not unrelated model/data archives.
    for folder in [
        "gym",
        "learning",
        "experiments/front_catch",
        "experiments/momentum_phase",
        "experiments/com_energy_phase",
    ]:
        files.update(
            p for p in Path(folder).rglob("*.py") if "__pycache__" not in p.parts
        )
    for name in [
        "experiments/morphology/task.py",
        "experiments/morphology/dynamics_audit.py",
        "experiments/morphology/learning/environment.py",
        "experiments/morphology/learning/takeoff.py",
        "experiments/morphology/learning/ppo.py",
        "scripts/__init__.py",
        "scripts/run_front_catch.py",
        "scripts/run_com_energy_phase.py",
        "scripts/audit_morphology_dynamics.py",
        "scripts/morphology_feasibility.py",
        "scripts/benchmark_front_catch.py",
        "scripts/report_front_catch.py",
        "scripts/report_front_catch_timing.py",
        "scripts/play_front_catch.py",
        "scripts/analyze_front_catch.py",
        "scripts/backup_front_catch.py",
        "MOLAB_START.py",
        "MOLAB_TRAIN.py",
        "MOLAB_BACKUP.py",
        "docs/molab_unified_deployment.md",
        "notebooks/molab_unified_launcher.ipynb",
        "scripts/build_molab_complete.py",
    ]:
        files.add(Path(name))
    for folder in [
        "resources/robots/go2",
        "deployment/molab",
        "output/front_catch_preparation_20261112_v3",
    ]:
        files.update(
            p
            for p in Path(folder).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.name != "manifest.json"
        )
    files.update(Path("gym/envs/go2/references").glob("*.json"))
    prep = Path("output/front_catch_preparation_20261112_v3/manifest.json")
    files.add(prep)
    prepared = json.loads(prep.read_text())
    initial = Path(prepared["initial"])
    files.add(initial)
    # Preserve exact tests but run them outside the unrelated repository conftest.
    # Close static local imports, including functions reached only at model build.
    scanned = set()
    while files - scanned:
        for file in list(files - scanned):
            scanned.add(file)
            if file.suffix != ".py":
                continue
            for node in ast.walk(ast.parse(file.read_bytes())):
                if (
                    not isinstance(node, ast.ImportFrom)
                    or node.level
                    or not node.module
                ):
                    continue
                if node.module.split(".")[0] not in {
                    "experiments",
                    "scripts",
                    "gym",
                    "learning",
                    "deployment",
                }:
                    continue
                path = Path(*node.module.split("."))
                for candidate in [path.with_suffix(".py"), path / "__init__.py"]:
                    if candidate.is_file():
                        files.add(candidate)
    mapped = {str(p): p.read_bytes() for p in sorted(files)}
    for name in ["test_front_catch.py", "test_front_catch_anticipatory.py"]:
        mapped["deployment/molab/tests/" + name] = Path(
            "tests/unit_tests", name
        ).read_bytes()
    mapped["README.md"] = Path("deployment/molab/README.md").read_bytes()
    # Fail the build on a missing static internal dependency, including lazy imports.
    modules = {
        str(p.with_suffix("")).replace("/", ".") for p in files if p.suffix == ".py"
    }
    for name, content in list(mapped.items()):
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(content, filename=name)):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                module = node.module
                if module.split(".")[0] not in {
                    "experiments",
                    "scripts",
                    "gym",
                    "learning",
                    "deployment",
                }:
                    continue
                if module not in modules and not any(
                    m.startswith(module + ".") for m in modules
                ):
                    raise RuntimeError(
                        "Missing internal import " + module + " in " + name
                    )
    hashes = {
        name: hashlib.sha256(content).hexdigest() for name, content in mapped.items()
    }
    actual_initial = hashes[str(initial)]
    if actual_initial != prepared["initial_sha256"]:
        raise ValueError("Preserved checkpoint checksum changed")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = dict(
        format_version=1,
        experiment="anticipatory_v1",
        files=hashes,
        source_revision=revision,
        source_tree=(
            "Workspace files identified by individual SHA256 hashes; "
            "includes untracked experiment source."
        ),
        initial_checkpoint=str(initial),
        initial_sha256=actual_initial,
        python="3.11.14",
        dependency_lock="deployment/molab/requirements.lock",
    )
    mapped["deployment/molab/manifest.json"] = json.dumps(manifest, indent=2).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(mapped.items()):
            archive.writestr(name, content)
    print(
        json.dumps(
            dict(
                path=str(args.output),
                bytes=args.output.stat().st_size,
                sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                files=len(mapped),
                initial_sha256=actual_initial,
            )
        )
    )


if __name__ == "__main__":
    main()
