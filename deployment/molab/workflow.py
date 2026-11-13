"""Resumable process supervisor, measured reports and atomic downloadable ZIPs."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deployment.molab.common import (
    ROOT,
    PYTHON,
    MANIFEST,
    atomic_json,
    child_env,
    digest,
    load,
    run,
    stamp,
    verify_manifest,
)


def inside_root(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Run directories must be inside the extracted project")
    return path


def effective_config(config):
    cfg = dict(config)
    n = cfg["num_envs"]
    if n not in (8, 16, 32, 64):
        raise ValueError("num_envs must be 8, 16, 32 or 64")
    if cfg.get("match_environment_budgets", True):
        budget = cfg.get("reference_num_envs", 16) * cfg.get("reference_horizon", 32)
        if budget % n:
            raise ValueError("Matched transition budget must be divisible by num_envs")
        cfg["horizon"] = budget // n
    if cfg["horizon"] not in (4, 8, 16, 32, 64, 128):
        raise ValueError("Choose a matched budget giving horizon 4/8/16/32/64/128")
    if not 1 <= cfg["updates"] <= 20:
        raise ValueError("The unchanged bounded trainer accepts 1--20 updates")
    if (
        cfg["num_envs"] * cfg["horizon"] * cfg["updates"] * 3 * len(cfg["seeds"])
        > 200000
    ):
        raise ValueError("Budget exceeds the unchanged trainer 200k pilot limit")
    if cfg["eval_every"] < 1 or cfg["checkpoint_every"] < 1:
        raise ValueError("Evaluation/checkpoint periods must be positive")
    if cfg["workers"] < 0:
        raise ValueError("workers must be nonnegative")
    if cfg["suite"] != "anticipatory_v1":
        raise ValueError("This package is configured for anticipatory_v1")
    return cfg


def valid_checkpoints(directory):
    """Ignore .tmp; preserve a corrupt committed file under quarantine, never delete."""
    import torch

    valid = []
    for path in sorted(directory.glob("checkpoint_*.pt")):
        if not re.fullmatch(r"checkpoint_\d{6}\.pt", path.name):
            continue
        try:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            assert saved["iteration"] == int(path.stem.rsplit("_", 1)[1])
            assert {
                "runs",
                "protocol",
                "history",
                "torch_rng",
                "numpy_rng",
            } <= saved.keys()
            for state in saved["runs"].values():
                assert {
                    "network",
                    "optimizer",
                    "torch_rng",
                    "worlds",
                    "actors",
                    "critics",
                } <= state.keys()
            valid.append((path, saved["iteration"], saved["protocol"]))
            del saved
        except Exception as error:
            quarantine = directory / "quarantine"
            quarantine.mkdir(exist_ok=True)
            destination = quarantine / (path.name + "." + stamp() + ".bad")
            path.rename(destination)
            atomic_json(
                destination.with_suffix(".json"),
                dict(original=str(path), error=str(error)),
            )
            print("Preserved invalid checkpoint under", destination, flush=True)
    return valid


def active_directory(explicit=None):
    if explicit:
        return inside_root(explicit)
    pointer = ROOT / ".molab/active_run.json"
    if not pointer.exists():
        raise RuntimeError(
            "No active run. Run MOLAB_TRAIN.py first, or specify --run-dir."
        )
    return inside_root(load(pointer)["directory"])


def export(directory, export_root, latest_only=False):
    """Snapshot sources, all full checkpoints and new outputs into a download ZIP."""
    export_root = Path(export_root).resolve()
    export_root.mkdir(parents=True, exist_ok=True)
    destination = export_root / ("bigredgym_backup_" + stamp() + ".zip")
    manifest = verify_manifest()
    files = {ROOT / name: "project/" + name for name in manifest["files"]}
    files[MANIFEST] = "project/" + str(MANIFEST.relative_to(ROOT))
    checkpoint_files = sorted(directory.glob("checkpoint_*.pt"))
    latest = checkpoint_files[-1] if checkpoint_files else None
    folders = [directory, ROOT / ".molab"]
    folders.extend((ROOT / "output").glob("molab_benchmark_*"))
    for folder in folders:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or path.name.endswith((".tmp", ".lock")):
                continue
            if any(
                part in ("exports", "__pycache__")
                for part in path.relative_to(folder).parts
            ):
                continue
            if (
                latest_only
                and path.name.startswith("checkpoint_")
                and path.suffix == ".pt"
                and path != latest
            ):
                continue
            files[path] = "project/" + str(path.relative_to(ROOT))
    with tempfile.NamedTemporaryFile(
        dir=export_root, suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        checksums = {}
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as archive:
            for path, name in sorted(files.items(), key=lambda x: x[1]):
                # Read a single snapshot of growing metrics; .pt files are immutable.
                content = path.read_bytes()
                import hashlib

                checksums[name] = hashlib.sha256(content).hexdigest()
                archive.writestr(name, content)
            archive.writestr(
                "EXPORT_MANIFEST.json",
                json.dumps(
                    dict(
                        files=checksums,
                        timestamp=stamp(),
                        run=str(directory.relative_to(ROOT)),
                        latest_only=latest_only,
                        checkpoint_names=[p.name for p in checkpoint_files],
                        source_revision=manifest["source_revision"],
                        restore=(
                            "Extract project/; run deployment/molab/bootstrap.py; "
                            "Resume the active run; keep relative output paths. "
                            "restore-paths can rebuild the active pointer."
                        ),
                    ),
                    indent=2,
                ),
            )
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print("BACKUP READY:", destination, "| SHA256", digest(destination), flush=True)
    return destination


def reports(directory):
    if not list(directory.glob("checkpoint_*.pt")):
        return None
    report = directory / "reports" / stamp()
    run([PYTHON, ROOT / "scripts/report_front_catch.py", directory, "--output", report])
    run(
        [
            PYTHON,
            ROOT / "scripts/report_front_catch_timing.py",
            directory,
            "--output",
            report / "anticipatory_timing",
        ]
    )
    return report


def train(args):
    verify_manifest()
    if not (ROOT / ".molab/setup_success.json").is_file():
        raise RuntimeError("Run MOLAB_START.py/bootstrap.py before training")
    cfg = effective_config(
        load(args.config or ROOT / "deployment/molab/experiment.json")
    )
    if args.resume_only:
        directory = active_directory(args.run_dir)
        cfg = load(directory / "deployment_run.json")["config"]
    else:
        directory = inside_root(args.run_dir or cfg["output"])
        pointer = ROOT / ".molab/active_run.json"
        if not args.run_dir and not args.new_run and pointer.exists():
            directory = active_directory()
        if args.new_run:
            directory = directory.with_name(directory.name + "_" + stamp())
    state = ROOT / ".molab"
    with (state / "training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another training supervisor is running; "
                "no duplicate PPO process started"
            ) from error
        checkpoints = valid_checkpoints(directory) if directory.exists() else []
        if directory.exists() and not checkpoints and any(directory.iterdir()):
            if args.resume_only:
                raise RuntimeError(
                    "No complete checkpoint to resume. Existing attempt preserved."
                )
            previous = directory
            directory = directory.with_name(directory.name + "_restart_" + stamp())
            print(
                "Preserving uncheckpointed attempt",
                previous,
                "; new attempt:",
                directory,
            )
        actual_steps = cfg["num_envs"] * cfg["horizon"] * cfg["updates"]
        print(
            "Matched training budget:",
            actual_steps,
            "per condition/seed;",
            actual_steps * 3 * len(cfg["seeds"]),
            "total;",
            cfg["horizon"],
            "steps/world/update",
        )
        # Native 16/32 runs invoke the original script; others use the CLI adapter.
        entry = (
            "scripts/run_front_catch.py"
            if cfg["num_envs"] in (16, 32)
            else "deployment/molab/entrypoint.py"
        )
        command = [
            str(PYTHON),
            str(ROOT / entry),
            "--output",
            str(directory.relative_to(ROOT)),
            "--workers",
            str(cfg["workers"]),
            "--max-seconds",
            str(cfg["max_seconds"]),
        ]
        if checkpoints:
            last, iteration, protocol = checkpoints[-1]
            for key in ("num_envs", "updates", "horizon", "seeds"):
                if cfg[key] != protocol[key]:
                    raise ValueError(
                        f"Resume {key} differs. Keep the budget or choose NEW_RUN=True."
                    )
            if protocol.get("comparison_suite") != "anticipatory_v1":
                raise ValueError("Existing run uses another objective suite")
            if iteration >= protocol["updates"]:
                print("RUN ALREADY COMPLETE:", directory, "; no training restarted.")
                return directory
            command += ["--resume", str(last.relative_to(ROOT))]
        else:
            command += [
                "--suite",
                cfg["suite"],
                "--preparation",
                cfg["preparation"],
                "--num-envs",
                str(cfg["num_envs"]),
                "--updates",
                str(cfg["updates"]),
                "--horizon",
                str(cfg["horizon"]),
                "--eval-every",
                str(cfg["eval_every"]),
                "--checkpoint-every",
                str(cfg["checkpoint_every"]),
                "--seeds",
            ]
            command += [str(seed) for seed in cfg["seeds"]]
        atomic_json(
            state / "active_run.json", dict(directory=str(directory.relative_to(ROOT)))
        )
        # The original trainer creates the run directory exclusively. Metadata lives
        # outside it until the subprocess has created it.
        metadata = dict(
            config=cfg,
            command=command,
            timestamp=stamp(),
            source_manifest_sha256=digest(MANIFEST),
            environment=load(state / "environment.json"),
        )
        logdir = state / "launches"
        logdir.mkdir(exist_ok=True)
        launch = logdir / (stamp() + ".json")
        atomic_json(launch, metadata)
        print("+", " ".join(command), flush=True)
        process = subprocess.Popen(command, cwd=ROOT, env=child_env())
        backed = {p[0] for p in checkpoints}
        exit_code = None
        try:
            while process.poll() is None:
                if (
                    directory.exists()
                    and not (directory / "deployment_run.json").exists()
                ):
                    atomic_json(directory / "deployment_run.json", metadata)
                if args.periodic_backups:
                    for checkpoint in sorted(directory.glob("checkpoint_*.pt")):
                        if checkpoint not in backed:
                            export(directory, args.export_directory, latest_only=True)
                            backed.add(checkpoint)
                time.sleep(2)
            exit_code = process.returncode
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if directory.exists():
                if not (directory / "deployment_run.json").exists():
                    atomic_json(directory / "deployment_run.json", metadata)
                try:
                    reports(directory)
                finally:
                    export(directory, args.export_directory)
        if exit_code:
            raise RuntimeError(
                f"Training exited with code {exit_code}; "
                "prior checkpoints and backup preserved"
            )
        return directory


def restore_paths(directory):
    """Original trainer records relative paths. Keep checkpoint bytes immutable."""
    meta = directory / "deployment_run.json"
    if not meta.exists():
        raise RuntimeError(
            "Missing deployment_run.json; cannot infer restored run identity"
        )
    value = load(meta)
    # A restored backup should retain project/output/<original run directory>.
    expected = Path(value["command"][value["command"].index("--output") + 1])
    if (ROOT / expected).resolve() != directory.resolve():
        raise ValueError(
            "Restore run to "
            + str(ROOT / expected)
            + "; checkpoint paths stay immutable"
        )
    atomic_json(ROOT / ".molab/active_run.json", dict(directory=str(expected)))
    print("Restored active pointer; relative checkpoint paths remain valid:", expected)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "action", choices=("train", "backup", "report", "restore-paths")
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-dir")
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--resume-only", action="store_true")
    parser.add_argument("--periodic-backups", action="store_true")
    parser.add_argument("--export-directory", type=Path, default=Path("/marimo"))
    args = parser.parse_args()
    if Path(sys.executable).absolute() != PYTHON.absolute():
        raise RuntimeError("Use the isolated interpreter: " + str(PYTHON))
    if args.action == "train":
        train(args)
    elif args.action == "backup":
        export(active_directory(args.run_dir), args.export_directory)
    elif args.action == "report":
        reports(active_directory(args.run_dir))
    else:
        restore_paths(active_directory(args.run_dir))


if __name__ == "__main__":
    main()
