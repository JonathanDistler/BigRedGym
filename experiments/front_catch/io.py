"""Append-only logs, exclusive atomic checkpoints and auditable source contracts."""

import csv
import hashlib
import os
from pathlib import Path
import tempfile
import torch


def append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    if path.exists():
        with path.open() as file:
            fields = next(csv.reader(file))
        if set().union(*(r.keys() for r in rows)) - set(fields):
            raise ValueError("CSV schema changed: " + str(path))
    else:
        fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".checkpoint_", suffix=".tmp", delete=False
    ) as temporary:
        name = temporary.name
    try:
        torch.save(state, name)
        with open(name, "rb") as file:
            os.fsync(file.fileno())
        # Link is exclusive even if another resume process races for this generation.
        os.link(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def source_hashes():
    files = set()
    for root in (
        "experiments/front_catch",
        "experiments/momentum_phase",
        "experiments/com_energy_phase",
    ):
        files.update(Path(root).glob("*.py"))
    files.update(
        Path(p)
        for p in (
            "experiments/morphology/learning/environment.py",
            "experiments/morphology/learning/takeoff.py",
            "experiments/morphology/learning/ppo.py",
            "scripts/run_front_catch.py",
        )
    )
    return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(files)}
