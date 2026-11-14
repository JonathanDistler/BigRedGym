"""Freeze a separate read-only, hash-locked reference without editing seed data."""

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.unified_go2 import setup
from experiments.unified_go2.core import Environment, load
import mujoco
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    prep = Path("output/retained_go2_preparation_v3")
    copies = {
        "capture.pt": Path("output/centroidal_capture_83_v3/best.pt"),
        "inputs.pt": Path("output/centroidal_capture_83_v3/inputs.pt"),
        "capture.gif": Path("output/centroidal_capture_83_v3/best.gif"),
        "contact_audit.pt": Path("output/centroidal_capture_best_audit_v2/audit.pt"),
        "continuation.pt": Path(
            "output/centroidal_capture_83_continuation_v1/replay.pt"
        ),
        "config.json": prep / "config.json",
    }
    for name, src in copies.items():
        shutil.copy2(src, a.output / name)
    cfg, data, bank, assets = setup(prep, a.output)
    e = Environment(assets, cfg)
    e.capture_centers = data["centers"]
    saved = load(a.output / "capture.pt")
    e.restore(copy.deepcopy(saved["standing"]))
    for u in saved["standing_actions"]:
        e.step(u)
    error = float(np.max(abs(e.state()["integration"] - saved["final"]["integration"])))
    if error:
        raise RuntimeError(f"Reference diverged: {error}")
    mujoco.mj_saveModel(e.m, str(a.output / "nominal.mjb"))
    np.savez(
        a.output / "actuator_contract.npz",
        effort=e.effort,
        speed=e.speed,
        bounds=e.bounds,
        qids=e.qids,
        vids=e.vids,
        nominal=e.nominal,
        armature=e.m.dof_armature,
        damping=e.m.dof_damping,
        friction=e.m.geom_friction,
    )
    torch.save(
        dict(physics=e.physics_trace, control=e.trace, final=e.state()),
        a.output / "prefix_physics.pt",
    )
    files = [p for p in a.output.glob("*") if p.is_file()]
    # Explicit data required by the legacy preparation loader, kept out of Git.
    files += [p for p in prep.glob("*") if p.is_file()]
    files += [
        Path("output/centroidal_capture_83_v3/best.pt"),
        Path("output/centroidal_capture_83_continuation_v1/replay.pt"),
    ]
    manifest = dict(
        reference=str(a.output),
        source_commit="15fab3f",
        seed=96000,
        control_dt=cfg.dt,
        physics_dt=float(e.m.opt.timestep),
        kp=cfg.kp,
        kd=cfg.kd,
        mujoco=mujoco.__version__,
        replay_max_error=error,
        files={
            str(p): dict(
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                bytes=p.stat().st_size,
            )
            for p in files
        },
    )
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    for p in a.output.glob("*"):
        if p.is_file():
            p.chmod(0o444)
    print(a.manifest, len(files), "hashed data files; reference replay error", error)


if __name__ == "__main__":
    main()
