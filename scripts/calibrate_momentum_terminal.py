"""Measure archived successful balance states in the current full-size Go2 model."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from experiments.com_energy_phase.core import (
    EnergyConfig,
    EnergyEnvironment,
    orientation_error,
)
from experiments.com_energy_phase.data import reference
from experiments.morphology.learning.environment import Assets
from experiments.momentum_phase.physics import sample
from scripts.morphology_feasibility import save_csv
from scripts.run_com_energy_phase import dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--reference", type=Path, default=Path("output/recoverability_maps_20261112")
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    assets = Assets(a.output / "assets")
    e = EnergyEnvironment(assets, EnergyConfig())
    ref = reference(e, a.reference)
    rows = []
    bank = np.load(a.reference / "recoverable_states.npz")
    for i in ref["checked_indices"]:
        mujoco.mj_setState(
            e.m, e.d, bank[f"{i}_integration"], mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(e.m, e.d)
        e.row = e.measure()
        e.pitch_axis = np.array([0.0, 1.0, 0.0])
        e.fixed_support = np.r_[e.row["feet"][:2, :2].mean(0), 0.0]
        s = sample(e, e.row)
        s["index"] = i
        s["orientation_error"] = orientation_error(
            e.d.xmat[e.base].reshape(3, 3), np.array(ref["rotation"]).reshape(3, 3)
        )
        rows.append(s)
    save_csv(a.output / "states.csv", rows)
    summary = {
        k: dict(
            min=float(min(r[k] for r in rows)),
            median=float(np.median([r[k] for r in rows])),
            max=float(max(r[k] for r in rows)),
        )
        for k in (
            "potential",
            "com_height",
            "body_angular_speed",
            "pitch_velocity",
            "com_speed",
            "support_region_distance",
            "calf_clearance",
            "L_com",
            "L_support",
            "orientation_error",
        )
    }
    dump(a.output / "summary.json", summary)
    dump(a.output / "reference.json", ref)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
