"""Nominal repository actuator inventory, not hardware system identification."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.front_catch.data import network, config
from experiments.adc_front_capture.core import CaptureEnvironment
from experiments.adc_front_capture.curriculum import effective_inertia, damping
from experiments.morphology.learning.environment import Assets
from scripts.run_com_energy_phase import dump


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    manifest = json.loads((a.preparation / "manifest.json").read_text())
    net, parent = network(manifest["initial"])
    cfg = config(parent, "B")
    cfg.full_trace = True
    cfg.adc_enabled = True
    cfg.comparison_suite = "anticipatory_v1"
    env = CaptureEnvironment(Assets(a.output / "assets"), cfg)
    inertia = effective_inertia(env)
    rows = []
    for i, name in enumerate(env.names):
        rows.append(
            dict(
                joint=name,
                nominal_k=cfg.kp,
                nominal_b=cfg.kd,
                effort_limit=float(env.effort[i]),
                speed_limit=float(env.speed[i]),
                position_bounds=env.bounds[i].tolist(),
                armature=float(env.m.dof_armature[env.vids[i]]),
                passive_damping=float(env.m.dof_damping[env.vids[i]]),
                local_effective_inertia=float(inertia[i]),
                critical_active_damping=float(damping(env, cfg.kp)[i]),
                nominal_local_damping_ratio=float(
                    (cfg.kd + env.m.dof_damping[env.vids[i]])
                    / (2 * np.sqrt(cfg.kp * inertia[i]))
                ),
            )
        )
    obs, _ = env.reset(1.0, 96000)
    while True:
        with torch.no_grad():
            action = net.actor(torch.tensor(obs)).tanh().numpy()
        (obs, _), _, done, info = env.step(action)
        if done:
            break
    trace = env.physics_trace
    time = np.array([r["time"] for r in trace])
    power = np.array([r["actuator_power"] for r in trace])
    command_trace = env.trace
    tracking = []
    for i, name in enumerate(env.names):
        target = np.array([r[f"target_{i}"] for r in command_trace])
        q = np.array([r[f"joint_{i}"] for r in command_trace])
        candidates = []
        for lag in range(0, min(round(0.1 / cfg.dt) + 1, len(q) - 2)):
            x = target[:-lag] if lag else target
            y = q[lag:] if lag else q
            corr = (
                float(np.corrcoef(x, y)[0, 1])
                if x.std() > 1e-8 and y.std() > 1e-8
                else 0.0
            )
            candidates.append(corr)
        tracking.append(
            dict(
                joint=name,
                tracking_rmse=float(np.sqrt(np.mean((target - q) ** 2))),
                apparent_target_to_position_lag_s=float(np.argmax(candidates) * cfg.dt),
                lag_correlation=max(candidates),
                note=(
                    "Closed-loop contact-coupled cross-correlation; "
                    "not identified motor delay"
                ),
            )
        )
    dump(
        a.output / "actuator_audit.json",
        dict(
            joints=rows,
            tracking=tracking,
            actuator_work=float(np.trapezoid(power, time)),
            positive_work=float(np.trapezoid(np.maximum(power, 0), time)),
            peak_power=float(np.max(abs(power))),
            policy_dt=cfg.dt,
            physics_dt=float(env.m.opt.timestep),
            target_rate_fraction=0.7,
            target_filter=1.0,
            action_scale=cfg.action_scale,
            measured_motor_delay=None,
            hardware_identification=None,
            nominal_source="go2_config and preserved checkpoint",
            model_mass=env.mass,
            limits_unchanged=True,
        ),
    )


if __name__ == "__main__":
    main()
