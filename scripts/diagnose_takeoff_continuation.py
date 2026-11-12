"""Independent full-horizon checkpoint diagnostics; no training changes."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.ppo import ActorCritic
from experiments.morphology.learning.takeoff import TakeoffConfig, TakeoffEnvironment
from scripts.morphology_feasibility import save_csv


class DiagnosticEnvironment(TakeoffEnvironment):
    def measure(self):
        row = super().measure()
        if hasattr(self, "diagnostics") and (
            not self.diagnostics or self.d.time > self.diagnostics[-1]["time"] + 1e-10
        ):
            mujoco.mj_subtreeVel(self.m, self.d)
            support = row["feet"][:2, :2].mean(0)
            self.diagnostics.append(
                dict(
                    time=float(self.d.time),
                    pitch=row["pitch"],
                    omega=row["pitch_velocity"],
                    angular_momentum_y=float(self.d.subtree_angmom[self.base, 1]),
                    com_support_x=float(row["com"][0] - support[0]),
                    com_support_y=float(row["com"][1] - support[1]),
                    front_support=bool(row["front_support"]),
                    collision=bool(row["illegal_contacts"]),
                    tracking_mse=float(
                        np.mean((self.target - self.d.qpos[self.qids]) ** 2)
                    ),
                )
            )
        return row


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads((a.run / "protocol.json").read_text())
    torch.set_num_threads(1)
    env = DiagnosticEnvironment(
        Assets(a.run / "diagnostic_assets"),
        TakeoffConfig(**dict(cfg["environment"], terminate_on_goal=False)),
    )
    destination = a.run / "rotation_diagnostics.csv"
    records = []
    if destination.exists():
        with destination.open() as stream:
            records = list(csv.DictReader(stream))
    for path in sorted(a.run.glob("*_seed*/checkpoint_*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=False)
        prior = [
            r
            for r in records
            if r["method"] == state["method"]
            and int(r["training_seed"]) == state["training_seed"]
            and int(r["iteration"]) == state["iteration"]
        ]
        if len(prior) == 2 * len(cfg["evaluation_seeds"]):
            continue
        net = ActorCritic(
            44, 74, cfg["ppo"]["hidden_dims"], log_std_min=cfg["ppo"]["log_std_min"]
        )
        net.load_state_dict(state["network"])
        for stochastic in (False, True):
            for seed in cfg["evaluation_seeds"]:
                env.diagnostics = []
                actor, _ = env.reset(1.0, seed)
                env.diagnostics = []
                rng = torch.Generator().manual_seed(seed + 1000000)
                while True:
                    with torch.no_grad():
                        latent = net.actor(torch.tensor(actor))
                        if stochastic:
                            latent += net.distribution(
                                torch.tensor(actor)
                            ).scale * torch.randn(12, generator=rng)
                    (actor, _), _, done, info = env.step(latent.tanh().numpy())
                    if done:
                        break
                trace = env.diagnostics
                peak = max(trace, key=lambda r: r["pitch"])
                lift = min(trace, key=lambda r: abs(r["time"] - info["takeoff_time"]))
                collisions = [r["time"] for r in trace if r["collision"]]
                records.append(
                    dict(
                        **info,
                        method=state["method"],
                        training_seed=state["training_seed"],
                        iteration=state["iteration"],
                        actual_steps=state["actual_steps"],
                        stochastic=stochastic,
                        evaluation_seed=seed,
                        pitch_velocity_liftoff=lift["omega"]
                        if info["takeoff"]
                        else float("nan"),
                        pitch_velocity_peak=peak["omega"],
                        angular_momentum_liftoff=lift["angular_momentum_y"]
                        if info["takeoff"]
                        else float("nan"),
                        angular_momentum_peak=peak["angular_momentum_y"],
                        com_support_x_liftoff=lift["com_support_x"]
                        if info["takeoff"]
                        else float("nan"),
                        com_support_x_peak=peak["com_support_x"],
                        joint_tracking_rms=float(
                            np.sqrt(np.mean([r["tracking_mse"] for r in trace]))
                        ),
                        first_nonfoot_collision_time=min(collisions)
                        if collisions
                        else -1.0,
                        front_support_at_peak=peak["front_support"],
                    )
                )
        save_csv(destination, records)
        print(path, flush=True)


if __name__ == "__main__":
    main()
