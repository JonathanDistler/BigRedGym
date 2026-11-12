"""Hard-gated BC, matched two-update smoke and six-update CPU PPO pilot.

The learning path is intentionally unreachable when physical references fail.
"""

import argparse
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from gym.envs.go2.go2kickup_pendulum import (
    Go2KickupPendulum,
    Go2KickupPendulumConfig,
    register_cpu_task,
)
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.pendulum_models import enrich
from experiments.morphology.task import write_json
from scripts.run_rotation_learning import RotationRun
from scripts.run_takeoff_ppo import REWARD_TERMS
from scripts.rotation_learning_tools import action
from scripts.morphology_feasibility import save_csv
from scripts.run_pendulum_reference import compatible_actor, new_env


def validated_gate(root):
    gate = json.loads((root / "trajectory_gate.json").read_text())
    if not gate["passed"]:
        return False
    cases = pd.read_csv(root / "robustness_cases.csv")
    for seed in (42, 43):
        selected = cases[(cases.training_seed == seed) & (cases["rank"] == 0)]
        if (
            len(selected) != 10
            or selected.reset.nunique() != 10
            or selected.success.mean() < 0.8
        ):
            raise ValueError("Gate disagrees with actual robustness cases")
    for name, digest in gate["demonstration_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Demonstration changed: " + name)
    return True


@dataclass
class ReferenceSchedule:
    method: str
    protocol: dict
    stage: int = 0
    stage_entry_step: int = 0
    outcomes: list = field(default_factory=list)
    passing_windows: int = 0
    events: list = field(default_factory=list)
    windows: list = field(default_factory=list)

    @property
    def scale(self):
        return 1.0

    def completed(self, episode, actual_steps):
        if self.stage == 3 or episode.get("reference_stage") != self.stage:
            return
        self.outcomes.append(bool(episode.get("reference_success", False)))
        if len(self.outcomes) < 32:
            return
        fraction = float(np.mean(self.outcomes))
        self.outcomes = []
        self.passing_windows = self.passing_windows + 1 if fraction >= 0.8 else 0
        self.windows.append(
            dict(
                actual_steps=actual_steps,
                stage=self.stage,
                controlled_fraction=fraction,
            )
        )
        if self.passing_windows >= 2 and actual_steps - self.stage_entry_step >= 1024:
            previous = self.stage
            self.stage += 1
            self.stage_entry_step = actual_steps
            self.passing_windows = 0
            self.events.append(
                dict(
                    actual_steps=actual_steps, previous_stage=previous, stage=self.stage
                )
            )


class ReferenceRun(RotationRun):
    config_class = Go2KickupPendulumConfig
    environment_class = Go2KickupPendulum
    reward_terms = REWARD_TERMS + (
        "reference_tracking",
        "reference_hold",
        "reference_slip",
        "reference_contact",
        "reference_failure",
    )

    def restore(self, path):
        super().restore(path)
        if self.p.get("name") == "go2kickup_pendulum_matched_v1":
            self.schedule = ReferenceSchedule(**asdict(self.schedule))
            self.cfg.reference_stage = self.schedule.stage

    def collect_and_update(self):
        self.cfg.reference_stage = self.schedule.stage
        super().collect_and_update()


def evaluate(run, seeds, label):
    cfg = copy.deepcopy(run.cfg)
    if label == "B":
        cfg.reference_stage = 3
    env = run.environment_class(run.envs[0].assets, cfg)
    rows = []
    for stochastic in (False, True):
        for seed in seeds:
            obs, _ = env.reset(1.0, seed)
            noise = torch.Generator().manual_seed(seed + 1000000)
            while True:
                with torch.no_grad():
                    d = run.net.distribution(torch.tensor(obs))
                    raw = d.loc
                    if stochastic:
                        raw = raw + d.scale * torch.randn(12, generator=noise)
                (obs, _), _, done, info = env.step(raw.tanh().numpy())
                if done:
                    break
            rows.append(
                dict(
                    **info,
                    method=label,
                    training_seed=run.seed,
                    stochastic=stochastic,
                    evaluation_seed=seed,
                    iteration=run.iteration,
                    actual_steps=run.actual_steps,
                )
            )
    return rows


def prepare_reference(root, output, seed, assets):
    net, p, _ = compatible_actor(root, seed, assets)
    cases = pd.read_csv(root / "robustness_cases.csv")
    successful = cases[
        (cases.training_seed == seed) & (cases["rank"] == 0) & cases.success
    ]
    picked = int(successful.reset.iloc[0])
    raw = pd.read_csv(root / f"seed{seed}_candidate0_reset{picked}_trajectory.csv")
    ref = enrich(raw, float(assets.models[1.0].body_mass.sum()))
    path = output / f"reference_seed{seed}.csv"
    ref.to_csv(path, index=False)
    stages = {
        "0": [
            str(root / f"seed{seed}_reset{r}_start.pt") for r in (65000, 65001, 65002)
        ],
        "1": [
            str(root / f"seed{seed}_reset{int(r)}_start.pt") for r in successful.reset
        ],
        "2": [],
    }
    for reset in (65003, 65004, 65005):
        env = new_env(assets, p)
        obs, _ = env.reset(1.0, reset)
        while env.row["pitch"] < np.deg2rad(20):
            (obs, _), _, done, _ = env.step(action(net, obs))
            if done:
                raise ValueError("No physically valid longer-approach state")
        state_path = output / f"approach_seed{seed}_reset{reset}.pt"
        torch.save(env.state(), state_path)
        stages["2"].append(str(state_path))
    stages_path = output / f"warm_stages_seed{seed}.json"
    write_json(stages_path, stages)
    frames = []
    for reset in successful.reset:
        data = np.load(root / f"seed{seed}_candidate0_reset{int(reset)}_states.npz")
        frames.append((data["observations"], data["actions"]))
    return (
        path,
        stages_path,
        np.concatenate([f[0] for f in frames]),
        np.concatenate([f[1] for f in frames]),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--offline-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    if not validated_gate(a.offline_run):
        write_json(
            a.output / "completion.json",
            dict(
                skipped=True,
                reason="reference_gate_failed",
                bc_updates=0,
                ppo_interactions=0,
                task_registered=False,
            ),
        )
        print("NO-GO: reference gate failed; BC, task activation, and PPO skipped.")
        return
    torch.set_num_threads(1)
    assets = Assets(a.output / "assets")
    rows = []
    outcomes = []
    assert "go2kickup_pendulum" in register_cpu_task()
    for seed in (42, 43):
        ref_path, warm_path, obs_demo, actions_demo = prepare_reference(
            a.offline_run, a.output, seed, assets
        )
        pair = []
        for label, cls in [("A", RotationRun), ("B", ReferenceRun)]:
            cp = (
                a.offline_run
                / "inputs/output/rotation_pilot_20261111_v1/global"
                / f"A_seed{seed}"
                / "checkpoint_000036.pt"
            )
            state = torch.load(cp, weights_only=False, map_location="cpu")
            p = copy.deepcopy(state["curriculum"]["protocol"])
            bc = a.offline_run / "inputs/output/multimorph_bc_20261111_v2/bc_actor.pt"
            run = cls(a.output / f"{label}_seed{seed}", "A", seed, p, assets, bc)
            run.restore(cp)
            assert all(
                torch.equal(v, run.net.state_dict()[k])
                for k, v in state["network"].items()
            )
            assert (
                run.optimizer.state_dict()["param_groups"]
                == state["optimizer"]["param_groups"]
            )
            p = copy.deepcopy(p)
            p["scales"] = [1.0]
            p["name"] = "go2kickup_pendulum_matched_v1"
            p["evaluation_seeds"] = list(range(65200, 65208))
            if label == "B":
                p["environment"].update(
                    rotation_enabled=False,
                    reference_file=str(ref_path),
                    warm_states=str(warm_path),
                    reference_stage=0,
                )
                run.cfg = Go2KickupPendulumConfig(**p["environment"])
                run.schedule = ReferenceSchedule(
                    "B", p, stage_entry_step=run.actual_steps
                )
                run.method = "B"
            else:
                run.schedule.protocol = p
            run.p = p
            for env in [*run.envs, run.eval_env]:
                env.cfg = run.cfg
            # Reference warm resets start at valid saved states, never injected
            # poses/velocities. No actor or critic random reinitialization.
            if label == "B":
                states = [
                    env.reset(1.0, run.episode_seed(i, run.episode_counts[i]))
                    for i, env in enumerate(run.envs)
                ]
                run.actors = np.stack([s[0] for s in states])
                run.critics = np.stack([s[1] for s in states])
                demo_x = torch.tensor(obs_demo, dtype=torch.float32)
                demo_y = torch.tensor(actions_demo, dtype=torch.float32)
                optimizer = torch.optim.Adam(run.net.actor.parameters(), lr=1e-5)
                bc_rng = torch.Generator().manual_seed(seed + 73000)
                for _ in range(120):
                    idx = torch.randint(len(demo_x), (40,), generator=bc_rng)
                    old = torch.randint(
                        len(run.bc_demonstrations[0]), (88,), generator=bc_rng
                    )
                    x = torch.cat([demo_x[idx], run.bc_demonstrations[0][old]])
                    y = torch.cat([demo_y[idx], run.bc_demonstrations[1][old]])
                    loss = (
                        (
                            (run.net.actor(x) - torch.atanh(y.clamp(-0.999, 0.999)))
                            / 0.05
                        )
                        .square()
                        .mean()
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(run.net.actor.parameters(), 1.0)
                    optimizer.step()
            # Expose all initialization/budget differences rather than calling
            # offline imitation a matched extra simulator budget.
            cfg = json.loads((run.directory / "configuration.json").read_text())
            cfg.update(
                protocol=p,
                config=asdict(run.cfg),
                method=label,
                source_checkpoint=str(cp),
                bc_updates=120 if label == "B" else 0,
                task="go2kickup_pendulum" if label == "B" else "original_fixed_global",
            )
            write_json(run.directory / "configuration.json", cfg)
            pair.append((label, run))
        for offset in range(9):
            if offset:
                for _, run in pair:
                    run.collect_and_update()
            if offset % 2 == 0:
                regression = False
                for label, run in pair:
                    current = evaluate(run, p["evaluation_seeds"], label)
                    rows.extend(current)
                    regression |= (
                        min(
                            np.mean(
                                [
                                    r["takeoff"]
                                    for r in current
                                    if r["stochastic"] == mode
                                ]
                            )
                            for mode in (False, True)
                        )
                        < 0.75
                    )
                    run.checkpoint_and_evaluate()
                save_csv(a.output / "probes.csv", rows)
                if regression or (
                    offset
                    and any(
                        run.training[-1]["post_update_kl"] > 0.03
                        or run.training[-1]["clip_fraction"] > 0.3
                        for _, run in pair
                    )
                ):
                    write_json(
                        a.output / "completion.json",
                        dict(
                            stopped=True,
                            reason="retention_or_divergence",
                            ppo_interactions=sum(
                                (r.actual_steps - 36864) for _, r in pair
                            ),
                            smoke_updates=2,
                            pilot_updates=6,
                        ),
                    )
                    return
        outcomes.extend(
            dict(
                method=label,
                seed=seed,
                updates=run.iteration - 36,
                ppo_interactions=run.actual_steps - 36864,
                stage=run.schedule.stage,
            )
            for label, run in pair
        )
    write_json(
        a.output / "completion.json",
        dict(
            outcomes=outcomes,
            ppo_interactions=sum(r["ppo_interactions"] for r in outcomes),
            smoke_updates=2,
            pilot_updates=6,
            extended_training=False,
        ),
    )


if __name__ == "__main__":
    main()
