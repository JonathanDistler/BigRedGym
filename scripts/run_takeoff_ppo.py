"""Prepared matched takeoff PPO, with explicit execution and validation gates."""

import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.morphology.learning.environment import Assets
from experiments.morphology.learning.takeoff import (
    TakeoffConfig,
    TakeoffEnvironment,
    TakeoffCurriculum,
)
from experiments.morphology.learning.ppo import advantages, update
from experiments.morphology.task import write_json
from scripts.run_physical_morphology_ppo import Run, protocol, evaluate
from scripts.morphology_feasibility import save_csv

REWARD_TERMS = (
    "push_preparation",
    "simultaneous_liftoff",
    "airborne_rotation",
    "supported_flight",
    "torque",
    "command_change",
    "failure",
)


def semantic_hash(path):
    return hashlib.sha256(
        ast.dump(ast.parse(Path(path).read_text()), include_attributes=False).encode()
    ).hexdigest()


def check_validation(path):
    gate = json.loads(Path(path).read_text())
    if (
        not gate["reward_negative_controls_passed"]
        or not gate["action_authority"]["passed"]
    ):
        raise ValueError("Reward/action-authority validation did not pass")
    if gate["reward_semantic_sha256"] != semantic_hash(
        "experiments/morphology/learning/takeoff.py"
    ):
        raise ValueError("Takeoff code differs from validated code; rerun validation")
    return gate


class TakeoffRun(Run):
    config_class = TakeoffConfig
    environment_class = TakeoffEnvironment
    reward_terms = REWARD_TERMS

    def __init__(self, directory, method, seed, p, assets, bc_actor=None):
        super().__init__(directory, method, seed, p, assets)
        self.cfg = self.config_class(**p["environment"])
        self.envs = [
            self.environment_class(assets, self.cfg) for _ in range(p["num_envs"])
        ]
        self.eval_env = self.environment_class(assets, self.cfg)
        self.schedule = TakeoffCurriculum(method, p)
        observations = [
            e.reset(self.schedule.scale, self.episode_seed(i, 0))
            for i, e in enumerate(self.envs)
        ]
        self.actors = np.stack([r[0] for r in observations])
        self.critics = np.stack([r[1] for r in observations])
        bc_meta = None
        if bc_actor:
            saved = torch.load(bc_actor, weights_only=False, map_location="cpu")
            expected = saved["action_configuration"]
            if (
                expected["action_scale"] != self.cfg.action_scale
                or expected["target_filter"] != self.cfg.target_filter
                or saved["actor_dim"] != 44
            ):
                raise ValueError("BC actor observation/action contract mismatch")
            self.net.actor.load_state_dict(saved["actor"])
            bc_meta = dict(
                path=str(bc_actor),
                sha256=hashlib.sha256(Path(bc_actor).read_bytes()).hexdigest(),
                dataset_sha256=saved["dataset_sha256"],
            )
        self.initial_parameters = self.net.actor[0].weight.detach().clone()
        self.bc_demonstrations = None
        regularizer = p.get("bc_regularization")
        if regularizer and regularizer.get("initial_weight", 0) > 0:
            dataset = Path(regularizer["dataset"])
            digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
            if digest != regularizer["dataset_sha256"]:
                raise ValueError("BC regularizer dataset hash mismatch")
            if bc_meta is None or digest != bc_meta["dataset_sha256"]:
                raise ValueError("BC regularizer must use the initialization dataset")
            demo = np.load(dataset)
            # Episode-held-out data remain independent of PPO's BC loss.
            mask = demo["seed"] < regularizer["validation_seed_start"]
            self.bc_demonstrations = (
                torch.tensor(demo["observations"][mask], dtype=torch.float32),
                torch.tensor(demo["actions"][mask], dtype=torch.float32),
            )
            if not len(self.bc_demonstrations[0]):
                raise ValueError("Empty BC regularizer training split")
            with torch.no_grad():
                self.initial_demo_mean = self.net.actor(
                    self.bc_demonstrations[0]
                ).clone()
                self.initial_demo_std = self.net.distribution(
                    self.bc_demonstrations[0]
                ).scale.clone()
            config_path = self.directory / "bc_regularizer.json"
            write_json(
                config_path, dict(**regularizer, training_frames=int(mask.sum()))
            )
        self.torch_state = torch.get_rng_state()
        for file in (
            Path(__file__),
            Path("experiments/morphology/learning/takeoff.py"),
        ):
            shutil.copyfile(file, self.source / file.name)
        config = json.loads((self.directory / "configuration.json").read_text())
        config.update(
            config=asdict(self.cfg),
            reward_version="takeoff_v2",
            bc_initialization=bc_meta,
            reward_semantic_sha256=semantic_hash(
                "experiments/morphology/learning/takeoff.py"
            ),
        )
        write_json(self.directory / "configuration.json", config)

    def restore(self, path):
        super().restore(path)
        self.schedule = TakeoffCurriculum(**asdict(self.schedule))

    def collect_and_update(self):
        torch.set_rng_state(self.torch_state)
        p = self.p
        hp = dict(p["ppo"])
        regularizer = p.get("bc_regularization")
        if regularizer:
            fraction = min(1.0, self.iteration / max(1, regularizer["decay_updates"]))
            hp.update(
                bc_weight=regularizer["initial_weight"] * (1 - fraction)
                + regularizer["final_weight"] * fraction,
                bc_latent_scale=regularizer.get("latent_scale", 0.05),
                bc_batch_size=regularizer.get("batch_size", 128),
            )
            if "phase_decay_start_update" in regularizer:
                hp["bc_phase_fraction"] = min(
                    1.0,
                    max(
                        0.0,
                        (self.iteration - regularizer["phase_decay_start_update"])
                        / regularizer["phase_decay_updates"],
                    ),
                )
        n = len(self.envs)
        t = p["steps_per_iteration"] // n
        obs = []
        priv = []
        raws = []
        logs = []
        values = []
        rewards = []
        dones = []
        nextvals = []
        scale_steps = {s: 0 for s in p["scales"]}
        for _ in range(t):
            actor = torch.tensor(self.actors)
            critic = torch.tensor(self.critics)
            with torch.no_grad():
                raw, log, value = self.net.sample(actor, critic)
            obs.append(self.actors.copy())
            priv.append(self.critics.copy())
            raws.append(raw.numpy())
            logs.append(log.numpy())
            values.append(value.numpy())
            reward_step = []
            done_step = []
            next_actors = []
            next_critics = []
            terminal_critics = []
            for i, e in enumerate(self.envs):
                scale = e.scale
                (a, c), reward, done, info = e.step(raw[i].tanh().numpy())
                self.actual_steps += 1
                scale_steps[scale] = scale_steps.get(scale, 0) + 1
                self.full_scale_steps += int(scale == 1.0)
                reward_step.append(reward)
                done_step.append(done)
                terminal_critics.append(c)
                if done:
                    record = dict(
                        **info,
                        actual_steps=self.actual_steps,
                        iteration=self.iteration + 1,
                        method=self.method,
                        training_seed=self.seed,
                        world=i,
                        episode_seed=e.seed,
                        episode_index=self.episode_counts[i],
                    )
                    record.update(
                        {f"episode_reward_{k}": v for k, v in e.terms_sum.items()}
                    )
                    self.episodes.append(record)
                    self.schedule.completed(record, self.actual_steps)
                    self.episode_counts[i] += 1
                    a, c = e.reset(
                        self.schedule.scale,
                        self.episode_seed(i, self.episode_counts[i]),
                    )
                next_actors.append(a)
                next_critics.append(c)
            with torch.no_grad():
                nv = (
                    self.net.critic(torch.tensor(np.stack(terminal_critics)))
                    .squeeze(-1)
                    .numpy()
                )
            nextvals.append(nv)
            rewards.append(reward_step)
            dones.append(done_step)
            self.actors = np.stack(next_actors)
            self.critics = np.stack(next_critics)
        a, returns = advantages(
            np.array(rewards),
            np.array(values),
            np.array(nextvals),
            np.array(dones),
            hp["gamma"],
            hp["gae_lambda"],
        )

        def tensor(x):
            x = np.asarray(x, dtype=np.float32)
            return torch.tensor(x.reshape((-1,) + x.shape[2:]))

        batch = (
            tensor(obs),
            tensor(priv),
            tensor(raws),
            tensor(logs),
            tensor(returns),
            tensor(a),
        )
        stats = update(self.net, self.optimizer, batch, hp, self.bc_demonstrations)
        if self.bc_demonstrations is not None:
            with torch.no_grad():
                current = self.net.distribution(self.bc_demonstrations[0])
                delta = current.loc - self.initial_demo_mean
                cumulative = (
                    torch.log(current.scale / self.initial_demo_std)
                    + (self.initial_demo_std.square() + delta.square())
                    / (2 * current.scale.square())
                    - 0.5
                )
                stats["demo_cumulative_gaussian_kl"] = float(cumulative.sum(-1).mean())
                stats["demo_action_shift_rms"] = float(
                    (current.loc.tanh() - self.initial_demo_mean.tanh())
                    .square()
                    .mean()
                    .sqrt()
                )
                for joint in range(12):
                    stats[f"demo_joint_{joint}_shift_rms"] = float(
                        delta[:, joint].square().mean().sqrt()
                    )
        self.torch_state = torch.get_rng_state()
        self.iteration += 1
        recent = [e for e in self.episodes if e["iteration"] == self.iteration]
        row = dict(
            iteration=self.iteration,
            actual_steps=self.actual_steps,
            physics_steps=self.actual_steps * 10,
            method=self.method,
            training_seed=self.seed,
            scale=self.schedule.scale,
            curriculum_stage=self.schedule.stage,
            full_scale_steps=self.full_scale_steps,
            completed_episodes=len(recent),
            wall_seconds=time.monotonic() - self.start,
            **stats,
        )
        for key in (
            "episode_reward",
            "episode_seconds",
            "max_supported_pitch",
            "max_pitch",
            "max_pitch_velocity",
            "takeoff",
            "rear_clearance",
            "front_support_fraction",
            "nonfoot_collision",
            "success",
            "recovery_seconds",
            "torque_saturation",
            "genuine_flight_pitch",
            "genuine_airborne_seconds",
        ):
            row[key] = float(np.mean([e[key] for e in recent])) if recent else ""
        for name in self.reward_terms:
            key = f"episode_reward_{name}"
            row[key] = float(np.mean([e[key] for e in recent])) if recent else ""
        row["actor_parameter_change"] = float(
            torch.linalg.vector_norm(
                self.net.actor[0].weight.detach() - self.initial_parameters
            )
        )
        for scale in p["scales"]:
            row[f"steps_scale_{scale:.2f}"] = scale_steps.get(scale, 0)
        self.training.append(row)
        save_csv(self.directory / "training.csv", self.training)
        if self.episodes:
            save_csv(self.directory / "episodes.csv", self.episodes)
        write_json(self.directory / "curriculum_events.json", self.schedule.events)
        write_json(self.directory / "curriculum_windows.json", self.schedule.windows)
        print(
            f"{self.method} seed={self.seed} iter={self.iteration} "
            f"steps={self.actual_steps} "
            f"scale={self.schedule.scale:.2f} reward={row['episode_reward']} "
            f"pitch={row['max_supported_pitch']} KL={stats['post_update_kl']:.4f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("experiments/morphology/learning/protocol_takeoff.json"),
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--bc-actor", type=Path)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--smoke", action="store_true")
    choice.add_argument("--execute-extended", action="store_true")
    args = parser.parse_args()
    gate = check_validation(args.validation)
    p = protocol(args.protocol)
    validated = json.loads((args.validation.parent / "configuration.json").read_text())[
        "authority_configurations"
    ]["direct_targets"]
    requested = TakeoffConfig(**p["environment"])
    for key in ("dt", "kp", "kd", "action_scale", "target_filter", "seconds"):
        if getattr(requested, key) != validated[key]:
            parser.error(f"Unvalidated action/observation contract: {key}")
    if args.smoke:
        p.update(
            training_seeds=[42],
            evaluation_seeds=[1000],
            num_envs=2,
            steps_per_iteration=128,
            base_training_steps=128,
            evaluation_interval=1,
        )
        p["ppo"].update(epochs=2, batch_size=64)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", p)
    write_json(args.output / "validation_gate.json", gate)
    assets = Assets(args.output / "assets")
    assets.directory = args.output / "assets"
    torch.set_num_threads(1)
    for seed in p["training_seeds"]:
        runs = [
            TakeoffRun(
                args.output / f"{m}_seed{seed}", m, seed, p, assets, args.bc_actor
            )
            for m in ("A", "B")
        ]
        for run in runs:
            run.checkpoint_and_evaluate()
        target = p["base_training_steps"]
        while runs[1].actual_steps < target:
            for run in runs:
                run.collect_and_update()
                if run.iteration % p["evaluation_interval"] == 0:
                    run.checkpoint_and_evaluate()
            b = runs[1]
            if b.schedule.scale == 1.0 and not args.smoke:
                target = max(
                    target,
                    b.actual_steps
                    + max(0, p["min_full_scale_steps"] - b.full_scale_steps),
                )
                target = min(target, p["hard_cap_steps"])
        if not args.smoke:
            for run in runs:
                final_records = []
                for scale in sorted(set([1.0, run.schedule.scale])):
                    records, traces, count = evaluate(
                        run.net,
                        run.eval_env,
                        p["final_evaluation_seeds"],
                        scale,
                        run.actual_steps,
                        run.method,
                        run.seed,
                        run.iteration,
                    )
                    final_records.extend(records)
                    run.evaluation_steps += count
                    save_csv(
                        run.directory / f"heldout_final_trace_{scale:.2f}.csv", traces
                    )
                save_csv(run.directory / "heldout_final.csv", final_records)
        for run in runs:
            # No perturbation benchmark is claimed as handstand recovery here;
            # this is a separate takeoff-first task with a partial-maneuver goal.
            write_json(
                run.directory / "outcome.json",
                dict(
                    method=run.method,
                    training_seed=seed,
                    actual_steps=run.actual_steps,
                    full_scale_steps=run.full_scale_steps,
                    full_scale_requirement_met=(
                        run.method == "A"
                        or (
                            run.schedule.scale == 1.0
                            and run.full_scale_steps >= p["min_full_scale_steps"]
                        )
                    ),
                    final_scale=run.schedule.scale,
                    events=run.schedule.events,
                    stalled=run.method == "B" and run.schedule.scale < 1.0,
                    software_smoke=args.smoke,
                ),
            )
        assert runs[0].actual_steps == runs[1].actual_steps
    write_json(
        args.output / "completion.json",
        dict(
            software_smoke=args.smoke,
            extended_training=not args.smoke,
            physical_morphology=True,
            cpg=False,
            posture_curriculum=False,
        ),
    )


if __name__ == "__main__":
    main()
