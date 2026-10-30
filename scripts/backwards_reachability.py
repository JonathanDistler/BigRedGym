"""Sample and plot empirical recovery sets.

Run with python -m scripts.backwards_reachability.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2reachability import Go2Reachability
from gym.envs.go2.go2reachability_config import (
    Go2ReachabilityCfg,
    Go2ReachabilityRunnerCfg,
)
from gym.utils.helpers import set_seed
from gym.utils.task_registry import task_registry


def plots(rows, out, label):
    plt.rcParams.update(
        {
            "figure.facecolor": "#f4f7fb",
            "axes.facecolor": "#ffffff",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "savefig.dpi": 180,
        }
    )
    colors = ["#31688e", "#35b779", "#fde725"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    accepted = rows["accepted"].astype(bool)
    for j, ax in enumerate(axes.flat):
        x, y = rows[f"q{j + 6}"], rows[f"v{j + 6}"]
        ax.scatter(
            x[~accepted],
            y[~accepted],
            c="#aeb8c4",
            marker="x",
            s=18,
            label="fallback replay",
        )
        sc = ax.scatter(
            x[accepted],
            y[accepted],
            c=rows["success"][accepted],
            cmap="viridis",
            vmin=0,
            vmax=1,
            s=24,
            alpha=0.8,
        )
        ax.set(
            xlabel="Initial angle [rad]",
            ylabel="Initial angular velocity [rad/s]",
            title=["RL hip", "RL thigh", "RL calf", "RR hip", "RR thigh", "RR calf"][j],
        )
        ax.grid(alpha=0.15)
    fig.colorbar(
        sc,
        ax=axes,
        label="Contact-preserving recovery (0 = fail, 1 = success)",
        shrink=0.7,
    )
    fig.suptitle(
        f"Empirical reachable-set projections • {label}\n"
        "Other state coordinates vary; these are not guaranteed boundaries"
    )
    fig.savefig(out / "phase_space.png")
    plt.close(fig)
    levels = np.unique(rows["level"])
    metrics = []
    for level in levels:
        mask = rows["level"] == level
        valid = mask & accepted
        n = int(valid.sum())
        k = int(rows["success"][valid].sum())
        p = k / max(n, 1)
        # Wilson 95% interval, conditional on admissible reset proposals.
        z = 1.96
        den = 1 + z * z / max(n, 1)
        center = (p + z * z / (2 * max(n, 1))) / den
        half = z * np.sqrt(p * (1 - p) / max(n, 1) + z * z / (4 * max(n, 1) ** 2)) / den
        metrics.append(
            dict(
                level=int(level),
                samples=int(mask.sum()),
                accepted=n,
                success_rate=p if n else None,
                wilson95=[max(0, center - half), min(1, center + half)] if n else None,
                acceptance_rate=float(accepted[mask].mean()),
                proposal_acceptance_rate=n
                / max(float(rows["attempts"][mask].sum()), 1),
                contact_failure_rate=float(rows["contact_failed"][valid].mean())
                if n
                else None,
            )
        )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, key, title, color in zip(
        axes,
        ["proposal_acceptance_rate", "success_rate", "contact_failure_rate"],
        [
            "Admissible proposals / all proposals",
            "Recovery probability | admissible",
            "Contact failure | admissible",
        ],
        colors,
    ):
        values = [m[key] if m[key] is not None else np.nan for m in metrics]
        ax.bar(levels, values, color=color, alpha=0.9)
        if key == "success_rate":
            for level, m in zip(levels, metrics):
                if m["wilson95"]:
                    ax.plot([level, level], m["wilson95"], color="#213547", linewidth=2)
        ax.set(title=title, xlabel="Outward curriculum level", ylim=(0, 1.05))
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle(label)
    fig.savefig(out / "recovery_metrics.png")
    plt.close(fig)
    return metrics


def torque_envelope(env, out):
    """Report one-axis conditional bounds at the nominal goal geometry."""
    sampler = env._sampler
    root = env.root_states[0].cpu().numpy()
    joints = env.initial_joints[0].cpu().numpy()
    bounds, inertias = [], []
    for axis in range(6):
        signed = []
        for sign in (-1, 1):
            sampler.load(root, joints)
            direction = np.zeros(6)
            direction[axis] = sign
            velocity, inertia = sampler.direction(direction)
            bound = sampler.bound(joints, direction, inertia)
            # Other motors also move to maintain front-foot constraints.
            limit = np.min(
                sampler.speed / np.maximum(np.abs(velocity[sampler.v]), 1e-12)
            )
            signed.append(min(bound, limit))
        bounds.append(signed)
        inertias.append(inertia)
    bounds = np.array(bounds)
    names = ["RL hip", "RL thigh", "RL calf", "RR hip", "RR thigh", "RR calf"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = np.arange(6)
    ax.bar(
        x - 0.18, bounds[:, 0], width=0.36, color="#31688e", label="Negative rotation"
    )
    ax.bar(
        x + 0.18, bounds[:, 1], width=0.36, color="#35b779", label="Positive rotation"
    )
    ax.set(
        xticks=x,
        xticklabels=names,
        ylabel="Angular speed [rad/s]",
        title="Torque-work sampling envelope at goal pose • frozen geometry",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.15)
    fig.savefig(out / "torque_velocity_envelope.png", dpi=180)
    plt.close(fig)
    report = dict(
        joints=names,
        negative_positive_rad_s=bounds.tolist(),
        effective_inertia_kg_m2=inertias,
        rear_torque_limits_Nm=sampler.torque[6:].tolist(),
        motor_speed_limits_rad_s=sampler.speed[6:].tolist(),
        assumptions=(
            "Start from rest; frozen contact-compatible inertia; "
            "rear torque work only; no gravity energy or front motor work."
        ),
    )
    (out / "torque_derivation.json").write_text(json.dumps(report, indent=2) + "\n")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, help="Compatible Go2 balance/reachability checkpoint"
    )
    parser.add_argument(
        "--samples", type=int, default=64, help="Samples per curriculum level"
    )
    parser.add_argument("--seconds", type=float, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path, default=Path("output_backwards_reachability")
    )
    args = parser.parse_args()
    if args.samples < 1 or args.seconds < 2.1:
        parser.error(
            "samples must be positive and seconds >= 2.1 (two-second success hold)"
        )
    torch.set_num_threads(1)
    set_seed(args.seed)
    cfg, train_cfg = Go2ReachabilityCfg(), Go2ReachabilityRunnerCfg()
    cfg.seed = args.seed
    cfg.env.num_envs = args.samples
    cfg.handstand_curriculum.replay_fraction = 0
    cfg.handstand_curriculum.enabled = True
    cfg.handstand_curriculum.min_episodes = 10**12
    task_registry.convert_frequencies_to_params(cfg, train_cfg)
    backend = MuJocoCPUBackend()
    env = Go2Reachability(cfg, "cpu", True, backend)
    args.output.mkdir(parents=True, exist_ok=True)
    torque_envelope(env, args.output)
    runner = None
    label = "Fixed goal-posture PD baseline (no trained policy)"
    try:
        if args.checkpoint:
            train_cfg.logging.enable_local_saving = False
            train_cfg.runner.resume = False
            task_registry.set_log_dir_name(train_cfg)
            runner = task_registry.make_alg_runner(env, train_cfg)
            runner.load(str(args.checkpoint), load_optimizer=False)
            runner.switch_to_eval()
            label = f"Policy: {args.checkpoint.name}"
        records = []
        for level in range(len(cfg.reachability.position_fraction)):
            env.curriculum_level = level
            env._reset_idx(torch.ones(env.num_envs, dtype=torch.bool))
            accepted = env.reset_accepted.cpu().numpy().copy()
            initial_q = env.initial_joints.cpu().numpy().copy()
            initial_v = env.initial_velocity.cpu().numpy().copy()
            envelope = env.velocity_envelope.cpu().numpy().copy()
            attempts = env.reset_attempts.cpu().numpy().copy()
            failed = torch.zeros(env.num_envs, dtype=torch.bool)
            contact = failed.clone()
            best = torch.zeros(env.num_envs)
            peak_speed = torch.zeros(env.num_envs)
            min_load = torch.full((env.num_envs,), float("inf"))
            max_slip = torch.zeros(env.num_envs)
            first_recovery = torch.full((env.num_envs,), float("nan"))
            for step in range(int(args.seconds / env.dt)):
                if runner:
                    runner.set_actions(
                        runner.actor_cfg["actions"],
                        runner.get_inference_actions(),
                        runner.actor_cfg["disable_actions"],
                    )
                else:
                    env.dof_pos_target.zero_()
                env.step()
                failed |= env.terminated.cpu()
                contact |= env.contact_failed.cpu()
                peak_speed = torch.maximum(
                    peak_speed, env.dof_vel[:, 6:].abs().amax(dim=1).cpu()
                )
                min_load = torch.minimum(
                    min_load,
                    env.contact_forces[:, env.feet_indices[:2], 2].amin(dim=1).cpu(),
                )
                max_slip = torch.maximum(
                    max_slip,
                    env._rigid_body_lin_vel[:, env.feet_indices[:2]]
                    .norm(dim=2)
                    .amax(dim=1)
                    .cpu(),
                )
                newly_held = (
                    (env._best_hold.cpu() >= cfg.handstand_curriculum.hold_seconds)
                    & ~failed
                    & torch.isnan(first_recovery)
                )
                first_recovery[newly_held] = (step + 1) * env.dt
                best = torch.maximum(
                    best, torch.where(failed, 0.0, env._best_hold.cpu())
                )
            success = (
                (best >= cfg.handstand_curriculum.hold_seconds)
                & ~failed
                & torch.from_numpy(accepted)
            )
            for i in range(env.num_envs):
                row = dict(
                    level=level,
                    accepted=int(accepted[i]),
                    attempts=int(attempts[i]),
                    envelope=float(envelope[i]),
                    success=int(success[i]),
                    contact_failed=int(contact[i]),
                    best_hold=float(best[i]),
                    peak_rear_speed=float(peak_speed[i]),
                    min_front_load_N=float(min_load[i]),
                    max_front_slip_m_s=float(max_slip[i]),
                    first_recovery_s=float(first_recovery[i]),
                )
                row.update({f"q{j}": float(initial_q[i, j]) for j in range(12)})
                row.update({f"v{j}": float(initial_v[i, j]) for j in range(12)})
                records.append(row)
            print(
                f"Level {level}: accepted {accepted.sum()}/{env.num_envs}, "
                f"recovered {success.sum().item()}",
                flush=True,
            )
        rows = {key: np.array([r[key] for r in records]) for key in records[0]}
        np.savez_compressed(args.output / "samples.npz", **rows)
        import csv

        with (args.output / "samples.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
        metrics = plots(rows, args.output, label)
        report = dict(
            controller=label,
            seed=args.seed,
            horizon_seconds=args.seconds,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            if args.checkpoint
            else None,
            metrics=metrics,
            interpretation=(
                "Conditional empirical recovery probabilities under the "
                "sampled distribution; no global reachability guarantee."
            ),
        )
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
