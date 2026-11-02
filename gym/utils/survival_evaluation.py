"""First-failure survival trials with a bounded pre-failure trace per robot."""

from collections import deque
import hashlib
import json
import math
from pathlib import Path

import torch


@torch.inference_mode()
def evaluate_survival(env, runner, seconds=60.0, seed=None, output=None):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if not hasattr(env, "_best_hold"):
        raise ValueError("Survival evaluation requires a handstand/balance task")
    path = Path(output or env.robot_metrics.path)
    path.mkdir(parents=True, exist_ok=True)
    n = env.num_envs
    failed = torch.zeros(n, dtype=torch.bool, device=env.device)
    trials = [
        dict(
            env_id=i,
            first_failure_s=None,
            observed_seconds=0.0,
            best_hold_s=0.0,
            front_contact_loss_steps=0,
            first_front_contact_loss_s=None,
        )
        for i in range(n)
    ]
    traces = [deque(maxlen=math.ceil(1 / env.dt) + 1) for _ in range(n)]
    steps = math.ceil(seconds / env.dt)
    for step in range(steps):
        active = ~failed
        runner.set_actions(
            runner.actor_cfg["actions"],
            runner.get_inference_actions(),
            runner.actor_cfg["disable_actions"],
        )
        env.step()
        elapsed = (step + 1) * env.dt
        lost = (env.contact_forces[:, env.feet_indices[:2], 2] < 1).any(dim=1)
        # Export physical target angles, not the policy's residual action.
        targets = env._standing_posture() + env.dof_pos_target
        for i in active.nonzero().flatten().tolist():
            trial = trials[i]
            trial["observed_seconds"] = elapsed
            trial["best_hold_s"] = max(trial["best_hold_s"], float(env._best_hold[i]))
            if elapsed >= 0.03 and bool(lost[i]):
                trial["front_contact_loss_steps"] += 1
                if trial["first_front_contact_loss_s"] is None:
                    trial["first_front_contact_loss_s"] = elapsed
            traces[i].append(
                dict(
                    time_s=elapsed,
                    root_state=env.root_states[i].cpu().tolist(),
                    body_linear_velocity_m_s=env.base_lin_vel[i].cpu().tolist(),
                    body_angular_velocity_rad_s=env.base_ang_vel[i].cpu().tolist(),
                    joint_angles_rad=env.dof_pos[i].cpu().tolist(),
                    joint_velocities_rad_s=env.dof_vel[i].cpu().tolist(),
                    joint_targets_rad=targets[i].cpu().tolist(),
                    joint_torques_Nm=env.torques[i].cpu().tolist(),
                    foot_positions_m=env._rigid_body_pos[i, env.feet_indices]
                    .cpu()
                    .tolist(),
                    foot_velocities_m_s=env._rigid_body_lin_vel[i, env.feet_indices]
                    .cpu()
                    .tolist(),
                    foot_forces_N=env.contact_forces[i, env.feet_indices]
                    .cpu()
                    .tolist(),
                )
            )
            if bool(env.terminated[i]):
                failed[i] = True
                trial["first_failure_s"] = elapsed
                (path / f"failure_env_{i}.json").write_text(
                    json.dumps(list(traces[i]), indent=2) + "\n"
                )
                traces[i].clear()
        if failed.all():
            break
    checkpoint = getattr(env, "evaluation_checkpoint", None)
    report = dict(
        seed=seed,
        horizon_seconds=seconds,
        dt_s=env.dt,
        checkpoint=checkpoint,
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if checkpoint
        else None,
        failures=int(failed.sum()),
        robots=n,
        survival_fraction=float((~failed).float().mean()),
        mean_observed_seconds=sum(t["observed_seconds"] for t in trials) / n,
        hold_seconds=env.cfg.handstand_curriculum.hold_seconds,
        pushes_enabled=env.cfg.balance_perturbations.enabled,
        joint_names=list(env.dof_names),
        foot_names=list(env.robot_layout.body_groups["feet"]),
        interpretation="Each trial ends at its first termination; survivors are "
        "right-censored at the horizon. Contact loss is diagnostic, not a fall. "
        "Traces contain final-substep samples, root quaternion xyzw, world foot "
        "coordinates and forces, and body-frame base velocities.",
        trials=trials,
    )
    (path / "survival.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Survived {seconds:g}s: {n - report['failures']}/{n}; "
        f"report: {path / 'survival.json'}"
    )
    return report
