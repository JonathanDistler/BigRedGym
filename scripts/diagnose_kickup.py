"""Matched-seed MuJoCo ablation; never trains or modifies a checkpoint."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import gym.envs  # noqa: F401
from gym.envs.go2.go2_config import FOOT_RADIUS
from gym.utils.helpers import class_to_dict, set_seed
from gym.utils.original_cfg import checkpoint_configs
from gym.utils.task_registry import task_registry


def run(args, mode):
    cfg, train = (
        checkpoint_configs(args.checkpoint, "go2kickup")
        if args.saved_config
        else task_registry.get_cfgs("go2kickup")
    )
    cfg.env.num_envs = args.episodes
    cfg.seed = 7
    cfg.push_robots.toggle = False
    train.runner.device = "cpu"
    train.runner.resume = False
    train.logging.enable_local_saving = False
    set_seed(7)
    task_registry.convert_frequencies_to_params(cfg, train)
    task_registry.set_log_dir_name(train)
    env = task_registry.make_env(
        "go2kickup", cfg, device="cpu", headless=True, backend="mujoco"
    )
    runner = task_registry.make_alg_runner(env, train)
    saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    runner.alg.actor.load_state_dict(saved["actor_state_dict"])
    runner.switch_to_eval()
    env.curriculum_level = args.level
    set_seed(71)
    env._reset_idx(torch.ones(args.episodes, dtype=torch.bool))
    initial = torch.cat((env.root_states, env.dof_pos, env.dof_vel), 1).clone()
    rows = []
    original = env._compute_torques
    m = cfg.kick_motion
    start = m.settle_seconds + m.crouch_seconds
    end = start + m.extend_seconds

    # Capture commanded targets and applied torque at physics frequency.
    def compute():
        clock = env._kick_clock + env._physics_offset
        ref = env._kick_reference(clock)[0]
        requested = ref + env.dof_pos_target
        limits = env.dof_pos_limits[env.actuated_dof_indices]
        desired = (
            requested.clamp(limits[:, 0], limits[:, 1])
            if getattr(cfg.kick_motion, "velocity_feedforward", False)
            else requested
        )
        torque = original()
        feet = env._rigid_body_pos[:, env.feet_indices]
        for i in range(args.episodes):
            row = dict(
                episode=i,
                time=float(clock[i]),
                phase=(
                    "settle"
                    if clock[i] < m.settle_seconds
                    else "crouch"
                    if clock[i] < start
                    else "extend"
                    if clock[i] < end
                    else "hold"
                ),
                terminated=bool(env.terminated[i]),
                timed_out=bool(env.timed_out[i]),
            )
            for j in range(12):
                row[f"reference_{j}"] = float(ref[i, j])
                row[f"residual_{j}"] = float(env.dof_pos_target[i, j])
                row[f"requested_{j}"] = float(requested[i, j])
                row[f"desired_{j}"] = float(desired[i, j])
                row[f"actual_{j}"] = float(env.dof_pos[i, env.actuated_dof_indices[j]])
                row[f"torque_{j}"] = float(torque[i, j])
            for j in range(4):
                row[f"force_{j}_N"] = float(
                    env.contact_forces[i, env.feet_indices[j], 2].clamp(min=0)
                )
                row[f"clearance_{j}_m"] = float(feet[i, j, 2] - FOOT_RADIUS)
                for k, axis in enumerate("xyz"):
                    row[f"foot_{j}_{axis}"] = float(feet[i, j, k])
            row["front_drift_m"] = float(env._front_displacement()[i].max())
            rows.append(row)
        return torque

    env._compute_torques = compute
    first_done = [None] * args.episodes
    with torch.inference_mode():
        for step in range(round(args.seconds / env.dt)):
            if mode == "reference":
                env.dof_pos_target.zero_()
                env.dof_vel_target.zero_()
                env.tau_ff.zero_()
            else:
                actions = runner.get_inference_actions()
                if mode == "zero":
                    actions = torch.zeros_like(actions)
                runner.set_actions(
                    runner.actor_cfg["actions"],
                    actions,
                    runner.actor_cfg["disable_actions"],
                )
            env.step()
            for i in range(args.episodes):
                if first_done[i] is None and (env.terminated[i] or env.timed_out[i]):
                    first_done[i] = (step + 1) * env.dt
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / f"{mode}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for i in range(args.episodes):
        r = [
            x
            for x in rows
            if x["episode"] == i
            and (first_done[i] is None or x["time"] < first_done[i])
        ]
        flight = [
            x
            for x in r
            if start <= x["time"] <= end + 0.65
            and min(x["clearance_2_m"], x["clearance_3_m"]) > 0.002
            and max(x["force_2_N"], x["force_3_N"])
            < 0.015 * float(env._body_weight()[i])
        ]
        push = [x for x in r if x["phase"] == "extend"]
        summaries.append(
            dict(
                first_done_s=first_done[i],
                max_min_clearance_m=max(
                    min(x["clearance_2_m"], x["clearance_3_m"]) for x in r
                ),
                rear_flight_s=len(flight) * (env.dt / cfg.control.decimation),
                supported_rear_flight_s=sum(
                    min(x["force_0_N"], x["force_1_N"])
                    >= cfg.reward_settings.min_front_load * float(env._body_weight()[i])
                    for x in flight
                )
                * (env.dt / cfg.control.decimation),
                max_front_drift_m=max(x["front_drift_m"] for x in r),
                push_mean_knee_residual_rad=[
                    sum(x[f"residual_{j}"] for x in push) / max(1, len(push))
                    for j in (8, 11)
                ],
                push_peak_knee_torque_Nm=[
                    max([abs(x[f"torque_{j}"]) for x in push], default=0)
                    for j in (8, 11)
                ],
            )
        )
    report = dict(
        mode=mode,
        initial_sha256=hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
        episodes=summaries,
        config=class_to_dict(cfg),
        phase_boundaries_s=[m.settle_seconds, start, end],
        auto_resets=0,
    )
    (dest / f"{mode}.json").write_text(json.dumps(report, indent=2))
    env._backend.close()
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--saved-config", action="store_true")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seconds", type=float, default=2)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    if args.episodes < 1 or args.seconds <= 0 or not 0 <= args.level <= 6:
        p.error("episodes/seconds must be positive and level must be 0 through 6")
    reports = [run(args, m) for m in ("reference", "zero", "policy")]
    assert len({r["initial_sha256"] for r in reports}) == 1, "Initial states differ"
    source = Path("gym/envs/go2/go2kickup.py")
    snapshot = Path(args.checkpoint).parent / "files" / source
    summary = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest(),
        saved_config=args.saved_config,
        controller_source=str(source),
        controller_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        controller_matches_snapshot=snapshot.is_file()
        and source.read_bytes() == snapshot.read_bytes(),
        cases=[{k: v for k, v in r.items() if k != "config"} for r in reports],
    )
    Path(args.output_dir, "comparison.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
