"""Deterministic, repeated four-second kicks with physical telemetry."""

import argparse
import importlib.util
import json
import hashlib
from pathlib import Path

import torch
import gym.envs  # noqa: F401
from gym.utils.original_cfg import checkpoint_configs
from gym.utils.task_registry import task_registry
from gym.utils.robot_metrics import RobotMetrics
from gym.utils.helpers import class_to_dict


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--saved-config", action="store_true")
    p.add_argument("--level", type=int, choices=range(7), default=0)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    if args.episodes < 1:
        p.error("--episodes must be positive")
    if args.saved_config:
        cfg, train = checkpoint_configs(args.checkpoint, "go2kickup")
        source = Path(args.checkpoint).parent / "files/gym/envs/go2/go2kickup.py"
        spec = importlib.util.spec_from_file_location("saved_kickup", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task_registry.task_classes["go2kickup"] = module.Go2Kickup
    else:
        cfg, train = task_registry.get_cfgs("go2kickup")
    cfg.env.num_envs = 16
    cfg.seed = 7
    train.runner.device = "cpu"
    train.runner.resume = False
    train.logging.enable_local_saving = False
    task_registry.convert_frequencies_to_params(cfg, train)
    task_registry.set_log_dir_name(train)
    env = task_registry.make_env(
        "go2kickup", cfg, device="cpu", headless=True, backend="mujoco"
    )
    runner = task_registry.make_alg_runner(env, train)
    saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    runner.alg.actor.load_state_dict(saved["actor_state_dict"])
    runner.alg.critic.load_state_dict(saved["critic_state_dict"])
    runner.switch_to_eval()
    env.curriculum_level = args.level
    metrics = RobotMetrics(env, args.output_dir, 16)
    results = []
    with torch.inference_mode():
        for _ in range((args.episodes + 15) // 16):
            env._reset_idx(torch.ones(16, dtype=torch.bool))
            peak = torch.zeros(16)
            drift = torch.zeros(16)
            support = torch.zeros(16)
            alive = torch.ones(16, dtype=torch.bool)
            loaded = torch.zeros(16, 2, dtype=torch.bool)
            events = torch.full((16, 2), -1.0)
            impulse = torch.zeros(16, 2)
            best_flight = torch.zeros(16)
            flight = torch.zeros(16)
            contact = torch.zeros(16, 2, dtype=torch.bool)
            goal = env._goal("clearance").clamp(min=0.003)
            tolerance = (
                env._goal("takeoff_tolerance")
                if hasattr(cfg.kick_curriculum, "takeoff_tolerance")
                else torch.full((16,), 0.12)
            )
            for _ in range(int(cfg.env.episode_length_s / env.dt)):
                runner.set_actions(
                    runner.actor_cfg["actions"],
                    runner.get_inference_actions(),
                    runner.actor_cfg["disable_actions"],
                )
                env.step()
                alive &= ~env.terminated
                m = cfg.kick_motion
                push = (env._kick_clock >= m.settle_seconds + m.crouch_seconds) & (
                    env._kick_clock
                    < m.settle_seconds + m.crouch_seconds + m.extend_seconds
                )
                lift = push | (
                    (
                        env._kick_clock
                        >= m.settle_seconds + m.crouch_seconds + m.extend_seconds
                    )
                    & (
                        env._kick_clock
                        < m.settle_seconds + m.crouch_seconds + m.extend_seconds + 0.65
                    )
                )
                rear_load = env._handstand_loads()[:, 2:]
                loaded |= rear_load > 0.03
                clear = env._rear_clearance()
                upward = env._rigid_body_lin_vel[:, env.feet_indices[2:], 2] > 0
                contact[:] = torch.where(contact, rear_load > 0.015, rear_load >= 0.03)
                off = ~contact & (rear_load < 0.015) & (clear > 0.002)
                first = loaded & off & upward & (events < 0) & lift[:, None]
                events[:] = torch.where(first, env._kick_clock[:, None], events)
                flying = (
                    off.all(1)
                    & (events >= 0).all(1)
                    & env._valid_kick_support()
                    & alive
                    & lift
                    & ((events[:, 0] - events[:, 1]).abs() <= tolerance)
                    & (clear >= goal[:, None]).all(1)
                )
                if args.level == 6:
                    flying &= (env.projected_gravity[:, 0] > 0.15) & (
                        env.base_ang_vel[:, 1].abs() < 3
                    )
                flight[:] = torch.where(flying, flight + env.dt, 0.0)
                best_flight = torch.maximum(best_flight, flight)
                impulse += (
                    rear_load
                    * (9.81 * env._backend.link_mass.sum(1))[:, None]
                    * env.dt
                    * push[:, None]
                )
                peak = torch.maximum(peak, env._rear_clearance().amin(1) * alive)
                drift = torch.maximum(drift, env._front_displacement().amax(1))
                support += (
                    env._handstand_loads()[:, :2] >= cfg.reward_settings.min_front_load
                ).all(1)
                for name in class_to_dict(train.critic.reward.weights):
                    assert torch.isfinite(getattr(env, "_reward_" + name)()).all(), name
                metrics.record()
            for i in range(16):
                results.append(
                    dict(
                        max_min_clearance_m=float(peak[i]),
                        max_front_drift_m=float(drift[i]),
                        front_contact_retention=float(
                            support[i] / int(cfg.env.episode_length_s / env.dt)
                        ),
                        simultaneous_takeoff=bool(best_flight[i] > 0),
                        event_timing_coordinated=bool(
                            (events[i] >= 0).all()
                            and (events[i, 0] - events[i, 1]).abs() <= tolerance[i]
                        ),
                        sustained_flight_s=float(best_flight[i]),
                        success=bool(
                            best_flight[i] >= 0.06
                            and alive[i]
                            and not env._support_failed[i]
                        ),
                        curriculum_success=bool(
                            env._best_hold[i] >= cfg.handstand_curriculum.hold_seconds
                        ),
                        push_impulse_Ns=impulse[i].tolist(),
                        push_impulse_bodyweight_seconds=(
                            impulse[i] / (9.81 * env._backend.link_mass[i].sum())
                        ).tolist(),
                        survived=bool(alive[i]),
                    )
                )
    metrics.close()
    env._backend.close()
    results = results[: args.episodes]
    report = dict(
        checkpoint=args.checkpoint,
        checkpoint_sha256=hashlib.sha256(
            Path(args.checkpoint).read_bytes()
        ).hexdigest(),
        saved_config=args.saved_config,
        level=args.level,
        episodes=results,
        success_percentage=100 * sum(x["success"] for x in results) / len(results),
        simultaneous_takeoff_rate=sum(x["simultaneous_takeoff"] for x in results)
        / len(results),
    )
    Path(args.output_dir, "kickup_evaluation.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps({k: v for k, v in report.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
