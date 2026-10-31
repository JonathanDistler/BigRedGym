"""Stream selected robots to CSV without retaining trajectories in memory."""

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


class RobotMetrics:
    def __init__(self, env, output_dir, num_envs=1):
        self.env = env
        self.count = min(num_envs, env.num_envs)
        if self.count < 1:
            raise ValueError("metrics_num_envs must be positive")
        self.path = Path(output_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.file = (self.path / "robot_metrics.csv").open("w", newline="")
        self.writer = None
        self.steps = 0
        self.episodes = [0] * self.count
        self.previous_episode_steps = [-1] * self.count
        self.sums = {}
        self.peaks = {}
        self.rows = 0
        metadata = {
            "feet": list(env.robot_layout.body_groups["feet"]),
            "contact_threshold_N": 1.0,
            "force_frame": "world",
            "base_velocity_frame": "body",
            "quaternion_order": "xyzw",
            "angles": "radians",
            "symmetry": "1 - abs(left-right)/(left+right); blank when unloaded",
            "sampling": (
                "end of each control step; forces are instantaneous, "
                "not substep averages"
            ),
            "num_logged_envs": self.count,
            "dt_s": env.dt,
            "contact_modes": getattr(
                getattr(env.cfg, "contact_cpg", None), "modes", None
            ),
            "cpg_enabled": getattr(env, "contact_cpg_enabled", False),
        }
        (self.path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    @torch.no_grad()
    def record(self):
        e = self.env
        self.steps += 1
        n = self.count
        forces = e.contact_forces[:n, e.feet_indices].detach().cpu()
        root = e.root_states[:n].detach().cpu()
        linear = e.base_lin_vel[:n].detach().cpu()
        angular = e.base_ang_vel[:n].detach().cpu()
        joints = e.dof_pos[:n].detach().cpu()
        joint_vel = e.dof_vel[:n].detach().cpu()
        steps = e.episode_length_buf[:n].cpu().tolist()
        phases = (
            e.leg_phase[:n].cpu() if getattr(e, "contact_cpg_enabled", False) else None
        )
        desired = e.cpg_contact_obs[:n].cpu() if phases is not None else None
        for i in range(n):
            if steps[i] <= self.previous_episode_steps[i]:
                self.episodes[i] += 1
            self.previous_episode_steps[i] = steps[i]
            row = dict(
                time_s=self.steps * e.dt,
                env_id=i,
                episode_id=self.episodes[i],
                episode_time_s=steps[i] * e.dt,
            )
            row["terminated"] = int(e.terminated[i])
            row["timed_out"] = int(e.timed_out[i])
            for axis, value in zip("xyz", root[i, :3]):
                row[f"base_position_{axis}_m"] = float(value)
            x, y, z, w = root[i, 3:7].tolist()
            for axis, value in zip("xyzw", (x, y, z, w)):
                row[f"base_quaternion_{axis}"] = value
            row["base_roll_rad"] = math.atan2(
                2 * (w * x + y * z), 1 - 2 * (x * x + y * y)
            )
            row["base_pitch_rad"] = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
            row["base_yaw_rad"] = math.atan2(
                2 * (w * z + x * y), 1 - 2 * (y * y + z * z)
            )
            # Angle between body +x and world down: zero at a nose-down handstand.
            row["handstand_angle_error_rad"] = math.acos(
                max(-1, min(1, -2 * (x * z - w * y)))
            )
            for axis, v, a in zip("xyz", linear[i], angular[i]):
                row[f"base_linear_velocity_{axis}_m_s"] = float(v)
                row[f"base_angular_velocity_{axis}_rad_s"] = float(a)
            row["base_angular_speed_rad_s"] = float(angular[i].norm())
            loads = forces[i, :, 2].clamp(min=0)
            total = float(loads.sum())
            row["total_vertical_load_N"] = total
            row["supporting_legs"] = int((loads > 1).sum())
            for j, name in enumerate(e.robot_layout.body_groups["feet"]):
                for axis, force in zip("xyz", forces[i, j]):
                    row[f"{name}_force_{axis}_N"] = float(force)
                row[f"{name}_force_magnitude_N"] = float(forces[i, j].norm())
                row[f"{name}_contact"] = int(loads[j] > 1)
                row[f"{name}_load_share"] = float(loads[j]) / total if total > 1 else ""
                if phases is not None:
                    row[f"{name}_phase_rad"] = float(phases[i, j])
                    row[f"{name}_desired_contact"] = float(desired[i, j])
            for name, left, right in (
                ("front", float(loads[0]), float(loads[1])),
                ("rear", float(loads[2]), float(loads[3])),
                ("left_right", float(loads[0] + loads[2]), float(loads[1] + loads[3])),
            ):
                row[f"{name}_load_difference_N"] = left - right
                row[f"{name}_force_symmetry"] = (
                    1 - abs(left - right) / (left + right) if left + right > 1 else ""
                )
            for j, name in enumerate(e.dof_names):
                row[f"{name}_angle_rad"] = float(joints[i, j])
                row[f"{name}_velocity_rad_s"] = float(joint_vel[i, j])
            if self.writer is None:
                self.writer = csv.DictWriter(self.file, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.rows += 1
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    acc = self.sums.setdefault(key, [0.0, 0])
                    acc[0] += value
                    acc[1] += 1
                    self.peaks[key] = max(self.peaks.get(key, 0), abs(value))
        if self.steps % 100 == 0:
            self.file.flush()

    def close(self):
        if self.file.closed:
            return
        self.file.close()
        summary = {
            "rows": self.rows,
            "control_steps": self.steps,
            "mean": {k: v[0] / v[1] for k, v in self.sums.items()},
            "peak_absolute": self.peaks,
        }
        (self.path / "summary.json").write_text(json.dumps(summary, indent=2))


def add_robot_options(parser):
    parser.add_argument(
        "--cpg",
        action="store_true",
        help="Enable Go2 force-coupled leg oscillators (train a new policy)",
    )
    parser.add_argument(
        "--contact_modes",
        nargs=4,
        choices=["phase", "stance", "swing", "ignore"],
        metavar="MODE",
        help="FL FR RL RR desired contact modes; requires --cpg or saved CPG config",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Metrics destination (default: output/<timestamp>)",
    )
    parser.add_argument(
        "--metrics_num_envs",
        type=int,
        default=1,
        help="Number of robots to log per control step",
    )


def configure_contact_cpg(args, env_cfg, train_cfg):
    enabled = getattr(args, "cpg", False)
    modes = getattr(args, "contact_modes", None)
    c = getattr(env_cfg, "contact_cpg", None)
    if (enabled or modes) and c is None:
        raise ValueError("CPG contact options require a Go2 task")
    if c is None:
        return
    if enabled:
        c.enabled = True
    if modes:
        if not c.enabled:
            raise ValueError(
                "--contact_modes requires --cpg or an enabled saved CPG config"
            )
        c.modes = modes
    if c.enabled:
        for network in (train_cfg.actor, train_cfg.critic):
            network.obs = list(network.obs)
            for name in ("cpg_phase_obs", "cpg_contact_obs"):
                if name not in network.obs:
                    network.obs.append(name)
        train_cfg.critic.reward.weights.cpg_contact = 1.0


def attach_robot_metrics(env, args):
    if not hasattr(env, "feet_indices") or len(env.feet_indices) != 4:
        return None
    output = getattr(args, "output_dir", None) or Path("output") / datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S_%fZ")
    recorder = RobotMetrics(env, output, getattr(args, "metrics_num_envs", 1))
    original_step = env.step

    def step_with_metrics():
        result = original_step()
        recorder.record()
        return result

    env.step = step_with_metrics
    env.robot_metrics = recorder
    print(f"Robot metrics: {recorder.path.resolve()}")
    return recorder
