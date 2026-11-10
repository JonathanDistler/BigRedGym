"""Constrained reference search on the task's unchanged native MuJoCo model."""

import argparse
import copy
import csv
import json
import hashlib
from pathlib import Path
import time
from dataclasses import replace

import mujoco
import numpy as np
import torch
import gym.envs  # noqa: F401
from gym.envs.go2.go2kickup_config import Go2KickupCfg, Go2KickupRunnerCfg
from gym.envs.go2.go2_config import FOOT_RADIUS
from gym.envs.go2.kickup_reference import (
    KickReference,
    reference,
    blend,
    physical_liftoff,
)
from gym.utils.helpers import set_seed, class_to_dict
from gym.utils.task_registry import task_registry


class Experiment:
    def __init__(self):
        cfg, train = Go2KickupCfg(), Go2KickupRunnerCfg()
        cfg.env.num_envs = 1
        cfg.seed = 7
        set_seed(7)
        task_registry.convert_frequencies_to_params(cfg, train)
        self.config = class_to_dict(cfg)
        env = task_registry.make_env(
            "go2kickup", cfg, device="cpu", headless=True, backend="mujoco"
        )
        env._reset_idx(torch.ones(1, dtype=torch.bool))
        b = env._backend
        self.model = copy.copy(b._model_for_env(0))
        self.initial = b._datas[0].qpos.copy()
        self.order = b._canonical_to_native_dof_np.copy()
        self.feet = b._canonical_to_native_body_np[env.feet_indices.cpu().numpy()]
        self.standing = env._standing_posture()[0].numpy().copy()
        self.limits = env.dof_pos_limits[env.actuated_dof_indices].numpy().copy()
        self.torque_limits = env.actuated_torque_limits.numpy().copy()
        self.velocity_limits = (
            env.dof_vel_limits[env.actuated_dof_indices].numpy().copy()
        )
        self.base = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base"))
        if self.base < 0:
            self.base = 1
        env._backend.close()

    def provenance(self):
        return dict(
            config=self.config,
            standing=self.standing.tolist(),
            native_joint_order=self.order.tolist(),
            initial_qpos=self.initial.tolist(),
            body_mass_kg=self.model.body_mass.tolist(),
            body_inertia=self.model.body_inertia.tolist(),
            torque_limits_Nm=self.torque_limits.tolist(),
            velocity_limits_rad_s=self.velocity_limits.tolist(),
            joint_limits_rad=self.limits.tolist(),
            timestep_s=self.model.opt.timestep,
            geom_solref=self.model.geom_solref.tolist(),
            urdf_sha256=hashlib.sha256(
                Path("resources/robots/go2/urdf/go2.urdf").read_bytes()
            ).hexdigest(),
        )

    def trial(
        self,
        p,
        seed=100,
        baseline=False,
        record=False,
        viewer=False,
        seconds=1.6,
        front_gain_override=None,
    ):
        rng = np.random.default_rng(seed)
        m = copy.copy(self.model)
        mass_scale = rng.uniform(0.95, 1.05)
        m.body_mass[:] *= mass_scale
        m.body_inertia[:] *= mass_scale
        m.geom_friction[:, 0] = rng.uniform(0.6, 1.0)
        mujoco.mj_setConst(m, mujoco.MjData(m))
        d = mujoco.MjData(m)
        d.qpos[:] = self.initial
        d.qpos[7 + self.order] += rng.normal(0, 0.008, 12)
        angles = rng.uniform(-0.012, 0.012, 3)
        angles[2] = rng.uniform(-0.15, 0.15)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, angles, "xyz")
        d.qpos[3:7] = quat
        mujoco.mj_forward(m, d)
        d.qpos[2] += FOOT_RADIUS + 0.001 - d.xpos[self.feet, 2].min()
        mujoco.mj_forward(m, d)
        anchor = d.xpos[self.feet[:2], :2].copy()
        kp = 60 * rng.uniform(0.95, 1.05, 12)
        kd = 3 * rng.uniform(0.95, 1.05, 12)
        bw = m.body_mass.sum() * 9.81
        h = m.opt.timestep
        impulse = np.zeros(2)
        excess_impulse = np.zeros(2)
        peak = 0.0
        supported_peak = 0.0
        driftmax = 0.0
        hold = 0.0
        besthold = 0.0
        hold30 = besthold30 = 0.0
        loaded = np.zeros(2, dtype=bool)
        frontloss = 0.0
        frontlossmax = 0.0
        bodyhit = False
        contact_bodies = set()
        first_body_contact = None
        maxpitch = 0.0
        maxmomentum = 0.0
        sat = np.zeros(12)
        track = np.zeros(12)
        samples = 0
        speedmax = 0.0
        allspeedmax = 0.0
        allsat = np.zeros(12)
        torquepeak = np.zeros(12)
        front_samples = support_samples = 0
        rows = []
        wrench = np.zeros(6)
        force = np.zeros((4, 3))
        cop = np.zeros(3)
        viewer_handle = None
        if viewer:
            from mujoco import viewer as mj_viewer

            viewer_handle = mj_viewer.launch_passive(m, d)
            viewer_handle.cam.distance = 1.4
            viewer_handle.cam.azimuth = 90
            viewer_handle.cam.elevation = -15
            viewer_handle.cam.lookat[:] = [0.0, 0.0, 0.25]
        for step in range(round(seconds / h)):
            t = step * h
            if baseline:
                # Exact stage-3 source reference, amplitude 0.85, no tuck.
                a, da = blend(t, 0.15, 0.55)
                b, db = blend(t, 0.7, 0.2)
                folded = self.standing.copy()
                folded[[7, 10]] = 1.15
                folded[[8, 11]] = -2.3
                thrust = folded.copy()
                thrust[[7, 10]] = 0.30
                thrust[[8, 11]] = -0.85
                q = self.standing + 0.85 * (
                    (1 - b) * a * (folded - self.standing)
                    + b * (thrust - self.standing)
                )
                v = 0.85 * (
                    (1 - b) * da * (folded - self.standing)
                    + db * (thrust - (self.standing + a * (folded - self.standing)))
                )
                fg = 0.65
                fd = 1.0
                ff = 1.0
                pushstart = 0.7
                pushend = 0.9
            else:
                q, v = reference(t, p, self.standing)
                fg = p.front_gain
                fd = p.front_damping
                ff = p.feedforward
                pushstart = p.settle + p.crouch
                pushend = pushstart + p.extend * max(
                    p.hip_delay + p.hip_duration, p.knee_delay + p.knee_duration
                )
            if front_gain_override is not None:
                fg = front_gain_override
            q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
            v = np.clip(v, -self.velocity_limits, self.velocity_limits)
            gains = np.ones(12)
            gains[:6] = fg
            damping = np.ones(12)
            damping[:6] = fd
            actual = d.qpos[7 + self.order]
            vel = d.qvel[6 + self.order]
            raw = kp * gains * (q - actual) + kd * damping * (ff * v - vel)
            tau = np.clip(raw, -self.torque_limits, self.torque_limits)
            d.qfrc_applied[6 + self.order] = tau
            mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            force[:] = 0
            pitch_moments = np.zeros(2)
            cop[:] = 0
            copweight = 0.0
            for k in range(d.ncon):
                contact = d.contact[k]
                b1 = m.geom_bodyid[contact.geom1]
                b2 = m.geom_bodyid[contact.geom2]
                if b1 != 0 and b2 != 0:
                    continue
                body = b2 if b1 == 0 else b1
                mujoco.mj_contactForce(m, d, k, wrench)
                world = contact.frame.reshape(3, 3).T @ wrench[:3]
                if b2 == 0:
                    world = -world
                if body in self.feet:
                    idx = int(np.flatnonzero(self.feet == body)[0])
                    force[idx] += world
                    pitch_moments[int(idx >= 2)] += np.cross(
                        contact.pos - d.subtree_com[self.base], world
                    )[1]
                elif t > 0.15 and contact.dist < 0:
                    bodyhit = True
                    contact_bodies.add(
                        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body)
                    )
                    if first_body_contact is None:
                        first_body_contact = t
                fz = max(0.0, world[2])
                cop += contact.pos * fz
                copweight += fz
            if copweight:
                cop /= copweight
            clearance = d.xpos[self.feet, 2] - FOOT_RADIUS
            drift = np.linalg.norm(d.xpos[self.feet[:2], :2] - anchor, axis=1).max()
            front = (force[:2, 2] >= 0.06 * bw).all()
            rearoff = (force[2:, 2] < 0.015 * bw).all()
            loaded |= force[2:, 2] > 0.03 * bw
            driftmax = max(driftmax, drift)
            if t > 0.15:
                support_samples += 1
                front_samples += int(front)
                frontloss = 0 if front else frontloss + h
                frontlossmax = max(frontlossmax, frontloss)
            if pushstart <= t <= pushend:
                impulse += np.maximum(force[2:, 2], 0) * h
                excess_impulse += np.maximum(force[2:, 2] - 0.25 * bw, 0) * h
            active = pushstart <= t <= pushend + 0.65
            peak = max(peak, float(clearance[2:].min()) if active else 0)
            physical = active and physical_liftoff(
                clearance, force[:, 2], loaded, bw, drift, bodyhit
            )
            hold = hold + h if physical else 0
            besthold = max(besthold, hold)
            hold30 = hold30 + h if physical and clearance[2:].min() >= 0.03 else 0.0
            besthold30 = max(besthold30, hold30)
            if active and front and rearoff and not bodyhit and drift <= 0.04:
                supported_peak = max(supported_peak, float(clearance[2:].min()))
            pitchrate = float(d.cvel[self.base, 1])
            maxpitch = max(maxpitch, abs(pitchrate))
            allspeedmax = max(
                allspeedmax,
                float(np.max(np.abs(d.qvel[6 + self.order]) / self.velocity_limits)),
            )
            allsat += np.abs(raw) >= self.torque_limits
            torquepeak = np.maximum(torquepeak, np.abs(tau))
            if record:
                mujoco.mj_subtreeVel(m, d)
                momentum = d.subtree_angmom[self.base].copy()
                maxmomentum = max(maxmomentum, abs(momentum[1]))
            if pushstart <= t <= pushend:
                sat += np.abs(raw) >= self.torque_limits
                track += (q - d.qpos[7 + self.order]) ** 2
                samples += 1
                speedmax = max(
                    speedmax,
                    float(
                        np.max(np.abs(d.qvel[6 + self.order]) / self.velocity_limits)
                    ),
                )
            if record:
                row = dict(
                    time=t,
                    clearance_min=float(clearance[2:].min()),
                    front_drift=drift,
                    front_contact=bool(front),
                    pitch_rate=pitchrate,
                    pitch_momentum=float(momentum[1]),
                    com_x=float(d.subtree_com[self.base, 0]),
                    com_y=float(d.subtree_com[self.base, 1]),
                    com_z=float(d.subtree_com[self.base, 2]),
                    cop_x=float(cop[0]),
                    cop_y=float(cop[1]),
                    front_pitch_moment_Nm=float(pitch_moments[0]),
                    rear_pitch_moment_Nm=float(pitch_moments[1]),
                    front_mid_x=float(d.xpos[self.feet[:2], 0].mean()),
                    pitch_rad=float(
                        np.arctan2(
                            -d.xmat[self.base].reshape(3, 3)[2, 0],
                            d.xmat[self.base].reshape(3, 3)[0, 0],
                        )
                    ),
                    body_contact=bodyhit,
                )
                for j in range(12):
                    for name, arr in [
                        ("desired", q),
                        ("actual", d.qpos[7 + self.order]),
                        ("torque", tau),
                        ("velocity", d.qvel[6 + self.order]),
                    ]:
                        row[f"{name}_{j}"] = float(arr[j])
                for j in range(4):
                    row[f"force_{j}"] = float(force[j, 2])
                    row[f"clearance_{j}"] = float(clearance[j])
                rows.append(row)
            if viewer_handle is not None:
                if not viewer_handle.is_running():
                    break
                viewer_handle.sync()
                time.sleep(h)
        if viewer_handle is not None:
            viewer_handle.close()
        result = dict(
            seed=seed,
            peak_clearance_m=peak,
            supported_peak_clearance_m=supported_peak,
            max_front_drift_m=driftmax,
            verified_10mm_hold_s=besthold,
            success_10mm=bool(besthold >= 0.02 and driftmax <= 0.04 and not bodyhit),
            verified_30mm_hold_s=besthold30,
            success_30mm=bool(besthold30 >= 0.02 and driftmax <= 0.04 and not bodyhit),
            front_contact_loss_max_s=frontlossmax,
            body_contact=bodyhit,
            nonfoot_contact_bodies=sorted(contact_bodies),
            first_nonfoot_contact_s=first_body_contact,
            peak_pitch_rate_rad_s=maxpitch,
            peak_pitch_momentum_Nms=maxmomentum,
            rear_push_impulse_Ns=impulse.tolist(),
            rear_excess_push_impulse_Ns=excess_impulse.tolist(),
            front_contact_retention=front_samples / max(1, support_samples),
            peak_torque_Nm=torquepeak.tolist(),
            saturation_fraction=(allsat / round(seconds / h)).tolist(),
            peak_velocity_limit_ratio=allspeedmax,
            body_weight_N=bw,
            friction=float(m.geom_friction[0, 0]),
            push_tracking_rmse_rad=np.sqrt(track / max(1, samples)).tolist(),
            push_saturation_fraction=(sat / max(1, samples)).tolist(),
            push_peak_velocity_limit_ratio=speedmax,
        )
        return result, rows


def score(results):
    return float(
        np.mean(
            [
                min(r["supported_peak_clearance_m"], 0.15)
                - 4 * max(0, r["max_front_drift_m"] - 0.035)
                - 2 * max(0, r["front_contact_loss_max_s"] - 0.04)
                - 0.5 * r["body_contact"]
                for r in results
            ]
        )
    )


def robust_score(results):
    scores = [
        min(r["supported_peak_clearance_m"], 0.06)
        - 5 * max(0, r["max_front_drift_m"] - 0.03)
        - 2 * max(0, r["front_contact_loss_max_s"] - 0.02)
        - 0.7 * r["body_contact"]
        - 0.1 * max(0, r["push_peak_velocity_limit_ratio"] - 1)
        for r in results
    ]
    return float(min(scores) + 0.02 * np.mean(scores))


def sample(rng, best=None):
    if best is not None:
        values = best.to_dict()
        bounds = {
            "extend": (0.07, 0.24),
            "crouch": (0.25, 0.65),
            "rear_crouch_hip": (0.8, 1.6),
            "rear_crouch_knee": (-2.65, -1.8),
            "rear_push_hip": (-0.3, 0.8),
            "rear_push_knee": (-1.35, -0.85),
            "front_crouch_hip": (0.45, 1.5),
            "front_crouch_knee": (-2.3, -1.0),
            "front_push_hip": (0.4, 1.8),
            "front_push_knee": (-2.6, -1.0),
            "front_gain": (0.1, 1.0),
            "front_damping": (0.4, 1.5),
            "hip_delay": (-0.25, 0.25),
            "knee_delay": (-0.25, 0.25),
            "hip_duration": (0.7, 1.3),
            "knee_duration": (0.7, 1.3),
            "tuck_delay": (0, 0.15),
            "tuck_duration": (0.12, 0.35),
            "rear_tuck_hip": (0.6, 1.7),
            "rear_tuck_knee": (-2.6, -1.5),
        }
        for key, (lo, hi) in bounds.items():
            values[key] = float(
                np.clip(values[key] + rng.normal(0, (hi - lo) * 0.13), lo, hi)
            )
        return KickReference(**values)
    return KickReference(
        crouch=rng.uniform(0.3, 0.6),
        extend=rng.uniform(0.08, 0.20),
        rear_crouch_hip=rng.uniform(0.9, 1.5),
        rear_crouch_knee=rng.uniform(-2.6, -2.0),
        rear_push_hip=rng.uniform(-0.2, 0.6),
        rear_push_knee=rng.uniform(-1.15, -0.85),
        front_crouch_hip=rng.uniform(0.6, 1.3),
        front_crouch_knee=rng.uniform(-2.2, -1.2),
        front_push_hip=rng.uniform(0.6, 1.6),
        front_push_knee=rng.uniform(-2.4, -1.2),
        front_gain=rng.uniform(0.15, 0.8),
        front_damping=rng.uniform(0.6, 1.3),
        hip_delay=rng.uniform(-0.2, 0.2),
        knee_delay=rng.uniform(-0.2, 0.2),
        rear_tuck_hip=rng.uniform(0.7, 1.5),
        rear_tuck_knee=rng.uniform(-2.5, -1.7),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--output-dir", default="output/reference_search")
    parser.add_argument("--parameters")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--playback-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--validation-seed", type=int, default=1000)
    parser.add_argument("--search-seed", type=int, default=482)
    parser.add_argument("--robust", action="store_true")
    parser.add_argument("--initial-parameters", nargs="*", default=[])
    args = parser.parse_args()
    if args.trials < 1 or args.candidates < 1 or args.repeats < 1:
        parser.error("trials, candidates and repeats must be positive")
    if args.playback_only and not args.parameters:
        parser.error("--playback-only requires --parameters")
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ex = Experiment()
    (dest / "provenance.json").write_text(
        json.dumps(dict(arguments=vars(args), **ex.provenance()), indent=2)
    )
    rng = np.random.default_rng(args.search_seed)
    history = []
    best = None
    bestscore = -np.inf
    if args.parameters:
        best = KickReference(**json.loads(Path(args.parameters).read_text()))
    else:
        for i in range(args.candidates):
            if i < len(args.initial_parameters):
                p = KickReference(
                    **json.loads(Path(args.initial_parameters[i]).read_text())
                )
            elif args.robust and best is not None:
                p = sample(rng, best)
            else:
                p = (
                    KickReference()
                    if i == 0
                    else sample(rng, best if i > 20 and rng.random() < 0.8 else None)
                )
            results = [
                ex.trial(p, seed=s)[0]
                for s in (range(200, 206) if args.robust else (100, 101))
            ]
            value = robust_score(results) if args.robust else score(results)
            history.append(
                dict(candidate=i, score=value, parameters=p.to_dict(), results=results)
            )
            if value > bestscore:
                best, bestscore = p, value
                (dest / "best_parameters.json").write_text(
                    json.dumps(p.to_dict(), indent=2)
                )
                print(
                    i,
                    round(value, 4),
                    [
                        (
                            round(r["supported_peak_clearance_m"] * 1000, 1),
                            round(r["max_front_drift_m"] * 100, 1),
                            r["success_10mm"],
                        )
                        for r in results
                    ],
                    flush=True,
                )
            if i % 20 == 0:
                (dest / "search.json").write_text(json.dumps(history, indent=2))
        (dest / "search.json").write_text(json.dumps(history, indent=2))
    if args.playback_only:
        if not args.parameters:
            parser.error("--playback-only requires --parameters")
        for _ in range(args.repeats):
            ex.trial(best, seed=1000, viewer=args.viewer)
        return
    (dest / "reference_parameters.json").write_text(
        json.dumps(best.to_dict(), indent=2)
    )
    report = {}
    for label, baseline, gain in [
        ("baseline", True, None),
        ("optimized", False, None),
        ("stiff_front", False, 1.0),
        ("soft_front", False, 0.1),
        ("fixed_front", False, None),
        ("no_feedforward", False, None),
        ("no_tuck", False, None),
    ]:
        report[label] = []
        for seed in range(args.validation_seed, args.validation_seed + args.trials):
            candidate = best
            if label == "fixed_front":
                candidate = replace(
                    best,
                    front_crouch_hip=float(ex.standing[1]),
                    front_push_hip=float(ex.standing[1]),
                    front_crouch_knee=float(ex.standing[2]),
                    front_push_knee=float(ex.standing[2]),
                )
            elif label == "no_feedforward":
                candidate = replace(best, feedforward=0.0)
            elif label == "no_tuck":
                candidate = replace(best, tuck=False)
            r, rows = ex.trial(
                candidate,
                seed=seed,
                baseline=baseline,
                record=True,
                viewer=args.viewer
                and label == "optimized"
                and seed == args.validation_seed,
                front_gain_override=gain,
            )
            report[label].append(r)
            with (dest / f"{label}_{seed}.csv").open("w") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
        print(
            label,
            "success",
            sum(r["success_10mm"] for r in report[label]),
            "clearance",
            np.mean([r["supported_peak_clearance_m"] for r in report[label]]),
            "drift",
            np.mean([r["max_front_drift_m"] for r in report[label]]),
            flush=True,
        )
    (dest / "validation.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
