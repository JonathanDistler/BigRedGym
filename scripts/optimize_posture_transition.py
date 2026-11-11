"""CPU MuJoCo search for an opt-in reference bridge between frozen actors.

All starts are replayed from standing kick trajectories. No PPO updates are
performed, and all experiment directories must be new.
"""

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import shutil
from types import MethodType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from gym.envs.go2.kickup_reference import KickReference
from gym.envs.go2.posture_transition import (
    BOUNDS,
    BoundedCMA,
    PostureTransition,
    smooth,
    rate_limited_target,
)
from scripts.evaluate_kickup_handoff import (
    Policy,
    Experiment,
    matched_reset,
    measurements,
    feature,
    states,
    restore,
    dump,
    wilson,
)
from scripts.reproduce_transition_hold import KICK, BALANCE, REFERENCE


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bank(policy, ex, seeds, source="original"):
    """Standing prefix on unchanged native physics; sample at every control step."""
    e = policy.env
    policy.reset()
    initial_reference = e.reference_parameters
    if source == "aggressive":
        e.reference_parameters = KickReference(
            **json.loads(Path(REFERENCE).read_text())
        )
    matched_reset(e, ex, seeds)
    seen = np.zeros(len(seeds), dtype=bool)
    peak = np.zeros(len(seeds))
    samples = {}
    post = e._post_physics_step

    def audit():
        post()
        for i, r in enumerate(measurements(e)):
            seen[i] |= r["body_contact"]
            peak[i] = max(peak[i], r["pitch_rad"])

    e._post_physics_step = audit
    try:
        for step in range(round(1.4 / e.dt)):
            policy.action()
            e.step()
            q, v = e._kick_reference(e._kick_clock + e._physics_offset)
            t = round((step + 1) * e.dt, 5)
            if t >= 0.01:
                samples[t] = dict(
                    states=states(e),
                    targets=(q + e.dof_pos_target).clone(),
                    velocity=v.clone(),
                    collision=seen.copy(),
                    peak=peak.copy(),
                    warmstart=[d.qacc_warmstart.copy() for d in e._backend._datas],
                )
    finally:
        e._post_physics_step = post
        e.reference_parameters = initial_reference
    return dict(
        seeds=seeds,
        samples=samples,
        source=source,
        models=[copy.copy(e._backend._model_for_env(i)) for i in range(len(seeds))],
        kp=e.p_gains.clone(),
        kd=e.d_gains.clone(),
        front_gain=initial_reference.front_gain if source == "original" else 0.9,
        front_damping=initial_reference.front_damping,
        standing=ex.standing.copy(),
    )


def install(balance, data, indices, t):
    e = balance.env
    balance.reset()
    b = e._backend
    b._models = [copy.copy(data["models"][i]) for i in indices]
    b._mjm = b._models[0]
    b._datas = [mujoco.MjData(m) for m in b._models]
    for i, m in enumerate(b._models):
        b._link_mass_t[i] = torch.tensor(
            m.body_mass[b._canonical_to_native_body_np], dtype=torch.float32
        )
        b._contact_friction_t[i] = float(m.geom_friction[0, 0])
    sample = data["samples"][t]
    restore(e, [sample["states"][i] for i in indices])
    for j, i in enumerate(indices):
        b._datas[j].qacc_warmstart[:] = sample["warmstart"][i]
    e.handstand_stage[:] = 2
    start = sample["targets"][indices].clone()
    e.dof_pos_target[:] = start - e._standing_posture()
    e.dof_pos_history[:] = e.dof_pos_target.repeat(1, 3)
    return sample, start


class Evaluator:
    def __init__(
        self,
        balance,
        atlas,
        late_bridge=False,
        reference_first=False,
        limit_target_rate=True,
    ):
        self.balance = balance
        self.late_bridge = late_bridge
        self.reference_first = reference_first
        self.limit_target_rate = limit_target_rate
        self.centers = np.array(
            [feature(r["initial"]) for a in atlas for r in a["trials"] if r["success"]]
        )
        if not len(self.centers):
            raise ValueError("Atlas contains no successful recovery states")
        self.source = KickReference(**json.loads(Path(REFERENCE).read_text()))

    def evaluate(
        self,
        parameters,
        data,
        indices,
        trace_path=None,
        viewer=False,
        view_slot=0,
        state_callback=None,
    ):
        p = parameters
        e = self.balance.env
        n = len(indices)
        assert e.num_envs == n
        t0 = min(data["samples"], key=lambda t: abs(t - p.takeover_s))
        sample, initial_target = install(self.balance, data, indices, t0)
        failed = sample["collision"][indices].copy()
        collisions = failed.copy()
        maximum_pitch = sample["peak"][indices].copy()
        weight = e._backend.link_mass.sum(1).numpy() * 9.81
        support_steps = np.zeros(n)
        active_steps = np.zeros(n)
        longest = np.zeros(n)
        stable_duration = np.zeros(n)
        hold = np.zeros(n)
        torque_sat = np.zeros(n)
        speed_ratio = np.zeros(n)
        jump = np.zeros(n)
        first_contact = [None] * n
        completed_at = [None] * n
        handoff = [None] * n
        nearest = np.full(n, np.inf)
        recorded = []
        last_target = initial_target.clone()
        policy_target = initial_target.clone()
        balance_kp = e.p_gains.clone()
        balance_kd = e.d_gains.clone()
        source_kp = data["kp"][indices]
        source_kd = data["kd"][indices]
        limits = e.dof_pos_limits[e.actuated_dof_indices]
        velocity_limits = e.dof_vel_limits[e.actuated_dof_indices]
        current_t = t0
        last_applied = initial_target.clone()
        cpg_started = False
        com_offset = torch.tensor([r["com_support_x"] for r in measurements(e)])
        policy_start = p.switch_s + (p.reference_s if self.reference_first else 0.0)
        handoff_time = policy_start + p.blend_s
        old_compute, old_post = e._compute_torques, e._post_physics_step
        physics_count = 0
        render = None
        if viewer:
            import mujoco.viewer
            import time

            viz_m = data["models"][indices[view_slot]]
            viz_d = mujoco.MjData(viz_m)
            handle = mujoco.viewer.launch_passive(viz_m, viz_d)
            handle.cam.distance = 1.5
            handle.cam.azimuth = 90

            def render(qpos):
                viz_d.qpos[:] = qpos
                mujoco.mj_forward(viz_m, viz_d)
                handle.sync()
                time.sleep(e.dt)

            # Playback the actual standing prefix before evaluating its continuation.
            for t, prefix in data["samples"].items():
                if t <= t0:
                    render(prefix["states"][indices[view_slot]][0])

        def compute(this):
            nonlocal last_applied
            local_t = (
                current_t
                + physics_count % this.cfg.control.decimation * this.cfg.sim_dt
            )
            q, v = p.reference(local_t, self.source, data["standing"])
            ref = torch.tensor(q, dtype=torch.float32).expand(n, -1).clone()
            rate = torch.tensor(v, dtype=torch.float32).expand(n, -1).clone()
            entry = float(smooth(local_t, t0, 0.06))
            # A short entry blend starts at the kick's actual target, in radians.
            ref = initial_target.lerp(ref, entry)
            rate = data["samples"][t0]["velocity"][indices].lerp(rate, entry)
            mix = float(smooth(local_t, policy_start, p.blend_s))
            if self.late_bridge:
                ref = initial_target.clone()
                for joints, offset in (
                    ([1, 4], p.front_hip_delta),
                    ([2, 5], p.front_knee_delta),
                    ([7, 10], p.rear_hip_delta),
                    ([8, 11], p.rear_knee_delta),
                ):
                    ref[:, joints] += 4 * mix * (1 - mix) * offset
                rate.zero_()
            if self.reference_first:
                goal = this._standing_posture().clone()
                for joints, offset in (
                    ([1, 4], p.front_hip_delta),
                    ([2, 5], p.front_knee_delta),
                    ([7, 10], p.rear_hip_delta),
                    ([8, 11], p.rear_knee_delta),
                ):
                    goal[:, joints] += offset
                ref = initial_target.lerp(
                    goal, float(smooth(local_t, t0, p.reference_s))
                )
                rate.zero_()
            # Small orientation feedback is active only in the intermediate phase.
            err = np.pi / 2 - torch.atan2(
                this.projected_gravity[:, 0], -this.projected_gravity[:, 2]
            )
            correction = (
                p.pitch_gain * err
                + p.pitch_rate_gain * this.base_ang_vel[:, 1]
                + p.com_gain * com_offset
            ).clamp(-0.25, 0.25)
            ref[:, [1, 4]] += correction[:, None] * entry * (1 - mix)
            desired = ref.lerp(policy_target, mix).clamp(limits[:, 0], limits[:, 1])
            if self.limit_target_rate:
                desired = rate_limited_target(
                    desired, last_target, velocity_limits, this.cfg.sim_dt
                )
            rate = ((1 - mix) * rate).clamp(-velocity_limits, velocity_limits)
            delta = (desired - last_target).abs().amax(1).numpy()
            jump[:] = np.maximum(jump, delta)
            last_target.copy_(desired)
            last_applied = desired
            this.dof_pos_target[:] = desired - this._standing_posture()
            gain = torch.ones_like(desired)
            gain[:, :6] = data["front_gain"] * (1 - entry) + p.front_gain * entry
            damping = torch.ones_like(desired)
            damping[:, :6] = (
                data["front_damping"] * (1 - entry) + p.front_damping * entry
            )
            kp = (source_kp * gain).lerp(balance_kp, mix)
            kd = (source_kd * damping).lerp(balance_kd, mix)
            if self.late_bridge:
                kp, kd = balance_kp, balance_kd
            raw = kp * (desired - this.dof_pos[:, this.actuated_dof_indices])
            raw += kd * (rate - this.dof_vel[:, this.actuated_dof_indices])
            raw += this.tau_ff
            torque_sat[:] += (
                (raw.abs() >= this.actuated_torque_limits).float().mean(1).numpy()
            )
            return raw.clamp(-this.actuated_torque_limits, this.actuated_torque_limits)

        def post(this):
            nonlocal physics_count
            old_post()
            physics_count += 1
            # Audit every physics substep; a collision is never erased by reset.
            for j, d in enumerate(this._backend._datas):
                m = this._backend._model_for_env(j)
                feet = this._backend._canonical_to_native_body_np[
                    this.feet_indices.numpy()
                ]
                hit = any(
                    c.dist < -1e-5
                    and (
                        (
                            m.geom_bodyid[c.geom1] == 0
                            and m.geom_bodyid[c.geom2] not in feet
                        )
                        or (
                            m.geom_bodyid[c.geom2] == 0
                            and m.geom_bodyid[c.geom1] not in feet
                        )
                    )
                    for c in d.contact
                )
                if hit and not collisions[j]:
                    first_contact[j] = physics_count * this.cfg.sim_dt
                collisions[j] |= hit
                failed[j] |= hit
            speed_ratio[:] = np.maximum(
                speed_ratio,
                (this.dof_vel[:, this.actuated_dof_indices].abs() / velocity_limits)
                .amax(1)
                .numpy(),
            )

        e._compute_torques = MethodType(compute, e)
        e._post_physics_step = MethodType(post, e)
        end = handoff_time + 6.5
        steps = 0
        try:
            while current_t < end - 1e-7:
                if current_t >= policy_start and not cpg_started:
                    e.leg_phase[:] = e._cpg_offsets
                    e._cpg_load.zero_()
                    e._update_cpg_obs()
                    cpg_started = True
                if cpg_started:
                    self.balance.action()
                    policy_target.copy_(e._standing_posture() + e.dof_pos_target)
                e.step()
                steps += 1
                current_t = t0 + steps * e.dt
                rows = measurements(e)
                com_offset[:] = torch.tensor([r["com_support_x"] for r in rows])
                for j, r in enumerate(rows):
                    if not failed[j]:
                        maximum_pitch[j] = max(maximum_pitch[j], r["pitch_rad"])
                    failed[j] |= bool(e.terminated[j])
                    supported = min(r["force_0"], r["force_1"]) >= 0.06 * weight[j]
                    support_steps[j] += supported and not failed[j]
                    active_steps[j] += not failed[j]
                    if not failed[j] and p.switch_s <= current_t <= handoff_time + 0.02:
                        distance = float(
                            np.linalg.norm(self.centers - feature(r), axis=1).min()
                        )
                        nearest[j] = min(nearest[j], distance)
                    if (
                        not failed[j]
                        and handoff[j] is None
                        and current_t >= handoff_time
                    ):
                        handoff[j] = dict(time=current_t, **r)
                    valid = (
                        current_t >= handoff_time
                        and abs(r["pitch_rad"] - np.pi / 2) <= np.deg2rad(15)
                        and abs(r["pitch_rate"]) <= 1.5
                        and supported
                        and max(r["force_2"], r["force_3"]) <= 0.015 * weight[j]
                        and min(r["clearance_2"], r["clearance_3"]) >= 0.2
                        and not failed[j]
                    )
                    stable_duration[j] += e.dt if valid else 0.0
                    hold[j] = hold[j] + e.dt if valid else 0.0
                    longest[j] = max(longest[j], hold[j])
                    if completed_at[j] is None and hold[j] >= 5 - 1e-6:
                        completed_at[j] = current_t
                    if trace_path:
                        r.update(
                            seed=data["seeds"][indices[j]],
                            time=current_t,
                            hold_s=hold[j],
                            collision=bool(collisions[j]),
                            terminated=bool(e.terminated[j]),
                            timed_out=bool(e.timed_out[j]),
                        )
                        for joint in range(12):
                            r[f"target_{joint}"] = float(last_applied[j, joint])
                            r[f"policy_target_{joint}"] = float(policy_target[j, joint])
                            r[f"torque_{joint}"] = float(e.torques[j, joint])
                        recorded.append(r)
                if state_callback is not None:
                    state_callback(e, current_t, handoff_time, failed.copy())
                if render:
                    render(e._backend._datas[view_slot].qpos)
                if failed.all():
                    break
        finally:
            e._compute_torques, e._post_physics_step = old_compute, old_post
            if viewer:
                handle.close()
        results = [
            dict(
                seed=data["seeds"][i],
                success=completed_at[j] is not None,
                completed_at=completed_at[j],
                post_success_collision=bool(
                    completed_at[j] is not None and collisions[j]
                ),
                hold_s=float(longest[j]),
                total_stable_duration_s=float(stable_duration[j]),
                collision=bool(collisions[j]),
                first_contact_relative_s=first_contact[j],
                max_pitch_deg=float(np.rad2deg(maximum_pitch[j])),
                handoff=handoff[j],
                support_fraction=float(support_steps[j] / max(steps, 1)),
                alive_support_fraction=float(
                    support_steps[j] / max(active_steps[j], 1)
                ),
                torque_saturation_fraction=float(torque_sat[j] / max(physics_count, 1)),
                max_velocity_limit_ratio=float(speed_ratio[j]),
                max_target_step_rad=float(jump[j]),
                atlas_distance=float(min(nearest[j], 100.0)),
                timed_out=bool(e.timed_out[j]),
                failed=bool(failed[j]),
            )
            for j, i in enumerate(indices)
        ]
        successes = sum(r["success"] for r in results)
        # Complete recovery dominates. Collision and speed are feasibility costs.
        score = (
            1000 * successes / n
            + 6 * float(np.mean(longest.clip(0, 5)))
            - 8 * float(collisions.mean())
            - 0.4 * float(np.mean(np.minimum(nearest, 30.0)))
            - 2 * float(np.maximum(speed_ratio - 1.0, 0).mean())
        )
        report = dict(
            parameters=p.to_dict(),
            source=data["source"],
            score=score,
            successes=successes,
            count=n,
            wilson95=wilson(successes, n),
            trials=results,
            simulated_s=steps * e.dt,
            actual_takeover_s=t0,
            hold_starts_after_blend=True,
            limit_target_rate=self.limit_target_rate,
        )
        if trace_path:
            with trace_path.open("w") as f:
                w = csv.DictWriter(f, fieldnames=recorded[0])
                w.writeheader()
                w.writerows(recorded)
        return report


def encode(p, bounds=BOUNDS):
    return np.array(
        [(getattr(p, k) - lo) / (hi - lo) for k, (lo, hi) in bounds.items()]
    )


def decode(x, bounds=BOUNDS, late_bridge=False):
    values = {
        k: float(lo + value * (hi - lo))
        for (k, (lo, hi)), value in zip(bounds.items(), x)
    }
    if late_bridge:
        values["takeover_s"] = values["switch_s"]
    return PostureTransition(**values)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--initial-parameters", type=Path)
    parser.add_argument("--sigma", type=float, default=0.14)
    parser.add_argument("--validation-seed", type=int, default=6000)
    parser.add_argument("--heldout-seed", type=int, default=7000)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--view-slot", type=int, default=0)
    parser.add_argument(
        "--legacy-target-steps",
        action="store_true",
        help="Reproduce historical unslewed policy commands",
    )
    parser.add_argument("--seed", type=int, default=3009)
    parser.add_argument("--late-bridge", action="store_true")
    parser.add_argument("--reference-first", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument(
        "--replay-screen",
        action="store_true",
        help="Replay the 16-world kick / four-world screening layout",
    )
    parser.add_argument("--replay-validation", action="store_true")
    parser.add_argument("--replay-heldout", action="store_true")
    parser.add_argument(
        "--source", choices=("original", "aggressive"), default="original"
    )
    args = parser.parse_args()
    if not 0 < args.sigma <= 0.5:
        parser.error("CMA sigma must be in (0, .5]")
    if args.reference_first and not args.late_bridge:
        parser.error("--reference-first requires --late-bridge")
    replay_count = sum(
        (args.replay_screen, args.replay_validation, args.replay_heldout)
    )
    if replay_count > 1 or (replay_count and not args.parameters):
        parser.error("Select one replay layout and supply --parameters")
    if args.generations < 0 or args.population < 4:
        parser.error(
            "Generations must be nonnegative; population must be at least four"
        )
    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=False)
    snapshot = dest / "source"
    snapshot.mkdir()
    for path in (Path(__file__), Path("gym/envs/go2/posture_transition.py")):
        shutil.copy2(path, snapshot / path.name)
    torch.set_num_threads(1)
    protected = [
        KICK,
        BALANCE,
        REFERENCE,
        "gym/envs/go2/references/kickup_front_yield.json",
        "output/handoff_cpg_comparison/atlas.json",
    ]
    hashes = {p: digest(p) for p in protected}
    atlas = json.loads(Path(protected[-1]).read_text())
    atlas_provenance = json.loads(
        (Path(protected[-1]).parent / "provenance.json").read_text()
    )
    if atlas_provenance["balance"]["sha256"] != hashes[BALANCE]:
        raise ValueError("The recovery atlas belongs to a different balance checkpoint")
    with torch.inference_mode():
        n = (
            (4 if args.replay_screen else 16 if replay_count else 1)
            if args.parameters
            else 16
        )
        if not 0 <= args.view_slot < n:
            parser.error("Viewer slot is outside the batch")
        full = Policy(BALANCE, "go2balance", n, True)
        kick_policy = Policy(KICK, "go2frontyield", 16 if args.replay_screen else n)
        ex = Experiment()
        dump(
            dest / "provenance.json",
            dict(
                balance=full.meta,
                kick=kick_policy.meta,
                protected_sha256=hashes,
                mujoco_version=mujoco.__version__,
                optimizer="bounded full-covariance CMA-ES",
                args=vars(args),
                training=False,
            ),
        )
        if args.parameters:
            p = PostureTransition(**json.loads(args.parameters.read_text()))
            start_seed = (
                3000
                if args.replay_screen
                else args.validation_seed
                if args.replay_validation
                else args.heldout_seed
            )
            seeds = (
                list(range(start_seed, start_seed + 16))
                if replay_count
                else [args.seed]
            )
            indices = [0, 4, 9, 12] if args.replay_screen else list(range(n))
            data = bank(kick_policy, ex, seeds, args.source)
            result = Evaluator(
                full,
                atlas,
                args.late_bridge,
                args.reference_first,
                not args.legacy_target_steps,
            ).evaluate(
                p, data, indices, dest / "playback.csv", args.viewer, args.view_slot
            )
            dump(dest / "playback.json", result)
        else:
            data = bank(kick_policy, ex, list(range(3000, 3016)), args.source)
            screening = Policy(BALANCE, "go2balance", 4, True)
            ev = Evaluator(
                screening,
                atlas,
                args.late_bridge,
                args.reference_first,
                not args.legacy_target_steps,
            )
            indices = [0, 4, 9, 12]
            history = []
            best = -np.inf
            bounds = (
                dict(
                    front_hip_delta=(-0.35, 0.35),
                    front_knee_delta=(-0.3, 0.3),
                    rear_hip_delta=(-0.6, 0.6),
                    rear_knee_delta=(-0.4, 0.4),
                    switch_s=(1.05, 1.32),
                    blend_s=(0.35, 1.1),
                )
                if args.late_bridge
                else BOUNDS
            )
            if args.reference_first:
                bounds["blend_s"] = (0.10, 0.4)
                bounds["reference_s"] = (0.25, 0.85)
            if args.adaptive:
                bounds.update(
                    pitch_gain=(-1.0, 1.0),
                    pitch_rate_gain=(-0.35, 0.35),
                    com_gain=(-3.0, 3.0),
                )
            start = (
                PostureTransition(takeover_s=1.2)
                if args.late_bridge
                else PostureTransition()
            )
            cma = BoundedCMA(
                encode(start, bounds), population=args.population, sigma=args.sigma
            )
            warm = (
                [PostureTransition(takeover_s=t, switch_s=t) for t in (1.1, 1.2, 1.25)]
                if args.late_bridge
                else [PostureTransition(takeover_s=t) for t in (0.45, 0.60, 0.70, 0.78)]
            )
            candidates = [encode(p, bounds) for p in warm]
            if args.reference_first:
                candidates = [
                    encode(
                        PostureTransition(takeover_s=t, switch_s=t, blend_s=0.2), bounds
                    )
                    for t in (1.05, 1.1, 1.2)
                ]
            if args.adaptive:
                candidates = [
                    encode(
                        PostureTransition(
                            takeover_s=1.2,
                            switch_s=1.2,
                            blend_s=0.2 if args.reference_first else 0.6,
                            pitch_gain=kp,
                            pitch_rate_gain=kd,
                        ),
                        bounds,
                    )
                    for kp, kd in ((0.0, 0.0), (-0.3, 0.1), (-0.6, 0.2), (0.3, -0.1))
                ]
            if args.initial_parameters:
                initial = PostureTransition(
                    **json.loads(args.initial_parameters.read_text())
                )
                candidates = [encode(initial, bounds)]
            for generation in range(args.generations + 1):
                if generation:
                    candidates = cma.ask()
                costs = []
                for x in candidates:
                    p = decode(x, bounds, args.late_bridge)
                    result = ev.evaluate(p, data, indices)
                    result["candidate"] = len(history)
                    result["generation"] = generation
                    history.append(result)
                    costs.append(-result["score"])
                    if result["score"] > best:
                        best = result["score"]
                        dump(dest / "screen_best_parameters.json", p.to_dict())
                        holds = [round(r["hold_s"], 2) for r in result["trials"]]
                        print(
                            f"best {len(history) - 1}: {best:.3f}, "
                            f"success {result['successes']}/4, "
                            f"holds {holds}",
                            flush=True,
                        )
                    dump(dest / "search.json", history)
                if generation:
                    cma.tell(candidates, costs)
                else:
                    best_warm = max(history, key=lambda r: r["score"])
                    cma = BoundedCMA(
                        encode(PostureTransition(**best_warm["parameters"]), bounds),
                        population=args.population,
                        sigma=args.sigma,
                    )
                print(f"generation {generation} complete", flush=True)
            screening.env._backend.close()
            # The two finalists each receive 16 NEW starts, not just screening seeds.
            validation_data = bank(
                kick_policy,
                ex,
                list(range(args.validation_seed, args.validation_seed + 16)),
                args.source,
            )
            finalists = sorted(history, key=lambda r: r["score"], reverse=True)[:2]
            validation = []
            evaluator = Evaluator(
                full,
                atlas,
                args.late_bridge,
                args.reference_first,
                not args.legacy_target_steps,
            )
            for finalist in finalists:
                p = PostureTransition(**finalist["parameters"])
                r = evaluator.evaluate(
                    p,
                    validation_data,
                    list(range(16)),
                    dest / f"validation_{finalist['candidate']}.csv",
                )
                r["candidate"] = finalist["candidate"]
                validation.append(r)
                print(f"validation {r['candidate']}: {r['successes']}/16", flush=True)
            selected = max(validation, key=lambda r: r["score"])
            p = PostureTransition(**selected["parameters"])
            dump(dest / "selected_parameters.json", p.to_dict())
            dump(dest / "validation.json", validation)
            heldout_data = bank(
                kick_policy,
                ex,
                list(range(args.heldout_seed, args.heldout_seed + 16)),
                args.source,
            )
            heldout = evaluator.evaluate(
                p, heldout_data, list(range(16)), dest / "heldout.csv"
            )
            dump(dest / "heldout.json", heldout)
            print(f"held-out: {heldout['successes']}/16", flush=True)
        for path, value in hashes.items():
            assert digest(path) == value, f"Protected input changed: {path}"
        full.env._backend.close()
        kick_policy.env._backend.close()


if __name__ == "__main__":
    main()
