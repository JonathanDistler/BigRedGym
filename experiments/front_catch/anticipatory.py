"""Additive anticipatory objectives; native contacts/actions and old A stay intact."""

import mujoco
import numpy as np
from experiments.front_catch.core import CatchEnvironment, CatchPhase, useful_swing_state


class AnticipatoryPhase(CatchPhase):
    """Phase 3 starts on certified liftoff, never on a detected pitch peak."""

    def update(self, time, loaded, takeoff, pitch, stable):
        super().update(time, loaded, takeoff, pitch, stable)
        if takeoff:
            self.progress = max(self.progress, 0.4)
        # Approach/catch still requires physical support/orientation events.
        if takeoff and self.current_support and pitch >= np.deg2rad(60):
            self.progress = max(self.progress, 0.6)


def placement_prediction(com, velocity, feet, origin, heading, gravity, cfg):
    """Short-horizon support forecast, NOT a certified dynamic capture basin.

    Bounded ballistic height and capture-time estimate; compare present and original
    feet against exactly the same forecast so COM motion alone cannot earn credit.
    Per-foot scoring allows a staggered step or loaded slide, not just two-foot moves.
    """
    tau = cfg.anticipation_seconds
    future = com + tau * velocity
    future[2] -= 0.5 * gravity * tau**2
    height = max(future[2] - 0.022, 0.05)
    landing = future + np.sqrt(height / gravity) * velocity
    target = float(landing @ heading)
    now = feet @ heading
    old = origin @ heading
    # Only request backward travel where the forecast actually lies behind a foot.
    request = np.clip(old - target, 0, cfg.backward_scale)
    desired = old - request
    improvement = np.clip((abs(old - desired) - abs(now - desired)) / cfg.backward_scale, 0, 1)
    distance = np.min(np.linalg.norm(feet[:, :2] - landing[:2], axis=1))
    recoverability = float(np.exp(-0.5 * (distance / cfg.anticipation_support_sigma)**2))
    return float(np.mean(improvement) * recoverability), landing, request, recoverability


def objective_gate(s, row, cs, started, safe, start_pitch, peak_pitch, progress):
    """Online rising-swing gate; never uses a future peak or pays after descent."""
    return bool(
        started and safe and cs['any_front_loaded'] and cs['impact_valid']
        and cs['head_clearance'] > 0 and s['calf_clearance'] > 0
        and useful_swing_state(s, row, start_pitch, peak_pitch)
        and s['pitch_velocity'] > 0.2 and s['com_velocity_z'] > 0
        and 0.4 <= progress < 0.8
    )


class AnticipatoryEnvironment(CatchEnvironment):
    def reset(self, *args, **kwargs):
        self.early_paid = self.sweep_paid = 0.0
        self.sweep_origin_inertia = None
        self.first_front_motion = self.first_rear_sweep = None
        self.peak_pitch_time = 0.0
        self.timing_peak_pitch = -np.inf
        result = super().reset(*args, **kwargs)
        if self.cfg.catch_mode != 'A':
            self.phase = AnticipatoryPhase(self.cfg.phase_times)
        return result

    def restore_start(self, state, kind):
        result = super().restore_start(state, kind)
        # Old banks contain old phase objects. Adapt event state without inventing events.
        if self.cfg.catch_mode != 'A':
            phase = AnticipatoryPhase(self.cfg.phase_times)
            phase.__dict__.update(self.phase.__dict__)
            self.phase = phase
        self.early_paid = self.sweep_paid = 0.0
        self.sweep_origin_inertia = None
        self.first_front_motion = self.first_rear_sweep = None
        self.peak_pitch_time = float(self.d.time)
        self.timing_peak_pitch = self.start_pitch
        return result

    def rear_motion(self):
        """Rear-link momentum relative to rigid-body trunk motion about whole COM.

        Negative relative rear momentum can redistribute rotation into the trunk;
        it does not create net angular momentum. Includes each link inertia/mass.
        """
        com = self.d.subtree_com[self.base]
        vcom = self.d.subtree_linvel[self.base]
        body_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY,
                                self.base, body_velocity, 0)
        omega = body_velocity[:3]
        relative = 0.0
        for body in self.robot_ids:
            if not self.m.body(int(body)).name.startswith(('RL_', 'RR_')):
                continue
            jp = np.zeros((3, self.m.nv))
            jr = np.zeros((3, self.m.nv))
            mujoco.mj_jacBodyCom(self.m, self.d, jp, jr, int(body))
            velocity = np.r_[jr @ self.d.qvel, jp @ self.d.qvel]
            r = self.d.xipos[body] - com
            rot = self.d.ximat[body].reshape(3, 3)
            spin = rot @ (self.m.body_inertia[body] * (rot.T @ (velocity[:3] - omega)))
            orbital = np.cross(r, self.m.body_mass[body] *
                               (velocity[3:] - vcom - np.cross(omega, r)))
            relative += float((spin + orbital) @ self.pitch_axis)
        rates = self.d.qvel[self.vids]
        hips = [i for i, n in enumerate(self.names) if n.startswith(('RL_', 'RR_')) and 'thigh' in n]
        knees = [i for i, n in enumerate(self.names) if n.startswith(('RL_', 'RR_')) and 'calf' in n]
        if len(hips) != 2 or len(knees) != 2:
            raise ValueError('Expected bilateral rear thigh/hip and calf/knee joints')
        h, k = rates[hips], rates[knees]
        coordination = float(h[0]*h[1] > 0 and k[0]*k[1] > 0) * float(
            np.clip(min(np.min(abs(h)), np.min(abs(k))) / 1.0, 0, 1))
        return relative, coordination, hips, knees

    def step(self, action):
        condition = self.cfg.catch_mode
        # B/C differ ONLY by the new sweep term, not legacy slide/catch gating.
        if condition == "B":
            self.cfg.catch_mode = "C"
        try:
            obs, reward, done, info = super().step(action)
        finally:
            self.cfg.catch_mode = condition
        self.trace[-1]["catch_mode"] = condition
        s, cs = self.momentum_sample, self.contact_sample
        out = self.trace[-1]
        started = self.tracker.confirmed or self.start_kind in ('post_liftoff', 'partial', 'near_vertical')
        gate = objective_gate(s, self.row, cs, started, self._momentum_prefix_safe,
                              self.start_pitch, self.peak_swing_pitch, self.phase.progress)
        heading = np.array([self.pitch_axis[1], -self.pitch_axis[0], 0.0])
        com = np.array([s[f'com_{a}'] for a in 'xyz'])
        velocity = np.array([s[f'com_velocity_{a}'] for a in 'xyz'])
        score, landing, request, recovery = placement_prediction(
            com, velocity, self.d.geom_xpos[self.foot_geoms[:2]], self.catch_origin,
            heading, np.linalg.norm(self.m.opt.gravity), self.cfg)
        rel, coordination, hips, knees = self.rear_motion()
        if started and self.sweep_origin_inertia is None:
            self.sweep_origin_inertia = float(s['I_com'])
        contraction = float(np.clip(1 - s['I_com'] / max(self.sweep_origin_inertia or s['I_com'], 1e-8), 0, 0.4) / 0.4)
        rotation = float(np.clip((s['pitch'] - self.start_pitch) / np.deg2rad(45), 0, 1))
        # Reward redistribution only with positive whole-body momentum and useful ascent.
        sweep = rotation * coordination * float(s['L_com'] > 0) * float(
            np.clip(-rel / self.cfg.rear_relative_momentum_scale, 0, 1)) * (0.75 + 0.25*contraction)
        active = self.cfg.catch_mode in ('B', 'C')
        early = score * rotation if gate and active else 0.0
        sweep = sweep if gate and self.cfg.catch_mode == 'C' else 0.0
        early_reward = self.cfg.early_placement_weight * max(0, early - self.early_paid)
        sweep_reward = self.cfg.rear_sweep_weight * max(0, sweep - self.sweep_paid)
        self.early_paid = max(self.early_paid, early)
        self.sweep_paid = max(self.sweep_paid, sweep)
        if started and self._momentum_prefix_safe:
            if s['pitch'] > self.timing_peak_pitch:
                self.timing_peak_pitch, self.peak_pitch_time = s['pitch'], float(self.d.time)
            if self.first_front_motion is None and max(cs['FL_backward'], cs['FR_backward']) > 0.005:
                self.first_front_motion = float(self.d.time)
            if self.first_rear_sweep is None and coordination > 0.1 and rel < -0.02:
                self.first_rear_sweep = float(self.d.time)
        extra = early_reward + sweep_reward
        self.reward_sum += extra
        for name, value in [('early_placement', early_reward), ('rear_sweep', sweep_reward)]:
            self.terms_sum[name] = self.terms_sum.get(name, 0.0) + value
            out['reward_' + name] = value
        out.update(reward=reward+extra, anticipation_gate=gate,
                   predicted_support_x=float(landing[0]), predicted_support_y=float(landing[1]),
                   predicted_recoverability=recovery, FL_requested_backward=float(request[0]),
                   FR_requested_backward=float(request[1]), rear_relative_momentum=rel,
                   rear_coordination=coordination, inertia_contraction=contraction,
                   early_placement_score=early, rear_sweep_score=sweep,
                   first_front_motion_time=self.first_front_motion,
                   first_rear_sweep_time=self.first_rear_sweep, peak_pitch_time=self.peak_pitch_time)
        for label, ids in [('rear_hip', hips), ('rear_knee', knees)]:
            for side, i in zip(('L', 'R'), ids, strict=True):
                out[f'{label}_{side}_velocity'] = float(self.d.qvel[self.vids[i]])
        info.update(reward=reward+extra, episode_reward=self.reward_sum, early_placement_credit=self.early_paid,
                    rear_sweep_credit=self.sweep_paid, first_front_motion_time=self.first_front_motion,
                    first_rear_sweep_time=self.first_rear_sweep, peak_pitch_time=self.peak_pitch_time)
        return obs, reward+extra, done, info


def environment_class(cfg):
    return AnticipatoryEnvironment if cfg.comparison_suite == 'anticipatory_v1' else CatchEnvironment
