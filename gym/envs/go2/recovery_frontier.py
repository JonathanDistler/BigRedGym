"""Native recovery replay, common success definition and authority diagnostics."""

import copy

import mujoco
import numpy as np
import torch

from gym.envs.go2.recovery_state import restore, sync


def native_contact_flags(env):
    b = env._backend
    feet = set(b._canonical_to_native_body_np[env.feet_indices.cpu().numpy()])
    hits = []
    for i, d in enumerate(b._datas):
        m = b._model_for_env(i)
        hits.append(
            any(
                c.dist < -1e-5
                and (
                    (m.geom_bodyid[c.geom1] == 0 and m.geom_bodyid[c.geom2] not in feet)
                    or (
                        m.geom_bodyid[c.geom2] == 0
                        and m.geom_bodyid[c.geom1] not in feet
                    )
                )
                for c in d.contact
            )
        )
    return np.array(hits)


def support_loads(env):
    b = env._backend
    feet = b._canonical_to_native_body_np[env.feet_indices.cpu().numpy()]
    loads = np.zeros((env.num_envs, 4))
    wrench = np.zeros(6)
    for i, d in enumerate(b._datas):
        m = b._model_for_env(i)
        for j, c in enumerate(d.contact):
            a, z = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            body = z if a == 0 else a if z == 0 else -1
            if body in feet:
                mujoco.mj_contactForce(m, d, j, wrench)
                force = c.frame.reshape(3, 3).T @ wrench[:3]
                k = int(np.flatnonzero(feet == body)[0])
                loads[i, k] += force[2] if a == 0 else -force[2]
    return np.maximum(loads, 0)


def valid_recovery(env, collision=None):
    loads = support_loads(env)
    weight = env._backend.link_mass.sum(1).numpy() * 9.81
    pitch = torch.atan2(
        env.projected_gravity[:, 0], -env.projected_gravity[:, 2]
    ).numpy()
    from gym.envs.go2.go2_config import FOOT_RADIUS

    clearance = env._rigid_body_pos[:, env.feet_indices[2:], 2].numpy() - FOOT_RADIUS
    valid = (
        (abs(pitch - np.pi / 2) <= np.pi / 12)
        & (abs(env.base_ang_vel[:, 1].numpy()) <= 1.5)
        & (loads[:, :2] >= 0.06 * weight[:, None]).all(1)
        & (loads[:, 2:] <= 0.015 * weight[:, None]).all(1)
        & (clearance >= 0.2).all(1)
    )
    if collision is not None:
        valid &= ~np.asarray(collision)
    return valid, (loads[:, :2] >= 0.06 * weight[:, None]).all(1)


def install_samples(policy, samples):
    policy.reset()
    e = policy.env
    b = e._backend
    # Replay models must be independent even for the frozen baseline task.
    b._models = [copy.copy(b._model_for_env(i)) for i in range(e.num_envs)]
    b._mjm = b._models[0]
    b._datas = [mujoco.MjData(m) for m in b._models]
    for i in range(e.num_envs):
        restore(e, i, samples[i % len(samples)])
    sync(e)
    e.episode_length_buf.zero_()
    e.terminated.zero_()
    e.timed_out.zero_()
    e._hold_time.zero_()
    e._best_hold.zero_()
    if hasattr(e, "_recovery_collision"):
        e._recovery_collision.zero_()


def sample_from_arrays(arrays, i):
    return {
        k: v[i]
        for k, v in arrays.items()
        if k == "native" or k.startswith(("model_", "controller_"))
    }


def authority(env, index):
    b = env._backend
    m, d = b._model_for_env(index), b._datas[index]
    order = b._canonical_to_native_dof_np
    q = d.qpos[7 + order]
    v = d.qvel[6 + order]
    target = (env._standing_posture() + env.dof_pos_target)[index].numpy()
    requested = (
        env.p_gains[index].numpy() * (target - q)
        + env.d_gains[index].numpy() * (env.dof_vel_target[index].numpy() - v)
        + env.tau_ff[index].numpy()
    )
    limits = env.actuated_torque_limits.numpy()
    native_limits = m.jnt_range[1:][order]
    joint_violation = np.maximum(native_limits[:, 0] - q, 0) + np.maximum(
        q - native_limits[:, 1], 0
    )
    feet = b._canonical_to_native_body_np[env.feet_indices[:2].numpy()]
    contacts = []
    wrench = np.zeros(6)
    for ci, c in enumerate(d.contact):
        a, z = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        body = z if a == 0 else a if z == 0 else -1
        if body in feet:
            mujoco.mj_contactForce(m, d, ci, wrench)
            force = c.frame.reshape(3, 3).T @ wrench[:3]
            if a != 0:
                force = -force
            mu = float(c.friction[0])
            contacts.append(
                dict(
                    position=c.pos.tolist(),
                    force_world_N=force.tolist(),
                    friction=mu,
                    friction_utilization=float(
                        np.linalg.norm(force[:2]) / max(mu * force[2], 1e-6)
                    ),
                )
            )
    mujoco.mj_subtreeVel(m, d)
    offset = d.subtree_com[1, :2] - d.xpos[feet, :2].mean(0)
    speed = float(np.linalg.norm(d.subtree_linvel[1, :2]))
    mu = min((c["friction"] for c in contacts), default=None)
    return dict(
        front_contacts=contacts,
        com_support_offset_xy_m=offset.tolist(),
        com_velocity_xy_m_s=d.subtree_linvel[1, :2].tolist(),
        constant_height_braking_distance_m=(speed**2 / (2 * mu * 9.81) if mu else None),
        requested_torque_limit_ratio=float(np.max(abs(requested) / limits)),
        torque_headroom_Nm=(limits - np.minimum(abs(requested), limits)).tolist(),
        maximum_joint_limit_violation_rad=float(joint_violation.max()),
        actual_joint_speed_limit_ratio=float(
            np.max(abs(v) / env.dof_vel_limits.numpy())
        ),
        cpg_phase=env.leg_phase[index].tolist(),
        cpg_filtered_load=env._cpg_load[index].tolist(),
        inference="Constraint diagnostics do not certify dynamic infeasibility.",
    )


@torch.inference_mode()
def replay_batch(policy, samples, horizon_s=6.5, trace=False):
    """No perturbations, blends, pose projection or automatic reset during replay."""
    from scripts.evaluate_kickup_handoff import measurements

    install_samples(policy, samples)
    e = policy.env
    initial = measurements(e)
    initial_authority = [authority(e, i) for i in range(len(samples))]
    collision = native_contact_flags(e)
    old = e._post_physics_step
    hold = np.zeros(e.num_envs)
    longest = hold.copy()
    total = hold.copy()
    support = hold.copy()
    saturation = hold.copy()
    alive_steps = hold.copy()
    com_motion = hold.copy()
    pitch_motion = hold.copy()
    first_collision_time = np.full(e.num_envs, np.nan)
    residual = hold.copy()
    records = []

    def post():
        old()
        collision[:] |= native_contact_flags(e)

    e._post_physics_step = post
    count = 0
    try:
        for step in range(round(horizon_s / e.dt)):
            policy.action()
            e.step()
            count += 1
            valid, supported = valid_recovery(e, collision)
            hold = np.where(valid, hold + e.dt, 0.0)
            longest = np.maximum(longest, hold)
            total += valid * e.dt
            alive = ~collision
            first = collision & np.isnan(first_collision_time)
            first_collision_time[first] = (step + 1) * e.dt
            alive_steps += alive
            support += supported & alive
            for j, d in enumerate(e._backend._datas):
                mujoco.mj_subtreeVel(e._backend._model_for_env(j), d)
                if alive[j]:
                    com_motion[j] += np.linalg.norm(d.subtree_linvel[1])
                    pitch_motion[j] += abs(float(e.base_ang_vel[j, 1]))
            saturation += (
                e.torques.abs() >= 0.99 * e.actuated_torque_limits
            ).float().mean(1).numpy() * alive
            residual += e.dof_pos_target.square().mean(1).numpy() * alive
            if trace and step % 20 == 0:
                records.append(
                    dict(
                        time=(step + 1) * e.dt,
                        pitch_rate=e.base_ang_vel[:, 1].tolist(),
                        hold_s=hold.tolist(),
                        phase=e.leg_phase.tolist(),
                    )
                )
            if collision.all():
                break
    finally:
        e._post_physics_step = old
    final = measurements(e)
    outcomes = []
    for i in range(len(samples)):
        success = bool(longest[i] >= 5 - 1e-6)
        invalid = bool(
            initial[i]["body_contact"]
            or initial_authority[i]["maximum_joint_limit_violation_rad"] > 0.03
        )
        outcomes.append(
            dict(
                success=success,
                hold_s=float(longest[i]),
                total_stable_duration_s=float(total[i]),
                collision=bool(collision[i]),
                front_support_retention=float(support[i] / max(count, 1)),
                alive_front_support_retention=float(
                    support[i] / max(alive_steps[i], 1)
                ),
                first_collision_time_s=(
                    None
                    if np.isnan(first_collision_time[i])
                    else float(first_collision_time[i])
                ),
                mean_alive_com_speed=float(com_motion[i] / max(alive_steps[i], 1)),
                mean_alive_abs_pitch_rate=float(
                    pitch_motion[i] / max(alive_steps[i], 1)
                ),
                motor_saturation_fraction=float(saturation[i] / max(alive_steps[i], 1)),
                residual_rms_rad=float(np.sqrt(residual[i] / max(alive_steps[i], 1))),
                initial=initial[i],
                final=final[i],
                authority=initial_authority[i],
                validity="inadmissible_initial_state" if invalid else "valid",
                feasibility="demonstrated_recoverable"
                if success
                else "constraint_violation_at_reset"
                if invalid
                else "controller_failure_physical_feasibility_unknown",
            )
        )
    return outcomes, records
