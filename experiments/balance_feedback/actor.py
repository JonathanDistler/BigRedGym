"""One observation-to-action linear neural actor initialized from native LQR.

Local handstand skill only. No simulator state, clock, or controller switching is
used at inference. All 93 existing actor inputs retain their original ordering;
34 physical channels drive the feedback. The original 73-channel ADC prefix is
unchanged. This is an analytical initializer, not a PPO-trained handstand actor.
"""

import numpy as np
import torch
from torch import nn

# gravity, angular velocity, joint offset/speed, COM height and COM velocity.
CHANNELS = np.r_[np.arange(30), 55, np.arange(56, 59)]


class BalanceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("channels", torch.tensor(CHANNELS, dtype=torch.long))
        self.feedback = nn.Linear(len(CHANNELS), 12, dtype=torch.float64)
        self.register_buffer("lower", torch.full((12,), -1.0, dtype=torch.float64))
        self.register_buffer("upper", torch.full((12,), 1.0, dtype=torch.float64))

    def forward(self, observations):
        """Return bounded normalized POSITION TARGET actions (already squashed)."""
        if observations.shape[-1] != 93:
            raise ValueError("Expected unchanged multi-impulse actor observation (93)")
        x = observations[..., self.channels].to(self.feedback.weight.dtype)
        return self.feedback(x).clamp(self.lower, self.upper).to(observations.dtype)


def initialize_from_teacher(env, teacher):
    """Project native LQR derivatives into policy-observable local coordinates."""
    import mujoco

    q = teacher.q.copy()
    data = mujoco.MjData(env.m)

    def features(qpos, qvel):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(env.m, data)
        jac = np.zeros((3, env.m.nv))
        mujoco.mj_jacSubtreeCom(env.m, data, jac, env.base)
        return np.r_[
            data.xmat[env.base].reshape(3, 3).T @ [0, 0, -1],
            qvel[3:6],
            qpos[env.qids] - env.nominal,
            qvel[env.vids] / env.speed,
            data.subtree_com[env.base, 2] / 0.6,
            jac @ qvel,
        ]

    center = features(q, np.zeros(env.m.nv))
    eps = 1e-5
    jacobian = np.zeros((len(CHANNELS), len(teacher.indices)))
    for column, index in enumerate(teacher.indices):
        samples = []
        for sign in (-1, 1):
            qq = q.copy()
            vv = np.zeros(env.m.nv)
            if index < env.m.nv:
                delta = np.zeros(env.m.nv)
                delta[index] = sign * eps
                mujoco.mj_integratePos(env.m, qq, delta, 1.0)
            else:
                vv[index - env.m.nv] = sign * eps
            samples.append(features(qq, vv))
        jacobian[:, column] = (samples[1] - samples[0]) / (2 * eps)
    rank = np.linalg.matrix_rank(jacobian)
    # Absolute heading is intentionally absent from the actor observations.
    # At vertical it is a different local tangent axis from the legacy teacher
    # yaw convention. Project that unobservable nuisance rotation out explicitly.
    if rank < len(teacher.indices) - 1:
        raise ValueError(f"Unexpected local observation rank {rank}")
    weights = -teacher.K @ np.linalg.pinv(jacobian) / env.cfg.action_scale
    reference_action = (teacher.target - env.nominal) / env.cfg.action_scale
    actor = BalanceActor()
    with torch.no_grad():
        actor.feedback.weight.copy_(torch.tensor(weights))
        actor.feedback.bias.copy_(torch.tensor(reference_action - weights @ center))
        actor.lower.copy_(
            torch.tensor(
                np.maximum(-1, (env.bounds[:, 0] - env.nominal) / env.cfg.action_scale)
            )
        )
        actor.upper.copy_(
            torch.tensor(
                np.minimum(1, (env.bounds[:, 1] - env.nominal) / env.cfg.action_scale)
            )
        )
    return actor, dict(
        rank=int(rank),
        state_dimensions=len(teacher.indices),
        unobservable_feedback_gain=float(
            np.max(abs(weights @ jacobian + teacher.K / env.cfg.action_scale))
        ),
        discarded_heading_dimensions=int(len(teacher.indices) - rank),
        selected_channels=CHANNELS.tolist(),
        observation_dim=93,
        action_dim=12,
        action_definition="bounded normalized joint-position target; DO NOT tanh again",
        dt=env.cfg.dt,
        kp=env.cfg.kp,
        kd=env.cfg.kd,
    )


def adapt_adc_observation(observation):
    """Pad the unchanged 73-channel ADC prefix; never reorder existing inputs."""
    if observation.shape[-1] == 93:
        return observation
    if observation.shape[-1] != 73:
        raise ValueError("Expected ADC73 or multi-impulse93 observations")
    return torch.nn.functional.pad(observation, (0, 20))
