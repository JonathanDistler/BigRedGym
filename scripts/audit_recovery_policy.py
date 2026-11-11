"""Inspect the verified frozen CPG balance checkpoint without modifying it."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.evaluate_kickup_handoff import Policy, dump
from scripts.reproduce_transition_hold import BALANCE


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    with torch.inference_mode():
        policy = Policy(BALANCE, "go2balance", 1, True)
        e, r = policy.env, policy.runner
        dump(
            a.output,
            dict(
                **policy.meta,
                actor_observations=r.get_obs(r.actor_cfg["obs"]).shape[1],
                critic_observations=r.get_obs(r.critic_cfg["obs"]).shape[1],
                actions=r.get_inference_actions().shape[1],
                actuator_order=e.dof_names,
                soft_position_limits=e.dof_pos_limits.tolist(),
                native_position_limits=e._backend._model_for_env(0).jnt_range.tolist(),
                torque_limits=e.actuated_torque_limits.tolist(),
                velocity_limits=e.dof_vel_limits.tolist(),
                residual_scale=e.scales["dof_pos_target"].tolist(),
                reference=e._standing_posture()[0].tolist(),
                phase=e.leg_phase[0].tolist(),
                contact_modes=e.cpg_contact_obs[0].tolist(),
                actor_frequency=r.actor_cfg["frequency"],
                control_dt=e.dt,
                cpg_role=(
                    "Force-coupled phase/contact observations; "
                    "fixed handstand joint reference plus PPO position residuals."
                ),
            ),
        )
        e._backend.close()


if __name__ == "__main__":
    main()
