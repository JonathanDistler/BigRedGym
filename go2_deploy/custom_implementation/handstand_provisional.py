"""Cornell fixdeploy hooks. Provisional team name; offline candidate only.

Stock runner fails closed until lossless sensor ingestion, explicit activation,
synchronized snapshots and bounded command acknowledgements are integrated.
"""

import torch

SLICES = {
    "projected_gravity": (0, 3),
    "base_ang_vel": (3, 6),
    "dof_pos_obs": (6, 18),
    "dof_vel": (18, 30),
    "handstand_previous_target": (30, 42),
    "handstand_elapsed": (42, 43),
    "handstand_com": (43, 47),
}


class CustomDeployConfig:
    obs_vector = list(SLICES)
    custom_obs_sizes = {key: end - start for key, (start, end) in SLICES.items()}

    class DeployScaling:
        projected_gravity = base_ang_vel = dof_pos_obs = dof_vel = 1.0
        handstand_previous_target = handstand_elapsed = handstand_com = 1.0
        dof_pos_target = 2.0  # Cornell RLController action multiplier only


def get_custom_buffers_dict():
    return {"handstand_session": None, "handstand_snapshot": None}


def update_custom_buffers(main_controller, t):
    if main_controller.handstand_session is None:
        raise RuntimeError(
            "NOT READY: lossless IMU ingestion and "
            "bounded-command acknowledgement required"
        )
    main_controller.handstand_snapshot = main_controller.handstand_session.snapshot(t)


def get_custom_obs_pieces():
    def getter(start, end):
        def piece(main_controller, lowstate_msg):
            if main_controller.handstand_snapshot is None:
                raise RuntimeError("Missing synchronized handstand snapshot")
            return torch.from_numpy(
                main_controller.handstand_snapshot[start:end].copy()
            )

        return piece

    return {key: getter(start, end) for key, (start, end) in SLICES.items()}
