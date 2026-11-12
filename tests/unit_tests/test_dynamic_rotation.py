"""Dynamic crossings are separate from terminal stability and native validity."""

import numpy as np
import pytest

from scripts.search_dynamic_rotation import assess, energy
from experiments.morphology.learning.pendulum_reference import score_trajectory


def trajectory():
    return [
        dict(
            time=float(t),
            pitch=float(np.deg2rad(30) + 1.2 * t),
            pitch_velocity=1.2,
            FL_load=80.0,
            FR_load=80.0,
            rear_force_z=0.0,
            FL_velocity_x=0.0,
            FL_velocity_y=0.0,
            FR_velocity_x=0.0,
            FR_velocity_y=0.0,
            FL_position_x=0.0,
            FR_position_x=0.0,
            com_x=-0.15,
            com_z=0.35,
            com_velocity_x=0.2,
            com_velocity_y=0.0,
            com_velocity_z=0.0,
            inertia_com_y=0.5,
            nonfoot_collision=False,
            self_collision=False,
            joint_violation=0.0,
            max_speed_ratio=0.5,
            torque_ratio=0.9,
            front_drift=0.0,
        )
        for t in np.arange(0, 0.402, 0.002)
    ]


def test_dynamic_crossing_allowed_without_intermediate_hold():
    trace = trajectory()
    new = assess(trace, 15.0)
    assert new["cross45"] and new["improved"]
    assert new["omega45"] == 1.2
    assert not new["terminal_recovery_60"]
    assert not score_trajectory(trace, 15 * 9.81)["success"]


def test_turnaround_at_45_is_not_progress_to_60():
    trace = trajectory()
    for r in trace:
        if r["pitch"] > np.deg2rad(45.2):
            r["pitch"] = np.deg2rad(45.2)
            r["pitch_velocity"] = -0.2
    result = assess(trace, 15.0)
    assert result["cross45"] and not result["improved"] and not result["cross60"]


def test_later_collision_cannot_be_cropped_from_acceptance():
    trace = trajectory()
    trace[-1]["nonfoot_collision"] = True
    result = assess(trace, 15.0)
    assert result["cross45"] and not result["full_valid"] and not result["improved"]


@pytest.mark.parametrize(
    "field,value",
    [("max_speed_ratio", 1.002), ("torque_ratio", 1.01), ("joint_violation", 0.002)],
)
def test_native_limits_unchanged(field, value):
    trace = trajectory()
    trace[-1][field] = value
    assert not assess(trace, 15.0)["improved"]


def test_body_and_com_angular_velocity_not_conflated():
    r = trajectory()[0]
    e = energy(r, 15.0)
    assert e["com_omega"] != r["pitch_velocity"]
    assert e["energy_deficit"] > 0
    r["com_velocity_x"] = -0.2
    e = energy(r, 15.0)
    assert e["rotational_energy"] == 0 and e["com_omega"] < 0


def test_pilot_blocks_support_loss_even_if_success_flag_is_wrong(tmp_path):
    import pandas as pd
    from scripts.run_dynamic_rotation_pilot import supported_gate

    root = tmp_path / "frontier"
    root.mkdir()
    rows = []
    for seed in (42, 43):
        for reset in range(10):
            rows.append(
                dict(
                    seed=seed,
                    label=0,
                    reset=reset,
                    improved=True,
                    cross45=True,
                    full_valid=True,
                    nonfoot_collision=False,
                    self_collision=False,
                    max_speed_ratio=0.8,
                    max_joint_violation=0.0,
                    max_torque_ratio=1.0,
                    contact_gap=0.34,
                    front_drift=0.7,
                    slip_distance=0.04,
                )
            )
    pd.DataFrame(rows).to_csv(root / "validation_cases.csv", index=False)
    assert not supported_gate(tmp_path)
