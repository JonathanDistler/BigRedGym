"""Independent reference timing and feedforward checks."""

import numpy as np
from gym.envs.go2.kickup_reference import KickReference, reference


def test_analytic_reference_velocity_matches_finite_difference():
    p = KickReference(hip_delay=-0.1, knee_delay=0.2)
    standing = np.tile([0.0, 0.58, -1.16], 4)
    for t in (0.1, 0.35, 0.63, 0.69, 0.8, 1.0, 1.3):
        q, v = reference(t, p, standing)
        numerical = (
            reference(t + 1e-6, p, standing)[0] - reference(t - 1e-6, p, standing)[0]
        ) / 2e-6
        np.testing.assert_allclose(v, numerical, atol=1e-7)
        assert np.isfinite(q).all()


def test_reference_starts_at_standing_and_ends_at_tuck():
    p = KickReference()
    standing = np.tile([0.0, 0.58, -1.16], 4)
    np.testing.assert_array_equal(reference(0, p, standing)[0], standing)
    q, v = reference(2, p, standing)
    np.testing.assert_allclose(q[[7, 10]], p.rear_tuck_hip)
    np.testing.assert_allclose(q[[8, 11]], p.rear_tuck_knee)
    np.testing.assert_array_equal(v, np.zeros(12))


def test_hip_knee_delay_controls_only_rear_extension():
    standing = np.tile([0.0, 0.58, -1.16], 4)
    p = KickReference(knee_delay=0.25)
    q, _ = reference(0.61, p, standing)
    assert q[7] < p.rear_crouch_hip
    assert q[8] == p.rear_crouch_knee
    np.testing.assert_allclose(q[6:9], q[9:12])


def test_force_release_without_geometric_separation_cannot_pass():
    from gym.envs.go2.kickup_reference import physical_liftoff

    clearance = np.array([0.001, 0.001, 0.001, 0.001])
    force = np.array([75.0, 75.0, 0.0, 0.0])
    assert not physical_liftoff(clearance, force, [True, True], 150.0, 0.01, False)
    clearance[2:] = 0.03
    assert physical_liftoff(clearance, force, [True, True], 150.0, 0.01, False)
    assert not physical_liftoff(clearance, force, [True, False], 150.0, 0.01, False)
    force[0] = 0
    assert not physical_liftoff(clearance, force, [True, True], 150.0, 0.01, False)
    force[0] = 75
    assert not physical_liftoff(clearance, force, [True, True], 150.0, 0.05, False)
    assert not physical_liftoff(clearance, force, [True, True], 150.0, 0.01, True)


def test_selected_reference_has_real_supported_takeoff_in_mujoco():
    import json
    from pathlib import Path
    from scripts.search_kickup_reference import Experiment

    p = KickReference(
        **json.loads(
            (
                Path(__file__).parents[2]
                / "gym/envs/go2/references/kickup_front_yield.json"
            ).read_text()
        )
    )
    ex = Experiment()
    result, rows = ex.trial(p, seed=3000, record=True)
    assert result["success_30mm"]
    assert result["verified_30mm_hold_s"] >= 0.02
    assert result["max_front_drift_m"] <= 0.04
    assert not result["body_contact"]
    assert np.all(np.asarray(result["peak_torque_Nm"]) <= ex.torque_limits + 1e-6)
    assert result["peak_velocity_limit_ratio"] <= 1.0
    assert any(
        min(row["clearance_2"], row["clearance_3"]) >= 0.03 and row["front_contact"]
        for row in rows
    )
