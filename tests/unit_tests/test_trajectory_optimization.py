"""Software-only constraint correctness and fail-closed trajectory export."""

from pathlib import Path
import json
import numpy as np
import pytest
from experiments.trajectory_optimization.model import Robot, Settings
from experiments.trajectory_optimization.transcription import (
    Problem,
    Y,
    V,
    A,
    TAU,
    FORCE,
    WIDTH,
)

REFERENCE = Path("artifacts/handstand_reference/trajectory.npz")


class StaticSettings(Settings):
    def contacts(self, k):
        return np.ones(4, dtype=bool)


@pytest.fixture(scope="module")
def endpoints():
    if not REFERENCE.exists():
        pytest.skip("Provide hash-locked external reference")
    r = Robot(StaticSettings(intervals=2, duration=0.04))
    return r, r.endpoints(REFERENCE)


@pytest.fixture
def static_problem(endpoints):
    r, (start, goal, anchors, nominal) = endpoints
    return Problem(r, start, start, anchors, nominal)


def test_objective_is_identically_zero(static_problem):
    p = static_problem
    x = p.seed(static=True)
    assert p.objective(x) == p.objective(x + 100) == 0
    np.testing.assert_array_equal(p.objective_gradient(x), np.zeros_like(x))


def test_crouched_boundary_and_manifold_roundtrip(endpoints):
    r, (start, goal, anchors, nominal) = endpoints
    assert start[2] == pytest.approx(0.27)
    np.testing.assert_allclose(r.pose(r.chart(goal)), goal, atol=1e-14)
    r.set(start)
    feet, _ = r.foot_kinematics()
    np.testing.assert_allclose(feet, anchors, atol=1e-7)
    assert abs(goal[2] - 0.57) < 1e-12
    assert np.linalg.norm(goal[3:7]) == pytest.approx(1)


def test_static_feasible_manipulator_and_contact_constraints(static_problem):
    p = static_problem
    x = p.seed(static=True)
    report = p.audit(x)
    assert report["mesh_feasible"], report
    assert report["maximum_scaled_equality_violation"] < 1e-8
    assert not report["hardware_ready"]


def test_gravity_needs_contact_forces(static_problem):
    p = static_problem
    rows = p.seed(static=True).reshape(-1, WIDTH)
    rows[:, FORCE] = 0
    report = p.audit(rows.ravel())
    assert report["groups"]["manipulator_dynamics"]["maximum_raw_violation"] > 100
    assert not report["mesh_feasible"]


def test_friction_cone_and_negative_normal_detected(static_problem):
    p = static_problem
    rows = p.seed(static=True).reshape(-1, WIDTH)
    rows[1, FORCE] = np.tile([1000, 0, -1], 4)
    g = p.audit(rows.ravel())["groups"]
    assert g["friction_cone"]["maximum_raw_violation"] > 999
    assert g["unilateral_force"]["maximum_raw_violation"] == 1


def test_speed_dependent_torque_limits(endpoints):
    r, _ = endpoints
    np.testing.assert_array_equal(r.torque_envelope(np.zeros(12)), r.effort)
    np.testing.assert_allclose(r.torque_envelope(0.5 * r.speed), 0.5 * r.effort)
    np.testing.assert_array_equal(r.torque_envelope(r.speed), np.zeros(12))
    np.testing.assert_array_equal(r.torque_envelope(-2 * r.speed), np.zeros(12))
    assert not r.cfg.torque_curve_calibrated


def test_contact_acceleration_and_velocity_detected(static_problem):
    p = static_problem
    rows = p.seed(static=True).reshape(-1, WIDTH)
    rows[1, A][0] = 1
    rows[1, V][0] = 0.1
    g = p.audit(rows.ravel())["groups"]
    assert g["stance_zero_acceleration"]["maximum_raw_violation"] > 0.5
    assert g["stance_velocity"]["maximum_raw_violation"] > 0.09


def test_swing_force_must_be_zero(endpoints):
    r, (start, goal, anchors, nominal) = endpoints
    other = Robot(Settings(intervals=2, duration=0.04))
    p = Problem(other, start, goal, anchors, nominal)
    rows = p.seed().reshape(-1, WIDTH)
    rows[-1, FORCE][6] = 1
    assert (
        p.audit(rows.ravel())["groups"]["swing_zero_force"]["maximum_raw_violation"]
        >= 1
    )


def test_ground_clearance_uses_calf_volume(static_problem):
    p = static_problem
    rows = p.seed(static=True).reshape(-1, WIDTH)
    rows[1, Y][2] = 0.1
    g = p.audit(rows.ravel())["groups"]
    assert g["body_calf_head_clearance"]["maximum_raw_violation"] > 0.01
    assert g["foot_ground_nonpenetration"]["maximum_raw_violation"] > 0.1


def test_torque_target_and_slew_constraints(static_problem):
    p = static_problem
    rows = p.seed(static=True).reshape(-1, WIDTH)
    rows[1, TAU] = p.robot.effort
    rows[1, Y][6:18] += 0.5
    report = p.audit(rows.ravel())
    assert report["groups"]["target_slew"]["maximum_raw_violation"] > 0


def test_euler_manipulator_integrator(static_problem):
    p = static_problem
    x = p.seed(static=True)
    rows = x.reshape(-1, WIDTH)
    rows[1, V][0] += 0.2
    g = p.audit(rows.ravel())["groups"]
    assert g["explicit_euler_velocity"]["maximum_raw_violation"] >= 0.2
    assert g["initial_configuration"]["maximum_raw_violation"] == 0


def test_sparse_dependency_structure(static_problem):
    p = static_problem
    x = p.seed(static=True)
    for kind in ["eq", "ineq"]:
        pattern = p.sparsity(x, kind)
        assert pattern.shape == (len(p.values(x, kind)), len(x))
        assert pattern.nnz < pattern.shape[0] * pattern.shape[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration": 0},
        {"friction": 0},
        {"intervals": 1},
        {"release_fraction": 1},
        {"crouch_height": float("nan")},
    ],
)
def test_invalid_problem_settings(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_nonfinite_decision_rejected(static_problem):
    p = static_problem
    x = p.seed()
    x[0] = np.nan
    with pytest.raises(ValueError):
        p.audit(x)


def test_ik_warm_start_is_not_claimed_feasible(endpoints):
    r, (start, goal, anchors, nominal) = endpoints
    moving = Robot(Settings(intervals=4, duration=2.4))
    p = Problem(moving, start, goal, anchors, nominal)
    x = p.kinematic_seed()
    assert np.isfinite(x).all()
    assert p.objective(x) == 0
    assert not p.audit(x)["hardware_ready"]


def _save_static_candidate(tmp_path, p):
    import hashlib
    from dataclasses import asdict

    x = p.seed(static=True)
    rows = p.unpack(x)
    np.savez_compressed(
        tmp_path / "candidate.npz",
        qpos=np.array([p.robot.pose(row[Y]) for row in rows]),
        qvel=rows[:, V],
        targets=np.array([p.target(row) for row in rows]),
        start=p.start,
        goal=p.goal,
        anchors=p.anchors,
        nominal=p.nominal,
        decision=x,
        time=np.linspace(0, p.cfg.duration, p.nodes),
    )
    meta = {
        "settings": asdict(p.cfg),
        "candidate_sha256": hashlib.sha256(
            (tmp_path / "candidate.npz").read_bytes()
        ).hexdigest(),
    }
    (tmp_path / "report.json").write_text(json.dumps(meta))


def test_modified_archive_rejected(static_problem, tmp_path):
    from experiments.trajectory_optimization.validation import audit_saved

    _save_static_candidate(tmp_path, static_problem)
    with open(tmp_path / "candidate.npz", "ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        audit_saved(tmp_path)


def test_export_rejects_unvalidated_trajectory(tmp_path):
    from experiments.trajectory_optimization.validation import export

    with pytest.raises(FileNotFoundError):
        export(tmp_path, tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_ik_projection_preserves_swing_joint_motion(endpoints):
    r, (start, goal, anchors, nominal) = endpoints
    moving = Robot(Settings(intervals=8, duration=2.4))
    p = Problem(moving, start, goal, anchors, nominal)
    rows = p.unpack(p.kinematic_seed())
    # A minimum-norm replacement would erase unconstrained rear motion and
    # introduce spurious Euler defects. Project a correction instead.
    swing_rows = [k for k in range(p.nodes - 1) if not p.cfg.contacts(k)[2]]
    assert np.linalg.norm(rows[swing_rows, 30:36]) > 0.1


def test_resampling_and_slew_match_shared_command_path():
    from experiments.trajectory_optimization.validation import position_commands

    speed = np.tile([30.1, 30.1, 15.7], 4)
    saved = {
        "time": np.array([0.0, 0.04]),
        "targets": np.array([np.zeros(12), np.ones(12) * 0.1]),
    }
    times, targets = position_commands(saved, speed, 0.04)
    np.testing.assert_allclose(targets[:3, 0], [0, 0.05, 0.1], atol=1e-15)
    assert (abs(np.diff(targets, axis=0)) <= 0.7 * speed * 0.02 + 1e-15).all()
    assert np.allclose(np.diff(times), 0.02)
