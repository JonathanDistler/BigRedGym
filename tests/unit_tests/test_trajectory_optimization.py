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


def test_contact_refined_mesh_and_validation():
    from experiments.trajectory_optimization.warm_start import contact_mesh

    times = contact_mesh(12, 2.4, 0.3)
    cfg = Settings(intervals=12, duration=2.4, mesh_times=times)
    assert len(times) == 13
    assert np.all(np.diff(times) > 0)
    release = np.flatnonzero(np.isclose(times, 0.72))[0]
    assert cfg.contacts(release - 1).all()
    np.testing.assert_array_equal(cfg.contacts(release), [True, True, False, False])
    assert sum(cfg.step(k) for k in range(12)) == pytest.approx(2.4)
    with pytest.raises(ValueError, match="Mesh"):
        Settings(intervals=2, mesh_times=(0, 0, 2.4))


def test_inverse_dynamics_repairs_without_corrupting_state(static_problem):
    from experiments.trajectory_optimization.warm_start import inverse_dynamics_seed

    p = static_problem
    original = p.seed(static=True).reshape(-1, WIDTH)
    original[:, TAU] = 0
    original[:, FORCE] = 0
    repaired, report = inverse_dynamics_seed(p, original.ravel())
    assert all(n["accepted"] for n in report), report
    repaired = repaired.reshape(-1, WIDTH)
    np.testing.assert_array_equal(repaired[:, Y], original[:, Y])
    np.testing.assert_array_equal(repaired[:, V], original[:, V])
    audit = p.audit(repaired.ravel())
    assert audit["groups"]["manipulator_dynamics"]["maximum_raw_violation"] < 1e-6
    assert audit["groups"]["friction_cone"]["maximum_raw_violation"] < 1e-7
    assert audit["groups"]["velocity_dependent_torque"]["maximum_raw_violation"] < 1e-7
    # Nodewise repair is not a temporal-feasibility certificate.


def test_prolongation_and_nonuniform_euler(static_problem):
    from experiments.trajectory_optimization.warm_start import prolong

    p = static_problem
    cfg = StaticSettings(
        intervals=4, duration=0.04, mesh_times=(0, 0.003, 0.01, 0.025, 0.04)
    )
    r = Robot(cfg)
    fine = Problem(r, p.start, p.goal, p.anchors, p.nominal)
    x = prolong(p, p.seed(static=True), fine)
    assert fine.audit(x)["mesh_feasible"]
    rows = fine.unpack(x).copy()
    rows[2, A][0] = 2
    blocks = fine.blocks(rows.ravel())
    velocity_defect = [
        v for _, n, v, _, d in blocks if n == "explicit_euler_velocity" and d == (2, 3)
    ][0]
    assert velocity_defect[0] == pytest.approx(-0.03)


def test_discrete_contact_only_removes_acceleration(endpoints):
    r0, (start, _, anchors, nominal) = endpoints
    r = Robot(StaticSettings(intervals=2, duration=0.04, contact_acceleration=False))
    p = Problem(r, start, start, anchors, nominal)
    names = {b[1] for b in p.blocks(p.seed(static=True))}
    assert "stance_zero_acceleration" not in names
    assert {
        "stance_ground_gap",
        "stance_velocity",
        "stance_drift",
        "friction_cone",
        "native_self_clearance",
        "velocity_dependent_torque",
    } <= names


def test_refinement_splits_largest_defect(static_problem):
    from experiments.trajectory_optimization.warm_start import refine_defects

    p = static_problem
    rows = p.unpack(p.seed(static=True)).copy()
    rows[1, A][0] = 10
    mesh, report = refine_defects(p, rows.ravel(), 1)
    assert report["split_intervals"] == [1]
    np.testing.assert_allclose(mesh, [0, 0.02, 0.03, 0.04])
    with pytest.raises(ValueError):
        refine_defects(p, rows.ravel(), 0)


def test_restoration_exact_dynamics_and_audit(static_problem):
    from scripts.optimize_go2_trajectory import solve

    p = static_problem
    initial = p.seed(static=True)
    result, _ = solve(
        p,
        initial,
        max_iterations=1,
        max_seconds=10,
        method="restoration",
        eliminate_dynamics=True,
    )
    audit = p.audit(result.x)
    assert audit["groups"]["manipulator_dynamics"]["maximum_raw_violation"] < 1e-9
    assert audit["maximum_bound_violation"] < 1e-9
    assert audit["objective"] == 0
    with pytest.raises(ValueError, match="restoration"):
        solve(p, initial, max_iterations=1, eliminate_dynamics=True)


def test_calf_surface_uses_full_collision_geometry(static_problem):
    from gym.envs.go2.handstand_replay import clearance

    r = static_problem.robot
    r.set(static_problem.start)
    expected = [min(clearance(r.m, r.d, g) for g in gs) for gs in r.calf_geoms]
    np.testing.assert_allclose(r.calf_heights(), expected)
    # Cylinder-center height is insufficient to detect the lowest calf surface.
    centers = [min(r.d.geom_xpos[g, 2] for g in gs) for gs in r.calf_geoms]
    assert np.all(r.calf_heights() < centers)


def test_swept_calf_constraints_and_penalty(endpoints):
    from dataclasses import replace

    _, (start, _, anchors, nominal) = endpoints
    cfg = StaticSettings(
        intervals=2,
        duration=0.04,
        sweep_substeps=2,
        calf_clearance=0.01,
        calf_buffer=0.12,
        calf_penalty_weight=100,
    )
    p = Problem(Robot(cfg), start, start, anchors, nominal)
    x = p.seed(static=True)
    samples = list(p.calf_samples(x))
    assert len(samples) == 7
    assert samples[1][1] == (0, 1)
    assert p.calf_penalty_sparsity(x).shape == (28, len(x))
    assert np.linalg.norm(p.calf_penalty(x)) > 0
    assert p.objective(x) == pytest.approx(0.5 * np.sum(p.calf_penalty(x) ** 2))
    assert p.objective_gradient(x)[2] < 0
    groups = p.audit(x)["groups"]
    assert groups["calf_surface_clearance"]["maximum_raw_violation"] == 0
    rows = p.unpack(x).copy()
    rows[1, Y][2] -= 0.15
    assert (
        p.audit(rows.ravel())["groups"]["calf_surface_clearance"][
            "maximum_raw_violation"
        ]
        > 0.04
    )
    assert np.linalg.norm(p.calf_penalty(rows.ravel())) > np.linalg.norm(
        p.calf_penalty(x)
    )
    for kwargs in [
        {"sweep_substeps": -1},
        {"calf_penalty_weight": -1},
        {"calf_clearance": 0.2, "calf_buffer": 0.1},
    ]:
        with pytest.raises(ValueError):
            replace(cfg, **kwargs)


def test_native_shooting_targets_and_determinism(static_problem, tmp_path):
    from experiments.trajectory_optimization.shooting import NativeShooting

    _save_static_candidate(tmp_path, static_problem)
    p = NativeShooting(tmp_path, knots=2, tail=0)
    grid, targets = p.commands(np.full(8, 0.6))
    np.testing.assert_array_equal(targets[0], p.saved["targets"][0])
    assert np.all(targets >= p.robot.bounds[:, 0])
    assert np.all(targets <= p.robot.bounds[:, 1])
    assert np.all(abs(np.diff(targets, axis=0)) <= 0.7 * p.robot.speed * 0.02 + 1e-12)
    r1, t1, c1 = p.simulate(np.zeros(8))
    r2, t2, c2 = p.simulate(np.zeros(8))
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(t1, t2)
    assert c1 == c2
    assert len(grid) == 2 and np.isfinite(r1).all()
    planned = p.planned_clearance()
    assert planned.shape[1] == 5 and planned[0, 0] == 0
    assert planned[-1, 0] == pytest.approx(0.04)
    assert np.all(planned[:, 1:] > 0.07)
    with pytest.raises(ValueError):
        p.commands(np.full(8, np.nan))


def test_command_replay_rejects_unsafe_override(static_problem, tmp_path):
    from experiments.trajectory_optimization.validation import replay, position_commands

    _save_static_candidate(tmp_path, static_problem)
    grid, commands = position_commands(
        {
            "time": static_problem.cfg.times,
            "targets": np.array(
                [
                    static_problem.target(row)
                    for row in static_problem.unpack(static_problem.seed(static=True))
                ]
            ),
        },
        static_problem.robot.speed,
        5.0,
    )
    commands[1, 0] = 100
    with pytest.raises(ValueError, match="limits"):
        replay(tmp_path, command_override=(grid, commands))
    with pytest.raises(ValueError, match="timing"):
        replay(tmp_path, command_override=(grid + 0.001, commands))


def test_native_prefix_horizon_does_not_change_full_command_trial(
    static_problem, tmp_path
):
    from experiments.trajectory_optimization.shooting import NativeShooting

    _save_static_candidate(tmp_path, static_problem)
    p = NativeShooting(tmp_path, knots=2, tail=0, horizon=0.02)
    residual, trace, _ = p.simulate(np.zeros(8))
    assert trace.shape[0] == 1
    grid, commands = p.commands(np.zeros(8), hold_seconds=5.0)
    assert grid[-1] >= 5
    assert commands.shape == (len(grid), 12)
    with pytest.raises(ValueError, match="Horizon"):
        NativeShooting(tmp_path, knots=2, tail=0, horizon=0.05)
    with pytest.raises(ValueError, match="initial"):
        p.optimize(max_evaluations=1, initial=np.full(8, np.nan))


def test_shooting_joint_margin_strengthens_command_limits(static_problem, tmp_path):
    from experiments.trajectory_optimization.shooting import NativeShooting

    _save_static_candidate(tmp_path, static_problem)
    p = NativeShooting(tmp_path, knots=2, tail=0, target_margin=0.04)
    _, targets = p.commands(np.full(8, 1.0))
    assert np.all(targets >= p.robot.bounds[:, 0] + 0.04)
    assert np.all(targets <= p.robot.bounds[:, 1] - 0.04)
    np.testing.assert_array_equal(targets[0], p.saved["targets"][0])
    with pytest.raises(ValueError, match="margin"):
        NativeShooting(tmp_path, target_margin=-0.1)
    with pytest.raises(ValueError, match="margin"):
        NativeShooting(tmp_path, target_margin=2.0)


def test_reference_seed_is_hash_locked_and_bounded(static_problem, tmp_path):
    from experiments.trajectory_optimization.shooting import NativeShooting

    _save_static_candidate(tmp_path, static_problem)
    p = NativeShooting(tmp_path, knots=3, target_margin=0.08, terminal_only=True)
    initial, meta = p.reference_seed(REFERENCE, bound=0.5)
    assert initial.shape == (12,)
    assert np.isfinite(initial).all() and np.max(abs(initial)) <= 0.5
    assert meta["fit_rms_rad"] >= 0
    _, commands = p.commands(initial)
    np.testing.assert_array_equal(commands[0], p.saved["targets"][0])
    invalid = tmp_path / "wrong_reference.npz"
    invalid.write_bytes(b"wrong release asset")
    with pytest.raises(ValueError, match="SHA256"):
        p.reference_seed(invalid)
