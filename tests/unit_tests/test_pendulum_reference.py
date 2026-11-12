"""Sign conventions, reference endpoints, strict sustained-control gate."""

import numpy as np
import pytest

from experiments.morphology.learning.pendulum_reference import (
    ReferenceGate,
    acceleration,
    quintic,
    score_trajectory,
)


def sample(time, pitch=45.0, speed=0.0, contact=True):
    return dict(
        time=time,
        pitch=np.deg2rad(pitch),
        pitch_velocity=speed,
        FL_load=80.0 if contact else 0.0,
        FR_load=80.0 if contact else 0.0,
        FL_velocity_x=0.0,
        FL_velocity_y=0.0,
        FR_velocity_x=0.0,
        FR_velocity_y=0.0,
        nonfoot_collision=False,
        self_collision=False,
        joint_violation=0.0,
        max_speed_ratio=0.0,
        torque_ratio=0.5,
        front_drift=0.0,
    )


def test_upward_convention_matches_equation():
    theta = -0.6
    mass = 15.0
    length = 0.35
    assert acceleration(theta, 0, length, 0, mass) == pytest.approx(
        9.81 / length * np.sin(theta)
    )
    assert acceleration(0, 0, length, 0, mass, torque=1) > 0
    assert acceleration(
        theta, 0, length, 0, mass, torque=-mass * 9.81 * length * np.sin(theta)
    ) == pytest.approx(0)


def test_radial_and_configuration_inertia_terms():
    mass = 10.0
    length = 0.4
    omega = 1.2
    assert acceleration(0, omega, length, -0.1, mass, gravity=0) == pytest.approx(0.6)
    assert acceleration(
        0, omega, length, 0, mass, inertia_com=0.5, inertia_rate=0.2, gravity=0
    ) == pytest.approx(-0.2 * omega / (mass * length**2 + 0.5))
    with pytest.raises(ValueError):
        acceleration(0, 0, 0, 0, mass)


def test_reference_matches_initial_and_controlled_terminal_state():
    times = np.array([0.0, 0.3, 0.6])
    p, v, a = quintic(0.52, 1.2, np.pi / 4, 0.3, times)
    np.testing.assert_allclose(p, [0.52, np.pi / 4, np.pi / 4])
    np.testing.assert_allclose(v, [1.2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(a, [0, 0, 0], atol=1e-10)


def test_gate_rejects_peak_without_low_speed_hold():
    trace = [sample(t, pitch=45.0, speed=2.0) for t in np.arange(0.002, 0.602, 0.002)]
    result = score_trajectory(trace, 160.0)
    assert not result["success"] and result["hold_seconds"] == 0
    assert result["peak_pitch"] == pytest.approx(np.pi / 4)


def test_gate_requires_reaching_45_not_just_entering_band():
    trace = [sample(t, pitch=43.0) for t in np.arange(0.002, 0.602, 0.002)]
    result = score_trajectory(trace, 160.0)
    assert result["hold_seconds"] > 0.3 and not result["success"]


def test_gate_requires_contact_native_limits_and_low_slip():
    trace = [sample(t) for t in np.arange(0.002, 0.602, 0.002)]
    assert score_trajectory(trace, 160.0)["success"]
    bad = [dict(r) for r in trace]
    for r in bad[100:120]:
        r["FL_load"] = 0
    assert not score_trajectory(bad, 160.0)["success"]
    bad = [dict(r) for r in trace]
    bad[-1]["nonfoot_collision"] = True
    assert not score_trajectory(bad, 160.0)["success"]
    bad = [dict(r) for r in trace]
    bad[-1]["max_speed_ratio"] = 1.01
    assert not score_trajectory(bad, 160.0)["success"]
    bad = [dict(r, FL_velocity_x=0.1) for r in trace]
    assert not score_trajectory(bad, 160.0)["success"]


def test_gate_threshold_is_eight_of_ten_not_one_lucky_case():
    gate = ReferenceGate()
    assert gate.required_fraction == 0.8
    assert 1 / 10 < gate.required_fraction


def test_reference_preserves_com_rate_independently_of_body_rate():
    from experiments.morphology.learning.pendulum_models import reference

    initial = dict(
        pitch=0.53,
        pitch_velocity=1.5,
        theta=-0.5,
        omega=-0.2,
        length=0.35,
        length_rate=-0.1,
        inertia_com_y=0.5,
    )
    model = dict(mass=15.0, reference_com_body_gain=0.1)
    ref = reference(initial, model)
    assert ref.omega.iloc[0] == pytest.approx(-0.2)
    assert ref.pitch_velocity.iloc[0] == pytest.approx(1.5)
    assert ref.length_rate.iloc[0] == pytest.approx(-0.1)
    assert ref.omega.iloc[-1] == pytest.approx(0.0, abs=1e-10)


def test_failed_gate_blocks_learning_before_demonstration_loading(tmp_path):
    import json
    from scripts.learn_pendulum_reference import validated_gate

    (tmp_path / "trajectory_gate.json").write_text(json.dumps(dict(passed=False)))
    assert not validated_gate(tmp_path)


def test_reference_curriculum_never_advances_on_takeoff_alone():
    from scripts.learn_pendulum_reference import ReferenceSchedule

    stage = ReferenceSchedule("B", {})
    for i in range(64):
        stage.completed(
            dict(reference_stage=0, takeoff=True, reference_success=False), 2048 + i
        )
    assert stage.stage == 0
    for i in range(64):
        stage.completed(
            dict(reference_stage=0, takeoff=True, reference_success=True), 4096 + i
        )
    assert stage.stage == 1


def test_cpu_task_registration_and_compatibility(tmp_path):
    from experiments.morphology.learning.environment import Assets
    from gym.envs.go2.go2kickup_pendulum import register_cpu_task

    task, config = register_cpu_task()["go2kickup_pendulum"]
    env = task(Assets(tmp_path / "assets"), config(reference_stage=3))
    actor, critic = env.reset(1.0, 42)
    assert actor.shape == (44,) and critic.shape == (74,)
    assert env.cfg.action_scale == 2 and env.cfg.kp == 60 and env.cfg.kd == 3
    _, reward, _, _ = env.step(np.zeros(12))
    assert np.isfinite(reward)


def test_forecasts_do_not_read_future_measured_state():
    import pandas as pd
    from experiments.morphology.learning.pendulum_models import (
        validate,
        MODELS,
        FEATURES,
    )

    d = pd.DataFrame(
        dict(
            time=np.arange(61) * 0.002,
            theta=-0.5,
            omega=0.3,
            length=0.35,
            length_rate=0.0,
            inertia_com_y=0.5,
            inertia_rate=0.0,
            fixed_length=0.35,
            pivot_term=0.0,
            theta_acceleration=-0.1,
            eligible=True,
            **{f"command_{i}": 0.0 for i in range(6)},
        )
    )
    regression = dict(
        mean=[0.0] * len(FEATURES),
        scale=[1.0] * len(FEATURES),
        coef=[0.0] * (len(FEATURES) + 1),
    )
    model = dict(
        mass=15.0,
        torques={name: regression for name in MODELS},
        radial=regression,
        inertia_rate=regression,
    )
    modified = d.copy()
    modified.loc[1:, "theta"] += 0.1
    modified.loc[1:, "length"] += 0.05
    _, a, _ = validate([("original", d)], model)
    _, b, _ = validate([("changed", modified)], model)
    a = pd.DataFrame(a)
    b = pd.DataFrame(b)
    a = a[a.start_time == 0]
    b = b[b.start_time == 0]
    np.testing.assert_array_equal(a.theta_predicted, b.theta_predicted)
    np.testing.assert_array_equal(a.length_predicted, b.length_predicted)
