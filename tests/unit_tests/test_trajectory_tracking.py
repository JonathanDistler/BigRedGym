"""Native trajectory, inference, and mocked Cornell transport contracts."""

from pathlib import Path
from types import SimpleNamespace as NS
import json
import hashlib
import os
import threading
import mujoco
import numpy as np
import pytest
import torch
from experiments.trajectory_optimization.model import Robot, REFERENCE_SHA
from experiments.trajectory_optimization.staged import (
    Preparation,
    wire_target,
    evaluate,
)
from experiments.trajectory_optimization.feedback import fit
from gym.envs.go2.handstand_replay import load_trajectory, STATE_SPEC
from go2_deploy.custom_implementation import go2trajectory_lqr as deploy


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    ref = Path(
        os.environ.get(
            "GO2_HANDSTAND_TRAJECTORY", "artifacts/handstand_reference/trajectory.npz"
        )
    )
    if not ref.exists():
        pytest.skip("Provide pinned release reference")
    policy, report = fit(ref, start=110, spacing=10, reduced=False)
    policy.gains.mul_(0.4)
    r = Robot()
    trajectory = load_trajectory(ref, REFERENCE_SHA, r.m)
    d = mujoco.MjData(r.m)
    mujoco.mj_setState(r.m, d, trajectory["initial_state"], STATE_SPEC)
    folder = tmp_path_factory.mktemp("trajectory-lqr")
    torch.jit.save(torch.jit.script(policy), str(folder / "policy.pt"))
    meta = dict(
        kind="native_trajectory_lqr_v1",
        num_obs=43,
        nominal_validation_pass=True,
        default_dof_pos=trajectory["nominal"].tolist(),
        joint_bounds=r.bounds.tolist(),
        initial_q=d.qpos.tolist(),
        initial_v=d.qvel.tolist(),
        scaling={k: 1 for k in deploy.SLICES} | {"dof_pos_target": 2},
        policy_sha256=hashlib.sha256((folder / "policy.pt").read_bytes()).hexdigest(),
    )
    calibration = dict(
        calibrated=True,
        source="SYNTHETIC TEST ONLY",
        imu=dict(
            body_from_imu=np.eye(3).tolist(),
            world_alignment=np.eye(3).tolist(),
            gyro_bias_imu=[0, -0.009, 0],
        ),
    )
    (folder / "calibration.json").write_text(json.dumps(calibration))
    cfg = NS(
        ctrl_freq=50,
        kp=60,
        kd=3,
        exp_moving_avg=False,
        bundle_dir=str(folder),
        bundle_meta=meta,
        obs_vector=list(deploy.SLICES),
        default_dof_pos=torch.tensor(trajectory["nominal"], dtype=torch.float32),
        loaded_policy=torch.jit.load(str(folder / "policy.pt")),
    )
    cfg.custom_runtime = deploy.prepare_runtime(cfg)
    return ref, trajectory, policy, cfg


@pytest.fixture
def c(model):
    cfg = model[3]
    c = NS(
        cfg=cfg,
        _state=NS(name="INTERMEDIATE"),
        last_lowstate_msg_lock=threading.Lock(),
        last_action=torch.zeros(2, 12),
        kp_mult=1.0,
        kd_mult=1.0,
        lowcmd_publisher=object(),
        **deploy.get_custom_buffers_dict(),
    )
    deploy.initialize_controller(c)
    q = model[1]["initial_target"].astype(np.float32)
    command = NS(motor_cmd=[NS(q=float(x)) for x in q[deploy.ORDER]])
    deploy.on_command_sent(c, command, True, 10.0)  # Explicit software publisher only.
    c._state = NS(name="CUSTOM_CTRL")
    runtime = c.trajectory_runtime
    msg = NS(
        tick=1000,
        imu_state=NS(
            quaternion=runtime["initial_q"][3:7].copy(),
            gyroscope=runtime["initial_v"][3:6] + [0, -0.009, 0],
            accelerometer=[np.nan] * 3,
        ),
        motor_state=[
            NS(q=float(q), dq=float(v))
            for q, v in zip(
                runtime["initial_q"][7:][deploy.ORDER],
                runtime["initial_v"][6:][deploy.ORDER],
            )
        ],
    )
    c.last_lowstate_msg = msg
    deploy.on_lowstate_msg(c, msg, 10.0)
    return c


def test_nominal_second_controller_completes_strict_hold(model):
    result = evaluate(model[1], policy=model[2], adapter_dry_run=True)
    assert result["maximum_adapter_parity_errors"]["observation"] < 1e-6
    assert result["maximum_adapter_parity_errors"]["target"] < 1e-6
    assert result["complete_success"] and result["strict_hold_seconds"] >= 5
    assert result["illegal_contacts"] == 0 and result["max_joint_limit_excess_rad"] == 0
    assert min(result["minimum_calf_clearance_m"]) > 0.06


@pytest.mark.parametrize("step", [0, 1, 47, 109, 110, 111, 319, 320, 400])
def test_complete_actor_script_phase_and_feedback_parity(model, step):
    policy = model[2]
    script = model[3].loaded_policy
    obs = torch.zeros(43)
    obs[:30] = policy.centers[min(step, 319)] + torch.linspace(-0.003, 0.003, 30)
    obs[42] = step / 400
    torch.testing.assert_close(policy(obs), script(obs), rtol=0, atol=1e-7)
    if step >= 319:
        obs[42] = 319 / 400
        torch.testing.assert_close(
            policy(obs),
            script(torch.cat((obs[:42], torch.tensor([step / 400])))),
            rtol=0,
            atol=1e-7,
        )
    assert (abs(policy(obs) - policy.actions[min(step, 319)]) <= 0.05 + 1e-7).all()


def test_actor_rejects_original_capture_observation_size(model):
    with pytest.raises(ValueError, match="43"):
        model[2](torch.zeros(47))


def test_wire_rounding_is_inward_at_position_and_slew_bounds(model):
    bounds = np.asarray(model[3].bundle_meta["joint_bounds"])
    previous = model[1]["initial_target"]
    for desired in [previous + 100, previous - 100]:
        actual = wire_target(desired, previous, bounds, deploy.SPEED)
        assert np.array_equal(actual, actual.astype(np.float32).astype(float))
        assert (actual >= bounds[:, 0]).all() and (actual <= bounds[:, 1]).all()
        assert (abs(actual - previous) <= 0.7 * deploy.SPEED * 0.02).all()


def test_real_hook_layout_target_mapping_and_publication(c, model):
    deploy.update_custom_buffers(c, 10.0)
    obs = torch.cat(
        [
            deploy.get_custom_obs_pieces()[name](c, c.last_lowstate_msg)
            for name in deploy.SLICES
        ]
    )
    assert obs.shape == (43,) and obs.dtype == torch.float32
    np.testing.assert_allclose(obs[:30], model[2].centers[0].numpy(), atol=1e-7)
    with torch.no_grad():
        action = model[3].loaded_policy(obs)
    target = deploy.bound_command(c, action * 2).numpy()
    expected = wire_target(
        model[1]["nominal"] + action.numpy() * 2,
        c.trajectory_applied,
        np.asarray(c.cfg.bundle_meta["joint_bounds"]),
        deploy.SPEED,
    )
    np.testing.assert_array_equal(target, expected)
    assert c.trajectory_step == 0
    deploy.on_command_sent(
        c, NS(motor_cmd=[NS(q=float(x)) for x in target[deploy.ORDER]]), True, 10.0
    )
    assert c.trajectory_step == 1
    np.testing.assert_array_equal(c.trajectory_applied, target)


@pytest.mark.parametrize(
    "failure",
    [
        "nan",
        "stale",
        "frozen",
        "wrong_pose",
        "unpublished",
        "cancel",
        "gain",
        "unsafe_history",
    ],
)
def test_bad_sensor_or_control_state_latches_and_requires_reset(c, failure):
    if failure == "nan":
        c.last_lowstate_msg.imu_state.gyroscope[0] = np.nan
    elif failure == "stale":
        pass
    elif failure == "frozen":
        deploy.on_lowstate_msg(c, c.last_lowstate_msg, 10.02)
    elif failure == "wrong_pose":
        c.last_lowstate_msg.motor_state[0].q += 0.1
    elif failure == "unsafe_history":
        c.trajectory_applied[0] += 1
    elif failure == "unpublished":
        c.trajectory_pending = np.zeros(12)
    elif failure == "cancel":
        c.trajectory_cancel_requested = True
    elif failure == "gain":
        deploy.update_custom_buffers(c, 10.0)
        c.kp_mult = 2
        with pytest.raises(ValueError):
            deploy.bound_command(c, torch.zeros(12))
        assert not c.trajectory_active
        return
    with pytest.raises(ValueError):
        deploy.update_custom_buffers(
            c, 10.02 if failure in ["stale", "frozen"] else 10.0
        )
    assert not c.trajectory_active and c.trajectory_observation is None
    with pytest.raises(ValueError, match="reset"):
        deploy.update_custom_buffers(c, 10.02)
    c._state = NS(name="INTERMEDIATE")
    deploy.update_custom_buffers(c, 10.02)
    assert not c.trajectory_fault_latched


def test_10_6ms_sensor_age_is_accepted_without_acceleration_integration(c):
    deploy.update_custom_buffers(c, 10.0106)
    assert c.trajectory_active and c.trajectory_observation.shape == (43,)
    assert not hasattr(c, "imu_observer")


def test_missing_or_failed_publication_does_not_advance(c):
    deploy.update_custom_buffers(c, 10.0)
    target = deploy.bound_command(c, torch.zeros(12)).numpy()
    with pytest.raises(ValueError, match="publication"):
        deploy.on_command_sent(
            c, NS(motor_cmd=[NS(q=float(x)) for x in target[deploy.ORDER]]), False, 10.0
        )
    assert c.trajectory_step == 0 and not c.trajectory_active


def test_calibration_preset_loads_but_activation_requires_confirmation(c):
    c.trajectory_runtime = {
        **c.trajectory_runtime,
        "calibration": {**c.trajectory_runtime["calibration"], "calibrated": False},
    }
    with pytest.raises(ValueError, match="calibration"):
        deploy.update_custom_buffers(c, 10.0)


def test_command_snapshot_age_and_bounds(c):
    deploy.update_custom_buffers(c, 10.0)
    c.lowstate_clock = lambda: 10.02001
    with pytest.raises(ValueError, match="stale"):
        deploy.bound_command(c, torch.zeros(12))


def test_preparation_fresh_forward_kinematics_and_pulse_start(model):
    p = Preparation(model[0], duration=0.4, settle=0)
    r = p.robot
    r.set(p.goal_q)
    assert np.max(abs(r.foot_kinematics()[0][:, 2])) < 0.003
    assert p.start[2] == 0.27 and p.goal_q[2] > 0.34
    assert p.goal_v[4] == pytest.approx(0.00616877, abs=1e-8)
    zero = p.commands(np.zeros(36))
    legacy = p.commands(np.zeros(24))
    np.testing.assert_array_equal(zero, legacy)
    pulse = np.r_[np.zeros(24), np.full(12, 0.05)]
    np.testing.assert_array_equal(p.commands(pulse)[0], zero[0])
    assert (
        abs(np.diff(p.commands(pulse), axis=0)) <= 0.7 * deploy.SPEED * 0.02 + 1e-9
    ).all()
    with pytest.raises(ValueError):
        p.commands(np.zeros(8))


@pytest.mark.parametrize("failure", ["kind", "scale", "order", "validation"])
def test_bundle_metadata_cannot_silently_change_controller_contract(model, failure):
    cfg = NS(**vars(model[3]))
    cfg.bundle_meta = {**cfg.bundle_meta}
    if failure == "kind":
        cfg.bundle_meta["num_obs"] = 47
    elif failure == "scale":
        cfg.bundle_meta["scaling"] = {**cfg.bundle_meta["scaling"], "dof_pos_target": 1}
    elif failure == "order":
        cfg.obs_vector = list(reversed(cfg.obs_vector))
    else:
        cfg.bundle_meta["nominal_validation_pass"] = False
    with pytest.raises(ValueError):
        deploy.prepare_runtime(cfg)


def test_viewer_close_cancels_without_claiming_success_or_leaking_renderer(model):
    closed = []

    class ClosedView:
        viewer = None

        def __init__(self, robot):
            pass

        def sync(self):
            return False

        def capture(self, **kwargs):
            pass

        def close(self):
            closed.append(True)

    result = evaluate(model[1], policy=model[2], presentation=ClosedView)
    assert result["cancelled"] and not result["complete_success"]
    assert result["failure_reason"] == "cancelled" and result["seconds"] == 0
    assert result["minimum_calf_clearance_m"] == [None] * 4
    assert closed == [True]
