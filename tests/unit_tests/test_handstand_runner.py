"""Cornell hooks exercised without constructing a robot/SDK controller."""

import ast
from pathlib import Path
import os
import json
import threading
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

from go2_deploy.custom_implementation import go2handstand as team
from go2_deploy.utility import deploy_utility, import_custom_impl
from go2_deploy.state import State


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    ckpt = Path(
        os.environ.get(
            "GO2_HANDSTAND_CHECKPOINT", "output/capture_ppo_conservative/latest.pt"
        )
    )
    ref = Path(
        os.environ.get(
            "GO2_HANDSTAND_REFERENCE", "artifacts/handstand_reference/trajectory.npz"
        )
    )
    if not ckpt.exists() or not ref.exists():
        pytest.skip("Supply pinned handstand release assets")
    folder = tmp_path_factory.mktemp("handstand") / "bundle"
    team.export_runtime_bundle(ckpt, ref, folder)
    policy, trajectory = team.load_assets(ckpt, ref)
    model = team.build_fk_model()
    d = __import__("mujoco").MjData(model)
    __import__("mujoco").mj_setState(
        model,
        d,
        trajectory["initial_state"],
        __import__("mujoco").mjtState.mjSTATE_INTEGRATION,
    )
    calibration = {
        "calibrated": True,
        "initial_height": float(d.qpos[2]),
        "tick_period_s": 0.001,
        "imu": {
            "body_from_imu": np.eye(3).tolist(),
            "world_alignment": np.eye(3).tolist(),
            "lever_arm_body": [0.0, 0.0, 0.0],
            "accel_bias_imu": [0.0, 0.0, 0.0],
            "gyro_bias_imu": [0.0, 0.0, 0.0],
        },
    }
    path = folder / "synthetic_calibration.json"
    path.write_text(json.dumps(calibration))
    return folder, policy, trajectory, model, calibration, path


@pytest.fixture
def controller(assets, monkeypatch):
    folder, policy, trajectory, model, calibration, _ = assets
    cfg = NS(
        ctrl_freq=50,
        kp=60.0,
        kd=3.0,
        exp_moving_avg=False,
        default_dof_pos=torch.tensor(trajectory["nominal"], dtype=torch.float32),
        obs_vector=list(team.SLICES),
        obs_sizes=team.CustomDeployConfig.custom_obs_sizes,
        DeployScaling=team.CustomDeployConfig.DeployScaling,
        custom_runtime=(policy, trajectory, model, calibration),
    )
    c = NS(
        cfg=cfg,
        _state=State.INTERMEDIATE,
        obs_vec_size=47,
        last_lowstate_msg=None,
        last_lowstate_msg_lock=threading.Lock(),
        last_action=torch.zeros(2, 12),
        kp_mult=1.0,
        kd_mult=1.0,
        lowcmd_publisher=object(),
    )
    for k, v in team.get_custom_buffers_dict().items():
        setattr(c, k, v)
    team.initialize_controller(c)
    # A real software publication in INTERMEDIATE establishes known history.
    target = trajectory["initial_target"].astype(np.float32).astype(float)
    cmd = NS(motor_cmd=[NS(q=float(x)) for x in target[team.PERMUTATION]])
    team.on_command_sent(c, cmd, True, 0.0)
    c._state = State.CUSTOM_CTRL
    monkeypatch.setattr(deploy_utility, "custom_impl", team)
    return c


def message(c, tick, quaternion=None):
    q = c.handstand_session.path.initial.copy()
    return NS(
        tick=tick,
        imu_state=NS(
            quaternion=[1.0, 0.0, 0.0, 0.0] if quaternion is None else quaternion,
            gyroscope=[0.0, 0.0, 0.0],
            accelerometer=[0.0, 0.0, 9.81],
        ),
        motor_state=[NS(q=float(x), dq=0.0) for x in q[team.PERMUTATION]],
        wireless_remote=b"",
    )


def receive(c, tick, now):
    msg = message(c, tick)
    with c.last_lowstate_msg_lock:
        c.last_lowstate_msg = msg
        c.last_lowstate_received_time = now
        team.on_lowstate_msg(c, msg, now)


def start(c):
    receive(c, 1000, 10.0)
    team.update_custom_buffers(c, 10.0)


def publish(c, accepted=True):
    obs = deploy_utility.lowstate_to_obs(c, c.last_lowstate_msg)
    with torch.no_grad():
        action = c.handstand_session.policy(obs) * 2
    c.default_lowcmd = NS(motor_cmd=[NS(q=0.0, kp=0.0, kd=0.0) for _ in range(20)])
    cmd = deploy_utility.action_to_lowcmd(c, action)
    team.on_command_sent(c, cmd, accepted, 10.0)
    return obs, cmd


def test_actual_lowstate_to_47_observations_and_bounded_publication(controller):
    c = controller
    start(c)
    previous = c.handstand_last_sent.copy()
    obs, cmd = publish(c)
    assert obs.shape == (47,) and obs.dtype == torch.float32
    np.testing.assert_allclose(obs[:3], [0, 0, -1])
    np.testing.assert_allclose(
        obs[30:42], (previous - c.handstand_session.path.nominal) / 2, atol=1e-7
    )
    target = np.array([m.q for m in cmd.motor_cmd[:12]])[team.PERMUTATION]
    np.testing.assert_array_equal(target, c.handstand_session.path.target)
    np.testing.assert_array_equal(target, target.astype(np.float32).astype(float))
    assert c.handstand_session.path.step == 1
    assert np.all(abs(target - previous) <= 0.7 * team.SPEED * 0.02)


def test_every_imu_interval_once_and_phase_advance_only_after_send(controller):
    c = controller
    start(c)
    publish(c)
    for i in range(1, 11):
        receive(c, 1000 + 2 * i, 10.0 + i * 0.002)
    team.update_custom_buffers(c, 10.02)
    assert c.handstand_session.estimator.samples == 10
    assert c.handstand_session.path.step == 1
    assert c.handstand_snapshot[42] == pytest.approx(0.02 / 8)
    assert not c.handstand_rx
    # A duplicate receipt does not integrate a second time or refresh unique time.
    receive(c, 1020, 10.021)
    assert c.handstand_session.estimator.samples == 10


@pytest.mark.parametrize(
    "failure",
    ["stale", "gap", "nan", "unpublished", "publish_false", "publish_none", "gain"],
)
def test_invalid_runner_state_does_not_advance_phase(controller, failure):
    c = controller
    start(c)
    if failure == "stale":
        with pytest.raises(ValueError, match="Stale"):
            team.update_custom_buffers(c, 10.01)
    elif failure == "gap":
        publish(c)
        receive(c, 1020, 10.02)
        with pytest.raises(ValueError, match="Missing"):
            team.update_custom_buffers(c, 10.02)
    elif failure == "nan":
        bad = message(c, 1002, [float("nan"), 0, 0, 0])
        team.on_lowstate_msg(c, bad, 10.002)
        c.last_lowstate_msg = bad
        c.last_lowstate_received_time = 10.002
        with pytest.raises(ValueError):
            team.update_custom_buffers(c, 10.002)
    elif failure == "unpublished":
        team.bound_command(c, torch.zeros(12))
        for i in range(1, 11):
            receive(c, 1000 + 2 * i, 10.0 + i * 0.002)
        with pytest.raises(ValueError, match="not published"):
            team.update_custom_buffers(c, 10.02)
    elif failure == "gain":
        with pytest.raises(ValueError, match="gains"):
            team.bound_command(c, torch.zeros(12), kp_mult=1.1)
    else:
        with pytest.raises(ValueError, match="publication"):
            publish(c, False if failure == "publish_false" else None)
    assert c.handstand_session.path.step == (1 if failure == "gap" else 0)


def test_tick_rollover_supported_and_duplicate_not_fresh(controller):
    c = controller
    receive(c, 2**32 - 2, 10.0)
    team.update_custom_buffers(c, 10.0)
    publish(c)
    for i in range(1, 11):
        receive(c, (2**32 - 2 + 2 * i) % (2**32), 10.0 + i * 0.002)
    team.update_custom_buffers(c, 10.02)
    assert c.handstand_session.estimator.samples == 10
    publish(c)
    receive(c, 18, 10.04)
    with pytest.raises(ValueError, match="fresh"):
        team.update_custom_buffers(c, 10.04)


def test_exit_and_reentry_reset_phase_and_cancel_pending(controller):
    c = controller
    start(c)
    publish(c)
    c._state = State.INTERMEDIATE
    receive(c, 1002, 10.002)
    team.update_custom_buffers(c, 10.002)
    assert not c.handstand_session.active
    c._state = State.CUSTOM_CTRL
    receive(c, 1004, 10.004)
    team.update_custom_buffers(c, 10.004)
    assert c.handstand_session.path.step == 0
    assert c.handstand_session.pending is None


def test_bundle_loads_through_cornell_loader_and_checks_calibration(
    assets, monkeypatch
):
    from go2_deploy.policy_bundle import load_bundle

    folder, _, _, _, _, calibration = assets
    loaded = import_custom_impl.get_custom_impl_module("go2handstand.py")
    monkeypatch.setattr(import_custom_impl, "custom_impl", loaded)
    monkeypatch.setenv("GO2_HANDSTAND_CALIBRATION", str(calibration))
    policy, cfg = load_bundle(folder)
    assert cfg.obs_dim() == 47 and cfg.ctrl_freq == 50 and not cfg.exp_moving_avg
    assert cfg.kp == 60 and cfg.kd == 3
    assert policy(torch.zeros(47)).shape == (12,)
    monkeypatch.delenv("GO2_HANDSTAND_CALIBRATION")
    with pytest.raises(ValueError, match="CALIBRATION"):
        load_bundle(folder)


def test_template_loader_rejects_traversal():
    with pytest.raises(ValueError):
        import_custom_impl.get_custom_impl_module("../deploy.py")


def extracted_main(clock=None):
    # Compile the exact MainController class, not its SDK imports/constructor.
    path = Path("go2_deploy/main_controller.py")
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MainController"
    )
    scope = {
        "torch": torch,
        "time": time if clock is None else NS(monotonic=lambda: clock),
        "threading": threading,
        "State": State,
        "custom_impl": team,
        "deploy_utility": deploy_utility,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["MainController"]


def test_exact_runner_callback_delivers_stream_and_fault_stops_before_publish(
    controller,
):
    cls = extracted_main()
    c = controller
    c.remote_controller = NS(parse=lambda _: None)
    c.rc_handler = NS(_process_input=lambda: None)
    c.csv_logger = NS(log_lowstate=lambda *_: None)
    c.lowstate_obs_count = 0
    cls._on_lowstate_msg(c, message(c, 1000))
    assert c.handstand_rx_seen and len(c.handstand_rx) == 1
    calls = []
    c._policy_control_cycle = lambda: (_ for _ in ()).throw(ValueError("stale"))
    c.request_emergency_stop = lambda: calls.append("estop")
    cls._control_loop(c)
    assert calls == ["estop"]


@pytest.mark.parametrize("accepted", [True, False])
def test_exact_control_cycle_uses_runner_telemetry_and_publisher_result(
    controller, accepted
):
    cls = extracted_main(clock=10.0)
    c = controller
    # No MainController.__init__, SDK clients or real publisher are constructed.
    receive(c, 1000, 10.0)
    c.last_command_lock = threading.Lock()
    c.last_sportmodestate_msg_lock = threading.Lock()
    c.last_obs_lock = threading.Lock()
    c.last_action_lock = threading.Lock()
    c.last_sportmodestate_msg = None
    c.last_obs = torch.zeros(47)
    c.last_command = torch.zeros(3)
    c.remote_controller = NS(lin_vel_x=0.0, lin_vel_y=0.0, yaw_vel=0.0)
    c._process_custom_buffers = lambda t: cls._process_custom_buffers(c, t)
    c.msg_to_obs = lambda msg: deploy_utility.lowstate_to_obs(c, msg)
    from go2_deploy.rl_controller import RLController

    actor = RLController(c.cfg, c.handstand_session.policy)
    c._act = actor.act
    c.default_lowcmd = NS(motor_cmd=[NS(q=0.0, kp=0.0, kd=0.0) for _ in range(20)])
    c.action_to_lowcmd = lambda action: deploy_utility.action_to_lowcmd(c, action)
    writes = []
    logs = []
    fault = []
    c.lowcmd_publisher = NS(Write=lambda cmd: writes.append(cmd) or accepted)
    c.crc = NS(Crc=lambda _: 123)
    c.csv_logger = NS(log_control=lambda *v: logs.append(v))
    c.action_count = 0
    c._policy_control_cycle = lambda: cls._policy_control_cycle(c)
    c.request_emergency_stop = lambda: fault.append(True)
    cls._control_loop(c)
    assert len(writes) == 1
    assert c.handstand_session.path.step == int(accepted)
    assert c.action_count == int(accepted) and len(logs) == int(accepted)
    assert bool(fault) == (not accepted)
    if accepted:
        assert writes[0].crc == 123
        assert all(m.kp == 60 and m.kd == 3 for m in writes[0].motor_cmd[:12])


def test_intermediate_pose_is_absolute_and_keyboard_q_requests_hold(
    controller, monkeypatch
):
    cls = extracted_main()
    c = controller
    c._state = State.RECOVERY
    c.last_lowstate_msg = message(c, 1000)
    c.motion_switcher_client = NS(
        ReleaseMode=lambda: None, CheckMode=lambda: (0, {"name": ""})
    )
    c.lowcmd_thread = NS(Start=lambda: None)
    cls.switch_to_intermediate(c)
    actual = deploy_utility.action_to_target_pos(c, cls._act(c, torch.zeros(47)))
    expected = torch.tensor(
        team.read_lowstate(c.last_lowstate_msg)[4], dtype=actual.dtype
    )
    torch.testing.assert_close(actual, expected)
    tree = ast.parse(Path("go2_deploy/keyboard_handler.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = {}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), "keyboard_handler.py", "exec"
        ),
        scope,
    )
    called = []
    handler = object.__new__(scope["KeyboardHandler"])
    handler.controller = NS(request_intermediate=lambda: called.append(True))
    monkeypatch.setattr("builtins.input", lambda: "q")
    handler._process_input()
    assert called == [True]


def test_bundle_cannot_substitute_a_weights_only_actor(assets, monkeypatch, tmp_path):
    import shutil
    import hashlib
    from go2_deploy.policy_bundle import load_bundle

    folder, *_, calibration = assets
    copied = tmp_path / "bad_bundle"
    shutil.copytree(folder, copied)
    policy = torch.jit.trace(torch.nn.Linear(47, 12), torch.zeros(47))
    torch.jit.save(policy, str(copied / "policy.pt"))
    meta = json.loads((copied / "deploy_meta.json").read_text())
    meta["policy_sha256"] = hashlib.sha256(
        (copied / "policy.pt").read_bytes()
    ).hexdigest()
    (copied / "deploy_meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(import_custom_impl, "custom_impl", team)
    monkeypatch.setenv("GO2_HANDSTAND_CALIBRATION", str(calibration))
    with pytest.raises(ValueError, match="complete pinned"):
        load_bundle(copied)


def test_viewer_cancellation_is_not_a_completed_trial(assets):
    from scripts.handstand_capture import trial

    _, policy, trajectory, *_ = assets
    closed = []

    class Cancelled:
        frames = 0
        viewer = None

        def sync(self):
            return False

        def capture(self, force=False):
            pass

        def close(self):
            closed.append(True)

    result = trial(policy, trajectory, 301001, True, lambda _: Cancelled())
    assert result["cancelled"] and not result["complete_success"]
    assert result["steps"] == 0 and closed == [True]


def test_presentation_style_does_not_change_simulated_outcome(assets):
    from scripts.handstand_capture import trial, Presentation

    _, policy, trajectory, *_ = assets
    headless = trial(policy, trajectory, 301006, True)
    styled = trial(policy, trajectory, 301006, True, lambda r: Presentation(r))
    assert headless == styled


def test_stream_overflow_and_missing_callback_fail_closed(controller):
    c = controller
    c.last_lowstate_msg = message(c, 1000)
    c.last_lowstate_received_time = 10.0
    with pytest.raises(ValueError, match="callback"):
        team.update_custom_buffers(c, 10.0)
    for i in range(260):
        receive(c, 1000 + 2 * i, 10.0 + i * 0.002)
    with pytest.raises(ValueError, match="overflow"):
        team.update_custom_buffers(c, 10.0 + 259 * 0.002)
    assert c.handstand_session.path.step == 0


def test_activation_waits_for_callback_without_inventing_a_frame(controller):
    c = controller
    c._state = State.INTERMEDIATE
    receive(c, 1000, 10.0)
    c._state = State.CUSTOM_CTRL
    assert team.update_custom_buffers(c, 10.0) is False
    assert not c.handstand_session.active
    receive(c, 1002, 10.002)
    team.update_custom_buffers(c, 10.002)
    assert c.handstand_session.active and c.handstand_session.estimator.samples == 0


def test_conflicting_duplicate_tick_is_rejected(controller):
    c = controller
    start(c)
    publish(c)
    changed = message(c, 1000)
    changed.imu_state.accelerometer[0] = 1.0
    with c.last_lowstate_msg_lock:
        c.last_lowstate_msg = changed
        c.last_lowstate_received_time = 10.02
        team.on_lowstate_msg(c, changed, 10.02)
    with pytest.raises(ValueError, match="Conflicting"):
        team.update_custom_buffers(c, 10.02)


def test_recorded_sensor_messages_match_47_channel_contract(
    controller, assets, record_property
):
    path = Path(
        os.environ.get(
            "GO2_HANDSTAND_RECORDING",
            "docs/validation/go2_handstand/nominal_sensors.npz",
        )
    )
    if not path.exists():
        pytest.skip("Provide pinned simulated sensor recording for parity")
    c = controller
    with np.load(path, allow_pickle=False) as f:
        samples, controls = f["samples"], f["controls"]
    policy = assets[1]
    j = 0
    errors = {"observation": 0.0, "action": 0.0, "target": 0.0}

    def packet(timestamp, quaternion, gyro, force, q, dq):
        return NS(
            tick=int(round(timestamp * 1000)) % (2**32),
            imu_state=NS(
                quaternion=quaternion.copy(),
                gyroscope=gyro.copy(),
                accelerometer=np.array(force),
            ),
            motor_state=[
                NS(q=float(a), dq=float(b))
                for a, b in zip(q[team.PERMUTATION], dq[team.PERMUTATION])
            ],
        )

    for i, row in enumerate(controls):
        now = float(row[0])
        incoming = []
        if i == 0:
            incoming.append(
                (
                    0.0,
                    packet(
                        0.0, row[1:5], row[5:8], [0, 0, 9.81], row[8:20], row[20:32]
                    ),
                )
            )
        while j < len(samples) and samples[j, 0] <= now + 1e-9:
            s = samples[j]
            quaternion, force = s[1:5], s[5:8]
            if abs(float(s[0]) - now) < 1e-9:
                # The legacy recording separates pre-step IMU orientation from
                # post-step control orientation. A LowState has one orientation;
                # re-express the same world acceleration in the current body frame.
                quaternion = row[1:5]
                force = team.rotation(quaternion).T @ team.rotation(s[1:5]) @ force
            incoming.append(
                (
                    float(s[0]),
                    packet(s[0], quaternion, row[5:8], force, row[8:20], row[20:32]),
                )
            )
            j += 1
        for t, msg in incoming:
            with c.last_lowstate_msg_lock:
                c.last_lowstate_msg = msg
                c.last_lowstate_received_time = 10.0 + t
                team.on_lowstate_msg(c, msg, 10.0 + t)
        team.update_custom_buffers(c, 10.0 + now)
        obs = deploy_utility.lowstate_to_obs(c, c.last_lowstate_msg)
        diff = abs(obs.numpy() - row[45:92])
        errors["observation"] = max(errors["observation"], float(diff.max()))
        with torch.no_grad():
            action = policy(obs)
            oracle = policy(torch.tensor(row[45:92], dtype=torch.float32))
        errors["action"] = max(errors["action"], float(abs(action - oracle).max()))
        target = team.bound_command(c, action * 2).numpy()
        if i + 1 < len(controls):
            errors["target"] = max(
                errors["target"], float(abs(target - controls[i + 1, 32:44]).max())
            )
        cmd = NS(motor_cmd=[NS(q=float(x)) for x in target[team.PERMUTATION]])
        team.on_command_sent(c, cmd, True, 10.0 + now)
        assert c.handstand_session.path.step == i + 1
    assert (
        errors["observation"] < 1e-6
        and errors["action"] < 2e-6
        and errors["target"] < 2e-6
    ), errors
    for name, value in errors.items():
        record_property("maximum_" + name + "_error", value)
    print("RUNNER_PARITY", json.dumps(errors))
