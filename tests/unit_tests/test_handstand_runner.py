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
        custom_impl=team.CustomDeployConfig(),
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
    assets, monkeypatch, tmp_path
):
    from go2_deploy.policy_bundle import load_bundle

    folder, _, _, _, _, calibration = assets
    loaded = import_custom_impl.load_custom_impl("go2handstand")
    assert isinstance(loaded, team.CustomDeployConfig.__bases__[0])
    monkeypatch.setenv("GO2_HANDSTAND_CALIBRATION", str(calibration))
    policy, cfg = load_bundle(folder)
    assert cfg.obs_dim() == 47 and cfg.ctrl_freq == 50 and not cfg.exp_moving_avg
    assert cfg.kp == 60 and cfg.kd == 3
    assert policy(torch.zeros(47)).shape == (12,)
    monkeypatch.delenv("GO2_HANDSTAND_CALIBRATION")
    _, model_cfg = load_bundle(folder)
    assert model_cfg.custom_runtime[3]["calibrated"] is False
    assert model_cfg.custom_runtime[3]["source"] == "model_and_ta_estimates"
    candidate = dict(model_cfg.custom_runtime[3], source="staff_log_candidate")
    candidate_file = tmp_path / "calibration.candidate.json"
    candidate_file.write_text(json.dumps(candidate))
    monkeypatch.setenv("GO2_HANDSTAND_CALIBRATION", str(candidate_file))
    _, candidate_cfg = load_bundle(folder)
    assert candidate_cfg.custom_runtime[3]["calibrated"] is False


def test_template_loader_rejects_unknown_task():
    with pytest.raises(LookupError):
        import_custom_impl.load_custom_impl("../deploy.py")


def test_canvas_config_is_discovered_once_and_buffers_are_independent(assets):
    from go2_deploy.custom_implementation.custom_config import CustomConfig
    from go2_deploy.deploy_config import DeployConfig

    first = import_custom_impl.load_custom_impl("go2handstand")
    second = import_custom_impl.load_custom_impl("go2handstand")
    assert isinstance(first, CustomConfig)
    assert first.task_name == "go2handstand"
    assert first.obs_vector == list(team.SLICES)
    assert sum(first.custom_obs_sizes.values()) == 47
    np.testing.assert_array_equal(first.default_dof_pos, assets[2]["nominal"])
    cfg = DeployConfig(first)
    assert cfg.obs_dim() == 47
    assert list(first.get_custom_obs_pieces()) == list(team.SLICES)
    a, b = first.get_custom_buffers_dict(), second.get_custom_buffers_dict()
    a["handstand_imu_queue"].append("sample")
    assert not b["handstand_imu_queue"]


def test_canvas_instance_hooks_process_actual_runner_messages(controller):
    c = controller
    impl = c.cfg.custom_impl
    impl.on_lowstate_msg(c, message(c, 1000), 10.0)
    c.last_lowstate_msg = message(c, 1000)
    c.last_lowstate_received_time = 10.0
    impl.update_custom_buffers(c, 10.0)
    assert c.handstand_session.active
    assert c.handstand_snapshot.shape == (47,)


def test_canvas_bundle_rejects_a_different_task_config(assets):
    from go2_deploy.policy_bundle import load_bundle

    with pytest.raises(ValueError, match="bundle is for task"):
        load_bundle(assets[0], import_custom_impl.load_custom_impl("go2trot"))


def test_retrained_candidate_is_separate_and_release_hash_remains_enforced(
    assets, tmp_path
):
    from go2_deploy.policy_bundle import load_bundle

    original = assets[0] / "capture_checkpoint.pt"
    saved = torch.load(original, map_location="cpu", weights_only=False)
    saved["test_candidate_provenance"] = "synthetic metadata change; no training"
    checkpoint = tmp_path / "candidate.pt"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="SHA256"):
        team.export_runtime_bundle(
            checkpoint, assets[0] / "trajectory.npz", tmp_path / "release"
        )
    folder = tmp_path / "candidate"
    team.export_runtime_bundle(
        checkpoint, assets[0] / "trajectory.npz", folder, candidate=True
    )
    policy, cfg = load_bundle(folder)
    assert cfg.task_name == "go2handstand_candidate"
    torch.testing.assert_close(policy(torch.zeros(47)), assets[1](torch.zeros(47)))
    meta = json.loads((folder / "deploy_meta.json").read_text())
    meta["task_name"] = "go2handstand"
    (folder / "deploy_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="SHA256"):
        load_bundle(folder)


@pytest.mark.parametrize("change", ["order", "observation_scale", "action_scale"])
def test_bundle_rejects_training_schema_drift(assets, tmp_path, change):
    import shutil
    from go2_deploy.policy_bundle import load_bundle

    folder = tmp_path / "bad_schema"
    shutil.copytree(assets[0], folder)
    meta = json.loads((folder / "deploy_meta.json").read_text())
    if change == "order":
        meta["obs_vector"][:2] = reversed(meta["obs_vector"][:2])
    elif change == "observation_scale":
        meta["scaling"]["base_ang_vel"] = 2.0
    else:
        meta["scaling"]["dof_pos_target"] = 1.0
    (folder / "deploy_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="training"):
        load_bundle(folder)


def test_submission_rejects_modified_archive_before_creating_log(tmp_path):
    from scripts.package_handstand_submission import package

    archive = tmp_path / "bad.tar.gz"
    archive.write_bytes(b"not the released training archive")
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        package(archive, archive, output)
    assert not output.exists()


@pytest.mark.parametrize("step", [0, 1, 47, 109, 110, 111, 319, 320, 400])
def test_canvas_loaded_hybrid_policy_matches_inference_at_boundaries(
    assets, step, record_property
):
    from go2_deploy.policy_bundle import load_bundle
    from go2_deploy.rl_controller import RLController

    folder, eager, trajectory, *_ = assets
    scripted, cfg = load_bundle(folder)
    obs = eager.centers[min(step, 319)].clone()
    obs[:42] += torch.linspace(-0.01, 0.01, 42)
    obs[42] = step / 400
    obs[43:] += 0.003
    with torch.no_grad():
        expected, capture, balance, reference = eager.components(obs)
        actual = scripted(obs)
        applied_offsets = RLController(cfg, scripted).act(obs)
    assert int(reference) == min(step, 319)
    correction = capture if step < 110 else balance
    law = (eager.actions[min(step, 319)] + correction).clamp(-1, 1)
    torch.testing.assert_close(expected, law, rtol=0, atol=1e-7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)
    torch.testing.assert_close(applied_offsets, expected * 2, rtol=0, atol=2e-7)
    np.testing.assert_allclose(cfg.default_dof_pos, trajectory["nominal"], atol=1e-7)
    record_property("max_action_error", float(abs(actual - expected).max()))


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


def test_instance_hook_can_defer_cycle_until_first_distinct_packet(controller):
    cls = extracted_main()
    c = controller
    c.cfg.custom_impl = NS(update_custom_buffers=lambda main_controller, t: False)
    assert cls._process_custom_buffers(c, 10.0) is False


def test_runtime_configuration_is_recorded_even_before_activation(controller):
    events = []
    controller.csv_logger = NS(log_event=lambda name, data: events.append((name, data)))
    team.initialize_controller(controller)
    assert events[0][0] == "handstand_runtime_config"
    assert events[0][1]["calibration"] == controller.handstand_calibration
    assert not controller.handstand_session.active


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


def test_model_calibration_is_filled_but_not_hardware_verified(assets):
    config = team.model_calibration(assets[2])
    committed = json.loads(
        Path(
            "go2_deploy/custom_implementation/go2handstand_calibration.json"
        ).read_text()
    )
    assert committed == config
    assert config["calibrated"] is False
    assert config["source"] == "model_and_ta_estimates"
    assert 0.34 < config["initial_height"] < 0.35
    np.testing.assert_array_equal(config["imu"]["body_from_imu"], np.eye(3))
    np.testing.assert_allclose(config["imu"]["lever_arm_body"], [-0.02557, 0, 0.04232])
    assert config["imu"]["accel_scale"] == pytest.approx(9.81 / 9.55)
    assert config["observer"]["kind"] == "contact"
    assert config["observer"]["max_interval"] == 0.012
    assert config["max_lowstate_age_s"] == 0.015


def test_unverified_model_config_loads_but_cannot_activate(controller, assets):
    controller.handstand_calibration = team.model_calibration(assets[2])
    receive(controller, 1000, 10.0)
    with pytest.raises(ValueError, match="staff calibration confirmation"):
        team.update_custom_buffers(controller, 10.0)
    assert not controller.handstand_session.active


def relaxed(c):
    c.handstand_calibration = json.loads(json.dumps(c.handstand_calibration))
    c.handstand_calibration["max_lowstate_age_s"] = 0.015
    c.handstand_calibration["imu"]["max_interval"] = 0.012
    c.handstand_session.observer_options = {"max_interval": 0.012, "max_age": 0.015}


def test_10_6ms_receipt_age_and_source_gap_are_bounded_and_counted(controller):
    c = controller
    relaxed(c)
    start(c)
    publish(c)
    receive(c, 1010, 10.010)
    team.update_custom_buffers(c, 10.0206)
    assert c.handstand_session.estimator.samples == 1
    assert c.handstand_session.estimator.timestamp == pytest.approx(0.010)
    assert c.handstand_gap_intervals == 1
    assert c.handstand_session.estimator.gap_intervals == 1
    assert c.handstand_session.path.step == 1
    assert np.isfinite(c.handstand_snapshot).all()


@pytest.mark.parametrize("failure", ["source_gap", "receipt_age", "frozen_tick"])
def test_relaxed_limits_still_reject_missing_or_frozen_stream(controller, failure):
    c = controller
    relaxed(c)
    start(c)
    publish(c)
    if failure == "source_gap":
        receive(c, 1013, 10.02)
        match, now = "Missing or out-of-order", 10.02
    elif failure == "receipt_age":
        receive(c, 1004, 10.004)
        match, now = "Stale", 10.020
    else:
        receive(c, 1000, 10.020)
        match, now = "distinct IMU", 10.020
    with pytest.raises(ValueError, match=match):
        team.update_custom_buffers(c, now)
    assert c.handstand_session.path.step == 1
    assert not c.handstand_session.active


def stationary_readings(c, moving=False, missing_loads=False):
    rows = []
    rng = np.random.default_rng(4110)
    for k in range(125):
        msg = message(c, 1000 + 2 * k)
        msg.foot_force = [35, 35, 35, 35]
        msg.imu_state.gyroscope = np.array([0.0, -0.009, 0.0]) + rng.normal(0, 0.007, 3)
        msg.imu_state.accelerometer = np.array([0.05, -0.05, 9.65]) + rng.normal(
            0, 0.03, 3
        )
        if moving:
            msg.motor_state[0].dq = 0.2
        packet = team.read_lowstate(msg)
        if missing_loads:
            packet = (*packet[:-1], None)
        rows.append((10 + k * 0.002, packet))
    return rows


def test_stationary_bias_estimate_uses_actual_sensor_samples(controller):
    parameters = dict(controller.handstand_calibration["imu"], accel_scale=9.81 / 9.55)
    result = team.calibrate_stationary(parameters, stationary_readings(controller))
    np.testing.assert_allclose(result["gyro_bias_imu"], [0, -0.009, 0], atol=0.002)
    np.testing.assert_allclose(result["accel_bias_imu"], [0.05, -0.05, 0.1], atol=0.01)


@pytest.mark.parametrize("failure", ["short", "moving", "loads", "timestamps"])
def test_stationary_calibration_refuses_unreliable_measurements(controller, failure):
    rows = stationary_readings(
        controller, moving=failure == "moving", missing_loads=failure == "loads"
    )
    if failure == "short":
        rows = rows[:99]
    elif failure == "timestamps":
        rows[-1] = (rows[-2][0], rows[-1][1])
    with pytest.raises(ValueError):
        team.calibrate_stationary(controller.handstand_calibration["imu"], rows)


def test_contact_kinematics_bounds_drift_and_requires_measured_loads(assets):
    _, policy, trajectory, model, *_ = assets
    data = __import__("mujoco").MjData(model)
    __import__("mujoco").mj_setState(
        model,
        data,
        trajectory["initial_state"],
        __import__("mujoco").mjtState.mjSTATE_INTEGRATION,
    )
    session = team.Session(
        policy,
        trajectory,
        model,
        {"kind": "contact", "max_interval": 0.012, "max_age": 0.015},
    )
    height = float(data.qpos[2])
    session.activate(0, height + 0.025, calibrated=True)
    q, dq = data.qpos[session.estimator.qids], np.zeros(12)
    for i in range(1001):
        session.ingest(
            i * 0.002, [0, 0, 10.06], [1, 0, 0, 0], np.zeros(3), q, dq, [35] * 4
        )
    assert session.estimator.corrections > 990
    assert abs(session.estimator.height - height) < 0.003
    assert abs(session.estimator.velocity[2]) < 0.011
    with pytest.raises(ValueError, match="measured foot loads"):
        session.ingest(2.002, [0, 0, 9.81], [1, 0, 0, 0], np.zeros(3), q, dq)
    assert not session.active


def test_no_contact_does_not_invent_a_ground_anchor(assets):
    model = assets[3]
    e = team.ContactEstimator(model, 0.34, 0, calibrated=True)
    for i in range(1, 101):
        e.update(i * 0.002, [0, 0, 10.06], [1, 0, 0, 0])
        e.correct_contacts(
            [1, 0, 0, 0],
            np.zeros(3),
            assets[2]["initial_target"],
            np.zeros(12),
            np.zeros(4),
        )
    assert e.corrections == 0
    assert e.velocity[2] == pytest.approx(0.05)


def test_runner_collects_and_consumes_stationary_samples_without_driver(controller):
    c = controller
    c.handstand_calibration = json.loads(json.dumps(c.handstand_calibration))
    c.handstand_calibration["stationary_samples"] = 100
    c.handstand_calibration["observer"] = {"contact_threshold": 10.0}
    c.handstand_calibration["imu"]["accel_scale"] = 9.81 / 9.55
    c._state = State.INTERMEDIATE
    for t, packet in stationary_readings(c):
        tick, quat, omega, force, q, dq, loads = packet
        msg = NS(
            tick=tick,
            imu_state=NS(quaternion=quat, gyroscope=omega, accelerometer=force),
            motor_state=[
                NS(q=float(a), dq=float(b))
                for a, b in zip(q[team.PERMUTATION], dq[team.PERMUTATION])
            ],
            foot_force=loads[[1, 0, 3, 2]],
        )
        with c.last_lowstate_msg_lock:
            c.last_lowstate_msg = msg
            c.last_lowstate_received_time = t
            team.on_lowstate_msg(c, msg, t)
    assert len(c.handstand_stationary_rx) == 125
    c._state = State.CUSTOM_CTRL
    msg.tick = 1250
    with c.last_lowstate_msg_lock:
        c.last_lowstate_received_time = 10.25
        team.on_lowstate_msg(c, msg, 10.25)
    team.update_custom_buffers(c, 10.25)
    np.testing.assert_allclose(
        c.handstand_effective_imu["accel_bias_imu"], [0.05, -0.05, 0.1], atol=0.01
    )
    assert not c.handstand_stationary_rx
    assert c.handstand_session.active


@pytest.mark.parametrize("source_interval", [0.0106, 0.012])
def test_explicit_tick_units_accept_10_6_and_12ms_intervals(
    controller, source_interval
):
    c = controller
    relaxed(c)
    c.handstand_calibration["tick_period_s"] = 0.0001
    start(c)
    publish(c)
    receive(c, 1000 + round(source_interval / 0.0001), 10 + source_interval)
    team.update_custom_buffers(c, 10.02)
    assert c.handstand_session.estimator.samples == 1
    assert c.handstand_session.estimator.timestamp == pytest.approx(source_interval)
    assert c.handstand_gap_intervals == 1


def test_sensor_estimator_uses_only_kinematics_not_dynamics_or_contacts(
    assets, monkeypatch
):
    import mujoco

    def forbidden(*args, **kwargs):
        pytest.fail("Deployable observer queried simulator dynamics/contact data")

    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_contactForce", forbidden)
    _, policy, trajectory, model, *_ = assets
    session = team.Session(policy, trajectory, model, {"kind": "contact"})
    session.activate(0, 0.34, calibrated=True)
    for i in range(11):
        session.ingest(
            i * 0.002,
            [0, 0, 9.81],
            [1, 0, 0, 0],
            np.zeros(3),
            trajectory["initial_target"],
            np.zeros(12),
            [35] * 4,
        )
    assert np.isfinite(session.snapshot(0.02)).all()


def test_callback_after_cycle_entry_uses_fresh_runner_clock(controller):
    c = controller
    relaxed(c)
    start(c)
    publish(c)
    for i in range(1, 11):
        receive(c, 1000 + 2 * i, 10 + i * 0.002 + 0.0005)
    c.lowstate_clock = lambda: 10.021
    team.update_custom_buffers(c, 10.020)
    assert c.handstand_session.active
    assert c.handstand_session.estimator.samples == 10


@pytest.mark.parametrize("age, accepted", [(0.020, True), (0.02001, False)])
def test_command_age_budget_covers_processing_but_is_not_unbounded(
    controller, age, accepted
):
    c = controller
    relaxed(c)
    c.handstand_calibration["max_command_state_age_s"] = 0.020
    start(c)
    c.lowstate_clock = lambda: 10 + age
    if accepted:
        publish(c)
        assert c.handstand_session.path.step == 1
    else:
        with pytest.raises(ValueError, match="snapshot exceeded age"):
            publish(c)
        assert c.handstand_session.path.step == 0
        assert not c.handstand_session.active


def test_packet_loss_preserves_underlying_timestamp_indexed_sensor_noise(
    assets, monkeypatch
):
    from scripts.handstand_capture import trial

    original_rng = np.random.default_rng
    streams = []

    class Logged:
        def __init__(self, rng):
            self.rng = rng
            self.values = []
            streams.append(self.values)

        def normal(self, *args, **kwargs):
            value = self.rng.normal(*args, **kwargs)
            self.values.append(value.copy())
            return value

        def __getattr__(self, name):
            return getattr(self.rng, name)

    def rng(seed):
        result = original_rng(seed)
        return Logged(result) if seed == 8301000 else result

    class TwoSteps:
        viewer = None
        frames = 0

        def __init__(self, _):
            self.calls = 0

        def sync(self):
            self.calls += 1
            return self.calls <= 2

        def capture(self, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(np.random, "default_rng", rng)
    for drops in (0, 5):
        result = trial(
            assets[1],
            assets[2],
            301000,
            False,
            TwoSteps,
            {"ta": True, "drop_count": drops},
        )
        assert result["cancelled"] and result["steps"] == 2
    assert len(streams) == 2
    assert len(streams[0]) == len(streams[1]) == 42
    np.testing.assert_array_equal(streams[0], streams[1])
