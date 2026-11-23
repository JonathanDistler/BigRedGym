"""Remote workflow, independent command path, calibration and fresh-run tooling.

All messages, publishers and training subprocesses here are software fakes.
No Unitree transport or PPO training is executed.
"""

import ast
import hashlib
import json
import csv
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from go2_deploy.custom_implementation.go2commandcheck import make_runtime
from go2_deploy.custom_implementation import go2handstand as team
from go2_deploy.state import State
from scripts import calibrate_handstand_logs as calibration
from scripts import train_handstand_staff as training


def remote_controller():
    path = Path("go2_deploy/unitree_remote_controller.py")
    tree = ast.parse(path.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RCHandler"
    )
    scope = {"State": State}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    buttons = "A B X Y Up Down Left Right F1".split()
    remote = NS(**{b: [0, 0] for b in buttons})
    calls = []
    c = NS(
        remote_controller=remote,
        cfg=NS(custom_impl=team.CustomDeployConfig()),
        _state=State.RECOVERY,
        kp_mult=1.0,
        kd_mult=1.0,
    )
    for action in ("intermediate", "custom_ctrl", "emergency_stop", "shutdown"):
        setattr(c, "request_" + action, lambda a=action: calls.append(a))
    return scope["RCHandler"](c), c, remote, calls


@pytest.mark.parametrize(
    "button,action", [("B", "intermediate"), ("X", "emergency_stop"), ("Y", "shutdown")]
)
def test_remote_modes_need_no_keyboard(button, action):
    handler, _, remote, calls = remote_controller()
    setattr(remote, button, [1, 0])
    handler._process_input()
    assert calls == [action]


def test_remote_a_requires_b_and_fixed_gains_cannot_be_changed():
    handler, c, remote, calls = remote_controller()
    remote.A = [1, 0]
    handler._process_input()
    assert not calls
    c._state = State.INTERMEDIATE
    handler._process_input()
    assert calls == ["custom_ctrl"]
    remote.A = [0, 0]
    remote.Up = [1, 0]
    handler._process_input()
    assert c.kp_mult == c.kd_mult == 1


def test_deployment_has_no_keyboard_import_or_thread():
    source = Path("go2_deploy/deploy.py").read_text()
    assert "KeyboardHandler" not in source
    assert "input(" not in source


@pytest.fixture
def diagnostic():
    actor, cfg = make_runtime(1, 0.01)
    c = NS(cfg=cfg, _state=State.INTERMEDIATE)
    for k, v in cfg.custom_impl.get_custom_buffers_dict().items():
        setattr(c, k, v)
    cfg.custom_impl.initialize_controller(c)
    now = [0.0]
    c.lowstate_clock = lambda: now[0]
    q = np.tile([0, 0.7334, -1.4655], 4)

    def sample(t, position=q, tick=None):
        now[0] = t
        msg = NS(
            tick=round(t * 1000) if tick is None else tick,
            motor_state=[NS(q=float(v), dq=0.0) for v in position[team.PERMUTATION]],
        )
        cfg.custom_impl.on_lowstate_msg(c, msg, t)
        return msg

    def publish(target, accepted=True):
        cmd = NS(motor_cmd=[NS(q=float(v)) for v in target.numpy()[team.PERMUTATION]])
        cfg.custom_impl.on_command_sent(c, cmd, accepted, now[0])

    return c, actor, now, q, sample, publish


def test_diagnostic_runs_without_imu_calibration_and_returns_to_hold(diagnostic):
    c, actor, now, q, sample, publish = diagnostic
    impl = c.cfg.custom_impl
    sample(0)
    impl.update_custom_buffers(c, 0)
    publish(impl.bound_command(c, torch.tensor(q)))
    initial = c.check_sent.copy()
    c._state = State.CUSTOM_CTRL
    targets = []
    for i in range(121):
        t = (i + 1) * 0.02
        sample(t, c.check_sent)
        impl.update_custom_buffers(c, t)
        assert torch.equal(actor(torch.zeros(12)), torch.zeros(12))
        target = impl.bound_command(c, torch.zeros(12))
        targets.append(target.numpy())
        publish(target)
    targets = np.array(targets)
    assert np.max(targets[:, 1] - initial[1]) == pytest.approx(0.01, abs=1e-7)
    np.testing.assert_array_equal(targets[-1], initial)
    np.testing.assert_array_equal(targets[:, 0], np.repeat(initial[0], len(targets)))
    assert c.check_step == 121


def test_diagnostic_hold_is_default_and_offset_is_bounded():
    _, cfg = make_runtime()
    assert cfg.custom_impl.offset == 0
    assert cfg.ctrl_freq == 50 and not cfg.exp_moving_avg
    with pytest.raises(ValueError):
        make_runtime(offset=0.021)
    with pytest.raises(ValueError):
        make_runtime(joint=12)


def test_exact_cornell_cycle_runs_diagnostic_with_mocked_publisher(diagnostic):
    from tests.unit_tests.test_handstand_runner import extracted_main
    from go2_deploy.utility import deploy_utility
    from go2_deploy.rl_controller import RLController

    c, actor, _, q, sample, _ = diagnostic
    cls = extracted_main(clock=0.02)
    c.last_lowstate_msg = sample(0.02)
    for name in (
        "last_command",
        "last_sportmodestate_msg",
        "last_obs",
        "last_action",
        "last_lowstate_msg",
    ):
        setattr(c, name + "_lock", threading.Lock())
    c.last_sportmodestate_msg = None
    c.last_action = torch.zeros(2, 12)
    c.obs_vec_size = 12
    c.kp_mult = c.kd_mult = 1.0
    c.intermediate_pos = torch.tensor(q)
    c.remote_controller = NS(lin_vel_x=0.0, lin_vel_y=0.0, yaw_vel=0.0)
    c.rl_controller = RLController(c.cfg, actor)
    c._process_custom_buffers = lambda t: cls._process_custom_buffers(c, t)
    c.msg_to_obs = lambda msg: deploy_utility.lowstate_to_obs(c, msg)
    c._act = lambda obs: cls._act(c, obs)
    c.default_lowcmd = NS(motor_cmd=[NS(q=0.0, kp=0.0, kd=0.0) for _ in range(20)])
    c.action_to_lowcmd = lambda action: deploy_utility.action_to_lowcmd(c, action)
    c.crc = NS(Crc=lambda _: 0)
    writes = []
    c.lowcmd_publisher = NS(Write=lambda command: writes.append(command) or True)
    c.csv_logger = NS(log_control=lambda *args: None)
    c.action_count = 0
    cls._policy_control_cycle(c)
    assert len(writes) == 1 and c.check_sent is not None
    np.testing.assert_allclose(c.check_sent, q, atol=1e-7)


def test_effective_calibration_events_are_persisted_off_control_thread(tmp_path):
    class FakeThread:
        def __init__(self, **kwargs):
            pass

        def Start(self):
            pass

        def Wait(self, **kwargs):
            pass

    path = Path("go2_deploy/utility/csv_logger.py")
    tree = ast.parse(path.read_text())
    tree.body = [
        n
        for n in tree.body
        if not (
            isinstance(n, ast.ImportFrom) and n.module == "go2_deploy.utility.thread"
        )
    ]
    scope = {"RecurrentThread": FakeThread}
    exec(compile(tree, str(path), "exec"), scope)
    _, cfg = make_runtime()
    c = NS(cfg=cfg, _state=State.INTERMEDIATE)
    logger = scope["CSVLogger"](c, run_dir=str(tmp_path))
    logger.log_event("handstand_calibration", {"gyro_bias": [0, -0.009, 0]})
    assert logger.queue.qsize() == 1
    logger.close()
    rows = list(csv.DictReader((tmp_path / "events.csv").open()))
    assert rows[0]["event"] == "handstand_calibration"
    assert json.loads(rows[0]["data_json"])["gyro_bias"] == [0, -0.009, 0]


@pytest.mark.parametrize(
    "failure", ["stale", "frozen", "invalid", "no_hold", "failed_publication"]
)
def test_diagnostic_rejects_invalid_command_paths(diagnostic, failure):
    c, _, now, q, sample, publish = diagnostic
    impl = c.cfg.custom_impl
    sample(0)
    if failure == "stale":
        now[0] = 0.016
    elif failure == "frozen":
        sample(0.02, tick=0)
    elif failure == "invalid":
        sample(0.002, np.full(12, np.nan))
    elif failure == "no_hold":
        c._state = State.CUSTOM_CTRL
    if failure == "failed_publication":
        impl.update_custom_buffers(c, 0)
        target = impl.bound_command(c, torch.tensor(q))
        with pytest.raises(ValueError, match="publication"):
            publish(target, False)
    else:
        with pytest.raises(ValueError):
            impl.update_custom_buffers(c, now[0])


def sensor_rows():
    q = np.tile([0, 0.7334, -1.4655], 4)
    return [
        (
            i * 0.002,
            (
                i * 2,
                np.array([1.0, 0, 0, 0]),
                np.array([0.0, -0.009, 0]),
                np.array([0.04, -0.03, 9.63]),
                q.copy(),
                np.zeros(12),
                np.full(4, 30.0),
            ),
            "INTERMEDIATE",
        )
        for i in range(200)
    ]


def test_log_calibration_estimates_bias_tick_height_but_never_activates():
    preset = json.loads(
        Path(
            "go2_deploy/custom_implementation/go2handstand_calibration.json"
        ).read_text()
    )
    candidate, report = calibration.estimate(sensor_rows(), preset)
    assert candidate["calibrated"] is False and not report["activation_enabled"]
    assert candidate["tick_period_s"] == pytest.approx(0.001)
    np.testing.assert_allclose(
        candidate["imu"]["accel_bias_imu"], [0.04, -0.03, 0.08], atol=1e-10
    )
    assert 0.30 < candidate["initial_height"] < 0.37
    assert report["reference_posture"]["within_existing_activation_bound"]
    assert len(report["requires_independent_confirmation"]) == 4
    assert "not identifiable" in report["timing"]["absolute_transport_latency"]


def test_initial_heading_alignment_is_an_explicit_frame_choice():
    preset = json.loads(
        Path(
            "go2_deploy/custom_implementation/go2handstand_calibration.json"
        ).read_text()
    )
    rows = sensor_rows()
    yaw = 0.4
    for _, packet, _ in rows:
        packet[1][:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    config, report = calibration.estimate(rows, preset, align_initial_heading=True)
    r = np.array(config["imu"]["world_alignment"]) @ team.rotation(rows[-1][1][1])
    np.testing.assert_allclose(r, np.eye(3), atol=1e-12)
    assert report["initial_heading_alignment"][
        "removed_initial_yaw_rad"
    ] == pytest.approx(yaw)
    assert not config["calibrated"]


def test_multi_pose_accelerometer_fit_and_single_pose_nonidentifiability():
    bias = np.array([0.04, -0.03, 0.08])
    poses = np.r_[np.eye(3), -np.eye(3)] * (9.81 / 0.97) + bias
    scale, estimated, rms = calibration.fit_accelerometer(poses)
    assert scale == pytest.approx(0.97)
    np.testing.assert_allclose(estimated, bias, atol=1e-12)
    assert rms < 1e-12
    with pytest.raises(ValueError, match="do not identify"):
        calibration.fit_accelerometer(np.tile(poses[0], (100, 1)))


def test_calibration_refuses_no_contact_or_moving_logs():
    preset = json.loads(
        Path(
            "go2_deploy/custom_implementation/go2handstand_calibration.json"
        ).read_text()
    )
    rows = sensor_rows()
    for _, packet, _ in rows:
        packet[-1][:] = 0
    with pytest.raises(ValueError, match="No valid"):
        calibration.estimate(rows, preset)


def test_stationary_encoder_noise_uses_window_rms_without_accepting_motion():
    preset = json.loads(
        Path(
            "go2_deploy/custom_implementation/go2handstand_calibration.json"
        ).read_text()
    )
    rows = sensor_rows()
    for i, (_, packet, _) in enumerate(rows):
        packet[5][0] = 0.12 if i % 20 == 0 else 0
    calibration.estimate(rows, preset)
    for _, packet, _ in rows:
        packet[5][0] = 0.08  # Sustained motion still fails, despite peak < .2.
    with pytest.raises(ValueError, match="No valid"):
        calibration.estimate(rows, preset)


def test_recovery_logs_are_only_used_when_explicitly_requested():
    preset = json.loads(
        Path(
            "go2_deploy/custom_implementation/go2handstand_calibration.json"
        ).read_text()
    )
    rows = [(t, p, "RECOVERY") for t, p, _ in sensor_rows()]
    with pytest.raises(ValueError, match="No valid"):
        calibration.estimate(rows, preset)
    candidate, report = calibration.estimate(rows, preset, allow_recovery=True)
    assert report["stationary_window_state"] == "RECOVERY"
    assert not candidate["calibrated"]


def test_empty_command_log_reports_no_activation(tmp_path):
    path = tmp_path / "control.csv"
    path.write_text("t,state\n")
    report = calibration.command_evidence(sensor_rows(), path)
    assert report["control_rows"] == 0 and "activation" in report["reason"]


def test_training_plan_uses_public_assets_and_fresh_bc_not_old_logs(tmp_path):
    plan = training.commands(tmp_path, "new", 4110, 100)
    assert len(plan) == 4
    assert str(tmp_path / "runs/new/bc/bc.pt") in plan[1]
    assert "--candidate" in plan[3]
    assert all("logs/" not in arg for command in plan for arg in command)
    assert "--balance" in plan[0] and "--feedback" in plan[0]


def test_training_snapshots_before_invoking_any_training(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for folder in ("experiments", "gym", "scripts", "configs", "resources"):
        (source / folder).mkdir(parents=True)
        (source / folder / "sample.py").write_text("# fresh source\n")
    calls = []

    def fake_run(command, **kwargs):
        run = tmp_path / "runs/new"
        assert (run / "files/gym/sample.py").is_file()
        manifest = json.loads((run / "source_sha256.json").read_text())
        assert (
            manifest["gym/sample.py"] == hashlib.sha256(b"# fresh source\n").hexdigest()
        )
        calls.append(command)

    monkeypatch.setattr(training.subprocess, "run", fake_run)
    training.train(tmp_path, "new", 4110, 100)
    assert len(calls) == 4
    with pytest.raises(FileExistsError):
        training.train(tmp_path, "new", 4110, 100)
