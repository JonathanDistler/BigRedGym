"""Offline complete inference, sensor synchronization and lifecycle regression."""

from pathlib import Path
from types import SimpleNamespace
import mujoco
import numpy as np
import pytest
import torch
from go2_deploy.handstand.policy import (
    load_assets,
    PERMUTATION,
    JOINTS,
    TargetPath,
    checked,
)
from go2_deploy.handstand.estimator import SensorEstimator
from go2_deploy.handstand.session import Session
from go2_deploy.custom_implementation import handstand_provisional as hooks
from gym.envs.go2.handstand_replay import build_model
from experiments.handstand_robustness.capture import InertialObserver
from scripts.handstand_takeoff import load

ROOT = Path(__file__).resolve().parents[2]
CP = ROOT / "output/capture_ppo_conservative/latest.pt"
REF = ROOT / "artifacts/handstand_reference/trajectory.npz"


@pytest.fixture
def assets():
    return load_assets(CP, REF)


@pytest.mark.parametrize("k", [0, 1, 47, 109, 110, 111, 319, 320, 400])
def test_complete_law_boundaries(assets, k):
    policy, t = assets
    original = load(CP)[0]
    torch.manual_seed(k)
    x = policy.centers[min(k, 319)].clone() + torch.randn(47) * 0.01
    x[42] = k / 400
    expected = original(x)
    torch.testing.assert_close(policy(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.jit.script(policy)(x), expected, rtol=0, atol=0)
    assert policy.components(x)[3] == min(k, 319)


def test_mapping_roundtrip_and_wrong_order():
    np.testing.assert_array_equal(
        np.arange(12)[PERMUTATION][PERMUTATION], np.arange(12)
    )
    with pytest.raises(ValueError, match="joint order"):
        SensorEstimator(
            build_model(), 0.34, 0, calibrated=True, joint_order=JOINTS[::-1]
        )


def test_timestamped_observer_matches_research():
    a = InertialObserver(0.34)
    b = SensorEstimator(build_model(), 0.34, 0, calibrated=True)
    rng = np.random.default_rng(4110)
    for i in range(4000):
        force = rng.normal(0, 0.1, 3) + [0, 0, 9.81]
        a.update(force, np.eye(3))
        b.update((i + 1) * 0.002, force, [1, 0, 0, 0])
    np.testing.assert_allclose(b.velocity, a.velocity, atol=1e-12)
    assert abs(b.height - a.height) < 1e-12
    assert b.samples == 4000


@pytest.mark.parametrize("timestamp", [0, -0.002, 0.02, float("nan")])
def test_bad_timestamps(timestamp):
    b = SensorEstimator(build_model(), 0.34, 0, calibrated=True)
    with pytest.raises(ValueError):
        b.update(timestamp, [0, 0, 9.81], [1, 0, 0, 0])


@pytest.mark.parametrize(
    "force,quat", [([np.nan, 0, 0], [1, 0, 0, 0]), ([0, 0, 9.81], [0, 0, 0, 0])]
)
def test_bad_sensors(force, quat):
    b = SensorEstimator(build_model(), 0.34, 0, calibrated=True)
    with pytest.raises(ValueError):
        b.update(0.002, force, quat)


def test_missing_or_corrupt_assets(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_assets(tmp_path / "missing", REF)
    p = tmp_path / "bad.pt"
    p.write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA256"):
        load_assets(p, REF)


def test_metadata_validation(assets, monkeypatch, tmp_path):
    import go2_deploy.handstand.policy as module

    saved = torch.load(CP, weights_only=False)
    saved["kind"] = "trot"
    p = tmp_path / "bad.pt"
    torch.save(saved, p)
    # Isolate metadata validation after the separate immutable hash test.
    monkeypatch.setattr(module, "checked", lambda path, digest: Path(path))
    with pytest.raises(ValueError, match="metadata"):
        load_assets(p, REF)


def test_session_activation_acknowledgement_and_cancel(assets):
    policy, t = assets
    m = build_model()
    s = Session(policy, t, m)
    with pytest.raises(RuntimeError):
        s.snapshot(0)
    with pytest.raises(ValueError, match="calibration"):
        s.activate(0, 0.34)
    s.activate(0, 0.3413013268255194, calibrated=True)
    for i in range(10):
        s.ingest(
            (i + 1) * 0.002,
            [0, 0, 9.81],
            [1, 0, 0, 0],
            [0, 0, 0],
            t["nominal"],
            np.zeros(12),
        )
    obs = s.snapshot(0.02)
    action, target = s.propose(0.02)
    assert obs.shape == (47,) and obs.dtype == np.float32
    assert s.path.step == 0
    with pytest.raises(RuntimeError):
        s.propose(0.02)
    s.acknowledge(target)
    assert s.path.step == 1
    np.testing.assert_array_equal(s.path.target, target)
    with pytest.raises(ValueError, match="Stale"):
        s.snapshot(0.04)
    assert not s.active
    s.activate(1, 0.34, calibrated=True)
    assert s.path.step == 0
    np.testing.assert_array_equal(s.path.target, t["initial_target"])
    s.cancel()
    with pytest.raises(RuntimeError):
        s.snapshot(1)


def test_hooks_fail_closed_and_exact_dimensions():
    c = SimpleNamespace(**hooks.get_custom_buffers_dict())
    with pytest.raises(RuntimeError, match="NOT READY"):
        hooks.update_custom_buffers(c, 1)
    c.handstand_snapshot = np.arange(47, dtype=np.float32)
    pieces = hooks.get_custom_obs_pieces()
    actual = torch.cat(
        [pieces[key](c, None) for key in hooks.CustomDeployConfig.obs_vector]
    )
    np.testing.assert_array_equal(actual, np.arange(47))
    assert sum(hooks.CustomDeployConfig.custom_obs_sizes.values()) == 47


def test_target_limits_slew_and_nonfinite(assets):
    _, t = assets
    m = build_model()
    bounds = m.jnt_range[[m.joint(n).id for n in JOINTS]]
    p = TargetPath(t, bounds)
    before = p.target.copy()
    target = p.apply(np.ones(12) * 10)
    assert (abs(target - before) <= np.tile([0.4214, 0.4214, 0.2198], 4) + 1e-15).all()
    for _ in range(20):
        target = p.apply(np.ones(12) * 10)
    assert (target <= bounds[:, 1]).all()
    with pytest.raises(ValueError):
        p.apply(np.full(12, np.nan))


def test_exact_cornell_loader_retains_unsafe_timing_defaults(
    assets, tmp_path, monkeypatch
):
    import importlib.util
    import json
    import sys
    import types
    import subprocess

    output = tmp_path / "bundle"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_handstand_deployment.py",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    fixtures = ROOT / "tests/fixtures/cornell_fixdeploy"
    custom_module = types.ModuleType("go2_deploy.utility.import_custom_impl")
    custom_module.custom_impl = hooks
    monkeypatch.setitem(
        sys.modules, "go2_deploy.utility.import_custom_impl", custom_module
    )

    def read_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    read_module("go2_deploy.deploy_config", fixtures / "deploy_config.py")
    bundle = read_module("cornell_test_bundle", fixtures / "policy_bundle.py")
    policy, cfg = bundle.load_bundle(str(output))
    utility = read_module("cornell_test_utility", fixtures / "deploy_utility.py")
    np.testing.assert_array_equal(utility.UNITREE_TO_QGYM_JOINT_IDX, PERMUTATION)
    np.testing.assert_array_equal(utility.QGYM_TO_UNITREE_JOINT_IDX, PERMUTATION)
    controller = SimpleNamespace(
        cfg=cfg,
        obs_vec_size=47,
        handstand_snapshot=np.arange(47, dtype=np.float32),
        default_lowcmd=SimpleNamespace(
            motor_cmd=[SimpleNamespace() for _ in range(20)]
        ),
    )
    np.testing.assert_array_equal(
        utility.lowstate_to_obs(controller, None), np.arange(47)
    )
    normalized = torch.linspace(-0.1, 0.1, 12)
    command = utility.action_to_lowcmd(controller, 2 * normalized)
    observed = torch.tensor([motor.q for motor in command.motor_cmd[:12]])
    expected = (cfg.default_dof_pos + 2 * normalized)[PERMUTATION]
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert cfg.obs_dim() == 47
    assert cfg.DeployScaling.dof_pos_target == 2
    assert cfg.DeployScaling.handstand_previous_target == 1
    # Demonstrated blocker: extra metadata does NOT override these defaults.
    assert cfg.ctrl_freq == 100 and cfg.exp_moving_avg is True
    with torch.no_grad():
        x = torch.zeros(47)
        x[42] = 400 / 400
        torch.testing.assert_close(policy(x), assets[0](x), rtol=0, atol=0)
    # Model bundle reload is self-contained and requires no simulator state.
    m = mujoco.MjModel.from_binary_path(str(output / "go2_fk.mjb"))
    assert m.nq == 19
    meta = json.loads((output / "deploy_meta.json").read_text())
    for name, digest in meta["files"].items():
        checked(output / name, digest)


@pytest.mark.parametrize("k", [0, 1, 47, 109, 110, 111, 319, 320, 400])
def test_all_observation_channels_and_estimator_at_boundaries(assets, k):
    from experiments.handstand_robustness.capture import CaptureTrainingEnvironment

    policy, t = assets
    e = CaptureTrainingEnvironment(build_model(), t, horizon=400)
    b = SensorEstimator(build_model(), e.imu_observer.height, 0, calibrated=True)
    # A recorded reference snapshot, used as synthetic encoder/IMU telemetry.
    from gym.envs.go2.handstand_replay import STATE_SPEC

    mujoco.mj_setState(e.m, e.d, t["reference_states"][min(k, 320)], STATE_SPEC)
    mujoco.mj_forward(e.m, e.d)
    e.index = k
    rng = np.random.default_rng(k)
    for i in range(k * 10):
        force = rng.normal(0, 0.1, 3) + [0, 0, 9.81]
        e.imu_observer.update(force, np.eye(3))
        b.update((i + 1) * 0.002, force, [1, 0, 0, 0])
    obs = b.observation(
        e.d.qpos[3:7],
        e.d.qvel[3:6],
        e.d.qpos[e.qids],
        e.d.qvel[e.vids],
        e.target,
        t["nominal"],
        k,
        k * 0.02,
    )
    np.testing.assert_allclose(obs, e.observation(), rtol=0, atol=1e-7)


def test_session_initial_sample_and_wrong_ack_fault(assets):
    p, t = assets
    s = Session(p, t, build_model())
    s.activate(5, 0.34, calibrated=True)
    s.ingest(5, [0, 0, 9.81], [1, 0, 0, 0], [0, 0, 0], t["nominal"], np.zeros(12))
    action, target = s.propose(5)
    assert s.estimator.samples == 0
    with pytest.raises(ValueError, match="Applied target"):
        s.acknowledge(target + 0.001)
    assert not s.active


def test_calibration_mount_bias_and_lever_correction():
    from go2_deploy.handstand.calibration import IMUCalibration

    mount = np.diag([-1.0, -1.0, 1.0])
    c = IMUCalibration(mount, np.eye(3), [0.1, 0, 0], [0.01, 0, 0], [0, 0.02, 0])
    # IMU rotated 180 degrees around z relative to the calibrated model body.
    quat, gyro, force = c.correct(0, [0, 0, 0, 1], [0, 0.02, 0], [0.01, 0, 9.81])
    np.testing.assert_allclose(gyro, [0, 0, 0], atol=1e-15)
    np.testing.assert_allclose(force, [0, 0, 9.81], atol=1e-15)
    from go2_deploy.handstand.estimator import rotation

    np.testing.assert_allclose(rotation(quat), np.eye(3), atol=1e-15)
    # 1 rad/s body z, alpha=500 rad/s²: lever acceleration [-.1,50,0].
    raw_force = mount.T @ np.array([-0.1, 50, 9.81]) + [0.01, 0, 0]
    quat, gyro, force = c.correct(0.002, [0, 0, 0, 1], [0, 0.02, 1], raw_force)
    np.testing.assert_allclose(force, [0, 0, 9.81], atol=1e-12)
    with pytest.raises(ValueError):
        c.correct(0.002, [0, 0, 0, 1], [0, 0, 0], [0, 0, 9.81])


def test_review_bundle_and_metadata_tamper(tmp_path):
    import json
    from go2_deploy.handstand.bundle import load_review_bundle
    import shutil

    source = ROOT / "artifacts/handstand_deployment"
    policy, model, meta = load_review_bundle(source)
    assert policy(torch.zeros(47)).shape == (12,) and model.nq == 19
    directory = tmp_path / "bundle"
    shutil.copytree(source, directory)
    meta["ctrl_freq"] = 100
    (directory / "deploy_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="ctrl_freq"):
        load_review_bundle(directory)


def test_missing_one_imu_sample_is_rejected():
    b = SensorEstimator(build_model(), 0.34, 0, calibrated=True)
    with pytest.raises(ValueError, match="missing"):
        b.update(0.004, [0, 0, 9.81], [1, 0, 0, 0])


def test_bad_control_cadence_faults_session(assets):
    p, t = assets
    s = Session(p, t, build_model())
    s.activate(0, 0.34, calibrated=True)
    for i in range(6):
        s.ingest(
            i * 0.002, [0, 0, 9.81], [1, 0, 0, 0], [0, 0, 0], t["nominal"], np.zeros(12)
        )
        if i == 0:
            _, target = s.propose(0)
            s.acknowledge(target)
    with pytest.raises(ValueError, match="cadence"):
        s.propose(0.01)
    assert not s.active
