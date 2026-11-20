"""Pure software interface tests; toy archives are NOT robot trajectories."""

import hashlib
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from go2_deploy.custom_implementation import go2trajectory as interface


@pytest.fixture
def toy_bundle(tmp_path):
    nominal = np.tile([0.0, 0.7334, -1.46549], 4)
    targets = np.stack([nominal, nominal + 0.001, nominal + 0.002])
    path = tmp_path / "position_reference.npz"
    np.savez(
        path,
        targets=targets,
        nominal=nominal,
        time=np.arange(3) * 0.02,
        initial_target=targets[0],
        joint_order=np.array(interface.JOINT_ORDER),
    )
    meta = {
        "kind": "constrained_position_trajectory_v1",
        "control_hz": 50,
        "kp": 60,
        "kd": 3,
        "ema": False,
        "action_scale": 2.0,
        "hardware_ready": False,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (tmp_path / "deploy_meta.json").write_text(json.dumps(meta))
    return tmp_path, nominal, targets


def test_hooks_disabled_by_default():
    c = SimpleNamespace(**interface.get_custom_buffers_dict())
    with pytest.raises(RuntimeError, match="NOT READY"):
        interface.update_custom_buffers(c, 0)
    assert sum(interface.CustomDeployConfig.custom_obs_sizes.values()) == 43


def test_full_software_action_path_and_phase(toy_bundle):
    directory, nominal, targets = toy_bundle
    c = SimpleNamespace(**interface.get_custom_buffers_dict())
    policy = interface.configure_software_validation(c, directory)
    for step in range(5):
        time = step * 0.02
        c.trajectory_sensor_snapshot = (
            time,
            [1, 0, 0, 0],
            [0, 0, 0],
            nominal,
            np.zeros(12),
        )
        interface.update_custom_buffers(c, time)
        obs = torch.cat(
            [
                interface.get_custom_obs_pieces()[name](c, None)
                for name in interface.CustomDeployConfig.obs_vector
            ]
        )
        assert obs.shape == (43,)
        action = policy(obs).numpy()
        np.testing.assert_allclose(
            nominal + 2 * action, targets[min(step, 2)], atol=1e-9
        )
        assert c.trajectory_step == step
        c.trajectory_acknowledgement = (step, targets[min(step, 2)].copy())
    assert interface.CustomDeployConfig.DeployScaling.dof_pos_target == 2
    assert interface.CustomDeployConfig.DeployScaling.trajectory_previous_target == 1


@pytest.mark.parametrize(
    "fault", ["stale", "nan", "wrong_ack", "missing_ack", "cancel", "hardware"]
)
def test_faults_disable_controller(toy_bundle, fault):
    directory, nominal, targets = toy_bundle
    c = SimpleNamespace(**interface.get_custom_buffers_dict())
    interface.configure_software_validation(c, directory)
    c.trajectory_sensor_snapshot = (0, [1, 0, 0, 0], [0, 0, 0], nominal, np.zeros(12))
    interface.update_custom_buffers(c, 0)
    c.trajectory_acknowledgement = (0, targets[0])
    c.trajectory_sensor_snapshot = (
        0.02,
        [1, 0, 0, 0],
        [0, 0, 0],
        nominal,
        np.zeros(12),
    )
    if fault == "stale":
        c.trajectory_sensor_snapshot = (
            0,
            [1, 0, 0, 0],
            [0, 0, 0],
            nominal,
            np.zeros(12),
        )
    if fault == "nan":
        c.trajectory_sensor_snapshot = (
            0.02,
            [1, 0, 0, 0],
            [np.nan, 0, 0],
            nominal,
            np.zeros(12),
        )
    if fault == "wrong_ack":
        c.trajectory_acknowledgement = (0, targets[1])
    if fault == "missing_ack":
        c.trajectory_acknowledgement = None
    if fault == "cancel":
        c.trajectory_cancel_requested = True
    if fault == "hardware":
        c.lowcmd_publisher = object()  # Sentinel; no SDK is imported.
    with pytest.raises((ValueError, RuntimeError)):
        interface.update_custom_buffers(c, 0.02)
    assert c.trajectory_mode == "disabled" and c.trajectory_observation is None


def test_bad_bundle_hash_and_metadata(toy_bundle):
    directory, nominal, targets = toy_bundle
    c = SimpleNamespace(**interface.get_custom_buffers_dict())
    path = directory / "deploy_meta.json"
    meta = json.loads(path.read_text())
    meta["sha256"] = "0" * 64
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="hash"):
        interface.configure_software_validation(c, directory)
    meta["control_hz"] = 100
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="metadata"):
        interface.configure_software_validation(c, directory)


def test_exact_cornell_file_loader(tmp_path, toy_bundle):
    import importlib.util
    from pathlib import Path
    import subprocess

    deployment = tmp_path / "cornell"
    utility = deployment / "utility"
    custom = deployment / "custom_implementation"
    utility.mkdir(parents=True)
    custom.mkdir()
    commit = "8ca9bf98373a234de662f9fb8242a228b7efde21"
    for original, destination in [
        ("utility/import_custom_impl.py", utility / "import_custom_impl.py"),
        ("custom_implementation/go2trot_base.py", custom / "go2trot_base.py"),
    ]:
        destination.write_bytes(
            subprocess.check_output(["git", "show", commit + ":go2_deploy/" + original])
        )
    (custom / "go2trajectory.py").write_bytes(Path(interface.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location(
        "cornell_loader_test", utility / "import_custom_impl.py"
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    candidate = loader.get_custom_impl_module("go2trajectory.py")
    assert set(candidate.get_custom_obs_pieces()) == set(
        candidate.CustomDeployConfig.obs_vector
    )
    controller = SimpleNamespace(**candidate.get_custom_buffers_dict())
    policy = candidate.configure_software_validation(controller, toy_bundle[0])
    assert policy(torch.zeros(43)).shape == (12,)


def test_explicit_reconfigure_resets_cancelled_state(toy_bundle):
    directory, nominal, targets = toy_bundle
    c = SimpleNamespace(**interface.get_custom_buffers_dict())
    interface.configure_software_validation(c, directory)
    c.trajectory_cancel_requested = True
    with pytest.raises(RuntimeError):
        interface.update_custom_buffers(c, 0)
    c.trajectory_acknowledgement = (99, targets[-1])
    interface.configure_software_validation(c, directory)
    assert c.trajectory_step == 0 and c.trajectory_acknowledgement is None
    assert not c.trajectory_cancel_requested and c.trajectory_observation is None
