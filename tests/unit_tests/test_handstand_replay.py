"""Strict scoring boundaries and optional hash-locked continuous replay."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from gym.envs.go2.handstand_replay import (
    TARGET_HEIGHT,
    Replay,
    build_model,
    load_trajectory,
    strict_balance,
)


def test_strict_gate_rejects_each_failure():
    valid = dict(
        safe=True,
        rear_clearance=0.2,
        rear_loads=np.zeros(2),
        front_support=True,
        orientation=0.0,
        base_height=0.57,
        com_height=TARGET_HEIGHT,
        com_speed=0.0,
        angular_speed=0.0,
        support_error=0.0,
        slip=0.0,
        calf_clearance=0.03,
        head_clearance=0.1,
        force_ratio=1.0,
    )
    assert strict_balance(**valid)
    failures = dict(
        safe=False,
        rear_clearance=0.1,
        rear_loads=np.array([0.02, 0]),
        front_support=False,
        orientation=np.deg2rad(10),
        base_height=0.64,
        com_height=0.49,
        com_speed=0.25,
        angular_speed=0.5,
        support_error=0.04,
        slip=0.05,
        calf_clearance=0.0,
        head_clearance=0.0,
        force_ratio=4.0,
    )
    for key, value in failures.items():
        assert not strict_balance(**(valid | {key: value})), key


def test_artifact_validation(tmp_path):
    model = build_model()
    path = tmp_path / "trajectory.npz"
    with pytest.raises(FileNotFoundError):
        load_trajectory(path, "0" * 64, model)
    np.savez(path, schema_version=1, mujoco_version="incompatible")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA256"):
        load_trajectory(path, "0" * 64, model)
    with pytest.raises(ValueError, match="MuJoCo version"):
        load_trajectory(path, digest, model)


def test_original_continuous_replay():
    path = os.environ.get("GO2_HANDSTAND_TRAJECTORY")
    digest = os.environ.get("GO2_HANDSTAND_SHA256")
    if not path or not digest:
        pytest.skip("Provide external reference NPZ and SHA256 to run cold replay")
    model = build_model()
    trajectory = load_trajectory(Path(path), digest, model)
    result = Replay(model, trajectory).run()
    assert result["replay_matches"] and result["max_state_error"] == 0
    assert result["certified_takeoff"] and result["front_capture"]
    assert result["legal_contacts"] and result["prefix_safe"]
    assert result["five_second_success"]
    assert result["strict_hold_seconds"] == pytest.approx(5.014, abs=1e-10)
    assert result["max_pitch_degrees"] == pytest.approx(92.05553383951612, abs=1e-10)
    trajectory["actions"] = np.zeros_like(trajectory["actions"])
    failure = Replay(model, trajectory).run()
    assert not failure["five_second_success"] and not failure["replay_matches"]
