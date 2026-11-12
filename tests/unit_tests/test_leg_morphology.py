"""Physical and curriculum invariants for isolated morphology experiments."""

import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from experiments.morphology.curriculum import Curriculum
from experiments.morphology.task import Config, Task
from gym.envs.go2.leg_morphology import (
    SCALES,
    SOURCE,
    actuator_parameters,
    build_model,
    generate,
)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    root = tmp_path_factory.mktemp("morphology")
    original = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    paths = {s: generate(s, root / f"{s}.urdf") for s in SCALES}
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == original
    return paths, {s: build_model(p) for s, p in paths.items()}


def test_identity_geometry_and_dynamics(models):
    paths, models = models
    assert paths[1.0].read_bytes() == SOURCE.read_bytes()
    original = build_model(SOURCE)
    identity = models[1.0]
    for name in (
        "body_pos",
        "body_quat",
        "body_ipos",
        "body_iquat",
        "body_mass",
        "body_inertia",
        "geom_type",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "jnt_pos",
        "jnt_axis",
        "jnt_range",
        "dof_armature",
        "dof_damping",
    ):
        np.testing.assert_array_equal(getattr(original, name), getattr(identity, name))
    d1, d2 = mujoco.MjData(original), mujoco.MjData(identity)
    rng = np.random.default_rng(71)
    for data in (d1, d2):
        data.qpos[2] = 0.38
        data.qpos[3] = 1
        data.qpos[7:] = [0, 0.58, -1.16] * 4
    for _ in range(500):
        force = rng.uniform(-2, 2, original.nv)
        d1.qfrc_applied[:] = force
        d2.qfrc_applied[:] = force
        mujoco.mj_step(original, d1)
        mujoco.mj_step(identity, d2)
        np.testing.assert_array_equal(d1.qpos, d2.qpos)
        np.testing.assert_array_equal(d1.qvel, d2.qvel)
        np.testing.assert_array_equal(d1.qacc, d2.qacc)


@pytest.mark.parametrize("scale", SCALES)
def test_geometry_mass_and_actuators(models, scale):
    paths, compiled = models
    original = ET.parse(SOURCE)
    variant = ET.parse(paths[scale])
    # Main body, motor housings and feet are untouched, including inertia.
    for name in ("base", "FL_hip", "FR_foot", "RL_calf_rotor"):
        assert ET.tostring(original.find(f"link[@name='{name}']")) == ET.tostring(
            variant.find(f"link[@name='{name}']")
        )
    for leg in ("FL", "FR", "RL", "RR"):
        for part in ("calf", "foot"):
            node = variant.find(f"joint[@name='{leg}_{part}_joint']/origin")
            assert float(node.get("xyz").split()[2]) == pytest.approx(-0.213 * scale)
        for part in ("thigh", "calf"):
            old = original.find(f"link[@name='{leg}_{part}']/inertial")
            new = variant.find(f"link[@name='{leg}_{part}']/inertial")
            assert float(new.find("mass").get("value")) == pytest.approx(
                float(old.find("mass").get("value")) * scale**1
            )
            inertia = {k: float(v) for k, v in old.find("inertia").attrib.items()}
            actual = {k: float(v) for k, v in new.find("inertia").attrib.items()}
            czz = (inertia["ixx"] + inertia["iyy"] - inertia["izz"]) / 2
            assert actual["ixx"] == pytest.approx(
                scale * (inertia["ixx"] + (scale**2 - 1) * czz)
            )
            assert actual["iyy"] == pytest.approx(
                scale * (inertia["iyy"] + (scale**2 - 1) * czz)
            )
            assert actual["izz"] == pytest.approx(scale * inertia["izz"])
            assert actual["ixz"] == pytest.approx(scale**2 * inertia["ixz"])
        for part in ("hip", "thigh", "calf"):
            name = f"{leg}_{part}_joint"
            assert (
                original.find(f"joint[@name='{name}']/limit").attrib
                == variant.find(f"joint[@name='{name}']/limit").attrib
            )
    model = compiled[scale]
    baseline = compiled[1.0]
    # Includes automatically inferred collision-link mass/inertia.
    for leg in ("FL", "FR", "RL", "RR"):
        for part in ("thigh", "calf", "calflower", "calflower1"):
            i = model.body(f"{leg}_{part}").id
            assert model.body_mass[i] == pytest.approx(baseline.body_mass[i] * scale**1)
            assert (model.body_inertia[i] > 0).all()
    # Axial shortening preserves body width and unchanged motor mounts.
    np.testing.assert_array_equal(model.body_pos[:, 1], baseline.body_pos[:, 1])
    np.testing.assert_allclose(model.jnt_range, baseline.jnt_range)
    np.testing.assert_array_equal(model.dof_armature, baseline.dof_armature)
    for a, b in zip(actuator_parameters(model), actuator_parameters(baseline)):
        np.testing.assert_array_equal(a, b)
    assert (model.body_inertia[model.body_mass > 0] > 0).all()
    # Principal inertias obey rigid-body triangle inequalities.
    inertias = model.body_inertia
    assert (2 * inertias.max(axis=1) <= inertias.sum(axis=1) + 1e-12).all()


def test_curriculum_window_and_episode_latching():
    schedule = Curriculum("C", 0.8, min_episodes=2, required_windows=2)
    episode_scale = schedule.scale
    for i in range(3):
        schedule.completed(0.8, True, i)
        assert schedule.scale == 0.8
    schedule.completed(0.8, True, 4)
    assert schedule.scale == 0.85
    assert episode_scale == 0.8
    schedule.completed(0.8, True, 5)  # old-stage episodes do not count
    assert schedule.outcomes == []
    fixed = Curriculum("D", 0.85, fixed_interval=10)
    fixed.iteration(10)
    assert fixed.scale == 0.9
    fixed.iteration(100)
    assert fixed.scale == 1.0


def test_task_stance_motor_limits_and_episode_geometry(tmp_path):
    task = Task(tmp_path / "assets", Config(seconds=0.02))
    heights = []
    for scale in SCALES:
        task.reset(scale, 17)
        heights.append(task.d.qpos[2])
        assert task.last["illegal_contacts"] == 0
        assert task.last["rear_clearance"] >= 0.0029
        current = task.m
        task.step(np.ones(12) * 100)
        assert task.m is current
        assert (np.abs(task.d.qfrc_applied[task.vids]) <= task.effort).all()
        task.step(np.zeros(12))
        assert task.done
        with pytest.raises(RuntimeError):
            task.step(np.zeros(12))
    assert np.all(np.diff(heights) > 0)


def test_no_overwrite_and_invalid_scale(tmp_path):
    with pytest.raises(ValueError):
        generate(0.81, tmp_path / "invalid.urdf")
    with pytest.raises(FileExistsError):
        generate(0.8, SOURCE)


def test_collision_mesh_preserves_transverse_box_dimensions(models):
    paths, _ = models
    original = ET.parse(SOURCE)
    box = original.find("link[@name='FL_thigh']/collision/geometry/box")
    dimensions = np.fromstring(box.get("size"), sep=" ")
    # Original thigh box has a 90-degree rotation about y.
    expected = dimensions[[2, 1, 0]]
    for scale in SCALES[:-1]:
        variant = ET.parse(paths[scale])
        mesh = variant.find("link[@name='FL_thigh']/collision/geometry/mesh")
        points = np.array(
            [
                np.fromstring(line[2:], sep=" ")
                for line in open(mesh.get("filename"))
                if line.startswith("v ")
            ]
        )
        np.testing.assert_allclose(np.ptp(points, axis=0), expected * [1, 1, scale])
