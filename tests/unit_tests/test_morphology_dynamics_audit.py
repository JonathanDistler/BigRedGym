"""Contact/IK/energy tests independent of PPO and any feasibility claim."""

import mujoco
import numpy as np
import pytest
from experiments.morphology.dynamics_audit import Audit, ground_clearance
from scripts.audit_morphology_dynamics import FEET, waypoints


@pytest.fixture(scope="module")
def task(tmp_path_factory):
    return Audit(tmp_path_factory.mktemp("audit") / "assets")


@pytest.mark.parametrize("scale", [1.0, 0.95, 0.9])
def test_ik_ground_clearance_and_static_wrench(task, scale):
    task.reset(scale, 0)
    standing, crouch, target = waypoints(task, "relative_extension")
    for q, active in [
        (standing, (0, 1, 2, 3)),
        (crouch, (0, 1, 2, 3)),
        (target, (0, 1)),
    ]:
        row = task.static(q, active)
        np.testing.assert_allclose(
            task.d.geom_xpos[task.foot_geoms[list(active)]],
            FEET[list(active)],
            atol=1e-6,
        )
        assert row["nonfoot_clearance"] > 0.01
        assert row["static_torque_ratio"] < 1
        assert row["static_base_residual"] < 1e-4
        assert row["min_normal_force"] > 0
        assert row["static_friction_ratio"] < task.m.geom_friction[0, 0]
        bounds = task.m.jnt_range[task.jids]
        assert (q[task.qids] >= bounds[:, 0]).all()
        assert (q[task.qids] <= bounds[:, 1]).all()
    assert abs(row["com_x"] - FEET[0, 0]) < 1e-6


def test_ground_support_functions():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="sphere" size=".1" pos="0 0 .3"/>
      <geom type="box" size=".1 .2 .3" pos="0 0 .5"/>
      <geom type="cylinder" size=".1 .2" pos="0 0 .4"/>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for geom in range(3):
        assert ground_clearance(model, data, geom) == pytest.approx(0.2)


def test_contact_sample_keeps_model_and_actuator_limits(task):
    task.reset(1.0, 0)
    q, _, _ = waypoints(task, "world_geometry")
    task.set_pose(q)
    before = task.m.geom_contype.copy()
    row, events = task.sample(q[task.qids], "settle")
    np.testing.assert_array_equal(task.m.geom_contype, before)
    assert row["torque_ratio"] <= 1
    assert not events
    # External impulse has SI angular-momentum units and finite numeric data.
    assert np.isfinite(row["external_angular_impulse"])


def test_pendulum_convention_and_energy_confound():
    from experiments.morphology.dynamics_audit import (
        pendulum_acceleration,
        frozen_energy_barrier,
    )

    assert pendulum_acceleration(0.1, 0.0, 0.4, 0.0) < 0  # downward convention
    assert pendulum_acceleration(0.0, 1.0, 0.4, -0.1) > 0.0  # shortening amplifies spin
    with pytest.raises(ValueError):
        pendulum_acceleration(0.0, 0.0, 0.0, 0.0)
    # Lowering COM at fixed horizontal offset reduces L but raises the barrier.
    assert frozen_energy_barrier(1.0, 0.2, 0.2) > frozen_energy_barrier(1.0, 0.2, 0.3)
    # At fixed angle, scaling the radius down does reduce the barrier.
    assert frozen_energy_barrier(1.0, 0.16, 0.24) < frozen_energy_barrier(1.0, 0.2, 0.3)
