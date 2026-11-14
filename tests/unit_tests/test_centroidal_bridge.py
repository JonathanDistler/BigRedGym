"""Independent centroidal sum and external contact sign on a floating robot."""

from types import SimpleNamespace
import mujoco
import numpy as np
from experiments.centroidal_bridge.physics import audit


def robot():
    m = mujoco.MjModel.from_xml_string(
        """<mujoco>
<option timestep="0.0002"/>
<worldbody>

<geom type="plane" size="2 2 .1"/>
<body name="trunk" pos="0 0 .099"><freejoint/>
<geom type="sphere" size=".1" mass="2"/>
<body name="leg" pos=".1 0 0"><joint axis="0 1 0"/>
<geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass=".2"/>
</body></body></worldbody></mujoco>"""
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return SimpleNamespace(m=m, d=d, base=1)


def test_centroidal_matrix_and_link_sum():
    e = robot()
    e.d.qvel[:] = np.arange(e.m.nv) * 0.13
    mujoco.mj_forward(e.m, e.d)
    x = audit(e)
    np.testing.assert_allclose(
        np.sum(list(x["links"].values()), axis=0), x["H"], atol=1e-12
    )
    assert x["subtree_error"] < 1e-12
    assert x["velocity_error"] < 1e-12
    assert x["com_error"] < 1e-12
    assert np.linalg.eigvalsh(x["inertia"]).min() > 0


def test_contact_sign_and_angular_impulse():
    e = robot()
    x = audit(e)
    assert x["force"][2] > 0
    before = np.array(x["H"])
    impulse = np.zeros(3)
    for _ in range(20):
        x = audit(e)
        impulse += np.array(x["moment"]) * e.m.opt.timestep
        mujoco.mj_step(e.m, e.d)
        mujoco.mj_forward(e.m, e.d)
    delta = np.array(audit(e)["H"]) - before
    np.testing.assert_allclose(delta, impulse, atol=2e-5)
